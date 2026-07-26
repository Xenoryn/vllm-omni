# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D, all_to_all_4D
from vllm_omni.diffusion.distributed.functional_collectives import (
    functional_all_gather_tensor,
    functional_all_to_all_single,
    launch_functional_all_to_all_single,
    wait_functional_collective,
)
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator
from vllm_omni.platforms import current_omni_platform


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def _positive_divisors(n: int) -> list[int]:
    if n <= 0:
        return []
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


@dataclass(frozen=True, slots=True)
class _UlyssesHeadPlan:
    """Head padding that preserves the original Q:KV ratio under UAA."""

    query_heads: int
    kv_heads: int
    padded_query_heads: int
    padded_kv_heads: int

    @classmethod
    def build(
        cls,
        *,
        query_heads: int,
        kv_heads: int,
        world_size: int,
        mode: str,
    ) -> _UlyssesHeadPlan:
        if query_heads <= 0 or kv_heads <= 0:
            raise ValueError(f"Q/KV head counts must be positive, got Q={query_heads}, KV={kv_heads}.")
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"Ulysses GQA requires query heads to be a multiple of KV heads, got Q={query_heads}, KV={kv_heads}."
            )
        if mode == "advanced_uaa":
            padded_kv_heads = _ceil_div(kv_heads, world_size) * world_size
            padded_query_heads = padded_kv_heads * (query_heads // kv_heads)
        else:
            padded_query_heads = query_heads
            padded_kv_heads = kv_heads
        return cls(
            query_heads=query_heads,
            kv_heads=kv_heads,
            padded_query_heads=padded_query_heads,
            padded_kv_heads=padded_kv_heads,
        )


@torch.compiler.disable
def _all_gather_int(pg: dist.ProcessGroup, value: int, *, device: torch.device) -> list[int]:
    """All-gather a scalar int across pg.

    Note: we use a device tensor so this works for NCCL subgroups (e.g. Ulysses/Ring).
    """
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return [int(value)]

    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t, group=pg)
    return [int(x.item()) for x in gathered]


def _all_gather_2d_mask(
    pg: dist.ProcessGroup,
    mask: torch.Tensor,
    *,
    communication_backend: str,
    world_size: int,
) -> torch.Tensor:
    mask = mask.contiguous()
    if communication_backend == "functional":
        return functional_all_gather_tensor(
            mask,
            gather_dim=1,
            group=pg,
        )

    gathered = [torch.empty_like(mask) for _ in range(world_size)]
    dist.all_gather(gathered, mask, group=pg)
    return torch.cat(gathered, dim=1)


def _ulysses_all_to_all_any_qkv(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_local, H, D)
    *,
    seq_lens: list[int],
    use_sync: bool,
    padded_head_cnt: int | None = None,
    communication_backend: str = "native",
    world_size: int | None = None,
) -> tuple[torch.Tensor, int]:
    """UAA forward all-to-all: (B, S_local, H, D) -> (B, S_global, H_local, D).

    Returns:
        (resharded, orig_head_cnt)
    """
    if world_size is None:
        world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x, int(x.shape[2])

    bsz, s_local, head_cnt, head_dim = x.shape
    orig_head_cnt = int(head_cnt)
    if padded_head_cnt is None:
        padded_head_cnt = _ceil_div(orig_head_cnt, world_size) * world_size
    if padded_head_cnt < orig_head_cnt or padded_head_cnt % world_size != 0:
        raise ValueError(
            f"Invalid padded head count {padded_head_cnt} for original heads={orig_head_cnt}, world_size={world_size}."
        )
    head_pad = padded_head_cnt - orig_head_cnt
    if head_pad:
        x = F.pad(x, (0, 0, 0, head_pad))

    head_cnt_local = padded_head_cnt // world_size

    # (B, S_local, H, D) -> (world_size, S_local, B, H_local, D)
    x_t = x.reshape(bsz, s_local, world_size, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    # (world_size, S_local, B, H_local, D) -> (world_size * S_local, B, H_local, D)
    x_t = x_t.flatten(0, 1)

    input_split_sizes = [s_local] * world_size
    output_split_sizes = seq_lens
    if communication_backend == "functional":
        out = functional_all_to_all_single(
            x_t,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=pg,
        )
    else:
        s_global = sum(output_split_sizes)
        out = torch.empty(
            (s_global, bsz, head_cnt_local, head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        dist.all_to_all_single(
            out,
            x_t,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=pg,
        )
        if use_sync:
            current_omni_platform.synchronize()

    # (S_global, B, H_local, D) -> (B, S_global, H_local, D)
    out = out.permute(1, 0, 2, 3).contiguous()
    return out, orig_head_cnt


def _launch_functional_ulysses_all_to_all_any_qkv(
    pg: dist.ProcessGroup,
    x: torch.Tensor,
    *,
    seq_lens: list[int],
    padded_head_cnt: int,
    world_size: int,
) -> tuple[torch.Tensor, int]:
    """Launch graphable UAA Q/K/V resharding and defer its wait."""
    if world_size == 1:
        return x, int(x.shape[2])

    batch_size, local_seq_len, head_count, head_dim = x.shape
    original_head_count = int(head_count)
    if padded_head_cnt < original_head_count or padded_head_cnt % world_size != 0:
        raise ValueError(
            f"Invalid padded head count {padded_head_cnt} for original "
            f"heads={original_head_count}, world_size={world_size}."
        )
    if padded_head_cnt != original_head_count:
        x = F.pad(
            x,
            (0, 0, 0, padded_head_cnt - original_head_count),
        )

    local_head_count = padded_head_cnt // world_size
    input_tensor = (
        x.reshape(
            batch_size,
            local_seq_len,
            world_size,
            local_head_count,
            head_dim,
        )
        .permute(2, 1, 0, 3, 4)
        .contiguous()
        .flatten(0, 1)
    )
    output = launch_functional_all_to_all_single(
        input_tensor,
        output_split_sizes=seq_lens,
        input_split_sizes=[local_seq_len] * world_size,
        group=pg,
    )
    return output, original_head_count


def _finish_functional_ulysses_all_to_all_any_qkv(
    output: torch.Tensor,
) -> torch.Tensor:
    output = wait_functional_collective(output)
    return output.permute(1, 0, 2, 3).contiguous()


def _ulysses_all_to_all_any_o(
    pg: dist.ProcessGroup,
    x: torch.Tensor,  # (B, S_global, H_local, D)
    *,
    seq_lens: list[int],
    local_seq_len: int,
    orig_head_cnt: int,
    use_sync: bool,
    communication_backend: str = "native",
    world_size: int | None = None,
) -> torch.Tensor:
    """UAA reverse all-to-all: (B, S_global, H_local, D) -> (B, S_local, H, D)."""
    if world_size is None:
        world_size = dist.get_world_size(pg)
    if world_size == 1:
        return x

    bsz, s_global, head_cnt_local, head_dim = x.shape
    s_local = int(local_seq_len)

    # (B, S_global, H_local, D) -> (S_global, B, H_local, D)
    x_t = x.permute(1, 0, 2, 3).contiguous()

    input_split_sizes = seq_lens
    output_split_sizes = [s_local] * world_size

    if communication_backend == "functional":
        out = functional_all_to_all_single(
            x_t,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=pg,
        )
    else:
        out = torch.empty(
            (world_size * s_local, bsz, head_cnt_local, head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        dist.all_to_all_single(
            out,
            x_t,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=pg,
        )
        if use_sync:
            current_omni_platform.synchronize()

    # (world_size * S_local, B, H_local, D) -> (B, S_local, H, D)
    out = out.reshape(world_size, s_local, bsz, head_cnt_local, head_dim).permute(2, 1, 0, 3, 4).contiguous()
    out = out.reshape(bsz, s_local, world_size * head_cnt_local, head_dim)

    if out.shape[2] != orig_head_cnt:
        out = out[:, :, :orig_head_cnt, :].contiguous()
    return out


@dataclass(frozen=True, slots=True)
class _UlyssesCtx(ParallelAttentionContext):
    """Per-forward context for Ulysses sequence-parallel attention."""

    ulysses_pg: dist.ProcessGroup
    scatter_idx: int
    gather_idx: int
    use_sync: bool
    communication_backend: str = "native"
    world_size: int = 1
    joint_len: int = 0
    joint_strategy: str = "front"
    # UAA (Ulysses Anything Attention) metadata
    use_uaa: bool = False
    uaa_seq_lens: tuple[int, ...] = ()
    uaa_local_seq_len: int = 0
    orig_head_cnt: int = 0
    joint_orig_head_cnt: int = 0


class UlyssesParallelAttention:
    """Ulysses sequence-parallel strategy (all-to-all over seq/head dims).

    This preserves the semantics previously implemented in
    `Attention._forward_ulysses`:
    - If `AttentionMetadata.joint_*` is provided, joint_query/key/value are
      concatenated *after* all-to-all.
    - joint_key/value are assumed to be replicated across SP ranks and are sliced
      by ulysses head rank before concatenation.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
        mode: str = "strict",
        communication_backend: str = "native",
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
    ) -> None:
        self._sp_group = sp_group
        self._ulysses_pg = sp_group.ulysses_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._use_sync = use_sync
        self._mode = mode
        self._communication_backend = communication_backend
        self._world_size = sp_group.ulysses_world_size
        if communication_backend not in {"native", "functional"}:
            raise ValueError(
                f"Ulysses communication backend must be 'native' or 'functional', got {communication_backend!r}."
            )
        if communication_backend == "functional":
            backend = str(dist.get_backend(self._ulysses_pg)).rsplit(".", 1)[-1].lower()
            if not current_omni_platform.is_cuda() or backend != "nccl":
                raise RuntimeError(
                    "sp_communication_backend='functional' currently supports CUDA/NCCL only, "
                    f"but platform={current_omni_platform.device_name}, "
                    f"distributed_backend={backend}."
                )
            if sp_group.ring_world_size > 1:
                raise ValueError(
                    "sp_communication_backend='functional' currently supports pure Ulysses only; set ring_degree=1."
                )
        self._head_plan_value = (
            _UlyssesHeadPlan.build(
                query_heads=num_heads,
                kv_heads=num_kv_heads or num_heads,
                world_size=self._world_size,
                mode=mode,
            )
            if num_heads is not None
            else None
        )

    def _head_plan(self) -> _UlyssesHeadPlan:
        if self._head_plan_value is None:
            raise RuntimeError("Ulysses requires Attention Q/KV head counts at strategy construction.")
        return self._head_plan_value

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ulysses"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        mode = self._mode
        head_plan = self._head_plan()
        joint_tensor_query = joint_tensor_key = joint_tensor_value = None
        joint_strategy = "front"
        joint_len = 0
        joint_orig_head_cnt = 0

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_tensor_key = attn_metadata.joint_key
            joint_tensor_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        is_joint = False
        if joint_tensor_query is not None and joint_tensor_key is not None and joint_tensor_value is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported."
                    f" supported joint strategy: {supported_joint_strategy}"
                )

            # Slice joint_query for this Ulysses rank
            # joint_query is (B, S, H, D). We split H (dim 2).
            ulysses_world_size = self._world_size
            ulysses_rank = self._sp_group.ulysses_rank
            joint_head_cnt = int(joint_tensor_query.shape[-2])
            joint_kv_head_cnt = int(joint_tensor_key.shape[-2])
            joint_orig_head_cnt = joint_head_cnt
            if joint_head_cnt != head_plan.query_heads or joint_kv_head_cnt != head_plan.kv_heads:
                raise ValueError(
                    "Joint Q/KV heads must match the Attention configuration, "
                    f"got Q={joint_head_cnt}, KV={joint_kv_head_cnt}, "
                    f"expected Q={head_plan.query_heads}, KV={head_plan.kv_heads}."
                )
            if int(joint_tensor_value.shape[-2]) != joint_kv_head_cnt:
                raise ValueError("Joint K/V head counts must match.")

            if mode == "advanced_uaa":
                joint_tensor_query = F.pad(
                    joint_tensor_query,
                    (0, 0, 0, head_plan.padded_query_heads - joint_head_cnt),
                )
                joint_tensor_key = F.pad(
                    joint_tensor_key,
                    (0, 0, 0, head_plan.padded_kv_heads - joint_kv_head_cnt),
                )
                joint_tensor_value = F.pad(
                    joint_tensor_value,
                    (0, 0, 0, head_plan.padded_kv_heads - joint_kv_head_cnt),
                )
                joint_head_cnt = head_plan.padded_query_heads
                joint_kv_head_cnt = head_plan.padded_kv_heads
            else:
                if joint_head_cnt % ulysses_world_size != 0 or joint_kv_head_cnt % ulysses_world_size != 0:
                    supported = sorted(
                        set(_positive_divisors(joint_head_cnt)) & set(_positive_divisors(joint_kv_head_cnt))
                    )
                    raise ValueError(
                        "Ulysses-SP strict mode requires joint Q/KV heads divisible "
                        f"by ulysses_degree. Q={joint_head_cnt}, KV={joint_kv_head_cnt}, "
                        f"ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            attn_heads_per_ulysses_rank = joint_head_cnt // ulysses_world_size

            joint_tensor_query = joint_tensor_query[
                ...,
                attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
                :,
            ]

            joint_len = joint_tensor_query.shape[1]

            is_joint = True
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        if is_joint:
            attn_heads_per_ulysses_rank_kv = joint_kv_head_cnt // ulysses_world_size

            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]

            # Update metadata with sliced tensors so Ring attention can use them if needed
            if attn_metadata is not None:
                attn_metadata.joint_key = joint_tensor_key
                attn_metadata.joint_value = joint_tensor_value

        ulysses_world_size = self._world_size
        if mode == "advanced_uaa":
            if self._scatter_idx != 2 or self._gather_idx != 1:
                raise ValueError(
                    "ulysses_mode='advanced_uaa' currently only supports scatter_idx=2, gather_idx=1 "
                    f"(got scatter_idx={self._scatter_idx}, gather_idx={self._gather_idx})."
                )

            if self._communication_backend == "functional":
                local_seq_len = query.shape[1]
                seq_lens = [local_seq_len] * ulysses_world_size
            else:
                local_seq_len = int(query.shape[1])
                seq_lens = _all_gather_int(
                    self._ulysses_pg,
                    local_seq_len,
                    device=query.device,
                )
            s_global = sum(seq_lens)

            # In hybrid Ulysses+Ring, Ring attention uses P2P send/recv with fixed-shape
            # buffers. This requires all ring ranks to have the same seq_len after the
            # Ulysses all-to-all (i.e. per-ring-rank S_global must match).
            if self._sp_group.ring_world_size > 1:
                ring_s_globals = _all_gather_int(self._sp_group.ring_group, s_global, device=query.device)
                if len(set(ring_s_globals)) != 1:
                    raise ValueError(
                        "ulysses_mode='advanced_uaa' with hybrid Ulysses+Ring requires the "
                        "post-Ulysses seq_len to be equal across ring ranks, but got "
                        f"{ring_s_globals} (ring_degree={self._sp_group.ring_world_size}). "
                        "This typically means the input sequence was not evenly shardable across the ring. "
                        "Try setting ring_degree=1, or choose a sequence length divisible by ring_degree."
                    )
            if self._communication_backend == "functional":
                query_work, orig_head_cnt = _launch_functional_ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    query,
                    seq_lens=seq_lens,
                    padded_head_cnt=head_plan.padded_query_heads,
                    world_size=ulysses_world_size,
                )
                key_work, _ = _launch_functional_ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    key,
                    seq_lens=seq_lens,
                    padded_head_cnt=head_plan.padded_kv_heads,
                    world_size=ulysses_world_size,
                )
                value_work, _ = _launch_functional_ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    value,
                    seq_lens=seq_lens,
                    padded_head_cnt=head_plan.padded_kv_heads,
                    world_size=ulysses_world_size,
                )
                query = _finish_functional_ulysses_all_to_all_any_qkv(query_work)
                key = _finish_functional_ulysses_all_to_all_any_qkv(key_work)
                value = _finish_functional_ulysses_all_to_all_any_qkv(value_work)
            else:
                query, orig_head_cnt = _ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    query,
                    seq_lens=seq_lens,
                    use_sync=self._use_sync,
                    padded_head_cnt=head_plan.padded_query_heads,
                    communication_backend=self._communication_backend,
                    world_size=ulysses_world_size,
                )
                key, _ = _ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    key,
                    seq_lens=seq_lens,
                    use_sync=self._use_sync,
                    padded_head_cnt=head_plan.padded_kv_heads,
                    communication_backend=self._communication_backend,
                    world_size=ulysses_world_size,
                )
                value, _ = _ulysses_all_to_all_any_qkv(
                    self._ulysses_pg,
                    value,
                    seq_lens=seq_lens,
                    use_sync=self._use_sync,
                    padded_head_cnt=head_plan.padded_kv_heads,
                    communication_backend=self._communication_backend,
                    world_size=ulysses_world_size,
                )
        else:
            # Strict mode: fail fast with actionable errors for head divisibility.
            for name, t in (("query", query), ("key", key), ("value", value)):
                head_cnt = int(t.shape[2])
                if head_cnt % ulysses_world_size != 0:
                    supported = _positive_divisors(head_cnt)
                    raise ValueError(
                        "Ulysses-SP strict mode requires head_cnt divisible by ulysses_degree. "
                        f"{name}_head_cnt={head_cnt}, ulysses_degree={ulysses_world_size}. "
                        f"Try ulysses_degree in {supported}, or set ulysses_mode='advanced_uaa'."
                    )

            # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
            if self._communication_backend == "functional":
                query = all_to_all_4D(
                    query,
                    scatter_idx=self._scatter_idx,
                    gather_idx=self._gather_idx,
                    group=self._ulysses_pg,
                    communication_backend="functional",
                    world_size=ulysses_world_size,
                )
                key = all_to_all_4D(
                    key,
                    scatter_idx=self._scatter_idx,
                    gather_idx=self._gather_idx,
                    group=self._ulysses_pg,
                    communication_backend="functional",
                    world_size=ulysses_world_size,
                )
                value = all_to_all_4D(
                    value,
                    scatter_idx=self._scatter_idx,
                    gather_idx=self._gather_idx,
                    group=self._ulysses_pg,
                    communication_backend="functional",
                    world_size=ulysses_world_size,
                )
            else:
                query = SeqAllToAll4D.apply(
                    self._ulysses_pg,
                    query,
                    self._scatter_idx,
                    self._gather_idx,
                    self._use_sync,
                )
                key = SeqAllToAll4D.apply(
                    self._ulysses_pg,
                    key,
                    self._scatter_idx,
                    self._gather_idx,
                    self._use_sync,
                )
                value = SeqAllToAll4D.apply(
                    self._ulysses_pg,
                    value,
                    self._scatter_idx,
                    self._gather_idx,
                    self._use_sync,
                )
            seq_lens = []
            local_seq_len = 0
            orig_head_cnt = 0

        if is_joint:
            # Concatenate joint query AFTER AllToAll
            # Image query is now (B, S, H/P, D). Joint query is (B, S_txt, H/P, D).
            # This is dimensionally consistent.
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)

        # Check if Ring Attention is also active (Hybrid mode)
        # If Ring is active, we should NOT concatenate joint_key/value to k/v here.
        # Instead, they should remain in attn_metadata and be passed to the Ring kernel.
        use_ring = self._sp_group.ring_world_size > 1

        if is_joint and not use_ring:
            # Concatenate joint key/value after all-to-all ONLY for pure Ulysses (Local Attention).
            if joint_strategy == "front":
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
            else:  # "rear"
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)

        if (
            not is_joint
            and attn_metadata is not None
            and attn_metadata.attn_mask is not None
            and attn_metadata.attn_mask.ndim == 2
        ):
            attn_metadata.attn_mask = _all_gather_2d_mask(
                self._ulysses_pg,
                attn_metadata.attn_mask,
                communication_backend=self._communication_backend,
                world_size=ulysses_world_size,
            )

        ctx = _UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
            communication_backend=self._communication_backend,
            world_size=ulysses_world_size,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
            use_uaa=(mode == "advanced_uaa"),
            uaa_seq_lens=tuple(seq_lens) if mode == "advanced_uaa" else (),
            uaa_local_seq_len=local_seq_len if mode == "advanced_uaa" else 0,
            orig_head_cnt=int(orig_head_cnt) if mode == "advanced_uaa" else 0,
            joint_orig_head_cnt=int(joint_orig_head_cnt) if mode == "advanced_uaa" else 0,
        )
        use_2d_mask = False
        if attn_metadata is not None:
            if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 2:
                use_2d_mask = True
            if attn_metadata.joint_attn_mask is not None and attn_metadata.joint_attn_mask.ndim == 2:
                use_2d_mask = True

        if attn_metadata is not None and use_2d_mask:
            if is_joint:
                if attn_metadata.joint_attn_mask is None and attn_metadata.attn_mask is None:
                    attn_metadata.attn_mask = None
                else:
                    if attn_metadata.attn_mask is None:
                        attn_metadata.attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.joint_attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    elif attn_metadata.joint_attn_mask is None:
                        attn_metadata.joint_attn_mask = torch.ones(
                            [query.shape[0], query.shape[1] - attn_metadata.attn_mask.shape[1]],
                            dtype=torch.bool,
                            device=query.device,
                        )
                    attn_metadata.attn_mask = (
                        torch.cat([attn_metadata.joint_attn_mask, attn_metadata.attn_mask], dim=1)
                        if joint_strategy == "front"
                        else torch.cat([attn_metadata.attn_mask, attn_metadata.joint_attn_mask], dim=1)
                    )

            if attn_metadata.attn_mask is not None:
                # the final attn_mask is ready, the length should be aligedn with query length
                assert attn_metadata.attn_mask.shape[1] == query.shape[1], (
                    f"attn_mask length: {attn_metadata.attn_mask.shape[1]} != query length: {query.shape[1]}"
                )
                attn_metadata.attn_mask = attn_metadata.attn_mask.bool().contiguous()
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        assert isinstance(ctx, _UlyssesCtx), f"Unexpected ctx type: {type(ctx)!r}"

        if ctx.joint_len > 0:
            joint_len = ctx.joint_len

            if ctx.joint_strategy == "front":
                output_joint = attn_output[:, :joint_len]
                output_img = attn_output[:, joint_len:]
            else:
                output_img = attn_output[:, :-joint_len]
                output_joint = attn_output[:, -joint_len:]

            # 1. Process Image part: Standard Ulysses Reverse (AllToAll)
            # (bs, seq_len, head_cnt/P, head_size) -> (bs, seq_len/P, head_cnt, head_size)
            # SeqAllToAll4D handles: Scatter gather_idx, Gather scatter_idx.
            # Forward: Scatter 2 (H), Gather 1 (S).
            # Reverse: Scatter 1 (S), Gather 2 (H).
            if ctx.use_uaa:
                output_img = _ulysses_all_to_all_any_o(
                    ctx.ulysses_pg,
                    output_img,
                    seq_lens=list(ctx.uaa_seq_lens),
                    local_seq_len=ctx.uaa_local_seq_len,
                    orig_head_cnt=ctx.orig_head_cnt,
                    use_sync=ctx.use_sync,
                    communication_backend=ctx.communication_backend,
                    world_size=ctx.world_size,
                )
            elif ctx.communication_backend == "functional":
                output_img = all_to_all_4D(
                    output_img,
                    scatter_idx=ctx.gather_idx,
                    gather_idx=ctx.scatter_idx,
                    group=ctx.ulysses_pg,
                    communication_backend="functional",
                    world_size=ctx.world_size,
                )
            else:
                output_img = SeqAllToAll4D.apply(
                    ctx.ulysses_pg, output_img, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync
                )

            # 2. Process Joint part: AllGather on Heads
            # Input: (B, JointLen, H/P, D). Output: (B, JointLen, H, D).
            # AllGather along dim 2.
            # Ensure tensor is contiguous for all_gather (slicing may create non-contiguous views)
            output_joint = output_joint.contiguous()
            if ctx.communication_backend == "functional":
                output_joint = functional_all_gather_tensor(
                    output_joint,
                    gather_dim=2,
                    group=ctx.ulysses_pg,
                )
            else:
                gathered_joint = [torch.zeros_like(output_joint) for _ in range(ctx.world_size)]
                dist.all_gather(
                    gathered_joint,
                    output_joint,
                    group=ctx.ulysses_pg,
                )
                output_joint = torch.cat(gathered_joint, dim=2)
            if ctx.use_uaa and ctx.joint_orig_head_cnt > 0 and output_joint.shape[2] != ctx.joint_orig_head_cnt:
                output_joint = output_joint[:, :, : ctx.joint_orig_head_cnt, :].contiguous()

            # 3. Recombine
            if ctx.joint_strategy == "front":
                return torch.cat([output_joint, output_img], dim=1)
            else:
                return torch.cat([output_img, output_joint], dim=1)

        # Standard Ulysses Reverse
        if ctx.use_uaa:
            return _ulysses_all_to_all_any_o(
                ctx.ulysses_pg,
                attn_output,
                seq_lens=list(ctx.uaa_seq_lens),
                local_seq_len=ctx.uaa_local_seq_len,
                orig_head_cnt=ctx.orig_head_cnt,
                use_sync=ctx.use_sync,
                communication_backend=ctx.communication_backend,
                world_size=ctx.world_size,
            )
        if ctx.communication_backend == "functional":
            return all_to_all_4D(
                attn_output,
                scatter_idx=ctx.gather_idx,
                gather_idx=ctx.scatter_idx,
                group=ctx.ulysses_pg,
                communication_backend="functional",
                world_size=ctx.world_size,
            )
        return SeqAllToAll4D.apply(ctx.ulysses_pg, attn_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)

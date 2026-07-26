# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare native and graphable BOOGU Ulysses SP on two CUDA GPUs.

Each backend runs in a fresh process group. Alternating separately compiled
native and functional graphs in one process measurably distorts the functional
latency, especially for small shapes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import statistics
import tempfile
import time
from collections.abc import Callable
from functools import partial

import torch
import torch.distributed as dist
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    set_forward_context,
)
from vllm_omni.platforms import current_omni_platform

DIM = 3360
NUM_HEADS = 28
NUM_KV_HEADS = 7
HEAD_DIM = 120
CASES = ("native", "functional")
DEFAULT_SHAPES = (
    (1024, 1),
    (1024, 2),
    (1024, 4),
    (1024, 8),
    (1536, 1),
    (1536, 2),
    (1536, 4),
    (2048, 1),
    (2048, 2),
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        resolution, batch_size = (int(part) for part in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be RESOLUTIONxBATCH, for example 1024x4") from exc
    if resolution <= 0 or batch_size <= 0 or resolution % 16 != 0:
        raise argparse.ArgumentTypeError("resolution and batch must be positive, and resolution must divide by 16")
    return resolution, batch_size


def _make_config(communication_backend: str) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="boogu-graphable-sp-benchmark",
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(
            sequence_parallel_size=2,
            ulysses_degree=2,
            ulysses_mode="advanced_uaa",
            sp_communication_backend=communication_backend,
        ),
    )


def _measure(
    functions: dict[str, Callable[[], torch.Tensor]],
    *,
    rank: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    names = tuple(functions)
    for _ in range(warmup):
        for name in names:
            functions[name]()
    torch.accelerator.synchronize()
    dist.barrier(device_ids=[rank])

    trials: dict[str, list[float]] = {name: [] for name in names}
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            dist.barrier(device_ids=[rank])
            torch.accelerator.synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                functions[name]()
            torch.accelerator.synchronize()
            elapsed = torch.tensor(
                [(time.perf_counter() - started) * 1000.0 / iterations],
                device=torch.device("cuda", rank),
                dtype=torch.float64,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            trials[name].append(float(elapsed.item()))

    medians = {name: statistics.median(values) for name, values in trials.items()}
    result = {
        "trials_ms": trials,
        "median_ms": medians,
        "min_ms": {name: min(values) for name, values in trials.items()},
        "max_ms": {name: max(values) for name, values in trials.items()},
    }
    if "native" in medians and "functional" in medians:
        result["functional_vs_native_pct"] = (medians["native"] / medians["functional"] - 1.0) * 100.0
    return result


def _worker(rank: int, port: int, args: argparse.Namespace) -> None:
    world_size = 2
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "DIFFUSION_ATTENTION_BACKEND": "FLASH_ATTN",
        }
    )
    device = torch.device(f"cuda:{rank}")
    current_omni_platform.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )

    try:
        from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import (
            BooguImageTransformerBlock,
        )

        run_config = _make_config("functional")
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=str(device)))),
            set_forward_context(omni_diffusion_config=run_config),
            set_current_diffusion_config(run_config),
            torch.inference_mode(),
        ):
            get_forward_context()._sp_shard_depth = 1
            common_kwargs = {
                "dim": DIM,
                "num_attention_heads": NUM_HEADS,
                "num_kv_heads": NUM_KV_HEADS,
                "multiple_of": 256,
                "ffn_dim_multiplier": None,
                "norm_eps": 1e-5,
                "modulation": False,
            }
            forwards: dict[
                str,
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            ] = {}
            reference_state = None
            for case in args.case:
                case_config = _make_config(case)
                with (
                    set_forward_context(omni_diffusion_config=case_config),
                    set_current_diffusion_config(case_config),
                ):
                    get_forward_context()._sp_shard_depth = 1
                    torch.manual_seed(args.seed)
                    block = BooguImageTransformerBlock(
                        **common_kwargs,
                    ).to(device=device, dtype=torch.bfloat16)
                block.eval()
                if reference_state is None:
                    reference_state = block.state_dict()
                else:
                    block.load_state_dict(reference_state)

                def forward(
                    hidden: torch.Tensor,
                    rope: torch.Tensor,
                    *,
                    selected_block=block,
                ) -> torch.Tensor:
                    return selected_block(hidden, None, rope)

                forwards[case] = (
                    torch.compile(
                        forward,
                        dynamic=True,
                        fullgraph=args.fullgraph,
                    )
                    if not args.eager
                    else forward
                )

            results = []
            for resolution, batch_size in args.shape:
                global_seq_len = (resolution // 16) ** 2
                local_seq_len = global_seq_len // world_size
                generator = torch.Generator(device=device).manual_seed(args.seed + rank)
                hidden_states = torch.randn(
                    batch_size,
                    local_seq_len,
                    DIM,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                rotary_emb = torch.ones(
                    batch_size,
                    local_seq_len,
                    HEAD_DIM // 2,
                    device=device,
                    dtype=torch.complex64,
                )
                functions = {name: partial(forward, hidden_states, rotary_emb) for name, forward in forwards.items()}

                max_abs_error = None
                if "native" in functions and "functional" in functions:
                    with torch.inference_mode():
                        native_output = functions["native"]()
                        functional_output = functions["functional"]()
                    max_abs_error = float((native_output - functional_output).abs().max().item())
                measurement = _measure(
                    functions,
                    rank=rank,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                )
                result = {
                    "resolution": resolution,
                    "batch_size": batch_size,
                    "global_seq_len": global_seq_len,
                    "local_seq_len": local_seq_len,
                    "compiled": not args.eager,
                    "fullgraph": args.fullgraph,
                    "cases": list(args.case),
                    "max_abs_error": max_abs_error,
                    **measurement,
                }
                results.append(result)
                if rank == 0 and getattr(args, "_emit_results", True):
                    print(
                        "BOOGU_GRAPHABLE_SP_POINT=" + json.dumps(result, sort_keys=True),
                        flush=True,
                    )
                del hidden_states, rotary_emb
                torch.accelerator.empty_cache()

            if rank == 0:
                payload = json.dumps(results, sort_keys=True)
                if getattr(args, "_emit_results", True):
                    print(
                        "BOOGU_GRAPHABLE_SP_RESULT=" + payload,
                        flush=True,
                    )
                if args.output_json:
                    with open(
                        args.output_json,
                        "w",
                        encoding="utf-8",
                    ) as output:
                        output.write(payload + "\n")
    finally:
        destroy_distributed_env()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        default=None,
        help="RESOLUTIONxBATCH; repeat for multiple points",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument(
        "--case",
        action="append",
        choices=CASES,
        default=None,
        help="backend to measure; repeat for an interleaved comparison",
    )
    args = parser.parse_args()
    args.shape = tuple(args.shape or DEFAULT_SHAPES)
    args.case = tuple(dict.fromkeys(args.case or CASES))
    for name in ("warmup", "iterations", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def _spawn_workers(args: argparse.Namespace) -> None:
    torch.multiprocessing.spawn(
        _worker,
        args=(_find_free_port(), args),
        nprocs=2,
    )


def _merge_isolated_results(
    results_by_case: dict[str, list[dict[str, object]]],
    cases: tuple[str, ...],
) -> list[dict[str, object]]:
    merged_results = []
    for index, template in enumerate(results_by_case[cases[0]]):
        merged = {
            name: template[name]
            for name in (
                "resolution",
                "batch_size",
                "global_seq_len",
                "local_seq_len",
                "compiled",
                "fullgraph",
            )
        }
        merged.update(
            {
                "cases": list(cases),
                "max_abs_error": None,
                "trials_ms": {},
                "median_ms": {},
                "min_ms": {},
                "max_ms": {},
            }
        )
        for case in cases:
            result = results_by_case[case][index]
            if (
                result["resolution"],
                result["batch_size"],
            ) != (
                merged["resolution"],
                merged["batch_size"],
            ):
                raise RuntimeError("Isolated benchmark result order differs.")
            for metric in (
                "trials_ms",
                "median_ms",
                "min_ms",
                "max_ms",
            ):
                merged[metric][case] = result[metric][case]
        medians = merged["median_ms"]
        if "native" in medians and "functional" in medians:
            merged["functional_vs_native_pct"] = (medians["native"] / medians["functional"] - 1.0) * 100.0
        merged_results.append(merged)
    return merged_results


def _run_isolated_cases(args: argparse.Namespace) -> None:
    """Run each backend in a fresh process group, then merge its measurements."""
    results_by_case = {}
    with tempfile.TemporaryDirectory(prefix="boogu_graphable_sp_") as temp_dir:
        for case in args.case:
            case_args = copy.copy(args)
            case_args.case = (case,)
            case_args.output_json = os.path.join(
                temp_dir,
                f"{case}.json",
            )
            case_args._emit_results = False
            _spawn_workers(case_args)
            with open(
                case_args.output_json,
                encoding="utf-8",
            ) as result_file:
                results_by_case[case] = json.load(result_file)

    results = _merge_isolated_results(results_by_case, args.case)
    for result in results:
        print(
            "BOOGU_GRAPHABLE_SP_POINT=" + json.dumps(result, sort_keys=True),
            flush=True,
        )
    payload = json.dumps(results, sort_keys=True)
    print("BOOGU_GRAPHABLE_SP_RESULT=" + payload, flush=True)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output:
            output.write(payload + "\n")


if __name__ == "__main__":
    parsed_args = _parse_args()
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 2:
        raise RuntimeError("This benchmark requires two CUDA GPUs.")
    if len(parsed_args.case) == 1:
        _spawn_workers(parsed_args)
    else:
        _run_isolated_cases(parsed_args)

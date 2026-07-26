# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Graphable tensor-based collectives for diffusion sequence parallelism.

PyTorch's functional collectives return tensors whose dependencies can be
captured by ``torch.compile``.  Keep the private PyTorch import isolated in
this module so callers have one compatibility boundary and the native
``torch.distributed`` path can remain the default fallback.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def launch_functional_all_to_all_single(
    input_tensor: torch.Tensor,
    *,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Launch a graphable all-to-all-single without forcing materialization."""
    import torch.distributed._functional_collectives as funcol

    return funcol.all_to_all_single(
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )


def wait_functional_collective(output: torch.Tensor) -> torch.Tensor:
    """Materialize a tensor returned by a functional collective."""
    import torch.distributed._functional_collectives as funcol

    return funcol.wait_tensor(output)


def functional_all_to_all_single(
    input_tensor: torch.Tensor,
    *,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run a graphable all-to-all-single and materialize its tensor result."""
    output = launch_functional_all_to_all_single(
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return wait_functional_collective(output)


def functional_all_gather_tensor(
    input_tensor: torch.Tensor,
    *,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run a graphable all-gather along ``gather_dim``."""
    import torch.distributed._functional_collectives as funcol

    # funcol.all_gather_tensor handles non-zero gather dimensions through
    # torch._utils._maybe_view_chunk_cat, which Dynamo intentionally skips.
    # Move the gather dimension to the front and call the graphable c10d
    # primitive directly instead.
    input_tensor = input_tensor.movedim(gather_dim, 0).contiguous()
    group_name = funcol._resolve_group_name(group, "")
    group_size = funcol.c10d._get_group_size_by_name(group_name)
    output = torch.ops._c10d_functional.all_gather_into_tensor(
        input_tensor,
        group_size,
        group_name,
    )
    output = funcol._maybe_wrap_tensor(output)
    output = wait_functional_collective(output)
    return output.movedim(0, gather_dim).contiguous()

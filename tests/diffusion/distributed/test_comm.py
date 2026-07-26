# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SeqAllToAll4D and SeqAllToAll5D communication primitives."""

import os

import pytest
import torch

from vllm_omni.diffusion.distributed.comm import (
    RingComm,
    SeqAllToAll4D,
    SeqAllToAll5D,
    all_to_all_4D,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_sharding import sp_gather
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
    pytest.mark.core_model,
    pytest.mark.gpu,
]


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        os.environ[k] = v


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len_per_rank", [8])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("use_sync", [False, True])
def test_4d_identity(
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Test that two consecutive all-to-all operations return the original input."""
    # Skip if not enough GPUs available
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Test requires {world_size} GPUs but only {available_gpus} available")

    # Ensure num_heads is divisible by world_size
    if num_heads % world_size != 0:
        pytest.skip(f"num_heads ({num_heads}) not divisible by world_size ({world_size})")

    # Run test with multiprocessing spawn
    torch.multiprocessing.spawn(
        _test_4d_identity_worker,
        args=(
            world_size,
            dtype,
            batch_size,
            seq_len_per_rank,
            num_heads,
            head_size,
            use_sync,
        ),
        nprocs=world_size,
    )


def _test_4d_identity_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Worker function for test_4d_identity."""
    # Set device
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    # Set environment variables for distributed training
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
    )

    # Initialize distributed environment
    init_distributed_environment()
    initialize_model_parallel(ulysses_degree=world_size)  # test ulysses sp by default
    sp_group = get_sp_group().ulysses_group  # get ulysses sp group not ring sp group

    # Create input tensor: (bs, seqlen/P, hc, hs)
    torch.manual_seed(42 + local_rank)
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    # Save original input for comparison
    original_input = input_tensor.clone()

    # First all-to-all: (bs, seqlen/P, hc, hs) -> (bs, seqlen, hc/P, hs)
    intermediate = SeqAllToAll4D.apply(
        sp_group,
        input_tensor,
        2,  # scatter head dimension
        1,  # gather sequence dimension
        use_sync,
    )

    # Verify intermediate shape
    expected_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_shape, (
        f"Intermediate shape mismatch: expected {expected_shape}, got {intermediate.shape}"
    )

    # Second all-to-all: (bs, seqlen, hc/P, hs) -> (bs, seqlen/P, hc, hs)
    output = SeqAllToAll4D.apply(
        sp_group,
        intermediate,
        1,  # scatter sequence dimension
        2,  # gather head dimension
        use_sync,
    )

    # Verify output shape matches input
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    # Verify output matches original input
    torch.testing.assert_close(
        output,
        original_input,
        rtol=1e-5,
        atol=1e-5,
        msg="Output does not match original input after two all-to-all operations",
    )

    # Cleanup distributed environment
    destroy_distributed_env()


def test_functional_4d_is_graphable_and_matches_native():
    """Functional 4D resharding must be equivalent and compile as one graph."""
    if not current_omni_platform.is_cuda():
        pytest.skip("Functional sequence-parallel collectives require CUDA/NCCL")
    if current_omni_platform.get_device_count() < 2:
        pytest.skip("Test requires 2 GPUs")

    torch.multiprocessing.spawn(
        _test_functional_4d_worker,
        args=(2,),
        nprocs=2,
    )


def _test_functional_4d_worker(local_rank: int, world_size: int):
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29502",
        }
    )

    init_distributed_environment()
    initialize_model_parallel(ulysses_degree=world_size)
    group = get_sp_group().ulysses_group
    try:
        torch.manual_seed(42 + local_rank)
        input_tensor = torch.randn(
            2,
            8,
            8,
            32,
            dtype=torch.bfloat16,
            device=device,
        )
        expected = all_to_all_4D(
            input_tensor,
            scatter_idx=2,
            gather_idx=1,
            group=group,
            communication_backend="native",
            world_size=world_size,
        )

        def functional_forward(tensor: torch.Tensor) -> torch.Tensor:
            return all_to_all_4D(
                tensor,
                scatter_idx=2,
                gather_idx=1,
                group=group,
                communication_backend="functional",
                world_size=world_size,
            )

        compiled_forward = torch.compile(
            functional_forward,
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled_forward(input_tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        def functional_reverse(tensor: torch.Tensor) -> torch.Tensor:
            return all_to_all_4D(
                tensor,
                scatter_idx=1,
                gather_idx=2,
                group=group,
                communication_backend="functional",
                world_size=world_size,
            )

        compiled_reverse = torch.compile(
            functional_reverse,
            fullgraph=True,
            dynamic=False,
        )
        restored = compiled_reverse(actual)
        torch.testing.assert_close(restored, input_tensor, rtol=0, atol=0)

        local_output = torch.full(
            (1, 3, 2),
            local_rank,
            dtype=torch.bfloat16,
            device=device,
        )

        def functional_gather(tensor: torch.Tensor) -> torch.Tensor:
            return sp_gather(
                tensor,
                dim=1,
                communication_backend="functional",
            )

        compiled_gather = torch.compile(
            functional_gather,
            fullgraph=True,
            dynamic=False,
        )
        gathered = compiled_gather(local_output)
        expected_gather = torch.cat(
            [torch.full_like(local_output, rank) for rank in range(world_size)],
            dim=1,
        )
        torch.testing.assert_close(
            gathered,
            expected_gather,
            rtol=0,
            atol=0,
        )
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len_per_rank", [8])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("use_sync", [False, True])
def test_5d_identity(
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Test that two consecutive all-to-all operations return the original input."""
    # Skip if not enough GPUs available
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Test requires {world_size} GPUs but only {available_gpus} available")

    # Ensure num_heads is divisible by world_size
    if num_heads % world_size != 0:
        pytest.skip(f"num_heads ({num_heads}) not divisible by world_size ({world_size})")

    # Run test with multiprocessing spawn
    torch.multiprocessing.spawn(
        _test_5d_identity_worker,
        args=(
            world_size,
            dtype,
            batch_size,
            seq_len_per_rank,
            num_heads,
            head_size,
            use_sync,
        ),
        nprocs=world_size,
    )


def _test_5d_identity_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    seq_len_per_rank: int,
    num_heads: int,
    head_size: int,
    use_sync: bool,
):
    """Worker function for test_5d_identity."""
    # Set device
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    # Set environment variables for distributed training
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }
    )

    # Initialize distributed environment
    init_distributed_environment()
    initialize_model_parallel(ulysses_degree=world_size)  # test ulysses sp by default
    sp_group = get_sp_group().ulysses_group  # get ulysses sp group not ring sp group

    # Create input tensor: (bs, seqlen/P, 3, hc, hs)
    # The '3' dimension is for Q, K, V
    torch.manual_seed(42 + local_rank)
    input_tensor = torch.randn(
        batch_size,
        seq_len_per_rank,
        3,  # Q, K, V
        num_heads,
        head_size,
        dtype=dtype,
        device=device,
    )

    # Save original input for comparison
    original_input = input_tensor.clone()

    # First all-to-all: (bs, seqlen/P, 3, hc, hs) -> (bs, seqlen, 3, hc/P, hs)
    intermediate = SeqAllToAll5D.apply(
        sp_group,
        input_tensor,
        3,  # scatter head dimension
        1,  # gather sequence dimension
        use_sync,
    )

    # Verify intermediate shape
    expected_shape = (
        batch_size,
        seq_len_per_rank * world_size,
        3,
        num_heads // world_size,
        head_size,
    )
    assert intermediate.shape == expected_shape, (
        f"Intermediate shape mismatch: expected {expected_shape}, got {intermediate.shape}"
    )

    # Second all-to-all: (bs, seqlen, 3, hc/P, hs) -> (bs, seqlen/P, 3, hc, hs)
    output = SeqAllToAll5D.apply(
        sp_group,
        intermediate,
        1,  # scatter sequence dimension
        3,  # gather head dimension
        use_sync,
    )

    # Verify output shape matches input
    assert output.shape == original_input.shape, (
        f"Output shape mismatch: expected {original_input.shape}, got {output.shape}"
    )

    # Verify output matches original input
    torch.testing.assert_close(
        output,
        original_input,
        rtol=1e-5,
        atol=1e-5,
        msg="Output does not match original input after two all-to-all operations",
    )

    # Cleanup distributed environment
    destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [128])
def test_ring_p2p(
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    num_heads: int,
    head_size: int,
):
    """Test Ring P2P communication (send_recv)."""
    # Skip if not enough GPUs available
    available_gpus = current_omni_platform.get_device_count()
    if available_gpus < world_size:
        pytest.skip(f"Test requires {world_size} GPUs but only {available_gpus} available")

    torch.multiprocessing.spawn(
        _test_ring_p2p_worker,
        args=(world_size, dtype, batch_size, num_heads, head_size),
        nprocs=world_size,
    )


def _test_ring_p2p_worker(
    local_rank: int,
    world_size: int,
    dtype: torch.dtype,
    batch_size: int,
    num_heads: int,
    head_size: int,
):
    """Worker for Ring P2P test."""
    import sys

    # Set device
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)

    # Set env vars
    # Use a different port to avoid conflict with other tests if run in parallel
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29501",
        }
    )

    # Init distributed
    try:
        init_distributed_environment()
        # Ring degree = world_size to test ring group
        initialize_model_parallel(ring_degree=world_size)
        sp_group = get_sp_group()

        print(f"[Rank {local_rank}] Initialized. Ring group size: {sp_group.ring_group.size()}")
        sys.stdout.flush()

        # Create RingComm
        comm = RingComm(sp_group.ring_group)

        # Create tensor: rank-specific data
        # (batch, num_heads, head_size)
        # Fill with rank value + 1 to avoid 0 and make verification easy
        input_tensor = torch.full(
            (batch_size, num_heads, head_size), fill_value=float(local_rank + 1), dtype=dtype, device=device
        )

        print(f"[Rank {local_rank}] Input sum: {input_tensor.sum().item()}")
        sys.stdout.flush()

        # Send input, receive from prev
        # RingComm.send_recv sends to next, receives from prev
        t0 = __import__("time").time()
        recv_tensor = comm.send_recv(input_tensor)
        comm.commit()
        comm.wait()
        t1 = __import__("time").time()

        print(f"[Rank {local_rank}] Communication done in {t1 - t0:.4f}s")

        # Verify
        # Expected value: from (rank - 1) % world_size
        prev_rank = (local_rank - 1 + world_size) % world_size
        expected_value = float(prev_rank + 1)

        recv_sum = recv_tensor.sum().item()
        print(f"[Rank {local_rank}] Received sum: {recv_sum}, Expected value: {expected_value}")
        sys.stdout.flush()

        expected_tensor = torch.full_like(recv_tensor, fill_value=expected_value)

        # Use a slightly loose tolerance for bfloat16
        torch.testing.assert_close(
            recv_tensor, expected_tensor, rtol=1e-3, atol=1e-3, msg=f"[Rank {local_rank}] Data mismatch!"
        )
        print(f"[Rank {local_rank}] Verification PASSED")

    except Exception as e:
        print(f"[Rank {local_rank}] FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        raise e
    finally:
        destroy_distributed_env()

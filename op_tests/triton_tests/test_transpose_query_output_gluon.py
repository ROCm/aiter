# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for transpose_query_gluon function.

This test file validates the Gluon kernel implementation with shared memory
against the original Triton kernel implementation.
"""

import hashlib
import pytest
import torch
import aiter
from typing import Optional

# Import the functions to test
from aiter.ops.triton.gluon.pa_decode_gluon import (
    transpose_query_gluon,
    transpose_output_gluon,
    GLUON_JIT_KERNEL_ENABLED,
)
from aiter.test_common import perftest

# Import AOT functions
try:
    from csrc.cpp_itfs.pa_gluon_aot.transpose_query_output_gluon_aot import (
        transpose_query_gluon_aot,
        transpose_output_gluon_aot,
    )
except ImportError:
    print("Warning: AOT functions not available")


# Set default device to CUDA
torch.set_default_device("cuda")
torch.manual_seed(42)


def calculate_bandwidth(
    batch_size: int,
    seq_len: int,
    num_query_heads: int,
    head_size: int,
    dtype: torch.dtype,
    include_scale: bool,
    time_us: float,
) -> float:
    """Calculate memory bandwidth in GB/s.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_query_heads: Number of query heads
        head_size: Head dimension size
        dtype: Data type of tensors
        include_scale: Whether scale tensors are included
        time_us: Execution time in microseconds

    Returns:
        Bandwidth in GB/s
    """
    # Get dtype size in bytes
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    # Calculate data size for query tensor
    # Input: read query [batch_size * seq_len, num_query_heads, head_size]
    # Output: write query_output [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
    # Both have the same number of elements
    query_elements = batch_size * seq_len * num_query_heads * head_size
    query_bytes = query_elements * dtype_size

    # Total data transfer: read input + write output
    total_bytes = 2 * query_bytes

    # Add scale tensors if present
    if include_scale:
        scale_elements = batch_size * seq_len * num_query_heads * 1
        scale_bytes = scale_elements * 4
        total_bytes += 2 * scale_bytes  # read + write

    # Convert to GB/s (1 GB = 10^9 bytes)
    # time_us is in microseconds, so multiply by 1e-6 to get seconds
    time_s = time_us * 1e-6
    bandwidth_gbs = (total_bytes / 1e9) / time_s

    return bandwidth_gbs


def generate_test_tensors(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
    dtype: torch.dtype = torch.float16,
    include_scale: bool = True,
):
    """
    Generate test input and output tensors for transpose operation.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of KV heads
        query_group_size: Query group size (GQA factor)
        head_size: Head dimension
        dtype: Data type for tensors
        include_scale: Whether to include scale tensors

    Returns:
        Tuple of (query, query_output, query_scale, query_scale_output)
    """
    num_query_heads = num_kv_heads * query_group_size

    # Input query: [batch_size * seq_len, num_query_heads, head_size]
    query = torch.randn(
        batch_size * seq_len,
        num_query_heads,
        head_size,
        dtype=torch.float32,
        device="cuda",
    ).to(dtype)

    # Output query: [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
    query_output = torch.empty(
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        head_size,
        dtype=dtype,
        device="cuda",
    )

    if include_scale:
        # Scale tensors have last dimension = 1
        query_scale = torch.randn(
            batch_size * seq_len, num_query_heads, 1, dtype=torch.float32, device="cuda"
        )

        query_scale_output = torch.empty(
            batch_size,
            num_kv_heads * seq_len * query_group_size,
            1,
            dtype=torch.float32,
            device="cuda",
        )
    else:
        query_scale = None
        query_scale_output = None

    return query, query_output, query_scale, query_scale_output


@perftest()
def run_transpose_query_pytorch_reference(
    query: torch.Tensor,
    query_output: torch.Tensor,
    query_scale: Optional[torch.Tensor],
    query_scale_output: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
) -> None:
    """Run PyTorch reference implementation with performance measurement.

    This function wraps the PyTorch reference implementation with the perftest
    decorator to measure its execution time.

    Args:
        query: Input query tensor [batch_size * seq_len, num_query_heads, head_size]
        query_output: Output query tensor [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
        query_scale: Optional scale tensor for query [batch_size * seq_len, num_query_heads, 1]
        query_scale_output: Optional output scale tensor [batch_size, num_kv_heads * seq_len * query_group_size, 1]
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of key-value heads
        query_group_size: Query group size (GQA factor)
        head_size: Head dimension size

    Returns:
        None (modifies query_output and query_scale_output in-place)
        Note: The @perftest() decorator wraps this to return (None, avg_time)
    """
    # PyTorch reference implementation for query
    query_ref = query.reshape(
        batch_size, seq_len, num_kv_heads, query_group_size, head_size
    )
    query_ref = query_ref.transpose(1, 2).reshape(
        batch_size, num_kv_heads * seq_len * query_group_size, head_size
    )
    query_output.copy_(query_ref)

    # PyTorch reference implementation for query_scale
    if query_scale is not None and query_scale_output is not None:
        query_scale_ref = query_scale.reshape(
            batch_size, seq_len, num_kv_heads, query_group_size, 1
        )
        query_scale_ref = query_scale_ref.transpose(1, 2).reshape(
            batch_size, num_kv_heads * seq_len * query_group_size, 1
        )
        query_scale_output.copy_(query_scale_ref)


@perftest()
def run_transpose_query_gluon(
    query: torch.Tensor,
    query_output: torch.Tensor,
    query_scale: Optional[torch.Tensor],
    query_scale_output: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
    use_aot: bool = False,
) -> None:
    """Run transpose_query_gluon with performance measurement.

    This function wraps transpose_query_gluon with the perftest
    decorator to measure its execution time.

    Args:
        query: Input query tensor [batch_size * seq_len, num_query_heads, head_size]
        query_output: Output query tensor [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
        query_scale: Optional scale tensor for query [batch_size * seq_len, num_query_heads, 1]
        query_scale_output: Optional output scale tensor [batch_size, num_kv_heads * seq_len * query_group_size, 1]
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of key-value heads
        query_group_size: Query group size (GQA factor)
        head_size: Head dimension size
        use_aot: Whether to use AOT compiled kernel

    Returns:
        None (modifies query_output and query_scale_output in-place)
        Note: The @perftest() decorator wraps this to return (None, avg_time)
    """
    if use_aot:
        # Use AOT compiled kernel
        transpose_query_gluon_aot(
            input_tensor=query,
            output_tensor=query_output,
            batch_size=batch_size,
            seq_len=seq_len,
            num_kv_heads=num_kv_heads,
            query_group_size=query_group_size,
            last_dim=head_size,
            input_scale=(
                query_scale
                if (query_scale is not None and query_scale_output is not None)
                else None
            ),
            output_scale=(
                query_scale_output
                if (query_scale is not None and query_scale_output is not None)
                else None
            ),
            run_compiled_kernel=True,
        )
    else:
        # Use JIT kernel
        transpose_query_gluon(
            query,
            query_output,
            query_scale,
            query_scale_output,
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
        )


@perftest()
def run_transpose_output_pytorch_reference(
    output_gluon: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
) -> None:
    """Run PyTorch reference implementation for output transpose with performance measurement.

    This function implements the transpose operation:
        output_final = output_gluon.reshape(batch_size, num_kv_heads, seq_len, query_group_size, head_size)
        output_final = output_final.transpose(1, 2).reshape(batch_size * seq_len, num_query_heads, head_size)

    Args:
        output_gluon: Input tensor [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
        output: Output tensor [batch_size * seq_len, num_query_heads, head_size]
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of key-value heads
        query_group_size: Query group size (GQA factor)
        head_size: Head dimension size

    Returns:
        None (modifies output in-place)
        Note: The @perftest() decorator wraps this to return (None, avg_time)
    """
    # PyTorch reference implementation for output transpose
    output_ref = output_gluon.reshape(
        batch_size, num_kv_heads, seq_len, query_group_size, head_size
    )
    output_ref = output_ref.transpose(1, 2).reshape(
        batch_size * seq_len, num_kv_heads * query_group_size, head_size
    )
    output.copy_(output_ref)


@perftest()
def run_transpose_output_gluon(
    output_gluon: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
    use_aot: bool = False,
) -> None:
    """Run transpose_output_gluon with performance measurement.

    This function wraps transpose_output_gluon with the perftest
    decorator to measure its execution time.

    Args:
        output_gluon: Input tensor [batch_size, num_kv_heads * seq_len * query_group_size, head_size] (3D physical)
                      Interpreted as [batch_size, num_kv_heads, seq_len, query_group_size, head_size] (5D logical)
        output: Output tensor [batch_size * seq_len, num_query_heads, head_size]
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of key-value heads
        query_group_size: Query group size (GQA factor)
        head_size: Head dimension size
        use_aot: Whether to use AOT compiled kernel

    Returns:
        None (modifies output in-place)
        Note: The @perftest() decorator wraps this to return (None, avg_time)
    """
    if use_aot:
        # Use AOT compiled kernel
        transpose_output_gluon_aot(
            input_tensor=output_gluon,
            output_tensor=output,
            batch_size=batch_size,
            seq_len=seq_len,
            num_kv_heads=num_kv_heads,
            query_group_size=query_group_size,
            last_dim=head_size,
            run_compiled_kernel=True,
        )
    else:
        # Use JIT kernel
        transpose_output_gluon(
            output_gluon,
            output,
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
        )


# @pytest.mark.parametrize("batch_size", [1, 4, 8])
# @pytest.mark.parametrize("seq_len", [1, 2, 4])
# @pytest.mark.parametrize("num_kv_heads", [4, 8, 16])
# @pytest.mark.parametrize("query_group_size", [1, 2, 4])
# @pytest.mark.parametrize("head_size", [64, 128, 256])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("include_scale", [True, False])
def run_transpose_query_gluon_vs_triton_test(
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
    dtype: torch.dtype,
    include_scale: bool,
    test_aot: bool = False,
):
    """
    Test that Gluon kernel produces the same results as Triton kernel.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of KV heads
        query_group_size: Query group size
        head_size: Head dimension
        dtype: Data type
        include_scale: Whether to test scale tensors
        test_aot: Whether to test AOT compiled kernel
    """
    # ==================== Test transpose_query_gluon ====================
    print("-" * 80)
    print("Testing transpose_query_gluon")
    print("-" * 80)

    # Generate test tensors
    query, _, query_scale, _ = generate_test_tensors(
        batch_size,
        seq_len,
        num_kv_heads,
        query_group_size,
        head_size,
        dtype,
        include_scale,
    )

    # Run PyTorch reference implementation for comparison
    query_output_ref = torch.empty(
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        head_size,
        dtype=dtype,
        device="cuda",
    )
    query_scale_output_ref = None
    if include_scale:
        query_scale_output_ref = torch.empty(
            batch_size,
            num_kv_heads * seq_len * query_group_size,
            1,
            dtype=torch.float32,
            device="cuda",
        )

    _, pytorch_time = run_transpose_query_pytorch_reference(
        query.clone(),
        query_output_ref,
        query_scale.clone() if query_scale is not None else None,
        query_scale_output_ref,
        batch_size,
        seq_len,
        num_kv_heads,
        query_group_size,
        head_size,
    )
    query_output_ref_md5 = hashlib.md5(
        query_output_ref.view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()
    num_query_heads = num_kv_heads * query_group_size
    pytorch_bandwidth = calculate_bandwidth(
        batch_size,
        seq_len,
        num_query_heads,
        head_size,
        dtype,
        include_scale,
        pytorch_time,
    )
    print(f"query_output_ref_md5={query_output_ref_md5}")
    print(
        f"PyTorch reference avg time: {pytorch_time:.2f} us, bandwidth: {pytorch_bandwidth:.2f} GB/s"
    )
    if include_scale:
        query_scale_output_ref_md5 = hashlib.md5(
            query_scale_output_ref.view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        print(f"query_scale_output_ref_md5={query_scale_output_ref_md5}")

    # Run Gluon kernel
    if GLUON_JIT_KERNEL_ENABLED:
        query_output_gluon = torch.empty(
            batch_size,
            num_kv_heads * seq_len * query_group_size,
            head_size,
            dtype=dtype,
            device="cuda",
        )
        query_scale_output_gluon = None
        if include_scale:
            query_scale_output_gluon = torch.empty(
                batch_size,
                num_kv_heads * seq_len * query_group_size,
                1,
                dtype=torch.float32,
                device="cuda",
            )

        _, gluon_time = run_transpose_query_gluon(
            query.clone(),
            query_output_gluon,
            query_scale.clone() if query_scale is not None else None,
            query_scale_output_gluon,
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
            use_aot=False,
        )
        query_output_gluon_md5 = hashlib.md5(
            query_output_gluon.view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        query_diff = (
            (query_output_gluon.to(torch.float32) - query_output_ref.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        gluon_bandwidth = calculate_bandwidth(
            batch_size,
            seq_len,
            num_query_heads,
            head_size,
            dtype,
            include_scale,
            gluon_time,
        )
        print(f"query_output_gluon_md5={query_output_gluon_md5}")
        print(
            f"Gluon vs PyTorch reference max diff: {query_diff}, "
            f"avg time: {gluon_time:.2f} us, bandwidth: {gluon_bandwidth:.2f} GB/s"
        )
        assert query_diff < 1e-5, f"Query transpose difference too large: {query_diff}"
        if include_scale:
            query_scale_output_gluon_md5 = hashlib.md5(
                query_scale_output_gluon.view(torch.uint8)
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
            print(f"query_scale_output_gluon_md5={query_scale_output_gluon_md5}")
            query_scale_diff = (
                (
                    query_scale_output_gluon.to(torch.float32)
                    - query_scale_output_ref.to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )
            print(
                f"Gluon vs PyTorch reference query scale max diff: {query_scale_diff}"
            )
            assert (
                query_scale_diff < 1e-5
            ), f"Query scale transpose difference too large: {query_scale_diff}"

    # ==================== Test transpose_query_gluon AOT ====================
    if test_aot:
        print("-" * 80)
        print("Testing transpose_query_gluon AOT")
        print("-" * 80)

        query_output_aot = torch.empty(
            batch_size,
            num_kv_heads * seq_len * query_group_size,
            head_size,
            dtype=dtype,
            device="cuda",
        )
        query_scale_output_aot = None
        if include_scale:
            query_scale_output_aot = torch.empty(
                batch_size,
                num_kv_heads * seq_len * query_group_size,
                1,
                dtype=torch.float32,
                device="cuda",
            )

        _, aot_time = run_transpose_query_gluon(
            query.clone(),
            query_output_aot,
            query_scale.clone() if query_scale is not None else None,
            query_scale_output_aot,
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
            use_aot=True,
        )
        query_output_aot_md5 = hashlib.md5(
            query_output_aot.view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        query_aot_diff = (
            (query_output_aot.to(torch.float32) - query_output_ref.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        aot_bandwidth = calculate_bandwidth(
            batch_size,
            seq_len,
            num_query_heads,
            head_size,
            dtype,
            include_scale,
            aot_time,
        )
        print(f"query_output_aot_md5={query_output_aot_md5}")
        print(
            f"AOT vs PyTorch reference max diff: {query_aot_diff}, "
            f"avg time: {aot_time:.2f} us, bandwidth: {aot_bandwidth:.2f} GB/s"
        )
        assert (
            query_aot_diff < 1e-5
        ), f"Query AOT transpose difference too large: {query_aot_diff}"

        if include_scale:
            query_scale_output_aot_md5 = hashlib.md5(
                query_scale_output_aot.view(torch.uint8)
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
            print(f"query_scale_output_aot_md5={query_scale_output_aot_md5}")
            query_scale_aot_diff = (
                (
                    query_scale_output_aot.to(torch.float32)
                    - query_scale_output_ref.to(torch.float32)
                )
                .abs()
                .max()
                .item()
            )
            print(
                f"AOT vs PyTorch reference query scale max diff: {query_scale_aot_diff}"
            )
            assert (
                query_scale_aot_diff < 1e-5
            ), f"Query scale AOT transpose difference too large: {query_scale_aot_diff}"

    # ==================== Test transpose_output_gluon ====================
    print("-" * 80)
    print("Testing transpose_output_gluon")
    print("-" * 80)

    output_dtype = torch.bfloat16 if dtype == aiter.dtypes.fp8 else dtype
    # Generate test data for output transpose
    # Input: [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
    output_gluon_input = torch.randn(
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    ).to(output_dtype)

    # Run PyTorch reference implementation
    output_ref = torch.empty(
        batch_size * seq_len,
        num_kv_heads * query_group_size,
        head_size,
        dtype=output_dtype,
        device="cuda",
    )

    _, pytorch_output_time = run_transpose_output_pytorch_reference(
        output_gluon_input.clone(),
        output_ref,
        batch_size,
        seq_len,
        num_kv_heads,
        query_group_size,
        head_size,
    )
    output_ref_md5 = hashlib.md5(
        output_ref.view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()
    pytorch_output_bandwidth = calculate_bandwidth(
        batch_size,
        seq_len,
        num_query_heads,
        head_size,
        output_dtype,
        False,
        pytorch_output_time,
    )
    print(f"output_ref_md5={output_ref_md5}")
    print(
        f"PyTorch output transpose avg time: {pytorch_output_time:.2f} us, bandwidth: {pytorch_output_bandwidth:.2f} GB/s"
    )

    # Run Gluon kernel
    if GLUON_JIT_KERNEL_ENABLED:
        output_gluon = torch.empty(
            batch_size * seq_len,
            num_kv_heads * query_group_size,
            head_size,
            dtype=output_dtype,
            device="cuda",
        )

        _, gluon_output_time = run_transpose_output_gluon(
            output_gluon_input.clone(),
            output_gluon,
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
            use_aot=False,
        )
        output_gluon_md5 = hashlib.md5(
            output_gluon.view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        output_diff = (
            (output_gluon.to(torch.float32) - output_ref.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        gluon_output_bandwidth = calculate_bandwidth(
            batch_size,
            seq_len,
            num_query_heads,
            head_size,
            output_dtype,
            False,
            gluon_output_time,
        )
        print(f"output_gluon_md5={output_gluon_md5}")
        print(
            f"Gluon output transpose vs PyTorch reference max diff: {output_diff}, "
            f"avg time: {gluon_output_time:.2f} us, bandwidth: {gluon_output_bandwidth:.2f} GB/s"
        )

        # Assert that the difference is small
        assert (
            output_diff < 1e-5
        ), f"Output transpose difference too large: {output_diff}"

    # ==================== Test transpose_output_gluon AOT ====================
    if test_aot:
        print("-" * 80)
        print("Testing transpose_output_gluon AOT")
        print("-" * 80)

        output_aot = torch.empty(
            batch_size * seq_len,
            num_kv_heads * query_group_size,
            head_size,
            dtype=output_dtype,
            device="cuda",
        )

        _, aot_output_time = run_transpose_output_gluon(
            output_gluon_input.clone(),
            output_aot,
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
            use_aot=True,
        )
        output_aot_md5 = hashlib.md5(
            output_aot.view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        output_aot_diff = (
            (output_aot.to(torch.float32) - output_ref.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        aot_output_bandwidth = calculate_bandwidth(
            batch_size,
            seq_len,
            num_query_heads,
            head_size,
            output_dtype,
            False,
            aot_output_time,
        )
        print(f"output_aot_md5={output_aot_md5}")
        print(
            f"AOT output transpose vs PyTorch reference max diff: {output_aot_diff}, "
            f"avg time: {aot_output_time:.2f} us, bandwidth: {aot_output_bandwidth:.2f} GB/s"
        )

        # Assert that the difference is small
        assert (
            output_aot_diff < 1e-5
        ), f"Output AOT transpose difference too large: {output_aot_diff}"


if __name__ == "__main__":
    print("=" * 80)
    print("Testing transpose_query_gluon")
    print("=" * 80)

    # Check if Gluon is available
    if GLUON_JIT_KERNEL_ENABLED:
        print("✓ Gluon JIT kernel is enabled")
    else:
        print(
            "⚠ Gluon JIT kernel is NOT enabled, so the kernel can only run in AOT mode"
        )

    # Format: (batch_size, seq_len, num_kv_heads, query_group_size, head_size, dtype, include_scale, test_aot)
    test_configs = [
        (128, 4, 4, 16, 128, torch.float16, True, True),
        (128, 4, 4, 16, 128, torch.bfloat16, True, True),
        (128, 2, 4, 16, 128, torch.bfloat16, True, True),
        (128, 2, 1, 16, 128, torch.bfloat16, True, True),
        (128, 4, 4, 16, 128, torch.float16, False, True),
        (128, 4, 4, 16, 128, torch.bfloat16, False, True),
        (128, 2, 4, 16, 128, torch.bfloat16, False, True),
        (128, 2, 1, 16, 128, torch.bfloat16, False, True),
        (128, 4, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (128, 3, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (128, 2, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (128, 1, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (64, 4, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (64, 3, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (64, 2, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (64, 1, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (1, 4, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (1, 1, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (16384, 4, 4, 16, 128, torch.float8_e4m3fnuz, True, True),
        (128, 4, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (128, 3, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (128, 2, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (128, 1, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (64, 4, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (64, 3, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (64, 2, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (64, 1, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (1, 4, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (1, 1, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
        (16384, 4, 4, 16, 128, torch.float8_e4m3fnuz, False, True),
    ]

    for (
        batch_size,
        seq_len,
        num_kv_heads,
        query_group_size,
        head_size,
        dtype,
        include_scale,
        test_aot,
    ) in test_configs:
        aot_status = " (with AOT)" if test_aot else ""
        print(
            f"\nTesting: batch={batch_size}, seq={seq_len}, kv_heads={num_kv_heads}, "
            f"q_group={query_group_size}, head={head_size}, dtype={dtype}, scale={include_scale}{aot_status}"
        )
        run_transpose_query_gluon_vs_triton_test(
            batch_size,
            seq_len,
            num_kv_heads,
            query_group_size,
            head_size,
            dtype,
            include_scale,
            test_aot,
        )
        print("✓ Passed")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)

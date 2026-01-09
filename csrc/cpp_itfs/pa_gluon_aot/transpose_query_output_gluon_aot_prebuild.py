# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import torch
import triton
import aiter
from itertools import product
from multiprocessing import Pool, cpu_count

from csrc.cpp_itfs.utils import BUILD_DIR, get_default_func_name
from csrc.cpp_itfs.pa_gluon_aot.transpose_query_output_gluon_aot import (
    transpose_query_gluon_aot,
    transpose_output_gluon_aot,
    MD_NAME_QUERY,
    MD_NAME_OUTPUT,
)


# Configuration options for prebuilding
HEAD_DIMENSION_OPTIONS = [64, 128, 192, 256]
HEAD_CONFIGURATIONS = [
    (5, 1),
    (8, 1),
    (10, 1),
    (16, 1),
    (64, 4),
    (64, 8),
]  # (num_query_heads, num_kv_heads)
QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
DATA_TYPE_OPTIONS = ["fp8", "bf16", "fp16"]

# Set default device
torch.set_default_device("cuda")


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "fp8": aiter.dtypes.fp8,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported data type string: {dtype_str}")
    return dtype_map[dtype_str]


def run_transpose_query_test(
    dtype_str: str,
    num_kv_heads: int,
    query_group_size: int,
    seq_len: int,
    head_dim: int,
    batch_size: int = 2,
    test_scale: bool = False,
):
    """
    Run a single test case for transpose_query_gluon_aot.

    This will trigger compilation if the kernel hasn't been compiled yet,
    and then run the kernel to verify it works.

    Args:
        dtype_str: Data type string ("bf16", "fp16", "fp8")
        num_kv_heads: Number of KV heads
        query_group_size: Query group size
        seq_len: Sequence length
        head_dim: Head dimension
        batch_size: Batch size
        test_scale: Whether to test scale tensor transpose
    """
    data_type = str_to_dtype(dtype_str)
    num_query_heads = num_kv_heads * query_group_size

    # Create input and output tensors
    input_tensor = torch.randn(
        batch_size * seq_len,
        num_query_heads,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    ).to(data_type)

    output_tensor = torch.empty(
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    ).to(data_type)

    # Create scale tensors if needed
    input_scale = None
    output_scale = None
    if test_scale:
        input_scale = torch.randn(
            batch_size * seq_len, num_query_heads, 1, dtype=torch.float32, device="cuda"
        )
        output_scale = torch.empty(
            batch_size,
            num_kv_heads * seq_len * query_group_size,
            1,
            dtype=torch.float32,
            device="cuda",
        )

    # Run the AOT kernel (this will trigger compilation)
    transpose_query_gluon_aot(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        batch_size=batch_size,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        last_dim=head_dim,
        input_scale=input_scale,
        output_scale=output_scale,
        run_compiled_kernel=True,
    )

    return True


def run_transpose_output_test(
    dtype_str: str,
    num_kv_heads: int,
    query_group_size: int,
    seq_len: int,
    head_dim: int,
    batch_size: int = 2,
):
    """
    Run a single test case for transpose_output_gluon_aot.

    This will trigger compilation if the kernel hasn't been compiled yet,
    and then run the kernel to verify it works.
    """
    data_type = str_to_dtype(dtype_str)
    output_dtype = torch.bfloat16 if data_type == aiter.dtypes.fp8 else data_type
    num_query_heads = num_kv_heads * query_group_size

    # Create input tensor with 3D physical shape
    # Physical shape: [batch_size, num_kv_heads * seq_len * query_group_size, head_dim]
    # Logical layout: [batch_size, num_kv_heads, seq_len, query_group_size, head_dim] (5D view)
    input_tensor = torch.randn(
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    ).to(output_dtype)

    output_tensor = torch.empty(
        batch_size * seq_len,
        num_query_heads,
        head_dim,
        dtype=output_dtype,
        device="cuda",
    )

    # Run the AOT kernel (this will trigger compilation)
    transpose_output_gluon_aot(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        batch_size=batch_size,
        seq_len=seq_len,
        num_kv_heads=num_kv_heads,
        query_group_size=query_group_size,
        last_dim=head_dim,
        run_compiled_kernel=True,
    )

    return True


def _run_single_test_wrapper(test_args):
    """Wrapper function for multiprocessing."""
    test_config, idx, total = test_args

    kernel_type = test_config["kernel_type"]
    dtype_str = test_config["dtype"]
    num_kv_heads = test_config["num_kv_heads"]
    query_group_size = test_config["query_group_size"]
    seq_len = test_config["seq_len"]
    head_dim = test_config["head_dim"]
    test_scale = test_config.get("test_scale", False)

    scale_info = " (with scale)" if test_scale else ""
    print(
        f"[{idx}/{total}] Running {kernel_type} test{scale_info}: "
        f"dtype={dtype_str}, num_kv_heads={num_kv_heads}, "
        f"query_group_size={query_group_size}, seq_len={seq_len}, head_dim={head_dim}"
    )

    try:
        if kernel_type == "transpose_query":
            run_transpose_query_test(
                dtype_str=dtype_str,
                num_kv_heads=num_kv_heads,
                query_group_size=query_group_size,
                seq_len=seq_len,
                head_dim=head_dim,
                test_scale=test_scale,
            )
        else:  # transpose_output
            run_transpose_output_test(
                dtype_str=dtype_str,
                num_kv_heads=num_kv_heads,
                query_group_size=query_group_size,
                seq_len=seq_len,
                head_dim=head_dim,
            )

        print(f"  ✓ [{idx}/{total}] {kernel_type} test passed")
        return {"status": "success", "config": test_config}

    except Exception as e:
        print(f"  ✗ [{idx}/{total}] {kernel_type} test failed: {e}")
        return {"status": "failed", "config": test_config, "error": str(e)}


def prebuild_transpose_query_gluon_aot_so(num_processes=None):
    """Prebuild all .so files for transpose_query_gluon_aot and transpose_output_gluon_aot."""

    print("=" * 80)
    print(
        "Starting prebuild of transpose_query_gluon_aot and transpose_output_gluon_aot"
    )
    print("=" * 80)

    # Configuration summary
    print("\nConfiguration:")
    print(f"  HEAD_DIMENSION_OPTIONS: {HEAD_DIMENSION_OPTIONS}")
    print(f"  HEAD_CONFIGURATIONS: {HEAD_CONFIGURATIONS}")
    print(f"  QUERY_LENGTH_OPTIONS: {QUERY_LENGTH_OPTIONS}")
    print(f"  DATA_TYPE_OPTIONS: {DATA_TYPE_OPTIONS}")

    # Generate all test configurations
    test_configs = []

    for dtype_str, (num_query_heads, num_kv_heads), seq_len, head_dim in product(
        DATA_TYPE_OPTIONS,
        HEAD_CONFIGURATIONS,
        QUERY_LENGTH_OPTIONS,
        HEAD_DIMENSION_OPTIONS,
    ):
        data_type = str_to_dtype(dtype_str)

        # Calculate parameters for transpose_query kernel
        query_group_size = num_query_heads // num_kv_heads
        merged_dim_size_query = num_kv_heads * seq_len * query_group_size
        merged_block_size_query = triton.next_power_of_2(merged_dim_size_query)
        block_size_last_query = triton.next_power_of_2(head_dim)

        # Calculate func_name for transpose_query (head_size)
        func_name_query = get_default_func_name(
            MD_NAME_QUERY,
            (
                data_type,
                merged_block_size_query,
                block_size_last_query,
            ),
        )

        # Add transpose_query test (head_size)
        test_configs.append(
            {
                "kernel_type": "transpose_query",
                "dtype": dtype_str,
                "num_kv_heads": num_kv_heads,
                "query_group_size": query_group_size,
                "seq_len": seq_len,
                "head_dim": head_dim,
                "func_name": func_name_query,
                "test_scale": False,
            }
        )

        # Calculate func_name for transpose_query with scale (last_dim=1)
        block_size_last_scale = 1
        func_name_query_scale = get_default_func_name(
            MD_NAME_QUERY,
            (
                torch.float32,
                merged_block_size_query,
                block_size_last_scale,
            ),
        )

        # Add transpose_query test for scale (last_dim=1)
        test_configs.append(
            {
                "kernel_type": "transpose_query",
                "dtype": dtype_str,
                "num_kv_heads": num_kv_heads,
                "query_group_size": query_group_size,
                "seq_len": seq_len,
                "head_dim": head_dim,
                "func_name": func_name_query_scale,
                "test_scale": True,
            }
        )

        # Calculate parameters for transpose_output kernel
        merged_dim_size_output = num_kv_heads * query_group_size
        merged_block_size_output = triton.next_power_of_2(merged_dim_size_output)
        block_size_last_output = triton.next_power_of_2(head_dim)

        output_dtype = torch.bfloat16 if data_type == aiter.dtypes.fp8 else data_type
        # Calculate func_name for transpose_output
        func_name_output = get_default_func_name(
            MD_NAME_OUTPUT,
            (
                output_dtype,
                merged_block_size_output,
                block_size_last_output,
            ),
        )

        # Add transpose_output test
        test_configs.append(
            {
                "kernel_type": "transpose_output",
                "dtype": dtype_str,
                "num_kv_heads": num_kv_heads,
                "query_group_size": query_group_size,
                "seq_len": seq_len,
                "head_dim": head_dim,
                "func_name": func_name_output,
                "test_scale": False,
            }
        )

    # Filter out duplicate test configs based on func_name
    total_before_dedup = len(test_configs)
    print(f"\nTotal test cases before deduplication: {total_before_dedup}")

    seen_func_names = set()
    unique_test_configs = []
    for test_config in test_configs:
        func_name = test_config.get("func_name")
        if func_name not in seen_func_names:
            seen_func_names.add(func_name)
            unique_test_configs.append(test_config)

    test_configs = unique_test_configs
    total = len(test_configs)
    print(f"Total test cases after deduplication: {total}")
    print(f"Removed {total_before_dedup - total} duplicate configurations")
    print("=" * 80)

    # Prepare arguments for multiprocessing
    test_args = [(config, idx + 1, total) for idx, config in enumerate(test_configs)]

    # Determine number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), total)
        num_processes = min(num_processes, 128)

    print(f"Using {num_processes} parallel processes\n")

    # Run tests in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(_run_single_test_wrapper, test_args)

    # Count successes and failures
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    print("\n" + "=" * 80)
    print("Compilation and Test Summary:")
    print(f"  Total test cases: {total}")
    print(f"  Successfully passed: {success_count}")
    print(f"  Failed: {failed_count}")
    print("=" * 80)

    if failed_count > 0:
        print("\nFailed test cases:")
        for result in results:
            if result["status"] == "failed":
                config = result["config"]
                print(
                    f"  - {config['kernel_type']}: dtype={config['dtype']}, "
                    f"num_kv_heads={config['num_kv_heads']}, "
                    f"seq_len={config['seq_len']}, head_dim={config['head_dim']}"
                )
                print(f"    Error: {result['error']}")

    # Get the total size of so files in aiter build directory
    try:
        du_result = subprocess.run(
            ["du", "-sh", BUILD_DIR], capture_output=True, text=True, timeout=100
        )
        if du_result.returncode == 0:
            total_size_of_so_files = du_result.stdout.split()[0]
            print(
                f"The total size of so files in aiter build directory: {total_size_of_so_files}"
            )
    except Exception as e:
        print(
            f"Warning: Could not get the total size of so files in aiter build directory: {e}"
        )

    # Get the number of so files in aiter build directory
    try:
        so_count_result = subprocess.run(
            ["sh", "-c", f"find {BUILD_DIR} -type f -name '*.so' | wc -l"],
            capture_output=True,
            text=True,
            timeout=100,
        )
        if so_count_result.returncode == 0:
            number_of_so_files = so_count_result.stdout.strip()
            print(
                f"The number of so files in aiter build directory: {number_of_so_files}"
            )
    except Exception as e:
        print(
            f"Warning: Could not get the number of so files in aiter build directory: {e}"
        )

    print("\n" + "=" * 80)
    print(
        "All the so files under different configurations have been built successfully!"
    )
    print("=" * 80)


if __name__ == "__main__":
    prebuild_transpose_query_gluon_aot_so()

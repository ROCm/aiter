import os
import time
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Optional
from jinja2 import Template
import triton
import aiter
import torch

GLUON_AOT_COMPILE_ENABLED = True
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    print(
        "Warning: triton.experimental.gluon or triton.experimental.gluon.language not exists, transpose_query_gluon_aot cannot use compile mode!"
    )
    GLUON_AOT_COMPILE_ENABLED = False

from csrc.cpp_itfs.gluon_aot_tools.compile_gluon import (
    compile_gluon_kernel,
    CompileGluonArgs,
)
from csrc.cpp_itfs.torch_utils import torch_to_c_types
from csrc.cpp_itfs.utils import (
    BUILD_DIR,
    AITER_CORE_DIR,
    get_default_func_name,
    compile_template_op,
    mp_lock,
    not_built,
    run_lib,
    logger,
)

MD_NAME_QUERY = "transpose_query_gluon_kernel"
MD_NAME_OUTPUT = "transpose_output_gluon_kernel"


def compile_query(
    data_type: torch.dtype,
    merged_block_size: int,
    block_size_last: int,
    func_name: str = None,
):
    """Compile the transpose_query_gluon_kernel."""

    if func_name is None:
        func_name = get_default_func_name(
            MD_NAME_QUERY,
            (
                data_type,
                merged_block_size,
                block_size_last,
            ),
        )

    if not_built(func_name):
        if not GLUON_AOT_COMPILE_ENABLED:
            raise RuntimeError(
                "This version triton is not support gluon aot compile, please upgrade to 3.5.0 or higher!"
            )

        from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import (
            clean_directory_except_so,
        )

        # Determine signature based on data type
        if data_type == torch.bfloat16:
            data_sig = "*bf16:16"
        elif data_type == torch.float16:
            data_sig = "*fp16:16"
        elif data_type == aiter.dtypes.fp8:
            data_sig = "*fp8e4b8:16"
        elif data_type == torch.float32:
            data_sig = "*fp32:16"
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Build signature for the kernel
        signature_parts = [
            data_sig,  # input_ptr
            data_sig,  # output_ptr
            "i32:16",  # batch_size
            "i32",  # seq_len
            "i32",  # num_kv_heads
            "i32:16",  # query_group_size
            "i32:16",  # last_dim
            "i32:16",  # stride_input_batch
            "i32:16",  # stride_input_seq
            "i32:16",  # stride_input_head
            "i32:16",  # stride_input_group
            "i32:16",  # stride_output_batch
            "i32:16",  # stride_output_merged
            # Grid dimensions as runtime parameters
            "i32:16",  # grid_dim_0 (batch_size)
            "i32:16",  # grid_dim_1 (merged_blocks)
            "i32:16",  # grid_dim_2 (last_blocks)
            f"{merged_block_size}",  # MERGED_BLOCK_SIZE
            f"{block_size_last}",  # BLOCK_SIZE_LAST
            f"{1}",  # STRIDE_LAST
        ]
        signature = ",".join(signature_parts)

        gluon_kernel_name = "transpose_query_gluon_kernel"

        current_dir = os.getcwd()
        aot_file_dir = f"{current_dir}/{func_name}"
        os.makedirs(aot_file_dir, exist_ok=True)

        compile_args = CompileGluonArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_gluon.py",
            kernel_name=gluon_kernel_name,
            signature=signature,
            grid="grid_dim_0,grid_dim_1,grid_dim_2",
            num_warps=4,
            waves_per_eu=1,
            num_stages=1,
            num_ctas=1,
            kpack=1,
            out_path=Path(aot_file_dir + f"/{MD_NAME_QUERY}"),
            out_name=f"{MD_NAME_QUERY}",
        )

        # Create lock directory and lock path
        lock_path = os.path.join(aot_file_dir, "lock_triton_aot_compile")
        start_ts = time.perf_counter()

        def main_func():
            """Main compilation function protected by multiprocessing lock."""
            logger.info(f"start build {func_name}")
            triton_kernel, output_files = compile_gluon_kernel(compile_args)

            # Find header and source files
            triton_header = None
            triton_source = None
            for output_file in output_files:
                if output_file.suffix == ".h":
                    triton_header = output_file
                elif output_file.suffix == ".cpp":
                    triton_source = output_file

            # Load template for C++ wrapper
            with open(
                f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon_aot/transpose_query_gluon_kernel.cpp.jinja",
                "r",
            ) as f:
                src_template = Template(f.read())

            compiled_func = compile_template_op(
                src_template,
                MD_NAME_QUERY,
                [triton_header],
                [triton_source],
                triton_header=triton_header,
                kernel_name=MD_NAME_QUERY,
                triton_kernel=triton_kernel,
                func_name=func_name,
            )
            return compiled_func

        def final_func():
            """Final function called after compilation completes."""
            logger.info(
                f"finish build {func_name}, cost {time.perf_counter()-start_ts:.8f}s"
            )

        # Use multiprocessing lock to protect the compilation process
        main_func_result = mp_lock(
            lock_path=lock_path, main_func=main_func, final_func=final_func
        )
        if main_func_result is not None:
            print(f"Cleaning aot temporary files: {aot_file_dir}")
            clean_aot_temporary_files_cmd = ["sh", "-c", f"rm -rf {aot_file_dir}"]
            result = subprocess.run(
                clean_aot_temporary_files_cmd,
                capture_output=True,
                text=True,
                timeout=100,
            )
            if result.returncode != 0 and result.stderr:
                print(f"Warning: {result.stderr}")
            print(f"Cleaning aot temporary files completed!")
            print(f"Cleaning aiter build cache directory: {BUILD_DIR}/{func_name}")
            clean_directory_except_so(f"{BUILD_DIR}/{func_name}")
            print(
                f"Cleaning aiter build cache directory completed, only *.so files are left!"
            )
            return main_func_result
        else:
            logger.info(f"{func_name} already built by another process")
            assert not not_built(func_name)
            return run_lib(func_name)
    else:
        return run_lib(func_name)


def transpose_query_gluon_aot(
    input_tensor: torch.Tensor,  # [batch_size * seq_len, num_query_heads, last_dim]
    output_tensor: torch.Tensor,  # [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    last_dim: int,
    input_scale: Optional[
        torch.Tensor
    ] = None,  # [batch_size * seq_len, num_query_heads, 1] or [1]
    output_scale: Optional[
        torch.Tensor
    ] = None,  # [batch_size, num_kv_heads * seq_len * query_group_size, 1]
    run_compiled_kernel: bool = True,
) -> None:
    """
    AOT compiled version of transpose_query_gluon_kernel.

    Args:
        input_tensor: Input tensor [batch_size * seq_len, num_query_heads, last_dim]
        output_tensor: Output tensor [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of KV heads
        query_group_size: Query group size
        last_dim: Last dimension (head_size or 1 for scale)
        input_scale: Optional scale tensor [batch_size * seq_len, num_query_heads, 1] or [1]
        output_scale: Optional output scale tensor [batch_size, num_kv_heads * seq_len * query_group_size, 1]
        run_compiled_kernel: Whether to run the compiled kernel

    Returns:
        None
    """
    # Validate input shapes
    num_query_heads = num_kv_heads * query_group_size
    assert input_tensor.shape == (
        batch_size * seq_len,
        num_query_heads,
        last_dim,
    ), f"Expected input shape ({batch_size * seq_len}, {num_query_heads}, {last_dim}), got {input_tensor.shape}"
    assert output_tensor.shape == (
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        last_dim,
    ), f"Expected output shape ({batch_size}, {num_kv_heads * seq_len * query_group_size}, {last_dim}), got {output_tensor.shape}"

    # Validate data types
    assert input_tensor.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"input tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got {input_tensor.dtype}"
    assert (
        output_tensor.dtype == input_tensor.dtype
    ), f"Output dtype {output_tensor.dtype} must match input dtype {input_tensor.dtype}"

    # Calculate strides using actual tensor strides to support non-contiguous tensors
    # Input shape: [batch_size * seq_len, num_kv_heads * query_group_size, last_dim]
    # Interpreted as 5D: [batch_size, seq_len, num_kv_heads, query_group_size, last_dim]
    input_stride_0 = input_tensor.stride(0)  # stride for dim 0 (batch*seq dimension)
    input_stride_1 = input_tensor.stride(1)  # stride for dim 1 (num_heads dimension)

    stride_input_batch = seq_len * input_stride_0  # skip seq_len rows
    stride_input_seq = input_stride_0  # skip 1 row
    stride_input_head = query_group_size * input_stride_1  # skip query_group_size heads
    stride_input_group = input_stride_1  # skip 1 head

    # Output strides using actual tensor strides
    # Output shape: [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    stride_output_batch = output_tensor.stride(0)
    stride_output_merged = output_tensor.stride(1)

    # Calculate block sizes
    merged_dim_size = num_kv_heads * seq_len * query_group_size
    merged_block_size = triton.next_power_of_2(merged_dim_size)
    block_size_last = triton.next_power_of_2(last_dim)

    # Calculate grid dimensions
    grid_dim_0 = batch_size
    grid_dim_1 = triton.cdiv(merged_dim_size, merged_block_size)
    grid_dim_2 = triton.cdiv(last_dim, block_size_last)

    # Compile the kernel
    compiled_func = compile_query(
        data_type=input_tensor.dtype,
        merged_block_size=merged_block_size,
        block_size_last=block_size_last,
    )

    assert compiled_func is not None, f"Compiled function is None"
    # Execute the compiled kernel
    if run_compiled_kernel:
        compiled_func(
            *torch_to_c_types(
                input_tensor,
                output_tensor,
                batch_size,
                seq_len,
                num_kv_heads,
                query_group_size,
                last_dim,
                stride_input_batch,
                stride_input_seq,
                stride_input_head,
                stride_input_group,
                stride_output_batch,
                stride_output_merged,
                grid_dim_0,
                grid_dim_1,
                grid_dim_2,
                torch.cuda.current_stream(output_tensor.device),
            )
        )

    # Handle query_scale if present
    if input_scale is not None and len(input_scale.shape) > 1:
        # For scale, last_dim = 1
        scale_last_dim = 1
        block_size_last_scale = 1

        # Calculate strides for query_scale using actual tensor strides
        # Input shape: [batch_size * seq_len, num_kv_heads * query_group_size, 1]
        input_scale_stride_0 = input_scale.stride(0)  # stride for dim 0
        input_scale_stride_1 = input_scale.stride(1)  # stride for dim 1

        stride_input_batch_scale = seq_len * input_scale_stride_0
        stride_input_seq_scale = input_scale_stride_0
        stride_input_head_scale = query_group_size * input_scale_stride_1
        stride_input_group_scale = input_scale_stride_1

        # Output strides using actual tensor strides
        # Output shape: [batch_size, num_kv_heads * seq_len * query_group_size, 1]
        stride_output_batch_scale = output_scale.stride(0)
        stride_output_merged_scale = output_scale.stride(1)

        # Calculate grid dimensions for scale
        grid_dim_0_scale = batch_size
        grid_dim_1_scale = triton.cdiv(merged_dim_size, merged_block_size)
        grid_dim_2_scale = 1  # last_dim = 1

        # Compile the kernel for scale (with different block_size_last)
        compiled_func_scale = compile_query(
            data_type=input_scale.dtype,
            merged_block_size=merged_block_size,
            block_size_last=block_size_last_scale,
        )

        assert compiled_func_scale is not None, f"Compiled function for scale is None"
        # Execute the compiled kernel for scale
        if run_compiled_kernel:
            compiled_func_scale(
                *torch_to_c_types(
                    input_scale,
                    output_scale,
                    batch_size,
                    seq_len,
                    num_kv_heads,
                    query_group_size,
                    scale_last_dim,
                    stride_input_batch_scale,
                    stride_input_seq_scale,
                    stride_input_head_scale,
                    stride_input_group_scale,
                    stride_output_batch_scale,
                    stride_output_merged_scale,
                    grid_dim_0_scale,
                    grid_dim_1_scale,
                    grid_dim_2_scale,
                    torch.cuda.current_stream(output_scale.device),
                )
            )


def compile_output(
    data_type: torch.dtype,
    merged_block_size: int,
    block_size_last: int,
    func_name: str = None,
):
    """Compile the transpose_output_gluon_kernel."""

    if func_name is None:
        func_name = get_default_func_name(
            MD_NAME_OUTPUT,
            (
                data_type,
                merged_block_size,
                block_size_last,
            ),
        )

    if not_built(func_name):
        if not GLUON_AOT_COMPILE_ENABLED:
            raise RuntimeError(
                "This version triton is not support gluon aot compile, please upgrade to 3.5.0 or higher!"
            )

        from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import (
            clean_directory_except_so,
        )

        # Determine signature based on data type
        if data_type == torch.bfloat16:
            data_sig = "*bf16:16"
        elif data_type == torch.float16:
            data_sig = "*fp16:16"
        elif data_type == aiter.dtypes.fp8:
            data_sig = "*fp8e4b8:16"
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Build signature for the kernel
        # Note: transpose_output has different stride names than transpose_query
        signature_parts = [
            data_sig,  # input_ptr
            data_sig,  # output_ptr
            "i32:16",  # batch_size
            "i32",  # seq_len
            "i32",  # num_kv_heads
            "i32:16",  # query_group_size
            "i32:16",  # last_dim
            "i32:16",  # stride_input_batch
            "i32:16",  # stride_input_kv_head
            "i32:16",  # stride_input_seq
            "i32:16",  # stride_input_group
            "i32:16",  # stride_output_batch_seq
            "i32:16",  # stride_output_merged
            # Grid dimensions as runtime parameters
            "i32:16",  # grid_dim_0 (batch_seq_size)
            "i32:16",  # grid_dim_1 (merged_blocks)
            "i32:16",  # grid_dim_2 (last_blocks)
            f"{merged_block_size}",  # MERGED_BLOCK_SIZE
            f"{block_size_last}",  # BLOCK_SIZE_LAST
            f"{1}",  # STRIDE_LAST
        ]
        signature = ",".join(signature_parts)

        gluon_kernel_name = "transpose_output_gluon_kernel"

        current_dir = os.getcwd()
        aot_file_dir = f"{current_dir}/{func_name}"
        os.makedirs(aot_file_dir, exist_ok=True)

        compile_args = CompileGluonArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_gluon.py",
            kernel_name=gluon_kernel_name,
            signature=signature,
            grid="grid_dim_0,grid_dim_1,grid_dim_2",
            num_warps=4,
            waves_per_eu=1,
            num_stages=1,
            num_ctas=1,
            kpack=1,
            out_path=Path(aot_file_dir + f"/{MD_NAME_OUTPUT}"),
            out_name=f"{MD_NAME_OUTPUT}",
        )

        # Create lock directory and lock path
        lock_path = os.path.join(aot_file_dir, "lock_triton_aot_compile")
        start_ts = time.perf_counter()

        def main_func():
            """Main compilation function protected by multiprocessing lock."""
            logger.info(f"start build {func_name}")
            triton_kernel, output_files = compile_gluon_kernel(compile_args)

            # Find header and source files
            triton_header = None
            triton_source = None
            for output_file in output_files:
                if output_file.suffix == ".h":
                    triton_header = output_file
                elif output_file.suffix == ".cpp":
                    triton_source = output_file

            # Load template for C++ wrapper
            with open(
                f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon_aot/transpose_output_gluon_kernel.cpp.jinja",
                "r",
            ) as f:
                src_template = Template(f.read())

            compiled_func = compile_template_op(
                src_template,
                MD_NAME_OUTPUT,
                [triton_header],
                [triton_source],
                triton_header=triton_header,
                kernel_name=MD_NAME_OUTPUT,
                triton_kernel=triton_kernel,
                func_name=func_name,
            )
            return compiled_func

        def final_func():
            """Final function called after compilation completes."""
            logger.info(
                f"finish build {func_name}, cost {time.perf_counter()-start_ts:.8f}s"
            )

        # Use multiprocessing lock to protect the compilation process
        main_func_result = mp_lock(
            lock_path=lock_path, main_func=main_func, final_func=final_func
        )
        if main_func_result is not None:
            print(f"Cleaning aot temporary files: {aot_file_dir}")
            clean_aot_temporary_files_cmd = ["sh", "-c", f"rm -rf {aot_file_dir}"]
            result = subprocess.run(
                clean_aot_temporary_files_cmd,
                capture_output=True,
                text=True,
                timeout=100,
            )
            if result.returncode != 0 and result.stderr:
                print(f"Warning: {result.stderr}")
            print(f"Cleaning aot temporary files completed!")
            print(f"Cleaning aiter build cache directory: {BUILD_DIR}/{func_name}")
            clean_directory_except_so(f"{BUILD_DIR}/{func_name}")
            print(
                f"Cleaning aiter build cache directory completed, only *.so files are left!"
            )
            return main_func_result
        else:
            logger.info(f"{func_name} already built by another process")
            assert not not_built(func_name)
            return run_lib(func_name)
    else:
        return run_lib(func_name)


def transpose_output_gluon_aot(
    input_tensor: torch.Tensor,  # [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    output_tensor: torch.Tensor,  # [batch_size * seq_len, num_query_heads, last_dim]
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    query_group_size: int,
    last_dim: int,
    run_compiled_kernel: bool = True,
) -> None:
    """
    AOT compiled version of transpose_output_gluon_kernel.

    Transpose output tensor from [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    to [batch_size * seq_len, num_query_heads, last_dim]

    Logical layout interpretation:
        Input: [batch_size, num_kv_heads, seq_len, query_group_size, last_dim] (5D logical view)
        Output: [batch_size * seq_len, num_query_heads, last_dim] (3D)

    Args:
        input_tensor: Input tensor [batch_size, num_kv_heads * seq_len * query_group_size, last_dim] (3D physical)
        output_tensor: Output tensor [batch_size * seq_len, num_query_heads, last_dim]
        batch_size: Batch size
        seq_len: Sequence length
        num_kv_heads: Number of KV heads
        query_group_size: Query group size
        last_dim: Last dimension (head_size or 1 for scale)
        run_compiled_kernel: Whether to run the compiled kernel

    Returns:
        None
    """
    # Validate input shapes
    num_query_heads = num_kv_heads * query_group_size
    assert input_tensor.shape == (
        batch_size,
        num_kv_heads * seq_len * query_group_size,
        last_dim,
    ), f"Expected input shape ({batch_size}, {num_kv_heads * seq_len * query_group_size}, {last_dim}), got {input_tensor.shape}"
    assert output_tensor.shape == (
        batch_size * seq_len,
        num_query_heads,
        last_dim,
    ), f"Expected output shape ({batch_size * seq_len}, {num_query_heads}, {last_dim}), got {output_tensor.shape}"

    # Validate data types
    assert input_tensor.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"input tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got {input_tensor.dtype}"
    assert (
        output_tensor.dtype == input_tensor.dtype
    ), f"Output dtype {output_tensor.dtype} must match input dtype {input_tensor.dtype}"

    # Calculate strides for input tensor using actual tensor strides
    # Input shape: [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    # Logical layout: [batch_size, num_kv_heads, seq_len, query_group_size, last_dim] (5D view)
    # Merged dimension is organized as: kv_head * (seq_len * group_size) + seq * group_size + group
    input_stride_0 = input_tensor.stride(0)  # stride for batch dimension
    input_stride_1 = input_tensor.stride(1)  # stride for merged dimension

    stride_input_batch = input_stride_0
    stride_input_kv_head = seq_len * query_group_size * input_stride_1
    stride_input_seq = query_group_size * input_stride_1
    stride_input_group = input_stride_1

    # Calculate strides for output tensor using actual tensor strides
    # Output shape: [batch_size * seq_len, num_query_heads, last_dim]
    stride_output_batch_seq = output_tensor.stride(0)
    stride_output_merged = output_tensor.stride(1)

    # Calculate block sizes
    merged_dim_size = num_kv_heads * query_group_size
    merged_block_size = triton.next_power_of_2(merged_dim_size)
    block_size_last = triton.next_power_of_2(last_dim)

    # Calculate grid dimensions
    batch_seq_size = batch_size * seq_len
    grid_dim_0 = batch_seq_size
    grid_dim_1 = triton.cdiv(merged_dim_size, merged_block_size)
    grid_dim_2 = triton.cdiv(last_dim, block_size_last)

    # Compile the kernel
    compiled_func = compile_output(
        data_type=input_tensor.dtype,
        merged_block_size=merged_block_size,
        block_size_last=block_size_last,
    )

    assert compiled_func is not None, f"Compiled function is None"
    # Execute the compiled kernel
    if run_compiled_kernel:
        compiled_func(
            *torch_to_c_types(
                input_tensor,
                output_tensor,
                batch_size,
                seq_len,
                num_kv_heads,
                query_group_size,
                last_dim,
                stride_input_batch,
                stride_input_kv_head,
                stride_input_seq,
                stride_input_group,
                stride_output_batch_seq,
                stride_output_merged,
                grid_dim_0,
                grid_dim_1,
                grid_dim_2,
                torch.cuda.current_stream(output_tensor.device),
            )
        )

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import hashlib
import aiter
import numpy as np
import argparse
import random
import tempfile
import torch
import triton
import triton.language as tl

from triton.tools.compile import compile_kernel, CompileArgs
from jinja2 import Template
from aiter.test_common import perftest, run_perftest
from aiter import pertoken_quant, per_tensor_quant
from csrc.cpp_itfs.torch_utils import torch_to_c_types
from csrc.cpp_itfs.gluon_aot_tools.compile_gluon import (
    compile_gluon_kernel,
    CompileGluonArgs,
)
from csrc.cpp_itfs.utils import (
    compile_template_op,
    AITER_CORE_DIR,
    get_default_func_name,
    not_built,
    run_lib,
)
from aiter.ops.triton.gluon.pa_decode_gluon import (
    paged_attention_decode_v2_gluon_dot_kernel,
    paged_attention_decode_v2_gluon_large_block_dot_kernel,
)
from op_tests.triton_tests.test_pa_decode_gluon import (
    torch_attention_compute,
    create_kv_cache,
    quantize_kv_cache_symmetric,
    quantize_kv_cache_per_tensor,
    shuffle_value_cache_layout,
)


# Global configuration from reference implementation
UNIFORM_RANGE = (-1, 1)
TORCH_TO_TL_DTYPE = {
    aiter.dtypes.fp8: tl.float8e4b8,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}
# os.environ['TRITON_CACHE_DIR'] = '/mnt/raid0/heyanguang/code/fa_triton/aiter/triton_cache'
compile_reduce_kernel_count = 0


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tensor_to_hash(tensor: torch.Tensor, algorithm: str = "md5") -> str:
    """
    Convert a PyTorch tensor to a hash value using the specified algorithm.

    Args:
        tensor (torch.Tensor): Input tensor
        algorithm (str): Hash algorithm, defaults to 'md5',
                        options: 'md5', 'sha1', 'sha256', etc.

    Returns:
        str: Hexadecimal string representation of the hash value
    """
    hash_func = getattr(hashlib, algorithm)()

    # Process tensor data
    tensor_data = tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()

    hash_func.update(tensor_data)
    return hash_func.hexdigest()


def compile_ttgir_with_triton(ttgir_content: str):
    """
    Compile TTGIR (Triton Tensor IR) content to executable artifact.

    This function takes TTGIR string content, writes it to a temporary file,
    and compiles it using the Triton compiler. The temporary file is cleaned up
    after compilation regardless of success or failure.

    Args:
        ttgir_content (str): The TTGIR (Triton Tensor IR) code as a string
                             to be compiled.

    Returns:
        object: The compiled artifact from the Triton compiler.

    Raises:
        Exception: Any exception raised during the compilation process
                  will be propagated to the caller.

    Note:
        This function uses a temporary file to work with the Triton compiler
        which expects file input. The file is automatically deleted after
        compilation to avoid leaving temporary files on disk.
    """
    # Create a temporary file to store the TTGIR content
    # Using NamedTemporaryFile with delete=False to control deletion manually
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ttgir", delete=False
    ) as temp_file:
        # Write TTGIR content to temporary file
        temp_file.write(ttgir_content)
        ttgir_file_path = temp_file.name

    try:
        # Compile the TTGIR file using Triton compiler
        # This converts the intermediate representation to executable code
        compiled_artifact = triton.compiler.compiler.compile(ttgir_file_path)
        return compiled_artifact

    finally:
        # Ensure temporary file is cleaned up even if compilation fails
        # This prevents leaving temporary files on the filesystem
        if os.path.exists(ttgir_file_path):
            os.unlink(ttgir_file_path)


def compile_attention_kernel(
    compute_type: tl.dtype,
    equivalent_query_group_size: int,
    head_size: int,
    kv_block_size: int,
    context_partition_size: int,
    query_quant_mode: int,
    kv_quant_mode: int,
    fp8_max_value: float,
    value_transposed: int,
    is_causal: int,
    md_name: str,
    func_name: str = None,
):
    """Compile the attention kernel for paged attention decode."""
    head_size_pow2 = triton.next_power_of_2(head_size)

    if equivalent_query_group_size < 16:
        equi_query_group_size_pow2 = 16
    else:
        equi_query_group_size_pow2 = triton.next_power_of_2(equivalent_query_group_size)

    if func_name is None:
        func_name = get_default_func_name(
            md_name,
            (
                compute_type,
                equi_query_group_size_pow2,
                head_size_pow2,
                kv_block_size,
                context_partition_size,
                query_quant_mode,
                kv_quant_mode,
                fp8_max_value,
                value_transposed,
                is_causal,
            ),
        )

    global compile_reduce_kernel_count
    compile_reduce_kernel_count += 1

    # if not_built(func_name):
    if compile_reduce_kernel_count == 1:
        kv_compute_block_size = 256
        waves_per_eu = 1
        # Select kernel implementation based on block size
        if kv_block_size > context_partition_size:
            # Use big block kernel for large block sizes
            if value_transposed:
                # Use smaller compute block size for better performance with transposed values
                kv_compute_block_size = 128
        else:
            # Use standard kernel for normal block sizes
            # Configure waves per EU based on query group size
            if equi_query_group_size_pow2 == 64:
                waves_per_eu = 3
            else:
                waves_per_eu = 4

        if compute_type == tl.float8e4b8 or compute_type == tl.bfloat16:
            if query_quant_mode >= 0:
                query_sig = "*fp8e4b8:16"
            else:
                query_sig = "*bf16:16"
            if kv_quant_mode >= 0:
                key_cache_sig = "*fp8e4b8:16"
                value_cache_sig = "*fp8e4b8:16"
            else:
                key_cache_sig = "*bf16:16"
                value_cache_sig = "*bf16:16"
            logits_sig = "*bf16:16"
        elif compute_type == tl.float16:
            if query_quant_mode >= 0:
                query_sig = "*fp8e4b8:16"
            else:
                query_sig = "*fp16:16"
            if kv_quant_mode >= 0:
                key_cache_sig = "*fp8e4b8:16"
                value_cache_sig = "*fp8e4b8:16"
            else:
                key_cache_sig = "*fp16:16"
                value_cache_sig = "*fp16:16"
            logits_sig = "*fp16:16"
        else:
            raise ValueError(f"Unsupported compute type: {compute_type}")
        # Build signature based on kernel parameters (combined from both kernels)
        signature_parts = [
            "*fp32:16",  # exp_sums_ptr
            "*fp32:16",  # max_logits_ptr
            logits_sig,  # logits_ptr
            query_sig,  # query_ptr
            key_cache_sig,  # key_cache_ptr
            value_cache_sig,  # value_cache_ptr
            "*i32:16",  # block_tables_ptr
            "*i32:16",  # context_lengths_ptr
            "fp32:16",  # softmax_scale
            "*fp32:16",  # query_scale
            "*fp32:16",  # key_scale
            "*fp32:16",  # value_scale
            "i32:16",  # stride_max_logits_seq
            "i32:16",  # stride_max_logits_head
            "i32:16",  # stride_max_logits_part
            "i32:16",  # stride_output_seq
            "i32:16",  # stride_output_head
            "i32:16",  # stride_output_part
            "i32:16",  # stride_output_group
            "i32:16",  # stride_query_seq
            "i32:16",  # stride_query_head
            "i32:16",  # stride_key_block
            "i32:16",  # stride_key_head
            "i32:16",  # stride_key_head_split
            "i32:16",  # stride_key_block_elem
            "i32:16",  # stride_value_block
            "i32:16",  # stride_value_head
            "i32:16",  # stride_value_head_size
            "i32:16",  # stride_block_table_seq
            "i32:16",  # query_scale_stride_0
            "i32:16",  # kv_scale_stride_0
            "i32:16",  # kv_scale_stride_1
            "i32:16",  # query_sequence_length
            "i32:16",  # query_group_size
            "i32:16",  # head_size
            "i32:16",  # num_seqs
            "i32:16",  # num_kv_heads
            "i32:16",  # max_context_partition_num
            f"{str(compute_type)}",
            f"{equi_query_group_size_pow2}",
            f"{head_size_pow2}",
            f"{kv_block_size}",
            f"{context_partition_size}",
            f"{kv_compute_block_size}",
            f"{query_quant_mode}",
            f"{kv_quant_mode}",
            f"{fp8_max_value}",
            f"{value_transposed}",
            f"{is_causal}",
        ]
        signature = ",".join(signature_parts)
        gluon_kernel_name = "paged_attention_decode_v2_gluon_dot_kernel"
        if kv_block_size > context_partition_size:
            gluon_kernel_name = "paged_attention_decode_v2_gluon_large_block_dot_kernel"

        compile_args = CompileGluonArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_gluon.py",
            kernel_name=gluon_kernel_name,
            signature=signature,
            grid="num_seqs,num_kv_heads,max_context_partition_num",
            num_warps=4,
            waves_per_eu=waves_per_eu,
            num_stages=1,
            num_ctas=1,
            kpack=1,
            out_name=md_name,
        )
        triton_kernel, output_files = compile_gluon_kernel(compile_args)
        triton_header = None
        triton_source = None
        for output_file in output_files:
            if output_file.suffix == ".h":
                triton_header = output_file
            elif output_file.suffix == ".cpp":
                triton_source = output_file

        with open(
            f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon_aot/pa_decode_attention_kernel.cpp.jinja",
            "r",
        ) as f:
            src_template = Template(f.read())

        return compile_template_op(
            src_template,
            md_name,
            [triton_header],
            [triton_source],
            triton_header=triton_header,
            kernel_name=md_name,
            triton_kernel=triton_kernel,
            func_name=func_name,
        )
    else:
        return run_lib(func_name)


@perftest()
def run_compiled_attention_kernel(
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    softmax_scale: float,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    num_sequences: int,
    num_kv_heads: int,
    max_context_partition_num: int,
    compute_type: str,
    kv_block_size: int,
    context_partition_size: int,
    query_quant_mode: int,
    kv_quant_mode: int,
    fp8_max_value: float,
    value_transposed: int,
    is_causal: int,
    md_name: str,
    func_name: str = None,
):
    """
    Compile and run the compiled attention kernel with perftest timing
    """
    func = compile_attention_kernel(
        compute_type=compute_type,
        equivalent_query_group_size=query_seq_len * query_group_size,
        head_size=head_size,
        kv_block_size=kv_block_size,
        context_partition_size=context_partition_size,
        query_quant_mode=query_quant_mode,
        kv_quant_mode=kv_quant_mode,
        fp8_max_value=fp8_max_value,
        value_transposed=int(value_transposed),
        is_causal=int(is_causal),
        md_name=md_name,
        func_name=func_name,
    )

    # Configure query quantization
    query_scale_stride_0 = 0
    if query_scale is not None:
        assert (
            isinstance(query_scale, torch.Tensor)
            and query_scale.dtype == aiter.dtypes.fp32
        ), f"query_scale tensor only support dtype == {aiter.dtypes.fp32}, but got query_scale.dtype == {query_scale.dtype}"

        if query_scale.numel() == 1:
            pass
        else:
            # Per-token quantization
            assert (
                len(query_scale.shape) == 3
            ), f"Expected 3D query_scale tensor, but got shape {query_scale.shape}"
            assert (
                query_scale.shape[-1] == 1
            ), f"Expected query_scale.shape[-1] == 1, but got query_scale.shape[-1]={query_scale.shape[-1]}"
            query_scale_stride_0 = query_scale.stride(0)

    # Configure KV quantization
    key_scale_stride_0 = 0
    key_scale_stride_1 = 0
    if key_scale is not None and value_scale is not None:
        assert (
            isinstance(key_scale, torch.Tensor) and key_scale.dtype == aiter.dtypes.fp32
        ), f"key_scale tensor only support dtype == {aiter.dtypes.fp32}, but got key_scale.dtype == {key_scale.dtype}"
        assert (
            isinstance(value_scale, torch.Tensor)
            and value_scale.dtype == aiter.dtypes.fp32
        ), f"value_scale tensor only support dtype == {aiter.dtypes.fp32}, but got value_scale.dtype == {value_scale.dtype}"

        if key_scale.numel() == 1:
            pass
        else:
            # Per-token quantization
            assert (
                len(key_scale.shape) == 4
            ), f"Expected 4D key_scale tensor, but got shape {key_scale.shape}"
            assert (
                key_scale.shape[-1] == 1
            ), f"Expected key_scale.shape[-1] == 1, but got key_scale.shape[-1]={key_scale.shape[-1]}"
            key_scale_stride_0 = key_scale.stride(0)
            key_scale_stride_1 = key_scale.stride(1)

        # Validate KV scale shape consistency
        assert (
            key_scale.shape == value_scale.shape
        ), f"Key and value scales must have same shape, but got key: {key_scale.shape}, value: {value_scale.shape}"

    func(
        *torch_to_c_types(
            exp_sums,
            max_logits,
            temporary_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            softmax_scale,
            query_scale,
            key_scale,
            value_scale,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_tables.stride(0),
            query_scale_stride_0,
            key_scale_stride_0,
            key_scale_stride_1,
            query_seq_len,
            query_group_size,
            head_size,
            num_sequences,
            num_kv_heads,
            max_context_partition_num,
            torch.cuda.current_stream(exp_sums.device),
        )
    )


@perftest()
def run_direct_attention_kernel(
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    softmax_scale: float,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    compute_type: tl.dtype,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    kv_block_size: int,
    context_partition_size: int,
    query_quant_mode: int,
    kv_quant_mode: int,
    fp8_max_value: float,
    value_transposed: int,
    is_causal: int,
):
    """
    Directly call the attention kernel from pa_decode_gluon.py with perftest timing
    """
    num_seqs = exp_sums.shape[0]
    num_kv_heads = exp_sums.shape[1]
    max_context_partition_num = exp_sums.shape[2]
    # Configure grid
    grid = (num_seqs, num_kv_heads, max_context_partition_num)

    # Configure query quantization
    query_scale_stride_0 = 0
    if query_scale is not None:
        assert (
            isinstance(query_scale, torch.Tensor)
            and query_scale.dtype == aiter.dtypes.fp32
        ), f"query_scale tensor only support dtype == {aiter.dtypes.fp32}, but got query_scale.dtype == {query_scale.dtype}"

        if query_scale.numel() == 1:
            pass
        else:
            # Per-token quantization
            assert (
                len(query_scale.shape) == 3
            ), f"Expected 3D query_scale tensor, but got shape {query_scale.shape}"
            assert (
                query_scale.shape[-1] == 1
            ), f"Expected query_scale.shape[-1] == 1, but got query_scale.shape[-1]={query_scale.shape[-1]}"
            query_scale_stride_0 = query_scale.stride(0)

    # Configure KV quantization
    key_scale_stride_0 = 0
    key_scale_stride_1 = 0
    if key_scale is not None and value_scale is not None:
        assert (
            isinstance(key_scale, torch.Tensor) and key_scale.dtype == aiter.dtypes.fp32
        ), f"key_scale tensor only support dtype == {aiter.dtypes.fp32}, but got key_scale.dtype == {key_scale.dtype}"
        assert (
            isinstance(value_scale, torch.Tensor)
            and value_scale.dtype == aiter.dtypes.fp32
        ), f"value_scale tensor only support dtype == {aiter.dtypes.fp32}, but got value_scale.dtype == {value_scale.dtype}"

        if key_scale.numel() == 1:
            pass
        else:
            # Per-token quantization
            assert (
                len(key_scale.shape) == 4
            ), f"Expected 4D key_scale tensor, but got shape {key_scale.shape}"
            assert (
                key_scale.shape[-1] == 1
            ), f"Expected key_scale.shape[-1] == 1, but got key_scale.shape[-1]={key_scale.shape[-1]}"
            key_scale_stride_0 = key_scale.stride(0)
            key_scale_stride_1 = key_scale.stride(1)

        # Validate KV scale shape consistency
        assert (
            key_scale.shape == value_scale.shape
        ), f"Key and value scales must have same shape, but got key: {key_scale.shape}, value: {value_scale.shape}"

    # Debug compilation path - kept for development and debugging purposes
    # if 1:
    if 0:
        ttgir_file_path = "/mnt/raid0/heyanguang/code/fa_triton/aiter/paged_attention_decode_v2_gluon_dot_kernel.ttgir"
        with open(ttgir_file_path, "r") as f:
            ttgir_content = f.read()
        try:
            compiled_kernel = compile_ttgir_with_triton(ttgir_content)
            compiled_kernel[grid](
                exp_sums,
                max_logits,
                temporary_output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                softmax_scale,
                query_scale,
                key_scale,
                value_scale,
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                temporary_output.stride(0),
                temporary_output.stride(1),
                temporary_output.stride(2),
                temporary_output.stride(3),
                query.stride(0),
                query.stride(1),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                key_cache.stride(3),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                block_tables.stride(0),
                query_scale_stride_0,
                key_scale_stride_0,
                key_scale_stride_1,
                grid[0],
                grid[1],
                grid[2],
            )
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        # Production path - select and launch appropriate kernel
        equi_query_group_size = query_seq_len * query_group_size
        if equi_query_group_size < 16:
            equi_query_group_size_pow2 = 16
        else:
            equi_query_group_size_pow2 = triton.next_power_of_2(equi_query_group_size)
        kv_compute_block_size = 256
        waves_per_eu = 1

        # Use standard kernel
        kernel = paged_attention_decode_v2_gluon_dot_kernel
        # Select kernel implementation based on block size
        if kv_block_size > context_partition_size:
            # Use large block kernel
            kernel = paged_attention_decode_v2_gluon_large_block_dot_kernel
            if value_transposed:
                # Use smaller compute block size for better performance with transposed values
                kv_compute_block_size = 128
        else:
            # Use standard kernel for normal block sizes
            # Configure waves per EU based on query group size
            if equi_query_group_size_pow2 == 64:
                waves_per_eu = 3
            else:
                waves_per_eu = 4

        # Launch the kernel directly
        kernel[grid](
            exp_sums,
            max_logits,
            temporary_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            softmax_scale,
            query_scale,
            key_scale,
            value_scale,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_tables.stride(0),
            query_scale_stride_0,
            key_scale_stride_0,
            key_scale_stride_1,
            query_seq_len=query_seq_len,
            query_group_size_original=query_group_size,
            head_size=head_size,
            num_seqs=num_seqs,
            num_kv_heads=num_kv_heads,
            max_context_partition_num=max_context_partition_num,
            COMPUTE_TYPE=compute_type,
            QUERY_GROUP_SIZE_POW2=equi_query_group_size_pow2,
            HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
            KV_BLOCK_SIZE=kv_block_size,
            CONTEXT_PARTITION_SIZE=context_partition_size,
            KV_COMPUTE_BLOCK_SIZE=kv_compute_block_size,
            QUERY_QUANT_MODE=query_quant_mode,
            KV_QUANT_MODE=kv_quant_mode,
            FP8_MAX_VALUE=fp8_max_value,
            VALUE_TRANSPOSED=value_transposed,
            IS_CAUSAL=is_causal,
            waves_per_eu=waves_per_eu,
            num_stages=1,
        )


def test_attention_kernel(kernel_type: str = "compiled"):
    """Test the attention kernel with provided parameters.

    Args:
        kernel_type: Type of kernel to test - "compiled" or "direct"
    """
    print(f"\n=== Testing Attention Kernel (Type: {kernel_type}) ===")
    setup_seed(123)
    device = "cuda:0"
    max_context_length = 16384
    compute_type = aiter.dtypes.bf16
    quant_q, quant_kv = [False, False]
    # compute_type = aiter.dtypes.fp8
    # quant_q, quant_kv = [True, True]
    data_type = compute_type
    # quant_mode = "per_token"
    quant_mode = "per_tensor"
    if compute_type == aiter.dtypes.fp8:
        data_type = aiter.dtypes.bf16
    compute_type = TORCH_TO_TL_DTYPE[compute_type]

    context_length = 2048
    batch_size = 128
    # batch_size = 128 * 8
    num_heads = (8, 1)
    # num_heads = (64, 8)
    # num_heads = (16, 1)
    # num_heads = (64, 4)
    head_size = 128
    block_size = 16
    query_length = 1
    query_quant_mode = -1
    kv_quant_mode = -1
    context_partition_size = 256
    trans_v = False
    kv_varlen = False

    # ==================== FP8 CONFIGURATION ====================
    fp8_max_value = 1.0
    if quant_kv:
        fp8_max_value = torch.finfo(aiter.dtypes.fp8).max

    # Derived parameters
    num_query_heads, num_kv_heads = num_heads
    query_seq_len = query_length
    query_group_size = num_query_heads // num_kv_heads
    kv_block_size = block_size

    # Set quantization modes based on quant_mode parameter
    if quant_q:
        query_quant_mode = (
            1 if quant_mode == "per_token" else 0
        )  # 1 for per-token, 0 for per-tensor
    if quant_kv:
        kv_quant_mode = (
            1 if quant_mode == "per_token" else 0
        )  # 1 for per-token, 0 for per-tensor
    value_transposed = 1 if trans_v else 0

    # Determine if causal masking is needed
    is_causal = query_length > 1
    num_sequences = batch_size
    max_context_partition_num = (
        context_length + context_partition_size - 1
    ) // context_partition_size
    softmax_scale = 1.0 / (head_size**0.5)

    # Calculate block table dimensions
    max_blocks_per_sequence = (max_context_length + block_size - 1) // block_size
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = (context_length + block_size - 1) // block_size

    # Create intermediate tensors for attention computation
    equivalent_query_group_size = query_length * (num_query_heads // num_kv_heads)
    intermediate_shape = (
        num_sequences,
        num_kv_heads,
        max_context_partition_num,
        equivalent_query_group_size,
    )

    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
    max_logits = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
    temporary_output = torch.empty(
        *intermediate_shape, head_size, dtype=data_type, device=device
    )

    num_query_heads, num_kv_heads = query_group_size, 1
    # Create query tensor following reference approach
    total_queries = batch_size * query_length
    qkv_tensor = torch.randn(
        total_queries,
        num_query_heads + 2 * num_kv_heads,
        head_size,
        dtype=data_type,
        device=device,
    )
    query, key, value = torch.split(
        qkv_tensor, [num_query_heads, num_kv_heads, num_kv_heads], dim=1
    )
    query.uniform_(*UNIFORM_RANGE)

    # Create key and value caches
    key_caches, value_caches = create_kv_cache(
        total_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        "auto",
        data_type,
        0,  # seed
        device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    num_query_heads, num_kv_heads = num_heads
    query = query.repeat(1, num_kv_heads, 1).contiguous()
    key_cache = key_cache.repeat(1, num_kv_heads, 1, 1, 1).contiguous()
    value_cache = value_cache.repeat(1, num_kv_heads, 1, 1).contiguous()

    # Quantization based on mode
    if quant_q:
        if quant_mode == "per_token":
            # Per-token quantization for query
            quantized_query, query_scale_factors = pertoken_quant(
                query, quant_dtype=aiter.dtypes.fp8
            )
        else:
            # Per-tensor quantization for query
            quantized_query, query_scale_factors = per_tensor_quant(
                query, quant_dtype=aiter.dtypes.fp8
            )
    else:
        quantized_query = query
        query_scale_factors = None

    if quant_kv:
        if quant_mode == "per_token":
            # Per-token quantization for KV cache
            (
                quantized_keys,
                key_scale_factors_flat,
                quantized_values,
                value_scale_factors_flat,
                key_scale_original,
                value_scale_original,
            ) = quantize_kv_cache_symmetric(
                key_cache, value_cache, quant_dtype=aiter.dtypes.fp8
            )
        else:
            # Per-tensor quantization for KV cache
            (
                quantized_keys,
                key_scale_factors_flat,
                quantized_values,
                value_scale_factors_flat,
                key_scale_original,
                value_scale_original,
            ) = quantize_kv_cache_per_tensor(
                key_cache, value_cache, quant_dtype=aiter.dtypes.fp8
            )
    else:
        quantized_keys = key_cache
        quantized_values = value_cache
        key_scale_factors_flat = None
        value_scale_factors_flat = None
        key_scale_original = None
        value_scale_original = None

    # Reshape query for Gluon kernel format
    quantized_query_gluon = quantized_query
    query_scale_gluon = query_scale_factors
    if query_length > 1:
        quantized_query_gluon = quantized_query.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        quantized_query_gluon = quantized_query_gluon.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )

        if (
            query_scale_factors is not None and len(query_scale_factors.shape) > 1
        ):  # per-token quantization
            query_scale_gluon = query_scale_factors.reshape(
                batch_size, query_length, num_kv_heads, query_group_size, 1
            )
            query_scale_gluon = query_scale_gluon.transpose(1, 2).reshape(
                batch_size, num_kv_heads * query_length * query_group_size, 1
            )

    # Transpose value cache if required
    if trans_v:
        quantized_values = shuffle_value_cache_layout(quantized_values)

    # Create sequence lengths and block tables
    if kv_varlen:
        kv_len_list = [
            random.randint(query_length, context_length) for _ in range(batch_size)
        ]
    else:
        kv_len_list = [context_length] * batch_size
    context_lengths = torch.tensor(kv_len_list, dtype=torch.int32, device=device)

    block_tables_list = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)
        ]
        block_tables_list.append(block_table)
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    # Assign to variables used by the kernel
    query = quantized_query_gluon
    key_cache = quantized_keys
    value_cache = quantized_values
    query_scale = query_scale_gluon
    key_scale = key_scale_original
    value_scale = value_scale_original

    # Execute kernel based on selected type
    if kernel_type == "compiled":
        # Compile and run the compiled kernel
        print("\n=== Running Compiled Kernel ===")
        _, compiled_time = run_compiled_attention_kernel(
            exp_sums,
            max_logits,
            temporary_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            softmax_scale,
            query_scale,
            key_scale,
            value_scale,
            query_seq_len,
            query_group_size,
            head_size,
            num_sequences,
            num_kv_heads,
            max_context_partition_num,
            compute_type=compute_type,
            kv_block_size=kv_block_size,
            context_partition_size=context_partition_size,
            query_quant_mode=query_quant_mode,
            kv_quant_mode=kv_quant_mode,
            fp8_max_value=fp8_max_value,
            value_transposed=value_transposed,
            is_causal=is_causal,
            md_name="pa_decode_attention_kernel",
        )
        print(f"Compiled kernel execution time: {compiled_time:.2f} us/iter")
    elif kernel_type == "direct":
        # Directly call the kernel from pa_decode_gluon.py
        print("\n=== Running Direct Kernel ===")
        _, direct_time = run_direct_attention_kernel(
            exp_sums,
            max_logits,
            temporary_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            softmax_scale,
            query_scale,
            key_scale,
            value_scale,
            compute_type,
            query_seq_len,
            query_group_size,
            head_size,
            kv_block_size,
            context_partition_size,
            query_quant_mode,
            kv_quant_mode,
            fp8_max_value,
            value_transposed,
            is_causal,
        )
        print(f"Direct kernel execution time: {direct_time:.2f} us/iter")
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Check for NaN values
    exp_sums_nan_cnt = torch.isnan(exp_sums).sum().item()
    max_logits_nan_cnt = torch.isnan(max_logits).sum().item()
    temporary_output_nan_cnt = torch.isnan(temporary_output).sum().item()

    print("\n=== Checking Results ===")
    print(f"exp_sums NaN count: {exp_sums_nan_cnt}")
    print(f"max_logits NaN count: {max_logits_nan_cnt}")
    print(f"temporary_output NaN count: {temporary_output_nan_cnt}")

    # Compare with torch_attention_compute reference
    print("\n=== Comparing with Reference Implementation ===")

    # Run reference implementation
    ref_exp_sums, ref_max_logits, ref_temporary_output = torch_attention_compute(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lengths=context_lengths,
        softmax_scale=softmax_scale,
        q_seq_len=query_seq_len,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        output_dtype=torch.bfloat16,
        kv_block_size=kv_block_size,
        context_partition_size=context_partition_size,
        is_causal=bool(is_causal),
    )

    # Compare exp_sums
    print("\n--- Comparing exp_sums ---")
    diff_exp_sums = (exp_sums - ref_exp_sums).abs()
    max_diff_exp_sums = diff_exp_sums.max().item()
    mean_diff_exp_sums = diff_exp_sums.mean().item()
    print(f"Max difference: {max_diff_exp_sums:.6e}")
    print(f"Mean difference: {mean_diff_exp_sums:.6e}")

    # Compare max_logits
    print("\n--- Comparing max_logits ---")
    diff_max_logits = (max_logits - ref_max_logits).abs()
    max_diff_max_logits = diff_max_logits.max().item()
    mean_diff_max_logits = diff_max_logits.mean().item()
    print(f"Max difference: {max_diff_max_logits:.6e}")
    print(f"Mean difference: {mean_diff_max_logits:.6e}")

    # Compare temporary_output
    print("\n--- Comparing temporary_output ---")
    diff_temporary_output = (temporary_output - ref_temporary_output).abs()
    max_diff_temporary_output = diff_temporary_output.max().item()
    mean_diff_temporary_output = diff_temporary_output.mean().item()
    print(f"Max difference: {max_diff_temporary_output:.6e}")
    print(f"Mean difference: {mean_diff_temporary_output:.6e}")

    # Detailed error analysis for temporary_output (largest tensor)
    if max_diff_temporary_output > 1e-4:
        print("\n=== Detailed Error Analysis (temporary_output) ===")
        # Find top 5 differences
        flat_diff = diff_temporary_output.flatten()
        top_k = 5
        top_k_indices = torch.topk(flat_diff, top_k).indices

        print(f"Top {top_k} differences:")
        for i in range(top_k):
            idx = top_k_indices[i]
            orig_idx = np.unravel_index(idx.cpu().numpy(), temporary_output.shape)
            print(
                f"  Position {orig_idx}: kernel={temporary_output[orig_idx].item():.6f}, ref={ref_temporary_output[orig_idx].item():.6f}, diff={flat_diff[idx].item():.6e}"
            )

    # Test results
    tolerance = 5e-3
    all_passed = True

    print("\n=== Test Results ===")
    if max_diff_exp_sums < tolerance:
        print(
            f"✅ exp_sums TEST PASSED: Max difference ({max_diff_exp_sums:.6e}) < tolerance ({tolerance})"
        )
    else:
        print(
            f"❌ exp_sums TEST FAILED: Max difference ({max_diff_exp_sums:.6e}) >= tolerance ({tolerance})"
        )
        all_passed = False

    if max_diff_max_logits < tolerance:
        print(
            f"✅ max_logits TEST PASSED: Max difference ({max_diff_max_logits:.6e}) < tolerance ({tolerance})"
        )
    else:
        print(
            f"❌ max_logits TEST FAILED: Max difference ({max_diff_max_logits:.6e}) >= tolerance ({tolerance})"
        )
        all_passed = False

    if max_diff_temporary_output < tolerance:
        print(
            f"✅ temporary_output TEST PASSED: Max difference ({max_diff_temporary_output:.6e}) < tolerance ({tolerance})"
        )
    else:
        print(
            f"❌ temporary_output TEST FAILED: Max difference ({max_diff_temporary_output:.6e}) >= tolerance ({tolerance})"
        )
        all_passed = False

    # MD5 hashes for verification
    ref_exp_sums_md5 = tensor_to_hash(ref_exp_sums)
    ref_max_logits_md5 = tensor_to_hash(ref_max_logits)
    ref_temporary_output_md5 = tensor_to_hash(ref_temporary_output)
    exp_sums_md5 = tensor_to_hash(exp_sums)
    # exp_sums_sha256 = tensor_to_hash(exp_sums, 'sha256')
    max_logits_md5 = tensor_to_hash(max_logits)
    temporary_output_md5 = tensor_to_hash(temporary_output)
    print(f"Reference exp_sums MD5: {ref_exp_sums_md5}")
    print(f"Reference max_logits MD5: {ref_max_logits_md5}")
    print(f"Reference temporary_output MD5: {ref_temporary_output_md5}")
    print(f"exp_sums MD5: {exp_sums_md5}")
    # print(f"exp_sums SHA256: {exp_sums_sha256}")
    print(f"max_logits MD5: {max_logits_md5}")
    print(f"temporary_output MD5: {temporary_output_md5}")

    # Test result
    if all_passed:
        print(
            f"\n✅ OVERALL TEST PASSED: All comparisons within tolerance and no NaN values detected"
        )
        return True
    else:
        print(
            f"\n❌ OVERALL TEST FAILED: Some comparisons failed or NaN values detected"
        )
        return False


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Test paged attention kernel")
    parser.add_argument(
        "--kernel-type",
        type=str,
        choices=["compiled", "direct"],
        default="compiled",
        help="Type of kernel to test: 'compiled' (default) or 'direct'",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=101,
        help="Number of iterations for performance testing",
    )
    parser.add_argument(
        "--num-warmup", type=int, default=2, help="Number of warmup iterations"
    )

    args = parser.parse_args()

    # Run the test
    result = test_attention_kernel(kernel_type=args.kernel_type)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()

import os
import sys
import hashlib
import time
import logging
from pathlib import Path
from unittest import result
from jinja2 import Template
import triton
import functools
import aiter
import torch
import triton
import triton.language as tl

from triton.tools.compile import compile_kernel, CompileArgs
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
    mp_lock,
)

MD_NAME = "pa_decode_attention_reduce_kernel"


def parse_version(version_str):
    """Parse version string into comparable tuple format, handling possible development version suffixes"""
    # Remove potential suffixes like .dev, +git etc.
    version_str = version_str.split("+")[0].split("-")[0]

    # Split version number and convert to integers
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break

    return tuple(parts)


TRITON_VERSION = parse_version(triton.__version__)
logger = logging.getLogger("aiter")


def compile(
    compute_type: tl.dtype,
    equivalent_query_group_size: int,
    head_size: int,
    kv_block_size: int,
    max_context_partition_num: int,
    context_partition_size: int,
    query_quant_mode: int,
    kv_quant_mode: int,
    fp8_max_value: float,
    value_transposed: int,
    is_causal: int,
    func_name: str = None,
):
    """Compile the combined attention and reduce kernel for paged attention decode."""
    max_context_partition_num_pow2 = triton.next_power_of_2(max_context_partition_num)
    head_size_pow2 = triton.next_power_of_2(head_size)

    if equivalent_query_group_size < 16:
        equi_query_group_size_pow2 = 16
    else:
        equi_query_group_size_pow2 = triton.next_power_of_2(equivalent_query_group_size)

    if func_name is None:
        func_name = get_default_func_name(
            MD_NAME,
            (
                compute_type,
                equi_query_group_size_pow2,
                head_size_pow2,
                kv_block_size,
                max_context_partition_num_pow2,
                context_partition_size,
                query_quant_mode,
                kv_quant_mode,
                fp8_max_value,
                value_transposed,
                is_causal,
            ),
        )

    if not_built(func_name):
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
            output_sig = "*bf16:16"
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
            output_sig = "*fp16:16"
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
        gluon_kernel_name = "paged_attention_decode_v2_gluon_fp8"
        if kv_block_size > context_partition_size:
            gluon_kernel_name = "paged_attention_decode_v2_gluon_large_block_fp8"

        current_dir = os.getcwd()
        aot_file_dir = f"{current_dir}/{func_name}"
        os.makedirs(aot_file_dir, exist_ok=True)

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
            out_path=Path(aot_file_dir + f"/{MD_NAME}_stage1"),
            out_name=f"{MD_NAME}_stage1",
        )

        # Compile reduce kernel separately
        reduce_signature_parts = [
            output_sig,  # output_ptr
            "*fp32:16",  # exp_sums_ptr
            "*fp32:16",  # max_logits_ptr
            logits_sig,  # logits_ptr
            "*i32:16",  # context_lengths_ptr
            "i32:16",  # stride_output_seq
            "i32:16",  # stride_output_head
            "i32:16",  # stride_exp_sums_seq
            "i32:16",  # stride_exp_sums_head
            "i32:16",  # stride_exp_sums_part
            "i32:16",  # stride_logits_seq
            "i32:16",  # stride_logits_head
            "i32:16",  # stride_logits_part
            "i32:16",  # stride_logits_group
            "i32:16",  # query_group_size
            "i32:16",  # head_size
            "i32:16",  # num_seqs
            "i32:16",  # num_kv_heads
            f"{equi_query_group_size_pow2}",
            f"{head_size_pow2}",
            f"{max_context_partition_num_pow2}",
            f"{context_partition_size}",
        ]
        reduce_signature = ",".join(reduce_signature_parts)

        reduce_kernel_name = "paged_attention_decode_v2_reduce_kernel_triton34"
        if TRITON_VERSION > (3, 4, 0):
            reduce_kernel_name = "paged_attention_decode_v2_reduce_kernel"

        reduce_compile_args = CompileArgs(
            path=f"{AITER_CORE_DIR}/aiter/ops/triton/gluon/pa_decode_gluon.py",
            kernel_name=reduce_kernel_name,
            signature=reduce_signature,
            grid="num_seqs,num_kv_heads,1",
            num_warps=4,
            num_stages=2,
            out_path=Path(aot_file_dir + f"/{MD_NAME}_stage2"),
            out_name=f"{MD_NAME}_stage2",
        )

        # Create lock directory and lock path
        lock_path = os.path.join(aot_file_dir, "lock_triton_aot_compile")
        start_ts = time.perf_counter()

        def main_func():
            """Main compilation function protected by multiprocessing lock."""
            logger.info(f"start build {func_name}")
            triton_kernel1, output_files1 = compile_gluon_kernel(compile_args)
            triton_kernel2, output_files2 = compile_kernel(reduce_compile_args)
            return triton_kernel1, output_files1, triton_kernel2, output_files2

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
            triton_kernel1, output_files1, triton_kernel2, output_files2 = (
                main_func_result
            )
            # Combine output files
            triton_header1 = None
            triton_source1 = None
            triton_header2 = None
            triton_source2 = None
            for output_file in output_files1:
                if output_file.suffix == ".h":
                    triton_header1 = output_file
                elif output_file.suffix == ".cpp":
                    triton_source1 = output_file
            for output_file in output_files2:
                if output_file.suffix == ".h":
                    triton_header2 = output_file
                elif output_file.suffix == ".cpp":
                    triton_source2 = output_file

            with open(
                f"{AITER_CORE_DIR}/csrc/cpp_itfs/pa_gluon_aot/pa_decode_attention_reduce_kernel.cpp.jinja",
                "r",
            ) as f:
                src_template = Template(f.read())

            return compile_template_op(
                src_template,
                MD_NAME,
                [triton_header1, triton_header2],
                [triton_source1, triton_source2],
                triton_header1=triton_header1,
                triton_header2=triton_header2,
                kernel_name=MD_NAME,
                triton_kernel1=triton_kernel1,
                triton_kernel2=triton_kernel2,
                func_name=func_name,
            )
        else:
            return None
    else:
        return run_lib(func_name)


def pa_decode_gluon_aot(
    output: torch.Tensor,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    query: torch.Tensor,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_sequence_length: int,
    max_context_length: int,
    context_partition_size: int,
    compute_type: tl.dtype,
    query_scale: torch.Tensor,  # [num_seqs, num_kv_heads * query_group_size, 1]
    key_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    max_logits: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    temporary_output: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
    alibi_slopes: torch.Tensor = None,
    run_compiled_kernel: bool = True,
) -> dict:
    # Extract tensor dimensions
    num_sequences = query.shape[0]
    num_query_heads_total = query.shape[1]
    num_query_heads_total = num_query_heads_total // query_sequence_length
    num_kv_heads = key_cache.shape[1]
    max_context_partition_num = int(
        (max_context_length + context_partition_size - 1) // context_partition_size
    )
    head_size = query.shape[-1]
    kv_block_size = key_cache.shape[-2]
    query_group_size = num_query_heads_total // num_kv_heads

    # Calculate equivalent group sizes for kernel configuration
    equivalent_query_group_size = query_sequence_length * query_group_size

    # Determine if causal masking is needed
    is_causal = query_sequence_length > 1

    # Calculate elements per 16B load based on data type
    kv_elements_per_16b = 16 // key_cache.dtype.itemsize

    assert (
        query_sequence_length <= 4
    ), f"query_sequence_length == {query_sequence_length} exceeds maximum of 4"
    # Validate input params constraint
    assert query.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"query tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got query.dtype == {query.dtype}"
    assert key_cache.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"key_cache tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got key_cache.dtype == {key_cache.dtype}"
    assert value_cache.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"value_cache tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got value_cache.dtype == {value_cache.dtype}"
    assert output.dtype in [
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"output tensor only support dtype in [{aiter.dtypes.bf16, aiter.dtypes.fp16}], but got output.dtype == {output.dtype}"
    assert (
        equivalent_query_group_size <= 64
    ), f"equivalent_query_group_size={equivalent_query_group_size} exceeds maximum of 64"
    assert kv_block_size in [
        16,
        64,
        1024,
    ], f"kv_block_size == {kv_block_size} not in [16, 64, 1024]"
    assert (
        len(output.shape) == 3
    ), f"Expected 3D output tensor, but got shape {output.shape}"
    assert (
        len(query.shape) == 3
    ), f"Expected 3D query tensor, but got shape {query.shape}"
    assert (
        len(key_cache.shape) == 5
    ), f"Expected 5D key_cache tensor, but got shape {key_cache.shape}"

    # ==================== QUANTIZATION MODE CONFIGURATION ====================
    query_scale_stride_0 = 0
    key_scale_stride_0 = 0
    key_scale_stride_1 = 0
    query_quant_mode = -1
    kv_quant_mode = -1

    # Configure query quantization
    if query_scale is not None:
        assert (
            isinstance(query_scale, torch.Tensor)
            and query_scale.dtype == aiter.dtypes.fp32
        ), f"query_scale tensor only support dtype == {aiter.dtypes.fp32}, but got query_scale.dtype == {query_scale.dtype}"

        if query_scale.numel() == 1:
            # Per-tensor quantization
            query_quant_mode = 0
        else:
            # Per-token quantization
            assert (
                len(query_scale.shape) == 3
            ), f"Expected 3D query_scale tensor, but got shape {query_scale.shape}"
            assert (
                query_scale.shape[-1] == 1
            ), f"Expected query_scale.shape[-1] == 1, but got query_scale.shape[-1]={query_scale.shape[-1]}"
            query_quant_mode = 1
            query_scale_stride_0 = query_scale.stride(0)

    # Configure KV quantization
    if key_scale is not None and value_scale is not None:
        assert (
            isinstance(key_scale, torch.Tensor) and key_scale.dtype == aiter.dtypes.fp32
        ), f"key_scale tensor only support dtype == {aiter.dtypes.fp32}, but got key_scale.dtype == {key_scale.dtype}"
        assert (
            isinstance(value_scale, torch.Tensor)
            and value_scale.dtype == aiter.dtypes.fp32
        ), f"value_scale tensor only support dtype == {aiter.dtypes.fp32}, but got value_scale.dtype == {value_scale.dtype}"

        if key_scale.numel() == 1:
            # Per-tensor quantization
            kv_quant_mode = 0
        else:
            # Per-token quantization
            assert (
                len(key_scale.shape) == 4
            ), f"Expected 4D key_scale tensor, but got shape {key_scale.shape}"
            assert (
                key_scale.shape[-1] == 1
            ), f"Expected key_scale.shape[-1] == 1, but got key_scale.shape[-1]={key_scale.shape[-1]}"
            kv_quant_mode = 1
            key_scale_stride_0 = key_scale.stride(0)
            key_scale_stride_1 = key_scale.stride(1)

        # Validate KV scale shape consistency
        assert (
            key_scale.shape == value_scale.shape
        ), f"Key and value scales must have same shape, but got key: {key_scale.shape}, value: {value_scale.shape}"

    # ==================== VALUE CACHE LAYOUT DETECTION ====================
    value_transposed = False
    if len(value_cache.shape) == 5:
        value_transposed = True
    elif len(value_cache.shape) == 4:
        value_transposed = False
    else:
        raise RuntimeError(f"Unsupported value cache shape: {value_cache.shape}")

    # ==================== FP8 CONFIGURATION ====================
    fp8_max_value = 1.0
    if value_cache.dtype == aiter.dtypes.fp8:
        fp8_max_value = torch.finfo(aiter.dtypes.fp8).max

    # Compile the combined attention and reduce kernel
    combined_func = compile(
        compute_type=compute_type,
        equivalent_query_group_size=equivalent_query_group_size,
        head_size=head_size,
        kv_block_size=kv_block_size,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        query_quant_mode=query_quant_mode,
        kv_quant_mode=kv_quant_mode,
        fp8_max_value=fp8_max_value,
        value_transposed=int(value_transposed),
        is_causal=int(is_causal),
    )

    # Execute the combined kernel
    if run_compiled_kernel and combined_func is not None:
        combined_func(
            *torch_to_c_types(
                output,
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
                output.stride(0),
                output.stride(1),
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
                num_sequences,
                num_kv_heads,
                max_context_partition_num,
                query_sequence_length,
                query_group_size,
                equivalent_query_group_size,
                head_size,
                torch.cuda.current_stream(output.device),
            )
        )

import os
import time
import shutil
import subprocess
from pathlib import Path
from jinja2 import Template
import torch
import aiter
import aiter.ops.triton.utils._triton.arch_info as arch_info
import triton
import triton.language as tl

GLUON_AOT_COMPILE_ENABLED = True
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    print(
        "Warning: triton.experimental.gluon or triton.experimental.gluon.language not exists, pa_decode_gluon_aot cannot use compile mode!"
    )
    GLUON_AOT_COMPILE_ENABLED = False

try:
    from triton.tools.compile import compile_kernel, CompileArgs
except ImportError:
    print("Warning: compile_kernel or CompileArgs is not in triton.tools.compile!")

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
from csrc.cpp_itfs.pa_gluon_aot.transpose_query_output_gluon_aot import (
    transpose_query_gluon_aot,
    transpose_output_gluon_aot,
)
from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version

MD_NAME = "pa_decode_attention_reduce_kernel"


def clean_directory_except_so(directory_path):
    """
    Delete all files and folders in the specified directory except for .so files.

    Args:
        directory_path (str): Path to the directory to clean
    """
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Check if the path is actually a directory
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        return

    # Walk through all files and directories
    for root, dirs, files in os.walk(directory_path, topdown=False):
        # Process files first
        for file in files:
            file_path = os.path.join(root, file)
            # Skip .so files
            if not file.endswith(".so"):
                try:
                    os.remove(file_path)
                    # print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

        # Process directories (after files have been processed)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if directory contains any .so files
            has_so_files = False
            try:
                for item in os.listdir(dir_path):
                    if item.endswith(".so"):
                        has_so_files = True
                        break
            except Exception as e:
                print(f"Error accessing directory {dir_path}: {e}")
                continue

            # Only delete directory if it doesn't contain .so files
            if not has_so_files:
                try:
                    shutil.rmtree(dir_path)
                    # print(f"Deleted directory: {dir_path}")
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {e}")


def compile(
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
    use_sinks: int,
    cdna_version: int,
    func_name: str = None,
):
    """Compile the combined attention and reduce kernel for paged attention decode."""
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
                context_partition_size,
                query_quant_mode,
                kv_quant_mode,
                fp8_max_value,
                value_transposed,
                is_causal,
                use_sinks,
                cdna_version,
            ),
        )

    if not_built(func_name):
        if not GLUON_AOT_COMPILE_ENABLED:
            raise RuntimeError(
                "This version triton is not support gluon aot compile, please upgrade to 3.5.0 or higher!"
            )

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
            "i32:16",  # query_length
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
            f"{cdna_version}",
        ]
        signature = ",".join(signature_parts)
        gluon_kernel_name = "paged_attention_decode_v2_gluon_dot_kernel"
        if kv_block_size > context_partition_size:
            gluon_kernel_name = "paged_attention_decode_v2_gluon_large_block_dot_kernel"

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
            "*fp32:16",  # sinks_ptr
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
            f"{context_partition_size}",
            f"{use_sinks}",
        ]
        reduce_signature = ",".join(reduce_signature_parts)
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
            # return triton_kernel1, output_files1, triton_kernel2, output_files2
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

            compiled_func = compile_template_op(
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
            print("Cleaning aot temporary files completed!")
            print(f"Cleaning aiter build cache directory: {BUILD_DIR}/{func_name}")
            clean_directory_except_so(f"{BUILD_DIR}/{func_name}")
            print(
                "Cleaning aiter build cache directory completed, only *.so files are left!"
            )
            return main_func_result
        else:
            logger.info(f"{func_name} already built by another process")
            assert not not_built(func_name)
            return run_lib(func_name)
    else:
        return run_lib(func_name)


def pa_decode_gluon_aot(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    output_gluon: torch.Tensor,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    query_gluon: torch.Tensor,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    query_scale_gluon: torch.Tensor,  # [num_seqs, num_kv_heads * query_length * query_group_size, 1] or [1]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    max_context_length: int,
    context_partition_size: int,
    compute_type: tl.dtype,
    query_scale: torch.Tensor,  # [num_seqs * query_length, num_query_heads, 1] or [1]
    key_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    max_logits: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    temporary_output: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
    alibi_slopes: torch.Tensor = None,
    run_compiled_kernel: bool = True,
    sinks: torch.Tensor = None,
) -> None:
    """
    Paged Attention Decode with FP8/BF16/FP16 Support.

    Implements the attention mechanism for transformer decoding with paged KV caches,
    supporting various quantization schemes and data types. This function performs
    attention computation in two phases: a partitioned attention kernel followed
    by a reduction kernel.

    Parameters
    ----------
    output : torch.Tensor
        Output tensor for final attention results.
        - Shape: [num_seqs * query_length, num_query_heads, head_size]
        - Dtype: torch.bfloat16, torch.float16

    output_gluon : torch.Tensor
        Intermediate output tensor in gluon layout for internal computation.
        - Shape: [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
        - Dtype: torch.bfloat16, torch.float16 (same as output)

    query : torch.Tensor
        Input query tensor in standard layout.
        - Shape: [num_seqs * query_length, num_query_heads, head_size]
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

    query_gluon : torch.Tensor
        Query tensor in gluon layout for internal computation.
        - Shape: [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16 (same as query)

    query_scale_gluon : torch.Tensor
        Quantization scales for query in gluon layout.
        - Shape: [1] (per-tensor) or [num_seqs, num_kv_heads * query_length * query_group_size, 1] (per-token)
        - Dtype: torch.float32

    key_cache : torch.Tensor
        Paged key cache in block layout with interleaved head dimension.
        - Shape: [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
          where x = 16 // dtype.itemsize (e.g., x=16 for fp8, x=8 for bf16/fp16)
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

    value_cache : torch.Tensor
        Paged value cache in block layout. Supports two layouts:
        - Non-transposed shape: [num_blocks, num_kv_heads, head_size, kv_block_size]
        - Transposed shape: [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
          where x = 16 // dtype.itemsize
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

    context_lengths : torch.Tensor
        Current context lengths (KV cache lengths) for each sequence.
        - Shape: [num_seqs]
        - Dtype: torch.int32

    block_tables : torch.Tensor
        Mapping from sequences to physical cache block indices.
        - Shape: [num_seqs, max_num_blocks_per_seq]
        - Dtype: torch.int32

    softmax_scale : float
        Scaling factor for attention scores, typically 1/sqrt(head_size).

    query_length : int
        Length of query sequences. Must be <= 4.

    max_context_length : int
        Maximum sequence length supported in the KV cache.

    context_partition_size : int
        Size of each context partition for partitioned attention computation.

    compute_type : tl.dtype
        Triton data type for computation.
        - Supported: tl.float8e4b8, tl.bfloat16, tl.float16

    query_scale : torch.Tensor
        Quantization scales for queries in standard layout. Required for FP8 queries.
        - Shape: [1] (per-tensor) or [num_seqs * query_length, num_query_heads, 1] (per-token)
        - Dtype: torch.float32

    key_scale : torch.Tensor
        Quantization scales for keys. Required for FP8 keys.
        - Shape: [1] (per-tensor) or [num_blocks, num_kv_heads, kv_block_size, 1] (per-token)
        - Dtype: torch.float32

    value_scale : torch.Tensor
        Quantization scales for values. Must have same shape as key_scale.
        - Shape: [1] (per-tensor) or [num_blocks, num_kv_heads, kv_block_size, 1] (per-token)
        - Dtype: torch.float32

    exp_sums : torch.Tensor
        Buffer for exponential sums used in online softmax computation.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
          where max_context_partition_num = ceil(max_context_length / context_partition_size)
        - Dtype: torch.float32

    max_logits : torch.Tensor
        Buffer for maximum logits used in online softmax computation.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
        - Dtype: torch.float32

    temporary_output : torch.Tensor
        Buffer for partial attention outputs from each context partition.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
        - Dtype: torch.float32

    alibi_slopes : torch.Tensor, optional
        ALiBi (Attention with Linear Biases) slopes for positional encoding.
        - Shape: [num_query_heads]
        - Dtype: torch.float32
        - Default: None (no ALiBi)

    Returns
    -------
    None
        Results are written directly to the output tensor.

    Notes
    -----
    - query_length * query_group_size must be <= 64
    - kv_block_size must be one of [16, 64, 1024]
    - When query_length > 1, automatic transpose operations are performed
      between standard and gluon layouts
    - For FP8 computation, query_scale and key_scale/value_scale are required
    - For BF16/FP16 computation, scales can be None
    """
    cdna_version = get_cdna_version()
    assert cdna_version in [
        3,
        4,
    ], f"pa_decode_gluon only supports gfx942 (CDNA3) and gfx950 (CDNA4) now, but got {arch_info.get_arch()}"

    # Extract tensor dimensions from input tensors
    num_query_heads = query.shape[1]
    head_size = query.shape[-1]
    batch_size = query.shape[0] // query_length
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads

    if query_length > 1:
        # Transpose query and query_scale from [num_seqs * query_length, num_query_heads, head_size]
        # to [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
        transpose_query_gluon_aot(
            input_tensor=query,
            output_tensor=query_gluon,
            batch_size=batch_size,
            seq_len=query_length,
            num_kv_heads=num_kv_heads,
            query_group_size=query_group_size,
            last_dim=head_size,
            input_scale=(
                query_scale
                if (query_scale is not None and len(query_scale.shape) > 1)
                else None
            ),
            output_scale=(
                query_scale_gluon
                if (query_scale is not None and len(query_scale.shape) > 1)
                else None
            ),
            run_compiled_kernel=run_compiled_kernel,
        )

    num_sequences = batch_size
    num_query_heads_total = num_query_heads
    max_context_partition_num = int(
        (max_context_length + context_partition_size - 1) // context_partition_size
    )
    head_size = query.shape[-1]
    kv_block_size = key_cache.shape[-2]
    query_group_size = num_query_heads_total // num_kv_heads

    # Calculate equivalent group sizes for kernel configuration
    equivalent_query_group_size = query_length * query_group_size

    # Determine if causal masking is needed
    is_causal = query_length > 1

    assert query_length <= 4, f"query_length == {query_length} exceeds maximum of 4"
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
        context_partition_size=context_partition_size,
        query_quant_mode=query_quant_mode,
        kv_quant_mode=kv_quant_mode,
        fp8_max_value=fp8_max_value,
        value_transposed=int(value_transposed),
        is_causal=int(is_causal),
        use_sinks=int(sinks is not None),
        cdna_version=cdna_version,
    )

    assert combined_func is not None, "Combined function is not compiled"
    # Execute the combined kernel
    if run_compiled_kernel:
        combined_func(
            *torch_to_c_types(
                output_gluon,
                exp_sums,
                max_logits,
                temporary_output,
                query_gluon,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                sinks,
                softmax_scale,
                query_scale_gluon,
                key_scale,
                value_scale,
                output_gluon.stride(0),
                output_gluon.stride(1),
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                temporary_output.stride(0),
                temporary_output.stride(1),
                temporary_output.stride(2),
                temporary_output.stride(3),
                query_gluon.stride(0),
                query_gluon.stride(1),
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
                query_length,
                query_group_size,
                equivalent_query_group_size,
                head_size,
                torch.cuda.current_stream(output.device),
            )
        )

    # Transpose output from [num_seqs, num_kv_heads, query_length, query_group_size, head_size]
    # back to [num_seqs * query_length, num_query_heads, head_size]
    # Only needed when query_length > 1
    if query_length > 1:
        transpose_output_gluon_aot(
            input_tensor=output_gluon,
            output_tensor=output,
            batch_size=batch_size,
            seq_len=query_length,
            num_kv_heads=num_kv_heads,
            query_group_size=query_group_size,
            last_dim=head_size,
            run_compiled_kernel=run_compiled_kernel,
        )

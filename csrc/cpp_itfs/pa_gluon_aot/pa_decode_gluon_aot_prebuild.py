# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import random
from typing import Optional, Tuple, Union, Dict
import subprocess
import multiprocessing
import concurrent.futures
from multiprocessing import cpu_count

import pandas as pd
import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.test_common import benchmark
from csrc.cpp_itfs.utils import (
    BUILD_DIR,
    get_default_func_name,
)
from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import (
    pa_decode_gluon_aot,
)

MD_NAME = "pa_decode_attention_reduce_kernel"

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# Triton to PyTorch dtype mapping
TL_TO_TORCH_DTYPE = {
    tl.float8e4b8: torch.float8_e4m3fnuz,
    tl.float8e4nv: torch.float8_e4m3fn,
    tl.bfloat16: torch.bfloat16,
    tl.float16: torch.float16,
}
TORCH_TO_TL_DTYPE = {
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
}

# Test configuration parameters
USE_TORCH_FLASH_REF_OPTIONS = [True]
USE_AOT_IMPL_OPTIONS = [True, False]
KV_VARLEN_OPTIONS = [False, True]
TRANS_V_OPTIONS = [False, True]
# QUANT_Q_AND_KV_OPTIONS = [[True, True], [False, False]]
QUANT_Q_AND_KV_OPTIONS = [[False, False]]
CONTEXT_PARTITION_SIZE_OPTIONS = [256]
COMPUTE_TYPE_OPTIONS = ["fp8", "bf16", "fp16"]
QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
HEAD_DIMENSION_OPTIONS = [128]

BLOCK_SIZE_OPTIONS = [16, 64, 1024]
HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1), (64, 4)]
QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
CONTEXT_LENGTH_OPTIONS = [512, 4096, 4097]
BATCH_SIZE_OPTIONS = [4, 80, 128]
SINKS_OPTIONS = [True, False]
SLIDING_WINDOW_OPTIONS = [0, 128]
COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = []


def run_gluon_kernel(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    query_length: int,
    max_context_partition_num: int,
    context_partition_size: int,
    compute_type: torch.dtype,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    use_aot_impl: bool = False,
) -> None:
    """Run Gluon FP8/BF16/FP16 kernel for paged attention.

    Args:
        output: Output tensor [batch_size * query_length, num_query_heads, head_size]
        query: Query tensor [batch_size * query_length, num_query_heads, head_size]
        key_cache: Key cache tensor [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
        value_cache: Value cache tensor [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
        context_lengths: Current context lengths for each sequence [num_seqs]
        block_tables: Mapping from sequences to physical cache blocks [num_seqs, max_num_blocks_per_seq]
        softmax_scale: Softmax scale (typically 1/sqrt(head_size))
        query_length: Query sequence length
        max_context_partition_num: Maximum number of context partitions
        context_partition_size: Context partition size
        compute_type: Compute data type (torch.dtype)
        query_scale: Query quantization scale
        key_scale: Key quantization scale
        value_scale: Value quantization scale
        exp_sums: Buffer for exponential sums
        max_logits: Buffer for maximum logits
        temporary_output: Buffer for partial outputs
        alibi_slopes: ALiBi slopes (optional)
        sinks: Sink tokens (optional)
        use_aot_impl: Whether to use AOT implementation

    Returns:
        None (modifies output in-place)

    This function can run in aot or jit mode based on use_aot_impl flag.
    """
    if use_aot_impl:
        pa_decode_gluon_aot(
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            query_length,
            max_context_partition_num,
            context_partition_size,
            compute_type,
            query_scale,
            key_scale,
            value_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=alibi_slopes,
            run_compiled_kernel=True,
            sinks=sinks,
        )


@benchmark()
def run_pa_gluon_test(
    context_length: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    compute_type: torch.dtype,
    query_length: int,
    quant_mode: str,
    context_partition_size: int,
    trans_v: bool,
    kv_varlen: bool,
    use_aot_impl: bool,
    quant_q: bool,
    quant_kv: bool,
) -> Dict[str, Union[float, str]]:
    """Test paged attention decode with gluon implementations."""
    results = {}
    data_type = compute_type
    if compute_type == aiter.dtypes.fp8:
        data_type = torch.bfloat16
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads
    assert (
        num_query_heads % num_kv_heads == 0
    ), "Query heads must be divisible by KV heads"

    max_context_length = max(1024, context_length)
    max_blocks_per_sequence = (max_context_length + block_size - 1) // block_size
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = (context_length + block_size - 1) // block_size
    total_queries = batch_size * query_length

    if kv_varlen:
        random.seed(123)
        kv_len_list = [
            random.randint(query_length, context_length) for _ in range(batch_size)
        ]
    else:
        kv_len_list = [context_length] * batch_size
    context_lengths = torch.tensor(kv_len_list, dtype=torch.int32, device=device)

    random.seed(123)
    block_tables_list = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)
        ]
        block_tables_list.append(block_table)

    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    # Create KV cache tensors directly with torch.empty
    elements_per_vector = 16 // data_type.itemsize
    key_cache_shape = (
        total_blocks,
        num_kv_heads,
        head_size // elements_per_vector,
        block_size,
        elements_per_vector,
    )
    key_cache = torch.empty(key_cache_shape, dtype=data_type, device=device)

    value_cache_shape = (total_blocks, num_kv_heads, head_size, block_size)
    value_cache = torch.empty(value_cache_shape, dtype=data_type, device=device)

    softmax_scale = 1.0 / (head_size**0.5)

    # Create query tensors directly
    query = torch.empty(
        total_queries, num_query_heads, head_size, dtype=data_type, device=device
    )

    # Quantization based on mode and flags
    if quant_mode == "per_token":
        # Per-token quantization for query (if enabled)
        if quant_q:
            quantized_query = torch.empty(
                total_queries,
                num_query_heads,
                head_size,
                dtype=aiter.dtypes.fp8,
                device=device,
            )
            # Per-token query scale factors: [total_queries, num_query_heads, 1]
            query_scale_factors = torch.empty(
                total_queries, num_query_heads, 1, dtype=torch.float32, device=device
            )
        else:
            quantized_query = query
            query_scale_factors = None

        # Per-token quantization for KV cache (if enabled)
        if quant_kv:
            quantized_keys = torch.empty(
                key_cache_shape, dtype=aiter.dtypes.fp8, device=device
            )
            # Determine value cache shape based on trans_v
            if trans_v:
                # Shuffled layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
                quant_elements_per_vector = 16 // aiter.dtypes.fp8.itemsize
                quantized_values_shape = (
                    total_blocks,
                    num_kv_heads,
                    block_size // quant_elements_per_vector,
                    head_size,
                    quant_elements_per_vector,
                )
            else:
                # Normal layout: [num_blocks, num_kv_heads, head_size, block_size]
                quantized_values_shape = value_cache_shape
            quantized_values = torch.empty(
                quantized_values_shape, dtype=aiter.dtypes.fp8, device=device
            )
            # Per-token KV scale factors: [num_blocks, num_kv_heads, block_size, 1]
            key_scale_original = torch.empty(
                total_blocks,
                num_kv_heads,
                block_size,
                1,
                dtype=torch.float32,
                device=device,
            )
            value_scale_original = torch.empty(
                total_blocks,
                num_kv_heads,
                block_size,
                1,
                dtype=torch.float32,
                device=device,
            )
        else:
            quantized_keys = key_cache
            # Determine value cache shape based on trans_v
            if trans_v:
                # Shuffled layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
                cache_elements_per_vector = 16 // data_type.itemsize
                quantized_values_shape = (
                    total_blocks,
                    num_kv_heads,
                    block_size // cache_elements_per_vector,
                    head_size,
                    cache_elements_per_vector,
                )
                quantized_values = torch.empty(
                    quantized_values_shape, dtype=data_type, device=device
                )
            else:
                quantized_values = value_cache
            key_scale_original = None
            value_scale_original = None
    else:  # per_tensor
        # Per-tensor quantization for query (if enabled)
        if quant_q:
            quantized_query = torch.empty(
                total_queries,
                num_query_heads,
                head_size,
                dtype=aiter.dtypes.fp8,
                device=device,
            )
            # Per-tensor query scale factor: scalar [1]
            query_scale_factors = torch.empty(1, dtype=torch.float32, device=device)
        else:
            quantized_query = query
            query_scale_factors = None

        # Per-tensor quantization for KV cache (if enabled)
        if quant_kv:
            quantized_keys = torch.empty(
                key_cache_shape, dtype=aiter.dtypes.fp8, device=device
            )
            # Determine value cache shape based on trans_v
            if trans_v:
                # Shuffled layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
                quant_elements_per_vector = 16 // aiter.dtypes.fp8.itemsize
                quantized_values_shape = (
                    total_blocks,
                    num_kv_heads,
                    block_size // quant_elements_per_vector,
                    head_size,
                    quant_elements_per_vector,
                )
            else:
                # Normal layout: [num_blocks, num_kv_heads, head_size, block_size]
                quantized_values_shape = value_cache_shape
            quantized_values = torch.empty(
                quantized_values_shape, dtype=aiter.dtypes.fp8, device=device
            )
            # Per-tensor KV scale factors: scalar [1]
            key_scale_original = torch.empty(1, dtype=torch.float32, device=device)
            value_scale_original = torch.empty(1, dtype=torch.float32, device=device)
        else:
            quantized_keys = key_cache
            # Determine value cache shape based on trans_v
            if trans_v:
                # Shuffled layout: [num_blocks, num_kv_heads, block_size // x, head_size, x]
                cache_elements_per_vector = 16 // data_type.itemsize
                quantized_values_shape = (
                    total_blocks,
                    num_kv_heads,
                    block_size // cache_elements_per_vector,
                    head_size,
                    cache_elements_per_vector,
                )
                quantized_values = torch.empty(
                    quantized_values_shape, dtype=data_type, device=device
                )
            else:
                quantized_values = value_cache
            key_scale_original = None
            value_scale_original = None

    # Test Gluon
    query_group_size = num_query_heads // num_kv_heads
    num_seqs = batch_size
    max_context_partition_num = (
        context_lengths.max().item() + context_partition_size - 1
    ) // context_partition_size
    query_group_size = num_query_heads // num_kv_heads
    equivalent_query_group_size = query_length * query_group_size
    intermediate_shape = (
        num_seqs,
        num_kv_heads,
        max_context_partition_num,
        equivalent_query_group_size,
    )
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
    max_logits = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
    temporary_output = torch.empty(
        *intermediate_shape,
        head_size,
        dtype=data_type,
        device=device,
    )
    # Create output tensor for final results
    output = torch.empty(
        total_queries, num_query_heads, head_size, dtype=data_type, device=device
    )

    run_gluon_kernel(
        output,
        quantized_query,
        quantized_keys,
        quantized_values,
        context_lengths,
        block_tables,
        softmax_scale,
        query_length,
        max_context_partition_num,
        context_partition_size,
        compute_type,
        query_scale=query_scale_factors,
        key_scale=key_scale_original,
        value_scale=value_scale_original,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
        use_aot_impl=use_aot_impl,
    )

    return results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test paged attention decode gluon implementation",
    )

    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="Compute type",
    )
    parser.add_argument(
        "-n",
        "--num_heads",
        type=dtypes.str2tuple,
        default=None,
        help="Number of heads (q_heads, kv_heads)",
    )
    parser.add_argument(
        "-q",
        "--query_length",
        type=int,
        choices=QUERY_LENGTH_OPTIONS,
        default=None,
        help="Query length",
    )
    parser.add_argument(
        "-c", "--context_length", type=int, default=None, help="Context length"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--block_size", type=int, default=None, help="Block size")
    parser.add_argument(
        "--quant_mode",
        type=str,
        choices=["per_token", "per_tensor", "both"],
        default=None,
        help="Quantization mode: per_token, per_tensor, or both",
    )
    parser.add_argument(
        "--quant_q_and_kv",
        type=dtypes.str2tuple,
        default=None,
        help=(
            "Tuple of bools specifying whether to quant_q and quant_kv, e.g. 0,0 "
            "First value is for quant_q, second for quant_kv."
        ),
    )
    parser.add_argument(
        "--trans_v",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Transpose value cache layout (True/False)",
    )
    parser.add_argument(
        "--kv_varlen",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="KV use varlen (True/False)",
    )
    parser.add_argument(
        "--use_torch_flash_ref",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Use torch flash reference implementation (True/False)",
    )
    parser.add_argument(
        "--use_aot_impl",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Use gluon AOT implementation (True/False)",
    )
    parser.add_argument(
        "--context_partition_size",
        type=int,
        default=None,
        help="Sequence partition size",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=32,
        help="Number of parallel processes to use (default: use 1 CPU cores)",
    )

    return parser


def process_arguments(args: argparse.Namespace) -> tuple:
    """Process command line arguments."""
    compute_types = COMPUTE_TYPE_OPTIONS
    block_sizes = BLOCK_SIZE_OPTIONS
    head_configs = HEAD_CONFIGURATIONS
    context_length = CONTEXT_LENGTH_OPTIONS
    batch_sizes = BATCH_SIZE_OPTIONS
    query_lengths = QUERY_LENGTH_OPTIONS
    quant_mode = QUANT_MODE_OPTIONS
    trans_v = TRANS_V_OPTIONS
    kv_varlen = KV_VARLEN_OPTIONS
    quant_q_and_kv = QUANT_Q_AND_KV_OPTIONS
    use_torch_flash_ref_options = USE_TORCH_FLASH_REF_OPTIONS
    use_aot_impl_options = USE_AOT_IMPL_OPTIONS
    context_partition_size_options = CONTEXT_PARTITION_SIZE_OPTIONS
    sinks_options = SINKS_OPTIONS
    sliding_window_options = SLIDING_WINDOW_OPTIONS

    if args.compute_type is not None:
        compute_types = [dtypes.d_dtypes[args.compute_type]]
    else:
        compute_types = [dtypes.d_dtypes[key] for key in compute_types]

    if args.num_heads is not None:
        head_configs = [args.num_heads]
    if args.query_length is not None:
        query_lengths = [args.query_length]
    if args.context_length is not None:
        context_length = [args.context_length]
    if args.batch_size is not None:
        batch_sizes = [args.batch_size]
    if args.block_size is not None:
        block_sizes = [args.block_size]
    if args.block_size is not None:
        block_sizes = [args.block_size]
    if args.quant_mode is not None:
        quant_mode = [args.quant_mode]
    if args.trans_v is not None:
        trans_v = [args.trans_v]
    if args.kv_varlen is not None:
        kv_varlen = [args.kv_varlen]
    if args.quant_q_and_kv is not None:
        quant_q_and_kv = [args.quant_q_and_kv]

    # Process new arguments
    if args.use_torch_flash_ref is not None:
        use_torch_flash_ref_options = [args.use_torch_flash_ref]
    if args.use_aot_impl is not None:
        use_aot_impl_options = [args.use_aot_impl]
    if args.context_partition_size is not None:
        context_partition_size_options = [args.context_partition_size]

    # Create compute_types_quant_q_and_kv list (same as test_pa_decode_gluon.py)
    compute_types_quant_q_and_kv = []
    for ct in compute_types:
        for quant_q, quant_kv in quant_q_and_kv:
            compute_types_quant_q_and_kv.append([ct, quant_q, quant_kv])
    if len(COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS) > 0:
        compute_types_quant_q_and_kv = COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
        for idx in range(len(compute_types_quant_q_and_kv)):
            if not isinstance(compute_types_quant_q_and_kv[idx][0], torch.dtype):
                compute_types_quant_q_and_kv[idx][0] = dtypes.d_dtypes[
                    compute_types_quant_q_and_kv[idx][0]
                ]

    return (
        block_sizes,
        head_configs,
        context_length,
        batch_sizes,
        query_lengths,
        quant_mode,
        trans_v,
        kv_varlen,
        compute_types_quant_q_and_kv,
        use_torch_flash_ref_options,
        use_aot_impl_options,
        context_partition_size_options,
        sinks_options,
        sliding_window_options,
    )


def _run_single_test(args):
    """
    Helper function to run a single test case.

    Args:
        args: Tuple containing (test_config, current, total)

    Returns:
        Dictionary containing test results
    """
    test_config, current, total = args

    print(
        f"\n[{current}/{total}] Testing: "
        f"use_torch_flash_ref={test_config['use_torch_flash_ref']}, "
        f"compute_type={test_config['compute_type']}, "
        f"quant_q_and_kv=({test_config['quant_q']}, {test_config['quant_kv']}), "
        f"use_aot_impl={test_config['use_aot_impl']}, "
        f"trans_v={test_config['trans_v']}, "
        f"kv_varlen={test_config['kv_varlen']}, "
        f"context_partition_size={test_config['context_partition_size']}, "
        f"quant_mode={test_config['quant_mode']}, "
        f"block_size={test_config['block_size']}, "
        f"num_heads={test_config['num_heads']}, "
        f"context_lengths={test_config['context_length']}, "
        f"batch_size={test_config['batch_size']}, "
        f"query_length={test_config['query_length']}, "
        f"head_size={test_config['head_size']}"
    )

    result = run_pa_gluon_test(
        context_length=test_config["context_length"],
        batch_size=test_config["batch_size"],
        num_heads=test_config["num_heads"],
        head_size=test_config["head_size"],
        block_size=test_config["block_size"],
        compute_type=test_config["compute_type"],
        query_length=test_config["query_length"],
        quant_mode=test_config["quant_mode"],
        context_partition_size=test_config["context_partition_size"],
        trans_v=test_config["trans_v"],
        kv_varlen=test_config["kv_varlen"],
        use_aot_impl=test_config["use_aot_impl"],
        quant_q=test_config["quant_q"],
        quant_kv=test_config["quant_kv"],
    )

    return result


def run_multi_pa_gluon_test(
    block_sizes,
    head_configs,
    context_length,
    batch_sizes,
    query_lengths,
    quant_mode,
    trans_v,
    kv_varlen,
    compute_types_quant_q_and_kv,
    use_torch_flash_ref_options,
    use_aot_impl_options,
    context_partition_size_options,
    num_processes=None,
    sinks_options=[False],
    sliding_window_options=[0],
) -> pd.DataFrame:
    """
    Run all tests using multiprocessing for parallel execution.

    Args:
        num_processes: Number of parallel processes to use.
                      If None, uses cpu_count().

    Returns:
        DataFrame containing all test results
    """
    cdna_version = 3
    # Generate all test configurations
    test_configs = []

    for use_torch_flash_ref in use_torch_flash_ref_options:
        for hc in head_configs:
            for ct, quant_q, quant_kv in compute_types_quant_q_and_kv:
                for trans_v_mode in trans_v:
                    for kv_varlen_mode in kv_varlen:
                        for context_partition_size in context_partition_size_options:
                            qm_cnt = 0
                            for qm in quant_mode:
                                qm_cnt += 1
                                if not quant_q and not quant_kv and qm_cnt > 1:
                                    continue
                                for bs in block_sizes:
                                    for head_size in HEAD_DIMENSION_OPTIONS:
                                        for ql in query_lengths:
                                            for bsz in batch_sizes:
                                                for cl in context_length:
                                                    for (
                                                        use_aot_impl
                                                    ) in use_aot_impl_options:
                                                        for sinks in sinks_options:
                                                            for (
                                                                sliding_window
                                                            ) in sliding_window_options:
                                                                test_config = {
                                                                    "use_torch_flash_ref": use_torch_flash_ref,
                                                                    "compute_type": ct,
                                                                    "quant_q": quant_q,
                                                                    "quant_kv": quant_kv,
                                                                    "trans_v": trans_v_mode,
                                                                    "kv_varlen": kv_varlen_mode,
                                                                    "context_partition_size": context_partition_size,
                                                                    "quant_mode": qm,
                                                                    "block_size": bs,
                                                                    "num_heads": hc,
                                                                    "context_length": cl,
                                                                    "batch_size": bsz,
                                                                    "query_length": ql,
                                                                    "head_size": head_size,
                                                                    "use_aot_impl": use_aot_impl,
                                                                    "sinks": sinks,
                                                                    "sliding_window": sliding_window,
                                                                }

                                                                # Calculate func_name to filter duplicate configs
                                                                (
                                                                    num_query_heads,
                                                                    num_kv_heads,
                                                                ) = hc
                                                                query_group_size = (
                                                                    num_query_heads
                                                                    // num_kv_heads
                                                                )

                                                                # Calculate power of 2 values
                                                                head_size_pow2 = triton.next_power_of_2(
                                                                    head_size
                                                                )

                                                                # Determine quantization modes
                                                                if quant_q:
                                                                    if (
                                                                        qm
                                                                        == "per_tensor"
                                                                    ):
                                                                        query_quant_mode = (
                                                                            0
                                                                        )
                                                                    else:  # per_token
                                                                        query_quant_mode = (
                                                                            1
                                                                        )
                                                                else:
                                                                    query_quant_mode = (
                                                                        -1
                                                                    )

                                                                if quant_kv:
                                                                    if (
                                                                        qm
                                                                        == "per_tensor"
                                                                    ):
                                                                        kv_quant_mode = (
                                                                            0
                                                                        )
                                                                    else:  # per_token
                                                                        kv_quant_mode = (
                                                                            1
                                                                        )
                                                                else:
                                                                    kv_quant_mode = -1

                                                                # Determine fp8_max_value
                                                                if kv_quant_mode >= 0:
                                                                    fp8_max_value = torch.finfo(
                                                                        aiter.dtypes.fp8
                                                                    ).max
                                                                else:
                                                                    fp8_max_value = 1.0

                                                                # Determine is_causal
                                                                is_causal = int(ql > 1)

                                                                # use_sinks for func_name (cdna_version already set at function start)
                                                                use_sinks = 0  # Default: no sinks

                                                                # Calculate func_name
                                                                func_name = get_default_func_name(
                                                                    MD_NAME,
                                                                    (
                                                                        ct,  # compute_type (torch.dtype)
                                                                        ql,  # query_seq_len
                                                                        query_group_size,  # one_query_group_size
                                                                        head_size_pow2,
                                                                        bs,  # kv_block_size
                                                                        context_partition_size,
                                                                        query_quant_mode,
                                                                        kv_quant_mode,
                                                                        fp8_max_value,
                                                                        int(
                                                                            trans_v_mode
                                                                        ),  # value_transposed
                                                                        is_causal,
                                                                        use_sinks,
                                                                        cdna_version,
                                                                    ),
                                                                )
                                                                # Store func_name in test_config for deduplication
                                                                test_config[
                                                                    "func_name"
                                                                ] = func_name
                                                                test_configs.append(
                                                                    test_config
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

    # Prepare arguments for multiprocessing
    test_args = [(config, idx + 1, total) for idx, config in enumerate(test_configs)]

    # Determine number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), total)
    else:
        num_processes = min(cpu_count(), num_processes)
        num_processes = min(total, num_processes)
    print(f"Using {num_processes} parallel processes\n")

    # Run tests in parallel using spawn context to avoid CUDA reinitialization issues
    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_processes, mp_context=mp_context
    ) as executor:
        results = list(executor.map(_run_single_test, test_args))

    return pd.DataFrame(results)


def parse_arg_and_run_test():
    """Parse arguments and run tests."""
    print(f"Triton location: {triton}")
    print(f"Triton version: {triton.__version__}")

    parser = create_argument_parser()
    args = parser.parse_args()
    (
        block_sizes,
        head_configs,
        context_length,
        batch_sizes,
        query_lengths,
        quant_mode,
        trans_v,
        kv_varlen,
        compute_types_quant_q_and_kv,
        use_torch_flash_ref_options,
        use_aot_impl_options,
        context_partition_size_options,
        sinks_options,
        sliding_window_options,
    ) = process_arguments(args)

    run_multi_pa_gluon_test(
        block_sizes,
        head_configs,
        context_length,
        batch_sizes,
        query_lengths,
        quant_mode,
        trans_v,
        kv_varlen,
        compute_types_quant_q_and_kv,
        use_torch_flash_ref_options,
        use_aot_impl_options,
        context_partition_size_options,
        num_processes=args.num_processes,
        sinks_options=sinks_options,
        sliding_window_options=sliding_window_options,
    )
    print("All the so under different configurations have been built successfully!")


def get_so_files_size_and_count():
    """Get the total size and number of so files in aiter build directory."""
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


def prebuild_normal_accuracy_cases_aot_so():
    """Prebuild normal accuracy cases AOT so files."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS

    USE_AOT_IMPL_OPTIONS = [True]
    SINKS_OPTIONS = [False]
    SLIDING_WINDOW_OPTIONS = [0]

    USE_TORCH_FLASH_REF_OPTIONS = [True]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]
    HEAD_DIMENSION_OPTIONS = [128]
    HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1)]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
    CONTEXT_LENGTH_OPTIONS = [1027]
    BATCH_SIZE_OPTIONS = [3, 81]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [False, True]
    BLOCK_SIZE_OPTIONS = [16, 64, 1024]
    parse_arg_and_run_test()

    # Test for different head dimensions
    HEAD_DIMENSION_OPTIONS = [64, 192, 256]
    HEAD_CONFIGURATIONS = [(8, 1)]
    QUERY_LENGTH_OPTIONS = [1, 3]
    QUANT_MODE_OPTIONS = ["per_token"]
    BATCH_SIZE_OPTIONS = [81]
    KV_VARLEN_OPTIONS = [True]
    parse_arg_and_run_test()


def prebuild_normal_performance_cases_aot_so():
    """Prebuild normal performance cases AOT so files."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS

    USE_AOT_IMPL_OPTIONS = [True]
    SINKS_OPTIONS = [False]
    SLIDING_WINDOW_OPTIONS = [0]

    USE_TORCH_FLASH_REF_OPTIONS = [True]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]
    HEAD_DIMENSION_OPTIONS = [128]
    CONTEXT_LENGTH_OPTIONS = [2048, 4096, 8192]
    BATCH_SIZE_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_tensor"]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [False]
    HEAD_CONFIGURATIONS = [(64, 4), (64, 8)]
    BLOCK_SIZE_OPTIONS = [16, 64]
    parse_arg_and_run_test()


if __name__ == "__main__":
    prebuild_normal_accuracy_cases_aot_so()
    prebuild_normal_performance_cases_aot_so()
    get_so_files_size_and_count()

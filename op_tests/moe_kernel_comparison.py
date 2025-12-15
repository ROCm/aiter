# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Kernel Variant Benchmarking Tool for MoE Operations
====================================================

This tool benchmarks ALL available kernel implementations for a FIXED MoE configuration
to find the optimal (fastest) kernel variant.

Purpose:
    For a specific workload (e.g., token=1024, expert=8), exhaustively test all kernel
    variants to determine which implementation provides the best performance.

Kernel Types Tested:
    1. 1-Stage Fused Kernels: Single kernel performs complete MoE operation
    2. 2-Stage Combinations: Pair of (stage1 + stage2) kernels working together

Usage Examples:
    # Test only 2-stage kernels (recommended - avoids GPU errors)
    python benchmark_kernel_variants.py --skip-1stage
    
    # Test specific configuration
    python benchmark_kernel_variants.py --token 1024 --expert 256 --topk 8 --skip-1stage
    
    # Test only 1-stage kernels (may have GPU issues)
    python benchmark_kernel_variants.py --only-1stage
"""

import torch
import aiter
import pandas as pd
import sys
import argparse
from aiter import QuantType, dtypes, ActivationType
from aiter.jit.core import get_asm_dir, AITER_CSRC_DIR
from aiter.test_common import run_perftest
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    asm_stage1,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter import ck_moe_stage1_fwd, ck_moe_stage2_fwd, dtype2str_dict
from aiter.ops.shuffle import shuffle_weight
from aiter.int4_utils import rearrange_4bit_elements, convert_int8_to_uint32_int4
from aiter.utility import fp4_utils
from aiter.jit.utils.chip_info import get_gfx

# Add CK codegen to path for kernel list generation
sys.path.insert(0, f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/")
from gemm_moe_ck2stages_common import get_gemm1_kernels_list, get_gemm2_kernels_list

torch.set_default_device("cuda")
torch.int4 = getattr(torch, "int4", torch.uint32)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_test_data(
    token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, blockM
):
    """
    Generate test data for MoE kernel benchmarking.
    
    Creates random input tensors and performs all necessary preprocessing:
    - Weight quantization
    - Input quantization
    - TopK expert selection
    - MoE token sorting
    - Weight shuffling for optimal memory access
    
    Args:
        token: Number of tokens
        model_dim: Model dimension  
        inter_dim: Intermediate (hidden) dimension
        expert: Number of experts
        topk: Number of experts to select per token
        dtype: Computation data type (bf16 or fp16)
        q_dtype_a: Activation quantization type
        q_dtype_w: Weight quantization type
        q_type: Quantization strategy (per_Token, per_Tensor, etc.)
        blockM: Block size for MoE sorting
        
    Returns:
        dict: All tensors needed for kernel benchmarking
    """
    torch.manual_seed(42)  # Fixed seed for reproducible benchmarks
    
    # Generate random input data
    input = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((expert, inter_dim * 2, model_dim), dtype=dtype) / 10  # Gate + Up weights
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtype) / 10      # Down weights
    
    # ==========================================
    # WEIGHT QUANTIZATION
    # ==========================================
    if q_type == QuantType.per_Tensor:
        # One scale per entire weight tensor
        w1_qt, w1_scale = aiter.pertoken_quant(w1.view(expert, -1), quant_dtype=q_dtype_w)
        w2_qt, w2_scale = aiter.pertoken_quant(w2.view(expert, -1), quant_dtype=q_dtype_w)
        w1_qt = w1_qt.view(w1.shape)
        w2_qt = w2_qt.view(w2.shape)
    elif q_type == QuantType.per_Token and q_dtype_w == torch.int4:
        # Per-token quantization with int4 weights
        w1_qt, w1_scale = aiter.pertoken_quant(w1, quant_dtype=dtypes.i8, dtypeMax=7)
        w2_qt, w2_scale = aiter.pertoken_quant(w2, quant_dtype=dtypes.i8, dtypeMax=7)
    else:
        # Standard per-token or no quantization
        torch_quant = aiter.get_torch_quant(q_type)
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=q_dtype_w)
        w2_qt, w2_scale = torch_quant(w2, quant_dtype=q_dtype_w)
    
    w1_qt = w1_qt.view(w1.shape)
    w2_qt = w2_qt.view(w2.shape)
    
    # ==========================================
    # EXPERT SELECTION (TopK)
    # ==========================================
    score = torch.randn((token, expert), dtype=dtype)
    topk_weights, topk_ids = fused_topk(input, score, topk, True)
    
    # ==========================================
    # INPUT QUANTIZATION
    # ==========================================
    torch_quant = aiter.get_torch_quant(q_type)
    a1_qt, a1_scale = torch_quant(input, quant_dtype=q_dtype_a)
    
    # ==========================================
    # WEIGHT SHUFFLING (for optimal GPU memory access)
    # ==========================================
    w1_qt_shffle = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shffle = shuffle_weight(w2_qt, (16, 16))
    w1_scale_shffle = fp4_utils.e8m0_shuffle(w1_scale)
    w2_scale_shffle = fp4_utils.e8m0_shuffle(w2_scale)
    
    # ==========================================
    # MOE TOKEN SORTING (group tokens by expert)
    # ==========================================
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, expert, model_dim, dtype, blockM
    )
    
    return {
        'input': input,
        'a1_qt': a1_qt,
        'w1_qt': w1_qt,
        'w2_qt': w2_qt,
        'w1_qt_shffle': w1_qt_shffle,
        'w2_qt_shffle': w2_qt_shffle,
        'a1_scale': a1_scale,
        'w1_scale': w1_scale,
        'w2_scale': w2_scale,
        'w1_scale_shffle': w1_scale_shffle,
        'w2_scale_shffle': w2_scale_shffle,
        'sorted_ids': sorted_ids,
        'sorted_weights': sorted_weights,
        'sorted_expert_ids': sorted_expert_ids,
        'num_valid_ids': num_valid_ids,
        'topk_weights': topk_weights,
        'topk_ids': topk_ids,
        'moe_buf': moe_buf,
    }


# ============================================================================
# KERNEL ENUMERATION - Get Available Kernel Variants
# ============================================================================

def get_asm_stage1_kernels_for_2stage(block_m, act_type, q_type, q_dtype_a, use_g1u1=True, doweight_stage1=False):
    """
    Get list of ASM stage1 kernels for 2-stage MoE from CSV inventory.
    
    ASM kernels are pre-compiled and listed in CSV files. This function reads
    the appropriate CSV file and returns kernels matching the given block size.
    
    Args:
        block_m: Block size (32, 64, or 128)
        act_type: Activation function (Silu or Gelu)
        q_type: Quantization type
        q_dtype_a: Activation data type
        use_g1u1: Whether using gate+up architecture
        doweight_stage1: Whether to apply routing weights in stage1
        
    Returns:
        list: Kernel names that match criteria, e.g.:
              ['_ZN5aiter47fmoe_stage1_bf16_pertokenFp8_g1u1_32x64_4tg_pf3E', ...]
    """
    acti_dir = "silu" if act_type == ActivationType.Silu else "gelu"
    up = 1 if use_g1u1 else 0
    
    # Determine quantization type string for file path
    if q_dtype_a == dtypes.i8:
        quantDtype = "Int8"
    elif q_dtype_a == dtypes.fp8:
        quantDtype = "Fp8"
    else:
        return []  # Unsupported quant type for ASM kernels
    
    # Build CSV file path
    extraInfo = "_doweight" if doweight_stage1 else ""
    kernels_list_csv = f"{get_asm_dir()}/fmoe_2stages/fmoe_stage1_bf16_pertoken{quantDtype}{extraInfo}_g1u1.csv"
    
    try:
        # Read CSV and filter by block size
        df = pd.read_csv(kernels_list_csv)
        kernels = df[df['tile_m'] == block_m]['knl_name'].tolist()
        return kernels
    except:
        return []


def get_1stage_fused_asm_kernels(act_type, q_type, q_dtype_a, use_g1u1=True, doweight_stage1=False):
    """
    Get list of 1-stage fused ASM kernels from CSV inventory.
    
    1-stage kernels perform the complete MoE operation in a single fused kernel.
    These are also pre-compiled and listed in CSV files.
    
    Args:
        act_type: Activation function
        q_type: Quantization type
        q_dtype_a: Activation data type
        use_g1u1: Whether using gate+up architecture
        doweight_stage1: Whether to apply routing weights
        
    Returns:
        list: 1-stage fused kernel names
    """
    acti_dir = "silu" if act_type == ActivationType.Silu else "gelu"
    up = 1 if use_g1u1 else 0
    
    if q_dtype_a == dtypes.i8:
        quantDtype = "Int8"
    elif q_dtype_a == dtypes.fp8:
        quantDtype = "Fp8"
    else:
        return []
    
    # Build CSV file path for 1-stage kernels
    kernels_list_csv = f"{get_asm_dir()}/fmoe/{acti_dir}/fmoe_bf16_pertoken{quantDtype}_g1u{up}_{acti_dir}.csv"
    
    try:
        df = pd.read_csv(kernels_list_csv)
        kernels = df['knl_name'].tolist()
        return kernels
    except:
        return []


def get_ck_stage1_kernels_for_2stage(block_m, dtype, q_dtype_a, q_dtype_w, q_type, act_type, doweight_stage1):
    """
    Get list of CK stage1 kernels for 2-stage MoE from codegen templates.
    
    Unlike ASM kernels which are pre-compiled and listed in CSVs, CK kernels are
    generated programmatically from templates. This function calls CK's codegen
    to get all possible stage1 kernel configurations.
    
    Note: Many returned kernel names won't have compiled binaries yet. They'll be
    JIT-compiled on first use or return "kernel not found" errors.
    
    Args:
        block_m: Block size
        dtype: Computation data type
        q_dtype_a: Activation quantization type
        q_dtype_w: Weight quantization type
        q_type: Quantization strategy
        act_type: Activation function
        doweight_stage1: Whether to apply routing weights in stage1
        
    Returns:
        list: CK stage1 kernel objects with properties (name, MPerBlock, etc.)
    """
    try:
        _, kernels_dict = get_gemm1_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            dtype2str_dict[dtype],
            False,  # bpreshuffle
            int(q_type),
            str(act_type).split(".")[-1].lower(),
            doweight_stage1,
            False,  # bpreshuffle
        )
        # Filter by block size
        return [k for k in kernels_dict.values() if k.MPerBlock == block_m]
    except:
        return []


def get_ck_stage2_kernels_for_2stage(block_m, dtype, q_dtype_a, q_dtype_w, q_type, act_type, doweight_stage1):
    """
    Get list of CK stage2 kernels for 2-stage MoE from codegen templates.
    
    CK stage2 kernels handle the second matrix multiplication in 2-stage MoE.
    Like stage1, these are generated from templates and JIT-compiled on demand.
    
    Args:
        block_m: Block size
        dtype: Computation data type
        q_dtype_a: Activation quantization type
        q_dtype_w: Weight quantization type
        q_type: Quantization strategy
        act_type: Activation function (not used by stage2 but passed for consistency)
        doweight_stage1: Whether weights were applied in stage1 (inverted for stage2)
        
    Returns:
        list: CK stage2 kernel objects with properties (name, MPerBlock, etc.)
    """
    try:
        _, kernels_dict = get_gemm2_kernels_list(
            dtype2str_dict[q_dtype_a],
            dtype2str_dict[q_dtype_w],
            dtype2str_dict[dtype],
            False,  # bpreshuffle
            int(q_type),
            not doweight_stage1,  # Note: inverted - if weights applied in s1, not in s2
            False,  # bpreshuffle
        )
        # Filter by block size
        return [k for k in kernels_dict.values() if k.MPerBlock == block_m]
    except:
        return []


# ============================================================================
# KERNEL BENCHMARKING FUNCTIONS
# ============================================================================

def benchmark_1stage_kernel(data, kernel_name, topk, q_type, act_type, q_dtype_a, doweight_stage1=False):
    """
    Benchmark a single 1-stage fused kernel.
    
    1-stage kernels perform the complete MoE operation:
    input → expert_selection → matmul1 → activation → matmul2 → output
    
    Args:
        data: Preprocessed test data dictionary
        kernel_name: Name of the 1-stage kernel to test
        topk: Number of experts per token
        q_type: Quantization type
        act_type: Activation function
        q_dtype_a: Activation data type
        doweight_stage1: Whether to apply routing weights
        
    Returns:
        tuple: (execution_time_us, error_message or None)
    """
    token = data['a1_qt'].shape[0]
    model_dim = data['w2_qt'].shape[1]
    
    # Allocate output buffer
    moe_buf = torch.zeros((token, model_dim), dtype=data['input'].dtype)
    
    # Select appropriate wrapper function based on quantization type
    # Different quant types require different kernel entry points
    if q_type == QuantType.per_128x128 or q_type == QuantType.per_1x128:
        fmoe_func = aiter.fmoe_fp8_blockscale_g1u1
    elif q_dtype_a == dtypes.fp8 and doweight_stage1:
        fmoe_func = aiter.fmoe_g1u1_tkw1
    else:
        fmoe_func = aiter.fmoe_g1u1
    
    try:
        _, time_us = run_perftest(
            fmoe_func,
            moe_buf,
            data['a1_qt'],
            data['w1_qt_shffle'],
            data['w2_qt_shffle'],
            data['sorted_ids'],
            data['sorted_weights'],
            data['sorted_expert_ids'],
            data['num_valid_ids'],
            topk,
            data['a1_scale'],
            data['w1_scale'],
            data['w2_scale'],
            kernel_name,
            fc2_smooth_scale=None,
            activation=act_type,
            num_iters=10,
            num_warmup=2,
        )
        return time_us, None
    except Exception as e:
        return float('inf'), str(e)


def benchmark_2stage_combination(data, stage1_kernel, stage1_type, stage2_kernel, 
                                 block_m, topk, q_type, act_type, doweight_stage1):
    """
    Benchmark a complete 2-stage kernel combination with detailed timing breakdown.
    
    2-stage approach splits MoE into three operations:
    - Stage1: input → matmul1 → activation → intermediate (full precision)
    - Quantization: intermediate full precision → quantized intermediate
    - Stage2: quantized intermediate → matmul2 → output
    
    This function benchmarks each component separately to identify bottlenecks.
    
    Args:
        data: Preprocessed test data
        stage1_kernel: ASM kernel name or CK kernel object for stage1
        stage1_type: 'asm' or 'ck' indicating kernel type
        stage2_kernel: CK kernel object for stage2
        block_m: Block size
        topk: Number of experts per token
        q_type: Quantization type
        act_type: Activation function
        doweight_stage1: Whether to apply routing weights in stage1
        
    Returns:
        tuple: (total_us, stage1_us, quant_us, stage2_us, error or None)
    """
    try:
        token = data['a1_qt'].shape[0]
        inter_dim = data['w2_qt'].shape[-1]
        model_dim = data['w2_qt'].shape[1]
        
        torch_quant = aiter.get_torch_quant(q_type)
        sorted_weights_s1 = data['sorted_weights'] if doweight_stage1 else None
        sorted_weights_s2 = None if doweight_stage1 else data['sorted_weights']
        
        # ==========================================
        # BENCHMARK STAGE 1
        # ==========================================
        def stage1_op():
            out1 = torch.empty((token, topk, inter_dim), dtype=data['input'].dtype)
            if stage1_type == 'asm':
                asm_stage1(
                    data['a1_qt'], data['w1_qt_shffle'], data['w2_qt_shffle'],
                    data['sorted_ids'], data['sorted_expert_ids'], data['num_valid_ids'],
                    out1, topk, block_m, stage1_kernel, 0, act_type, q_type,
                    data['a1_scale'], data['w1_scale'], None
                )
            else:
                ck_moe_stage1_fwd(
                    data['a1_qt'], data['w1_qt_shffle'], data['w2_qt_shffle'],
                    data['sorted_ids'], data['sorted_expert_ids'], data['num_valid_ids'],
                    out1, topk, stage1_kernel.name, data['w1_scale_shffle'], data['a1_scale'],
                    block_m, sorted_weights_s1, q_type, act_type
                )
            return out1
        
        out1, stage1_us = run_perftest(stage1_op, num_iters=10, num_warmup=2)
        
        # ==========================================
        # BENCHMARK QUANTIZATION
        # ==========================================
        def quant_op():
            a2_qt, a2_scale = torch_quant(out1, quant_dtype=data['a1_qt'].dtype)
            return a2_qt, a2_scale
        
        (a2_qt_result, a2_scale_result), quant_us = run_perftest(quant_op, num_iters=10, num_warmup=2)
        a2_qt_result = a2_qt_result.view(token, topk, inter_dim)
        
        # ==========================================
        # BENCHMARK STAGE 2
        # ==========================================
        def stage2_op():
            out2 = torch.zeros((token, model_dim), dtype=data['input'].dtype)
            ck_moe_stage2_fwd(
                a2_qt_result, data['w1_qt_shffle'], data['w2_qt_shffle'],
                data['sorted_ids'], data['sorted_expert_ids'], data['num_valid_ids'],
                out2, topk, stage2_kernel.name, data['w2_scale_shffle'], a2_scale_result,
                block_m, sorted_weights_s2, q_type, act_type
            )
            return out2
        
        _, stage2_us = run_perftest(stage2_op, num_iters=10, num_warmup=2)
        
        # Calculate total time
        total_us = stage1_us + quant_us + stage2_us
        
        return total_us, stage1_us, quant_us, stage2_us, None
        
    except Exception as e:
        return float('inf'), float('inf'), float('inf'), float('inf'), str(e)


# ============================================================================
# MAIN BENCHMARKING ORCHESTRATION
# ============================================================================

def benchmark_all_variants(
    token=1087,
    model_dim=4096,
    inter_dim=1536,
    expert=16,
    topk=8,
    dtype=dtypes.bf16,
    q_dtype_a=dtypes.bf16,
    q_dtype_w=dtypes.bf16,
    q_type=QuantType.No,
    act_type=ActivationType.Silu,
    doweight_stage1=False,
    skip_1stage=False,
    only_1stage=False,
):
    """
    Main benchmarking function - tests all kernel variants for a fixed configuration.
    
    For each block size (32, 64, 128):
        1. Enumerates available ASM stage1, CK stage1, and CK stage2 kernels
        2. Tests all combinations of (stage1 + stage2) kernels
        3. Records performance of each combination
    
    Optionally tests 1-stage fused kernels (if not skipped).
    
    Args:
        Configuration parameters for the MoE operation
        skip_1stage: If True, skip 1-stage tests (useful to avoid GPU errors)
        only_1stage: If True, only test 1-stage kernels
        
    Returns:
        list: Benchmark results for all tested combinations
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking Kernel Variants")
    print(f"Configuration:")
    print(f"  token={token}, model_dim={model_dim}, inter_dim={inter_dim}")
    print(f"  expert={expert}, topk={topk}")
    print(f"  quant_type={q_type}, activation={act_type}")
    print(f"{'='*80}\n")
    
    results = []
    block_sizes = [32, 64, 128] if not only_1stage else []
    
    # ============================================================================
    # PART 1: Test 2-Stage Kernel Combinations
    # ============================================================================
    
    for block_m in block_sizes:
        print(f"\nTesting block_m={block_m}...")
        
        # Generate test data for this block size
        data = generate_test_data(
            token, model_dim, inter_dim, expert, topk,
            dtype, q_dtype_a, q_dtype_w, q_type, block_m
        )
        
        # Enumerate available kernels for this block size
        asm1_kernels = get_asm_stage1_kernels_for_2stage(block_m, act_type, q_type, q_dtype_a, True, doweight_stage1)
        ck1_kernels = get_ck_stage1_kernels_for_2stage(block_m, dtype, q_dtype_a, q_dtype_w, q_type, act_type, doweight_stage1)
        ck2_kernels = get_ck_stage2_kernels_for_2stage(block_m, dtype, q_dtype_a, q_dtype_w, q_type, act_type, doweight_stage1)
        
        print(f"  Found {len(asm1_kernels)} ASM stage1, {len(ck1_kernels)} CK stage1, {len(ck2_kernels)} CK stage2 kernels")
        
        # Test all ASM stage1 + CK stage2 combinations
        print(f"  Testing {len(asm1_kernels)*len(ck2_kernels)} ASM+CK combinations...")
        for asm_kernel in asm1_kernels:
            for ck2_kernel in ck2_kernels:
                total_us, stage1_us, quant_us, stage2_us, error = benchmark_2stage_combination(
                    data, asm_kernel, 'asm', ck2_kernel, block_m, topk, q_type, act_type, doweight_stage1
                )
                results.append({
                    'approach': '2-stage',
                    'kernel_types': 'asm-ck',
                    'stage1_kernel': asm_kernel,
                    'stage2_kernel': ck2_kernel.name,
                    'block_m': block_m,
                    'time_us': round(total_us, 2) if total_us != float('inf') else float('inf'),
                    'stage1_us': round(stage1_us, 2) if stage1_us != float('inf') else float('inf'),
                    'quant_us': round(quant_us, 2) if quant_us != float('inf') else float('inf'),
                    'stage2_us': round(stage2_us, 2) if stage2_us != float('inf') else float('inf'),
                    'status': 'passed' if error is None else 'error',
                    'error': error or ''
                })
        
        # Test all CK stage1 + CK stage2 combinations
        print(f"  Testing {len(ck1_kernels)*len(ck2_kernels)} CK+CK combinations...")
        for ck1_kernel in ck1_kernels:
            for ck2_kernel in ck2_kernels:
                total_us, stage1_us, quant_us, stage2_us, error = benchmark_2stage_combination(
                    data, ck1_kernel, 'ck', ck2_kernel, block_m, topk, q_type, act_type, doweight_stage1
                )
                results.append({
                    'approach': '2-stage',
                    'kernel_types': 'ck-ck',
                    'stage1_kernel': ck1_kernel.name,
                    'stage2_kernel': ck2_kernel.name,
                    'block_m': block_m,
                    'time_us': round(total_us, 2) if total_us != float('inf') else float('inf'),
                    'stage1_us': round(stage1_us, 2) if stage1_us != float('inf') else float('inf'),
                    'quant_us': round(quant_us, 2) if quant_us != float('inf') else float('inf'),
                    'stage2_us': round(stage2_us, 2) if stage2_us != float('inf') else float('inf'),
                    'status': 'passed' if error is None else 'error',
                    'error': error or ''
                })
    
    # ============================================================================
    # PART 2: Test 1-Stage Fused Kernels
    # ============================================================================
    
    if not skip_1stage:
        print(f"\nTesting 1-stage fused kernels...")
        fused_kernels = get_1stage_fused_asm_kernels(act_type, q_type, q_dtype_a, True, doweight_stage1)
        print(f"  Found {len(fused_kernels)} 1-stage fused kernels")
    else:
        print(f"\nSkipping 1-stage tests (--skip-1stage flag set)")
        fused_kernels = []
    
    if fused_kernels:
        # Clear GPU state before 1-stage tests to avoid memory corruption
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            # Generate test data (use block_m=32 for 1-stage)
            data = generate_test_data(
                token, model_dim, inter_dim, expert, topk,
                dtype, q_dtype_a, q_dtype_w, q_type, 32
            )
            
            for kernel_name in fused_kernels:
                try:
                    time_us, error = benchmark_1stage_kernel(
                        data, kernel_name, topk, q_type, act_type, q_dtype_a, doweight_stage1
                    )
                    results.append({
                        'approach': '1-stage',
                        'kernel_types': 'asm-fused',
                        'stage1_kernel': kernel_name,
                        'stage2_kernel': 'N/A',
                        'block_m': 32,
                        'time_us': round(time_us, 2) if time_us != float('inf') else float('inf'),
                        'status': 'passed' if error is None else 'error',
                        'error': error or ''
                    })
                except Exception as e:
                    # GPU errors can occur - record and continue
                    results.append({
                        'approach': '1-stage',
                        'stage1_kernel': kernel_name,
                        'stage2_kernel': 'N/A',
                        'block_m': 32,
                        'time_us': float('inf'),
                        'status': 'error',
                        'error': f"GPU error: {str(e)[:100]}"
                    })
                    # Clear GPU state after error
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error setting up 1-stage tests: {e}")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    return results


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """
    Command-line interface for kernel benchmarking tool.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark all MoE kernel variants for a fixed configuration",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Benchmark 2-stage kernels only (recommended)
  python benchmark_kernel_variants.py --skip-1stage
  
  # Benchmark with large expert count
  python benchmark_kernel_variants.py --expert 256 --topk 8 --skip-1stage
  
  # Test only 1-stage kernels
  python benchmark_kernel_variants.py --only-1stage
        """
    )
    
    token=1087,
    model_dim=4096,
    inter_dim=1536,
    expert=16,
    topk=8,

    # Configuration parameters
    parser.add_argument('--token', type=int, default=1087, 
                       help='Number of tokens (default: 1024)')
    parser.add_argument('--model_dim', type=int, default=4096, 
                       help='Model dimension (default: 7168)')
    parser.add_argument('--inter_dim', type=int, default=1536, 
                       help='Intermediate dimension (default: 256)')
    parser.add_argument('--expert', type=int, default=16, 
                       help='Number of experts (default: 8)')
    parser.add_argument('--topk', type=int, default=8, 
                       help='Number of experts to select per token (default: 2)')
    
    # Data types
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp16'],
                       help='Computation data type (default: bf16)')
    parser.add_argument('--quant', type=str, default='No', 
                       choices=['No', 'per_Tensor', 'per_Token'],
                       help='Quantization strategy (default: per_Token)')
    parser.add_argument('--activation', type=str, default='Silu', 
                       choices=['Silu', 'Gelu'],
                       help='Activation function (default: Silu)')
    
    # Output and control
    parser.add_argument('--output', type=str, default='kernel_benchmark_results.csv', 
                       help='Output CSV file (default: kernel_benchmark_results.csv)')
    parser.add_argument('--skip-1stage', action='store_true',
                       help='Skip 1-stage kernel tests (use if GPU errors occur)')
    parser.add_argument('--only-1stage', action='store_true',
                       help='Only test 1-stage kernels (separate run to avoid GPU state issues)')
    
    args = parser.parse_args()
    
    # Convert string arguments to proper types
    dtype = dtypes.bf16 if args.dtype == 'bf16' else dtypes.fp16
    q_type = getattr(QuantType, args.quant)
    act_type = getattr(ActivationType, args.activation)
    q_dtype_a = dtypes.fp8  # Activation quantization type
    q_dtype_w = dtypes.fp8  # Weight quantization type
    
    # Run benchmarks
    results = benchmark_all_variants(
        token=args.token,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        expert=args.expert,
        topk=args.topk,
        dtype=dtype,
        q_dtype_a=q_dtype_a,
        q_dtype_w=q_dtype_w,
        q_type=q_type,
        act_type=act_type,
        doweight_stage1=False,
        skip_1stage=args.skip_1stage,
        only_1stage=args.only_1stage,
    )
    
    # ============================================================================
    # Save Results to CSV
    # ============================================================================
    
    df = pd.DataFrame(results)
    df = df.sort_values('time_us')  # Sort by performance (fastest first)
    
    # Write CSV with configuration metadata as header comments
    with open(args.output, 'w') as f:
        f.write(f"# Kernel Variant Benchmark Results\n")
        f.write(f"# Configuration:\n")
        f.write(f"#   token={args.token}\n")
        f.write(f"#   model_dim={args.model_dim}\n")
        f.write(f"#   inter_dim={args.inter_dim}\n")
        f.write(f"#   expert={args.expert}\n")
        f.write(f"#   topk={args.topk}\n")
        f.write(f"#   dtype={args.dtype}\n")
        f.write(f"#   quant_type={args.quant}\n")
        f.write(f"#   activation={args.activation}\n")
        f.write(f"#   hardware={get_gfx()}\n")
        f.write(f"#\n")
        df.to_csv(f, index=False)
    
    # ============================================================================
    # Print Summary
    # ============================================================================
    
    print(f"\n{'='*80}")
    print(f"Benchmark Results Summary")
    print(f"{'='*80}")
    
    passed = df[df['status'] == 'passed']
    
    if not passed.empty:
        fastest = passed.iloc[0]
        print(f"\n✅ Fastest Kernel Combination:")
        print(f"   Approach: {fastest['approach']}")
        print(f"   Stage1: {fastest['stage1_kernel'][:60]}")
        print(f"   Stage2: {fastest['stage2_kernel'][:60]}")
        print(f"   Block Size: {fastest['block_m']}")
        print(f"   Time: {fastest['time_us']:.2f} us")
        
        print(f"\n📊 Top 5 Kernel Combinations:")
        for idx, row in passed.head(5).iterrows():
            s1 = row['stage1_kernel'][:40]
            s2 = row['stage2_kernel'][:40]
            print(f"   {idx+1}. {s1:40s} + {s2:40s} "
                  f"block_m={row['block_m']}, time={row['time_us']:.2f}us")
    
    print(f"\n📈 Statistics:")
    print(f"   Total combinations tested: {len(df)}")
    print(f"   Successful: {len(passed)}")
    print(f"   Failed: {len(df[df['status'] == 'error'])}")
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

######################################## Imports ######################################## 
from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
import sys

# Add aiter to path for imports
sys.path.insert(0, '/home/upandey/kernelgen/openevolve/TriVolve/aiter')

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

dtype_mapping = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}
######################################## Imports ######################################## 


@triton.autotune(
    configs=[
        # Small matrices - favor smaller blocks for better occupancy
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        
        # Medium matrices - balanced block sizes
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        
        # Large matrices - larger blocks for better compute density
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        
        # Very large matrices - maximize compute throughput
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
        
        # High K dimension cases - optimize for K-reduction
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        
        # Memory bandwidth optimized configs
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _gemm_a16_w16_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    next_M_2,
    # Meta-parameters (now provided by autotuner)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid = remap_xcd(pid, num_pid_m * num_pid_n)
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    if ADD_BIAS:
        accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
        accumulator = tl.broadcast_to(
            accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
        )
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    # Optimized loop with better memory access pattern
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_blocks):
        # Compute mask once for efficiency
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Load the next block of A and B with optimized access pattern
        if EVEN_K:
            a = tl.load(a_ptrs)
            # Use .CA cache modifier for better L2 utilization on AMD
            b = tl.load(b_ptrs, cache_modifier=".ca")
        else:
            k_mask_a = offs_k[None, :] < k_remaining
            k_mask_b = offs_k[:, None] < k_remaining
            a = tl.load(a_ptrs, mask=k_mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=k_mask_b, other=0.0, cache_modifier=".ca")

        # Use higher precision dot product for better accuracy
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W16-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config
            return _get_config._config_dict["default"]["any"]

    bounds = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for bound in bounds:
        if M <= bound and f"M_LEQ_{bound}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"M_LEQ_{bound}"]
    return _get_config._config_dict[key]["any"]


def gemm_a16w16(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 16 bit matmul Y = X x W

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - bias: Vector with shape (N).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(f"GEMM_A16W16: x={tuple(x.shape)} w={tuple(w.shape)}")
    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, K = w.shape
    w = w.T
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # Use autotuning - no manual config needed
    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    # Call kernel with autotuning - parameters are automatically selected
    _gemm_a16_w16_kernel[grid](
        x,
        w,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        next_M_2=triton.next_power_of_2(M),
        ADD_BIAS=(bias is not None),
    )

    return y

##################################################################################################################################################  

import numpy as np
import random
import pytest
from torch.testing import assert_close
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

result_gold = {}

######################################## HELPERS for Eval ######################################## 

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_gemm_tflops(params: Dict, ms: float) -> float:
    """Calculate TFLOPS for GEMM operation."""
    M = params['M']
    N = params['N']
    K = params['K']
    # For GEMM: 2 * M * N * K FLOPs (multiply-add counts as 2 ops)
    flops = 2 * M * N * K
    tflops = flops / (ms / 1000) / 1e12
    return tflops


def calculate_gemm_gbps(params: Dict, ms: float) -> float:
    """Calculate GB/s for GEMM operation."""
    M = params['M']
    N = params['N']
    K = params['K']
    dtype_str = params['dtype_str']
    dtype = dtype_mapping[dtype_str]
    
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Read A (M x K), Read B (K x N), Write C (M x N)
    total_bytes = (M * K + K * N + M * N) * bytes_per_element
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

######################################## HELPERS for Eval ######################################## 


# Test cases inspired by aiter op_tests/test_gemm_a16w16.py
@pytest.mark.parametrize(
    'M,N,K,dtype_str,use_bias',
    [
        # Basic test cases
        (128, 32, 8192, 'bfloat16', False),
        (64, 256, 5120, 'bfloat16', False),
        # Square matrices
        (512, 512, 512, 'bfloat16', False),
        (1024, 1024, 1024, 'bfloat16', False),
        (2048, 2048, 2048, 'bfloat16', False),
        # With different dtypes
        (512, 512, 512, 'float16', False),
        (1024, 1024, 1024, 'float16', False),
        # With bias
        (1024, 1024, 1024, 'bfloat16', True),
        (512, 512, 512, 'float16', True),
        # Different output types
        (256, 256, 256, 'bfloat16', False),
        (256, 256, 256, 'float16', False),
    ]
)
def test_gemm_a16w16(M, N, K, dtype_str, use_bias, request):
    """Test correctness of GEMM A16W16 kernel."""
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    x = torch.randn(M, K, device='cuda', dtype=dtype)
    w = torch.randn(N, K, device='cuda', dtype=dtype)
    bias = torch.randn(N, device='cuda', dtype=dtype) if use_bias else None
    
    # Triton kernel
    y_triton = gemm_a16w16(x, w, bias=bias, dtype=dtype)
    
    # PyTorch reference (F.linear expects weight as (out_features, in_features))
    y_torch = torch.matmul(x, w.T)
    if use_bias:
        y_torch = y_torch + bias
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    # Save result
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = y_triton.clone().detach().cpu()
    
    # Check correctness
    torch.set_printoptions(profile='full')
    assert_close(y_triton, y_torch, rtol=1e-2, atol=1e-2, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "gemm_a16w16_perf"


@pytest.mark.parametrize(
    'M,N,K,dtype_str,use_bias',
    [
        # Performance test cases inspired by aiter tests
        (128, 32, 8192, 'bfloat16', False),
        (64, 256, 5120, 'bfloat16', False),
        (1024, 1024, 1024, 'bfloat16', False),
        (2048, 2048, 2048, 'bfloat16', False),
        (4096, 4096, 4096, 'bfloat16', False),
        (1024, 1024, 1024, 'float16', False),
        (2048, 2048, 2048, 'float16', False),
    ]
)
def test_performance(M, N, K, dtype_str, use_bias, request):
    """Benchmark performance of GEMM A16W16 kernel."""
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    x = torch.randn(M, K, device='cuda', dtype=dtype)
    w = torch.randn(N, K, device='cuda', dtype=dtype)
    bias = torch.randn(N, device='cuda', dtype=dtype) if use_bias else None
    
    # Create op_lambda for benchmarking
    op_lambda = lambda: gemm_a16w16(x, w, bias=bias, dtype=dtype)
    
    # Benchmarking
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(
        op_callable=op_lambda,
        op_name=OP_NAME_FOR_BENCHMARK,
        config=bench_config
    )
    
    current_params = {
        "M": M, "N": N, "K": K,
        "dtype_str": dtype_str,
        "use_bias": use_bias
    }
    
    benchmarker.run_benchmark(
        current_params_dict=current_params,
        gbps_calculator=calculate_gemm_gbps,
        tflops_calculator=calculate_gemm_tflops
    )


######################################## HELPERS for Eval ########################################     

def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################
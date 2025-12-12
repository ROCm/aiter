"""
Benchmark script for SageAttention and FA3 FP8 kernels using captured inputs from CogVideo inference.

Usage:
    # First, run inference with input capture:
    python op_tests/sagev1_tests/sageattn_cogvideo.py --attention_type sage --save_inputs --input_dir ./captured_inputs

    # Then run this benchmark with the captured inputs:
    python op_tests/op_benchmarks/triton/bench_cogvideo.py --input_dir ./captured_inputs -sage_fa3 -metric throughput
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import torch
import os
import glob
import argparse
import logging

import triton

from aiter.ops.triton.mha_v3 import flash_attn_fp8_func
from aiter.ops.triton.mha_v3 import flash_attn_func as flash_attn_v3_func
from aiter.ops.triton.mha import flash_attn_func as flash_attn_v2_func
from aiter.ops.triton.attn_qk_int8_per_block import per_block_int8
from aiter.ops.triton.sage_v1 import sage_attn_v1_func
from aiter.ops.triton._triton_kernels.sage_attn_triton_amd import get_fwd_configs
from op_tests.op_benchmarks.triton.bench_diffusion_attention import qk_int8_forward_func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_metadata(input_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata file if it exists.
    
    Args:
        input_dir: Directory containing captured files
        
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_files = glob.glob(os.path.join(input_dir, "*_metadata.pt"))
    if metadata_files:
        return torch.load(metadata_files[0], weights_only=False)
    return None


def load_captured_inputs(input_dir: str, max_inputs: Optional[int] = None, 
                         sample_rate: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load captured input tensors from disk.
    
    Args:
        input_dir: Directory containing captured .pt files
        max_inputs: Maximum number of inputs to load (None = all)
        sample_rate: Sample every Nth input (None = load all)
        
    Returns:
        List of dictionaries containing q, k, v tensors and metadata
    """
    input_files = sorted(glob.glob(os.path.join(input_dir, "*_input_*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No captured input files found in {input_dir}")
    
    # Apply sampling if requested
    if sample_rate is not None and sample_rate > 1:
        input_files = input_files[::sample_rate]
        logger.info(f"Sampling every {sample_rate}th input: {len(input_files)} files selected")
    
    # Apply max limit if requested
    if max_inputs is not None and len(input_files) > max_inputs:
        input_files = input_files[:max_inputs]
        logger.info(f"Limiting to first {max_inputs} inputs")
    
    inputs = []
    for i, f in enumerate(input_files):
        data = torch.load(f, weights_only=False)
        inputs.append(data)
        if i < 5 or i == len(input_files) - 1:  # Log first 5 and last
            logger.info(f"Loaded [{i}] {os.path.basename(f)}: q={tuple(data['q_shape'])}, k={tuple(data['k_shape'])}, v={tuple(data['v_shape'])}")
        elif i == 5:
            logger.info(f"  ... (loading {len(input_files) - 6} more files)")
    
    logger.info(f"Loaded {len(inputs)} input configurations for benchmarking")
    
    # Load and display metadata if available
    metadata = load_metadata(input_dir)
    if metadata:
        total_calls = metadata.get('total_calls', 'unknown')
        saved_count = metadata.get('saved_count', total_calls)
        max_captures = metadata.get('max_captures', None)
        unique_shapes = len(metadata.get('captured_shapes', {}))
        
        limit_info = f" (limit: {max_captures})" if max_captures else ""
        logger.info(f"Metadata: total calls = {total_calls}, saved = {saved_count}{limit_info}, "
                   f"unique shapes = {unique_shapes}")
    
    return inputs


def sage_fa3_forward_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
    """
    Create a lambda for benchmarking sage_attn_v1_func kernel (fused on fa3).
    
    Args:
        q, k, v: Input tensors in BHSD format (batch, heads, seqlen, dim)
        is_causal: Whether to use causal masking
        
    Returns:
        Lambda function that executes the attention
    """
    # Convert from BHSD to BSHD (NHD) for sage_attn_v1_func
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    
    config, _ = get_fwd_configs(False)
    assert len(config) == 1, f"Number of best config is expected to be 1, got {len(config)}"
    config = config[0].all_kwargs()
    BLKQ = config["BLOCK_M"]
    BLKK = config["BLOCK_N"]

    head_dim = q_bshd.shape[-1]
    softmax_scale = head_dim**-0.5
    k_mean = None
    
    # Quantization with softmax scale and RCP_LN2 
    q_int8, q_descale, k_int8, k_descale, _ = per_block_int8(
        q_bshd, k_bshd, km=k_mean, sm_scale=softmax_scale, BLKQ=BLKQ, BLKK=BLKK, tensor_layout="NHD"
    )
    v_fp16 = v_bshd.to(torch.float16)
    
    return lambda: sage_attn_v1_func(
        q_int8, 
        k_int8, 
        v_fp16,
        q_descale,
        k_descale,
        causal=is_causal,
    )


def fa3_fp8_forward_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          softmax_scale: Optional[float] = None, is_causal: bool = False):
    """
    Create a lambda for benchmarking flash_attn_fp8_func kernel.
    
    Args:
        q, k, v: Input tensors in BHSD format (batch, heads, seqlen, dim)
        softmax_scale: Softmax scale (defaults to 1/sqrt(head_dim))
        is_causal: Whether to use causal masking
        
    Returns:
        Lambda function that executes the attention
    """
    # Convert from BHSD to BSHD for flash_attn_fp8_func
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    
    return lambda: flash_attn_fp8_func(q_bshd, k_bshd, v_bshd, softmax_scale=softmax_scale, causal=is_causal)


def sdpa_forward_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                       softmax_scale: Optional[float] = None, is_causal: bool = False):
    """
    Create a lambda for benchmarking PyTorch's native scaled_dot_product_attention.
    
    Args:
        q, k, v: Input tensors in BHSD format (batch, heads, seqlen, dim)
        softmax_scale: Softmax scale (defaults to 1/sqrt(head_dim))
        is_causal: Whether to use causal masking
        
    Returns:
        Lambda function that executes the attention
    """
    # PyTorch SDPA expects BHSD format, which matches our input
    return lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=softmax_scale
    )


def fa2_forward_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                       softmax_scale: Optional[float] = None, is_causal: bool = False):
    """
    Create a lambda for benchmarking flash_attn_func (fa2) kernel.
    
    Args:
        q, k, v: Input tensors in BHSD format (batch, heads, seqlen, dim)
        softmax_scale: Softmax scale (defaults to 1/sqrt(head_dim))
        is_causal: Whether to use causal masking
        
    Returns:
        Lambda function that executes the attention
    """
    # Convert from BHSD to BSHD for flash_attn_v2_func
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    
    return lambda: flash_attn_v2_func(q_bshd, k_bshd, v_bshd, dropout_p=0.0, softmax_scale=softmax_scale, causal=is_causal)


def fa3_forward_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                       softmax_scale: Optional[float] = None, is_causal: bool = False):
    """
    Create a lambda for benchmarking flash_attn_func v3 (fa3) kernel.
    
    Args:
        q, k, v: Input tensors in BHSD format (batch, heads, seqlen, dim)
        softmax_scale: Softmax scale (defaults to 1/sqrt(head_dim))
        is_causal: Whether to use causal masking
        
    Returns:
        Lambda function that executes the attention
    """
    # Convert from BHSD to BSHD for flash_attn_v3_func
    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()
    
    return lambda: flash_attn_v3_func(q_bshd, k_bshd, v_bshd, softmax_scale=softmax_scale, causal=is_causal)


def create_benchmark_configs(inputs: List[Dict[str, Any]], args):
    """
    Create triton.testing.Benchmark configurations from captured inputs.
    
    Args:
        inputs: List of captured input dictionaries
        args: Parsed command-line arguments
        
    Returns:
        List of triton.testing.Benchmark objects
    """
    # Extract x_vals from loaded inputs
    x_vals_list = []
    for i, inp in enumerate(inputs):
        # Shape from BHSD format: (batch, heads, seqlen, dim)
        q_shape = inp['q_shape']
        batch = q_shape[0]
        hq = q_shape[1]
        seq_q = q_shape[2]
        d_head = q_shape[3]
        
        k_shape = inp['k_shape']
        hk = k_shape[1]
        seq_k = k_shape[2]
        
        v_shape = inp['v_shape']
        d_head_v = v_shape[3]
        
        # Get the original call index from inference (for tracking pipeline step)
        original_call_idx = inp.get('call_idx', i)
        
        x_vals_list.append((i, original_call_idx, batch, hq, hk, seq_q, seq_k, d_head, d_head_v))
    
    x_names = ["INPUT_IDX", "CALL_IDX", "BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K", "D_HEAD", "D_HEAD_V"]
    
    # Determine line_vals based on metric
    if args.metric == "all":
        line_vals = ["time(ms)", "throughput(TFLOPS)", "bandwidth(GB/s)", "arithmetic_intensity(FLOP/byte)"]
    else:
        metric_map = {
            "time": "time(ms)",
            "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
            "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        }
        line_vals = [metric_map.get(args.metric, "throughput(TFLOPS)")]
    
    plot_name = f"bench_cogvideo_{args.kernel}"
    
    configs = [
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-"), ("blue", "-")],
            ylabel="",
            plot_name=plot_name,
            args={
                "inputs": inputs,
                "kernel": args.kernel,
                "causal": args.causal,
                "tensor_layout": args.tensor_layout,
                "no_k_smooth": args.no_k_smooth,
            },
        )
    ]
    return configs


def run_benchmark(args):
    """
    Main benchmark function that loads inputs and runs the benchmark.
    """
    torch.manual_seed(42)
    
    # Load captured inputs with optional sampling
    inputs = load_captured_inputs(
        args.input_dir, 
        max_inputs=args.max_inputs,
        sample_rate=args.sample_rate
    )
    
    @triton.testing.perf_report(create_benchmark_configs(inputs, args))
    def bench_attention(
        INPUT_IDX,
        CALL_IDX,
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        inputs,
        kernel,
        causal,
        tensor_layout,
        no_k_smooth,
        provider,
        device="cuda",
    ):
        """
        Benchmark function for attention kernels using captured inputs.
        INPUT_IDX: Index in the loaded inputs list
        CALL_IDX: Original call index from inference (pipeline step)
        """
        # Get the input tensors for this configuration
        inp = inputs[INPUT_IDX]
        
        # Load tensors to GPU
        q = inp['q'].to(device)
        k = inp['k'].to(device)
        v = inp['v'].to(device)
        
        # Get kwargs from capture
        kwargs = inp.get('kwargs', {})
        is_causal = kwargs.get('is_causal', causal)
        softmax_scale = kwargs.get('softmax_scale', None)
        
        # Calculate FLOPS: 2 * batch * heads * seq_q * seq_k * (head_dim + head_dim_v)
        total_flops = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)
        
        # Select kernel and create benchmark function
        if kernel == "sagev1":
            # Captured inputs are in BHSD/HND format (batch, heads, seq, dim)
            # qk_int8_forward_func expects NHD format (batch, seq, heads, dim), so transpose first
            q_nhd = q.transpose(1, 2).contiguous()
            k_nhd = k.transpose(1, 2).contiguous()
            v_nhd = v.transpose(1, 2).contiguous()
            # Note: qk_int8_forward_func only accepts (q, k, v, tensor_layout, sm_scale, k_smooth)
            # After transpose, tensors are in NHD format
            fn = qk_int8_forward_func(q_nhd, k_nhd, v_nhd, tensor_layout="NHD", 
                                       sm_scale=softmax_scale, k_smooth=not no_k_smooth)
        elif kernel == "sage_fa3":
            fn = sage_fa3_forward_func(q, k, v, is_causal=is_causal)
        elif kernel == "sdpa":
            fn = sdpa_forward_func(q, k, v, softmax_scale=softmax_scale, is_causal=is_causal)
        elif kernel == "fa2":
            fn = fa2_forward_func(q, k, v, softmax_scale=softmax_scale, is_causal=is_causal)
        elif kernel == "fa3":
            fn = fa3_forward_func(q, k, v, softmax_scale=softmax_scale, is_causal=is_causal)
        elif kernel == "fa3_fp8":
            fn = fa3_fp8_forward_func(q, k, v, softmax_scale=softmax_scale, is_causal=is_causal)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        
        # Run benchmark
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        
        # Calculate memory transfer
        # sage_fa3 and sagev1 use int8 quantized Q/K; fa3_fp8 uses fp8; fa2 and fa3 use original dtype
        q_element_size = 1 if kernel in ["sagev1", "sage_fa3", "fa3_fp8"] else q.element_size()
        k_element_size = 1 if kernel in ["sagev1", "sage_fa3", "fa3_fp8"] else k.element_size()
        v_element_size = 1 if kernel == "fa3_fp8" else v.element_size()
        # Output is always in bfloat16 (2 bytes), regardless of input quantization
        o_element_size = 2
        
        total_num_tokens_q = BATCH * N_CTX_Q
        total_num_tokens_k = BATCH * N_CTX_K
        q_size = total_num_tokens_q * HQ * D_HEAD * q_element_size
        k_size = total_num_tokens_k * HK * D_HEAD * k_element_size
        v_size = total_num_tokens_k * HK * D_HEAD_V * v_element_size
        o_size = total_num_tokens_q * HQ * D_HEAD_V * o_element_size
        
        mem = q_size + k_size + v_size + o_size
        
        # Return appropriate metric
        if "ms" in provider:
            return ms
        elif "TFLOPS" in provider:
            return total_flops / ms * 1e-9
        elif "GB/s" in provider:
            return mem / ms * 1e-6
        elif "arithmetic_intensity" in provider:
            return total_flops / mem
        
        return ms  # default
    
    bench_attention.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SageAttention and FA3 FP8 kernels using captured CogVideo inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing captured input .pt files from sageattn_cogvideo.py --save_inputs",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="sagev1",
        choices=["sagev1", "sage_fa3", "sdpa", "fa2", "fa3", "fa3_fp8"],
        help="Kernel to benchmark: sagev1 (qk_int8_per_block), sage_fa3 (sage_attn_v1_func fused on fa3), sdpa (PyTorch native), fa2 (flash_attn_func v2), fa3 (flash_attn_func v3), fa3_fp8",
    )
    parser.add_argument(
        "-metric",
        type=str,
        default="throughput",
        choices=["all", "time", "throughput", "bandwidth", "arithmetic_intensity"],
        help="Metric to report",
    )
    parser.add_argument(
        "-causal",
        action="store_true",
        default=False,
        help="Use causal masking (overrides captured kwargs)",
    )
    parser.add_argument(
        "-no_k_smooth",
        action="store_true",
        default=False,
        help="Disable key smoothing for sagev1 kernel",
    )
    parser.add_argument(
        "--tensor_layout",
        type=str,
        default="HND",
        choices=["HND", "NHD"],
        help="Tensor layout for sagev1 kernel",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write performance results to CSV file",
    )
    parser.add_argument(
        "--max_inputs",
        type=int,
        default=None,
        help="Maximum number of inputs to benchmark (None = all)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="Sample every Nth input for benchmarking (None = all)",
    )
    # Shorthand flags
    parser.add_argument("-sage_fa3", action="store_true", help="Shorthand for --kernel sage_fa3")
    parser.add_argument("-sagev1", action="store_true", help="Shorthand for --kernel sagev1")
    parser.add_argument("-sdpa", action="store_true", help="Shorthand for --kernel sdpa")
    parser.add_argument("-fa2", action="store_true", help="Shorthand for --kernel fa2")
    parser.add_argument("-fa3", action="store_true", help="Shorthand for --kernel fa3")
    parser.add_argument("-fa3_fp8", action="store_true", help="Shorthand for --kernel fa3_fp8")
    
    return parser.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(7)
    # Handle shorthand flags
    if args.sage_fa3:
        args.kernel = "sage_fa3"
    elif args.sagev1:
        args.kernel = "sagev1"
    elif args.sdpa:
        args.kernel = "sdpa"
    elif args.fa2:
        args.kernel = "fa2"
    elif args.fa3:
        args.kernel = "fa3"
    elif args.fa3_fp8:
        args.kernel = "fa3_fp8"
    
    logger.info(f"Benchmarking kernel: {args.kernel}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Metric: {args.metric}")
    
    run_benchmark(args)


if __name__ == "__main__":
    main()


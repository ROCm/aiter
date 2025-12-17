from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import torch
import os
import glob

import sys
import argparse
import itertools
import aiter
import triton

from aiter.ops.triton.mha import (
    flash_attn_func,
)
from aiter.ops.triton.attn_qk_int8_per_block import (
    attn_qk_int8_per_block,
    per_block_int8,
    _get_config,
)
from aiter.test_mha_common import (
    attention_ref,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.test_mha import check_attention_outputs
from op_tests.op_benchmarks.triton.mha_correctness_utils import (
    primary_output,
    print_output_comparison_stats,
)
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3
from aiter.ops.triton.mha_v3 import _quantize_bshd

from aiter.ops.triton.fav3_sage import (
    fav3_sage_func,
)
from aiter.ops.triton._triton_kernels.sage_attn_triton_amd import (
    get_fwd_configs,
    quantize_v_fp8
)


# test_mha.py configures root logging to DEBUG on import; reset to INFO to avoid noisy deps
import logging

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def load_captured_inputs(input_dir: str) -> List[Dict[str, Any]]:
    """
    Load captured input tensors from disk.

    Args:
        input_dir: Directory containing captured .pt files

    Returns:
        List of dictionaries containing q, k, v tensors and metadata
    """
    input_files = sorted(glob.glob(os.path.join(input_dir, "*_input_*.pt")))
    if not input_files:
        raise FileNotFoundError(f"No captured input files found in {input_dir}")

    inputs = []
    for i, f in enumerate(input_files):
        data = torch.load(f, weights_only=False)
        inputs.append(data)
        # logger.info(f"Loaded [{i}] {os.path.basename(f)}: q={tuple(data['q_shape'])}")

    logger.info(f"Loaded {len(inputs)} captured inputs for benchmarking")
    return inputs


# taken from mha_v3.py
def fav3_fp8_forward_func(
    q: torch.Tensor,  # High precision (BF16/FP32)
    k: torch.Tensor,  # High precision (BF16/FP32)
    v: torch.Tensor,  # High precision (BF16/FP32)
    softmax_scale: Optional[float],
    causal: bool,
    window_size: Tuple[int, int],
    attention_chunk: int,
    softcap: float,
    deterministic: bool,
    sm_margin: int,
):
    batch, seqlen, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    # Quantize inputs to FP8
    fp8_dtype = torch.float8_e4m3fnuz

    # For GQA/MQA: quantize query with grouped scaling
    group_size = num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None
    q_fp8, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
    k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
    v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)

    # Verify descale shapes for GQA/MQA
    assert q_descale.shape == (
        batch,
        num_kv_heads,
    ), f"q_descale shape {q_descale.shape} != expected {(batch, num_kv_heads)}"
    assert k_descale.shape == (
        batch,
        num_kv_heads,
    ), f"k_descale shape {k_descale.shape} != expected {(batch, num_kv_heads)}"
    assert v_descale.shape == (
        batch,
        num_kv_heads,
    ), f"v_descale shape {v_descale.shape} != expected {(batch, num_kv_heads)}"

    # Derive softmax scale if not provided
    if softmax_scale is None:
        softmax_scale = head_dim ** (-0.5)

    # Validate unsupported features
    if attention_chunk not in (0, 1):
        raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
    if softcap != 0.0:
        raise NotImplementedError("softcap not implemented in FP8 high-precision API")
    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in FP8 high-precision API"
        )

    # Call flash attention forward
    return lambda: flash_attn_3.fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        None,
        None,
        None,
        None,  # k_new, v_new, qv, out
        None,
        None,
        None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
        None,
        None,
        None,
        None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
        None,
        None,
        None,  # page_table, kv_batch_idx, leftpad_k
        None,
        None,
        None,  # rotary_cos, rotary_sin, seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        attention_chunk,
        softcap,
        False,  # rotary_interleaved
        None,
        1,
        None,
        sm_margin,  # scheduler_metadata, num_splits, pack_gqa, sm_margin
    )


def sagev1_forward_func(q, k, v, tensor_layout, sm_scale, k_smooth=True):
    # tensor_layout for attn_qk_int8_per_block kernel: "HND" (batch, heads, seq, dim) or "NHD" (batch, seq, heads, dim)
    # tensor_layout = args.sagev1_layout
    config = _get_config()
    # Original tensors are in NHD format (batch, seq, heads, dim)
    if tensor_layout == "HND":
        # Convert from NHD to HND for better memory access
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous().to(torch.float16)
    else:  # NHD
        # Keep original layout
        v = v.to(torch.float16)
    # k_mean = k.mean(dim=1, keepdim=True) if k_smooth else None # provide km as None and it gets computed inside per_block_int8
    # sm_scale = D_HEAD**-0.5
    q, q_scale, k, k_scale, _ = per_block_int8(
        q,
        k,
        km=None,
        BLKQ=config["BLOCK_SIZE_M"],
        BLKK=config["BLOCK_SIZE_N"],
        sm_scale=sm_scale,
        tensor_layout=tensor_layout,
        smooth_k=k_smooth,
    )
    q_scale = q_scale.to(torch.float32).unsqueeze(-1).contiguous()
    k_scale = k_scale.to(torch.float32).unsqueeze(-1).contiguous()
    return lambda: attn_qk_int8_per_block(
        q,
        k,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=torch.bfloat16,
        config=config,
    )


def fav2_forward_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: Optional[float],
    causal: bool,
    return_lse: bool,
    return_attn_probs: bool,
):
    return lambda: flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )


def fav3_sage_forward_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    inference_mode: bool,  # not return softmax_lse
):
    config = get_fwd_configs(False)
    # assert (
    #     len(config) == 1
    # ), f"Number of best config is expected to be 1, got {len(config)}"
    # config = config[0].all_kwargs()
    BLKQ = config["BLOCK_M"]
    BLKK = config["BLOCK_N"]

    head_dim = q.shape[-1]
    softmax_scale = head_dim**-0.5
    k_mean = None
    ## following quantization already considered softmax scale and RCP_LN2
    q_int8, q_descale, k_int8, k_descale, _ = per_block_int8(
        q,
        k,
        km=k_mean,
        sm_scale=softmax_scale,
        BLKQ=BLKQ,
        BLKK=BLKK,
        tensor_layout="NHD",
    )

    fp8_dtype = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(fp8_dtype).max
    v_fp8, v_descale = quantize_v_fp8(v, FP8_MAX, BLKK=BLKK, tensor_layout="NHD")

    return lambda: fav3_sage_func(
        q_int8,
        k_int8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        FP8_MAX,
        causal=causal,
        inference_mode=inference_mode,
    )


def nonvarlen_benchmark_configs():
    batch_sizes = [1, 4, 16]
    N_HEADS = [16, 48]
    seq_len_q = [1, 1024, 4096]
    seq_len_k = [163, 8192]
    configs = list(itertools.product(batch_sizes, N_HEADS, seq_len_q, seq_len_k))
    configs = [
        (batch_size, N_HEAD, N_HEAD, seq_len_q, seq_len_k)
        for batch_size, N_HEAD, seq_len_q, seq_len_k in configs
    ]
    return configs


def create_benchmark_configs(args):
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    head_size_v = head_size if not args.dv else args.dv
    mode = args.mode
    x_names = ["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"]
    causal = False

    configs = []
    plot_name = get_caller_name_no_ext()
    extra_args = {
        "D_HEAD": head_size,
        "D_HEAD_V": head_size_v,
        "dtype": dtype,
        "causal": causal,
        "mode": mode,
    }
    x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    unit = ""
    line_vals = [
        "time(ms)",
        "throughput(TFLOPS)",
        "bandwidth(GB/s)",
        "arithmetic_intensity(FLOP/byte)",
    ]

    # if comparing to reference, or specific metric provided, adjust line_vals accordingly
    if args.compare_to_ref or (args.metric and args.metric != "all"):
        if args.compare_to_ref:
            line_vals = [
                "time(ms)"
            ]  # avoid redundant runs of other metrics when comparing to reference. default to time only.
        else:
            metric_map = {
                "time": "time(ms)",
                "throughput": "throughput(TFLOPS)",
                "bandwidth": "bandwidth(GB/s)",
                "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
            }
            line_vals = [metric_map[args.metric]]

    configs.append(
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red", "-"), ("green", "-"), ("yellow", "-"), ("blue", "-")],
            ylabel=unit,
            plot_name=plot_name,
            args=extra_args,
        )
    )
    return configs


def create_benchmark_configs_from_captured(inputs: List[Dict[str, Any]], args):
    """
    Create triton.testing.Benchmark configurations from captured inputs.

    Captured inputs are in BHSD format (batch, heads, seqlen, dim).
    """
    # Extract x_vals from loaded inputs
    x_vals_list = []
    for i, inp in enumerate(inputs):
        # Shape from BSHD format: (batch, seqlen, heads, dim)
        batch, seq_q, hq, d_head = inp["q"].shape
        _, seq_k, hk, _ = inp["k"].shape
        d_head_v = inp["v_shape"][-1]

        x_vals_list.append((i, batch, seq_q, seq_k, hq, hk, d_head, d_head_v))

    x_names = [
        "INPUT_IDX",
        "BATCH",
        "N_CTX_Q",
        "N_CTX_K",
        "HQ",
        "HK",
        "D_HEAD",
        "D_HEAD_V",
    ]

    # Determine line_vals based on metric
    if args.metric == "all" or args.metric is None:
        line_vals = [
            "time(ms)",
            "throughput(TFLOPS)",
            "bandwidth(GB/s)",
            "arithmetic_intensity(FLOP/byte)",
        ]
    else:
        metric_map = {
            "time": "time(ms)",
            "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
            "arithmetic_intensity": "arithmetic_intensity(FLOP/byte)",
        }
        line_vals = [metric_map.get(args.metric, "throughput(TFLOPS)")]

    plot_name = "bench_diffusion_attention_captured"

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
                "causal": False,
                "mode": args.mode,
            },
        )
    ]
    return configs


def bench_kernel(q, k, v, args, provider):
    # Default softmax scale
    BATCH, N_CTX_Q, HQ, D_HEAD = q.shape
    _, N_CTX_K, HK, D_HEAD_V = v.shape
    softmax_scale = 1.0 / (D_HEAD**0.5)
    k_smooth = not args.no_k_smooth

    # FLOPS calculation variables
    total_flops = 0.0
    total_flops += 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)
    if args.fav3_fp8:  #  fav3 fp8
        fn = fav3_fp8_forward_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            deterministic=False,
            sm_margin=0,
        )
    elif args.sagev1:  # sage v1
        fn = sagev1_forward_func(
            q,
            k,
            v,
            tensor_layout=args.sagev1_layout,
            sm_scale=softmax_scale,
            k_smooth=k_smooth,
        )
    elif args.fav3_sage:  # sage v1, fused on fav3
        fn = fav3_sage_forward_func(
            q,
            k,
            v,
            causal=False,
            inference_mode=True,
        )
    else:  # fav2 (no quantization)
        fn = fav2_forward_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            return_lse=False,
            return_attn_probs=False,
        )

    ms = triton.testing.do_bench(fn)

    if args.compare_to_ref:
        current_output = fn()
        assert current_output is not None
        current_primary = primary_output(current_output)

        def reference_output():
            if args.ref == "fav3_fp8":  #  fav3 fp8
                fn = fav3_fp8_forward_func(
                    q,
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=False,
                    window_size=(-1, -1),
                    attention_chunk=0,
                    softcap=0.0,
                    deterministic=False,
                    sm_margin=0,
                )
            elif args.ref == "sagev1":  # sage v1
                fn = sagev1_forward_func(
                    q,
                    k,
                    v,
                    tensor_layout=args.sagev1_layout,
                    sm_scale=softmax_scale,
                    k_smooth=k_smooth,
                )
            elif args.ref == "fav2":  # fav2 (no quantization)
                fn = fav2_forward_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=False,
                    return_lse=False,
                    return_attn_probs=False,
                )
            else:
                fn = lambda: attention_ref(
                    q, k, v, dropout_p=0.0, dropout_mask=None, causal=False
                )
            return fn()

        reference_primary = primary_output(reference_output())
        check_attention_outputs(current_primary, reference_primary, fp8=False)
        if args.print_compare_stats:
            print_output_comparison_stats(current_primary, reference_primary)

    q_element_size = (
        1 if args.sagev1 or args.fav3_fp8 or args.fav3_sage else q.element_size()
    )
    k_element_size = (
        1 if args.sagev1 or args.fav3_fp8 or args.fav3_sage else k.element_size()
    )
    v_element_size = 1 if args.fav3_fp8 else v.element_size()

    total_num_tokens_q = BATCH * N_CTX_Q
    total_num_tokens_k = BATCH * N_CTX_K
    q_size = total_num_tokens_q * HQ * D_HEAD * q_element_size
    k_size = total_num_tokens_k * HK * D_HEAD * k_element_size
    v_size = total_num_tokens_k * HK * D_HEAD_V * v_element_size
    o_size = total_num_tokens_q * HQ * D_HEAD_V * q_element_size

    # read q, k, v
    mem_read = q_size + k_size + v_size
    # write o
    mem_write = o_size
    mem = mem_read + mem_write

    # return ms
    if "ms" in provider:
        return ms
    elif "TFLOPS" in provider:
        return total_flops / ms * 1e-9
    elif "GB/s" in provider:  # GB/s
        return mem / ms * 1e-6
    elif "arithmetic_intensity" in provider:
        return total_flops / mem
    return ms


def run_benchmark_captured(args):
    """
    Run benchmark using captured inputs from disk.
    Captured inputs are in BHSD format and need to be transposed to BSHD for kernels.
    """
    torch.manual_seed(20)

    # Load captured inputs
    inputs = load_captured_inputs(args.captured_dir)
    # logger.info(f"Loaded {len(inputs)} captured inputs for benchmarking")

    @triton.testing.perf_report(create_benchmark_configs_from_captured(inputs, args))
    def bench_mha_captured(
        INPUT_IDX,
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        inputs,
        causal,
        mode,
        provider,
        device="cuda",
    ):
        """
        Benchmark function for attention kernels using captured inputs.
        INPUT_IDX: Index in the loaded inputs list
        """
        # Get the input tensors for this configuration
        inp = inputs[INPUT_IDX]

        # Load tensors to GPU - captured inputs are in BSHD format (batch, seq, heads, dim)
        q = inp["q"].to(device)
        k = inp["k"].to(device)
        v = inp["v"].to(device)
        return bench_kernel(q, k, v, args, provider)

    bench_mha_captured.run(save_path="." if args.o else None, print_data=True)


saved_output_keys = set()


def run_benchmark(args):
    torch.manual_seed(20)

    @triton.testing.perf_report(create_benchmark_configs(args))
    def bench_mha(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        dtype,
        causal,
        mode,
        provider,
        dropout=0.0,
        model=None,
        sm_scale=None,
        device="cuda",
    ):
        """
        Benchmark function for attention kernels with generated random inputs.
        """
        assert args.sagev1_layout == "NHD"
        assert args.layout == "bshd"
        assert dropout <= 0.0, "Dropout not supported in this benchmark."
        assert causal == False, "Causal not supported in this benchmark."

        # Generate base inputs
        q = torch.randn((BATCH, N_CTX_Q, HQ, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, N_CTX_K, HK, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, N_CTX_K, HK, D_HEAD_V), device=device, dtype=dtype)
        q.requires_grad = False
        k.requires_grad = False
        v.requires_grad = False

        return bench_kernel(q, k, v, args, provider)

    bench_mha.run(save_path="." if args.o else None, print_data=True)


def supported_layouts():
    layouts = (
        "bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]. "
        "thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]. "
    )
    return layouts


# argparse lacks support for boolean argument type (sigh...)
def str2bool(v):
    if isinstance(v, bool) or v is None:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = get_parser(kernel_name="FlashAttention")
    parser.add_argument(
        "-mode", type=str, default="fwd", help="fwd:forward kernel, bwd:backward kernel"
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument(
        "-equal_seqlens",
        action="store_true",
        default=False,
        help="If specified, uses equal sequence lengths with thd layout, i.e t = b * sq",
    )
    parser.add_argument(
        "-d",
        type=int,
        default=0,
        help="Q and K head size, if -dv is absent then -d specifies V head size too",
    )
    parser.add_argument("-dv", type=int, default=0, help="optional V head size")
    # parser.add_argument("-causal", type=str2bool, default=None)
    parser.add_argument(
        "-fav3_fp8",
        action="store_true",
        default=False,
        help="Use fav3 fp8 kernel: per tensor quantization, QK and PV in fp8, accumulation in fp32",
    )
    parser.add_argument(
        "-sagev1",
        action="store_true",
        default=False,
        help="Use sage v1 kernel: per block quantization, QK in int8, PV and accumulation in fp16",
    )
    parser.add_argument(
        "-fav3_sage",
        action="store_true",
        default=False,
        help="fav3 fp8 sagev1 hybrid kernel: per block quantization for Q/K, per tensor quantization for V, QK in int8, PV in fp8, accumulation in fp32.",
    )
    parser.add_argument("-no_k_smooth", action="store_true", default=False)
    parser.add_argument(
        "-sagev1_layout",
        type=str,
        default="NHD",
        choices=["HND", "NHD"],
        help="Tensor layout for sagev1: HND (batch, heads, seq, dim) or NHD (batch, seq, heads, dim). Default: NHD.",
    )
    parser.add_argument("-quantize_p", action="store_true", default=False)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("-bench_torch", action="store_true", default=False)
    parser.add_argument("-fused_bwd", action="store_true", default=False)
    parser.add_argument("-print_vgpr", action="store_true", default=False)
    parser.add_argument("-print_compare_stats", action="store_true", default=False)

    parser.add_argument("--layout", type=str, default=None, help=supported_layouts())
    parser.add_argument(
        "-ref",
        type=str,
        default=None,
        help="fp8, qk_int8, fav2 or torch ref (default).",
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="throughput",
        choices=["all", "time", "throughput", "bandwidth", "arithmetic_intensity"],
        default=None,
        help="Metrics for the kernel benchmark.",
    )
    parser.add_argument(
        "-persistent",
        nargs="?",
        const="fixed",
        choices=["fixed", "dynamic"],
        default=None,
        help="Enable persistent kernels. Use '-persistent dynamic' for dynamic scheduling of the tiles.",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        default=False,
        help="Store one representative tensor per benchmark configuration for later comparisons.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store tensors when --save_output is used.",
    )
    parser.add_argument(
        "--compare_to_ref",
        action="store_true",
        help="also execute the reference kernel (-ref) and assert outputs match.",
    )
    # Captured input loading
    parser.add_argument(
        "--load_captured",
        action="store_true",
        help="Load captured inputs from disk instead of generating random tensors",
    )
    parser.add_argument(
        "--captured_dir",
        type=str,
        default="./captured_inputs",
        help="Directory containing captured input .pt files",
    )
    return parser.parse_args()


arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def main():
    args = parse_args()
    # hardcode non-variable length and non-causal for now
    args.causal = False
    args.layout = "bshd"

    # Handle captured input mode separately
    if args.load_captured:
        logger.info(f"Running benchmark with captured inputs from: {args.captured_dir}")
        run_benchmark_captured(args)
        return 0

    if not args.dv:
        args.dv = args.d
    assert (
        args.b and args.hq and args.sq and args.d and args.dv
    ), "If not running on captured (--load_captured) please provide \
            all of batch, number of Q heads, Q sequence length \
            and head size."

    assert (
        args.dtype in arg_to_torch_dtype
    ), "Only fp16, bf16 and f32 types currently supported."

    if args.print_vgpr:
        assert not args.bench_torch, "Do not use -bench_torch with -print_vgpr."
        print("Retrieving VGPR usage for Triton kernels...")

        def fun():
            return run_benchmark(args)

        print_vgpr(fun, get_caller_name_no_ext())
        return 0

    run_benchmark(args)


if __name__ == "__main__":
    import sys

    sys.exit(main())

from __future__ import annotations
from typing import List, Dict, Any
import torch
import os
import glob
import sys
import argparse
import triton
import logging

from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_wrapper,
)

from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
    create_hadamard_matrix,
)

from op_tests.triton_tests.attention.test_fav3_sage import (
    compare_accuracy,
    input_helper,
)
from op_tests.op_benchmarks.triton.bench_fav3_sage import fav2_forward_func
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    print_vgpr,
)

# Configuration
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def layout_preprocess(q, k, v, layout: str, target_layout: str = "bshd"):
    if layout != target_layout:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def bench_kernel(q, k, v, args, provider):
    """Main benchmarking logic for a single configuration."""
    if args.layout == "bshd":
        BATCH, N_CTX_Q, HQ, D_HEAD = q.shape
        _, N_CTX_K, HK, D_HEAD_V = v.shape
    else:
        BATCH, HQ, N_CTX_Q, D_HEAD = q.shape
        _, HK, N_CTX_K, D_HEAD_V = v.shape

    BLOCK_R = args.BLOCK_R
    R = create_hadamard_matrix(BLOCK_R, q.device) / (BLOCK_R**0.5)

    def fn():
        return fav3_sage_mxfp4_wrapper(
            q,
            k,
            v,
            causal=args.causal,
            layout=args.layout,
            q_smooth=args.qsmooth,
            hadamard_rotation=args.hadamard_rotate,
            R=R,
        )

    ms = triton.testing.do_bench(fn)
    # print("kernel (ms)", ms)

    # Metrics calculation (MXFP4 treats elements as 0.5 bytes in memory traffic for Q/K)
    total_flops = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * (D_HEAD + D_HEAD_V)

    if "ms" in provider:
        return ms
    if "TFLOPS" in provider:
        return total_flops / ms * 1e-9
    return ms


def run_benchmark(args):
    torch.manual_seed(20)

    # Define benchmark configs
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"],
            x_vals=x_vals_list,
            line_arg="provider",
            line_vals=["time(ms)", "throughput(TFLOPS)"],
            line_names=["time(ms)", "throughput(TFLOPS)"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="Metric Value",
            plot_name="MXFP4_Attention_Performance",
            args={
                "D_HEAD": args.d,
                "D_HEAD_V": args.dv,
                "dtype": arg_to_torch_dtype[args.dtype],
                "layout": args.layout,
            },
        )
    )
    def bench_mha(
        BATCH,
        HQ,
        HK,
        N_CTX_Q,
        N_CTX_K,
        D_HEAD,
        D_HEAD_V,
        dtype,
        layout,
        provider,
        device="cuda",
    ):
        q = torch.randn((BATCH, HQ, N_CTX_Q, D_HEAD), device=device, dtype=dtype)
        k = torch.randn((BATCH, HK, N_CTX_K, D_HEAD), device=device, dtype=dtype)
        v = torch.randn((BATCH, HK, N_CTX_K, D_HEAD_V), device=device, dtype=dtype)

        q, k, v = layout_preprocess(q, k, v, layout="bhsd", target_layout=layout)
        return bench_kernel(q, k, v, args, provider)

    bench_mha.run(save_path=None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Simplified MXFP4 Attention Benchmark")
    parser.add_argument("-b", type=int, required=True, help="Batch size")
    parser.add_argument("-hq", type=int, required=True, help="Number of Q heads")
    parser.add_argument("-hk", type=int, default=0, help="Number of K heads (GQA)")
    parser.add_argument("-sq", type=int, required=True, help="Q Sequence length")
    parser.add_argument("-sk", type=int, default=0, help="K Sequence length")
    parser.add_argument("-d", type=int, required=True, help="Head dimension")
    parser.add_argument("-dv", type=int, default=0, help="V head dimension")
    parser.add_argument("-dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("-layout", type=str, default="bhsd", choices=["bshd", "bhsd"])
    parser.add_argument(
        "-captured_dir",
        type=str,
        default=None,
        help="Provide dir for captured inputs, for accuracy comparison.",
    )
    parser.add_argument(
        "-hadamard_rotate",
        type=lambda v: bool(int(v)),
        default=True,
        help="whether to apply hadamard rotate (1) or not (0). Default 1.",
    )

    parser.add_argument(
        "-BLOCK_R",
        type=int,
        default=128,
        help="Hadamard matrix size. Should be <= d",
    )
    parser.add_argument(
        "-qsmooth",
        action="store_true",
        help="Do q smoothing (Warning! Smoothing Q requires bias addition which drops the perf as of now!)",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        help="Print VGPR usage of the called Triton kernels",
    )
    parser.add_argument(
        "-causal",
        action="store_true",
        help="Causal masking",
    )
    return parser.parse_args()


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
    for f in input_files:
        data = torch.load(f, weights_only=False)
        inputs.append(data)

    return inputs


def test_accuracy(q, k, v, args):

    BLOCK_R = args.BLOCK_R
    R = create_hadamard_matrix(BLOCK_R, q.device) / (BLOCK_R**0.5)

    triton_out = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=args.causal,
        layout=args.layout,
        q_smooth=args.qsmooth,
        hadamard_rotation=args.hadamard_rotate,
        R=R,
    )
    # permute because FAv2 assumes bshd
    if args.layout == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

    print("Using as ref: Triton FAv2")
    sm_scale = q.shape[-1] ** -0.5
    ref_out = fav2_forward_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=sm_scale,
        causal=False,
        return_lse=False,
        return_attn_probs=False,
    )()
    if args.layout == "bhsd":
        ref_out = ref_out.permute(0, 2, 1, 3)

    assert ref_out.shape == triton_out.shape
    compare_accuracy(triton_out, ref_out)


def test_accuracy_with_captured_inputs(args):
    input_dir = args.captured_dir
    inputs = load_captured_inputs(input_dir)
    n_ = len(inputs)

    for input_i in range(n_):
        # Get the input tensors for this configuration
        inp = inputs[input_i]
        q = inp["q"].to("cuda")
        k = inp["k"].to("cuda")
        v = inp["v"].to("cuda")
        print("Testing accuracy on captured input:")
        print("q.shape: ", q.shape)
        print("k.shape: ", k.shape)
        print("v.shape: ", v.shape)
        test_accuracy(q, k, v, args)


def test_accuracy_with_shape(
    args,
    dtype=torch.bfloat16,
):
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q, k, v = input_helper(
        args.b,
        args.hq,
        args.hk,
        args.sq,
        args.sk,
        args.d,
        args.dv,
        dtype,
        args.layout,
    )
    print("Testing accuracy on shape:")
    print("q.shape: ", q.shape)
    print("k.shape: ", k.shape)
    print("v.shape: ", v.shape)
    test_accuracy(q, k, v, args)


def main():
    args = parse_args()
    if not args.dv:
        args.dv = args.d
    if not args.sk:
        args.sk = args.sq
    if not args.hk:
        args.hk = args.hq

    assert args.BLOCK_R <= args.d, "Rotation block size should be <= d"

    if args.captured_dir is not None:
        test_accuracy_with_captured_inputs(args)
    else:
        test_accuracy_with_shape(args)

    if args.print_vgpr:
        print("Retrieving VGPR usage for Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), "MXFP4_Attention_Performance")
        return 0

    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())

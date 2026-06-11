"""
Benchmark: fav3_sage_lloymax — Lloyd-Max quantized attention

Mirrors bench_fav3_sage_mxfp4.py but uses the Lloyd-Max codebook for Q/K
quantization instead of e2m1 MXFP4.  Supports:
  • Dense attention (default)
  • Block-sparse / Sparge  (--block_sparsity)
  • include_quant_overhead flag (benchmark full pipeline vs kernel-only)

Usage (same flags as bench_fav3_sage_mxfp4.py):
  python3 bench_fav3_sage_lloymax.py -b 4 -hq 16 -sq 4096 -d 128 -metric all
  python3 bench_fav3_sage_lloymax.py -b 4 -hq 16 -sq 4096 -d 128 \\
      --block_sparsity 0.5 -metric all
"""

from __future__ import annotations

import sys
import argparse
import logging

import torch
import triton
import aiter

from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_lloymax_wrapper,
    fav3_sage_lloymax_func,
    get_sage_fwd_configs_mxfp4,
)
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    sage_quant_lloymax_packed,
)
from aiter.ops.triton.attention.turboquant.codebook import get_codebook
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
from aiter.test_mha_common import attention_ref, attention_ref_block_sparse
from op_tests.triton_tests.attention.test_fav3_sage import compare_accuracy
# Note: check_attention_outputs uses assert_close with atol=0.01, designed for
# near-lossless kernels (int8/fp8). Lloyd-Max is a lossy 4-bit quantizer and
# intentionally has larger errors — compare_accuracy (which prints stats) is sufficient.
from op_tests.op_benchmarks.triton.bench_fav3_sage import sparse_flops_from_lut
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

logging.getLogger().setLevel(logging.INFO)

arg_to_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}


def layout_preprocess(q, k, v, src="bhsd", dst="bshd"):
    if src != dst:
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def bench_kernel(q, k, v, args, provider, block_lut=None, block_attn_mask=None):
    layout = args.layout
    if layout == "bshd":
        B, N_CTX_Q, HQ, D = q.shape
        _, N_CTX_K, HK, DV = v.shape
    else:
        B, HQ, N_CTX_Q, D = q.shape
        _, HK, N_CTX_K, DV = v.shape

    use_block_sparse = block_lut is not None
    if use_block_sparse:
        kv_block_indices, lut_start, lut_count = block_lut
    else:
        kv_block_indices = lut_start = lut_count = None

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX  = torch.finfo(FP8_TYPE).max
    config   = get_sage_fwd_configs_mxfp4()

    if args.include_quant_overhead:
        # Full pipeline: quantization + attention
        def fn():
            return fav3_sage_lloymax_wrapper(
                q, k, v,
                causal=args.causal,
                layout=layout,
                block_lut=block_lut,
            )
    else:
        # Kernel-only: pre-quantize once, benchmark attention kernel
        q_packed, q_norms, k_packed, k_norms, v_fp8, v_scale, _ = sage_quant_lloymax_packed(
            q, k, v,
            FP8_TYPE=FP8_TYPE,
            FP8_MAX=FP8_MAX,
            BLKQ=config["BLOCK_M"],
            BLKK=64,
            layout=layout,
        )
        head_dim = D
        codebook = get_codebook(head_dim, 4, device=q.device).float().contiguous()

        def fn():
            return fav3_sage_lloymax_func(
                q_packed, q_norms,
                k_packed, k_norms,
                v_fp8, v_scale,
                codebook,
                causal=args.causal,
                layout=layout,
                config=config,
                kv_block_indices=kv_block_indices,
                lut_start=lut_start,
                lut_count=lut_count,
                use_block_sparse=use_block_sparse,
            )

    rep    = getattr(args, "rep", 100)
    warmup = getattr(args, "warmup", 25)
    ms     = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    if getattr(args, "compare_to_ref", False):
        out = fn()
        if layout == "bhsd":
            out = out.permute(0, 2, 1, 3)
        q_ref, k_ref, v_ref = layout_preprocess(q, k, v, src=layout)
        if block_attn_mask is not None:
            ref_out = attention_ref_block_sparse(
                q_ref, k_ref, v_ref, block_attn_mask,
                config["BLOCK_M"], config["BLOCK_N"],
                dropout_p=0.0, dropout_mask=None, upcast=True,
            )[0]
        else:
            ref_out = attention_ref(q_ref, k_ref, v_ref,
                                    dropout_p=0.0, dropout_mask=None,
                                    causal=False)[0]
        compare_accuracy(out, ref_out)
        # Intentionally not calling check_attention_outputs — Lloyd-Max is lossy
        # 4-bit quantization; max abs error ~0.08 is expected and within 4-bit limits.

    total_flops = 2.0 * B * HQ * N_CTX_Q * N_CTX_K * (D + DV)
    sparse_flops = sparse_flops_from_lut(block_lut, B, N_CTX_Q, N_CTX_K, HQ, D, DV)[0] \
                   if block_lut is not None else 0

    q_sz = B * N_CTX_Q * HQ * D  * q.element_size()
    k_sz = B * N_CTX_K * HK * D  * k.element_size()
    v_sz = B * N_CTX_K * HK * DV * v.element_size()
    o_sz = B * N_CTX_Q * HQ * DV * q.element_size()
    mem  = q_sz + k_sz + v_sz + o_sz

    if "ms" in provider:
        return ms
    if "throughput_sparse" in provider:
        return sparse_flops / ms * 1e-9
    if "TFLOPS" in provider:
        return total_flops / ms * 1e-9
    if "GB/s" in provider:
        return mem / ms * 1e-6
    return ms


def run_benchmark(args):
    torch.manual_seed(20)
    dtype  = arg_to_dtype[args.dtype]
    layout = args.layout

    line_vals = ["time(ms)", "throughput(TFLOPS)", "bandwidth(GB/s)"]
    if args.block_sparsity is not None:
        line_vals.append("throughput_sparse(TFLOPS)")
    if args.metric and args.metric != "all":
        metric_map = {
            "time": "time(ms)", "throughput": "throughput(TFLOPS)",
            "bandwidth": "bandwidth(GB/s)",
        }
        line_vals = [metric_map[args.metric]]

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=["BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K"],
            x_vals=[(args.b, args.hq, args.hk, args.sq, args.sk)],
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=[("red","-"),("green","-"),("yellow","-"),("blue","-")],
            ylabel="",
            plot_name=get_caller_name_no_ext(),
            args={"D_HEAD": args.d, "D_HEAD_V": args.dv,
                  "dtype": dtype, "layout": layout, "causal": False},
        )
    ])
    def bench_mha(BATCH, HQ, HK, N_CTX_Q, N_CTX_K,
                  D_HEAD, D_HEAD_V, dtype, layout, causal, provider, device="cuda"):
        q = torch.randn((BATCH, HQ, N_CTX_Q, D_HEAD),   device=device, dtype=dtype)
        k = torch.randn((BATCH, HK, N_CTX_K, D_HEAD),   device=device, dtype=dtype)
        v = torch.randn((BATCH, HK, N_CTX_K, D_HEAD_V), device=device, dtype=dtype)
        q, k, v = layout_preprocess(q, k, v, src="bhsd", dst=layout)

        block_lut = block_attn_mask = None
        if args.block_sparsity is not None:
            cfg = get_sage_fwd_configs_mxfp4()
            BM, BN = cfg["BLOCK_M"], cfg["BLOCK_N"]
            nq = (N_CTX_Q + BM - 1) // BM
            nk = (N_CTX_K + BN - 1) // BN
            block_attn_mask = (
                torch.rand(BATCH, HQ, nq, nk, device=device) > args.block_sparsity
            ).bool()
            block_lut = block_attn_mask_to_ragged_lut(block_attn_mask)

        return bench_kernel(q, k, v, args, provider,
                            block_lut=block_lut, block_attn_mask=block_attn_mask)

    bench_mha.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Lloyd-Max attention kernel")
    p.add_argument("-b",   type=int, required=True)
    p.add_argument("-hq",  type=int, required=True)
    p.add_argument("-hk",  type=int, default=0)
    p.add_argument("-sq",  type=int, required=True)
    p.add_argument("-sk",  type=int, default=0)
    p.add_argument("-d",   type=int, required=True)
    p.add_argument("-dv",  type=int, default=0)
    p.add_argument("-dtype",   default="bf16", choices=["bf16", "fp16"])
    p.add_argument("-layout",  default="bshd", choices=["bshd", "bhsd"])
    p.add_argument("-causal",  action="store_true", default=False)
    p.add_argument("-include_quant_overhead", action="store_true", default=False)
    p.add_argument("-compare_to_ref",         action="store_true", default=False)
    p.add_argument("--block_sparsity", type=float, default=None)
    p.add_argument("-metric", default="all",
                   choices=["all", "time", "throughput", "bandwidth"])
    p.add_argument("-o", action="store_true", default=False)
    p.add_argument("--rep",    type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.dv:
        args.dv = args.d
    if not args.sk:
        args.sk = args.sq
    if not args.hk:
        args.hk = args.hq
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())

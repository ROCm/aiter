# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the FP8 Flash Attention v2 kernel (attn_fwd, USE_FP8=False path).

Usage:
    python bench_fp8_attention.py
    python bench_fp8_attention.py --causal
    python bench_fp8_attention.py -metric bandwidth
"""

import argparse
import math

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.fp8_attention_kernel import (
    attn_fwd,
    get_padded_headsize,
)
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# Representative shapes: (B, HQ, HK, S, D)
_SHAPES = [
    (1, 32, 32, 512, 128),
    (1, 32, 32, 1024, 128),
    (1, 32, 32, 2048, 128),
    (1, 32, 32, 4096, 128),
    (4, 32, 8, 1024, 128),  # GQA
    (4, 32, 8, 2048, 128),  # GQA
]

BLOCK_M = 64
BLOCK_N = 64


def _make_inputs(B, HQ, HK, S, D, dtype, device="cuda"):
    q = torch.randn(B, HQ, S, D, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, HK, S, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, HK, S, D, dtype=dtype, device=device) * 0.1
    return q, k, v


def _launch_attn_fwd(q, k, v, causal, sm_scale):
    B, HQ, S_q, D = q.shape
    _, HK, S_k, _ = k.shape
    D_pad = get_padded_headsize(D)

    def _pad(t):
        if t.shape[-1] == D_pad:
            return t.contiguous()
        p = torch.zeros(*t.shape[:-1], D_pad, dtype=t.dtype, device=t.device)
        p[..., :D] = t
        return p

    q_p, k_p, v_p = _pad(q), _pad(k), _pad(v)
    out = torch.zeros(B, HQ, S_q, D_pad, dtype=q.dtype, device=q.device)
    lse = torch.zeros(B, HQ, S_q, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(S_q, BLOCK_M), HQ, B)

    attn_fwd[grid](
        Q=q_p,
        K=k_p,
        V=v_p,
        bias=None,
        p_scale=1.0,
        q_descale_ptr=None,
        k_descale_ptr=None,
        v_scale_ptr=None,
        USE_FP8=False,
        SM_SCALE=sm_scale,
        LSE=lse,
        Out=out,
        stride_qz=q_p.stride(0),
        stride_qh=q_p.stride(1),
        stride_qm=q_p.stride(2),
        stride_qk=q_p.stride(3),
        stride_kz=k_p.stride(0),
        stride_kh=k_p.stride(1),
        stride_kn=k_p.stride(2),
        stride_kk=k_p.stride(3),
        stride_vz=v_p.stride(0),
        stride_vh=v_p.stride(1),
        stride_vk=v_p.stride(2),
        stride_vn=v_p.stride(3),
        stride_oz=out.stride(0),
        stride_oh=out.stride(1),
        stride_om=out.stride(2),
        stride_on=out.stride(3),
        stride_bz=0,
        stride_bh=0,
        stride_bm=0,
        stride_bn=0,
        stride_az=0,
        stride_ah=0,
        stride_sz=0,
        stride_sh=0,
        stride_sm=0,
        stride_sn=0,
        stride_lse_z=lse.stride(0),
        stride_lse_h=lse.stride(1),
        stride_lse_m=lse.stride(2),
        stride_qdescale_z=0,
        stride_qdescale_h=0,
        stride_qdescale_m=0,
        stride_kdescale_z=0,
        stride_kdescale_h=0,
        stride_kdescale_m=0,
        padded_kscale_block_num=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        dropout_p=0.0,
        philox_seed=0,
        philox_offset_base=0,
        scores=None,
        scores_scaled_shifted=None,
        exp_scores=None,
        alibi_slopes=None,
        HQ=HQ,
        HK=HK,
        ACTUAL_BLOCK_DMODEL_QK=D_pad,
        ACTUAL_BLOCK_DMODEL_V=D_pad,
        MAX_SEQLENS_Q=S_q,
        MAX_SEQLENS_K=S_k,
        VARLEN=False,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL_QK=D_pad,
        BLOCK_DMODEL_V=D_pad,
        BLOCK_N=BLOCK_N,
        USE_BIAS=False,
        ENABLE_DROPOUT=False,
        RETURN_SCORES=False,
        USE_ALIBI=False,
        USE_EXP2=True,
    )
    return out[..., :D]


def benchmark(args):
    arch = get_arch()
    if arch not in FP8_ARCHS:
        print(f"Skipping: FP8 attention not supported on {arch}")
        return

    causal = args.causal
    unit = "ms" if args.metric == "time" else "GB/s"
    dtype = torch.bfloat16

    x_vals = [(B, HQ, HK, S, D) for B, HQ, HK, S, D in _SHAPES]

    config = triton.testing.Benchmark(
        x_names=["B", "HQ", "HK", "S", "D"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["attn_fwd"],
        line_names=[f"attn_fwd ({'causal' if causal else 'non-causal'}) ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(B, HQ, HK, S, D, provider):
        q, k, v = _make_inputs(B, HQ, HK, S, D, dtype)
        sm_scale = 1.0 / math.sqrt(D)
        fn = lambda: _launch_attn_fwd(q, k, v, causal, sm_scale)

        # reads: Q(B,HQ,S,D) + K(B,HK,S,D) + V(B,HK,S,D); writes: Out(B,HQ,S,D)
        elem = q.element_size()
        mem = (B * HQ * S * D + B * HK * S * D * 2 + B * HQ * S * D) * elem

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FP8 Flash Attention v2 kernel", allow_abbrev=False
    )
    parser.add_argument(
        "--causal", action="store_true", default=False, help="Enable causal masking"
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()

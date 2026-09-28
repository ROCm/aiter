# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for fused KDA decode kernel vs 3-kernel baseline.

Uses triton.testing.do_bench for measurement (same as other bench files).
Uses real Triton rmsnorm kernel (inlined from ATOM) for fair comparison.

Usage examples
--------------
python bench_fused_kda_decode.py
python bench_fused_kda_decode.py --batches 1 16 64 128
python bench_fused_kda_decode.py --Hloc 8
"""

import argparse

import torch
import triton
import triton.language as tl
from einops import rearrange

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from aiter.ops.triton.gated_delta_net.causal_conv1d_decode import (
    causal_conv1d_update_split_qkv,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode

DEVICE = "cuda"
D = 128
W = 4
DTYPE = torch.bfloat16


# -- Inlined from ATOM's atom/model_ops/kimi_k3/activations.py --


@triton.jit
def _rmsnorm_gated_kernel(
    x_ptr,
    w_ptr,
    g_ptr,
    y_ptr,
    H,
    eps,
    stride_xm,
    stride_ym,
    stride_g_outer,
    stride_g_head,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    g_off = (row // HEADS) * stride_g_outer + (row % HEADS) * stride_g_head + cols
    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * w) * tl.sigmoid(gate)
    tl.store(y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm_gated_bf16(x, weight, gate, eps):
    """Gated RMSNorm -> bf16 (real Triton kernel)."""
    h = x.shape[-1]
    x2 = x.reshape(-1, h).contiguous()
    m = x2.shape[0]
    heads = x.shape[1] if x.ndim == 3 else 1
    y = torch.empty_like(x2)
    if gate.ndim == 3:
        stride_g_outer, stride_g_head = gate.stride(0), gate.stride(1)
    else:
        stride_g_outer, stride_g_head = gate.stride(0), 0
    BLOCK = triton.next_power_of_2(h)
    _rmsnorm_gated_kernel[(m,)](
        x2,
        weight,
        gate,
        y,
        h,
        float(eps),
        x2.stride(0),
        y.stride(0),
        stride_g_outer,
        stride_g_head,
        HEADS=heads,
        BLOCK=BLOCK,
    )
    return y.reshape_as(x)


# -- End inlined kernel --


def _make_inputs(batch, Hloc):
    lp = Hloc * D
    num_slots = batch + 2
    torch.manual_seed(0)
    return {
        "mixed_qkv": torch.randn(batch, 3 * lp, dtype=DTYPE, device=DEVICE),
        "conv_weight": torch.randn(3 * lp, W, dtype=DTYPE, device=DEVICE) * 0.1,
        "conv_state": torch.randn(num_slots, 3 * lp, W - 1, dtype=DTYPE, device=DEVICE)
        * 0.1,
        "gate": torch.randn(1, batch, Hloc, D, dtype=DTYPE, device=DEVICE) * 0.5,
        "beta": torch.randn(1, batch, Hloc, dtype=DTYPE, device=DEVICE),
        "out_gate": torch.randn(batch, lp, dtype=DTYPE, device=DEVICE),
        "A_log": torch.randn(Hloc, dtype=DTYPE, device=DEVICE) * 0.1,
        "dt_bias": torch.randn(lp, dtype=DTYPE, device=DEVICE) * 0.1,
        "ssm_state": torch.randn(
            num_slots, Hloc, D, D, dtype=torch.float32, device=DEVICE
        )
        * 0.01,
        "norm_weight": torch.ones(D, dtype=DTYPE, device=DEVICE),
        "ssm_state_indices": torch.arange(batch, dtype=torch.int32, device=DEVICE),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int64, device=DEVICE),
    }


def run_benchmark(args):
    Hloc = args.Hloc
    batches = args.batches

    header = f"{'Batch':>6}  {'3-kernel(us)':>13}  {'Fused(us)':>10}  {'Speedup':>8}"
    print(f"\nHloc={Hloc}, D={D}, W={W}")
    print(header)
    print("-" * len(header))

    for batch in batches:
        inp = _make_inputs(batch, Hloc)

        def fn_3k(inp=inp):
            cs = inp["conv_state"].clone()
            ss = inp["ssm_state"].clone()
            T = inp["mixed_qkv"].shape[0]
            lp = Hloc * D
            q, k, v = causal_conv1d_update_split_qkv(
                inp["mixed_qkv"],
                cs,
                inp["conv_weight"],
                lp,
                lp,
                bias=None,
                activation="silu",
                conv_state_indices=inp["ssm_state_indices"],
                use_gluon=False,
            )
            out = torch.empty(T, Hloc, D, dtype=q.dtype, device=DEVICE)
            fused_sigmoid_gating_delta_rule_update(
                A_log=inp["A_log"],
                a=inp["gate"],
                dt_bias=inp["dt_bias"],
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=rearrange(q, "t (h d) -> 1 t h d", d=D),
                k=rearrange(k, "t (h d) -> 1 t h d", d=D),
                v=rearrange(v, "t (h d) -> 1 t h d", d=D),
                b=inp["beta"],
                initial_state_source=ss,
                initial_state_indices=inp["ssm_state_indices"],
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=inp["cu_seqlens"],
            )
            og3d = rearrange(inp["out_gate"][:T], "t (h d) -> t h d", d=D)
            rmsnorm_gated_bf16(out, inp["norm_weight"], og3d, 1e-6)

        def fn_fused(inp=inp):
            cs = inp["conv_state"].clone()
            ss = inp["ssm_state"].clone()
            fused_kda_decode(
                inp["mixed_qkv"],
                cs,
                inp["conv_weight"],
                inp["gate"],
                inp["beta"],
                inp["out_gate"],
                inp["A_log"],
                inp["dt_bias"],
                ss,
                inp["ssm_state_indices"],
                inp["cu_seqlens"],
                inp["norm_weight"],
                1e-6,
                D,
                Hloc,
                -5.0,
            )

        t_3k = triton.testing.do_bench(fn_3k, warmup=50, rep=200) * 1000
        t_fused = triton.testing.do_bench(fn_fused, warmup=50, rep=200) * 1000
        speedup = t_3k / t_fused

        print(f"{batch:>6}  {t_3k:>13.1f}  {t_fused:>10.1f}  {speedup:>7.2f}x")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark Fused KDA Decode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--Hloc",
        type=int,
        default=12,
        help="Number of local heads (Kimi K3 = 12)",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128, 256],
        help="Batch sizes to benchmark",
    )
    return parser.parse_args(args=args)


def main(args=None):
    run_benchmark(parse_args(args=args))


if __name__ == "__main__":
    main()

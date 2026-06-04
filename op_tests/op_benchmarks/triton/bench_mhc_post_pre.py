# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse

import torch
import triton

import aiter
from aiter.ops.triton.fusions.mhc import mhc_post_pre as triton_mhc_post_pre
from aiter.test_common import checkAllclose
from op_tests.triton_tests.fusions.test_mhc import _alphas
from op_tests.triton_tests.utils.mhc_ref import (
    generate_mhc_inputs,
    generate_mhc_post_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HIP mhc_post_pre against Triton mhc_post_pre"
    )
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("-C", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert args.n == 4, "HIP mhc_post_pre currently supports n == 4"
    assert torch.cuda.is_available(), "CUDA/ROCm device is required"
    assert hasattr(aiter, "mhc_post_pre"), "aiter.mhc_post_pre is not available"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dtype = torch.bfloat16
    layer_input, residual_in, post_mix, comb_mix = generate_mhc_post_inputs(
        args.M, args.n, args.C, dtype
    )
    _x_unused, phi, alpha_pre, alpha_post, alpha_res, bias, _n = generate_mhc_inputs(
        args.M, args.n, args.C, dtype
    )
    alphas = _alphas(alpha_pre, alpha_post, alpha_res, device=layer_input.device)
    phi_triton = phi.T.contiguous().T
    fn_hip = phi.T.contiguous().float()
    hc_base = bias.float().contiguous()

    def run_triton():
        return triton_mhc_post_pre(
            layer_input,
            residual_in,
            post_mix,
            comb_mix,
            phi_triton,
            alphas,
            bias,
            args.n,
            hc_pre_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_iters=20,
            asymmetric_exp_domain=True,
            hc_sinkhorn_eps=1e-6,
        )

    def run_hip():
        return aiter.mhc_post_pre(
            layer_input,
            residual_in,
            post_mix,
            comb_mix,
            fn_hip,
            alphas.float(),
            hc_base,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )

    tri = run_triton()
    hip = run_hip()
    for name, h, t, atol, rtol in (
        ("post_mix", hip[0], tri[0], 4e-2, 2e-2),
        ("comb_mix", hip[1], tri[1], 4e-2, 2e-2),
        ("layer_input", hip[2], tri[2], 8e-2, 2e-2),
        ("residual_out", hip[3], tri[3], 4e-2, 2e-2),
    ):
        pct = checkAllclose(
            h.float(),
            t.float(),
            atol=atol,
            rtol=rtol,
            tol_err_ratio=0.05,
            msg=f"{name} HIP vs Triton",
            printLog=True,
        )
        if pct > 0.05:
            raise AssertionError(f"{name} mismatch: bad_element_ratio={pct:.2%}")

    hip_ms = triton.testing.do_bench(run_hip, warmup=args.warmup, rep=args.rep)
    triton_ms = triton.testing.do_bench(run_triton, warmup=args.warmup, rep=args.rep)
    print(
        f"M={args.M} n={args.n} C={args.C}: "
        f"hip={hip_ms:.4f} ms, triton={triton_ms:.4f} ms, "
        f"triton/hip={triton_ms / hip_ms:.3f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

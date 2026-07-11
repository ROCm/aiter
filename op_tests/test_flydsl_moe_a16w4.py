# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL MOE a16w4 (bf16 activations, fp4x2/per_1x32 weights).

This is the "fp4_bf16" path: activations stay bf16 (no activation quant), and
FP4 E2M1 weights + E8M0 block scales are dequantised to bf16 in-kernel.

It uses gate_mode="separated" with the klane-inner preshuffle layout produced by
``shuffle_weight_a16w4_sep_fp4`` / ``shuffle_scale_a16w4_sep_fp4`` -- the layout
the fp4_bf16 kernels address (klane_inner=True for is_fp4_bf16 in both stages).

Tests:
  - Stage1 (gate+up GEMM + silu): flydsl_moe_stage1  a_dtype="bf16" b_dtype="fp4bf16"
  - Stage2 (down-proj GEMM):      flydsl_moe_stage2  a_dtype="bf16" b_dtype="fp4bf16"
  - End-to-end (stage1 -> stage2 via FlyDSL)

Usage:
    python op_tests/test_flydsl_moe_a16w4.py                 # all tests
    python op_tests/test_flydsl_moe_a16w4.py --stage stage1  # stage1 only
    python op_tests/test_flydsl_moe_a16w4.py -t 16 64        # specific token counts
"""

import argparse
import sys

import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import (
    fused_topk,
    moe_sorting,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.shuffle import (
    shuffle_weight_a16w4_sep_fp4,
    shuffle_scale_a16w4_sep_fp4,
)

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_W = dtypes.fp4x2


def _generate_a16w4_data(token, model_dim, inter_dim, E, topk, block_m,
                         dtype=torch.bfloat16):
    """Quantised a16w4 data + torch references (bf16 activations, no act quant)."""
    torch_quant = aiter.get_torch_quant(Q_TYPE)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    a1_qt = inp.to(dtypes.bf16)  # a16w4: bf16 activations, no scale

    ref1 = torch_moe_stage1(
        a1_qt, w1_qt, w2_qt, topk_weights, topk_ids, dtype=dtype,
        activation=ActivationType.Silu, quant_type=Q_TYPE,
        a1_scale=None, w1_scale=w1_scale, doweight=False,
    )
    ref2 = torch_moe_stage2(
        ref1.view(token, topk, -1), w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, quant_type=Q_TYPE, w2_scale=w2_scale, a2_scale=None,
        doweight=True,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    return dict(
        ref_stage1=ref1, ref_stage2=ref2,
        a1_qt=a1_qt,
        w1_shuf=shuffle_weight_a16w4_sep_fp4(w1_qt),
        w1_scale_shuf=shuffle_scale_a16w4_sep_fp4(w1_scale, E),
        w2_shuf=shuffle_weight_a16w4_sep_fp4(w2_qt),
        w2_scale_shuf=shuffle_scale_a16w4_sep_fp4(w2_scale, E),
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        dtype=dtype, token=token, topk=topk, inter_dim=inter_dim,
    )


def _check(ref, out, label, atol=1.0, rtol=0.05, pass_pct=95.0, ld_tol=1e-2):
    ref_f, out_f = ref.float().reshape(-1), out.float().reshape(-1)
    max_delta = (ref_f - out_f).abs().max().item()
    pct = torch.isclose(ref_f, out_f, atol=atol, rtol=rtol).float().mean().item() * 100
    denom = (ref_f * ref_f + out_f * out_f).sum()
    logits_diff = (1 - 2 * (ref_f * out_f).sum() / denom).item()
    passed = (pct > pass_pct) and (logits_diff <= ld_tol)
    print(f"  [{label}] max_delta={max_delta:.4f}, {pct:.1f}% close, "
          f"logits_diff={logits_diff:.6f} --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta, logits_diff


def test_stage1(token=16, model_dim=7168, inter_dim=256, E=384, topk=8, block_m=16):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
    print(f"\n[TEST] stage1 a16w4: token={token} dim=({model_dim},{inter_dim}) "
          f"E={E} topk={topk} bm={block_m}")
    d = _generate_a16w4_data(token, model_dim, inter_dim, E, topk, block_m)
    out = flydsl_moe_stage1(
        a=d["a1_qt"], w1=d["w1_shuf"],
        sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"], topk=topk,
        tile_m=block_m, tile_n=128, tile_k=128,
        a_dtype="bf16", b_dtype="fp4bf16", out_dtype="bf16", act="silu",
        w1_scale=d["w1_scale_shuf"], a1_scale=None, sorted_weights=None,
        use_async_copy=True, gate_mode="separated",
    )
    torch.cuda.synchronize()
    return _check(d["ref_stage1"], out, "stage1")


def test_stage2(token=16, model_dim=7168, inter_dim=256, E=384, topk=8, block_m=16):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    print(f"\n[TEST] stage2 a16w4: token={token} dim=({model_dim},{inter_dim}) "
          f"E={E} topk={topk} bm={block_m}")
    d = _generate_a16w4_data(token, model_dim, inter_dim, E, topk, block_m)
    out = flydsl_moe_stage2(
        inter_states=d["ref_stage1"].view(token, topk, -1), w2=d["w2_shuf"],
        sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
        num_valid_ids=d["num_valid_ids"], topk=topk,
        tile_m=block_m, tile_n=128, tile_k=128,
        a_dtype="bf16", b_dtype="fp4bf16", out_dtype="bf16", mode="atomic",
        w2_scale=d["w2_scale_shuf"], a2_scale=None, sorted_weights=d["sorted_weights"],
    )
    torch.cuda.synchronize()
    return _check(d["ref_stage2"], out, "stage2")


def test_e2e(token=16, model_dim=7168, inter_dim=256, E=384, topk=8, block_m=16,
             bench=False):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    print(f"\n[TEST] e2e a16w4: token={token} dim=({model_dim},{inter_dim}) "
          f"E={E} topk={topk} bm={block_m}")
    d = _generate_a16w4_data(token, model_dim, inter_dim, E, topk, block_m)

    def _s1():
        return flydsl_moe_stage1(
            a=d["a1_qt"], w1=d["w1_shuf"],
            sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
            num_valid_ids=d["num_valid_ids"], topk=topk,
            tile_m=block_m, tile_n=128, tile_k=128,
            a_dtype="bf16", b_dtype="fp4bf16", out_dtype="bf16", act="silu",
            w1_scale=d["w1_scale_shuf"], a1_scale=None, sorted_weights=None,
            use_async_copy=True, gate_mode="separated",
        )

    def _s2(s1):
        return flydsl_moe_stage2(
            inter_states=s1.view(token, topk, -1), w2=d["w2_shuf"],
            sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
            num_valid_ids=d["num_valid_ids"], topk=topk,
            tile_m=block_m, tile_n=128, tile_k=128,
            a_dtype="bf16", b_dtype="fp4bf16", out_dtype="bf16", mode="atomic",
            w2_scale=d["w2_scale_shuf"], a2_scale=None, sorted_weights=d["sorted_weights"],
        )

    s1 = _s1(); torch.cuda.synchronize()
    out = _s2(s1); torch.cuda.synchronize()
    if bench:
        from aiter.test_common import run_perftest
        _, us1 = run_perftest(_s1, num_iters=20, num_warmup=5)
        _, us2 = run_perftest(_s2, s1, num_iters=20, num_warmup=5)
        print(f"  [BENCH] moe_gemm1(stage1)={us1:.2f} us  "
              f"moe_gemm2(stage2)={us2:.2f} us  (token={token})")
    return _check(d["ref_stage2"], out, "e2e", pass_pct=90.0)


def main():
    p = argparse.ArgumentParser(description="FlyDSL MOE a16w4 (fp4_bf16) unit tests")
    p.add_argument("-t", "--tokens", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--model-dim", type=int, default=7168)
    p.add_argument("--inter-dim", type=int, default=256)
    p.add_argument("-E", "--experts", type=int, default=384)
    p.add_argument("-k", "--topk", type=int, default=8)
    p.add_argument("--block-m", type=int, nargs="+", default=[16])
    p.add_argument("--stage", type=str, nargs="+", default=["stage1", "stage2", "e2e"],
                   choices=["stage1", "stage2", "e2e"])
    p.add_argument("--bench", action="store_true",
                   help="Also print per-stage (moe_gemm1/moe_gemm2) us via run_perftest (e2e only)")
    args = p.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available
    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available.")
        sys.exit(0)

    fns = {"stage1": test_stage1, "stage2": test_stage2, "e2e": test_e2e}
    results = []
    for token in args.tokens:
        for bm in args.block_m:
            for st in args.stage:
                _kw = dict(
                    token=token, model_dim=args.model_dim, inter_dim=args.inter_dim,
                    E=args.experts, topk=args.topk, block_m=bm,
                )
                if st == "e2e":
                    _kw["bench"] = args.bench
                passed, md, ld = fns[st](**_kw)
                results.append((f"{st}_t{token}_bm{bm}", passed, md, ld))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    for name, ok, md, ld in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:28s} max_delta={md:.4f} "
              f"logits_diff={ld:.6f}")
    print(f"\n  {n_pass}/{len(results)} passed")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()

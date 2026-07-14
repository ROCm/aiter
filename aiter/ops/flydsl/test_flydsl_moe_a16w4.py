#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
FlyDSL MOE A16W4 tests: bf16 activations, MXFP4 (FP4 E2M1 + E8M0) weights.

  - Stage1 (gate+up GEMM): flydsl_moe_stage1 with a_dtype="bf16", b_dtype="fp4bf16"
  - Stage2 (down-proj GEMM): flydsl_moe_stage2 with a_dtype="bf16", b_dtype="fp4bf16"
  - End-to-end (stage1 + stage2 combined via FlyDSL)

Usage (matches test_flydsl_moe_a4w4.py CLI):

    python test_flydsl_moe_a16w4.py --stage stage1          # stage1 only
    python test_flydsl_moe_a16w4.py --stage e2e             # end-to-end only
    python test_flydsl_moe_a16w4.py --stage e2e -t 8 --model-dim 4096 --inter-dim 512 -E 512 -k 10

Profiling:

    rocprofv3 --kernel-trace --output-format csv -d /tmp/pf_a16w4 -- \\
      env AITER_FLYDSL_FORCE=1 HIP_VISIBLE_DEVICES=2 \\
      python3 test_flydsl_moe_a16w4.py --stage e2e \\
      --model-dim 4096 --inter-dim 512 -E 512 -k 10 -t 8
"""

import argparse
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch

torch.set_default_device("cuda")

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2
from aiter.ops.shuffle import shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.test_common import checkAllclose

Q_TYPE = QuantType.per_1x32
Q_DTYPE_W = dtypes.fp4x2


def _moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_m):
    from aiter.fused_moe import moe_sorting

    return moe_sorting(topk_ids, topk_weights, E, model_dim, dtype, block_m)


def _generate_a16w4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    dtype=torch.bfloat16,
    doweight_stage1: bool = False,
    gate_mode: str = "interleave",
):
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # Quantize weights only (activations stay bf16)
    w1_qt_list, w1_scale_list = [], []
    for e in range(E):
        qt, sc = torch_quant(w1[e : e + 1], quant_dtype=Q_DTYPE_W)
        w1_qt_list.append(qt)
        w1_scale_list.append(sc)
    w1_qt = torch.cat(w1_qt_list).view(E, inter_dim * 2, model_dim // 2)
    w1_scale = torch.cat(w1_scale_list)
    del w1_qt_list, w1_scale_list

    w2_qt_list, w2_scale_list = [], []
    for e in range(E):
        qt, sc = torch_quant(w2[e : e + 1], quant_dtype=Q_DTYPE_W)
        w2_qt_list.append(qt)
        w2_scale_list.append(sc)
    w2_qt = torch.cat(w2_qt_list).view(E, model_dim, inter_dim // 2)
    w2_scale = torch.cat(w2_scale_list)
    del w2_qt_list, w2_scale_list
    torch.cuda.empty_cache()

    # Torch reference: stage1 (bf16 activation, fp4 weight)
    ref1 = torch_moe_stage1(
        inp,
        w1_qt.view(E, inter_dim * 2, model_dim // 2),
        w2_qt.view(E, model_dim, inter_dim // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Swiglu,
        quant_type=Q_TYPE,
        a1_scale=None,
        w1_scale=w1_scale,
        doweight=doweight_stage1,
    )

    # Torch reference: stage2 (bf16 activation, fp4 weight)
    stage1_for_ref2 = ref1.view(token, topk, inter_dim)
    ref2 = torch_moe_stage2(
        stage1_for_ref2,
        w1_qt.view(E, inter_dim * 2, model_dim // 2),
        w2_qt.view(E, model_dim, inter_dim // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=None,
        doweight=not doweight_stage1,
    )

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = _moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    if doweight_stage1:
        sorted_weights_s1 = sorted_weights
        sorted_weights_s2 = None
    else:
        sorted_weights_s1 = None
        sorted_weights_s2 = sorted_weights

    # Pad sorted_ids if needed
    needed = sorted_expert_ids.shape[0] * block_m
    if sorted_ids.shape[0] < needed:
        pad = torch.full(
            (needed - sorted_ids.shape[0],),
            token,
            dtype=sorted_ids.dtype,
            device=sorted_ids.device,
        )
        sorted_ids = torch.cat([sorted_ids, pad])
        sorted_weights = torch.cat(
            [
                sorted_weights,
                torch.zeros(
                    pad.shape[0],
                    dtype=sorted_weights.dtype,
                    device=sorted_weights.device,
                ),
            ]
        )

    # Shuffle weights for INTERLEAVE (klane_inner) or SEPARATED
    klane_inner = gate_mode == "interleave"
    gate_up = gate_mode == "interleave"
    w1_qt_shuf = shuffle_weight_a16w4(w1_qt, 16, gate_up, klane_inner=klane_inner)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, E, gate_up, klane_inner=klane_inner)
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False, klane_inner=klane_inner)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False, klane_inner=klane_inner)

    return dict(
        ref_stage1=ref1,
        ref_stage2=ref2,
        inp=inp,
        w1_qt=w1_qt,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale=w1_scale,
        w1_scale_shuf=w1_scale_shuf,
        w2_qt=w2_qt,
        w2_qt_shuf=w2_qt_shuf,
        w2_scale=w2_scale,
        w2_scale_shuf=w2_scale_shuf,
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_weights_s1=sorted_weights_s1,
        sorted_weights_s2=sorted_weights_s2,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        dtype=dtype,
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
    )


def _check_result(ref_out, test_out, label, atol=1.0, rtol=0.05, pass_pct=95.0):
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct

    print(
        f"  max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    return passed, max_delta, pct_close


# ---------------------------------------------------------------------------
# Stage1 test
# ---------------------------------------------------------------------------


def test_flydsl_stage1_a16w4(
    token: int = 16,
    model_dim: int = 4096,
    inter_dim: int = 512,
    E: int = 512,
    topk: int = 10,
    block_m: int = 16,
    gate_mode: str = "interleave",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage1 A16W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, gate_mode={gate_mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        gate_mode=gate_mode,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=128,
        a_dtype="bf16",
        b_dtype="fp4bf16",
        out_dtype=out_dtype_str,
        act="swiglu",
        w1_scale=data["w1_scale_shuf"],
        gate_mode=gate_mode,
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage1"]
    return _check_result(ref, out, f"stage1_a16w4_{gate_mode}", atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Stage2 test
# ---------------------------------------------------------------------------


def test_flydsl_stage2_a16w4(
    token: int = 16,
    model_dim: int = 4096,
    inter_dim: int = 512,
    E: int = 512,
    topk: int = 10,
    block_m: int = 16,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage2 A16W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"
    a2 = data["ref_stage1"].view(token, topk, inter_dim)

    out = flydsl_moe_stage2(
        inter_states=a2,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=128,
        a_dtype="bf16",
        b_dtype="fp4bf16",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage2"]
    return _check_result(
        ref, out, f"stage2_a16w4_{mode}", atol=atol, rtol=rtol, pass_pct=90.0
    )


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


def test_flydsl_e2e_a16w4(
    token: int = 16,
    model_dim: int = 4096,
    inter_dim: int = 512,
    E: int = 512,
    topk: int = 10,
    block_m: int = 16,
    mode: str = "atomic",
    gate_mode: str = "interleave",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL E2E A16W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}, gate_mode={gate_mode}"
    )
    print(f"{'='*70}")

    data = _generate_a16w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
        gate_mode=gate_mode,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    # Stage1: bf16 activation, fp4 weight
    stage1_out = flydsl_moe_stage1(
        a=data["inp"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=128,
        a_dtype="bf16",
        b_dtype="fp4bf16",
        out_dtype=out_dtype_str,
        act="swiglu",
        w1_scale=data["w1_scale_shuf"],
        gate_mode=gate_mode,
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()

    # Stage2: bf16 activation (stage1 output), fp4 weight
    a2 = stage1_out.view(token, topk, inter_dim)
    e2e_out = flydsl_moe_stage2(
        inter_states=a2,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=128,
        tile_k=128,
        a_dtype="bf16",
        b_dtype="fp4bf16",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage2"]
    return _check_result(
        ref, e2e_out, f"e2e_a16w4_{gate_mode}_{mode}", atol=atol, rtol=rtol, pass_pct=90.0
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL MOE A16W4 tests (bf16 activation, MXFP4 weights)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128],
        help="Token counts to test (default: 1 8 32 128)",
    )
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=512)
    parser.add_argument("-E", "--experts", type=int, default=512)
    parser.add_argument("-k", "--topk", type=int, default=10)
    parser.add_argument("--block-m", type=int, nargs="+", default=[16])
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        default=["atomic"],
        choices=["atomic", "reduce"],
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="interleave",
        choices=["interleave", "separated"],
    )
    parser.add_argument(
        "--stage",
        type=str,
        nargs="+",
        default=["stage1", "stage2", "e2e"],
        choices=["stage1", "stage2", "e2e"],
        help="Which tests to run (default: all)",
    )
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL is not available. Install flydsl package first.")
        sys.exit(0)

    results = []

    for token in args.tokens:
        for bm in args.block_m:
            if "stage1" in args.stage:
                try:
                    passed, max_delta, pct = test_flydsl_stage1_a16w4(
                        token=token,
                        model_dim=args.model_dim,
                        inter_dim=args.inter_dim,
                        E=args.experts,
                        topk=args.topk,
                        block_m=bm,
                        gate_mode=args.gate_mode,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    results.append(
                        (f"stage1_t{token}_bm{bm}", "PASS" if passed else "FAIL")
                    )
                except Exception as e:
                    print(f"  [ERROR] stage1 t={token} bm={bm}: {e}")
                    import traceback

                    traceback.print_exc()
                    results.append((f"stage1_t{token}_bm{bm}", f"ERROR: {e}"))

            for mode in args.mode:
                if "stage2" in args.stage:
                    try:
                        passed, max_delta, pct = test_flydsl_stage2_a16w4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            mode=mode,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append(
                            (
                                f"stage2_{mode}_t{token}_bm{bm}",
                                "PASS" if passed else "FAIL",
                            )
                        )
                    except Exception as e:
                        print(f"  [ERROR] stage2 {mode} t={token} bm={bm}: {e}")
                        import traceback

                        traceback.print_exc()
                        results.append(
                            (f"stage2_{mode}_t{token}_bm{bm}", f"ERROR: {e}")
                        )

                if "e2e" in args.stage:
                    try:
                        passed, max_delta, pct = test_flydsl_e2e_a16w4(
                            token=token,
                            model_dim=args.model_dim,
                            inter_dim=args.inter_dim,
                            E=args.experts,
                            topk=args.topk,
                            block_m=bm,
                            mode=mode,
                            gate_mode=args.gate_mode,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        results.append(
                            (
                                f"e2e_{mode}_t{token}_bm{bm}",
                                "PASS" if passed else "FAIL",
                            )
                        )
                    except Exception as e:
                        print(f"  [ERROR] e2e {mode} t={token} bm={bm}: {e}")
                        import traceback

                        traceback.print_exc()
                        results.append(
                            (f"e2e_{mode}_t{token}_bm{bm}", f"ERROR: {e}")
                        )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, status in results:
        print(f"  {status:6s}  {name}")

    n_fail = sum(1 for _, s in results if s != "PASS")
    if n_fail:
        print(f"\n{n_fail} test(s) failed or errored.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} test(s) passed.")


if __name__ == "__main__":
    main()

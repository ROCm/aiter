# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL MOE a4w4 (fp4x2, per_1x32) kernels.

Tests:
  - Stage1 (gate+up GEMM): flydsl_moe_stage1 with a_dtype="fp4", b_dtype="fp4"
  - Stage2 (down-proj GEMM): flydsl_moe_stage2 with a_dtype="fp4", b_dtype="fp4"
  - End-to-end (stage1 + stage2 combined via FlyDSL)

Usage:
    python op_tests/test_flydsl_moe_a4w4.py                         # all tests
    python op_tests/test_flydsl_moe_a4w4.py --stage stage1          # stage1 only
    python op_tests/test_flydsl_moe_a4w4.py --stage stage2          # stage2 only
    python op_tests/test_flydsl_moe_a4w4.py --stage e2e             # end-to-end only
    python op_tests/test_flydsl_moe_a4w4.py -t 16 -t 128            # specific token counts
    python op_tests/test_flydsl_moe_a4w4.py --block-m 16 32 64      # specific block sizes
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
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.utility.fp4_utils import (
    e8m0_shuffle,
    moe_mxfp4_sort,
)

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
Q_DTYPE_W = dtypes.fp4x2


# ---------------------------------------------------------------------------
# Shared data generation
# ---------------------------------------------------------------------------


def _generate_a4w4_data(
    token: int,
    model_dim: int,
    inter_dim: int,
    E: int,
    topk: int,
    block_m: int,
    dtype=torch.bfloat16,
    doweight_stage1: bool = False,
):
    """Generate quantised a4w4 data with torch reference outputs for stage1 and stage2."""
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # Quantize weights
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    # Quantize activation (stage1 input)
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=Q_DTYPE_A)

    # Torch reference: stage1
    ref1 = torch_moe_stage1(
        a1_qt,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=Q_TYPE,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        doweight=doweight_stage1,
    )

    # Quantize stage2 activation (stage1 output)
    a2_qt, a2_scale = torch_quant(ref1, quant_dtype=Q_DTYPE_A)
    a2_qt = a2_qt.view(token, topk, -1)

    # Torch reference: stage2
    ref2 = torch_moe_stage2(
        a2_qt,
        w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2),
        w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2),
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=Q_TYPE,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=not doweight_stage1,
    )

    # MoE sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    if doweight_stage1:
        sorted_weights_s1 = sorted_weights
        sorted_weights_s2 = None
    else:
        sorted_weights_s1 = None
        sorted_weights_s2 = sorted_weights

    # Stage1 now follows CK preshuffle for fp4 weights/scales.
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    # Sort activation scales for MoE dispatch
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )
    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[: token * topk, :].view(token, topk, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )

    return dict(
        # References
        ref_stage1=ref1,
        ref_stage2=ref2,
        # Quantised tensors
        a1_qt=a1_qt,
        a1_scale=a1_scale,
        a1_scale_sort=a1_scale_sort,
        a2_qt=a2_qt,
        a2_scale=a2_scale,
        a2_scale_sort=a2_scale_sort,
        w1_qt=w1_qt,
        w1_qt_shuf=w1_qt_shuf,
        w1_scale=w1_scale,
        w1_scale_shuf=w1_scale_shuf,
        w2_qt=w2_qt,
        w2_qt_shuf=w2_qt_shuf,
        w2_scale=w2_scale,
        w2_scale_shuf=w2_scale_shuf,
        # Sorting results
        sorted_ids=sorted_ids,
        sorted_weights=sorted_weights,
        sorted_weights_s1=sorted_weights_s1,
        sorted_weights_s2=sorted_weights_s2,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        # Shape info
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
    """Compare outputs and print result. Returns (passed, max_delta, pct_close)."""
    max_delta = (ref_out.float() - test_out.float()).abs().max().item()
    close_mask = torch.isclose(ref_out.float(), test_out.float(), atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct

    print(
        f"  max_delta={max_delta:.4f}, {pct_close:.1f}% close (atol={atol}, rtol={rtol})"
    )
    print(f"  ref  sample: {ref_out.reshape(-1)[:8]}")
    print(f"  test sample: {test_out.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta, pct_close


# ---------------------------------------------------------------------------
# Stage1 test: FlyDSL flydsl_moe_stage1 a4w4
# ---------------------------------------------------------------------------


def test_flydsl_stage1_a4w4(
    token: int = 16,
    model_dim: int = 7168,
    inter_dim: int = 256,
    E: int = 256,
    topk: int = 8,
    block_m: int = 32,
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage1 A4W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}"
    )
    print(f"{'='*70}")

    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a1_scale_sort"],
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage1"]
    return _check_result(ref, out, "stage1_a4w4", atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Stage2 test: FlyDSL flydsl_moe_stage2 a4w4
# ---------------------------------------------------------------------------


def test_flydsl_stage2_a4w4(
    token: int = 16,
    model_dim: int = 7168,
    inter_dim: int = 256,
    E: int = 256,
    topk: int = 8,
    block_m: int = 32,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL stage2 A4W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    out = flydsl_moe_stage2(
        inter_states=data["a2_qt"],
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage2"]
    return _check_result(ref, out, f"stage2_a4w4_{mode}", atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# End-to-end test: FlyDSL stage1 + stage2 combined
# ---------------------------------------------------------------------------


def test_flydsl_e2e_a4w4(
    token: int = 16,
    model_dim: int = 7168,
    inter_dim: int = 256,
    E: int = 256,
    topk: int = 8,
    block_m: int = 32,
    mode: str = "atomic",
    atol: float = 1.0,
    rtol: float = 0.05,
):
    """End-to-end test: FlyDSL stage1 output -> quantise -> FlyDSL stage2."""
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    print(f"\n{'='*70}")
    print(
        f"[TEST] FlyDSL E2E A4W4: token={token}, dim=({model_dim},{inter_dim}), "
        f"E={E}, topk={topk}, block_m={block_m}, mode={mode}"
    )
    print(f"{'='*70}")

    torch_quant = aiter.get_torch_quant(Q_TYPE)

    data = _generate_a4w4_data(
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        block_m=block_m,
    )

    out_dtype_str = "bf16" if data["dtype"] == torch.bfloat16 else "f16"

    # Stage1: FlyDSL
    stage1_out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a1_scale_sort"],
        sorted_weights=data["sorted_weights_s1"],
    )
    torch.cuda.synchronize()

    # Quantise stage1 output for stage2 input
    stage1_flat = stage1_out.view(-1, inter_dim)
    a2_qt_e2e, a2_scale_e2e = torch_quant(stage1_flat, quant_dtype=Q_DTYPE_A)
    a2_qt_e2e = a2_qt_e2e.view(token, topk, -1)

    a2_scale_sort_e2e = moe_mxfp4_sort(
        a2_scale_e2e[: token * topk, :].view(token, topk, -1),
        sorted_ids=data["sorted_ids"],
        num_valid_ids=data["num_valid_ids"],
        token_num=token,
        block_size=block_m,
    )

    # Stage2: FlyDSL
    e2e_out = flydsl_moe_stage2(
        inter_states=a2_qt_e2e,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=256,
        tile_k=256,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype_str,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=a2_scale_sort_e2e,
        sorted_weights=data["sorted_weights_s2"],
    )
    torch.cuda.synchronize()

    ref = data["ref_stage2"]
    return _check_result(
        ref, e2e_out, f"e2e_a4w4_{mode}", atol=atol, rtol=rtol, pass_pct=90.0
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FlyDSL MOE A4W4 FP4 unit tests",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=[16, 64, 256],
        help="Token counts to test (default: 16 64 256)",
    )
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=256)
    parser.add_argument("-k", "--topk", type=int, default=8)
    parser.add_argument("--block-m", type=int, nargs="+", default=[32])
    parser.add_argument(
        "--mode", type=str, nargs="+", default=["atomic"], choices=["atomic", "reduce"]
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
            # Stage1 tests
            if "stage1" in args.stage:
                try:
                    passed, max_delta, pct = test_flydsl_stage1_a4w4(
                        token=token,
                        model_dim=args.model_dim,
                        inter_dim=args.inter_dim,
                        E=args.experts,
                        topk=args.topk,
                        block_m=bm,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    results.append(
                        (
                            f"stage1_a4w4_t{token}_bm{bm}",
                            "PASS" if passed else "FAIL",
                            max_delta,
                            pct,
                        )
                    )
                except Exception:
                    import traceback

                    traceback.print_exc()
                    results.append((f"stage1_a4w4_t{token}_bm{bm}", "ERROR", 0, 0))

            # Stage2 tests
            if "stage2" in args.stage:
                for mode in args.mode:
                    try:
                        passed, max_delta, pct = test_flydsl_stage2_a4w4(
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
                                f"stage2_a4w4_t{token}_bm{bm}_{mode}",
                                "PASS" if passed else "FAIL",
                                max_delta,
                                pct,
                            )
                        )
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append(
                            (f"stage2_a4w4_t{token}_bm{bm}_{mode}", "ERROR", 0, 0)
                        )

            # End-to-end tests
            if "e2e" in args.stage:
                for mode in args.mode:
                    try:
                        passed, max_delta, pct = test_flydsl_e2e_a4w4(
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
                                f"e2e_a4w4_t{token}_bm{bm}_{mode}",
                                "PASS" if passed else "FAIL",
                                max_delta,
                                pct,
                            )
                        )
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        results.append(
                            (f"e2e_a4w4_t{token}_bm{bm}_{mode}", "ERROR", 0, 0)
                        )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, status, delta, pct in results:
        print(
            f"  {status:>5s}  {name:<50s}  max_delta={delta:>8.4f}  close={pct:>5.1f}%"
        )

    n_pass = sum(1 for _, s, _, _ in results if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")

    if any(s in ("FAIL", "ERROR") for _, s, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

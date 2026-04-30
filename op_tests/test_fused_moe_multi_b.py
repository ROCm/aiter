# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for multi-B DWDP support in aiter fused_moe() entry point.

Tests the full stack: fused_moe() -> _fused_moe_multi_b() -> fused_moe_2stages()
-> FlyDSL stage1/stage2 with List[Tensor] weights.

Comparison strategy:
  FP4: compute torch reference (torch_moe_stage1/stage2) with raw quantized
       weights, compare multi-B FlyDSL output against it.
  FP8: single-tensor fused_moe() as reference, compare multi-B against it.

Important: FP4 FlyDSL requires different weight/scale layouts for stage1 vs stage2:
  - w1 (stage1): shuffle_weight(w1, (16,16)) + e8m0_shuffle(w1_scale)
  - w2 (stage2): shuffle_weight_a16w4(w2, 16, False) + shuffle_scale_a16w4(w2_scale, E, False)

Usage:
    python test_fused_moe_multi_b.py                              # all tests
    python test_fused_moe_multi_b.py --precision fp4              # FP4 only
    python test_fused_moe_multi_b.py --precision fp8              # FP8 only
    python test_fused_moe_multi_b.py -t 1 16 64                   # specific tokens
    python test_fused_moe_multi_b.py --config small               # small model
    python test_fused_moe_multi_b.py --config deepseek            # production
"""

import argparse
import gc
import sys
import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import (
    fused_moe, fused_topk, torch_moe_stage1, torch_moe_stage2,
)
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle

torch.set_default_device("cuda")


# ============================================================================
# Helpers
# ============================================================================

def _split_tensor(t, splits, dim=0):
    """Split tensor along dim and .clone() each part for independent memory."""
    parts = torch.split(t, splits, dim=dim)
    return [p.clone() for p in parts]


def _split_weights_and_scales(w, w_scale, expert_splits):
    """Split weight and scale tensors into multi-B partitions.

    Handles FP4 scale layout where scale dim0 = E * rows_per_expert.
    Returns (w_list, scale_list) with cloned independent tensors.
    """
    w_list = _split_tensor(w, expert_splits, dim=0)

    if w_scale is None:
        return w_list, None

    if w_scale.shape[0] == w.shape[0]:
        # FP8: scale dim0 == expert count
        scale_list = _split_tensor(w_scale, expert_splits, dim=0)
    else:
        # FP4/MXFP4: scale dim0 = E * rows_per_expert
        total_experts = sum(expert_splits)
        rows_per_expert = w_scale.shape[0] // total_experts
        scale_splits = [s * rows_per_expert for s in expert_splits]
        scale_list = _split_tensor(w_scale, scale_splits, dim=0)

    return w_list, scale_list


def _check_result(ref_out, test_out, label, atol=1.0, rtol=0.15, pass_pct=92.0):
    """Compare outputs. Returns (passed, max_delta, pct_close)."""
    ref_f = ref_out.float()
    test_f = test_out.float()
    max_delta = (ref_f - test_f).abs().max().item()
    close_mask = torch.isclose(ref_f, test_f, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100
    passed = pct_close > pass_pct
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: max_delta={max_delta:.4f}, "
          f"{pct_close:.1f}% close (atol={atol}, rtol={rtol})")
    if not passed:
        print(f"    ref  sample: {ref_f.reshape(-1)[:8]}")
        print(f"    test sample: {test_f.reshape(-1)[:8]}")
    return passed, max_delta, pct_close


def _build_split_configs(E):
    """Generate split configs for testing.

    NOTE: Only 2-way splits are validated. 3-way and 4-way splits have a
    known issue in the high-level _fused_moe_multi_b -> fused_moe_2stages
    path where the output can be garbage for certain routing patterns, even
    though the low-level FlyDSL kernels work correctly with those splits.
    """
    configs = []
    # 2-way even — primary DWDP config
    if E >= 2 and E % 2 == 0:
        configs.append([E // 2, E // 2])
    # 2-way uneven
    if E >= 3:
        configs.append([E // 3, E - E // 3])
    return configs


def _build_experimental_split_configs(E):
    """Generate 3-way and 4-way split configs (known issues, experimental)."""
    configs = []
    if E >= 6:
        a = E // 3
        b = E // 3
        configs.append([a, b, E - a - b])
    if E >= 8:
        base = E // 4
        rem = E % 4
        splits = [base] * 4
        for i in range(rem):
            splits[i] += 1
        configs.append(splits)
    return configs


# ============================================================================
# Data generation
# ============================================================================

def generate_fp4_data(token, model_dim, inter_dim, E, topk, dtype=torch.bfloat16):
    """Generate quantized FP4 (MXFP4) data for fused_moe testing.

    FP4 FlyDSL kernels require specific shuffle layouts:
      - w1 (stage1): shuffle_weight(w1, (16,16)) + e8m0_shuffle(w1_scale)
      - w2 (stage2): shuffle_weight_a16w4(w2, 16, False) + shuffle_scale_a16w4(w2_scale, E, False)

    Also generates torch reference outputs for comparison.
    """
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    torch.manual_seed(42)

    x = torch.randn((token, model_dim), dtype=dtype) / 10

    # Quantize weights
    w1_bf16 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2_bf16 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10

    w1_qt, w1_scale = torch_quant(w1_bf16, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_scale = torch_quant(w2_bf16, quant_dtype=dtypes.fp4x2)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)

    # Routing
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    # --- Torch reference ---
    print("  Computing torch reference (stage1)...")
    ref_s1 = torch_moe_stage1(
        x, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=None, w1_scale=w1_scale,
        doweight=False,
    )
    print("  Computing torch reference (stage2)...")
    ref_s1_qt, ref_s1_scale = torch_quant(ref_s1, quant_dtype=dtypes.fp4x2)
    ref_s1_qt = ref_s1_qt.view(token, topk, -1)
    ref_out = torch_moe_stage2(
        ref_s1_qt, w1_qt, w2_qt, topk_weights, topk_ids,
        dtype=dtype, quant_type=QuantType.per_1x32,
        w2_scale=w2_scale, a2_scale=ref_s1_scale,
        doweight=True,
    )
    del ref_s1, ref_s1_qt, ref_s1_scale, w1_bf16, w2_bf16

    # --- Preshuffle for FlyDSL ---
    # Stage1: standard fp4 shuffle
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    # Stage2: a16w4-specific shuffle (required by FlyDSL stage2 FP4 kernel)
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    del w1_qt, w2_qt, w1_scale, w2_scale
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
        x=x,
        w1_qt_shuf=w1_qt_shuf,
        w2_qt_shuf=w2_qt_shuf,
        w1_scale_shuf=w1_scale_shuf,
        w2_scale_shuf=w2_scale_shuf,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        ref_out=ref_out,
        E=E, topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtype,
    )


def generate_fp8_data(token, model_dim, inter_dim, E, topk, dtype=torch.bfloat16):
    """Generate quantized FP8 data for fused_moe testing."""
    torch_quant = aiter.get_torch_quant(QuantType.per_Token)
    torch.manual_seed(42)

    x = torch.randn((token, model_dim), dtype=dtype)
    w1_bf16 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2_bf16 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10

    w1_qt, w1_scale = torch_quant(w1_bf16, quant_dtype=dtypes.fp8)
    w2_qt, w2_scale = torch_quant(w2_bf16, quant_dtype=dtypes.fp8)
    w1_qt = w1_qt.view(w1_bf16.shape)
    w2_qt = w2_qt.view(w2_bf16.shape)

    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(x, score, topk, True)

    # Preshuffle (same for CK and FlyDSL in FP8)
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shuf = shuffle_weight(w2_qt, (16, 16))

    return dict(
        x=x,
        w1_qt_shuf=w1_qt_shuf,
        w2_qt_shuf=w2_qt_shuf,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        E=E, topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        dtype=dtype,
    )


# ============================================================================
# Test functions -- FP4 (MXFP4)
# ============================================================================

def test_fp4_multi_b(data, splits, ref_out, **check_kw):
    """FP4: multi-B split vs torch reference."""
    label = f"fp4_multi_b_{len(splits)}way_{splits}"
    print(f"\n--- FP4 fused_moe: {label} ---")

    w1_list, w1s_list = _split_weights_and_scales(
        data["w1_qt_shuf"], data["w1_scale_shuf"], splits,
    )
    w2_list, w2s_list = _split_weights_and_scales(
        data["w2_qt_shuf"], data["w2_scale_shuf"], splits,
    )
    for w in w1_list + w2_list:
        w.is_shuffled = True

    out = fused_moe(
        data["x"],
        w1_list, w2_list,
        data["topk_weights"], data["topk_ids"],
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        w1_scale=w1s_list, w2_scale=w2s_list,
    )
    return _check_result(ref_out, out, label, **check_kw), out


def test_fp4_consistency(data, splits_a, splits_b, **check_kw):
    """FP4: check that different splits produce same output."""
    label_a = f"{len(splits_a)}way"
    label_b = f"{len(splits_b)}way"
    label = f"fp4_consistency_{label_a}_vs_{label_b}"
    print(f"\n--- FP4 consistency: {label} ---")

    def _run(splits):
        w1_list, w1s_list = _split_weights_and_scales(
            data["w1_qt_shuf"], data["w1_scale_shuf"], splits)
        w2_list, w2s_list = _split_weights_and_scales(
            data["w2_qt_shuf"], data["w2_scale_shuf"], splits)
        for w in w1_list + w2_list:
            w.is_shuffled = True
        return fused_moe(
            data["x"], w1_list, w2_list,
            data["topk_weights"], data["topk_ids"],
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Swiglu,
            w1_scale=w1s_list, w2_scale=w2s_list)

    out_a = _run(splits_a)
    out_b = _run(splits_b)
    # Consistency check uses tighter tolerance
    return _check_result(out_a, out_b, label,
                         atol=0.001, rtol=0.001, pass_pct=99.0)


# ============================================================================
# Test functions -- FP8
# ============================================================================

def test_fp8_single(data, **check_kw):
    """FP8: single-tensor baseline through fused_moe()."""
    print("\n--- FP8 fused_moe: single tensor ---")
    data["w1_qt_shuf"].is_shuffled = True
    data["w2_qt_shuf"].is_shuffled = True

    out = fused_moe(
        data["x"],
        data["w1_qt_shuf"],
        data["w2_qt_shuf"],
        data["topk_weights"],
        data["topk_ids"],
        quant_type=QuantType.per_Token,
        activation=ActivationType.Swiglu,
        w1_scale=data["w1_scale"],
        w2_scale=data["w2_scale"],
    )
    print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"  Output sample: {out.reshape(-1)[:8]}")
    return out


def test_fp8_multi_b(data, splits, ref_out, **check_kw):
    """FP8: multi-B split vs single-tensor reference."""
    label = f"fp8_multi_b_{len(splits)}way_{splits}"
    print(f"\n--- FP8 fused_moe: {label} ---")

    w1_list, w1s_list = _split_weights_and_scales(
        data["w1_qt_shuf"], data["w1_scale"], splits,
    )
    w2_list, w2s_list = _split_weights_and_scales(
        data["w2_qt_shuf"], data["w2_scale"], splits,
    )
    for w in w1_list + w2_list:
        w.is_shuffled = True

    out = fused_moe(
        data["x"],
        w1_list, w2_list,
        data["topk_weights"], data["topk_ids"],
        quant_type=QuantType.per_Token,
        activation=ActivationType.Swiglu,
        w1_scale=w1s_list, w2_scale=w2s_list,
    )
    return _check_result(ref_out, out, label, **check_kw)


# ============================================================================
# Main
# ============================================================================

# FP4 requires inter_dim >= 256 for shuffle_scale_a16w4 compatibility
MODEL_CONFIGS = {
    "small": dict(model_dim=256, inter_dim=256, E=8, topk=2),
    "medium": dict(model_dim=1024, inter_dim=512, E=32, topk=4),
    "deepseek": dict(model_dim=7168, inter_dim=2048, E=257, topk=8),
}

SEP = "=" * 70


def main():
    parser = argparse.ArgumentParser(description="Multi-B DWDP fused_moe tests")
    parser.add_argument("--precision", choices=["fp4", "fp8", "all"], default="all")
    parser.add_argument("-t", "--tokens", type=int, nargs="+",
                        default=[4, 16, 32, 64])
    parser.add_argument("--config", choices=list(MODEL_CONFIGS.keys()), default="small")
    parser.add_argument("--atol-fp4", type=float, default=0.1)
    parser.add_argument("--rtol-fp4", type=float, default=0.15)
    parser.add_argument("--atol-fp8", type=float, default=0.5)
    parser.add_argument("--rtol-fp8", type=float, default=0.1)
    parser.add_argument("--pass-pct", type=float, default=95.0)
    parser.add_argument("--experimental", action="store_true",
                        help="Also test 3/4-way splits (known issues, results not counted)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.config]
    E = cfg["E"]
    split_configs = _build_split_configs(E)
    exp_splits = _build_experimental_split_configs(E) if args.experimental else []

    print(SEP)
    print(f"Multi-B DWDP fused_moe tests -- config={args.config}")
    print(f"  E={E}, model_dim={cfg['model_dim']}, inter_dim={cfg['inter_dim']}, topk={cfg['topk']}")
    print(f"  Tokens: {args.tokens}")
    print(f"  Splits: {split_configs}")
    if exp_splits:
        print(f"  Experimental splits: {exp_splits}")
    print(SEP)

    results = []

    for token in args.tokens:
        print(f"\n{SEP}")
        print(f"Token count: {token}")
        print(SEP)

        # --- FP4 tests ---
        if args.precision in ("fp4", "all"):
            fp4_check = dict(atol=args.atol_fp4, rtol=args.rtol_fp4,
                             pass_pct=args.pass_pct)
            try:
                fp4_data = generate_fp4_data(token=token, **cfg)
                ref_out = fp4_data["ref_out"]
                print(f"  Torch ref sample: {ref_out.reshape(-1)[:6]}")

                # Multi-B vs torch reference
                multi_b_outputs = {}
                for splits in split_configs:
                    (p, _, _), out = test_fp4_multi_b(
                        fp4_data, splits, ref_out, **fp4_check)
                    results.append((f"fp4_{len(splits)}way_t{token}", p))
                    multi_b_outputs[tuple(splits)] = out

                # Consistency: different splits should give same output
                if len(split_configs) >= 2:
                    p, _, _ = test_fp4_consistency(
                        fp4_data, split_configs[0], split_configs[-1])
                    results.append((f"fp4_consistency_t{token}", p))

                # Experimental 3/4-way splits (not counted toward pass/fail)
                for splits in exp_splits:
                    try:
                        (p, _, _), _ = test_fp4_multi_b(
                            fp4_data, splits, ref_out, **fp4_check)
                        tag = "XPASS" if p else "XFAIL"
                        results.append((f"fp4_{len(splits)}way_t{token} [experimental]", None))
                        print(f"    [{tag}] (experimental, not counted)")
                    except Exception as e:
                        print(f"    [XERR] {len(splits)}-way experimental: {e}")

            except Exception:
                import traceback
                traceback.print_exc()
                results.append((f"fp4_ERROR_t{token}", False))
            finally:
                try:
                    del fp4_data, ref_out, multi_b_outputs
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

        # --- FP8 tests ---
        if args.precision in ("fp8", "all"):
            fp8_check = dict(atol=args.atol_fp8, rtol=args.rtol_fp8,
                             pass_pct=args.pass_pct)
            try:
                fp8_data = generate_fp8_data(token=token, **cfg)
                ref_out = test_fp8_single(fp8_data, **fp8_check)
                results.append((f"fp8_single_t{token}", True))

                for splits in split_configs:
                    p, _, _ = test_fp8_multi_b(
                        fp8_data, splits, ref_out, **fp8_check)
                    results.append((f"fp8_{len(splits)}way_t{token}", p))

            except Exception:
                import traceback
                traceback.print_exc()
                results.append((f"fp8_ERROR_t{token}", False))
            finally:
                try:
                    del fp8_data, ref_out
                except NameError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    counted = [(l, p) for l, p in results if p is not None]
    n_pass = sum(1 for _, p in counted if p)
    n_total = len(counted)

    for label, passed in results:
        if passed is None:
            print(f"  [SKIP] {label}")
        else:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    print()
    print(f"  {n_pass}/{n_total} passed")
    print(SEP)

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())

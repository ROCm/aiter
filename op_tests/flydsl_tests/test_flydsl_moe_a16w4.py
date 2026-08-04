# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Correctness tests for the FlyDSL fp4_bf16 MoE kernels via fused_moe.

Tests the INTERLEAVE and SEPARATED dispatch paths with bf16 activations and
fp4 weights (a16w4). Compares FlyDSL output against a CK-tile reference by
calling fused_moe twice — once with AITER_FLYDSL_FORCE=0 (CK) and once with
AITER_FLYDSL_FORCE=1 (FlyDSL).

This specifically exercises:
  - The INTERLEAVE bypass in fused_moe_ / fused_moe_2stages (q_dtype_a guard)
  - The SEPARATED branch in fused_moe_2stages (q_dtype_a guard)
  - The FlyDSL kernel compilation and correctness for both gate modes

Run:
    python3 op_tests/flydsl_tests/test_flydsl_moe_a16w4.py
"""

import os

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_gfx
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import checkAllclose


def _setup(model_dim, inter_dim, E, topk, tokens, gate_mode, seed=42):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = dtypes.bf16
    qfn = aiter.get_torch_quant(QuantType.per_1x32)
    gui = gate_mode == "interleave"

    inp = torch.randn(tokens, model_dim, dtype=dtype, device=device)

    w1 = torch.randn(E, 2 * inter_dim, model_dim, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(E, model_dim, inter_dim, dtype=dtype, device=device) * 0.02

    # Quantize
    w1_q, w1_sc = qfn(w1.view(E * 2 * inter_dim, model_dim))
    w1_q = w1_q.view(E, 2 * inter_dim, -1)
    w1_sc = w1_sc.view(E * 2 * inter_dim, -1)
    w2_q, w2_sc = qfn(w2.view(E * model_dim, inter_dim))
    w2_q = w2_q.view(E, model_dim, -1)
    w2_sc = w2_sc.view(E * model_dim, -1)

    # Preshuffle
    w1_s = shuffle_weight_a16w4(w1_q, 16, gui)
    w1_sc_s = shuffle_scale_a16w4(w1_sc, E, gui)
    w2_s = shuffle_weight_a16w4(w2_q, 16, False)
    w2_sc_s = shuffle_scale_a16w4(w2_sc, E, False)

    score = torch.randn(tokens, E, dtype=dtype, device=device)
    tw, tid = fused_topk(inp, score, topk, True)

    return inp, w1_s, w2_s, w1_sc_s, w2_sc_s, tw, tid


def _call_fused_moe(inp, w1, w2, w1_sc, w2_sc, tw, tid, gate_mode, flydsl_force):
    old = os.environ.get("AITER_FLYDSL_FORCE")
    os.environ["AITER_FLYDSL_FORCE"] = "1" if flydsl_force else "0"
    try:
        return fused_moe(
            inp,
            w1,
            w2,
            tw,
            tid,
            w1_scale=w1_sc,
            w2_scale=w2_sc,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            gate_mode=gate_mode,
        )
    finally:
        if old is None:
            os.environ.pop("AITER_FLYDSL_FORCE", None)
        else:
            os.environ["AITER_FLYDSL_FORCE"] = old


SHAPES = [
    # (model_dim, inter_dim, E, topk, tokens)
    (3072, 512, 128, 4, 1),  # GPT-OSS tok=1 decode
    (3072, 512, 128, 4, 4),  # GPT-OSS tok=4 decode
    (7168, 256, 256, 8, 1),  # DSR1-like tok=1 decode
    (7168, 256, 256, 8, 4),  # DSR1-like tok=4 decode
]

A16W4_TUNED_CONFIGS = [
    "gptoss_a16w4_tuned_fmoe.csv",
    "qwen3_5_397b_a16w4_tuned_fmoe.csv",
]


def _a16w4_tuned_rows():
    """Yield (config, row_index, kernelName1, kernelName2) for shipped a16w4 configs."""
    import csv
    from pathlib import Path

    cfg_dir = Path(aiter.__file__).parent / "configs" / "model_configs"
    for cfg in A16W4_TUNED_CONFIGS:
        path = cfg_dir / cfg
        if not path.exists():
            continue
        with open(path, newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                kn1 = (row.get("kernelName1") or "").strip()
                kn2 = (row.get("kernelName2") or "").strip()
                if kn1.startswith("flydsl_") or kn2.startswith("flydsl_"):
                    yield cfg, i, kn1, kn2


@pytest.mark.parametrize("cfg,row,kn1,kn2", list(_a16w4_tuned_rows()))
def test_a16w4_tuned_config_kernels_are_valid(cfg, row, kn1, kn2):
    """Every flydsl kernel a shipped a16w4 config selects must exist and be buildable."""
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    for kn in (kn1, kn2):
        if not kn.startswith("flydsl_"):
            continue
        params = get_flydsl_kernel_params(kn)
        assert params is not None, f"{cfg} row {row}: unknown flydsl kernel {kn!r}"

        if params["stage"] == 1 and params.get("b_dtype") == "fp4bf16":
            assert kn.endswith("_gui") or "_gui_" in kn, (
                f"{cfg} row {row}: {kn!r} is a SEPARATED fp4_bf16 stage1 kernel, but "
                "flydsl fp4_bf16 stage1 is INTERLEAVE-only"
            )
            tile_n = params["tile_n"]
            assert tile_n % 128 == 0, (
                f"{cfg} row {row}: {kn!r} has tile_n={tile_n}; INTERLEAVE needs "
                "tile_n >= 128 so each wave holds at least one gate+up pair"
            )


@pytest.mark.skipif(get_gfx() not in ["gfx950"], reason="fp4_bf16 requires gfx950")
@pytest.mark.parametrize("model_dim,inter_dim,E,topk,tokens", SHAPES)
@pytest.mark.parametrize("gate_mode", ["interleave", "separated"])
def test_fp4_bf16_fused_moe(model_dim, inter_dim, E, topk, tokens, gate_mode):
    inp, w1, w2, w1_sc, w2_sc, tw, tid = _setup(
        model_dim, inter_dim, E, topk, tokens, gate_mode
    )

    # Reference: CK-tile path
    ref = _call_fused_moe(
        inp, w1, w2, w1_sc, w2_sc, tw, tid, gate_mode, flydsl_force=False
    )
    # FlyDSL path
    out = _call_fused_moe(
        inp, w1, w2, w1_sc, w2_sc, tw, tid, gate_mode, flydsl_force=True
    )

    err = checkAllclose(
        ref,
        out,
        atol=0.5,
        rtol=0.1,
        msg=f"{gate_mode} {model_dim}x{inter_dim} E={E} k={topk} t={tokens}",
    )
    assert err < 0.05, (
        f"logits_diff too large ({err:.4f}) for {gate_mode} {tokens}t — "
        "FlyDSL and CK-tile outputs diverge."
    )


if __name__ == "__main__":
    if get_gfx() not in ["gfx950"]:
        print("Skipping: fp4_bf16 requires gfx950")
        raise SystemExit(0)

    for model_dim, inter_dim, E, topk, tokens in SHAPES:
        for gate_mode in ["interleave", "separated"]:
            label = f"{gate_mode} {model_dim}x{inter_dim} E={E} k={topk} t={tokens}"
            print(f"\n--- {label} ---")
            inp, w1, w2, w1_sc, w2_sc, tw, tid = _setup(
                model_dim, inter_dim, E, topk, tokens, gate_mode
            )
            ref = _call_fused_moe(
                inp, w1, w2, w1_sc, w2_sc, tw, tid, gate_mode, flydsl_force=False
            )
            out = _call_fused_moe(
                inp, w1, w2, w1_sc, w2_sc, tw, tid, gate_mode, flydsl_force=True
            )
            err = checkAllclose(ref, out, atol=0.5, rtol=0.1, msg=label)
            assert err < 0.05, f"FAIL: logits_diff={err:.4f}"
            print(f"  PASS (logits_diff={err:.4f})")

    print("\nAll tests passed.")

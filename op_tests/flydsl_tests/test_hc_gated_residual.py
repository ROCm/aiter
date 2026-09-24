# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the two-stage Hyper-Connection Gated-Residual op
(SILOTIGER-1042), ``aiter.ops.flydsl.kernels.hyper_connection_gated_residual``.

Exercises the three ticket entry points (``combine_and_mix`` / ``mix`` /
``combine``), the final mixer (no inject), and the decode tail -- all against the
shared float32 oracle in the package's ``reference`` module. Both weight modes
are covered: ``fold_w=False`` (K1 applies the RMSNorm affine) and ``fold_w=True``
((1+w) pre-folded into the down weight). The tile-aligned sweep spans both K1
reduction branches: split-K (M<3072) and the decoupled async-LDS pipe (M>=3072).

Aiter script convention (not pytest): run directly; exits non-zero on failure.

    python op_tests/flydsl_tests/test_hc_gated_residual.py
    python op_tests/flydsl_tests/test_hc_gated_residual.py --tokens 512 4096
"""

import argparse
import sys

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
    flydsl_gr_two_stage_combine,
    flydsl_gr_two_stage_combine_and_mix,
    flydsl_gr_two_stage_mix,
    fold_norm_weight,
    gr_combine,
    gr_combine_and_mix,
    gr_mix,
    merge_gr_two_stage_weight,
)
from aiter.test_common import checkAllclose

# Shipped model dimensions (Qwen3.8-Flash-Next).
HC, HS, LOWRANK = 4, 2560, 320
EPS = 1e-6

_FAILURES = []


def _check(ref, out, label, atol=0.06, rtol=0.05, max_err_ratio=0.03):
    if out.isnan().any() or out.isinf().any():
        print(f"[FAIL] {label}: output has NaN/Inf")
        _FAILURES.append(label)
        return
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    if not (err == 0 or err <= max_err_ratio):
        print(f"[FAIL] {label}: mismatch ratio {err} > {max_err_ratio}")
        _FAILURES.append(label)


def _make_inputs(tokens, *, with_inject=True, seed=0):
    torch.manual_seed(seed)
    hidden = HC * HS

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    return {
        "residual": _rand(tokens, hidden),
        "block_output": _rand(tokens, HS),
        "injection": _rand(tokens, HC),
        "norm_weight": _rand(HS, scale=0.1),
        "w_down": _rand(LOWRANK, hidden, scale=hidden**-0.5),
        "w_up": _rand(hidden, LOWRANK, scale=LOWRANK**-0.5),
        "w_inject": _rand(HC, hidden, scale=hidden**-0.5) if with_inject else None,
    }


def _ref(inp, which):
    """Oracle output for the given entry point (production bf16 numerics)."""
    if which == "combine_and_mix":
        return gr_combine_and_mix(
            inp["residual"], inp["block_output"], inp["injection"],
            inp["norm_weight"], inp["w_down"], inp["w_up"], inp["w_inject"],
            HC, EPS, round_bf16=True,
        )
    if which == "mix":
        block_input, inj_next = gr_mix(
            inp["residual"], inp["norm_weight"], inp["w_down"], inp["w_up"],
            inp["w_inject"], HC, EPS, round_bf16=True,
        )
        return inp["residual"].float(), block_input, inj_next
    if which == "combine":
        return gr_combine(
            inp["residual"], inp["block_output"], inp["injection"], HC,
            round_bf16=True,
        )
    raise ValueError(which)


def _merged(inp, fold_w):
    """Pre-folded merged weight when fold_w, else the plain merge (or None)."""
    if not fold_w:
        return None
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
    return fold_norm_weight(merged, inp["norm_weight"], HC)


def test_combine_and_mix(tokens, fold_w):
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        inp["w_down"], inp["w_up"], inp["w_inject"], HC, EPS,
        w_down_merged=_merged(inp, fold_w), fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
    _check(r2_ref.to(r2.dtype), r2, f"cmix[{tag}][M={tokens}] r2", atol=0.05, rtol=0.02)
    _check(x_ref.to(x.dtype), x, f"cmix[{tag}][M={tokens}] x")
    _check(inj_ref.to(inj.dtype), inj.contiguous(), f"cmix[{tag}][M={tokens}] inj", atol=0.05)


def test_mix(tokens, fold_w):
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens, seed=1)
    r2, x, inj = flydsl_gr_two_stage_mix(
        inp["residual"], inp["norm_weight"], inp["w_down"], inp["w_up"],
        inp["w_inject"], HC, EPS, w_down_merged=_merged(inp, fold_w), fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "mix")
    _check(r2_ref.to(r2.dtype), r2, f"mix[{tag}][M={tokens}] r2 (==residual)", atol=0.0, rtol=0.0)
    _check(x_ref.to(x.dtype), x, f"mix[{tag}][M={tokens}] x")
    _check(inj_ref.to(inj.dtype), inj.contiguous(), f"mix[{tag}][M={tokens}] inj", atol=0.05)


def test_combine(tokens):
    inp = _make_inputs(tokens, seed=2)
    r2 = flydsl_gr_two_stage_combine(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        HC, EPS,
    )
    torch.cuda.synchronize()
    r2_ref = _ref(inp, "combine")
    _check(r2_ref.to(r2.dtype), r2, f"combine[M={tokens}] r2", atol=0.05, rtol=0.02)


def test_final_mixer_no_inject(tokens, fold_w):
    """Final mixer: ``w_inject=None`` -> no inject columns, ``inj_next=None``."""
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens, with_inject=False, seed=3)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        inp["w_down"], inp["w_up"], None, HC, EPS,
        w_down_merged=_merged(inp, fold_w), fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, _ = _ref(inp, "combine_and_mix")
    if inj is not None:
        print(f"[FAIL] final_mixer[{tag}][M={tokens}]: must not emit inject logits")
        _FAILURES.append(f"final_mixer[{tag}][M={tokens}] inj")
    _check(r2_ref.to(r2.dtype), r2, f"final_mixer[{tag}][M={tokens}] r2", atol=0.05, rtol=0.02)
    _check(x_ref.to(x.dtype), x, f"final_mixer[{tag}][M={tokens}] x")


def test_decode(tokens):
    """Decode M (fused-K2 + fold_w): skinny GEMV on gfx942, padded fused on
    gfx950 -- both at non-tile-multiple M vs the oracle."""
    inp = _make_inputs(tokens, seed=7)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"], inp["block_output"], inp["injection"], inp["norm_weight"],
        inp["w_down"], inp["w_up"], inp["w_inject"], HC, EPS,
        w_down_merged=_merged(inp, True), fold_w=True,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
    _check(r2_ref.to(r2.dtype), r2, f"decode[M={tokens}] r2", atol=0.05, rtol=0.02)
    _check(x_ref.to(x.dtype), x, f"decode[M={tokens}] x")
    _check(inj_ref.to(inj.dtype), inj.contiguous(), f"decode[M={tokens}] inj", atol=0.05)


parser = argparse.ArgumentParser(
    description="Correctness test for the two-stage HC Gated-Residual op (SILOTIGER-1042)."
)
parser.add_argument(
    "--tokens", type=int, nargs="+", default=[512, 2048, 4096],
    help="tile-aligned token counts (spans split-K <3072 and decouple >=3072).",
)
parser.add_argument(
    "--decode-tokens", type=int, nargs="+", default=[1, 3, 8, 32],
    help="small (decode) token counts for the fused-K2 skinny/tail-path check.",
)
args = parser.parse_args()

_arch = get_gfx()
if _arch not in ("gfx950", "gfx942"):
    print(f"[skip] two-stage HC gated-residual requires gfx950/gfx942 FlyDSL, got {_arch}")
    sys.exit(0)

for m in args.tokens:
    for fold_w in (False, True):
        test_combine_and_mix(m, fold_w)
        test_mix(m, fold_w)
        test_final_mixer_no_inject(m, fold_w)
    test_combine(m)
test_combine(64)  # extra tile-aligned combine size
for m in args.decode_tokens:
    test_decode(m)

if _FAILURES:
    print(f"\n{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
    sys.exit(1)
print("\nall checks passed")

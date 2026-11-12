# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the two-stage Hyper-Connection Gated-Residual op
(``aiter.ops.flydsl.kernels.hyper_connection_gated_residual``).

Checks all public entry points, the no-injection final mixer, decode and tiled
token counts, both norm-weight layouts, and folded/unfolded weights against the
float32 oracle. Also checks strided injection, direct-K1 extent guards, skinny
counter reuse across streams and graphs, and invalid up weights. Finished-gate
seam checks use FP64 references, four seeds, stress fixtures, both tiled K1 paths,
and exact BF16 combine parity.

    python op_tests/test_hc_gated_residual.py
    python op_tests/test_hc_gated_residual.py --tokens 512 4096
"""

import argparse
import dataclasses
import sys
from unittest.mock import patch

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl.hc_seam import (
    hc_gated_combine,
    hc_seam,
    hc_seam_prepare_weights,
    hc_seam_prepared,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import (
    flydsl_gr_two_stage_combine,
    flydsl_gr_two_stage_combine_and_mix,
    flydsl_gr_two_stage_mix,
    flydsl_k1_combine_norm_down,
    flydsl_up_gate_mix_norm,
    fold_norm_weight,
    gr_combine,
    gr_combine_and_mix,
    gr_grouped_rmsnorm,
    gr_mix,
    merge_gr_two_stage_weight,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.common import (
    merge_down_inject,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.k1 import (
    _build_down_deferred_partial_pipe,
)
from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.reference import (
    seam_combine_only_ref,
    seam_deferred_partials_ref,
    seam_ref_deferred,
    seam_ref_fp64,
    seam_ref_operational,
)
from aiter.test_common import checkAllclose

# Model dimensions covered by this test.
HC, HS, LOWRANK = 4, 2560, 320
EPS = 1e-6

# SGLang PR #46881 test_hc_seam.py asserts 2e-4 gate error on only 10
# sampled cases. Operational-vs-FP64 maxima over 208 cases per weight scale
# are 2.36e-4 (1x) and 3.80e-4 (2x): FP32/FP64 normalization flips BF16(X)
# boundaries. The approved 1e-3 gives 2.6–4x headroom; real gate bugs are
# around 1e-1. Mixed retains SGLang's 6e-3, without a mismatch allowance.
SEAM_MIXED_REL_L2 = 6e-3
SEAM_GATE_MAX_ABS = 1e-3
# Formulation allowance for forced-deferred calls against FP64: saturated-gates,
# M=255, seed 3 gives 1.219e-3 in the deferred three-launch
# formulation. Default hc_seam dispatch keeps SEAM_GATE_MAX_ABS.
SEAM_GATE_MAX_ABS_DEFERRED = 1.5e-3
# Bounds kernel error against its own formulation reference (measured maximum 3.4e-4).
SEAM_DEFERRED_KERNEL_REL_L2 = 5e-4
SEAM_SEEDS = (0, 1, 2, 3)
SEAM_STRESS = ("heavy-norm", "saturated-gates", "near-zero-rms", "stream-scales")
SEAM_ROWS = (1, 2, 7, 16, 17, 64, 65, 128, 129, 193, 224, 255, 256)

SEAM_UNDER_TEST = hc_seam
SEAM_COMBINE_UNDER_TEST = hc_gated_combine

_FAILURES = []
_SEAM_METRICS = []


def _check(ref, out, label, atol=0.06, rtol=0.05):
    """Every element must be within ``atol + rtol * |ref|``.

    No mismatch budget: a wrong row or column is a small fraction of a large
    output and would hide inside a ratio threshold.
    """
    if out.shape != ref.shape:
        print(f"[FAIL] {label}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        _FAILURES.append(label)
        return
    if out.isnan().any() or out.isinf().any():
        print(f"[FAIL] {label}: output has NaN/Inf")
        _FAILURES.append(label)
        return
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    if err != 0:
        bad = (out.float() - ref.float()).abs() > atol + rtol * ref.float().abs()
        rows = bad.reshape(bad.shape[0], -1).any(-1).nonzero().flatten()
        print(
            f"[FAIL] {label}: mismatch ratio {err} in {rows.numel()} row(s), "
            f"first {rows[:8].tolist()}"
        )
        _FAILURES.append(label)


def _expect_value_error(label, fn):
    try:
        fn()
    except ValueError:
        return
    print(f"[FAIL] {label}: expected ValueError")
    _FAILURES.append(label)


def _make_inputs(tokens, *, with_inject=True, full_norm=False, seed=0):
    torch.manual_seed(seed)
    hidden = HC * HS

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    return {
        "residual": _rand(tokens, hidden),
        "block_output": _rand(tokens, HS),
        "injection": _rand(tokens, HC),
        # full_norm selects per-stream weights instead of one shared stream weight.
        "norm_weight": _rand(hidden if full_norm else HS, scale=0.1),
        "w_down": _rand(LOWRANK, hidden, scale=hidden**-0.5),
        "w_up": _rand(hidden, LOWRANK, scale=LOWRANK**-0.5),
        "w_inject": _rand(HC, hidden, scale=hidden**-0.5) if with_inject else None,
    }


def _ref(inp, which):
    """Oracle output with the kernel's bf16 rounding order."""
    if which == "combine_and_mix":
        return gr_combine_and_mix(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            HC,
            EPS,
            round_bf16=True,
        )
    if which == "mix":
        block_input, inj_next = gr_mix(
            inp["residual"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            HC,
            EPS,
            round_bf16=True,
        )
        return inp["residual"].float(), block_input, inj_next
    if which == "combine":
        return gr_combine(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            HC,
            round_bf16=True,
        )
    raise ValueError(which)


def _merged(inp, fold_w):
    """Pre-folded merged weight when fold_w, else the plain merge (or None)."""
    if not fold_w:
        return None
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
    return fold_norm_weight(merged, inp["norm_weight"], HC)


def _run_combine_and_mix(tokens, fold_w):
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        w_down_merged=_merged(inp, fold_w),
        fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
    _check(r2_ref.to(r2.dtype), r2, f"cmix[{tag}][M={tokens}] r2", atol=0.05, rtol=0.02)
    _check(x_ref.to(x.dtype), x, f"cmix[{tag}][M={tokens}] x")
    _check(
        inj_ref.to(inj.dtype),
        inj.contiguous(),
        f"cmix[{tag}][M={tokens}] inj",
        atol=0.05,
    )


def _run_mix(tokens, fold_w):
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens, seed=1)
    r2, x, inj = flydsl_gr_two_stage_mix(
        inp["residual"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        w_down_merged=_merged(inp, fold_w),
        fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "mix")
    _check(
        r2_ref.to(r2.dtype),
        r2,
        f"mix[{tag}][M={tokens}] r2 (==residual)",
        atol=0.0,
        rtol=0.0,
    )
    _check(x_ref.to(x.dtype), x, f"mix[{tag}][M={tokens}] x")
    _check(
        inj_ref.to(inj.dtype),
        inj.contiguous(),
        f"mix[{tag}][M={tokens}] inj",
        atol=0.05,
    )


def _run_combine(tokens):
    inp = _make_inputs(tokens, seed=2)
    r2 = flydsl_gr_two_stage_combine(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        HC,
        EPS,
    )
    torch.cuda.synchronize()
    r2_ref = _ref(inp, "combine")
    _check(r2_ref.to(r2.dtype), r2, f"combine[M={tokens}] r2", atol=0.05, rtol=0.02)


def _run_strided_combine():
    """Feed the packed inj_next view directly into the standalone combine."""
    tokens = 3
    source = _make_inputs(tokens, seed=13)
    _, _, injection = flydsl_gr_two_stage_combine_and_mix(
        source["residual"],
        source["block_output"],
        source["injection"],
        source["norm_weight"],
        source["w_down"],
        source["w_up"],
        source["w_inject"],
        HC,
        EPS,
        w_down_merged=_merged(source, True),
        fold_w=True,
    )
    if injection.is_contiguous() or injection.stride(0) == HC:
        print("[FAIL] strided_combine: setup did not produce a strided injection")
        _FAILURES.append("strided_combine setup")
        return

    target = _make_inputs(tokens, seed=14)
    r2 = flydsl_gr_two_stage_combine(
        target["residual"],
        target["block_output"],
        injection,
        target["norm_weight"],
        HC,
        EPS,
    )
    torch.cuda.synchronize()
    ref = gr_combine(
        target["residual"], target["block_output"], injection, HC, round_bf16=True
    )
    _check(ref.to(r2.dtype), r2, "strided_combine r2", atol=0.05, rtol=0.02)


def _run_w_up_validation():
    """Invalid K2 weight dtype/layout must fail before K1 launches."""
    inp = _make_inputs(1, seed=15)
    bad_weights = {
        "float32": inp["w_up"].float(),
        "noncontiguous": inp["w_up"].t().contiguous().t(),
    }
    for label, w_up in bad_weights.items():
        _expect_value_error(
            f"w_up_validation[{label}]",
            lambda w_up=w_up: flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                w_up,
                inp["w_inject"],
                HC,
                EPS,
            ),
        )
    inp = _make_inputs(8, seed=31)  # M=8 coverage beyond the M=1 cases above
    weight = inp["w_up"]
    invalid = (
        (weight.float(), ValueError, "w_up must have dtype torch.bfloat16"),
        (weight.t().contiguous().t(), ValueError, "w_up must be row-major contiguous"),
        (weight.reshape(-1), ValueError, "w_up must have shape"),
        (weight[:-1].contiguous(), ValueError, "w_up must have shape"),
        (weight.cpu(), ValueError, "w_up must be on"),
    )
    for fold_w in (False, True):
        merged = _merged(inp, fold_w)
        for which in ("combine_and_mix", "mix"):
            for bad, error_type, message in invalid:
                args = {**inp, "w_up": bad}
                if which == "mix":
                    args.pop("block_output")
                    args.pop("injection")
                op = (
                    flydsl_gr_two_stage_mix
                    if which == "mix"
                    else flydsl_gr_two_stage_combine_and_mix
                )
                try:
                    op(
                        **args,
                        hc_count=HC,
                        eps=EPS,
                        w_down_merged=merged,
                        fold_w=fold_w,
                    )
                except error_type as error:
                    assert message in str(error), str(error)
                else:
                    torch.cuda.synchronize()
                    raise AssertionError(
                        f"{which} fold={fold_w} accepted invalid w_up: {message}"
                    )
    print("w_up validation passed")


def _cmix_call(inp, **overrides):
    args = dict(inp)
    args.update(overrides)
    fold_w = args.pop("fold_w", True)
    return lambda: flydsl_gr_two_stage_combine_and_mix(
        args["residual"],
        args["block_output"],
        args["injection"],
        args["norm_weight"],
        args["w_down"],
        args["w_up"],
        args["w_inject"],
        HC,
        EPS,
        fold_w=fold_w,
    )


def _run_input_validation():
    """Malformed shapes must raise before any kernel reads past a buffer."""
    for tokens in (1, 64):  # skinny decode and tiled dispatch
        inp = _make_inputs(tokens, seed=21)
        cases = {
            # One inject row short: the old merge zero-filled it silently.
            "short_w_inject": {"w_inject": inp["w_inject"][: HC - 1]},
            "w_inject_dtype": {"w_inject": inp["w_inject"].float()},
            # Neither HS (shared) nor HC*HS (per stream).
            "norm_2xHS": {"norm_weight": inp["norm_weight"].repeat(2)},
            "block_output_short": {"block_output": inp["block_output"][:, :-8]},
            "injection_narrow": {"injection": inp["injection"][:, : HC - 1]},
            "residual_rows": {"residual": inp["residual"][:-1]},
        }
        for name, overrides in cases.items():
            for fold_w in (False, True):
                _expect_value_error(
                    f"validation[{name}][M={tokens}][fold={fold_w}]",
                    _cmix_call(inp, fold_w=fold_w, **overrides),
                )
        merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
        _expect_value_error(
            f"validation[fold_norm_2xHS][M={tokens}]",
            lambda m=merged, w=inp["norm_weight"]: fold_norm_weight(m, w.repeat(2), HC),
        )
    inp = _make_inputs(1, seed=22)
    _expect_value_error(
        "validation[merge_short_w_inject]",
        lambda: merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"][:1], HC),
    )


def _run_zero_tokens():
    """``M=0`` returns correctly shaped empty outputs without launching."""
    inp = _make_inputs(0, seed=23)
    hidden = HC * HS
    for fold_w in (False, True):
        r2, x, inj = _cmix_call(inp, fold_w=fold_w)()
        mr2, mx, minj = flydsl_gr_two_stage_mix(
            inp["residual"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            HC,
            EPS,
            fold_w=fold_w,
        )
        got = [t.shape for t in (r2, x, inj, mr2, mx, minj)]
        want = [(0, hidden), (0, HS), (0, HC)] * 2
        if got != [torch.Size(s) for s in want]:
            print(f"[FAIL] zero_tokens[fold={fold_w}]: shapes {got}")
            _FAILURES.append(f"zero_tokens[fold={fold_w}]")
    r2 = flydsl_gr_two_stage_combine(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        HC,
        EPS,
    )
    if r2.shape != (0, hidden):
        print(f"[FAIL] zero_tokens[combine]: shape {tuple(r2.shape)}")
        _FAILURES.append("zero_tokens[combine]")


def _run_k2_override_limits():
    """Explicit K2 tiles beyond workgroup/LDS limits must be rejected."""
    tokens = 256
    lora = torch.randn(tokens, LOWRANK, device="cuda").bfloat16()
    r2 = torch.randn(tokens, HC * HS, device="cuda").bfloat16()
    rrms = torch.ones(tokens, HC, device="cuda")
    norm_weight = torch.zeros(HS, device="cuda").bfloat16()
    w_up = torch.randn(HC * HS, LOWRANK, device="cuda").bfloat16()
    w_up_il = torch.zeros(160, 4, 16, LOWRANK, device="cuda").bfloat16()
    cases = {
        "threads": {"block_m": 256, "block_n": 64, "m_waves": 8, "n_waves": 4},
        "lds": {"block_m": 256, "block_n": 256, "m_waves": 1, "n_waves": 1},
        "tiling": {"block_m": 64, "block_n": 24, "m_waves": 1, "n_waves": 1},
    }
    for name, cfg in cases.items():
        _expect_value_error(
            f"k2_override[{name}]",
            lambda cfg=cfg: flydsl_up_gate_mix_norm(
                lora, r2, rrms, norm_weight, w_up, HC, **cfg
            ),
        )
    # The register-branch kernel hardcodes hidden=10240, a full-width norm and
    # lowrank=320; anything else must be rejected rather than read out of bounds.
    reg = {"block_m": 64, "block_n": 64, "m_waves": 1, "n_waves": 1}
    full_norm = torch.zeros(HC * HS, device="cuda").bfloat16()
    bad = {
        "hidden": (lora, r2[:, :8192].contiguous(), rrms, full_norm[:8192]),
        "shared_norm": (lora, r2, rrms, norm_weight),
        "lowrank": (lora[:, :160].contiguous(), r2, rrms, full_norm),
    }
    for name, (lo, rr, rm, nw) in bad.items():
        for mode in ("combined", "separate"):
            _expect_value_error(
                f"k2_register_branch[{name},{mode}]",
                lambda lo=lo, rr=rr, rm=rm, nw=nw, mode=mode: flydsl_up_gate_mix_norm(
                    lo, rr, rm, nw, w_up_il, HC, branch_mode=mode, **reg
                ),
            )


def _run_side_stream_pressure():
    """Temporaries must belong to the launch stream, not the ambient stream.

    The side stream is held behind a busy kernel, then the ambient stream
    allocates and poisons memory. Temporaries mistakenly owned by the ambient
    stream are recycled into the poison buffers, which the delayed kernels then
    overwrite. The ambient stream is not the legacy default stream, which would
    implicitly wait for the side stream and hide the race.
    """
    ambient, side = torch.cuda.Stream(), torch.cuda.Stream()
    for tokens in (4, 64):  # skinny decode and tiled K1/K2
        inp = _make_inputs(tokens, seed=24)
        merged = _merged(inp, True)
        torch.cuda.synchronize()
        with torch.cuda.stream(ambient):
            with torch.cuda.stream(side):
                torch.cuda._sleep(50_000_000)
            out = flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                HC,
                EPS,
                w_down_merged=merged,
                fold_w=True,
                stream=side,
            )
            poison = [
                torch.full((n,), float("nan"), device="cuda")
                for n in (1 << k for k in range(4, 23))
                for _ in range(4)
            ]
        torch.cuda.synchronize()
        label = f"side_stream[M={tokens}]"
        if not all(p.isnan().all() for p in poison):
            print(f"[FAIL] {label}: launch-stream writes hit recycled memory")
            _FAILURES.append(f"{label} recycled")
        del poison
        r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
        r2, x, inj = out
        _check(r2_ref.to(r2.dtype), r2, f"{label} r2", atol=0.05, rtol=0.02)
        _check(x_ref.to(x.dtype), x, f"{label} x")
        _check(inj_ref.to(inj.dtype), inj.contiguous(), f"{label} inj", atol=0.05)


def _run_skinny_publication_stress(reps=300):
    """Repeat skinny decode; the last-arriving wave must never read stale partials.

    Workgroups of one column group may land on different XCDs, whose L2s are
    not coherent. The cross-wave sum is accumulated in a fixed order, so every
    replay must be bit-identical to the first oracle-checked result.
    """
    for tokens in (1, 3, 8):
        inp = _make_inputs(tokens, seed=26)
        merged = _merged(inp, True)

        def run(inp=inp, merged=merged):
            return flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                HC,
                EPS,
                w_down_merged=merged,
                fold_w=True,
            )

        _, x0, inj0 = run()
        torch.cuda.synchronize()
        _, x_ref, inj_ref = _ref(inp, "combine_and_mix")
        label = f"skinny_stress[M={tokens}]"
        _check(x_ref.to(x0.dtype), x0, f"{label} x")
        _check(inj_ref.to(inj0.dtype), inj0.contiguous(), f"{label} inj", atol=0.05)
        outs = [run() for _ in range(reps)]
        torch.cuda.synchronize()
        bad = sum(
            not (torch.equal(x, x0) and torch.equal(inj, inj0)) for _, x, inj in outs
        )
        if bad:
            print(f"[FAIL] {label}: {bad}/{reps} replays differ from the first")
            _FAILURES.append(f"{label} replay")


def _make_shape_inputs(tokens, hc_count, stream_dim, seed):
    torch.manual_seed(seed)
    hidden = hc_count * stream_dim

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    return {
        "residual": _rand(tokens, hidden),
        "block_output": _rand(tokens, stream_dim),
        "injection": _rand(tokens, hc_count),
        "norm_weight": _rand(stream_dim, scale=0.1),
        "w_down": _rand(LOWRANK, hidden, scale=hidden**-0.5),
        "w_up": _rand(hidden, LOWRANK, scale=LOWRANK**-0.5),
        "w_inject": _rand(hc_count, hidden, scale=hidden**-0.5),
    }


def _run_untuned_hidden():
    """Tuned plans are keyed by tokens only; another hidden must still work.

    ``hidden = 2 * 2560`` hits entries measured for ``hidden = 4 * 2560``; any
    plan that cannot tile it must fall back to the heuristic.
    """
    hc_count = 2
    for tokens in (1, 64, 512):
        inp = _make_shape_inputs(tokens, hc_count, HS, seed=25)
        for fold_w in (False, True):
            merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], hc_count)
            if fold_w:
                merged = fold_norm_weight(merged, inp["norm_weight"], hc_count)
            r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                hc_count,
                EPS,
                w_down_merged=merged,
                fold_w=fold_w,
            )
            torch.cuda.synchronize()
            r2_ref, x_ref, inj_ref = gr_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                hc_count,
                EPS,
                round_bf16=True,
            )
            label = f"hc2[fold={fold_w}][M={tokens}]"
            _check(r2_ref.to(r2.dtype), r2, f"{label} r2", atol=0.05, rtol=0.02)
            _check(x_ref.to(x.dtype), x, f"{label} x")
            _check(inj_ref.to(inj.dtype), inj.contiguous(), f"{label} inj", atol=0.05)


def _make_hc_inputs(tokens, hc_count, seed):
    """Inputs with fixed hidden width and a variable number of HC streams."""
    torch.manual_seed(seed)
    # 8192 keeps both HC=4 and HC=8 stream widths divisible by the skinny
    # kernel's 512-element per-iteration width.
    hidden = 8192
    stream_dim = hidden // hc_count

    def _rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda") * scale).bfloat16()

    return {
        "residual": _rand(tokens, hidden),
        "block_output": _rand(tokens, stream_dim),
        "injection": _rand(tokens, hc_count),
        "norm_weight": _rand(stream_dim, scale=0.1),
        "w_down": _rand(LOWRANK, hidden, scale=hidden**-0.5),
        "w_up": _rand(hidden, LOWRANK, scale=LOWRANK**-0.5),
        "w_inject": _rand(hc_count, hidden, scale=hidden**-0.5),
    }


def _run_counter_hc_isolation():
    """Persistent skinny-counter phases must not leak across hc_count values."""

    def _run(inp, hc_count, stream=None):
        merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], hc_count)
        return flydsl_gr_two_stage_combine_and_mix(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            hc_count,
            EPS,
            w_down_merged=fold_norm_weight(merged, inp["norm_weight"], hc_count),
            fold_w=True,
            stream=stream,
        )

    def _check_case(inp, hc_count, out, label):
        r2, x, inj = out
        r2_ref, x_ref, inj_ref = gr_combine_and_mix(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            hc_count,
            EPS,
            round_bf16=True,
        )
        _check(r2_ref.to(r2.dtype), r2, f"{label} r2", atol=0.05, rtol=0.02)
        _check(x_ref.to(x.dtype), x, f"{label} x")
        _check(inj_ref.to(inj.dtype), inj.contiguous(), f"{label} inj", atol=0.05)

    # Both calls have n_pad=384 and m_groups=1. The old cache reused the HC=4
    # counter phase for HC=8 and selected the wrong last-arriving wave.
    _run(_make_hc_inputs(3, 4, 16), 4)
    inp = _make_hc_inputs(3, 8, 17)
    out = _run(inp, 8)
    torch.cuda.synchronize()
    _check_case(inp, 8, out, "counter_hc8")

    # Release two side streams together. They use identical counter dimensions
    # and would race if the cache did not isolate storage by stream.
    inputs = [_make_hc_inputs(8, 4, seed) for seed in (19, 20)]
    streams = [torch.cuda.Stream() for _ in inputs]
    start = torch.cuda.Event()
    for stream in streams:
        stream.wait_event(start)
    outputs = [_run(i, 4, s) for i, s in zip(inputs, streams)]
    start.record()
    torch.cuda.synchronize()
    for index, (case, result) in enumerate(zip(inputs, outputs)):
        _check_case(case, 4, result, f"counter_stream{index}")


def _run_skinny_concurrent_graphs():
    """HC4 and HC8 skinny on separate streams, eager and under graph replay, M=2 and 8.

    Counter storage is per stream and per hc_count; concurrent launches or
    replays of both widths must each match the oracle.
    """
    for tokens in (2, 8):
        cases = []
        for hc_count, seed in ((4, 31), (8, 32)):
            inp = _make_hc_inputs(tokens, hc_count, seed)
            merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], hc_count)
            merged = fold_norm_weight(merged, inp["norm_weight"], hc_count)
            ref = gr_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                hc_count,
                EPS,
                round_bf16=True,
            )
            cases.append((inp, merged, hc_count, ref))
        streams = (torch.cuda.Stream(), torch.cuda.Stream())

        def launch(case, stream):
            inp, merged, hc_count, _ = case
            return flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                hc_count,
                EPS,
                w_down_merged=merged,
                fold_w=True,
                stream=stream,
            )

        def verify(outs, label, cases=cases, tokens=tokens):
            for case, out in zip(cases, outs):
                r2, x, inj = out
                r2_ref, x_ref, inj_ref = case[3]
                tag = f"{label}[HC{case[2]},M={tokens}]"
                _check(r2_ref.to(r2.dtype), r2, f"{tag} r2", atol=0.05, rtol=0.02)
                _check(x_ref.to(x.dtype), x, f"{tag} x")
                _check(inj_ref.to(inj.dtype), inj.contiguous(), f"{tag} inj", atol=0.05)

        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for _ in range(20):
            outs = [launch(c, s) for c, s in zip(cases, streams)]
            torch.cuda.synchronize()
            verify(outs, "skinny_concurrent")
        graphs, outs = [], []
        for case, stream in zip(cases, streams):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outs.append(launch(case, stream))
            graphs.append(graph)
        for _ in range(20):
            for graph, stream in zip(graphs, streams):
                with torch.cuda.stream(stream):
                    graph.replay()
            torch.cuda.synchronize()
            verify(outs, "skinny_graph")
    print("skinny concurrent HC4/HC8 streams and graph replay passed")


def _check_direct_k1(inp, merged, hc_count, r2, packed, label):
    """Compare a direct unfolded K1 call against the oracle."""
    r2_ref = gr_combine(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        hc_count,
        round_bf16=True,
    )
    xn = gr_grouped_rmsnorm(r2_ref, inp["norm_weight"], hc_count, EPS)
    raw = xn.bfloat16().float() @ merged.t().float()
    lora = raw[:, :LOWRANK] / hc_count
    lora = lora * torch.sigmoid(lora)
    _check(r2_ref.to(r2.dtype), r2, f"{label} r2", atol=0.05, rtol=0.02)
    _check(lora.to(packed.dtype), packed[:, :LOWRANK], f"{label} lora")
    _check(
        raw[:, LOWRANK : LOWRANK + hc_count].to(packed.dtype),
        packed[:, LOWRANK : LOWRANK + hc_count],
        f"{label} inj",
        atol=0.05,
    )


def _run_direct_k1(inp, hc_count, label, **kwargs):
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], hc_count)
    r2, packed = flydsl_k1_combine_norm_down(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        merged,
        LOWRANK,
        hc_count,
        EPS,
        **kwargs,
    )
    torch.cuda.synchronize()
    _check_direct_k1(inp, merged, hc_count, r2, packed, label)


def _run_direct_k1_untiled_tokens():
    """A nearest tuned split-K tile must not overrun a direct M=224 call.

    Row counts no tile divides are rejected; ``gemm_pad`` makes them valid.
    """
    _run_direct_k1(_make_inputs(224, seed=18), HC, "direct_k1[M=224]")
    tokens = 100
    inp = _make_inputs(tokens, seed=19)
    for split_k in ("auto", 1, 4):
        _expect_value_error(
            f"direct_k1[M={tokens}][sk={split_k}]",
            lambda split_k=split_k: _run_direct_k1(inp, HC, "unused", split_k=split_k),
        )
        merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
        r2, packed = flydsl_k1_combine_norm_down(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            merged,
            LOWRANK,
            HC,
            EPS,
            split_k=split_k,
            gemm_pad=128,
        )
        torch.cuda.synchronize()
        _check_direct_k1(
            inp,
            merged,
            HC,
            r2[:tokens],
            packed[:tokens],
            f"direct_k1[M={tokens}][gemm_pad=128][sk={split_k}]",
        )


def _run_wide_stream_lds():
    """Staged ``(1+w)`` and rrms scratch count against LDS on every arch.

    ``stream_dim=8192`` unfolded makes the M=24576 tuned decoupled tile
    (128x384x64, 2 stages) need 165,888 B > gfx950's 163,840 B, so the N tile
    must shrink. ``stream_dim=40960`` cannot stage ``(1+w)`` beside any tile,
    so both pipelines must read it from global memory instead (on gfx942 the
    split-K path keeps its register-prefetch kernel, which never stages it).
    """
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.k1 import (
        _fit_pipe_tile,
        _pipe_lds_bytes,
    )

    arch = get_gfx()
    cap = get_lds_capacity_bytes(arch)
    if arch == "gfx950":
        full = _pipe_lds_bytes(2, 128, 384, 64, 8192, 4)
        bn, nw, w_lds = _fit_pipe_tile(
            arch, 384, 384, 128, 64, 2, 4, 2, 16, 8192, 4, True
        )
        used = _pipe_lds_bytes(2, 128, bn, 64, 8192 if w_lds else 1, 4)
        if full <= cap or used > cap or not w_lds:
            print(f"[FAIL] lds_fit: full={full} fit=({bn},{nw},{w_lds}) used={used}")
            _FAILURES.append("lds_fit")
        inp = _make_shape_inputs(24576, 4, 8192, seed=31)
        _run_combine_and_mix_shape(inp, 4, False, "wide_stream[M=24576,HS=8192]")
        del inp
        torch.cuda.empty_cache()
    # hc_count=2 keeps the [stream_dim] weight distinct from a full [hidden] one.
    inp = _make_shape_inputs(64, 2, 40960, seed=32)
    for split_k in (1, "auto"):
        _run_direct_k1(
            inp, 2, f"wide_stream_k1[HS=40960][sk={split_k}]", split_k=split_k
        )


def _run_combine_and_mix_shape(inp, hc_count, fold_w, label):
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], hc_count)
    if fold_w:
        merged = fold_norm_weight(merged, inp["norm_weight"], hc_count)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        hc_count,
        EPS,
        w_down_merged=merged,
        fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = gr_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        hc_count,
        EPS,
        round_bf16=True,
    )
    _check(r2_ref.to(r2.dtype), r2, f"{label} r2", atol=0.05, rtol=0.02)
    del r2_ref
    _check(x_ref.to(x.dtype), x, f"{label} x")
    _check(inj_ref.to(inj.dtype), inj.contiguous(), f"{label} inj", atol=0.05)


def _run_pipeline_depth():
    """Deep pipelines must cover their preload; short K is rejected, not read."""
    if get_gfx() != "gfx950":
        # gfx942 cannot fit an exact-preload pipeline: block_k=256 overflows
        # its 64 KiB LDS, and block_k=128 needs stages >= 5, which exceeds the
        # GEMM's 63-load vmcnt budget.
        print("[skip] pipe_depth: no exact-preload config fits gfx942")
        return
    # stream_dim=512, block_k=256 -> 4 K tiles: 5 stages preload exactly 4.
    inp = _make_shape_inputs(64, 2, 512, seed=33)
    kwargs = {"split_k": 1, "block_k": 256, "dn_block_m": 32, "use_tuned": False}
    _run_direct_k1(inp, 2, "pipe_depth[stages=5]", stages=5, **kwargs)
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], 2)
    for split_k, stages in ((1, 6), (2, 4)):
        _expect_value_error(
            f"pipe_depth[sk={split_k}][stages={stages}]",
            lambda split_k=split_k, stages=stages: flydsl_k1_combine_norm_down(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                merged,
                LOWRANK,
                2,
                EPS,
                **{**kwargs, "split_k": split_k, "stages": stages},
            ),
        )


def _run_stream_dim_validation():
    """A width the prologue cannot tile is a ValueError on every entry point."""
    hs = HS + 64  # multiple of hc_count but not of 512
    inp = _make_shape_inputs(4, HC, hs, seed=34)
    _expect_value_error(
        "stream_dim[cmix]",
        _cmix_call(inp, fold_w=False),
    )
    _expect_value_error(
        "stream_dim[mix]",
        lambda: flydsl_gr_two_stage_mix(
            inp["residual"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            HC,
            EPS,
        ),
    )
    _expect_value_error(
        "stream_dim[combine]",
        lambda: flydsl_gr_two_stage_combine(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            HC,
            EPS,
        ),
    )


def _run_arch_gating():
    """Unvalidated architectures are rejected before any dispatch."""
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import common

    inp = _make_inputs(1, seed=35)
    calls = {
        "skinny": _cmix_call(inp, fold_w=True),
        "combine": lambda: flydsl_gr_two_stage_combine(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            HC,
            EPS,
        ),
    }
    real = common.arch_name
    common.arch_name = lambda device=None: "gfx1100"
    try:
        for name, fn in calls.items():
            try:
                fn()
            except NotImplementedError:
                continue
            print(f"[FAIL] arch_gating[{name}]: expected NotImplementedError")
            _FAILURES.append(f"arch_gating[{name}]")
    finally:
        common.arch_name = real


def _run_stream_device_mismatch():
    """A launch stream on another device is rejected."""
    if torch.cuda.device_count() < 2:
        print("[skip] stream_device: needs two visible devices")
        return
    inp = _make_inputs(1, seed=36)
    other = torch.cuda.Stream(device=1)
    args = (
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        HC,
        EPS,
    )
    _expect_value_error(
        "stream_device[combine]",
        lambda: flydsl_gr_two_stage_combine(*args, stream=other),
    )


def _run_counter_cache_bound():
    """Distinct streams must not grow the skinny counter cache without bound.

    PyTorch pools 32 streams per device and priority, so the limits are lowered
    to make eviction and the graph-pinned bound reachable with distinct handles.
    """
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import skinny

    inp = _make_inputs(2, seed=37)
    merged = _merged(inp, True)
    _, x_ref, inj_ref = _ref(inp, "combine_and_mix")

    def run(stream=None):
        return flydsl_gr_two_stage_combine_and_mix(
            inp["residual"],
            inp["block_output"],
            inp["injection"],
            inp["norm_weight"],
            inp["w_down"],
            inp["w_up"],
            inp["w_inject"],
            HC,
            EPS,
            w_down_merged=merged,
            fold_w=True,
            stream=stream,
        )

    def check(out, label):
        _check(x_ref.to(out[1].dtype), out[1], f"{label} x")
        _check(inj_ref.to(out[2].dtype), out[2].contiguous(), f"{label} inj", atol=0.05)

    limits = (skinny._MAX_COUNTER_SETS, skinny._MAX_PINNED_SETS)
    try:
        skinny._MAX_COUNTER_SETS = 4
        streams, handles = [], set()
        while len(streams) < 12:
            stream = torch.cuda.Stream()
            if stream.cuda_stream not in handles:
                handles.add(stream.cuda_stream)
                streams.append(stream)
        for stream in streams:
            run(stream)
            stream.synchronize()
        unpinned = [k for k in skinny._COUNTERS if k not in skinny._PINNED_COUNTERS]
        kept = {k[-1] for k in unpinned}
        if len(unpinned) > 4 or streams[0].cuda_stream in kept:
            print(f"[FAIL] counter_cache: {len(unpinned)} unpinned sets, no eviction")
            _FAILURES.append("counter_cache evict")
        # The first stream's set was evicted; a fresh one must work.
        out = run(streams[0])
        streams[0].synchronize()
        check(out, "counter_cache[reused]")

        # At most one more graph-pinned set; later captures own their counters.
        skinny._MAX_PINNED_SETS = len(skinny._PINNED_COUNTERS) + 1
        graphs = []
        for stream in streams[:3]:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = run()
            graphs.append((graph, out))
        if len(skinny._PINNED_COUNTERS) > skinny._MAX_PINNED_SETS:
            print(f"[FAIL] counter_cache: {len(skinny._PINNED_COUNTERS)} pinned sets")
            _FAILURES.append("counter_cache pinned")
        for rep in range(3):
            for index, (graph, out) in enumerate(graphs):
                graph.replay()
                torch.cuda.synchronize()
                check(out, f"counter_cache[graph{index}][replay{rep}]")
        skinny.release_counters()
        if any(k not in skinny._PINNED_COUNTERS for k in skinny._COUNTERS):
            print("[FAIL] counter_cache: release kept unpinned sets")
            _FAILURES.append("counter_cache release")
        graphs[0][0].replay()
        torch.cuda.synchronize()
        check(graphs[0][1], "counter_cache[graph0][after release]")
    finally:
        skinny._MAX_COUNTER_SETS, skinny._MAX_PINNED_SETS = limits


def _run_skinny_n_pad_8():
    """Skinny decode keeps 8-row ``n_pad`` granularity; tiled K1 rejects it."""
    for tokens in (1, 8, 64):
        inp = _make_inputs(tokens, seed=36)
        merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
        folded = fold_norm_weight(merged, inp["norm_weight"], HC)
        narrow = folded[: LOWRANK + 8].contiguous()

        def call(w, inp=inp):
            return flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                inp["w_up"],
                inp["w_inject"],
                HC,
                EPS,
                w_down_merged=w,
                fold_w=True,
            )

        if tokens > 8:
            _expect_value_error(f"n_pad_8[M={tokens}]", lambda w=narrow: call(w))
            continue
        got, want = call(narrow), call(folded)
        if not all(torch.equal(g, w) for g, w in zip(got, want)):
            print(f"[FAIL] n_pad_8[M={tokens}]: differs from the 16-row-padded weight")
            _FAILURES.append(f"n_pad_8[M={tokens}]")


def _run_heuristic_marker():
    """A tuner ``heuristic`` entry must route to the heuristic, not a neighbour."""
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual import tuned

    real = tuned._table
    table = {
        "gfxtest": {
            "k1": {
                "256": {"method": "splitk", "split_k": 4},
                "512": {"method": "heuristic"},
            }
        }
    }
    tuned._table = lambda: table
    try:
        got = (tuned.k1_plan("gfxtest", 512), tuned.k1_plan("gfxtest", 256))
    finally:
        tuned._table = real
    if got[0] is not None or got[1] is None:
        print(f"[FAIL] heuristic_marker: {got}")
        _FAILURES.append("heuristic_marker")


def _k2_inputs(tokens, seed, hs=HS):
    """Valid K2 operands and the oracle ``block_input``."""
    torch.manual_seed(seed)
    hidden = HC * hs
    lora = (torch.randn(tokens, LOWRANK, device="cuda") * 0.5).bfloat16()
    r2 = torch.randn(tokens, hidden, device="cuda").bfloat16()
    rrms = torch.rsqrt(r2.float().view(tokens, HC, hs).pow(2).mean(-1) + EPS)
    norm_weight = (torch.randn(hs, device="cuda") * 0.1).bfloat16()
    w_up = (torch.randn(hidden, LOWRANK, device="cuda") * LOWRANK**-0.5).bfloat16()
    xn = r2.float().view(tokens, HC, hs) * rrms.unsqueeze(-1)
    xn = (xn * (1.0 + norm_weight.float())).bfloat16().float()
    gate = (lora.float() @ w_up.float().t()).view(tokens, HC, hs)
    ref = (torch.sigmoid(gate) * xn).mean(1)
    return (lora, r2, rrms.contiguous(), norm_weight, w_up), ref


def _run_direct_k2():
    """Public K2: fresh and caller-provided outputs, row-tail rejection/padding."""
    args, ref = _k2_inputs(128, seed=38)
    out = flydsl_up_gate_mix_norm(*args, HC)
    torch.cuda.synchronize()
    _check(ref.to(out.dtype), out, "direct_k2[M=128]")

    block_input = torch.full_like(out, float("nan"))
    got = flydsl_up_gate_mix_norm(*args, HC, block_input=block_input)
    torch.cuda.synchronize()
    if got.data_ptr() != block_input.data_ptr():
        print("[FAIL] direct_k2[out]: did not write the caller's buffer")
        _FAILURES.append("direct_k2[out] buffer")
    _check(ref.to(got.dtype), block_input, "direct_k2[out]")
    for name, bad in {
        "out_shape": block_input[:-1],
        "out_dtype": block_input.float(),
        "out_strided": torch.empty(128, 2 * HS, device="cuda").bfloat16()[:, ::2],
        "out_in_r2": args[1].view(-1)[: 128 * HS].view(128, HS),
        "out_in_w_up": args[4].view(-1)[-128 * HS :].view(128, HS),
    }.items():
        _expect_value_error(
            f"direct_k2[{name}]",
            lambda bad=bad: flydsl_up_gate_mix_norm(*args, HC, block_input=bad),
        )

    tail = 100
    (lora, r2, rrms, norm_weight, w_up), ref = _k2_inputs(tail, seed=39)
    _expect_value_error(
        "direct_k2[tail]",
        lambda: flydsl_up_gate_mix_norm(lora, r2, rrms, norm_weight, w_up, HC),
    )
    pad = 128 - tail

    def _pad(t):
        return torch.cat([t, t.new_zeros(pad, *t.shape[1:])])

    padded = flydsl_up_gate_mix_norm(
        _pad(lora), _pad(r2), _pad(rrms), norm_weight, w_up, HC
    )
    torch.cuda.synchronize()
    _check(ref.to(padded.dtype), padded[:tail], "direct_k2[padded tail]")

    # K2 has no prologue, so streams narrower than K1's 512 alignment are valid.
    args, ref = _k2_inputs(64, seed=40, hs=64)
    narrow = flydsl_up_gate_mix_norm(*args, HC)
    torch.cuda.synchronize()
    _check(ref.to(narrow.dtype), narrow, "direct_k2[stream_dim=64]")


def _run_direct_k1_buffers():
    """Caller K1 buffers are validated, also at M=0, and written in place."""
    tokens = 64
    inp = _make_inputs(tokens, seed=41)
    merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
    n_pad = merged.shape[0]

    def call(src=inp, **kwargs):
        return flydsl_k1_combine_norm_down(
            src["residual"],
            src["block_output"],
            src["injection"],
            src["norm_weight"],
            merged,
            LOWRANK,
            HC,
            EPS,
            **kwargs,
        )

    def bf16(*shape):
        return torch.empty(*shape, device="cuda").bfloat16()

    hidden = HC * HS
    bad = {
        "r2_cpu": {"r2_out": torch.empty(tokens, hidden, dtype=torch.bfloat16)},
        "r2_dtype": {"r2_out": torch.empty(tokens, hidden, device="cuda")},
        "r2_rows": {"r2_out": bf16(tokens - 1, hidden)},
        "out_dtype": {"out": torch.empty(tokens, n_pad, device="cuda")},
        "out_strided": {"out": bf16(tokens, 2 * n_pad)[:, ::2]},
        "rrms_dtype": {"rrms_out": torch.empty(tokens, HC, device="cuda").int()},
        "rrms_rows": {"rrms_out": torch.empty(tokens + 1, HC, device="cuda")},
        "gemm_pad": {"gemm_pad": tokens - 1},
        "stages": {"stages": 1},
    }
    # Outputs aliasing an input or each other: the prologue would overwrite
    # operands the GEMM still reads.
    r2_buf = bf16(tokens, hidden)
    out_buf = bf16(tokens, n_pad)
    bad.update(
        {
            "r2_is_weight": {"r2_out": merged[:tokens]},
            "r2_is_residual": {"r2_out": inp["residual"]},
            "out_in_block_output": {
                "out": inp["block_output"]
                .view(-1)[: tokens * n_pad]
                .view(tokens, n_pad)
            },
            "out_in_r2": {
                "r2_out": r2_buf,
                "out": r2_buf.view(-1)[-tokens * n_pad :].view(tokens, n_pad),
            },
            "rrms_in_out": {
                "out": out_buf,
                "rrms_out": out_buf.view(torch.float32)
                .view(-1)[: tokens * HC]
                .view(tokens, HC),
            },
        }
    )
    for name, kwargs in bad.items():
        _expect_value_error(f"k1_buffers[{name}]", lambda kwargs=kwargs: call(**kwargs))
    empty = _make_inputs(0, seed=42)
    for name, kwargs in {
        "r2_cpu": {"r2_out": torch.empty(0, hidden)},
        "out_int": {"out": torch.empty(0, n_pad, dtype=torch.int32)},
    }.items():
        _expect_value_error(
            f"k1_buffers[M=0][{name}]",
            lambda kwargs=kwargs: call(src=empty, **kwargs),
        )

    # Adjacent but disjoint outputs in one allocation are accepted.
    flat = bf16(tokens * (hidden + n_pad))
    r2_out = flat[: tokens * hidden].view(tokens, hidden)
    out = flat[tokens * hidden :].view(tokens, n_pad)
    rrms_out = torch.empty(tokens, HC, device="cuda")
    r2, packed = call(r2_out=r2_out, out=out, rrms_out=rrms_out)
    torch.cuda.synchronize()
    if r2.data_ptr() != r2_out.data_ptr() or packed.data_ptr() != out.data_ptr():
        print("[FAIL] k1_buffers: did not write the caller's buffers")
        _FAILURES.append("k1_buffers in place")
    _check_direct_k1(inp, merged, HC, r2, packed, "k1_buffers")
    r2_ref = gr_combine(
        inp["residual"], inp["block_output"], inp["injection"], HC, round_bf16=True
    )
    rr_ref = torch.rsqrt(r2_ref.view(tokens, HC, HS).pow(2).mean(-1) + EPS)
    _check(rr_ref, rrms_out, "k1_buffers rrms", atol=1e-3, rtol=1e-3)


def _run_final_mixer_no_inject(tokens, fold_w):
    """Final mixer: ``w_inject=None`` -> no inject columns, ``inj_next=None``."""
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens, with_inject=False, seed=3)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        None,
        HC,
        EPS,
        w_down_merged=_merged(inp, fold_w),
        fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, _ = _ref(inp, "combine_and_mix")
    if inj is not None:
        print(f"[FAIL] final_mixer[{tag}][M={tokens}]: must not emit inject logits")
        _FAILURES.append(f"final_mixer[{tag}][M={tokens}] inj")
    _check(
        r2_ref.to(r2.dtype),
        r2,
        f"final_mixer[{tag}][M={tokens}] r2",
        atol=0.05,
        rtol=0.02,
    )
    _check(x_ref.to(x.dtype), x, f"final_mixer[{tag}][M={tokens}] x")


def _run_decode(tokens, fold_w):
    """Check skinny folded decode and the padded unfolded path."""
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens, seed=7)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        w_down_merged=_merged(inp, fold_w),
        fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
    _check(
        r2_ref.to(r2.dtype), r2, f"decode[{tag}][M={tokens}] r2", atol=0.05, rtol=0.02
    )
    _check(x_ref.to(x.dtype), x, f"decode[{tag}][M={tokens}] x")
    _check(
        inj_ref.to(inj.dtype),
        inj.contiguous(),
        f"decode[{tag}][M={tokens}] inj",
        atol=0.05,
    )


def _run_full_norm_weight(tokens, fold_w):
    """Check per-stream norm weights in K1, K2, and skinny decode."""
    tag = "fold" if fold_w else "nofold"
    inp = _make_inputs(tokens, full_norm=True, seed=5)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        w_down_merged=_merged(inp, fold_w),
        fold_w=fold_w,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
    _check(
        r2_ref.to(r2.dtype), r2, f"fullw[{tag}][M={tokens}] r2", atol=0.05, rtol=0.02
    )
    _check(x_ref.to(x.dtype), x, f"fullw[{tag}][M={tokens}] x")
    _check(
        inj_ref.to(inj.dtype),
        inj.contiguous(),
        f"fullw[{tag}][M={tokens}] inj",
        atol=0.05,
    )


def _run_pad336_merged(tokens):
    """Check a caller-padded 336-column merged weight."""
    inp = _make_inputs(tokens, seed=11)
    merged = merge_down_inject(inp["w_down"], inp["w_inject"], 336)
    r2, x, inj = flydsl_gr_two_stage_combine_and_mix(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        HC,
        EPS,
        w_down_merged=fold_norm_weight(merged, inp["norm_weight"], HC),
        fold_w=True,
    )
    torch.cuda.synchronize()
    r2_ref, x_ref, inj_ref = _ref(inp, "combine_and_mix")
    _check(r2_ref.to(r2.dtype), r2, f"pad336[M={tokens}] r2", atol=0.05, rtol=0.02)
    _check(x_ref.to(x.dtype), x, f"pad336[M={tokens}] x")
    _check(
        inj_ref.to(inj.dtype),
        inj.contiguous(),
        f"pad336[M={tokens}] inj",
        atol=0.05,
    )


def _run_direct_k1_extent():
    """Guarded-allocation sentinels for row tails; M=224 is _run_direct_k1_untiled_tokens."""
    for tokens in (225, 255, 256):
        inp = _make_inputs(tokens, seed=17)
        merged = merge_gr_two_stage_weight(inp["w_down"], inp["w_inject"], HC)
        n_pad = merged.shape[0]
        # Contract: a token count that is not a multiple of the 32-row
        # K1 tile needs gemm_pad, and caller buffers span gemm_pad rows.
        rows = (tokens + 31) // 32 * 32
        gemm_pad = None if rows == tokens else rows
        sentinel = 123.0
        r2_storage = torch.full(
            (tokens + 64, HC * HS), sentinel, device="cuda", dtype=torch.bfloat16
        )
        out_storage = torch.full(
            (tokens + 64, n_pad), sentinel, device="cuda", dtype=torch.bfloat16
        )
        rr_storage = torch.full(
            (tokens + 64, HC), sentinel, device="cuda", dtype=torch.float32
        )
        # Keep an overrun inside a guarded allocation so it is observable without
        # corrupting unrelated allocator state. The logical partial shape is unchanged.
        partial_storage = torch.full(
            (4 * rows + 4 * 64, n_pad), sentinel, device="cuda", dtype=torch.float32
        )
        empty = torch.empty

        def guarded_empty(
            *shape,
            rows=rows,
            n_pad=n_pad,
            partial_storage=partial_storage,
            empty=empty,
            **kwargs,
        ):
            if shape == (4 * rows, n_pad) and kwargs.get("dtype") == torch.float32:
                return partial_storage[: 4 * rows]
            return empty(*shape, **kwargs)

        with patch("torch.empty", guarded_empty):
            r2, out = flydsl_k1_combine_norm_down(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                merged,
                LOWRANK,
                HC,
                EPS,
                r2_out=r2_storage[:rows],
                out=out_storage[:rows],
                rrms_out=rr_storage[:rows],
                split_k=4,
                sk_block_m=64,
                gemm_pad=gemm_pad,
            )
            r2_ref = _ref(inp, "combine")
            xn = (
                gr_grouped_rmsnorm(r2_ref, inp["norm_weight"], HC, EPS)
                .bfloat16()
                .float()
            )
            projection = xn @ merged.float().t()
            projection[:, :LOWRANK] = torch.nn.functional.silu(
                projection[:, :LOWRANK] / HC
            )
            _check(
                r2_ref.bfloat16(),
                r2[:tokens],
                f"direct K1[M={tokens}] r2",
                atol=0.05,
                rtol=0.02,
            )
            _check(projection.bfloat16(), out[:tokens], f"direct K1[M={tokens}] packed")
        torch.cuda.synchronize()
        assert torch.all(r2_storage[rows:] == sentinel)
        assert torch.all(out_storage[rows:] == sentinel)
        assert torch.all(rr_storage[rows:] == sentinel)
        assert torch.all(
            partial_storage[4 * rows :] == sentinel
        ), f"partial overrun at M={tokens}"
        if gemm_pad is not None:
            try:
                flydsl_k1_combine_norm_down(
                    inp["residual"],
                    inp["block_output"],
                    inp["injection"],
                    inp["norm_weight"],
                    merged,
                    LOWRANK,
                    HC,
                    EPS,
                    split_k=4,
                    sk_block_m=64,
                )
            except ValueError as error:
                assert "multiple of the 32-row K1 tile" in str(error), str(error)
            else:
                raise AssertionError(f"M={tokens} accepted without gemm_pad")
    print("direct K1 extent passed (unaligned rows need gemm_pad)")


def _make_seam_inputs(
    rows, *, combine=True, next_gate=True, seed=0, w_scale=1.0, device="cuda"
):
    """Seam-contract inputs: finished FP32 gate_prev in [0, 2]; weights at
    SGLang's test scales (norm 0.1, down 0.04, up 0.1, inject 0.04) times w_scale."""
    gen = torch.Generator(device=device).manual_seed(seed * 1000 + rows)
    hidden = HC * HS

    def _rnd(*shape, std):
        return (torch.randn(*shape, generator=gen, device=device) * std).bfloat16()

    gate_prev = 2.0 * torch.sigmoid(torch.randn(rows, HC, generator=gen, device=device))
    return {
        "y": _rnd(rows, HS, std=1.0) if combine else None,
        "residual": _rnd(rows, hidden, std=3.0),
        "gate_prev": gate_prev.contiguous() if combine else None,
        "norm_weight": _rnd(hidden, std=0.1 * w_scale),
        "w_down": _rnd(LOWRANK, hidden, std=0.04 * w_scale),
        "w_up": _rnd(hidden, LOWRANK, std=0.1 * w_scale),
        "w_inject": _rnd(HC, hidden, std=0.04 * w_scale) if next_gate else None,
    }


def _seam_args(inp):
    return (
        inp["y"],
        inp["residual"],
        inp["gate_prev"],
        inp["norm_weight"],
        inp["w_down"],
        inp["w_up"],
        inp["w_inject"],
        EPS,
    )


def _seam_rel_l2(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def _seam_errors(fn, inp, ref_fn=seam_ref_fp64):
    """Return (mixed rel-L2, gate max-abs or None) of fn against ref_fn."""
    _, mixed, gate = fn(*_seam_args(inp))
    _, mixed_ref, gate_ref = ref_fn(*_seam_args(inp))
    gerr = None
    if gate_ref is not None:
        gerr = (gate.double() - gate_ref.double()).abs().max().item()
    return _seam_rel_l2(mixed, mixed_ref), gerr


def _run_seam_case(rows, combine, next_gate, fn=None, *, seed=0, stress=None):
    fn = fn or SEAM_UNDER_TEST
    tag = (
        f"seam[M={rows},combine={int(combine)},gate={int(next_gate)},"
        f"seed={seed},stress={stress}]"
    )
    inp = _make_seam_inputs(rows, combine=combine, next_gate=next_gate, seed=seed)
    if stress == "heavy-norm":
        # Sparse affine outliers retain the near-zero typical norm weight.
        inp["norm_weight"][::257] *= 8
    elif stress == "saturated-gates":
        if combine:
            inp["gate_prev"] = 2 * torch.sigmoid(
                torch.linspace(-40, 40, rows * HC, device="cuda").view(rows, HC)
            )
        if next_gate:
            inp["w_inject"] *= torch.tensor([0.1, 1, 4, 10], device="cuda").view(HC, 1)
    elif stress == "near-zero-rms":
        inp["residual"] *= 1e-5
        if combine:
            inp["y"] *= 1e-5
    elif stress == "stream-scales":
        inp["residual"] = (
            (
                inp["residual"].view(rows, HC, HS)
                * torch.tensor([0.01, 0.1, 10, 100], device="cuda").view(1, HC, 1)
            )
            .bfloat16()
            .view(rows, HC * HS)
        )
    snap = {k: v.clone() for k, v in inp.items() if v is not None}
    r_new, mixed, gate = fn(*_seam_args(inp))
    ok = True

    def _fail(msg):
        nonlocal ok
        print(f"[FAIL] {tag}: {msg}")
        ok = False

    if r_new.dtype != torch.bfloat16 or r_new.shape != inp["residual"].shape:
        _fail(f"residual_new {r_new.dtype} {tuple(r_new.shape)}")
    if mixed.dtype != torch.bfloat16 or tuple(mixed.shape) != (rows, HS):
        _fail(f"mixed {mixed.dtype} {tuple(mixed.shape)}")
    if next_gate:
        if (
            gate is None
            or gate.dtype != torch.float32
            or tuple(gate.shape) != (rows, HC)
        ):
            _fail("gate_next must be FP32 [M,4]")
    elif gate is not None:
        _fail("gate_next must be None without w_inject")
    # No combine returns the input residual object; combine allocates new storage.
    if combine == (r_new is inp["residual"]):
        _fail(f"alias contract violated (combine={combine})")
    for k, v in snap.items():
        if not torch.equal(v, inp[k]):
            _fail(f"input {k} was mutated")
    if not (
        torch.isfinite(r_new.float()).all() and torch.isfinite(mixed.float()).all()
    ):
        _fail("non-finite output")
    if gate is not None and not (
        torch.isfinite(gate).all() and gate.min() >= 0 and gate.max() <= 2
    ):
        _fail("gate_next outside [0, 2]")
    # Residual: exactly the combine-only reference (or untouched when no combine).
    r_exp = (
        SEAM_COMBINE_UNDER_TEST(inp["y"], inp["residual"], inp["gate_prev"])
        if combine
        else inp["residual"]
    )
    if not torch.equal(r_new.view(torch.int16), r_exp.view(torch.int16)):
        _fail("residual_new differs from combine-only reference")
    if combine:
        split = fn(
            *_seam_args({**inp, "residual": r_exp, "y": None, "gate_prev": None})
        )
        if _seam_rel_l2(mixed, split[1]) >= SEAM_MIXED_REL_L2:
            _fail("fused differs from combine-only plus mix-only")
        if next_gate and (gate - split[2]).abs().max() >= SEAM_GATE_MAX_ABS:
            _fail("fused gate differs from combine-only plus mix-only")
    l2, gerr = _seam_errors(fn, inp)
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import (
        seam_deferred_default,
    )

    is_deferred = fn is _deferred_seam or (
        fn is hc_seam and seam_deferred_default(rows)
    )
    _SEAM_METRICS.append((is_deferred, l2, gerr or 0.0))
    if not (l2 < SEAM_MIXED_REL_L2):
        _fail(f"mixed rel-L2 {l2:.3e} >= {SEAM_MIXED_REL_L2:.0e}")
    gate_limit = (
        SEAM_GATE_MAX_ABS_DEFERRED if fn is _deferred_seam else SEAM_GATE_MAX_ABS
    )
    if gerr is not None and not (gerr < gate_limit):
        _fail(f"gate max-abs {gerr:.3e} >= {gate_limit:.1e}")
    if not ok and stress is not None:
        floor_l2, floor_gate = _seam_errors(seam_ref_operational, inp)
        print(f"[FLOOR] {tag} mixed_l2={floor_l2:.3e} gate_abs={floor_gate}")
    if is_deferred:
        deferred_ref = seam_ref_deferred(*_seam_args(inp))
        kernel_l2 = _seam_rel_l2(mixed, deferred_ref[1])
        if not (kernel_l2 < SEAM_DEFERRED_KERNEL_REL_L2):
            _fail(
                f"mixed rel-L2 vs deferred ref {kernel_l2:.3e} >= "
                f"{SEAM_DEFERRED_KERNEL_REL_L2:.0e}"
            )
    if fn is _deferred_seam:
        operational = seam_ref_operational(*_seam_args(inp))
        print(
            f"deferred-errors M={rows} seed={seed} stress={stress} "
            f"operational_l2={_seam_rel_l2(mixed, operational[1]):.9g} "
            f"deferred_l2={_seam_rel_l2(mixed, deferred_ref[1]):.9g} "
            f"operational_gate={0 if gate is None else (gate - operational[2]).abs().max().item():.9g}"
        )
    if ok:
        print(
            f"[PASS] {tag} mixed_l2={l2:.2e}"
            + (f" gate_abs={gerr:.2e}" if gerr is not None else "")
        )
    else:
        _FAILURES.append(tag)


def _run_seam_combine_only(rows, fn=None):
    fn = fn or SEAM_COMBINE_UNDER_TEST
    tag = f"seam-combine[M={rows}]"
    inp = _make_seam_inputs(rows, combine=True, next_gate=False)
    snap = {key: inp[key].clone() for key in ("y", "residual", "gate_prev")}
    out = fn(inp["y"], inp["residual"], inp["gate_prev"])
    assert all(torch.equal(value, inp[key]) for key, value in snap.items())
    ref = seam_combine_only_ref(inp["y"], inp["residual"], inp["gate_prev"])
    bad = out.dtype != torch.bfloat16 or out.shape != ref.shape
    mism = (out.view(torch.int16) != ref.view(torch.int16)).float().mean().item()
    if bad or mism != 0 or out is inp["residual"]:
        print(f"[FAIL] {tag}: dtype/shape/alias or mismatch frac {mism:.2e}")
        _FAILURES.append(tag)
    else:
        print(f"[PASS] {tag} mismatch_frac={mism:.2e}")


def _run_seam_paths_and_views():
    k1 = flydsl_k1_combine_norm_down
    for rows in (16, 64, 128, 256):
        with patch(
            "aiter.ops.flydsl.hc_seam.flydsl_k1_combine_norm_down",
            side_effect=lambda *args, **kw: k1(*args, **{**kw, "split_k": 1}),
        ):
            for combine in (False, True):
                for next_gate in (False, True):
                    _run_seam_case(rows, combine, next_gate)
    for rows in (1, 64):
        inp = _make_seam_inputs(rows, seed=19)
        base = hc_seam(*_seam_args(inp))
        storage = torch.empty(rows, 8, device="cuda", dtype=torch.float32)
        storage[:, ::2] = inp["gate_prev"]
        views = [storage[:, ::2]]
        for expanded in (
            inp["gate_prev"][:1].expand(rows, HC),
            inp["gate_prev"][:, :1].expand(rows, HC),
        ):
            assert torch.equal(
                hc_gated_combine(inp["y"], inp["residual"], expanded),
                seam_combine_only_ref(inp["y"], inp["residual"], expanded),
            )
        if rows == 1:
            views.append(torch.as_strided(inp["gate_prev"], (1, HC), (99, 1)))
        for view in views:
            result = hc_seam(*_seam_args({**inp, "gate_prev": view}))
            for got, expected in zip(result, base):
                assert torch.equal(got, expected)
            assert torch.equal(
                hc_gated_combine(inp["y"], inp["residual"], view),
                seam_combine_only_ref(inp["y"], inp["residual"], view),
            )
        for next_gate in (False, True):
            full_inp = {**inp, "w_inject": inp["w_inject"] if next_gate else None}
            fused = hc_seam(*_seam_args(full_inp))
            combined = hc_gated_combine(inp["y"], inp["residual"], inp["gate_prev"])
            separate = hc_seam(
                *_seam_args(
                    {**full_inp, "residual": combined, "y": None, "gate_prev": None}
                )
            )
            assert separate[0] is combined
            assert _seam_rel_l2(fused[1], separate[1]) < SEAM_MIXED_REL_L2
            if next_gate:
                assert (fused[2] - separate[2]).abs().max() < SEAM_GATE_MAX_ABS
    for rows in (16, 64):
        inp = _make_seam_inputs(rows, seed=23)
        result = hc_seam(*_seam_args(inp))
        # A returned finished gate must remain a valid next-combine input.
        chained = {**inp, "residual": result[0], "gate_prev": result[2]}
        r_new, _, _ = hc_seam(*_seam_args(chained))
        assert torch.equal(
            r_new,
            seam_combine_only_ref(
                chained["y"], chained["residual"], chained["gate_prev"]
            ),
        )
        f32_norm = hc_seam(
            *_seam_args({**inp, "norm_weight": inp["norm_weight"].float()})
        )
        for got, expected in zip(f32_norm, result):
            assert torch.equal(got, expected)
    inp = _make_seam_inputs(16, seed=31)
    invalid = (
        ({"y": None}, ValueError, "both be present"),
        ({"gate_prev": None}, ValueError, "both be present"),
        ({"gate_prev": inp["gate_prev"].bfloat16()}, ValueError, "gate_prev"),
        ({"gate_prev": inp["gate_prev"][:, :3]}, ValueError, "gate_prev"),
        ({"norm_weight": inp["norm_weight"][:HS]}, ValueError, "norm_w"),
        ({"w_down": inp["w_down"][:-1]}, ValueError, "w_down"),
        ({"w_inject": inp["w_inject"].cpu()}, ValueError, "same device"),
        ({"w_up": inp["w_up"].float()}, ValueError, "w_up"),
        ({"w_up": inp["w_up"].t().contiguous().t()}, ValueError, "contiguous"),
    )
    for changes, error_type, message in invalid:
        try:
            hc_seam(*_seam_args({**inp, **changes}))
        except error_type as error:
            assert message in str(error)
        else:
            raise AssertionError(f"seam accepted invalid input: {message}")
    print(
        "seam decoupled paths, FP32 norm, strided gates, chaining and validation passed"
    )


def _fail_prepared(msg):
    print(f"[FAIL] {msg}")


def _bits_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (
        a.shape == b.shape
        and torch.equal(
            a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
        )
        if a.dtype == torch.float32
        else torch.equal(
            a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        )
    )


def _prepare(inp):
    return hc_seam_prepare_weights(
        inp["norm_weight"], inp["w_down"], inp["w_up"], inp["w_inject"]
    )


def _run_k2_branch_parity():
    """Both register schedules retain residual/gate bits and prepared row panels."""
    maximum = 0.0
    for rows in (1, 16, 65, 224, 255, 256):
        for combine in (False, True):
            inp = _make_seam_inputs(rows, combine=combine, next_gate=True)
            prep = _prepare(inp)
            unpacked = (
                prep.w_up_interleaved.permute(1, 0, 2, 3).contiguous().view(10240, 320)
            )
            assert torch.equal(unpacked, prep.w_up)
            snapshot = prep.w_up_interleaved.clone()
            for deferred in (False, True):

                def call(mode, inp=inp, prep=prep, deferred=deferred):
                    return hc_seam_prepared(
                        inp["y"],
                        inp["residual"],
                        inp["gate_prev"],
                        prep,
                        EPS,
                        deferred=deferred,
                        k2_branch_mode=mode,
                    )

                old = call("branchwise")
                for mode in ("combined", "separate"):
                    new = call(mode)
                    assert _bits_equal(old[0], new[0]) and _bits_equal(old[2], new[2])
                    diff = (old[1].float() - new[1].float()).abs().max().item()
                    maximum = max(maximum, diff)
                    assert _seam_rel_l2(new[1], old[1]) < SEAM_MIXED_REL_L2
            assert torch.equal(snapshot, prep.w_up_interleaved)
    print(f"K2 branch parity: max_mixed_abs={maximum:.9g}; residual/gate bit-identical")


@pytest.mark.parametrize("rows", (64, 128, 65, 256))
@pytest.mark.parametrize("variant", ("batch", "structural"))
def test_deferred_structural_knobs(rows, variant):
    """Load batching and grid permutations preserve bits, including tail tiles."""
    if get_gfx_runtime() != "gfx950":
        pytest.skip("deferred HC seam requires gfx950")
    inp = _make_seam_inputs(rows, seed=7, combine=True, next_gate=True)
    prep = _prepare(inp)
    args = (inp["y"], inp["residual"], inp["gate_prev"], prep, EPS)
    geometry = {"deferred": True}
    old = hc_seam_prepared(*args, **geometry)
    knobs = {"k1_a_batch": 1}
    if variant != "batch":
        knobs.update(k1_grid_swap=1, k2_grid_swap=1)
    new = hc_seam_prepared(*args, **geometry, **knobs)
    ref = seam_ref_deferred(*_seam_args(inp))
    assert all(_bits_equal(a, b) for a, b in zip(old, new))
    assert _bits_equal(new[0], ref[0])
    mixed_error = _seam_rel_l2(new[1], ref[1])
    gate_error = (new[2] - ref[2].float()).abs().max().item()
    assert mixed_error < SEAM_MIXED_REL_L2
    assert gate_error < SEAM_GATE_MAX_ABS
    print(
        f"structural M={rows} variant={variant}: bit-identical; "
        f"mixed_rel_l2={mixed_error:.9g}; gate_max_abs={gate_error:.9g}"
    )


def test_structural_default_selector_range():
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import (
        seam_deferred_structural_default as sel,
    )

    on = {"k1_a_batch": 1, "k1_grid_swap": 1, "k2_grid_swap": 1}
    off = dict.fromkeys(on, 0)
    for rows in (1, 4, 63, 129, 256):
        assert sel(rows, True, True) == off, rows
    for rows in (64, 65, 96, 112, 128):
        assert sel(rows, True, True) == on, rows
        assert sel(rows, False, True) == off, rows
        assert sel(rows, True, False) == off, rows


@pytest.mark.parametrize("rows", (64, 65, 128, 256))
def test_structural_default_dispatch(rows):
    """Default dispatch turns the knobs on in 64..128 only; explicit kwargs win."""
    if get_gfx_runtime() != "gfx950":
        pytest.skip("deferred HC seam requires gfx950")
    import aiter.ops.flydsl.hc_seam as mod

    inp = _make_seam_inputs(rows, seed=7, combine=True, next_gate=True)
    prep = _prepare(inp)
    args = (inp["y"], inp["residual"], inp["gate_prev"], prep, EPS)

    def resolved(**kw):
        seen = {}
        k1, k2 = mod._build_down_deferred_partial_pipe, mod.flydsl_up_gate_mix_norm

        def k1_spy(**c):
            seen["k1"] = (c.get("k1_a_batch", 0), c.get("k1_grid_swap", 0))
            return k1(**c)

        def k2_spy(*a, **c):
            seen["k2"] = c.get("grid_swap", 0)
            return k2(*a, **c)

        with (
            patch.object(mod, "_build_down_deferred_partial_pipe", k1_spy),
            patch.object(mod, "flydsl_up_gate_mix_norm", k2_spy),
        ):
            out = mod.hc_seam_prepared(*args, **kw)
        return seen, out

    want = 1 if 64 <= rows <= 128 else 0
    seen, out = resolved()
    if rows <= 128:
        assert seen["k1"] == (want, want) and seen["k2"] == want, (rows, seen)
    else:
        assert "k1" not in seen, (rows, seen)
    if rows <= 128:
        seen, forced = resolved(deferred=True)
        assert seen["k1"] == (0, 0) and seen["k2"] == 0, (rows, seen)
        seen, _ = resolved(k1_a_batch=0, k1_grid_swap=0, k2_grid_swap=0)
        assert seen["k1"] == (0, 0) and seen["k2"] == 0, (rows, seen)
        assert all(_bits_equal(a, b) for a, b in zip(out, forced))
    ref = seam_ref_deferred(*_seam_args(inp)) if rows <= 128 else None
    if ref is not None:
        assert _bits_equal(out[0], ref[0])
        assert _seam_rel_l2(out[1], ref[1]) < SEAM_MIXED_REL_L2
        assert (out[2] - ref[2].float()).abs().max().item() < SEAM_GATE_MAX_ABS


def _run_k2_default_schedule():
    """Default K2 schedule per M: separate through the threshold, else branchwise."""
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import (
        SEAM_K2_SEPARATE_MAX_ROWS as T,
    )
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import (
        seam_k2_default,
    )

    m64 = {"block_m": 32, "block_n": 32, "m_waves": 2, "n_waves": 1, "slp": False}
    m16 = {"block_m": 16, "block_n": 32, "m_waves": 1, "n_waves": 1, "slp": True}
    expected = {1: m16, 16: m16, 32: m16, 64: m64, T: m64}
    for rows, geometry in expected.items():
        mode, got = seam_k2_default(rows)
        assert (mode, got) == ("separate", geometry), (rows, mode, got)
    for rows in (T + 1, 256):
        assert seam_k2_default(rows) == ("branchwise", {}), rows
    # End to end: above the deferred range the default call is the four-launch
    # path and must equal the forced schedule bit for bit. Smaller M dispatches
    # skinny or deferred, which do not use seam_k2_default.
    for rows in (129, 256):
        inp = _make_seam_inputs(rows, combine=True, next_gate=True)
        prep = _prepare(inp)
        args = (inp["y"], inp["residual"], inp["gate_prev"], prep, EPS)
        got = hc_seam_prepared(*args)
        ref = hc_seam_prepared(*args, k2_branch_mode="branchwise")
        for a, b in zip(got, ref):
            assert _bits_equal(a, b), rows
    # Within the deferred range the default call is the deferred path; the
    # explicit deferred= control must reproduce it bit for bit.
    for rows in (64, 96):
        inp = _make_seam_inputs(rows, combine=True, next_gate=True)
        prep = _prepare(inp)
        args = (inp["y"], inp["residual"], inp["gate_prev"], prep, EPS)
        got = hc_seam_prepared(*args)
        ref = hc_seam_prepared(*args, deferred=True)
        for a, b in zip(got, ref):
            assert _bits_equal(a, b), ("default vs deferred", rows)
    print(f"K2 default schedule: separate for M<={T}, branchwise above")


def _run_merged_width_rejected():
    """Hand-built 336-row merged weights are rejected before any launch."""
    inp = _make_seam_inputs(8)
    prep = dataclasses.replace(
        _prepare(inp),
        merged=merge_down_inject(inp["w_down"], inp["w_inject"], 336),
    )
    try:
        hc_seam_prepared(inp["y"], inp["residual"], inp["gate_prev"], prep, EPS)
    except ValueError as exc:
        assert "hc_seam_prepare_weights" in str(exc), exc
    else:
        raise AssertionError("336-row merged weight was accepted")
    print("merged-width check: 336-row merged weight rejected")


def _run_prepared_weights_validation():
    """HcSeamPreparedWeights.__post_init__ rejects inconsistent hand-built objects."""
    inp = _make_seam_inputs(8)
    prep = _prepare(inp)
    bad = {
        "has_next_gate": {"has_next_gate": not prep.has_next_gate},
        "w_up_interleaved shape": {"w_up_interleaved": prep.w_up_interleaved[:80]},
        "w_down shape": {"w_down": prep.w_down[:160]},
        "w_inject shape": {"w_inject": prep.w_inject[:2]},
        "merged ndim": {"merged": prep.merged.reshape(-1)},
        "merged columns": {"merged": prep.merged[:, :5120]},
        "cpu tensor": {"w_down": prep.w_down.cpu()},
    }
    for label, kw in bad.items():
        try:
            dataclasses.replace(prep, **kw)
        except ValueError:
            continue
        _fail_prepared(f"prepared weights accepted {label}")
        _FAILURES.append("prepared-weights accepted " + label)
    for label, kw in (
        ("valid", {}),
        (
            "336-row merged",
            {"merged": merge_down_inject(inp["w_down"], inp["w_inject"], 336)},
        ),
    ):
        try:
            dataclasses.replace(prep, **kw)
        except ValueError as exc:
            _fail_prepared(f"prepared weights rejected {label}: {exc}")
            _FAILURES.append("prepared-weights rejected " + label)
    print("prepared-weights validation passed")


def _run_deferred_padded_extent_rejected():
    """The deferred path needs a 64-multiple padded_extent; raised before launch."""
    inp = _make_seam_inputs(8)
    prep = _prepare(inp)
    for bad in (32, 96 + 16):
        try:
            hc_seam_prepared(
                inp["y"],
                inp["residual"],
                inp["gate_prev"],
                prep,
                EPS,
                deferred=True,
                padded_extent=bad,
            )
        except ValueError as exc:
            assert "multiple of 64" in str(exc), exc
        else:
            raise AssertionError(f"deferred padded_extent={bad} was accepted")
    print("deferred padded_extent check: non-multiples of 64 rejected")


def _run_seam_default_paths():
    for rows in (1, 3, 4, 8, 16, 64, 96, 128, 129, 256):
        inp = _make_seam_inputs(rows)
        prep = _prepare(inp)
        args = (inp["y"], inp["residual"], inp["gate_prev"], prep, EPS)
        got = hc_seam_prepared(*args)
        if 4 <= rows <= 128:
            ref = hc_seam_prepared(*args, deferred=True)
        elif rows <= 3:
            ref = hc_seam_prepared(*args, fallback="skinny")
        else:
            ref = hc_seam_prepared(*args, k2_branch_mode="branchwise")
        assert all(_bits_equal(a, b) for a, b in zip(got, ref)), rows
    print("seam default path boundaries and forced-path parity passed")


def _run_seam_prepared():
    """hc_seam_prepared is bit-identical to hc_seam, reusable, non-mutating."""
    if get_gfx_runtime() != "gfx950":
        return
    shared = _make_seam_inputs(256, combine=True, next_gate=True, seed=0)
    shared_prep = _prepare(shared)
    snap = (
        shared_prep.merged.clone(),
        shared_prep.norm_w.clone(),
        shared_prep.w_up.clone(),
    )
    for rows in SEAM_ROWS:
        for combine in (False, True):
            for next_gate in (False, True):
                tag = f"seam-prepared[M={rows},combine={int(combine)},gate={int(next_gate)}]"
                inp = _make_seam_inputs(
                    rows, combine=combine, next_gate=next_gate, seed=0
                )
                ref = hc_seam(*_seam_args(inp))
                out = hc_seam_prepared(
                    inp["y"], inp["residual"], inp["gate_prev"], _prepare(inp), EPS
                )
                if not all(_bits_equal(a, b) for a, b in zip(ref, out)):
                    _fail_prepared(f"{tag} differs from hc_seam")
                    _FAILURES.append(tag)
                # One shared prepared object across all M values and modes.
                shared_in = {
                    **inp,
                    **{k: shared[k] for k in ("norm_weight", "w_down", "w_up")},
                }
                shared_in["w_inject"] = shared["w_inject"] if next_gate else None
                prep = shared_prep if next_gate else _prepare(shared_in)
                ref = hc_seam(*_seam_args(shared_in))
                out = hc_seam_prepared(
                    inp["y"], inp["residual"], inp["gate_prev"], prep, EPS
                )
                if not all(_bits_equal(a, b) for a, b in zip(ref, out)):
                    _fail_prepared(f"{tag} shared prepared differs from hc_seam")
                    _FAILURES.append(tag + "-shared")
    after = (shared_prep.merged, shared_prep.norm_w, shared_prep.w_up)
    if not all(torch.equal(a, b) for a, b in zip(snap, after)):
        _fail_prepared("prepared weights were mutated by calls")
        _FAILURES.append("seam-prepared-mutation")
    inp = _make_seam_inputs(16, combine=True, next_gate=False)
    no_inj = _prepare(inp)
    for label, call, exc in (
        (
            "device mismatch",
            lambda: hc_seam_prepared(
                inp["y"].cpu(),
                inp["residual"].cpu(),
                inp["gate_prev"].cpu(),
                no_inj,
                EPS,
            ),
            ValueError,
        ),
    ):
        try:
            call()
        except exc:
            pass
        else:
            _fail_prepared(f"prepared accepted {label}")
            _FAILURES.append("seam-prepared-" + label)
    try:
        hc_seam_prepare_weights(
            inp["norm_weight"], inp["w_down"], inp["w_up"], inp["w_down"][:2]
        )
    except ValueError:
        pass
    else:
        _fail_prepared("prepare accepted a mis-shaped w_inject_next")
        _FAILURES.append("prepare accepted a mis-shaped w_inject_next")
    _, _, gate = hc_seam_prepared(
        inp["y"], inp["residual"], inp["gate_prev"], no_inj, EPS
    )
    if gate is not None or no_inj.has_next_gate:
        _fail_prepared("prepared without inject returned a next gate")
        _FAILURES.append("prepared without inject returned a next gate")
    print("seam prepared-weights equivalence, reuse, immutability, misuse passed")


def _run_seam_sentinel_extents(bm1, bm2):
    """Real-row sentinel guards with K1 row tile bm1 and K2 row tile bm2."""
    kw = {"k1_row_tile": bm1, "k2_row_tile": bm2}
    for fallback in ("splitk", "decouple"):
        for rows in (1, 17, 65, 224):
            for combine in (False, True):
                inp = _make_seam_inputs(rows, combine=combine, next_gate=True)
                prep = _prepare(inp)
                sentinel = 123.0
                residual_storage = torch.full(
                    (rows + 2, HC * HS),
                    sentinel,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                residual_storage[1 : rows + 1].copy_(inp["residual"])
                inp["residual"] = residual_storage[1 : rows + 1]
                guards = []
                empty = torch.empty

                def guarded_empty(
                    *shape,
                    rows=rows,
                    sentinel=sentinel,
                    guards=guards,
                    empty=empty,
                    **kwargs,
                ):
                    if shape in ((rows, HC * HS), (rows, HS)):
                        storage = torch.full(
                            (rows + 2, shape[1]),
                            sentinel,
                            device="cuda",
                            dtype=kwargs["dtype"],
                        )
                        guards.append(storage)
                        return storage[1 : rows + 1]
                    return empty(*shape, **kwargs)

                # The next-gate allocation uses Tensor.new_empty, so inject its
                # guarded real-row destination at the K1 boundary.
                gate_storage = torch.full(
                    (rows + 2, HC), sentinel, device="cuda", dtype=torch.float32
                )
                k1 = flydsl_k1_combine_norm_down

                def guarded_k1(
                    *args, k1=k1, gate_storage=gate_storage, rows=rows, **kw
                ):
                    return k1(*args, **{**kw, "gate_out": gate_storage[1 : rows + 1]})

                with (
                    patch("torch.empty", guarded_empty),
                    patch(
                        "aiter.ops.flydsl.hc_seam.flydsl_k1_combine_norm_down",
                        side_effect=guarded_k1,
                    ),
                ):
                    out = hc_seam_prepared(
                        inp["y"],
                        inp["residual"],
                        inp["gate_prev"],
                        prep,
                        EPS,
                        **kw,
                        fallback=fallback,
                    )
                torch.cuda.synchronize()
                for storage in [residual_storage, gate_storage, *guards]:
                    assert torch.all(storage[0] == sentinel) and torch.all(
                        storage[-1] == sentinel
                    )
                if not combine:
                    assert out[0] is inp["residual"]
                ref = seam_ref_fp64(*_seam_args(inp))
                assert _seam_rel_l2(out[1], ref[1]) < SEAM_MIXED_REL_L2
                assert (
                    gate_storage[1 : rows + 1] - ref[2]
                ).abs().max() < SEAM_GATE_MAX_ABS


def _run_seam_row_tiles():
    for tile in (16, 32, 64):

        def forced(y, residual, gate, norm, down, up, inject, eps, tile=tile):
            prep = hc_seam_prepare_weights(norm, down, up, inject)
            return hc_seam_prepared(y, residual, gate, prep, eps, row_tile=tile)

        for rows in SEAM_ROWS:
            for combine in (False, True):
                for next_gate in (False, True):
                    for seed in SEAM_SEEDS:
                        _run_seam_case(rows, combine, next_gate, fn=forced, seed=seed)
                        if rows in (16, 64, 128, 256):
                            for stress in SEAM_STRESS:
                                _run_seam_case(
                                    rows,
                                    combine,
                                    next_gate,
                                    fn=forced,
                                    seed=seed,
                                    stress=stress,
                                )
        _run_seam_sentinel_extents(tile, tile)
        print(
            f"row_tile={tile} forced matrix, split-K/decoupled sentinel extents passed"
        )
    for bm1, bm2 in ((16, 32), (32, 16), (32, 64), (64, 32), (16, 64), (64, 16)):
        _run_seam_sentinel_extents(bm1, bm2)
    print("mixed BM1 != BM2 sentinel extents passed")
    inp = _make_seam_inputs(16)
    try:
        hc_seam_prepared(
            inp["y"],
            inp["residual"],
            inp["gate_prev"],
            _prepare(inp),
            EPS,
            row_tile=16,
            k2_n_tile=16,
            k2_branch_mode="branchwise",
        )
    except ValueError as error:
        assert "vector-work divisibility" in str(error)
    else:
        raise AssertionError("BM16/BN16 must fail vector-work validation")


def _run_deferred_partials(
    rows_set=(16, 64, 65, 128, 256),
    seeds=(0, 1, 2, 3),
    stresses=(None, *SEAM_STRESS),
    **geometry,
):
    """Private deferred down: chunk/branch numerics and guarded scratch stores."""
    if get_gfx_runtime() != "gfx950":
        return
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    chunk = geometry.get("chunk_width", 256)
    bm = geometry.get("block_m", 64)
    splits = HC * HS // chunk
    max_stats, max_dot, max_oracle_stats = 0.0, 0.0, 0.0
    count = 0
    for rows in rows_set:
        padded = ((rows + bm - 1) // bm) * bm
        for combine in (False, True):
            builder = _build_down_deferred_partial_pipe(has_combine=combine, **geometry)
            compiled = None
            for seed in seeds:
                for stress in stresses:
                    inp = _make_seam_inputs(
                        rows, combine=combine, next_gate=False, seed=seed
                    )
                    if stress == "heavy-norm":
                        inp["norm_weight"][::257] *= 8
                    elif stress == "saturated-gates" and combine:
                        inp["gate_prev"] = 2 * torch.sigmoid(
                            torch.linspace(-40, 40, rows * HC, device="cuda").view(
                                rows, HC
                            )
                        )
                    elif stress == "near-zero-rms":
                        inp["residual"] *= 1e-5
                        if combine:
                            inp["y"] *= 1e-5
                    elif stress == "stream-scales":
                        inp["residual"] = (
                            (
                                inp["residual"].view(rows, HC, HS)
                                * torch.tensor(
                                    [0.01, 0.1, 10, 100], device="cuda"
                                ).view(1, HC, 1)
                            )
                            .bfloat16()
                            .view(rows, HC * HS)
                        )
                    if geometry.get("w_bf16") is False:
                        inp["norm_weight"] = inp["norm_weight"].float()
                    r_ref, d_ref, ss_ref = seam_deferred_partials_ref(
                        inp["y"],
                        inp["residual"],
                        inp["gate_prev"],
                        inp["norm_weight"],
                        inp["w_down"],
                        chunk_width=chunk,
                    )
                    oracle_ss = (
                        r_ref.float()
                        .view(rows, HC, HS // chunk, chunk)
                        .square()
                        .sum(-1)
                        .permute(1, 2, 0)
                    )
                    oracle_err = (
                        ((oracle_ss.double() - ss_ref).abs() / ss_ref.clamp_min(1e-300))
                        .max()
                        .item()
                    )
                    max_oracle_stats = max(max_oracle_stats, oracle_err)
                    assert oracle_err <= 1e-5
                    r_back = torch.full(
                        (rows + 3, HC * HS), 123, dtype=torch.bfloat16, device="cuda"
                    )
                    r_back[:rows].copy_(inp["residual"])
                    residual = r_back[:rows]
                    r_snap = r_back.clone()
                    out_back = torch.full_like(r_back, 117)
                    p_back = torch.full(
                        (splits * padded + 3, LOWRANK),
                        113,
                        dtype=torch.float32,
                        device="cuda",
                    )
                    p = p_back[: splits * padded]
                    ss_back = torch.full(
                        (splits * rows + 17,), 109, dtype=torch.float32, device="cuda"
                    )
                    ss = ss_back[: splits * rows]
                    args = (
                        residual,
                        inp["y"] if combine else residual,
                        inp["gate_prev"] if combine else residual,
                        inp["norm_weight"],
                        inp["w_down"],
                        out_back[:rows],
                        p,
                        ss,
                        fx.Stream(torch.cuda.current_stream().cuda_stream),
                    )
                    if compiled is None:
                        compiled = flyc.compile(builder, *args)
                    else:
                        compiled(*args)
                    torch.cuda.synchronize()
                    assert torch.equal(r_back, r_snap), "caller residual changed"
                    if combine:
                        assert torch.equal(
                            out_back[:rows], r_ref
                        ), "BF16 residual parity"
                    else:
                        assert torch.all(out_back == 117), "mix-only residual store"
                    assert torch.all(out_back[rows:] == 117), "residual tail store"
                    assert torch.all(
                        p_back[splits * padded :] == 113
                    ), "partial allocation tail"
                    assert torch.all(
                        p.view(splits, padded, LOWRANK)[:, rows:] == 113
                    ), "partial padded rows"
                    assert torch.all(ss_back[splits * rows :] == 109), "stat tail store"
                    got_ss = ss.view(HC, HS // chunk, rows).double()
                    stat_err = (
                        ((got_ss - ss_ref).abs() / ss_ref.clamp_min(1e-300))
                        .max()
                        .item()
                    )
                    branch_ref = ss_ref.sum(1)
                    branch_err = (
                        (
                            (got_ss.sum(1) - branch_ref).abs()
                            / branch_ref.clamp_min(1e-300)
                        )
                        .max()
                        .item()
                    )
                    stat_err = max(stat_err, branch_err)
                    got_d = p.view(HC, HS // chunk, padded, LOWRANK)[:, :, :rows]
                    dot_err = _seam_rel_l2(got_d, d_ref)
                    # Check branch-reduced intermediates too, not only chunk dots.
                    dot_err = max(dot_err, _seam_rel_l2(got_d.sum(1), d_ref.sum(1)))
                    assert stat_err <= 1e-5, (rows, combine, seed, stress, stat_err)
                    assert dot_err <= 1e-4, (rows, combine, seed, stress, dot_err)
                    max_stats, max_dot = max(max_stats, stat_err), max(max_dot, dot_err)
                    count += 1
    print(
        f"deferred gate: cases={count}, max_stat_rel={max_stats:.9g}, "
        f"max_dot_rel_l2={max_dot:.9g}, oracle_stat_rel={max_oracle_stats:.9g}; "
        "residual bit parity and all sentinels passed"
    )
    return max_stats, max_dot


def _run_deferred_pipeline(
    rows_set=(64, 65, 128), seeds=(0, 1, 2, 3), stresses=(None, *SEAM_STRESS)
):
    """K1 block_k/stages overrides, norm dtype and partial M tiles.

    BK changes the statistics reduction tree; the existing 1e-5 stats and
    1e-4 dot-relative-L2 gates apply, without requiring default bit identity.
    """
    from aiter.ops.flydsl.kernels.hyper_connection_gated_residual.tuned import (
        seam_deferred_k1_config,
    )

    for rows in rows_set:
        for bk, stages in ((64, 3), (128, 2)):
            cfg = seam_deferred_k1_config(rows)
            cfg.update(block_k=bk, stages=stages)
            _run_deferred_partials(
                rows_set=(rows,), seeds=seeds, stresses=stresses, **cfg
            )
        cfg = seam_deferred_k1_config(rows)
        _run_deferred_partials(
            rows_set=(rows,), seeds=seeds, stresses=stresses, w_bf16=False, **cfg
        )


def _run_deferred_legality():
    for chunk in (256, 512, 640, 1280, 2560):
        _build_down_deferred_partial_pipe(chunk_width=chunk)
    for bad in (
        {"chunk_width": 384},
        {"block_k": 96},
        {"block_k": 640, "chunk_width": 1280},
        {"block_m": 48},
        {"block_n": 96},
        {"stages": 0},
        {"m_waves": 0},
        {"slp": None},
        {"chunk_width": 256, "stages": 6},
    ):
        try:
            _build_down_deferred_partial_pipe(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"deferred accepted illegal geometry: {bad}")
    print(
        "deferred legality: five chunk values accepted; nine invalid configs rejected"
    )


def _run_deferred_reference():
    for rows in (16, 64, 65, 128, 256):
        for combine in (False, True):
            inp = _make_seam_inputs(rows, combine=combine, next_gate=False)
            snap = {k: v.clone() for k, v in inp.items() if v is not None}
            r, dots, ss = seam_deferred_partials_ref(
                inp["y"],
                inp["residual"],
                inp["gate_prev"],
                inp["norm_weight"],
                inp["w_down"],
            )
            assert dots.shape == (HC, HS // 256, rows, LOWRANK)
            assert dots.dtype == torch.float32 and ss.dtype == torch.float64
            assert ss.shape == (HC, HS // 256, rows)
            assert torch.isfinite(dots).all() and torch.isfinite(ss).all()
            if combine:
                assert torch.equal(
                    r,
                    seam_combine_only_ref(inp["y"], inp["residual"], inp["gate_prev"]),
                )
            else:
                assert r is inp["residual"]
            expected = r.double().view(rows, HC, HS).square().sum(-1).t()
            assert torch.allclose(ss.sum(1), expected, rtol=1e-14, atol=0)
            assert all(torch.equal(inp[k], v) for k, v in snap.items())
    print("deferred reference shapes, dtypes, combine parity and nonmutation passed")


def _deferred_seam(y, residual, gate, norm, down, up, inject, eps):
    prep = hc_seam_prepare_weights(norm, down, up, inject)
    return hc_seam_prepared(y, residual, gate, prep, eps, deferred=True)


def _run_deferred_seam():
    for rows in SEAM_ROWS:
        for combine in (False, True):
            for next_gate in (False, True):
                for seed in SEAM_SEEDS:
                    for stress in (None, *SEAM_STRESS):
                        _run_seam_case(
                            rows,
                            combine,
                            next_gate,
                            fn=_deferred_seam,
                            seed=seed,
                            stress=stress,
                        )
    # FP32 norm and noncanonical gate strides retain the exact-X gate contract.
    for rows in (1, 65, 224, 256):
        inp = _make_seam_inputs(rows)
        inp["norm_weight"] = inp["norm_weight"].float()
        inp["gate_prev"] = inp["gate_prev"].t().contiguous().t()
        prep = _prepare(inp)
        out = hc_seam_prepared(
            inp["y"],
            inp["residual"],
            inp["gate_prev"],
            prep,
            EPS,
            deferred=True,
            k2_row_tile=16,
            padded_extent=320,
        )
        ref = seam_ref_deferred(*_seam_args(inp))
        assert torch.equal(out[0], ref[0])
        assert _seam_rel_l2(out[1], ref[1]) < SEAM_MIXED_REL_L2
        assert (out[2] - ref[2]).abs().max() < SEAM_GATE_MAX_ABS
    inp = _make_seam_inputs(65)
    prep = _prepare(inp)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, results = [], []
    torch.cuda.synchronize()
    for stream in streams:
        with torch.cuda.stream(stream):
            hc_seam_prepared(
                inp["y"], inp["residual"], inp["gate_prev"], prep, EPS, deferred=True
            )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = hc_seam_prepared(
                inp["y"], inp["residual"], inp["gate_prev"], prep, EPS, deferred=True
            )
        graphs.append(graph)
        results.append(result)
    for _ in range(20):
        for graph, stream in zip(graphs, streams):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(*results))
    print("deferred seam FP32 norm, gate views, extent and concurrent graphs passed")


def _skinny_seam(y, residual, gate, norm, down, up, inject, eps):
    prepared = hc_seam_prepare_weights(norm, down, up, inject)
    return hc_seam_prepared(y, residual, gate, prepared, eps, fallback="skinny")


def _run_skinny_seam():
    for rows in range(1, 9):
        for combine in (False, True):
            for next_gate in (False, True):
                for seed in SEAM_SEEDS:
                    for stress in (None, *SEAM_STRESS):
                        _run_seam_case(
                            rows,
                            combine,
                            next_gate,
                            fn=_skinny_seam,
                            seed=seed,
                            stress=stress,
                        )
    for rows in (1, 4, 8):
        inp = _make_seam_inputs(rows)
        inp["norm_weight"] = inp["norm_weight"].float()
        inp["gate_prev"] = torch.stack((inp["gate_prev"], inp["gate_prev"]), dim=-1)[
            ..., 0
        ]
        got = _skinny_seam(*_seam_args(inp))
        ref = seam_ref_fp64(*_seam_args(inp))
        assert _seam_rel_l2(got[1], ref[1]) < SEAM_MIXED_REL_L2
        assert (got[2].double() - ref[2]).abs().max() < SEAM_GATE_MAX_ABS
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    graphs, outputs, expected, inputs = [], [], [], []
    for rows, stream in zip((1, 8), streams):
        inp = _make_seam_inputs(rows, seed=29)
        prepared = hc_seam_prepare_weights(
            inp["norm_weight"], inp["w_down"], inp["w_up"], inp["w_inject"]
        )

        inputs.append((inp, prepared))

        def launch(inp=inp, prepared=prepared):
            return hc_seam_prepared(
                inp["y"],
                inp["residual"],
                inp["gate_prev"],
                prepared,
                EPS,
                fallback="skinny",
            )

        expected.append(launch())
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(launch())
        graphs.append(graph)
    for _ in range(100):
        for graph, stream in zip(graphs, streams):
            with torch.cuda.stream(stream):
                graph.replay()
        for stream in streams:
            stream.synchronize()
        for out, ref in zip(outputs, expected):
            for actual, want in zip(out, ref):
                if not torch.equal(actual, want):
                    _FAILURES.append("skinny seam concurrent graph replay")
                    return
    print("skinny seam lifecycle passed")


def _run_seam_reference():
    if get_gfx_runtime() != "gfx950":
        print("seam requires gfx950; skipping finished-gate fixtures")
        return
    for rows in SEAM_ROWS:
        for combine in (False, True):
            for next_gate in (False, True):
                for seed in SEAM_SEEDS:
                    _run_seam_case(rows, combine, next_gate, seed=seed)
                    if rows in (16, 64, 128, 256):
                        for stress in SEAM_STRESS:
                            _run_seam_case(
                                rows, combine, next_gate, seed=seed, stress=stress
                            )
        _run_seam_combine_only(rows)
    _run_seam_paths_and_views()
    _run_seam_prepared()
    _run_seam_row_tiles()


def main():
    parser = argparse.ArgumentParser(
        description="Correctness test for the two-stage HC Gated-Residual op."
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[512, 2048, 4096],
        help="tile-aligned token counts covering split-K and decoupled K1 paths.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        nargs="+",
        default=[1, 3, 4, 5, 6, 7, 8, 32],
        help="small token counts covering skinny and padded decode paths.",
    )
    args = parser.parse_args()

    arch = get_gfx()
    if arch not in ("gfx950", "gfx942"):
        print(
            f"[skip] two-stage HC gated-residual requires gfx950/gfx942 FlyDSL, got {arch}"
        )
        return 0

    if arch == "gfx950":
        _run_deferred_legality()
        _run_deferred_reference()
        _run_deferred_partials()
        _run_deferred_pipeline()
        _run_deferred_seam()
        _run_k2_branch_parity()
        _run_k2_default_schedule()
        _run_seam_default_paths()
        _run_merged_width_rejected()
        _run_prepared_weights_validation()
        _run_deferred_padded_extent_rejected()
        # Pytest-only knob tests; main() is what CI runs.
        for rows in (64, 128, 65, 256):
            for variant in ("batch", "structural"):
                test_deferred_structural_knobs(rows, variant)
        test_structural_default_selector_range()
        for rows in (64, 65, 128, 256):
            test_structural_default_dispatch(rows)
    if arch == "gfx950":
        _run_skinny_seam()
    _run_seam_reference()
    _run_direct_k1_extent()
    _run_w_up_validation()
    for m in args.tokens:
        for fold_w in (False, True):
            _run_combine_and_mix(m, fold_w)
            _run_mix(m, fold_w)
            _run_final_mixer_no_inject(m, fold_w)
        _run_combine(m)
    if arch == "gfx942" and 8192 not in args.tokens:
        # gfx942's 64 KiB LDS makes high-M decoupled plans capacity-sensitive.
        for fold_w in (False, True):
            _run_combine_and_mix(8192, fold_w)
    if arch == "gfx950":
        # Validate every tuned range above the ordinary sweep against the
        # independent PyTorch oracle, not against the untuned kernel.
        for m in (16384, 24576, 32768, 49152, 65536):
            _run_combine_and_mix(m, False)
        _run_direct_k1_untiled_tokens()
    _run_combine(64)
    _run_strided_combine()
    _run_input_validation()
    _run_zero_tokens()
    _run_k2_override_limits()
    _run_side_stream_pressure()
    _run_skinny_publication_stress()
    _run_untuned_hidden()
    _run_counter_hc_isolation()
    _run_counter_cache_bound()
    _run_skinny_concurrent_graphs()
    _run_wide_stream_lds()
    _run_pipeline_depth()
    _run_stream_dim_validation()
    _run_arch_gating()
    _run_stream_device_mismatch()
    _run_heuristic_marker()
    _run_skinny_n_pad_8()
    _run_direct_k2()
    _run_direct_k1_buffers()
    for m in args.decode_tokens:
        # Unfolded decode uses the padded tail because skinny K1 requires folding.
        for fold_w in (False, True):
            _run_decode(m, fold_w)
        # The final mixer exercises the narrower no-injection output.
        _run_final_mixer_no_inject(m, True)
    for m in (3, 512):
        for fold_w in (False, True):
            _run_full_norm_weight(m, fold_w)
    for m in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ):
        _run_pad336_merged(m)

    for deferred in (False, True):
        metrics = [x for x in _SEAM_METRICS if x[0] == deferred]
        if metrics:
            print(
                f"seam maxima deferred={deferred}: mixed_rel_l2={max(x[1] for x in metrics):.9g} gate_max_abs={max(x[2] for x in metrics):.9g}"
            )
    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

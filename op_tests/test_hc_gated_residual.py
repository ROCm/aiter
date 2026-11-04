# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for the two-stage Hyper-Connection Gated-Residual op
(``aiter.ops.flydsl.kernels.hyper_connection_gated_residual``).

Checks all public entry points, the no-injection final mixer, decode and tiled
token counts, both norm-weight layouts, and folded/unfolded weights against the
float32 oracle.

    python op_tests/test_hc_gated_residual.py
    python op_tests/test_hc_gated_residual.py --tokens 512 4096
"""

import argparse
import sys

import torch

from aiter.jit.utils.chip_info import get_gfx
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
from aiter.test_common import checkAllclose

# Model dimensions covered by this test.
HC, HS, LOWRANK = 4, 2560, 320
EPS = 1e-6

_FAILURES = []


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
    _run_w_up_validation()
    _run_input_validation()
    _run_zero_tokens()
    _run_k2_override_limits()
    _run_side_stream_pressure()
    _run_skinny_publication_stress()
    _run_untuned_hidden()
    _run_counter_hc_isolation()
    _run_counter_cache_bound()
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

    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

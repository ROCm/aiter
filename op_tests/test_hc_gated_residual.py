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


def _check(ref, out, label, atol=0.06, rtol=0.05, max_err_ratio=0.03):
    if out.isnan().any() or out.isinf().any():
        print(f"[FAIL] {label}: output has NaN/Inf")
        _FAILURES.append(label)
        return
    err = checkAllclose(ref, out, msg=label, atol=atol, rtol=rtol)
    if not (err == 0 or err <= max_err_ratio):
        print(f"[FAIL] {label}: mismatch ratio {err} > {max_err_ratio}")
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
        try:
            flydsl_gr_two_stage_combine_and_mix(
                inp["residual"],
                inp["block_output"],
                inp["injection"],
                inp["norm_weight"],
                inp["w_down"],
                w_up,
                inp["w_inject"],
                HC,
                EPS,
            )
        except ValueError:
            continue
        print(f"[FAIL] w_up_validation[{label}]: expected ValueError")
        _FAILURES.append(f"w_up_validation[{label}]")


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


def _run_direct_k1_untiled_tokens():
    """A nearest tuned split-K tile must not overrun a direct M=224 call."""
    tokens = 224
    inp = _make_inputs(tokens, seed=18)
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
    )
    torch.cuda.synchronize()
    r2_ref = gr_combine(
        inp["residual"],
        inp["block_output"],
        inp["injection"],
        HC,
        round_bf16=True,
    )
    xn = gr_grouped_rmsnorm(r2_ref, inp["norm_weight"], HC, EPS).bfloat16().float()
    raw = xn @ merged.t().float()
    lora = raw[:, :LOWRANK] / HC
    lora = lora * torch.sigmoid(lora)
    _check(r2_ref.to(r2.dtype), r2, "direct_k1[M=224] r2", atol=0.05, rtol=0.02)
    _check(lora.to(packed.dtype), packed[:, :LOWRANK], "direct_k1[M=224] lora")
    _check(
        raw[:, LOWRANK : LOWRANK + HC].to(packed.dtype),
        packed[:, LOWRANK : LOWRANK + HC],
        "direct_k1[M=224] inj",
        atol=0.05,
    )


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
    _run_counter_hc_isolation()
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

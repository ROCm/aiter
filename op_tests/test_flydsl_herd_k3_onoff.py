# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 (TP8 shard) HERD on/off on one gfx1250 GPU: routing vs MoE time.

off: biased_grouped_topk(K) + fused_moe(K routes), as ATOM routes Kimi-K3.
on:  biased_grouped_topk(K+1, raw scores) + fused_moe(herd_topk=K), as ATOM's
     _flydsl_herd_step routes it; the grouped MoE drops one candidate per row
     inside moe_herd_route_psum (up to HERD_ROUTE_BLOCK rows).

Kernel device times come from torch.profiler, split into topk, route, GEMM
(gemm1 + gemm2) and other (activation quant, gather-reduce).

    python op_tests/test_flydsl_herd_k3_onoff.py               # benchmark table
    python op_tests/test_flydsl_herd_k3_onoff.py --kernels     # per-kernel times
    pytest op_tests/test_flydsl_herd_k3_onoff.py               # kernel set + outputs
"""

import argparse
import collections
import functools

import pytest
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from aiter.jit.utils.chip_info import get_gfx_runtime

EXPERTS, TOPK, MODEL_DIM, INTER_DIM = 896, 16, 3584, 384
ROUTED_SCALE = 2.5
TOKENS = (16, 32, 64, 128, 256, 512, 1024)
# Equivalent concurrency of the reference table: 5 tokens per decode request.
TOKENS_PER_REQUEST = 5
# Kernel name keys per stage. "route" builds the grouped MoE's per-expert row
# layout from the top-k: off = counter zero-fill + moe_route + psum_remap, on =
# moe_herd_route_psum.
STAGES = {
    "topk": ("sigmoid_biased_topk",),
    "route": ("FillFunctor", "moe_route", "psum_remap", "herd"),
    "gemm": ("mxfp4_tdm",),
}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="fused_moe(herd_topk) runs on the gfx1250 grouped MoE only",
)


@functools.cache
def _moe_weights():
    """Kimi-K3 TP8 MXFP4 GUGU expert weights and the fused_moe arguments."""
    from aiter import ActivationType, QuantType, benchmark_data_init
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
    from aiter.utility import dtypes

    generator = torch.Generator(device="cuda").manual_seed(2713)

    def weight(rows, cols, gate_up):
        plain = (
            benchmark_data_init.fill_fp4((EXPERTS * rows, cols), "uniform", generator)
            .view(EXPERTS, rows, cols // 2)
            .view(dtypes.fp4x2)
        )
        return moe_shuffle_weight(
            plain, experts_cnt=EXPERTS, is_guinterleave=True, gate_up=gate_up
        )

    def scale(rows, cols, gate_up):
        plain = torch.full(
            (EXPERTS, rows, cols // 32), 127, device="cuda", dtype=torch.uint8
        ).view(dtypes.fp8_e8m0)
        return moe_shuffle_scale(
            plain, experts_cnt=EXPERTS, is_guinterleave=True, gate_up=gate_up
        )

    w1 = weight(2 * INTER_DIM, MODEL_DIM, True)
    w2 = weight(MODEL_DIM, INTER_DIM, False)
    kwargs = {
        "w1_scale": scale(2 * INTER_DIM, MODEL_DIM, True),
        "w2_scale": scale(MODEL_DIM, INTER_DIM, False),
        "activation": ActivationType.Situv2,
        "quant_type": QuantType.per_1x32,
        "gate_mode": GateMode.INTERLEAVE.value,
    }
    return w1, w2, kwargs


def _inputs(tokens, skew):
    """Hidden states and router logits whose expert popularity has a fixed
    per-expert N(0, skew^2) bias on top of the per-token N(0, 1) logits."""
    generator = torch.Generator(device="cuda").manual_seed(1250 + tokens)
    hidden = (
        torch.randn(
            tokens, MODEL_DIM, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.1
    )
    logits = torch.randn(tokens, EXPERTS, device="cuda", generator=generator)
    if skew:
        popularity = torch.Generator(device="cuda").manual_seed(7)
        logits += skew * torch.randn(EXPERTS, device="cuda", generator=popularity)
    correction_bias = torch.zeros(EXPERTS, device="cuda", dtype=torch.bfloat16)
    num_valid = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    return hidden, logits.bfloat16(), correction_bias, num_valid


def _route(logits, correction_bias, num_valid, herd):
    from aiter import biased_grouped_topk

    rows = logits.shape[0]
    width = TOPK + 1 if herd else TOPK
    weights = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    ids = torch.empty(rows, width, dtype=torch.int32, device="cuda")
    # HERD candidates carry raw scores: fused_moe renormalizes and scales.
    biased_grouped_topk(
        logits,
        correction_bias,
        weights,
        ids,
        1,
        1,
        not herd,
        1.0 if herd else ROUTED_SCALE,
        num_valid,
    )
    return weights, ids


def _step(tokens, skew, herd):
    """One MoE layer's routing + experts, as ATOM's Mxfp4MoEMethod runs it."""
    from aiter.fused_moe import fused_moe

    w1, w2, kwargs = _moe_weights()
    hidden, logits, correction_bias, num_valid = _inputs(tokens, skew)
    herd_kwargs = (
        {"herd_topk": TOPK, "herd_routed_scaling_factor": ROUTED_SCALE} if herd else {}
    )

    def run():
        weights, ids = _route(logits, correction_bias, num_valid, herd)
        return fused_moe(
            hidden,
            w1,
            w2,
            weights,
            ids,
            **kwargs,
            num_local_tokens=num_valid,
            **herd_kwargs,
        )

    return run


def _active_experts(tokens, skew, herd):
    from aiter.ops.flydsl.herd_topk import herd_finalize_candidates

    _, logits, correction_bias, num_valid = _inputs(tokens, skew)
    weights, ids = _route(logits, correction_bias, num_valid, herd)
    if herd:
        _, ids = herd_finalize_candidates(weights, ids, EXPERTS, ROUTED_SCALE)
    return ids.unique().numel()


def _kernel_times(run, warmup, iters):
    """Mean device time per call of each kernel run() launches, in us."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    times = collections.defaultdict(float)
    for event in prof.events():
        if event.device_type == DeviceType.CUDA:
            times[event.name] += event.time_range.elapsed_us() / iters
    return dict(times)


def _stage(kernel):
    for stage, keys in STAGES.items():
        if any(key in kernel for key in keys):
            return stage
    return "other"  # activation quant, gather-reduce


def _split(times):
    stages = dict.fromkeys((*STAGES, "other"), 0.0)
    for kernel, t in times.items():
        stages[_stage(kernel)] += t
    return stages


def _logits_diff(actual, expected):
    x, y = actual.float(), expected.float()
    return float((x - y).square().sum() / (x.square().sum() + y.square().sum()))


def _kernel_names(run):
    times = _kernel_times(run, warmup=2, iters=1)
    return " ".join(times)


@pytest.mark.parametrize("tokens", TOKENS)
def test_herd_on_fuses_route_and_matches_router_drop(tokens):
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.herd_topk import herd_finalize_candidates

    off, on = _step(tokens, 1.0, False), _step(tokens, 1.0, True)
    off_kernels, on_kernels = _kernel_names(off), _kernel_names(on)
    assert "moe_herd_route_psum" in on_kernels
    for gone in ("herd_finalize", "moe_route", "psum_remap"):
        assert gone not in on_kernels, f"{gone} still launched with HERD on"
    assert "moe_herd_route_psum" not in off_kernels

    # HERD on equals the K-route MoE on the routes the HERD drop keeps.
    w1, w2, kwargs = _moe_weights()
    hidden, logits, correction_bias, num_valid = _inputs(tokens, 1.0)
    values, ids = _route(logits, correction_bias, num_valid, herd=True)
    kept_weights, kept_ids = herd_finalize_candidates(
        values, ids, EXPERTS, ROUTED_SCALE
    )
    expected = fused_moe(hidden, w1, w2, kept_weights, kept_ids, **kwargs)
    actual = on()
    assert torch.isfinite(actual).all()
    assert _logits_diff(actual, expected) < 1e-4


def _bench(args):
    rows = []
    for tokens in args.tokens:
        row = {"tokens": tokens}
        for dist, skew in (("uniform", 0.0), ("skew", args.skew)):
            for herd in (False, True):
                times = _kernel_times(
                    _step(tokens, skew, herd), args.warmup, args.iters
                )
                tag = f"{dist}_{'on' if herd else 'off'}"
                row[tag] = _split(times)
                row[f"{tag}_experts"] = _active_experts(tokens, skew, herd)
                if args.kernels:
                    print(f"\n[tokens={tokens} {tag}]")
                    for name, t in sorted(times.items(), key=lambda kv: -kv[1]):
                        print(f"  {_stage(name):>5} {t:8.2f} us  {name}")
        rows.append(row)
    _print_summary(rows, args.skew)
    _print_detail(rows, args.skew)
    return rows


def _change(before, after):
    delta = after - before
    return f"{delta:+.1f} ({delta / before * 100:+.1f}%)"


def _print_summary(rows, skew):
    """The reference table's columns: route and GEMM (gemm1 + gemm2) in us."""
    print(
        f"\n{'tokens':>6} {'conc':>6} | {'off route':>9} {'on route':>9} |"
        f" {'GEMM change (uniform)':>21} | {f'GEMM change (skew {skew})':>21} |"
        f" {'experts (uniform)':>17} | {f'experts (skew {skew})':>18}"
    )
    for row in rows:
        tokens = row["tokens"]
        uoff, uon, soff, son = (
            row[k] for k in ("uniform_off", "uniform_on", "skew_off", "skew_on")
        )
        print(
            f"{tokens:>6} {tokens / TOKENS_PER_REQUEST:>6.1f} |"
            f" {uoff['route']:>9.1f} {uon['route']:>9.1f} |"
            f" {_change(uoff['gemm'], uon['gemm']):>21} |"
            f" {_change(soff['gemm'], son['gemm']):>21} |"
            f" {row['uniform_off_experts']:>7} -> {row['uniform_on_experts']:<6} |"
            f" {row['skew_off_experts']:>8} -> {row['skew_on_experts']:<6}"
        )


def _print_detail(rows, skew):
    """Absolute stage times (us) per distribution, off / on."""
    for dist, label in (("uniform", "uniform"), ("skew", f"skew {skew}")):
        print(
            f"\n[{label}] {'tokens':>6} | {'topk off/on':>13} | {'route off/on':>13} |"
            f" {'GEMM off/on':>15} | {'other off/on':>13} | {'total off/on':>15}"
            f" {'total change':>17}"
        )
        for row in rows:
            off, on = row[f"{dist}_off"], row[f"{dist}_on"]
            total_off, total_on = sum(off.values()), sum(on.values())
            pad = " " * (len(label) + 3)
            print(
                f"{pad}{row['tokens']:>6} |"
                f" {off['topk']:>6.1f}/{on['topk']:<6.1f} |"
                f" {off['route']:>6.1f}/{on['route']:<6.1f} |"
                f" {off['gemm']:>7.1f}/{on['gemm']:<7.1f} |"
                f" {off['other']:>6.1f}/{on['other']:<6.1f} |"
                f" {total_off:>7.1f}/{total_on:<7.1f}"
                f" {_change(total_off, total_on):>17}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--skew", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--kernels", action="store_true", help="print each config's kernel times"
    )
    _bench(parser.parse_args())

# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paired A/B harness for the DSv4 sparse-MLA prefill kernel.

Best-of-N wall timing on this kernel had a ~0.6-1% noise floor, the same size as
the structural wins we were trying to land, so "flat" and "no effect" could not be
told apart. The dominant cause turned out not to be the statistics: this box has
eight GPUs shared with other jobs (several holding 100+ GB resident), and a
neighbour bursting mid-run inflates our samples. Clocks are on ``auto`` and DPM
level moves around underneath us as the power budget shifts.

That noise is *one-sided* -- contention and downclocking only ever make us slower,
never faster -- which drives four choices:

1. Many short samples rather than few long ones, so a burst spoils a few samples
   instead of the whole measurement. (Raising iterations per sample made things
   dramatically worse: 60 iters read 2.08 ms against 1.67 ms at 10.)
2. Tight interleaving with the order reversed on alternate rounds, so a burst and
   any linear drift land on every variant roughly equally.
3. Contamination filtering. A round counts only if every variant's sample sits
   within ``--clean-pct`` of that variant's own fastest sample; slow rounds are
   dropped rather than averaged in, and the kept count is reported.
4. Paired differences against a baseline measured moments earlier, with a
   bootstrap CI on the median, so residual common-mode drift cancels.

``--aa`` enters the baseline a second time under another label. Its true effect is
zero, so whatever interval it reports is the harness's own noise floor -- if an A/A
run claims significance, distrust the A/B beside it.

Pairing needs both versions live in one binary, so an optimization must sit behind
a flag to be measurable here; comparing two source revisions in separate processes
reintroduces exactly the drift this harness removes.

Run:
    python op_tests/flydsl_tests/ab_measure.py --aa
    python op_tests/flydsl_tests/ab_measure.py --aa --variant kvdb:kv_double_buffer=1
    python op_tests/flydsl_tests/ab_measure.py --aa --rounds 60 --gpu 4
"""

import argparse
import os
import random
import re
import statistics
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_AITER_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)


def _smi(*flags):
    try:
        return subprocess.run(
            ["rocm-smi", *flags], capture_output=True, text=True, timeout=20
        ).stdout
    except Exception:
        return ""


def _gpu_state(idx):
    """(sclk MHz, power W, busy %) for one GPU, any field None if unavailable."""
    out = _smi("--showgpuclocks", "--showpower", "--showuse")
    sclk = power = busy = None
    for line in out.splitlines():
        m = re.match(rf"GPU\[{idx}\]\s*:\s*(.*)", line.strip())
        if not m:
            continue
        rest = m.group(1)
        if "sclk" in rest:
            g = re.search(r"\((\d+)\s*Mhz\)", rest, re.I)
            if g:
                sclk = int(g.group(1))
        elif "Power" in rest:
            g = re.search(r":\s*([\d.]+)", rest)
            if g:
                power = float(g.group(1))
        elif "use" in rest:
            g = re.search(r":\s*(\d+)", rest)
            if g:
                busy = int(g.group(1))
    return sclk, power, busy


def _neighbours_busy(self_idx):
    """Total busy% across the other GPUs -- a proxy for contention risk."""
    out = _smi("--showuse")
    tot = 0
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*GPU use \(%\):\s*(\d+)", line.strip())
        if m and int(m.group(1)) != self_idx:
            tot += int(m.group(2))
    return tot


def _event_ms(fn, warmup, iters):
    """ms per call, timed with GPU events over a back-to-back batch.

    Launches are async and the kernel is ~1.7 ms, so the host stays well ahead and
    the GPU never idles between iterations; elapsed/iters is therefore kernel time
    without the profiler's per-event bookkeeping, which was itself distorting long
    samples.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _parse_variant(spec):
    """``label:key=val,key=val`` -> (label, kwargs); empty kwargs means default."""
    label, _, rest = spec.partition(":")
    kw = {}
    for item in filter(None, (s.strip() for s in rest.split(","))):
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if v.lower() in ("1", "true", "yes", "on"):
            kw[k] = True
        elif v.lower() in ("0", "false", "no", "off"):
            kw[k] = False
        elif v.lstrip("-").isdigit():
            kw[k] = int(v)
        else:
            kw[k] = v
    return label.strip(), kw


def _bootstrap_ci(vals, iters=4000, conf=0.95, seed=0):
    if len(vals) < 3:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    boots = sorted(
        statistics.median([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(iters)
    )
    return boots[int((1 - conf) / 2 * iters)], boots[min(iters - 1, int((1 + conf) / 2 * iters))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", default=[],
                    help="label:key=val,... (repeatable); the baseline is implicit")
    ap.add_argument("--aa", action="store_true", help="add a null control: baseline vs itself")
    ap.add_argument("--gpu", type=int, default=None,
                    help="pin to this GPU (pick an idle one on a shared box)")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--iters", type=int, default=10, help="calls per sample; keep small")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--clean-pct", type=float, default=1.5,
                    help="keep a round only if every sample is within this %% of its own best")
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--topk-main", type=int, default=512)
    ap.add_argument("--topk-extra", type=int, default=128)
    ap.add_argument("--main-tokens", type=int, default=65536)
    ap.add_argument("--extra-tokens", type=int, default=32768)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--locality", choices=("none", "realistic"), default="none",
                    help="'none' draws rows i.i.d. uniform and has no cross-query "
                         "reuse, so cache/XCD experiments must use 'realistic'")
    ap.add_argument("--corr-len", type=int, default=128,
                    help="queries over which region1 top-k decorrelates (realistic only)")
    ap.add_argument("--fast-cache", action="store_true",
                    help="pre-convert both caches (region1 nope OCP->fnuz, both rope "
                         "tails bf16->fp8) so every variant runs the DMA load paths")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from bench_dsv4_ab import _burn  # noqa: E402  (after device pinning)
    from bench_sparse_mla_prefill import (  # noqa: E402
        _ensure_flydsl,
        _is_gfx942_fnuz,
        build_b2_inputs,
    )

    def _runner(entry, inp, extra):
        """Local rather than bench_dsv4_ab's: that one pins extra_is_fnuz, which a
        variant needs to own so the converted-cache paths can be selected."""
        kw = dict(main_is_fnuz=_is_gfx942_fnuz(), extra_is_fnuz=False)
        kw.update(extra or {})

        def run_once():
            entry(
                inp.q, inp.out, inp.main_cache, inp.main_indices, inp.main_indptr,
                inp.main_bt, inp.extra_cache, inp.extra_indices, inp.extra_indptr,
                inp.extra_bt, block_size=inp.block_size, attn_sink=inp.sink, **kw,
            )

        return run_once

    _ensure_flydsl()
    from aiter.ops.flydsl import flydsl_sparse_mla_prefill_dsv4 as dsv4

    phys = args.gpu if args.gpu is not None else 0
    variants = [("baseline", {})]
    if args.aa:
        variants.append(("A/A control", {}))
    variants += [_parse_variant(s) for s in args.variant]

    inp = build_b2_inputs(
        T=args.T, topk_main=args.topk_main, topk_extra=args.topk_extra,
        main_tokens=args.main_tokens, extra_tokens=args.extra_tokens,
        block_size=args.block_size, seed=args.seed,
        locality=args.locality, corr_len=args.corr_len,
    )
    if args.fast_cache:
        from convert_extra_cache import convert_extra_cache_, quantize_rope_
        inp.main_cache.copy_(quantize_rope_(inp.main_cache.clone()))
        inp.extra_cache.copy_(quantize_rope_(convert_extra_cache_(inp.extra_cache.clone())))
        fast = dict(extra_is_fnuz=True, extra_scale_mode="none", rope_fp8=True)
        variants = [(lab, {**fast, **kw}) for lab, kw in variants]
    runners = [(lab, _runner(dsv4, inp, kw)) for lab, kw in variants]

    for _lab, fn in runners:      # compile every variant before timing
        fn()
    torch.cuda.synchronize()
    print(f"GPU {phys}, locality={args.locality}"
          + (f" corr_len={args.corr_len}" if args.locality == "realistic" else "")
          + ", burning clocks ...")
    _burn()

    sclk0, pw0, _ = _gpu_state(phys)
    nb0 = _neighbours_busy(phys)
    samples = {lab: [] for lab, _ in runners}
    clk = []
    for rnd in range(args.rounds):
        order = list(range(len(runners)))
        if rnd % 2:
            order.reverse()
        for i in order:
            lab, fn = runners[i]
            samples[lab].append(_event_ms(fn, args.warmup, args.iters))
        if rnd % max(1, args.rounds // 8) == 0:
            s, _p, _b = _gpu_state(phys)
            if s:
                clk.append(s)
        print(f"\r  round {rnd + 1}/{args.rounds}", end="", flush=True)
    print()
    sclk1, pw1, _ = _gpu_state(phys)
    nb1 = _neighbours_busy(phys)

    print(f"\nsclk {sclk0} -> {sclk1} MHz"
          + (f" (sampled {min(clk)}-{max(clk)})" if clk else "")
          + f"   power {pw0} -> {pw1} W")
    if max(nb0, nb1):
        print(f"  WARNING: other GPUs busy ({nb0}% -> {nb1}% summed) -- shared box, "
              f"contention may have inflated samples")

    # One-sided noise: keep a round only if neither side of *this* comparison was
    # slow in it. Judging every variant at once would compound the rejection rate
    # with variant count and throw away usable pairs.
    best = {lab: min(v) for lab, v in samples.items()}
    tol = 1 + args.clean_pct / 100

    def clean_pairs(lab):
        return [
            r for r in range(args.rounds)
            if samples["baseline"][r] <= best["baseline"] * tol
            and samples[lab][r] <= best[lab] * tol
        ]

    print(f"{args.rounds} rounds x {args.iters} calls, interleaved; pairwise-clean "
          f"within {args.clean_pct}% of each side's own best\n")
    print(f"  {'variant':<16}{'best ms':>9}{'clean med':>11}{'vs base':>10}"
          f"{'95% CI':>20}{'n':>5}  verdict")
    print(f"  {'baseline':<16}{best['baseline']:>9.4f}"
          f"{statistics.median(samples['baseline']):>11.4f}{'--':>10}{'--':>20}")
    stats = {}
    for lab, _ in runners:
        if lab == "baseline":
            continue
        keep = clean_pairs(lab)
        if len(keep) < 3:
            print(f"  {lab:<16}{best[lab]:>9.4f}{'':>11}{'':>10}{'too few clean pairs':>20}"
                  f"{len(keep):>5}")
            continue
        deltas = [
            100.0 * (samples[lab][r] - samples["baseline"][r]) / samples["baseline"][r]
            for r in keep
        ]
        d = statistics.median(deltas)
        lo, hi = _bootstrap_ci(deltas)
        verdict = "SLOWER" if lo > 0 else "FASTER" if hi < 0 else "inconclusive"
        med = statistics.median([samples[lab][r] for r in keep])
        print(f"  {lab:<16}{best[lab]:>9.4f}{med:>11.4f}{d:>+9.2f}%"
              f"{f'[{lo:+.2f}%, {hi:+.2f}%]':>20}{len(keep):>5}  {verdict}")
        stats[lab] = deltas

    ctl = "A/A control" if args.aa else next(iter(stats), None)
    if ctl in stats:
        sd = statistics.stdev(stats[ctl])
        n = len(stats[ctl])
        print(f"\n  paired noise sd {sd:.2f}% (from {ctl}) -> minimum detectable effect "
              f"~{1.96 * sd / n ** 0.5:.2f}% on {n} clean pairs")
        print(f"  to resolve 0.5% you would need ~{max(3, int((1.96 * sd / 0.5) ** 2) + 1)} "
              f"clean pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import contextlib
import itertools
import os
import statistics
import sys
import warnings
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.test_common import benchmark
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    calc_diff,
    e4m3_type,
    generate_cp_test_data,
    per_custom_dims_cast_to_fp8,
    ref_fp8_mqa_logits,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
# `e4m3_type` is arch-dependent (get_fp8_dtypes): FNUZ on gfx942, FN on gfx950.
# So on gfx950 both keys resolve to float8_e4m3fn and the two cases coincide --
# only gfx942 has a genuine FN/FNUZ split.
DTYPE_MAP = {"fnuz": e4m3_type, "fn": torch.float8_e4m3fn}

# Default operand-dtype sweep, per arch.
#
# gfx942's native MFMA operand format is FNUZ, so the kernel takes FN operands
# by patching them (see `convert_q_fn`/`convert_kv_fn`). Both fnuz/fnuz and the
# live DeepSeek-V4 indexer combo fn/fnuz are therefore real, distinct paths.
#
# gfx950's native format is FN and its CDNA4 scaled atoms reject FNUZ outright,
# so the kernel never converts there and fn/fn is the only combination that can
# occur.
_DEFAULT_Q_DTYPES = ["fn"] if get_gfx() == "gfx950" else ["fnuz", "fn"]
_DEFAULT_KV_DTYPES = ["fn"] if get_gfx() == "gfx950" else ["fnuz"]

MAX_REL_DELTA = 1e-3

try:
    from aiter.ops.flydsl import flydsl_fp8_mqa_logits
except ImportError:
    flydsl_fp8_mqa_logits = None

# Bench timing knobs, set from argv in main(), read by _time_us.
BENCH_WARMUP = 10
BENCH_SAMPLES = 20
BENCH_REPLAYS = 50


@contextlib.contextmanager
def _fill_output_with_nan(s_q, s_k):
    """NaN-fill the output buffer a launcher allocates inside this block."""
    real_empty = torch.empty
    nan_filled = []

    def _empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if (
            t.dtype == torch.float32
            and t.dim() == 2
            and t.shape[0] >= s_q
            and t.shape[1] >= s_k
        ):
            t.fill_(float("nan"))
            nan_filled.append(tuple(t.shape))
        return t

    torch.empty = _empty
    try:
        yield nan_filled
    finally:
        torch.empty = real_empty


def _make_windows(s_q, s_k, mode):
    if mode == "causal":
        ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
        ke = torch.arange(s_q, dtype=torch.int, device="cuda") + (s_k - s_q)
        return ks, ke
    if mode == "cp":
        return generate_cp_test_data(s_q, s_k)
    if mode == "misaligned":
        rows = torch.arange(s_q, device="cuda")
        ks = ((rows * 53 + 100) % max(1, s_k // 2)).to(torch.int32)
        ke = torch.minimum(ks + max(1, s_k // 3), torch.full_like(ks, s_k)).to(
            torch.int32
        )
        return ks, ke
    if mode == "empty":
        # Rows with no window at all, interleaved with normal ones so a single
        # block's union window mixes the two. Both spellings of "empty" appear:
        #
        #   r%3==0  cu_ends < 0        -- what a causal mask yields whenever
        #                                 s_kv < s_q, so this is ordinary input,
        #                                 not a synthetic edge case
        #   r%3==1  cu_ends <= cu_starts
        #   r%3==2  an ordinary non-empty window
        #
        # Both leave the union tile_end below tile_start, which the kernel must
        # collapse to zero width before the (unsigned) grid.y split arithmetic.
        rows = torch.arange(s_q, device="cuda")
        ks = torch.where(rows % 3 == 1, min(100, s_k), 0)
        ke = torch.where(
            rows % 3 == 0,
            -1 - (rows % 7),
            torch.where(rows % 3 == 1, 0, torch.minimum(rows + 1, ks + s_k)),
        )
        return ks.to(torch.int32), ke.to(torch.int32)
    if mode == "past_end":
        # cu_starts beyond seq_len_kv, interleaved with ordinary rows. A window
        # that starts past the end of KV is legal input and simply empty, but it
        # is the one case where the kernel's -inf fill must clamp cu_starts:
        # the fill's first range is [0, cu_starts), the per-row output view is a
        # buffer descriptor covering 4 GiB from the row base (no hardware OOB
        # net), and seq_len_kv is often the row stride exactly.
        # An unclamped fill would run straight into the next row's live columns.
        # The reference masks with an unclamped `col >= cu_starts`, so it agrees:
        # these rows are entirely -inf.
        rows = torch.arange(s_q, device="cuda")
        ks = torch.where(rows % 2 == 0, s_k + 17, 0)
        ke = torch.where(rows % 2 == 0, s_k + 64, torch.minimum(rows + 1, ks + s_k))
        return ks.to(torch.int32), ke.to(torch.int32)
    raise ValueError(f"unknown window mode: {mode}")


def _rehydrate(out, ks, ke, s_q, s_k, clean_logits):
    if clean_logits:
        return out
    full = torch.full((s_q, s_k), float("-inf"), device="cuda")
    # Clamp into [0, s_k] before slicing: cu_ends is allowed to be negative or
    # to sit below cu_starts (both mean "this row has no window"), and a raw
    # negative bound would wrap into a from-the-end slice and copy most of the
    # row instead of none of it.
    lo = ks.clamp(0, s_k)
    hi = ke.clamp(0, s_k)
    for i in range(s_q):
        a, b = int(lo[i]), int(hi[i])
        if a < b:
            full[i, a:b] = out[i, a:b]
    return full


def _kv_in_dtype(kv_fp8_fnuz, kv_dtype):
    if kv_dtype == e4m3_type:
        return kv_fp8_fnuz
    return kv_fp8_fnuz.to(torch.float32).to(kv_dtype)


Inputs = namedtuple("Inputs", "q kv q_fp8 kv_fp8 scales weights ks ke")


def _make_inputs(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, window):
    """Build one case's operands. `q`/`kv` are the bf16 grading inputs."""
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, dtype=torch.float32)

    ks, ke = _make_windows(s_q, s_k, window)

    q_fp8 = q.to(DTYPE_MAP[q_dtype])
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    # A no-op when the request is already the arch-native format.
    kv_fp8 = _kv_in_dtype(kv_fp8, DTYPE_MAP[kv_dtype])

    # Grade against exactly what the kernels consume, in fp32. The launcher is
    # handed (q_fp8, kv_fp8, scales) and works from kv_fp8 * scales, so that
    # product -- not the bf16 tensor it was quantized from -- is the kernel's
    # real input.
    q = q_fp8.to(torch.float32)
    kv = kv_fp8.to(torch.float32) * scales.reshape(-1, 1)

    return Inputs(q, kv, q_fp8, kv_fp8, scales, weights, ks, ke)


def _candidates(inp, kv_dtype, clean_logits):
    candidates = {
        "flydsl": lambda: flydsl_fp8_mqa_logits(
            inp.q_fp8, inp.kv_fp8, inp.scales, inp.weights, inp.ks, inp.ke, clean_logits
        ),
    }
    if DTYPE_MAP[kv_dtype] == e4m3_type:
        candidates["triton"] = lambda: triton_logits(
            inp.q_fp8, inp.kv_fp8, inp.scales, inp.weights, inp.ks, inp.ke, clean_logits
        )
    return candidates


def _case_tag(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window):
    """One-line case identifier, repeated into every failure message."""
    return (
        f"s_q={s_q} s_k={s_k} nh={num_heads} hd={head_dim} q={q_dtype} "
        f"kv={kv_dtype} clean_logits={bool(clean_logits)} window={window}"
    )


def _grade(name, fn, inp, ref, ref_mask, s_q, s_k, clean_logits, tag):
    """Run `fn` exactly once and compare against `ref`.

    Returns `(calc_diff, rel_delta)`. Every failure raises an AssertionError
    naming the candidate, the case, and the first offending position, so a
    single log line identifies what broke and where.
    """
    # Intercept the regular torch.empty call the launcher makes to allocate the
    # output buffer and fill it with NaN. Any position the kernel fails to write
    # stays NaN, which the -inf mask check below then catches.
    #
    # Without this the mask check is close to vacuous: PyTorch's caching
    # allocator hands back a block a previous case already left holding the
    # correct -inf, so an under-fill (a missed grid.y chunk, an off-by-one at a
    # range boundary) would pass. It also hardens `clean_logits=False` by proving
    # the epilogue writes every in-window position.
    #
    # Applied to triton too where it essentially does nothing since its launcher
    # still pre-fills with torch.full when clean_logits=True.
    with torch.inference_mode(), _fill_output_with_nan(s_q, s_k) as nan_filled:
        out = fn()
    if name == "flydsl" and not nan_filled:
        raise AssertionError(f"{name}: output buffer was not intercepted [{tag}]")

    out = _rehydrate(out, inp.ks, inp.ke, s_q, s_k, clean_logits)

    out_mask = out == float("-inf")
    if not torch.equal(out_mask, ref_mask):
        wrong = (out_mask != ref_mask).nonzero()
        r, c = wrong[0].tolist()
        raise AssertionError(
            f"{name}: -inf mask mismatch at {len(wrong)} of {ref.numel()} "
            f"positions, first at (row={r}, col={c}) "
            f"out={out[r, c].item()} ref={ref[r, c].item()} [{tag}]"
        )

    if ref_mask.all():
        return 0.0, 0.0

    ref_f = ref.masked_fill(ref_mask, 0).to(dtypes.fp32)
    out_f = out.masked_fill(out_mask, 0).to(dtypes.fp32)

    diff = calc_diff(out_f, ref_f)
    # calc_diff is 1 - 2xy/(x^2+y^2), an aggregate similarity. Over a single
    # finite element it degenerates to (a-b)^2/(a^2+b^2), where one borderline
    # ReLU term (a dot product near zero flipping sign between fp32 accumulation
    # orders) moves it by percent. Both the FlyDSL and Triton kernels land on the
    # same value there and differ from the fp32 reference identically, so assert
    # the aggregate only where it is meaningful; MAX_REL_DELTA below bounds the
    # magnitude in every case.
    if int((~ref_mask).sum()) > 1 and not diff < 1e-3:
        raise AssertionError(f"{name}: calc_diff={diff.item():.3e} >= 1e-3 [{tag}]")

    delta = (ref_f - out_f).abs()
    scale = ref_f.abs().max()
    rel = (delta.max() / scale).item() if scale > 0 else delta.max().item()
    if not rel < MAX_REL_DELTA:
        r, c = divmod(int(delta.argmax()), ref.shape[1])
        raise AssertionError(
            f"{name}: max|ref-out|/|ref|max = {rel:.3e} >= {MAX_REL_DELTA:.3e} "
            f"at (row={r}, col={c}) out={out_f[r, c].item()} "
            f"ref={ref_f[r, c].item()} [{tag}]"
        )

    return diff.item(), rel


@benchmark()
def verify_fp8_mqa_logits(
    s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window
):
    """Grade one call per candidate against the fp32 reference. No timing,
    each kernel runs exactly once.
    """
    inp = _make_inputs(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, window)
    tag = _case_tag(
        s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window
    )

    with torch.inference_mode():
        ref, _ = ref_fp8_mqa_logits(
            q=inp.q,
            kv=inp.kv,
            weights=inp.weights,
            cu_seqlen_ks=inp.ks,
            cu_seqlen_ke=inp.ke,
        )
    ref_mask = ref == float("-inf")

    ret = {"gfx": get_gfx(), "status": "ok"}
    for name, fn in _candidates(inp, kv_dtype, clean_logits).items():
        err, rel = _grade(name, fn, inp, ref, ref_mask, s_q, s_k, clean_logits, tag)
        ret[f"{name} err"] = err
        ret[f"{name} rel"] = rel

    return ret


def _captured_node_count(graph):
    """Captured node count, or None if this torch build does not expose it."""
    for attr in ("num_nodes", "_num_nodes"):
        probe = getattr(graph, attr, None)
        if probe is None:
            continue
        value = probe() if callable(probe) else probe
        if isinstance(value, int):
            return value
    return None


def _bench_graph_us(fn):
    """HIP graph-replay steady state for `fn`, as a median over samples."""
    # Warm on the capture stream. The floor of 3 ensures at least one call lands
    # after the FlyDSL cache miss, which runs the kernel twice (once for the real
    # output, then one canary launch from flyc.compile).
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(max(3, BENCH_WARMUP)):
            fn()
    torch.cuda.synchronize()

    # torch.cuda.graph makes capture_stream current, so a launcher that threads
    # torch.cuda.current_stream() is recorded. One that omits the stream runs on
    # the HIP NULL stream and is dropped, leaving an empty graph that would time
    # as a meaningless ~1 us -- detect that instead of reporting it.
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.cuda.graph(graph, stream=capture_stream):
            fn()
    nodes = _captured_node_count(graph)
    if nodes == 0 or (
        nodes is None and any("empty" in str(w.message).lower() for w in caught)
    ):
        raise RuntimeError(
            "graph capture recorded zero nodes: the kernel launched on the NULL "
            "stream and was not captured"
        )

    # Each sample brackets BENCH_REPLAYS back-to-back replays under one event
    # pair and divides. Replays serialize on the stream FIFO, so this is serial
    # device time; bracketing amortizes the per-replay event and dispatch cost.
    samples = []
    for _ in range(BENCH_SAMPLES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(BENCH_REPLAYS):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / BENCH_REPLAYS * 1000.0)  # ms -> us
    return statistics.median(samples)


def _time_us(name, fn, tag):
    """Median graph-replay latency of one candidate, or NaN if not measurable."""
    try:
        return _bench_graph_us(fn)
    except (RuntimeError, torch.OutOfMemoryError) as exc:
        aiter.logger.warning("%s: timing failed: %s [%s]", name, exc, tag)
        return float("nan")


@benchmark()
def bench_fp8_mqa_logits(
    s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window
):
    """Time each candidate, and grade it exactly as verify does.

    The graded call stays separate from the timed replays: the replayed graph
    writes into the buffer captured with it, so grading it would lose the NaN
    interception `_grade` depends on.
    """
    inp = _make_inputs(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, window)
    tag = _case_tag(
        s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window
    )

    with torch.inference_mode():
        ref, cost = ref_fp8_mqa_logits(
            q=inp.q,
            kv=inp.kv,
            weights=inp.weights,
            cu_seqlen_ks=inp.ks,
            cu_seqlen_ke=inp.ke,
        )
    ref_mask = ref == float("-inf")

    flops = cost.item() * num_heads * head_dim * 2
    nbytes = (
        s_q * num_heads * head_dim
        + s_k * head_dim  # Q + KV (fp8)
        + (s_k + s_q * num_heads) * 4  # scales + weights
        + 2 * s_q * 4  # ks + ke
        + s_q * s_k * 4  # output
    )

    ret = {"gfx": get_gfx(), "status": "ok"}
    times = {}
    for name, fn in _candidates(inp, kv_dtype, clean_logits).items():
        err, rel = _grade(name, fn, inp, ref, ref_mask, s_q, s_k, clean_logits, tag)
        us = _time_us(name, fn, tag)
        times[name] = us
        ret[f"{name} us"] = us
        # NaN, not 0, when the windows select nothing: every row of this sweep
        # with an empty window has flops == 0, and a 0 in a TFLOPS column reads
        # as a broken measurement rather than as "there was no work to do". The
        # kernel still writes the -inf fill, so TB/s stays meaningful.
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if flops > 0 else float("nan")
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} rel"] = rel

    # Perf ratio straight off the runtimes rather than off TFLOPS.
    ret["speedup"] = (
        times["triton"] / times["flydsl"] if "triton" in times else float("nan")
    )

    return ret


Case = namedtuple(
    "Case", "s_q s_k num_heads head_dim q_dtype kv_dtype clean_logits window"
)


def _log_speedup(ratios):
    """Headline FlyDSL-over-Triton figure for a bench sweep.

    Geometric mean, because these are ratios: an arithmetic mean over a sweep
    spanning 0.13x to 70x would just report the widest win.
    """
    r = pd.to_numeric(ratios, errors="coerce")
    r = r[r.gt(0) & r.lt(float("inf"))].dropna()
    if r.empty:
        return
    aiter.logger.info(
        "fp8_mqa_logits bench: FlyDSL speedup over Triton on %d of %d cases: "
        "geomean=%.2fx min=%.2fx max=%.2fx, FlyDSL faster on %d (>1 is faster)",
        len(r),
        len(ratios),
        float(np.exp(np.log(r).mean())),
        r.min(),
        r.max(),
        int(r.gt(1).sum()),
    )


def _cp_eligible(s_q, s_k):
    return s_k % s_q == 0 and s_q % 2 == 0


def _full_set(args):
    """The cartesian product of every axis -- ~500 cases on the defaults."""
    cases = []
    for (s_q, s_k), nh, hd, qd, kvd, cl, win in itertools.product(
        args.shapes,
        args.num_heads,
        args.head_dim,
        args.q_dtype,
        args.kv_dtype,
        args.clean_logits,
        args.window,
    ):
        if win == "cp" and not _cp_eligible(s_q, s_k):
            continue
        cases.append(Case(s_q, s_k, nh, hd, qd, kvd, bool(cl), win))
    return cases


def _reduced_set(args):
    """One case per (shape, window) pair, with the remaining axes rotated.

    56 cases on the defaults, which is what the CI lane runs. Shape and window
    are the axes that reach genuinely distinct kernel paths -- the grid.y split,
    the negative/empty window collapse, the cu_starts clamp -- so they are
    covered exhaustively. num_heads, head_dim, clean_logits and the operand
    dtype pair only select between tile shapes and epilogues, so they rotate
    instead. The strides below are powers of two against 5 window modes per
    shape, so each of those values still lands on many different shapes and
    windows rather than tracking one of them.
    """
    dtype_pairs = list(itertools.product(args.q_dtype, args.kv_dtype))
    cases = []
    for s_q, s_k in args.shapes:
        for win in args.window:
            if win == "cp" and not _cp_eligible(s_q, s_k):
                continue
            i = len(cases)
            qd, kvd = dtype_pairs[(i // 8) % len(dtype_pairs)]
            cases.append(
                Case(
                    s_q,
                    s_k,
                    args.num_heads[i % len(args.num_heads)],
                    args.head_dim[(i // 2) % len(args.head_dim)],
                    qd,
                    kvd,
                    bool(args.clean_logits[(i // 4) % len(args.clean_logits)]),
                    win,
                )
            )
    return cases


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fp8_mqa_logits unsupported on %s; skipping", get_gfx())
        return
    if flydsl_fp8_mqa_logits is None:
        aiter.logger.warning("flydsl package not installed; skipping")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fp8_mqa_logits correctness + perf sweep",
    )
    parser.add_argument(
        "--scenario",
        choices=("verify", "bench", "all"),
        default="verify",
        help="verify: one call per candidate, graded against the fp32 reference\n"
        "bench:  the same grading, plus graph-replay timings\n"
        "all:    both, verify first\n"
        "(default: verify)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="warmup calls before graph capture"
    )
    parser.add_argument(
        "--bench-samples", type=int, default=20, help="timed samples per candidate"
    )
    parser.add_argument(
        "--replay-iters",
        type=int,
        default=50,
        help="graph replays bracketed under one event pair per sample",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="sweep the full cartesian product of every axis (~500 cases)\n"
        "instead of the default covering set (one case per shape x window)",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1),
            (1, 16),
            (1, 113),
            (17, 76),
            (61, 113),
            (61, 1024),
            (128, 1024),
            (1024, 1024),
            (1024, 1560),
            # Small-M / long-KV. The row grid alone is far too small to fill the
            # device here (64 rows is a 16-block grid on the gfx950 default
            # variant), so the launcher splits each row's KV window hard across
            # grid.y -- these are the only shapes reaching a split count high
            # enough that most blocks end up owning an empty column range.
            (64, 2048),
            (64, 8192),
            # s_kv < s_q. A causal mask then puts cu_ends below zero on the
            # leading rows, so these cover the negative-window path end to end.
            (128, 64),
            (1024, 1000),
        ],
    )
    parser.add_argument("--num-heads", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--head-dim", type=int, nargs="*", default=[64, 128])
    parser.add_argument(
        "--q-dtype",
        type=str,
        nargs="*",
        default=_DEFAULT_Q_DTYPES,
        choices=["fnuz", "fn"],
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        nargs="*",
        default=_DEFAULT_KV_DTYPES,
        choices=["fnuz", "fn"],
    )
    parser.add_argument(
        "--clean-logits",
        type=int,
        nargs="*",
        default=[0, 1],
        choices=[0, 1],
    )
    parser.add_argument(
        "-w",
        "--window",
        type=str,
        nargs="*",
        default=["causal", "cp", "misaligned", "empty", "past_end"],
        choices=["causal", "cp", "misaligned", "empty", "past_end"],
    )
    args = parser.parse_args()

    cases = _full_set(args) if args.full else _reduced_set(args)
    if not cases:
        aiter.logger.warning("fp8_mqa_logits: the requested axes select no cases")
        return
    scenarios = ("verify", "bench") if args.scenario == "all" else (args.scenario,)
    aiter.logger.info("fp8_mqa_logits: %d cases x %s", len(cases), "+".join(scenarios))

    if "bench" in scenarios:
        global BENCH_WARMUP, BENCH_SAMPLES, BENCH_REPLAYS
        BENCH_WARMUP = args.warmup
        BENCH_SAMPLES = args.bench_samples
        BENCH_REPLAYS = args.replay_iters
        aiter.logger.info(
            "fp8_mqa_logits: graph-replay timing, warmup=%d samples=%d "
            "replays/sample=%d (us columns are medians)",
            args.warmup,
            args.bench_samples,
            args.replay_iters,
        )

    failures = []
    for scenario in scenarios:
        run = verify_fp8_mqa_logits if scenario == "verify" else bench_fp8_mqa_logits
        rows = []
        for case in cases:
            try:
                rows.append(run(*case))
            except Exception as exc:  # noqa: BLE001 - record, keep sweeping
                # Keep sweeping. One bad shape should not hide the verdict on
                # the other 55, and the exit code below still fails the run.
                # AssertionErrors already name the case; anything else is
                # unexpected, so keep its traceback.
                aiter.logger.error(
                    "FAILED %s case: %s\n    %s",
                    scenario,
                    _case_tag(*case),
                    exc,
                    exc_info=not isinstance(exc, AssertionError),
                )
                failures.append((scenario, case, exc))
                rows.append({**case._asdict(), "gfx": get_gfx(), "status": "FAIL"})
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "fp8_mqa_logits %s summary (markdown):\n%s",
            scenario,
            df.to_markdown(index=False),
        )
        if "speedup" in df:
            _log_speedup(df["speedup"])

    total = len(cases) * len(scenarios)
    if failures:
        aiter.logger.error(
            "fp8_mqa_logits: %d of %d case runs FAILED\n%s",
            len(failures),
            total,
            "\n".join(
                f"  [{sc}] {_case_tag(*case)}\n        {exc}"
                for sc, case, exc in failures
            ),
        )
        sys.exit(1)
    aiter.logger.info("fp8_mqa_logits: all %d case runs passed", total)


if __name__ == "__main__":
    main()

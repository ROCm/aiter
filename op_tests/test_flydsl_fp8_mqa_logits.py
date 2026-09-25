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
import triton

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention import fp8_mqa_logits as _triton_mod
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.test_common import benchmark
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
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

# `triton_logits` is an entry point, not an implementation: it dispatches to the
# hand-written Gluon kernel on the arches that ship one (gfx950, gfx1250) and
# falls back to the generic Triton kernel elsewhere. Name the column after
# whichever actually runs -- on gfx950 the competing implementation is the Gluon
# kernel of PR #5216, and reporting it as "triton" would hide that.
REF_IMPL = (
    "gluon"
    if (
        _triton_mod.TRITON_GE_36
        and _triton_mod._gluon_fp8_mqa_logits_kernel is not None
    )
    else "triton"
)

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


def _make_windows(s_q, s_k, mode, batch=1):
    if mode == "batch_causal":
        # Block-diagonal causal windows: `batch` independent sequences packed
        # into one (s_q, s_k) call, where the rows of sequence b see only KV
        # block b.
        if s_q % batch or s_k % batch:
            raise ValueError(
                f"batch={batch} does not divide s_q={s_q} / s_k={s_k} evenly"
            )
        q_l, kv_l = s_q // batch, s_k // batch
        rows = torch.arange(s_q, device="cuda")
        ks = (rows // q_l) * kv_l
        ke = ks + (kv_l - q_l) + (rows % q_l) + 1
        return ks.to(torch.int32), ke.to(torch.int32)
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


def _rehydrate(out_chunk, ks_chunk, ke_chunk, s_k, clean_logits):
    """Force the out-of-window positions of a row block to -inf.

    `clean_logits=False` leaves them unspecified, so the candidate may have
    written anything (or, thanks to the NaN pre-fill, nothing). Masking them
    here is what lets the -inf mask check below still prove the epilogue wrote
    every *in*-window position.

    The window bounds are applied as the reference applies them -- `col >= ks`
    and `col < ke`, unclamped -- so a negative `ke` or a `ks` past the end of KV
    simply selects nothing, which is exactly what those inputs mean.
    """
    if clean_logits:
        return out_chunk
    cols = torch.arange(s_k, device=out_chunk.device)
    in_window = (cols[None, :] >= ks_chunk[:, None]) & (
        cols[None, :] < ke_chunk[:, None]
    )
    return out_chunk.where(in_window, float("-inf"))


# Rows graded per block, as a budget on the (head, row, col) fp32 elements the
# reference materializes. `ref_fp8_mqa_logits` builds the full [num_heads, rows,
# s_k] score tensor, so grading a model shape in one shot is out of reach: 64
# heads x 16k rows x 64k columns is 274 GB before any of the downstream masks
# and calc_diff's float64 cast. 2**28 elements keeps that tensor near 1 GB, and
# everything downstream of it is a factor of num_heads smaller.
_GRADE_BLOCK_ELEMS = 1 << 28


def _row_block_rows(s_k, num_heads):
    return max(1, _GRADE_BLOCK_ELEMS // max(1, num_heads * s_k))


def _run_candidate(name, fn, s_q, s_k):
    """Run `fn` exactly once, with its output buffer NaN-poisoned first."""
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
    # Applied to triton/gluon too where it essentially does nothing since that
    # launcher still pre-fills with torch.full when clean_logits=True.
    with torch.inference_mode(), _fill_output_with_nan(s_q, s_k) as nan_filled:
        out = fn()
    if name == "flydsl" and not nan_filled:
        raise AssertionError(f"{name}: output buffer was not intercepted [{s_q}x{s_k}]")
    return out


def _grade_all(outs, inp, s_q, s_k, num_heads, clean_logits, tag):
    """Compare every candidate's output against the fp32 reference.

    Returns `{name: (calc_diff, rel_delta)}`. Every failure raises an
    AssertionError naming the candidate, the case, and the first offending
    position, so a single log line identifies what broke and where.

    Streams the comparison a row block at a time and accumulates, rather than
    materializing the reference for the whole matrix (see `_GRADE_BLOCK_ELEMS`).
    Both statistics are exactly the whole-matrix ones: calc_diff is a ratio of
    two plain sums, and `rel` divides the global max absolute error by the
    global max |ref|, so each is a pair of running accumulators.
    """
    names = list(outs)
    # calc_diff = 1 - 2*sum(xy)/sum(x^2+y^2), accumulated in float64 to match
    # the shared helper's `.double()` cast.
    xy = {n: 0.0 for n in names}
    xx_yy = {n: 0.0 for n in names}
    max_delta = {n: 0.0 for n in names}
    max_at = {n: None for n in names}
    ref_absmax = 0.0
    finite = 0

    block = _row_block_rows(s_k, num_heads)
    for r0 in range(0, s_q, block):
        r1 = min(r0 + block, s_q)
        with torch.inference_mode():
            ref, _ = ref_fp8_mqa_logits(
                q=inp.q[r0:r1],
                kv=inp.kv,
                weights=inp.weights[r0:r1],
                cu_seqlen_ks=inp.ks[r0:r1],
                cu_seqlen_ke=inp.ke[r0:r1],
            )
        ref_mask = ref == float("-inf")
        ref_f = ref.masked_fill(ref_mask, 0).to(dtypes.fp32)
        finite += int((~ref_mask).sum())
        ref_absmax = max(ref_absmax, float(ref_f.abs().max()))

        for name in names:
            out = _rehydrate(
                outs[name][r0:r1], inp.ks[r0:r1], inp.ke[r0:r1], s_k, clean_logits
            )
            out_mask = out == float("-inf")
            if not torch.equal(out_mask, ref_mask):
                wrong = (out_mask != ref_mask).nonzero()
                r, c = wrong[0].tolist()
                raise AssertionError(
                    f"{name}: -inf mask mismatch at {len(wrong)} of {ref.numel()} "
                    f"positions in rows [{r0},{r1}), first at "
                    f"(row={r0 + r}, col={c}) out={out[r, c].item()} "
                    f"ref={ref[r, c].item()} [{tag}]"
                )
            out_f = out.masked_fill(out_mask, 0).to(dtypes.fp32)

            x, y = out_f.double(), ref_f.double()
            xy[name] += float((x * y).sum())
            xx_yy[name] += float((x * x + y * y).sum())

            delta = (ref_f - out_f).abs()
            local = float(delta.max()) if delta.numel() else 0.0
            if local > max_delta[name]:
                max_delta[name] = local
                r, c = divmod(int(delta.argmax()), ref.shape[1])
                max_at[name] = (r0 + r, c, float(out_f[r, c]), float(ref_f[r, c]))

    results = {}
    for name in names:
        if finite == 0:
            results[name] = (0.0, 0.0)
            continue
        diff = 1.0 - (2 * xy[name] / xx_yy[name] if xx_yy[name] else 1.0)
        # calc_diff is 1 - 2xy/(x^2+y^2), an aggregate similarity. Over a single
        # finite element it degenerates to (a-b)^2/(a^2+b^2), where one
        # borderline ReLU term (a dot product near zero flipping sign between
        # fp32 accumulation orders) moves it by percent. Both the FlyDSL and
        # Gluon kernels land on the same value there and differ from the fp32
        # reference identically, so assert the aggregate only where it is
        # meaningful; MAX_REL_DELTA below bounds the magnitude in every case.
        if finite > 1 and not diff < 1e-3:
            raise AssertionError(f"{name}: calc_diff={diff:.3e} >= 1e-3 [{tag}]")

        rel = max_delta[name] / ref_absmax if ref_absmax > 0 else max_delta[name]
        if not rel < MAX_REL_DELTA:
            r, c, got, want = max_at[name]
            raise AssertionError(
                f"{name}: max|ref-out|/|ref|max = {rel:.3e} >= "
                f"{MAX_REL_DELTA:.3e} at (row={r}, col={c}) out={got} "
                f"ref={want} [{tag}]"
            )
        results[name] = (diff, rel)
    return results


def _kv_in_dtype(kv_fp8_fnuz, kv_dtype):
    if kv_dtype == e4m3_type:
        return kv_fp8_fnuz
    return kv_fp8_fnuz.to(torch.float32).to(kv_dtype)


Inputs = namedtuple("Inputs", "q kv q_fp8 kv_fp8 scales weights ks ke")


def _make_inputs(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, window, batch=1):
    """Build one case's operands. `q`/`kv` are the bf16 grading inputs."""
    torch.manual_seed(0)
    q = torch.randn(s_q, num_heads, head_dim, dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, dtype=torch.bfloat16)
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads, dtype=torch.float32)

    ks, ke = _make_windows(s_q, s_k, window, batch)

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
        candidates[REF_IMPL] = lambda: triton_logits(
            inp.q_fp8, inp.kv_fp8, inp.scales, inp.weights, inp.ks, inp.ke, clean_logits
        )
    return candidates


def _case_tag(
    s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window, batch=1
):
    """One-line case identifier, repeated into every failure message."""
    return (
        f"s_q={s_q} s_k={s_k} batch={batch} nh={num_heads} hd={head_dim} "
        f"q={q_dtype} kv={kv_dtype} clean_logits={bool(clean_logits)} "
        f"window={window}"
    )


def _shape_label(s_q, s_k, batch):
    def _k(n):
        return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)

    return f"{batch}x{_k(s_q // batch)}x{_k(s_k // batch)}"


@benchmark()
def verify_fp8_mqa_logits(
    s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window, batch=1
):
    """Grade one call per candidate against the fp32 reference. No timing,
    each kernel runs exactly once.
    """
    inp = _make_inputs(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, window, batch)
    tag = _case_tag(
        s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window, batch
    )

    candidates = _candidates(inp, kv_dtype, clean_logits)
    outs = {n: _run_candidate(n, fn, s_q, s_k) for n, fn in candidates.items()}
    graded = _grade_all(outs, inp, s_q, s_k, num_heads, clean_logits, tag)

    ret = {"shape": _shape_label(s_q, s_k, batch), "gfx": get_gfx(), "status": "ok"}
    for name, (err, rel) in graded.items():
        ret[f"{name} err"] = err
        ret[f"{name} rel"] = rel

    return ret


_FLUSH_CACHE = None


def _l2_flush_cache():
    """The scratch buffer whose zeroing evicts the LLC, allocated once."""
    global _FLUSH_CACHE
    if _FLUSH_CACHE is None:
        _FLUSH_CACHE = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    return _FLUSH_CACHE


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

    # Each replay gets its own event pair, preceded by an LLC flush that stays
    # outside that pair -- the same shape as Triton's `do_bench`.
    #
    # Replaying back-to-back without the flush instead leaves KV, the scales and
    # the previous replay's output resident in the 256 MiB LLC. Everything is
    # serialized on the stream FIFO, so the flush is complete before the replay
    # it precedes starts, and its own cost falls outside the bracket.
    cache = _l2_flush_cache()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_REPLAYS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(BENCH_REPLAYS)]
    samples = []
    for _ in range(BENCH_SAMPLES):
        for i in range(BENCH_REPLAYS):
            cache.zero_()
            starts[i].record()
            graph.replay()
            ends[i].record()
        torch.cuda.synchronize()
        # Mean within a sample (as do_bench reports), median across samples.
        samples.append(
            statistics.mean(s.elapsed_time(e) for s, e in zip(starts, ends)) * 1000.0
        )  # ms -> us
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
    s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window, batch=1
):
    """Time each candidate, and grade it exactly as verify does.

    The graded call stays separate from the timed replays: the replayed graph
    writes into the buffer captured with it, so grading it would lose the NaN
    interception `_run_candidate` depends on.
    """
    inp = _make_inputs(s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, window, batch)
    tag = _case_tag(
        s_q, s_k, num_heads, head_dim, q_dtype, kv_dtype, clean_logits, window, batch
    )

    with torch.inference_mode():
        cost = ref_fp8_mqa_logits(
            q=inp.q,
            kv=inp.kv,
            weights=inp.weights,
            cu_seqlen_ks=inp.ks,
            cu_seqlen_ke=inp.ke,
            cost_only=True,
        )

    flops = cost.item() * num_heads * head_dim * 2
    # clean_logits=True writes the full [s_q, s_k] matrix (window + -inf fill);
    # clean_logits=False writes only the selected window (`cost` positions), so
    # counting s_q*s_k there would claim full-output bandwidth even when the
    # window is empty and nothing was written at all.
    out_elems = s_q * s_k if clean_logits else cost.item()
    nbytes = (
        s_q * num_heads * head_dim
        + s_k * head_dim  # Q + KV (fp8)
        + (s_k + s_q * num_heads) * 4  # scales + weights
        + 2 * s_q * 4  # ks + ke
        + out_elems * 4  # output
    )

    candidates = _candidates(inp, kv_dtype, clean_logits)
    outs = {n: _run_candidate(n, fn, s_q, s_k) for n, fn in candidates.items()}
    graded = _grade_all(outs, inp, s_q, s_k, num_heads, clean_logits, tag)
    del outs  # the graded buffers are dead; the timed replays allocate their own

    ret = {"shape": _shape_label(s_q, s_k, batch), "gfx": get_gfx(), "status": "ok"}
    times = {}
    for name, fn in candidates.items():
        err, rel = graded[name]
        us = _time_us(name, fn, tag)
        times[name] = us
        ret[f"{name} us"] = us
        # NaN, not 0, when the windows select nothing: every row of this sweep
        # with an empty window has flops == 0, and a 0 in a TFLOPS column reads
        # as a broken measurement rather than as "there was no work to do".
        # TB/s stays meaningful either way: clean_logits=True still counts the
        # -inf fill traffic via out_elems == s_q*s_k above.
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if flops > 0 else float("nan")
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} rel"] = rel

    # Perf ratio straight off the runtimes rather than off TFLOPS.
    ret["speedup"] = (
        times[REF_IMPL] / times["flydsl"] if REF_IMPL in times else float("nan")
    )

    return ret


Case = namedtuple(
    "Case",
    "s_q s_k num_heads head_dim q_dtype kv_dtype clean_logits window batch",
    defaults=(1,),
)


def _log_speedup(ratios):
    """Headline FlyDSL-over-reference figure for a bench sweep.

    Geometric mean, because these are ratios: an arithmetic mean over a sweep
    spanning 0.13x to 70x would just report the widest win.
    """
    r = pd.to_numeric(ratios, errors="coerce")
    r = r[r.gt(0) & r.lt(float("inf"))].dropna()
    if r.empty:
        return
    aiter.logger.info(
        f"fp8_mqa_logits bench: FlyDSL speedup over {REF_IMPL} on %d of %d cases: "
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


_DSv4_AND_GLM_SHAPES = [
    (1, 4096, 4096),
    (1, 8192, 8192),
    (2, 8192, 8192),
    (4, 8192, 8192),
    (1, 8192, 32768),
    (2, 8192, 32768),
]

_MODELS = {"dsv4": 64, "glm5.2": 32}


def _model_set(args):
    """Measurement grid for DSv4 and GLM 5.2: 6 shapes x 2 q-head counts."""
    cases = []
    for cl in args.clean_logits:
        for _, nh in sorted(_MODELS.items()):
            for b, q_l, kv_l in _DSv4_AND_GLM_SHAPES:
                cases.append(
                    Case(
                        s_q=b * q_l,
                        s_k=b * kv_l,
                        num_heads=nh,
                        head_dim=128,
                        q_dtype=args.q_dtype[0],
                        kv_dtype=args.kv_dtype[0],
                        clean_logits=bool(cl),
                        window="batch_causal",
                        batch=b,
                    )
                )
    return cases


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

    65 cases on the defaults, which is what the CI lane runs. Shape and window
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
        help="individually timed graph replays per sample, each preceded by an\n"
        "LLC flush (the flush itself is not timed)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="sweep the full cartesian product of every axis (~500 cases)\n"
        "instead of the default covering set (one case per shape x window)",
    )
    parser.add_argument(
        "--model-shapes",
        action="store_true",
        help="Run only the shapes for DSv4 and GLM 5.2:\n"
        "6 (batch, seq_q, seq_kv) shapes x {dsv4: 64, glm5.2: 32} q-heads,\n"
        "head_dim 128, block-diagonal causal windows. Overrides --shapes,\n"
        "--num-heads, --head-dim and --window; pairs with --scenario bench.",
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
            # r4 auto-select: 256*8192 == 2**21. 257 is odd, so the launcher
            # pads seq_len up to a multiple of RPB=4.
            (256, 8192),
            (257, 8192),
            # s_kv < s_q. A causal mask then puts cu_ends below zero on the
            # leading rows, so these cover the negative-window path end to end.
            (128, 64),
            (1024, 1000),
        ],
    )
    parser.add_argument("--num-heads", type=int, nargs="*", default=[32, 64, 128])
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
        choices=["causal", "cp", "misaligned", "empty", "past_end", "batch_causal"],
    )
    args = parser.parse_args()

    if args.model_shapes:
        cases = _model_set(args)
    elif args.full:
        cases = _full_set(args)
    else:
        cases = _reduced_set(args)
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
            "fp8_mqa_logits: graph-replay timing with a per-replay LLC flush, "
            "warmup=%d samples=%d replays/sample=%d (us columns are medians "
            "over samples of the mean replay)",
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

# Benchmark the shuffled-cache unified attention paths.
"""A/B FlyDSL-shuffled vs Triton-shuffled (the REAL production layout/kernel)
on the newer vLLM-v1 trace's real prefill shapes, weighted by the trace's call
counts (Stage 5a, shuffled-cache-support-investigation.md).

This is bench_trace_shapes.py's linear-cache harness adapted so BOTH sides run
the production 5D shuffled fp8 KV-cache layout (`shuffled_kv_cache=True`) via
the repo's own `shuffle_kv_cache()` writer-layout helper (through
`generate_data`, the same builder test_stage4_matrix.py uses for its
correctness gate) -- not a hand-rolled cache. The BASELINE here is
Triton-shuffled, the kernel/layout that actually runs in production, not
Triton-linear. Calibration (auto KV-depth so modeled time reproduces the
trace's recorded per-call time) uses Triton-shuffled regardless of --baseline.
All backends are then timed on the same inputs; trace-ratio totals are legacy
projections reported separately from measured totals.

The V-side of the FlyDSL shuffled path is Stage-3/4 correctness-first (a
bank-conflict-heavy 16-byte-store scatter per lane), explicitly NOT perf-tuned
(shuffled-cache-support-investigation.md, Stage 3 SOLVED note; Stage 5 is the
coalescing rewrite). This benchmark measures whether that unoptimized V
scatter is already competitive with Triton-shuffled's scalar per-page load, or
whether Stage 5 is required before the projected win can materialize.

Run from the worktree root you want to measure:
    cd ~/projects/aiter/flydsl-unified-attention
    rm -rf ~/.flydsl/cache
    ENABLE_CK=0 python3 -u <this>
    ... --quick     # two shapes (M=32768, M=8290), for a fast sanity check
"""

from __future__ import annotations

import argparse
import os
import sys
from unittest import mock

import torch

sys.path.insert(0, os.getcwd())
import aiter.ops.flydsl.unified_attention_kernels as uak
import aiter.ops.unified_attention as ua
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
)

PAGE = 64
D = 128
H, HKV = 64, 4
DEV = "cuda"
BLOCK_Q = 8  # BLOCK_M(128) // num_queries_per_kv(16), from grid inversion
FLYDSL_DECLINE = "FlyDSL unified_attention backend is unavailable or does not support this configuration"
NUM_BLOCKS = 65536  # margin over generate_data's default 32768 for deep-kv
# calibration probes (mult up to 24x) at large M/n.

# (M, trace_calls, trace_mean_us, grid_y) -- from new-trace-analysis.md tables.
TRACE = [
    (32768, 3196, 1784.8, 4138),
    (29262, 94, 1585.1, 3695),
    (29261, 94, 1591.7, 3694),
    (25613, 94, 1422.3, 3243),
    (25610, 188, 1442.0, 3240),
    (25609, 282, 1457.3, 3239),
    (11795, 94, 801.7, 1538),
    (11793, 94, 791.9, 1538),
    (8303, 94, 618.3, 1101),
    (8294, 94, 609.9, 1100),
    (8291, 94, 613.7, 1100),
    (8290, 94, 611.2, 1100),
    (3697, 94, 309.8, 506),
    (3694, 188, 305.3, 502),
    (3693, 282, 306.7, 501),
]


def build_shuffled(query_lens, kv_lens, num_blocks=NUM_BLOCKS):
    """Build a production-shaped batch: fp8 5D shuffled KV cache via the
    repo's own shuffle_kv_cache() (through generate_data), the same builder
    the Stage 4 correctness matrix uses -- not a hand-rolled layout."""
    seq_lens = list(zip(query_lens, kv_lens))
    data = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=PAGE,
        head_size=D,
        num_heads=(H, HKV),
        q_dtype=e4m3_dtype,
        kv_dtype=e4m3_dtype,
        out_dtype=torch.bfloat16,
        shuffled_kv_cache=True,
        device=DEV,
    )
    (
        _query,
        _key_orig,
        _value_orig,
        shuf_key,
        shuf_value,
        _sinks,
        output,
        cu_query_lens,
        kv_lens_t,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        maybe_quant_query,
        _query_scales,
        q_descale,
        k_descale,
        v_descale,
        _output_scale,
    ) = data
    return {
        "q": maybe_quant_query,
        "k": shuf_key,
        "v": shuf_value,
        "out": output,
        "cu_seqlens_q": cu_query_lens,
        "max_seqlen_q": max_query_len,
        "seqused_k": kv_lens_t,
        "max_seqlen_k": max_kv_len,
        "softmax_scale": scale,
        "causal": True,
        "window_size": window_size,
        "block_table": block_tables,
        "softcap": 0,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "shuffled_kv_cache": True,
    }


def split_equal(M, n):
    base, rem = divmod(M, n)
    return [base + 1] * rem + [base] * (n - rem)


def time_us(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000.0 / iters


def _triton_shuffled_us(qls, mult):
    kw = build_shuffled(qls, [max(1, round(q * mult)) for q in qls])
    return time_us(lambda: ua.unified_attention(**kw, backend="triton"))


def bench_one(M, gy, trace_us):
    # num_seqs from grid: grid.y = M//BLOCK_Q + num_seqs (unified_attention.py:464).
    n = max(1, gy - M // BLOCK_Q)
    qls = split_equal(M, n)

    # Calibrate KV-depth multiplier so modeled Triton-SHUFFLED time reproduces
    # the trace's recorded per-call time -- Triton-shuffled is the production
    # kernel, so calibration is against it, not the linear baseline.
    t1, t4 = _triton_shuffled_us(qls, 1.0), _triton_shuffled_us(qls, 4.0)
    slope = (t4 - t1) / 3.0
    mult = 1.0 + (trace_us - t1) / slope if slope > 1e-6 else 1.0
    mult = min(24.0, max(1.0, mult))

    kvs = [max(1, round(q * mult)) for q in qls]
    kw = build_shuffled(qls, kvs)

    def flydsl():
        return ua.unified_attention(**kw, backend="flydsl")

    real, seen = uak.flydsl_unified_attention, {}

    def spy(*a, **k):
        r = real(*a, **k)
        seen["served"] = r is not None
        return r

    with mock.patch.object(uak, "flydsl_unified_attention", spy):
        try:
            ua.unified_attention(**kw, backend="flydsl")
        except RuntimeError as exc:
            if str(exc) != FLYDSL_DECLINE:
                raise
    served = seen.get("served", False)

    times = {}
    for backend in ("flydsl", "triton", "gluon", "default"):
        selected = (
            ("flydsl" if served else None)
            if backend == "flydsl"
            else (None if backend == "default" else backend)
        )
        try:
            call = (
                flydsl
                if backend == "flydsl" and served
                else (
                    lambda selected=selected: ua.unified_attention(
                        **kw, backend=selected
                    )
                )
            )
            times[backend] = time_us(call)
        except Exception as e:
            if backend == "triton":
                raise
            times[backend] = float("nan")
            print(f"  !! {backend}-shuffled RAISED for M={M}: {type(e).__name__}: {e}")
    return n, mult, times, served


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--baseline", choices=("triton", "gluon"), default="triton")
    a = p.parse_args()
    print("--baseline is deprecated and ignored; calibration uses Triton.")
    rows = [r for r in TRACE if r[0] in (32768, 8290)] if a.quick else TRACE

    import aiter

    print(f"aiter: {aiter.__file__}")
    print(f"cache={os.environ.get('FLYDSL_RUNTIME_ENABLE_CACHE','(default on)')}\n")
    print(
        f"{'M':>6} {'calls':>6} {'nseq':>5} {'kvmult':>6} "
        f"{'fly_us':>8} {'tri_us':>8} {'glu_us':>8} {'def_us':>8} "
        f"{'tri/fly':>8} {'glu/fly':>8} {'served':>7}"
    )
    totals = {backend: 0.0 for backend in ("flydsl", "triton", "gluon", "default")}
    projected_tri = projected_fly = 0.0
    completed = 0
    ceded = 0
    for M, calls, trace_us, gy in rows:
        n, mult, times, served = bench_one(M, gy, trace_us)
        completed += 1
        ceded += not served
        for backend, us in times.items():
            totals[backend] += us * calls / 1000.0
        ratio = times["triton"] / times["flydsl"]
        projected_tri += trace_us * calls / 1000.0
        projected_fly += trace_us * calls / ratio / 1000.0
        print(
            f"{M:>6} {calls:>6} {n:>5} {mult:>6.1f} "
            f"{times['flydsl']:>8.1f} {times['triton']:>8.1f} "
            f"{times['gluon']:>8.1f} {times['default']:>8.1f} "
            f"{ratio:>8.2f} {times['gluon']/times['flydsl']:>8.2f} "
            f"{'yes' if served else 'ceded':>7}"
        )

    print(f"ceded shapes: {ceded}")
    if completed:
        print(
            "TOTAL prefill (measured, trace-weighted ms): "
            + "  ".join(f"{backend} {ms:.1f}" for backend, ms in totals.items())
        )
        print(
            "TOTAL prefill projected (legacy method, trace_us × calls / Triton ratio): "
            f"triton {projected_tri:.1f} ms  flydsl {projected_fly:.1f} ms  "
            f"saved {projected_tri - projected_fly:.1f} ms "
            f"({100 * (projected_tri - projected_fly) / projected_tri:.1f}%)"
        )
    else:
        print("No shapes completed -- see failures above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

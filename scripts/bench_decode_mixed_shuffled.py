# Benchmark shuffled-cache decode and mixed batches.
"""FlyDSL-shuffled vs Triton-shuffled on the DECODE (3d, all-decode) and MIXED
(2d, prefill-chunk + riding decodes) regimes, to complete the trace-weighted
total-attention picture started by bench_trace_shapes_shuffled.py (prefill
only). SILOTIGER-877.

Both regimes build the production 5D shuffled fp8 KV cache via the repo's own
`generate_data(..., shuffled_kv_cache=True)` (which calls `shuffle_kv_cache()`
internally) -- same builder Stage 4's correctness matrix uses, not a
hand-rolled layout.

DECODE calibration problem: the new trace (new-trace-analysis.md) records
3d-decode kernel times by segment tier (16-seg: 18046 launches, mean 53.7us;
32-seg: 1034 launches, mean 50.9us) but NOT the decode batch size `num_seqs`,
because decode runs under a replayed CUDA graph with a null grid. So instead
of grid-inverting num_seqs (the prefill script's trick), we calibrate: for a
fixed representative context depth C chosen to land the segment-count
arithmetic exactly on the target tier (`select_3d_config`'s
`ceil(target_num_prgms/num_2d_prgms)` clipped by
`MAX_SEGMENTS=min(128, ceil(C/TILE_SIZE))`), sweep the decode batch size `b`
and match the whole-call (attn + internal reduce_segments) Triton-shuffled
time to the trace's per-tier mean (decode-kernel-mean + reduce-mean, since one
`unified_attention()` python call issues both kernels for NUM_SEGMENTS>1).
`b` is the real lever here: C only moves segment count in a ~64-token-wide
window (one TILE_SIZE), so it barely changes latency; `b` (sequences served by
one launch) is what actually scales total work.

MIXED batch models a chunked-prefill step with riding decodes in the same
call (2d path) -- the regime bench_unified_attention_dispatch.py already
covers on the LINEAR cache; here both sides run the 5D shuffled cache. This
regime has no call count in the new trace (its decode is separate, pure
all-decode -- new-trace-analysis.md "Overturned conclusion 2"), so it is
reported standalone, not folded into the trace-weighted total.

Run from the worktree root you want to measure:
    cd ~/projects/aiter/flydsl-unified-attention
    rm -rf ~/.flydsl/cache
    ENABLE_CK=0 python3 -u <this>
    ... --quick     # skip the calibration sweep tables, just final numbers
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from unittest import mock

import torch

sys.path.insert(0, os.getcwd())
import aiter.ops.flydsl.unified_attention_kernels as uak
import aiter.ops.triton.attention.unified_attention as triton_ua
import aiter.ops.unified_attention as ua
from aiter.ops.triton.utils.types import e4m3_dtype
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

PAGE = 64
D = 128
H, HKV = 64, 4
DEV = "cuda"
NUM_BLOCKS = 65536
FLYDSL_DECLINE = "FlyDSL unified_attention backend is unavailable or does not support this configuration"

# Trace facts (new-trace-analysis.md "Attention GPU-time budget").
PREFILL_TRI_MS, PREFILL_FLY_MS = 7372.1, 6699.0
DECODE16_LAUNCHES, DECODE16_ATTN_US, DECODE16_REDUCE_US = 18046, 53.7, 12.1
DECODE32_LAUNCHES, DECODE32_ATTN_US, DECODE32_REDUCE_US = 1034, 50.9, 13.2
DECODE16_BUDGET_MS = (
    DECODE16_LAUNCHES * (DECODE16_ATTN_US + DECODE16_REDUCE_US) / 1000.0
)
DECODE32_BUDGET_MS = (
    DECODE32_LAUNCHES * (DECODE32_ATTN_US + DECODE32_REDUCE_US) / 1000.0
)
TOTAL_DECODE_MS = DECODE16_BUDGET_MS + DECODE32_BUDGET_MS  # ~= 1252.5ms


def clear_flydsl_cache():
    d = os.path.expanduser("~/.flydsl/cache")
    if os.path.isdir(d):
        shutil.rmtree(d)
        print(f"cleared {d}")
    else:
        print(f"{d} did not exist, nothing to clear")


def build_shuffled_full(query_lens, kv_lens, num_blocks=NUM_BLOCKS):
    """Production-shaped batch on the 5D shuffled fp8 cache, plus the linear
    (key_cache_orig/value_cache_orig) materials generate_data also returns,
    for a torch-reference correctness check with the SAME block_tables."""
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
        query,
        key_orig,
        value_orig,
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
    kw = {
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
    ref_kw = {
        "query": query,
        "key_cache": key_orig,
        "value_cache": value_orig,
        "query_lens": list(query_lens),
        "kv_lens": list(kv_lens),
        "block_tables": block_tables,
        "scale": scale,
        "out_dtype": torch.float32,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "causal": 1,
    }
    return kw, ref_kw, output


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


def served_check(kw):
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
    return seen.get("served", False)


def captured_segments(kw):
    """Read back the Triton KV-split config's segment count without changing kernels."""
    real = triton_ua.get_unified_attention_config
    seen = {}

    def spy(*a, **k):
        config = real(*a, **k)
        if a[0] == "kv_split":
            seen["segments"] = config["NUM_SEGMENTS"]
        return config

    with mock.patch.object(triton_ua, "get_unified_attention_config", spy):
        ua.unified_attention(**kw, backend="triton")
    return seen.get("segments")


def correctness(kw, ref_kw, output, atol_err=1e-1, min_cos=0.99, served=True):
    ua.unified_attention(**kw, backend="flydsl" if served else None)
    of = output.float()
    want = ref_paged_attn(**ref_kw)
    err = (of - want).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        of.flatten(), want.flatten(), dim=0
    ).item()
    ok = err < atol_err and cos > min_cos
    return err, cos, ok


def bench_decode_shape(b, C):
    kw, ref_kw, output = build_shuffled_full([1] * b, [C] * b)
    segments = captured_segments(kw)
    served = served_check(kw)
    err, cos, ok = correctness(kw, ref_kw, output, served=served)

    def flydsl():
        return ua.unified_attention(**kw, backend="flydsl" if served else None)

    times = {
        backend: time_us(
            flydsl
            if backend == "flydsl"
            else lambda backend=backend: ua.unified_attention(
                **kw, backend=None if backend == "default" else backend
            )
        )
        for backend in ("flydsl", "triton", "gluon", "default")
    }
    return {
        "b": b,
        "C": C,
        "segments": segments,
        "served": served,
        "err": err,
        "cos": cos,
        "ok": ok,
        "times": times,
        "ratio": times["triton"] / times["flydsl"],
    }


def calibrate_decode(label, C, target_us, b_candidates):
    print(
        f"\n--- calibrating {label}: C={C}, target whole-call Triton "
        f"us={target_us:.1f} (decode-kernel + reduce_segments trace means) ---"
    )
    print(f"{'b':>4} {'segments':>9} {'Triton_us':>9} {'target':>8} {'delta':>8}")
    rows = []
    for b in b_candidates:
        r = bench_decode_shape(b, C)
        rows.append(r)
        print(
            f"{b:>4} {r['segments']:>9} {r['times']['triton']:>8.1f} {target_us:>8.1f} "
            f"{r['times']['triton'] - target_us:>8.1f}"
        )
    best = min(rows, key=lambda r: abs(r["times"]["triton"] - target_us))
    print(
        f"  chosen: b={best['b']}, C={C}, segments={best['segments']}, "
        f"Triton_us={best['times']['triton']:.1f}  "
        f"Triton/trace={best['times']['triton']/target_us:.2f})"
    )
    return best


def mixed_cases(quick):
    chunks = [4023] if quick else [512, 4023]
    ndecs = [7] if quick else [7, 8]
    ctxs = [4096] if quick else [4096, 16384]
    out = []
    for chunk in chunks:
        for ndec in ndecs:
            for ctx in ctxs:
                out.append((chunk, ndec, ctx))
    return out


def bench_mixed(chunk, ndec, ctx):
    qls = [chunk] + [1] * ndec
    kvs = [chunk + ctx] + [ctx] * ndec
    kw, ref_kw, output = build_shuffled_full(qls, kvs)
    served = served_check(kw)
    err, cos, ok = correctness(kw, ref_kw, output, served=served)

    def flydsl():
        return ua.unified_attention(**kw, backend="flydsl" if served else None)

    times = {
        backend: time_us(
            flydsl
            if backend == "flydsl"
            else lambda backend=backend: ua.unified_attention(
                **kw, backend=None if backend == "default" else backend
            )
        )
        for backend in ("flydsl", "triton", "gluon", "default")
    }
    return {
        "chunk": chunk,
        "ndec": ndec,
        "ctx": ctx,
        "served": served,
        "err": err,
        "cos": cos,
        "ok": ok,
        "times": times,
        "ratio": times["triton"] / times["flydsl"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--baseline", choices=("triton", "gluon"), default="triton")
    a = p.parse_args()
    print("--baseline is deprecated and ignored; calibration uses Triton.")

    import aiter

    print(f"aiter: {aiter.__file__}")
    print(f"cache={os.environ.get('FLYDSL_RUNTIME_ENABLE_CACHE', '(default on)')}")
    print(
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}  "
        f"device={torch.cuda.get_device_name()}"
    )

    # ---------------- PURE DECODE ----------------
    # C chosen so ceil(C/TILE_SIZE=64) clips MAX_SEGMENTS to exactly the target
    # tier (select_3d_config, else-branch: TILE_SIZE=block_size=64 under
    # shuffled_kv_cache). 16-seg window is C in (960,1024], pick C=1024.
    # 32-seg window is C in (1984,2048], pick C=2048 ("deeper C", per the task).
    b16_candidates = [8, 16, 32, 48, 64] if not a.quick else [16, 32]
    b32_candidates = [4, 8, 16, 24, 32] if not a.quick else [8, 16]

    target16_us = DECODE16_ATTN_US + DECODE16_REDUCE_US
    target32_us = DECODE32_ATTN_US + DECODE32_REDUCE_US

    best16 = calibrate_decode("16-seg tier", 1024, target16_us, b16_candidates)
    best32 = calibrate_decode("32-seg tier", 2048, target32_us, b32_candidates)

    print("\n=== DECODE regime result ===")
    decode_rows = [best16, best32]
    print(f"ceded decode shapes: {sum(not r['served'] for r in decode_rows)}")
    for tag, best, target in (
        ("16-seg", best16, target16_us),
        ("32-seg", best32, target32_us),
    ):
        print(
            f"{tag}: b={best['b']} C={best['C']} segments={best['segments']} "
            f"served={'yes' if best['served'] else 'ceded'} err={best['err']:.3f} cos={best['cos']:.4f} "
            f"correct={'yes' if best['ok'] else 'NO'}  "
            + " ".join(
                f"{backend}_us={us:.1f}" for backend, us in best["times"].items()
            )
            + f"  triton/flydsl={best['ratio']:.2f} "
            f"gluon/flydsl={best['times']['gluon']/best['times']['flydsl']:.2f} "
            f"triton/trace={best['times']['triton']/target:.2f}"
        )

    # ---------------- MIXED BATCH ----------------
    print(
        "\n=== MIXED regime (2d, prefill-chunk + riding decodes; NOT in the "
        "new trace's call counts -- reported as a standalone regime ratio) ==="
    )
    print(
        f"{'chunk':>6} {'ndec':>5} {'ctx':>6} {'fly_us':>8} {'tri_us':>8} "
        f"{'glu_us':>8} {'def_us':>8} {'tri/fly':>8} {'glu/fly':>8} "
        f"{'served':>7} {'cos':>7} {'correct':>8}"
    )
    mixed_rows = []
    for chunk, ndec, ctx in mixed_cases(a.quick):
        r = bench_mixed(chunk, ndec, ctx)
        mixed_rows.append(r)
        print(
            f"{chunk:>6} {ndec:>5} {ctx:>6} "
            + " ".join(f"{us:>8.1f}" for us in r["times"].values())
            + f" {r['ratio']:>8.2f} "
            f"{r['times']['gluon']/r['times']['flydsl']:>8.2f} "
            f"{'yes' if r['served'] else 'ceded':>7} {r['cos']:>7.4f} "
            f"{'yes' if r['ok'] else 'NO':>8}"
        )
    if mixed_rows:
        print(f"ceded mixed shapes: {sum(not r['served'] for r in mixed_rows)}")
        geo = math.exp(sum(math.log(r["ratio"]) for r in mixed_rows) / len(mixed_rows))
        print(f"mixed geomean FlyDSL x = {geo:.2f}")

    # The prefill constants are historical trace projections, not measured
    # alongside this decode run, so keep them separate from measured totals.
    print("\n=== DECODE TOTALS (measured, trace-weighted ms) ===")
    for backend in best16["times"]:
        total = sum(
            best["times"][backend] * launches / 1000.0
            for best, launches in (
                (best16, DECODE16_LAUNCHES),
                (best32, DECODE32_LAUNCHES),
            )
        )
        print(f"{backend}: {total:.1f} ms")

    projected_tri = PREFILL_TRI_MS + TOTAL_DECODE_MS
    projected_fly = (
        PREFILL_FLY_MS
        + DECODE16_BUDGET_MS / best16["ratio"]
        + DECODE32_BUDGET_MS / best32["ratio"]
    )
    print(
        "TOTAL attention projected (legacy method, historical prefill + trace decode ratios): "
        f"triton {projected_tri:.1f} ms  flydsl {projected_fly:.1f} ms  "
        f"saved {projected_tri - projected_fly:.1f} ms "
        f"({100 * (projected_tri - projected_fly) / projected_tri:.1f}%)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

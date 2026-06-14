# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-model decode tuner for the Triton unified-attention 2D / 3D kernels.

Reads an untuned shape table (``aiter/configs/untuned_ua.csv`` by default),
sweeps 2D and 3D launch configs for each (context-bucket, exact-batch) decode
operating point, and writes the winning config to a tuned table
(``aiter/configs/tuned_ua.csv``) in the schema consumed at runtime by
``aiter/ops/triton/utils/ua_config.py``.

A row is only written when the best swept config beats the built-in heuristic by
at least ``--min_improvement_pct`` (default 3%) at the sampled point; otherwise
the heuristic is left in place.

Usage (inside the container with vLLM + AITER installed):

    python3 csrc/unified_attention_tune/unified_attention_tune.py \
        -i aiter/configs/untuned_ua.csv \
        -o aiter/configs/tuned_ua.csv

To tune a single model into its own merged file:

    python3 csrc/unified_attention_tune/unified_attention_tune.py \
        -i my_model_shapes.csv \
        -o aiter/configs/model_configs/mymodel_tuned_ua.csv
"""
import argparse
import csv
import itertools
import os

import torch
import triton

import aiter
import aiter.ops.triton.attention.unified_attention as ua_module
from aiter.ops.triton.attention.unified_attention import (
    select_3d_config as orig_select_3d_config,
    select_2d_config as orig_select_2d_config,
    use_2d_kernel as orig_use_2d_kernel,
    unified_attention,
)
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils import ua_config
from aiter.jit.utils.chip_info import get_gfx_runtime

_INT_MAX = ua_config._INT_MAX

# Default sweep grids. Override via CLI flags.
DEFAULT_SWEEP_3D = {
    "num_segments": [8, 16, 32, 64],
    "num_stages": [1, 2, 3],
    "waves_per_eu": [2, 4],
    "num_warps": [1, 2],
    "tile_size": [16, 32, 64],
}
DEFAULT_SWEEP_2D = {
    "num_stages": [1, 2, 3],
    "waves_per_eu": [2, 4],
    "num_warps": [2, 4],
    "tile_size": [32, 64],
}


def parse_dtype(s):
    s = str(s).strip()
    table = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float8_e4m3fn": aiter.dtypes.fp8,
        "torch.float8_e4m3fnuz": aiter.dtypes.fp8,
        "fp8_e4m3": aiter.dtypes.fp8,  # ua_config's normalized fp8 token
        "fp8": aiter.dtypes.fp8,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if s not in table:
        raise ValueError(f"Unsupported dtype string: {s!r}")
    return table[s]


def sample_ctx(ctx_bucket):
    """Representative context length for a bucket (its upper edge, capped)."""
    return 131072 if int(ctx_bucket) >= _INT_MAX else int(ctx_bucket)


def bench_fn(fn, warmup, iters, trim_pct=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    trim = max(1, int(len(times) * trim_pct / 100))
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    return sum(trimmed) / len(trimmed)


def make_inputs(spec, ctx_len, bs, has_sinks, dev="cuda:0"):
    """Build decode (max_seqlen_q=1) unified_attention inputs."""
    nqh, nkvh, hs, block = (
        spec["num_query_heads"],
        spec["num_kv_heads"],
        spec["head_size"],
        spec["block_size"],
    )
    kv_dtype = spec["kv_dtype"]
    q_dtype = spec["q_dtype"]
    bps = (ctx_len + block - 1) // block
    num_blocks = max(4096, bs * (bps + 1) + 64)

    kf = torch.randn(num_blocks, block, nkvh, hs, device=dev, dtype=torch.bfloat16)
    vf = torch.randn(num_blocks, block, nkvh, hs, device=dev, dtype=torch.bfloat16)
    if kv_dtype != torch.bfloat16:
        kf, vf = kf.to(kv_dtype), vf.to(kv_dtype)
    q = torch.randn(bs, nqh, hs, device=dev, dtype=torch.bfloat16)
    if q_dtype != torch.bfloat16:
        q = q.to(q_dtype)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    sl = torch.full((bs,), ctx_len, dtype=torch.int32, device=dev)
    cu = torch.arange(0, bs + 1, dtype=torch.int32, device=dev)
    bt = torch.arange(bs * bps, device=dev, dtype=torch.int32).reshape(bs, bps)
    bt = torch.cat([bt, torch.zeros(bs, 16, device=dev, dtype=torch.int32)], dim=1)
    ks = torch.tensor([1.0], dtype=torch.float32, device=dev).unsqueeze(0)
    vs = torch.tensor([1.0], dtype=torch.float32, device=dev).unsqueeze(0)
    sinks = torch.randn(nqh, dtype=torch.float32, device=dev) if has_sinks else None
    return dict(
        q=q, k=kf, v=vf, out=out, cu=cu, sl=sl, bt=bt, ks=ks, vs=vs,
        sinks=sinks, q_len=1, hs=hs,
    )


def run_ua(inp, sliding_window):
    inp["out"].zero_()
    window = (-1, -1) if sliding_window <= 0 else (sliding_window - 1, 0)
    unified_attention(
        q=inp["q"], k=inp["k"], v=inp["v"], out=inp["out"], cu_seqlens_q=inp["cu"],
        max_seqlen_q=inp["q_len"], seqused_k=inp["sl"],
        max_seqlen_k=int(inp["sl"].max().item()),
        softmax_scale=1.0 / (inp["hs"] ** 0.5), causal=True,
        window_size=window, block_table=inp["bt"], softcap=0.0,
        q_descale=None, k_descale=inp["ks"], v_descale=inp["vs"],
        sinks=inp["sinks"],
    )


def make_custom_2d(tile, stages, warps, wpe, block_m, nqpkv):
    cfg = {
        "BLOCK_M": block_m,
        "BLOCK_Q": max(1, block_m // nqpkv),
        "TILE_SIZE": tile,
        "num_warps": warps,
        "num_stages": stages,
        "waves_per_eu": wpe,
    }
    return lambda *_a, **_kw: dict(cfg)


def clamp_segments(seg, ctx_len, tile):
    """Same NUM_SEGMENTS clamp select_3d_config applies at runtime."""
    max_seg = min(128, (ctx_len + tile - 1) // tile)
    min_seg = min(8, max_seg)
    return max(min(seg, max_seg), min_seg), min_seg


def make_custom_3d(seg, warps, stages, wpe, tile, reduce_warps):
    attn = {
        "TILE_SIZE": tile,
        "NUM_SEGMENTS_PER_SEQ": seg,
        "num_warps": warps,
        "num_stages": stages,
        "waves_per_eu": wpe,
    }
    reduce = {
        "TILE_SIZE": tile,
        "NUM_SEGMENTS_PER_SEQ": seg,
        "num_warps": reduce_warps,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
    return lambda *_a, **_kw: (dict(attn), dict(reduce))


def _restore_dispatch():
    ua_module.select_2d_config = orig_select_2d_config
    ua_module.select_3d_config = orig_select_3d_config
    ua_module.use_2d_kernel = orig_use_2d_kernel


def tune_row(row, sweep_2d, sweep_3d, warmup, iters, dev):
    spec = {
        "num_query_heads": int(row["num_query_heads"]),
        "num_kv_heads": int(row["num_kv_heads"]),
        "head_size": int(row["head_size"]),
        "block_size": int(row["block_size"]),
        "q_dtype": parse_dtype(row["q_dtype"]),
        "kv_dtype": parse_dtype(row["kv_dtype"]),
    }
    sliding_window = int(float(row["sliding_window"]))
    has_sinks = int(float(row["has_sinks"])) != 0
    ctx = sample_ctx(row["ctx_bucket"])
    bs = int(row["bs"])
    forced_2d = sliding_window > 0 or ctx <= 512

    # fp8 q+kv requires TILE_SIZE >= 32 (matches select_*_config); never sweep
    # below it, so the tuned value is exactly what the runtime applies.
    is_fp8 = spec["q_dtype"] == aiter.dtypes.fp8 and spec["kv_dtype"] == aiter.dtypes.fp8
    min_tile = 32 if is_fp8 else 1
    max_tile = max(64, spec["block_size"])

    nqpkv = spec["num_query_heads"] // spec["num_kv_heads"]
    block_m = 16 if nqpkv <= 16 else triton.next_power_of_2(nqpkv)

    inp = make_inputs(spec, ctx, bs, has_sinks, dev)

    # Baseline: built-in heuristic (table disabled).
    _restore_dispatch()
    base = bench_fn(lambda: run_ua(inp, sliding_window), warmup, iters)

    candidates = []  # (kernel, cfg_dict, time)

    # 2D sweep.
    ua_module.use_2d_kernel = lambda *a, **k: True
    for tile, stages, warps, wpe in itertools.product(
        [t for t in sweep_2d["tile_size"] if min_tile <= t <= max_tile],
        sweep_2d["num_stages"], sweep_2d["num_warps"], sweep_2d["waves_per_eu"],
    ):
        ua_module.select_2d_config = make_custom_2d(
            tile, stages, warps, wpe, block_m, nqpkv
        )
        try:
            t = bench_fn(lambda: run_ua(inp, sliding_window), warmup, iters)
        except Exception:
            t = float("inf")
        candidates.append(
            ("2d", {"TILE_SIZE": tile, "NUM_SEGMENTS": 0, "BLOCK_M": 0,
                    "num_warps": warps, "num_stages": stages, "waves_per_eu": wpe}, t)
        )
    _restore_dispatch()

    # 3D sweep, skipped only when the dispatch forces 2D (SWA / ctx<=512).
    # Segments are clamped to the runtime range so the measured and recorded
    # NUM_SEGMENTS are exactly what the runtime applies.
    if not forced_2d:
        ua_module.use_2d_kernel = lambda *a, **k: False
        seen = set()
        for seg, stages, warps, wpe, tile in itertools.product(
            sweep_3d["num_segments"], sweep_3d["num_stages"],
            sweep_3d["num_warps"], sweep_3d["waves_per_eu"],
            [t for t in sweep_3d["tile_size"] if min_tile <= t <= max_tile],
        ):
            seg_c, min_seg = clamp_segments(seg, ctx, tile)
            combo = (seg_c, stages, warps, wpe, tile)
            if combo in seen:
                continue
            seen.add(combo)
            reduce_warps = 1 if seg_c == min_seg else 2
            ua_module.select_3d_config = make_custom_3d(
                seg_c, warps, stages, wpe, tile, reduce_warps
            )
            try:
                t = bench_fn(lambda: run_ua(inp, sliding_window), warmup, iters)
            except Exception:
                t = float("inf")
            candidates.append(
                ("3d", {"TILE_SIZE": tile, "NUM_SEGMENTS": seg_c, "BLOCK_M": 0,
                        "num_warps": warps, "num_stages": stages, "waves_per_eu": wpe}, t)
            )
        _restore_dispatch()

    if not candidates:
        return "none", None, float("inf"), base, 0.0
    best_kernel, best_cfg, best_time = min(candidates, key=lambda c: c[2])
    speedup = base / best_time if best_time > 0 else 0.0
    return best_kernel, best_cfg, best_time, base, speedup


def load_untuned_shapes(path):
    """Read untuned shapes, dropping duplicates and the stray header rows that
    concurrent (TP) capture can interleave into the file."""
    cols = ua_config.UNTUNED_COLS
    with open(path, newline="") as f:
        raw = list(csv.DictReader(f))
    rows, seen = [], set()
    for r in raw:
        if any(not str(r.get(c, "")).strip() for c in cols):
            continue
        if r.get("num_query_heads") == "num_query_heads":  # duplicated header
            continue
        sig = tuple(r[c] for c in cols)
        if sig in seen:
            continue
        seen.add(sig)
        rows.append(r)
    return rows


def write_tuned(path, rows, out_cols):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for e in rows:
            w.writerow({c: e.get(c, "") for c in out_cols})


def main():
    ap = argparse.ArgumentParser(description="Unified-attention per-model tuner")
    ap.add_argument("-i", "--untune_file", default="aiter/configs/untuned_ua.csv")
    ap.add_argument("-o", "--tune_file", default="aiter/configs/tuned_ua.csv")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--min_improvement_pct", type=float, default=3.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    gfx = get_gfx_runtime()
    cu_num = get_num_sms()
    print(f"Tuning UA on {gfx} ({cu_num} CUs)")

    # Disable the runtime table during tuning so that forced-kernel sweeps are
    # not short-circuited by an existing tuned row for the same shape.
    ua_config.get_ua_config = lambda **kwargs: None

    rows = load_untuned_shapes(args.untune_file)
    out_cols = list(ua_config.KEY_COLS + ua_config.RESULT_COLS + ua_config.META_COLS)
    key_cols = list(ua_config.KEY_COLS)

    existing = []
    if os.path.exists(args.tune_file):
        with open(args.tune_file, newline="") as f:
            existing = list(csv.DictReader(f))

    written = 0
    for i, row in enumerate(rows, 1):
        try:
            kernel, cfg, t, base, sp = tune_row(
                row, DEFAULT_SWEEP_2D, DEFAULT_SWEEP_3D,
                args.warmup, args.iters, args.device,
            )
        except Exception as e:  # one bad shape must not abort the whole run
            print(f"[{i}/{len(rows)}] FAILED {dict(row)}: {e}")
            continue
        keep = cfg is not None and sp >= 1.0 + args.min_improvement_pct / 100.0
        print(
            f"[{i}/{len(rows)}] ctx<={row['ctx_bucket']} bs={row['bs']}: "
            f"best={kernel} {cfg} {t*1000:.1f}us vs default {base*1000:.1f}us "
            f"({sp:.2f}x) {'KEEP' if keep else 'skip'}"
        )
        if not keep:
            continue
        out_row = {
            "gfx": gfx, "cu_num": cu_num,
            "num_query_heads": row["num_query_heads"],
            "num_kv_heads": row["num_kv_heads"],
            "head_size": row["head_size"], "block_size": row["block_size"],
            "q_dtype": row["q_dtype"], "kv_dtype": row["kv_dtype"],
            "sliding_window": int(float(row["sliding_window"])),
            "has_sinks": int(float(row["has_sinks"])),
            "phase": row["phase"], "ctx_bucket": row["ctx_bucket"],
            "bs": row["bs"], "kernel": kernel,
            "TILE_SIZE": cfg["TILE_SIZE"], "NUM_SEGMENTS": cfg["NUM_SEGMENTS"],
            "BLOCK_M": cfg["BLOCK_M"], "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"], "waves_per_eu": cfg["waves_per_eu"],
            "us": round(t * 1000, 4), "errRatio": 0.0, "_tag": "",
        }
        existing = [
            e for e in existing
            if not all(str(e.get(c)) == str(out_row.get(c)) for c in key_cols)
        ]
        existing.append(out_row)
        written += 1
        # Persist after every kept shape so a later failure can't lose work.
        write_tuned(args.tune_file, existing, out_cols)

    write_tuned(args.tune_file, existing, out_cols)
    print(f"Wrote {written} tuned rows to {args.tune_file}")
    ua_config.clear_cache()


if __name__ == "__main__":
    main()

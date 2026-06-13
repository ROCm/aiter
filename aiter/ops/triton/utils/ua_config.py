# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-model launch-config table for the Triton unified-attention kernels.

The attention analogue of the FMOE/GEMM tuned-config tables under
``aiter/configs/``. A row maps an attention-shape signature plus an operating
point (phase, coarse context-length bucket, coarse batch bucket) to a kernel
choice (2d/3d) and its launch meta-params. Lookup runs per call; a miss keeps
the built-in heuristics in ``unified_attention.py``.

Buckets are coarse to bound the number of distinct Triton compilations. On a
miss for the exact bucket the lookup falls *down* to a shorter-context bucket
first (a config tuned for a shorter context degrades gracefully on a longer
one; the reverse can be worse than the heuristic), then down the batch bucket,
and up only as a last resort. ``NUM_SEGMENTS`` is clamped to the valid range for
the live sequence length by the caller.
"""
import logging
import math
import os
import functools

logger = logging.getLogger("aiter")
_LOG_TUNED_CONFIG = int(os.getenv("AITER_LOG_TUNED_CONFIG", "0"))
# Append the live shape/bucket of every lookup miss to an untuned-shape CSV the
# tuner can consume (mirrors FMOE's online-tune capture). "1" -> a file in the
# cwd; any other value is treated as an explicit path. Unset disables capture.
_DUMP_UNTUNED = os.getenv("AITER_UA_DUMP_UNTUNED", "")

_INT_MAX = 2**31 - 1

# Context-length (max_seqlen_k) bucket upper bounds. The first edge is the
# dispatch boundary (max_seqlen_k <= 512 always uses 2D), so a bucket never
# straddles the 2D/3D split.
CTX_BUCKETS = (512, 2048, 8192, 32768, 65536, 131072, _INT_MAX)
# Batch (num_seqs) bucket upper bounds.
BS_BUCKETS = (4, 16, 64, 256, _INT_MAX)

KEY_COLS = (
    "gfx",
    "cu_num",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "block_size",
    "q_dtype",
    "kv_dtype",
    "sliding_window",
    "has_sinks",
    "phase",
    "ctx_bucket",
    "bs_bucket",
)
RESULT_COLS = (
    "kernel",
    "TILE_SIZE",
    "NUM_SEGMENTS",
    "BLOCK_M",
    "num_warps",
    "num_stages",
    "waves_per_eu",
)
META_COLS = ("us", "errRatio", "_tag")

# Untuned-shape columns (no gfx/cu_num; the tuner fills those from the device).
UNTUNED_COLS = KEY_COLS[2:]

# fp8 e4m3 is fn on gfx950/gfx12 and fnuz on gfx942; both are 1-byte e4m3 and
# pick the same launch config (gfx is already in the key), so fold them to one
# token. This keeps a CSV portable across arches and avoids a silent miss when
# str(runtime_dtype) differs from the authored string.
_DTYPE_ALIASES = {
    "torch.float8_e4m3fn": "fp8_e4m3",
    "torch.float8_e4m3fnuz": "fp8_e4m3",
    "torch.float8_e5m2": "fp8_e5m2",
    "torch.float8_e5m2fnuz": "fp8_e5m2",
}


def _norm_dtype(dtype):
    s = str(dtype).strip()
    return _DTYPE_ALIASES.get(s, s)

# Set membership keeps logging and shape-capture to one line per unique shape.
_logged = set()
_dumped = set()


def _bypass_active():
    """Global escape hatch shared with FMOE/GEMM: force the built-in heuristics."""
    return os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0").strip().lower() not in (
        "",
        "0",
        "false",
    )


def bucket_value(value, boundaries):
    """Smallest boundary >= value (the bucket the value falls into)."""
    for b in boundaries:
        if value <= b:
            return b
    return boundaries[-1]


def _shape_key(
    gfx,
    cu_num,
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    q_dtype,
    kv_dtype,
    sliding_window,
    has_sinks,
    phase,
):
    """Normalized shape signature shared by table indexing and lookup."""
    return (
        str(gfx).strip(),
        int(cu_num),
        int(num_query_heads),
        int(num_kv_heads),
        int(head_size),
        int(block_size),
        _norm_dtype(q_dtype),
        _norm_dtype(kv_dtype),
        int(sliding_window),
        int(float(has_sinks) != 0),
        str(phase).strip().lower(),
    )


def _coerce_int(value, allow_zero=False):
    """``int(value)`` or ``None`` for missing/NaN (and non-positive unless allowed)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or (f < 0) or (f == 0 and not allow_zero):
        return None
    return int(f)


def _parse_row(raw):
    kernel = str(raw.get("kernel", "")).strip().lower()
    if kernel not in ("2d", "3d"):
        return None
    return {
        "kernel": kernel,
        "TILE_SIZE": _coerce_int(raw.get("TILE_SIZE")),
        "NUM_SEGMENTS": _coerce_int(raw.get("NUM_SEGMENTS")),
        "BLOCK_M": _coerce_int(raw.get("BLOCK_M")),
        "num_warps": _coerce_int(raw.get("num_warps")),
        "num_stages": _coerce_int(raw.get("num_stages")),
        "waves_per_eu": _coerce_int(raw.get("waves_per_eu"), allow_zero=True),
    }


def _resolve_config_path():
    """Resolve the merged tuned-UA CSV path via jit-core (same as FMOE/GEMM),
    falling back to the env var / packaged default for standalone contexts."""
    try:
        from aiter.jit.core import AITER_CONFIGS

        return AITER_CONFIGS.AITER_CONFIG_UA_FILE
    except Exception:
        env = os.environ.get("AITER_CONFIG_UA")
        if env:
            return env.split(os.pathsep)[0]
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(
            os.path.join(here, "..", "..", "..", "configs", "tuned_ua.csv")
        )


@functools.lru_cache(maxsize=4)
def _load_table(path):
    """Index the tuned-UA CSV as ``{key_tuple: parsed_row}`` (empty if absent)."""
    if not path or not os.path.exists(path):
        return {}
    import pandas as pd

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or any(c not in df.columns for c in KEY_COLS):
        return {}
    table = {}
    for raw in df.to_dict("records"):
        parsed = _parse_row(raw)
        if parsed is None:
            continue
        try:
            key = _shape_key(*(raw[c] for c in KEY_COLS[:-2])) + (
                int(raw["ctx_bucket"]),
                int(raw["bs_bucket"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
        # First write wins; upstream merge/dedup already kept the best row.
        table.setdefault(key, parsed)
    return table


def _bucket_order(target, boundaries):
    """Buckets to try in priority order: exact, then down, then up."""
    return [b for b in boundaries if b <= target][::-1] + [
        b for b in boundaries if b > target
    ]


def clear_cache():
    """Drop the cached table (call after the tuned CSV is regenerated)."""
    _load_table.cache_clear()
    _logged.clear()
    _dumped.clear()


def _untuned_dump_path():
    if _DUMP_UNTUNED.strip().lower() in ("1", "true"):
        return os.path.join(os.getcwd(), "untuned_ua_captured.csv")
    return _DUMP_UNTUNED


def _dump_untuned(fixed, ctx_b, bs_b):
    """Append this shape/bucket to the untuned CSV once per process (best effort)."""
    row = fixed[2:] + (ctx_b, bs_b)  # drop gfx, cu_num -> tuner re-derives them
    if row in _dumped:
        return
    _dumped.add(row)
    path = _untuned_dump_path()
    try:
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a") as f:
            if new_file:
                f.write(",".join(UNTUNED_COLS) + "\n")
            f.write(",".join(str(v) for v in row) + "\n")
    except OSError as e:
        logger.warning("unified_attention: cannot write untuned shapes to %s: %s", path, e)


def get_ua_config(
    *,
    gfx,
    cu_num,
    num_query_heads,
    num_kv_heads,
    head_size,
    block_size,
    q_dtype,
    kv_dtype,
    sliding_window,
    has_sinks,
    phase,
    max_seqlen_k,
    num_seqs,
):
    """Tuned UA config for this shape/operating-point, or ``None`` for default."""
    # AITER_BYPASS_TUNE_CONFIG forces the heuristics (same switch as FMOE/GEMM):
    # behave as if no table exists, but still allow shape capture.
    # Fast path: no tuned table and not capturing (the common case) costs one
    # cached lookup. With capture on we still need the key to record the miss.
    table = {} if _bypass_active() else _load_table(_resolve_config_path())
    if not table and not _DUMP_UNTUNED:
        if _LOG_TUNED_CONFIG:
            _log_once(("no_table",), logging.INFO, "unified_attention: no tuned UA table")
        return None

    fixed = _shape_key(
        gfx,
        cu_num,
        num_query_heads,
        num_kv_heads,
        head_size,
        block_size,
        q_dtype,
        kv_dtype,
        sliding_window,
        has_sinks,
        phase,
    )
    ctx_b = bucket_value(int(max_seqlen_k), CTX_BUCKETS)
    bs_b = bucket_value(int(num_seqs), BS_BUCKETS)

    # Context is the dominant axis (it decides 2D vs 3D and the kernel regime),
    # so match the context bucket first, then the batch bucket.
    for ctx in _bucket_order(ctx_b, CTX_BUCKETS):
        for bs in _bucket_order(bs_b, BS_BUCKETS):
            row = table.get(fixed + (ctx, bs))
            if row is not None:
                if _LOG_TUNED_CONFIG:
                    _log_once(
                        ("hit",) + fixed + (ctx_b, bs_b),
                        logging.INFO,
                        "unified_attention: %s config for %s ctx<=%d bs<=%d",
                        row["kernel"],
                        fixed,
                        ctx,
                        bs,
                    )
                return row

    if _DUMP_UNTUNED:
        _dump_untuned(fixed, ctx_b, bs_b)
    if table:
        # Table has tuning, but not for this shape/bucket: a real, actionable gap.
        _log_once(
            ("miss",) + fixed + (ctx_b, bs_b),
            logging.WARNING,
            "unified_attention: no tuned config for %s ctx<=%d bs<=%d; "
            "using default heuristic",
            fixed,
            ctx_b,
            bs_b,
        )
    elif _LOG_TUNED_CONFIG:
        _log_once(("no_table",), logging.INFO, "unified_attention: no tuned UA table")
    return None


def _log_once(key, level, msg, *args):
    if key in _logged:
        return
    _logged.add(key)
    logger.log(level, msg, *args)

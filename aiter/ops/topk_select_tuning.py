# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline table, presets, and recorder for `topk_select`.

The runtime looks up a backend by (gfx, cu_num, k, dtype, ragged, tie,
deterministic) and a half-octave band on rows and width. Misses, promise
violations, backends that cannot serve the shape, and a tuned backend that
raised at dispatch all fall through to the shape-only router.

Recording is opt-in via ``AITER_TOPK_SELECT_RECORD`` and is skipped while a
CUDA graph is being captured.
"""

from __future__ import annotations

import csv
import fcntl
import math
import os
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIGS, AITER_LOG_TUNED_CONFIG

PRESETS = (
    "randn",
    "uniform16",
    "bf16_randn",
    "lognormal",
    "recency",
    "equal",
    "adversarial",
    "inf",
)

TUNED_COLUMNS = (
    "gfx",
    "cu_num",
    "rows_lo",
    "rows_hi",
    "width_lo",
    "width_hi",
    "k",
    "dtype",
    "ragged",
    "tie",
    "deterministic",
    "mode",
    "backend",
    "us",
    "default_backend",
    "us_default",
    "worst_ratio_vs_default",
    "dists",
    "n_samples",
    "aiter_rev",
)

UNTUNED_COLUMNS = (
    "rows",
    "width",
    "k",
    "dtype",
    "ragged",
    "tie",
    "deterministic",
    "dist",
)

_OVERRIDE_ROWS: list[dict] | None = None
_RECORD_WARNED = False
_RECORD_OFF = False
_ENV_WARNED: set[str] = set()
_RECORD_LOCK = threading.Lock()
_RECORD_COUNTS: dict[tuple, int] = {}
_AITER_REV_WARNED = False
_TABLE_ERR_WARNED = False
_BAD_BACKEND_WARNED: set[str] = set()
# (rows, width, k, ragged, tie, deterministic, dtype, backend): a tuned backend
# that raised at dispatch on that shape, so no table row sends it there again
# in this process.
_RETIRED: set[tuple] = set()

DTYPES = ("float32", "bfloat16", "float16")
_DTYPE_ALIASES = {
    "fp32": "float32",
    "float": "float32",
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
}


def normalize_dtype(v) -> str:
    """Canonical dtype name; empty means float32, the dtype tables predate."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "float32"
    s = str(v).strip().lower().removeprefix("torch.")
    if not s:
        return "float32"
    s = _DTYPE_ALIASES.get(s, s)
    if s not in DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(DTYPES)}; got {v!r}")
    return s


def _band_edge(e: int) -> int:
    """``ceil(2 ** (e / 2))``, exactly."""
    x = 1 << e
    r = math.isqrt(x)
    return r if r * r == x else r + 1


def half_octave_bounds(n: int) -> tuple[int, int]:
    """Inclusive [lo, hi] half-octave band that contains ``n``.

    Edges sit on ``ceil(2**(e/2))``, so 4096 covers 4096..5792 and 512 covers
    512..724. They are computed in integers: rounding the irrational edges
    leaves values such as 11, 362, 2896 and 11585 outside the band computed
    for them. A table cell is a region along every axis it is indexed by.
    """
    n = int(n)
    if n <= 1:
        return 1, 1
    e = int(math.log2(n) * 2)
    while _band_edge(e + 1) <= n:
        e += 1
    while _band_edge(e) > n:
        e -= 1
    return _band_edge(e), _band_edge(e + 1) - 1


def normalize_tie(tie) -> str | None:
    if tie is None:
        return None
    if isinstance(tie, float) and math.isnan(tie):
        return None
    s = str(tie).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s


def normalize_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", "", "nan", "none"):
        return False
    raise ValueError(f"not a bool: {v!r}")


def gen_preset(dist: str, m: int, n: int, seed: int, device) -> torch.Tensor:
    """Build one of the eight synthetic distributions from the dist-flip grid."""
    if dist not in PRESETS:
        raise ValueError(
            f"unknown preset {dist!r}; choose from {PRESETS} or a .pt path"
        )
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    if dist == "randn":
        return torch.randn(m, n, generator=g, device=device)
    if dist == "uniform16":
        q = torch.randint(0, 65536, (m, n), generator=g, device=device)
        return (q - 32768).float() / 32768.0
    if dist == "bf16_randn":
        return torch.randn(m, n, generator=g, device=device).bfloat16().float()
    if dist == "lognormal":
        return torch.exp(2.0 * torch.randn(m, n, generator=g, device=device))
    if dist == "recency":
        col = torch.arange(n, device=device, dtype=torch.float32) / n
        return 3.0 * col + 0.5 * torch.randn(m, n, generator=g, device=device)
    i = torch.arange(m * n, device=device, dtype=torch.int64).view(m, n) + int(seed)
    low = (i & 0xFFFF).float() * 1e-6
    if dist == "equal":
        return torch.ones(m, n, device=device)
    if dist == "adversarial":
        col = torch.arange(n, device=device).view(1, n)
        return torch.where(col >= n - 3000, 100.0 + (i & 0xF).float(), low)
    return torch.where((i & 0xFF) == 0, torch.full_like(low, float("inf")), low)


def expand_dist_field(dist: str) -> list[str]:
    d = (dist or "").strip()
    if d == "" or d.lower() == "all":
        return list(PRESETS)
    return [p.strip() for p in d.split(";") if p.strip()]


def is_sample_path(dist: str) -> bool:
    s = dist.strip()
    return s.endswith((".pt", ".pth")) or os.path.sep in s or s.startswith(".")


@lru_cache(maxsize=1)
def aiter_rev() -> str:
    """Short git rev of the aiter tree, or "unknown". A subprocess -- call once."""
    try:
        import subprocess

        from aiter.jit.core import AITER_ROOT_DIR

        out = subprocess.check_output(
            ["git", "-C", AITER_ROOT_DIR, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _tuned_csv_path() -> str | None:
    """The table `AITER_CONFIGS` resolves, as for every tuned aiter op.

    A failed merge (two tables claiming one band) falls back to the router
    with a warning rather than failing the `topk_select` call.
    """
    global _TABLE_ERR_WARNED
    try:
        return AITER_CONFIGS.AITER_CONFIG_TOPK_SELECT_FILE
    except Exception as e:  # noqa: BLE001
        if not _TABLE_ERR_WARNED:
            logger.warning(
                "topk_select tuned table not loaded (%s: %s); using the shape router",
                type(e).__name__,
                e,
            )
            _TABLE_ERR_WARNED = True
        return None


def _parse_tuned_row(raw: dict) -> dict | None:
    try:
        backend = str(raw.get("backend", "")).strip()
        if not backend:
            return None
        return {
            "gfx": str(raw.get("gfx", "")).strip(),
            "cu_num": int(float(raw["cu_num"])),
            "rows_lo": int(float(raw["rows_lo"])),
            "rows_hi": int(float(raw["rows_hi"])),
            "width_lo": int(float(raw["width_lo"])),
            "width_hi": int(float(raw["width_hi"])),
            "k": int(float(raw["k"])),
            "dtype": normalize_dtype(raw.get("dtype")),
            "ragged": normalize_bool(raw.get("ragged", False)),
            "tie": normalize_tie(raw.get("tie")),
            "deterministic": normalize_bool(raw.get("deterministic", False)),
            "mode": str(raw.get("mode", "graph") or "graph").strip() or "graph",
            "backend": backend,
            "aiter_rev": str(raw.get("aiter_rev", "") or "").strip(),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _load_tuned_rows_from_path(path: str) -> tuple[dict, ...]:
    if not path:
        return ()
    if not os.path.exists(path):
        logger.warning("topk_select tuned table %s does not exist; ignoring it", path)
        return ()
    from aiter.ops.topk_select import _BACKENDS_BY_TIE

    known = _BACKENDS_BY_TIE[None]
    rows, bad, unknown = [], 0, set()
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            # Hand-edited tables carry spaces after the commas.
            raw = {
                str(k).strip(): v.strip() if isinstance(v, str) else v
                for k, v in raw.items()
                if k is not None
            }
            parsed = _parse_tuned_row(raw)
            if parsed is None:
                bad += 1
            elif parsed["backend"] not in known:
                unknown.add(parsed["backend"])
            else:
                parsed["src"] = path
                rows.append(parsed)
    if bad:
        logger.warning(
            "topk_select tuned table %s: ignored %d malformed row(s)", path, bad
        )
    if unknown:
        logger.warning(
            "topk_select tuned table %s: ignored rows naming unknown backend(s) %s; "
            "known: %s",
            path,
            ", ".join(sorted(unknown)),
            ", ".join(known),
        )
    return tuple(rows)


@lru_cache(maxsize=8)
def _cached_tuned_rows(path: str | None) -> tuple[dict, ...]:
    rows = _load_tuned_rows_from_path(path) if path else ()
    if rows:
        logger.info("topk_select: %d tuned row(s) loaded from %s", len(rows), path)
    return rows


def reload_tuned_table() -> None:
    """Drop the CSV cache. Tests and the tuner call this after writing a table."""
    global _AITER_REV_WARNED, _TABLE_ERR_WARNED
    _cached_tuned_rows.cache_clear()
    AITER_CONFIGS.get_config_file.cache_clear()
    _AITER_REV_WARNED = False
    _TABLE_ERR_WARNED = False


@contextmanager
def override_tuned_rows(rows: list[dict] | None):
    """Process-local table, used by the tuner to time one candidate through the entry."""
    global _OVERRIDE_ROWS
    prev = _OVERRIDE_ROWS
    _OVERRIDE_ROWS = list(rows) if rows is not None else None
    try:
        yield
    finally:
        _OVERRIDE_ROWS = prev


def current_tuned_rows() -> tuple[dict, ...]:
    if _OVERRIDE_ROWS is not None:
        return tuple(_OVERRIDE_ROWS)
    return _cached_tuned_rows(_tuned_csv_path())


def lookup_tuned(
    rows: int,
    width: int,
    k: int,
    ragged: bool,
    tie: str | None,
    deterministic: bool,
    gfx: str | None = None,
    cu_num: int | None = None,
    mode: str | None = None,
    dtype: str = "float32",
    device: int | None = None,
) -> str | None:
    """Name the tuned backend for this call on ``device``, or None to keep the
    shape router."""
    table = current_tuned_rows()
    if not table:
        return None
    tie_n = normalize_tie(tie)
    if gfx is None or cu_num is None:
        gfx, cu_num = _runtime_chip(device)
    want_mode = mode or os.getenv("AITER_TOPK_SELECT_TUNE_MODE", "graph")
    hits = []
    for rec in table:
        if rec["k"] != k or rec["dtype"] != dtype:
            continue
        if rec["ragged"] != bool(ragged):
            continue
        if rec["tie"] != tie_n:
            continue
        if rec["deterministic"] != bool(deterministic):
            continue
        if rec["gfx"] and gfx and rec["gfx"] != gfx:
            continue
        if rec["cu_num"] and cu_num and rec["cu_num"] != cu_num:
            continue
        if not (rec["rows_lo"] <= rows <= rec["rows_hi"]):
            continue
        if not (rec["width_lo"] <= width <= rec["width_hi"]):
            continue
        hits.append(rec)
    if not hits:
        return None
    prefer = [r for r in hits if r["mode"] == want_mode] or hits
    prefer.sort(
        key=lambda r: (
            (r["rows_hi"] - r["rows_lo"]) + (r["width_hi"] - r["width_lo"]),
            r["rows_hi"] - r["rows_lo"],
        )
    )
    rec = prefer[0]
    shape = (rows, width, k, bool(ragged), tie_n, bool(deterministic), dtype)
    if (*shape, rec["backend"]) in _RETIRED:
        return None
    global _AITER_REV_WARNED
    if not _AITER_REV_WARNED and rec["aiter_rev"] not in ("", "unknown"):
        live = aiter_rev()
        if live != "unknown" and rec["aiter_rev"] != live:
            logger.warning(
                "topk_select tuned table aiter_rev=%s does not match live %s; "
                "using the table anyway",
                rec["aiter_rev"],
                live,
            )
            _AITER_REV_WARNED = True
    if AITER_LOG_TUNED_CONFIG:
        logger.info(
            f"topk_select shape is rows:{rows}, width:{width}, k:{k}, dtype:{dtype}, "
            f"ragged:{bool(ragged)}, tie:{tie_n}, deterministic:{bool(deterministic)}, "
            f"is tuned on gfx = {gfx}, cu_num = {cu_num} in "
            f"{rec.get('src', 'override')} (band rows {rec['rows_lo']}-"
            f"{rec['rows_hi']} x width {rec['width_lo']}-{rec['width_hi']}, "
            f"mode = {rec['mode']}), backend is {rec['backend']}!"
        )
    return rec["backend"]


@lru_cache(maxsize=1)
def _runtime_chip(device: int | None = None) -> tuple[str, int]:
    """``(gfx, cu_num)`` of one GPU, the current one by default: a process can
    drive different GPUs side by side, as `topk_select`'s router allows for."""
    try:
        from aiter.ops.topk import _device_arch

        dev = torch.cuda.current_device() if device is None else device
        props = torch.cuda.get_device_properties(dev)
        return _device_arch(dev), int(props.multi_processor_count)
    except Exception:  # noqa: BLE001
        return "", 0


def retire_tuned(
    backend, router, error, rows, width, k, ragged, tie, deterministic, dtype
) -> bool:
    """A tuned backend raised at dispatch: stop sending this shape to it.

    Returns False, retiring nothing, while the tuner times a candidate
    (`override_tuned_rows`): the caller re-raises, so the failure is charged to
    the candidate instead of being hidden behind the router's answer.
    """
    if _OVERRIDE_ROWS is not None:
        return False
    tie_n = normalize_tie(tie)
    _RETIRED.add(
        (rows, width, k, bool(ragged), tie_n, bool(deterministic), dtype, backend)
    )
    logger.warning(
        "topk_select tuned backend %r raised %s: %s at rows=%d width=%d k=%d; "
        "this call and later ones of this shape use the shape router's %r",
        backend,
        type(error).__name__,
        error,
        rows,
        width,
        k,
        router,
    )
    return True


def warn_unavailable_tuned(backend: str, rows: int, width: int, k: int) -> None:
    key = f"{backend}:{rows}:{width}:{k}"
    if key in _BAD_BACKEND_WARNED:
        return
    _BAD_BACKEND_WARNED.add(key)
    logger.warning(
        "topk_select tuned backend %r cannot serve rows=%s width=%s k=%s; "
        "falling back to the shape router",
        backend,
        rows,
        width,
        k,
    )


def load_sample_file(path: str, device) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or "input" not in blob:
        raise ValueError(f"{path} is not a topk_select sample dict")
    x = blob["input"].to(device)
    row_lens = blob["row_lens"].to(device=device, dtype=torch.int32)
    return {
        "input": x,
        "row_lens": row_lens,
        "rows": int(blob.get("rows", x.shape[0])),
        "width": int(blob.get("width", x.shape[1])),
        "k": int(blob["k"]) if "k" in blob else None,
        "ragged": bool(blob.get("ragged", False)),
        "tie": normalize_tie(blob.get("tie")),
        "deterministic": normalize_bool(blob.get("deterministic", False)),
        "path": path,
    }


def tile_sample_to_rows(
    sample: dict, rows: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat recorded rows until the call's row count is met."""
    src = sample["input"]
    lens = sample["row_lens"]
    nsrc = src.shape[0]
    if nsrc == 0:
        raise ValueError("empty sample")
    reps = (rows + nsrc - 1) // nsrc
    x = src.repeat(reps, 1)[:rows].contiguous().to(device)
    row_lens = lens.repeat(reps)[:rows].contiguous().to(device)
    return x, row_lens


def oracle_check(
    idx: torch.Tensor, x: torch.Tensor, row_lens: torch.Tensor, k: int
) -> str:
    """Return 'ok' or a short tag. Live slots must match torch.topk of the clipped row."""
    if idx.dtype != torch.int32:
        return "index_dtype"
    m, kk = idx.shape
    if kk != k or m != x.shape[0]:
        return "shape"
    width = x.shape[1]
    i64 = idx.long()
    lens = row_lens.long().view(m, 1).clamp(max=width)
    if bool((i64 >= width).any()):
        return "index_out_of_range"
    if bool(((i64 < 0) & (i64 != -1)).any()):
        return "index_out_of_range"
    live = i64 >= 0
    if bool((live & (i64 >= lens)).any()):
        return "index_past_end"
    s = torch.where(live, i64, -1).sort(dim=1).values
    if bool(((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any()):
        return "duplicate_index"
    nlive = live.sum(dim=1)
    expect = lens.view(m).clamp(max=k)
    bad = (nlive != expect).nonzero()
    if bad.numel():
        r = int(bad[0, 0])
        return f"live_count={int(nlive[r])} expected={int(expect[r])}"
    neg = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    step = max(1, (1 << 26) // width)
    for r0 in range(0, m, step):
        xs = x[r0 : r0 + step]
        if bool(torch.isnan(xs).any()):
            return "input contains NaN (top-k over NaN is undefined)"
        col = torch.arange(width, device=x.device).view(1, width)
        masked = torch.where(col < lens[r0 : r0 + step], xs, neg)
        ref = torch.topk(masked, k, dim=1, sorted=True).values
        li, lv = i64[r0 : r0 + step], live[r0 : r0 + step]
        got = torch.where(lv, xs.gather(1, li.clamp(min=0)), neg)
        if not torch.equal(got.sort(dim=1, descending=True).values, ref):
            return "value_mismatch"
    return "ok"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        value = int(raw) if raw.strip() else default
    except ValueError:
        value = 0
    if value < 1:
        if name not in _ENV_WARNED:
            _ENV_WARNED.add(name)
            logger.warning(
                "%s=%r is not a positive integer; using %d", name, raw, default
            )
        return default
    return value


def maybe_record(input, row_lens, k, ragged, tie, deterministic) -> None:
    """Store a subsample of this call if recording is on. No-op in graph capture.

    Never raises: a recording that fails (unwritable directory, full disk) is
    switched off for the process with one warning, and the call goes on.
    """
    global _RECORD_OFF
    dest = os.getenv("AITER_TOPK_SELECT_RECORD")
    if not dest or _RECORD_OFF:
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            return
    except Exception:  # noqa: BLE001,S110
        pass
    try:
        _record(dest, input, row_lens, k, ragged, tie, deterministic)
    except Exception as e:  # noqa: BLE001
        _RECORD_OFF = True
        logger.warning(
            "topk_select recording to %s failed (%s: %s); recording is off for "
            "this process, topk_select itself is unaffected",
            dest,
            type(e).__name__,
            e,
        )


def _record(dest, input, row_lens, k, ragged, tie, deterministic) -> None:
    global _RECORD_WARNED
    if not _RECORD_WARNED:
        logger.warning(
            "AITER_TOPK_SELECT_RECORD=%s is on: each recorded call copies logits "
            "to the host. Use this only during eager calibration, never while "
            "serving or capturing a graph.",
            dest,
        )
        _RECORD_WARNED = True
    max_calls = _env_int("AITER_TOPK_SELECT_RECORD_CALLS", 4)
    max_rows = _env_int("AITER_TOPK_SELECT_RECORD_ROWS", 64)
    rows, width = int(input.shape[0]), int(input.shape[1])
    dtype = normalize_dtype(input.dtype)
    # Capped per table band, not per exact shape: prefill varies `rows` on every
    # call, and the tuner merges a band's samples into one decision anyway.
    key = (
        half_octave_bounds(rows),
        half_octave_bounds(width),
        int(k),
        dtype,
        bool(ragged),
        normalize_tie(tie),
        bool(deterministic),
    )
    with _RECORD_LOCK:
        n = _RECORD_COUNTS.get(key, 0)
        if n >= max_calls:
            return
        _RECORD_COUNTS[key] = n + 1
        call_i = n
    take = min(max_rows, rows)
    if take < rows:
        idx = torch.linspace(0, rows - 1, take, device=input.device).long()
        x = input.index_select(0, idx).detach().cpu()
        lens = row_lens.index_select(0, idx).detach().cpu()
    else:
        x = input.detach().cpu()
        lens = row_lens.detach().cpu()
    torch.cuda.synchronize()
    out_dir = Path(dest)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # The pid keeps ranks of one job that record into one directory apart.
    fname = (
        f"k{int(k)}_w{width}_r{rows}_{dtype}_rag{int(ragged)}_"
        f"t{normalize_tie(tie) or 'none'}_d{int(bool(deterministic))}_"
        f"p{os.getpid()}_{call_i}.pt"
    )
    path = sample_dir / fname
    torch.save(
        {
            "input": x,
            "row_lens": lens,
            "rows": rows,
            "width": width,
            "k": int(k),
            "ragged": bool(ragged),
            "tie": normalize_tie(tie),
            "deterministic": bool(deterministic),
        },
        path,
    )
    csv_path = out_dir / "topk_untuned.csv"
    row = {
        "rows": rows,
        "width": width,
        "k": int(k),
        "dtype": dtype,
        "ragged": bool(ragged),
        "tie": normalize_tie(tie) or "",
        "deterministic": bool(deterministic),
        "dist": f"samples/{fname}",
    }
    with _RECORD_LOCK, open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=UNTUNED_COLUMNS)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)

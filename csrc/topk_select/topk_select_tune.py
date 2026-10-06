# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline tuner for aiter.ops.topk_select.

See csrc/topk_select/README.md.
"""

from __future__ import annotations

import copy
import math
import os
import random
import re
import sys
import time
import traceback
import zlib
from typing import Any, ClassVar

import pandas as pd
import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIG_TOPK_SELECT, AITER_ROOT_DIR
from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of
from aiter.ops.topk_select import (
    _BACKENDS_BY_TIE,
    _allowed,
    _choose,
    _router_choice,
    _sampled_ok,
    _servable,
    topk_select,
)
from aiter.ops.topk_select_tuning import (
    PRESETS,
    TUNED_COLUMNS,
    aiter_rev,
    expand_dist_field,
    gen_preset,
    half_octave_bounds,
    is_sample_path,
    load_sample_file,
    normalize_bool,
    normalize_dtype,
    normalize_tie,
    oracle_check,
    override_tuned_rows,
    reload_tuned_table,
    tile_sample_to_rows,
)
from aiter.utility.base_tuner import TunerCommon

LOOKUP_KEYS = [
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
]
RESULT_COLS = [
    "backend",
    "us",
    "default_backend",
    "us_default",
    "worst_ratio_vs_default",
    "dists",
    "n_samples",
    "aiter_rev",
]

ROT_BYTES = 256 << 20
CHUNK_BYTES = 8 << 30
DRAWS = 16
DRAW_BYTES = 4 << 30
MAX_DRAWS = 128
MAX_COPIES = 32
TARGET_MS = 3.0
NO_REGRESS = 0.02
FAULT_NAME = "__fault__"
_SPEC_RE = re.compile(r"^(?P<dist>.+)@(?P<rows>\d+)x(?P<width>\d+)$")


def _geomean(xs):
    xs = [float(x) for x in xs if x is not None and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# ---------------------------------------------------------------- samples / keys


def format_spec(dist: str, rows: int, width: int) -> str:
    return f"{dist}@{int(rows)}x{int(width)}"


def parse_spec(spec: str) -> tuple[str, int, int] | None:
    m = _SPEC_RE.match(str(spec).strip())
    if not m:
        return None
    return m["dist"], int(m["rows"]), int(m["width"])


def short_spec(spec: str) -> str:
    p = parse_spec(spec)
    if p is None:
        return str(spec)
    dist, rows, width = p
    name = os.path.basename(dist) if is_sample_path(dist) else dist
    return format_spec(name, rows, width)


def band_key(
    gfx, cu_num, rows, width, k, ragged, tie, det, mode, dtype="float32"
) -> dict:
    rlo, rhi = half_octave_bounds(rows)
    wlo, whi = half_octave_bounds(width)
    return {
        "gfx": str(gfx),
        "cu_num": int(cu_num),
        "rows_lo": rlo,
        "rows_hi": rhi,
        "width_lo": wlo,
        "width_hi": whi,
        "k": int(k),
        "dtype": normalize_dtype(dtype),
        "ragged": bool(ragged),
        "tie": normalize_tie(tie),
        "deterministic": bool(det),
        "mode": str(mode),
    }


def key_tuple(d) -> tuple:
    return (
        str(d["gfx"]),
        int(d["cu_num"]),
        int(d["rows_lo"]),
        int(d["rows_hi"]),
        int(d["width_lo"]),
        int(d["width_hi"]),
        int(d["k"]),
        normalize_dtype(d.get("dtype")),
        normalize_bool(d["ragged"]),
        normalize_tie(d["tie"]),
        normalize_bool(d["deterministic"]),
        str(d["mode"]),
    )


def band_label(key) -> str:
    dt = f", {key['dtype']}" if key.get("dtype", "float32") != "float32" else ""
    tie = f", tie={key['tie']}" if key["tie"] else ""
    det = ", deterministic" if key["deterministic"] else ""
    rag = ", ragged" if key["ragged"] else ""
    return (
        f"rows {key['rows_lo']}-{key['rows_hi']} x width "
        f"{key['width_lo']}-{key['width_hi']}, k={key['k']}{dt}{rag}{tie}{det}, "
        f"{key['mode']}"
    )


def check_entry(rows, width, k, ragged, tie, dist, dtype="float32") -> str | None:
    """Why this untuned entry cannot be tuned, or None."""
    if rows < 1 or width < 1:
        return "rows and width must be >= 1"
    if not 1 <= k <= width:
        return f"k={k} must be in [1, width={width}]"
    if dtype != "float32" and k != 1:
        return f"topk_select serves {dtype} only at k=1 (the reduction); got k={k}"
    if tie not in _BACKENDS_BY_TIE:
        return f"tie must be empty, 'low' or 'high'; got {tie!r}"
    if not is_sample_path(dist):
        if dist not in PRESETS:
            return (
                f"unknown dist {dist!r}; use one of {', '.join(PRESETS)}, "
                "'all', or a recorded .pt path"
            )
        if ragged:
            return "ragged rows need a recorded .pt (presets fill every row)"
        return None
    if not os.path.exists(dist):
        return f"sample file not found: {dist}"
    try:
        try:
            blob = torch.load(dist, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # torch without mmap=
            blob = torch.load(dist, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        return f"cannot load {dist}: {type(e).__name__}: {e}"
    if not isinstance(blob, dict) or "input" not in blob:
        return f"{dist} is not a sample written by AITER_TOPK_SELECT_RECORD"
    x = blob["input"]
    if normalize_dtype(x.dtype) != dtype:
        return (
            f"{dist} was recorded as {normalize_dtype(x.dtype)}, the row says {dtype}"
        )
    rec_w = int(blob.get("width", x.shape[1]))
    if rec_w != width:
        return f"{dist} was recorded at width={rec_w}, the row says {width}"
    if "k" in blob and int(blob["k"]) != k:
        return f"{dist} was recorded at k={blob['k']}, the row says {k}"
    if bool(blob.get("ragged", False)) != bool(ragged):
        return f"{dist} was recorded with ragged={bool(blob.get('ragged'))}"
    return None


def _nan_is_empty(v, default):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    s = str(v).strip()
    return s if s else default


def _resolve_dist(dist: str, base_dir: str | None) -> str:
    if is_sample_path(dist) and base_dir and not os.path.isabs(dist):
        return os.path.normpath(os.path.join(base_dir, dist))
    return dist


def groups_from_untuned(df: pd.DataFrame, gfx, cu_num, mode, base_dir=None):
    """One group per table band; same-band entries are pooled into one decision.

    Relative sample paths are taken from ``base_dir`` -- the untuned CSV's
    directory, so a recording directory can be moved as a whole.
    """
    groups: dict[tuple, dict] = {}
    rejected = []
    for raw in df.to_dict("records"):
        try:
            rows, width, k = int(raw["rows"]), int(raw["width"]), int(raw["k"])
            ragged = normalize_bool(_nan_is_empty(raw.get("ragged"), "False"))
            tie = normalize_tie(raw.get("tie"))
            det = normalize_bool(_nan_is_empty(raw.get("deterministic"), "False"))
            dtype = normalize_dtype(raw.get("dtype"))
        except (KeyError, TypeError, ValueError) as e:
            shown = {c: raw.get(c) for c in ("rows", "width", "k") if c in raw}
            rejected.append((f"row {shown}", f"unreadable: {e}"))
            continue
        for dist in expand_dist_field(_nan_is_empty(raw.get("dist"), "all")):
            dist = _resolve_dist(dist, base_dir)
            why = check_entry(rows, width, k, ragged, tie, dist, dtype)
            if why:
                label = f"rows={rows} width={width} k={k} dist={dist}"
                rejected.append(
                    (label + ("" if dtype == "float32" else f" {dtype}"), why)
                )
                continue
            key = band_key(gfx, cu_num, rows, width, k, ragged, tie, det, mode, dtype)
            g = groups.setdefault(key_tuple(key), {"key": key, "specs": []})
            spec = format_spec(dist, rows, width)
            if spec not in g["specs"]:
                g["specs"].append(spec)
    return list(groups.values()), rejected


def groups_from_tuned(df: pd.DataFrame, gfx, cu_num, base_dir=None):
    """Rebuild groups from a tuned table, to re-tune it in place (``--all``)."""
    groups, rejected = [], []
    for raw in df.to_dict("records"):
        if _nan_is_empty(raw.get("gfx"), gfx) != gfx or int(
            float(_nan_is_empty(raw.get("cu_num"), cu_num))
        ) not in (0, int(cu_num)):
            rejected.append(
                (
                    f"row rows {raw.get('rows_lo')}-{raw.get('rows_hi')} k={raw.get('k')}",
                    f"tuned for {raw.get('gfx')}/{raw.get('cu_num')}, not this GPU",
                )
            )
            continue
        key = {c: raw.get(c) for c in LOOKUP_KEYS}
        key.update(gfx=gfx, cu_num=int(cu_num))
        key = dict(zip(LOOKUP_KEYS, key_tuple(key)))
        label = band_label(key)
        listed = [s for s in str(_nan_is_empty(raw.get("dists"), "")).split(";") if s]
        if not listed:
            rejected.append((label, "the row lists no samples (dists) to re-tune on"))
            continue
        specs = []
        for spec in listed:
            p = parse_spec(spec)
            if p is None:
                rejected.append(
                    (
                        f"{label}: {spec}",
                        "no @ROWSxWIDTH suffix; re-tune it from an untuned CSV",
                    )
                )
                continue
            dist = _resolve_dist(p[0], base_dir)
            spec = format_spec(dist, p[1], p[2])
            why = check_entry(
                p[1], p[2], key["k"], key["ragged"], key["tie"], dist, key["dtype"]
            )
            if why:
                rejected.append((f"{label}: {short_spec(spec)}", why))
            else:
                specs.append(spec)
        if specs:
            groups.append({"key": key, "specs": specs})
    return groups, rejected


def _entry_args(key, rows, width, wave) -> tuple:
    """`_choose` / `_router_choice` positional arguments for one call shape on
    the current GPU, as `topk_select` builds them."""
    device = torch.cuda.current_device()
    return (
        rows,
        width,
        key["k"],
        wave,
        key["ragged"],
        key["tie"],
        key["deterministic"],
        key["dtype"] == "float32",
        device,
        _sampled_ok(device),
    )


def plan_backends(g, wave) -> dict:
    """Per sample: the backends topk_select would let a table pick, and the
    router's own choice -- both asked of topk_select, so a backend added there
    is a candidate here with no change."""
    key = g["key"]
    allowed = _allowed(key["tie"], key["deterministic"])
    avail, router = {}, {}
    for spec in g["specs"]:
        _, rows, width = parse_spec(spec)
        args = _entry_args(key, rows, width, wave)
        avail[spec] = set(_servable(*args[:5], *args[7:]) & allowed)
        try:
            router[spec] = _router_choice(*args)
        except RuntimeError:
            router[spec] = None
    common = set.intersection(*avail.values()) if avail else set()
    return {"available": avail, "router": router, "common": common}


def sample_chunks(spec, device, draw=0, draws=DRAWS, dtype="float32"):
    """A sample, as chunks of tensors to time: one recorded tensor, or
    ``draws`` independent draws of a preset in chunks of at most CHUNK_BYTES.

    Some backends' time is a property of the draw, not of the distribution:
    `sampled` either takes its fast path or a ~160us fallback, on 6-45% of
    draws depending on the cell (randn 8x524288 k=64: 15us or ~170us).
    Serving pays the mean over draws, so that is what is measured, over at
    least ``draws`` draws and up to MAX_DRAWS while they fit in DRAW_BYTES --
    a 20% fallback rate needs far more than 16 draws to resolve, and small
    shapes, where it is commonest, are the cheap ones. ``draw`` selects an
    independent set, so a second measurement never sees the first one's data.
    """
    dist, rows, width = parse_spec(spec)
    if is_sample_path(dist):
        x, lens = tile_sample_to_rows(load_sample_file(dist, device), rows, device)
        yield {"spec": spec, "x": x, "row_lens": lens, "xs": [(x, lens)]}
        return
    lens = torch.full((rows,), width, dtype=torch.int32, device=device)
    tdtype = getattr(torch, dtype)
    nbytes = rows * width * torch.empty(0, dtype=tdtype).element_size()
    draws = max(draws, min(MAX_DRAWS, DRAW_BYTES // nbytes))
    per = max(1, min(MAX_COPIES, CHUNK_BYTES // nbytes))
    base = 7 + zlib.crc32(spec.encode()) % 100003 + 1000003 * int(draw)
    for c0 in range(0, draws, per):
        xs = [
            (gen_preset(dist, rows, width, base + 7919 * j, device).to(tdtype), lens)
            for j in range(c0, min(draws, c0 + per))
        ]
        yield {"spec": spec, "x": xs[0][0], "row_lens": lens, "xs": xs}
        del xs


def measure_spec(spec, backends, key, wave, opts, device, check=None):
    """Mean us/call per backend over all of one sample's draws.

    Chunk by chunk: every backend in ``check`` (default: all) must pass the
    oracle on each draw, then the live backends are timed interleaved on that
    chunk; chunk means are weighted by their draw count. Returns
    ``(us, dropped, wrong)`` -- ``wrong`` holds just the oracle failures.
    """
    check = set(backends) if check is None else set(check)
    tot, cnt, dropped, wrong = {}, {}, {}, {}
    chunks = sample_chunks(
        spec,
        device,
        getattr(opts, "draw", 0),
        getattr(opts, "draws", DRAWS),
        key["dtype"],
    )
    for chunk in chunks:
        for b in backends:
            if b in check and b not in dropped:
                st = run_oracle(b, chunk, key, wave, opts)
                if st != "ok":
                    dropped[b] = wrong[b] = f"{short_spec(spec)}: {st}"
        live = [b for b in backends if b not in dropped]
        us, fails = interleave_time(
            live, chunk, key, key["mode"], opts.warmup, opts.iters
        )
        for b, why in fails.items():
            dropped.setdefault(b, f"{short_spec(spec)}: {why}")
        n = len(chunk["xs"])
        for b, t in us.items():
            tot[b] = tot.get(b, 0.0) + t * n
            cnt[b] = cnt.get(b, 0) + n
        del chunk
        torch.cuda.empty_cache()
    full = max(cnt.values(), default=0)
    us = {b: tot[b] / cnt[b] for b in tot if b not in dropped and cnt[b] == full}
    return us, dropped, wrong


# ---------------------------------------------------------------- oracle / timing


def _rec_for(key, backend) -> dict:
    return {**key, "backend": backend, "aiter_rev": ""}


def _call_select(x, k, row_lens, key):
    end = row_lens if key["ragged"] else None
    return topk_select(
        x, k, end=end, tie=key["tie"], deterministic=key["deterministic"]
    )[1]


def run_oracle(backend, sample, key, wave, opts=None) -> str:
    """Correctness on every draw of the sample, through the entry point."""
    if backend == FAULT_NAME:
        os.abort()
    if backend in (getattr(opts, "inject_wrong", None) or ()):
        return "value_mismatch (injected by --inject-wrong)"
    k = key["k"]
    rows, width = sample["x"].shape
    status = "ok"
    with override_tuned_rows([_rec_for(key, backend)]):
        _choose.cache_clear()
        got = _choose(*_entry_args(key, rows, width, wave), dtype=key["dtype"])
        if got != backend:
            return f"entry dispatched {got} instead"
        draws = sample.get("xs") or [(sample["x"], sample["row_lens"])]
        for j, (x, lens) in enumerate(draws):
            idx = _call_select(x, k, lens, key)
            torch.cuda.synchronize()
            status = oracle_check(idx, x, lens, k)
            if key["deterministic"] and status == "ok":
                idx2 = _call_select(x, k, lens, key)
                torch.cuda.synchronize()
                a = torch.sort(idx.long(), dim=1).values
                b = torch.sort(idx2.long(), dim=1).values
                if not torch.equal(a, b):
                    status = "nondeterministic"
            if status != "ok":
                status = f"{status} (draw {j})" if len(draws) > 1 else status
                break
    _choose.cache_clear()
    return status


def _rotation(sample):
    """Buffers one timed sequence cycles through: the draws, or clones of a
    recorded tensor, enough to step past the 256 MB MALL either way."""
    draws = sample.get("xs") or [(sample["x"], sample["row_lens"])]
    if len(draws) > 1:
        return draws, max(len(draws), 4)
    x0, lens0 = draws[0]
    nbytes = x0.numel() * x0.element_size()
    copies = max(2, min(MAX_COPIES, math.ceil(ROT_BYTES / max(nbytes, 1))))
    xs = [(x0, lens0)] + [(x0.clone(), lens0.clone()) for _ in range(copies - 1)]
    return xs, max(copies, 4)


def _make_timer(backend, xs, ncalls, key, mode):
    k = key["k"]
    seq = [xs[j % len(xs)] for j in range(ncalls)]

    def body():
        for x, lens in seq:
            _call_select(x, k, lens, key)

    with override_tuned_rows([_rec_for(key, backend)]):
        _choose.cache_clear()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if mode == "eager":
            return body, "eager"
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            graph.replay()
            torch.cuda.synchronize()
            return graph.replay, "graph"
        except Exception as e:  # noqa: BLE001
            torch.cuda.synchronize()
            return None, f"graph capture failed ({type(e).__name__})"


def _time_fn(fn, ncalls, reps):
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * ncalls)


def interleave_time(backends, sample, key, mode, warmup, iters):
    """Median us per call for each backend, timed through `topk_select` itself.

    Each backend gets ``warmup`` untimed passes (at least one, which also sizes
    its timed batch), then ``iters`` rounds in a shuffled order. Eager bodies do
    not clear `_choose`; each backend's table row is activated and the memo
    primed before its slot instead, so the timed calls are the serving path's
    memo hits.
    """
    xs, ncalls = _rotation(sample)
    timers, fails = {}, {}
    for b in backends:
        fn, tag = _make_timer(b, xs, ncalls, key, mode)
        if fn is None:
            fails[b] = tag
        else:
            timers[b] = fn

    def slot(b, reps):
        with override_tuned_rows([_rec_for(key, b)]):
            if mode == "eager":
                _choose.cache_clear()
                timers[b]()
                torch.cuda.synchronize()
            return _time_fn(timers[b], ncalls, reps)

    reps = {}
    for b in timers:
        for _ in range(max(1, warmup)):
            once = slot(b, 1)
        reps[b] = max(3, int(TARGET_MS * 1e3 / max(once * ncalls, 1.0)))
    acc = {b: [] for b in timers}
    rng = random.Random(zlib.crc32(sample["spec"].encode()))
    for _ in range(iters):
        order = list(timers)
        rng.shuffle(order)
        for b in order:
            acc[b].append(slot(b, reps[b]))
    _choose.cache_clear()
    return {b: sorted(v)[len(v) // 2] for b, v in acc.items()}, fails


# ---------------------------------------------------------------- decision


def _pick(
    objective,
    us_by,
    router,
    min_improvement_pct,
    candidates=None,
    no_regress=NO_REGRESS,
):
    """Backend for the band, or None to keep the router.

    ``us_by`` is ``{sample: {backend: us}}``; ``router`` is a backend name or
    ``{sample: name}`` -- the router may pick differently across one band.
    Returns ``(backend, stats)`` or ``(None, {backend_or_"_": reason})``.
    """
    specs = list(us_by)
    if not specs:
        return None, {"_": "no sample was timed"}
    rt = router if isinstance(router, dict) else dict.fromkeys(specs, router)
    for s in specs:
        if rt.get(s) not in us_by[s]:
            return None, {
                "_": f"the router's choice ({rt.get(s)}) has no time on "
                f"{short_spec(s)}, so there is nothing to compare against"
            }
    pool = (
        candidates if candidates is not None else {b for s in specs for b in us_by[s]}
    )
    pool = sorted(b for b in pool if all(b in us_by[s] for s in specs))
    base = {s: us_by[s][rt[s]] for s in specs}

    def ratios(b):
        return {s: us_by[s][b] / base[s] for s in specs}

    floor = 1.0 + min_improvement_pct / 100.0
    if objective == "minimax":
        best = {s: min(us_by[s].values()) for s in specs}

        def regret(b):
            return max(us_by[s][b] / best[s] for s in specs)

        router_regret = max(base[s] / best[s] for s in specs)
        ranked = sorted(pool, key=regret)
        if not ranked:
            return None, {"_": "no backend was timed on every sample"}
        b = ranked[0]
        if all(b == rt[s] for s in specs) or regret(b) * floor > router_regret:
            return None, {
                "_": f"the router is already within {router_regret:.2f}x of the "
                f"per-sample best everywhere; {b} would be {regret(b):.2f}x"
            }
        r = ratios(b)
        worst = max(r, key=r.get)
        return b, {
            "speedup": _geomean([1 / v for v in r.values()]),
            "worst_ratio": r[worst],
            "worst_sample": worst,
            "worst_regret": regret(b),
            "router_worst_regret": router_regret,
        }
    reasons, ok = {}, []
    for b in pool:
        if all(b == rt[s] for s in specs):
            continue
        r = ratios(b)
        worst = max(r, key=r.get)
        if r[worst] > 1.0 + no_regress:
            reasons[b] = (
                f"{r[worst]:.2f}x slower than the router ({rt[worst]}) on "
                f"{short_spec(worst)}"
            )
            continue
        sp = _geomean([1 / v for v in r.values()])
        if sp < floor:
            reasons[b] = f"only {sp:.3f}x faster than the router (needs {floor:.2f}x)"
            continue
        ok.append((sp, b, r[worst], worst))
    if not ok:
        return None, reasons or {"_": "no backend other than the router's choice"}
    ok.sort(reverse=True)
    sp, b, wr, ws = ok[0]
    return b, {"speedup": sp, "worst_ratio": wr, "worst_sample": ws}


def router_failures(summary) -> dict:
    """``{sample: reason}`` where the router's own choice failed: wrong results,
    no graph capture, or a crash. Such a band has no baseline to beat, and
    keeping the router there keeps the failure."""
    failed = {**(summary.get("dropped") or {}), **(summary.get("crashed") or {})}
    return {
        s: failed[b] for s, b in (summary.get("router") or {}).items() if b in failed
    }


def _fastest_working(us_by, candidates) -> str | None:
    """The backend with the lowest geomean time among those timed on every sample."""
    pool = [b for b in candidates if all(b in t for t in us_by.values())]
    if not us_by or not pool:
        return None
    return min(pool, key=lambda b: _geomean([us_by[s][b] for s in us_by]))


def decide(g, plan, us_by, opts, summary):
    router = {s: plan["router"][s] for s in us_by}
    broken = router_failures(summary)
    if broken:
        pick = _fastest_working(us_by, plan["common"])
        why = {"router_failed": broken}
        if pick is None:
            why["_"] = (
                "the router's choice failed and no other backend passed on every "
                "sample, so topk_select keeps the router's choice here"
            )
    else:
        pick, why = _pick(
            opts.objective,
            us_by,
            router,
            opts.min_improvement_pct,
            candidates=plan["common"],
        )
    summary["pick"], summary["why"] = pick, why
    if pick is None:
        return None, summary
    return _row(g, pick, us_by, router, why), summary


def _row(g, pick, us_by, router, why) -> dict:
    """The tuned CSV row; a fallback row has no router time to compare with."""
    specs = list(us_by)
    fallback = "router_failed" in why
    return {
        **g["key"],
        "backend": pick,
        "us": round(_geomean([us_by[s][pick] for s in specs]), 4),
        "default_backend": "|".join(sorted({router[s] for s in specs})),
        "us_default": (
            "" if fallback else round(_geomean([us_by[s][router[s]] for s in specs]), 4)
        ),
        "worst_ratio_vs_default": "" if fallback else round(why["worst_ratio"], 4),
        "dists": ";".join(specs),
        "n_samples": len(specs),
        "aiter_rev": aiter_rev(),
    }


def _failure_reason(summary) -> str:
    """Why a band that wrote no row has no usable backend at all, else ""."""
    if not summary.get("us"):
        return "no backend could be timed"
    if router_failures(summary):
        return "the router's choice failed and no other backend passed"
    return ""


def _measured(summary) -> bool:
    """The band's "no row" rests on a timing of the router on every sample,
    not on a sample that could not be timed."""
    us, rt = summary.get("us") or {}, summary.get("router") or {}
    return bool(us) and all(rt.get(s) in t for s, t in us.items())


def confirm(g, first, second, opts):
    """The row to write once a second process has re-measured the band.

    A backend is written only if it beats the router (by ``_pick``'s rules)
    in both measurements; among those, the one whose worse speedup is
    largest. Requiring the *same top pick* instead throws away real wins
    between near-ties -- `decode` and `sampled` both 1.23x over the router at
    recency 64x8192 k=512 swap places from one set of draws to the next.

    If the router's choice failed in either measurement, the row goes to the
    backend that worked on every sample in both, fastest by its slower of the
    two geomeans.
    """
    shared = set(first.get("common") or ()) & set(second.get("common") or ())
    # A router failure in either process counts: one that does not recur in
    # the second may still recur in serving.
    broken = {**router_failures(second), **router_failures(first)}
    if broken:
        timed = [s["us"] for s in (first, second)]
        pool = [
            b
            for b in sorted(shared)
            if all(us and all(b in t for t in us.values()) for us in timed)
        ]
        if not pool:
            first["why"] = {
                "router_failed": broken,
                "_": "the router's choice failed and no other backend passed on "
                "every sample in both processes, so topk_select keeps it here",
            }
            return None, first
        slower = {
            b: max(_geomean([t[b] for t in us.values()]) for us in timed) for b in pool
        }
        b = min(slower, key=slower.get)
        second["pick"], second["why"] = b, {"router_failed": broken}
        second.setdefault("notes", []).append(
            f"{b} passed on every sample in two processes on independent draws"
        )
        router = {s: second["router"][s] for s in second["us"]}
        return _row(g, b, second["us"], router, second["why"]), second
    both = []
    for b in sorted(shared):
        picks = [
            _pick(
                opts.objective,
                s["us"],
                s["router"],
                opts.min_improvement_pct,
                candidates={b},
            )
            for s in (first, second)
        ]
        if all(p == b for p, _w in picks):
            sp = [w["speedup"] for _p, w in picks]
            both.append((min(sp), b, sp))
    if not both:
        top = first.get("pick")
        first.setdefault("notes", []).append(
            f"first process picked {top}, but no backend beat the router in a "
            "second process on fresh draws as well; not written"
        )
        first["why"] = {"_": "the first-process win did not reproduce"}
        return None, first
    _, b, (sp1, sp2) = max(both)
    plan = {"router": second["router"], "common": {b}}
    rec, summary = decide(g, plan, second["us"], opts, second)
    summary.setdefault("notes", []).append(
        f"{b} beat the router in two processes on independent draws: "
        f"{sp1:.2f}x then {sp2:.2f}x"
    )
    return rec, summary


def tune_group(g, opts):
    """Oracle + interleaved timing for one band, in the current process."""
    device = torch.device("cuda", torch.cuda.current_device())
    wave = wave_size_of(device.index)
    key = g["key"]
    plan = plan_backends(g, wave)
    cands = sorted(plan["common"] | {b for b in plan["router"].values() if b})
    if opts.inject_fault:
        cands.append(FAULT_NAME)
    summary = {
        "key": key,
        "router": plan["router"],
        "candidates": cands,
        "common": sorted(plan["common"]),
    }
    dropped, us_by = {}, {}
    for spec in g["specs"]:
        here = [
            b
            for b in cands
            if (b == FAULT_NAME or b in plan["available"][spec]) and b not in dropped
        ]
        us, why, _wrong = measure_spec(spec, here, key, wave, opts, device)
        for b, w in why.items():
            dropped.setdefault(b, w)
        us_by[spec] = us
    for times in us_by.values():
        for b in dropped:
            times.pop(b, None)
    summary["dropped"], summary["us"] = dropped, us_by
    return decide(g, plan, us_by, opts, summary)


def time_one_backend(g, backend, opts):
    """Crash fallback: one backend over a band's samples, alone in its process."""
    if backend == FAULT_NAME:
        os.abort()
    device = torch.device("cuda", torch.cuda.current_device())
    wave = wave_size_of(device.index)
    plan = plan_backends(g, wave)
    us = {}
    for spec in g["specs"]:
        if backend not in plan["available"][spec]:
            continue
        t, why, _wrong = measure_spec(spec, [backend], g["key"], wave, opts, device)
        if backend in why:
            return {"dropped": why[backend], "us": us}
        us[spec] = t[backend]
    return {"dropped": None, "us": us}


def compare_group(g, opts):
    """``--run_config``: the live table's choice vs the router, per sample."""
    device = torch.device("cuda", torch.cuda.current_device())
    wave = wave_size_of(device.index)
    key = g["key"]
    reload_tuned_table()
    out = {}
    for spec in g["specs"]:
        dist, rows, width = parse_spec(spec)
        args = _entry_args(key, rows, width, wave)
        _choose.cache_clear()
        tuned = _choose(*args, dtype=key["dtype"])
        router = _router_choice(*args)
        _choose.cache_clear()
        if is_sample_path(dist) and not os.path.exists(dist):
            out[spec] = {
                "table": tuned,
                "router": router,
                "oracle": "ok",
                "table_us": None,
                "router_us": None,
                "fail": f"sample file not found: {dist}",
            }
            continue
        # A fallback row exists because the router's choice failed; check that
        # it still does, rather than timing the table against it.
        fallback = bool(g.get("fallback")) and tuned != router
        check = set() if tuned == router else {tuned, router} if fallback else {tuned}
        us, why, wrong = measure_spec(
            spec, sorted({tuned, router}), key, wave, opts, device, check
        )
        out[spec] = {
            "table": tuned,
            "router": router,
            "oracle": wrong[tuned].split(": ", 1)[-1] if tuned in wrong else "ok",
            "table_us": us.get(tuned),
            "router_us": us.get(router),
            "fail": "; ".join(
                f"{b}: {w}"
                for b, w in why.items()
                if b not in wrong and not (fallback and b == router)
            ),
            "fallback": fallback,
            "router_failed": why.get(router) if fallback else None,
        }
    return out


RUN_CONFIG_REGRESS = 0.05


def summarize_run_config(label, per_spec):
    """``(text, status, table_us)`` for one band of ``--run_config``.

    Status, in the base tuner's vocabulary: ``mismatch: ...`` if the table's
    backend returned a wrong answer on any sample; ``error: ...`` if a sample
    could not be timed, or the table's backend is more than
    ``RUN_CONFIG_REGRESS`` slower than the router on any sample (a stale row);
    else ``ok``. A fallback row -- written because the router's choice failed --
    is not timed against the router: it is ``error`` once the router's choice
    passes on every sample again, since the row may then cost speed for nothing.
    """
    lines = [
        label,
        f"  {'sample':<34}{'table':>9}{'us':>9}{'router':>9}{'us':>9}{'ratio':>8}",
    ]
    tus, rus = [], []
    worst = bad = untimed = None
    fallback = [m for m in per_spec.values() if m.get("fallback")]
    for spec, m in per_spec.items():
        t, u = m["table_us"], m["router_us"]
        ratio = t / u if t and u else float("nan")
        tus.append(t)
        rus.append(u)
        if m.get("oracle", "ok") != "ok" and bad is None:
            bad = (spec, m)
        if not (t and (u or m.get("fallback"))) and untimed is None:
            untimed = (spec, m)
        if t and u and not m.get("fallback") and (worst is None or ratio > worst[0]):
            worst = (ratio, spec, m)
        if m.get("router_failed"):
            lines.append(
                f"  (the router's {m['router']} still fails on {short_spec(spec)}: "
                f"{m['router_failed'].split(': ', 1)[-1]})"
            )
        router_us = "failed" if m.get("router_failed") else f"{(u or -1):.1f}"
        lines.append(
            f"  {short_spec(spec):<34}{m['table']:>9}{(t or -1):9.1f}"
            f"{m['router']:>9}{router_us:>9}{ratio:8.3f}"
            + (f"  ({m['fail']})" if m.get("fail") else "")
        )
    tu, ru = _geomean(tus), _geomean(rus)
    ok = math.isfinite(tu) and math.isfinite(ru)
    if any(m.get("router_failed") for m in fallback):
        lines.append(f"  geomean: table {tu:.1f} us (the router's choice failed)")
    else:
        lines.append(
            f"  geomean: table {tu:.1f} us vs router {ru:.1f} us"
            + (f" -> {tu / ru:.3f}x" if ok else "")
        )
    if bad is not None:
        spec, m = bad
        status = f"mismatch: {m['table']} on {short_spec(spec)}: {m['oracle']}"
    elif untimed is not None:
        spec, m = untimed
        status = f"error: {short_spec(spec)} was not timed ({m.get('fail') or '?'})"
    elif fallback and not any(m.get("router_failed") for m in fallback):
        status = (
            f"error: the router's choice ({fallback[0]['router']}) passes on every "
            "sample again, so this fallback row may only cost speed; re-tune this band"
        )
    elif worst is not None and worst[0] > 1.0 + RUN_CONFIG_REGRESS:
        ratio, spec, m = worst
        status = (
            f"error: the table's {m['table']} takes {ratio:.2f}x the router's "
            f"({m['router']}) time on {short_spec(spec)}; re-tune this band"
        )
    else:
        status = "ok"
    return "\n".join(lines), status, (tu if math.isfinite(tu) else -1)


# ---------------------------------------------------------------- isolation


def _child_main(fn, payload, device_index, conn):
    try:
        torch.cuda.set_device(device_index)
        out = ("ok", fn(**payload))
    except BaseException as e:  # noqa: BLE001
        out = ("error", f"{type(e).__name__}: {e}", traceback.format_exc())
    try:
        conn.send(out)
    finally:
        conn.close()


def run_isolated(fn, payloads, devices, timeout, on_done=None, quiet=True):
    """``fn(**payload)`` per payload, each in its own spawned process.

    At most one process per device at a time. Returns, aligned with
    ``payloads``, ``(status, value, device)`` where status is ``ok``, ``error``
    (Python exception), ``died`` (value = exit code) or ``timeout``. A process
    killed by a GPU fault surfaces as ``died`` -- a ``multiprocessing.Pool``
    would instead wait forever for the task it lost. ``quiet`` sets
    ``AITER_LOG_LEVEL=ERROR`` in children unless the caller set it.
    """
    ctx = torch.multiprocessing.get_context("spawn")
    results: list = [None] * len(payloads)
    queue = list(range(len(payloads)))
    free = list(devices)
    running = {}
    child_env = {}
    if quiet and "AITER_LOG_LEVEL" not in os.environ:
        child_env["AITER_LOG_LEVEL"] = "ERROR"
    while queue or running:
        while queue and free:
            i, dev = queue.pop(0), free.pop(0)
            recv, send = ctx.Pipe(duplex=False)
            p = ctx.Process(
                target=_child_main, args=(fn, payloads[i], dev, send), daemon=True
            )
            saved = {k: os.environ.get(k) for k in child_env}
            os.environ.update(child_env)
            try:
                p.start()
            finally:
                for k_, v in saved.items():
                    if v is None:
                        os.environ.pop(k_, None)
                    else:
                        os.environ[k_] = v
            send.close()
            running[i] = (p, recv, dev, time.monotonic())
        for i, (p, recv, dev, t0) in list(running.items()):
            res = None
            if recv.poll():
                try:
                    msg = recv.recv()
                    res = (msg[0], msg[1] if msg[0] == "ok" else msg[1:], dev)
                except EOFError:
                    res = ("died", None, dev)
            elif not p.is_alive():
                res = ("died", p.exitcode, dev)
            elif timeout and time.monotonic() - t0 > timeout:
                p.kill()
                res = ("timeout", timeout, dev)
            if res is None:
                continue
            p.join(timeout=30)
            if res[0] == "died":
                res = ("died", p.exitcode, dev)
            recv.close()
            results[i] = res
            del running[i]
            free.append(dev)
            if on_done:
                on_done(i, res)
        time.sleep(0.02)
    return results


def _failure_text(res) -> str:
    status, value, _dev = res
    if status == "died":
        return f"process died (exit code {value})"
    if status == "timeout":
        return f"timed out after {value}s"
    return str(value[0]) if isinstance(value, tuple) else str(value)


# ---------------------------------------------------------------- reporting


def print_group(idx, total, g, rec, summary):
    key = g["key"]
    out = [f"[{idx}/{total}] {band_label(key)}  ({len(g['specs'])} sample(s))"]
    us_by = summary.get("us") or {}
    router = summary.get("router") or {}
    backends = sorted({b for m in us_by.values() for b in m})
    if backends:
        width = max(len(short_spec(s)) for s in us_by) + 2
        out.append("  us/call".ljust(width + 2) + "".join(f"{b:>10}" for b in backends))
        for s, m in us_by.items():
            cells = []
            for b in backends:
                v = m.get(b)
                mark = "*" if router.get(s) == b else " "
                cells.append(f"{v:9.1f}{mark}" if v is not None else f"{'-':>9} ")
            out.append(f"  {short_spec(s):<{width}}" + "".join(cells))
        if set(router.values()) & set(backends):
            out.append("  (* = what the router picks today)")
    for b, why in (summary.get("dropped") or {}).items():
        out.append(f"  dropped {b}: {why}")
    for b, why in (summary.get("crashed") or {}).items():
        out.append(f"  crashed {b}: {why}")
    for note in summary.get("notes") or []:
        out.append(f"  note: {note}")
    why = summary.get("why") or {"_": summary.get("skip", "no decision")}
    broken = why.get("router_failed") or {}
    for s, reason in broken.items():
        out.append(
            f"  WARNING: the router's choice ({router.get(s)}) failed on "
            f"{short_spec(s)}: {reason.split(': ', 1)[-1]}"
        )
        if "mismatch" in reason or "duplicate" in reason or "live_count" in reason:
            out.append(
                "           wrong results from an aiter backend: please report "
                "it with this sample"
            )
    if rec is not None and broken:
        out.append(
            f"  -> WROTE {rec['backend']}: the fastest backend that passed on every "
            "sample, so topk_select stops using the failing one here"
        )
    elif rec is not None:
        out.append(
            f"  -> WROTE {rec['backend']}: {why['speedup']:.2f}x faster than the router "
            f"(geomean); worst sample {short_spec(why['worst_sample'])} runs at "
            f"{why['worst_ratio']:.2f}x of the router's time"
        )
    else:
        out.append("  -> no row; the router keeps this band:")
        for b, reason in why.items():
            if b == "router_failed":
                continue
            out.append(f"       {reason}" if b == "_" else f"       {b}: {reason}")
    print("\n".join(out), flush=True)


class _ArgBox:
    def __init__(self, args):
        self.mode = args.mode
        self.warmup = args.warmup
        self.iters = args.iters
        self.objective = args.objective
        self.min_improvement_pct = args.min_improvement_pct
        self.inject_fault = getattr(args, "inject_fault", False)
        self.inject_wrong = tuple(getattr(args, "inject_wrong", None) or ())
        self.draws = getattr(args, "draws", DRAWS)
        # Which independent set of preset draws to time: 0 tunes, 1 confirms,
        # 2 is --run_config's, so a verdict is never on data the tuner saw.
        self.draw = 0


def _with_draw(opts, draw):
    o = copy.copy(opts)
    o.draw = draw
    return o


def _untuned_row(g) -> dict:
    row = dict(g["key"])
    row["tie"] = row["tie"] if row["tie"] else float("nan")
    row["samples"] = ";".join(g["specs"])
    return row


class TopkSelectTuner(TunerCommon):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **TunerCommon.ARG_DEFAULTS,
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/topk_select_untuned.csv",
        "tune_file": f"{AITER_CONFIG_TOPK_SELECT}",
        "config_env_name": "AITER_CONFIG_TOPK_SELECT",
        "warmup": 5,
        "iters": 5,
        "batch": 100,
        "sort": False,
    }

    def __init__(self):
        super().__init__(
            "topk_select_tuned",
            list(LOOKUP_KEYS),
            RESULT_COLS,
            "topk_select backend tuner (shape + distribution)",
        )
        self._groups: list[dict] = []
        self._written = 0
        self._done = 0
        self._out_file = None
        self.run_config_failed = False

    def _clear_op_caches(self):
        reload_tuned_table()
        _choose.cache_clear()

    def _restore_config_env(self, env_name, old_val, old_rebuild=0):
        super()._restore_config_env(env_name, old_val, old_rebuild)
        self._clear_op_caches()

    def _setup_specific_arguments(self):
        import argparse

        self.parser.add_argument(
            "--mode",
            choices=("graph", "eager"),
            default="graph",
            help="time HIP-graph replay (default) or eager calls",
        )
        self.parser.add_argument(
            "--draws",
            type=int,
            default=DRAWS,
            help="independent draws per preset sample; the time is their mean",
        )
        self.parser.add_argument(
            "--objective",
            choices=("no_regress", "minimax"),
            default="no_regress",
            help="no_regress: never more than 2%% slower than the router on any "
            "sample; minimax: smallest worst-sample gap to the per-sample best",
        )
        self.parser.add_argument(
            "--inject-fault", action="store_true", help=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--inject-wrong", action="append", metavar="BACKEND", help=argparse.SUPPRESS
        )

    def getKernelName(self, kernel_id):
        return str(kernel_id)

    def calculate(self, results, inbpe=2, outbpe=2):
        return results

    def result_to_df(self, rets):
        if isinstance(rets, pd.DataFrame):
            return rets
        return pd.DataFrame(rets or [], columns=self.columns)

    def _devices(self, args) -> list[int]:
        n = max(1, min(int(getattr(args, "mp", 1) or 1), torch.cuda.device_count()))
        return list(range(n))

    _GEMM_ONLY_FLAGS = (
        "errRatio",
        "splitK",
        "shape_grouped",
        "e2e_tune",
    )

    def pre_process(self, args):
        gfx, cu = self.get_gfx(), self.get_cu_num()
        self._opts = _ArgBox(args)
        self._args = args
        self._out_file = self.get_out_file(args.tune_file)
        for name in ("draws", "iters"):
            if getattr(args, name) < 1:
                raise SystemExit(f"--{name} must be >= 1, got {getattr(args, name)}")
        for name in ("warmup", "min_improvement_pct"):
            if getattr(args, name) < 0:
                raise SystemExit(f"--{name} must be >= 0, got {getattr(args, name)}")
        ignored = [
            f"--{f}"
            for f in self._GEMM_ONLY_FLAGS
            if getattr(args, f, None) != self.parser.get_default(f)
        ]
        if ignored:
            print(
                f"note: the topk_select tuner ignores {', '.join(ignored)} "
                "(GEMM-tuner options)",
                flush=True,
            )
        # The base class re-sorts -o after every batch and when there is nothing
        # to tune, and a run that keeps the router everywhere writes no row:
        # -o has to exist either way, so the table the summary points at does.
        if not os.path.exists(self._out_file):
            parent = os.path.dirname(os.path.abspath(self._out_file))
            os.makedirs(parent, exist_ok=True)
            pd.DataFrame(columns=self.columns).to_csv(self._out_file, index=False)
        df = self.get_untuned_gemm_list(args.untune_file)
        base_dir = os.path.dirname(os.path.abspath(args.untune_file))
        if "rows_lo" in df.columns:
            groups, rejected = groups_from_tuned(df, gfx, cu, base_dir)
        else:
            missing = {"rows", "width", "k"} - set(df.columns)
            if missing:
                raise SystemExit(
                    f"{args.untune_file}: missing column(s) {sorted(missing)}. "
                    "Needed: rows,width,k; optional: dtype,ragged,tie,deterministic,"
                    "dist"
                )
            groups, rejected = groups_from_untuned(df, gfx, cu, args.mode, base_dir)
        wave = wave_size_of(0)
        keep = []
        for g in groups:
            plan = plan_backends(g, wave)
            if not plan["common"] or not all(plan["router"].values()):
                rejected.append((band_label(g["key"]), "no backend serves it"))
            elif all(plan["available"][s] <= {plan["router"][s]} for s in g["specs"]):
                only = ", ".join(sorted(set(plan["router"].values())))
                rejected.append(
                    (
                        band_label(g["key"]),
                        f"only {only} can serve it; nothing to choose",
                    )
                )
            else:
                keep.append(g)
        if rejected:
            print(f"Skipped {len(rejected)} input entries:", flush=True)
            for what, why in rejected:
                print(f"  {what}: {why}", flush=True)
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not args.all:
            fresh = [g for g in keep if not self._already_tuned(g)]
            if len(fresh) < len(keep):
                print(
                    f"{len(keep) - len(fresh)} band(s) already in {args.tune_file} "
                    "with these samples; skipped (pass --all to re-tune them)",
                    flush=True,
                )
            keep = fresh
        self._groups = keep
        self.untunedf = pd.DataFrame(
            [_untuned_row(g) for g in keep], columns=[*LOOKUP_KEYS, "samples"]
        )

    def _already_tuned(self, g) -> bool:
        df = self.tunedf
        if df is None or df.empty:
            return False
        want = key_tuple(g["key"])
        for raw in df.to_dict("records"):
            try:
                if key_tuple(raw) != want:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            have = set(str(_nan_is_empty(raw.get("dists"), "")).split(";"))
            if set(g["specs"]) <= have:
                return True
        return False

    def tune(self, untunedf, tunedf, args):
        want = {key_tuple(r) for r in untunedf.to_dict("records")}
        groups = [g for g in self._groups if key_tuple(g["key"]) in want]
        if not groups:
            return []
        total = len(self._groups)

        def report(g, rec, summary):
            self._done += 1
            print_group(self._done, total, g, rec, summary)

        def final_if_no_row(i, res):
            if res[0] == "ok" and res[1][0] is None:
                report(groups[i], *res[1])

        first = self._measure(groups, args, final_if_no_row)
        cand = [i for i, (rec, _s) in enumerate(first) if rec is not None]
        for i, (rec, summary) in enumerate(first):
            if rec is None and summary.get("crashed") is not None:
                report(groups[i], rec, summary)
        # A process can be slow for one backend in every round, which the
        # interleaved median cannot see, and a preset draw can be unlucky; a
        # row is written only if a fresh process on fresh draws picks the same.
        second = self._measure(
            [groups[i] for i in cand], args, opts=_with_draw(self._opts, 1)
        )
        recs = []
        decided = {i: (None, s) for i, (r, s) in enumerate(first) if r is None}
        for i, (_rec2, summary2) in zip(cand, second):
            rec, summary = confirm(groups[i], first[i][1], summary2, self._opts)
            decided[i] = (rec, summary)
            if rec is not None:
                recs.append(rec)
            report(groups[i], rec, summary)
        failed = [
            {**groups[i]["key"], "reason": _failure_reason(s)}
            for i, (rec, s) in sorted(decided.items())
            if rec is None and _failure_reason(s)
        ]
        if failed:
            self.failed = pd.DataFrame([*self.failed.to_dict("records"), *failed])
        stale = [
            groups[i]
            for i, (rec, s) in decided.items()
            if rec is None and _measured(s) and self._row_for(groups[i]) is not None
        ]
        if stale:
            self._drop_stale(stale, args)
        if getattr(args, "profile_file", ""):
            self._write_profile(
                args.profile_file,
                [(g, 1, s) for g, (_r, s) in zip(groups, first)]
                + [(groups[i], 2, s) for i, (_r, s) in zip(cand, second)],
            )
        self._written += len(recs)
        return recs

    def _row_for(self, g):
        df = self.tunedf
        if df is None or df.empty:
            return None
        want = key_tuple(g["key"])
        for raw in df.to_dict("records"):
            try:
                if key_tuple(raw) == want:
                    return raw
            except (KeyError, TypeError, ValueError):
                continue
        return None

    def _drop_stale(self, groups, args):
        """Bands re-measured to "keep the router" lose their earlier row: a row
        that the new samples do not support would otherwise still route."""
        labels = ", ".join(band_label(g["key"]) for g in groups)
        if getattr(args, "compare", False):
            print(
                f"note: {self._out_file} still has an earlier row for {labels}, "
                "which these samples no longer support; re-run without --compare "
                "to remove it",
                flush=True,
            )
            return
        drop = {key_tuple(g["key"]) for g in groups}
        old = self.get_tuned_gemm_list(self._out_file)
        keep = [r for r in old.to_dict("records") if key_tuple(r) not in drop]
        pd.DataFrame(keep, columns=old.columns).to_csv(self._out_file, index=False)
        self.tunedf = self.get_tuned_gemm_list(self._out_file)
        print(
            f"removed the earlier row for {labels} from {self._out_file}: the new "
            "samples no longer support it, so the router keeps it",
            flush=True,
        )

    _PROFILE_COLUMNS = (
        *LOOKUP_KEYS,
        "pass",
        "sample",
        "backend",
        "us",
        "router",
        "status",
    )

    def _write_profile(self, path, entries):
        """Every measurement behind the decisions, one row per (band, pass,
        sample, backend), including backends that were dropped or crashed."""
        rows = []
        for g, npass, summary in entries:
            key = dict(g["key"])
            key["tie"] = key["tie"] or ""
            router = summary.get("router") or {}
            for spec, times in (summary.get("us") or {}).items():
                for b, t in sorted(times.items()):
                    rows.append(
                        {
                            **key,
                            "pass": npass,
                            "sample": spec,
                            "backend": b,
                            "us": round(t, 4),
                            "router": router.get(spec, ""),
                            "status": "ok",
                        }
                    )
            for kind in ("dropped", "crashed"):
                for b, why in (summary.get(kind) or {}).items():
                    rows.append(
                        {
                            **key,
                            "pass": npass,
                            "sample": "",
                            "backend": b,
                            "us": "",
                            "router": "",
                            "status": f"{kind}: {why}",
                        }
                    )
        if not rows:
            return
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        pd.DataFrame(rows, columns=self._PROFILE_COLUMNS).to_csv(
            path, mode="a", header=new, index=False
        )

    def _measure(self, groups, args, on_done=None, opts=None):
        """``(rec, summary)`` per group: one isolated process each, with the
        per-backend fallback for a group whose process died."""
        opts = opts or self._opts
        payloads = [{"g": g, "opts": opts} for g in groups]
        res = run_isolated(
            tune_group,
            payloads,
            self._devices(args),
            args.timeout,
            on_done,
            quiet=not getattr(args, "verbose", False),
        )
        out = []
        for g, r in zip(groups, res):
            if r[0] == "ok":
                out.append(r[1])
            else:
                out.append(self._retry_per_backend(g, opts, r, args))
        return out

    def _retry_per_backend(self, g, opts, first, args):
        plan = plan_backends(g, wave_size_of(0))
        cands = sorted(plan["common"] | {b for b in plan["router"].values() if b})
        if opts.inject_fault:
            cands.append(FAULT_NAME)
        payloads = [{"g": g, "backend": b, "opts": opts} for b in cands]
        res = run_isolated(
            time_one_backend,
            payloads,
            [first[2]],
            args.timeout,
            quiet=not getattr(args, "verbose", False),
        )
        us_by = {s: {} for s in g["specs"]}
        dropped, crashed = {}, {}
        for b, r in zip(cands, res):
            if r[0] != "ok":
                crashed[b] = _failure_text(r)
            elif r[1]["dropped"]:
                dropped[b] = r[1]["dropped"]
            else:
                for s, t in r[1]["us"].items():
                    us_by[s][b] = t
        us_by = {s: m for s, m in us_by.items() if m}
        summary = {
            "key": g["key"],
            "router": plan["router"],
            "common": sorted(plan["common"]),
            "dropped": dropped,
            "crashed": crashed,
            "us": us_by,
            "notes": [
                (
                    f"the band's process failed ({_failure_text(first)}); each "
                    "backend was re-run alone, so these times are not interleaved"
                )
            ],
        }
        return decide(g, plan, us_by, opts, summary)

    def post_process(self, results, args, topk=-1, fast_mode=False):
        if isinstance(results, list):
            results = pd.DataFrame(results)
        if not isinstance(results, pd.DataFrame) or results.empty:
            return pd.DataFrame(columns=self.columns)
        for col in self.columns:
            if col not in results.columns:
                results[col] = pd.NA
        return results.loc[:, self.columns].reset_index(drop=True)

    def result_to_csv(self, results, file, concat=False):
        old = self.get_tuned_gemm_list(file)
        for col in self.columns:
            if col not in old.columns:
                old[col] = pd.NA
        if results is None or results.empty:
            if not os.path.exists(file):
                pd.DataFrame(columns=self.columns).to_csv(file, index=False)
            return
        results = results.loc[:, self.columns].copy()
        self.success = pd.concat([self.success, results], ignore_index=True)
        old["tie"] = old["tie"].fillna("").astype(str)
        results["tie"] = results["tie"].fillna("").astype(str)
        merged = self.update_tunedf(old, results)
        ordered = [c for c in TUNED_COLUMNS if c in merged.columns]
        ordered.extend(c for c in merged.columns if c not in ordered)
        merged[ordered].to_csv(file, index=False)

    def run_config(self, args):
        gfx, cu = self.get_gfx(), self.get_cu_num()
        df = self.untunedf
        if df is None or df.empty:
            return []
        opts = _with_draw(getattr(self, "_opts", None) or _ArgBox(args), 2)
        groups = []
        for raw in df.to_dict("records"):
            key = dict(zip(LOOKUP_KEYS, key_tuple({**raw, "gfx": gfx, "cu_num": cu})))
            text = _nan_is_empty(raw.get("samples"), "") or _nan_is_empty(
                raw.get("dists"), ""
            )
            specs = []
            for spec in text.split(";"):
                if not spec:
                    continue
                p = parse_spec(spec)
                if p is None:
                    p = (spec, key["rows_lo"], key["width_lo"])
                dist = os.path.abspath(p[0]) if is_sample_path(p[0]) else p[0]
                specs.append(format_spec(dist, p[1], p[2]))
            # A tuned row with no router time was written because the router's
            # choice failed while tuning (see `decide`).
            fallback = (
                "backend" in raw and _nan_is_empty(raw.get("us_default"), "") == ""
            )
            groups.append({"key": key, "specs": specs, "fallback": fallback})
        verifiable = [g for g in groups if g["specs"]]
        res = iter(
            run_isolated(
                compare_group,
                [{"g": g, "opts": opts} for g in verifiable],
                self._devices(args),
                args.timeout,
                quiet=not getattr(args, "verbose", False),
            )
        )
        results = []
        for g in groups:
            label = band_label(g["key"])
            if not g["specs"]:
                why = "the row lists no samples (dists), so it cannot be verified"
                print(f"{label}\n  ERROR: {why}", flush=True)
                results.append(
                    {"shape": label, "e2e_us": -1, "status": f"error: {why}"}
                )
                continue
            r = next(res)
            if r[0] != "ok":
                why = _failure_text(r)
                print(f"{label}\n  ERROR: {why}", flush=True)
                results.append(
                    {"shape": label, "e2e_us": -1, "status": f"error: {why}"}
                )
                continue
            line, status, tu = summarize_run_config(label, r[1])
            print(line, flush=True)
            results.append({"shape": label, "e2e_us": tu, "status": status})
        self.run_config_failed = any(r["status"] != "ok" for r in results)
        return results

    def tune_summary(self, status):
        tuning_time = round(time.time() - getattr(self, "tune_start_time", 0), 4)
        logger.info("============= Tuning results Summary: ==============")
        logger.info(
            f"Tuning {status}. tune {len(self._groups)} bands, wrote "
            f"{self._written} row(s), total tuning time is {tuning_time} seconds"
        )
        if not self.success.empty:
            logger.info("Successfully tuned shapes:")
            print(self.success, flush=True)
        if self._written and self._out_file:
            out = os.path.abspath(self._out_file)
            print(
                f"  verify: python3 {os.path.abspath(__file__)} --run_config {out}\n"
                f"  use:    export AITER_CONFIG_TOPK_SELECT={out}",
                flush=True,
            )
        profile = getattr(getattr(self, "_args", None), "profile_file", "")
        if profile:
            print(f"  every measurement: {profile}", flush=True)
        if not self.failed.empty:
            logger.info("Failed shapes:")
            print(self.failed, flush=True)
            logger.error(
                "\033[91m[Tuning not Finished]\033[0m some bands have no backend "
                "that passed; see the per-band blocks above"
            )
            sys.exit(1)


if __name__ == "__main__":
    tuner = TopkSelectTuner()
    args = tuner.parse_args()
    tuner.run(args)
    if args.run_config and tuner.run_config_failed:
        sys.exit(1)

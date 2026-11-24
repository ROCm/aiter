#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the launch knobs of the RDNA attention kernels, and apply them.

Decode, per (Hq, Hkv, D, window, M): every point of the knob grid is checked
against a float reference and timed over --contexts; the finalists are timed
against the current row in an interleaved A/B. A point may lose at most
loss_limit(S) at any context, and wins on the mean speedup weighted by
sqrt(S / 128) if above --min-gain. With --batch, the batched build, timed
over batch specs instead of contexts.

Prefill (--prefill), per (Hq, Hkv, D, window): one row per step of M, the
closest to the roofline in an interleaved A/B among the finalists that lose at
most --prefill-loss-limit to the current row at any cell.

Prefill batches (--prefill-batch): per cell, the build and head group.

`--apply` writes accepted decisions into the table of the GPU family they were
measured on in csrc/kernels/rdna_attn/rdna_*_configs.h, refusing those
measured on another kernel source or row, and removes the built
module_rdna_attn* so they are rebuilt from the new tables.

    python op_tests/op_benchmarks/rdna_attn/tune.py \\
        --hq 32 --hkv 8 --head-dim 128 --m 1 \\
        --points points.jsonl --decisions decisions.jsonl
"""

import argparse
import functools
import gc
import itertools
import json
import math
import os
import platform
import statistics
import sys
import re
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import utils  # noqa: E402

CONTEXTS = [128, 512, 1024, 4096, 8192, 16384, 32768]
LONG_S, LONG_TARGET = 16384, 0.90
KNOBS = utils.KNOBS
# Relative error bound per dtype, for decode.
RTOL = {"fp16": 1e-3, "bf16": 8e-3}
# Absolute error bound per dtype against the float reference, for prefill.
PREFILL_ATOL = {"fp16": 5e-3, "bf16": 3e-2}

# Search space; Hkv * row_groups * max_segments is capped at MAX_WG.
GRID = {
    "waves": (2, 4, 8),
    "max_segments": (1, 2, 3, 4, 5, 8),
    "min_segment_blocks": (1, 2, 4),
    "pipe": ("none", "prefetch", "v_in_lds"),
}
DOT_GRID = {
    "waves": (2, 4, 8),
    "max_segments": (1, 2, 3, 4, 5, 8),
}
MAX_WG = 64
# A point stuck for HANG_S is recorded as a hang and the process exits with
# HANG_RC; rerun with --reuse.
HANG_S = 240
HANG_RC = 75
# Collect garbage every GC_EVERY points.
GC_EVERY = 25


def dspl_rule(d):
    """The kernel's default HEAD_DIM_SPLIT."""
    return d // 128 if d >= 256 else 1


def grid(hq, hkv, d, m, batch=False):
    """Valid, distinct knob sets for one configuration; no dot product for a
    batch."""
    gqa = hq // hkv
    rule = dspl_rule(d)
    out = []
    # Row groups split the gqa * m rows evenly.
    rows = gqa * m
    for row_groups in (r for r in range(1, rows + 1) if rows % r == 0):
        for waves, max_segments, pipe, min_segment_blocks in itertools.product(
            GRID["waves"],
            GRID["max_segments"],
            GRID["pipe"],
            GRID["min_segment_blocks"],
        ):
            if hkv * row_groups * max_segments > MAX_WG:
                continue
            if max_segments == 1 and min_segment_blocks != 1:
                continue
            for head_dim_split in sorted({rule, 2 * rule}):
                lane_dims = -(-d // 16) // head_dim_split
                if (
                    lane_dims - lane_dims % 2 not in (2, 4, 6, 8)
                    or waves % head_dim_split
                ):
                    continue
                # As the kernel's asserts: whole chunks per part, an odd lane
                # slice unsplit, the score exchange inside a wave's K tile.
                row_tiles = -(-(rows // row_groups) // 16)
                if (
                    -(-d // 16) % head_dim_split
                    or (lane_dims % 2 and head_dim_split != 1)
                    or (
                        head_dim_split > 1
                        and row_tiles * 1024 > 16 * ((d // head_dim_split) * 2 + 16)
                    )
                ):
                    continue
                out.append(
                    {
                        "waves": waves,
                        "row_groups": row_groups,
                        "max_segments": max_segments,
                        "min_segment_blocks": min_segment_blocks,
                        "prefetch": int(pipe == "prefetch"),
                        "v_in_lds": int(pipe == "v_in_lds"),
                        # Write the kernel's default head_dim_split as 0.
                        "head_dim_split": (
                            0 if head_dim_split == rule else head_dim_split
                        ),
                    }
                )
    # The dot path needs whole dot2 pairs per lane (not D=96).
    if not batch and (8 if d == 128 else max(2, d // 32)) % 2 == 0:
        for waves, max_segments in itertools.product(*DOT_GRID.values()):
            out.append({"dot_product": 1, "waves": waves, "max_segments": max_segments})
    return out


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def rank(contexts, pct):
    # Unserved cells are left out.
    served = [(s, c) for s, c in zip(contexts, pct) if c > 0]
    long_ = [c for s, c in served if s >= LONG_S]
    return (
        min(min(long_), LONG_TARGET) if long_ else LONG_TARGET,
        geomean([c for _, c in served]) if served else 0.0,
    )


def cell_ratio(ref_us, us):
    """Speedup over the reference at one cell; inf (unserved) is neutral for
    both, a loss for the point alone."""
    if math.isinf(ref_us):
        return 1.0
    return ref_us / us


def loss_limit(s):
    return min(0.05, max(0.01, 0.05 - 0.005 * (math.log2(s) - 7)))


def weighted(contexts, ratio):
    w = [math.sqrt(s / 128) for s in contexts]
    return sum(a * b for a, b in zip(w, ratio)) / sum(w)


def within_limits(contexts, ratio):
    return all(r >= 1 - loss_limit(s) for s, r in zip(contexts, ratio))


# Subspaces whose best point is also a finalist.
PINS = {
    "no_dot": lambda k, r: not k["dot_product"],
    "only_dot": lambda k, r: bool(k["dot_product"]),
    "no_prefetch": lambda k, r: not k["prefetch"],
    "no_v_in_lds": lambda k, r: not k["v_in_lds"],
    "no_head_dim_split": lambda k, r: k["head_dim_split"] in (0, r),
    "min_segment_blocks_1": lambda k, r: (
        k["min_segment_blocks"] == 1 or k["dot_product"]
    ),
    "max_segments_pow2": lambda k, r: k["max_segments"] & (k["max_segments"] - 1) == 0,
    "no_waves_2": lambda k, r: k["waves"] != 2,
    "simple": lambda k, r: (
        not (k["dot_product"] or k["prefetch"] or k["v_in_lds"])
        and k["head_dim_split"] in (0, r)
        and k["min_segment_blocks"] == 1
    ),
}


def canonical(k, d):
    """The knobs, normalized so equivalent builds compare equal."""
    if k["dot_product"]:
        return {n: k[n] for n in ("dot_product", "waves", "max_segments")}
    rule = dspl_rule(d)
    return {
        **k,
        "min_segment_blocks": 1 if k["max_segments"] == 1 else k["min_segment_blocks"],
        "head_dim_split": 0 if k["head_dim_split"] == rule else k["head_dim_split"],
    }


def knobs_of(v):
    return {k: getattr(v, k) for k in KNOBS}


def label(k):
    if k["dot_product"]:
        return f"dot waves={k['waves']} max_segments={k['max_segments']}"
    extra = [f"{n}={k[n]}" for n in ("head_dim_split", "prefetch", "v_in_lds") if k[n]]
    head = " ".join(
        f"{n}={k[n]}"
        for n in ("waves", "row_groups", "max_segments", "min_segment_blocks")
    )
    return " ".join([head, *extra])


# -------------------------------------------------------------------- checks --


def versions() -> dict:
    """What a record was measured with."""
    import importlib.metadata

    import torch

    try:
        aiter = importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        aiter = "unknown"
    return {"aiter": aiter, "torch": torch.__version__, "host": platform.node()}


def check_batch(module, v, repeat=2):
    """Worst relative error of a batched build over a long, a short and a
    padded sequence; the padded rows must stay unwritten."""
    import torch

    from aiter.ops.rdna_attn import make_scratch
    from aiter.ops.torch_ref.rdna_attn import paged_attention

    base = max(1040, v.window + 64, v.page_size + 1000)
    lens = [base, 48, 0]
    torch.manual_seed(0)
    width = -(-base // v.page_size)
    kv = (
        torch.randn(
            len(lens) * width,
            v.num_kv_heads,
            v.page_size,
            2 * v.head_size,
            device="cuda",
            dtype=v.dtype,
        )
        * 0.5
    )
    table = torch.arange(len(lens) * width, device="cuda", dtype=torch.int32)
    table = table.view(len(lens), width)
    q = (
        torch.randn(
            len(lens) * v.max_query_len,
            v.num_q_heads,
            v.head_size,
            device="cuda",
            dtype=v.dtype,
        )
        * 0.5
    )
    sl = torch.tensor(lens, device="cuda", dtype=torch.int32)
    scratch = make_scratch(v, torch.device("cuda"), max_seqs=len(lens))
    worst = 0.0
    for _ in range(repeat):
        out = torch.full_like(q, 7.0)
        module.decode_attn(q, kv, table, out, *scratch, sl, v.head_size**-0.5)
        torch.cuda.synchronize()
        for i, s in enumerate(lens):
            rows = out[i * v.max_query_len : (i + 1) * v.max_query_len].float()
            if s == 0:
                if not (rows == 7.0).all():
                    return float("inf")
                continue
            if not torch.isfinite(rows).all():
                return float("inf")
            ref = paged_attention(
                q[i * v.max_query_len : (i + 1) * v.max_query_len],
                kv,
                s,
                v.window,
                block_table=table[i],
            )
            err = ((rows - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            worst = max(worst, err)
    return worst


def check(module, v, repeat=2):
    """Worst relative error over several contexts, each on new inputs and a
    NaN-filled output, launched `repeat` times on one scratch.
    """
    import torch

    from aiter.ops.rdna_attn import make_scratch
    from aiter.ops.torch_ref.rdna_attn import paged_attention

    worst = 0.0
    scratch = make_scratch(v, torch.device("cuda"))
    base = max(1040, v.window + 64, v.page_size + 1000)
    for seed, s in enumerate((48, base, base + 1000, 300)):
        torch.manual_seed(seed)
        nb = -(-s // v.page_size)
        kv = (
            torch.randn(
                nb,
                v.num_kv_heads,
                v.page_size,
                2 * v.head_size,
                device="cuda",
                dtype=v.dtype,
            )
            * 0.5
        )
        q = (
            torch.randn(
                v.max_query_len,
                v.num_q_heads,
                v.head_size,
                device="cuda",
                dtype=v.dtype,
            )
            * 0.5
        )
        table = torch.arange(nb, device="cuda", dtype=torch.int32)
        ref = paged_attention(q, kv, s, v.window)
        sl = torch.tensor([s], device="cuda", dtype=torch.int32)
        out = torch.empty_like(q)
        for _ in range(repeat):
            out.fill_(float("nan"))
            module.decode_attn(q, kv, table, out, *scratch, sl, v.head_size**-0.5)
            torch.cuda.synchronize()
            if not torch.isfinite(out).all():
                return float("inf")
            err = ((out.float() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
            worst = max(worst, err)
    return worst


# -------------------------------------------------------------------- tuning --


def tune(a) -> None:
    import harness
    import jit

    from aiter.ops.rdna_attn import KernelVariant, variant_for

    jit.install()
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, m, win, bs = a.hq, a.hkv, a.head_dim, a.m, a.window, a.block_size
    batch = list(a.batch or [])
    # Contexts, or batch specs weighted as their longest sequence.
    cells = batch or list(a.contexts)
    if batch:
        from batch_spec import parse_batch_spec

        for spec in batch:
            if any(r.q_len != m for r in parse_batch_spec(spec)):
                raise SystemExit(f"{spec}: every sequence of a batch takes M={m}")
        ctxs = [max(r.kv_len for r in parse_batch_spec(spec)) for spec in batch]
    else:
        ctxs = cells
    tag = (
        f"{hq}/{hkv}/{d} M={m}"
        + (f" w{win}" if win else "")
        + (" batch" if batch else "")
        + f" {a.dtype}"
    )
    fixed = {
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "m": m,
        "window": win,
        "dtype": a.dtype,
        "block_size": bs,
        "source": jit.source_digest(),
        "family": gpu_family(),
        "contexts": cells,
        "batch": bool(batch),
    }
    meta = versions()
    if batch:
        roof = [
            utils.spec_roofline(spec, hq, hkv, d, block_size=bs, window=win)[0]
            for spec in batch
        ]
    else:
        roof = [
            utils.roofline_us(hq, hkv, d, m, s, block_size=bs, window=win) for s in ctxs
        ]

    def variant(knobs):
        return KernelVariant(
            d,
            hq,
            hkv,
            m,
            bs,
            1,
            **knobs,
            dtype=dtype,
            window=win,
            batched=int(bool(batch)),
        )

    def decode(v, cell):
        """(us, spread %) of m tokens at context `cell`, or of batch spec `cell`,
        on v; (inf, 0) for a batch the decode kernel does not serve."""
        with jit.override(knobs_of(v)):
            r = harness.run(
                cell if batch else f"q{m}s{cell}",
                hq,
                hkv,
                d,
                dtype=dtype,
                block_size=bs,
                window=win,
                kernels=jit,
            )
        progress["t"] = time.time()
        if batch and r.path == "unserved":
            return math.inf, 0.0
        if r.path != "kernel":
            raise RuntimeError(f"ran {r.path} at {cell}: {r.rejected}")
        return r.median_us, r.spread_pct

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    known: dict[str, dict] = {}
    if a.reuse and Path(a.points).exists():
        for line in Path(a.points).read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # truncated line
                continue
            if all(rec.get(k) == v for k, v in fixed.items()):
                known[pid(rec["knobs"])] = rec

    current = variant_for(hq, hkv, d, m, bs, 1, win, dtype, bool(batch))
    raw = (
        [json.loads(x) for x in Path(a.candidates).read_text().splitlines() if x]
        if a.candidates
        else grid(hq, hkv, d, m, bool(batch))
    )
    cands: dict = {} if current is None else {current: knobs_of(current)}
    space = set()
    for k in raw:
        try:
            v = variant(k)
        except (TypeError, ValueError):
            continue
        cands.setdefault(v, knobs_of(v))
        space.add(pid(canonical(knobs_of(v), d)))
    outside = current is not None and pid(canonical(knobs_of(current), d)) not in space
    items = list(cands.items())[: a.limit or None]
    todo = [v for v, k in items if pid(k) not in known]
    t0 = time.time()
    # Reused points too: any may be a finalist.
    jit.precompile([v for v, _ in items])
    jit.seal()
    print(
        f"# {tag}: {len(items)} points, {len(items) - len(todo)} reused, "
        f"built in {time.time() - t0:.0f} s",
        flush=True,
    )
    time.sleep(5)  # let clocks settle after the build

    # Watchdog for hung points.
    progress: dict = {"t": time.time(), "rec": None}

    def watchdog():
        while True:
            time.sleep(5)
            rec = progress["rec"]
            if rec is not None and time.time() - progress["t"] > HANG_S:
                rec = {
                    **rec,
                    "status": "hang",
                    "error": f"no progress for {HANG_S} s (GPU hang)",
                }
                with open(a.points, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                print(f"# HANG: {rec['knobs']}", flush=True)
                os._exit(HANG_RC)

    threading.Thread(target=watchdog, daemon=True).start()
    tol = RTOL[a.dtype]
    scored = []
    with open(a.points, "a") as fh:
        for i, (v, k) in enumerate(items):
            rec = known.get(pid(k))
            if rec is None:
                rec = {
                    **fixed,
                    "knobs": k,
                    "current": v == current,
                    **meta,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                progress["t"], progress["rec"] = time.time(), dict(rec)
                try:
                    mod = jit.load(v)
                    if mod is None:
                        raise RuntimeError(jit.failure(v))
                    rec["max_rel"] = err = (check_batch if batch else check)(mod, v)
                    if err > tol:
                        rec["status"] = "wrong"
                    else:
                        us, spread = zip(*(decode(v, c) for c in cells))
                        if all(math.isinf(u) for u in us):
                            raise RuntimeError("unserved at every batch")
                        rec.update(
                            status="ok",
                            us=[round(u, 3) for u in us],
                            spread_pct=[round(x, 2) for x in spread],
                            pct_roof=[round(r / u, 4) for r, u in zip(roof, us)],
                        )
                except Exception as e:  # noqa: BLE001 - recorded, the search goes on
                    rec["status"] = "build" if "does not build" in str(e) else "error"
                    rec["error"] = f"{type(e).__name__}: {e}"[:400]
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                known[pid(k)] = rec
                progress["rec"] = None
                # Free the batched scratch the layers share.
                harness.clear_scratch()
            if (i + 1) % GC_EVERY == 0 or rec.get("status") == "error":
                gc.collect()
            if rec.get("status") == "ok":
                scored.append((rank(ctxs, rec["pct_roof"]), v, k))
            if (i + 1) % 100 == 0:
                print(
                    f"#   {i + 1}/{len(items)} points, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )

    # Rank against the current row, or by %roof without one.
    cur = None if current is None else known.get(pid(knobs_of(current)))
    ref_us = cur["us"] if cur is not None and cur.get("status") == "ok" else None
    if ref_us is not None:
        scored = [
            ((within_limits(ctxs, r), weighted(ctxs, r)), v, k)
            for _, v, k in scored
            for r in [[cell_ratio(b, u) for b, u in zip(ref_us, known[pid(k)]["us"])]]
        ]
    scored.sort(key=lambda x: x[0], reverse=True)
    rule = dspl_rule(d)
    finalists: dict = {}
    for _, v, k in scored[: a.top]:
        finalists.setdefault(v, (k, ["top"]))
    for name, pred in PINS.items():
        best_in = next((x for x in scored if pred(x[2], rule)), None)
        if best_in is not None:
            finalists.setdefault(best_in[1], (best_in[2], []))[1].append(name)
    if ref_us is not None:
        # Also the best points that lose at no context.
        safe = [
            x
            for x in scored
            if x[1] != current
            and all(b >= u for b, u in zip(ref_us, known[pid(x[2])]["us"]))
        ]
        for _, v, k in safe[:3]:
            finalists.setdefault(v, (k, []))[1].append("safe")
    finalists.pop(current, None)

    # Interleaved A/B over all contexts.
    runs = [(v, k, tags) for v, (k, tags) in finalists.items()]
    if current is not None:
        runs.insert(0, (current, knobs_of(current), ["current"]))
    # Points reused from --points were not built in this run.
    if jit.precompile([v for v, _, _ in runs]):
        time.sleep(5)  # let clocks settle after the build
    times = [{c: [] for c in cells} for _ in runs]
    for _ in range(a.confirm_rounds):
        for c in cells:
            for (v, _, _), t in zip(runs, times):
                try:
                    t[c].append(decode(v, c)[0])
                except Exception:  # noqa: BLE001
                    t[c].append(float("inf"))
    med = [[statistics.median(t[c]) for c in cells] for t in times]
    base = med[0] if current is not None else None
    # A current row that failed at every cell is no baseline: candidates are
    # ranked against the roofline and none is accepted over it.
    base_ok = base is None or any(map(math.isfinite, base))
    results = []
    for (v, k, tags), us, t in zip(runs, med, times):
        if "current" in tags:
            continue
        # Against the row, or roof without one.
        ratio = (
            [cell_ratio(b, u) for b, u in zip(base, us)]
            if base and base_ok
            else [r / u for r, u in zip(roof, us)]
        )
        score = weighted(ctxs, ratio)
        accepted = base is None or (
            base_ok and score > a.min_gain and within_limits(ctxs, ratio)
        )
        results.append(
            {
                "knobs": k,
                "for": tags,
                "us": us,
                "samples": [t[c] for c in cells],
                "ratio": ratio,
                "score": score,
                "worst": min(ratio),
                "accepted": accepted,
            }
        )
    ok = [r for r in results if r["accepted"]]
    best = max(ok, key=lambda r: r["score"]) if ok else None
    replaced = best is None and outside and bool(results)
    if replaced:
        best = max(results, key=lambda r: (within_limits(ctxs, r["ratio"]), r["score"]))
    winner = None if current is None else knobs_of(current)
    if best:
        winner = best["knobs"]
    decision = {
        **fixed,
        "current": None if current is None else knobs_of(current),
        "current_us": base,
        "current_samples": None if current is None else [times[0][c] for c in cells],
        "finalists": results,
        "accepted": best is not None,
        # The row is outside the search space and nothing beat it.
        "replaced": replaced,
        "winner": winner,
        "score": best["score"] if best else 1.0,
        "points": len(items),
        "ok_points": len(scored),
        **meta,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.decisions, "a") as out:
        out.write(json.dumps(decision) + "\n")
    if best:
        print(
            f"{tag}: {'REPLACE (row outside the grid)' if replaced else 'ACCEPT'} "
            f"{label(best['knobs'])}  score {best['score']:.3f}, "
            f"worst cell {best['worst']:.3f}",
            flush=True,
        )
    elif current is not None:
        print(f"{tag}: keep current {label(knobs_of(current))}", flush=True)
    else:
        print(f"{tag}: no point measured correct", flush=True)
    for r in results:
        print(
            f"   {','.join(r['for']):<24} score {r['score']:.3f} worst "
            f"{r['worst']:.3f}  {label(r['knobs'])}",
            flush=True,
        )


# ------------------------------------------------------------------- prefill --

# Steps of M; a row serves M from its step up to the next.
STEPS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
PREFIXES = (0, 1024, 16384)
# A step keeps the previous step's row when within this factor of the winner.
PREFILL_SLACK = 1.02
# Most a step's winner may lose at any cell against the current row.
PREFILL_LOSS_LIMIT = 0.02


def prefill_grid(d):
    """Prefill knob sets for head size d."""
    split = max(1, d // 128)
    if d <= 128:
        axes = {
            "waves": (4, 8),
            "key_tile": (16, 32, 64),
            "v_t": (0, 1),
            "key_waves": (1, 2),
            "double_buffer": (0,),
        }
    elif d == 256:
        axes = {
            "waves": (4, 8, 16),
            "key_tile": (16, 32),
            "v_t": (1,),
            "key_waves": (1, 2),
            "double_buffer": (0, 1),
        }
    else:
        axes = {
            "waves": (8, 16),
            "key_tile": (16,),
            "v_t": (1,),
            "key_waves": (1, 2),
            "double_buffer": (0, 1),
        }
    axes |= {"max_segments": (1, 4, 16), "head_group": (1, 2, 4, 8)}
    out = []
    for values in itertools.product(*axes.values()):
        k = dict(zip(axes, values), head_dim_split=split)
        if (
            k["waves"] % (split * k["key_waves"])
            or k["key_tile"] // 16 % k["key_waves"]
            # An odd lane slice (D=48, 80) or a padded one (D=72) needs V
            # transposed.
            or ((-(-d // 16) // split % 2 or d % 16) and not k["v_t"])
        ):
            continue
        out.append(k)
    return out


def prefill_knobs_of(v):
    return {k: getattr(v, k) for k in utils.PREFILL_KNOBS}


def prefill_label(k):
    return " ".join(f"{n}={k[n]}" for n in utils.PREFILL_KNOBS)


def weighted_speedup(weights, ratio):
    """Weighted geometric mean of the speedups (other / ours)."""
    return math.exp(sum(w * math.log(r) for w, r in zip(weights, ratio)) / sum(weights))


def prefill_weights(hq, d, win, cells):
    flops = [utils.attention_flops(hq, d, m, m + p, win) for m, p in cells]
    total = sum(flops)
    return {c: math.sqrt(f / total) for c, f in zip(cells, flops)}


def choose_prefill_steps(cells, weight, med, roof, finalists, current, loss_limit):
    """Per step of M, the finalist to keep: among those losing at most
    `loss_limit` at any cell against the current row, the closest to the
    roofline, weighted by sqrt(FLOPs); the previous step's winner stays within
    PREFILL_SLACK.

    `med` maps (point id, cell) to us and `roof` a cell to its roofline;
    `finalists` and `current` map M to point ids. Returns (M, winner, scored)
    per step, `scored` mapping a point id to (weighted fraction of the
    roofline, the fraction per cell).
    """
    out, prev = [], None
    for m in sorted({c[0] for c in cells}):
        sc = [c for c in cells if c[0] == m]
        cur = current.get(m)
        allowed = [
            k
            for k in sorted(finalists[m])
            if cur not in finalists[m]
            or all(med[(cur, c)] / med[(k, c)] >= 1 - loss_limit for c in sc)
        ]
        w = [weight[c] for c in sc]
        scored = {}
        for k in allowed:
            ratio = [roof[c] / med[(k, c)] for c in sc]
            scored[k] = (weighted_speedup(w, ratio), ratio)
        best = max(scored, key=lambda k: scored[k][0])
        if prev in scored and scored[prev][0] * PREFILL_SLACK >= scored[best][0]:
            best = prev
        out.append((m, best, scored))
        prev = best
    return out


def tune_prefill(a) -> None:
    import harness
    import jit
    import torch

    from aiter.ops import rdna_attn as R
    from aiter.ops.torch_ref.rdna_attn import paged_attention

    jit.install()
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, win, bs = a.hq, a.hkv, a.head_dim, a.window, a.block_size
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + f" prefill {a.dtype}"
    ms = sorted(a.steps)
    cells = [(m, p) for m in ms for p in a.prefixes]
    weight = prefill_weights(hq, d, win, cells)
    itemsize = torch.finfo(dtype).bits // 8
    roof = {
        (m, p): utils.roofline(hq, hkv, d, m, m + p, itemsize, bs, win, True)[0]
        for m, p in cells
    }
    tol = PREFILL_ATOL[a.dtype]
    fixed = {
        "kind": "prefill",
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "window": win,
        "dtype": a.dtype,
        "block_size": bs,
        "source": jit.source_digest(prefill=True),
        "family": gpu_family(),
        "cells": [list(c) for c in cells],  # as read back from --points
    }
    meta = versions()

    def variant(knobs):
        return R.PrefillVariant(d, hq, hkv, win, bs, STEPS[0], **knobs, dtype=dtype)

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    current = {m: R.prefill_variant_for(hq, hkv, d, bs, 1, win, dtype, m) for m in ms}
    raw = (
        [json.loads(x) for x in Path(a.candidates).read_text().splitlines() if x]
        if a.candidates
        else prefill_grid(d)
    )
    cands: dict[str, R.PrefillVariant] = {}
    for k in [*raw, *(prefill_knobs_of(v) for v in current.values() if v)]:
        try:
            v = variant(k)
        except (TypeError, ValueError):
            continue
        # By every knob, the ones the grid leaves at their defaults included.
        cands.setdefault(pid(prefill_knobs_of(v)), v)
    items = list(cands.items())[: a.limit or None]
    known: dict[str, dict] = {}
    if a.reuse and Path(a.points).exists():
        for line in Path(a.points).read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(rec.get(k) == v for k, v in fixed.items()):
                known[pid(rec["knobs"])] = rec
    t0 = time.time()
    jit.precompile([v for p, v in items if p not in known])
    jit.seal()
    print(
        f"# {tag}: {len(items)} points, built in {time.time() - t0:.0f} s", flush=True
    )
    time.sleep(5)  # let clocks settle after the build

    dev = torch.device("cuda")

    data: dict = {}

    def inputs(m, p, seed):
        """A cell's Q, cache, block table, length and float reference."""
        if (m, p) not in data:
            torch.manual_seed(seed)
            s = m + p
            pages = -(-s // bs)
            kv = (torch.randn(pages, hkv, bs, 2 * d, device=dev) * 0.5).to(dtype)
            table = torch.randperm(pages, device=dev).to(torch.int32)
            q = (torch.randn(m, hq, d, device=dev) * 0.5).to(dtype)
            sl = torch.tensor([s], device=dev, dtype=torch.int32)
            ref = paged_attention(q, kv, s, win, block_table=table)
            data[m, p] = (q, kv, table, sl, ref)
        return data[m, p]

    scratch = {}

    def op(module, v, q, kv, table, sl, out):
        key = v.scratch_rows(dev)
        if key not in scratch:
            scratch[key] = R.make_prefill_scratch([v], dev)
        s = int(sl.item())
        return lambda: module.prefill_attn(
            q, kv, table, out, *scratch[key], sl, s, d**-0.5
        )

    # 1. Check and time every point on the op.
    with open(a.points, "a") as fh:
        for i, (key, v) in enumerate(items):
            if key in known:
                continue
            rec = {
                **fixed,
                "knobs": prefill_knobs_of(v),
                **meta,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            module = jit.load(v)
            if module is None:
                rec.update(status="build", error=(jit.failure(v) or "")[:400])
            else:
                us = []
                for seed, (m, p) in enumerate(cells):
                    q, kv, table, sl, ref = inputs(m, p, seed)
                    out = torch.full_like(q, float("nan"))
                    run = op(module, v, q, kv, table, sl, out)
                    run()
                    torch.cuda.synchronize()
                    err = (out.float().nan_to_num(1e9) - ref).abs().max()
                    if not err.item() <= tol:
                        rec.update(status="wrong", cell=[m, p], err=err.item())
                        break
                    us.append(harness.median_us(run))
                else:
                    rec.update(status="ok", us=[round(u, 2) for u in us])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            known[key] = rec
            if (i + 1) % 20 == 0:
                print(
                    f"#   {i + 1}/{len(items)} points, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    ok = {
        k: dict(zip(map(tuple, cells), r["us"]))
        for k, r in known.items()
        if r.get("status") == "ok"
    }
    if not ok:
        print(f"{tag}: no point measured correct", flush=True)
        return

    # 2. Time the finalists of each step as a server runs them.
    def step_cells(m):
        return [c for c in cells if c[0] == m]

    finalists: dict[int, set[str]] = {}
    for m in ms:
        sc = step_cells(m)
        fastest = {c: min(t[c] for t in ok.values()) for c in sc}
        ranked = sorted(
            ok,
            key=lambda k: (
                -weighted_speedup(
                    [weight[c] for c in sc], [fastest[c] / ok[k][c] for c in sc]
                )
            ),
        )
        finalists[m] = set(ranked[: a.top])
        if current[m] is not None and pid(prefill_knobs_of(current[m])) in ok:
            finalists[m].add(pid(prefill_knobs_of(current[m])))
    for m, prev in zip(ms[1:], ms):
        finalists[m] |= finalists[prev]
    # Points reused from --points were not built in this run.
    if jit.precompile([cands[k] for m in ms for k in finalists[m]]):
        time.sleep(5)  # let clocks settle after the build

    def served(m, p, knobs):
        with jit.override(knobs):
            r = harness.run(
                f"q{m}s{m + p}",
                hq,
                hkv,
                d,
                dtype=dtype,
                block_size=bs,
                window=win,
                use_cuda_graphs=False,
                max_model_len=max(ms) * 2 + max(a.prefixes),
                kernels=jit,
            )
        if r.path != "prefill":
            raise RuntimeError(f"ran {r.path} at q{m}s{m + p}: {r.rejected}")
        return r.median_us

    times: dict = {}
    for _ in range(a.confirm_rounds):
        for m in ms:
            for c in step_cells(m):
                for k in sorted(finalists[m]):
                    times.setdefault((k, c), []).append(served(*c, dict(json.loads(k))))
    med = {key: statistics.median(t) for key, t in times.items()}

    # 3. Per step, the winner within the loss limit against the current row.
    cur_ids = {m: pid(prefill_knobs_of(v)) for m, v in current.items() if v is not None}
    steps = []
    for m, best, scored in choose_prefill_steps(
        cells, weight, med, roof, finalists, cur_ids, a.prefill_loss_limit
    ):
        sc = step_cells(m)
        cur = None if current[m] is None else prefill_knobs_of(current[m])
        steps.append(
            {
                "m": m,
                "current": cur,
                "winner": dict(json.loads(best)),
                "roof": scored[best][0],
                "worst": min(scored[best][1]),
                "finalists": [dict(json.loads(k)) for k in sorted(finalists[m])],
                "cells": {
                    f"{c[0]}:{c[1]}": {"ours": med[(best, c)], "roofline": roof[c]}
                    for c in sc
                },
            }
        )
        print(
            f"{tag} M>={m}: {prefill_label(steps[-1]['winner'])}  "
            f"{steps[-1]['roof'] * 100:.1f} % of the roofline, "
            f"worst cell {steps[-1]['worst'] * 100:.1f} %",
            flush=True,
        )
    decision = {
        **fixed,
        "current": {m: s["current"] for m, s in zip(ms, steps)},
        "steps": steps,
        # Every A/B run: the finalists of each step (the current row among
        # them), per cell.
        "ab": [
            {"cell": list(c), "knobs": dict(json.loads(who)), "samples": t}
            for (who, c), t in times.items()
        ],
        "roofline": [roof[c] for c in cells],
        "loss_limit": a.prefill_loss_limit,
        "accepted": True,
        **meta,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.decisions, "a") as out:
        out.write(json.dumps(decision) + "\n")


def reselect_prefill(path: str, out: str, loss_limit: float) -> None:
    """Choose the steps of stored prefill decisions again from their A/B
    samples, with `loss_limit`; no timing."""

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    with open(out, "a") as fh:
        for line in Path(path).read_text().splitlines():
            dec = json.loads(line)
            if dec.get("kind") != "prefill" or "ab" not in dec:
                continue
            cells = [tuple(c) for c in dec["cells"]]
            weight = prefill_weights(dec["hq"], dec["d"], dec["window"], cells)
            med = {
                (pid(x["knobs"]), tuple(x["cell"])): statistics.median(x["samples"])
                for x in dec["ab"]
            }
            roof = dict(zip(cells, dec["roofline"]))
            finalists = {s["m"]: {pid(k) for k in s["finalists"]} for s in dec["steps"]}
            current = {s["m"]: pid(s["current"]) for s in dec["steps"] if s["current"]}
            chosen = choose_prefill_steps(
                cells, weight, med, roof, finalists, current, loss_limit
            )
            for step, (m, best, scored) in zip(dec["steps"], chosen):
                sc = [c for c in cells if c[0] == m]
                step.update(
                    winner=dict(json.loads(best)),
                    roof=scored[best][0],
                    worst=min(scored[best][1]),
                    cells={
                        f"{c[0]}:{c[1]}": {"ours": med[(best, c)], "roofline": roof[c]}
                        for c in sc
                    },
                )
            dec["loss_limit"] = loss_limit
            fh.write(json.dumps(dec) + "\n")


# Non-causal (an encoder's segments, L = S): the sequence lengths tuned, the
# steps' own and the traced mm-encoder calls'.
NON_CAUSAL_CELLS = tuple(
    sorted({*STEPS, 320, 577, 784, 1200, 2040, 3200, 3520, 4292, 8192})
)
# Finer than STEPS: one row per size band of the encoders' calls.
NON_CAUSAL_STEPS = (*STEPS[:6], 768, 1024, 1536, 2048, 3072, 4096, 8192)
NON_CAUSAL_GRID = {
    "waves": (4, 8, 16),
    "key_tile": (16, 32, 64),
    "v_t": (0, 1),
    "key_waves": (1, 2),
    "double_buffer": (0, 1),
    "max_segments": (1, 4, 16),
    "head_group": (1, 2, 4, 8),
    "row_tiles": (1, 2),
}


def non_causal_grid(d):
    """Non-causal knob sets for head size d: the whole space, as the masking
    it drops changes the register pressure."""
    split = max(1, d // 128)
    out = []
    for values in itertools.product(*NON_CAUSAL_GRID.values()):
        k = dict(zip(NON_CAUSAL_GRID, values), head_dim_split=split)
        if (
            k["waves"] % (split * k["key_waves"])
            or k["key_tile"] // 16 % k["key_waves"]
            or ((-(-d // 16) // split % 2 or d % 16) and not k["v_t"])
            or (
                k["row_tiles"] > 1
                and (
                    split > 1
                    or k["key_waves"] > 1
                    or k["key_tile"] > 32
                    # compute2 reads V^T.
                    or not k["v_t"]
                )
            )
        ):
            continue
        out.append(k)
    return out


def tune_prefill_non_causal(a) -> None:
    """Prefill rows for one sequence of L = S tokens, every token seeing every
    key: per step of M, the knobs closest to the roofline over
    NON_CAUSAL_CELLS."""
    import harness
    import jit
    import torch

    from aiter.ops import rdna_attn as R
    from aiter.ops.torch_ref.rdna_attn import full_attention

    jit.install()
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, bs = a.hq, a.hkv, a.head_dim, a.block_size
    if a.window:
        raise SystemExit("a non-causal build has no window")
    tag = f"{hq}/{hkv}/{d} non-causal prefill {a.dtype}"
    cells = [m for m in NON_CAUSAL_CELLS if m >= STEPS[0]]
    flops = [4.0 * hq * d * m * m for m in cells]
    weight = {c: math.sqrt(f / sum(flops)) for c, f in zip(cells, flops)}
    roof = {m: utils.roofline_noncausal(hq, hkv, d, [m])[0] for m in cells}
    tol = PREFILL_ATOL[a.dtype]
    fixed = {
        "kind": "prefill",
        "causal": 0,
        "hq": hq,
        "hkv": hkv,
        "d": d,
        "window": 0,
        "dtype": a.dtype,
        "block_size": bs,
        "source": jit.source_digest(prefill=True),
        "family": gpu_family(),
        "cells": cells,
    }
    meta = versions()

    def variant(knobs):
        return R.PrefillVariant(
            d, hq, hkv, 0, bs, NON_CAUSAL_STEPS[0], **knobs, dtype=dtype, causal=0
        )

    def pid(knobs):
        return json.dumps(sorted(knobs.items()))

    current = {
        m: R.prefill_variant_for(hq, hkv, d, bs, 1, 0, dtype, m, causal=False)
        for m in NON_CAUSAL_STEPS
    }
    raw = (
        [json.loads(x) for x in Path(a.candidates).read_text().splitlines() if x]
        if a.candidates
        else non_causal_grid(d)
    )
    cands: dict[str, R.PrefillVariant] = {}
    for k in [*raw, *(prefill_knobs_of(v) for v in current.values() if v)]:
        try:
            v = variant(k)
        except (TypeError, ValueError):
            continue
        # By every knob, the ones the grid leaves at their defaults included.
        cands.setdefault(pid(prefill_knobs_of(v)), v)
    items = list(cands.items())[: a.limit or None]
    known: dict[str, dict] = {}
    if a.reuse and Path(a.points).exists():
        for line in Path(a.points).read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(rec.get(k) == v for k, v in fixed.items()):
                known[pid(rec["knobs"])] = rec
    t0 = time.time()
    # Reused points run again as finalists.
    jit.precompile([v for _, v in items])
    jit.seal()
    print(
        f"# {tag}: {len(items)} points, built in {time.time() - t0:.0f} s", flush=True
    )
    time.sleep(5)  # let clocks settle after the build

    dev = torch.device("cuda")
    data = {}

    def inputs(m):
        if m not in data:
            torch.manual_seed(m)
            q = (torch.randn(m, hq, d, device=dev) * 0.5).to(dtype)
            k = (torch.randn(m, hkv, d, device=dev) * 0.5).to(dtype)
            v = (torch.randn(m, hkv, d, device=dev) * 0.5).to(dtype)
            sl = torch.tensor([m], device=dev, dtype=torch.int32)
            data[m] = (q, k, v, sl, full_attention(q, k, v))
        return data[m]

    scratch = {}

    def op(module, var, m, out):
        q, k, v, sl, _ = inputs(m)
        key = var.scratch_rows(dev)
        if key not in scratch:
            scratch[key] = R.make_prefill_scratch([var], dev)
        return lambda: module.encoder_attn(q, k, v, out, *scratch[key], sl, m, d**-0.5)

    # 1. Check and time every point.
    with open(a.points, "a") as fh:
        for i, (key, v) in enumerate(items):
            if key in known:
                continue
            rec = {
                **fixed,
                "knobs": prefill_knobs_of(v),
                **meta,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            module = jit.load(v)
            if module is None:
                rec.update(status="build", error=(jit.failure(v) or "")[:400])
            else:
                us = []
                for m in cells:
                    out = torch.full_like(inputs(m)[0], float("nan"))
                    run = op(module, v, m, out)
                    run()
                    torch.cuda.synchronize()
                    err = (out.float().nan_to_num(1e9) - inputs(m)[4]).abs().max()
                    if not err.item() <= tol:
                        rec.update(status="wrong", cell=m, err=err.item())
                        break
                    us.append(harness.median_us(run))
                else:
                    rec.update(status="ok", us=[round(u, 2) for u in us])
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            known[key] = rec
            if (i + 1) % 20 == 0:
                print(
                    f"#   {i + 1}/{len(items)} points, "
                    f"{(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    ok = {
        k: dict(zip(cells, r["us"]))
        for k, r in known.items()
        if r.get("status") == "ok"
    }
    if not ok:
        print(f"{tag}: no point measured correct", flush=True)
        return

    def bucket(m):
        return max(x for x in NON_CAUSAL_STEPS if x <= m)

    step_cells = {s_: [c for c in cells if bucket(c) == s_] for s_ in NON_CAUSAL_STEPS}

    # 2. Finalists of each step, re-timed in turn.
    finalists: dict[int, set[str]] = {}
    for st in NON_CAUSAL_STEPS:
        sc = step_cells[st]
        fastest = {c: min(t[c] for t in ok.values()) for c in sc}
        ranked = sorted(
            ok,
            key=lambda k: -weighted_speedup(
                [weight[c] for c in sc], [fastest[c] / ok[k][c] for c in sc]
            ),
        )
        finalists[st] = set(ranked[: a.top])
        if current[st] is not None and pid(prefill_knobs_of(current[st])) in ok:
            finalists[st].add(pid(prefill_knobs_of(current[st])))
    for st, prev in zip(NON_CAUSAL_STEPS[1:], NON_CAUSAL_STEPS):
        finalists[st] |= finalists[prev]
    times: dict = {}
    for _ in range(a.confirm_rounds):
        for st in NON_CAUSAL_STEPS:
            for c in step_cells[st]:
                for k in sorted(finalists[st]):
                    out = torch.empty_like(inputs(c)[0])
                    run = op(jit.load(cands[k]), cands[k], c, out)
                    times.setdefault((k, c), []).append(harness.median_us(run))
    med = {key: statistics.median(t) for key, t in times.items()}

    # 3. Per step, the closest to the roofline; the previous step's winner
    # stays within PREFILL_SLACK.
    steps, prev = [], None
    for st in NON_CAUSAL_STEPS:
        sc = step_cells[st]
        w = [weight[c] for c in sc]
        scored = {}
        for k in finalists[st]:
            ratio = [roof[c] / med[(k, c)] for c in sc]
            scored[k] = (weighted_speedup(w, ratio), ratio)
        best = max(scored, key=lambda k: scored[k][0])
        if prev in scored and scored[prev][0] * PREFILL_SLACK >= scored[best][0]:
            best = prev
        cur = None if current[st] is None else prefill_knobs_of(current[st])
        steps.append(
            {
                "m": st,
                "current": cur,
                "winner": dict(json.loads(best)),
                "roof": scored[best][0],
                "worst": min(scored[best][1]),
                "cells": {
                    str(c): {"ours": med[(best, c)], "roofline": roof[c]} for c in sc
                },
            }
        )
        prev = best
        print(
            f"{tag} M>={st}: {prefill_label(steps[-1]['winner'])}  "
            f"{steps[-1]['roof'] * 100:.1f} % of the roofline, "
            f"worst cell {steps[-1]['worst'] * 100:.1f} %",
            flush=True,
        )
    decision = {
        **fixed,
        "current": {st: s_["current"] for st, s_ in zip(NON_CAUSAL_STEPS, steps)},
        "steps": steps,
        "accepted": True,
        **meta,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(a.decisions, "a") as out_:
        out_.write(json.dumps(decision) + "\n")


def apply_prefill(dec: dict, block_sizes: list[int]) -> int:
    """Write one prefill decision into the prefill configs."""
    import jit
    import torch

    from aiter.ops import rdna_attn as R

    hq, hkv, d, win = dec["hq"], dec["hkv"], dec["d"], dec["window"]
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + " prefill"
    if dec["source"] != jit.source_digest(prefill=True):
        print(f"{tag}: refused, measured on another kernel source")
        return 0
    causal = dec.get("causal", 1)
    family = dec.get("family") or gpu_family()
    lines, rows = read_configs(configs_h("prefill"), family, R.PREFILL_CONFIG_FIELDS)
    col = {f: i for i, f in enumerate(R.PREFILL_CONFIG_FIELDS)}
    dtype = utils.torch_dtype(dec["dtype"])
    for step in dec["steps"]:
        now = R.prefill_variant_for(
            hq, hkv, d, dec["block_size"], 1, win, dtype, step["m"], causal
        )
        if step["current"] != (None if now is None else prefill_knobs_of(now)):
            print(f"{tag}: refused, the rows changed since they were measured")
            return 0

    def mine(r):
        return (
            r[col["head_size"]],
            r[col["num_q_heads"]],
            r[col["num_kv_heads"]],
            r[col["window"]],
            r[col["causal"]],
        ) == (d, hq, hkv, win, causal)

    sizes = sorted({r[col["page_size"]] for r in rows if mine(r)}) or block_sizes
    kept = [r for r in rows if not mine(r)]
    # A decided step's winner serves M from its step up to the next step of
    # its grid (NON_CAUSAL_STEPS for a non-causal build); outside them the
    # entries stay.
    decided = {s["m"]: s["winner"] for s in dec["steps"]}
    grid = STEPS if causal else NON_CAUSAL_STEPS
    ends = dict(zip(grid, (*grid[1:], None)))
    written = 0
    for bf16, bs in itertools.product((0, 1), sizes):
        group = sorted(
            (
                r
                for r in rows
                if mine(r) and (r[col["dtype"]], r[col["page_size"]]) == (bf16, bs)
            ),
            key=lambda r: r[col["min_query_len"]],
        )
        starts = {r[col["min_query_len"]] for r in group}
        bounds = sorted(starts | set(decided) | {e for e in ends.values() if e})
        out = []
        for b in bounds:
            m = max((m for m in decided if m <= b), default=None)
            if m is not None and (ends[m] is None or b < ends[m]):
                knobs = decided[m]
            else:
                below = [r for r in group if r[col["min_query_len"]] <= b]
                if not below:
                    continue
                knobs = {k: below[-1][col[k]] for k in utils.PREFILL_KNOBS}
            if not out or out[-1][1] != knobs:
                out.append((b, knobs))
        for b, knobs in out:
            v = R.PrefillVariant(
                d,
                hq,
                hkv,
                win,
                bs,
                b,
                **knobs,
                dtype=torch.bfloat16 if bf16 else torch.float16,
                causal=causal,
            )
            kept.append(R.prefill_config_fields(v))
            written += 1
    write_configs(configs_h("prefill"), lines, kept, R.PREFILL_CONFIG_FIELDS, family)
    for m, knobs in sorted(decided.items()):
        print(f"{tag} M>={m}: {prefill_label(knobs)}")
    return written


# ------------------------------------------------------------- prefill batch --

# A cell's representative mean M and keys, by its lower bounds.
BATCH_M = {1: 4, 16: 32, 64: 128, 256: 512, 1024: 2048, 4096: 4096}
BATCH_KEYS = {0: 256, 512: 1024, 2048: 4096, 8192: 16384}
# Tokens of a cell's larger batch (the smaller has a quarter), within
# BATCH_KV_BYTES of KV.
BATCH_TOKENS = 4096
BATCH_KV_BYTES = 1 << 30
# Launches rotate over KV copies this large, so caches do not flatter small
# batches.
BATCH_WORKING_SET = 96 << 20


def prefill_batch_cells(window, kv_bytes_per_key, causal=True):
    """((min mean M, min keys), representative batches) of every cell.

    Non-causal (an encoder's segments, L = S): a batch's mean M and keys are
    both L, so only the cells holding some L >= PREFILL_MIN_M are reached;
    each takes batches of a total near BATCH_TOKENS and a quarter of it.
    """
    import numpy as np

    from aiter.ops import rdna_attn as R

    out = []
    if not causal:
        ms = (*R.PREFILL_BATCH_MEAN_M, 1 << 30)
        ks = (*R.PREFILL_BATCH_KEYS, 1 << 30)
        for i, m_lo in enumerate(R.PREFILL_BATCH_MEAN_M):
            for j, k_lo in enumerate(R.PREFILL_BATCH_KEYS):
                lo = max(m_lo, k_lo, R.PREFILL_MIN_M)
                hi = min(ms[i + 1], ks[j + 1], 16384)
                if lo >= hi:
                    continue
                length = min(hi - 1, max(lo, round(math.sqrt(lo * hi))))
                n = max(2, BATCH_TOKENS // length)
                specs = []
                for nseq in dict.fromkeys((n, max(2, n // 4))):
                    starts = np.arange(nseq + 1) * length
                    lens = np.full(nseq, length)
                    specs.append((starts, lens, f"{nseq}q{length}s{length}"))
                out.append(((m_lo, k_lo), specs))
        return out
    for i, m_lo in enumerate(R.PREFILL_BATCH_MEAN_M):
        m = BATCH_M[m_lo]
        for j, k_lo in enumerate(R.PREFILL_BATCH_KEYS):
            k_hi = (R.PREFILL_BATCH_KEYS + (float("inf"),))[j + 1]
            context = max(0, BATCH_KEYS[k_lo] - (m + 1) // 2)
            n = min(64, max(2, BATCH_TOKENS // m))
            n = max(2, min(n, BATCH_KV_BYTES // (kv_bytes_per_key * (m + context))))
            specs = []
            for nseq in dict.fromkeys((n, max(2, n // 4))):
                starts = np.arange(nseq + 1) * m
                lens = np.full(nseq, m + context)
                mean_m, keys = R.prefill_batch_features(starts, lens, window)
                if m_lo <= mean_m and k_lo <= keys < k_hi:
                    specs.append((starts, lens, f"{nseq}q{m}s{m + context}"))
            if specs:
                out.append(((m_lo, k_lo), specs))
    return out


def tune_prefill_batch(a) -> None:
    """Pick, per cell of a shape, the fastest prefill build and head group."""
    import jit
    import torch

    from aiter.ops import rdna_attn as R

    jit.install()
    dtype = utils.torch_dtype(a.dtype)
    hq, hkv, d, win, bs = a.hq, a.hkv, a.head_dim, a.window, a.block_size
    causal = not a.non_causal
    tag = (
        f"{hq}/{hkv}/{d}"
        + (f" w{win}" if win else "")
        + ("" if causal else " non-causal")
        + f" batch {a.dtype}"
    )
    dev = torch.device("cuda")
    rows = R.prefill_variants(hq, hkv, d, bs, win, dtype, causal)
    if not rows:
        raise SystemExit(f"{tag}: no prefill rows")
    # Rows differing in max_segments or head_group build the same kernel.
    builds: dict = {}
    for v in rows:
        k = prefill_knobs_of(v)
        k.pop("max_segments"), k.pop("head_group")
        builds.setdefault(json.dumps(sorted(k.items())), v)
    groups = [g for g in range(1, hkv + 1) if hkv % g == 0]
    jit.precompile(list(builds.values()))
    jit.seal()
    time.sleep(5)  # a parallel build moves the SoC clock
    scratch = R.make_prefill_scratch(rows, dev)
    target = R.prefill_target_workgroups(dev)

    def timed(launches, iters=25):
        """Median us of one launch, `launches` taken in turn."""
        for fn in launches[:3]:
            fn()
        torch.cuda.synchronize()
        ts = []
        for i in range(iters):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            launches[i % len(launches)]()
            end.record()
            end.synchronize()
            ts.append(start.elapsed_time(end) * 1000)
        return statistics.median(ts)

    cells = []
    kv_bytes_per_key = hkv * 2 * d * torch.finfo(dtype).bits // 8
    for (m_lo, k_lo), specs in prefill_batch_cells(win, kv_bytes_per_key, causal):
        times: dict = {}
        for starts, lens, spec in specs:
            torch.manual_seed(0)
            nseq, pages = len(lens), -(-int(lens[0]) // bs)
            # Causal: a paged cache; non-causal: K and V (in place).
            shape = (
                (nseq * pages, hkv, bs, 2 * d)
                if causal
                else (2, int(starts[-1]), hkv, d)
            )
            kv = (torch.randn(shape, device=dev) * 0.5).to(dtype)
            kvs = [kv] + [
                kv.clone()
                for _ in range(BATCH_WORKING_SET // (kv.numel() * kv.element_size()))
            ]
            table = torch.randperm(nseq * pages, device=dev).to(torch.int32)
            table = table.view(nseq, pages)
            q = (torch.randn(int(starts[-1]), hq, d, device=dev) * 0.5).to(dtype)
            out = torch.empty_like(q)
            sl = torch.tensor(lens, dtype=torch.int32, device=dev)
            qsl = torch.tensor(starts, dtype=torch.int32, device=dev)
            for v in builds.values():
                mod = jit.load(v)
                for g in groups:
                    plan, hg = R.plan_prefill(starts, lens, v, win, target, g)
                    plan = torch.from_numpy(plan).to(dev)
                    t = timed(
                        [
                            (
                                functools.partial(
                                    mod.prefill_attn,
                                    q,
                                    cache,
                                    table,
                                    out,
                                    *scratch,
                                    sl,
                                    int(lens.max()),
                                    d**-0.5,
                                    plan=plan,
                                    query_start_loc=qsl,
                                    head_group=hg,
                                )
                                if causal
                                else functools.partial(
                                    mod.encoder_attn,
                                    q,
                                    cache[0],
                                    cache[1],
                                    out,
                                    *scratch,
                                    sl,
                                    int(lens.max()),
                                    d**-0.5,
                                    plan=plan,
                                    query_start_loc=qsl,
                                    head_group=hg,
                                )
                            )
                            for cache in kvs
                        ]
                    )
                    times.setdefault((v.min_query_len, g), {})[spec] = t
        best = {spec: min(t[spec] for t in times.values()) for _, _, spec in specs}
        score = {
            c: geomean([t[spec] / best[spec] for spec in t]) for c, t in times.items()
        }
        (row, group), s_ = min(score.items(), key=lambda x: x[1])
        cells.append(
            {
                "min_mean_m": m_lo,
                "min_keys": k_lo,
                "row": row,
                "head_group": group,
                "score": s_,
                "specs": [spec for _, _, spec in specs],
                "us": {spec: times[(row, group)][spec] for spec in best},
            }
        )
        print(
            f"{tag}: mean M >= {m_lo:4}, keys >= {k_lo:4}: row {row:5} head group "
            f"{group}, {s_:.3f} of the best per batch",
            flush=True,
        )
    with open(a.decisions, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "prefill_batch",
                    "hq": hq,
                    "hkv": hkv,
                    "d": d,
                    "window": win,
                    "causal": int(causal),
                    "dtype": a.dtype,
                    "block_size": bs,
                    "source": jit.source_digest(prefill=True),
                    "family": gpu_family(),
                    "rows": [
                        prefill_knobs_of(v) | {"min_query_len": v.min_query_len}
                        for v in rows
                    ],
                    "cells": cells,
                    "host": platform.node(),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )


def apply_prefill_batch(dec: dict, block_sizes: list[int]) -> int:
    """Write a shape's cells into the prefill batch configs; refused if its
    prefill builds changed since."""
    import torch

    from aiter.ops import rdna_attn as R

    hq, hkv, d, win = dec["hq"], dec["hkv"], dec["d"], dec["window"]
    causal = dec.get("causal", 1)
    tag = f"{hq}/{hkv}/{d}" + (f" w{win}" if win else "") + " batch"
    rows = R.prefill_variants(hq, hkv, d, dec["block_size"], win, torch.float16, causal)
    now = [prefill_knobs_of(v) | {"min_query_len": v.min_query_len} for v in rows]
    if now != dec["rows"]:
        print(f"{tag}: refused, the prefill builds changed since it was measured")
        return 0
    family = dec.get("family") or gpu_family()
    lines, table = read_configs(
        configs_h("prefill_batch"), family, R.PREFILL_BATCH_CONFIG_FIELDS
    )
    pages = (
        sorted(
            {
                k[3]
                for k in R._prefill_table()
                if k[:3] == (hq, hkv, d) and k[4] == win and k[6] == causal
            }
        )
        or block_sizes
    )
    table = [
        r for r in table if (r[0], r[1], r[2], r[3], r[9]) != (d, hq, hkv, win, causal)
    ]
    for page in pages:
        for c in dec["cells"]:
            table.append(
                [
                    d,
                    hq,
                    hkv,
                    win,
                    page,
                    c["min_mean_m"],
                    c["min_keys"],
                    c["row"],
                    c["head_group"],
                    causal,
                ]
            )
    write_configs(
        configs_h("prefill_batch"),
        lines,
        table,
        R.PREFILL_BATCH_CONFIG_FIELDS,
        family,
    )
    print(f"{tag}: {len(dec['cells'])} cells at pages {pages}")
    return len(dec["cells"]) * len(pages)


# ------------------------------------------------------------------- --apply --


_CSRC = _HERE.parents[3] / "csrc" / "kernels" / "rdna_attn"


def configs_h(kind: str) -> Path:
    """The configs header of `kind`: decode, decode_batch, prefill or
    prefill_batch."""
    return _CSRC / f"rdna_{kind}_configs.h"


# One config entry per line.
_ENTRY = re.compile(r"^\s*\{([\w:, ]+)\},$")

# The GPU family of each arch: its config tables (Arch in rdna_attn_types.h).
FAMILIES = {
    "gfx1150": "Gfx1150",
    "gfx1151": "Gfx1151",
    "gfx1152": "Gfx1150",
    "gfx1153": "Gfx1150",
    "gfx1200": "Gfx1201",
    "gfx1201": "Gfx1201",
}


def gpu_family() -> str:
    """The family of the GPU tune.py measures on."""
    import torch

    return FAMILIES[torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]]


def _table_lines(lines: list[str], family: str) -> range:
    """The line indices of family's entries in a configs header."""
    start = 1 + next(
        i
        for i, line in enumerate(lines)
        if re.search(rf"\bk{family}\w*Configs\[\] = \{{$", line)
    )
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("};"))
    return range(start, end)


def read_configs(
    path: Path, family: str, names: tuple[str, ...] | None = None
) -> tuple[list[str], list[list[int]]]:
    """The lines of a configs header, and the entries of family's table; with
    `names`, left out trailing fields read as their defaults."""
    import jit

    lines = path.read_text().splitlines()
    at = _table_lines(lines, family)
    rows = [jit.parse_cpp_fields(m[1]) for i in at if (m := _ENTRY.match(lines[i]))]
    if names:
        tail = _trailing_defaults(names)
        rows = [r + tail[len(tail) - (len(names) - len(r)) :] for r in rows]
    return lines, rows


def _trailing_defaults(names: tuple[str, ...]) -> list[int]:
    """Defaults of the fields an entry may leave out, at the end of `names`."""
    defaults = {"causal": 1, "row_tiles": 1}
    tail: list[int] = []
    for name in reversed(names):
        if name not in defaults:
            break
        tail.insert(0, defaults[name])
    return tail


def write_configs(
    path: Path,
    lines: list[str],
    rows: list[list[int]],
    names: tuple[str, ...],
    family: str,
) -> None:
    """Replace the entries of family's table in a configs header with `rows`,
    sorted; `names` are their fields."""
    import jit

    at = _table_lines(lines, family)
    # Trailing fields at their defaults are left out.
    tail = _trailing_defaults(names)
    first_tail = len(names) - len(tail)

    def kept(r):
        n = len(r)
        while n > first_tail and r[n - 1] == tail[n - 1 - first_tail]:
            n -= 1
        return n

    entries = [
        f"    {{{jit.cpp_fields(r[: kept(r)], names[: kept(r)])}}},"
        for r in sorted(rows)
    ]
    text = lines[: at.start] + entries + lines[at.stop :]
    path.write_text("\n".join(text) + "\n")


def apply(path: str, block_sizes: list[int]) -> None:
    """Write the accepted decisions of `path` into the configs headers."""
    import jit
    import torch

    from aiter.ops import rdna_attn as R

    # The shape: the first seven fields.
    assert R.DECODE_CONFIG_FIELDS[:7] == (
        "head_size",
        "num_q_heads",
        "num_kv_heads",
        "window",
        "max_query_len",
        "dtype",
        "page_size",
    ), R.DECODE_CONFIG_FIELDS
    headers = (configs_h("decode"), configs_h("decode_batch"))
    # Each (family, batched) table, keyed by shape.
    tables: dict[tuple[str, int], dict[tuple, list[int]]] = {}

    def table(family: str, batched: int) -> dict[tuple, list[int]]:
        if (family, batched) not in tables:
            _, entries = read_configs(headers[batched], family)
            tables[family, batched] = {tuple(r[:7]): r for r in entries}
        return tables[family, batched]

    source = jit.source_digest()
    written = 0
    for line in Path(path).read_text().splitlines():
        dec = json.loads(line)
        if dec.get("kind") == "prefill":
            written += apply_prefill(dec, block_sizes)
            continue
        if dec.get("kind") == "prefill_batch":
            written += apply_prefill_batch(dec, block_sizes)
            continue
        if not dec["accepted"]:
            continue
        hq, hkv, d, m, win = dec["hq"], dec["hkv"], dec["d"], dec["m"], dec["window"]
        batched = int(dec.get("batch", False))
        tag = (
            f"{hq}/{hkv}/{d} M={m}"
            + (f" w{win}" if win else "")
            + (" batch" if batched else "")
        )
        if dec["source"] != source:
            print(f"{tag}: refused, measured on another kernel source")
            continue
        now = R.variant_for(
            hq,
            hkv,
            d,
            m,
            dec["block_size"],
            1,
            win,
            utils.torch_dtype(dec["dtype"]),
            bool(batched),
        )
        if dec["current"] != (None if now is None else knobs_of(now)):
            print(f"{tag}: refused, the row changed since it was measured")
            continue
        rows = table(dec.get("family") or gpu_family(), batched)
        sizes = sorted({k[6] for k in rows if k[:5] == (d, hq, hkv, win, m)})
        for bf16, bs in itertools.product((0, 1), sizes or block_sizes):
            v = R.KernelVariant(
                d,
                hq,
                hkv,
                m,
                bs,
                1,
                **dec["winner"],
                window=win,
                dtype=torch.bfloat16 if bf16 else torch.float16,
                batched=batched,
            )
            row = R.config_fields(v)
            rows[tuple(row[:7])] = row
            written += 1
        print(f"{tag}: {label(dec['winner'])}")
    for (family, batched), rows in tables.items():
        lines = headers[batched].read_text().splitlines()
        write_configs(
            headers[batched], lines, list(rows.values()), R.DECODE_CONFIG_FIELDS, family
        )
    print(f"# {written} rows written")
    if written:
        rebuild_modules()


def rebuild_modules() -> None:
    """Remove the built module_rdna_attn*: they hold the tables as they were,
    and are rebuilt from the headers at their next use."""
    from aiter.jit import core

    jit_dir = Path(core.get_user_jit_dir())
    removed = 0
    for so in jit_dir.glob("module_rdna_attn*.so"):
        core.clear_build(so.stem)
        so.unlink()
        removed += 1
    print(f"# {removed} built module_rdna_attn* removed, to rebuild from the tables")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--apply", metavar="DECISIONS", help="write decisions to the configs"
    )
    p.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[16],
        help="with --apply: page sizes of a configuration not built yet",
    )
    p.add_argument(
        "--prefill",
        action="store_true",
        help="tune the prefill kernel: every step of M of one configuration",
    )
    p.add_argument(
        "--prefill-batch",
        action="store_true",
        help="tune the prefill batch configs: the prefill build and "
        "head group of every cell of one configuration's batches",
    )
    p.add_argument(
        "--non-causal",
        action="store_true",
        help="with --prefill or --prefill-batch: every query token sees every "
        "key of its sequence (an encoder's segments, L = S)",
    )
    p.add_argument(
        "--prefixes",
        type=int,
        nargs="+",
        default=list(PREFIXES),
        help="with --prefill: cached tokens before each step's M",
    )
    p.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(STEPS),
        choices=STEPS,
        help="with --prefill: the steps of M to choose; --apply keeps the others",
    )
    p.add_argument("--hq", type=int)
    p.add_argument("--hkv", type=int)
    p.add_argument("--head-dim", type=int)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--m", type=int)
    utils.add_dtype_argument(p)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    p.add_argument(
        "--batch",
        nargs="+",
        help="tune the batched row over these batch specs instead of --contexts",
    )
    p.add_argument("--points", help="append every point (JSONL)")
    p.add_argument("--decisions", help="append the decision (JSONL)")
    p.add_argument("--reuse", action="store_true", help="skip points in --points")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--confirm-rounds", type=int, default=3)
    p.add_argument(
        "--min-gain", type=float, default=1.01, help="weighted score to accept"
    )
    p.add_argument("--limit", type=int, default=0, help="first N points (smoke)")
    p.add_argument("--candidates", help="JSONL of knob sets instead of the grid")
    p.add_argument(
        "--prefill-loss-limit",
        type=float,
        default=PREFILL_LOSS_LIMIT,
        help="with --prefill: most a winner may lose at a cell against the current row",
    )
    p.add_argument(
        "--reselect",
        metavar="DECISIONS",
        help="choose stored prefill decisions again from their A/B (to --decisions)",
    )
    a = p.parse_args()
    if a.reselect:
        if not a.decisions:
            p.error("--reselect needs --decisions")
        reselect_prefill(a.reselect, a.decisions, a.prefill_loss_limit)
        return
    if a.apply:
        apply(a.apply, a.block_sizes)
        return
    needed = ("hq", "hkv", "head_dim", "decisions")
    if not a.prefill_batch:
        needed += ("points",)
    batchless = a.prefill or a.prefill_batch
    missing = [
        n for n in (*needed, *(() if batchless else ("m",))) if getattr(a, n) is None
    ]
    if missing:
        p.error("tuning needs --" + ", --".join(n.replace("_", "-") for n in missing))
    if a.prefill_batch:
        tune_prefill_batch(a)
    elif a.prefill and a.non_causal:
        tune_prefill_non_causal(a)
    elif a.prefill:
        tune_prefill(a)
    else:
        tune(a)


if __name__ == "__main__":
    main()

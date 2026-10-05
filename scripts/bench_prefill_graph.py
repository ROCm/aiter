# Benchmark graph-replayed shuffled-cache prefill and mixed attention.
"""Run from this checkout with ENABLE_CK=0, PYTHONPATH set, and an assigned HIP device.

Example: python3 -u scripts/bench_prefill_graph.py --cells proto --output /tmp/prefill-graph.csv
A and C use separate processes because AITER_PREFILL_CONVENTIONAL is read at import.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.bench_decode_mixed_shuffled import build_shuffled_full, ref_paged_attn
from scripts.bench_trace_shapes_shuffled import TRACE, split_equal

CALLS_PER_GRAPH = 20
REPLAYS = 50
REPEATS = 2
FIELDS = (
    "date",
    "commit",
    "device",
    "hip_visible_devices",
    "cells",
    "M",
    "nseq",
    "chunk",
    "ndec",
    "ctx",
    "kv_mult",
    "flydsl_A_us",
    "flydsl_C_us",
    "gluon_us",
    "triton_us",
    "winner",
    "flydsl_best_over_baseline",
    "served_A",
    "served_C",
    "auto_A_us",
    "auto_C_us",
    "auto_backend_A",
    "auto_backend_C",
    "timing_mode",
    "flydsl_A_eager_us",
    "flydsl_C_eager_us",
    "gluon_eager_us",
    "triton_eager_us",
    "auto_A_eager_us",
    "auto_C_eager_us",
)
PROTO = (
    (32768, [781] * 8 + [780] * 34, [2124] * 8 + [2122] * 34),
    (3693, [93] * 13 + [92] * 27, [837] * 13 + [828] * 27),
)


def graph_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CALLS_PER_GRAPH):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (REPLAYS * CALLS_PER_GRAPH))
    us = sum(times) / len(times)
    if not us > 0:
        raise AssertionError(f"Empty graph or invalid timing: {times}")
    return us


def cases(group):
    if group == "proto":
        return [{"M": m, "query_lens": q, "kv_lens": k} for m, q, k in PROTO]
    if group == "mixed":
        return [
            {
                "M": chunk + ndec,
                "chunk": chunk,
                "ndec": ndec,
                "ctx": ctx,
                "query_lens": [chunk] + [1] * ndec,
                "kv_lens": [chunk + ctx] + [ctx] * ndec,
            }
            for chunk in (512, 4023)
            for ndec in (7, 8)
            for ctx in (4096, 16384)
        ]
    return [
        {
            "M": m,
            "trace_us": target_us,
            "query_lens": split_equal(m, max(1, gy - m // 8)),
        }
        for m, _, target_us, gy in TRACE
    ]


def eager_us(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(2):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 20)
    return statistics.median(samples)


def calibrate_trace(cell, triton_ua, eager=False):
    qls = cell["query_lens"]

    def probe(mult):
        kvs = [max(1, round(q * mult)) for q in qls]
        kw, _, _ = build_shuffled_full(qls, kvs)
        fn = lambda: triton_ua.unified_attention(**kw, backend="triton")
        return eager_us(fn) if eager else graph_us(fn)

    t1, t4 = probe(1), probe(4)
    slope = (t4 - t1) / 3
    mult = (
        min(24.0, max(1.0, 1 + (cell["trace_us"] - t1) / slope))
        if slope > 1e-6
        else 1.0
    )
    cell["kv_mult"] = mult
    cell["kv_lens"] = [max(1, round(q * mult)) for q in qls]


def eager_profile(fn, label):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    elapsed = []
    for i in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        torch.cuda.nvtx.range_push(f"{label}_call{i:02d}")
        fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        elapsed.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(elapsed)


def worker(group, variant, specs_path, result_path, profile_mixed=False, eager=False):
    import aiter.ops.flydsl.unified_attention_kernels as uak
    import aiter.ops.triton.attention.unified_attention as triton_ua
    import aiter.ops.unified_attention as ua

    expected = variant == "C"
    if uak._PREFILL_CONVENTIONAL != expected:
        raise AssertionError("AITER_PREFILL_CONVENTIONAL was not read at import")
    specs = json.loads(specs_path.read_text()) if specs_path.exists() else cases(group)
    rows = []
    for cell in specs:
        if "kv_lens" not in cell:
            calibrate_trace(cell, triton_ua, eager=eager)
        qls, kvs = cell["query_lens"], cell["kv_lens"]
        kw, ref_kw, output = build_shuffled_full(qls, kvs)
        want = ref_paged_attn(**ref_kw)
        row = {k: cell.get(k, "") for k in ("M", "chunk", "ndec", "ctx", "kv_mult")}
        row["nseq"] = len(qls)
        backends = ("flydsl", "gluon", "triton") if variant == "A" else ("flydsl",)
        for backend in backends:
            fn = (
                (lambda kw=kw: ua.unified_attention(**kw, backend="flydsl"))
                if backend == "flydsl"
                else (
                    lambda backend=backend, kw=kw: triton_ua.unified_attention(
                        **kw, backend=backend
                    )
                )
            )
            if backend == "flydsl":
                real, seen = uak.flydsl_unified_attention, {}

                def spy(*args, _real=real, _seen=seen, **kwargs):
                    result = _real(*args, **kwargs)
                    _seen["served"] = result is not None
                    return result

                with mock.patch.object(uak, "flydsl_unified_attention", spy):
                    if group == "mixed":
                        ua.unified_attention(**kw)
                    else:
                        fn()
                if group == "mixed":
                    row[f"served_{variant}"] = (
                        "served" if seen.get("served", False) else "declined"
                    )
                    if not seen["served"]:
                        row[f"flydsl_{variant}_us"] = ""
                        print(
                            f"checked mixed M={cell['M']} {variant}/flydsl served=False declined",
                            flush=True,
                        )
                        continue
                elif not seen.get("served", False):
                    raise AssertionError(
                        f"FlyDSL {variant} did not serve M={cell['M']}"
                    )
            else:
                fn()
            of = output.float()
            err = (of - want).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                of.flatten(), want.flatten(), dim=0
            ).item()
            if not (err < 1e-1 and cos > 0.99):
                raise AssertionError(
                    f"{group} M={cell['M']} {variant}/{backend}: max_err={err:.6f} cos={cos:.6f}"
                )
            us = (
                eager_profile(
                    fn,
                    f"bpg_mixed_{cell['chunk']}_{cell['ndec']}_{cell['ctx']}_{variant}_{backend}",
                )
                if profile_mixed
                else eager_us(fn) if eager else graph_us(fn)
            )
            metric = (
                f"{backend}_{variant}_us" if backend == "flydsl" else f"{backend}_us"
            )
            row[metric] = us
            if eager and not profile_mixed:
                row[
                    (
                        f"{backend}_{variant}_eager_us"
                        if backend == "flydsl"
                        else f"{backend}_eager_us"
                    )
                ] = us
            print(
                f"checked {group} M={cell['M']} {variant}/{backend} served={seen.get('served') if backend == 'flydsl' else '-'} max_err={err:.4f} cos={cos:.5f} graph_us={us:.2f}",
                flush=True,
            )
        if group == "mixed":
            auto = lambda kw=kw: ua.unified_attention(**kw)
            auto()
            of = output.float()
            err = (of - want).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(
                of.flatten(), want.flatten(), dim=0
            ).item()
            if not (err < 1e-1 and cos > 0.99):
                raise AssertionError(
                    f"mixed M={cell['M']} {variant}/auto: max_err={err:.6f} cos={cos:.6f}"
                )
            row[f"auto_backend_{variant}"] = (
                "flydsl" if row[f"served_{variant}"] == "served" else "fallback"
            )
            row[f"auto_{variant}_us"] = (
                eager_profile(
                    auto,
                    f"bpg_mixed_{cell['chunk']}_{cell['ndec']}_{cell['ctx']}_{variant}_auto",
                )
                if profile_mixed
                else eager_us(auto) if eager else graph_us(auto)
            )
            if eager and not profile_mixed:
                row[f"auto_{variant}_eager_us"] = row[f"auto_{variant}_us"]
            print(
                f"checked mixed M={cell['M']} {variant}/auto backend={row[f'auto_backend_{variant}']} max_err={err:.4f} cos={cos:.5f} graph_us={row[f'auto_{variant}_us']:.2f}",
                flush=True,
            )
        rows.append(row)
        del kw, want, output
        torch.cuda.empty_cache()
    if variant == "A":
        specs_path.write_text(json.dumps(specs))
    result_path.write_text(json.dumps(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", choices=("proto", "trace", "mixed"), required=True)
    p.add_argument("--output", type=Path, default=Path("/tmp/bench-prefill-graph.csv"))
    p.add_argument(
        "--profile-mixed",
        action="store_true",
        help="eager marked kernel profiling for mixed rows",
    )
    p.add_argument(
        "--eager", action="store_true", help="unprofiled CUDA-event eager timing"
    )
    p.add_argument("--worker", choices=("A", "C"), help=argparse.SUPPRESS)
    p.add_argument("--specs", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.profile_mixed and (a.cells != "mixed" or a.eager):
        p.error("--profile-mixed requires --cells mixed without --eager")
    if a.worker:
        worker(a.cells, a.worker, a.specs, a.result, a.profile_mixed, a.eager)
        return
    if os.environ.get("ENABLE_CK") != "0":
        p.error("ENABLE_CK=0 is required")
    if not os.environ.get("HIP_VISIBLE_DEVICES"):
        p.error("set HIP_VISIBLE_DEVICES to the assigned GPU")
    with tempfile.TemporaryDirectory() as tmp:
        specs = Path(tmp) / "specs.json"
        results = {}
        for variant in ("A", "C"):
            result = Path(tmp) / f"{variant}.json"
            env = os.environ.copy()
            env["AITER_PREFILL_CONVENTIONAL"] = "1" if variant == "C" else "0"
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--cells",
                    a.cells,
                    "--worker",
                    variant,
                    *(("--profile-mixed",) if a.profile_mixed else ()),
                    *(("--eager",) if a.eager else ()),
                    "--specs",
                    str(specs),
                    "--result",
                    str(result),
                ],
                check=True,
                env=env,
            )
            results[variant] = json.loads(result.read_text())
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    date = datetime.now(timezone.utc).isoformat()
    rows = []
    for ar, cr in zip(results["A"], results["C"]):
        keys = ("M", "nseq", "chunk", "ndec", "ctx", "kv_mult")
        if any(ar.get(key) != cr.get(key) for key in keys):
            raise AssertionError("Variant cells do not match")
        row = {
            **ar,
            **cr,
            "date": date,
            "commit": commit,
            "device": torch.cuda.get_device_name(),
            "hip_visible_devices": os.environ["HIP_VISIBLE_DEVICES"],
            "cells": a.cells,
            "timing_mode": (
                "eager_profiled_wall"
                if a.profile_mixed
                else "eager_unprofiled" if a.eager else "graph"
            ),
        }
        times = {
            name: row[f"{name}_us"]
            for name in ("flydsl_A", "flydsl_C", "gluon", "triton")
            if row.get(f"{name}_us") != ""
        }
        row["winner"] = min(times, key=times.get)
        flydsl_times = [
            times[name] for name in ("flydsl_A", "flydsl_C") if name in times
        ]
        row["flydsl_best_over_baseline"] = (
            min(flydsl_times) / min(times["gluon"], times["triton"])
            if flydsl_times
            else ""
        )
        rows.append(row)
    with a.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"date={date} commit={commit} device={rows[0]['device']} HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']}"
    )
    if a.cells == "mixed":
        print(
            f"{'chunk':>5} {'ndec':>4} {'ctx':>5} {'A':>9} {'C':>9} {'Gluon':>9} {'Triton':>9} {'auto_A':>9} {'auto_C':>9} {'served_A':>10} {'served_C':>10} {'auto_backend_A':>14} {'auto_backend_C':>14} {'winner':>10} {'fly/best':>9}"
        )
        for r in rows:
            fmt = lambda key, r=r: f"{r[key]:.2f}" if r[key] != "" else "-"
            ratio = (
                f"{r['flydsl_best_over_baseline']:.3f}"
                if r["flydsl_best_over_baseline"] != ""
                else "-"
            )
            print(
                f"{r['chunk']:>5} {r['ndec']:>4} {r['ctx']:>5} {fmt('flydsl_A_us'):>9} {fmt('flydsl_C_us'):>9} {fmt('gluon_us'):>9} {fmt('triton_us'):>9} {fmt('auto_A_us'):>9} {fmt('auto_C_us'):>9} {r['served_A']:>10} {r['served_C']:>10} {r['auto_backend_A']:>14} {r['auto_backend_C']:>14} {r['winner']:>10} {ratio:>9}"
            )
    else:
        print(
            f"{'M':>6} {'nseq':>5} {'A_us':>9} {'C_us':>9} {'Gluon_us':>9} {'Triton_us':>10} {'winner':>10} {'fly/best':>9}"
        )
        for r in rows:
            print(
                f"{r['M']:>6} {r['nseq']:>5} {r['flydsl_A_us']:>9.2f} {r['flydsl_C_us']:>9.2f} {r['gluon_us']:>9.2f} {r['triton_us']:>10.2f} {r['winner']:>10} {r['flydsl_best_over_baseline']:>9.3f}"
            )
    print(f"CSV: {a.output}")


if __name__ == "__main__":
    main()

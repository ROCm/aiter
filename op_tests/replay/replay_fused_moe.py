#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Replay fused_moe calls recorded with AITER_RECORD.

Record (in the framework, eager mode):

    AITER_RECORD=/data/rec AITER_RECORD_OPS=fused_moe vllm serve ... --enforce-eager

Replay one bundle (one process / rank):

    python op_tests/replay/replay_fused_moe.py /data/rec/rank0-host-pid123 \\
        --pad-to-graph-sizes 1,2,4,8,16,24,32 --out results/

The recorded AITER_* environment and tuned configs are restored before aiter
is imported, so fused_moe makes the same dispatch decision the framework made.
Every replayed call is checked against the recorded kernels.

Outputs (in --out):
    replay_calls.csv    one row per recorded call
    replay_summary.csv  one row per M bucket (aiter's tuned-config token key)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Per-token arguments of fused_moe; their first dimension is M.
PER_TOKEN_ARGS = ("hidden_states", "topk_ids", "topk_weight", "output", "a1_scale")
# Expert weights; --shared-weights uses one set of these for every layer.
WEIGHT_ARGS = ("w1", "w2", "w1_scale", "w2_scale", "bias1", "bias2")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("bundle", type=Path, help="bundle directory written by AITER_RECORD")
    p.add_argument("--out", type=Path, default=Path("replay_out"))
    p.add_argument(
        "--pad-to-graph-sizes",
        default="",
        help="comma-separated batch sizes the framework captured graphs for; "
        "each call's M is padded up to the next size, as graph mode would",
    )
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--no-flush",
        action="store_true",
        help="keep caches warm between iterations (default: flush them, "
        "since a framework runs other layers between two MoE calls)",
    )
    p.add_argument("--flush-mb", type=int, default=1024, help="cache flush buffer size")
    p.add_argument(
        "--sleep-cycles",
        type=int,
        default=2_000_000,
        help="GPU busy-wait queued before each timed call so host-side "
        "dispatch overhead is not timed (0 disables)",
    )
    p.add_argument(
        "--check-only", action="store_true", help="verify kernels, no timing"
    )
    p.add_argument("--limit", type=int, default=0, help="replay at most N calls")
    p.add_argument(
        "--include-capture",
        action="store_true",
        help="also replay calls recorded during graph capture (dummy routing)",
    )
    p.add_argument(
        "--shared-weights",
        action="store_true",
        help="one weight set for all layers (less memory, warmer caches)",
    )
    p.add_argument("--no-env", action="store_true", help="do not restore AITER_* env")
    p.add_argument(
        "--no-configs", action="store_true", help="do not use bundle configs"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Environment (must run before torch/aiter are imported)
# ---------------------------------------------------------------------------


def restore_environment(session: dict, bundle: Path, args) -> list[str]:
    notes = []
    if not args.no_env:
        for key, value in session.get("env", {}).items():
            if key.startswith("AITER_RECORD"):
                continue
            # Paths from the recording host (JIT dirs, caches) may not exist here.
            if value.startswith("/") and not os.path.exists(value):
                notes.append(f"env {key}={value} skipped: path does not exist here")
                continue
            os.environ[key] = value
    if not args.no_configs:
        configs = session.get("ops", {}).get("fused_moe", {}).get("configs", {})
        for env_var, info in configs.items():
            os.environ[env_var] = str((bundle / info["file"]).resolve())
        if "AITER_CONFIG_FMOE" not in configs:
            notes.append("bundle has no fused_moe tuned config; using local configs")
    for key in [k for k in os.environ if k.startswith("AITER_RECORD")]:
        del os.environ[key]  # never record while replaying
    return notes


def check_compat(session: dict, notes: list[str]) -> None:
    from aiter import op_recorder
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    rec_gpu = session.get("gpu", {})
    here = {"gfx": get_gfx(), "cu_num": get_cu_num()}
    for key, value in here.items():
        if rec_gpu.get(key) not in (None, value):
            notes.append(
                f"recorded on {key}={rec_gpu[key]}, replaying on {value}: "
                "tuned-config lookups may pick different kernels"
            )
    rec_aiter = session.get("aiter", {})
    local = op_recorder.aiter_info()
    if rec_aiter.get("commit") and rec_aiter.get("commit") != local.get("commit"):
        notes.append(
            f"recorded with aiter {rec_aiter['commit'][:12]}, replaying with "
            f"{(local.get('commit') or 'unknown')[:12]}"
        )
    if rec_aiter.get("dirty"):
        notes.append("recording aiter tree had local changes")
    patched = session.get("ops", {}).get("fused_moe", {}).get("patched", {})
    if patched:
        notes.append(
            "framework patched aiter during recording (replay uses stock aiter): "
            + ", ".join(f"{k} -> {v}" for k, v in patched.items())
        )


# ---------------------------------------------------------------------------
# Call preparation
# ---------------------------------------------------------------------------


def recorded_m(call: dict) -> int:
    return int(call["args"]["hidden_states"]["__tensor__"]["shape"][0])


def target_m(m: int, graph_sizes: list[int]) -> int:
    for size in graph_sizes:
        if size >= m:
            return size
    return m  # larger than every captured size: runs eagerly, unpadded


def pad_per_token(kwargs: dict, m: int, padded_m: int, torch) -> None:
    """Pad per-token arguments from m to padded_m rows.

    Padding rows reuse the routing of real rows (cyclically). Real graph-mode
    padding rows route whatever the router produces for their (stale) hidden
    states, which is not recorded.
    """
    if padded_m == m:
        return
    extra = torch.arange(padded_m - m) % m
    for name in PER_TOKEN_ARGS:
        t = kwargs.get(name)
        if not isinstance(t, torch.Tensor) or t.dim() == 0 or t.shape[0] != m:
            continue
        attrs = dict(vars(t)) if hasattr(t, "__dict__") else {}
        padded = torch.cat([t, t[extra.to(t.device)]], dim=0).contiguous()
        for k, v in attrs.items():
            setattr(padded, k, v)
        kwargs[name] = padded


def kernel_key(dispatch: list[dict]) -> str:
    return " | ".join("+".join(d.get("kernels", [])) for d in dispatch)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    def __init__(self, torch, args):
        self.torch = torch
        self.iters = args.iters
        self.warmup = args.warmup
        self.sleep_cycles = args.sleep_cycles
        self.flush_buf = None
        if not args.no_flush and args.flush_mb > 0:
            self.flush_buf = torch.empty(
                args.flush_mb * (1 << 20) // 4, dtype=torch.int32, device=args.device
            )
        if self.sleep_cycles and not hasattr(torch.cuda, "_sleep"):
            self.sleep_cycles = 0

    def run(self, fn) -> list[float]:
        torch = self.torch
        for _ in range(self.warmup):
            fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(self.iters):
            if self.flush_buf is not None:
                self.flush_buf.add_(1)
            if self.sleep_cycles:
                torch.cuda._sleep(self.sleep_cycles)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CALL_FIELDS = [
    "call_id",
    "op_seq",
    "status",
    "m",
    "replay_m",
    "bucket",
    "weights_id",
    "recorded_kernels",
    "replayed_kernels",
    "kernels_match",
    "median_us",
    "min_us",
    "mean_us",
    "stdev_us",
    "iters",
    "detail",
]

SUMMARY_FIELDS = [
    "bucket",
    "calls",
    "share_pct",
    "m_values",
    "kernels",
    "kernels_match_pct",
    "median_us",
    "mean_us",
    "total_us",
]


def summarize(rows: list[dict]) -> list[dict]:
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            by_bucket[row["bucket"]].append(row)
    total_calls = sum(len(v) for v in by_bucket.values()) or 1
    summary = []
    for bucket in sorted(by_bucket):
        group = by_bucket[bucket]
        timed = [r["median_us"] for r in group if r["median_us"] != ""]
        compared = [r["kernels_match"] for r in group if r["kernels_match"] != ""]
        m_counts = Counter(r["replay_m"] for r in group)
        summary.append(
            {
                "bucket": bucket,
                "calls": len(group),
                "share_pct": round(100.0 * len(group) / total_calls, 1),
                "m_values": " ".join(f"{m}:{n}" for m, n in sorted(m_counts.items())),
                "kernels": Counter(r["replayed_kernels"] for r in group).most_common(1)[
                    0
                ][0],
                "kernels_match_pct": (
                    round(100.0 * sum(compared) / len(compared), 1) if compared else ""
                ),
                "median_us": round(statistics.median(timed), 2) if timed else "",
                "mean_us": round(statistics.fmean(timed), 2) if timed else "",
                "total_us": round(sum(timed), 1) if timed else "",
            }
        )
    return summary


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict]) -> None:
    header = f"{'bucket':>7} {'calls':>7} {'share':>6} {'match':>6} {'median_us':>10} {'total_us':>11}  kernels"
    print(header)
    for s in summary:
        print(
            f"{s['bucket']:>7} {s['calls']:>7} {s['share_pct']:>5}% "
            f"{s['kernels_match_pct']:>5}% {s['median_us']!s:>10} {s['total_us']!s:>11}  {s['kernels']}"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    bundle = args.bundle
    session = json.loads((bundle / "session.json").read_text())
    notes = restore_environment(session, bundle, args)

    import torch

    from aiter import op_recorder
    from aiter.fused_moe import fused_moe, get_padded_M

    check_compat(session, notes)
    for note in notes:
        print(f"[replay] note: {note}")

    graph_sizes = sorted(
        int(s) for s in args.pad_to_graph_sizes.split(",") if s.strip()
    )
    payloads = op_recorder.load_payloads(bundle)
    factory = op_recorder.ArgumentFactory(
        payloads,
        device=args.device,
        seed=args.seed,
        ignore_ptr_for=WEIGHT_ARGS if args.shared_weights else (),
    )
    timer = None if args.check_only else Timer(torch, args)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    replayed = 0
    for call in op_recorder.iter_calls(bundle, op="fused_moe"):
        if args.limit and replayed >= args.limit:
            break
        m = recorded_m(call)
        row = {f: "" for f in CALL_FIELDS}
        row.update(
            call_id=call["call_id"],
            op_seq=call["op_seq"],
            m=m,
            weights_id=call["args"]["w1"]["__tensor__"]["data_ptr"],
            recorded_kernels=kernel_key(call.get("dispatch", [])),
        )
        rows.append(row)
        if call.get("capturing") and not args.include_capture:
            row["status"] = "skipped:capturing"
            continue
        if call.get("error"):
            row.update(status="skipped:error", detail=call["error"][:200])
            continue
        unsupported = op_recorder.find_unsupported(call["args"])
        if unsupported:
            row.update(status="skipped:unsupported", detail="; ".join(unsupported))
            continue

        try:
            kwargs = factory.build_args(call["args"])
        except Exception as exc:  # noqa: BLE001 - report the call, keep replaying
            row.update(status="skipped:rebuild", detail=repr(exc)[:200])
            continue
        replay_m = m if call.get("capturing") else target_m(m, graph_sizes)
        pad_per_token(kwargs, m, replay_m, torch)
        row.update(replay_m=replay_m, bucket=get_padded_M(replay_m))

        def fn(kwargs=kwargs):
            return fused_moe(**kwargs)

        try:
            with op_recorder.collect_dispatch() as dispatch:
                fn()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - report the call, keep replaying
            row.update(status="error", detail=f"{type(exc).__name__}: {exc}"[:300])
            continue
        row["replayed_kernels"] = kernel_key(dispatch)
        # Tuned configs are keyed by the M bucket; when padding moves a call to
        # another bucket the recorded kernels no longer apply.
        if get_padded_M(replay_m) == get_padded_M(m):
            row["kernels_match"] = row["replayed_kernels"] == row["recorded_kernels"]
        else:
            row["detail"] = "padding changed the M bucket; kernels not compared"
        if replay_m != m:
            row["detail"] = (row["detail"] + f" padded from M={m}").strip()
        if timer is not None:
            times = timer.run(fn)
            row.update(
                median_us=round(statistics.median(times), 3),
                min_us=round(min(times), 3),
                mean_us=round(statistics.fmean(times), 3),
                stdev_us=round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
                iters=len(times),
            )
        row["status"] = "ok"
        replayed += 1

    summary = summarize(rows)
    write_csv(args.out / "replay_calls.csv", CALL_FIELDS, rows)
    write_csv(args.out / "replay_summary.csv", SUMMARY_FIELDS, summary)

    statuses = Counter(r["status"] for r in rows)
    mismatched = sum(
        1 for r in rows if r["status"] == "ok" and r["kernels_match"] is False
    )
    print(
        f"[replay] {len(rows)} recorded calls: "
        + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    )
    if mismatched:
        print(
            f"[replay] WARNING: {mismatched} calls ran different kernels than recorded"
        )
    for warning in sorted(set(factory.warnings)):
        print(f"[replay] warning: {warning}")
    print_summary(summary)
    print(f"[replay] results in {args.out}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())

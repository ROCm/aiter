# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared timing and reporting for the FlyDSL all-reduce tests' ``bench`` scenario.

``test_flydsl_one_shot_allreduce.py`` and ``test_flydsl_quick_allreduce_int4.py``
each compare a FlyDSL kernel with the stock kernels production runs on the same
payloads. This module is how both of them time and report that comparison, so
the two tests cannot drift apart on method.

The comparison is meant to be unbiased between the candidates. Every rule
below removes a difference that is in the harness, not in the kernel:

* **One process, one input.** Every candidate runs in the same rank process,
  on the same input tensor at the same address, so allocation, placement and
  input data cannot differ between them. Inputs and outputs live for the whole
  spawn: ``CustomAllreduce`` keeps the IPC mapping of a registered buffer by
  its address, so a later buffer at a recycled address would reuse a stale
  peer mapping.
* **Production's call.** Each thunk calls what production calls for that
  kernel, with the arguments production passes: the ``cdr`` kernels in place
  with the input IPC-registered at capture, as
  ``CustomAllreduce.custom_all_reduce`` does under graph capture; RCCL as
  PyNccl's out-of-place call; the HIP quick all-reduce through
  ``QuickAllReduce.quick_all_reduce``.
* **Graph replay, not eager calls.** Every candidate is captured into a CUDA
  graph of ``GRAPH_INNER`` back-to-back calls and the replay is timed. That is
  what a captured decode step sees, and it removes the host cost of each
  candidate (pybind, ``_run_compiled``, ctypes), which differs by family and
  would otherwise rank the candidates by how much Python sits in front of
  their kernel. ``graph_capture()`` is necessary: on exit it IPC-registers the
  buffers the captured ``cdr`` calls use, and it owns the side stream.
* **No JIT, no cold start in the timing.** Each thunk runs before its capture,
  and each graph is replayed once, untimed, before the first timed sample.
* **No rank skew in the timing.** Each sample is one untimed replay and then
  ``reps`` timed replays queued behind it. The untimed replay is a collective,
  so it ends at about the same moment on every rank, and the timed region
  starts on the device when it ends. The skew of the host barrier before the
  sample is therefore outside the timed region.
* **No order effects.** The candidates are timed in ``rounds`` rounds, and the
  order rotates by one candidate per round. No candidate always runs first,
  or always runs after one particular other. A rank's time is the median of
  its rounds.
* **The slowest rank.** A collective finishes when its slowest rank does, so
  a candidate's time is the maximum over ranks. ``spread us`` is the gap to
  the fastest rank.
* **Graded on a replay.** The output is filled with NaN, then the graph is
  replayed once more, and that output is graded. A capture that dropped a
  launch or replayed a stale buffer fails the grade.
* **An accuracy floor.** A candidate below its own SQNR floor is a failure
  and cannot be the best stock kernel.

All-reduces captured back-to-back are a best case for the FlyDSL schedules,
whose double-buffered inboxes let a rank start call *i+1* while its peers
still reduce call *i*. That was measured on MI350P, with a 2048^3 GEMM
captured before every collective and its own time subtracted, and showed no
consistent bias towards FlyDSL.
"""

from __future__ import annotations

import os
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

# Calls captured in each timed CUDA graph. Several back-to-back calls per
# replay keep the graph-launch cost out of the time per call.
GRAPH_INNER = 10

# Timed replays per sample, and samples per candidate (one per round).
DEFAULT_REPS = 10
DEFAULT_ROUNDS = 12

# The C++ dispatch's LL window: per world size, ordered rungs of (max element
# count, kernel), and the first rung that fits wins. Element counts, because
# that is what the C++ compares. A world size missing here (6) has no LL path.
_LL_THRESHOLDS = {
    2: ((4 * 8192, "ll"), (16 * 8192, "ll128_256"), (64 * 7168, "ll128_512")),
    4: ((4 * 7168, "ll"), (4 * 8192, "ll128_256")),
    8: ((4 * 8192, "ll"),),
}
_ONESTAGE_MAX_BYTES = {4: 160 * 1024, 6: 80 * 1024, 8: 80 * 1024}
_ONESTAGE_MAX_BYTES_NAIVE = {4: 512 * 1024, 6: 256 * 1024, 8: 256 * 1024}


def cdr_variant(
    world_size: int, nbytes: int, *, use_new: bool, arch: str | None, elem_size: int = 2
) -> str:
    """The kernel ``CustomAllreduce::allreduce`` runs for this payload.

    One of ``ll`` and ``ll128_<block>`` (the LL kernels: ``use_new``, 2-byte
    dtypes only), ``1stage`` and ``2stage`` (``cross_device_reduce_*``), and
    their ``_naive`` and ``_write`` forms. The C++ host dispatch in
    ``csrc/include/custom_all_reduce.cuh`` picks the kernel and reports
    nothing, so this mirrors it by hand. Keep the two in step.
    """
    if use_new and elem_size == 2:
        numel = nbytes // elem_size
        for max_numel, name in _LL_THRESHOLDS.get(world_size, ()):
            if numel <= max_numel:
                return name
    if world_size == 2:
        return "1stage"
    if not use_new:
        limit = _ONESTAGE_MAX_BYTES_NAIVE.get(world_size)
        if limit is None:
            return "?"
        return "1stage_naive" if nbytes < limit else "2stage_naive"
    limit = _ONESTAGE_MAX_BYTES.get(world_size)
    if limit is None:
        return "?"
    if nbytes < limit:
        return "1stage"
    # The 2-stage dispatch drops to the naive kernel when the payload does not
    # split into world_size * 16 B chunks, and always at world size 6.
    if world_size == 6 or nbytes % (world_size * 16):
        return "2stage_naive"
    if world_size == 8 and nbytes > 512 * 4096 * 2 and arch == "gfx942":
        return "2stage_write"
    return "2stage"


def capture(thunk, inner: int = GRAPH_INNER) -> torch.cuda.CUDAGraph:
    """A CUDA graph of *inner* back-to-back calls of *thunk*.

    *thunk* runs once before the capture, so no JIT compile or first-call
    allocation is captured. ``graph_capture()`` is a collective when
    ``CustomAllreduce`` is built, so every rank must capture the same
    candidates in the same order.
    """
    from aiter.dist.parallel_state import graph_capture

    thunk()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
        for _ in range(inner):
            thunk()
    torch.cuda.synchronize()
    return g


def graded_replay(g, out: torch.Tensor, group) -> torch.Tensor:
    """Fill *out* with NaN, replay *g* once and return *out*.

    The barriers on both sides of the fill keep a peer that is still
    replaying from racing it.
    """
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier(group=group)
    out.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier(group=group)
    g.replay()
    torch.cuda.synchronize()
    return out


def time_graphs(
    graphs: dict, group, *, reps: int, rounds: int, inner: int = GRAPH_INNER
) -> dict[str, float]:
    """Microseconds per call of each graph from ``capture``, on this rank.

    Refer to the module docstring for the method: rotated order per round, an
    untimed aligning replay before each sample, and the median of the rounds.
    Every rank must pass the same graphs in the same order.
    """
    import torch.distributed as dist

    names = list(graphs)
    for g in graphs.values():
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: dict[str, list[float]] = {n: [] for n in names}
    for r in range(rounds):
        k = r % len(names)
        for name in names[k:] + names[:k]:
            g = graphs[name]
            dist.barrier(group=group)
            g.replay()
            start.record()
            for _ in range(reps):
                g.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1e3 / (reps * inner))
    return {n: statistics.median(v) for n, v in samples.items()}


# Every FlyDSL schedule bakes its rank into the kernel symbol, so each rank
# reports a different string for the same variant.
_RANK_FIELD = re.compile(r"_r\d+_")


def agree_variant(per_rank: list[str | None]) -> str | None:
    """One variant name for all ranks, or ``MIXED: ...`` if they differ.

    A time taken as the maximum over ranks is meaningless if the ranks ran
    different binaries, so this is checked rather than assumed.
    """
    seen = {_RANK_FIELD.sub("_r*_", v) for v in per_rank if v is not None}
    if not seen:
        return None
    if len(seen) > 1:
        return "MIXED: " + " | ".join(sorted(seen))
    return seen.pop()


# The columns a bench row has per candidate, as ``<candidate> <field>``.
CANDIDATE_FIELDS = ("us", "spread us", "SQNR dB", "variant")


def collapse(per_rank: list[dict]) -> dict:
    """One candidate's per-rank results as its ``CANDIDATE_FIELDS``."""
    us = [r["us"] for r in per_rank]
    return {
        "us": max(us),
        "spread us": max(us) - min(us),
        "SQNR dB": min(r["sqnr_db"] for r in per_rank),
        "variant": agree_variant([r.get("variant") for r in per_rank]),
    }


def speedup(base_us: float | None, us: float | None) -> float | None:
    """``base_us / us``: above 1.0 means *us* is the faster one."""
    if base_us is None or us is None:
        return None
    return base_us / us


def add_bench_args(parser) -> None:
    """The command-line options of the ``bench`` scenario."""
    parser.add_argument(
        "--scenario",
        choices=("verify", "bench"),
        default="verify",
        help="verify (default): check correctness, and time the FlyDSL kernel.\n"
        "bench: compare the FlyDSL kernel with the stock kernels production\n"
        "runs on the same payloads.",
    )
    parser.add_argument(
        "--shape-csv",
        metavar="PATH",
        default=None,
        help="bench: read (tokens, hidden) pairs from a CSV with M,K columns,\n"
        "instead of -s/--mnk.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"Timed graph replays per sample. Default {DEFAULT_REPS}.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help="Samples per candidate, the median of which is its time. In bench,\n"
        f"the candidate order rotates each round. Default {DEFAULT_ROUNDS}.",
    )
    parser.add_argument(
        "--report",
        metavar="PATH",
        default=None,
        help="bench: also write the tables to PATH as markdown, with a header\n"
        "that records what changes the numbers (device, link, GPUs, method,\n"
        "command). Overwrites PATH.",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        default=None,
        help="bench: also write every row and column to PATH as CSV.",
    )


def check_bench_args(parser, args) -> None:
    if args.reps < 1 or args.rounds < 1:
        parser.error("--reps and --rounds must be positive")
    if args.scenario != "bench":
        for flag, val in (
            ("--shape-csv", args.shape_csv),
            ("--report", args.report),
            ("--csv", args.csv),
        ):
            if val is not None:
                parser.error(f"{flag} needs --scenario bench")
    if args.shape_csv is not None:
        if args.mnk is not None:
            parser.error("--shape-csv and -s/--mnk are mutually exclusive")
        args.mnk = load_shapes_csv(args.shape_csv)


def load_shapes_csv(path: str) -> list[tuple[int, int]]:
    """``(M, K)`` pairs from a CSV with ``M`` and ``K`` columns.

    Other columns are ignored, and repeated shapes are dropped.
    """
    df = pd.read_csv(path)
    missing = {"M", "K"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    shapes = []
    for i, row in df.iterrows():
        m, k = int(row["M"]), int(row["K"])
        if m < 1 or k < 1 or (m * k * 2) % 16:
            raise ValueError(
                f"{path} row {i}: {m}x{k} must be positive and a multiple of 16 B"
            )
        shapes.append((m, k))
    return list(dict.fromkeys(shapes))


def _gpu_numa_map() -> str:
    """``device:NUMA node`` for every visible GPU.

    On a multi-socket PCIe host the GPUs a run uses change the peer bandwidth,
    so two reports are not comparable without this.
    """
    try:
        parts = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
            node = Path(f"/sys/bus/pci/devices/{bdf}/numa_node")
            parts.append(f"{i}:{node.read_text().strip() if node.exists() else '?'}")
        return ", ".join(parts) if parts else "none"
    except (OSError, AttributeError, RuntimeError):
        return "unknown"


def provenance(args, link: str, extra: list[str] = ()) -> list[str]:
    """Header lines that record everything that changes the numbers."""
    import aiter

    p = torch.cuda.get_device_properties(0)
    return [
        f"- generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
        (
            f"- device: {torch.cuda.get_device_name(0)} [{p.gcnArchName}, "
            f"{p.multi_processor_count} CU, {p.total_memory / 2**30:.0f} GiB]"
        ),
        f"- peer link: {link}",
        f"- HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', '(unset)')}",
        f"- GPU NUMA node: {_gpu_numa_map()}",
        f"- aiter package: {Path(aiter.__file__).resolve().parent}",
        (
            f"- timing: CUDA graph of {GRAPH_INNER} calls, {args.reps} timed "
            f"replays per sample, median of {args.rounds} rotated rounds, "
            "slowest rank"
        ),
        *extra,
        f"- command: {' '.join(sys.argv)}",
    ]


def bench_tables(rows: list[dict], *, candidates, drop=(), title: str) -> list:
    """The ``(title, DataFrame)`` tables of a bench run.

    The first table has the comparisons: every column that does not belong to
    one candidate. The second has each candidate's ``us``, ``spread us``,
    ``SQNR dB`` and ``variant``. The two share the columns that identify the
    row.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows).drop(columns=list(drop), errors="ignore")
    ids = [c for c in ("gfx", "tp", "tokens", "hidden", "KiB") if c in df]
    per_cand = [
        f"{n} {field}"
        for n in candidates
        for field in CANDIDATE_FIELDS
        if f"{n} {field}" in df
    ]
    # The comparisons need each candidate's time, but not the rest.
    times = [f"{n} us" for n in candidates if f"{n} us" in df]
    summary = [c for c in df.columns if c not in per_cand and c not in ids]
    return [
        (title, df[ids + times + summary]),
        (f"{title}: per candidate", df[ids + per_cand]),
    ]


def to_markdown(df: pd.DataFrame) -> str:
    # Payload sizes in full; ".4g" would print 32768 KiB as 3.277e+04.
    fmt = tuple(".10g" if c == "KiB" else ".4g" for c in df.columns)
    return df.to_markdown(index=False, floatfmt=fmt, missingval="n/a")


def write_outputs(args, title: str, header: list[str], tables, rows) -> None:
    """Write ``--report`` and ``--csv``, if they were given.

    *tables* is a list of ``(title, DataFrame)``; *rows* is every bench row.
    """
    import aiter

    if args.report:
        lines = [f"# {title}", "", *header, ""]
        for name, df in tables:
            lines += [f"## {name}", "", to_markdown(df), ""]
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
        aiter.logger.info("wrote %s", out)
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        aiter.logger.info("wrote %s", out)

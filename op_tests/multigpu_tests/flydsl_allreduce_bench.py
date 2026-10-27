# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared timing and reporting for the FlyDSL all-reduce tests' ``bench`` scenario.

``test_flydsl_one_shot_allreduce.py`` and ``test_flydsl_quick_allreduce.py``
each compare a FlyDSL kernel with the stock kernels production runs on the same
payloads, for the plain all-reduce (``--operation ar``) and for all-reduce +
residual add + RMSNorm (``--operation ar_rmsnorm``). This module is how both of
them time and report that comparison, so the two tests cannot drift apart on
method. It also reads the candidate files (``load_candidates``) through which
the tuning suite pins FlyDSL configurations.

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
from dataclasses import dataclass
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


def graded_replay(g, out, group):
    """Fill *out* with NaN, replay *g* once and return *out*.

    *out* is a tensor or a tuple of them, for a kernel with several outputs.
    The barriers on both sides of the fill keep a peer that is still
    replaying from racing it.
    """
    import torch.distributed as dist

    torch.cuda.synchronize()
    dist.barrier(group=group)
    for t in out if isinstance(out, tuple) else (out,):
        t.fill_(float("nan"))
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


def gpu_count() -> int:
    n = torch.cuda.device_count()
    if n == 0 and torch.version.hip and torch.cuda.is_available():
        n = torch._C._cuda_getDeviceCount()
    return n


def select_world_sizes(
    requested, *, explicit: bool, n_gpu: int, supported
) -> list[int]:
    """The world sizes of *requested* that can run on *n_gpu* GPUs.

    A size that cannot run is skipped with a warning when the sweep is the
    default one (*explicit* False); when it was named on the command line
    (*explicit* True) it is an error, because the caller asked for it. An empty
    result is also an error, because a run with no checks must not look like a
    pass; name a size that fits with ``--tp`` instead.
    """
    import logging

    log = logging.getLogger("aiter")
    tps = []
    for tp in requested:
        reason = None
        if tp not in supported:
            reason = f"world_size={tp} is not supported (one of {tuple(supported)})"
        elif n_gpu < tp:
            reason = f"tp={tp} needs {tp} GPUs, have {n_gpu}"
        if reason is None:
            tps.append(tp)
        elif explicit:
            raise SystemExit(reason)
        else:
            log.warning("%s; skipping", reason)
    if not tps:
        raise SystemExit(
            f"no world size can run on {n_gpu} GPUs, so nothing was checked"
        )
    return tps


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
    parser.add_argument(
        "--candidates",
        metavar="PATH",
        default=None,
        help="bench: also time the pinned FlyDSL configurations in this\n"
        "candidate file (see load_candidates in flydsl_allreduce_bench.py),\n"
        "next to the shipping engine, as columns named after each row.",
    )
    parser.add_argument(
        "--candidate-set",
        metavar="NAME",
        default=None,
        help="bench: only the --candidates rows tagged NAME in their sets\n"
        "column. Default every row.",
    )
    parser.add_argument(
        "--baselines",
        choices=BASELINES,
        default="all",
        help="bench: which stock columns to time next to the FlyDSL engines.\n"
        "all (default): every stock kernel and unfused path.\n"
        "reference: only the stock kernel production dispatch.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print every FlyDSL configuration this test can pin for\n"
        "--operation (default ar) as a --candidates file (CSV) on stdout.\n",
    )


def check_bench_args(parser, args) -> None:
    if args.reps < 1 or args.rounds < 1:
        parser.error("--reps and --rounds must be positive")
    if args.list_candidates:
        for flag, val in (
            ("--candidates", args.candidates),
            ("--report", args.report),
            ("--csv", args.csv),
        ):
            if val is not None:
                parser.error(f"{flag} and --list-candidates are mutually exclusive")
    elif args.scenario != "bench":
        for flag, val in (
            ("--shape-csv", args.shape_csv),
            ("--report", args.report),
            ("--csv", args.csv),
            ("--candidates", args.candidates),
            ("--candidate-set", args.candidate_set),
        ):
            if val is not None:
                parser.error(f"{flag} needs --scenario bench")
    if args.candidate_set is not None and args.candidates is None:
        parser.error("--candidate-set needs --candidates")
    if args.baselines != "all" and args.scenario != "bench":
        parser.error("--baselines needs --scenario bench")
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


# The operations a bench run times: the plain all-reduce, or all-reduce +
# residual add + RMSNorm.
OPERATIONS = ("ar", "ar_rmsnorm")

# The RMSNorm epsilon every ``ar_rmsnorm`` candidate is run with.
RMS_EPS = 1e-6

# Knobs a candidate row may pin, per algorithm and operation. An empty cell is
# the engine's own value.
CANDIDATE_KNOBS = {
    ("oneshot", "ar"): ("atoms", "grid_cap", "fanout", "block"),
    ("oneshot", "ar_rmsnorm"): ("atoms", "grid_cap", "fanout", "block", "split"),
    ("mesh", "ar"): ("super_tile", "grid_cap", "block"),
    ("ring", "ar"): ("super_tile", "grid_cap", "block"),
    ("mesh", "ar_rmsnorm"): (
        "super_tile",
        "grid_cap",
        "block",
        "atoms_per_row",
        "low_latency",
    ),
    ("ring", "ar_rmsnorm"): ("super_tile", "grid_cap", "atoms_per_row"),
}
_CANDIDATE_COLUMNS = (
    "name",
    "operation",
    "algorithm",
    "atoms",
    "grid_cap",
    "fanout",
    "block",
    "split",
    "super_tile",
    "atoms_per_row",
    "low_latency",
    "sets",
)
# Columns a candidate file may leave out; a missing one is empty on every row.
_OPTIONAL_CANDIDATE_COLUMNS = ("low_latency",)


@dataclass(frozen=True)
class Candidate:
    """One pinned FlyDSL configuration from a candidate file."""

    name: str
    operation: str
    algorithm: str
    knobs: dict
    sets: tuple[str, ...]


def load_candidates(
    path: str,
    *,
    operation: str,
    algorithms: tuple[str, ...],
    candidate_set: str | None = None,
) -> list[Candidate]:
    """The rows of the candidate file *path* for *operation* and *algorithms*.

    A candidate file is a CSV with the columns ``name, operation, algorithm``,
    one column per knob (``atoms, grid_cap, fanout, block, split, super_tile,
    atoms_per_row``, and the optional ``low_latency``, 0 or 1) and ``sets``. ``operation`` is ``ar`` or ``ar_rmsnorm``,
    ``algorithm`` is ``oneshot``, ``mesh`` or ``ring``. A knob cell is empty
    when the row leaves it to the engine, and a row may only pin the knobs of
    its algorithm and operation (``CANDIDATE_KNOBS``). ``sets`` is a list of
    tags separated by ``;`` that ``--candidate-set`` selects on. Names are
    unique: each is a column prefix in the bench tables.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    for col in _OPTIONAL_CANDIDATE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    missing = set(_CANDIDATE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    names = list(df["name"])
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate candidate name(s) {dup}")
    out = []
    for i, row in df.iterrows():
        key = (row["algorithm"], row["operation"])
        if key not in CANDIDATE_KNOBS:
            raise ValueError(f"{path} row {i}: unknown algorithm/operation {key}")
        allowed = CANDIDATE_KNOBS[key]
        knobs = {}
        for col in _CANDIDATE_COLUMNS[3:-1]:
            cell = row[col].strip()
            if not cell:
                continue
            if col not in allowed:
                raise ValueError(
                    f"{path} row {i} ({row['name']}): {col} is not a knob of "
                    f"{key[0]} {key[1]}"
                )
            if col == "fanout":
                knobs[col] = cell
            elif col == "low_latency":
                if cell not in ("0", "1"):
                    raise ValueError(
                        f"{path} row {i} ({row['name']}): low_latency must be "
                        f"0 or 1, got {cell!r}"
                    )
                knobs[col] = cell == "1"
            else:
                knobs[col] = int(cell)
        sets = tuple(t for t in row["sets"].replace(" ", "").split(";") if t)
        cand = Candidate(row["name"], row["operation"], row["algorithm"], knobs, sets)
        if (
            cand.operation == operation
            and cand.algorithm in algorithms
            and (candidate_set is None or candidate_set in cand.sets)
        ):
            out.append(cand)
    return out


def baselines_line(level: str, reference) -> str:
    """The report header line that records the ``--baselines`` *level*, and
    the *reference* columns it keeps."""
    if level == "all":
        return "- baselines: all"
    return "- baselines: reference only, " + ", ".join(f"`{n}`" for n in reference)


def list_grid_caps(limit: int, *, at_least: int = 1) -> tuple[int, ...]:
    """``--list-candidates``: the grid caps listed for a launch that keeps at
    most *limit* workgroups resident -- powers of two from
    ``LIST_MIN_GRID_CAP`` (and from *at_least*) up to *limit*."""
    caps, cap = [], LIST_MIN_GRID_CAP
    while cap <= limit:
        if cap >= at_least:
            caps.append(cap)
        cap *= 2
    return tuple(caps)


def write_candidates(cands, out=None) -> None:
    """Write the ``Candidate``\\ s *cands* as a candidate file
    (``load_candidates``) to *out*, default stdout."""
    import csv

    names = [c.name for c in cands]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate candidate name(s) {dup}")
    w = csv.DictWriter(
        out or sys.stdout, fieldnames=_CANDIDATE_COLUMNS, lineterminator="\n"
    )
    w.writeheader()
    for c in cands:
        row = dict.fromkeys(_CANDIDATE_COLUMNS, "")
        row.update(name=c.name, operation=c.operation, algorithm=c.algorithm)
        for knob, value in c.knobs.items():
            row[knob] = int(value) if isinstance(value, bool) else value
        row["sets"] = ";".join(c.sets)
        w.writerow(row)


def fused_reference(total: torch.Tensor, residual, weight, eps: float = RMS_EPS):
    """``(out, residual_out)`` of all-reduce + residual add + RMSNorm in fp32,
    from the fp32 all-reduced *total*.

    The sum is rounded to bf16 before the residual add, as every fused kernel
    and the unfused path do.
    """
    acc = total.to(torch.bfloat16).float() + residual.float()
    res_out = acc.to(torch.bfloat16)
    rstd = torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + eps)
    out = (acc * rstd * weight.float()).to(torch.bfloat16)
    return out, res_out


# ``ar_rmsnorm``: the exact stock fused paths, in the order each spawn captures
# them. ``cdr_fused_*`` is ``CustomAllreduce``'s fused kernel at both stage
# counts; ``<ar>+norm`` is that all-reduce and then ``rmsnorm2d_fwd_with_add``.
EXACT_FUSED_STOCK = ("cdr_fused_1stage", "cdr_fused_2stage", "cdr+norm", "rccl+norm")
# ``--baselines``: every stock column, or only the test's ``REFERENCE`` --
# what production runs where the FlyDSL kernel declines.
BASELINES = ("all", "reference")
# ``--list-candidates``: the smallest grid cap listed. Below it a persistent
# launch leaves most of the device idle at any payload a sweep times.
LIST_MIN_GRID_CAP = 8
# ``ar_rmsnorm``: the accuracy floor of an exact candidate, on ``out`` and on
# ``residual_out``. ``out`` goes through an approximate rsqrt, so it sits well
# below the plain all-reduce's floor.
EXACT_FUSED_SQNR_FLOOR_DB = 30.0


def exact_fused_thunks(
    ca, nccl, outs, x, residual, weight, ar_buf, eps, keep=EXACT_FUSED_STOCK
) -> dict:
    """The ``EXACT_FUSED_STOCK`` thunks of *keep* that can run on *x*, by name.

    *ca* is the group's ``CustomAllreduce`` and *nccl* its PyNccl, either None
    when absent. Each candidate is gated as ``CudaCommunicator`` gates it, and
    writes into its preallocated ``outs[name]``: ``fused_ar_rms`` allocates
    whatever it is not given, and an allocation inside a capture is at best a
    surprise.
    """
    from aiter import rmsnorm2d_fwd_with_add

    thunks = {}
    reg = ca is not None and ca.enable_register_for_capturing

    def _registered():
        # As CustomAllreduce: in place under capture, through its pool otherwise.
        return reg and torch.cuda.is_current_stream_capturing()

    custom = ca is not None and ca.should_custom_ar(x)
    if custom and x.shape[-1] <= 16384:
        for name, one in (("cdr_fused_1stage", True), ("cdr_fused_2stage", False)):

            def _cdr_fused(o=outs[name], one=one):
                ca.fused_ar_rms(
                    x,
                    residual,
                    res_out=o[1],
                    out=o[0],
                    w=weight,
                    eps=eps,
                    registered=_registered(),
                    use_1stage=one,
                )

            thunks[name] = _cdr_fused
    if custom:

        def _cdr_norm(o=outs["cdr+norm"]):
            ca.all_reduce(x, out=ar_buf, registered_input=_registered())
            rmsnorm2d_fwd_with_add(o[0], ar_buf, residual, o[1], weight, eps)

        thunks["cdr+norm"] = _cdr_norm
    if nccl is not None:

        def _rccl_norm(o=outs["rccl+norm"]):
            # What CudaCommunicator.all_reduce falls back to, then the norm.
            nccl.all_reduce(x, ar_buf)
            rmsnorm2d_fwd_with_add(o[0], ar_buf, residual, o[1], weight, eps)

        thunks["rccl+norm"] = _rccl_norm
    return {n: t for n, t in thunks.items() if n in keep}


def mark_staged(res: dict, ca) -> None:
    """Name the ``cdr`` candidates of *res* ``staged`` where production copies
    the input into the IPC pool instead of registering it under capture."""
    if ca is None or ca.enable_register_for_capturing:
        return
    for name in ("cdr_fused_1stage", "cdr_fused_2stage", "cdr+norm"):
        if name in res:
            res[name]["variant"] = "staged"


def best_candidate(got: dict, names, floor_db: float):
    """``(name, us)`` of the fastest of *names* in *got* that clears
    *floor_db*, or ``(None, None)``. *got* maps a name to its ``collapse``."""
    passing = [n for n in names if n in got and got[n]["SQNR dB"] >= floor_db]
    best = min(passing, key=lambda n: got[n]["us"], default=None)
    return best, (got[best]["us"] if best else None)


def _gpu_numa_map() -> str:
    """``device:NUMA node`` for every visible GPU.

    On a multi-socket PCIe host the GPUs a run uses change the peer bandwidth,
    so two reports are not comparable without this.
    """
    try:
        parts = []
        for i in range(gpu_count()):
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
    from aiter.jit.utils.chip_info import get_gfx

    p = torch.cuda.get_device_properties(0)
    return [
        f"- generated: {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}",
        (
            f"- device: {torch.cuda.get_device_name(0)} [{get_gfx()}, "
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


def geomean_table(df: pd.DataFrame, speedups, by=("tp",)) -> pd.DataFrame | None:
    """Per group of *by*, the geometric mean, minimum and count of each
    ``X vs Y`` column of *speedups*, over the rows where it is defined."""
    by = [c for c in by if c in df]
    speedups = [c for c in speedups if c in df]
    if not by or not speedups:
        return None
    out = []
    for key, g in df.groupby(by, sort=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["rows"] = len(g)
        for col in speedups:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            v = v[v > 0]
            row[f"{col} geomean"] = (
                float(statistics.geometric_mean(v)) if len(v) else None
            )
            row[f"{col} min"] = float(v.min()) if len(v) else None
            row[f"{col} n"] = len(v)
        out.append(row)
    return pd.DataFrame(out)


def bench_tables(
    rows: list[dict],
    *,
    candidates,
    drop=(),
    title: str,
    times=None,
    sections=(),
    geomean=(),
    geomean_by=("tp",),
) -> list:
    """The ``(title, DataFrame)`` tables of a bench run.

    The first table has the comparisons: every column that does not belong to
    one candidate or to a section, and the ``us`` of *times* (default every
    candidate). Each ``(name, columns)`` of *sections* is a table of its own.
    *geomean* adds a table of those ``X vs Y`` columns' geometric means per
    *geomean_by* group. The last table has each candidate's ``us``,
    ``spread us``, ``SQNR dB`` and ``variant``. Every table but the geomean one
    leads with the columns that identify the row.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows).drop(columns=list(drop), errors="ignore")
    ids = [c for c in ("gfx", "tp", "tokens", "hidden", "KiB") if c in df]
    # Every row of a table has the same operation; the CSV keeps the column.
    df = df.drop(columns=["operation"], errors="ignore")
    per_cand = [
        f"{n} {field}"
        for n in candidates
        for field in CANDIDATE_FIELDS
        if f"{n} {field}" in df
    ]
    # The comparisons need each candidate's time, but not the rest.
    shown = candidates if times is None else times
    time_cols = [f"{n} us" for n in shown if f"{n} us" in df]
    in_section = {c for _name, cols in sections for c in cols}
    summary = [
        c
        for c in df.columns
        if c not in per_cand and c not in ids and c not in in_section
    ]
    tables = [(title, df[ids + time_cols + summary])]
    for name, cols in sections:
        cols = [c for c in cols if c in df]
        if cols:
            tables.append((f"{title}: {name}", df[ids + cols]))
    gm = geomean_table(df, geomean, by=geomean_by)
    if gm is not None:
        tables.append((f"{title}: geomean speedup", gm))
    tables.append((f"{title}: per candidate", df[ids + per_cand]))
    return tables


def to_markdown(df: pd.DataFrame) -> str:
    # Payload sizes in full; ".4g" would print 32768 KiB as 3.277e+04.
    fmt = tuple(".10g" if c == "KiB" else ".4g" for c in df.columns)
    # A missing value is NaN once pandas has a float column, and tabulate only
    # writes ``missingval`` for None.
    # A bit-exact result has no noise: its SQNR is inf, or float32's maximum
    # where the SQNR is computed in float32.
    df = df.map(lambda v: "inf" if isinstance(v, float) and abs(v) >= 1e30 else v)
    df = df.astype(object).where(df.notna(), None)
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

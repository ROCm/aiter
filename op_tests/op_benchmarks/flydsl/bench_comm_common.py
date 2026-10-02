# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Machinery shared by the communication benchmarks in ``bench_comm.py``.

``bench_comm_ar`` (all-reduce) and ``bench_comm_a2a`` (all-to-all) differ in
their candidates, references and tables; how they time a collective, grade it,
spawn ranks and write a report is the same, and lives here.
"""

import logging
import math
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, set_start_method
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

logger = logging.getLogger("aiter")

# Ranks initialize HIP, which a forked child cannot.
set_start_method("spawn", force=True)

SUPPORTED_GFX = ["gfx942", "gfx950"]

TIMING_CHOICES = ("graph", "eager")
TIMING_DEFAULT = "graph"
GRAPH_INNER_DEFAULT = 10

try:
    from aiter.ops.flydsl.allreduce_shared import has_xgmi_peer_links
except Exception:  # noqa: BLE001
    has_xgmi_peer_links = None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _aiter_origin() -> str:
    """Filesystem root of the ``aiter`` package this process actually imported."""
    try:
        import aiter as _a

        return str(Path(_a.__file__).resolve().parent)
    except Exception:  # noqa: BLE001
        return "unknown"


def _peer_link_type() -> str:
    """GPU-to-GPU link type for the provenance header.

    Shares FlyQuickAllReduce's KFD probe rather than reimplementing it, so the report can
    never disagree with the dispatch decision the kernel actually made. Says so
    plainly when flydsl is absent and the probe is unavailable, rather than
    guessing -- a wrong link type here would misattribute a whole class of
    performance difference.
    """
    if has_xgmi_peer_links is None:
        return "unknown (flydsl unavailable)"
    return "xGMI" if has_xgmi_peer_links() else "PCIe (no xGMI)"


def _gpu_numa_map() -> str:
    """``cuda:i -> NUMA node`` for every visible GPU, for the provenance header.

    *Which* GPUs a run used is not cosmetic on a multi-socket PCIe host. A GPU
    hangs off one socket's root complex, so a pair on one node reaches each
    other through that socket's switch while a pair spanning nodes also crosses
    the inter-socket link. Picking devices 1,2,3,4 rather than 0,1,2,3 on the
    reference box moves 3 of 6 pairs across that boundary to 4 of 6, and
    measured 21% on the roofline -- a swing large enough that two reports
    without this line are simply not comparable.

    Read from sysfs via the BDF torch reports, since neither torch nor the HIP
    runtime exposes the NUMA node directly. Degrades to a plain "unknown"
    rather than guessing.
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


def device_description() -> str:
    """Marketing name plus enough detail to pin the SKU when it is generic.

    ``get_device_name`` is the marketing string the driver reports -- on a
    properly provisioned card that is e.g. "AMD Instinct MI355X", but many
    hosts report a generic "AMD Radeon Graphics". CU count and memory
    disambiguate the SKU in that case (256 CU / 288 GiB is an MI355X), which is
    the whole point of putting it in a provenance header.

    ``pci_device_id`` is deliberately not used: torch reports 0 for it on ROCm,
    so it would look like real provenance while carrying none. ``rocm-smi
    --showproductname`` has the real one (Card Model) if you need it.
    """
    p = torch.cuda.get_device_properties(0)
    return (
        f"{torch.cuda.get_device_name(0)} "
        f"[{p.gcnArchName}, {p.multi_processor_count} CU, "
        f"{p.total_memory / 2**30:.0f} GiB]"
    )


def dtype2str(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def provenance_lines(title: str, args, visible: int) -> list[str]:
    """The report header every benchmark shares: what ran, where, and how.

    The point of saving a report is comparing a later run against it, so the
    header records everything that changes the numbers: arch, visible GPU
    count, iteration counts, and the exact command. Without those a saved
    table is unfalsifiable. Callers append their own lines.
    """
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return [
        f"# {title}",
        "",
        f"- generated: {stamp}",
        f"- device: {device_description()}",
        f"- arch: {get_gfx()} ({visible} GPU(s) visible)",
        # Arch does not identify the fabric -- MI350X (xGMI) and MI350P
        # (PCIe-only) both report gfx950, and peer bandwidth between them
        # differs by an order of magnitude. Without this line two reports from
        # the two machines are indistinguishable on the axis that explains
        # most of the gap between them.
        f"- peer links: {_peer_link_type()}",
        # Which devices, not just how many: on a PCIe host the NUMA split of
        # the chosen subset is worth ~21% on the roofline (see _gpu_numa_map).
        f"- HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', '(unset)')}",
        f"- GPU NUMA node: {_gpu_numa_map()}",
        f"- iters: {args.iters} (warmup {args.warmup})",
        f"- aiter package: {_aiter_origin()}",
        # What the `us` column means. There is exactly one time per candidate,
        # so a report is unreadable without this line.
        (
            f"- timing: **{args.timing}** -- `us` is "
            + (
                f"HIP-graph replay ({args.graph_inner} collectives per capture, "
                "back-to-back)"
                if args.timing == "graph"
                else "eager hipEvent wall time, host path included"
            )
        ),
        f"- baseline: {args.baseline}",
    ]


def write_report(path, header: list[str], sections) -> None:
    """*header* lines, then one ``## title`` section per ``(title, table)``."""
    lines = list(header)
    lines += [f"- command: {' '.join(sys.argv)}", ""]
    for title, table in sections:
        lines += [f"## {title}", "", table, ""]
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("wrote %s", out)


def write_csv(path, df: pd.DataFrame, dtype_name: str, per_dtype: bool) -> None:
    """Dump the un-collapsed dataframe for one dtype, one file per dtype."""
    out = Path(path)
    if per_dtype:
        out = out.with_name(f"{out.stem}_{dtype_name}{out.suffix or '.csv'}")
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d row(s) x %d column(s))", out, len(df), len(df.columns))


# ---------------------------------------------------------------------------
# Grading and shapes
# ---------------------------------------------------------------------------


def sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB, matching the #4970 test.

    The only accuracy metric that spans exact and quantized candidates.
    Returns +inf when the result is bit-exact against the reference.
    """
    got = got.to(dtypes.fp32)
    ref = ref.to(dtypes.fp32)
    mse = float(((got - ref) ** 2).mean().item())
    ref_pow = float((ref * ref).mean().item())
    if not math.isfinite(mse) or not math.isfinite(ref_pow):
        return float("-inf")
    if ref_pow <= 0.0:
        return float("inf") if mse <= 0.0 else float("-inf")
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(ref_pow / mse)


def load_shapes_csv(path: str) -> list[tuple[int, int]]:
    """``(M, K)`` pairs from a CSV, for sweeps too long to put on a command line.

    ``M`` and ``K`` columns, uppercase, any extra columns ignored -- the same
    contract as ``test_gemm_a8w8_blockscale.py``'s ``--csv``, so a shape file is
    readable across the op_tests. An extra ``label`` column is conventional here
    for naming what a row is probing; it is carried nowhere and exists for
    whoever reads the file.

    The point is the dispatch sweeps: the crossovers this benchmark exists to
    find sit between the shapes the defaults measure, and bracketing them takes
    ~50 sizes per world size. That is a file, not an argument list, and it wants
    to be committed next to the report it produced so the run is reproducible.

    Duplicates are dropped preserving order rather than silently timed twice.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"shape CSV not found: {p}")
    df = pd.read_csv(p)
    missing = {"M", "K"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{p}: missing column(s) {sorted(missing)}; expected M,K "
            f"(got {list(df.columns)})"
        )
    shapes = []
    for i, row in df.iterrows():
        m, k = int(row["M"]), int(row["K"])
        if m < 1 or k < 1:
            raise ValueError(f"{p} row {i}: M and K must be positive, got {m},{k}")
        # Every custom path requires a 16 B-aligned payload and would be
        # silently dropped to the rccl-only row by `applicable()`. Refuse the
        # shape instead: in a shape file that is a typo, not a request.
        nbytes = m * k * 2
        if nbytes % 16 != 0:
            raise ValueError(
                f"{p} row {i}: {m}x{k} is {nbytes} B at 2 B/element, not a "
                "multiple of 16; every candidate but rccl would be skipped"
            )
        shapes.append((m, k))
    return list(dict.fromkeys(shapes))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _bench_graph(thunk, *, num_iters, num_warmup, inner, group, label="candidate"):
    """hipEvent-timed HIP-graph replay. Returns ``(output, us_per_call)``.

    Same ``(data, time)`` order ``run_perftest`` returns, so the two timing
    paths unpack identically.

    Why not ``run_perftest(testGraph=True)``: it times the replay through the
    torch profiler, and this bench cannot use the profiler at all -- spawned
    ranks get one that records CPU ops but no GPU activity, and the custom-AR
    and FlyDSL candidates register no aten op, so the event table comes back
    empty and ``get_trace_perf`` raises on the missing ``host_time_sum``.

    ``graph_capture()`` is required, not cosmetic. It enters ``ca_comm.capture()``,
    whose exit flushes the buffer addresses the graph recorded
    (``custom_all_reduce.py:capture``) and whose ``_IS_CAPTURING`` routes
    ``custom_fused_ar_rms`` down its capture branch, and it owns the side stream
    RCCL capture needs. ``stream=gc.stream`` is what puts the capture on *that*
    stream rather than on ``torch.cuda.graph``'s own class-level one. Capturing
    without either records a different code path than the one that replays.

    ``inner`` calls per graph, replayed back-to-back: no host and no other
    device work between consecutive collectives. That is deliberate, and was
    checked for bias. The FlyDSL schedules double-buffer their inbox and close
    each call on a single handshake, so a rank that finishes call *i* early
    can start pushing call *i+1* while its peers are still reducing call *i*.
    The ``cross_device_reduce_*`` kernels close every call with an end barrier
    and cannot run ahead. A normal deployment puts model compute between two
    collectives and the concern was that this timing methodology flatters FlyDSL.

    What comes back is whatever the thunk returns -- a tensor, or a tuple of
    them -- and it is what a **replay** produced, not what a subsequent eager
    call produced. Every returned buffer is poisoned and the graph replayed once
    more before it is read, so a capture that dropped a launch or replayed a
    stale buffer fails the grading gate. Grading an eager call instead would
    score that capture clean, because every thunk writes into the same
    preallocated output and the eager call would simply overwrite the evidence.
    """
    from aiter.dist.parallel_state import graph_capture

    for _ in range(max(1, num_warmup)):
        thunk()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    out = None
    try:
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(inner):
                out = thunk()
    except Exception as exc:
        # Not every candidate is guaranteed capturable. Say which one and how
        # to get a number anyway, rather than dying on a bare HIP error.
        raise RuntimeError(
            f"{label}: HIP graph capture failed ({exc}). Re-run with "
            "--timing eager to measure this candidate on the host path instead."
        ) from exc
    torch.cuda.synchronize()
    dist.barrier(group=group)

    reps = max(1, num_iters // inner)
    graph.replay()  # one untimed replay: the first is cold
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / (reps * inner)

    # Barriered on both sides of the poison: a peer still replaying into this
    # rank's inbox would otherwise race the fill_, and every rank must reach
    # the grading replay having issued the same number of colour increments.
    dist.barrier(group=group)
    for buf in out if isinstance(out, tuple) else (out,):
        buf.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph.replay()
    torch.cuda.synchronize()
    return out, us


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


def run_ranks(tp_size: int, worker, args_for_rank) -> list:
    """``worker(*args_for_rank(rank))`` on *tp_size* spawned ranks; their
    results in rank order.

    Not ``pool.join()``: every rank shares one process group, so a
    ``dist.barrier()`` a few lines into any candidate needs all ranks to reach
    it. If one rank returns early -- success or exception, fewer collective
    calls than its peers either way -- the barrier sequence desyncs permanently
    and the remaining ranks spin in that barrier forever. ``pool.join()`` waits
    for every worker process to exit, so it would hang right along with them,
    silently sitting on top of a result (or exception). Poll instead, and the
    moment any one rank's result is ready, fetch it -- an exception surfaces
    immediately instead of waiting behind peers that will now never finish.
    """
    with Pool(processes=tp_size) as pool:
        rets = [pool.apply_async(worker, args=args_for_rank(r)) for r in range(tp_size)]
        pool.close()
        pending = set(range(len(rets)))
        while pending:
            for i in sorted(pending):
                if not rets[i].ready():
                    continue
                pending.discard(i)
                try:
                    rets[i].get()
                except Exception:
                    stuck = sorted(pending)
                    logger.error(
                        "rank %d failed (see traceback below); rank(s) %s were "
                        "still running and will now be killed -- a per-rank "
                        "failure desyncs the barrier sequence, so they were "
                        "never going to finish on their own",
                        i,
                        stuck,
                    )
                    pool.terminate()
                    pool.join()
                    raise
            if pending:
                time.sleep(1.0)
        pool.join()
    return [r.get() for r in rets]

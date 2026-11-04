# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for ``QuickAllReduceInt4.reduce_scatter`` / ``all_gather``.

Same layout as ``test_flydsl_quick_allreduce_int4.py``: ``python3`` this file
first runs untimed validity, then a ``@benchmark`` / markdown sweep. Every rank
is a ``multiprocessing`` spawn worker with its own ``QuickAllReduceInt4``
engine; the oracle is an untimed fp32 NCCL all-reduce / all-gather of the same
per-rank inputs. INT4 is lossy, so accuracy is NMSE against the exact fp32
result (plus ``checkAllclose`` with the all-reduce test's tolerances); the
other checks are exact. Inputs are normal, heavy-tailed (Student-t, 3 degrees
of freedom), ranged (rank r's partial is ``4**(r % 4)`` times as large, so that
the rounding order of the sum shows) and all zeros:

* RS block placement and accuracy: rank r holds rows ``[r*T/W, (r+1)*T/W)`` of
  the sum, NMSE below a bound that a wrong block, offset or store would exceed
  by two orders of magnitude;
* a row's RS value does not depend on its owner: rolling the input row blocks
  by one block rolls the outputs by one rank, bit for bit;
* AG accuracy, bitwise-equal output on every rank, and the own block is
  round-tripped through the codec (it does not equal the input);
* all-zero inputs give all-zero outputs (either sign: the codec's zero group
  decodes to ``-0.0``, which still compares equal to zero), never NaN or a
  stray value;
* inputs unchanged, and every output element written and nothing past it (the
  outputs start as a sentinel and are followed by a guard);
* back-to-back launches of both modes, alternating between two inputs, with
  one rank delayed (``torch.cuda._sleep``, a private torch API), each
  reproduce the isolated output of their own input bit for bit;
* tensors the engine must refuse are refused: wrong size ratio, a partial
  tile, overlapping and misaligned buffers; so are engines the modes cannot run
  on (the ring, a fine-grained inbox);
* after ``compile_sp`` (which launches nothing and leaves its tensors alone)
  both modes capture into one CUDA graph, and every replay with new inputs
  equals an eager run of the same inputs bit for bit; capturing on an engine
  that has not loaded its builds is refused;
* all-reduce, reduce-scatter and all-gather alternate on one engine in every
  ordered pair, with each rank slowed in turn, and every output equals the
  isolated result of its call bit for bit: once on a one-rung ST=1 engine at a
  small grid cap (several tiles per block), once on the shipping ladder.

``rows * hidden * 2`` must be a whole number of 32 KiB tiles and ``rows`` a
multiple of the world size; the default hidden 2560 needs rows % 32 == 0.
Needs at least TP GPUs. Runs on gfx942/gfx950 at TP in {2, 4, 8}; other archs
skip, and ``main()`` skips a world size when fewer GPUs are visible.
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from contextlib import ExitStack, closing
from functools import partial
from multiprocessing import Pool, freeze_support, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

import aiter
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

set_start_method("spawn", force=True)

from aiter.ops.flydsl.kernels.comm.allreduce_shared import (
    BLOCK,
    DEFAULT_GRID_CAP,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    TwoShotRung,
)

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None
SUPPORTED_ARCHS = ("gfx942", "gfx950")
# Expected NMSE of one INT4 group-16 E4M3 codec application on N(0, sigma) data is
# about 6e-3 for the RS sum of two partials (and for an AG round trip), and about
# 1e-2 on Student-t(3) data; 2e-2 is a bound a correct kernel stays under with
# margin and a wrong block / offset (NMSE ~ 1-2) exceeds.
NMSE_MAX = 2e-2
# checkAllclose tolerances as in test_flydsl_quick_allreduce_int4.py: calibrated
# to the INT4 group-16 codec, not bit identity. NMSE stays the accuracy gate.
CLOSE_RTOL = 1e-1
CLOSE_ATOL = 1e-1
CLOSE_ERR_RATIO = 0.5
_FILLS = ("normal", "heavy", "ranged", "zeros")
# Output buffers start as this value, exactly representable in bf16 and far from
# the data (std 0.1), so a tile the kernel never wrote shows as a huge NMSE.
_SENTINEL = -7.0
_GUARD_ELEMS = 64
# Launches of each mode per delayed-rank round (a count, not tuned).
_BACK_TO_BACK = 40
# Argument of torch.cuda._sleep: device clock cycles, not calibrated here. It
# only has to be long against one launch (tens to hundreds of microseconds).
_SKEW_CYCLES = 20_000_000
# Fixed so the refusal cases below hit the check they name at every TP: 3 * tp
# rows are never a whole number of 32 KiB tiles at this width, 32 * tp rows are.
_REFUSE_HIDDEN = 2560
# (tokens, hidden). At TP2 the grid is 3 resident workgroups on each of 256 CUs
# (768; 4 per CU, 1024, at TP4/8): 1024 is 160 tiles (one per block), 4928 is
# 770 tiles (just above the TP2 grid: some blocks run two) and 16384 gives 3-4
# tiles per block. The cap-64 pass of main() gives many tiles per block at every
# TP. The single-phase kernels are super_tile=1 only, so the engine is built
# with it.
_VALIDITY_SHAPES = {
    2: [(1024, 2560), (4928, 2560), (16384, 2560)],
    4: [(1024, 2560), (4928, 2560)],
    8: [(1024, 2560), (4928, 2560)],
}
# Graph test: rows of the full tensor (160 tiles at hidden 2560, a whole number
# of tiles and of ranks at every TP) and replays, each with new inputs.
_GRAPH_ROWS = 1024
_GRAPH_REPLAYS = 4
# Interleave test. The ordered-transition cycle repeats this often. The pinned
# engine runs at this grid cap, so a block runs several tiles of the
# single-phase size; the shipping ladder runs the larger one, which is above the
# TP2 and TP8 caps.
_INTERLEAVE_ROUNDS = 3
_PINNED_CAP = 64
_INTERLEAVE_SP_ROWS = {"pinned": 1024, "shipped": 4096}
# All-reduce sizes tried for each engine, largest first: one is kept for every
# binary the engine picks (at TP8 the shipping ladder has an ST=8 rung and, for
# payloads under eight tiles, the ST=1 engine that the single-phase modes share).
_AR_ROW_CANDIDATES = (16384, 1024, 32)
# fp32 matmuls the slowed rank runs on a side stream: real work that competes
# with its kernels for the CUs, like the drag of the all-reduce test's run-ahead.
_DRAG_STEPS = 200
_OPS = ("ar", "rs", "ag")
_OP_METHODS = {"ar": "allreduce", "rs": "reduce_scatter", "ag": "all_gather"}


def _make_inp(
    tokens: int, hidden: int, *, rank: int, salt: int, fill: str, device
) -> torch.Tensor:
    if fill == "zeros":
        return torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=device)
    gen = torch.Generator().manual_seed(1000 * salt + rank)
    z = torch.randn((tokens, hidden), generator=gen, dtype=torch.float32)
    if fill == "heavy":
        # Student-t: a normal over sqrt(chi-square / dof); a few large values
        # stretch the scale of their group of 16.
        chi = sum(
            torch.randn((tokens, hidden), generator=gen, dtype=torch.float32) ** 2
            for _ in range(3)
        )
        z = z / torch.sqrt(chi / 3.0)
    if fill == "ranged":
        # Rank r's partial is 4**(r % 4) times as large, up to 64x (the values stay
        # far below the fp16 maximum). A dequantized product is exact in fp16, and
        # so is a sum of terms of similar magnitude, which makes the result
        # independent of the accumulation order for the other fills; terms of
        # very different magnitude round, so with this one the order (rank order
        # for every owner) shows in the bits.
        z = z * 4.0 ** (rank % 4)
    return (z * 0.1).to(device=device, dtype=torch.bfloat16)


def _sentinel_buffer(shape: tuple[int, int], device):
    """A sentinel-filled bf16 tensor of *shape* and the guard elements after it."""
    n = shape[0] * shape[1]
    buf = torch.full(
        (n + _GUARD_ELEMS,), _SENTINEL, dtype=torch.bfloat16, device=device
    )
    return buf[:n].view(shape), buf[n:]


def _nmse(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.double(), ref.double()
    return float(((got - ref) ** 2).sum() / (ref**2).sum())


def _close_err(ref: torch.Tensor, got: torch.Tensor, msg: str) -> float:
    return float(
        checkAllclose(
            ref,
            got.float(),
            rtol=CLOSE_RTOL,
            atol=CLOSE_ATOL,
            tol_err_ratio=CLOSE_ERR_RATIO,
            printLog=False,
            msg=msg,
        )
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _gather(t: torch.Tensor, world: int) -> list[torch.Tensor]:
    parts = [torch.empty_like(t) for _ in range(world)]
    torch.distributed.all_gather(parts, t.contiguous())
    return parts


def _expect_refused(call, pattern: str, label: str) -> None:
    try:
        call()
    except ValueError as exc:
        if not re.search(pattern, str(exc)):
            raise AssertionError(f"{label}: refused with another error: {exc}") from exc
    else:
        raise AssertionError(f"{label}: accepted")


def _check_refusals(fly, tp: int, device) -> None:
    """Malformed tensors raise ValueError before any launch (all ranks alike)."""
    rows = 32 * tp  # a whole number of tiles at hidden 2560
    shard_rows = rows // tp

    def zeros(n_rows: int) -> torch.Tensor:
        return torch.zeros(
            (n_rows, _REFUSE_HIDDEN), dtype=torch.bfloat16, device=device
        )

    full, shard = zeros(rows), zeros(shard_rows)
    big_shard = zeros(2 * shard_rows)  # the full tensor is not tp times this
    part, part_shard = zeros(3 * tp), zeros(3)  # tp * 3 rows are a partial tile
    # a shard that starts 16 bytes (8 elements) into the full tensor
    overlap_full = zeros(rows)
    overlap_shard = overlap_full.view(-1)[8 : 8 + shard.numel()].view_as(shard)
    # a full tensor that starts one element (2 bytes) into its allocation
    raw = torch.zeros(full.numel() + 8, dtype=torch.bfloat16, device=device)
    skewed = raw[1 : 1 + full.numel()].view_as(full)

    ratio, tile = "times the shard", "whole number of"
    overlap, align = "non-overlapping", "16-byte-aligned"
    cases = (
        ("rs size ratio", lambda: fly.reduce_scatter(full, big_shard), ratio),
        ("ag size ratio", lambda: fly.all_gather(big_shard, full), ratio),
        ("rs partial tile", lambda: fly.reduce_scatter(part, part_shard), tile),
        ("ag partial tile", lambda: fly.all_gather(part_shard, part), tile),
        (
            "rs overlap",
            lambda: fly.reduce_scatter(overlap_full, overlap_shard),
            overlap,
        ),
        (
            "ag overlap",
            lambda: fly.all_gather(overlap_shard, overlap_full),
            overlap,
        ),
        ("rs misaligned input", lambda: fly.reduce_scatter(skewed, shard), align),
        ("ag misaligned output", lambda: fly.all_gather(shard, skewed), align),
    )
    for label, call, pattern in cases:
        _expect_refused(call, pattern, label)


def _measure(
    fly, tp: int, rank: int, ntok: int, hidden: int, fill: str, *, device, time_it
) -> dict:
    import torch.distributed as dist

    shard_rows = ntok // tp
    row = {
        "tokens": ntok,
        "hidden": hidden,
        "fill": fill,
        "us_rs": None,
        "us_ag": None,
    }

    # reduce-scatter against the exact fp32 sum
    x = _make_inp(ntok, hidden, rank=rank, salt=1, fill=fill, device=device)
    x0 = x.clone()
    out, out_tail = _sentinel_buffer((shard_rows, hidden), device)
    fly.reduce_scatter(x, out)
    torch.cuda.synchronize()
    ref = x.float().clone()
    dist.all_reduce(ref)
    ref = ref[rank * shard_rows : (rank + 1) * shard_rows]
    row["rs_nmse"] = _nmse(out.float(), ref)
    row["rs_err"] = _close_err(ref, out, f"rs rank {rank}")
    row["rs_zero"] = bool((out == 0).all())
    if fill == "zeros":
        row["rs_zero_bits"] = torch.unique(_bits(out)).tolist()
    row["rs_clean"] = torch.equal(_bits(x), _bits(x0)) and bool(
        (out_tail == _SENTINEL).all()
    )

    # a row's value does not depend on the owner: rolled input -> rolled output
    rolled = torch.roll(x, -shard_rows, dims=0).contiguous()
    out_rolled = torch.empty_like(out)
    fly.reduce_scatter(rolled, out_rolled)
    torch.cuda.synchronize()
    first, second = _gather(out.contiguous(), tp), _gather(out_rolled, tp)
    row["rs_owner_independent"] = all(
        torch.equal(_bits(second[r]), _bits(first[(r + 1) % tp])) for r in range(tp)
    )

    # all-gather against the exact concatenation
    s = _make_inp(shard_rows, hidden, rank=rank, salt=2, fill=fill, device=device)
    s0 = s.clone()
    full, full_tail = _sentinel_buffer((ntok, hidden), device)
    fly.all_gather(s, full)
    torch.cuda.synchronize()
    ref_ag = torch.cat([p.float() for p in _gather(s, tp)])
    row["ag_nmse"] = _nmse(full.float(), ref_ag)
    row["ag_err"] = _close_err(ref_ag, full, f"ag rank {rank}")
    row["ag_zero"] = bool((full == 0).all())
    if fill == "zeros":
        row["ag_zero_bits"] = torch.unique(_bits(full)).tolist()
    outs = _gather(full, tp)
    row["ag_ranks_equal"] = all(torch.equal(_bits(outs[0]), _bits(o)) for o in outs[1:])
    own = full[rank * shard_rows : (rank + 1) * shard_rows]
    row["ag_own_equal_frac"] = float((own.float() == s.float()).float().mean())
    row["ag_clean"] = torch.equal(_bits(s), _bits(s0)) and bool(
        (full_tail == _SENTINEL).all()
    )

    # back-to-back launches, one rank delayed, alternating between two inputs:
    # a launch that read a stale or half-overwritten payload of its neighbour
    # differs from the isolated result of its own input
    x2 = _make_inp(ntok, hidden, rank=rank, salt=3, fill=fill, device=device)
    s2 = _make_inp(shard_rows, hidden, rank=rank, salt=4, fill=fill, device=device)
    out2, full2 = torch.empty_like(out), torch.empty_like(full)
    fly.reduce_scatter(x2, out2)
    fly.all_gather(s2, full2)
    torch.cuda.synchronize()
    pairs = ((x, s, out, full), (x2, s2, out2, full2))
    bad = 0
    for slow in range(tp):
        dist.barrier()
        if rank == slow:
            torch.cuda._sleep(_SKEW_CYCLES)
        rs_outs = [torch.empty_like(out) for _ in range(_BACK_TO_BACK)]
        ag_outs = [torch.empty_like(full) for _ in range(_BACK_TO_BACK)]
        for i in range(_BACK_TO_BACK):
            src_rs, src_ag, _, _ = pairs[i % 2]
            fly.reduce_scatter(src_rs, rs_outs[i])
            fly.all_gather(src_ag, ag_outs[i])
        torch.cuda.synchronize()
        for i in range(_BACK_TO_BACK):
            _, _, want_rs, want_ag = pairs[i % 2]
            bad += not torch.equal(_bits(rs_outs[i]), _bits(want_rs))
            bad += not torch.equal(_bits(ag_outs[i]), _bits(want_ag))
    row["back_to_back_mismatches"] = bad
    dist.barrier()

    if time_it:
        for key, op, src, dst in (
            ("us_rs", fly.reduce_scatter, x, out),
            ("us_ag", fly.all_gather, s, full),
        ):
            dist.barrier()
            torch.cuda.synchronize()

            def _call(op=op, src=src, dst=dst):
                op(src, dst)
                return dst

            # use_cuda_event is mandatory: run_perftest's default timer
            # collects no device rows inside a spawn worker.
            _, us = run_perftest(_call, use_cuda_event=True)
            row[key] = float(us)
    return row


def _chain_text(exc: BaseException) -> str:
    """The messages of *exc* and of what it was raised from or during."""
    texts = []
    while exc is not None:
        texts.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    return " | ".join(texts)


def _graph_replay(fly, rank: int, tp: int, device, gloo) -> dict:
    """``compile_sp``, then both modes captured into one CUDA graph and replayed.

    Every replay gets new inputs written in place and must reproduce an eager
    run of the same inputs bit for bit and stay within NMSE_MAX of a fresh fp32
    reference: a graph that baked in the capture-time data, or a launch that was
    captured wrongly, fails one of the two. No eager launch precedes the
    capture, so it also shows that ``compile_sp`` alone keeps the JIT and the
    module load out of it.
    """
    import torch.distributed as dist

    rows, hidden = _GRAPH_ROWS, _REFUSE_HIDDEN
    shard_rows = rows // tp
    x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device)
    s = torch.zeros((shard_rows, hidden), dtype=torch.bfloat16, device=device)
    rs_out, ag_out = torch.empty_like(s), torch.empty_like(x)
    # compile_sp launches nothing and leaves its tensors alone
    full, shard = torch.full_like(x, _SENTINEL), torch.full_like(s, _SENTINEL)
    fly.compile_sp(full, shard)
    torch.cuda.synchronize()
    res = {
        "capture_error": None,
        "compile_sp_untouched": bool(
            (full == _SENTINEL).all() and (shard == _SENTINEL).all()
        ),
        "replay_mismatches": 0,
        "replay_nmse": 0.0,
    }
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            fly.reduce_scatter(x, rs_out)
            fly.all_gather(s, ag_out)
    except Exception as exc:  # noqa: BLE001 -- reported, and ends the test
        res["capture_error"] = f"{type(exc).__name__}: {exc}"
    errors = [None] * tp
    dist.all_gather_object(errors, res["capture_error"], group=gloo)
    # A rank that replays alone hangs on its peers' kernels.
    if any(errors):
        return res
    for k in range(_GRAPH_REPLAYS):
        x.copy_(
            _make_inp(
                rows, hidden, rank=rank, salt=10 + k, fill="normal", device=device
            )
        )
        s.copy_(
            _make_inp(
                shard_rows, hidden, rank=rank, salt=20 + k, fill="normal", device=device
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        rs_eager, ag_eager = torch.empty_like(rs_out), torch.empty_like(ag_out)
        fly.reduce_scatter(x, rs_eager)
        fly.all_gather(s, ag_eager)
        torch.cuda.synchronize()
        res["replay_mismatches"] += int(
            not torch.equal(_bits(rs_out), _bits(rs_eager))
        ) + int(not torch.equal(_bits(ag_out), _bits(ag_eager)))
        ref = x.float().clone()
        dist.all_reduce(ref)
        ref_ag = torch.cat([p.float() for p in _gather(s, tp)])
        res["replay_nmse"] = max(
            res["replay_nmse"],
            _nmse(rs_out.float(), ref[rank * shard_rows : (rank + 1) * shard_rows]),
            _nmse(ag_out.float(), ref_ag),
        )
    return res


def _cold_capture_refused(fly, tp: int, device) -> bool:
    """A capture of a single-phase mode on an engine that has not loaded its
    builds (no ``compile_sp``, no eager launch) is refused with a message that
    names ``compile_sp``: loading cannot run under capture."""
    full = torch.zeros(
        (_GRAPH_ROWS, _REFUSE_HIDDEN), dtype=torch.bfloat16, device=device
    )
    shard = torch.empty(
        (_GRAPH_ROWS // tp, _REFUSE_HIDDEN), dtype=torch.bfloat16, device=device
    )
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=torch.cuda.Stream()):
            fly.reduce_scatter(full, shard)
    except Exception as exc:  # noqa: BLE001 -- the text of the refusal is checked
        return "compile_sp" in _chain_text(exc)
    return False


def _op_rows(op: str, rows: int, tp: int) -> tuple[int, int]:
    """(input rows, output rows) of *op* on a *rows*-row full tensor."""
    return {"ar": (rows, rows), "rs": (rows, rows // tp), "ag": (rows // tp, rows)}[op]


def _ar_rows(fly) -> list[int]:
    """The all-reduce row counts that run each distinct binary *fly* would pick."""
    seen, rows_for = set(), []
    for rows in _AR_ROW_CANDIDATES:
        symbol = fly.variant(rows * _REFUSE_HIDDEN * 2).split("/")[0]
        if symbol not in seen:
            seen.add(symbol)
            rows_for.append(rows)
    return rows_for


def _interleave_ops(ar_rows: list[int], sp_rows: int) -> list[tuple[str, int]]:
    """``(op, full-tensor rows)`` calls: for every all-reduce size a cycle through
    all six ordered transitions between the three ops, repeated."""
    ops = [
        step
        for rows in ar_rows
        for step in (
            ("ar", rows),
            ("rs", sp_rows),
            ("ag", sp_rows),
            ("ar", rows),
            ("ag", sp_rows),
            ("rs", sp_rows),
        )
    ] * _INTERLEAVE_ROUNDS
    pairs = {(a, b) for (a, _), (b, _) in itertools.pairwise(ops)}
    assert pairs == {(a, b) for a in _OPS for b in _OPS if a != b}, pairs
    return ops


def _interleave(fly, rank: int, tp: int, device, sp_rows: int) -> dict:
    """Back-to-back all-reduce / reduce-scatter / all-gather calls with no sync
    between them, each with its own input, one rank slowed in turn; every output
    must equal the isolated result of its call bit for bit.

    The three modes share the engine's inbox and colors, so a launch that starts
    on its neighbour's leftovers (a slow reader, a flag of the other mode) reads
    the payload of the wrong call and differs. The slowed rank runs fp32 matmuls
    on a side stream.
    """
    import torch.distributed as dist

    ops = _interleave_ops(_ar_rows(fly), sp_rows)
    ins, isolated = [], []
    for i, (op, rows) in enumerate(ops):
        n_in, n_out = _op_rows(op, rows, tp)
        ins.append(
            _make_inp(
                n_in,
                _REFUSE_HIDDEN,
                rank=rank,
                salt=100 + i,
                fill="normal",
                device=device,
            )
        )
        out = torch.empty((n_out, _REFUSE_HIDDEN), dtype=torch.bfloat16, device=device)
        dist.barrier()
        getattr(fly, _OP_METHODS[op])(ins[-1], out)
        torch.cuda.synchronize()
        isolated.append(out)
    mismatches = 0
    for slow in range(tp):
        outs = [torch.empty_like(o) for o in isolated]
        dist.barrier()
        if rank == slow:
            with torch.cuda.stream(torch.cuda.Stream()):
                drag = torch.randn(2048, 2048, device=device)
                for _ in range(_DRAG_STEPS):
                    drag = drag @ drag.T * 1e-6
        for (op, _), inp, out in zip(ops, ins, outs):
            getattr(fly, _OP_METHODS[op])(inp, out)
        torch.cuda.synchronize()
        mismatches += sum(
            not torch.equal(_bits(o), _bits(iso)) for o, iso in zip(outs, isolated)
        )
    return {"launches": len(ops) * tp, "mismatches": mismatches}


def _check_unsupported_engines(rank: int, tp: int, device, gloo) -> None:
    """Engines the single-phase modes cannot run on refuse them before any
    launch: the ring, and a fine-grained inbox (validated on the uncached one)."""
    from aiter.ops.flydsl import QuickAllReduceInt4

    full = torch.zeros((32 * tp, _REFUSE_HIDDEN), dtype=torch.bfloat16, device=device)
    shard = torch.zeros((32, _REFUSE_HIDDEN), dtype=torch.bfloat16, device=device)
    # The ring has no default ladder on xGMI.
    ring_rung = TwoShotRung(
        min_bytes=0, super_tile=8, grid_cap=_PINNED_CAP, block=BLOCK
    )
    for label, pattern, kw in (
        (
            "fine-grained inbox",
            "uncached inbox",
            {"inbox_memory": "finegrained", "ladder": _sp_ladder(_PINNED_CAP)},
        ),
        ("ring", "mesh algorithm", {"algorithm": "ring", "ladder": (ring_rung,)}),
    ):
        with closing(
            QuickAllReduceInt4(
                group=gloo, device=device, rank=rank, world_size=tp, **kw
            )
        ) as fly:
            for name, call in (
                ("reduce_scatter", partial(fly.reduce_scatter, full, shard)),
                ("all_gather", partial(fly.all_gather, shard, full)),
                ("compile_sp", partial(fly.compile_sp, full, shard)),
            ):
                _expect_refused(call, pattern, f"{label}: {name}")


def _init_rank(rank: int, tp: int, init_method: str):
    """Set up this rank's device and process groups; (device, gloo group)."""
    import torch.distributed as dist

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=tp,
        rank=rank,
        device_id=device,
    )
    # QuickAllReduceInt4 exchanges IPC metadata over a non-NCCL group;
    # NCCL stays for the fp32 references.
    return device, dist.new_group(backend="gloo")


def _sp_ladder(grid_cap: int):
    """The single-phase kernels are super_tile=1 only."""
    return (TwoShotRung(min_bytes=0, super_tile=1, grid_cap=grid_cap, block=BLOCK),)


def _run_rank(
    rank: int,
    tp: int,
    init_method: str,
    tokens: list[int],
    hiddens: list[int],
    fills: tuple[str, ...],
    grid_cap: int,
    time_it: bool,
) -> list[dict]:
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4

    device, gloo = _init_rank(rank, tp, init_method)
    fly = QuickAllReduceInt4(
        group=gloo,
        device=device,
        rank=rank,
        world_size=tp,
        ladder=_sp_ladder(grid_cap),
    )
    rows = []
    try:
        _check_refusals(fly, tp, device)
        for ntok, hidden in zip(tokens, hiddens, strict=True):
            fly.compile_sp(
                torch.zeros((ntok, hidden), dtype=torch.bfloat16, device=device),
                torch.zeros((ntok // tp, hidden), dtype=torch.bfloat16, device=device),
            )
            dist.barrier()
            for fill in fills:
                row = _measure(
                    fly,
                    tp,
                    rank,
                    ntok,
                    hidden,
                    fill,
                    device=device,
                    time_it=time_it and fill == "normal",
                )
                row["grid_cap"] = grid_cap
                rows.append(row)
                torch.cuda.empty_cache()
    finally:
        fly.close()
        dist.destroy_process_group()
    return rows


def _run_rank_extras(rank: int, tp: int, init_method: str) -> dict:
    """The graph, interleave and unsupported-engine checks of one rank."""
    import torch.distributed as dist

    from aiter.ops.flydsl import QuickAllReduceInt4

    device, gloo = _init_rank(rank, tp, init_method)
    res = {}
    try:
        with ExitStack() as stack:
            # Pinned: the one ST=1 rung the single-phase modes need, at a small
            # grid cap; the all-reduce runs on the same engine.
            pinned = stack.enter_context(
                closing(
                    QuickAllReduceInt4(
                        group=gloo,
                        device=device,
                        rank=rank,
                        world_size=tp,
                        ladder=_sp_ladder(_PINNED_CAP),
                    )
                )
            )
            res["graph"] = _graph_replay(pinned, rank, tp, device, gloo)
            res["interleave_pinned"] = _interleave(
                pinned, rank, tp, device, _INTERLEAVE_SP_ROWS["pinned"]
            )
            shipped = stack.enter_context(
                closing(
                    QuickAllReduceInt4(
                        group=gloo, device=device, rank=rank, world_size=tp
                    )
                )
            )
            res["interleave_shipped"] = _interleave(
                shipped, rank, tp, device, _INTERLEAVE_SP_ROWS["shipped"]
            )
            _check_unsupported_engines(rank, tp, device, gloo)
            # Last, on an engine nothing has launched on: a refused capture must
            # not leave the others in a state that fails their checks.
            cold = stack.enter_context(
                closing(
                    QuickAllReduceInt4(
                        group=gloo,
                        device=device,
                        rank=rank,
                        world_size=tp,
                        ladder=_sp_ladder(_PINNED_CAP),
                    )
                )
            )
            res["cold_capture_refused"] = _cold_capture_refused(cold, tp, device)
            dist.barrier()
    finally:
        dist.destroy_process_group()
    return res


def _run_pool(world_size, worker, **kwds):
    """Run *worker* on every rank of a fresh spawn pool; its results by rank."""
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    n_gpu = torch.cuda.device_count()
    if n_gpu < world_size:
        raise RuntimeError(f"QuickAllReduceInt4 needs {world_size} GPUs, have {n_gpu}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", "3600"))
    pool = Pool(processes=world_size)
    try:
        results = [
            pool.apply_async(
                worker,
                kwds={
                    "rank": rank,
                    "tp": world_size,
                    "init_method": init_method,
                    **kwds,
                },
            )
            for rank in range(world_size)
        ]
        ranks = [fut.get(timeout=timeout) for fut in results]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return ranks


def _spawn(world_size, pairs, *, fills, time_it, grid_cap=DEFAULT_GRID_CAP):
    return _run_pool(
        world_size,
        _run_rank,
        tokens=[t for t, _ in pairs],
        hiddens=[h for _, h in pairs],
        fills=fills,
        grid_cap=grid_cap,
        time_it=time_it,
    )


def _assert_validity(ranks, *, world_size, label) -> dict:
    fails = []
    for rank, rows in enumerate(ranks):
        for row in rows:
            where = (
                f"{label} tp={world_size} rank {rank} tokens={row['tokens']} "
                f"hidden={row['hidden']} fill={row['fill']}"
            )
            if row["fill"] == "zeros":
                # NMSE is 0 / 0 here: the all-zero check is the accuracy gate
                if not row["rs_zero"]:
                    fails.append(
                        f"{where}: RS of all-zero partials is not zero "
                        f"(int16 bit patterns {row['rs_zero_bits']})"
                    )
                if not row["ag_zero"]:
                    fails.append(
                        f"{where}: AG of all-zero shards is not zero "
                        f"(int16 bit patterns {row['ag_zero_bits']})"
                    )
            else:
                # written so that a NaN fails
                if not row["rs_nmse"] <= NMSE_MAX:
                    fails.append(f"{where}: RS NMSE {row['rs_nmse']:.3e} > {NMSE_MAX}")
                if not row["ag_nmse"] <= NMSE_MAX:
                    fails.append(f"{where}: AG NMSE {row['ag_nmse']:.3e} > {NMSE_MAX}")
                if row["ag_own_equal_frac"] > 0.3:
                    fails.append(
                        f"{where}: own AG block equals its input "
                        f"({row['ag_own_equal_frac']:.2f}); no codec round trip"
                    )
            for mode in ("rs", "ag"):
                if row[f"{mode}_err"] >= CLOSE_ERR_RATIO:
                    fails.append(
                        f"{where}: {mode.upper()} checkAllclose err "
                        f"{row[f'{mode}_err']:.3f} >= {CLOSE_ERR_RATIO}"
                    )
            if not row["rs_clean"]:
                fails.append(f"{where}: RS modified its input or wrote past its output")
            if not row["ag_clean"]:
                fails.append(f"{where}: AG modified its input or wrote past its output")
            if not row["rs_owner_independent"]:
                fails.append(
                    f"{where}: a row's RS value depends on the rank that owns it"
                )
            if not row["ag_ranks_equal"]:
                fails.append(f"{where}: AG output differs between ranks")
            if row["back_to_back_mismatches"]:
                fails.append(
                    f"{where}: {row['back_to_back_mismatches']} delayed "
                    "back-to-back outputs differ from their isolated result"
                )
    if fails:
        raise AssertionError("; ".join(fails))
    return ranks[0][0]


def _assert_extras(ranks, *, world_size) -> None:
    fails = []
    for rank, res in enumerate(ranks):
        where = f"tp={world_size} rank {rank}"
        graph = res["graph"]
        if graph["capture_error"]:
            fails.append(f"{where}: graph capture failed: {graph['capture_error']}")
        else:
            if graph["replay_mismatches"]:
                fails.append(
                    f"{where}: {graph['replay_mismatches']} graph replay outputs "
                    "differ from an eager run of the same inputs"
                )
            # written so that a NaN fails
            if not graph["replay_nmse"] <= NMSE_MAX:
                fails.append(f"{where}: graph replay NMSE {graph['replay_nmse']:.3e}")
        if not graph["compile_sp_untouched"]:
            fails.append(f"{where}: compile_sp wrote to its tensors")
        for name in ("interleave_pinned", "interleave_shipped"):
            if res[name]["mismatches"]:
                fails.append(
                    f"{where}: {name}: {res[name]['mismatches']} of "
                    f"{res[name]['launches']} outputs differ from their isolated result"
                )
        if not res["cold_capture_refused"]:
            fails.append(f"{where}: capture without compile_sp was not refused")
    if fails:
        raise AssertionError("; ".join(fails))


@benchmark()
def test_quick_reduce_scatter_all_gather(
    tokens, hidden, dtype, tp, grid_cap=DEFAULT_GRID_CAP
):
    ranks = _spawn(
        tp, [(tokens, hidden)], fills=_FILLS[:1], time_it=True, grid_cap=grid_cap
    )
    row = _assert_validity(ranks, world_size=tp, label="bench")
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element of the reduce-scatter sum, none in the
    # all-gather; codec ALU work is not counted. TB/s counts the full tensor.
    flops = tokens * hidden * (tp - 1)
    us_rs = max(r[0]["us_rs"] for r in ranks)
    us_ag = max(r[0]["us_ag"] for r in ranks)
    return {
        "gfx": ARCH,
        "tp": tp,
        "grid_cap": row["grid_cap"],
        "flydsl rs us": us_rs,
        "flydsl rs TFLOPS": (flops / us_rs / 1e6) if us_rs else 0.0,
        "flydsl rs TB/s": (nbytes / us_rs / 1e6) if us_rs else 0.0,
        "flydsl rs err": max(r[0]["rs_err"] for r in ranks),
        "flydsl rs nmse": max(r[0]["rs_nmse"] for r in ranks),
        "flydsl ag us": us_ag,
        "flydsl ag TB/s": (nbytes / us_ag / 1e6) if us_ag else 0.0,
        "flydsl ag err": max(r[0]["ag_err"] for r in ranks),
        "flydsl ag nmse": max(r[0]["ag_nmse"] for r in ranks),
    }


test_quick_reduce_scatter_all_gather.__test__ = False


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("QuickAllReduceInt4 unsupported on %s; skipping", ARCH)
        return
    n_gpu = torch.cuda.device_count()
    parser = argparse.ArgumentParser(
        description="FlyDSL INT4 reduce-scatter / all-gather"
    )
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=[2],
        help="world sizes (2, 4 or 8). Default 2.",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=lambda v: tuple(int(p) for p in v.split(",")),
        nargs="*",
        default=[(1024, 2560), (8192, 2560), (16384, 2560)],
        help="(tokens, hidden) pairs for the timed sweep, e.g. -s 8192,2560",
    )
    parser.add_argument("--grid-cap", type=int, default=DEFAULT_GRID_CAP)
    args = parser.parse_args()

    for tp, pairs in _VALIDITY_SHAPES.items():
        if n_gpu < tp:
            aiter.logger.warning(
                "needs %s GPUs, have %s; skipping validity tp=%s", tp, n_gpu, tp
            )
            continue
        for grid_cap in (args.grid_cap, 64):
            ranks = _spawn(tp, pairs, fills=_FILLS, time_it=False, grid_cap=grid_cap)
            _assert_validity(
                ranks,
                world_size=tp,
                label=f"validity cap={grid_cap}",
            )
        _assert_extras(_run_pool(tp, _run_rank_extras), world_size=tp)

    df = []
    for tp, mnk in itertools.product(args.tp, args.mnk):
        if tp not in SUPPORTED_WORLDS or n_gpu < tp:
            aiter.logger.warning("skipping tp=%s (%s GPUs)", tp, n_gpu)
            continue
        tokens, hidden = mnk
        if tokens % tp or (tokens * hidden * 2) % TILE_BYTES:
            aiter.logger.warning(
                "skipping %sx%s: not a whole number of tiles at tp=%s",
                tokens,
                hidden,
                tp,
            )
            continue
        df.append(
            test_quick_reduce_scatter_all_gather(
                tokens, hidden, torch.bfloat16, tp, grid_cap=args.grid_cap
            )
        )
    if df:
        aiter.logger.info(
            "flydsl quick reduce-scatter / all-gather INT4 summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    freeze_support()
    main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Harness shared by the FlyDSL communication tests.

The all-reduce (``test_flydsl_quick_allreduce``, ``test_flydsl_one_shot_allreduce``)
and all-to-all (``test_flydsl_quick_alltoall``) tests all run the same way:
each world size is one ``multiprocessing`` pool of rank workers, cases are
registered up front and grouped by the engine they need so each engine is built
by exactly one spawn, and every case is graded on SQNR against an fp32
reference. This module is that machinery; the tests keep their cases, workers
and gates.

A rank worker is a module-level function (the spawn start method pickles it by
name) called as ``rank_fn(rank=, tp=, init_method=, engine_kw=, cases=,
window=)``.
"""

from __future__ import annotations

import math
import os
import sys
from multiprocessing import Pool, set_start_method

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch

import aiter
from aiter.dist.utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.collectives_shared import SUPPORTED_WORLDS

# Rank workers initialize HIP, which a forked child cannot.
set_start_method("spawn", force=True)

SUPPORTED_ARCHS = ("gfx942", "gfx950")

# Seconds to wait for each rank of a spawn. The kernels spin on flags written
# by peers, so a protocol bug is a hang rather than an error; this turns it
# into a failure.
SPAWN_TIMEOUT_S = 600

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


def run_on_ranks(world_size: int, rank_fn, timeout: float = SPAWN_TIMEOUT_S, **kwds):
    """Run ``rank_fn(rank=, tp=, init_method=, **kwds)`` on *world_size* ranks.

    Returns each rank's result, in rank order. A rank that does not return
    within *timeout* seconds fails the spawn: the kernels spin on peers' flags,
    so a protocol bug is a hang, and this is what turns it into an error.
    """
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={world_size}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=world_size)
    try:
        results = [
            pool.apply_async(
                rank_fn,
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


def spawn_ranks(
    world_size: int,
    rank_fn,
    engine_kw: dict,
    cases: list,
    window: tuple[int, int] | None = None,
) -> list:
    """:func:`run_on_ranks` for a case-list worker, ``rank_fn(..., engine_kw=,
    cases=, window=)``."""
    return run_on_ranks(
        world_size, rank_fn, engine_kw=engine_kw, cases=cases, window=window
    )


class SpawnRegistry:
    """Cases grouped by the engine they need, so each engine spawns once.

    A key is ``(tp, sorted engine kwargs)`` (:meth:`key`); a case is whatever
    tuple the rank worker understands. Register every case first, then read
    results: the first read of a key spawns its engine with every case
    registered for it.
    """

    def __init__(self, rank_fn):
        self._rank_fn = rank_fn
        self.cases: dict[tuple, list] = {}
        # Production dispatch window of an engine whose kernel coverage is
        # checked, per key.
        self.windows: dict[tuple, tuple[int, int]] = {}
        self._ranks: dict[tuple, list] = {}

    @staticmethod
    def key(tp: int, **engine_kw) -> tuple:
        return (tp, tuple(sorted(engine_kw.items())))

    def register(self, key: tuple, case) -> None:
        cases = self.cases.setdefault(key, [])
        if case not in cases:
            cases.append(case)

    def ranks(self, key: tuple) -> list:
        """Every rank's raw result for *key*, spawning on first use."""
        if key not in self._ranks:
            self._ranks[key] = spawn_ranks(
                key[0],
                self._rank_fn,
                dict(key[1]),
                self.cases[key],
                self.windows.get(key),
            )
        return self._ranks[key]

    def result(self, key: tuple, case) -> list:
        """Per-rank rows for *case*, for a worker that returns one row per case."""
        i = self.cases[key].index(case)
        return [rank_rows[i] for rank_rows in self.ranks(key)]


class FailureLog:
    """Collects gate failures so a sweep reports all of them, then fails once."""

    def __init__(self):
        self.failures: list[str] = []

    def check(self, label: str, fails: list[str]) -> bool:
        if fails:
            msg = f"{label}: " + "; ".join(fails)
            aiter.logger.error(msg)
            self.failures.append(msg)
        return not fails

    def raise_if_failed(self, what: str) -> None:
        """Exit non-zero naming every failure, if there were any."""
        if self.failures:
            raise SystemExit(
                f"{len(self.failures)} {what} check(s) failed:\n  "
                + "\n  ".join(self.failures)
            )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def sqnr(ref_pow: torch.Tensor, mse: torch.Tensor) -> torch.Tensor:
    """``10 log10(ref_pow / mse)``: +inf when both are zero, -inf on NaN."""
    score = torch.where(
        (ref_pow <= 0) & (mse <= 0),
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10(ref_pow / mse),
    )
    return torch.nan_to_num(
        score, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )


def sqnr_db(got: torch.Tensor, reference: torch.Tensor) -> float:
    got = got.to(torch.float32)
    reference = reference.to(torch.float32)
    return float(
        sqnr((reference * reference).mean(), ((got - reference) ** 2).mean()).item()
    )


def min_tile_sqnr_db(
    got: torch.Tensor, reference: torch.Tensor, tile_bytes: int, elem_bytes: int = 2
) -> float:
    """Worst per-tile SQNR, so one unwritten tile cannot be averaged away."""
    tile_elems = tile_bytes // elem_bytes
    g = got.reshape(-1).to(torch.float32)
    r = reference.reshape(-1).to(torch.float32)
    n = int(g.numel())
    n_full = (n // tile_elems) * tile_elems
    vals = []
    if n_full:
        gt = g[:n_full].view(-1, tile_elems)
        rt = r[:n_full].view(-1, tile_elems)
        mse = ((gt - rt) ** 2).mean(dim=1)
        pow_ = (rt * rt).mean(dim=1)
        vals.append(float(sqnr(pow_, mse).min().item()))
    if n > n_full:
        vals.append(sqnr_db(g[n_full:], r[n_full:]))
    return min(vals) if vals else sqnr_db(got, reference)


def rel_mae(got: torch.Tensor, reference: torch.Tensor) -> float:
    scale = float(reference.abs().mean().item())
    err = float((got.to(torch.float32) - reference).abs().mean().item())
    return err / scale if scale else 0.0


def lanes_differing(out: torch.Tensor, group=None) -> int:
    """16-bit lanes where any rank's output differs, bit for bit, from rank 0's.

    Widened to int32 because gloo rejects int16 ("Invalid scalar type"); the
    widening is exact, so the comparison is still on bits. Moved to the CPU
    when *group* is a gloo group.
    """
    import torch.distributed as dist

    bits = out.contiguous().view(torch.int16).to(torch.int32)
    if dist.get_backend(group) == dist.Backend.GLOO:
        bits = bits.cpu()
    gathered = [torch.empty_like(bits) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, bits, group=group)
    return max(int((g != gathered[0]).sum().item()) for g in gathered)


def worst(a: dict, b: dict) -> dict:
    """Per-field worst of two metric dicts, for a row checked several times.

    Fields ending in ``sqnr_db`` take the minimum; ``rel_mae`` takes the larger
    with NaN winning; ``first_bad`` keeps the first one seen; every other
    numeric field takes the maximum.
    """
    out = dict(a)
    for key, bv in b.items():
        av = a.get(key)
        if key.endswith("sqnr_db"):
            out[key] = min(av, bv)
        elif key == "rel_mae":
            if not math.isfinite(bv) or bv > av:
                out[key] = bv
        elif key == "first_bad":
            out[key] = bv if av < 0 else av
        elif isinstance(bv, (int, float)) and not isinstance(bv, bool):
            out[key] = max(av, bv)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_bytes(nbytes: int, inf_at: int | None = None) -> str:
    if inf_at is not None and nbytes >= inf_at:
        return "inf"
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if nbytes >= 1 << shift:
            return f"{nbytes / (1 << shift):g} {unit}"
    return f"{nbytes} B"


def summarize(name: str, rows: list[dict], drop=()) -> None:
    if rows:
        df = pd.DataFrame(rows).drop(columns=list(drop), errors="ignore")
        aiter.logger.info(
            "%s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )

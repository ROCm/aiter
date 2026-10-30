# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and timing for the AttnRes form of the one-shot all-reduce
(``OneShotAllReduceAttnRes``: all-reduce + Kimi-K3's attention-residual
aggregation, SILOTIGER-1277).

``--scenario`` selects what a run does: ``verify`` (the default) checks the
kernel against a torch reference, and ``bench`` compares the fused kernel with
the two kernels production runs today.

**verify** -- ``test_one_shot_allreduce_attn_res`` checks five things per
shape, and the last three matter more than an error norm does:

1. ``out`` is right against an fp32 torch reference, at the bf16 rounding
   floor (SQNR, and a max-ulp count that tolerates the single bf16 rounding).
2. ``prefix_out`` is **bit-exact** against the reference. Both sides sum the
   ranks in order in fp32 and round exactly once, so equality is the correct
   expectation, and it is the only cheap way to catch a missing bf16 round
   trip before the residual add -- an error an SQNR check on ``out`` is far
   too loose to see and that compounds per layer in a real model.
3. With ``write_bank``, ``bank[t, nvb]`` is bit-exact against the same
   ``prefix_out``, and no other bank row is touched. The snapshot is what the
   next decode step scores against, so a row written to the wrong offset would
   surface a layer later, not here.
4. Every rank produces **bit-identical** ``out`` and ``prefix_out``. The
   all-reduce accumulates in a fixed rank order for exactly this reason, and
   an accuracy check cannot see an ordering bug -- both answers would be
   equally "accurate". The scores are a softmax over block-wide sums, so a
   reduction that is not thread-stable shows up here too.
5. All of the above again after a CUDA-graph replay, since that is how the
   decode loop calls it.

The sweep walks ``tokens`` x ``nvb`` x ``write_bank`` x ``out_norm``:
``nvb`` is a build constant -- the bank rows are unrolled -- so each bank
depth is its own binary, as is each ``(write_bank, out_norm)``.

**bench** -- ``bench_one_shot_allreduce_attn_res`` times three paths on the
same shapes, all CUDA-graphed:

* ``fused`` -- ``OneShotAllReduceAttnRes.allreduce_attn_res``, this kernel.
* ``ar`` -- the plain one-shot all-reduce alone, so ``epilogue us`` (the
  difference) is what the aggregation costs on top of the collective. That
  difference, not the absolute time, is what a change to the epilogue moves,
  and taking both in the same process makes it immune to host drift.

There is deliberately no unfused candidate. The kernel this replaces is
SGLang's Triton ``attn_res_hip``, which does not live in this repo, and
timing the torch reference as a stand-in measured 100-190 us against the
fused path's 9-27 us -- an eager-torch number that says nothing about the
kernel. Compare against Triton in SGLang, or against ``epilogue us`` here.

Every rank is a ``multiprocessing`` spawn worker; one spawn per world size
runs every sweep. The kernel runs on gfx942/gfx950; other archs skip, and
``main()`` skips a world size when fewer GPUs are visible than TP.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import itertools
import os
import sys
import traceback
from multiprocessing import Pool, freeze_support, set_start_method

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# The ranks are spawned, so they re-import this file and need the same paths.
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.one_shot_allreduce import (
    OneShotAllReduce,
    OneShotAllReduceAttnRes,
)
from aiter.test_common import benchmark

set_start_method("spawn", force=True)

try:
    ARCH = get_gfx_runtime()
except (KeyError, RuntimeError):
    ARCH = None

SUPPORTED_GFX = ("gfx942", "gfx950")
SUPPORTED_WORLDS = (2, 4, 8)

HIDDEN = 7168
#: Bank rows the model keeps per token. ``nvb`` of them are scored and the
#: snapshot is written to row ``nvb``, so ``NB`` has to clear ``max(nvb) + 1``.
NB = 9
SCORE_EPS = 1e-6
OUT_EPS = 1e-6

VERIFY_TOKENS = (1, 2, 4, 8, 16, 32, 64)
VERIFY_NVB = (1, 4, 8)
BENCH_TOKENS = (1, 4, 16, 64)
BENCH_NVB = (4, 8)

#: bf16 keeps 8 explicit mantissa bits, so one rounding is 2**-8 relative.
#: The reference rounds once and the kernel rounds once, at different points
#: in the same expression, which is worth at most a couple of those steps.
SQNR_FLOOR_DB = 40.0
MAX_ULP = 2.0

GRAPH_INNER = 10
GRAPH_REPLAYS = 20
ROUNDS = 8
SPAWN_TIMEOUT_S = 900

#: The harness pins ``max_bytes`` so the sweep can reach shapes the dispatch
#: policy would route elsewhere; see ``resolve_fused_oneshot_max_bytes``.
MAX_BYTES = 1 << 30

_RESULTS: dict[int, list] = {}
_FAILURES: list[str] = []


def run_torch(prefix_in, bank, cw, ow, nvb, score_eps, out_eps, addend):
    """Reference for the AttnRes aggregation, fp32 throughout.

    Same order of operations as the kernel: the residual add is rounded to
    bf16 *before* it is scored, because that is what downstream readers see;
    the scores take a global max before ``exp``; the mix is over the raw rows.
    Returns ``(out, prefix_out)``.
    """
    hidden = prefix_in.shape[-1]
    prefix = (prefix_in.float() + addend.float()).to(prefix_in.dtype)
    pv = prefix.float()
    tile = bank[:, :nvb].float()
    cwf = cw.float()
    p_score = (pv * cwf).sum(-1) / torch.sqrt((pv * pv).sum(-1) / hidden + score_eps)
    b_score = (tile * cwf).sum(-1) / torch.sqrt(
        (tile * tile).sum(-1) / hidden + score_eps
    )
    m = torch.maximum(b_score.max(-1).values, p_score)
    b_w = torch.exp(b_score - m[:, None])
    p_w = torch.exp(p_score - m)
    inv = 1.0 / (b_w.sum(-1) + p_w)
    acc = pv * (p_w * inv)[:, None] + (b_w[..., None] * inv[:, None, None] * tile).sum(
        1
    )
    if ow is not None:
        acc = (
            acc
            / torch.sqrt((acc * acc).sum(-1, keepdim=True) / hidden + out_eps)
            * ow.float()
        )
    return acc.to(prefix_in.dtype), prefix


def _sqnr_db(ref, got) -> float:
    ref = ref.float()
    err = (got.float() - ref).pow(2).sum().item()
    sig = ref.pow(2).sum().item()
    if err == 0.0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(sig / err)).item()


def _max_ulp(ref, got) -> float:
    """Largest difference in bf16 steps of *ref*, which has 8 mantissa bits.

    The clamp keeps a denormal-sized reference from turning a last-bit
    difference into a huge ratio.
    """
    ref = ref.float()
    step = ref.abs().clamp_min(1e-3) * 2.0**-7
    return ((got.float() - ref).abs() / step).max().item()


def _inputs(tokens, nvb, case_idx, tp, device):
    """The same inputs on every rank: seeded on the host, then moved."""
    gen = torch.Generator(device="cpu").manual_seed(1000 + case_idx)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device)

    parts = [rnd(tokens, HIDDEN).bfloat16() for _ in range(tp)]
    prefix_in = rnd(tokens, HIDDEN, scale=4.0).bfloat16()
    bank = rnd(tokens, NB, HIDDEN, scale=4.0).bfloat16()
    cw = rnd(HIDDEN, scale=0.05).float()
    ow = (1.0 + 0.1 * rnd(HIDDEN)).bfloat16()
    return parts, prefix_in, bank, cw, ow


def _rank_order_sum(parts):
    """The fp32 rank-order sum the kernel computes, rounded once."""
    acc = None
    for part in parts:
        acc = part.float() if acc is None else acc + part.float()
    return acc.bfloat16()


def _verify_case(eng, tokens, nvb, write_bank, out_norm, case_idx, rank, tp, device):
    parts, prefix_in, bank, cw, ow_full = _inputs(tokens, nvb, case_idx, tp, device)
    ow = ow_full if out_norm else None
    addend = _rank_order_sum(parts)
    ref_out, ref_prefix = run_torch(
        prefix_in, bank, cw, ow, nvb, SCORE_EPS, OUT_EPS, addend
    )

    bank_got = bank.clone()
    out, prefix_out = eng.allreduce_attn_res(
        parts[rank],
        prefix_in,
        bank_got,
        nvb,
        cw,
        ow,
        SCORE_EPS,
        OUT_EPS,
        write_bank=write_bank,
    )
    torch.cuda.synchronize()

    # The decode loop calls this under a graph, so check the replay too.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eng.allreduce_attn_res(
            parts[rank],
            prefix_in,
            bank_got,
            nvb,
            cw,
            ow,
            SCORE_EPS,
            OUT_EPS,
            out=out,
            prefix_out=prefix_out,
            write_bank=write_bank,
        )
    graph.replay()
    torch.cuda.synchronize()

    untouched = [j for j in range(NB) if j != (nvb if write_bank else -1)]
    row = {
        "sqnr_db": _sqnr_db(ref_out, out),
        "max_ulp": _max_ulp(ref_out, out),
        "prefix_exact": bool(torch.equal(prefix_out, ref_prefix)),
        "snapshot_exact": (
            bool(torch.equal(bank_got[:, nvb], prefix_out)) if write_bank else True
        ),
        "bank_untouched": bool(torch.equal(bank_got[:, untouched], bank[:, untouched])),
        "finite": bool(torch.isfinite(out.float()).all()),
        # Doubles so a rank-to-rank comparison is exact rather than rounded.
        "out_sig": float(out.float().double().sum()),
        "prefix_sig": float(prefix_out.float().double().sum()),
    }
    # Hand the graph back before it is collected, so its pool is released
    # before the next case captures.
    del graph

    # SILOTIGER-1277 acc-cache (Kevin Mi sglang#44988 port): the same op fed the
    # caller-precomputed (acc, ms) must land on the same bf16 output and the same
    # bit-exact prefix/snapshot -- the fused read then never touches bank values.
    bank_acc = eng.precompute_bank_acc(bank, cw, nvb, SCORE_EPS)
    bank_acc_got = bank.clone()
    out_acc, prefix_acc = eng.allreduce_attn_res(
        parts[rank],
        prefix_in,
        bank_acc_got,
        nvb,
        cw,
        ow,
        SCORE_EPS,
        OUT_EPS,
        write_bank=write_bank,
        bank_acc=bank_acc,
    )
    torch.cuda.synchronize()
    row.update(
        acc_sqnr_db=_sqnr_db(ref_out, out_acc),
        acc_max_ulp=_max_ulp(ref_out, out_acc),
        acc_prefix_exact=bool(torch.equal(prefix_acc, ref_prefix)),
        acc_snapshot_exact=(
            bool(torch.equal(bank_acc_got[:, nvb], prefix_acc))
            if write_bank
            else True
        ),
        acc_bank_untouched=bool(
            torch.equal(bank_acc_got[:, untouched], bank[:, untouched])
        ),
        acc_finite=bool(torch.isfinite(out_acc.float()).all()),
    )

    # SILOTIGER-1277 iter004 split ladder: the same acc-cache op with the decode
    # split auto-selected (AITER_FLY_AR_FUSED_SPLIT=1, no manual pin) must land on
    # the same bf16 output and the same bit-exact prefix/snapshot. The engine
    # splits the fused epilogue over several workgroups only on the acc_cache path
    # and only inside the measured band (nvb>=8: T<=18, nvb>=4: T<=8); larger T
    # falls back to the unsplit build. Every rank runs the same case under the
    # same flag, so the lazy split-engine build stays a lockstep collective.
    prev_split = os.environ.get("AITER_FLY_AR_FUSED_SPLIT")
    os.environ["AITER_FLY_AR_FUSED_SPLIT"] = "1"
    try:
        bank_split_got = bank.clone()
        bank_acc_split = eng.precompute_bank_acc(bank, cw, nvb, SCORE_EPS)
        out_split, prefix_split = eng.allreduce_attn_res(
            parts[rank],
            prefix_in,
            bank_split_got,
            nvb,
            cw,
            ow,
            SCORE_EPS,
            OUT_EPS,
            write_bank=write_bank,
            bank_acc=bank_acc_split,
        )
        torch.cuda.synchronize()
    finally:
        if prev_split is None:
            os.environ.pop("AITER_FLY_AR_FUSED_SPLIT", None)
        else:
            os.environ["AITER_FLY_AR_FUSED_SPLIT"] = prev_split
    row.update(
        split_used=int(eng._attn_ladder_split(tokens, nvb)),
        split_max_ulp=_max_ulp(ref_out, out_split),
        split_prefix_exact=bool(torch.equal(prefix_split, ref_prefix)),
        split_snapshot_exact=(
            bool(torch.equal(bank_split_got[:, nvb], prefix_split))
            if write_bank
            else True
        ),
        split_bank_untouched=bool(
            torch.equal(bank_split_got[:, untouched], bank[:, untouched])
        ),
        split_finite=bool(torch.isfinite(out_split.float()).all()),
    )
    return row


def _time_graph(fn, group):
    """Microseconds per call, median over ``ROUNDS`` of replayed graphs."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_INNER):
            fn()

    samples = []
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        dist.barrier(group=group)
        graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(GRAPH_REPLAYS):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / (GRAPH_REPLAYS * GRAPH_INNER))
    samples.sort()
    return samples[len(samples) // 2], samples[-1] - samples[0]


def _bench_case(eng, plain, tokens, nvb, case_idx, rank, tp, device, group):
    parts, prefix_in, bank, cw, ow = _inputs(tokens, nvb, case_idx, tp, device)
    out = torch.empty_like(prefix_in)
    prefix_out = torch.empty_like(prefix_in)
    reduced = torch.empty_like(prefix_in)
    mine = parts[rank]

    def fused():
        eng.allreduce_attn_res(
            mine,
            prefix_in,
            bank,
            nvb,
            cw,
            ow,
            SCORE_EPS,
            OUT_EPS,
            out=out,
            prefix_out=prefix_out,
            write_bank=True,
        )

    def ar_only():
        plain.allreduce(mine, reduced)

    fused_us, fused_spread = _time_graph(fused, group)
    ar_us, _ = _time_graph(ar_only, group)
    return {
        "fused us": fused_us,
        "fused spread us": fused_spread,
        "ar us": ar_us,
    }


def _run_rank(rank, tp, init_method, scenario, cases):
    """One rank of a spawn: build the engines, walk *cases*, tear down."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", init_method=init_method, world_size=tp, rank=rank)
    group = dist.group.WORLD
    eng = OneShotAllReduceAttnRes(
        group=group, device=device, rank=rank, world_size=tp, max_bytes=MAX_BYTES
    )
    plain = (
        OneShotAllReduce(
            group=group, device=device, rank=rank, world_size=tp, max_bytes=MAX_BYTES
        )
        if scenario == "bench"
        else None
    )
    rows = []
    try:
        for case_idx, case in enumerate(cases):
            if scenario == "verify":
                tokens, nvb, write_bank, out_norm = case
                rows.append(
                    _verify_case(
                        eng,
                        tokens,
                        nvb,
                        write_bank,
                        out_norm,
                        case_idx,
                        rank,
                        tp,
                        device,
                    )
                )
            else:
                tokens, nvb = case
                rows.append(
                    _bench_case(
                        eng, plain, tokens, nvb, case_idx, rank, tp, device, group
                    )
                )
    finally:
        eng.close()
        if plain is not None:
            plain.close()
        dist.destroy_process_group()
    return {"rank": rank, "rows": rows}


def _spawn(tp: int, scenario: str, cases: list) -> list:
    """``_run_rank`` on every rank of a new spawn, with the rendezvous."""
    if tp not in SUPPORTED_WORLDS:
        raise ValueError(f"unsupported world_size={tp}")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=tp)
    try:
        futures = [
            pool.apply_async(
                _run_rank,
                kwds={
                    "rank": rank,
                    "tp": tp,
                    "init_method": init_method,
                    "scenario": scenario,
                    "cases": cases,
                },
            )
            for rank in range(tp)
        ]
        timeout = float(os.environ.get("FLYDSL_QR_TIMEOUT", str(SPAWN_TIMEOUT_S)))
        ranks = [fut.get(timeout=timeout) for fut in futures]
    except Exception:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return ranks


def _check(label: str, fails: list[str]) -> None:
    if fails:
        msg = f"{label}: " + "; ".join(fails)
        aiter.logger.error(msg)
        _FAILURES.append(msg)


def _graded_benchmark(fn):
    """``@benchmark()`` for a test written with plain ``assert``.

    A failed assert is recorded through ``_check`` instead of raised, so one
    bad case does not stop the sweep and ``main`` reports every failure at the
    end. The row ``benchmark`` builds gains ``passed``.
    """
    bench = benchmark()(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            row = bench(*args, **kwargs)
        except AssertionError as exc:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            row = inspect.getcallargs(fn, *args, **kwargs)
            given = ", ".join(f"{k}={v}" for k, v in row.items())
            where = f"{os.path.basename(frame.filename)}:{frame.lineno}"
            _check(f"{fn.__name__}({given})", [f"{exc or frame.line} [{where}]"])
            return {**row, "passed": False}
        return {**row, "passed": True}

    return wrapper


@_graded_benchmark
def test_one_shot_allreduce_attn_res(tokens, nvb, write_bank, out_norm, tp):
    rows = [
        r["rows"][_CASE_INDEX[(tokens, nvb, write_bank, out_norm)]]
        for r in _RESULTS[tp]
    ]
    fails = []
    for rank, row in enumerate(rows):
        if not row["finite"]:
            fails.append(f"rank {rank}: out has non-finite lanes")
        if row["sqnr_db"] < SQNR_FLOOR_DB:
            fails.append(
                f"rank {rank}: SQNR {row['sqnr_db']:.2f} dB below the "
                f"{SQNR_FLOOR_DB} dB bf16 floor"
            )
        if row["max_ulp"] > MAX_ULP:
            fails.append(
                f"rank {rank}: {row['max_ulp']:.2f} bf16 ulp from the reference, "
                f"above {MAX_ULP}"
            )
        if not row["prefix_exact"]:
            fails.append(
                f"rank {rank}: prefix_out is not bit-exact against the "
                "rank-order reference"
            )
        if not row["snapshot_exact"]:
            fails.append(f"rank {rank}: bank[:, {nvb}] does not hold prefix_out")
        if not row["bank_untouched"]:
            fails.append(f"rank {rank}: a bank row other than {nvb} was written")
        # SILOTIGER-1277 acc-cache path: same reference, same gates.
        if not row["acc_finite"]:
            fails.append(f"rank {rank}: acc-cache out has non-finite lanes")
        if row["acc_sqnr_db"] < SQNR_FLOOR_DB:
            fails.append(
                f"rank {rank}: acc-cache SQNR {row['acc_sqnr_db']:.2f} dB below "
                f"the {SQNR_FLOOR_DB} dB bf16 floor"
            )
        if row["acc_max_ulp"] > MAX_ULP:
            fails.append(
                f"rank {rank}: acc-cache {row['acc_max_ulp']:.2f} bf16 ulp from "
                f"the reference, above {MAX_ULP}"
            )
        if not row["acc_prefix_exact"]:
            fails.append(
                f"rank {rank}: acc-cache prefix_out is not bit-exact against the "
                "rank-order reference"
            )
        if not row["acc_snapshot_exact"]:
            fails.append(
                f"rank {rank}: acc-cache bank[:, {nvb}] does not hold prefix_out"
            )
        if not row["acc_bank_untouched"]:
            fails.append(
                f"rank {rank}: acc-cache wrote a bank row other than {nvb}"
            )
        # SILOTIGER-1277 iter004 split-ladder path: auto-selected decode split on
        # the acc-cache read; same reference, same gates.
        if not row["split_finite"]:
            fails.append(f"rank {rank}: split out has non-finite lanes")
        if row["split_max_ulp"] > MAX_ULP:
            fails.append(
                f"rank {rank}: split (split={row['split_used']}) "
                f"{row['split_max_ulp']:.2f} bf16 ulp from the reference, above "
                f"{MAX_ULP}"
            )
        if not row["split_prefix_exact"]:
            fails.append(
                f"rank {rank}: split prefix_out is not bit-exact against the "
                "rank-order reference"
            )
        if not row["split_snapshot_exact"]:
            fails.append(
                f"rank {rank}: split bank[:, {nvb}] does not hold prefix_out"
            )
        if not row["split_bank_untouched"]:
            fails.append(
                f"rank {rank}: split wrote a bank row other than {nvb}"
            )
    identical = len({(r["out_sig"], r["prefix_sig"]) for r in rows}) == 1
    if not identical:
        fails.append("ranks do not agree bit for bit")
    _check(
        f"tp={tp} {tokens}x{HIDDEN} nvb={nvb} wb={write_bank} norm={out_norm}", fails
    )
    return {
        "gfx": ARCH,
        "sqnr_db": min(r["sqnr_db"] for r in rows),
        "max_ulp": max(r["max_ulp"] for r in rows),
        "prefix_exact": all(r["prefix_exact"] for r in rows),
        "snapshot_exact": all(r["snapshot_exact"] for r in rows),
        "bit_identical": identical,
        "acc_sqnr_db": min(r["acc_sqnr_db"] for r in rows),
        "acc_max_ulp": max(r["acc_max_ulp"] for r in rows),
        "acc_prefix_exact": all(r["acc_prefix_exact"] for r in rows),
    }


@benchmark()
def bench_one_shot_allreduce_attn_res(tokens, nvb, tp):
    rows = [r["rows"][_CASE_INDEX[(tokens, nvb)]] for r in _RESULTS[tp]]
    # The slowest rank is when the collective finishes.
    fused_us = max(r["fused us"] for r in rows)
    ar_us = max(r["ar us"] for r in rows)
    # Bytes the epilogue moves on top of the collective: the prefix in and
    # out, the nvb bank rows it scores, the snapshot, and ``out``.
    nbytes = tokens * HIDDEN * 2 * (nvb + 4)
    return {
        "gfx": ARCH,
        "fused us": fused_us,
        "ar us": ar_us,
        "epilogue us": fused_us - ar_us,
        "fused TB/s": nbytes / fused_us / 1e6,
        "spread us": max(r["fused spread us"] for r in rows),
    }


#: Case -> its index in the per-rank row list, filled by ``main``.
_CASE_INDEX: dict[tuple, int] = {}


def _summarize(title: str, rows: list) -> None:
    frame = pd.DataFrame(rows)
    aiter.logger.info("%s:\n%s", title, frame.to_markdown(index=False))


def main():
    # The whole-op arch gate goes here, not inside the test functions:
    # ``benchmark`` always returns the call-args dict, so a return from inside
    # one still emits an args-only row.
    if ARCH not in SUPPORTED_GFX:
        aiter.logger.warning(
            "OneShotAllReduceAttnRes is unsupported on %s; skipping", ARCH
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        nargs="*",
        default=["verify"],
        help="verify (check the kernel) and/or bench (time it against the "
        "collective alone)",
    )
    parser.add_argument(
        "-t",
        "--tp",
        type=int,
        nargs="*",
        default=[2],
        help="world sizes to run; one spawn each",
    )
    parser.add_argument(
        "-m",
        "--tokens",
        type=int,
        nargs="*",
        default=None,
        help="token counts to sweep (default: the scenario's own list)",
    )
    parser.add_argument(
        "-v",
        "--nvb",
        type=int,
        nargs="*",
        default=None,
        help="bank depths to sweep; each is its own binary",
    )
    args = parser.parse_args()

    visible = torch.cuda.device_count()
    for tp in args.tp:
        if tp not in SUPPORTED_WORLDS:
            aiter.logger.warning("world size %d is not supported; skipping", tp)
            continue
        if visible < tp:
            aiter.logger.warning(
                "tp=%d needs %d GPUs, %d visible; skipping", tp, tp, visible
            )
            continue

        if "verify" in args.scenario:
            tokens = args.tokens or list(VERIFY_TOKENS)
            nvbs = args.nvb or list(VERIFY_NVB)
            cases = [
                (m, v, wb, norm)
                for m, v, wb, norm in itertools.product(
                    tokens, nvbs, (False, True), (True, False)
                )
            ]
            _CASE_INDEX.clear()
            _CASE_INDEX.update({case: i for i, case in enumerate(cases)})
            _RESULTS[tp] = _spawn(tp, "verify", cases)
            _summarize(
                f"one-shot all-reduce + AttnRes, tp={tp}, hidden={HIDDEN} (verify)",
                [test_one_shot_allreduce_attn_res(*case, tp) for case in cases],
            )

        if "bench" in args.scenario:
            tokens = args.tokens or list(BENCH_TOKENS)
            nvbs = args.nvb or list(BENCH_NVB)
            cases = list(itertools.product(tokens, nvbs))
            _CASE_INDEX.clear()
            _CASE_INDEX.update({case: i for i, case in enumerate(cases)})
            _RESULTS[tp] = _spawn(tp, "bench", cases)
            _summarize(
                f"one-shot all-reduce + AttnRes, tp={tp}, hidden={HIDDEN} (bench)",
                [bench_one_shot_allreduce_attn_res(*case, tp) for case in cases],
            )

    if _FAILURES:
        raise AssertionError(
            f"{len(_FAILURES)} case(s) failed:\n" + "\n".join(_FAILURES)
        )


if __name__ == "__main__":
    freeze_support()
    main()

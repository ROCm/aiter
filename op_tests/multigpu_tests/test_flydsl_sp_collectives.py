# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness for ``SequenceParallelCollectives``, the INT4
sequence-parallel router over ``FlyQuickAllReduce.reduce_scatter`` /
``all_gather``.

``python3`` this file runs a spawn worker per rank, each with its own
``FlyQuickAllReduce``, on two configurations: TP2 through the factory
``create_sp_collectives`` with the platform's window, and TP4 with a router
built directly over an engine that has passed ``compile_sp`` and a window that
admits every size (the table has no TP4 row; the padding rules differ by world
size). The accuracy oracle does not use the router's padding rule, and the
engine comparison takes its unit from ``block_row_unit``, which
``tests/test_sp_collectives.py`` pins to hand-derived values:

* accuracy: NMSE of the router's output against the exact fp32 sum
  (reduce-scatter) or concatenation (all-gather), below the 7444 test's bound,
  so a wrong block placement or unpadding, which gives an NMSE near 1, fails;
* bit identity with the engine called directly on an input padded in the test
  (zero rows after each block), and the output equal to its unpadded prefix;
* padding rows must be zero: the router pads into a block the caching allocator
  just held NaN in, and a hidden size of 2568 with an odd block of rows puts a
  codec group across a real row and a padding row, so an unzeroed padding row
  shows as NaN or a large error in a real row;
* every output the shape the caller expects, equal on every rank for the
  all-gather, and the input left unchanged;
* a payload below the window is not served (``None``);
* both modes captured into one CUDA graph after the factory's ``compile_sp``
  replay with new inputs, bit for bit equal to an eager run of the same inputs;
* ``checkAllclose`` against the same exact fp32 result with the 7444 test's
  tolerances and mismatch bound. It is supplementary there too: the NMSE bound
  and the bit identity are the strict checks.

After the checks, a timed sweep of the same shapes (``@benchmark``,
``run_perftest`` with CUDA events, a markdown summary): the router, the engine on
an input padded here (no padding copy, no copy of the result back to the unpadded
shape) and the exact collective (RCCL), each output checked against the exact
result. Times are reported, not asserted. A world size whose factory returned
None has no router and is not timed.

Needs 2 GPUs, and 4 for the TP4 configuration: a size named with ``--tp`` that
cannot run is an error, the default sweep skips the sizes that do not fit and
fails if none does. The factory must return a router exactly on gfx950 / xGMI /
TP2, the one platform with a window row, and None elsewhere. The expectation is
written out here, so a changed window table or a misdetected link fails the
TP2 configuration; where the factory returns None it checks nothing more. Other
archs skip.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import freeze_support

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# The ranks are spawned, so they re-import this file and need the same paths.
for _path in (_REPO_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flydsl_allreduce_bench as fab
import pandas as pd
import torch
from test_flydsl_quick_reduce_scatter_all_gather import (
    CLOSE_ERR_RATIO,
    NMSE_MAX,
    SUPPORTED_ARCHS,
    _bits,
    _close_err,
    _gather,
    _init_rank,
    _make_inp,
    _nmse,
    _run_pool,
)

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, run_perftest

# (tokens, hidden): padded and unpadded blocks, and a hidden size with an odd
# block of rows whose last real row ends mid codec group (1651 * 2568 elements
# is 8 mod 16, as is 511 * 2568), so a padding row shares a group with it. TP2
# sizes are inside the 8 MiB window (the bytes sent: 1640 * 2560 * 2 is 8,396,800).
# Below it: 1.3 MB, and two rows of hidden 2568, which pad to 10.5 MB (321
# tiles) but send 10 KB.
# The one platform with a window row (allreduce_policy.SP_INT4_WINDOWS): the
# factory returns a router there and None everywhere else.
_ROUTER_PLATFORM = ("gfx950", "xgmi", 2)
_TP2_CASES = ((1640, 2560), (4096, 2560), (3302, 2568))
_TP2_BELOW_WINDOW = ((256, 2560), (2, 2568))
_TP4_CASES = ((40, 2560), (2044, 2568), (4096, 2560))
_GRAPH_CASE = (1640, 2560)
_GRAPH_REPLAYS = 3


def _poison(elements: int, device) -> None:
    """Free a block of *elements* bf16 NaNs, which the caching allocator hands
    to the next request of that size: the router's padded buffer."""
    garbage = torch.full((elements,), float("nan"), dtype=torch.bfloat16, device=device)
    del garbage


def _padded(fly, rows: int, hidden: int) -> int:
    from aiter.dist.device_communicators import sp_collectives as sp

    unit = sp.block_row_unit(fly.world_size, hidden)
    return -(-rows // unit) * unit


def _engine_rs(fly, x: torch.Tensor, rows: int, hidden: int, device) -> torch.Tensor:
    """The engine's reduce-scatter of *x*, padded here with zero rows."""
    tp, padded = fly.world_size, _padded(fly, rows, hidden)
    src = torch.zeros((tp, padded, hidden), dtype=torch.bfloat16, device=device)
    src[:, :rows] = x.view(tp, rows, hidden)
    out = torch.empty((padded, hidden), dtype=torch.bfloat16, device=device)
    fly.reduce_scatter(src.view(tp * padded, hidden), out)
    torch.cuda.synchronize()
    return out[:rows]


def _engine_ag(fly, s: torch.Tensor, device) -> torch.Tensor:
    """The engine's all-gather of *s*, padded here with zero rows."""
    tp, (rows, hidden) = fly.world_size, s.shape
    padded = _padded(fly, rows, hidden)
    src = torch.zeros((padded, hidden), dtype=torch.bfloat16, device=device)
    src[:rows] = s
    full = torch.empty((tp * padded, hidden), dtype=torch.bfloat16, device=device)
    fly.all_gather(src, full)
    torch.cuda.synchronize()
    return full.view(tp, padded, hidden)[:, :rows].reshape(tp * rows, hidden)


def _nmse_or_inf(got: torch.Tensor, ref: torch.Tensor) -> float:
    """The NMSE, or infinity for a wrong shape, which `_failures` reports."""
    return _nmse(got, ref) if got.shape == ref.shape else float("inf")


def _close(ref: torch.Tensor, got: torch.Tensor, msg: str) -> float:
    """The mismatch fraction of ``checkAllclose``, or 1 for a wrong shape, which
    `_failures` reports."""
    return _close_err(ref, got, msg) if got.shape == ref.shape else 1.0


def _measure(router, fly, rank: int, tokens: int, hidden: int, device) -> dict:
    tp, rows = fly.world_size, tokens // fly.world_size
    row = {"tokens": tokens, "hidden": hidden}
    padded = _padded(fly, rows, hidden)

    x = _make_inp(tokens, hidden, rank=rank, salt=1, fill="normal", device=device)
    x0 = x.clone()
    _poison(tp * padded * hidden, device)
    out = router.reduce_scatter(x)
    torch.cuda.synchronize()
    row["rs_served"] = out is not None
    if out is not None:
        ref = x.float().clone()
        torch.distributed.all_reduce(ref)
        row["rs_shape"] = tuple(out.shape)
        want = ref[rank * rows : (rank + 1) * rows]
        row["rs_nmse"] = _nmse_or_inf(out.float(), want)
        row["rs_err"] = _close(want, out, "rs")
        row["rs_engine_equal"] = torch.equal(
            _bits(out), _bits(_engine_rs(fly, x, rows, hidden, device))
        )
        row["rs_input_unchanged"] = torch.equal(_bits(x), _bits(x0))

    s = _make_inp(rows, hidden, rank=rank, salt=2, fill="normal", device=device)
    s0 = s.clone()
    _poison(padded * hidden, device)
    full = router.all_gather(s)
    torch.cuda.synchronize()
    row["ag_served"] = full is not None
    if full is not None:
        ref_ag = torch.cat([p.float() for p in _gather(s, tp)])
        row["ag_shape"] = tuple(full.shape)
        row["ag_nmse"] = _nmse_or_inf(full.float(), ref_ag)
        row["ag_err"] = _close(ref_ag, full, "ag")
        row["ag_engine_equal"] = torch.equal(
            _bits(full), _bits(_engine_ag(fly, s, device))
        )
        row["ag_ranks_equal"] = all(
            torch.equal(_bits(full), _bits(o)) for o in _gather(full, tp)
        )
        row["ag_input_unchanged"] = torch.equal(_bits(s), _bits(s0))
    return row


def _not_served(router, fly, rank: int, shape, device) -> dict:
    tokens, hidden = shape
    x = _make_inp(tokens, hidden, rank=rank, salt=1, fill="normal", device=device)
    s = _make_inp(
        tokens // fly.world_size,
        hidden,
        rank=rank,
        salt=2,
        fill="normal",
        device=device,
    )
    return {
        "rs_none": router.reduce_scatter(x) is None,
        "ag_none": router.all_gather(s) is None,
    }


def _graph_replay(router, rank: int, tp: int, device) -> dict:
    tokens, hidden = _GRAPH_CASE
    rows = tokens // tp
    x = _make_inp(tokens, hidden, rank=rank, salt=9, fill="normal", device=device)
    s = _make_inp(rows, hidden, rank=rank, salt=8, fill="normal", device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        rs_out = router.reduce_scatter(x)
        ag_out = router.all_gather(s)
    res = {"served": rs_out is not None and ag_out is not None, "mismatches": 0}
    for k in range(_GRAPH_REPLAYS):
        x.copy_(
            _make_inp(
                tokens, hidden, rank=rank, salt=10 + k, fill="normal", device=device
            )
        )
        s.copy_(
            _make_inp(
                rows, hidden, rank=rank, salt=20 + k, fill="normal", device=device
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        rs_eager, ag_eager = router.reduce_scatter(x), router.all_gather(s)
        torch.cuda.synchronize()
        res["mismatches"] += int(not torch.equal(_bits(rs_out), _bits(rs_eager)))
        res["mismatches"] += int(not torch.equal(_bits(ag_out), _bits(ag_eager)))
    return res


def _make_engine(rank: int, tp: int, init_method: str):
    """This rank's device and its ``FlyQuickAllReduce``."""
    from aiter.ops.flydsl import FlyQuickAllReduce

    device, gloo = _init_rank(rank, tp, init_method)
    # The single-phase kernels run on the uncached inbox only; "auto" picks the
    # fine-grained one at TP4 and TP8 on a host whose GPUs are linked by PCIe.
    fly = FlyQuickAllReduce(
        group=gloo, device=device, rank=rank, world_size=tp, inbox_memory="uncached"
    )
    return device, fly


def _make_router(fly, factory: bool, device):
    """The router under test: from the factory (TP2), or over an engine that has
    passed ``compile_sp`` and a window that admits every size (TP4)."""
    from aiter.dist.device_communicators import sp_collectives as sp
    from aiter.ops.flydsl.allreduce_policy import Window

    if factory:
        return sp.create_sp_collectives(fly)
    elements = sp.TILE_BYTES // 2
    fly.compile_sp(
        torch.zeros(elements, dtype=torch.bfloat16, device=device),
        torch.zeros(elements // fly.world_size, dtype=torch.bfloat16, device=device),
    )
    return sp.SequenceParallelCollectives(fly, Window(0, None))


def _run_rank(rank: int, tp: int, init_method: str, factory: bool) -> dict:
    import torch.distributed as dist

    from aiter.dist.device_communicators import sp_collectives as sp
    from aiter.ops.flydsl.allreduce_policy import detect_arch

    device, fly = _make_engine(rank, tp, init_method)
    res = {"factory_none": None, "rows": [], "below": None, "graph": None}
    try:
        res["router_expected"] = (detect_arch(), fly.link, tp) == _ROUTER_PLATFORM
        res["tile_bytes"] = (sp.TILE_BYTES, fly._sp_engine().tile_bytes)
        router = _make_router(fly, factory, device)
        if factory:
            res["factory_none"] = router is None
        if router is not None:
            cases = _TP2_CASES if factory else _TP4_CASES
            res["rows"] = [_measure(router, fly, rank, t, h, device) for t, h in cases]
            if factory:
                res["below"] = {
                    f"{shape} {op}": none
                    for shape in _TP2_BELOW_WINDOW
                    for op, none in _not_served(
                        router, fly, rank, shape, device
                    ).items()
                }
                res["graph"] = _graph_replay(router, rank, tp, device)
    finally:
        fly.close()
        dist.destroy_process_group()
    return res


def _failures(ranks, *, tp: int, factory: bool) -> list[str]:
    fails = []
    for rank, res in enumerate(ranks):
        where = f"tp={tp} rank {rank}"
        if res["tile_bytes"][0] != res["tile_bytes"][1]:
            fails.append(f"{where}: router tile {res['tile_bytes']} != engine tile")
        if factory and res["factory_none"] == res["router_expected"]:
            fails.append(
                f"{where}: the factory returned "
                f"{'None' if res['factory_none'] else 'a router'}, expected "
                f"{'a router' if res['router_expected'] else 'None'} on this "
                f"platform (a router exactly on {_ROUTER_PLATFORM})"
            )
        for row in res["rows"]:
            at = f"{where} tokens={row['tokens']} hidden={row['hidden']}"
            rows = row["tokens"] // tp
            for mode, shape in (
                ("rs", (rows, row["hidden"])),
                ("ag", (row["tokens"], row["hidden"])),
            ):
                if not row[f"{mode}_served"]:
                    fails.append(f"{at}: {mode} was not served")
                    continue
                if row[f"{mode}_shape"] != shape:
                    fails.append(
                        f"{at}: {mode} shape {row[f'{mode}_shape']} != {shape}"
                    )
                # written so that a NaN fails
                if not row[f"{mode}_nmse"] <= NMSE_MAX:
                    fails.append(
                        f"{at}: {mode} NMSE {row[f'{mode}_nmse']:.3e} > {NMSE_MAX}"
                    )
                if row[f"{mode}_err"] >= CLOSE_ERR_RATIO:
                    fails.append(
                        f"{at}: {mode} checkAllclose err {row[f'{mode}_err']:.3f} "
                        f">= {CLOSE_ERR_RATIO}"
                    )
                if not row[f"{mode}_engine_equal"]:
                    fails.append(
                        f"{at}: {mode} differs from the engine on a padded input"
                    )
                if not row[f"{mode}_input_unchanged"]:
                    fails.append(f"{at}: {mode} modified its input")
            if row["ag_served"] and not row["ag_ranks_equal"]:
                fails.append(f"{at}: AG output differs between ranks")
        if res["below"] is not None and not all(res["below"].values()):
            fails.append(
                f"{where}: a payload below the window was served: {res['below']}"
            )
        if res["graph"] is not None and (
            not res["graph"]["served"] or res["graph"]["mismatches"]
        ):
            fails.append(f"{where}: graph replay {res['graph']}")
    return fails


def _bench(router, fly, rank: int, tokens: int, hidden: int, device) -> dict:
    """The router, the engine on an input padded here and the exact collective
    (RCCL) on one shape: per candidate the time, the bytes of the full tensor
    per second and the ``checkAllclose`` mismatch fraction against the exact
    fp32 result. The engine's input is padded, and its output left padded,
    outside the timed call, so the router's time includes the padding and
    unpadding copies that the engine's does not (a shape that needs no padding
    has none)."""
    import torch.distributed as dist

    tp, rows = fly.world_size, tokens // fly.world_size
    padded = _padded(fly, rows, hidden)
    x = _make_inp(tokens, hidden, rank=rank, salt=1, fill="normal", device=device)
    s = _make_inp(rows, hidden, rank=rank, salt=2, fill="normal", device=device)
    ref = x.float().clone()
    dist.all_reduce(ref)
    ref_rs = ref[rank * rows : (rank + 1) * rows]
    ref_ag = torch.cat([p.float() for p in _gather(s, tp)])

    rs_src = torch.zeros((tp, padded, hidden), dtype=torch.bfloat16, device=device)
    rs_src[:, :rows] = x.view(tp, rows, hidden)
    rs_src = rs_src.view(tp * padded, hidden)
    rs_padded = torch.empty((padded, hidden), dtype=torch.bfloat16, device=device)
    ag_src = torch.zeros((padded, hidden), dtype=torch.bfloat16, device=device)
    ag_src[:rows] = s
    ag_padded = torch.empty((tp * padded, hidden), dtype=torch.bfloat16, device=device)
    rs_exact, ag_exact = torch.empty_like(s), torch.empty_like(x)

    def engine_rs():
        fly.reduce_scatter(rs_src, rs_padded)
        return rs_padded

    def engine_ag():
        fly.all_gather(ag_src, ag_padded)
        return ag_padded

    def exact_rs():
        dist.reduce_scatter_tensor(rs_exact, x)
        return rs_exact

    def exact_ag():
        dist.all_gather_into_tensor(ag_exact, s)
        return ag_exact

    def unpad_ag(t):
        return t.view(tp, padded, hidden)[:, :rows].reshape(tokens, hidden)

    # name -> (call, the part of its output to compare, the exact result)
    candidates = {
        "rs router": (lambda: router.reduce_scatter(x), lambda t: t, ref_rs),
        "rs engine": (engine_rs, lambda t: t[:rows], ref_rs),
        "rs rccl": (exact_rs, lambda t: t, ref_rs),
        "ag router": (lambda: router.all_gather(s), lambda t: t, ref_ag),
        "ag engine": (engine_ag, unpad_ag, ref_ag),
        "ag rccl": (exact_ag, lambda t: t, ref_ag),
    }
    nbytes = tokens * hidden * 2
    # (tp - 1) adds per element of the reduce-scatter sum, none in the all-gather;
    # codec ALU work is not counted. TB/s counts the full tensor.
    flops = tokens * hidden * (tp - 1)
    row = {}
    for name, (call, part, want) in candidates.items():
        dist.barrier()
        torch.cuda.synchronize()
        # use_cuda_event is mandatory: run_perftest's default timer collects no
        # device rows inside a spawn worker.
        out, us = run_perftest(call, use_cuda_event=True)
        row[f"{name} us"] = float(us)
        if name.startswith("rs"):
            row[f"{name} TFLOPS"] = flops / us / 1e6
        row[f"{name} TB/s"] = nbytes / us / 1e6
        row[f"{name} err"] = _close(want, part(out), name)
    return row


def _bench_rank(
    rank: int, tp: int, init_method: str, tokens: int, hidden: int, factory: bool
) -> dict:
    import torch.distributed as dist

    device, fly = _make_engine(rank, tp, init_method)
    try:
        router = _make_router(fly, factory, device)
        return _bench(router, fly, rank, tokens, hidden, device)
    finally:
        fly.close()
        dist.destroy_process_group()


@benchmark()
def test_sp_router(tokens, hidden, tp):
    ranks = _run_pool(tp, _bench_rank, tokens=tokens, hidden=hidden, factory=tp == 2)
    # the slowest rank's time, the worst rank's mismatch fraction
    ret = {"gfx": get_gfx_runtime()}
    for column in ranks[0]:
        ret[column] = max(r[column] for r in ranks)
    bad = {c: v for c, v in ret.items() if c.endswith(" err") and v >= CLOSE_ERR_RATIO}
    assert (
        not bad
    ), f"tp={tp} {tokens}x{hidden}: checkAllclose err >= {CLOSE_ERR_RATIO}: {bad}"
    return ret


test_sp_router.__test__ = False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tp",
        type=int,
        nargs="*",
        default=None,
        help="World sizes to run (2, 4). Default both, skipping sizes with fewer\n"
        "visible GPUs. A size that is named but cannot run is an error.",
    )
    args = parser.parse_args()
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_ARCHS:
        print(f"skipped: {arch} is not one of {SUPPORTED_ARCHS}")
        return
    tps = fab.select_world_sizes(
        args.tp or [2, 4],
        explicit=bool(args.tp),
        n_gpu=torch.cuda.device_count(),
        supported=(2, 4),
    )
    fails = []
    served = []  # the world sizes with a router here, which the sweep times
    for tp in tps:
        factory = tp == 2
        ranks = _run_pool(tp, _run_rank, factory=factory)
        fails += _failures(ranks, tp=tp, factory=factory)
        if not ranks[0]["factory_none"]:
            served.append(tp)
        print(
            f"tp={tp}: {len(ranks[0]['rows'])} shapes checked on {tp} ranks"
            f"{', the factory returned None, as expected here' if ranks[0]['factory_none'] and not ranks[0]['router_expected'] else ''}"
        )
    if fails:
        raise AssertionError("; ".join(fails))
    df = [
        test_sp_router(tokens, hidden, tp)
        for tp in served
        for tokens, hidden in (_TP2_CASES if tp == 2 else _TP4_CASES)
    ]
    if df:
        aiter.logger.info(
            "sequence-parallel router summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )
    print("all checks passed")


if __name__ == "__main__":
    freeze_support()
    main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bandwidth roofline: FlyDSL quick all-reduce (mesh, ring) and RCCL.

    HIP_VISIBLE_DEVICES=4,5,6,7 python3 \\
        op_tests/op_benchmarks/collective_bw/bench_bw_quick_allreduce.py -tp 4

The setup (process group, engines, inputs) and the timing are the ones
``op_tests/multigpu_tests/test_flydsl_quick_allreduce.py`` uses in its bench
scenario: a CUDA graph of back-to-back calls, rotated rounds, median, slowest
rank (``flydsl_allreduce_bench.capture`` / ``time_graphs``). RCCL runs through
the same timer. Each engine reports its own traffic (``traffic(nbytes)``),
which TransferBench replays as the roof once the ranks have exited.

Accuracy is not checked here. ``test_flydsl_quick_allreduce.py`` does that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for _path in (str(_HERE), str(_REPO), str(_REPO / "op_tests" / "multigpu_tests")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import report

DEFAULT_TOKENS = (64, 256, 1024, 4096, 8192)
HIDDEN = 7168
ALGORITHMS = ("mesh", "ring")


def _worker(
    *, rank, world_size, init_method, algorithms, cases, reps, rounds, rccl_backend
):
    import flydsl_allreduce_bench as fab
    import test_flydsl_quick_allreduce as qt
    import torch
    from measure import FnTimer, gather, measure
    from rccl import RcclBaseline

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    tp_group = qt._init_dist(rank, world_size, init_method, hip=False)
    gloo = tp_group.cpu_group
    # Built in a fixed order: each engine's IPC exchange is a collective.
    engines = {
        f"fly_{a}": qt._engine(rank, world_size, gloo, device, {"algorithm": a})
        for a in algorithms
    }
    if rccl_backend == "torch":
        rccl = RcclBaseline(device=device, group=tp_group.device_group)
    else:
        rccl = RcclBaseline(
            device=device, pynccl=tp_group.device_communicator.pynccl_comm
        )

    def _graph_time(thunks):
        graphs = {name: fab.capture(thunk) for name, thunk in thunks.items()}
        return fab.time_graphs(graphs, gloo, reps=reps, rounds=rounds)

    timer = FnTimer("graph", _graph_time)

    # Fixed addresses for the whole spawn; see flydsl_allreduce_bench.
    max_bytes = max(m * hidden * 2 for m, hidden in cases)
    bufs = {
        name: torch.empty(max_bytes, dtype=torch.uint8, device=device)
        for name in ("inp", *engines)
    }

    def _view(name, m, hidden):
        return bufs[name][: m * hidden * 2].view(torch.bfloat16).view(m, hidden)

    records = []
    try:
        for m, hidden in cases:
            nbytes = m * hidden * 2
            x = _view("inp", m, hidden)
            x.copy_(qt._make_inp(m, hidden, "normal", rank=rank, device=device))
            outs = {name: _view(name, m, hidden) for name in engines}
            ops = {
                name: (
                    lambda e=eng, x=x, o=outs[name]: e.allreduce(x, o),
                    eng.traffic(nbytes),
                )
                for name, eng in engines.items()
            }
            variants = {name: eng.variant(nbytes) for name, eng in engines.items()}
            first_traffic = next(iter(ops.values()))[1]
            records += measure(
                f"{m}x{hidden}",
                ops,
                timer,
                rccl=rccl.thunk(first_traffic),
                variants=variants,
            )
        merged = gather(records, gloo)
    finally:
        qt._teardown(engines.values(), gloo)
    return merged


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-tp", "--world-size", type=int, default=4, choices=(2, 4, 8))
    p.add_argument(
        "-m",
        "--tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKENS),
        help=f"token counts; each case is (tokens, {HIDDEN}) bf16",
    )
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    p.add_argument(
        "--reps", type=int, default=None, help="default: flydsl_allreduce_bench's"
    )
    p.add_argument(
        "--rounds", type=int, default=None, help="default: flydsl_allreduce_bench's"
    )
    p.add_argument(
        "--rccl-backend",
        choices=("torch", "pynccl"),
        default="torch",
        help="torch.distributed, or aiter's PyNccl where torch collectives do not capture",
    )
    report.add_args(p)
    args = p.parse_args(argv)

    import flydsl_allreduce_bench as fab
    import torch
    from measure import spawn

    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    if fab.gpu_count() < args.world_size:
        print(f"SKIP: tp={args.world_size} needs {args.world_size} GPUs")
        return 0
    cases = [(m, args.hidden) for m in args.tokens]
    reps = args.reps or fab.DEFAULT_REPS
    rounds = args.rounds or fab.DEFAULT_ROUNDS
    ranks = spawn(
        _worker,
        args.world_size,
        init_method=get_distributed_init_method(get_ip(), get_open_port()),
        algorithms=tuple(args.algorithms),
        cases=cases,
        reps=reps,
        rounds=rounds,
        rccl_backend=args.rccl_backend,
    )
    header = [
        f"- device: {torch.cuda.get_device_name(0)}",
        (
            f"- timing: CUDA graph of {fab.GRAPH_INNER} calls, {reps} timed replays "
            f"per sample, median of {rounds} rotated rounds, slowest rank"
        ),
        f"- RCCL backend: {args.rccl_backend}",
    ]
    report.finish(
        ranks[0],
        title=f"FlyDSL quick all-reduce vs TransferBench roof, TP{args.world_size}",
        args=args,
        header=header,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

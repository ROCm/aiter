# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bandwidth roofline: fused attention A2A (Q/K/V trio) and RCCL.

    HIP_VISIBLE_DEVICES=4,5,6,7 python3 \\
        op_tests/op_benchmarks/collective_bw/bench_bw_attention_a2a.py -w 4

Shapes, recipes, inputs, the inbox-memory policy and the timing are those of
``op_tests/multigpu_tests/bench_attention_a2a.py``. Its ``_time_roles`` gives
the eager median of the whole trio, cycling ``_ROTATIONS`` input sets. RCCL
(three ``all_to_all_single`` of the bf16 role tensors) runs through the same
timer with as many rotations. The op reports its own traffic
(``AttentionA2AIntraNodeOp.traffic()``), which TransferBench replays as the
roof once the ranks have exited.

The RCCL baseline is the bare collective. ``bench_attention_a2a.py`` also
times the yunchang layout transposes and the MHA V4 packers, and it checks
the fused output bit for bit. Neither is repeated here.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_COLLECTIVE_BW = _HERE.parent / "collective_bw"
for _path in (
    str(_HERE),
    str(_COLLECTIVE_BW),
    str(_REPO),
    str(_REPO / "op_tests" / "multigpu_tests"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import bench_attention_a2a as a2a
import report


def _no_role(role, index, start, end):
    """``_time_roles`` also times single roles; only the trio is graded."""
    start.record()
    end.record()


def _worker(
    *, rank, world_size, init_method, shapes, recipes, warmup, iterations, inbox
):
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "12G")
    import torch
    import torch.distributed as dist
    from measure import FnTimer, gather, measure
    from rccl import RcclBaseline

    from aiter.ops.flydsl.attention_a2a_intranode import AttentionA2AIntraNodeOp
    from aiter.ops.mha_v4 import AttentionPack

    ms = None
    if inbox == "uncached":
        import mori.shmem as ms
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    merged = None
    try:
        cpu_group = dist.new_group(backend="gloo")
        if ms is not None:
            torch._C._distributed_c10d._register_process_group("mori", cpu_group)
            ms.shmem_torch_process_group_init("mori")

        def _eager_trio(thunks):
            return {
                name: a2a._time_roles(
                    lambda index, t=thunk: t(), _no_role, warmup, iterations, group
                )["trio"]
                for name, thunk in thunks.items()
            }

        timer = FnTimer("eager", _eager_trio)
        rccl = RcclBaseline(device=device, group=group, rotations=a2a._ROTATIONS)
        records = []
        for shape_name in shapes:
            requested = a2a._SHAPES[shape_name]
            shape = a2a._local_shape(requested, world_size)
            inputs = [
                tuple(
                    a2a._make_input(rank, shape, rotation * 3 + role, device)
                    for role in range(3)
                )
                for rotation in range(a2a._ROTATIONS)
            ]
            # Padded tokens carry zeros, as a real caller would send.
            valid = max(0, min(shape[1], requested[1] - rank * shape[1]))
            for values in inputs:
                for value in values:
                    value[:, valid:].zero_()
            for recipe in recipes:
                packed = recipe.packers is not None
                op = AttentionA2AIntraNodeOp(
                    rank=rank,
                    world_size=world_size,
                    shape=shape,
                    quant=recipe.quant,
                    return_packed=packed,
                    v_pack=(
                        AttentionPack.V_FOR_FP6_P
                        if recipe.v_fp6_p
                        else AttentionPack.DEFAULT
                    ),
                    softmax_scale=shape[-1] ** -0.5 if packed else None,
                    hadamard=recipe.hadamard,
                    group=cpu_group,
                    inbox_memory=inbox,
                )
                counter = itertools.count()

                def trio(op=op, counter=counter, inputs=inputs):
                    values = inputs[next(counter) % len(inputs)]
                    for role in range(3):
                        op.submit_role(role, values[role])

                traffic = op.traffic()
                records += measure(
                    f"{shape_name} {recipe.name}",
                    {"fused": (trio, traffic)},
                    timer,
                    rccl=rccl.thunk(traffic),
                )
                torch.cuda.synchronize()
                op.close()
            del inputs
        merged = gather(records, cpu_group)
    finally:
        # Skip finalize while an exception unwinds: a peer may be mid-collective.
        if sys.exc_info()[0] is None:
            try:
                if ms is not None:
                    ms.shmem_finalize()
            finally:
                dist.destroy_process_group()
    return merged


def main(argv=None) -> int:
    names = [r.name for r in a2a._RECIPES]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-w", "--world-size", type=int, default=4, choices=(2, 4, 8))
    p.add_argument(
        "--shapes", choices=tuple(a2a._SHAPES), nargs="+", default=list(a2a._SHAPES)
    )
    p.add_argument("--recipes", choices=["all", *names], nargs="+", default=["all"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=15)
    p.add_argument(
        "--inbox-memory",
        choices=a2a.INBOX_MEMORY_MODES,
        default="auto",
        help="memory of the peer-written buffers (auto: the op's own policy)",
    )
    report.add_args(p)
    args = p.parse_args(argv)
    selected = names if "all" in args.recipes else args.recipes
    recipes = [r for r in a2a._RECIPES if r.name in selected]

    import torch
    from measure import spawn

    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    gfx = a2a.get_gfx_runtime()
    if gfx != "gfx950":
        print(f"SKIP: attention A2A benchmark requires gfx950, found {gfx}")
        return 0
    if torch.cuda.device_count() < args.world_size:
        print(f"SKIP: world={args.world_size} needs {args.world_size} GPUs")
        return 0
    link, inbox = a2a._resolve_inbox(args.inbox_memory, args.world_size)
    ranks = spawn(
        _worker,
        args.world_size,
        init_method=get_distributed_init_method(get_ip(), get_open_port()),
        shapes=tuple(args.shapes),
        recipes=recipes,
        warmup=args.warmup,
        iterations=args.iterations,
        inbox=inbox,
    )
    header = [
        f"- device: {torch.cuda.get_device_name(0)} ({gfx}), link {link}, inbox {inbox}",
        (
            f"- timing: eager trio, median of {args.iterations} iterations after "
            f"{args.warmup} warmup, {a2a._ROTATIONS} rotated input sets, slowest rank"
        ),
    ]
    report.finish(
        ranks[0],
        title=f"Attention A2A trio vs TransferBench roof, world {args.world_size}",
        args=args,
        header=header,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

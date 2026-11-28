# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx942's MoE time in sglang's steps: DeepSeek-V4-Flash, EP8 on 8 x MI300X.

sglang (``--moe-a2a-backend megamoe``, 8 DP attention ranks, EAGLE with 4 draft tokens)
calls the layer once per MoE layer (43) of every step, and its MTP layer's MoE three
times per decode step (draft extend on 4 rows per request, two draft steps on one) and
once per extend step. When every rank decodes (or idles), the step runs small_M at the
decode capacity (``forward_small_m``), otherwise large_M (``forward``) at
max_tok_per_rank 8192; a rank without tokens brings one padding row (id == experts).
This bench replays those calls:

- ``--trace DIR``: the steps of a recorded sglang run, with each rank's token count and
  real top-6 ids per layer (the routing skew included). The MTP layer's ids are not
  recorded: its calls take layer 42's ids of the same rows.
- otherwise ``--cases``: synthetic steps with uniform routing.

Each call is timed alone, as in sglang, where attention runs between two MoE layers:
L2 and MALL flushed by a 512 MiB read, the experts one of ``--copies`` copies (rotated;
a layer's experts are 384 MiB per rank, more than the MALL), the ranks aligned by a GPU
spin and an all-reduce, CUDA events around the call, the slowest rank's time (the
median of ``--rounds``). The configurations run each call back to back, in alternating
order:

- sglang: decode steps on small_M, extend steps on large_M (what sglang runs)
- main128+large: decode steps on the main path at the decode capacity (sglang before
  the small kernel; what a8w8's small_M runs), extend steps on large_M
- small+main: decode steps on small_M, extend steps on the main path
- main: every step on the main path (neither small_M nor large_M)
- large: every step on large_M (sglang without a decode capacity)
- sglang#2: sglang again, the noise of the measurement

To record a trace, serve with the recorder and an admin key (the recorder copies every
step's ids to the host, which slows the server down; with EAGLE, its per-token gatherer
must skip the MTP layer, which has no layer index):

    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/tmp/edr sglang serve ... \\
        --moe-a2a-backend megamoe --expert-distribution-recorder-mode per_token \\
        --admin-api-key KEY
    curl -X POST -H "Authorization: Bearer KEY" $URL/start_expert_distribution_record
    (the workload)
    curl -X POST -H "Authorization: Bearer KEY" $URL/stop_expert_distribution_record
    curl -X POST -H "Authorization: Bearer KEY" $URL/dump_expert_distribution_record

then

    python3 op_tests/multigpu_tests/bench_mega_moe_gfx942.py --trace /tmp/edr
"""

import argparse
import gc
import glob
import logging
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_mega_moe_gfx942 import (
    SWIGLU_LIMIT,
    TOPK,
    WORLD,
    E,
    H,
    I,
    _free_port,
    _make_weights,
    _skip_reason,
)

import aiter
from aiter import dtypes

LAYERS = 43
MTPR = 8192  # sglang's max_tok_per_rank (chunked prefill 16384 fits on every rank)
DRAFT_TOKENS = 4  # --speculative-num-draft-tokens: rows per request of a verify step
EXTEND, MIXED = 1, 3  # sglang ForwardMode values of steps with extending ranks
# (path of decode steps, path of extend steps)
CONFIGS = {
    "sglang": ("small", "large"),
    "main128+large": ("main128", "large"),
    "small+main": ("small", "main"),
    "main": ("main", "main"),
    "large": ("large", "large"),
    "sglang#2": ("small", "large"),
}
SPIN = 2_000_000  # GPU cycles each rank spins before the aligning all-reduce


def _trace_steps(path, rank):
    """This rank's recorded steps: {forward_pass_id: (mode, ids [layers, T, TOPK])}."""
    files = glob.glob(os.path.join(path, f"expert_distribution_recorder_*_{rank}.pt"))
    if len(files) != 1:
        raise SystemExit(f"{path}: expected one dump of rank {rank}, found {files}")
    records = torch.load(files[0], weights_only=True)["records"]
    return {
        r["forward_pass_id"]: (r["forward_mode"], r["topk_ids_of_layer"][:, :, :TOPK])
        for r in records
    }


def _steps(args, rank):
    """The steps in the same order on every rank: (kind, tokens of each rank, this
    rank's ids of a layer)."""
    if args.trace:
        local = _trace_steps(args.trace, rank)
        meta = [None] * WORLD
        dist.all_gather_object(
            meta, {p: (m, ids.shape[1]) for p, (m, ids) in local.items()}
        )
        steps = []
        for p in sorted(set.intersection(*(set(m) for m in meta))):
            tokens = [m[p][1] for m in meta]
            if not any(tokens):
                continue
            kind = (
                "extend" if any(m[p][0] in (EXTEND, MIXED) for m in meta) else "decode"
            )
            steps.append((kind, tokens, lambda layer, ids=local[p][1]: ids[layer]))
    else:
        steps = []
        for c, case in enumerate(args.cases):
            kind, counts = case.split(":")
            counts = [int(n) for n in counts.split(",")]
            tokens = counts * WORLD if len(counts) == 1 else counts
            if kind not in ("decode", "extend") or len(tokens) != WORLD:
                raise SystemExit(f"--cases {case}: decode|extend:N or 8 counts")

            def ids(layer, n=tokens[rank], c=c):
                gen = torch.Generator(device="cuda").manual_seed(
                    (c * LAYERS + layer) * WORLD + rank
                )
                scores = torch.randn((n, E), device="cuda", generator=gen)
                return scores.topk(TOPK, dim=-1).indices

            steps.append((kind, tokens, ids))
    if args.steps and args.steps < len(steps):
        stride = len(steps) / args.steps
        steps = [steps[int(i * stride)] for i in range(args.steps)]
    return steps


def _calls(steps, mtp):
    """(step, kind, part, layer, rows of each rank, row stride) of every call."""
    calls = []
    for s, (kind, tokens, _) in enumerate(steps):
        for layer in range(LAYERS):
            calls.append((s, kind, "target", layer, tokens, 1))
        if not mtp:
            continue
        calls.append((s, kind, "mtp", LAYERS - 1, tokens, 1))
        if kind == "decode":
            rows = [-(-n // DRAFT_TOKENS) for n in tokens]
            calls += [(s, kind, "mtp", LAYERS - 1, rows, DRAFT_TOKENS)] * 2
    return calls


def _worker(rank, port, args):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=WORLD, device_id=device
    )
    import mori.shmem as ms

    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    if rank:
        aiter.logger.setLevel(logging.WARNING)
    try:
        _bench(rank, args)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        ms.shmem_finalize()
        dist.destroy_process_group()


def _bench(rank, args):
    from aiter.ops.flydsl.kernels.mega_moe_gfx942 import MegaMoEGfx942

    steps = _steps(args, rank)
    calls = _calls(steps, args.mtp)
    weights, _ = _make_weights(rank, args.quant)
    weights = MegaMoEGfx942.prepare_weights(
        *weights, w1_gate_up_interleaved=False, quant=args.quant
    )
    copies = [weights] + [
        tuple(t.clone() for t in weights) for _ in range(args.copies - 1)
    ]

    def layer(mtpr, small, large):
        return MegaMoEGfx942(
            rank=rank,
            world_size=WORLD,
            model_dim=H,
            inter_dim=I,
            experts=E,
            topk=TOPK,
            w1=weights[0],
            w1_scale=weights[1],
            w2=weights[2],
            w2_scale=weights[3],
            max_tok_per_rank=mtpr,
            quant=args.quant,
            swiglu_limit=SWIGLU_LIMIT,
            small_m_mtprs=(args.decode_mtpr,),
            small_m_enabled=small,
            large_m_enabled=large,
        )

    # path: (instance, call). As sglang builds them: small_M and large_M on one
    # instance; the main path on another.
    paths = {p for c in args.configs for p in CONFIGS[c]}
    sl = layer(MTPR, True, True) if paths & {"small", "large"} else None
    small = (
        sl,
        lambda m, x, w, i: m.forward_small_m(x, w, i, max_tokens=args.decode_mtpr),
    )
    run_path = {"small": small, "large": (sl, lambda m, x, w, i: m.forward(x, w, i))}
    for name, mtpr in (("main", MTPR), ("main128", args.decode_mtpr)):
        if name in paths:
            run_path[name] = (
                layer(mtpr, False, False),
                lambda m, x, w, i: m.forward(x, w, i),
            )
    gen = torch.Generator(device="cuda").manual_seed(rank)
    x_all = torch.randn((MTPR, H), dtype=dtypes.bf16, device="cuda", generator=gen)
    w_all = torch.randn((MTPR, TOPK), device="cuda", generator=gen).softmax(dim=-1)
    pad = (
        torch.zeros((1, H), dtype=dtypes.bf16, device="cuda"),
        torch.zeros((1, TOPK), device="cuda"),
        torch.full((1, TOPK), E, dtype=torch.int32, device="cuda"),
    )

    def inputs(call):
        s, _, _, lay, rows, stride = call
        n = rows[rank]
        if not n:
            return pad
        ids = steps[s][2](lay)[::stride][:n]
        return x_all[:n], w_all[:n], ids.to(device="cuda", dtype=torch.int32)

    def run(cfg, kind, copy, x, w, ids):
        mega, call = run_path[CONFIGS[cfg][kind == "extend"]]
        mega.bind_weights(*copies[copy])
        return call(mega, x, w, ids)

    for _, kind, _, _, rows, _ in calls:
        if kind == "decode" and max(rows) > args.decode_mtpr:
            raise SystemExit(f"a decode step has {max(rows)} rows > --decode-mtpr")

    flush = torch.ones(1 << 29, dtype=torch.uint8, device="cuda")
    bar = torch.zeros(1, device="cuda")
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    def timed(cfg, kind, copy, x, w, ids):
        flush.max()
        dist.barrier()
        torch.cuda._sleep(SPIN)
        dist.all_reduce(bar)
        ev0.record()
        run(cfg, kind, copy, x, w, ids)
        ev1.record()
        torch.cuda.synchronize()
        return ev0.elapsed_time(ev1) * 1e3

    # Compile and warm every path once per kind of step.
    for kind in ("decode", "extend"):
        first = next((c for c in calls if c[1] == kind), None)
        for cfg in args.configs if first else ():
            for copy in range(min(2, args.copies)):
                timed(cfg, kind, copy, *inputs(first))

    times = torch.zeros((len(calls), len(args.configs)), device="cuda")
    rounds = torch.zeros((args.rounds, len(args.configs)), device="cuda")
    order = list(enumerate(args.configs))
    gc.collect()
    gc.disable()
    for i, call in enumerate(calls):
        x, w, ids = inputs(call)
        for r in range(args.rounds):
            for j, cfg in order if (i + r) % 2 == 0 else reversed(order):
                rounds[r, j] = timed(cfg, call[1], (i + r) % args.copies, x, w, ids)
        # The slowest rank of each round, then the median round: a host stall on any
        # rank (all of them wait for it) spoils one round, not the call.
        dist.all_reduce(rounds, op=dist.ReduceOp.MAX)
        times[i] = rounds.median(dim=0).values
    gc.enable()
    if rank == 0:
        _report(args, steps, calls, times.cpu())


def _bucket(kind, rows):
    t = max(rows)
    return t if kind == "decode" else min(MTPR, 1 << max(8, (t - 1).bit_length()))


def _report(args, steps, calls, times):
    cfgs = args.configs
    df = pd.DataFrame(
        [(kind, part, _bucket(kind, rows)) for _, kind, part, _, rows, _ in calls],
        columns=["step", "part", "rows/rank <="],
    )
    df[cfgs] = times.numpy()
    if args.csv:
        df.to_csv(args.csv, index=False)
    kinds = pd.Series([k for k, _, _ in steps]).value_counts()
    log = aiter.logger.info
    title = f"MegaMoEGfx942 {args.quant} in sglang's steps " + (
        f"(trace {args.trace})" if args.trace else "(synthetic)"
    )
    log(
        "%s: %d decode and %d extend steps, %d calls, slowest rank, cold",
        title,
        kinds.get("decode", 0),
        kinds.get("extend", 0),
        len(calls),
    )
    total = df.groupby(["step", "part"])[cfgs].sum() / 1e3
    total.loc[("all", ""), :] = total.sum()
    for cfg in cfgs[1:]:
        total[f"{cfg} vs {cfgs[0]}"] = (total[cfg] / total[cfgs[0]] - 1).map(
            "{:+.1%}".format
        )
    log("MoE ms over the steps (markdown):\n%s", total.to_markdown(floatfmt=".1f"))
    per_step = df.groupby("step")[cfgs].sum().div(kinds, axis=0) / 1e3
    log("MoE ms per step (markdown):\n%s", per_step.to_markdown(floatfmt=".2f"))
    target = df[df.part == "target"]
    calls_by = target.groupby(["step", "rows/rank <="])[cfgs]
    table = calls_by.mean().join(calls_by.size().rename("calls"))
    log("us per target call (markdown):\n%s", table.to_markdown(floatfmt=".0f"))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MegaMoEGfx942 in sglang's steps on 8 GPUs",
    )
    parser.add_argument(
        "-q", "--quant", choices=["a8w4", "a8w8"], default="a8w4", help="experts"
    )
    parser.add_argument(
        "--trace",
        help="directory of an sglang per-token expert recorder dump (one file per "
        "rank); without it, --cases",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[
            "decode:4",
            "decode:16",
            "decode:24",
            "extend:512",
            "extend:2048",
            "extend:2048,0,2048,2048,2048,0,2048,2048",
        ],
        help="synthetic steps: decode|extend:N (every rank) or 8 counts",
    )
    parser.add_argument(
        "--steps", type=int, default=0, help="replay this many steps, evenly spaced"
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        choices=list(CONFIGS),
        default=list(CONFIGS),
        help="the first one is the reference of the comparison",
    )
    parser.add_argument(
        "--decode-mtpr",
        type=int,
        default=128,
        help="small_M capacity (sglang's SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR)",
    )
    parser.add_argument(
        "--mtp",
        type=dtypes.str2bool,
        default=True,
        help="add the MTP layer's MoE calls (1) or not (0)",
    )
    parser.add_argument(
        "--copies", type=int, default=8, help="copies of the experts, rotated"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="time each call this many times and keep the median (a host stall on "
        "any rank makes a round ms long)",
    )
    parser.add_argument("--csv", help="save the time of every call to this file")
    args = parser.parse_args()
    reason = _skip_reason(args.quant)
    if reason:
        aiter.logger.warning("bench_mega_moe_gfx942 skipped: %s", reason)
        return
    mp.spawn(_worker, args=(_free_port(), args), nprocs=WORLD, join=True)


if __name__ == "__main__":
    main()

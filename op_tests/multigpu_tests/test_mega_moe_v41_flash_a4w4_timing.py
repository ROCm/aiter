# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
r"""DeepSeek-V4.1-Flash EP8 A4W4 MegaMoEV2 timing with the MXFP4 combine.

Times the whole call (quant, Stage1, Stage2 when launched, combine) as CUDA-graph
replays and reports the slowest rank's median per size:

  decode  (MTPR 512, fixed-slot, AITER_MEGA_FIXED_SLOT_MAX_MTPR=1023): 160 / 192 / 224
  prefill (MTPR 4096, compact):                                         1024 / 2048 / 4096

tokens per rank, every rank with the same count and uniform random routing from
a fixed seed, through the routed interface (routed=True, forward(max_tokens=...)).
--shape draft runs the DSpark draft MoE (16 experts per rank) instead.

--reference compares the slowest-rank times against earlier runs' JSON (--out)
and fails on a slowdown beyond --tolerance; --compare-outputs checks every
output bitwise against the ones an earlier run saved with --save-outputs.

  torchrun --nproc-per-node 8 test_mega_moe_v41_flash_a4w4_timing.py \
      --out run1.json --save-outputs outputs/
  torchrun --nproc-per-node 8 test_mega_moe_v41_flash_a4w4_timing.py \
      --reference run1.json run2.json run3.json --tolerance 0.03 \
      --compare-outputs outputs/
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# (experts, topk) of the target MoE and of the DSpark draft MoE (16 experts per rank at EP8).
SHAPES = {"target": (384, 6), "draft": (128, 3)}
PHASES = {"decode": (512, (160, 192, 224)), "prefill": (4096, (1024, 2048, 4096))}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--shape", choices=tuple(SHAPES), default="target")
    parser.add_argument("--phases", default="decode,prefill")
    parser.add_argument(
        "--sizes",
        default="",
        help="comma-separated tokens per rank for every phase (default: the phase's sizes)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", type=Path, nargs="+")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--save-outputs", type=Path)
    parser.add_argument("--compare-outputs", type=Path)
    args = parser.parse_args()
    args.phases = [p for p in args.phases.split(",") if p]
    for phase in args.phases:
        if phase not in PHASES:
            parser.error(f"unknown phase {phase!r}")
    args.sizes = [int(s) for s in args.sizes.split(",") if s]
    return args


ARGS = _parse_args()
# Read when mega_moe_config is imported: MTPR 512 dispatches fixed-slot.
os.environ["AITER_MEGA_FIXED_SLOT_MAX_MTPR"] = "1023"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
from test_mega_moe_v2 import (
    NETWORKS,
    _barrier,
    _cleanup,
    _quantize_weights,
    _setup_dist,
)

import aiter
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2


def _versions():
    tree = Path(aiter.__file__).resolve().parent

    def git(*cmd):
        try:
            return subprocess.check_output(
                ["git", "-C", str(tree), *cmd], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    def dist_version(*names):
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        return "not installed"

    return {
        "aiter_tree": str(tree.parent),
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "torch": torch.__version__,
        "flydsl": dist_version("flydsl"),
        "mori": dist_version("mori", "amd_mori"),
    }


def _make_inputs(tokens, model_dim, experts, topk, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    values, ids = torch.topk(scores, topk, dim=-1)
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def _forward(moe, x, weights, ids):
    return moe.forward(x, weights, ids, max_tokens=x.shape[0])


def _time_size(moe, x, weights, ids, args, device):
    """(eager output, per-rank median us of ``trials`` x ``replays`` graph replays)."""
    _barrier()
    eager = _forward(moe, x, weights, ids)[: x.shape[0]].clone()
    _barrier()
    state = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        state["out"] = _forward(moe, x, weights, ids)
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(state["out"][: x.shape[0]], eager):
        raise AssertionError(f"tokens={x.shape[0]}: graph replay differs from eager")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    trials = []
    for _ in range(args.trials):
        _barrier()
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        trials.append(start.elapsed_time(end) * 1000.0 / args.replays)
    del graph
    local = torch.tensor([statistics.median(trials)], device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return eager, [float(t.item()) for t in gathered]


def _check_outputs(path, eager, compare):
    if not compare:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(eager.cpu(), path)
        return True
    if not path.exists():
        raise FileNotFoundError(f"no saved output {path}")
    return torch.equal(eager.cpu(), torch.load(path))


def _reference(paths):
    """Per-key median slowest-rank time over the reference runs, and the runs' relative spread."""
    runs = [json.loads(p.read_text())["results"] for p in paths]
    ref, spread = {}, {}
    for key in runs[0]:
        values = [run[key]["max_us"] for run in runs if key in run]
        ref[key] = statistics.median(values)
        spread[key] = (max(values) - min(values)) / ref[key]
    return ref, spread


def main():
    args = ARGS
    rank, world, device = _setup_dist()
    if world != 8:
        raise SystemExit("the V4.1-Flash EP8 layout needs 8 ranks")
    experts, topk = SHAPES[args.shape]
    network = dict(NETWORKS["v41_flash"], experts=experts, topk=topk)
    packed = _quantize_weights(
        network["model_dim"],
        network["inter_dim"],
        experts // world,
        rank,
        args.seed,
        device,
    )
    w1, w1_scale, w2, w2_scale = packed[:4]
    del packed
    plan = [
        (phase, PHASES[phase][0], args.sizes or list(PHASES[phase][1]))
        for phase in args.phases
    ]
    max_tokens = max(max(sizes) for _, _, sizes in plan)
    x_all, weights_all, ids_all = _make_inputs(
        max_tokens, network["model_dim"], experts, topk, rank, args.seed, device
    )
    output_dir = args.compare_outputs or args.save_outputs
    results, mismatches = {}, []
    for phase, mtpr, sizes in plan:
        if max(sizes) > mtpr:
            raise SystemExit(f"{phase}: sizes {sizes} exceed MTPR {mtpr}")
        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a4w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=mtpr,
            stage2_p2p_quant="fp4_blockwise_1x32",
            routed=True,
            **network,
        )
        for tokens in sizes:
            x = x_all[:tokens].contiguous()
            weights = weights_all[:tokens].contiguous()
            ids = ids_all[:tokens].contiguous()
            eager, per_rank = _time_size(moe, x, weights, ids, args, device)
            key = f"{phase}:{tokens}"
            results[key] = {
                "mtpr": mtpr,
                "max_us": max(per_rank),
                "mean_us": sum(per_rank) / len(per_rank),
                "per_rank_us": per_rank,
            }
            if output_dir is not None:
                path = output_dir / f"{args.shape}_{phase}_{tokens}_rank{rank}.pt"
                same = _check_outputs(path, eager, args.compare_outputs is not None)
                flag = torch.tensor(int(same), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if not int(flag.item()):
                    mismatches.append(key)
            if rank == 0:
                print(
                    f"[TIMING] {args.shape} {key} mtpr={mtpr} "
                    f"max={results[key]['max_us']:.1f}us mean={results[key]['mean_us']:.1f}us",
                    flush=True,
                )
        del moe
        gc.collect()
        torch.cuda.empty_cache()
        _barrier()

    failed = []
    if rank == 0:
        meta = _versions()
        meta.update(shape=args.shape, seed=args.seed)
        meta.update(trials=args.trials, replays=args.replays)
        print(f"[META] {json.dumps(meta)}", flush=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps({"meta": meta, "results": results}, indent=2) + "\n"
            )
        if args.reference:
            ref, spread = _reference(args.reference)
            print(
                f"{'size':<16}{'reference us':>14}{'spread':>9}{'this run us':>14}{'delta':>9}"
            )
            for key, result in results.items():
                if key not in ref:
                    continue
                delta = result["max_us"] / ref[key] - 1.0
                if delta > args.tolerance:
                    failed.append(key)
                print(
                    f"{key:<16}{ref[key]:>14.1f}{spread[key]:>8.1%}"
                    f"{result['max_us']:>14.1f}{delta:>+8.1%}"
                    f"{'  SLOWER' if key in failed else ''}"
                )
        if mismatches:
            print(f"[OUTPUTS] differ from {output_dir}: {mismatches}", flush=True)
        elif args.compare_outputs is not None:
            print(f"[OUTPUTS] bitwise equal to {output_dir}", flush=True)
    status = torch.tensor(int(bool(failed or mismatches)), device=device)
    dist.broadcast(status, src=0)
    _cleanup()
    if int(status.item()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

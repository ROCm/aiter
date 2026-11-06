# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE layer (MXFP4 weights, gfx950) vs the pure-torch EP oracle: accuracy
over token counts and routings, ending in a markdown summary table::

    torchrun --nproc_per_node=8 op_tests/multigpu_tests/test_mega_moe_ep.py \\
        -q a8w4 --mtpr 256 8192 -t 1 64 300 2048 8192 -r uniform zipf:1 spike

Run with plain ``python3`` it relaunches itself under torchrun on up to 8 GPUs, or
skips without >= 2 gfx950 GPUs. Exits 1 if any case failed.

Shape: DeepSeek-V4-Pro (hidden 7168, inter 3072, 48 experts per rank, top-6). Each
case runs one eager forward, then captures it in a CUDA graph and checks the output
of the third replay; the kernel's error flag must stay 0.  ``--mtpr`` <= 256 selects
the small-batch kernel, larger values the large one.

Routings (``route_ids``):

* ``uniform``: top-k of softmax(randn) logits -- every rank and expert evenly loaded.
* ``zipf:S``: the expert of popularity rank r (a fixed random permutation) gets a logit
  bias of -S * log(1 + r).  The hot experts are scattered over the ranks, so the load
  is uneven *between* ranks; a larger S concentrates more tokens on fewer experts.
* ``spike[:P1,P2,..]``: the shape the full model shows after expert-load rebalancing
  (EPLB): hot expert j sits on rank j and is picked by a fraction P_j of the tokens
  (default 0.8, 0.5, 0.35, 0.26, 0.2, 0.17); the other routes go to the remaining
  experts with rank weight max(K / W - P_r, 0.1).  The ranks stay roughly balanced but
  a hot rank's rows mostly belong to ONE expert -- the imbalance is *within* a rank.
* ``same``: every token routes to experts 0..K-1 (one rank takes everything).

``--uneven``: rank r runs ``t * (W + r) // (2 W)`` tokens instead of ``t``.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import subprocess
import sys
import traceback

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

H, I, EPR, K, SWIGLU_LIMIT = 7168, 3072, 48, 6, 10.0
TOL = {"a8w4": 0.10, "a4w4": 0.25}  # rel_l2 vs the oracle
SPIKE = (0.8, 0.5, 0.35, 0.26, 0.2, 0.17)
SUPPORTED_GFX = ("gfx950",)


def route_ids(route, tokens, experts, world, rank, seed, device):
    """``(x, wts, ids)`` of ``tokens`` tokens on this rank for routing ``route``."""
    epr = experts // world
    g = torch.Generator(device=device).manual_seed(777 + rank * 131 + tokens + seed)
    x = torch.randn(tokens, H, dtype=torch.bfloat16, device=device, generator=g) * 0.1
    logits = torch.randn(tokens, experts, device=device, generator=g)
    if route.startswith("zipf:"):
        perm = torch.randperm(experts, generator=torch.Generator().manual_seed(5)).to(
            device
        )
        logits -= float(route[5:]) * torch.log1p(perm.float())
    elif route not in ("uniform", "same") and not route.startswith("spike"):
        raise ValueError(f"unknown route {route!r}")
    wts, ids = torch.topk(torch.softmax(logits, -1), K, -1)
    if route.startswith("spike"):
        ps = [float(v) for v in route[6:].split(",")] if ":" in route else list(SPIKE)
        ps = torch.tensor(ps, device=device)
        hot = torch.tensor(
            [j * epr + (7 * j) % epr for j in range(len(ps))], device=device
        )
        gs = torch.Generator(device=device).manual_seed(
            4242 + rank * 131 + tokens + seed
        )
        pick = torch.rand(tokens, len(ps), device=device, generator=gs) < ps
        wr = torch.full((world,), K / world, device=device)
        wr[: len(ps)] = (wr[: len(ps)] - ps).clamp(min=0.1)
        pe = wr.repeat_interleave(epr)
        pe[hot] = 0
        rest = torch.multinomial(
            pe.expand(tokens, -1), K, replacement=False, generator=gs
        )
        cand = torch.cat([torch.where(pick, hot.expand(tokens, -1), -1), rest], 1)
        cand = torch.gather(
            cand, 1, torch.argsort((cand < 0).int(), dim=1, stable=True)
        )
        ids = cand[:, :K]
    if route == "same":
        ids = torch.arange(K, device=device).expand(tokens, K)
    return x, wts.float().contiguous(), ids.int().contiguous()


def local_weights(rank, device):
    """This rank's local experts: natural MXFP4 ``(w1, w1_scale, w2, w2_scale)``."""
    qz = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    g = torch.Generator(device=device).manual_seed(1234 + rank)
    out = []
    for rows, cols in ((2 * I, H), (H, I)):
        w = torch.randn(
            EPR, rows, cols, dtype=torch.bfloat16, device=device, generator=g
        )
        q, s = qz(w * cols**-0.25, quant_dtype=dtypes.fp4x2)
        out += [
            q.view(EPR, rows, cols // 2).contiguous(),
            s.view(EPR, rows, cols // 32).contiguous(),
        ]
    return tuple(out)


def check(layer, ws, route, tokens, uneven, rank, world, device):
    nb = max(1, tokens * (world + rank) // (2 * world)) if uneven else tokens
    x, wts, ids = route_ids(route, nb, EPR * world, world, rank, 0, device)
    part = layer.engine._part.local  # zeroed so a missed scatter cannot reuse a result
    part.zero_()
    out = layer(x, wts, ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer(x, wts, ids)
    for _ in range(3):
        torch.cuda.synchronize()
        dist.barrier()
        part.zero_()
        graph.replay()
    torch.cuda.synchronize()
    err = layer.poll_errors()
    ref = layer.reference_forward(x, wts, ids, ws).float()
    rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
    stats = torch.tensor([rel, float(err)], device=device)
    allv = [torch.empty_like(stats) for _ in range(world)]
    dist.all_gather(allv, stats)
    del graph
    return max(v[0].item() for v in allv), int(max(v[1].item() for v in allv))


def main() -> int:
    if get_gfx() not in SUPPORTED_GFX:
        if int(os.environ.get("RANK", "0")) == 0:
            aiter.logger.warning("MegaMoE EP unsupported on %s; skipping", get_gfx())
        return 0
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-q",
        "--quant",
        nargs="*",
        default=["a8w4", "a4w4"],
        choices=list(TOL),
        help="Activation quant.\n    e.g.: -q a8w4",
    )
    parser.add_argument(
        "--mtpr",
        type=int,
        nargs="*",
        default=[256, 8192],
        help="max_tok_per_rank (power of two).\n    e.g.: --mtpr 8192",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 7, 64, 256, 300, 2048, 8192],
        help="Tokens per rank (cases above mtpr are skipped).",
    )
    parser.add_argument(
        "-r",
        "--routes",
        nargs="*",
        default=["uniform", "zipf:1", "zipf:2", "spike"],
        help="Routings (see above).\n    e.g.: -r uniform spike:0.9",
    )
    parser.add_argument("--uneven", action="store_true", help="Uneven per-rank tokens.")
    args = parser.parse_args()

    from aiter.ops.flydsl.mega_moe_ep import MegaMoeEP, MegaMoeEPConfig

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    device = torch.device("cuda", lr)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    if rank != 0:
        aiter.logger.setLevel("WARNING")
    ws = local_weights(rank, device)

    rows = []
    for quant, mtpr in itertools.product(args.quant, args.mtpr):
        cfg = MegaMoeEPConfig(
            rank=rank,
            world_size=world,
            model_dim=H,
            inter_dim=I,
            experts=EPR * world,
            topk=K,
            max_tok_per_rank=mtpr,
            quant=quant,
            swiglu_limit=SWIGLU_LIMIT,
        )
        layer = MegaMoeEP(
            cfg, w1=ws[0], w1_scale=ws[1], w2=ws[2], w2_scale=ws[3], device=device
        )
        layer.prepare()
        for tokens, route in itertools.product(args.tokens, args.routes):
            if tokens > mtpr:
                continue
            row = {
                "quant": quant,
                "mtpr": mtpr,
                "tokens": tokens,
                "route": route,
                "uneven": args.uneven,
            }
            try:
                rel, err = check(
                    layer, ws, route, tokens, args.uneven, rank, world, device
                )
                row.update(
                    rel_l2=f"{rel:.4f}",
                    err=err,
                    status="PASS" if rel < TOL[quant] and not err else "FAIL",
                )
            except Exception:  # noqa: BLE001 - one failed case must not end the sweep
                if rank == 0:
                    traceback.print_exc()
                row.update(rel_l2="-", err="-", status="ERROR")
            rows.append(row)
            if rank == 0:
                aiter.logger.info("%s", row)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

    failed = [r for r in rows if r["status"] != "PASS"]
    if rank == 0:
        aiter.logger.info(
            "MegaMoE EP ep%d accuracy:\n%s",
            world,
            pd.DataFrame(rows).to_markdown(index=False),
        )
        aiter.logger.info(
            "MegaMoE EP: %d/%d passed", len(rows) - len(failed), len(rows)
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if failed else 0


def _relaunch() -> int:
    """Plain ``python3`` run (CI): torchrun on up to 8 GPUs, or skip."""
    from aiter.ops.flydsl.mega_moe_ep import mega_moe_ep_supported

    n = torch.cuda.device_count()
    if n < 2 or not mega_moe_ep_supported():
        print("test_mega_moe_ep: skipped (needs >= 2 gfx950 GPUs)", flush=True)
        return 0
    n = 8 if n >= 8 else 4 if n >= 4 else 2
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    cmd += [f"--nproc_per_node={n}", os.path.abspath(__file__), *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main() if "RANK" in os.environ else _relaunch())

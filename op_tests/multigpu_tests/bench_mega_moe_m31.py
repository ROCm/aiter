# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
r"""MegaMoEV2 EP4 at MiniMax-M3.1 decode shapes, on captured M3.1 routing.

Each point gives the decode requests per rank; a request is 8 DSpark verify rows.
Every rank takes distinct captured requests of the layer (MoE input hidden states,
topk ids and weights as the server routed them), so the cross-rank routing is real.
The instance is the one ATOM builds: a8w4, --mtpr 512, swiglu_limit 7, staged route
unless --routed. Weights are random MXFP4: the op streams them, its time does not
depend on their values. Per point it reports the graph-replay time of one call
(median of --trials replays of a --calls-call graph, max over ranks), the relL2
against an emulation of the op's own MXFP8 quantization (the error check), the
relL2 against the fp32 reference of its activation, and the rows each rank receives.
With --baseline, a point whose emulation error exceeds the baseline run's largest
one fails the run.

The capture is a directory of part*.pt files, each a list of per-MoE-call records
{"layer", "rows", "hidden", "topk_ids", "topk_w"}.

    torchrun --nproc-per-node 4 bench_mega_moe_m31.py --capture DIR --act silu --out base.json
    torchrun --nproc-per-node 4 bench_mega_moe_m31.py --capture DIR --act swigluoai \
        --baseline base.json --out oai.json
"""

import argparse
import glob
import json
import random

import test_mega_moe_v2 as T
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

NET = T.NETWORKS["m31"]
ROWS_PER_REQ = 8
# Requests per rank: the c140 / c150 decode points (mean and p90 batch over 4 DP
# ranks under least-requests routing) and a uniform 4..40 sweep around them.
POINTS = {
    "u4": [4, 4, 4, 4],
    "u8": [8, 8, 8, 8],
    "u12": [12, 12, 12, 12],
    "c140_mean": [13, 13, 12, 12],
    "c150_mean": [13, 13, 13, 13],
    "u16": [16, 16, 16, 16],
    "c150_p90": [18, 17, 17, 17],
    "c140_p90": [19, 18, 18, 18],
    "u24": [24, 24, 24, 24],
    "u32": [32, 32, 32, 32],
    "u40": [40, 40, 40, 40],
}


def load_requests(capture, layer):
    reqs = []
    for path in sorted(glob.glob(f"{capture}/part*.pt")):
        for rec in torch.load(path):
            if rec["layer"] != layer or "hidden" not in rec or "topk_ids" not in rec:
                continue
            h = rec["hidden"]
            w = rec["topk_w"][:, : NET["topk"]]
            i = rec["topk_ids"][:, : NET["topk"]]
            for s in range(0, rec["rows"] - ROWS_PER_REQ + 1, ROWS_PER_REQ):
                e = s + ROWS_PER_REQ
                reqs.append((h[s:e], w[s:e], i[s:e]))
    if not reqs:
        raise ValueError(f"no captured requests of layer {layer} in {capture}")
    return reqs


def point_inputs(reqs, counts, rank, seed, device):
    """This rank's requests of a point: a seeded shuffle, split by rank, no request shared."""
    order = list(range(len(reqs)))
    random.Random(seed).shuffle(order)
    need = sum(counts)
    # shortcut: a capture smaller than the point reuses requests; capture more to avoid it.
    order = (order * (need // len(order) + 1))[:need]
    start = sum(counts[:rank])
    mine = [reqs[j] for j in order[start : start + counts[rank]]]
    x = torch.cat([m[0] for m in mine]).to(device).contiguous()
    w = torch.cat([m[1] for m in mine]).to(device).float().contiguous()
    ids = torch.cat([m[2] for m in mine]).to(device).to(torch.int32).contiguous()
    return x, w, ids


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--capture", required=True, help="directory of part*.pt captures")
    p.add_argument("--act", choices=("silu", "swigluoai"), default="swigluoai")
    p.add_argument("--layers", default="3,20,40,59")
    p.add_argument("--points", default="", help="JSON {name: [requests per rank]}")
    p.add_argument("--mtpr", type=int, default=512)
    p.add_argument(
        "--routed", action="store_true", help="build routed=True, pass max_tokens"
    )
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="JSON results (rank 0)")
    p.add_argument(
        "--baseline", help="JSON of a baseline run: gate relL2_vs_emul on its max"
    )
    args = p.parse_args()
    points = json.loads(args.points) if args.points else POINTS
    emul_limit = None
    if args.baseline:
        with open(args.baseline) as f:
            emul_limit = max(r["relL2_vs_emul"] for r in json.load(f)["results"])

    rank, world, device = T._setup_dist()
    if world != 4 or any(len(c) != world for c in points.values()):
        raise ValueError("M3.1 decode is EP4: run 4 ranks, 4 counts per point")
    w1, w1s, w2, w2s, *ref_w = T._quantize_weights(
        NET["model_dim"], NET["inter_dim"], NET["experts"] // world, rank, 0, device
    )
    moe = MegaMoEV2(
        rank=rank,
        world_size=world,
        quant="a8w4",
        w1=w1,
        w1_scale=w1s,
        w2=w2,
        w2_scale=w2s,
        max_tok_per_rank=args.mtpr,
        routed=args.routed,
        **{**NET, "act": args.act},
    )
    results = []
    for layer in (int(v) for v in args.layers.split(",")):
        reqs = load_requests(args.capture, layer)
        for name, counts in points.items():
            x, w, ids = point_inputs(reqs, counts, rank, args.seed, device)
            max_tokens = max(counts) * ROWS_PER_REQ
            kw = {"max_tokens": max_tokens} if args.routed else {}

            def fn(x=x, w=w, ids=ids, kw=kw):
                return moe.forward(x, w, ids, **kw)

            out = fn()[: x.shape[0]].clone()
            T._barrier()
            rel = T._rel_l2(out, T._moe_reference(moe, x, w, ids, ref_w))
            emul = T._rel_l2(out, T._moe_reference(moe, x, w, ids, ref_w, True))
            us = T._time_replays(fn, args.calls, args.trials, device)
            recv = torch.bincount((ids.long() // moe.epr).flatten(), minlength=world)
            dist.all_reduce(recv)
            row = {
                "layer": layer,
                "point": name,
                "requests_per_rank": counts,
                "max_tokens": max_tokens,
                "us": round(us, 1),
                "relL2_vs_emul": round(emul, 6),
                "relL2_vs_fp32": round(rel, 6),
                "recv_rows_per_rank": recv.tolist(),
            }
            results.append(row)
            if rank == 0:
                print(
                    f"[M31] act={args.act} layer={layer} {name:<10} "
                    f"rows={max_tokens:>3} us={us:7.1f} emul={emul:.5f} relL2={rel:.4f}",
                    flush=True,
                )
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
    T._barrier()
    T._cleanup()
    worse = [
        r for r in results if emul_limit is not None and r["relL2_vs_emul"] > emul_limit
    ]
    if worse:
        raise AssertionError(
            f"relL2_vs_emul above the baseline's {emul_limit} at "
            + ", ".join(f"layer {r['layer']} {r['point']}" for r in worse)
        )


if __name__ == "__main__":
    main()

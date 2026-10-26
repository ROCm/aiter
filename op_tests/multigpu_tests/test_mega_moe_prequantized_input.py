# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoE on a quantizing dispatch wire, fed bf16 or already-quantized input.

A caller that produces the dispatch payload itself (e.g. a GEMM with an MXFP4
epilogue) passes it with its e8m0 rows as ``a1_scale``; the layer output must be
bit-identical to handing MegaMoE the bf16 tokens and letting dispatch quantize.

Launch (4x gfx1250):
    torchrun --standalone --nproc_per_node=4 test_mega_moe_prequantized_input.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["AITER_FORCE_A8W4"] = "0"  # fp4 wire <-> a4w4 grouped GEMM

from bench_mega_moe import (
    Dist,
    _import_mori_comm,
    make_routings,
    make_shared_weights,
    raw_quant_weights,
    resolve_spec,
    shuffle_group,
)

from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250
from aiter.ops.quant import dynamic_per_group_scaled_quant
from aiter.utility import dtypes


def run(dist_ctx, args, stage1_fused):
    dev = torch.device("cuda", dist_ctx.local_rank)
    r, world = dist_ctx.rank, dist_ctx.world
    E, hdim, idim, topk, ct = (
        args.experts,
        args.hidden,
        args.inter,
        args.topk,
        args.tokens,
    )
    epr = E // world
    spec = resolve_spec("a4w4_mxfp4")

    w1_bf, w2_bf, _, _ = make_shared_weights(E, hdim, idim, dtypes.bf16, dev)
    q1, s1, q2, s2 = raw_quant_weights(
        w1_bf[r * epr : (r + 1) * epr].contiguous(),
        w2_bf[r * epr : (r + 1) * epr].contiguous(),
    )
    w1, w2, w1_s, w2_s = shuffle_group(q1, s1, q2, s2, spec, epr)

    Communicator = _import_mori_comm()
    uid = dist_ctx.bcast_uid(Communicator.get_unique_id() if r == 0 else None)
    comm = Communicator.init(world, r, uid, per_rank_vmm=16 * 1024**3)
    mega = MegaMoEGfx1250(
        communicator=comm,
        rank=r,
        world_size=world,
        model_dim=hdim,
        inter_dim=idim,
        experts=E,
        topk=topk,
        max_tokens_per_rank=ct,
        activation=spec["activation"],
        gate_mode=spec["gate_mode"].value,
        quant_type=spec["aiter_qtype"],
        dispatch_wire="fp4",
        dispatch_backend="flydsl" if stage1_fused else "mori",
        stage1_fused=stage1_fused,
    )
    comm.barrier()

    routings = make_routings(args.steps, ct, E, topk, dev, seed=4242 + 100 * r)
    gen = torch.Generator(device=dev).manual_seed(1000 + r)
    mismatched = 0
    for step, (ids, wts) in enumerate(routings):
        tokens = ct if step % 2 == 0 else max(1, ct // 3)
        ids, wts = ids[:tokens].contiguous(), wts[:tokens].contiguous()
        x = torch.randn(tokens, hdim, generator=gen, device=dev).to(torch.bfloat16)
        kw = {"w1": w1, "w2": w2, "w1_scale": w1_s, "w2_scale": w2_s}
        q = torch.empty((tokens, hdim // 2), dtype=dtypes.fp4x2, device=dev)
        s = torch.empty((tokens, hdim // 32), dtype=torch.uint8, device=dev)
        dynamic_per_group_scaled_quant(q, x, s, 32, shuffle_scale=False)
        # Prequantized first: dispatch's own quant buffer still holds the
        # previous step, so reading it instead of q/s cannot match.
        y = mega(q, wts, ids, a1_scale=s, **kw).clone()
        y_ref = mega(x, wts, ids, **kw).clone()
        # The scales must reach the experts: doubling them doubles the output.
        y_scaled = mega(q, wts, ids, a1_scale=s + 1, **kw).clone()
        torch.cuda.synchronize()
        if not torch.equal(y, y_ref) or torch.equal(y_scaled, y_ref):
            mismatched += 1
            diff = (y.float() - y_ref.float()).abs().max().item()
            print(f"rank {r} step {step} tokens {tokens}: max diff {diff}", flush=True)
    mismatched = dist_ctx.allreduce_sum(mismatched)
    mega.close()
    comm.barrier()
    return mismatched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--hidden", type=int, default=3584)
    p.add_argument("--inter", type=int, default=512)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--steps", type=int, default=6)
    args = p.parse_args()
    dist_ctx = Dist()
    failed = []
    for stage1_fused in (False, True):
        bad = run(dist_ctx, args, stage1_fused)
        if dist_ctx.rank == 0:
            print(f"stage1_fused={stage1_fused}: {bad} mismatched (rank, step) pairs")
        if bad:
            failed.append(stage1_fused)
    dist_ctx.shutdown()
    assert not failed, f"prequantized input diverged for stage1_fused in {failed}"


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 EP4/EP8 check: MoonEP planning in front of an unchanged MegaMoEGfx1250.

Runs the same skewed routing through a plain MegaMoEGfx1250 and through one
over ``R * (EPR + B)`` experts fed the virtual ids ``MoonEPPlanner`` rewrites
the routes to.  The wide instance reads each layer's ``[EPR home | B slots]``
windows from a ``MoonEPWeightPool`` whose slots the pool's own prefetch kernel
fills.  Planner and pools live on their own cco communicator, apart from the
one MegaMoE uses.  Outputs must match.

    torchrun --nproc-per-node 4 op_tests/multigpu_tests/test_moonep_mega_gfx1250.py
"""

import argparse

import torch
import torch.distributed as dist
from bench_mega_moe import Dist, raw_quant_weights, resolve_spec, shuffle_group
from mori.cco import Communicator

from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250
from aiter.ops.flydsl.kernels.moonep_plan import MoonEPPlanner
from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 7168, 3072, 384, 6


def _home_weights(layer, rank, epr, spec, device):
    """This rank's shuffled a8w4 experts ``[w1, w1_scale, w2, w2_scale]``."""
    gen = torch.Generator(device=device).manual_seed(7 + 31 * layer + 1009 * rank)

    def w(*shape):
        return (
            torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen) * 0.1
        )

    w1 = w(epr, 2 * INTER_DIM, MODEL_DIM)
    w2 = w(epr, MODEL_DIM, INTER_DIM)
    w1_a, w2_a, w1_s, w2_s = shuffle_group(*raw_quant_weights(w1, w2), spec, epr)
    return [t.contiguous() for t in (w1_a, w1_s, w2_a, w2_s)]


def _expert_rows(t, epr):
    """Per-expert shape of ``t``; scales come flattened as ``[epr * rows, cols]``."""
    if t.shape[0] == epr:
        return tuple(t.shape[1:])
    return (t.shape[0] // epr, *t.shape[1:])


def _like(window, t):
    """A pool window in the layout the kernel was given ``t`` in."""
    return window if window.dim() == t.dim() else window.reshape(-1, *t.shape[1:])


def _routing(tokens, rank, world, epr, step, device, *, hot):
    gen = torch.Generator(device=device).manual_seed(1234 + 97 * step + rank)
    scores = torch.randn((tokens, EXPERTS), device=device, generator=gen)
    if hot:
        # Two hot ranks with many similar experts: several prefetches per rank.
        bias = torch.zeros(EXPERTS, device=device)
        for h in (step % world, (step + 1) % world):
            bias[h * epr : (h + 1) * epr] = 1.5
        scores = scores + bias
    values, ids = torch.topk(scores, TOPK, dim=-1)
    x = torch.randn(
        (tokens, MODEL_DIM), dtype=torch.bfloat16, device=device, generator=gen
    )
    return x, values.softmax(-1).contiguous(), ids.to(torch.int32).contiguous()


def _communicator(ctx, per_rank_vmm):
    uid = ctx.bcast_uid(Communicator.get_unique_id() if ctx.rank == 0 else None)
    return Communicator.init(ctx.world, ctx.rank, uid, per_rank_vmm=per_rank_vmm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="weight pools sharing one instance, like model layers",
    )
    parser.add_argument(
        "--stage1-fused",
        action="store_true",
        help="MegaMoE's compact stage-1 plan (fused routing + dispatch)",
    )
    parser.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="negative control: leave the slots stale",
    )
    parser.add_argument(
        "--shared-state",
        action="store_true",
        help="negative control: all layers share one slot state",
    )
    args = parser.parse_args()
    ctx = Dist()
    rank, world = ctx.rank, ctx.world
    device = torch.device("cuda", ctx.local_rank)
    epr = EXPERTS // world
    B = args.slots
    spec = resolve_spec("a8w4_mxfp4")
    mega_comm = moonep_comm = None
    try:
        mega_comm = _communicator(ctx, 16 * 1024**3)
        layers = [
            {"home": _home_weights(layer, rank, epr, spec, device)}
            for layer in range(args.layers)
        ]
        home0 = layers[0]["home"]
        parts = [(_expert_rows(t, epr), t.dtype) for t in home0]
        pool_bytes = sum((epr + B) * t.numel() // epr * t.element_size() for t in home0)
        # Planner buffers are a few KiB; leave 1 GiB of headroom per rank.
        moonep_comm = _communicator(ctx, args.layers * pool_bytes + (1 << 30))

        common = {
            "rank": rank,
            "world_size": world,
            "model_dim": MODEL_DIM,
            "inter_dim": INTER_DIM,
            "topk": TOPK,
            "max_tokens_per_rank": args.tokens,
            "activation": spec["activation"],
            "gate_mode": spec["gate_mode"].value,
            "quant_type": spec["aiter_qtype"],
            "dispatch_wire": "fp8",
            "dispatch_backend": "flydsl",
            "stage1_fused": args.stage1_fused,
        }
        plain = MegaMoEGfx1250(communicator=mega_comm, experts=EXPERTS, **common)
        wide = MegaMoEGfx1250(
            communicator=mega_comm, experts=world * (epr + B), **common
        )
        planner = MoonEPPlanner(
            rank=rank,
            world_size=world,
            experts=EXPERTS,
            slots=B,
            max_routes=args.tokens * TOPK,
            communicator=moonep_comm,
        )
        shared = planner.new_slot_state()
        for layer in layers:
            pool = MoonEPWeightPool(
                rank=rank,
                world_size=world,
                experts_per_rank=epr,
                prefetch_slots=B,
                parts=parts,
                communicator=moonep_comm,
            )
            pool.stage_home([t.view(epr, *_expert_rows(t, epr)) for t in layer["home"]])
            layer["pool"] = pool
            layer["windows"] = [_like(w, t) for w, t in zip(pool.local, layer["home"])]
            layer["state"] = shared if args.shared_state else planner.new_slot_state()

        def forward(moe, weights, x, wts, ids):
            w1, w1s, w2, w2s = weights
            return moe.forward(
                x, wts, ids, w1=w1, w1_scale=w1s, w2=w2, w2_scale=w2s
            ).clone()

        def run(layer, x, wts, ids, balance):
            if balance:
                virtual = planner.plan(ids, layer["state"])
                if not args.skip_prefetch:
                    layer["pool"].prefetch(
                        layer["state"]["placed"][rank], layer["state"]["prev"][rank]
                    )
            else:
                virtual = planner.home_ids(ids)
            return forward(wide, layer["windows"], x, wts, virtual)

        ok = True
        for step in range(args.steps):
            for index, layer in enumerate(layers):
                for balance in (True, False) if step == 0 else (True,):
                    # Layers see different hot experts, so their slots diverge.
                    x, wts, ids = _routing(
                        args.tokens,
                        rank,
                        world,
                        epr,
                        step + 5 * index,
                        device,
                        hot=step % 2 == 0,
                    )
                    ref = forward(plain, layer["home"], x, wts, ids)
                    dist.barrier()
                    out = run(layer, x, wts, ids, balance)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    diff = (out.float() - ref.float()).abs()
                    scale = ref.float().abs().max().clamp(min=1e-6)
                    max_rel = float(diff.max() / scale)
                    exact = bool(torch.equal(out, ref))
                    placed, prev = layer["state"]["placed"], layer["state"]["prev"]
                    prefetched = int((placed >= 0).sum())
                    kept = int(((placed >= 0) & (placed == prev)).sum())
                    stats = ctx.gather_objects((max_rel, exact, prefetched, kept))
                    if rank == 0:
                        rels = [s[0] for s in stats]
                        exacts = sum(int(s[1]) for s in stats)
                        step_ok = max(rels) < 2e-2
                        ok &= step_ok
                        print(
                            f"step={step} layer={index} balance={balance} "
                            f"hot={step % 2 == 0} placed={stats[0][2]} "
                            f"kept={stats[0][3]} exact_ranks={exacts}/{world} "
                            f"max_rel={max(rels):.3e} -> "
                            f"{'OK' if step_ok else 'FAIL'}",
                            flush=True,
                        )
        if rank == 0:
            print("ALL_OK" if ok else "SOME_FAILED", flush=True)
    finally:
        torch.cuda.synchronize(device)
        dist.barrier()
        for comm in (moonep_comm, mega_comm):
            if comm is not None:
                comm.destroy()
        ctx.shutdown()


if __name__ == "__main__":
    main()

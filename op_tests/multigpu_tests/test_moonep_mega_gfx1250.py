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
import os
import sys

import torch
import torch.distributed as dist
from bench_mega_moe import Dist, raw_quant_weights, resolve_spec, shuffle_group
from mori.cco import Communicator

from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import MegaMoEGfx1250
from aiter.ops.flydsl.kernels.moonep_plan import MoonEPPlanner
from aiter.ops.flydsl.kernels.moonep_weights import MoonEPWeightPool

MODEL_DIM, INTER_DIM, EXPERTS, TOPK = 7168, 3072, 384, 6
# The dispatch wire must carry what the expert GEMM reads as its A operand.
DISPATCH_WIRE = {"a4w4": "fp4", "a8w4": "fp8"}


def _home_weights(layer, rank, epr, spec, device):
    """This rank's shuffled mxfp4 experts ``[w1, w1_scale, w2, w2_scale]``."""
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


def _stale_slots(layer_index, layer, rank, epr, spec, device):
    """Slots of this rank whose bytes differ from their owner's home rows."""
    by_owner = {}
    for slot, expert in enumerate(layer["state"]["held"][rank].tolist()):
        if expert >= 0:
            by_owner.setdefault(expert // epr, []).append((slot, expert % epr))
    stale = []
    for owner, slots in sorted(by_owner.items()):
        # Rebuilt from the owner's seed: the weights are deterministic.
        home = _home_weights(layer_index, owner, epr, spec, device)
        for i, t in enumerate(home):
            rows = t.view(epr, *_expert_rows(t, epr))
            for slot, local in slots:
                got = layer["pool"].prefetched[i][slot].contiguous().view(torch.uint8)
                if not torch.equal(got, rows[local].contiguous().view(torch.uint8)):
                    stale.append(slot)
        del home
    return sorted(set(stale))


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
        "--quant",
        choices=sorted(DISPATCH_WIRE),
        default="a4w4",
        help="activation dtype of the expert GEMM (gfx1250 defaults to a4w4)",
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
    parser.add_argument(
        "--same-routing",
        action="store_true",
        help="one hot routing for every step and layer: from step 1 on every "
        "slot is a sticky hit, which --shared-state must then get wrong",
    )
    parser.add_argument(
        "--aa",
        action="store_true",
        help="control: compare plain against itself (MegaMoE's own run-to-run noise)",
    )
    parser.add_argument(
        "--max-flaky-rate",
        type=float,
        default=0.3,
        help="largest share of comparisons that may pass only on the rerun "
        "(plain against itself fails ~9%% of them on one rank)",
    )
    args = parser.parse_args()
    if args.quant == "a8w4":
        os.environ["AITER_FORCE_A8W4"] = "1"
    ctx = Dist()
    rank, world = ctx.rank, ctx.world
    device = torch.device("cuda", ctx.local_rank)
    epr = EXPERTS // world
    B = args.slots
    spec = resolve_spec(f"{args.quant}_mxfp4")
    mega_comm = moonep_comm = planner = None
    layers = []
    try:
        mega_comm = _communicator(ctx, 16 * 1024**3)
        layers = [
            {"home": _home_weights(layer, rank, epr, spec, device)}
            for layer in range(args.layers)
        ]
        parts = [(_expert_rows(t, epr), t.dtype) for t in layers[0]["home"]]
        moonep_comm = _communicator(
            ctx,
            MoonEPPlanner.vmm_bytes(world, EXPERTS)
            + args.layers * MoonEPWeightPool.vmm_bytes(epr, B, parts),
        )

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
            "dispatch_wire": DISPATCH_WIRE[args.quant],
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
            """``(output, ids the instance ran)``; ``--aa`` reruns plain instead."""
            if args.aa:
                return forward(plain, layer["home"], x, wts, ids), ids
            if balance:
                virtual = planner.plan(ids, layer["state"])
                if not args.skip_prefetch:
                    layer["pool"].prefetch(
                        layer["state"]["placed"][rank], layer["state"]["prev"][rank]
                    )
            else:
                virtual = planner.home_ids(ids)
            return forward(wide, layer["windows"], x, wts, virtual), virtual

        def rerun(layer, x, wts, ran):
            """The same ids again, no new plan or prefetch."""
            if args.aa:
                return forward(plain, layer["home"], x, wts, ran)
            return forward(wide, layer["windows"], x, wts, ran)

        def compare(out, ref):
            diff = (out.float() - ref.float()).abs()
            scale = ref.float().abs().max().clamp(min=1e-6)
            return float(diff.max() / scale), bool(torch.equal(out, ref))

        ok, flakes, compares = True, 0, 0
        for step in range(args.steps):
            for index, layer in enumerate(layers):
                for balance in (True, False) if step == 0 else (True,):
                    # Layers see different hot experts, so their slots diverge.
                    x, wts, ids = _routing(
                        args.tokens,
                        rank,
                        world,
                        epr,
                        0 if args.same_routing else step + 5 * index,
                        device,
                        hot=args.same_routing or step % 2 == 0,
                    )
                    ref = forward(plain, layer["home"], x, wts, ids)
                    dist.barrier()
                    out, ran = run(layer, x, wts, ids, balance)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    max_rel, exact = compare(out, ref)
                    placed, prev = layer["state"]["placed"], layer["state"]["prev"]
                    prefetched = int((placed >= 0).sum())
                    kept = int(((placed >= 0) & (placed == prev)).sum())
                    # Every rank computes the same placement; the tables must agree.
                    tables = tuple(
                        tuple(layer["state"][name].flatten().tolist())
                        for name in ("held", "placed")
                    )
                    stats = ctx.gather_objects(
                        (max_rel, exact, prefetched, kept, tables)
                    )
                    rels = [s[0] for s in stats]
                    step_ok = max(rels) < 2e-2
                    agree = len({s[4] for s in stats}) == 1
                    compares += 1
                    retry, flaky = "", False
                    if not step_ok:
                        # A mismatch is MegaMoE's own noise (--aa shows it)
                        # only if every slot holds its owner's bytes and the
                        # same ids, with no new plan or prefetch, then pass.
                        stale = []
                        if balance and not args.aa:
                            stale = _stale_slots(index, layer, rank, epr, spec, device)
                        slots = ctx.gather_objects(stale)
                        ref = forward(plain, layer["home"], x, wts, ids)
                        dist.barrier()
                        again = compare(rerun(layer, x, wts, ran), ref)
                        torch.cuda.synchronize(device)
                        redo = ctx.gather_objects(again)
                        slots_ok = not any(slots)
                        flaky = slots_ok and max(m for m, _ in redo) < 2e-2
                        retry = " retry=" + ",".join(
                            f"r{r}:{'exact' if e else f'{m:.2e}'}"
                            for r, (m, e) in enumerate(redo)
                        )
                        if not slots_ok:
                            retry += " stale_slots=" + ",".join(
                                f"r{r}:{s}" for r, s in enumerate(slots) if s
                            )
                    if rank == 0:
                        exacts = sum(int(s[1]) for s in stats)
                        ok &= (step_ok or flaky) and agree
                        flakes += int(flaky)
                        bad = [r for r, s in enumerate(stats) if s[0] >= 2e-2]
                        verdict = (
                            "OK"
                            if step_ok and agree
                            else (
                                f"FLAKY ranks={bad}"
                                if flaky and agree
                                else f"FAIL ranks={bad}"
                            )
                        )
                        print(
                            f"step={step} layer={index} balance={balance} "
                            f"hot={step % 2 == 0} placed={stats[0][2]} "
                            f"kept={stats[0][3]} tables_agree={agree} "
                            f"exact_ranks={exacts}/{world} "
                            f"max_rel={max(rels):.3e} -> {verdict}{retry}",
                            flush=True,
                        )
        if rank == 0:
            limit = int(args.max_flaky_rate * compares)
            print(f"flaky_steps={flakes}/{compares} limit={limit}", flush=True)
            ok &= flakes <= limit
            print("ALL_OK" if ok else "SOME_FAILED", flush=True)
    finally:
        torch.cuda.synchronize(device)
        dist.barrier()
        for layer in layers:
            layer.pop("windows", None)
            if "pool" in layer:
                layer.pop("pool").close()
        if planner is not None:
            planner.close()
        for comm in (moonep_comm, mega_comm):
            if comm is not None:
                comm.destroy()
        ctx.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

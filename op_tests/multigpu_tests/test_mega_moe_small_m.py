# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Small-input path of MegaMoEV2 (forward_small_m).

One MegaMoEV2 built at MTPR 8192 with small-input runtimes next to it
(small_m_mtprs, default 256 and 512). Checks accuracy against an fp32
reference and the main path (uneven, idle, padded ranks), per-layer weight
rebinding with interleaved main calls, CUDA graph replay, and paired timing.

    AITER_MEGA_SMALL_M=1 torchrun --nproc-per-node 8 test_mega_moe_small_m.py
    # against a same-MTPR main path, timing only:
    AITER_MEGA_SMALL_M=1 torchrun --nproc-per-node 8 test_mega_moe_small_m.py \
        --mtpr 256 --time-only --time-bs 140,168,196,224,252
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mega_moe_v2 import (
    NETWORKS,
    _barrier,
    _cleanup,
    _make_inputs,
    _quantize_weights,
    _reduce_float,
    _reference,
    _setup_dist,
)

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2


def _rel_l2(out, ref):
    err = torch.sum((out.float() - ref.float()) ** 2)
    den = torch.sum(ref.float() ** 2)
    dist.all_reduce(err)
    dist.all_reduce(den)
    return float(torch.sqrt(err / den.clamp_min(1e-30)))


def _all_true(flag, device):
    t = torch.tensor(int(bool(flag)), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())


def _main_forward(moe, x, w, ids):
    return moe(x, w, ids)[: x.shape[0]]


def _rank_case(tokens_per_rank, rank, x, w, ids, experts, pad_rows=0):
    """Local inputs for one case; 0 tokens means an idle rank that sends one sentinel row."""
    n = tokens_per_rank[rank]
    if n == 0:
        xi = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
        wi = torch.zeros((1, w.shape[1]), dtype=w.dtype, device=w.device)
        ii = torch.full((1, ids.shape[1]), experts, dtype=ids.dtype, device=ids.device)
        return xi, wi, ii, 0
    xi, wi, ii = x[:n].clone(), w[:n].clone(), ids[:n].clone()
    if pad_rows:
        # _fill_padded_rows: rows past num_token_non_padded route to `experts`.
        ii[n - pad_rows :] = experts
    return xi, wi, ii, n - pad_rows


def run(args):
    rank, world, device = _setup_dist()
    net = NETWORKS["v4_pro"]
    experts, topk = net["experts"], net["topk"]
    local_experts = experts // world
    try:
        packs = [
            _quantize_weights(
                net["model_dim"],
                net["inter_dim"],
                local_experts,
                rank,
                args.seed + 17 * i,
                device,
            )
            for i in range(2)
        ]
        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=packs[0][0],
            w1_scale=packs[0][1],
            w2=packs[0][2],
            w2_scale=packs[0][3],
            max_tok_per_rank=args.mtpr,
            large_m_state_slots=args.slots,
            small_m_mtprs=tuple(args.small_m_mtprs),
            **net,
        )
        if moe.small_m_mtpr != max(args.small_m_mtprs):
            raise AssertionError(
                f"small-input path not enabled (small_m_mtpr={moe.small_m_mtpr}); set AITER_MEGA_SMALL_M=1"
            )
        large_m = moe._large_m is not None
        if rank == 0:
            print(
                f"[SETUP] mtpr={moe.mtpr} small_m_mtpr={moe.small_m_mtpr} large_m={large_m} world={world}",
                flush=True,
            )

        x, w, ids = _make_inputs(
            moe.small_m_mtpr,
            net["model_dim"],
            experts,
            topk,
            rank,
            args.seed,
            device,
            force_fanout_boundary=False,
            inject_invalid_route=False,
            force_padding_boundary=False,
        )
        refw = [p[4:] for p in packs]
        binds = [p[:4] for p in packs]

        def bind(slot, which):
            moe.bind_layer(slot, *binds[which])

        # ---- 1. accuracy vs fp32 reference and vs main forward ----------
        cases = [
            ("uniform1", [1] * world, 0),
            ("uniform8", [8] * world, 0),
            ("uniform32", [32] * world, 0),
            ("uniform51", [51] * world, 0),
            ("uniform64", [64] * world, 0),
            ("uneven", [64, 1, 33, 7, 50, 64, 2, 17][:world], 0),
            ("idle_ranks", [0, 64, 0, 13, 64, 0, 40, 0][:world], 0),
            ("padded", [64, 48, 64, 32, 56, 64, 24, 64][:world], 5),
            ("all_idle", [0] * world, 0),
            ("uniform70", [70] * world, 0),
            ("uniform126", [126] * world, 0),
            ("uniform168", [168] * world, 0),
            ("uniform224", [224] * world, 0),
            ("uniform256", [256] * world, 0),
            ("dec_uneven", [168, 126, 126, 84, 84, 70, 56, 1][:world], 0),
            ("dec_idle", [224, 0, 7, 0, 140, 0, 1, 0][:world], 0),
            ("dec_padded", [224, 224, 224, 224, 224, 224, 224, 224][:world], 37),
            ("uniform280", [280] * world, 0),
            ("uniform364", [364] * world, 0),
            ("uniform448", [448] * world, 0),
            ("uniform512", [512] * world, 0),
            ("c384_uneven", [336, 182, 280, 119, 224, 308, 154, 210][:world], 0),
            ("c384_padded", [448] * world, 160),
            ("c384_idle", [448, 0, 7, 0, 364, 1, 0, 280][:world], 0),
        ]
        cases = [c for c in cases if max(c[1]) <= moe.small_m_mtpr]
        failures = []
        small_m_out_cache = {}
        if args.time_only:
            cases = []
        for name, tpr, pad in cases:
            for which in range(2):
                bind(which, which)
                xi, wi, ii, real = _rank_case(tpr, rank, x, w, ids, experts, pad)
                out_g = moe.forward_small_m(xi, wi, ii, max_tokens=max(max(tpr), 1))[
                    : xi.shape[0]
                ].clone()
                _barrier()
                out_m = _main_forward(moe, xi, wi, ii).clone()
                _barrier()
                rows = slice(0, real)
                if max(tpr) == 0:
                    rel_main, rel_ref = 0.0, 0.0
                else:
                    ref = _reference(
                        xi[rows],
                        wi[rows],
                        ii[rows],
                        refw[which],
                        rank,
                        world,
                        net["model_dim"],
                        net["inter_dim"],
                        experts,
                        net["swiglu_limit"],
                    )
                    rel_ref = _rel_l2(out_g[rows], ref)
                    rel_main = _rel_l2(out_g[rows], out_m[rows])
                # Padded rows are discarded by the caller; they only must not be
                # NaN/Inf. Whether they are exactly zero is reported, not gated.
                pad_ok = _all_true(bool(torch.isfinite(out_g).all()), device)
                pad_zero = (
                    _all_true(bool((out_g[real:] == 0).all()), device) if pad else True
                )
                small_m_out_cache[(name, which)] = (
                    xi,
                    wi,
                    ii,
                    out_g,
                    real,
                    max(max(tpr), 1),
                )
                ok = rel_ref < args.rtol and rel_main < args.rtol and pad_ok
                if rank == 0:
                    print(
                        f"[ACC] {name:10s} w{which} tpr={tpr} pad={pad} relL2_vs_ref={rel_ref:.6f} "
                        f"relL2_vs_main={rel_main:.6f} finite={pad_ok} pad_rows_zero={pad_zero} {'PASS' if ok else 'FAIL'}",
                        flush=True,
                    )
                if not ok:
                    failures.append(f"acc {name} w{which}")

        # ---- 1b. truly empty ranks and rank-local -1 slots -----------------
        if not args.time_only:
            bind(0, 0)
            tpr = [0, 64, 0, 13, 64, 0, 40, 0][:world]
            n = tpr[rank]
            xi, wi, ii = x[:n].clone(), w[:n].clone(), ids[:n].clone()
            if rank % 2:
                # The combine input still holds the rows of the calls above: a -1 slot must not read them.
                gen = torch.Generator(device=device).manual_seed(args.seed + rank)
                ii[torch.rand(ii.shape, generator=gen, device=device) < 1 / 3] = -1
            out = moe.forward_small_m(xi, wi, ii, max_tokens=max(tpr))[:n].clone()
            _barrier()
            ref = _reference(
                xi,
                wi,
                ii,
                refw[0],
                rank,
                world,
                net["model_dim"],
                net["inter_dim"],
                experts,
                net["swiglu_limit"],
            )
            rel_ref = _rel_l2(out, ref)
            ok = rel_ref < args.rtol and _all_true(
                bool(torch.isfinite(out).all()), device
            )
            if rank == 0:
                print(
                    f"[EDGE] empty_ranks+masked_slots tpr={tpr} relL2_vs_ref={rel_ref:.6f} "
                    f"{'PASS' if ok else 'FAIL'}",
                    flush=True,
                )
            if not ok:
                failures.append("empty ranks / masked slots")
            # Skewed routes: the rank holding the hot experts sends FP8 rows and reads cached weights, the others
            # keep BF16 rows, so the combine reads both formats.
            hot = torch.arange(topk, dtype=ids.dtype, device=device)
            for name, n, share in (
                ("hot60", 128, 3),
                ("one_rank", 64, 5),
                ("hot60", 448, 3),
            ):
                if n > moe.small_m_mtpr:
                    continue
                xi, wi, ii = x[:n].clone(), w[:n].clone(), ids[:n].clone()
                ii[(torch.arange(n, device=device) % 5) < share] = hot
                out = moe.forward_small_m(xi, wi, ii, max_tokens=n)[:n].clone()
                _barrier()
                ref = _reference(
                    xi,
                    wi,
                    ii,
                    refw[0],
                    rank,
                    world,
                    net["model_dim"],
                    net["inter_dim"],
                    experts,
                    net["swiglu_limit"],
                )
                rel_ref = _rel_l2(out, ref)
                ok = rel_ref < args.rtol and _all_true(
                    bool(torch.isfinite(out).all()), device
                )
                if rank == 0:
                    print(
                        f"[EDGE] {name} tpr={n} relL2_vs_ref={rel_ref:.6f} {'PASS' if ok else 'FAIL'}",
                        flush=True,
                    )
                if not ok:
                    failures.append(f"skewed {name} {n}")

        # ---- 2. layer rebinding + main/small-input alternation (eager) ------
        if args.time_only:
            args.alt_steps = 0
        gen = torch.Generator().manual_seed(args.seed)
        mism = 0
        main_calls = 0
        keys = [k for k in small_m_out_cache if k[0] != "all_idle"]
        cs = torch.cuda.Stream()
        for step in range(args.alt_steps):
            slot = step % args.slots
            which = (step + slot // 7) & 1
            bind(slot, which)
            if torch.rand((), generator=gen).item() < 0.3:
                # A larger main call on the same bound layer between small-input calls.
                n = 64 + int(torch.randint(0, 200, (), generator=gen).item())
                xm = torch.randn(
                    (n, net["model_dim"]), dtype=torch.bfloat16, device=device
                )
                wm = w[:1].expand(n, -1).contiguous()
                im = ids[:1].expand(n, -1).contiguous()
                _main_forward(moe, xm, wm, im)
                main_calls += 1
            k = keys[int(torch.randint(0, len(keys), (), generator=gen).item())]
            name = k[0]
            xi, wi, ii, expect, real, mt = small_m_out_cache[(name, which)]
            out = moe.forward_small_m(xi, wi, ii, max_tokens=mt)[:real]
            # Only real rows: sentinel/idle rows are never reduced, so they hold
            # whatever the combine buffer last had (the caller discards them).
            mism += int((out != expect[:real]).any(dim=-1).sum().item())
        mism_t = torch.tensor(mism, device=device)
        dist.all_reduce(mism_t)
        ok = int(mism_t.item()) == 0
        if rank == 0:
            print(
                f"[ALT] steps={args.alt_steps} slots={args.slots} main_calls={main_calls} "
                f"mismatched_rows_vs_first_call={int(mism_t.item())} {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
        if not ok:
            failures.append("alternation")
        _barrier()

        # ---- 3. CUDA graph capture/replay --------------------------------
        n = args.graph_bs
        if args.time_only:
            args.graph_reps = 0
        sx = x[:n].clone()
        sw = w[:n].clone()
        si = ids[:n].clone()
        graph_out = []

        def graph_body():
            graph_out.clear()
            for layer in range(args.graph_layers):
                which = layer & 1
                bind(layer, which)
                # The output is the operator's combine buffer (as in main), valid
                # until its next call: copy it out before the next call.
                graph_out.append(
                    moe.forward_small_m(sx, sw, si, max_tokens=n)[:n].clone()
                )

        # Warm (eager) so dispatch tables exist before capture.
        graph_body()
        _barrier()
        g = torch.cuda.CUDAGraph()
        cs = torch.cuda.Stream()
        with torch.cuda.graph(g, stream=cs):
            graph_body()
        gmism = 0
        for rep in range(args.graph_reps):
            ri = rep % 3
            gx, gw, gi = _make_inputs(
                n,
                net["model_dim"],
                experts,
                topk,
                rank,
                args.seed + 100 + ri,
                device,
                force_fanout_boundary=False,
                inject_invalid_route=False,
                force_padding_boundary=False,
            )
            sx.copy_(gx)
            sw.copy_(gw)
            si.copy_(gi)
            _barrier()
            g.replay()
            torch.cuda.synchronize()
            replay = [o.clone() for o in graph_out]
            _barrier()
            for layer in range(args.graph_layers):
                bind(layer, layer & 1)
                eager = moe.forward_small_m(gx, gw, gi, max_tokens=n)[:n]
                gmism += int((eager != replay[layer]).any(dim=-1).sum().item())
            _barrier()
        gm = torch.tensor(gmism, device=device)
        dist.all_reduce(gm)
        ok = int(gm.item()) == 0
        if rank == 0:
            print(
                f"[GRAPH] layers={args.graph_layers} reps={args.graph_reps} "
                f"replay_vs_eager_mismatched_rows={int(gm.item())} {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
        if not ok:
            failures.append("graph")
        del g
        _barrier()

        # ---- 4. paired timing (graph of `reps` calls) -----------------------
        for n in args.time_bs:
            if n <= x.shape[0]:
                tx, tw, ti = (
                    x[:n].contiguous(),
                    w[:n].contiguous(),
                    ids[:n].contiguous(),
                )
            else:
                tx, tw, ti = _make_inputs(
                    n,
                    net["model_dim"],
                    experts,
                    topk,
                    rank,
                    args.seed + 500,
                    device,
                    force_fanout_boundary=False,
                    inject_invalid_route=False,
                    force_padding_boundary=False,
                )
            bind(0, 0)
            res = {}
            paths = [
                (
                    "main" if not large_m else "large_m",
                    lambda tx=tx, tw=tw, ti=ti: moe(tx, tw, ti),
                )
            ]
            if n <= moe.small_m_mtpr:
                paths.append(
                    (
                        "small_m",
                        lambda tx=tx, tw=tw, ti=ti, n=n: moe.forward_small_m(
                            tx, tw, ti, max_tokens=n
                        ),
                    )
                )
            for label, fn in paths:
                for _ in range(3):
                    fn()
                _barrier()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=cs):
                    for _ in range(args.time_calls):
                        fn()
                for _ in range(3):
                    gr.replay()
                _barrier()
                times = []
                for _ in range(args.time_trials):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    _barrier()
                    s.record()
                    gr.replay()
                    e.record()
                    torch.cuda.synchronize()
                    times.append(s.elapsed_time(e) * 1000.0 / args.time_calls)
                times.sort()
                med = times[len(times) // 2]
                res[label] = _reduce_float(med, device, dist.ReduceOp.MAX)
                del gr
            if rank == 0 and len(res) == 1:
                (a,) = res.items()
                print(
                    f"[TIME] bs/rank={n:5d} {a[0]}={a[1]:8.1f}us (median of {args.time_trials}, max over ranks)",
                    flush=True,
                )
            if rank == 0 and len(res) == 2:
                a, b = list(res.items())
                print(
                    f"[TIME] bs/rank={n:3d} {a[0]}={a[1]:8.1f}us small_m={b[1]:8.1f}us "
                    f"speedup={a[1] / b[1]:.3f}x (median of {args.time_trials}, max over ranks)",
                    flush=True,
                )

        _barrier()
        if failures:
            raise AssertionError("FAILED: " + ", ".join(failures))
        if rank == 0:
            print("[RESULT] ALL PASS", flush=True)
    finally:
        _cleanup()


def main():
    if get_gfx() != "gfx950":
        print(f"[SKIP] the small- and large-input paths need gfx950, got {get_gfx()}")
        return
    p = argparse.ArgumentParser()
    p.add_argument("--mtpr", type=int, default=8192)
    p.add_argument("--slots", type=int, default=61)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rtol", type=float, default=0.08)
    p.add_argument("--alt-steps", type=int, default=600)
    p.add_argument("--graph-layers", type=int, default=8)
    p.add_argument("--graph-reps", type=int, default=6)
    p.add_argument(
        "--time-bs",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[7, 14, 28, 56, 70, 84, 126, 168, 224, 280, 336, 448],
    )
    p.add_argument(
        "--small-m-mtprs",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[256, 512],
        help="small-input runtime MTPRs",
    )
    p.add_argument("--graph-bs", type=int, default=168)
    p.add_argument("--time-only", action="store_true")
    p.add_argument("--time-calls", type=int, default=50)
    p.add_argument("--time-trials", type=int, default=15)
    run(p.parse_args())


if __name__ == "__main__":
    main()

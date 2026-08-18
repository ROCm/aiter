# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Correctness + A/B for the TDM dispatch transport (gfx1250).

Runs the same routing through ``dispatch_transport='vector'`` and
``'tdm'`` and checks both against a CPU model of what every rank should
receive. The two transports assign recv slots in different orders -- vector
takes one remote atomic per route, TDM reserves a contiguous run per (block,
peer) -- so the check is order-independent: it keys each received row by the
``(src_pe, src_tok)`` its srcmap names and compares the row's payload, indices
and weights against that source token.

    torchrun --standalone --nproc_per_node=4 test_dispatch_tdm.py
    HIDDEN=7168 TOPK=8 EPR=32 SWEEP=8,128,512 BENCH=1 torchrun ...
"""
import os
import sys

import torch
import torch.distributed as dist

from aiter.ops.flydsl.dispatch_combine_v2 import (
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)
from mori.cco import Communicator

HIDDEN = int(os.environ.get("HIDDEN", 2048))
TOPK = int(os.environ.get("TOPK", 8))
EPR = int(os.environ.get("EPR", 8))
SWEEP = [int(x) for x in os.environ.get("SWEEP", "8,64,512").split(",")]
BENCH = os.environ.get("BENCH") == "1"
ITERS = int(os.environ.get("ITERS", 50))
WARMUP = int(os.environ.get("WARMUP", 10))
DTYPE = torch.bfloat16


def _bootstrap():
    """torchrun + gloo, used only to carry the cco unique id and pass/fail counts."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    uid = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    return rank, world, local, uid[0]


def _allreduce_sum(v):
    t = torch.tensor([v], dtype=torch.int64)
    dist.all_reduce(t)
    return int(t[0])


def _expected_rows(all_idx, all_tok, rank, world, epr, topk, n_tok):
    """{(src_pe, src_tok): sorted expert ids} for every token routed to `rank`.

    A token routed to several of this rank's experts arrives once (the sender
    dedups per destination PE), and it carries its whole topk index/weight row,
    not just the experts that live here.
    """
    want = {}
    for pe in range(world):
        idx = all_idx[pe]
        for t in range(n_tok):
            row = [int(e) for e in idx[t]]
            if any(e // epr == rank for e in row if e >= 0):
                want[(pe, t)] = row
    return want


def _check(op, cfg, comm, rank, world, all_idx, all_wts, all_tok, n_tok, tag):
    out, out_w, _out_s, out_i, total_recv, routing = op.dispatch(
        all_tok[rank][:n_tok],
        all_wts[rank][:n_tok],
        None,
        all_idx[rank][:n_tok],
        return_routing=True,
    )
    torch.cuda.synchronize()
    comm.barrier()

    recv = int(total_recv.item())
    src = routing.disp_tok_id_to_src_tok_id_local[:recv].cpu()
    got_tok = out[:recv].float().cpu()
    got_idx = out_i[:recv].cpu()
    got_wts = out_w[:recv].cpu()

    want = _expected_rows(all_idx, all_tok, rank, world, cfg.num_experts_per_rank, TOPK, n_tok)
    errs = []
    if recv != len(want):
        errs.append(f"recv={recv} expected={len(want)}")
    seen = set()
    max_tok = cfg.max_num_inp_token_per_rank
    for r in range(recv):
        enc = int(src[r])
        key = (enc // max_tok, enc % max_tok)
        if key not in want:
            errs.append(f"row {r}: unexpected source {key}")
            break
        if key in seen:
            errs.append(f"row {r}: duplicate source {key}")
            break
        seen.add(key)
        pe, t = key
        if not torch.equal(got_idx[r], all_idx[pe][t].cpu()):
            errs.append(f"row {r} src={key}: indices mismatch")
            break
        if not torch.allclose(got_wts[r], all_wts[pe][t].cpu(), atol=0, rtol=0):
            errs.append(f"row {r} src={key}: weights mismatch")
            break
        if not torch.equal(got_tok[r], all_tok[pe][t].float().cpu()):
            errs.append(f"row {r} src={key}: payload mismatch")
            break
    ok = not errs
    bad = _allreduce_sum(0 if ok else 1)
    if rank == 0:
        print(
            f"# {tag} ct={n_tok}: {'PASS' if bad == 0 else 'FAIL'} (recv={recv})",
            flush=True,
        )
    if errs and rank == 0:
        print(f"    rank{rank}: {errs[0]}", flush=True)
    return bad == 0


def _time(op, comm, all_idx, all_wts, all_tok, rank, n_tok):
    """(us per dispatch, tokens landed in this rank's recv buffer)."""
    inp, w, i = all_tok[rank][:n_tok], all_wts[rank][:n_tok], all_idx[rank][:n_tok]
    for _ in range(WARMUP):
        _out, _w, _s, _i, total_recv = op.dispatch(inp, w, None, i)
    torch.cuda.synchronize()
    comm.barrier()
    recv = int(total_recv.item())
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        op.dispatch(inp, w, None, i)
    e.record()
    torch.cuda.synchronize()
    comm.barrier()
    return s.elapsed_time(e) / ITERS * 1000.0, recv


def main():
    rank, world, local, uid = _bootstrap()
    max_tok = max(SWEEP)
    n_experts = EPR * world
    torch.manual_seed(1234)

    # Same routing on every rank's view: each rank generates its own tokens but
    # all of them are needed to model what any one rank receives, so they are
    # generated from a per-rank seed the whole world can reproduce.
    all_idx, all_wts, all_tok = [], [], []
    for pe in range(world):
        g = torch.Generator().manual_seed(1000 + pe)
        idx = torch.stack(
            [torch.randperm(n_experts, generator=g)[:TOPK] for _ in range(max_tok)]
        ).int()
        wts = torch.rand(max_tok, TOPK, generator=g)
        tok = torch.randn(max_tok, HIDDEN, generator=g).to(DTYPE)
        dev = f"cuda:{local}"
        all_idx.append(idx.to(dev))
        all_wts.append(wts.to(dev))
        all_tok.append(tok.to(dev))

    comm = Communicator.init(world, rank, uid)

    failures = 0
    timings = {}
    for transport in ("vector", "tdm"):
        cfg = EpDispatchCombineConfig(
            rank=rank,
            world_size=world,
            hidden_dim=HIDDEN,
            max_num_inp_token_per_rank=max_tok,
            num_experts_per_rank=EPR,
            num_experts_per_token=TOPK,
            data_type=DTYPE,
            dispatch_transport=transport,
        )
        op = EpDispatchCombineOp(cfg, comm)
        try:
            for ct in SWEEP:
                if not _check(
                    op, cfg, comm, rank, world, all_idx, all_wts, all_tok, ct, transport
                ):
                    failures += 1
            if BENCH:
                timings[transport] = {
                    ct: _time(op, comm, all_idx, all_wts, all_tok, rank, ct)
                    for ct in SWEEP
                }
        finally:
            op.close()

    if BENCH and rank == 0:
        # Payload landed per rank: recv tokens of a hidden row each. Counts the
        # rank's own tokens too -- those never cross the fabric, so at world=4
        # roughly a quarter of this is local traffic. Same for both transports,
        # so the ratio is unaffected; the absolute number is not a link figure.
        row_b = HIDDEN * DTYPE.itemsize
        print(f"\n# dispatch (hidden={HIDDEN} topk={TOPK} world={world})")
        print(f"# GB/s = recv tokens x {row_b}B per rank / time")
        print(
            f"{'tokens':>8} {'recv':>8} {'vector':>9} {'tdm':>9} {'speedup':>8}"
            f" {'vec GB/s':>9} {'tdm GB/s':>9}"
        )
        for ct in SWEEP:
            (v, recv), (t, _) = timings["vector"][ct], timings["tdm"][ct]
            gb = recv * row_b / 1e9
            print(
                f"{ct:>8} {recv:>8} {v:>9.1f} {t:>9.1f} {v / t:>7.2f}x"
                f" {gb / (v * 1e-6):>9.1f} {gb / (t * 1e-6):>9.1f}"
            )

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

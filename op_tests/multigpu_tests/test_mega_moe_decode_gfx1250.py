# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx1250's single-kernel paths on gfx1250: small_m (AITER_MEGA_DECODE)
and large_m (AITER_MEGA_LARGE).

    python3 test_mega_moe_decode_gfx1250.py [--world N]

Spawns one rank per GPU (4 by default, 2 is the minimum). Anywhere else -- not
gfx1250, fewer than two GPUs, no mori -- it prints why and exits 0, since CI's
MULTIGPU job runs every file in this directory on its gfx950 runners.

Covered, on the compact dispatch with the fp4 wire (a4w4) and the fused bf16
combine, with and without a fused shared expert, for each megakernel:
  * the megakernel against the fp32 reference;
  * the megakernel against the staged path of the same instance (its per-call
    switch), bit for bit and both ways round, at several token counts per
    rank, uneven across ranks;
  * the path follows the step's receive bound: small_m runs the steps bounded
    to AITER_MEGA_DECODE, large_m every larger one, and a compact plan started
    for a step small_m serves is dropped;
  * a captured graph's replay against the eager result, bit for bit.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# The weights, routings, fp32 reference and its tolerances are the bench's.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HIDDEN, INTER, TOPK = 7168, 1024, 6
EXPERTS_PER_RANK = 8
# The instance's tokens per rank, and the steps the megakernel serves
# (AITER_MEGA_DECODE): those whose receive bound is at most world * SMALL_TOKENS.
MAX_TOKENS = 64
SMALL_TOKENS = 32
# Tokens per rank of each step: rank r runs max(1, t - r), so the ranks differ.
STEPS = (1, 7, 16, 32)
# large_m's steps (unbounded, so past small_m): few tokens, then up to the
# instance's MAX_TOKENS.
LARGE_STEPS = (5, 33, 48, 64)


def _skip_reason() -> str | None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return "needs two or more GPUs"
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx1250":
        return f"MegaMoEGfx1250 is gfx1250-only, this is {get_gfx()}"
    try:
        import mori.cco  # noqa: F401
    except (ImportError, OSError) as error:
        return f"needs mori with cco ({error})"
    return None


class _Checks:
    """Rank-local checks raised on every rank together, so a rank that fails
    one does not leave its peers inside a collective."""

    def __init__(self, label):
        self.label, self.failed = label, []

    def __call__(self, ok, what):
        if not ok:
            self.failed.append(what)

    def settle(self):
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, self.failed)
        self.failed = []
        bad = {rank: failed for rank, failed in enumerate(gathered) if failed}
        if bad:
            raise AssertionError(f"{self.label}: {bad}")


class _Case:
    """One MegaMoE with the megakernel built, its weights and a step source."""

    def __init__(self, rank, world, comm, device, n_shared):
        import bench_mega_moe as B

        from aiter.ops.flydsl.kernels.mega_moe_gfx1250 import (
            MegaMoEGeometry,
            MegaMoEGfx1250,
            MegaMoERecipe,
        )

        self.B, self.rank, self.world, self.device = B, rank, world, device
        self.n_shared = n_shared
        self.experts = EXPERTS_PER_RANK * world
        self.spec = B.resolve_spec("a4w4_mxfp4")
        w1, w2, sw1, sw2 = B.make_shared_weights(
            self.experts, HIDDEN, INTER, torch.bfloat16, device, n_shared=n_shared
        )
        lo, hi = rank * EXPERTS_PER_RANK, (rank + 1) * EXPERTS_PER_RANK
        w1_l, w2_l = w1[lo:hi], w2[lo:hi]
        if n_shared:
            w1_l, w2_l = torch.cat((w1_l, sw1)), torch.cat((w2_l, sw2))
        q1, s1, q2, s2 = B.raw_quant_weights(w1_l.contiguous(), w2_l.contiguous())
        self.weights = B.shuffle_group(
            q1, s1, q2, s2, self.spec, EXPERTS_PER_RANK + n_shared
        )
        self.ref = B.RefModel(w1, w2, sw1, sw2, self.spec, device)
        recipe = MegaMoERecipe.from_legacy(
            stage1_fused=True,
            stage2_fused=True,
            dispatch_wire="fp4",
            combine_quant="none",
            activation=self.spec["activation"],
            gate_mode=self.spec["gate_mode"].value,
            quant_type=self.spec["aiter_qtype"],
        )
        geometry = MegaMoEGeometry(
            rank=rank,
            world_size=world,
            model_dim=HIDDEN,
            inter_dim=INTER,
            experts=self.experts + n_shared * world,
            topk=TOPK + n_shared,
            max_tokens_per_rank=MAX_TOKENS,
        )
        self.mega = MegaMoEGfx1250.from_recipe(recipe, geometry, communicator=comm)
        self.dmk = self.mega._ep.decode_mega
        self.lmk = self.mega._ep.large_mega
        assert self.dmk is not None, "AITER_MEGA_DECODE did not build small_m"
        assert self.lmk is not None, "AITER_MEGA_LARGE did not build large_m"

    def step(self, tokens, seed):
        """(x, dispatch ids, weights, logical ids) of one step on this rank."""
        B = self.B
        ids, wts = B.make_routings(
            1, tokens, self.experts, TOPK, self.device, seed=seed + 100 * self.rank
        )[0]
        gen = torch.Generator().manual_seed(seed + 500 + self.rank)
        x = torch.randn(tokens, HIDDEN, generator=gen).to(torch.bfloat16)
        x = x.to(self.device)
        if not self.n_shared:
            return x, ids, wts, ids, wts
        ((d_ids, d_wts),) = B.shared_dispatch_routings(
            [(ids, wts)], EXPERTS_PER_RANK, self.rank, self.n_shared
        )
        ((l_ids, l_wts),) = B.shared_logical_routings(
            [(ids, wts)], self.experts, self.n_shared
        )
        return x, d_ids, d_wts, l_ids, l_wts

    @property
    def small_bound(self):
        return self.world * SMALL_TOKENS

    def layer(self, x, ids, wts, *, mega=True, bound="small", large=False):
        """One layer; ``bound`` is the step's recv_token_bound ("small": the
        largest small_m serves). ``mega`` switches the megakernels on, large_m
        only with ``large`` (otherwise larger steps run the staged path)."""
        self.dmk.enabled = mega
        self.lmk.enabled = mega and large
        w1, w2, w1s, w2s = self.weights
        y = self.mega(
            self.B._rmsnorm(x),
            wts,
            ids,
            w1=w1,
            w2=w2,
            w1_scale=w1s,
            w2_scale=w2s,
            recv_token_bound=self.small_bound if bound == "small" else bound,
        )
        return x + y

    def scramble(self, seed, tokens, **kw):
        """One staged step on other inputs: the regions the paths share
        (comb_inp, and large_m's rows / rowmap) then hold none of the next
        step's bytes, so a megakernel that skipped a write cannot pass on
        leftovers."""
        x, ids, wts, _, _ = self.step(max(1, tokens - self.rank), seed=seed)
        self.layer(x, ids, wts, mega=False, **kw)

    def close(self):
        self.mega.close()


def _check_small(case, comm):
    label = f"small_m, shared={case.n_shared}"
    check = _Checks(label)
    B = case.B
    for i, t in enumerate(STEPS):
        tokens = max(1, t - case.rank)
        x, ids, wts, l_ids, l_wts = case.step(tokens, seed=17 * (i + 1))
        mega = case.layer(x, ids, wts, mega=True).clone()
        staged = case.layer(x, ids, wts, mega=False).clone()
        case.scramble(seed=1000 + i, tokens=SMALL_TOKENS)
        again = case.layer(x, ids, wts, mega=True).clone()
        torch.cuda.synchronize()
        what = f"tokens={tokens}"
        check(torch.equal(mega, staged), f"{what}: megakernel != staged")
        check(torch.equal(mega, again), f"{what}: megakernel not deterministic")
        ref = (x.float() + case.ref.layer(x.float(), l_ids, l_wts)).to(torch.bfloat16)
        diff = B._calc_diff(mega, ref)
        tol = B.default_logits_tol("a4w4_mxfp4", "none", 1)
        check(diff <= tol, f"{what}: diff {diff:.5f} > tol {tol:.5f}")
        check(case.dmk.errors() == 0, f"{what}: expert slot overflow")
        check.settle()

    # The path follows the bound: the megakernel serves only steps bounded to
    # world * SMALL_TOKENS, whatever the instance holds.
    dmk = case.dmk
    check(dmk.serves(case.small_bound), "small bound not served")
    check(not dmk.serves(case.small_bound + 1), "bound past the slots served")
    check(not dmk.serves(None), "unbounded step served (instance > small)")
    check.settle()
    x, ids, wts, _, _ = case.step(SMALL_TOKENS - case.rank, seed=71)
    small = case.layer(x, ids, wts).clone()
    unbounded = case.layer(x, ids, wts, bound=None).clone()  # staged
    # A plan started for a step the megakernel then serves is dropped, and the
    # next staged step plans its own routing.
    x2, ids2, wts2, _, _ = case.step(MAX_TOKENS - case.rank, seed=73)
    staged_only = case.layer(x2, ids2, wts2, bound=None).clone()
    case.mega.prefetch_compact_plan(ids, recv_token_bound=None)
    small_again = case.layer(x, ids, wts).clone()
    after = case.layer(x2, ids2, wts2, bound=None).clone()
    torch.cuda.synchronize()
    check(torch.equal(small, unbounded), "bounded megakernel != unbounded staged")
    check(torch.equal(small, small_again), "megakernel after a pending plan differs")
    check(torch.equal(after, staged_only), "staged step after a dropped plan differs")
    check.settle()

    # A captured step replays the eager bits.
    x, ids, wts, _, _ = case.step(SMALL_TOKENS - case.rank, seed=99)
    eager = case.layer(x, ids, wts, mega=True).clone()
    torch.cuda.synchronize()
    comm.barrier()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        case.layer(x, ids, wts, mega=True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    comm.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = case.layer(x, ids, wts, mega=True)
    case.scramble(seed=3001, tokens=SMALL_TOKENS)
    graph.replay()
    torch.cuda.synchronize()
    check(torch.equal(captured, eager), "graph replay != eager")
    check.settle()
    del graph


def _check_large(case, comm):
    label = f"large_m, shared={case.n_shared}"
    check = _Checks(label)
    B = case.B
    ep = case.mega._ep
    case.dmk.enabled = case.lmk.enabled = True
    check(ep.mega_for(case.small_bound) is case.dmk, "small bound not on small_m")
    check(ep.mega_for(case.small_bound + 1) is case.lmk, "larger bound not on large_m")
    check(ep.mega_for(None) is case.lmk, "unbounded step not on large_m")
    check.settle()
    # Unbounded steps, then one bounded between small_m's and the instance's.
    steps = [(t, None) for t in LARGE_STEPS] + [(40, case.world * 48)]
    for i, (t, bound) in enumerate(steps):
        tokens = max(1, t - case.rank)
        x, ids, wts, l_ids, l_wts = case.step(tokens, seed=23 * (i + 1))
        mega = case.layer(x, ids, wts, bound=bound, large=True).clone()
        staged = case.layer(x, ids, wts, mega=False, bound=bound).clone()
        case.scramble(seed=2000 + i, tokens=MAX_TOKENS, bound=None)
        again = case.layer(x, ids, wts, bound=bound, large=True).clone()
        torch.cuda.synchronize()
        what = f"tokens={tokens} bound={bound}"
        check(torch.equal(mega, staged), f"{what}: large_m != staged")
        check(torch.equal(mega, again), f"{what}: large_m not deterministic")
        ref = (x.float() + case.ref.layer(x.float(), l_ids, l_wts)).to(torch.bfloat16)
        diff = B._calc_diff(mega, ref)
        tol = B.default_logits_tol("a4w4_mxfp4", "none", 1)
        check(diff <= tol, f"{what}: diff {diff:.5f} > tol {tol:.5f}")
        check.settle()

    # A captured step replays the eager bits.
    x, ids, wts, _, _ = case.step(MAX_TOKENS - case.rank, seed=97)
    eager = case.layer(x, ids, wts, bound=None, large=True).clone()
    torch.cuda.synchronize()
    comm.barrier()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        case.layer(x, ids, wts, bound=None, large=True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    comm.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = case.layer(x, ids, wts, bound=None, large=True)
    case.scramble(seed=3002, tokens=MAX_TOKENS, bound=None)
    graph.replay()
    torch.cuda.synchronize()
    check(torch.equal(captured, eager), "graph replay != eager")
    check.settle()
    del graph


def _worker(local_rank, world, port, failures):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(local_rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(local_rank),
        AITER_FORCE_A8W4="0",
        AITER_MEGA_DECODE=str(SMALL_TOKENS),
        AITER_MEGA_LARGE="1",
    )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo", rank=local_rank, world_size=world)
    from mori.cco import Communicator

    uid = [Communicator.get_unique_id() if local_rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = Communicator.init(world, local_rank, uid[0], per_rank_vmm=8 * 1024**3)
    comm.barrier()
    failed = []
    for n_shared in (0, 1):
        name = f"megakernels shared={n_shared}"
        case = None
        try:
            case = _Case(local_rank, world, comm, device, n_shared)
            comm.barrier()
            _check_small(case, comm)
            _check_large(case, comm)
            status = "ok"
        except Exception:  # noqa: BLE001
            failed.append(name)
            status = "FAILED\n" + traceback.format_exc()
        finally:
            torch.cuda.synchronize()
            comm.barrier()
            if case is not None:
                case.close()
            comm.barrier()
        if local_rank == 0:
            print(f"[{name}] {status}", flush=True)
        dist.barrier()
    comm.destroy()
    dist.destroy_process_group()
    failures[local_rank] = len(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=int, default=4)
    args = parser.parse_args()
    reason = _skip_reason()
    if reason is not None:
        print(f"SKIP test_mega_moe_decode_gfx1250: {reason}")
        return 0
    world = min(args.world, torch.cuda.device_count())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    failures = mp.get_context("spawn").Manager().dict()
    mp.spawn(_worker, args=(world, port, failures), nprocs=world, join=True)
    bad = sum(failures.values())
    print("test_mega_moe_decode_gfx1250:", "FAILED" if bad else "passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

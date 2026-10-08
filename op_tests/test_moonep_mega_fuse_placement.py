# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Single-GPU check of the MoonEP placement fused into MegaMoE prepare.

Runs ``emit_moonep_placement`` + ``emit_moonep_virtual_counts`` in one
standalone CTA on a synthetic gathered histogram and compares against
``build_prefill_reference_plan`` and a Python virtual-count split.
"""

import argparse
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.ops.flydsl.kernels.mega_moe.moonep_fuse import (
    emit_moonep_placement,
    emit_moonep_virtual_counts,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.moonep import MoonEPPlanConfig, build_prefill_reference_plan

R, E, B, K = 8, 384, 8, 6
EPN = E // R
VS = EPN + B
COUNT_STRIDE = E + R
VIRTUAL_STRIDE = R * VS + R
WAVES = 8

_LAUNCHERS = {}


def _launcher(rank: int, balance: bool):
    key = (rank, balance)
    if key in _LAUNCHERS:
        return _LAUNCHERS[key]

    @fx.struct
    class Lds:
        alloc: fx.Array[fx.Int32, E * R, 16]
        key: fx.Array[fx.Int32, R * (E + 1), 16]
        ecount: fx.Array[fx.Int32, E, 16]
        rem: fx.Array[fx.Int32, R * 64, 16]
        quota: fx.Array[fx.Int32, R * R, 16]
        bal: fx.Array[fx.Int32, R, 16]
        etc: fx.Array[fx.Int32, R * B, 16]
        target: fx.Array[fx.Int32, R * B, 16]

    @flyc.kernel(
        name=f"test_moonep_fuse_r{rank}_b{int(balance)}",
        known_block_size=[WAVES * 64, 1, 1],
    )
    def kernel(
        count: fx.Int64,
        alloc_cumsum: fx.Int64,
        expert_to_slot: fx.Int64,
        slot_held: fx.Int64,
        slot_prev: fx.Int64,
        slot_placed: fx.Int64,
        logical_pair_base: fx.Int64,
        virtual_count: fx.Int64,
        virtual_hist: fx.Int64,
        virtual_pair_base: fx.Int64,
    ):
        lds = fx.SharedAllocator().allocate(Lds).peek()
        emit_moonep_placement(
            count, alloc_cumsum, expert_to_slot, slot_held, slot_prev, slot_placed,
            lds.alloc.ptr, lds.key.ptr, lds.ecount.ptr, lds.rem.ptr,
            lds.quota.ptr, lds.bal.ptr, lds.etc.ptr, lds.target.ptr,
            num_waves=WAVES, npes=R, experts=E, slots=B,
            count_stride=COUNT_STRIDE, balance=balance,
        )
        emit_moonep_virtual_counts(
            count, alloc_cumsum, expert_to_slot, logical_pair_base,
            virtual_count, virtual_hist, virtual_pair_base,
            num_waves=WAVES, npes=R, experts=E, slots=B, rank=rank,
            count_stride=COUNT_STRIDE, virtual_stride=VIRTUAL_STRIDE,
        )

    @flyc.jit
    def launch(
        count: fx.Int64, alloc_cumsum: fx.Int64, expert_to_slot: fx.Int64,
        slot_held: fx.Int64, slot_prev: fx.Int64, slot_placed: fx.Int64,
        logical_pair_base: fx.Int64, virtual_count: fx.Int64,
        virtual_hist: fx.Int64, virtual_pair_base: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            count, alloc_cumsum, expert_to_slot, slot_held, slot_prev, slot_placed,
            logical_pair_base, virtual_count, virtual_hist, virtual_pair_base,
        ).launch(grid=(1, 1, 1), block=(WAVES * 64, 1, 1), stream=stream)

    _LAUNCHERS[key] = launch
    return launch


def _routing(tokens: int, skew: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if skew < 0:
        # Hot home: many similar experts on two ranks, so every destination
        # wants several remote experts and the top-B cut and rollback run.
        popularity = torch.ones(E, dtype=torch.float64)
        popularity[: 2 * EPN] = -skew
        popularity *= 1 + 0.3 * torch.rand(E, generator=gen, dtype=torch.float64)
    else:
        popularity = torch.arange(1, E + 1, dtype=torch.float64) ** (-skew)
        popularity = popularity[torch.randperm(E, generator=gen)]
    ids = torch.multinomial(
        popularity.expand(R * tokens, E), K, replacement=False, generator=gen
    )
    return ids.view(R, tokens, K).to(torch.int32)


def _settle_reference(want, held):
    target = torch.full_like(want, -1)
    for r in range(R):
        taken = set()
        for s in range(B):
            e = int(want[r, s])
            hits = [t for t in range(B) if e >= 0 and int(held[r, t]) == e]
            if hits:
                target[r, s] = hits[0]
                taken.add(hits[0])
        free = iter(t for t in range(B) if t not in taken)
        for s in range(B):
            if want[r, s] >= 0 and target[r, s] < 0:
                target[r, s] = next(free)
    placed = torch.full_like(want, -1)
    for r in range(R):
        for s in range(B):
            if target[r, s] >= 0:
                placed[r, target[r, s]] = want[r, s]
    return target, placed, torch.where(placed >= 0, placed, held)


def _virtual_reference(tpe, alloc, expert_to_slot, rank, pair_base):
    vcount = torch.zeros(R, VIRTUAL_STRIDE, dtype=torch.int64)
    vbase = torch.zeros(VIRTUAL_STRIDE, dtype=torch.int64)
    cumsum = alloc.t().cumsum(dim=1)  # [E, R]
    for e in range(E):
        begin = 0
        for s in range(R):
            end = begin + int(tpe[s, e])
            low = 0
            for d in range(R):
                high = int(cumsum[e, d])
                lo, hi = max(begin, low), min(end, high)
                if hi > lo:
                    column = d * VS + int(expert_to_slot[d, e])
                    vcount[s, column] = hi - lo
                    if s == rank:
                        vbase[column] = int(pair_base[e]) + lo - begin
                low = high
            begin = end
    return vcount, vbase


def run_case(rank: int, skew: float, seed: int, *, sticky_held: bool, balance=True):
    dev = torch.device("cuda")
    tokens = 2048
    ids = _routing(tokens, skew, seed)
    tpe = torch.stack([torch.bincount(ids[r].flatten().long(), minlength=E) for r in range(R)])
    count = torch.zeros(R, COUNT_STRIDE, dtype=torch.int32)
    count[:, :E] = tpe.to(torch.int32)
    logical_pair_base = torch.zeros(COUNT_STRIDE, dtype=torch.int32)
    logical_pair_base[1:] = count[rank].cumsum(0)[:-1].to(torch.int32)

    config = MoonEPPlanConfig(
        rank=rank, world_size=R, num_tokens=tokens, top_k=K, num_experts=E,
        prefetch_slots=B,
    )
    ref = build_prefill_reference_plan(config, ids[rank].to(dev), tpe.to(dev))
    ref_alloc = ref.alloc.cpu().long()  # [dest, E]
    ref_slot = ref.expert_to_slot.cpu().long()
    ref_etc = ref.experts_to_copy.cpu().long()
    if not balance:
        ref_alloc = torch.zeros(R, E, dtype=torch.int64)
        for e in range(E):
            ref_alloc[e // EPN, e] = int(tpe[:, e].sum())
        ref_etc = torch.full((R, B), -1, dtype=torch.int64)
        ref_slot = torch.full((R, E), -1, dtype=torch.int64)
        for d in range(R):
            ref_slot[d, d * EPN:(d + 1) * EPN] = torch.arange(EPN)

    held = torch.full((R, B), -1, dtype=torch.int64)
    if sticky_held:
        gen = torch.Generator().manual_seed(seed + 1)
        for d in range(R):
            pool = [e for e in range(E) if e // EPN != d]
            wanted = [int(x) for x in ref_etc[d] if x >= 0]
            picks = wanted[: B // 2] + [pool[int(i)] for i in torch.randperm(len(pool), generator=gen)[:B]]
            uniq = []
            for e in picks:
                if e not in uniq:
                    uniq.append(e)
            order = torch.randperm(B, generator=gen)
            for j, slot in enumerate(order.tolist()):
                held[d, slot] = uniq[j] if j < len(uniq) else -1
    target, ref_placed, ref_held = _settle_reference(ref_etc, held)
    ref_slot_settled = ref_slot.clone()
    for d in range(R):
        for s in range(B):
            if ref_etc[d, s] >= 0:
                ref_slot_settled[d, ref_etc[d, s]] = EPN + int(target[d, s])
    ref_vcount, ref_vbase = _virtual_reference(tpe, ref_alloc, ref_slot_settled, rank, logical_pair_base)

    t = lambda x: x.to(dev, torch.int32).contiguous()  # noqa: E731
    g_count = t(count)
    g_cumsum = torch.zeros(E * R, dtype=torch.int32, device=dev)
    g_slot = torch.zeros(R * E, dtype=torch.int32, device=dev)
    g_held = t(held)
    g_prev = torch.zeros(R * B, dtype=torch.int32, device=dev)
    g_placed = torch.zeros(R * B, dtype=torch.int32, device=dev)
    g_pair_base = t(logical_pair_base)
    g_vcount = torch.full((R * VIRTUAL_STRIDE,), 77, dtype=torch.int32, device=dev)
    g_vhist = torch.full((VIRTUAL_STRIDE,), 77, dtype=torch.int32, device=dev)
    g_vbase = torch.full((VIRTUAL_STRIDE,), 77, dtype=torch.int32, device=dev)
    args = [g_count, g_cumsum, g_slot, g_held, g_prev, g_placed, g_pair_base, g_vcount, g_vhist, g_vbase]
    _run_compiled(
        _launcher(rank, balance), *[fx.Int64(a.data_ptr()) for a in args],
        fx.Stream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    got_alloc = g_cumsum.view(E, R).cpu().long()
    got_alloc = torch.cat([got_alloc[:, :1], got_alloc.diff(dim=1)], dim=1).t()
    vcount = g_vcount.view(R, VIRTUAL_STRIDE).cpu().long()
    checks = {
        "alloc": torch.equal(got_alloc, ref_alloc),
        "expert_to_slot": torch.equal(g_slot.view(R, E).cpu().long(), ref_slot_settled),
        "slot_prev": torch.equal(g_prev.view(R, B).cpu().long(), held),
        "slot_placed": torch.equal(g_placed.view(R, B).cpu().long(), ref_placed),
        "slot_held": torch.equal(g_held.view(R, B).cpu().long(), ref_held),
        "virtual_count": torch.equal(vcount, ref_vcount),
        "virtual_hist": torch.equal(g_vhist.cpu().long(), ref_vcount[rank]),
        "virtual_pair_base": torch.equal(
            torch.where(ref_vcount[rank] > 0, g_vbase.cpu().long(), 0), ref_vbase
        ),
        "conservation": torch.equal(vcount.sum(0).view(-1)[: R * VS].view(R, VS).sum(1),
                                    ref_alloc.sum(1)),
    }
    moved = int((ref_alloc.sum(1) - torch.tensor([int(tpe[:, d * EPN:(d + 1) * EPN].sum()) for d in range(R)])).abs().sum())
    bad = [k for k, ok in checks.items() if not ok]
    if os.environ.get("MOONEP_TEST_DUMP"):
        print("placed", g_placed.view(R, B).cpu().tolist()[:2])
        print("ref   ", ref_placed.tolist()[:2])
    print(f"rank={rank} skew={skew} seed={seed} sticky={sticky_held} balance={balance} "
          f"moved={moved} prefetch={int((ref_etc >= 0).sum())} -> {'OK' if not bad else 'FAIL ' + ','.join(bad)}")
    return not bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    cases = [(0, 1.2, 1, False, True), (3, 1.2, 2, True, True), (7, 0.0, 3, False, True),
             (5, 0.8, 4, True, True), (2, 1.2, 5, True, False),
             (1, -6.0, 6, False, True), (4, -6.0, 7, True, True), (6, -3.0, 8, True, True)]
    if args.quick:
        cases = cases[:1]
    ok = all([run_case(r, s, seed, sticky_held=st, balance=bal) for r, s, seed, st, bal in cases])
    print("ALL_OK" if ok else "SOME_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

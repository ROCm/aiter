# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MoonEP weight prefetch timing, with the byte accounting done properly.

Two things make the existing prefetch number untrustworthy:

1. **Slot count is routing dependent.**  Upstream's ``prefetch`` row moves one
   ``[epn, H, Hp]`` matrix; ours is three independent ops (gate/up/down) each
   moving one, so the comparable unit is *one* of ours.  But how much any of
   them moves depends on how many entries of ``experts_to_copy[rank]`` are
   populated -- at uniform routing that was 1 slot out of 48, i.e. the measured
   time was almost entirely fixed overhead.

2. **Not every slot is remote.**  A slot whose expert lives on this rank is a
   local HBM copy, not wire traffic.  Counting all slots as remote inflates the
   apparent bandwidth by ~R/(R-1) at best, and much more when routing is skewed
   toward local experts.

That second point matters beyond prefetch: a previous run recorded 42 slots /
1.23 GB / 2.87 ms = 430 GB/s for this kernel, which is a *remote read*
(moonep_weight_prefetch.py:82 loads from the peer, :88 stores locally).  That
would contradict ``moonep_link_probe``, which measured remote reads pinned at
230-240 GB/s regardless of depth, block count, or cache policy.  Both cannot be
right, and the conclusion "do not build a multi-stage combine pipeline" rests on
the probe being right.  So this script reports remote bytes only, and re-runs
the streaming probe in the same process at the same byte count as a control.

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 \
        op_tests/benchmark_moonep_prefetch.py
"""

from __future__ import annotations

import json
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.moonep_link_probe import (
    make_link_probe_jit,
    probe_stride_dwords,
)
from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)
from aiter.ops.flydsl.moonep import MoonEPPlanConfig, build_reference_plan
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP
from op_tests.profile_moonep_gfx950_real_shape import _inputs
from op_tests.test_moonep_gfx950_real_shape import (
    B,
    E,
    H,
    I,
    K,
    R,
    S,
    TOKEN_PADDING,
    _identity_home_weights,
    _share_shmem_unique_id,
)

WARMUP = int(os.environ.get("MOONEP_BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("MOONEP_BENCH_ITERS", "20"))
BLOCK_NUMS = [
    int(x) for x in os.environ.get("MOONEP_BLOCK_NUMS", "128,256,1024").split(",")
]
# "profile" reproduces the harness routing (token*K + k + rank*13) % B, which
# touches only B of the E experts and therefore fills many prefetch slots.
# "uniform" is realistic routing and barely migrates anything.
ROUTINGS = os.environ.get("MOONEP_ROUTINGS", "uniform,profile").split(",")


def time_gpu_op(launch_fn, group):
    for _ in range(WARMUP):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        launch_fn()
    end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / ITERS * 1e3
    world = dist.get_world_size(group=group)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(world)]
    dist.all_gather(outs, t, group=group)
    allr = torch.cat(outs)
    return allr.mean().item(), allr.max().item()


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    assert (
        ms.shmem_init_attr(
            ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
            rank,
            world_size,
            _share_shmem_unique_id(rank),
        )
        == 0
    )

    experts_per_rank = E // R
    results = {}
    stats = {}

    for routing in ROUTINGS:
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=S,
            top_k=K,
            num_experts=E,
            prefetch_slots=B,
            token_padding=TOKEN_PADDING,
        )
        hidden, route_weights, topk, local_tpe = _inputs(rank, device)
        if routing == "uniform":
            gen = torch.Generator(device="cpu").manual_seed(20260810 + rank)
            topk = (
                torch.rand(S, E, generator=gen).topk(K, dim=1).indices.to(torch.int32)
            ).to(device)
            local_tpe = torch.bincount(
                topk.reshape(-1).to(torch.int64), minlength=E
            ).to(torch.int32)
        gathered = [torch.empty_like(local_tpe) for _ in range(R)]
        dist.all_gather(gathered, local_tpe)
        plan = build_reference_plan(config, topk, torch.stack(gathered))
        torch.cuda.synchronize(device)
        dist.barrier()

        # Remote bytes only: a slot whose expert already lives here is a local
        # HBM copy, not wire traffic.
        sel = plan.experts_to_copy[rank]
        valid = sel[sel >= 0]
        owners = (valid // experts_per_rank).to(torch.int64)
        slots = int(valid.numel())
        remote_slots = int((owners != rank).sum().item())
        weight_numel = H * I
        remote_bytes = remote_slots * weight_numel * 2
        total_bytes = slots * weight_numel * 2
        # Slot counts are wildly unequal across ranks (0..B), so a single
        # rank's byte count divided by the cross-rank mean time is meaningless.
        # Gather them and report the busiest rank, which is the critical path.
        counts = torch.zeros(2, dtype=torch.int64, device=device)
        counts[0] = slots
        counts[1] = remote_slots
        allc = [torch.empty(2, dtype=torch.int64, device=device) for _ in range(R)]
        dist.all_gather(allc, counts)
        per_rank_slots = [int(c[0].item()) for c in allc]
        per_rank_remote = [int(c[1].item()) for c in allc]
        weight_bytes = weight_numel * 2
        stats[routing] = {
            "slots": slots,
            "remote_slots": remote_slots,
            "remote_bytes": remote_bytes,
            "total_bytes": total_bytes,
            "per_rank_slots": per_rank_slots,
            "per_rank_remote": per_rank_remote,
            "max_rank_remote_bytes": max(per_rank_remote) * weight_bytes,
            "sum_remote_bytes": sum(per_rank_remote) * weight_bytes,
        }
        if rank == 0:
            print(
                f"[setup] routing={routing}: per-rank slots={per_rank_slots} "
                f"remote={per_rank_remote}; busiest rank moves "
                f"{max(per_rank_remote) * weight_bytes / 1e6:.1f} MB, "
                f"system total {sum(per_rank_remote) * weight_bytes / 1e6:.1f} MB",
                flush=True,
            )

        for block_num in BLOCK_NUMS:
            ep = MoonEPBF16ReferenceEP(
                config,
                H,
                I,
                dispatch_block_num=block_num,
                prefetch_block_num=block_num,
            )
            hg, hu, hd = _identity_home_weights(device)
            ep.load_home_weights(hg, hu, hd)
            del hg, hu, hd
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            dist.barrier()

            def prefetch_one():
                # The op takes the full [R, B] table and selects its own row.
                ep.gate_op.prefetch(plan.experts_to_copy)

            # The op compiles-without-running on its first call.
            prefetch_one()
            prefetch_one()
            torch.cuda.synchronize(device)
            dist.barrier()

            gate = ep.gate_op
            local_sel = plan.experts_to_copy[rank].contiguous()
            fast_jit = make_moonep_weight_prefetch_fast_jit(
                experts_per_rank=gate.experts_per_rank,
                prefetch_slots=gate.prefetch_slots,
                weight_numel=gate.weight_numel,
                block_num=block_num,
                block_threads=gate.block_threads,
            )
            stream = torch.cuda.current_stream(device)
            fast_args = (
                local_sel.data_ptr(),
                gate.peer_home_weight_ptrs.data_ptr(),
                gate.prefetched_weights.data_ptr(),
            )
            fast_compiled = flyc.compile(
                fast_jit, *(fx.Int64(p) for p in fast_args), stream
            )
            fast_raw = fast_args + (stream,)

            def prefetch_fast():
                fast_compiled(*fast_raw)

            # ---- correctness ------------------------------------------
            # A prefetch is a pure copy, so the tuned version must be
            # bit-identical, not merely close.
            gate.prefetched_weights.zero_()
            torch.cuda.synchronize(device)
            dist.barrier()
            prefetch_one()
            torch.cuda.synchronize(device)
            dist.barrier()
            ref_out = gate.prefetched_weights.clone()

            gate.prefetched_weights.zero_()
            torch.cuda.synchronize(device)
            dist.barrier()
            prefetch_fast()
            torch.cuda.synchronize(device)
            dist.barrier()
            same = torch.equal(ref_out, gate.prefetched_weights)

            # Independent ground truth for the slots this rank owns: the copy
            # must reproduce our own home weights exactly.  Guards against both
            # kernels sharing a bug, which bit-equality alone cannot catch.
            local_checked = 0
            local_ok = True
            sel_cpu = local_sel.cpu()
            for s in range(gate.prefetch_slots):
                e = int(sel_cpu[s].item())
                if e < 0 or e // gate.experts_per_rank != rank:
                    continue
                want = gate.home_weights[e % gate.experts_per_rank]
                got = gate.prefetched_weights[s].view(want.shape)
                local_ok = local_ok and torch.equal(want, got)
                local_checked += 1
            nonzero = ref_out.abs().max().item() > 0
            if rank == 0:
                print(
                    f"[check] {routing}/bn{block_num}: fast==ref {same}, "
                    f"ref nonzero {nonzero}, local-owned slots checked "
                    f"{local_checked} ok={local_ok}",
                    flush=True,
                )
            if slots > 0 and not nonzero:
                raise AssertionError(
                    f"rank{rank}: reference prefetch wrote nothing for "
                    f"{slots} slots"
                )
            if not same:
                bad = (ref_out != gate.prefetched_weights).sum().item()
                raise AssertionError(
                    f"rank{rank} {routing}/bn{block_num}: fast prefetch "
                    f"differs in {bad} elements"
                )
            if not local_ok:
                raise AssertionError(
                    f"rank{rank}: prefetched local-owned slot does not match "
                    f"home_weights"
                )
            dist.barrier()

            mean_us, max_us = time_gpu_op(prefetch_one, dist.group.WORLD)
            results[f"prefetch/{routing}/bn{block_num}"] = (mean_us, max_us)
            torch.cuda.synchronize(device)
            dist.barrier()
            mean_us, max_us = time_gpu_op(prefetch_fast, dist.group.WORLD)
            results[f"prefetch_fast/{routing}/bn{block_num}"] = (mean_us, max_us)
            torch.cuda.synchronize(device)
            dist.barrier()

            ep.close()
            del ep
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            dist.barrier()

    # Control: the streaming read probe, in this same process, over the same
    # order of bytes.  If prefetch beats this, the probe's ceiling is wrong and
    # the "no multi-stage pipeline" conclusion has to be revisited.
    from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

    slab_mib = 128
    slab_dwords = slab_mib * 1024 * 1024 // 4
    slab = mori_shmem_create_tensor((slab_dwords,), torch.int32)
    slab.fill_(0x5A5A5A5A)
    torch.cuda.synchronize(device)
    ms.shmem_barrier_all()
    remote_peers = [p for p in range(world_size) if p != rank]
    remote_ptrs = torch.tensor(
        [ms.shmem_ptr_p2p(slab.data_ptr(), rank, p) for p in remote_peers],
        dtype=torch.int64,
        device=device,
    )
    sink = torch.zeros(2048, dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream(device)
    for bn in (256, 1024):
        depth = 4
        if slab_dwords % probe_stride_dwords(
            block_num=bn, block_threads=256, depth=depth
        ):
            continue
        jit = make_link_probe_jit(
            covered_dwords=slab_dwords,
            num_peers=len(remote_peers),
            depth=depth,
            block_num=bn,
            direction="read",
        )
        compiled = flyc.compile(
            jit,
            fx.Int64(remote_ptrs.data_ptr()),
            fx.Int64(sink.data_ptr()),
            stream,
        )
        raw = (remote_ptrs.data_ptr(), sink.data_ptr(), stream)
        mean_us, max_us = time_gpu_op(lambda: compiled(*raw), dist.group.WORLD)
        results[f"probe_read/bn{bn}"] = (mean_us, max_us)
        stats[f"probe_read/bn{bn}"] = {
            "remote_bytes": slab_dwords * 4 * len(remote_peers)
        }
        torch.cuda.synchronize(device)
        dist.barrier()

    if rank == 0:
        print()
        print(f"shape S={S} H={H} I={I} K={K} E={E} R={R} B={B}")
        print(f"one weight matrix per expert = {H * I * 2 / 1e6:.1f} MB")
        print(f"warmup={WARMUP} iters={ITERS} (eager, cross-rank mean)")
        print("upstream 8xB300 prefetch (one matrix): 161.5 us")
        print()
        print(
            f"{'measurement':<32}{'mean us':>9}{'max us':>9}"
            f"{'busiest MB':>12}{'busiest GB/s':>14}"
        )
        for key in sorted(results):
            mean_us, max_us = results[key]
            if key.startswith("probe_read"):
                rb = stats[key]["remote_bytes"]
            else:
                rb = stats[key.split("/")[1]]["max_rank_remote_bytes"]
            # Busiest rank's bytes over the slowest rank's time: the honest
            # per-GPU number when the work is unbalanced.
            gb = rb / max_us / 1e3 if rb else float("nan")
            print(
                f"{key:<32}{mean_us:>9.1f}{max_us:>9.1f}"
                f"{rb / 1e6:>12.1f}{gb:>14.1f}"
            )
        print()
        print("MOONEP_PREFETCH_JSON " + json.dumps(
            {
                "timing": {
                    k: {"mean_us": v[0], "max_us": v[1]} for k, v in results.items()
                },
                "stats": stats,
            }
        ))

    dist.barrier()
    ms.shmem_barrier_all()
    mori_shmem_free_tensor(slab)
    ms.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end MoonEP forward: reference EP vs the EP with tuned stages wired in.

Every stage so far has been measured in isolation.  This runs the whole
``forward`` -- planning, dispatch, epilogue, expert MLP, combine, weight
prefetch -- so the per-stage wins can be checked against what actually comes out
the other end, including the costs isolation hides (extra barriers for the push
path's reverse map, and the plan-rebuild each step).

The two EPs are built one at a time, not side by side: at NvS~78k each one owns
~10 GB of symmetric buffers.  Inputs and weights are deterministic, so both see
identical data and their outputs are directly comparable.

Correctness: the push path folds duplicates in the prologue, which rounds to
bf16 once more than summing all K in fp32 inside combine, so the two forwards
agree to a tolerance rather than bit-for-bit.

Run under torchrun with one process per GPU::

    torchrun --standalone --nproc-per-node=8 op_tests/benchmark_moonep_e2e.py
"""

from __future__ import annotations

import json
import os

import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.moonep import MoonEPPlanConfig, build_reference_plan
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP
from aiter.ops.flydsl.moonep_ep_fast import MoonEPBF16FastEP
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

WARMUP = int(os.environ.get("MOONEP_BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("MOONEP_BENCH_ITERS", "10"))
BLOCK_NUM = int(os.environ.get("MOONEP_BLOCK_NUM", "1024"))
ROUTING = os.environ.get("MOONEP_ROUTING", "uniform")
# "prebuilt" reuses one plan across iterations (planning excluded from the
# loop); "fresh" rebuilds it every step, which is what serving does.
PLAN_MODE = os.environ.get("MOONEP_PLAN_MODE", "prebuilt,fresh").split(",")


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
    if ROUTING == "uniform":
        gen = torch.Generator(device="cpu").manual_seed(20260810 + rank)
        topk = (
            torch.rand(S, E, generator=gen).topk(K, dim=1).indices.to(torch.int32)
        ).to(device)
        local_tpe = torch.bincount(
            topk.reshape(-1).to(torch.int64), minlength=E
        ).to(torch.int32)
    gathered = [torch.empty_like(local_tpe) for _ in range(R)]
    dist.all_gather(gathered, local_tpe)
    tpe_all = torch.stack(gathered)
    base_plan = build_reference_plan(config, topk, tpe_all)
    torch.cuda.synchronize(device)
    dist.barrier()

    results = {}
    outputs = {}

    for tag in ("reference", "fast"):
        if tag == "reference":
            ep = MoonEPBF16ReferenceEP(
                config,
                H,
                I,
                dispatch_block_num=BLOCK_NUM,
                prefetch_block_num=BLOCK_NUM,
            )
        else:
            ep = MoonEPBF16FastEP(
                config,
                H,
                I,
                dispatch_block_num=BLOCK_NUM,
                prefetch_block_num=BLOCK_NUM,
                combine_block_num=BLOCK_NUM,
            )
        hg, hu, hd = _identity_home_weights(device)
        ep.load_home_weights(hg, hu, hd)
        del hg, hu, hd
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        dist.barrier()

        def fwd_prebuilt():
            ep.forward(hidden, route_weights, plan=base_plan)

        def fwd_fresh():
            ep.forward(
                hidden,
                route_weights,
                topk_experts=topk,
                tokens_per_expert=tpe_all,
            )

        # Prime: the ops compile on their first call without executing, so a
        # single untimed pass is not enough to leave a valid result behind.
        fwd_prebuilt()
        fwd_prebuilt()
        torch.cuda.synchronize(device)
        dist.barrier()
        out, _, _ = ep.forward(hidden, route_weights, plan=base_plan)
        torch.cuda.synchronize(device)
        dist.barrier()
        outputs[tag] = out.float().clone()
        scale = outputs[tag].abs().max().item()
        if rank == 0:
            print(f"[check] {tag}: |out|={scale:.4e}", flush=True)
        if scale <= 0:
            raise AssertionError(f"{tag}: forward produced an all-zero output")

        for mode in PLAN_MODE:
            fn = fwd_prebuilt if mode == "prebuilt" else fwd_fresh
            if mode == "fresh" and tag == "reference":
                # The reference planner is the 802 ms CPU path; timing 10 of
                # those would dominate the run and tells us nothing new.
                if rank == 0:
                    print(
                        "[skip] reference/fresh: CPU planner is ~802 ms/step",
                        flush=True,
                    )
                continue
            fn()
            torch.cuda.synchronize(device)
            dist.barrier()
            mean_us, max_us = time_gpu_op(fn, dist.group.WORLD)
            results[f"{tag}/{mode}"] = (mean_us, max_us)
            torch.cuda.synchronize(device)
            dist.barrier()

        ep.close()
        del ep
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        dist.barrier()

    if rank == 0:
        ref = outputs["reference"]
        fast = outputs["fast"]
        err = (ref - fast).abs().max().item()
        scale = ref.abs().max().item()
        print()
        print(f"EP={R} S={S} H={H} K={K} E={E} Hp={I} block_num={BLOCK_NUM}")
        print(f"warmup={WARMUP} iters={ITERS} routing={ROUTING} (eager)")
        print(
            f"[check] fast vs reference: max|err|={err:.3e} "
            f"max|ref|={scale:.3e} rel={err / max(scale, 1e-30):.2e}"
        )
        print()
        print(f"{'forward':<28}{'mean us':>10}{'max us':>10}")
        for key in sorted(results):
            mean_us, max_us = results[key]
            print(f"{key:<28}{mean_us:>10.1f}{max_us:>10.1f}")
        if "reference/prebuilt" in results and "fast/prebuilt" in results:
            r = results["reference/prebuilt"][0]
            f = results["fast/prebuilt"][0]
            print(f"\nspeedup (prebuilt plan): {r / f:.2f}x")
        print("\nMOONEP_E2E_JSON " + json.dumps(
            {
                "timing": {
                    k: {"mean_us": v[0], "max_us": v[1]} for k, v in results.items()
                },
                "max_abs_err": err,
                "max_abs_ref": scale,
            }
        ))
        if err > 5e-2 * max(scale, 1e-30):
            raise AssertionError(
                f"fast forward differs from reference by rel "
                f"{err / max(scale, 1e-30):.3e}"
            )

    dist.barrier()
    ms.shmem_barrier_all()
    ms.shmem_finalize()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

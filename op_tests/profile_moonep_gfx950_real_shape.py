# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Stage and GPU-trace profile for the eight-rank MoonEP 8k forward."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext

import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.moonep import (
    MoonEPPlanConfig,
    build_reference_plan,
    hipblaslt_moonep_mlp_reference,
)
from aiter.ops.flydsl.moonep_ep import MoonEPBF16ReferenceEP
from op_tests.test_moonep_gfx950_real_shape import (
    B,
    E,
    H,
    I,
    K,
    R,
    S,
    TOKEN_PADDING,
    _expected_output,
    _identity_home_weights,
    _share_shmem_unique_id,
)


STAGES = (
    "dispatch_zero_copy",
    "prefetch_gate",
    "prefetch_up",
    "prefetch_down",
    "expert_mlp",
    "route_weight",
    "expert_to_shard",
    "combine_comm",
    "output_copy",
)


def _inputs(rank: int, device: torch.device):
    token = torch.arange(S, dtype=torch.int64, device=device)[:, None]
    k_idx = torch.arange(K, dtype=torch.int64, device=device)[None, :]
    topk = ((token * K + k_idx + rank * 13) % B).to(torch.int32)
    tokens_per_expert = torch.bincount(
        topk.reshape(-1).to(torch.int64), minlength=E
    ).to(torch.int32)
    raw_weights = (
        k_idx.to(torch.float32)
        + 1.0
        + (token % 7).to(torch.float32) * 0.125
    )
    route_weights = (
        raw_weights / raw_weights.sum(dim=1, keepdim=True)
    ).contiguous()
    generator = torch.Generator(device=device).manual_seed(20260804 + rank)
    hidden = torch.randn(
        S, H, dtype=torch.bfloat16, device=device, generator=generator
    )
    return hidden, route_weights, topk, tokens_per_expert


def _record(name: str, enabled: bool):
    if enabled:
        return torch.autograd.profiler.record_function(f"moonep/{name}")
    return nullcontext()


def _run_pipeline(
    ep: MoonEPBF16ReferenceEP,
    plan,
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    collect_events: bool,
    annotate: bool = False,
):
    events = {}

    def run_stage(name, fn):
        if collect_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        with _record(name, annotate):
            result = fn()
        if collect_events:
            end.record()
            events[name] = (start, end)
        return result

    dispatched, dispatched_weights, _, _ = run_stage(
        "dispatch_zero_copy",
        lambda: ep.dispatch(
            hidden, route_weights, plan=plan, zero_copy=True
        ),
    )
    prefetched_gate = run_stage(
        "prefetch_gate", lambda: ep.gate_op.prefetch(plan.experts_to_copy)
    )
    prefetched_up = run_stage(
        "prefetch_up", lambda: ep.up_op.prefetch(plan.experts_to_copy)
    )
    prefetched_down = run_stage(
        "prefetch_down", lambda: ep.down_op.prefetch(plan.experts_to_copy)
    )

    def expert_mlp():
        return hipblaslt_moonep_mlp_reference(
            dispatched,
            ep.gate_op.home_weights,
            ep.up_op.home_weights,
            ep.down_op.home_weights,
            prefetched_gate,
            prefetched_up,
            prefetched_down,
            plan.cu_seqlens,
            plan.group_expert_ids,
            rank=ep.config.rank,
            experts_per_rank=ep.config.experts_per_rank,
            num_experts=ep.config.num_experts,
            output=ep.dispatch_op.expert_output,
        )

    expert_output = run_stage("expert_mlp", expert_mlp)
    run_stage(
        "route_weight",
        lambda: expert_output.mul_(dispatched_weights[:, None]),
    )
    run_stage(
        "expert_to_shard",
        lambda: ep.dispatch_op.recv_hidden.copy_(expert_output),
    )
    combined, gathered = run_stage(
        "combine_comm", lambda: ep.dispatch_op.combine(plan)
    )
    output, output_weights = run_stage(
        "output_copy", lambda: (combined.clone(), gathered.clone())
    )
    return output, output_weights, events


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == R
    warmup = int(os.environ.get("MOONEP_PROFILE_WARMUP", "3"))
    iterations = int(os.environ.get("MOONEP_PROFILE_ITERS", "10"))
    trace_enabled = os.environ.get("MOONEP_PROFILE_TRACE", "1") == "1"
    trace_dir = os.environ.get(
        "MOONEP_PROFILE_TRACE_DIR",
        "/it-share/jiaolyu/moonep_gfx950_20260803/profile_8k_traces",
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl")
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        _share_shmem_unique_id(rank),
    )
    assert status == 0

    ep = None
    try:
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=S,
            top_k=K,
            num_experts=E,
            prefetch_slots=B,
            token_padding=TOKEN_PADDING,
        )
        ep = MoonEPBF16ReferenceEP(
            config,
            H,
            I,
            dispatch_block_num=128,
            prefetch_block_num=128,
        )
        hidden, route_weights, topk, local_tpe = _inputs(rank, device)
        home_gate, home_up, home_down = _identity_home_weights(device)
        ep.load_home_weights(home_gate, home_up, home_down)
        torch.cuda.synchronize(device)
        del home_gate, home_up, home_down
        torch.cuda.empty_cache()

        gathered_tpe = [torch.empty_like(local_tpe) for _ in range(R)]
        for _ in range(3):
            dist.all_gather(gathered_tpe, local_tpe)
        torch.cuda.synchronize(device)
        dist.barrier()
        allgather_samples = []
        for _ in range(10):
            begin = time.perf_counter()
            dist.all_gather(gathered_tpe, local_tpe)
            torch.cuda.synchronize(device)
            allgather_samples.append((time.perf_counter() - begin) * 1e3)
        tpe_allgather_ms = sum(allgather_samples) / len(allgather_samples)
        all_tpe = torch.stack(gathered_tpe)

        build_reference_plan(config, topk, all_tpe)
        torch.cuda.synchronize(device)
        dist.barrier()
        planning_samples = []
        for _ in range(3):
            begin = time.perf_counter()
            plan = build_reference_plan(config, topk, all_tpe)
            torch.cuda.synchronize(device)
            planning_samples.append((time.perf_counter() - begin) * 1e3)
        planning_ms = sum(planning_samples) / len(planning_samples)

        checked_output, checked_weights, _ = _run_pipeline(
            ep, plan, hidden, route_weights, collect_events=False
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            checked_weights, route_weights, rtol=0, atol=0
        )
        expected_first = _expected_output(hidden, route_weights)
        torch.testing.assert_close(
            checked_output[:, :I], expected_first, rtol=0, atol=0
        )
        assert int(torch.count_nonzero(checked_output[:, I:]).item()) == 0
        local_dynamic_slots = torch.tensor(
            int((plan.experts_to_copy[rank] >= 0).sum().item()),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_dynamic_slots, op=dist.ReduceOp.SUM)
        assert int(local_dynamic_slots.item()) > 0
        if rank == 0:
            print(
                "MOONEP_PROFILE_ACCURACY_PASS "
                "route_weights_rtol=0_atol=0 "
                "hidden_rtol=0_atol=0 "
                f"dynamic_slots={int(local_dynamic_slots.item())}",
                flush=True,
            )
        del checked_output, checked_weights, expected_first

        for _ in range(warmup):
            _run_pipeline(
                ep, plan, hidden, route_weights, collect_events=False
            )
        torch.cuda.synchronize(device)
        dist.barrier()

        local_samples = []
        local_totals = []
        for _ in range(iterations):
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            _, _, events = _run_pipeline(
                ep, plan, hidden, route_weights, collect_events=True
            )
            total_end.record()
            total_end.synchronize()
            local_samples.append(
                [events[name][0].elapsed_time(events[name][1]) for name in STAGES]
            )
            local_totals.append(total_start.elapsed_time(total_end))

        local_stage_mean = torch.tensor(
            local_samples, dtype=torch.float64, device=device
        ).mean(dim=0)
        local_total_mean = torch.tensor(
            local_totals, dtype=torch.float64, device=device
        ).mean()
        stage_max = local_stage_mean.clone()
        stage_mean = local_stage_mean.clone()
        total_max = local_total_mean.clone()
        total_mean = local_total_mean.clone()
        dist.all_reduce(stage_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(stage_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_mean, op=dist.ReduceOp.SUM)
        stage_mean /= R
        total_mean /= R

        host_times = torch.tensor(
            [tpe_allgather_ms, planning_ms], dtype=torch.float64, device=device
        )
        dist.all_reduce(host_times, op=dist.ReduceOp.MAX)

        if trace_enabled:
            os.makedirs(trace_dir, exist_ok=True)
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                _run_pipeline(
                    ep,
                    plan,
                    hidden,
                    route_weights,
                    collect_events=False,
                    annotate=True,
                )
                torch.cuda.synchronize(device)
            profiler.export_chrome_trace(
                os.path.join(trace_dir, f"rank{rank}.json")
            )
            dist.barrier()

        if rank == 0:
            result = {
                "shape": {
                    "S": S,
                    "H": H,
                    "K": K,
                    "E": E,
                    "I": I,
                    "R": R,
                    "B": B,
                    "token_padding": TOKEN_PADDING,
                },
                "warmup": warmup,
                "iterations": iterations,
                "host_max_ms": {
                    "tokens_per_expert_allgather": host_times[0].item(),
                    "reference_planning": host_times[1].item(),
                },
                "stage_rank_max_ms": {
                    name: value
                    for name, value in zip(STAGES, stage_max.cpu().tolist())
                },
                "stage_rank_mean_ms": {
                    name: value
                    for name, value in zip(STAGES, stage_mean.cpu().tolist())
                },
                "pipeline_rank_max_ms": total_max.item(),
                "pipeline_rank_mean_ms": total_mean.item(),
                "trace_dir": trace_dir if trace_enabled else None,
            }
            print("MOONEP_PROFILE_JSON " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        if ep is not None:
            ep.close()
        ms.shmem_barrier_all()
        ms.shmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

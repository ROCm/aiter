# SPDX-License-Identifier: MIT
"""Probe exact-M Stage2 accumulation directly into the shared-expert result.

This is intentionally a narrow TP8 experiment for M=1/2.  It compares:

  baseline:  zero(output) -> atomic Stage2 -> add(shared)
  candidate: copy(shared, output) -> atomic Stage2

and repeats the comparison with the existing one-stage custom AllReduce
appended.  The copy in the candidate is a conservative stand-in for the
production contract where the shared-expert GEMM writes ``output`` directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from types import SimpleNamespace

import flydsl.expr as fx
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from aiter import dtypes
from aiter.dist.parallel_state import get_tp_group
from aiter.fused_moe import stage2_uses_route_reduce
from aiter.ops.flydsl.comm_fused_moe_host import _register
from aiter.ops.flydsl.kernels.comm_fused_moe import small_m_allreduce
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled
from op_tests.multigpu_tests import moe_tp_stage2_test_utils as fixtures
from op_tests.multigpu_tests.test_flydsl_comm_fused_full_tp8_perf import (
    resolve_ordinary_stage2,
)


TP = 8
H = 7168


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--route", choices=("uniform", "skew"), default="uniform")
    return parser.parse_args()


def make_shared(m: int, rank: int, device: torch.device) -> torch.Tensor:
    token = (
        torch.arange(m, device=device, dtype=torch.float32)
        .remainder(7)
        .mul_(1.0 / 32.0)
        .view(-1, 1)
    )
    column = (
        torch.arange(H, device=device, dtype=torch.float32)
        .remainder(17)
        .mul_(1.0 / 128.0)
        .view(1, -1)
    )
    return (token + column + float(rank + 1) / 16.0).to(torch.bfloat16)


def rank_max(local_us: float, world: int, group) -> float:
    local = torch.tensor([local_us], dtype=torch.float64, device="cuda")
    values = torch.empty(world, dtype=torch.float64, device="cuda")
    dist.all_gather_into_tensor(values, local, group=group)
    return float(values.max().item())


def measure(graph, *, rounds: int, iterations: int, world: int, group):
    samples = []
    for _ in range(rounds):
        fixtures.barrier(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(
            rank_max(start.elapsed_time(end) * 1000.0 / iterations, world, group)
        )
    return {
        "median_rank_max_us": statistics.median(samples),
        "min_rank_max_us": min(samples),
        "max_rank_max_us": max(samples),
        "samples_us": samples,
    }


def capture(body, group):
    fixtures.barrier(group)
    body()
    fixtures.barrier(group)
    graph = torch.cuda.CUDAGraph()
    with get_tp_group().graph_capture() as capture_context:
        with torch.cuda.graph(graph, stream=capture_context.stream):
            body()
    for _ in range(3):
        graph.replay()
    fixtures.barrier(group)
    return graph


def error(actual: torch.Tensor, expected: torch.Tensor, group):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = actual_f32 - expected_f32
    values = torch.stack(
        (
            diff.abs().max(),
            diff.norm() / expected_f32.norm().clamp_min(1.0e-12),
        )
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    return {"max_abs": float(values[0]), "rel_l2": float(values[1])}


def main() -> None:
    cli = parse_args()
    rank, world, _local_rank, device, group = fixtures.setup_distributed(TP)
    shared_w2 = None
    shared_w2_scale = None
    keepalive = []
    results = []

    for m in (1, 2):
        args = SimpleNamespace(
            tokens=m,
            target_tokens=m,
            logical_tokens=m,
            tp_size=TP,
            hidden=H,
            inter_dim=384,
            experts=384,
            topk=6,
            tile_m=32,
            tile_n=256,
            tile_k=128,
            route=cli.route,
            seed=20260827,
            low_memory_w2_quant=True,
        )
        metadata, kernel_name, requires_zero = resolve_ordinary_stage2(args)
        if not requires_zero or stage2_uses_route_reduce(metadata.stage2):
            raise RuntimeError(
                "shared-accumulator probe requires an atomic Stage2 kernel; "
                f"selected {kernel_name!r} for M={m}"
            )
        args.tile_m = int(metadata.block_m)
        case = fixtures.make_stage2_case(
            args,
            rank,
            device,
            shared_w2=shared_w2,
            shared_w2_scale=shared_w2_scale,
            accumulate=True,
        )
        if shared_w2 is None:
            shared_w2 = case.w2
            shared_w2_scale = case.w2_scale

        shared = make_shared(m, rank, device)
        baseline_partial = torch.empty_like(case.partial_out)
        candidate_partial = torch.empty_like(case.partial_out)
        flydsl_partial = symm_mem.empty(
            tuple(case.partial_out.shape),
            dtype=torch.bfloat16,
            device=device,
        )
        flydsl_state = symm_mem.empty(
            (small_m_allreduce.state_bytes(m, H),),
            dtype=torch.uint8,
            device=device,
        )
        flydsl_state.zero_()
        flydsl_output = torch.empty_like(case.partial_out)
        flydsl_comm, flydsl_windows, flydsl_bases = _register(
            get_tp_group(), rank, TP, (flydsl_partial, flydsl_state)
        )
        keepalive.append((flydsl_comm, flydsl_windows))
        flydsl_partial_base, flydsl_state_base = flydsl_bases
        flydsl_ar = small_m_allreduce.compile_bf16_one_stage(m, H)
        w2_scale = (
            case.w2_scale.view(dtypes.fp8_e8m0)
            if case.w2_scale.element_size() == 1
            else case.w2_scale
        )

        def stage2(output: torch.Tensor):
            metadata.stage2(
                case.inter_states,
                None,
                case.w2,
                case.sorted_token_ids,
                case.sorted_expert_ids,
                case.num_valid_ids,
                output,
                case.topk,
                w2_scale=w2_scale,
                a2_scale=case.a2_scale,
                block_m=int(metadata.block_m),
                sorted_weights=case.sorted_weights,
            )

        def baseline_local():
            baseline_partial.zero_()
            stage2(baseline_partial)
            baseline_partial.add_(shared)
            return baseline_partial

        def candidate_local():
            candidate_partial.copy_(shared)
            stage2(candidate_partial)
            return candidate_partial

        def flydsl_local():
            flydsl_partial.copy_(shared)
            stage2(flydsl_partial)
            return flydsl_partial

        baseline_local()
        candidate_local()
        flydsl_local()
        torch.cuda.synchronize(device)
        local_error = error(candidate_partial, baseline_partial, group)
        flydsl_local_error = error(flydsl_partial, baseline_partial, group)
        if (
            not torch.isfinite(candidate_partial).all()
            or local_error["max_abs"] > 1.0
            or local_error["rel_l2"] > 0.05
            or not torch.isfinite(flydsl_partial).all()
            or flydsl_local_error["max_abs"] > 1.0
            or flydsl_local_error["rel_l2"] > 0.05
        ):
            raise AssertionError(
                f"local shared accumulation failed for M={m}: "
                f"custom={local_error}, flydsl={flydsl_local_error}"
            )

        def baseline_e2e():
            partial = baseline_local()
            return get_tp_group().all_reduce(partial, ca_fp8_quant=False)

        def candidate_e2e():
            partial = candidate_local()
            return get_tp_group().all_reduce(partial, ca_fp8_quant=False)

        def flydsl_e2e():
            flydsl_local()
            stream = torch.cuda.current_stream(device)
            _run_compiled(
                flydsl_ar,
                (
                    fx.Int64(flydsl_partial_base),
                    ptr_arg(flydsl_state),
                    fx.Int64(flydsl_state_base),
                    ptr_arg(flydsl_output),
                    rank,
                    stream,
                ),
            )
            return flydsl_output

        baseline_output = baseline_e2e()
        candidate_output = candidate_e2e()
        flydsl_result = flydsl_e2e()
        torch.cuda.synchronize(device)
        e2e_error = error(candidate_output, baseline_output, group)
        flydsl_e2e_error = error(flydsl_result, baseline_output, group)
        if (
            not torch.isfinite(candidate_output).all()
            or e2e_error["max_abs"] > 1.0
            or e2e_error["rel_l2"] > 0.05
            or not torch.isfinite(flydsl_result).all()
            or flydsl_e2e_error["max_abs"] > 1.0
            or flydsl_e2e_error["rel_l2"] > 0.05
        ):
            raise AssertionError(
                f"end-to-end shared accumulation failed for M={m}: "
                f"custom={e2e_error}, flydsl={flydsl_e2e_error}"
            )

        baseline_local_graph = capture(baseline_local, group)
        candidate_local_graph = capture(candidate_local, group)
        flydsl_local_graph = capture(flydsl_local, group)
        baseline_e2e_graph = capture(baseline_e2e, group)
        candidate_e2e_graph = capture(candidate_e2e, group)
        flydsl_e2e_graph = capture(flydsl_e2e, group)

        timings = {
            "baseline_local": measure(
                baseline_local_graph,
                rounds=cli.rounds,
                iterations=cli.iterations,
                world=world,
                group=group,
            ),
            "candidate_local": measure(
                candidate_local_graph,
                rounds=cli.rounds,
                iterations=cli.iterations,
                world=world,
                group=group,
            ),
            "flydsl_local": measure(
                flydsl_local_graph,
                rounds=cli.rounds,
                iterations=cli.iterations,
                world=world,
                group=group,
            ),
            "baseline_e2e": measure(
                baseline_e2e_graph,
                rounds=cli.rounds,
                iterations=cli.iterations,
                world=world,
                group=group,
            ),
            "candidate_e2e": measure(
                candidate_e2e_graph,
                rounds=cli.rounds,
                iterations=cli.iterations,
                world=world,
                group=group,
            ),
            "flydsl_e2e": measure(
                flydsl_e2e_graph,
                rounds=cli.rounds,
                iterations=cli.iterations,
                world=world,
                group=group,
            ),
        }
        result = {
            "m": m,
            "route": cli.route,
            "kernel": kernel_name,
            "expert_blocks": int(case.sorted_expert_ids.numel()),
            "local_error": local_error,
            "flydsl_local_error": flydsl_local_error,
            "e2e_error": e2e_error,
            "flydsl_e2e_error": flydsl_e2e_error,
            "timings": timings,
        }
        results.append(result)
        if rank == 0:
            print(
                "SHARED_ACCUM_PROBE_RESULT " + json.dumps(result, sort_keys=True),
                flush=True,
            )

        del baseline_output, candidate_output, flydsl_result
        del baseline_partial, candidate_partial, flydsl_partial
        del flydsl_state, flydsl_output, shared, case
        torch.cuda.empty_cache()
        fixtures.barrier(group)

    if rank == 0:
        print(
            "SHARED_ACCUM_PROBE_SUMMARY " + json.dumps(results, sort_keys=True),
            flush=True,
        )
    keepalive.clear()
    fixtures.cleanup_distributed(rank)


if __name__ == "__main__":
    main()

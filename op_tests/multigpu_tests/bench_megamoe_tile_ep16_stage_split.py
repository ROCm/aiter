# SPDX-License-Identifier: MIT
"""Stage-split benchmark for the production EP16 two-kernel candidate.

This is deliberately separate from the communication-only probe.  It builds
the normal :class:`MegaMoETileA4W4` operator with the real A4W4 weights and
executes the exact production Stage-1 and Stage-2 launchers.  The only
difference from the public ``forward`` method is that two HIP event pairs are
inserted around the private launcher calls so their full kernel durations are
visible independently::

    generation += 1
    event(stage1); _launch_stage1(...); event(stage1_end)
    event(stage2); _launch_stage2(...); event(stage2_end)

The Gloo barrier remains outside every measured interval.  Constructor/JIT,
the public-forward prime, protocol inspection, output comparisons, and host
copies are also outside timing.  This benchmark does not attempt to label any
sub-region as communication-only: Stage-1 still contains quant/GMM1 and
Stage-2 still contains GMM2/direct-LSA-combine.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.megamoe_tile import MegaMoETileA4W4
from aiter.ops.flydsl.kernels.megamoe_tile.markers import profiler_pause, profiler_resume, roctx_range
from aiter.ops.flydsl.kernels.megamoe_tile.stage1_abi import validate_public_stage1_contract
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    HipStageTimer,
    IterationTiming,
    SharedInputs,
    _comparison_metrics,
    _global_max_timing,
    _sample_stats,
    _setup_dist,
    _shared_inputs,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    PathResult,
    TwoKernelCandidatePath,
    _debug_iteration_output_matrix,
    _validate_direct_tile_debug_snapshot,
    _validate_direct_tile_operator,
    _validate_rank_balanced_routes,
)


STAGE1_FIELD = "stage1_full_kernel"
STAGE2_FIELD = "stage2_full_kernel"
STAGE_SUM_FIELD = "stage1_plus_stage2"
TIMING_FIELDS = (STAGE1_FIELD, STAGE2_FIELD, STAGE_SUM_FIELD)


class FullCandidateStageSplitPath(TwoKernelCandidatePath):
    """Production candidate with an event boundary between its two launches."""

    name = "ep16_full_candidate_stage_split"
    stage_names = (STAGE1_FIELD, STAGE2_FIELD)

    def _begin_generation(self):
        op = self.operator
        if getattr(op, "_closed", False):
            raise RuntimeError("MegaMoETileA4W4 is closed")
        run_tokens = validate_public_stage1_contract(
            self.shared.x,
            self.shared.route_weights,
            self.shared.topk_ids,
            hidden=op.model_dim,
            topk=op.topk,
            max_tokens=op.mtpr,
        )
        if run_tokens != op.mtpr:
            raise ValueError("stage-split benchmark requires exactly 128 tokens/rank")
        op._generation += 1
        return run_tokens, op._generation, op._flydsl_stream(None)

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        op = self.operator
        run_tokens, generation, stream = self._begin_generation()

        def launch_stage1() -> None:
            op._launch_stage1(
                self.shared.x,
                self.shared.route_weights,
                self.shared.topk_ids,
                run_tokens,
                generation,
                stream,
            )

        def launch_stage2() -> None:
            op._launch_stage2(run_tokens, generation, stream)

        timer.stage(STAGE1_FIELD, launch_stage1)
        timer.stage(STAGE2_FIELD, launch_stage2)
        return op._output[:run_tokens]


def _build_full_candidate(
    shared: SharedInputs,
    shape: BenchmarkShape,
    *,
    rank: int,
    device: torch.device,
) -> FullCandidateStageSplitPath:
    weights = shared.prepared_weights
    operator = MegaMoETileA4W4(
        rank=rank,
        world_size=shape.ep_size,
        model_dim=shape.hidden,
        inter_dim=shape.inter,
        experts=shape.experts,
        topk=shape.topk,
        quant="a4w4",
        w1=weights.w1,
        w1_scale=weights.w1_scale,
        w2=weights.w2,
        w2_scale=weights.w2_scale,
        max_tok_per_rank=shape.tokens,
        mega_scheme="hierarchical",
        swiglu_limit=0.0,
    )
    try:
        _validate_direct_tile_operator(operator)
        if type(operator) is not MegaMoETileA4W4:
            raise TypeError("stage-split benchmark must use production MegaMoETileA4W4")
    except Exception:
        operator.close()
        raise
    return FullCandidateStageSplitPath(operator, shared, shape, device)


def _add_stage_sum(sample: IterationTiming) -> None:
    sample.stage_us[STAGE_SUM_FIELD] = (
        sample.stage_us[STAGE1_FIELD] + sample.stage_us[STAGE2_FIELD]
    )


def _run_stage_split(
    path: FullCandidateStageSplitPath,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> PathResult:
    # The prime intentionally uses the inherited public forward path.  Besides
    # validating the public output, it runs the normal direct-tile protocol
    # snapshot checker before any private launcher is called by this harness.
    prime = path.prime_and_check().clone()
    torch.cuda.synchronize(device)
    debug_generation_outputs: list[tuple[int, torch.Tensor]] = [
        (int(path.operator._generation), prime.detach().cpu())
    ]
    dist.barrier()
    timer = HipStageTimer(device, path.stage_names)

    for _ in range(warmup):
        torch.cuda.synchronize(device)
        dist.barrier()
        timer.begin_iteration()
        path.run_iteration(timer)
        sample = timer.finish_iteration()
        _add_stage_sum(sample)

    local_samples: list[IterationTiming] = []
    rank_max_samples: list[IterationTiming] = []
    output = None
    profiler_resume()
    for iteration in range(iterations):
        # Device completion before the CPU barrier prevents a fast rank from
        # entering the next epoch while another rank still owns the old parity.
        # Both operations precede begin_iteration(), so neither is timed.
        torch.cuda.synchronize(device)
        dist.barrier()
        timer.begin_iteration()
        with roctx_range(f"MEGAMOE_EP16_FULL_STAGE_SPLIT_TIMED_{iteration}"):
            output = path.run_iteration(timer)
            local = timer.finish_iteration()
        _add_stage_sum(local)
        local_samples.append(local)
        # Max each field only after the complete local event sample is known.
        # STAGE_SUM_FIELD is reduced directly; it is not reconstructed by adding
        # Stage-1 and Stage-2 maxima that may have come from different ranks.
        rank_max_samples.append(_global_max_timing(local, TIMING_FIELDS))
        if len(debug_generation_outputs) < 4:
            debug_generation_outputs.append(
                (int(path.operator._generation), output.detach().cpu())
            )
    profiler_pause()

    if output is None:
        raise AssertionError("timed loop produced no output")

    # A completed timed epoch must retain the same protocol invariants as the
    # public-forward prime.  This D2H inspection is deliberately after timing.
    path._validate_output(output)
    timed_state = path.operator.debug_direct_tile_snapshot()
    _validate_direct_tile_debug_snapshot(timed_state)
    print(
        "MEGAMOE_FULL_STAGE_SPLIT_SNAPSHOT "
        + json.dumps(
            {
                "rank": dist.get_rank(),
                "phase": "timed",
                "state": timed_state,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return PathResult(
        prime,
        output.clone(),
        local_samples,
        rank_max_samples,
        tuple(debug_generation_outputs),
    )


def _timing_summary(result: PathResult, tail_iterations: int) -> dict[str, object]:
    tail = result.rank_max_samples[-tail_iterations:]

    def values(field: str) -> list[float]:
        if field == "gpu_e2e":
            return [sample.gpu_e2e_us for sample in tail]
        if field == "host_critical":
            return [sample.host_critical_us for sample in tail]
        return [sample.stage_us[field] for sample in tail]

    fields = (*TIMING_FIELDS, "gpu_e2e", "host_critical")
    return {
        "path": FullCandidateStageSplitPath.name,
        "tail_iterations": tail_iterations,
        "tail_rank_max_stats_us": {
            field: _sample_stats(values(field)) for field in fields
        },
        "rank_max_samples_us": [
            {
                STAGE1_FIELD: sample.stage_us[STAGE1_FIELD],
                STAGE2_FIELD: sample.stage_us[STAGE2_FIELD],
                STAGE_SUM_FIELD: sample.stage_us[STAGE_SUM_FIELD],
                "gpu_e2e": sample.gpu_e2e_us,
                "host_critical": sample.host_critical_us,
            }
            for sample in result.rank_max_samples
        ],
    }


def _global_max_scalar(value: float) -> float:
    tensor = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _shape() -> BenchmarkShape:
    shape = BenchmarkShape(
        tokens=128,
        hidden=7168,
        inter=3072,
        experts=896,
        topk=16,
        ep_size=16,
        gpus_per_node=8,
        activation="silu",
    )
    shape.validate()
    return shape


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--tail-iters", type=int, default=10)
    parser.add_argument("--rel-l2-threshold", type=float, default=1.0e-2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be >= 1")
    if not 1 <= args.tail_iters <= args.iters:
        raise ValueError("tail-iters must be in [1, iters]")
    if args.rel_l2_threshold <= 0.0:
        raise ValueError("rel-l2-threshold must be positive")

    shape = _shape()
    contract = {
        "shape": shape.__dict__,
        "operator": "aiter.ops.flydsl.kernels.megamoe_tile.MegaMoETileA4W4",
        "candidate": "production_full_two_kernel",
        "event_order": [
            "iteration_gpu_start",
            "stage1_start",
            "_launch_stage1",
            "stage1_end",
            "stage2_start",
            "_launch_stage2",
            "stage2_end",
            "iteration_gpu_end",
        ],
        "stage_fields": [STAGE1_FIELD, STAGE2_FIELD, STAGE_SUM_FIELD],
        "barrier": "gloo_before_each_iteration_outside_timing",
        "prime": "public_forward_plus_direct_tile_protocol_snapshot",
        "correctness": "prime_vs_timed_and_four_generation_pairwise_matrix",
        "statistic": "per_iteration_ep16_rank_max_then_tail_mean_p50_p95",
        "warmup": args.warmup,
        "iterations": args.iters,
        "tail_iterations": args.tail_iters,
    }
    if args.plan_only:
        print("MEGAMOE_EP16_FULL_STAGE_SPLIT_PLAN " + json.dumps(contract, sort_keys=True))
        return 0

    rank, world, _local_rank, device = _setup_dist(needs_mori=False)
    if world != shape.ep_size:
        raise ValueError(f"stage-split benchmark requires world=16, got {world}")
    profiler_pause()
    shared = _shared_inputs(shape, rank, world, device)
    _validate_rank_balanced_routes(shared, shape)
    path = _build_full_candidate(shared, shape, rank=rank, device=device)

    result = _run_stage_split(
        path,
        device,
        warmup=args.warmup,
        iterations=args.iters,
    )
    diagnostics = [
        _comparison_metrics(
            result.prime,
            result.timed,
            rank=rank,
            label="full_stage_split_prime_vs_timed",
        )
    ]
    diagnostics.extend(
        _debug_iteration_output_matrix(
            result,
            rank=rank,
            path_name=path.name,
        )
    )
    rank_max_rel_l2 = _global_max_scalar(float(diagnostics[0]["rel_l2"]))
    failed = rank_max_rel_l2 >= args.rel_l2_threshold

    gathered_diagnostics = [None] * world if rank == 0 else None
    dist.gather_object(diagnostics, gathered_diagnostics, dst=0)
    if rank == 0:
        print(
            "MEGAMOE_EP16_FULL_STAGE_SPLIT_BENCH "
            + json.dumps(
                {
                    **contract,
                    "timing": _timing_summary(result, args.tail_iters),
                    "prime_vs_timed_rank_max_rel_l2": rank_max_rel_l2,
                    "rel_l2_threshold": args.rel_l2_threshold,
                    "correctness_by_rank": gathered_diagnostics,
                    "kernel_symbols": {
                        "stage1": path.operator.stage1_kernel_name,
                        "stage2": path.operator.stage2_kernel_name,
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.barrier()
    path.close()
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

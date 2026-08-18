# SPDX-License-Identifier: MIT
"""Apples-to-apples unfused-small-op versus fused Stage1 benchmark.

Both paths start from the same BF16 activations and finish after GMM1, SiLU,
and A4 requant.  The unfused path uses the existing production operators:

    per_1x32_f4_quant -> MORI dispatch -> route/sort/scale prepare -> GMM1

GMM2, MORI combine, and Stage2 are excluded from the measured Stage1 event.
MORI combine is issued only after the end event and device synchronization to
close the dispatch lifecycle before the next iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
import torch.distributed as dist

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_comm_only import (
    FusedCommunicationProbePath,
    _lightweight_shared_inputs,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
    BenchmarkShape,
    HipStageTimer,
    IterationTiming,
    _prepare_local_a4w4,
    _run_local_h1,
    _sample_stats,
    _setup_dist,
)
from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    MoriBf16A4W4BaselinePath,
    _global_max_timing,
)
from op_tests.multigpu_tests.megamoe_tile_comm_probe_factory import (
    MegaMoETileA4W4CommProbe,
)


UNFUSED_FULL = "unfused_stage1_full_quant_dispatch_prepare_gmm1_silu_requant"
UNFUSED_QUANT = "unfused_bf16_to_a4"
UNFUSED_DISPATCH = "unfused_mori_dispatch"
UNFUSED_PREPARE = "unfused_route_sort_scale_prepare"
UNFUSED_GMM1 = "unfused_gmm1_silu_requant"
UNFUSED_FIELDS = (
    UNFUSED_FULL,
    UNFUSED_QUANT,
    UNFUSED_DISPATCH,
    UNFUSED_PREPARE,
    UNFUSED_GMM1,
)


class UnfusedMoriStage1Path(MoriBf16A4W4BaselinePath):
    """Existing small operators with one continuous Stage1 timing envelope."""

    name = "unfused_smallop_stage1"
    stage_names = UNFUSED_FIELDS
    full_stage_field = UNFUSED_FULL

    def __init__(self, *args, record_components: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        from aiter.ops.quant import dynamic_per_group_scaled_quant
        from aiter.utility import dtypes

        self._quant_op = dynamic_per_group_scaled_quant
        self.record_components = bool(record_components)
        self.stage_names = (
            UNFUSED_FIELDS if self.record_components else (UNFUSED_FULL,)
        )
        self._quant_q = torch.empty(
            (self.shape.tokens, self.shape.hidden // 2),
            dtype=dtypes.fp4x2,
            device=self.shared.x.device,
        )
        self._quant_scale = torch.empty(
            (self.shape.tokens, self.shape.hidden // 32),
            dtype=torch.uint8,
            device=self.shared.x.device,
        ).view(dtypes.fp8_e8m0)
        self._prepare_workspace: dict[str, torch.Tensor] | None = None
        self._context: dict[str, object] | None = None
        self._cleanup_partial: torch.Tensor | None = None
        self._cleanup_output: torch.Tensor | None = None
        self.prime_contract: dict[str, object] | None = None

    def _input_quant_stage(self):
        self._quant_op(
            self._quant_q,
            self.shared.x,
            self._quant_scale,
            32,
            shuffle_scale=False,
        )
        self._iteration_a_quant = self._quant_q
        self._iteration_a_scale = self._quant_scale
        return self._quant_q

    def _prepare_stage(self):
        if self._dispatch is None:
            raise RuntimeError("dispatch stage has not run")
        dispatched, weights, scales, ids, _ = self._dispatch
        self._context = _prepare_local_a4w4(
            dispatched[: self.valid_recv],
            scales[: self.valid_recv],
            ids[: self.valid_recv],
            weights[: self.valid_recv],
            self.shape,
            self.rank,
            validate_routes=self.validate_routes,
            output_workspace=self._prepare_workspace,
            expand_local_routes=self.expand_local_routes,
        )
        if self._prepare_workspace is None:
            self._prepare_workspace = {
                name: self._context[name]
                for name in ("inter_q", "inter_s", "hidden_dummy")
            }
        if self._cleanup_partial is None:
            self._cleanup_partial = torch.zeros(
                (dispatched.shape[0], self.shape.hidden),
                dtype=torch.bfloat16,
                device=dispatched.device,
            )
        return self._context

    def _gmm1_stage(self):
        if self._context is None:
            raise RuntimeError("local prepare stage has not run")
        _run_local_h1(self._context, self.shared.prepared_weights, self.shape)
        return self._context["inter_q"]

    def _run_components(self, timer: HipStageTimer | None):
        if timer is None or not self.record_components:
            self._input_quant_stage()
            self._dispatch_stage()
            self._prepare_stage()
            return self._gmm1_stage()
        timer.stage(UNFUSED_QUANT, self._input_quant_stage)
        timer.stage(UNFUSED_DISPATCH, self._dispatch_stage)
        timer.stage(UNFUSED_PREPARE, self._prepare_stage)
        return timer.stage(UNFUSED_GMM1, self._gmm1_stage)

    def run_iteration(self, timer: HipStageTimer):
        return timer.stage(UNFUSED_FULL, lambda: self._run_components(timer))

    def _cleanup_dispatch_lifecycle(self) -> None:
        if self._cleanup_partial is None:
            raise RuntimeError("cleanup buffer was not allocated during prime")
        self._cleanup_partial.zero_()
        result = self.op.combine(
            self._cleanup_partial,
            None,
            self.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=4,
        )
        self._cleanup_output = result[0] if isinstance(result, tuple) else result
        torch.cuda.synchronize(self._cleanup_partial.device)

    def after_iteration(self) -> None:
        # Called only after finish_iteration() recorded/synchronized the end
        # event, so this protocol cleanup is outside the Stage1 envelope.
        self._cleanup_dispatch_lifecycle()

    def prime_and_check(self):
        output = self._run_components(None)
        torch.cuda.synchronize(output.device)
        if self._dispatch is None or self._context is None:
            raise AssertionError("unfused Stage1 prime did not build its context")
        recv_count = int(self._dispatch[4].item())
        if recv_count != self.valid_recv:
            raise AssertionError(
                f"unfused recv_count={recv_count}, expected={self.valid_recv}"
            )
        valid_routes = int(self._context["route_count"])
        expected_routes = self.shape.tokens * self.shape.ep_size
        if valid_routes != expected_routes:
            raise AssertionError(
                f"unfused valid_routes={valid_routes}, expected={expected_routes}"
            )
        expected_topk = 2 if self.expand_local_routes else 1
        if int(self._context["gmm1_topk"]) != expected_topk:
            raise AssertionError("unfused GMM1 top-k multiplicity mismatch")
        q = self._context["inter_q"].contiguous().view(torch.uint8)
        s = self._context["inter_s"].contiguous().view(torch.uint8)
        self.prime_contract = {
            "recv_records": recv_count,
            "expanded_routes": valid_routes,
            "gmm1_topk": int(self._context["gmm1_topk"]),
            "h1_q_sha256": hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest(),
            "h1_scale_sha256": hashlib.sha256(
                s.cpu().numpy().tobytes()
            ).hexdigest(),
        }
        self._cleanup_dispatch_lifecycle()
        self.validate_routes = False
        return output


class FusedStage1Path(FusedCommunicationProbePath):
    """Production fused Stage1 exposed through the existing probe harness."""

    @property
    def full_stage_field(self) -> str:
        return self.stage_names[0]


def _run_path(path, device, *, warmup: int, iterations: int):
    path.prime_and_check()
    torch.cuda.synchronize(device)
    dist.barrier()
    timer = HipStageTimer(device, path.stage_names)

    for _ in range(warmup):
        torch.cuda.synchronize(device)
        dist.barrier()
        timer.begin_iteration()
        path.run_iteration(timer)
        timer.finish_iteration()
        cleanup = getattr(path, "after_iteration", None)
        if cleanup is not None:
            cleanup()

    local_samples: list[IterationTiming] = []
    for _ in range(iterations):
        torch.cuda.synchronize(device)
        dist.barrier()
        timer.begin_iteration()
        path.run_iteration(timer)
        local_samples.append(timer.finish_iteration())
        cleanup = getattr(path, "after_iteration", None)
        if cleanup is not None:
            cleanup()

    rank_max_samples = [
        _global_max_timing(sample, path.stage_names) for sample in local_samples
    ]
    return local_samples, rank_max_samples


def _run_interleaved(paths, device, *, warmup: int, iterations: int):
    """Alternate path order every round to reduce whole-run order drift."""

    for path in paths:
        path.prime_and_check()
        torch.cuda.synchronize(device)
        dist.barrier()
    timers = {
        path.name: HipStageTimer(device, path.stage_names) for path in paths
    }

    def one(path):
        torch.cuda.synchronize(device)
        dist.barrier()
        timer = timers[path.name]
        timer.begin_iteration()
        path.run_iteration(timer)
        sample = timer.finish_iteration()
        cleanup = getattr(path, "after_iteration", None)
        if cleanup is not None:
            cleanup()
        return sample

    for iteration in range(warmup):
        order = paths if iteration % 2 == 0 else tuple(reversed(paths))
        for path in order:
            one(path)

    local_samples = {path.name: [] for path in paths}
    for iteration in range(iterations):
        order = paths if iteration % 2 == 0 else tuple(reversed(paths))
        for path in order:
            local_samples[path.name].append(one(path))

    return {
        path.name: (
            local_samples[path.name],
            [
                _global_max_timing(sample, path.stage_names)
                for sample in local_samples[path.name]
            ],
        )
        for path in paths
    }


def _summary(path, local_samples, rank_max_samples, tail: int):
    local_tail = local_samples[-tail:]
    rank_tail = rank_max_samples[-tail:]

    def values(samples, field):
        if field == "gpu_e2e":
            return [sample.gpu_e2e_us for sample in samples]
        if field == "host_critical":
            return [sample.host_critical_us for sample in samples]
        return [sample.stage_us[field] for sample in samples]

    fields = (*path.stage_names, "gpu_e2e", "host_critical")
    return {
        "path": path.name,
        "full_stage_field": path.full_stage_field,
        "tail_iterations": tail,
        "tail_rank_max_stats_us": {
            field: _sample_stats(values(rank_tail, field)) for field in fields
        },
        "tail_local_rank_stats_us": {
            field: _sample_stats(values(local_tail, field)) for field in fields
        },
    }


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
    parser.add_argument(
        "--order",
        choices=("unfused-first", "fused-first", "alternating"),
        default="unfused-first",
    )
    parser.add_argument(
        "--route-pattern",
        choices=(
            "rank-balanced-hot",
            "paired-rank-half-remote",
            "paired-rank-local-only",
        ),
        default="paired-rank-half-remote",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tail-iters", type=int, default=50)
    parser.add_argument("--component-events", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    if not 1 <= args.tail_iters <= args.iters:
        raise ValueError("tail-iters must be in [1,iters]")

    shape = _shape()
    contract = {
        "shape": shape.__dict__,
        "route_pattern": args.route_pattern,
        "order": args.order,
        "same_bf16_start": True,
        "same_gmm1_silu_a4_requant_end": True,
        "same_topk_weights_and_w1": True,
        "gmm2_in_stage_event": False,
        "combine_in_stage_event": False,
        "unfused_full_stage": UNFUSED_FULL,
        "fused_full_stage": "stage1_full_quant_transport_fanout_gmm1_silu_requant",
        "warmup": args.warmup,
        "iterations": args.iters,
        "tail_iterations": args.tail_iters,
        "component_events": args.component_events,
        "statistic": "per_iteration_ep16_rank_max_then_tail_mean_p50_p95",
    }
    if args.plan_only:
        print("MEGAMOE_STAGE1_SMALL_OP_AB_PLAN " + json.dumps(contract, sort_keys=True))
        return 0

    rank, world, _local_rank, device = _setup_dist(needs_mori=True)
    if world != shape.ep_size:
        raise ValueError(f"Stage1 A/B requires world=16, got {world}")
    shared = _lightweight_shared_inputs(
        shape,
        rank,
        world,
        device,
        quantize_for_mori=False,
        prepare_stage1_weights=True,
        route_pattern=args.route_pattern,
    )

    paired = args.route_pattern != "rank-balanced-hot"
    valid_recv = shape.tokens * world // 2 if paired else shape.tokens * world
    unfused = UnfusedMoriStage1Path(
        shape,
        shared,
        rank,
        world,
        valid_recv=valid_recv,
        expand_local_routes=paired,
        record_components=args.component_events,
    )
    weights = shared.prepared_weights
    fused_op = MegaMoETileA4W4CommProbe(
        rank=rank,
        world_size=world,
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
        stage1_transport="sparse_wqe",
        probe_stage="stage1",
        stage1_mode="internodev1_tilepipe",
        stage1_cco_geometry="sparse_wqe",
        stage1_phase="full",
        stage1_tile_pipeline_fanout_shards=16,
    )
    fused = FusedStage1Path(
        fused_op,
        shared,
        shape,
        device,
        probe_stage="stage1",
        stage1_mode="internodev1_tilepipe",
        stage1_phase="full",
        cco_chunks_per_flush=1,
        cco_geometry="sparse_wqe",
        quant_two_cta_per_token=False,
        prequant_input=False,
        stage2_mode="full",
        canonical_stage1=False,
        route_pattern=args.route_pattern,
    )
    paths = (
        (unfused, fused)
        if args.order in ("unfused-first", "alternating")
        else (fused, unfused)
    )

    results = {}
    try:
        if args.order == "alternating":
            raw_results = _run_interleaved(
                paths,
                device,
                warmup=args.warmup,
                iterations=args.iters,
            )
            for path in paths:
                local, rank_max = raw_results[path.name]
                results[path.name] = _summary(
                    path, local, rank_max, args.tail_iters
                )
        else:
            for path in paths:
                local, rank_max = _run_path(
                    path,
                    device,
                    warmup=args.warmup,
                    iterations=args.iters,
                )
                results[path.name] = _summary(
                    path, local, rank_max, args.tail_iters
                )
                dist.barrier()

        gathered = [None] * world if rank == 0 else None
        dist.gather_object(
            {"rank": rank, "summaries": results}, gathered, dst=0
        )
        if rank == 0:
            all_rank_mean = {}
            for path in paths:
                field = path.full_stage_field
                all_rank_mean[path.name] = sum(
                    float(
                        row["summaries"][path.name][
                            "tail_local_rank_stats_us"
                        ][field]["mean"]
                    )
                    for row in gathered
                ) / world
            unfused_stats = results[unfused.name]["tail_rank_max_stats_us"][
                unfused.full_stage_field
            ]
            fused_stats = results[fused.name]["tail_rank_max_stats_us"][
                fused.full_stage_field
            ]
            comparison = {
                key: {
                    "unfused_us": unfused_stats[key],
                    "fused_us": fused_stats[key],
                    "fused_change_pct": 100.0
                    * (fused_stats[key] / unfused_stats[key] - 1.0),
                }
                for key in ("mean", "p50", "p95")
            }
            comparison["all_rank_mean"] = {
                "unfused_us": all_rank_mean[unfused.name],
                "fused_us": all_rank_mean[fused.name],
                "fused_change_pct": 100.0
                * (
                    all_rank_mean[fused.name]
                    / all_rank_mean[unfused.name]
                    - 1.0
                ),
            }
            print(
                "MEGAMOE_STAGE1_SMALL_OP_AB "
                + json.dumps(
                    {
                        **contract,
                        "unfused_prime_contract": unfused.prime_contract,
                        "summaries": results,
                        "tail_all_rank_sample_mean_us": all_rank_mean,
                        "comparison": comparison,
                        "by_rank": gathered,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        fused.close()
        import mori.shmem as ms

        ms.shmem_finalize()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

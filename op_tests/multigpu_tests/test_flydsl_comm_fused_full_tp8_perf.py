# SPDX-License-Identifier: MIT
"""Full Stage2 accuracy smoke test for the lightweight TP8 runner."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.distributed as dist
import flydsl.expr as fx

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    get_2stage_cfgs,
    get_padded_M,
    stage2_uses_route_reduce,
)
from aiter.dist.parallel_state import get_tp_group
from aiter.ops.flydsl.comm_fused_moe_host import _barrier
from aiter.ops.flydsl.comm_fused_moe_host import _stage2_args
from aiter.ops.flydsl.comm_fused_moe_host import create_flydsl_comm_fused_runners
from aiter.ops.flydsl.kernels.comm_fused_moe import atomic_compressed
from aiter.ops.flydsl.kernels.comm_fused_moe import persistent_window
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled
from op_tests.multigpu_tests import moe_tp_stage2_test_utils as fixtures


TOKENS = int(os.environ.get("COMM_FUSED_M", "2048"))
ROUTE = os.environ.get("COMM_FUSED_ROUTE", "uniform")
COMPUTE_BACKEND = os.environ.get("COMM_FUSED_COMPUTE", "flydsl")
PERF_ROUNDS = int(os.environ.get("COMM_FUSED_PERF_ROUNDS", "3"))
PERF_ITERS = int(os.environ.get("COMM_FUSED_PERF_ITERS", "20"))
PROFILE_ONLY = os.environ.get("COMM_FUSED_PROFILE_ONLY") == "1"
PROFILE_REPLAYS = int(os.environ.get("COMM_FUSED_PROFILE_REPLAYS", "1"))
PROFILE_COMPONENT = os.environ.get("COMM_FUSED_PROFILE_COMPONENT", "")
PROFILE_GRAPH = os.environ.get("COMM_FUSED_PROFILE_GRAPH") == "1"
TUNER_SMOKE = os.environ.get("COMM_FUSED_TUNER_SMOKE") == "1"
COLLECTIVE_BLOCK = int(os.environ.get("COMM_FUSED_COLLECTIVE_BLOCK", "0"))
RS_BLOCK = int(os.environ.get("COMM_FUSED_RS_BLOCK", str(COLLECTIVE_BLOCK)))
AG_BLOCK = int(os.environ.get("COMM_FUSED_AG_BLOCK", str(COLLECTIVE_BLOCK)))
RS_GRID = int(os.environ.get("COMM_FUSED_RS_GRID", "0"))
AG_GRID = int(os.environ.get("COMM_FUSED_AG_GRID", "0"))


def resolve_ordinary_stage2(args):
    metadata = get_2stage_cfgs(
        get_padded_M(args.target_tokens),
        args.hidden,
        args.inter_dim,
        args.experts,
        args.topk,
        dtypes.bf16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        gate_mode="separated",
        is_ep=False,
        has_stage2_bias=False,
        opus_weights_shuffled=True,
    )
    kernel_name = str(
        getattr(metadata.stage2, "keywords", {}).get("kernelName")
        or getattr(metadata.stage2, "keywords", {}).get("kernelName2")
        or ""
    )
    return metadata, kernel_name, not stage2_uses_route_reduce(metadata.stage2)


def run_ordinary_stage2_allreduce(
    case, metadata, *, requires_output_zero, shared_partial
):
    if requires_output_zero:
        case.partial_out.zero_()
    w2_scale = (
        case.w2_scale.view(dtypes.fp8_e8m0)
        if case.w2_scale.element_size() == 1
        else case.w2_scale
    )
    metadata.stage2(
        case.inter_states,
        None,
        case.w2,
        case.sorted_token_ids,
        case.sorted_expert_ids,
        case.num_valid_ids,
        case.partial_out,
        case.topk,
        w2_scale=w2_scale,
        a2_scale=case.a2_scale,
        block_m=int(metadata.block_m),
        sorted_weights=case.sorted_weights,
    )
    case.partial_out.add_(shared_partial)
    return get_tp_group().all_reduce(case.partial_out, ca_fp8_quant=False)


def main() -> None:
    rank, world, _local_rank, device, group = fixtures.setup_distributed(8)

    args = SimpleNamespace(
        tokens=TOKENS,
        target_tokens=TOKENS,
        logical_tokens=TOKENS,
        tp_size=8,
        hidden=7168,
        inter_dim=384,
        experts=384,
        topk=6,
        tile_m=64,
        tile_n=256,
        tile_k=128,
        route=ROUTE,
        seed=20260819,
        low_memory_w2_quant=True,
    )
    metadata, kernel_name, requires_zero = resolve_ordinary_stage2(args)
    args.tile_m = int(metadata.block_m)
    case = fixtures.make_stage2_case(
        args, rank, device, accumulate=requires_zero
    )
    token_term = (
        torch.arange(args.tokens, device=device, dtype=torch.float32)
        .remainder(7)
        .mul_(1.0 / 32.0)
        .view(-1, 1)
    )
    column_term = (
        torch.arange(args.hidden, device=device, dtype=torch.float32)
        .remainder(17)
        .mul_(1.0 / 128.0)
        .view(1, -1)
    )
    shared = (token_term + column_term + float(rank + 1) / 16.0).to(
        torch.bfloat16
    )
    w2_scale = (
        case.w2_scale.view(dtypes.fp8_e8m0)
        if case.w2_scale.element_size() == 1
        else case.w2_scale
    )

    ordinary = None
    if not PROFILE_COMPONENT:
        ordinary = run_ordinary_stage2_allreduce(
            case,
            metadata,
            requires_output_zero=requires_zero,
            shared_partial=shared,
        ).clone()
        fixtures.barrier(group)

    if TUNER_SMOKE:
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.ops.flydsl.comm_fused_moe_host import ShapeKey, winners_for
        from aiter.ops.flydsl.comm_fused_moe_tuner import benchmark, kid

        config = winners_for(
            ShapeKey(
                get_gfx_runtime(),
                args.hidden,
                args.inter_dim,
                args.experts,
                args.topk,
                8,
                "fp8",
                "fp4",
            )
        )[TOKENS]
        result = benchmark(
            tp_group=get_tp_group(),
            process_group=group,
            config=config,
            stage2_args=(
                case.inter_states,
                None,
                case.w2,
                case.sorted_token_ids,
                case.sorted_expert_ids,
                case.num_valid_ids,
                case.partial_out,
                case.topk,
            ),
            stage2_kwargs={
                "w2_scale": w2_scale,
                "a2_scale": case.a2_scale,
                "block_m": case.block_m,
                "sorted_weights": case.sorted_weights,
            },
            shared_partial=shared,
            reference=ordinary,
            rounds=2,
            iterations=20,
        )
        if rank == 0:
            print(f"COMM_FUSED_TUNER_SMOKE kid={kid(config)} result={result}")
        fixtures.cleanup_distributed(rank)
        return

    runner = create_flydsl_comm_fused_runners(
        tp_group=get_tp_group(),
        model_dim=args.hidden,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
    )[args.tokens]
    if rank == 0:
        schedule = getattr(runner, "schedule", None)
        print(
            "PRODUCTION_RUNNER "
            f"type={type(runner).__name__} "
            f"schedule={schedule}",
            flush=True,
        )
    if isinstance(runner.config, atomic_compressed.Config):
        if RS_GRID or AG_GRID:
            runner.config = replace(
                runner.config,
                reduce_scatter_grid=RS_GRID or runner.config.reduce_scatter_grid,
                all_gather_grid=AG_GRID or runner.config.all_gather_grid,
            )
        default_block = atomic_compressed.BLOCK
        if RS_BLOCK:
            atomic_compressed.BLOCK = RS_BLOCK
            atomic_compressed.compile_stage2_tp_reduce_scatter(runner.config)
        if AG_BLOCK:
            atomic_compressed.BLOCK = AG_BLOCK
            atomic_compressed.compile_stage2_tp_all_gather(runner.config)
        atomic_compressed.BLOCK = default_block

    if PROFILE_COMPONENT:
        if not isinstance(runner.config, persistent_window.Config):
            raise AssertionError(
                "component profiling requires the persistent-window runner, "
                f"got {type(runner).__name__}"
            )
        if PROFILE_COMPONENT not in ("g0", "cycle0"):
            raise ValueError(
                "COMM_FUSED_PROFILE_COMPONENT must be g0 or cycle0, "
                f"got {PROFILE_COMPONENT!r}"
            )
        stream = torch.cuda.current_stream(device)
        stage2_args = (
            case.inter_states,
            None,
            case.w2,
            case.sorted_token_ids,
            case.sorted_expert_ids,
            case.num_valid_ids,
            case.partial_out,
            case.topk,
        )
        stage2_kwargs = {
            "w2_scale": w2_scale,
            "a2_scale": case.a2_scale,
            "block_m": case.block_m,
            "sorted_weights": case.sorted_weights,
        }
        common = _stage2_args(
            stage2_args,
            stage2_kwargs,
            persistent_window,
            runner.config,
        )

        def run_g0():
            _run_compiled(
                persistent_window.compile_stage2_compute(runner.config, 0),
                (ptr_arg(runner.routes[0]), *common, stream),
            )

        def run_cycle0():
            _run_compiled(
                persistent_window.compile_persistent_cycle(runner.config, 0),
                (
                    ptr_arg(runner.routes[1]),
                    *common,
                    *runner._local_args(0, shared),
                    ptr_arg(runner.state),
                    stream,
                ),
            )

        # Compile and populate route0 before the measured component replay.
        run_g0()
        if PROFILE_COMPONENT == "cycle0":
            run_cycle0()
        torch.cuda.synchronize(device)
        component = run_g0 if PROFILE_COMPONENT == "g0" else run_cycle0
        for _ in range(PROFILE_REPLAYS):
            component()
        torch.cuda.synchronize(device)
        fixtures.cleanup_distributed(rank)
        if rank == 0:
            print(
                "FLYDSL_COMM_FUSED_PROFILE_COMPONENT_TP8_OK "
                f"component={PROFILE_COMPONENT} replays={PROFILE_REPLAYS}",
                flush=True,
            )
        return
    consumes_shared = isinstance(runner.config, atomic_compressed.Config)

    if COMPUTE_BACKEND == "flydsl":

        def run_comm_fused(shared_partial):
            return runner(
                stage2_args=(
                    case.inter_states,
                    None,
                    case.w2,
                    case.sorted_token_ids,
                    case.sorted_expert_ids,
                    case.num_valid_ids,
                    case.partial_out,
                    case.topk,
                ),
                stage2_kwargs={
                    "w2_scale": w2_scale,
                    "a2_scale": case.a2_scale,
                    "block_m": case.block_m,
                    "sorted_weights": case.sorted_weights,
                },
                shared_partial=shared_partial,
                ordinary_stage2=metadata.stage2,
            )

    else:
        from aiter.ops.flydsl.kernels.comm_fused_moe import full_width as k
        from aiter.ops.opus.moe_stage2_a8w4 import (
            opus_moe_stage2_a8w4_decode_fwd,
        )
        from aiter.ops.opus.moe_stage2_a8w4_meta import (
            opus_a8w4_kid_from_name,
        )

        opus_names = {
            "opus_occ4": "opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ4_rbn2240",
            "opus_occ5": "opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ5_rbn2304",
        }
        if not hasattr(runner, "route"):
            raise ValueError(
                f"COMM_FUSED_COMPUTE={COMPUTE_BACKEND!r} requires a "
                "full-width production runner"
            )
        opus_kid = opus_a8w4_kid_from_name(opus_names[COMPUTE_BACKEND])

        def run_comm_fused(shared_partial):
            stream = torch.cuda.current_stream(device)
            opus_moe_stage2_a8w4_decode_fwd(
                case.inter_states,
                case.w2,
                case.a2_scale,
                w2_scale,
                case.sorted_token_ids,
                case.sorted_weights,
                case.sorted_expert_ids,
                case.num_valid_ids,
                block_m=32,
                inter_dim_pad=0,
                out=runner.route.view(TOKENS * k.TOPK, k.H + k.H // 8),
                kernel_id=opus_kid,
                return_per_slot=True,
                token_num=TOKENS,
                topk=k.TOPK,
            )
            _run_compiled(
                k.compile_stage2_local_reduce(runner.config),
                (
                    ptr_arg(runner.route),
                    ptr_arg(runner.partial),
                    ptr_arg(shared_partial),
                    stream,
                ),
            )
            _barrier(
                runner.partial,
                runner.partial_flat_base,
                runner.partial_ready,
                stream,
            )
            _run_compiled(
                k.compile_stage2_tp_reduce_scatter(runner.config),
                (
                    fx.Int64(runner.partial_flat_base),
                    ptr_arg(runner.reduced_shard),
                    ptr_arg(runner.reduced_payload),
                    ptr_arg(runner.reduced_scale),
                    runner.rank,
                    stream,
                ),
            )
            _barrier(
                runner.reduced_payload,
                runner.reduced_payload_base,
                runner.reduced_ready,
                stream,
            )
            _run_compiled(
                k.compile_stage2_tp_all_gather(runner.config),
                (
                    fx.Int64(runner.reduced_payload_base),
                    fx.Int64(runner.reduced_scale_base),
                    ptr_arg(runner.output),
                    runner.rank,
                    stream,
                ),
            )
            return runner.output

    runner_shared = shared.clone() if consumes_shared else shared
    if consumes_shared:
        runner.output.fill_(float("nan"))
    fused = run_comm_fused(runner_shared)
    fixtures.barrier(group)
    max_abs, rel_l2 = fixtures.error_stats(fused, ordinary, group)
    if not torch.isfinite(fused).all() or not torch.isfinite(ordinary).all():
        local_finite = torch.tensor(
            [
                int(torch.isfinite(ordinary).sum().item()),
                int(torch.isfinite(fused).sum().item()),
                int(torch.isfinite(runner_shared).sum().item()),
                ordinary.numel(),
            ],
            dtype=torch.int64,
            device=device,
        )
        finite_by_rank = torch.empty(world * 4, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(finite_by_rank, local_finite, group=group)
        if rank == 0 and hasattr(runner, "partial"):
            scales = runner.partial[
                TOKENS * args.hidden : runner.partial_ready
            ].view(TOKENS, -1)
            print(
                f"FULL_RUNNER_M{TOKENS}_FINITE "
                f"{finite_by_rank.view(world, 4).cpu().tolist()}"
            )
            print(
                f"FULL_RUNNER_M{TOKENS}_FINITE_SHARDS "
                f"{torch.isfinite(fused).view(8, -1).sum(1).cpu().tolist()}"
            )
            print(
                f"FULL_RUNNER_M{TOKENS}_SCALE_HALVES "
                f"{[(int(part.min()), int(part.max()), int((part == 255).sum())) for part in scales.chunk(2)]}"
            )
    if rank == 0:
        print(
            f"FULL_RUNNER_M{TOKENS} route={ROUTE} kernel={kernel_name} "
            f"max_abs={max_abs:.6f} rel_l2={rel_l2:.6f}"
        )
    if max_abs > 1.0 or rel_l2 > 0.05:
        raise AssertionError(
            f"full runner accuracy failed: max_abs={max_abs}, rel_l2={rel_l2}"
        )
    if PROFILE_ONLY:
        fixtures.barrier(group)
        if PROFILE_GRAPH:
            profile_graph = fixtures.capture_graph(
                lambda: run_comm_fused(runner_shared),
                group,
                warmup_replays=3,
            )
            fixtures.barrier(group)
            for _ in range(PROFILE_REPLAYS):
                profile_graph.replay()
        else:
            for _ in range(PROFILE_REPLAYS):
                run_comm_fused(runner_shared)
        torch.cuda.synchronize(device)
        fixtures.barrier(group)
        fixtures.cleanup_distributed(rank)
        if rank == 0:
            print(
                f"FLYDSL_COMM_FUSED_PROFILE_ONLY_TP8_OK "
                f"replays={PROFILE_REPLAYS} graph={PROFILE_GRAPH}"
            )
        return
    def run_ordinary():
        return run_ordinary_stage2_allreduce(
            case,
            metadata,
            requires_output_zero=requires_zero,
            shared_partial=shared,
        )

    def run_fused():
        return run_comm_fused(runner_shared)

    def time_us(body, iterations=10, reset=None):
        for _ in range(3):
            if reset is not None:
                reset()
            body()
        fixtures.barrier(group)
        elapsed = 0.0
        if reset is None:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                body()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) * 1000.0 / iterations
        else:
            for _ in range(iterations):
                reset()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                body()
                end.record()
                end.synchronize()
                elapsed += start.elapsed_time(end) * 1000.0 / iterations
        local = torch.tensor(
            [elapsed],
            dtype=torch.float64,
            device=device,
        )
        values = torch.empty(world, dtype=torch.float64, device=device)
        dist.all_gather_into_tensor(values, local, group=group)
        return float(values.max().item())

    ordinary_us = time_us(run_ordinary, reset=(lambda: None) if consumes_shared else None)
    def reset_fused():
        if consumes_shared:
            runner_shared.copy_(shared)
            torch.cuda.synchronize(device)

    fused_us = time_us(run_fused, reset=reset_fused if consumes_shared else None)
    if rank == 0:
        print(
            f"FULL_RUNNER_M{TOKENS}_EAGER ordinary_us={ordinary_us:.3f} "
            f"fused_us={fused_us:.3f} speedup={ordinary_us / fused_us:.4f}x"
        )

    ordinary_graph = fixtures.capture_graph(run_ordinary, group, warmup_replays=3)
    if consumes_shared:
        reset_fused()
    fused_graph = fixtures.capture_graph(run_fused, group, warmup_replays=3)
    samples = {"ordinary": [], "fused": []}
    graphs = {"ordinary": ordinary_graph, "fused": fused_graph}
    for round_index in range(PERF_ROUNDS):
        order = ("ordinary", "fused")
        if round_index & 1:
            order = tuple(reversed(order))
        for name in order:
            if consumes_shared:
                elapsed = 0.0
                fixtures.barrier(group)
                for _ in range(PERF_ITERS):
                    if name == "fused":
                        reset_fused()
                    fixtures.barrier(group)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    elapsed += start.elapsed_time(end) * 1000.0 / PERF_ITERS
                local = torch.tensor([elapsed], dtype=torch.float64, device=device)
                ranks = torch.empty(world, dtype=torch.float64, device=device)
                dist.all_gather_into_tensor(ranks, local, group=group)
                samples[name].append([float(value) for value in ranks.cpu()])
            else:
                samples[name].append(
                    fixtures.time_graph(
                        graphs[name],
                        PERF_ITERS,
                        group,
                        world,
                        marker=f"comm_fused_moe.{name}.round{round_index}",
                    )
                )
    summary = fixtures.summarize(samples)
    ordinary_graph_us = summary["ordinary"]["median_rank_max_us"]
    fused_graph_us = summary["fused"]["median_rank_max_us"]
    ordinary_graph.replay()
    if consumes_shared:
        reset_fused()
    fused_graph.replay()
    torch.cuda.synchronize(device)
    graph_max_abs, graph_rel_l2 = fixtures.error_stats(fused, ordinary, group)
    if rank == 0:
        print(f"FULL_RUNNER_M{TOKENS}_GRAPH_SUMMARY {summary}")
        print(
            f"FULL_RUNNER_M{TOKENS}_GRAPH max_abs={graph_max_abs:.6f} "
            f"rel_l2={graph_rel_l2:.6f} ordinary_us={ordinary_graph_us:.3f} "
            f"fused_us={fused_graph_us:.3f} "
            f"speedup={ordinary_graph_us / fused_graph_us:.4f}x"
        )

    ordinary_graph = None
    fused_graph = None

    fixtures.cleanup_distributed(rank)
    if rank == 0:
        print("FLYDSL_COMM_FUSED_FULL_TP8_OK")


if __name__ == "__main__":
    main()

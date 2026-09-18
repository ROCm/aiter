# SPDX-License-Identifier: MIT
"""Full-forward graph replay and Stage2 attribution for the breakdown driver.

Formal runner settings are warmup=10/iters=40/tail=20; shorter invocations are
correctness smokes, not comparable performance baselines. Stage2 is attributed
from each rank's GPU trace, and full-forward events are measured separately.
See scripts/megamoe_tile/FUSED_STAGE2_DESIGN_20260909.md for the exact contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import MoriFusedMoeBaselinePath
from scripts.megamoe_tile.stage2_graph_inputs import (
    IMMUTABLE_INPUTS, PUBLIC_INPUTS, WEIGHTS, aggregate_input_identity,
    changed_inputs, fingerprint_shared_inputs, performance_exclusions, route_histogram,
)
from scripts.megamoe_tile.stage2_graph_protocol import validate_protocol_snapshot
from scripts.megamoe_tile.summarize_stage2_graph import aggregate, parse_trace


def _check(reference, actual, threshold, label):
    expected, output = reference.float(), actual.float()
    finite = bool(torch.isfinite(expected).all().item() and torch.isfinite(output).all().item())
    expected_norm = float(expected.norm().item())
    output_norm = float(output.norm().item())
    value = float((output - expected).norm().item() / max(expected_norm, 1e-12)) if finite else float("inf")
    local_value = value
    error = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    value = float(error.item())
    if value >= threshold:
        print("MEGAMOE_STAGE2_GRAPH_NUMERICAL_FAILURE " + json.dumps({
            "rank": dist.get_rank(), "label": label, "finite": finite,
            "local_rel_l2": local_value, "reference_norm": expected_norm,
            "output_norm": output_norm,
            "cosine": float((output * expected).sum().item() / max(expected_norm * output_norm, 1e-12)),
        }), flush=True)
        raise AssertionError(f"{label}: rank-max relL2={value}, threshold={threshold}")
    return {"label": label, "rank_max_rel_l2": value, "threshold": threshold}


def _device_epoch(operator):
    from aiter.ops.flydsl.kernels.megamoe_tile.window_view import read_window_u64

    base = int(operator._runtime.window.local_ptr)
    return int(read_window_u64(base + operator.stage1_layout.offset("epoch_gate"), 1)[0])


def _check_protocol(operator, expected_generation, label):
    import os
    if os.environ.get("MEGAMOE_TWO_KERNEL") == "1":
        # 两 kernel 路径用 plane_slot_inbox 取代了 rank_push_inbox,这套协议的
        # 区域没人写。数值判据仍是 candidate_or_mori_vs_mori 的 relL2。
        return {"rank": dist.get_rank(), "label": label, "error": None,
                "state": "skipped: two-kernel stage2 replaces the rank-push protocol"}
    local = {"rank": dist.get_rank(), "label": label, "error": None}
    try:
        local["state"] = validate_protocol_snapshot(
            operator.debug_direct_tile_snapshot(),
            expected_generation=expected_generation,
            readiness=operator.stage2_ready_granularity,
            tokens=operator.mtpr,
            accumulation_mode=operator.stage2_rank_accumulation_mode,
        )
    except (KeyError, TypeError, ValueError) as error:
        local["error"] = str(error)
    all_ranks = [None] * dist.get_world_size()
    dist.all_gather_object(all_ranks, local)
    errors = [row for row in all_ranks if row["error"]]
    if errors:
        raise AssertionError(f"Graph protocol validation failed: {errors}")
    return local


def _check_input_integrity(shared, original, fields, label):
    changed = changed_inputs(original, fingerprint_shared_inputs(shared, fields))
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, {"rank": dist.get_rank(), "changed": changed})
    failures = [row for row in ranks if row["changed"]]
    if failures:
        raise AssertionError(f"{label}: input contents changed: {failures}")


def run_graph_profile(path, shared, shape, args, contract, rank, world, device):
    if not 1 <= args.tail_iters <= args.iters or args.warmup < 1:
        raise ValueError("graph profiling needs warmup >= 1 and 1 <= tail_iters <= iters")
    if args.direct_packed_weights:
        raise ValueError("graph performance requires random packed weights; constant weights are diagnostic only")
    if os.environ.get("MEGAMOE_TILE_PROFILE_REGIONS", "0") == "1":
        raise ValueError("torch.profiler graph mode cannot use external ROCTx profiler pause/resume")
    candidate = args.path == "candidate"
    if args.graph_coalesce_reference_duplicates and not candidate:
        raise ValueError("coalesced reference is only supported for candidate correctness checks")
    # Record actual constructed data before either path overwrites the mutable
    # reference quantization buffers. Full-forward inputs are BF16 x and original
    # TopK/weights; a_quant/a_scale are only initial reference buffers. Their
    # post-check values can legitimately reflect the changed-input eager oracle.
    initial_inputs = fingerprint_shared_inputs(shared)
    properties = torch.cuda.get_device_properties(device)
    local_inputs = {
        "rank": rank, "tensors": initial_inputs,
        "routes": route_histogram(shared.topk_ids.cpu().tolist(), rank=rank, shape=shape.__dict__),
        "device": {"name": properties.name, "cu_count": properties.multi_processor_count,
                   "architecture": getattr(properties, "gcnArchName", None)},
    }
    input_rows = [None] * world
    dist.all_gather_object(input_rows, local_inputs)
    input_identity = aggregate_input_identity(input_rows, shape=shape.__dict__)
    trace_dir = Path(args.torch_profiler_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"rank{rank}.inputs.json").write_text(json.dumps(local_inputs, indent=2) + "\n")
    if args.route_pattern == "cross_node" and not input_identity["eplb"]:
        raise AssertionError("cross_node inputs do not satisfy the measured EPLB workload")
    if candidate:
        # The public wrapper consumes contiguous byte views of these exact
        # weights. Check aliases so the recorded shared weights are also the
        # actual candidate weights, not an independently prepared copy.
        for name in WEIGHTS:
            actual, expected = getattr(path.operator, "_" + name), getattr(shared.prepared_weights, name)
            if actual.data_ptr() != expected.data_ptr() or actual.numel() * actual.element_size() != expected.numel() * expected.element_size():
                raise AssertionError(f"candidate {name} is not the recorded shared weight storage")
    reference_path = MoriFusedMoeBaselinePath(
        shape, shared, rank, world,
        valid_recv=shape.tokens * world // 2,
        combine_quant_type="none",
        block_num=args.mori_block_num,
        rdma_block_num=args.mori_rdma_block_num,
        exact_capacity=True,
    ) if candidate else path.baseline
    if reference_path.shared is not shared:
        raise AssertionError("MORI must consume the same recorded SharedInputs")
    def reference_forward(*, debug_sync=False):
        return reference_path._forward(
            debug_sync=debug_sync,
            reference_coalesce_duplicates=args.graph_coalesce_reference_duplicates,
        )

    reference = reference_forward(debug_sync=args.graph_debug_reference)[:shape.tokens].clone()
    torch.cuda.synchronize(device)
    dist.barrier()
    if candidate:
        call = lambda: path.operator.forward(shared.x, shared.route_weights, shared.topk_ids)
    else:
        call = path.baseline._forward
    eager = call()[:shape.tokens].clone()
    torch.cuda.synchronize(device)
    checks = [_check(reference, eager, args.graph_rel_l2_threshold, "candidate_or_mori_vs_mori")]
    # Explicit static buffers avoid Dynamo mutation/input/output copies. The
    # graph contains one full forward; the replay loop stays outside capture.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        graph_output = call()
    # Capture enqueues no execution. Verify both parities across replays and
    # refresh input contents without changing the captured pointer addresses.
    start_epoch = _device_epoch(path.operator) if candidate else None
    graph.replay()
    torch.cuda.synchronize(device)
    checks.append(_check(eager, graph_output[:shape.tokens], 1e-2, "first_replay_vs_eager"))
    original = shared.x.clone()
    shared.x.mul_(-0.75)
    updated_reference = reference_forward()[:shape.tokens].clone()
    torch.cuda.synchronize(device)
    dist.barrier()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)
    checks.append(_check(updated_reference, graph_output[:shape.tokens], args.graph_rel_l2_threshold, "changed_input_after_five_replays"))
    shared.x.copy_(original)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize(device)
    checks.append(_check(eager, graph_output[:shape.tokens], 1e-2, "restored_input_after_four_replays"))
    protocol_checks = []
    if candidate:
        # Ten correctness replays above: 1 original + 5 changed + 4 restored.
        # Check actual device epochs and BOTH named stage counters; a missing
        # snapshot field must fail rather than silently becoming error_count=0.
        protocol_checks.append(_check_protocol(path.operator, start_epoch + 10, "after_input_replays"))
    correctness_replays = 10
    if args.graph_check_routing_replay:
        original_ids = shared.topk_ids.clone()
        # Swapping the two expert nodes preserves duplicate-slot relations and each
        # token's per-rank route capacity. For local-only fixtures this changes
        # every node mask from present to absent, exercising stale BF16 payload
        # clearing within the same arena. Four replays reuse both parities.
        try:
            shared.topk_ids.copy_((original_ids + shape.experts // 2) % shape.experts)
            swapped_reference = reference_forward()[:shape.tokens].clone()
            torch.cuda.synchronize(device)
            dist.barrier()
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize(device)
            correctness_replays += 4
            checks.append(_check(swapped_reference, graph_output[:shape.tokens],
                                 args.graph_rel_l2_threshold, "swapped_expert_nodes_after_four_replays"))
            if candidate:
                protocol_checks.append(_check_protocol(path.operator, start_epoch + correctness_replays,
                                                       "after_swapped_routing_replays"))
        finally:
            shared.topk_ids.copy_(original_ids)
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize(device)
        correctness_replays += 4
        checks.append(_check(eager, graph_output[:shape.tokens], 1e-2,
                             "restored_routing_after_four_replays"))
        if candidate:
            protocol_checks.append(_check_protocol(path.operator, start_epoch + correctness_replays,
                                                   "after_restored_routing_replays"))
    if rank == 0:
        print("MEGAMOE_STAGE2_GRAPH_CORRECTNESS " + json.dumps(checks), flush=True)
    _check_input_integrity(shared, initial_inputs, PUBLIC_INPUTS, "before timing")
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    # A separate unprofiled event batch cross-checks full-forward graph timing.
    # Events are read only after all replays; no per-iteration host/rank sync.
    event_pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
    for start, end in event_pairs:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize(device)
    event_us = [start.elapsed_time(end) * 1000 for start, end in event_pairs]
    dist.barrier()
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for iteration in range(args.iters):
            with record_function(f"stage2_graph_{args.path}_tpr{shape.tokens}_iter{iteration}"):
                graph.replay()
        torch.cuda.synchronize(device)
    trace_file = trace_dir / f"rank{rank}.json"
    prof.export_chrome_trace(str(trace_file))
    checks.append(_check(eager, graph_output[:shape.tokens], 1e-2, "final_replay_vs_eager"))
    if candidate:
        # Include the unprofiled event batch as well as the profiler batch.
        # Protocol errors during timing must not escape the earlier input check.
        expected_epoch = start_epoch + correctness_replays + args.warmup + 2 * args.iters
        protocol_checks.append(_check_protocol(path.operator, expected_epoch, "after_all_replays"))
    # Recheck all immutable inputs, including complete weights, after timing.
    # Hashing never adds transfers or GPU work to a captured/profiled forward.
    _check_input_integrity(shared, initial_inputs, IMMUTABLE_INPUTS, "after timing")
    local = {"rank": rank, "trace": str(trace_file), "checks": checks, "protocol_checks": protocol_checks, "event_full_forward_us": event_us, "parse_error": None}
    if args.device_timeline:
        from op_tests.multigpu_tests.bench_megamoe_tile_ep16_stage2_breakdown import TIMELINE_INTERVALS

        local["final_replay_timeline"] = path.operator.debug_device_timeline()
        ticks = local["final_replay_timeline"]["ticks"]
        scale = 1000.0 / path._wall_clock_rate_khz
        local["final_replay_intervals_us"] = {
            name + "_us": (ticks[end] - ticks[begin]) * scale
            for name, (begin, end) in TIMELINE_INTERVALS.items()
            if ticks[begin] > 0 and ticks[end] > 0
        }
        if args.device_timeline_history_depth:
            # Read only after BOTH timing batches have finished. Device stores
            # use generation & (depth-1), so the captured graph still has exactly
            # two kernels and the replay loop gets no copy, barrier, or sync.
            # Keep event-batch history too: long tails must not be explained
            # solely by a profiler artifact if they also occur without profiling.
            first_generation = expected_epoch - 2 * args.iters + 1
            history = []
            for offset in range(2 * args.iters):
                snapshot = path.operator.debug_device_timeline(first_generation + offset)
                ticks = snapshot["ticks"]
                history.append({
                    **snapshot,
                    "batch": "event" if offset < args.iters else "profiler",
                    "iteration": offset % args.iters,
                    "intervals_us": {
                        name + "_us": (ticks[end] - ticks[begin]) * scale
                        for name, (begin, end) in TIMELINE_INTERVALS.items()
                        if ticks[begin] > 0 and ticks[end] > 0
                    },
                })
            local["replay_timelines"] = history
    try:
        local["samples"] = parse_trace(json.loads(trace_file.read_text()), path=args.path, tokens=shape.tokens, iterations=args.iters, tail=args.tail_iters)
    except (ValueError, KeyError) as error:
        local["parse_error"] = str(error)
    (trace_dir / f"rank{rank}.samples.json").write_text(json.dumps(local, indent=2) + "\n")
    all_ranks = [None] * world
    dist.all_gather_object(all_ranks, local)
    errors = {row["rank"]: row["parse_error"] for row in all_ranks if row["parse_error"]}
    if errors:
        raise RuntimeError(f"incomplete profiler evidence; preserve traces and rerun: {errors}")
    summary = aggregate(all_ranks, expected_world=world)
    root = Path(__file__).resolve().parents[2]
    source_files = [
        "aiter/ops/flydsl/kernels/megamoe_tile/stage1.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/stage2.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/mega_moe_tile_a4w4.py",
        "op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py",
        "op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py",
        "op_tests/multigpu_tests/megamoe_stage2_graph.py",
        "scripts/megamoe_tile/summarize_stage2_graph.py",
        "scripts/megamoe_tile/stage2_graph_window.py",
        "scripts/megamoe_tile/stage2_graph_protocol.py",
        "scripts/megamoe_tile/stage2_graph_reference.py",
        "scripts/megamoe_tile/stage2_graph_inputs.py",
        "scripts/megamoe_tile/compare_stage2_graph.py",
        "op_tests/multigpu_tests/bench_mega_moe_v2.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/stage1_abi.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/stage2_abi.py",
        "aiter/ops/flydsl/kernels/megamoe_tile/rank_push_layout.py",
        "op_tests/multigpu_tests/bench_megamoe_tile_ep16_dual_path.py",
    ]
    tune = os.environ.get("AITER_CONFIG_FMOE")
    result = {
        **contract,
        "measurement": "one_full_forward_per_explicit_cuda_graph_replay",
        "input_identity": input_identity,
        "input_integrity_passed": True,
        "graph_debug_reference": args.graph_debug_reference,
        "graph_coalesce_reference_duplicates": args.graph_coalesce_reference_duplicates,
        "reference_route_semantics": (
            "sum duplicate expert weights; zero-weight same-rank filler experts; eager reference only"
            if args.graph_coalesce_reference_duplicates else "original routing"
        ),
        "diagnostic_final_replay_intervals_us": {row["rank"]: row["final_replay_intervals_us"] for row in all_ranks} if args.device_timeline else None,
        "diagnostic_timeline_history": {
            "depth": args.device_timeline_history_depth,
            "records_per_rank": 2 * args.iters,
            "batches": ["event", "profiler"],
            "artifact": "node*/rank*.samples.json: replay_timelines",
            "readback": "after both batches; no per-replay copies or synchronization",
        } if args.device_timeline_history_depth else None,
        "candidate_stage1_input": "fresh fused Stage1 every replay",
        "candidate_stage1_timing": "included in graph; Stage2 extracted from GPU trace",
        "statistic": {
            "pooled_min": "minimum actual Stage2 span across all ranks and selected tail replays (EPLB)",
            "rank_mean_of_min": "per-rank minimum actual Stage2 span over tail replays, then all-rank mean",
            "rank_mean": "per-rank mean actual Stage2 span over tail replays, then all-rank mean",
        }[args.graph_comparison_statistic],
        "comparison_metric": "timing.metrics.stage2_span_us." + args.graph_comparison_statistic,
        "graph_comparison_statistic": args.graph_comparison_statistic,
        "warmup": args.warmup, "iterations": args.iters, "tail_iterations": args.tail_iters,
        "graph_check_routing_replay": args.graph_check_routing_replay,
        "correctness_replays": correctness_replays,
        "graph_contains_output_clone": False,
        "graph_input_quantization": "BF16-to-FP4 included on both paths",
        "activation_parameters": {"beta": 1.0, "linear_beta": 1.0} if shape.activation == "situv2" else None,
        "mori_blocks": args.mori_block_num, "mori_rdma_blocks": args.mori_rdma_block_num,
        "environment": {key: os.environ.get(key) for key in ("MORI_RDMA_DEVICES", "MORI_NUM_QP_PER_PE", "MORI_DEVICE_NIC", "MORI_IB_GID_INDEX", "AITER_SITUV2_A4W4", "AITER_CONFIG_FMOE", "AMD_SERIALIZE_KERNEL")},
        "torch_version": torch.__version__,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "tune_sha256": hashlib.sha256(Path(tune).read_bytes()).hexdigest() if tune else None,
        "timing": summary,
        "full_forward_event_rank_mean_us": sum(sum(row["event_full_forward_us"][-args.tail_iters:]) / args.tail_iters for row in all_ranks) / world,
        "correctness": checks,
        "protocol_checks": {row["rank"]: row["protocol_checks"] for row in all_ranks},
    }
    result["performance_exclusions"] = performance_exclusions(result, input_identity)
    result["performance_eligible"] = not result["performance_exclusions"]
    if rank == 0:
        (trace_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print("MEGAMOE_EP16_STAGE2_GRAPH_RESULT " + json.dumps(result, sort_keys=True), flush=True)

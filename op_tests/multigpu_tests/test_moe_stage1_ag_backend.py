# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end TP8 validation for the production Stage1 AG backend.

Run with::

    torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_moe_stage1_ag_backend.py

The reference path gathers BF16 activations and route metadata before the
ordinary two-stage MoE.  The candidate path gathers compact route metadata and
lets :class:`TpStage1AgBackend` overlap a quantized CCO payload AllGather with
the ready-aware Stage1 kernel.  Both paths reuse the ordinary Stage2 contract.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import os
import statistics
from dataclasses import replace

import torch
import torch.distributed as dist

from aiter import ActivationType, QuantType, dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    init_distributed_environment,
)
from aiter.fused_moe import _fused_moe_impl
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.flydsl.moe_stage1_ag import (
    TP_STAGE1_AG_KID,
    TpStage1AgConfig,
    close_tp_stage1_ag_backends,
    get_tp_stage1_ag_backend,
)
from aiter.ops.flydsl.moe_stage1_ready import Stage1ReadyPlan
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight


def setup_distributed(expected_world: int):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != expected_world:
        raise ValueError(f"this test requires {expected_world} ranks, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ["AITER_MOE_STAGE1_CCO"] = "1"
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    ensure_model_parallel_initialized(world, 1)
    group = get_tp_group()
    dist.all_reduce(
        torch.zeros(1, dtype=torch.int32, device=device),
        group=group.device_group,
    )
    torch.cuda.synchronize(device)
    return rank, world, device, group


def cleanup_distributed() -> None:
    close_tp_stage1_ag_backends()
    for get_group in (
        get_tp_group,
        get_pcp_group,
        get_pp_group,
        get_dp_group,
        get_ep_group,
        get_dcp_group,
    ):
        group = get_group()
        communicator = group.device_communicator
        if communicator is not None:
            communicator.destroy()
            group.device_communicator = None
        group.mq_broadcaster = None
    dist.destroy_process_group()


def barrier(group) -> None:
    torch.cuda.synchronize()
    dist.barrier(group=group.device_group)


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    return tensor_model_parallel_all_gather(tensor, use_custom=False, dim=0)


def make_inputs(args, rank: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(args.seed + rank * 1009)
    local_tokens = args.local_tokens - rank * args.rank_token_step
    if args.zero_last_rank and rank == args.world - 1:
        local_tokens = 0
    if local_tokens < 0:
        raise ValueError("--rank-token-step leaves a rank with negative tokens")
    hidden = torch.randn(
        (local_tokens, args.hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    token = torch.arange(
        local_tokens, dtype=torch.int32, device=device
    )[:, None]
    slot = torch.arange(args.topk, dtype=torch.int32, device=device)[None, :]
    topk_ids = (token * 17 + slot * 29 + rank * 37) % args.experts
    logits = torch.randn(
        (local_tokens, args.topk),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    topk_weights = logits.softmax(dim=-1)
    return hidden.contiguous(), topk_weights.contiguous(), topk_ids.contiguous()


def make_weights(args, device: torch.device):
    # A small e8m0 scale keeps the deterministic packed-FP4 operands finite
    # without allocating multi-GB BF16 staging weights on every rank.
    scale = 112
    w1_nibble = (
        torch.arange(2 * args.inter, dtype=torch.uint8, device=device) % 6 + 1
    )
    w1_packed = (w1_nibble | (w1_nibble << 4)).view(1, -1, 1)
    w1_q = (
        w1_packed.expand(args.experts, -1, args.hidden // 2)
        .contiguous()
        .view(dtypes.fp4x2)
    )
    w1_scale = torch.full(
        (args.experts * 2 * args.inter, args.hidden // 32),
        scale,
        dtype=torch.uint8,
        device=device,
    )
    w2_nibble = (
        torch.arange(args.inter // 2, dtype=torch.uint8, device=device) % 6 + 1
    )
    w2_packed = (w2_nibble | (w2_nibble << 4)).view(1, 1, -1)
    w2_q = (
        w2_packed.expand(args.experts, args.hidden, -1)
        .contiguous()
        .view(dtypes.fp4x2)
    )
    w2_scale = torch.full(
        (args.experts * args.hidden, args.inter // 32),
        scale,
        dtype=torch.uint8,
        device=device,
    )
    return (
        moe_shuffle_weight(
            w1_q,
            args.experts,
            is_guinterleave=True,
            gate_up=True,
        ).contiguous(),
        moe_shuffle_scale(
            w1_scale,
            args.experts,
            is_guinterleave=True,
            gate_up=True,
        ).contiguous(),
        moe_shuffle_weight(
            w2_q,
            args.experts,
            is_guinterleave=False,
            gate_up=False,
        ).contiguous(),
        moe_shuffle_scale(
            w2_scale,
            args.experts,
            is_guinterleave=False,
            gate_up=False,
        ).contiguous(),
    )


def pad_local_inputs(hidden, topk_weights, topk_ids, padded_rows):
    if hidden.shape[0] == padded_rows:
        return hidden, topk_weights, topk_ids
    hidden_padded = hidden.new_empty((padded_rows, hidden.shape[1]))
    weights_padded = topk_weights.new_zeros((padded_rows, topk_weights.shape[1]))
    ids_padded = topk_ids.new_zeros((padded_rows, topk_ids.shape[1]))
    hidden_padded[: hidden.shape[0]].copy_(hidden)
    if hidden.shape[0]:
        hidden_padded[hidden.shape[0] :].copy_(
            hidden[:1].expand(padded_rows - hidden.shape[0], -1)
        )
    else:
        hidden_padded.fill_(1.0)
    weights_padded[: topk_weights.shape[0]].copy_(topk_weights)
    ids_padded[: topk_ids.shape[0]].copy_(topk_ids)
    return hidden_padded, weights_padded, ids_padded


def reference_call(
    hidden,
    topk_weights,
    topk_ids,
    weights,
    padded_rows,
    capture_calls=None,
    force_plain_tpag_stage1: bool = False,
):
    w1, w1_scale, w2, w2_scale = weights
    hidden, topk_weights, topk_ids = pad_local_inputs(
        hidden, topk_weights, topk_ids, padded_rows
    )
    global_hidden = all_gather(hidden)
    global_weights = all_gather(topk_weights)
    global_ids = all_gather(topk_ids)
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    fused_moe_module.kernel_bench_callable = capture_calls
    metadata_transform = None
    if force_plain_tpag_stage1:
        def metadata_transform(metadata):
            return replace(
                metadata,
                stage1=functools.partial(
                    fused_moe_module._flydsl_stage1_wrapper,
                    kernelName=TP_STAGE1_AG_KID,
                    activation=ActivationType.Silu,
                    inter_dim_pad=0,
                    model_dim_pad=0,
                ),
                block_m=64,
                ksplit=1,
                run_1stage=False,
                has_bias=False,
                fuse_quant="fp8",
                prequant=True,
            )
    try:
        return _fused_moe_impl(
            hidden_states=global_hidden,
            w1=w1,
            w2=w2,
            topk_weight=global_weights,
            topk_ids=global_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            dtype=torch.bfloat16,
            gate_mode=GateMode.INTERLEAVE.value,
            _q_dtype_a=dtypes.fp8 if force_plain_tpag_stage1 else None,
            _metadata_transform=metadata_transform,
        )
    finally:
        fused_moe_module.kernel_bench_callable = None


def candidate_call(
    backend,
    hidden,
    topk_weights,
    topk_ids,
    weights,
    sizes,
    capture_calls=None,
):
    w1, w1_scale, w2, w2_scale = weights
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    fused_moe_module.kernel_bench_callable = capture_calls
    try:
        return backend.apply(
            hidden,
            topk_weights,
            topk_ids,
            w1,
            w2,
            sizes=sizes,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            hidden_pad=0,
            intermediate_pad=0,
            swiglu_limit=None,
            gate_mode=GateMode.INTERLEAVE.value,
        )
    finally:
        fused_moe_module.kernel_bench_callable = None


def time_call(fn, group, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    barrier(group)
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        sample = torch.tensor(start.elapsed_time(end), device="cuda")
        dist.all_reduce(sample, op=dist.ReduceOp.MAX, group=group.device_group)
        samples.append(float(sample.item()))
    return statistics.median(samples)


def time_call_with_setup(
    fn, setup, group, warmup: int, iterations: int
) -> float:
    """Time only ``fn`` while keeping deterministic state reset out of events."""

    for _ in range(warmup):
        setup()
        fn()
    barrier(group)
    samples = []
    for _ in range(iterations):
        setup()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        sample = torch.tensor(start.elapsed_time(end), device="cuda")
        dist.all_reduce(sample, op=dist.ReduceOp.MAX, group=group.device_group)
        samples.append(float(sample.item()))
    return statistics.median(samples)


def clone_stage1_output(output):
    tensors = output if isinstance(output, tuple) else (output,)
    cloned = tuple(tensor.clone() for tensor in tensors)
    return cloned if isinstance(output, tuple) else cloned[0]


def validate_stage1_all_tensors(reference, candidate, group, name: str) -> dict:
    reference_tensors = reference if isinstance(reference, tuple) else (reference,)
    candidate_tensors = candidate if isinstance(candidate, tuple) else (candidate,)
    if len(reference_tensors) != len(candidate_tensors):
        raise AssertionError(f"{name}: Stage1 return arity changed")
    mismatches = torch.zeros(
        len(reference_tensors),
        dtype=torch.int64,
        device=reference_tensors[0].device,
    )
    for index, (expected, actual) in enumerate(
        zip(reference_tensors, candidate_tensors)
    ):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                f"{name}: Stage1 tensor contract changed at {index}: "
                f"{expected.shape}/{expected.dtype} vs "
                f"{actual.shape}/{actual.dtype}"
            )
        mismatches[index] = (
            expected.contiguous().view(torch.uint8)
            != actual.contiguous().view(torch.uint8)
        ).sum()
    dist.all_reduce(mismatches, op=dist.ReduceOp.SUM, group=group.device_group)
    result = {
        f"tensor_{index}_byte_mismatches": int(value)
        for index, value in enumerate(mismatches.cpu().tolist())
    }
    if any(result.values()):
        raise AssertionError(f"{name}: {result}")
    return result


def compact_logical_rows(
    output: torch.Tensor, sizes: list[int], padded_rows: int
) -> torch.Tensor:
    return torch.cat(
        [
            output[source * padded_rows : source * padded_rows + rows]
            for source, rows in enumerate(sizes)
        ],
        dim=0,
    )


def validate_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    group,
    *,
    sizes: list[int],
    padded_rows: int,
) -> dict:
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"output shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    # Physical padding rows carry zero route weights but their intermediate
    # values are deliberately unspecified.  Production reduce-scatter drops
    # those rows; compare only the same compact logical contract here.
    reference = compact_logical_rows(reference, sizes, padded_rows)
    candidate = compact_logical_rows(candidate, sizes, padded_rows)
    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    reference_finite = torch.isfinite(reference_f32)
    candidate_finite = torch.isfinite(candidate_f32)
    if not reference_finite.all() or not candidate_finite.all():
        diagnostics = torch.tensor(
            [
                (~reference_finite).sum(),
                (~candidate_finite).sum(),
                reference_f32[reference_finite].abs().max(),
                candidate_f32[candidate_finite].abs().max(),
            ],
            dtype=torch.float64,
            device=reference.device,
        )
        dist.all_reduce(
            diagnostics, op=dist.ReduceOp.MAX, group=group.device_group
        )
        raise AssertionError(
            "non-finite output in production backend comparison: "
            f"reference_nonfinite={int(diagnostics[0].item())}, "
            f"candidate_nonfinite={int(diagnostics[1].item())}, "
            f"reference_finite_max={diagnostics[2].item()}, "
            f"candidate_finite_max={diagnostics[3].item()}"
        )
    difference = (reference_f32 - candidate_f32).abs()
    denominator = reference_f32.abs().clamp_min(1e-5)
    metrics = torch.tensor(
        [
            difference.max(),
            (difference / denominator).max(),
            difference.mean(),
            (reference.view(torch.int16) != candidate.view(torch.int16)).sum(),
        ],
        dtype=torch.float64,
        device=reference.device,
    )
    dist.all_reduce(metrics[:3], op=dist.ReduceOp.MAX, group=group.device_group)
    dist.all_reduce(metrics[3:], op=dist.ReduceOp.SUM, group=group.device_group)
    result = {
        "max_abs": float(metrics[0].item()),
        "max_rel": float(metrics[1].item()),
        "mean_abs_max_rank": float(metrics[2].item()),
        "bit_mismatches_all_ranks": int(metrics[3].item()),
    }
    if result["max_abs"] > 0.02 or result["max_rel"] > 0.08:
        raise AssertionError(f"production Stage1 AG output mismatch: {result}")
    return result


def validate_workspace(
    backend, hidden, topk_weights, topk_ids, sizes, group
) -> dict:
    padded_rows = backend.padded_rows(hidden.shape[0], sizes)
    workspace_set = backend._workspaces[padded_rows]
    slot = workspace_set.slots[(workspace_set.index - 1) % len(workspace_set.slots)]
    hidden, topk_weights, topk_ids = pad_local_inputs(
        hidden, topk_weights, topk_ids, padded_rows
    )
    local_q, local_scale = per_1x32_mx_quant(hidden, quant_mode="fp8")
    logical_q = all_gather(local_q.view(torch.uint8)).view(dtypes.fp8)
    logical_scale = all_gather(local_scale)
    expected_ids = all_gather(topk_ids)
    expected_weights = all_gather(topk_weights)
    chunks = backend.config.chunks_per_source
    rows_per_chunk = padded_rows // chunks
    route_slices = [
        slice(backend.rank * padded_rows, (backend.rank + 1) * padded_rows)
    ]
    for chunk in range(chunks):
        for offset in range(1, backend.world):
            source = (backend.rank + offset) % backend.world
            begin = source * padded_rows + chunk * rows_per_chunk
            route_slices.append(slice(begin, begin + rows_per_chunk))
    expected_ids = torch.cat([expected_ids[index] for index in route_slices])
    expected_weights = torch.cat(
        [expected_weights[index] for index in route_slices]
    )
    expected_q = (
        logical_q.view(backend.world, chunks, rows_per_chunk, backend.config.hidden)
        .permute(1, 0, 2, 3)
        .reshape_as(slot.values)
    )
    expected_scale = (
        logical_scale.view(
            backend.world,
            chunks,
            rows_per_chunk,
            backend.config.hidden // 32,
        )
        .permute(1, 0, 2, 3)
        .reshape_as(slot.scales)
    )
    metrics = torch.tensor(
        [
            (slot.values.view(torch.uint8) != expected_q.view(torch.uint8)).sum(),
            (slot.scales != expected_scale).sum(),
            (slot.global_topk_ids != expected_ids).sum(),
            (
                slot.global_topk_weights.view(torch.int32)
                != expected_weights.view(torch.int32)
            ).sum(),
            slot.ready[0],
        ],
        dtype=torch.int64,
        device=hidden.device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group.device_group)
    result = {
        "value_byte_mismatches": int(metrics[0].item()),
        "scale_byte_mismatches": int(metrics[1].item()),
        "route_id_mismatches": int(metrics[2].item()),
        "route_weight_bit_mismatches": int(metrics[3].item()),
        "ready": int(metrics[4].item()),
    }
    if any(result[key] for key in tuple(result)[:4]):
        raise AssertionError(f"production Stage1 AG workspace mismatch: {result}")
    if result["ready"] != -1:
        raise AssertionError(f"production Stage1 AG did not publish full ready: {result}")
    return result


def validate_queue(plan, num_valid_ids, group, *, tile_m: int) -> dict:
    if not isinstance(plan, Stage1ReadyPlan):
        raise TypeError("production Stage1 must use Stage1ReadyPlan")
    valid_tiles = (num_valid_ids[0] + tile_m - 1) // tile_m
    valid_tile_count = int(valid_tiles.item())
    claimed_per_partition = plan.tile_claimed[:valid_tile_count].sum(dim=0)
    trailing_claims = plan.tile_claimed[valid_tile_count:].sum()
    metrics = torch.stack(
        (
            valid_tiles,
            claimed_per_partition.min(),
            claimed_per_partition.max(),
            trailing_claims,
        )
    ).to(torch.int64)
    minimum = metrics.clone()
    maximum = metrics.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group.device_group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group.device_group)
    result = {
        "valid_tiles_minmax": (int(minimum[0]), int(maximum[0])),
        "claimed_tiles_minmax": (int(minimum[1]), int(maximum[2])),
        "trailing_claims_minmax": (int(minimum[3]), int(maximum[3])),
    }
    expected = minimum[0]
    if not (
        torch.all(minimum[:3] == expected)
        and torch.all(maximum[:3] == expected)
        and minimum[3] == 0
        and maximum[3] == 0
    ):
        raise AssertionError(f"Stage1 AG lost or duplicated tiles: {result}")
    return result


def capture_candidate(fn, group):
    fn()
    barrier(group)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output = fn()
    barrier(group)
    return graph, captured_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--local-tokens", type=int, default=256)
    parser.add_argument(
        "--rank-token-step",
        type=int,
        default=0,
        help="Subtract this many local tokens per rank to test uneven eager DP.",
    )
    parser.add_argument(
        "--zero-last-rank",
        action="store_true",
        help="Give the last rank no local tokens while keeping a production-size bucket.",
    )
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--inter", type=int, default=384)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--graph-replays", type=int, default=20)
    parser.add_argument(
        "--strict-stage1-phases",
        action="store_true",
        help=(
            "Measure production-slot R-only, G/all-ready, strict R->G, "
            "and ready-aware R||G with identical prepared Stage1 work."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world, device, group = setup_distributed(args.world)
    try:
        hidden, topk_weights, topk_ids = make_inputs(args, rank, device)
        weights = make_weights(args, device)
        local_size = torch.tensor([hidden.shape[0]], dtype=torch.int32, device=device)
        gathered_sizes = torch.empty(world, dtype=torch.int32, device=device)
        dist.all_gather_into_tensor(
            gathered_sizes, local_size, group=group.device_group
        )
        sizes = [int(size) for size in gathered_sizes.cpu().tolist()]
        backend = get_tp_stage1_ag_backend(
            group,
            TpStage1AgConfig(
                hidden=args.hidden,
                experts=args.experts,
                topk=args.topk,
                min_global_tokens=sum(sizes),
            ),
        )
        padded_rows = backend.padded_rows(hidden.shape[0], sizes)

        reference_calls = []
        candidate_calls = []
        reference = reference_call(
            hidden,
            topk_weights,
            topk_ids,
            weights,
            padded_rows,
            reference_calls,
        )
        flydsl_reference_calls = []
        flydsl_reference = reference_call(
            hidden,
            topk_weights,
            topk_ids,
            weights,
            padded_rows,
            flydsl_reference_calls,
            force_plain_tpag_stage1=True,
        )
        candidate = candidate_call(
            backend,
            hidden,
            topk_weights,
            topk_ids,
            weights,
            sizes,
            candidate_calls,
        )
        candidate_stage1_call = next(
            call for name, call in candidate_calls if name == "stage1"
        )
        candidate_ready_plan = candidate_stage1_call.keywords["ready_plan"]
        barrier(group)
        eager_metrics = validate_outputs(
            reference,
            candidate,
            group,
            sizes=sizes,
            padded_rows=padded_rows,
        )
        flydsl_reference_metrics = validate_outputs(
            reference,
            flydsl_reference,
            group,
            sizes=sizes,
            padded_rows=padded_rows,
        )
        queue_metrics = validate_queue(
            candidate_ready_plan,
            candidate_stage1_call.args[5],
            group,
            tile_m=64,
        )
        candidate_slot = backend._workspaces[padded_rows].slots[
            (backend._workspaces[padded_rows].index - 1)
            % len(backend._workspaces[padded_rows].slots)
        ]
        workspace_metrics = validate_workspace(
            backend, hidden, topk_weights, topk_ids, sizes, group
        )
        reference_stage1_call = next(
            call for name, call in reference_calls if name == "stage1"
        )
        reference_stage2_call = next(
            call for name, call in reference_calls if name == "stage2"
        )
        flydsl_reference_stage1_call = next(
            call for name, call in flydsl_reference_calls if name == "stage1"
        )
        candidate_stage2_call = next(
            call for name, call in candidate_calls if name == "stage2"
        )

        def run_candidate_stage1_all_ready():
            candidate_ready_plan.ready.fill_(-1)
            candidate_ready_plan.expert_cursor.zero_()
            candidate_ready_plan.completed_tiles.zero_()
            candidate_ready_plan.tile_claimed.zero_()
            return candidate_stage1_call()

        strict_stage1_metrics = {}
        if args.strict_stage1_phases:
            workspace_set = backend._workspaces[padded_rows]
            strict_slot = next(
                slot
                for slot in workspace_set.slots
                if slot.ready.data_ptr()
                == candidate_ready_plan.ready.data_ptr()
            )
            strict_expected_values = strict_slot.values.clone()
            strict_expected_scales = strict_slot.scales.clone()
            barrier(group)

            def poison_strict_remote_destination():
                for chunk in range(backend.config.chunks_per_source):
                    for source in range(world):
                        if source == rank:
                            continue
                        strict_slot.value_layout[chunk, source].view(
                            torch.uint8
                        ).fill_(0xFF)
                        strict_slot.scale_layout[chunk, source].zero_()

            def reset_strict_queue():
                candidate_ready_plan.expert_cursor.zero_()
                candidate_ready_plan.completed_tiles.zero_()
                candidate_ready_plan.tile_claimed.zero_()

            def setup_strict_r():
                strict_slot.ready.fill_(strict_slot.local_ready_mask)

            def setup_strict_g():
                strict_slot.ready.fill_(-1)
                reset_strict_queue()

            def setup_strict_rg():
                strict_slot.ready.fill_(strict_slot.local_ready_mask)
                reset_strict_queue()

            def setup_strict_rg_poison():
                poison_strict_remote_destination()
                setup_strict_rg()

            def run_strict_r():
                strict_slot.launch_transport(torch.cuda.current_stream(device))

            def run_strict_g():
                return candidate_stage1_call()

            def run_strict_serial():
                strict_slot.launch_transport(torch.cuda.current_stream(device))
                return candidate_stage1_call()

            def run_strict_fused():
                current_stream = torch.cuda.current_stream(device)
                transport_stream = backend.transport_stream
                transport_stream.wait_stream(current_stream)
                with torch.cuda.stream(transport_stream):
                    strict_slot.launch_transport(transport_stream)
                output = candidate_stage1_call()
                current_stream.wait_stream(transport_stream)
                return output

            strict_r_ms = time_call_with_setup(
                run_strict_r,
                setup_strict_r,
                group,
                args.warmup,
                args.iterations,
            )
            strict_g_ms = time_call_with_setup(
                run_strict_g,
                setup_strict_g,
                group,
                args.warmup,
                args.iterations,
            )
            strict_serial_ms = time_call_with_setup(
                run_strict_serial,
                setup_strict_rg,
                group,
                args.warmup,
                args.iterations,
            )
            strict_fused_ms = time_call_with_setup(
                run_strict_fused,
                setup_strict_rg,
                group,
                args.warmup,
                args.iterations,
            )
            setup_strict_g()
            strict_g_output = clone_stage1_output(run_strict_g())
            setup_strict_rg_poison()
            strict_serial_output = clone_stage1_output(run_strict_serial())
            setup_strict_rg_poison()
            strict_fused_output = clone_stage1_output(run_strict_fused())
            barrier(group)
            strict_serial_correctness = validate_stage1_all_tensors(
                strict_g_output,
                strict_serial_output,
                group,
                "strict serial Stage1",
            )
            strict_fused_correctness = validate_stage1_all_tensors(
                strict_g_output,
                strict_fused_output,
                group,
                "strict fused Stage1",
            )
            strict_workspace_mismatches = torch.tensor(
                [
                    (
                        strict_slot.values.view(torch.uint8)
                        != strict_expected_values.view(torch.uint8)
                    ).sum(),
                    (
                        strict_slot.scales != strict_expected_scales
                    ).sum(),
                ],
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(
                strict_workspace_mismatches,
                op=dist.ReduceOp.SUM,
                group=group.device_group,
            )
            if bool(strict_workspace_mismatches.any()):
                raise AssertionError(
                    "strict Stage1 transport left poisoned payload: "
                    f"{strict_workspace_mismatches.cpu().tolist()}"
                )
            strict_status = torch.tensor(
                [strict_slot.ready[0]],
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(
                strict_status,
                op=dist.ReduceOp.MAX,
                group=group.device_group,
            )
            if int(strict_status[0]) != -1:
                raise AssertionError(
                    "strict Stage1 transport did not drain: "
                    f"ready={int(strict_status[0])}"
                )
            strict_exposed_ms = max(0.0, strict_fused_ms - strict_g_ms)
            strict_stage1_metrics = {
                "r_only_ms": strict_r_ms,
                "g_all_ready_ms": strict_g_ms,
                "r_plus_g_ms": strict_r_ms + strict_g_ms,
                "serial_r_then_g_ms": strict_serial_ms,
                "fused_r_parallel_g_ms": strict_fused_ms,
                "serial_sum_delta_ms": (
                    strict_serial_ms - strict_r_ms - strict_g_ms
                ),
                "serial_over_fused": strict_serial_ms / strict_fused_ms,
                "sum_over_fused": (
                    strict_r_ms + strict_g_ms
                ) / strict_fused_ms,
                "exposed_r_ms": strict_exposed_ms,
                "unhidden_r": (
                    strict_exposed_ms / strict_r_ms if strict_r_ms else 0.0
                ),
                "serial_correctness": strict_serial_correctness,
                "fused_correctness": strict_fused_correctness,
                "workspace_byte_mismatches": (
                    strict_workspace_mismatches.cpu().tolist()
                ),
                "ready": int(strict_status[0]),
            }

        reference_stage1_ms = time_call(
            reference_stage1_call, group, args.warmup, args.iterations
        )
        reference_stage2_ms = time_call(
            reference_stage2_call, group, args.warmup, args.iterations
        )
        flydsl_reference_stage1_ms = time_call(
            flydsl_reference_stage1_call,
            group,
            args.warmup,
            args.iterations,
        )
        candidate_stage1_ms = time_call(
            run_candidate_stage1_all_ready,
            group,
            args.warmup,
            args.iterations,
        )
        candidate_stage2_ms = time_call(
            candidate_stage2_call, group, args.warmup, args.iterations
        )
        reference_ms = time_call(
            lambda: reference_call(
                hidden, topk_weights, topk_ids, weights, padded_rows
            ),
            group,
            args.warmup,
            args.iterations,
        )
        flydsl_reference_ms = time_call(
            lambda: reference_call(
                hidden,
                topk_weights,
                topk_ids,
                weights,
                padded_rows,
                force_plain_tpag_stage1=True,
            ),
            group,
            args.warmup,
            args.iterations,
        )
        candidate_ms = time_call(
            lambda: candidate_call(
                backend, hidden, topk_weights, topk_ids, weights, sizes
            ),
            group,
            args.warmup,
            args.iterations,
        )
        graph_metrics = None
        if args.graph_replays:
            graph, graph_output = capture_candidate(
                lambda: candidate_call(
                    backend, hidden, topk_weights, topk_ids, weights, sizes
                ),
                group,
            )
            for _ in range(args.graph_replays):
                graph.replay()
            barrier(group)
            graph_metrics = validate_outputs(
                reference,
                graph_output,
                group,
                sizes=sizes,
                padded_rows=padded_rows,
            )

        if rank == 0:
            print(
                {
                    "logical_global_tokens": sum(sizes),
                    "physical_global_tokens": padded_rows * world,
                    "sizes": sizes,
                    "eager_correctness": eager_metrics,
                    "flydsl_reference_correctness": flydsl_reference_metrics,
                    "workspace_correctness": workspace_metrics,
                    "queue_correctness": queue_metrics,
                    "graph_correctness": graph_metrics,
                    "reference_bf16_ag_plus_moe_ms": reference_ms,
                    "reference_bf16_ag_plus_flydsl_moe_ms": flydsl_reference_ms,
                    "fused_quantized_ag_stage1_plus_stage2_ms": candidate_ms,
                    "reference_stage1_ms": reference_stage1_ms,
                    "reference_flydsl_stage1_ms": flydsl_reference_stage1_ms,
                    "reference_stage2_ms": reference_stage2_ms,
                    "candidate_stage1_all_ready_ms": candidate_stage1_ms,
                    "candidate_stage2_ms": candidate_stage2_ms,
                    "candidate_non_gemm_residual_ms": (
                        candidate_ms - candidate_stage1_ms - candidate_stage2_ms
                    ),
                    "speedup": reference_ms / candidate_ms,
                    "speedup_vs_flydsl_reference": (
                        flydsl_reference_ms / candidate_ms
                    ),
                    "graph_replays": args.graph_replays,
                    "strict_stage1_phases": strict_stage1_metrics,
                },
                flush=True,
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

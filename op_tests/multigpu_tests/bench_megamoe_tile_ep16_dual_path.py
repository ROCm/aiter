# SPDX-License-Identifier: MIT
"""EP16 MORI-baseline versus hierarchical-cascade benchmark harness.

The baseline is executable today.  Candidate execution is loaded through an
explicit factory so the current cascade can be refactored into reusable stages
without duplicating its CCO lifecycle in this file.  Both paths receive the
same :class:`SharedInputs`; timing never runs a Gloo numerical reference.

Default target: tokens/rank=128, H=7168, I=3072, E=896 (56/rank), topk=16,
SiLU, A4W4.  Routes use the exact ``rank-balanced-hot`` generator from
``bench_mega_moe_v2.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import struct
import time
from typing import Callable, Protocol

import torch
import torch.distributed as dist

import aiter
from aiter.ops.flydsl.kernels.megamoe_tile import (
    K3DispatchWireLayout,
    K3PartialWireLayout,
    PreparedA4W4Weights,
)


BASELINE_STAGES = ("dispatch", "local_a4w4", "combine")
CANDIDATE_STAGES = (
    "dispatch_pack",
    "inter_node_dispatch",
    "lsa_fanout_unpack_sort",
    "h1",
    "h2",
    "node_reduce",
    "partial_pack_return",
    "final_combine",
)

TWO_KERNEL_ROUTE_PATTERNS = (
    "rank-balanced-hot",
    "paired-rank-half-remote",
    "paired-rank-local-only",
    "permuted-arbitrary-topk",
)

# Deliberately non-adjacent slot groups. Every token has eight local-node and
# eight remote-node routes, multiple slots map to each selected rank, and at
# least two non-adjacent slots for every selected rank repeat the exact same
# expert ID. Keep this fixture here so Stage1 stress and real-GMM validation
# cannot silently diverge.
_ARBITRARY_REMOTE = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
_ARBITRARY_RANK_DELTA = (0, 3, 2, 3, 0, 5, 2, 5, 0, 3, 6, 7, 0, 5, 6, 7)
_ARBITRARY_EXPERT_VARIANT = (0, 1, 2, 1, 3, 4, 2, 4, 0, 5, 6, 7, 3, 4, 6, 7)


@dataclass(frozen=True)
class BenchmarkShape:
    tokens: int = 128
    hidden: int = 7168
    inter: int = 3072
    experts: int = 896
    topk: int = 16
    ep_size: int = 16
    gpus_per_node: int = 8
    activation: str = "silu"

    def validate(self) -> None:
        if self.ep_size != 16 or self.gpus_per_node != 8:
            raise ValueError("this benchmark targets EP16 as two 8-GPU nodes")
        if self.experts != 896:
            raise ValueError(
                "this benchmark requires E=896; E=384 is the DSV4 expert "
                "configuration and must not be mixed into this run"
            )
        if self.experts % self.ep_size:
            raise ValueError("experts must divide EP size")
        if self.local_experts != 56:
            raise ValueError("E896/EP16 must produce exactly 56 experts/rank")
        if self.topk != self.ep_size:
            raise ValueError(
                "the current EP16 specialization requires topk=16"
            )
        if self.activation != "silu":
            raise ValueError("the target benchmark activation is SiLU")

    @property
    def local_experts(self) -> int:
        return self.experts // self.ep_size


@dataclass(frozen=True)
class CandidateWirePlan:
    chunk_bytes: int
    num_qp: int
    ring_depth: int
    dispatch_batch_per_qp: int
    dispatch_record_bytes: int
    dispatch_records_per_chunk: int
    dispatch_active_bytes_per_chunk: int
    dispatch_records_per_rank: int
    dispatch_chunks_per_rank: int
    partial_batch_per_qp: int
    partial_record_bytes: int
    partial_records_per_chunk: int
    partial_active_bytes_per_chunk: int
    partial_records_per_rank: int
    partial_chunks_per_rank: int
    remote_fanout_entries_per_rank: int

    @classmethod
    def for_shape(
        cls,
        shape: BenchmarkShape,
        *,
        chunk_bytes: int = 512 * 1024,
        num_qp: int = 4,
        ring_depth: int = 8,
        dispatch_batch_per_qp: int = 32,
        partial_batch_per_qp: int = 8,
    ) -> "CandidateWirePlan":
        dispatch = K3DispatchWireLayout(
            hidden_bytes=shape.hidden // 2,
            scale_bytes=shape.hidden // 32,
            topk=shape.topk,
        )
        partial = K3PartialWireLayout(hidden=shape.hidden)
        # rank-balanced topk16 over EP16 selects one expert on every rank. One
        # token record is sent to the remote-node proxy, then expanded to all
        # eight destination LSA ranks on that node.
        dispatch_records = shape.tokens
        partial_records = shape.tokens
        dispatch_per_chunk = num_qp * dispatch_batch_per_qp
        partial_per_chunk = num_qp * partial_batch_per_qp
        dispatch_active_bytes = dispatch_per_chunk * dispatch.record_bytes
        partial_active_bytes = partial_per_chunk * partial.record_bytes
        if dispatch_active_bytes > chunk_bytes:
            raise ValueError("dispatch QP batch exceeds one chunk")
        if partial_active_bytes > chunk_bytes:
            raise ValueError("partial QP batch exceeds one chunk")
        return cls(
            chunk_bytes=chunk_bytes,
            num_qp=num_qp,
            ring_depth=ring_depth,
            dispatch_batch_per_qp=dispatch_batch_per_qp,
            dispatch_record_bytes=dispatch.record_bytes,
            dispatch_records_per_chunk=dispatch_per_chunk,
            dispatch_active_bytes_per_chunk=dispatch_active_bytes,
            dispatch_records_per_rank=dispatch_records,
            dispatch_chunks_per_rank=math.ceil(
                dispatch_records / dispatch_per_chunk
            ),
            partial_batch_per_qp=partial_batch_per_qp,
            partial_record_bytes=partial.record_bytes,
            partial_records_per_chunk=partial_per_chunk,
            partial_active_bytes_per_chunk=partial_active_bytes,
            partial_records_per_rank=partial_records,
            partial_chunks_per_rank=math.ceil(
                partial_records / partial_per_chunk
            ),
            remote_fanout_entries_per_rank=shape.tokens
            * shape.gpus_per_node,
        )


@dataclass
class SharedInputs:
    x: torch.Tensor
    a_quant: torch.Tensor
    a_scale: torch.Tensor
    route_weights: torch.Tensor
    topk_ids: torch.Tensor
    prepared_weights: PreparedA4W4Weights
    local_expert_mask: torch.Tensor


@dataclass
class IterationTiming:
    stage_us: dict[str, float]
    gpu_e2e_us: float
    host_critical_us: float


class TimedPath(Protocol):
    name: str
    stage_names: tuple[str, ...]

    def run_iteration(self, timer: "HipStageTimer") -> torch.Tensor: ...


class HipStageTimer:
    """One-iteration HIP-event recorder with host critical-path timing."""

    def __init__(
        self,
        device: torch.device,
        stage_names: tuple[str, ...] = (),
    ):
        self.device = device
        self._records: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._stage_events = {
            name: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for name in stage_names
        }
        self._seen_stage_names: set[str] = set()
        self._host_start = 0.0
        # Construct every known timing event before begin_iteration().  Event
        # construction on an otherwise empty stream can otherwise become a
        # host-enqueue bubble inside a very short measured GPU interval.
        self._gpu_start = torch.cuda.Event(enable_timing=True)
        self._gpu_end = torch.cuda.Event(enable_timing=True)
        self._prime_events()

    def _prime_events(self) -> None:
        """Materialize lazy HIP event handles outside every timed interval."""

        events = [self._gpu_start, self._gpu_end]
        for start, end in self._stage_events.values():
            events.extend((start, end))
        stream = torch.cuda.current_stream(self.device)
        for event in events:
            event.record(stream)
        stream.synchronize()

    def begin_iteration(self) -> None:
        self._records.clear()
        self._seen_stage_names.clear()
        self._host_start = time.perf_counter()
        self._gpu_start.record()

    def stage(self, name: str, body: Callable[[], object]):
        if name in self._seen_stage_names:
            raise RuntimeError(f"timing stage {name!r} was recorded twice")
        self._seen_stage_names.add(name)
        pair = self._stage_events.get(name)
        if pair is None:
            pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            self._stage_events[name] = pair
        start, end = pair
        start.record()
        result = body()
        end.record()
        self._records.append((name, start, end))
        return result

    def finish_iteration(self) -> IterationTiming:
        self._gpu_end.record()
        torch.cuda.synchronize(self.device)
        host_us = (time.perf_counter() - self._host_start) * 1.0e6
        return IterationTiming(
            stage_us={
                name: float(start.elapsed_time(end) * 1000.0)
                for name, start, end in self._records
            },
            gpu_e2e_us=float(
                self._gpu_start.elapsed_time(self._gpu_end) * 1000.0
            ),
            host_critical_us=host_us,
        )


def _load_factory(spec: str):
    if ":" not in spec:
        raise ValueError("candidate factory must be module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def _setup_dist(needs_mori: bool):
    rank = int(__import__("os").environ["RANK"])
    world = int(__import__("os").environ["WORLD_SIZE"])
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # MORI owns the GPU/IB transport. Benchmark coordination and rank-max
    # statistics use CPU tensors only, so do not create an unrelated NCCL/RCCL
    # fabric that may select different RoCE rails/GID indices.
    dist.init_process_group("gloo")
    if needs_mori:
        import mori.shmem as ms
        import torch._C._distributed_c10d as c10d

        c10d._register_process_group("default", dist.group.WORLD)
        ms.shmem_torch_process_group_init("default")
    return rank, world, local_rank, device


def _permuted_arbitrary_topk_cpu(
    shape: BenchmarkShape, source_rank: int
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    """Build the shared deterministic arbitrary-Top-K CPU fixture."""

    token = torch.arange(shape.tokens, dtype=torch.int64).view(-1, 1)
    slot = torch.arange(shape.topk, dtype=torch.int64).view(1, -1)
    source_node = int(source_rank) // shape.gpus_per_node
    source_local_rank = int(source_rank) % shape.gpus_per_node
    remote = torch.tensor(_ARBITRARY_REMOTE, dtype=torch.int64).view(1, -1)
    rank_delta = torch.tensor(
        _ARBITRARY_RANK_DELTA, dtype=torch.int64
    ).view(1, -1)
    owner_local = (source_local_rank + rank_delta) % shape.gpus_per_node
    owner_node = torch.where(
        remote != 0,
        torch.full_like(remote, 1 - source_node),
        torch.full_like(remote, source_node),
    )
    owner = (owner_node * shape.gpus_per_node + owner_local).expand(
        shape.tokens, -1
    )

    expert_variant = torch.tensor(
        _ARBITRARY_EXPERT_VARIANT, dtype=torch.int64
    ).view(1, -1)
    expert_base = (token * 7 + int(source_rank) * 3) % shape.local_experts
    local_expert = (expert_base + expert_variant) % shape.local_experts
    topk_ids = (owner * shape.local_experts + local_expert).to(torch.int32)

    # Preserve each original Top-K slot: all weights are distinct within one
    # token and depend on both source rank and token.
    numerators = (
        (token * 17 + slot * 29 + int(source_rank) * 11) % 251 + 1
    ).to(torch.float32)
    route_weights = numerators / numerators.sum(dim=1, keepdim=True)

    rank_slot_masks: list[list[int]] = []
    for token_index in range(shape.tokens):
        masks = [0] * shape.ep_size
        ids_row = topk_ids[token_index].tolist()
        weights_row = route_weights[token_index]
        for topk_slot, expert in enumerate(ids_row):
            masks[int(expert) // shape.local_experts] |= 1 << topk_slot
        if sum(int(mask).bit_count() for mask in masks) != shape.topk:
            raise AssertionError("arbitrary fixture lost a Top-K slot")
        if sum(int(mask != 0) for mask in masks) != 6:
            raise AssertionError("arbitrary fixture must select six ranks")
        local_routes = sum(
            int(mask).bit_count()
            for owner_rank, mask in enumerate(masks)
            if owner_rank // shape.gpus_per_node == source_node
        )
        if local_routes != shape.topk // 2:
            raise AssertionError(
                "arbitrary fixture must split routes equally across nodes"
            )
        non_adjacent_ranks = sum(
            int(
                mask != 0
                and any(
                    (mask & (1 << left))
                    and (mask & (1 << right))
                    and right - left > 1
                    for left in range(shape.topk)
                    for right in range(left + 1, shape.topk)
                )
            )
            for mask in masks
        )
        if non_adjacent_ranks != 6:
            raise AssertionError("arbitrary fixture rank slots must be non-adjacent")
        if torch.unique(weights_row).numel() != shape.topk:
            raise AssertionError("arbitrary fixture weights must be slot-distinct")
        for owner_rank, mask in enumerate(masks):
            slots = [
                topk_slot
                for topk_slot in range(shape.topk)
                if mask & (1 << topk_slot)
            ]
            if not slots:
                continue
            experts = [int(ids_row[topk_slot]) for topk_slot in slots]
            if len(set(experts)) == len(experts):
                raise AssertionError(
                    f"rank {owner_rank} has no repeated exact expert"
                )
        rank_slot_masks.append(masks)
    return topk_ids, route_weights, rank_slot_masks


def _permuted_arbitrary_destination_oracle(
    shape: BenchmarkShape, destination_rank: int
) -> dict[str, object]:
    """Summarize all source routes received by one destination rank."""

    expert_count = [0] * shape.local_experts
    metadata = []
    unique_sources = set()
    for source_rank in range(shape.ep_size):
        ids, weights, _ = _permuted_arbitrary_topk_cpu(shape, source_rank)
        weight_bits = weights.contiguous().view(torch.int32)
        for token in range(shape.tokens):
            source = source_rank * shape.tokens + token
            for topk_slot in range(shape.topk):
                expert = int(ids[token, topk_slot].item())
                if expert // shape.local_experts != destination_rank:
                    continue
                local_expert = expert % shape.local_experts
                expert_count[local_expert] += 1
                unique_sources.add(source)
                metadata.append(
                    (
                        source | (topk_slot << 24),
                        local_expert,
                        int(weight_bits[token, topk_slot].item()) & 0xFFFFFFFF,
                    )
                )
    metadata.sort()
    metadata_sha = hashlib.sha256()
    for packed_source, local_expert, weight_bits in metadata:
        metadata_sha.update(
            struct.pack("<III", packed_source, local_expert, weight_bits)
        )
    tiles = sum((count + 31) // 32 for count in expert_count)
    return {
        "routes": len(metadata),
        "tiles": tiles,
        "expert_count": expert_count,
        "unique_sources": len(unique_sources),
        "metadata_sha256": metadata_sha.hexdigest(),
    }


def _shared_inputs(
    shape: BenchmarkShape,
    rank: int,
    world: int,
    device: torch.device,
    *,
    route_pattern: str = "rank-balanced-hot",
    direct_packed_weights: bool = False,
) -> SharedInputs:
    # Reuse the exact MegaMoE input generator, then optionally replace only
    # routing metadata with deterministic duplicate-rank fixtures.
    from op_tests.multigpu_tests.bench_mega_moe_v2 import make_inputs

    x, route_weights, topk_ids = make_inputs(
        shape.tokens,
        rank,
        world,
        shape.hidden,
        shape.experts,
        shape.topk,
        "rank-balanced-hot",
        0.6,
        device,
    )
    if route_pattern in (
        "paired-rank-half-remote",
        "paired-rank-local-only",
    ):
        token = torch.arange(shape.tokens, dtype=torch.int64).view(-1, 1)
        slot = torch.arange(shape.topk, dtype=torch.int64).view(1, -1)
        source_node = rank // shape.gpus_per_node
        if route_pattern == "paired-rank-half-remote":
            target_node = torch.where(
                (token & 1) == 0,
                torch.full_like(token, source_node),
                torch.full_like(token, 1 - source_node),
            )
        else:
            target_node = torch.full_like(token, source_node)
        owner = target_node * shape.gpus_per_node + slot // 2
        pair_member = slot & 1
        local_expert = (
            token % (shape.local_experts // 2)
        ) * 2 + pair_member
        topk_ids = (
            owner * shape.local_experts + local_expert
        ).to(torch.int32).to(device)
        route_weights = (
            ((slot + token) % shape.topk + 1).to(torch.float32)
            / float(shape.topk * (shape.topk + 1) // 2)
        ).to(device)
    elif route_pattern == "permuted-arbitrary-topk":
        topk_ids_cpu, route_weights_cpu, _ = _permuted_arbitrary_topk_cpu(
            shape, rank
        )
        topk_ids = topk_ids_cpu.to(device)
        route_weights = route_weights_cpu.to(device)
    elif route_pattern != "rank-balanced-hot":
        raise ValueError(f"unsupported two-kernel route_pattern={route_pattern!r}")
    generator = torch.Generator(device=device).manual_seed(90_000 + rank)
    # Prepare the two packed weight sets sequentially.  Holding both BF16
    # sources while FP4 quantization materializes its FP32 working tensor adds
    # several GiB of avoidable peak memory at this production shape.
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    if direct_packed_weights:
        # Constant packed values are invariant under the weight-layout
        # permutations and avoid the large BF16->FP32 quantization workspace.
        # This is a memory-light correctness mode for contended systems, not a
        # performance workload.
        from aiter.utility import dtypes

        w1q = torch.full(
            (shape.local_experts, 2 * shape.inter, shape.hidden // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        ).view(dtypes.fp4x2)
        w1s = torch.full(
            (shape.local_experts, 2 * shape.inter, shape.hidden // 32),
            120,
            dtype=torch.uint8,
            device=device,
        ).view(dtypes.fp8_e8m0)
        w2q = torch.full(
            (shape.local_experts, shape.hidden, shape.inter // 2),
            0x11,
            dtype=torch.uint8,
            device=device,
        ).view(dtypes.fp4x2)
        w2s = torch.full(
            (shape.local_experts, shape.hidden, shape.inter // 32),
            120,
            dtype=torch.uint8,
            device=device,
        ).view(dtypes.fp8_e8m0)
    else:
        w1 = torch.randn(
            (shape.local_experts, 2 * shape.inter, shape.hidden),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w1.mul_(shape.hidden**-0.25)
        w1q, w1s = per_1x32_f4_quant(w1, shuffle=False)
        del w1
        w1q = shuffle_weight(w1q, layout=(16, 16))
        w1s = e8m0_shuffle(w1s)
        torch.cuda.empty_cache()

        w2 = torch.randn(
            (shape.local_experts, shape.hidden, shape.inter),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w2.mul_(shape.inter**-0.25)
        w2q, w2s = per_1x32_f4_quant(w2, shuffle=False)
        del w2
        w2q = shuffle_weight_a16w4(w2q, 16, False)
        w2s = shuffle_scale_a16w4(w2s, shape.local_experts, False)
        torch.cuda.empty_cache()
    prepared = PreparedA4W4Weights(
        w1q,
        w1s,
        w2q,
        w2s,
        shape.local_experts,
        shape.hidden,
        shape.inter,
        None,
        None,
    )

    a_quant, a_scale = per_1x32_f4_quant(x, shuffle=False)
    local_mask = torch.zeros(
        shape.experts, dtype=torch.int32, device=device
    )
    first = rank * shape.local_experts
    local_mask[first : first + shape.local_experts] = 1
    return SharedInputs(
        x,
        a_quant,
        a_scale,
        route_weights,
        topk_ids,
        prepared,
        local_mask,
    )


def _prepare_local_a4w4(
    dispatched_q: torch.Tensor,
    dispatched_scale: torch.Tensor,
    dispatched_ids: torch.Tensor,
    dispatched_weights: torch.Tensor,
    shape: BenchmarkShape,
    rank: int,
    *,
    validate_routes: bool,
    output_workspace: dict[str, torch.Tensor] | None = None,
    expand_local_routes: bool = False,
) -> dict[str, object]:
    """Select/sort this rank's routes from each deduplicated token record."""

    from aiter.fused_moe import moe_sorting
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    rows = int(dispatched_q.shape[0])
    owners = torch.div(
        dispatched_ids,
        shape.local_experts,
        rounding_mode="floor",
    )
    local_route = owners == rank
    if validate_routes and not expand_local_routes and torch.count_nonzero(
        local_route.sum(dim=1) != 1
    ).item():
        raise RuntimeError("rank-balanced dispatch row must have one local route")
    if expand_local_routes:
        if validate_routes and torch.count_nonzero(
            local_route.sum(dim=1) < 1
        ).item():
            raise RuntimeError(
                "deduplicated MORI row has no route owned by this rank"
            )
        route_rows, route_slots = torch.nonzero(local_route, as_tuple=True)
        route_rows = route_rows.to(torch.int64)
        route_slots = route_slots.to(torch.int64)
        route_count = int(route_rows.numel())
        local_ids = (
            dispatched_ids[route_rows, route_slots]
            - rank * shape.local_experts
        ).to(torch.int32).view(route_count, 1)
        local_weights = dispatched_weights[
            route_rows, route_slots
        ].float().view(route_count, 1)
        route_sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
            local_ids,
            local_weights,
            shape.local_experts,
            shape.hidden,
            torch.bfloat16,
            32,
            accumulate=False,
        )
        packed_route = route_sorted_ids.to(torch.int64) & 0xFFFFFFFF
        route_index = packed_route & 0x00FFFFFF
        valid_sorted = route_index < route_count
        safe_route = torch.where(
            valid_sorted, route_index, torch.zeros_like(route_index)
        )
        source_rows = route_rows[safe_route]
        source_slots = route_slots[safe_route]
        sorted_ids = torch.where(
            valid_sorted,
            (source_slots << 24) | source_rows,
            packed_route,
        ).to(torch.int32).contiguous()
        m_indices = torch.where(
            valid_sorted, source_rows, torch.zeros_like(source_rows)
        ).to(torch.int32).contiguous()
        scale_input = dispatched_scale[route_rows].contiguous()
        scale_sorted_ids = route_sorted_ids
        scale_input_rows = route_count
        gmm1_topk = max(
            1, int(local_route.sum(dim=1).max().item())
        )
        gmm2_topk = shape.topk
        epilog = "atomic"
    else:
        route_count = rows
        slot = local_route.to(torch.int32).argmax(dim=1).to(torch.int64)
        row = torch.arange(rows, device=dispatched_ids.device)
        local_ids = (
            dispatched_ids[row, slot] - rank * shape.local_experts
        ).to(torch.int32).view(rows, 1)
        local_weights = dispatched_weights[row, slot].float().view(rows, 1)
        sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
            local_ids,
            local_weights,
            shape.local_experts,
            shape.hidden,
            torch.bfloat16,
            32,
            accumulate=False,
        )
        m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()
        scale_input = dispatched_scale
        scale_sorted_ids = sorted_ids
        scale_input_rows = rows
        gmm1_topk = 1
        gmm2_topk = 1
        epilog = "reduce"
    a1ss = moe_mxfp4_sort(
        scale_input.view(scale_input_rows, 1, shape.hidden // 32),
        scale_sorted_ids,
        nvalid,
        scale_input_rows,
        32,
    )
    max_sorted = int(sorted_ids.shape[0])
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((shape.inter // 32) + 7) // 8 * 8
    expected_workspace = {
        "inter_q": ((max_sorted, shape.inter // 2), torch.uint8),
        "inter_s": ((scale_rows * scale_cols,), torch.uint8),
        "hidden_dummy": ((rows, shape.hidden), torch.bfloat16),
    }
    if output_workspace is None:
        output_workspace = {
            name: torch.zeros(expected_shape, dtype=dtype, device=dispatched_q.device)
            for name, (expected_shape, dtype) in expected_workspace.items()
        }
    else:
        for name, (expected_shape, dtype) in expected_workspace.items():
            tensor = output_workspace.get(name)
            if tensor is None:
                raise ValueError(f"local A4W4 workspace is missing {name}")
            if (
                tuple(tensor.shape) != expected_shape
                or tensor.dtype != dtype
                or tensor.device != dispatched_q.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"local A4W4 workspace {name} is incompatible")
    inter_q = output_workspace["inter_q"]
    inter_s = output_workspace["inter_s"]
    hidden_dummy = output_workspace["hidden_dummy"]
    return {
        "rows": rows,
        "route_count": route_count,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_eids": sorted_eids,
        "nvalid": nvalid,
        "m_indices": m_indices,
        "a_quant": dispatched_q,
        "a_scale_sorted": a1ss,
        "inter_q": inter_q,
        "inter_s": inter_s,
        "hidden_dummy": hidden_dummy,
        "gmm1_topk": gmm1_topk,
        "gmm2_topk": gmm2_topk,
        "epilog": epilog,
    }


def _run_local_h1(
    context: dict[str, object],
    prepared: PreparedA4W4Weights,
    shape: BenchmarkShape,
) -> None:
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1

    flydsl_mxfp4_gemm1(
        a_quant=context["a_quant"],
        a_scale_sorted_shuffled=context["a_scale_sorted"],
        w1_u8=prepared.w1.view(torch.uint8),
        w1_scale_u8=prepared.w1_scale.view(torch.uint8),
        sorted_expert_ids=context["sorted_eids"],
        cumsum_tensor=context["nvalid"],
        m_indices=context["m_indices"],
        inter_sorted_quant=context["inter_q"],
        inter_sorted_shuffled_scale=context["inter_s"],
        hidden_states=context["hidden_dummy"],
        n_tokens=context["rows"],
        BM=32,
        use_nt=True,
        inline_quant=False,
        NE=shape.local_experts,
        D_HIDDEN=shape.hidden,
        D_INTER=shape.inter,
        topk=context["gmm1_topk"],
        act="silu",
    )


def _run_local_h2(
    context: dict[str, object],
    prepared: PreparedA4W4Weights,
    shape: BenchmarkShape,
) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2

    rows = int(context["rows"])
    out = torch.zeros(
        (rows, shape.hidden),
        dtype=torch.bfloat16,
        device=context["inter_q"].device,
    )
    mxfp4_moe_gemm2(
        inter_sorted_quant=context["inter_q"],
        inter_sorted_shuffled_scale=context["inter_s"],
        w2_u8=prepared.w2.view(torch.uint8),
        w2_scale_u8=prepared.w2_scale.view(torch.uint8),
        sorted_expert_ids=context["sorted_eids"],
        cumsum_tensor=context["nvalid"],
        sorted_token_ids=context["sorted_ids"],
        sorted_weights=context["sorted_weights"],
        out=out,
        M_logical=rows,
        max_sorted=context["inter_q"].shape[0],
        NE=shape.local_experts,
        D_HIDDEN=shape.hidden,
        D_INTER=shape.inter,
        topk=context["gmm2_topk"],
        BM=32,
        BN=128,
        BK=128,
        a_dtype="fp4",
        # Rank-balanced uses one local route/record and can store directly.
        # Arbitrary routing preserves all local slots and accumulates their
        # independently weighted expert contributions into the source row.
        epilog=context["epilog"],
        SBM=32,
        HIDDEN_MAX=shape.hidden,
        INTER_MAX=shape.inter,
    )
    return out


def _explicit_local_a4w4(
    dispatched_q: torch.Tensor,
    dispatched_scale: torch.Tensor,
    dispatched_ids: torch.Tensor,
    dispatched_weights: torch.Tensor,
    prepared: PreparedA4W4Weights,
    shape: BenchmarkShape,
    rank: int,
    *,
    validate_routes: bool,
    expand_local_routes: bool = False,
) -> torch.Tensor:
    context = _prepare_local_a4w4(
        dispatched_q,
        dispatched_scale,
        dispatched_ids,
        dispatched_weights,
        shape,
        rank,
        validate_routes=validate_routes,
        expand_local_routes=expand_local_routes,
    )
    _run_local_h1(context, prepared, shape)
    return _run_local_h2(context, prepared, shape)


class MoriBaselinePath:
    name = "mori_baseline"
    stage_names = BASELINE_STAGES

    def __init__(
        self,
        shape: BenchmarkShape,
        shared: SharedInputs,
        rank: int,
        world: int,
        *,
        valid_recv: int | None = None,
    ):
        import mori

        kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1LL
        config = mori.ops.EpDispatchCombineConfig(
            data_type=shared.a_quant.dtype,
            rank=rank,
            world_size=world,
            hidden_dim=shape.hidden,
            scale_dim=shape.hidden // 32,
            scale_type_size=1,
            max_token_type_size=torch.bfloat16.itemsize,
            max_num_inp_token_per_rank=max(128, shape.tokens),
            max_total_recv_tokens=0,
            num_experts_per_rank=shape.local_experts,
            num_experts_per_token=shape.topk,
            warp_num_per_block=8,
            block_num=256,
            kernel_type=kernel_type,
            rdma_block_num=128,
            gpu_per_node=shape.gpus_per_node,
            quant_type="none",
        )
        self.shape = shape
        self.shared = shared
        self.rank = rank
        self.op = mori.ops.EpDispatchCombineOp(config)
        self._dispatch = None
        self._local_full = None
        self.valid_recv = (
            shape.tokens * world if valid_recv is None else int(valid_recv)
        )
        if not 0 <= self.valid_recv <= shape.tokens * world:
            raise ValueError("valid_recv is outside MORI receive capacity")
        self.validate_routes = True

    def _dispatch_stage(self):
        self._dispatch = self.op.dispatch(
            self.shared.a_quant,
            self.shared.route_weights,
            self.shared.a_scale,
            self.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=8,
        )
        return self._dispatch

    def _compute_stage(self):
        if self._dispatch is None:
            raise RuntimeError("dispatch stage has not run")
        dispatched, weights, scales, ids, _ = self._dispatch
        valid = self.valid_recv
        if valid == 0:
            self._local_full = torch.zeros(
                (dispatched.shape[0], self.shape.hidden),
                dtype=torch.bfloat16,
                device=dispatched.device,
            )
            return self._local_full
        local = _explicit_local_a4w4(
            dispatched[:valid],
            scales[:valid],
            ids[:valid],
            weights[:valid],
            self.shared.prepared_weights,
            self.shape,
            self.rank,
            validate_routes=self.validate_routes,
            expand_local_routes=getattr(
                self, "expand_local_routes", False
            ),
        )
        self._local_full = torch.zeros(
            (dispatched.shape[0], self.shape.hidden),
            dtype=torch.bfloat16,
            device=dispatched.device,
        )
        self._local_full[:valid].copy_(local)
        return self._local_full

    def _combine_stage(self):
        if self._local_full is None:
            raise RuntimeError("local compute stage has not run")
        result = self.op.combine(
            self._local_full,
            None,
            self.shared.topk_ids,
            block_num=256,
            rdma_block_num=128,
            warp_per_block=4,
        )
        return result[0] if isinstance(result, tuple) else result

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        timer.stage("dispatch", self._dispatch_stage)
        timer.stage("local_a4w4", self._compute_stage)
        return timer.stage("combine", self._combine_stage)

    def prime_and_check(self) -> torch.Tensor:
        dispatched = self._dispatch_stage()
        recv_count = int(dispatched[4].item())
        if recv_count != self.valid_recv:
            raise AssertionError(
                f"rank-balanced recv_count={recv_count}, expected {self.valid_recv}"
            )
        local = self._compute_stage()
        output = self._combine_stage()
        torch.cuda.synchronize(output.device)
        if output.shape[0] < self.shape.tokens:
            raise AssertionError("combine output is shorter than local tokens")
        if not torch.isfinite(output[: self.shape.tokens].float()).all():
            raise AssertionError("baseline warmup produced non-finite output")
        if local.shape[1] != self.shape.hidden:
            raise AssertionError("local partial has the wrong hidden dimension")
        self.validate_routes = False
        return output


class CascadeCandidatePath:
    """Reusable staged form of the CCO/LSA cascade for benchmark bring-up."""

    name = "cco_lsa_candidate"
    stage_names = CANDIDATE_STAGES

    def __init__(
        self,
        shape: BenchmarkShape,
        shared: SharedInputs,
        rank: int,
        world: int,
        local_rank: int,
        device: torch.device,
        wire: CandidateWirePlan,
    ):
        from mori.cco import (
            Communicator,
            CCODevCommRequirements,
            GDA_CONNECTION_RAIL,
            UniqueId,
        )
        from mori.tensor_utils import from_gpu_ptr
        from aiter.ops.flydsl.kernels.megamoe_tile import cco
        from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
            build_dispatch_fanout_lsa,
            build_dispatch_record_module,
            build_partial_record_module,
            compile_final_combine,
            compile_node_partial_reduce_lsa,
            compile_rank_partial_epoch_gate_lsa,
        )
        from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

        if wire.dispatch_record_bytes != 4096 or wire.partial_record_bytes != 14592:
            raise ValueError("candidate factory requires the H7168 wire geometry")
        self.shape = shape
        self.shared = shared
        self.rank = rank
        self.world = world
        self.local_rank = local_rank
        self.device = device
        self.stream = torch.cuda.current_stream(device)
        self.node = rank // shape.gpus_per_node
        self.remote_node = 1 - self.node
        self.remote_rank = self.remote_node * shape.gpus_per_node + local_rank
        self.global_tokens = shape.tokens * world
        self.wire = wire
        self.epoch = 0

        uid_obj = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
        dist.broadcast_object_list(uid_obj, src=0)
        uid = UniqueId.from_bytes(uid_obj[0])
        self._comm_context = Communicator.init(
            world, rank, uid, per_rank_vmm=128 * 1024 * 1024
        )
        self.comm = self._comm_context.__enter__()
        self.layout = HierCcoArenaLayout.create(
            ring_depth=wire.ring_depth,
            num_qp=wire.num_qp,
            chunk_bytes=wire.chunk_bytes,
            max_m_tiles=256,
            max_source_tokens=self.global_tokens,
            max_h1_n_blocks=32,
            max_fanout_records=self.global_tokens,
            fanout_record_bytes=wire.dispatch_record_bytes,
        )
        rank_partial_bytes = self.global_tokens * shape.hidden * 2
        self.rank_partial_offset = (
            self.layout.total_bytes + 4095
        ) // 4096 * 4096
        total_bytes = self.rank_partial_offset + rank_partial_bytes
        if total_bytes >= 128 * 1024 * 1024:
            raise AssertionError("candidate registered window exceeds 128 MiB")
        self.memory = self.comm.alloc_mem(total_bytes)
        self.window = self.comm.register_window(
            self.memory.ptr, self.memory.size
        )
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_RAIL
        reqs.gda_context_count = wire.num_qp
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.lsa_barrier_count = 0
        reqs.rail_gda_barrier_count = 0
        reqs.barrier_count = 0
        self.dc = self.comm.create_dev_comm(reqs)
        self.rank_partial = from_gpu_ptr(
            self.window.local_ptr + self.rank_partial_offset,
            (self.global_tokens, shape.hidden),
            torch.bfloat16,
        )
        cco.zero_window(self.window.local_ptr, total_bytes)
        self.comm.barrier()

        self.h1_sidecar = cco.CcoStage1Sidecar.create(
            self.layout,
            batch_per_qp=wire.dispatch_batch_per_qp,
            segment_bytes=wire.dispatch_record_bytes,
            team=cco.TEAM_RAIL,
        )
        self.h2_sidecar = cco.CcoStage2ReturnSidecar.create(
            self.layout,
            batch_per_qp=wire.partial_batch_per_qp,
            record_bytes=wire.partial_record_bytes,
            team=cco.TEAM_RAIL,
        )
        if self.h1_sidecar.module.payload_bytes != wire.dispatch_active_bytes_per_chunk:
            raise AssertionError("H1 sidecar payload does not match active chunk")
        if self.h2_sidecar.module.payload_bytes != wire.partial_active_bytes_per_chunk:
            raise AssertionError("H2 sidecar payload does not match active chunk")
        if self.h2_sidecar.module.payload_bytes > wire.chunk_bytes:
            raise AssertionError("H2 sidecar payload exceeds aggregate chunk")
        self.dispatch_record = build_dispatch_record_module(
            shape.hidden, shape.topk
        )
        self.fanout = build_dispatch_fanout_lsa(
            self.layout, node_ranks=shape.gpus_per_node
        )
        self.fanout_local_slot_base = self.local_rank * shape.tokens
        self.fanout_remote_slot_base = (
            shape.tokens * shape.gpus_per_node
            + self.fanout_local_slot_base
        )
        self.fanout.validate_broadcast_plan(
            shape.tokens,
            self.fanout_local_slot_base,
            self.fanout_remote_slot_base,
        )
        self.partial_record = build_partial_record_module(shape.hidden)
        self.node_reduce = compile_node_partial_reduce_lsa(
            D_HIDDEN=shape.hidden,
            NUM_RANKS=shape.gpus_per_node,
            output_dtype="bf16",
        )
        self.rank_partial_epoch_gate = compile_rank_partial_epoch_gate_lsa(
            NUM_RANKS=shape.gpus_per_node
        )
        self.final_combine = compile_final_combine(
            D_HIDDEN=shape.hidden,
            local_dtype="bf16",
            remote_dtype="bf16",
            output_dtype="bf16",
        )
        self.source_ids = torch.arange(
            rank * shape.tokens,
            (rank + 1) * shape.tokens,
            dtype=torch.int32,
            device=device,
        )
        self.route_masks = torch.full(
            (shape.tokens,),
            (1 << shape.topk) - 1,
            dtype=torch.int64,
            device=device,
        )
        self.pack_error = torch.zeros(1, dtype=torch.int32, device=device)
        self.fanout_error = torch.zeros(1, dtype=torch.int32, device=device)
        self.dispatch_unpack_error = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        self.partial_error = torch.zeros(1, dtype=torch.int32, device=device)
        self.wait_scratch = torch.zeros(1, dtype=torch.int32, device=device)
        self.dispatch_row_ids = torch.arange(
            shape.tokens, dtype=torch.int32, device=device
        )
        self.fanout_q = torch.empty(
            (self.global_tokens, shape.hidden // 2),
            dtype=torch.uint8,
            device=device,
        )
        self.fanout_scales = torch.empty(
            (self.global_tokens, shape.hidden // 32),
            dtype=torch.uint8,
            device=device,
        )
        self.fanout_ids = torch.empty(
            (self.global_tokens, shape.topk),
            dtype=torch.int32,
            device=device,
        )
        self.fanout_weights = torch.empty_like(
            self.fanout_ids, dtype=torch.float32
        )
        self.fanout_sources = torch.empty(
            self.global_tokens, dtype=torch.int32, device=device
        )
        self.fanout_masks = torch.empty(
            self.global_tokens, dtype=torch.int64, device=device
        )
        self.route_expected = torch.full(
            (self.global_tokens,),
            shape.gpus_per_node,
            dtype=torch.int32,
            device=device,
        )
        self.route_ready = self.route_expected.clone()
        self.node_partial = torch.zeros(
            (self.global_tokens, shape.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        self.node_ready = torch.zeros(
            self.global_tokens, dtype=torch.int64, device=device
        )
        self.remote_rows = torch.zeros(
            (shape.tokens, shape.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        self.remote_sources = torch.empty(
            wire.partial_records_per_chunk,
            dtype=torch.int32,
            device=device,
        )
        self.partial_chunk_rows = torch.empty(
            (wire.partial_records_per_chunk, shape.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        self.final = torch.zeros(
            (shape.tokens, shape.hidden),
            dtype=torch.bfloat16,
            device=device,
        )
        self.final_ready = torch.zeros(
            shape.tokens, dtype=torch.int64, device=device
        )
        self.remote_ready = torch.zeros(
            shape.tokens, dtype=torch.int64, device=device
        )
        self.partial_source_ids = torch.arange(
            self.remote_rank * shape.tokens,
            (self.remote_rank + 1) * shape.tokens,
            dtype=torch.int32,
            device=device,
        ).view(wire.partial_chunks_per_rank, wire.partial_records_per_chunk)
        self.partial_dummy_ready = torch.zeros(
            (
                wire.partial_chunks_per_rank,
                wire.partial_records_per_chunk,
            ),
            dtype=torch.int64,
            device=device,
        )
        if wire.dispatch_chunks_per_rank > self.layout.ring_depth:
            raise AssertionError("dispatch active span exceeds the ring")
        if wire.partial_chunks_per_rank > self.layout.ring_depth:
            raise AssertionError("partial return active span exceeds the ring")
        dispatch_record_span = shape.tokens * wire.dispatch_record_bytes
        dispatch_active_span = (
            wire.dispatch_chunks_per_rank * wire.chunk_bytes
        )
        if dispatch_record_span > dispatch_active_span:
            raise AssertionError("dispatch records exceed the active chunks")
        for region_name in ("dispatch_tx", "dispatch_rx"):
            region = self.layout.region(region_name)
            if region.nbytes != wire.ring_depth * wire.chunk_bytes:
                raise AssertionError(f"{region_name} ring allocation is malformed")
            offsets = [
                self.layout.ring_chunk_offset(region_name, slot)
                for slot in range(self.layout.ring_depth)
            ]
            if any(
                right - left != self.layout.chunk_bytes
                for left, right in zip(offsets, offsets[1:])
            ):
                raise AssertionError(f"{region_name} ring slots are not contiguous")
            active_end = (
                offsets[wire.dispatch_chunks_per_rank - 1] + wire.chunk_bytes
            )
            if active_end - offsets[0] < dispatch_record_span:
                raise AssertionError(
                    f"{region_name} active prefix cannot hold packed records"
                )
            if offsets[0] != region.offset:
                raise AssertionError(
                    f"{region_name} pack does not start at the first slot"
                )
        self.local_context = None
        self.local_partial = None
        self.validate_routes = True

    def _generation(self, offset: int = 0) -> int:
        # Odd epoch stride makes generation parity alternate.  Fanout ready
        # and count have two parity banks; the old even stride selected the
        # same bank forever and turned every epoch reset into a live-buffer
        # race.
        return self.epoch * 1001 + offset + 1

    def _dispatch_pack(self):
        from aiter.ops.flydsl.kernels.megamoe_tile import cco

        self.epoch += 1
        self.pack_error.zero_()
        self.fanout_error.zero_()
        self.dispatch_unpack_error.zero_()
        self.partial_error.zero_()
        self.node_ready.zero_()
        self.remote_rows.zero_()
        self.final_ready.zero_()
        self.remote_ready.zero_()
        # Every one of the 2048 inbox records is fully overwritten before
        # unpack.  Clearing this live LSA destination is both unnecessary and
        # unsafe without a cross-rank clear-complete handshake.
        parity = self._generation() & 1
        cco.zero_window(
            self.window.local_ptr
            + self.layout.offset("fanout_ready", parity=parity),
            self.layout.region("fanout_ready").nbytes // 2,
        )
        cco.zero_window(
            self.window.local_ptr
            + self.layout.offset("fanout_count", parity=parity),
            self.layout.region("fanout_count").nbytes // 2,
        )
        # Two active 256-KiB TX slots are one contiguous 128-record slab.
        tx = self.window.local_ptr + self.layout.ring_chunk_offset(
            "dispatch_tx", 0
        )
        self.dispatch_record.launch_pack(
            self.shared.a_quant.data_ptr(),
            self.shared.a_scale.data_ptr(),
            self.shared.topk_ids.data_ptr(),
            self.shared.route_weights.data_ptr(),
            self.source_ids.data_ptr(),
            self.route_masks.data_ptr(),
            self.dispatch_row_ids.data_ptr(),
            tx,
            self.shape.tokens,
            self.shape.tokens,
            self.shape.experts,
            self.global_tokens,
            self.pack_error.data_ptr(),
            stream=self.stream,
        )

    def _inter_node_dispatch(self):
        generation = self._generation()
        for slot in range(self.wire.dispatch_chunks_per_rank):
            self.h1_sidecar.module.launch_send(
                self.dc.ptr,
                self.window.handle,
                self.window.local_ptr,
                self.remote_node,
                slot,
                generation,
                stream=self.stream,
            )
        for slot in range(self.wire.dispatch_chunks_per_rank):
            self.h1_sidecar.module.launch_mark_chunk_ready(
                self.window.local_ptr,
                slot,
                generation,
                self.wait_scratch.data_ptr(),
                0,
                0,
                1,
                stream=self.stream,
            )

    def _fanout_unpack_sort(self):
        # All ranks must have completed their epoch-local reset and inter-node
        # receive before any peer starts LSA writes into their fanout arena.
        # This is the bring-up epoch boundary; a production device-side EOS
        # protocol can replace it later.
        self.comm.barrier()
        generation = self._generation()
        parity = generation & 1
        local_records = self.window.local_ptr + self.layout.ring_chunk_offset(
            "dispatch_tx", 0
        )
        remote_records = self.window.local_ptr + self.layout.ring_chunk_offset(
            "dispatch_rx", 0
        )
        # One 2048-CTA launch broadcasts both 128-record slabs to all eight
        # LSA destinations.  This preserves the exact compact inbox slot map
        # while removing the former 16 host/kernel launch boundaries.
        self.fanout.launch_broadcast(
            local_records,
            remote_records,
            self.window.handle,
            self.shape.tokens,
            self.fanout_local_slot_base,
            self.fanout_remote_slot_base,
            generation,
            parity,
            self.fanout_error.data_ptr(),
            stream=self.stream,
        )
        torch.cuda.synchronize(self.device)
        self.comm.barrier()
        records = (
            self.window.local_ptr + self.layout.region("fanout_inbox").offset
        )
        self.dispatch_record.launch_unpack(
            records,
            self.fanout_q.data_ptr(),
            self.fanout_scales.data_ptr(),
            self.fanout_ids.data_ptr(),
            self.fanout_weights.data_ptr(),
            self.fanout_sources.data_ptr(),
            self.fanout_masks.data_ptr(),
            self.global_tokens,
            self.shape.experts,
            self.global_tokens,
            self.dispatch_unpack_error.data_ptr(),
            stream=self.stream,
        )
        scale_typed = self.fanout_scales.view(self.shared.a_scale.dtype)
        self.local_context = _prepare_local_a4w4(
            self.fanout_q,
            scale_typed,
            self.fanout_ids,
            self.fanout_weights,
            self.shape,
            self.rank,
            validate_routes=self.validate_routes,
        )
        self.local_sources = self.fanout_sources
        for slot in range(self.wire.dispatch_chunks_per_rank):
            self.h1_sidecar.module.launch_return_credit(
                self.dc.ptr,
                self.window.handle,
                self.remote_node,
                slot,
                generation,
                stream=self.stream,
            )
            self.h1_sidecar.module.launch_reclaim(
                self.dc.ptr,
                self.window.local_ptr,
                slot,
                generation,
                stream=self.stream,
            )

    def _h1(self):
        _run_local_h1(self.local_context, self.shared.prepared_weights, self.shape)

    def _h2(self):
        self.local_partial = _run_local_h2(
            self.local_context, self.shared.prepared_weights, self.shape
        )
        # This is the single rank-partial clear for the epoch. Keep it adjacent
        # to the scatter so ranks with no local route retain the zero-row ABI.
        self.rank_partial.zero_()
        self.rank_partial.index_copy_(
            0, self.local_sources.to(torch.int64), self.local_partial
        )

    def _node_reduce(self):
        generation = self._generation()
        ptrs = self.layout.epoch_pointers(
            self.window.local_ptr, generation
        )
        self.rank_partial_epoch_gate(
            self.window.handle,
            ptrs.rank_partial_epoch_ready,
            self.layout.offset("partial_eos", parity=ptrs.parity),
            generation,
            stream=self.stream,
        )
        # Every rank needs only its local-source rows for final combine and the
        # aligned remote-source rows for return. Reduce those two contiguous
        # 128-row ranges instead of redundantly reducing all 2048 sources on
        # every GPU. The epoch gate above still covers the complete rank slab.
        row_bytes = self.shape.hidden * 2
        for first_source in (
            self.rank * self.shape.tokens,
            self.remote_rank * self.shape.tokens,
        ):
            self.node_reduce(
                self.window.handle,
                self.rank_partial_offset + first_source * row_bytes,
                self.route_ready.data_ptr() + first_source * 4,
                self.route_expected.data_ptr() + first_source * 4,
                self.node_partial.data_ptr() + first_source * row_bytes,
                self.node_ready.data_ptr() + first_source * 8,
                generation,
                self.shape.tokens,
                stream=self.stream,
            )

    def _partial_return(self):
        per_chunk = self.wire.partial_records_per_chunk
        group_width = self.layout.ring_depth
        for group_first in range(
            0, self.wire.partial_chunks_per_rank, group_width
        ):
            group_chunks = list(
                range(
                    group_first,
                    min(
                        group_first + group_width,
                        self.wire.partial_chunks_per_rank,
                    ),
                )
            )
            # Fill every active TX slot (four in perf-v4) before posting.
            for chunk in group_chunks:
                slot = chunk % group_width
                tx = self.window.local_ptr + self.layout.ring_chunk_offset(
                    "partial_tx", slot
                )
                self.partial_record.launch_pack(
                    self.node_partial.data_ptr(),
                    self.partial_source_ids[chunk].data_ptr(),
                    tx,
                    per_chunk,
                    self.global_tokens,
                    self.partial_error.data_ptr(),
                    stream=self.stream,
                )
            # Submit the whole active group so network progress can overlap
            # before the receiver waits on any individual slot.
            for chunk in group_chunks:
                slot = chunk % group_width
                generation = self._generation(100 + chunk)
                self.h2_sidecar.module.launch_send(
                    self.dc.ptr,
                    self.window.handle,
                    self.window.local_ptr,
                    self.remote_node,
                    slot,
                    generation,
                    stream=self.stream,
                )
            # Do not clear RX: each PUT fully overwrites four active records.
            # Wait/unpack/copy/credit/reclaim each posted slot before the next
            # group reuses its ring index.
            for chunk in group_chunks:
                first = chunk * per_chunk
                slot = chunk % group_width
                generation = self._generation(100 + chunk)
                rx = self.window.local_ptr + self.layout.ring_chunk_offset(
                    "partial_rx", slot
                )
                self.h2_sidecar.module.launch_publish_received(
                    self.window.local_ptr,
                    slot,
                    generation,
                    self.partial_dummy_ready[chunk].data_ptr(),
                    0,
                    per_chunk,
                    stream=self.stream,
                )
                self.partial_record.launch_unpack(
                    rx,
                    self.partial_chunk_rows.data_ptr(),
                    self.remote_sources.data_ptr(),
                    per_chunk,
                    self.global_tokens,
                    self.partial_error.data_ptr(),
                    stream=self.stream,
                )
                self.remote_rows[first : first + per_chunk].copy_(
                    self.partial_chunk_rows
                )
                self.h2_sidecar.module.launch_return_credit(
                    self.dc.ptr,
                    self.window.handle,
                    self.remote_node,
                    slot,
                    generation,
                    stream=self.stream,
                )
                self.h2_sidecar.module.launch_reclaim(
                    self.dc.ptr,
                    self.window.local_ptr,
                    slot,
                    generation,
                    stream=self.stream,
                )
        self.remote_ready.fill_(self._generation())

    def _final(self):
        local_first = self.rank * self.shape.tokens
        self.final_combine(
            self.node_partial[local_first:].data_ptr(),
            self.remote_rows.data_ptr(),
            self.node_ready[local_first:].data_ptr(),
            self.remote_ready.data_ptr(),
            self.final.data_ptr(),
            self.final_ready.data_ptr(),
            self._generation(),
            self.shape.tokens,
            stream=self.stream,
        )
        return self.final

    def run_iteration(self, timer: HipStageTimer) -> torch.Tensor:
        timer.stage("dispatch_pack", self._dispatch_pack)
        timer.stage("inter_node_dispatch", self._inter_node_dispatch)
        timer.stage("lsa_fanout_unpack_sort", self._fanout_unpack_sort)
        timer.stage("h1", self._h1)
        timer.stage("h2", self._h2)
        timer.stage("node_reduce", self._node_reduce)
        timer.stage("partial_pack_return", self._partial_return)
        return timer.stage("final_combine", self._final)

    def prime_and_check(self) -> torch.Tensor:
        from aiter.ops.flydsl.kernels.megamoe_tile import cco

        timer = HipStageTimer(self.device, self.stage_names)
        timer.begin_iteration()
        output = self.run_iteration(timer)
        timer.finish_iteration()
        if not torch.isfinite(output.float()).all():
            raise AssertionError("candidate warmup produced non-finite output")
        if self.pack_error.item() or self.fanout_error.item():
            raise AssertionError("candidate dispatch pack/fanout error")
        if self.dispatch_unpack_error.item() or self.partial_error.item():
            raise AssertionError("candidate unpack/partial record error")
        ptrs = self.layout.epoch_pointers(
            self.window.local_ptr, self._generation()
        )
        fanout_count = cco.read_window_u32(ptrs.fanout_count, 1)[0]
        if fanout_count != self.global_tokens:
            raise AssertionError(
                f"fanout count={fanout_count}, expected {self.global_tokens}"
            )
        expected_last = (self.rank + 1) * self.shape.tokens - 1
        if int(self.remote_sources[-1].item()) != expected_last:
            raise AssertionError("partial return source mapping is incorrect")
        if torch.count_nonzero(
            self.final_ready != self._generation()
        ).item():
            raise AssertionError("final generation publication is incomplete")
        self.validate_routes = False
        return output

    def close(self) -> None:
        self._comm_context.__exit__(None, None, None)


def build_candidate_path(shape, shared, rank, world, local_rank, device, wire):
    """Default candidate-factory spec exported by this benchmark module."""

    return CascadeCandidatePath(
        shape, shared, rank, world, local_rank, device, wire
    )


def _global_max_timing(
    timing: IterationTiming, stage_names: tuple[str, ...]
) -> IterationTiming:
    values = [timing.stage_us[name] for name in stage_names]
    values.append(timing.gpu_e2e_us)
    values.append(timing.host_critical_us)
    tensor = torch.tensor(values, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return IterationTiming(
        dict(zip(stage_names, tensor[: len(stage_names)].tolist())),
        float(tensor[-2].item()),
        float(tensor[-1].item()),
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) & 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _sample_stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": float(sum(values) / len(values)),
        "p50": float(_median(values)),
        "p95": float(ordered[p95_index]),
    }


def _comparison_metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rank: int,
    label: str,
) -> dict[str, float | int | str]:
    reference = reference.float()
    actual = actual.float()
    if not torch.isfinite(reference).all().item():
        raise AssertionError(f"{label}: reference contains non-finite values")
    if not torch.isfinite(actual).all().item():
        raise AssertionError(f"{label}: actual contains non-finite values")
    delta = actual - reference
    rel_l2 = delta.norm() / reference.norm().clamp_min(1.0e-12)
    reference64 = reference.double()
    actual64 = actual.double()
    denominator = reference64.square().sum() + actual64.square().sum()
    logits_diff = float(
        0.0
        if denominator.item() == 0.0
        else 1.0
        - 2.0
        * (reference64 * actual64).sum().item()
        / denominator.item()
    )
    per_token = delta.norm(dim=1) / reference.norm(dim=1).clamp_min(1.0e-12)
    per_token_values = per_token.double().cpu().tolist()
    return {
        "rank": rank,
        "label": label,
        "rel_l2": float(rel_l2.item()),
        "logits_diff": logits_diff,
        "max_abs": float(delta.abs().max().item()),
        "norm_ratio": float(
            actual64.norm().item() / max(reference64.norm().item(), 1.0e-12)
        ),
        "per_token_rel_l2_min": float(min(per_token_values)),
        "per_token_rel_l2_p50": float(_median(per_token_values)),
        "per_token_rel_l2_max": float(max(per_token_values)),
    }


def _run_path(
    path: TimedPath,
    device: torch.device,
    warmup: int,
    iters: int,
    barrier_mode: str,
):
    prime = getattr(path, "prime_and_check", None)
    prime_output = None
    if prime is not None:
        prime_output = prime().clone()
        dist.barrier()
    timer = HipStageTimer(device, path.stage_names)
    for _ in range(warmup):
        timer.begin_iteration()
        path.run_iteration(timer)
        timer.finish_iteration()
    # JIT/prime and every warmup launch are outside timing. Align all ranks
    # exactly once before a continuous timed sequence.
    torch.cuda.synchronize(device)
    dist.barrier()
    local_samples = []
    output = None
    for _ in range(iters):
        if barrier_mode == "each":
            # This barrier is deliberately before begin_iteration(), so neither
            # HIP E2E nor host critical-path timing contains Gloo latency.
            dist.barrier()
        timer.begin_iteration()
        output = path.run_iteration(timer)
        local = timer.finish_iteration()
        local_samples.append(local)
    # Preserve continuous-ops timing in once mode: rank-max collectives happen
    # only after every local timed sample has been captured.
    max_samples = [
        _global_max_timing(local, path.stage_names) for local in local_samples
    ]
    return prime_output, output.clone(), local_samples, max_samples


def _summary(path: TimedPath, local_samples, max_samples, tail_iters: int):
    local_stage_sum = [sum(item.stage_us.values()) for item in local_samples]
    max_stage_sum = [sum(item.stage_us.values()) for item in max_samples]
    tail_count = len(max_samples) if tail_iters == 0 else tail_iters
    tail_local = local_samples[-tail_count:]
    tail_max = max_samples[-tail_count:]
    per_iteration = {
        name: [item.stage_us[name] for item in max_samples]
        for name in path.stage_names
    }
    per_iteration["gpu_e2e"] = [item.gpu_e2e_us for item in max_samples]
    per_iteration["host_critical"] = [
        item.host_critical_us for item in max_samples
    ]
    tail_stats = {
        name: _sample_stats([item.stage_us[name] for item in tail_max])
        for name in path.stage_names
    }
    tail_stats["gpu_e2e"] = _sample_stats(
        [item.gpu_e2e_us for item in tail_max]
    )
    tail_stats["host_critical"] = _sample_stats(
        [item.host_critical_us for item in tail_max]
    )
    return {
        "path": path.name,
        "local_median_us": {
            name: _median([item.stage_us[name] for item in local_samples])
            for name in path.stage_names
        },
        "global_max_median_us": {
            name: _median([item.stage_us[name] for item in max_samples])
            for name in path.stage_names
        },
        "local_host_critical_median_us": _median(
            [item.host_critical_us for item in local_samples]
        ),
        "global_max_host_critical_median_us": _median(
            [item.host_critical_us for item in max_samples]
        ),
        "local_gpu_e2e_median_us": _median(
            [item.gpu_e2e_us for item in local_samples]
        ),
        "global_max_gpu_e2e_median_us": _median(
            [item.gpu_e2e_us for item in max_samples]
        ),
        "local_stage_sum_median_us": _median(local_stage_sum),
        "global_max_stage_sum_median_us": _median(max_stage_sum),
        "local_gpu_e2e_minus_stage_sum_median_us": _median(
            [
                item.gpu_e2e_us - stage_sum
                for item, stage_sum in zip(local_samples, local_stage_sum)
            ]
        ),
        "global_host_minus_gpu_e2e_median_us": _median(
            [
                item.host_critical_us - item.gpu_e2e_us
                for item in max_samples
            ]
        ),
        "tail_iters": tail_count,
        "tail_global_max_stats_us": tail_stats,
        "tail_local_gpu_e2e_stats_us": _sample_stats(
            [item.gpu_e2e_us for item in tail_local]
        ),
        "global_max_per_iteration_us": per_iteration,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--inter", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=896)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument(
        "--candidate-chunk-bytes", type=int, default=512 * 1024
    )
    parser.add_argument("--candidate-num-qp", type=int, default=4)
    parser.add_argument("--candidate-ring-depth", type=int, default=8)
    parser.add_argument(
        "--candidate-dispatch-batch-per-qp", type=int, default=32
    )
    parser.add_argument(
        "--candidate-partial-batch-per-qp", type=int, default=8
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--tail-iters",
        type=int,
        default=0,
        help="Summarize the final N timed iterations; 0 uses all iterations.",
    )
    parser.add_argument(
        "--barrier-mode",
        choices=("once", "each"),
        default="each",
        help=(
            "each is the safe bring-up latency mode and adds an out-of-timing "
            "barrier before every sample; once requires a complete device-side "
            "EOS/epoch ownership protocol before ring reuse"
        ),
    )
    parser.add_argument(
        "--diagnostic-flat-sum",
        action="store_true",
        help=(
            "Outside timing, Gloo-all-reduce the candidate rank partial in "
            "FP32 and compare the flat sum with baseline/candidate outputs."
        ),
    )
    parser.add_argument(
        "--paths", default="baseline", help="baseline,candidate,or both"
    )
    parser.add_argument(
        "--candidate-factory",
        default="",
        help=(
            "module:function returning a TimedPath from "
            "(shape, shared, rank, world, local_rank, device, wire_plan)"
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.iters <= 0 or args.warmup < 0:
        raise ValueError("iters must be positive and warmup non-negative")
    if args.tail_iters < 0 or args.tail_iters > args.iters:
        raise ValueError("tail-iters must be in [0, iters]")
    shape = BenchmarkShape(
        args.tokens,
        args.hidden,
        args.inter,
        args.experts,
        args.topk,
    )
    shape.validate()
    wire = CandidateWirePlan.for_shape(
        shape,
        chunk_bytes=args.candidate_chunk_bytes,
        num_qp=args.candidate_num_qp,
        ring_depth=args.candidate_ring_depth,
        dispatch_batch_per_qp=args.candidate_dispatch_batch_per_qp,
        partial_batch_per_qp=args.candidate_partial_batch_per_qp,
    )
    if args.plan_only:
        print(
            "MEGAMOE_EP16_BENCH_PLAN "
            + json.dumps(
                {"shape": shape.__dict__, "candidate_wire": wire.__dict__},
                sort_keys=True,
            )
        )
        return 0

    requested = {item.strip() for item in args.paths.split(",") if item.strip()}
    needs_mori = "baseline" in requested or "both" in requested
    rank, world, local_rank, device = _setup_dist(needs_mori)
    if world != shape.ep_size:
        raise ValueError(f"requires world={shape.ep_size}, got {world}")
    shared = _shared_inputs(shape, rank, world, device)
    paths: list[TimedPath] = []
    if "baseline" in requested or "both" in requested:
        paths.append(MoriBaselinePath(shape, shared, rank, world))
    if "candidate" in requested or "both" in requested:
        factory = (
            _load_factory(args.candidate_factory)
            if args.candidate_factory
            else build_candidate_path
        )
        paths.append(
            factory(shape, shared, rank, world, local_rank, device, wire)
        )
    if not paths:
        raise ValueError("no benchmark path selected")

    summaries = []
    outputs = {}
    for path in paths:
        prime_output, output, local_samples, max_samples = _run_path(
            path,
            device,
            args.warmup,
            args.iters,
            args.barrier_mode,
        )
        outputs[path.name] = {"prime": prime_output, "timed": output}
        summaries.append(
            _summary(path, local_samples, max_samples, args.tail_iters)
        )
    correctness_rel_l2 = None
    correctness_diagnostic = None
    correctness_failed = False
    same_path_diagnostics = []
    for path in paths:
        prime = outputs[path.name]["prime"]
        timed = outputs[path.name]["timed"]
        if prime is not None:
            same_path_diagnostics.append(
                _comparison_metrics(
                    prime[: shape.tokens],
                    timed[: shape.tokens],
                    rank=rank,
                    label=f"{path.name}_prime_vs_timed",
                )
            )
    if len(paths) == 2:
        baseline = outputs[paths[0].name]["timed"][: shape.tokens].float()
        candidate = outputs[paths[1].name]["timed"][: shape.tokens].float()
        correctness_diagnostic = _comparison_metrics(
            baseline,
            candidate,
            rank=rank,
            label="candidate_vs_baseline_timed",
        )
        # Correctness is outside timing. Coordination is Gloo-only, so move
        # the scalar to a CPU float64 tensor before the rank-max collective.
        rel_l2_cpu = torch.tensor(
            float(correctness_diagnostic["rel_l2"]), dtype=torch.float64
        )
        dist.all_reduce(rel_l2_cpu, op=dist.ReduceOp.MAX)
        correctness_rel_l2 = float(rel_l2_cpu.item())
        correctness_failed = correctness_rel_l2 >= 1.0e-2
        if args.diagnostic_flat_sum:
            candidate_path = paths[1]
            rank_partial = getattr(candidate_path, "rank_partial", None)
            if rank_partial is None:
                raise ValueError(
                    "--diagnostic-flat-sum requires candidate.rank_partial"
                )
            flat_sum = rank_partial.float().cpu()
            dist.all_reduce(flat_sum, op=dist.ReduceOp.SUM)
            local_first = rank * shape.tokens
            flat_local = flat_sum[
                local_first : local_first + shape.tokens
            ]
            same_path_diagnostics.append(
                _comparison_metrics(
                    flat_local,
                    baseline,
                    rank=rank,
                    label="baseline_vs_fp32_flat_candidate_partials",
                )
            )
            same_path_diagnostics.append(
                _comparison_metrics(
                    flat_local,
                    candidate,
                    rank=rank,
                    label="candidate_vs_fp32_flat_candidate_partials",
                )
            )
    correctness_payload = {
        "same_path": same_path_diagnostics,
        "candidate_vs_baseline": correctness_diagnostic,
    }
    gathered = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(summaries, gathered, dst=0)
    gathered_diagnostics = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(correctness_payload, gathered_diagnostics, dst=0)
    if rank == 0:
        comparison_complete = len(paths) == 2
        marker = "MEGAMOE_EP16_DUAL_PATH" if comparison_complete else (
            "MEGAMOE_EP16_BASELINE_SMOKE"
            if paths[0].name == "mori_baseline"
            else "MEGAMOE_EP16_CANDIDATE_SMOKE"
        )
        print(
            marker + " "
            + json.dumps(
                {
                    "shape": shape.__dict__,
                    "wire": wire.__dict__,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "tail_iters": args.tail_iters or args.iters,
                    "barrier_mode": args.barrier_mode,
                    "iteration_semantics": (
                        "independent_round_latency_cuda_sync_each_iteration"
                    ),
                    "diagnostic_flat_sum": args.diagnostic_flat_sum,
                    "comparison_complete": comparison_complete,
                    "candidate_vs_baseline_rel_l2": correctness_rel_l2,
                    "correctness_by_rank": gathered_diagnostics,
                    "rank_summaries": gathered,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    for path in paths:
        close = getattr(path, "close", None)
        if close is not None:
            close()
    if needs_mori:
        import mori.shmem as ms

        ms.shmem_finalize()
    dist.destroy_process_group()
    return 1 if correctness_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

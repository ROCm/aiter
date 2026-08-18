# SPDX-License-Identifier: MIT
"""Prepacked EP16 (2 nodes x 8 GPUs) hierarchical cascade E2E.

Launch once on each node::

    MORI_SOCKET_IFNAME=enp193s0f1np1 \
    GLOO_SOCKET_IFNAME=enp193s0f1np1 \
    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=${NODE_RANK} \
      --master-addr=${MASTER_ADDR} --master-port=29618 \
      op_tests/multigpu_tests/test_megamoe_tile_ep16_cascade.py

The default ``MEGAMOE_CASCADE_ROUTE_MODE=fanout`` sends remote routes to
non-aligned local ranks through the aligned cross-node proxy and CCO LSA.
Set ``MEGAMOE_CASCADE_ROUTE_MODE=aligned`` for the no-fanout regression path.

Source A4 quantization is packed into real 2048-byte dispatch records. The
remote half of H1 input and its raw scales/routing metadata therefore comes
from CCO; the local half stays local. The remaining seam is generic
destination counting, expert sort and node-local fanout. Everything after that
seam is the production-shaped cascade: ready H1 -> ready H2 into a CCO window
-> EP8 LSA node reduction -> CCO RAIL return -> final two-node combine. TP
all-reduce is not part of this driver and remains a downstream operation.
"""

from __future__ import annotations

import ctypes
from datetime import timedelta
import os
import sys
import traceback

import torch
import torch.distributed as dist

from mori.cco import (
    Communicator,
    CCODevCommRequirements,
    GDA_CONNECTION_RAIL,
    UniqueId,
)

from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.megamoe_tile import (
    LogicalTopology,
    build_route_plan,
    cco,
    prepare_local_a4w4_weights,
)
from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
    build_dispatch_record_module,
    build_dispatch_fanout_lsa,
    build_partial_record_module,
    compile_final_combine,
    compile_hier_stage1_ready_a4w4,
    compile_hier_stage2_partial_a4w4,
    compile_node_partial_reduce_lsa,
    partial_record_format,
)
from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort


LOCAL_WORLD = 8
WORLD = 16
TOKENS = 16
GLOBAL_TOKENS = WORLD * TOKENS
HIDDEN = 1024
INTER = 256
GLOBAL_EXPERTS = 32
LOCAL_EXPERTS = 2
TOPK = 2
BM = 32
NUM_QP = 4
SLOT = 0
GENERATION = 1
CHUNK_BYTES = 64 * 1024
DISPATCH_SEGMENT_BYTES = 2048
DISPATCH_BATCH_PER_QP = 8
PARTIAL_FORMAT = partial_record_format(HIDDEN)
PARTIAL_RECORD_BYTES = PARTIAL_FORMAT.record_bytes
PARTIAL_ROW_BYTES = PARTIAL_FORMAT.payload_bytes
PARTIAL_BATCH_PER_QP = 4
PARTIAL_RECORDS = NUM_QP * PARTIAL_BATCH_PER_QP
TIMEOUT_SECONDS = int(os.environ.get("MEGAMOE_CASCADE_TIMEOUT_S", "900"))
ROUTE_MODE = os.environ.get("MEGAMOE_CASCADE_ROUTE_MODE", "fanout").lower()


def _broadcast_cco_uid(rank: int) -> UniqueId:
    obj = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    payload = obj[0]
    if not isinstance(payload, bytes) or len(payload) != 128:
        raise RuntimeError("invalid CCO unique id broadcast")
    return UniqueId.from_bytes(payload)


def _barrier(stage: str) -> None:
    """Gloo-monitored stage boundary so peer loss raises on every rank."""

    try:
        dist.monitored_barrier(timeout=timedelta(seconds=TIMEOUT_SECONDS))
    except Exception as exc:
        raise RuntimeError(f"cascade barrier failed at {stage}") from exc


def _cpu_random(shape, seed: int, scale: float, dtype=torch.bfloat16):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return (torch.randn(shape, generator=generator) * float(scale)).to(dtype)


def _source_hidden(source_rank: int, device: torch.device) -> torch.Tensor:
    return _cpu_random(
        (TOKENS, HIDDEN), 10_000 + source_rank, 0.1
    ).to(device)


def _local_weights(rank: int, device: torch.device):
    # Global experts [2*rank, 2*rank+1] live on this EP rank. The deterministic
    # route below selects the first expert; the second keeps the true E32/EP16
    # local shape and catches accidental NE=1 specialization.
    w1 = _cpu_random(
        (LOCAL_EXPERTS, 2 * INTER, HIDDEN), 20_000 + rank, 0.03
    ).to(device)
    w2 = _cpu_random(
        (LOCAL_EXPERTS, HIDDEN, INTER), 30_000 + rank, 0.03
    ).to(device)
    return prepare_local_a4w4_weights(w1, w2)


def _remote_destination(local_rank: int, token: int, mode: str) -> int:
    if mode == "aligned":
        return int(local_rank)
    if mode == "fanout":
        return (int(local_rank) + 1 + int(token) % LOCAL_WORLD) % LOCAL_WORLD
    raise ValueError(f"unknown cascade route mode {mode!r}")


def _topk_experts(source_rank: int, device: torch.device) -> torch.Tensor:
    source_node = int(source_rank) // LOCAL_WORLD
    source_lsa = int(source_rank) % LOCAL_WORLD
    tokens = torch.arange(TOKENS, dtype=torch.int64, device=device)
    remote_dest = (
        source_lsa + 1 + tokens.remainder(LOCAL_WORLD)
    ).remainder(LOCAL_WORLD)
    if ROUTE_MODE == "aligned":
        remote_dest.fill_(source_lsa)
    node0_dest = (
        torch.full_like(remote_dest, source_lsa)
        if source_node == 0
        else remote_dest
    )
    node1_dest = (
        torch.full_like(remote_dest, source_lsa)
        if source_node == 1
        else remote_dest
    )
    return torch.stack(
        [
            node0_dest * LOCAL_EXPERTS,
            (LOCAL_WORLD + node1_dest) * LOCAL_EXPERTS,
        ],
        dim=1,
    ).to(torch.int32)


def _fanout_table(mode: str):
    """Return unique compact target slots for all proxy/token records."""

    counters = [0] * LOCAL_WORLD
    table: dict[tuple[int, int], tuple[int, int]] = {}
    for proxy_lsa in range(LOCAL_WORLD):
        for token in range(TOKENS):
            dest = _remote_destination(proxy_lsa, token, mode)
            slot = counters[dest]
            counters[dest] += 1
            table[(proxy_lsa, token)] = (dest, slot)
    if counters != [TOKENS] * LOCAL_WORLD:
        raise AssertionError(f"fanout target counts are not balanced: {counters}")
    return table


def _proxy_fanout_plan(local_rank: int, device: torch.device):
    table = _fanout_table(ROUTE_MODE)
    destinations = []
    slots = []
    for token in range(TOKENS):
        dest, slot = table[(local_rank, token)]
        destinations.append(dest)
        slots.append(slot)
    return (
        torch.tensor(destinations, dtype=torch.int32, device=device),
        torch.tensor(slots, dtype=torch.int32, device=device),
        torch.ones(TOKENS, dtype=torch.int32, device=device),
    )


def _expected_target_records(
    target_lsa: int, source_node: int, device: torch.device
):
    """Materialize expected remote wire fields in compact target-slot order."""

    table = _fanout_table(ROUTE_MODE)
    entries = []
    for proxy_lsa in range(LOCAL_WORLD):
        for token in range(TOKENS):
            dest, slot = table[(proxy_lsa, token)]
            if dest == target_lsa:
                entries.append((slot, proxy_lsa, token))
    entries.sort()
    if [slot for slot, _, _ in entries] != list(range(TOKENS)):
        raise AssertionError("target fanout slots are not compact")

    q_rows = []
    scale_rows = []
    ids_rows = []
    source_rows = []
    cached_quant = {}
    for _, proxy_lsa, token in entries:
        source_rank = source_node * LOCAL_WORLD + proxy_lsa
        if proxy_lsa not in cached_quant:
            hidden = _source_hidden(source_rank, device)
            q, scales = per_1x32_f4_quant(hidden, shuffle=False)
            cached_quant[proxy_lsa] = (
                q,
                scales.view(TOKENS, HIDDEN // 32),
            )
        q, scales = cached_quant[proxy_lsa]
        q_rows.append(q[token].view(torch.uint8))
        scale_rows.append(scales[token])
        ids_rows.append(_topk_experts(source_rank, device)[token])
        source_rows.append(source_rank * TOKENS + token)
    return {
        "q": torch.stack(q_rows).contiguous(),
        "scales": torch.stack(scale_rows).contiguous(),
        "ids": torch.stack(ids_rows).contiguous(),
        "weights": torch.full(
            (TOKENS, TOPK), 0.5, dtype=torch.float32, device=device
        ),
        "sources": torch.tensor(
            source_rows, dtype=torch.int32, device=device
        ),
        "masks": torch.full(
            (TOKENS,), 1 << (1 - source_node), dtype=torch.int64, device=device
        ),
    }


def _hip_memcpy_d2d(dst: int, src: int, nbytes: int) -> None:
    """Synchronous D2D copy used only to inspect the H2 CCO output."""

    from mori.jit.hip_driver import _check, _get_hip_lib

    # hipMemcpyDeviceToDevice = 3.
    _check(
        _get_hip_lib().hipMemcpy(
            ctypes.c_void_p(int(dst)),
            ctypes.c_void_p(int(src)),
            ctypes.c_size_t(int(nbytes)),
            ctypes.c_int(3),
        ),
        "hipMemcpyDeviceToDevice",
    )


def _logits_diff(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.float().double()
    got = actual.float().double()
    denominator = ref.square().sum() + got.square().sum()
    if denominator.item() == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (ref * got).sum() / denominator)


def _build_source_payload(
    rank: int, node: int, local_rank: int, device: torch.device
):
    """Quantize the local 16 tokens and build their real two-node route plan."""

    hidden = _source_hidden(rank, device)
    hidden_q, raw_scales = per_1x32_f4_quant(hidden, shuffle=False)
    raw_scales = raw_scales.view(TOKENS, HIDDEN // 32).contiguous()
    topk_ids = _topk_experts(rank, device)
    topk_weights = torch.full(
        (TOKENS, TOPK), 0.5, dtype=torch.float32, device=device
    )
    topology = LogicalTopology(world_size=WORLD, gpus_per_node=LOCAL_WORLD)
    plan = build_route_plan(
        topk_ids, num_experts=GLOBAL_EXPERTS, topology=topology
    )
    remote_lsa = torch.tensor(
        [
            _remote_destination(local_rank, token, ROUTE_MODE)
            for token in range(TOKENS)
        ],
        dtype=torch.int64,
        device=device,
    )
    expected_ranks = torch.empty_like(plan.destination_rank)
    if node == 0:
        expected_ranks[:, 0] = rank
        expected_ranks[:, 1] = LOCAL_WORLD + remote_lsa
    else:
        expected_ranks[:, 0] = remote_lsa
        expected_ranks[:, 1] = rank
    if torch.count_nonzero(plan.destination_rank != expected_ranks).item():
        raise AssertionError("route plan has an unexpected destination rank")
    if torch.count_nonzero(plan.local_expert).item():
        raise AssertionError("cascade must select the first local expert")

    slots = torch.arange(TOPK, dtype=torch.int64, device=device)
    remote_node = 1 - node
    remote_mask = (
        (plan.destination_node == remote_node).to(torch.int64)
        * (torch.ones_like(slots) << slots)
    ).sum(dim=1)
    local_slot = torch.full((TOKENS,), node, dtype=torch.int32, device=device)
    source_tokens = torch.arange(
        rank * TOKENS,
        (rank + 1) * TOKENS,
        dtype=torch.int32,
        device=device,
    )
    return {
        # The wire ABI is packed bytes; avoid comparing/concatenating Torch's
        # experimental float4 scalar type with a uint8 unpack destination.
        "hidden_q": hidden_q.view(torch.uint8).contiguous(),
        "raw_scales": raw_scales,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "source_tokens": source_tokens,
        "remote_mask": remote_mask.contiguous(),
        "row_ids": torch.arange(TOKENS, dtype=torch.int32, device=device),
        "local_expert": plan.local_expert[:, node].to(torch.int32).contiguous(),
        "local_weight": topk_weights[:, node].contiguous(),
        "local_slot": local_slot,
        "prepared": _local_weights(rank, device),
    }


def _assemble_h1_prepacked(
    local_payload: dict[str, torch.Tensor],
    remote_q: torch.Tensor,
    remote_scales: torch.Tensor,
    remote_ids: torch.Tensor,
    remote_weights: torch.Tensor,
    remote_sources: torch.Tensor,
    remote_masks: torch.Tensor,
    rank: int,
    node: int,
):
    """Select this destination's routes, then rebuild sorted H1 metadata."""

    remote_slots = torch.where(
        (remote_masks & 1) != 0,
        torch.zeros_like(remote_masks),
        torch.ones_like(remote_masks),
    ).to(torch.int64)
    rows = torch.arange(TOKENS, device=remote_ids.device)
    remote_global_expert = remote_ids[rows, remote_slots]
    remote_local_expert = remote_global_expert - rank * LOCAL_EXPERTS
    if torch.count_nonzero(
        (remote_local_expert < 0) | (remote_local_expert >= LOCAL_EXPERTS)
    ).item():
        raise AssertionError("remote route is not owned by this EP rank")
    remote_weight = remote_weights[rows, remote_slots]

    a1q = torch.cat([local_payload["hidden_q"], remote_q], dim=0).contiguous()
    raw_scales = torch.cat(
        [local_payload["raw_scales"], remote_scales], dim=0
    ).contiguous()
    source_tokens = torch.cat(
        [local_payload["source_tokens"], remote_sources], dim=0
    ).contiguous()
    route_slots = torch.cat(
        [local_payload["local_slot"], remote_slots.to(torch.int32)], dim=0
    ).contiguous()
    local_experts = torch.cat(
        [local_payload["local_expert"], remote_local_expert.to(torch.int32)],
        dim=0,
    ).view(-1, 1)
    route_weights = torch.cat(
        [local_payload["local_weight"], remote_weight], dim=0
    ).view(-1, 1)
    local_rows = int(a1q.shape[0])
    sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
        local_experts,
        route_weights,
        LOCAL_EXPERTS,
        HIDDEN,
        torch.bfloat16,
        BM,
        accumulate=False,
    )
    a1ss = moe_mxfp4_sort(
        raw_scales.view(local_rows, 1, HIDDEN // 32),
        sorted_ids,
        nvalid,
        local_rows,
        BM,
    )
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()
    packed = sorted_ids.to(torch.int64) & 0xFFFFFFFF
    local_index = packed & 0x00FFFFFF
    valid = local_index < local_rows
    safe_index = torch.where(valid, local_index, torch.zeros_like(local_index))
    global_stids = (
        route_slots[safe_index].to(torch.int64) << 24
    ) | source_tokens[safe_index].to(torch.int64)
    global_stids = torch.where(valid, global_stids, packed).to(torch.int32)
    return {
        "prepared": local_payload["prepared"],
        "sorted_ids": sorted_ids,
        "global_stids": global_stids.contiguous(),
        "sorted_weights": sorted_weights,
        "sorted_eids": sorted_eids,
        "nvalid": nvalid,
        "a1q": a1q,
        "a1ss": a1ss,
        "m_indices": m_indices,
        "local_rows": local_rows,
    }


def _create_windows_and_rail(comm, layout: HierCcoArenaLayout):
    """Register one collective window before the DevComm snapshot.

    ROCm rejects a second fabric-VMM access mapping in this EP16 environment.
    Keep control/rings and rank partials in one allocation and address the
    latter by an aligned byte offset.  This also reduces registration cost.
    """

    rank_partial_bytes = GLOBAL_TOKENS * HIDDEN * 2
    partial_offset = (layout.total_bytes + 4095) // 4096 * 4096
    combined_bytes = partial_offset + rank_partial_bytes
    arena_memory = comm.alloc_mem(combined_bytes)
    arena_window = comm.register_window(arena_memory.ptr, arena_memory.size)

    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_RAIL
    reqs.gda_context_count = NUM_QP
    reqs.gda_signal_count = 0
    reqs.gda_counter_count = 0
    reqs.lsa_barrier_count = 0
    reqs.rail_gda_barrier_count = 0
    reqs.barrier_count = 0
    dev_comm = comm.create_dev_comm(reqs)
    return (
        arena_memory,
        arena_window,
        dev_comm,
        rank_partial_bytes,
        partial_offset,
        combined_bytes,
    )


def _run_cascade(rank: int, local_rank: int, uid: UniqueId) -> tuple[int, float]:
    node = rank // LOCAL_WORLD
    remote_node = 1 - node
    remote_rank = remote_node * LOCAL_WORLD + local_rank
    device = torch.device("cuda", local_rank)
    stream = torch.cuda.current_stream(device)
    local_failures = 0
    source_payload = _build_source_payload(rank, node, local_rank, device)

    layout = HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=NUM_QP,
        chunk_bytes=CHUNK_BYTES,
        max_m_tiles=8,
        max_source_tokens=GLOBAL_TOKENS,
        max_h1_n_blocks=4,
        max_fanout_records=TOKENS,
    )
    h1_sidecar = cco.CcoStage1Sidecar.create(
        layout,
        batch_per_qp=DISPATCH_BATCH_PER_QP,
        segment_bytes=DISPATCH_SEGMENT_BYTES,
        team=cco.TEAM_RAIL,
    )
    h2_sidecar = cco.CcoStage2ReturnSidecar.create(
        layout,
        batch_per_qp=PARTIAL_BATCH_PER_QP,
        record_bytes=PARTIAL_RECORD_BYTES,
        team=cco.TEAM_RAIL,
    )
    if h1_sidecar.module.payload_bytes != CHUNK_BYTES:
        raise AssertionError("Stage-1 cascade must send a real 64-KiB chunk")
    if h2_sidecar.module.payload_bytes != TOKENS * PARTIAL_RECORD_BYTES:
        raise AssertionError("Stage-2 payload must contain exactly 16 rows")
    if PARTIAL_RECORDS != TOKENS:
        raise AssertionError("Stage-2 QP geometry must own all source records")

    with Communicator.init(
        WORLD,
        rank,
        uid,
        per_rank_vmm=64 * 1024 * 1024,
    ) as comm:
        (
            arena_memory,
            arena_window,
            dc,
            rank_partial_bytes,
            rank_partial_offset,
            combined_bytes,
        ) = _create_windows_and_rail(comm, layout)
        # Keep the allocation owner referenced for the window lifetime.
        _ = arena_memory
        if dc.lsa_size != LOCAL_WORLD or dc.lsa_rank != local_rank:
            raise RuntimeError(
                f"unexpected LSA map size={dc.lsa_size} rank={dc.lsa_rank}"
            )

        cco.zero_window(arena_window.local_ptr, combined_bytes)
        rank_partial_ptr = arena_window.local_ptr + rank_partial_offset
        comm.barrier()
        _barrier("registered-windows")

        ptrs = layout.epoch_pointers(arena_window.local_ptr, GENERATION)
        tx_offset = layout.ring_chunk_offset("dispatch_tx", SLOT)
        rx_offset = layout.ring_chunk_offset("dispatch_rx", SLOT)
        tx_ptr = arena_window.local_ptr + tx_offset
        rx_ptr = arena_window.local_ptr + rx_offset
        dispatch_module = build_dispatch_record_module(HIDDEN, TOPK)
        if dispatch_module.layout.record_bytes != DISPATCH_SEGMENT_BYTES:
            raise AssertionError("dispatch record must remain exactly 2048 bytes")
        dispatch_pack_error = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        dispatch_module.launch_pack(
            source_payload["hidden_q"].data_ptr(),
            source_payload["raw_scales"].data_ptr(),
            source_payload["topk_ids"].data_ptr(),
            source_payload["topk_weights"].data_ptr(),
            source_payload["source_tokens"].data_ptr(),
            source_payload["remote_mask"].data_ptr(),
            source_payload["row_ids"].data_ptr(),
            tx_ptr,
            TOKENS,
            TOKENS,
            GLOBAL_EXPERTS,
            GLOBAL_TOKENS,
            dispatch_pack_error.data_ptr(),
            stream=stream,
        )
        cco.zero_window(rx_ptr, CHUNK_BYTES)
        torch.cuda.synchronize(device)
        comm.barrier()

        h1_sidecar.post_dispatch(
            dc.ptr,
            arena_window.handle,
            arena_window.local_ptr,
            remote_node,
            SLOT,
            GENERATION,
            stream=stream,
        )
        # Wait for all four remote-QP generations without publishing H1 tile
        # readiness: expert sort and shuffled-scale materialization happen only
        # after the wire fields are consumer-visible.
        dispatch_wait_scratch = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        h1_sidecar.module.launch_mark_chunk_ready(
            arena_window.local_ptr,
            SLOT,
            GENERATION,
            dispatch_wait_scratch.data_ptr(),
            0,
            0,
            1,
            stream=stream,
        )

        records_for_unpack = rx_ptr
        if ROUTE_MODE == "fanout":
            fanout = build_dispatch_fanout_lsa(
                layout, node_ranks=LOCAL_WORLD
            )
            fanout_dest, fanout_slot, fanout_valid = _proxy_fanout_plan(
                local_rank, device
            )
            fanout_error = torch.zeros(
                1, dtype=torch.int32, device=device
            )
            fanout.launch(
                rx_ptr,
                arena_window.handle,
                fanout_dest.data_ptr(),
                fanout_slot.data_ptr(),
                fanout_valid.data_ptr(),
                TOKENS,
                GENERATION,
                ptrs.parity,
                fanout_error.data_ptr(),
                stream=stream,
            )
            torch.cuda.synchronize(device)
            # Every proxy has drained its remote writes before the collective;
            # each target can now validate its preallocated contiguous slots.
            comm.barrier()
            fanout_count = cco.read_window_u32(ptrs.fanout_count, 1)[0]
            fanout_ready = cco.read_window_u64(ptrs.fanout_ready, TOKENS)
            local_failures += int(fanout_error.item() != 0)
            local_failures += int(fanout_count != TOKENS)
            local_failures += sum(
                int(value != GENERATION) for value in fanout_ready
            )
            records_for_unpack = (
                arena_window.local_ptr + layout.region("fanout_inbox").offset
            )

        remote_q = torch.empty(
            (TOKENS, HIDDEN // 2), dtype=torch.uint8, device=device
        )
        remote_scales = torch.empty(
            (TOKENS, HIDDEN // 32), dtype=torch.uint8, device=device
        )
        remote_ids = torch.empty(
            (TOKENS, TOPK), dtype=torch.int32, device=device
        )
        remote_weights = torch.empty(
            (TOKENS, TOPK), dtype=torch.float32, device=device
        )
        remote_sources = torch.empty(
            TOKENS, dtype=torch.int32, device=device
        )
        remote_masks = torch.empty(
            TOKENS, dtype=torch.int64, device=device
        )
        dispatch_unpack_error = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        dispatch_module.launch_unpack(
            records_for_unpack,
            remote_q.data_ptr(),
            remote_scales.data_ptr(),
            remote_ids.data_ptr(),
            remote_weights.data_ptr(),
            remote_sources.data_ptr(),
            remote_masks.data_ptr(),
            TOKENS,
            GLOBAL_EXPERTS,
            GLOBAL_TOKENS,
            dispatch_unpack_error.data_ptr(),
            stream=stream,
        )
        torch.cuda.synchronize(device)

        expected_remote = _expected_target_records(
            local_rank, remote_node, device
        )
        expected_remote_q = expected_remote["q"].view(torch.uint8)
        expected_remote_scales = expected_remote["scales"]
        remote_scales_typed = remote_scales.view(expected_remote_scales.dtype)
        local_failures += int(dispatch_pack_error.item() != 0)
        local_failures += int(dispatch_unpack_error.item() != 0)
        local_failures += int(not torch.equal(remote_q, expected_remote_q))
        local_failures += int(
            not torch.equal(
                remote_scales,
                expected_remote_scales.view(torch.uint8),
            )
        )
        local_failures += int(
            not torch.equal(remote_sources, expected_remote["sources"])
        )
        local_failures += int(
            not torch.equal(remote_ids, expected_remote["ids"])
        )
        local_failures += int(
            not torch.equal(remote_masks, expected_remote["masks"])
        )
        local_failures += int(
            not torch.equal(remote_weights, expected_remote["weights"])
        )

        prepacked = _assemble_h1_prepacked(
            source_payload,
            remote_q,
            remote_scales_typed,
            remote_ids,
            remote_weights,
            remote_sources,
            remote_masks,
            rank,
            node,
        )
        max_sorted = int(prepacked["sorted_ids"].shape[0])
        max_m_tiles = (max_sorted + BM - 1) // BM
        active_m_tiles = int(prepacked["nvalid"][0].item()) // BM
        if active_m_tiles <= 0 or max_m_tiles > layout.max_m_tiles:
            raise RuntimeError("rebuilt route has invalid M-tile geometry")

        h1_sidecar.publish_plan_expected(
            GENERATION,
            ptrs,
            active_m_tiles,
            expected_per_tile=1,
            stream=stream,
        )
        h1_sidecar.mark_chunk_ready(
            arena_window.local_ptr,
            SLOT,
            GENERATION,
            ptrs,
            0,
            active_m_tiles,
            delta=1,
            stream=stream,
        )

        scale_rows = (max_sorted + 255) // 256 * 256
        scale_cols = ((INTER // 32) + 7) // 8 * 8
        h1_q = torch.zeros(
            (max_sorted, INTER // 2), dtype=torch.uint8, device=device
        )
        h1_s = torch.zeros(
            scale_rows * scale_cols, dtype=torch.uint8, device=device
        )
        dummy_hidden = torch.zeros(
            (prepacked["local_rows"], HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        h1 = compile_hier_stage1_ready_a4w4(
            D_HIDDEN=HIDDEN,
            D_INTER=INTER,
            NE=LOCAL_EXPERTS,
            TOPK=TOPK,
            activation="silu",
        )
        h1_work = active_m_tiles * h1.num_n_blocks
        h1(
            ptrs.plan_ready,
            ptrs.h1_input_ready,
            ptrs.h1_input_expected,
            ptrs.h1_output_done,
            ptrs.h1_output_ready,
            GENERATION,
            prepacked["a1q"].data_ptr(),
            prepacked["a1ss"].data_ptr(),
            prepacked["prepared"].w1.data_ptr(),
            prepacked["prepared"].w1_scale.data_ptr(),
            prepacked["sorted_eids"].data_ptr(),
            prepacked["nvalid"].data_ptr(),
            prepacked["m_indices"].data_ptr(),
            prepacked["local_rows"],
            h1_work,
            h1_q.data_ptr(),
            h1_s.data_ptr(),
            dummy_hidden.data_ptr(),
            stream=stream,
        )
        torch.cuda.synchronize(device)

        h1_done = cco.read_window_u32(ptrs.h1_output_done, active_m_tiles)
        h1_ready = cco.read_window_u64(ptrs.h1_output_ready, active_m_tiles)
        local_failures += sum(int(value != h1.num_n_blocks) for value in h1_done)
        local_failures += sum(int(value != GENERATION) for value in h1_ready)

        h1_sidecar.return_credit(
            dc.ptr,
            arena_window.handle,
            remote_node,
            SLOT,
            GENERATION,
            stream=stream,
        )
        h1_sidecar.reclaim_dispatch(
            dc.ptr,
            arena_window.local_ptr,
            SLOT,
            GENERATION,
            stream=stream,
        )
        torch.cuda.synchronize(device)
        comm.barrier()

        # Existing rank-local GMM2 is the numerical reference before the Gloo
        # all-reduce. The cascade H2 writes the same weighted partial directly
        # into the aligned CCO/LSA-visible rank-partial region.
        reference_rank = torch.zeros(
            (GLOBAL_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device
        )
        mxfp4_moe_gemm2(
            inter_sorted_quant=h1_q,
            inter_sorted_shuffled_scale=h1_s,
            w2_u8=prepacked["prepared"].w2.view(torch.uint8),
            w2_scale_u8=prepacked["prepared"].w2_scale.view(torch.uint8),
            sorted_expert_ids=prepacked["sorted_eids"],
            cumsum_tensor=prepacked["nvalid"],
            sorted_token_ids=prepacked["global_stids"],
            sorted_weights=prepacked["sorted_weights"],
            out=reference_rank,
            M_logical=GLOBAL_TOKENS,
            max_sorted=max_sorted,
            NE=LOCAL_EXPERTS,
            D_HIDDEN=HIDDEN,
            D_INTER=INTER,
            topk=TOPK,
            BM=BM,
            BN=128,
            BK=128,
            a_dtype="fp4",
            epilog="atomic",
            SBM=BM,
            HIDDEN_MAX=HIDDEN,
            INTER_MAX=INTER,
        )
        h2 = compile_hier_stage2_partial_a4w4(
            D_HIDDEN=HIDDEN,
            D_INTER=INTER,
            NE=LOCAL_EXPERTS,
            TOPK=TOPK,
        )
        h2_work = active_m_tiles * h2.num_n_blocks
        h2(
            ptrs.h1_output_ready,
            ptrs.h2_output_done,
            ptrs.h2_output_ready,
            GENERATION,
            h1_q.data_ptr(),
            h1_s.data_ptr(),
            prepacked["prepared"].w2.data_ptr(),
            prepacked["prepared"].w2_scale.data_ptr(),
            prepacked["sorted_eids"].data_ptr(),
            prepacked["nvalid"].data_ptr(),
            prepacked["global_stids"].data_ptr(),
            prepacked["sorted_weights"].data_ptr(),
            GLOBAL_TOKENS,
            max_m_tiles,
            h2_work,
            rank_partial_ptr,
            stream=stream,
        )
        torch.cuda.synchronize(device)
        h2_done = cco.read_window_u32(ptrs.h2_output_done, active_m_tiles)
        h2_ready = cco.read_window_u64(ptrs.h2_output_ready, active_m_tiles)
        local_failures += sum(int(value != h2.num_n_blocks) for value in h2_done)
        local_failures += sum(int(value != GENERATION) for value in h2_ready)

        rank_partial_copy = torch.empty_like(reference_rank)
        _hip_memcpy_d2d(
            rank_partial_copy.data_ptr(),
            rank_partial_ptr,
            rank_partial_bytes,
        )
        local_failures += int(
            _logits_diff(reference_rank, rank_partial_copy) >= 1.0e-2
        )

        # The explicit Gloo reference sums every existing rank-local partial.
        # It is CPU/Gloo by design and therefore independent of CCO/LSA.
        reference_world = reference_rank.float().cpu()
        dist.all_reduce(reference_world, op=dist.ReduceOp.SUM)
        comm.barrier()
        _barrier("rank-partials-complete")

        route_expected = torch.full(
            (GLOBAL_TOKENS,), LOCAL_WORLD, dtype=torch.int32, device=device
        )
        route_ready = route_expected.clone()
        node_partial = torch.zeros(
            (GLOBAL_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device
        )
        node_ready = torch.zeros(
            GLOBAL_TOKENS, dtype=torch.int64, device=device
        )
        lsa_reduce = compile_node_partial_reduce_lsa(
            D_HIDDEN=HIDDEN,
            NUM_RANKS=LOCAL_WORLD,
            output_dtype="bf16",
        )
        lsa_reduce(
            arena_window.handle,
            rank_partial_offset,
            route_ready.data_ptr(),
            route_expected.data_ptr(),
            node_partial.data_ptr(),
            node_ready.data_ptr(),
            GENERATION,
            GLOBAL_TOKENS,
            stream=stream,
        )
        torch.cuda.synchronize(device)
        local_start = rank * TOKENS
        remote_start = remote_rank * TOKENS
        local_node_norm = float(
            node_partial[local_start : local_start + TOKENS]
            .float()
            .norm()
            .item()
        )
        remote_node_norm = float(
            node_partial[remote_start : remote_start + TOKENS]
            .float()
            .norm()
            .item()
        )
        local_failures += int(local_node_norm == 0.0)
        local_failures += int(remote_node_norm == 0.0)

        # Device pack selects the 16 rows owned by the aligned remote source
        # rank and writes BF16 payload + source ID + checked zero padding.
        partial_tx_offset = layout.ring_chunk_offset("partial_tx", SLOT)
        partial_rx_offset = layout.ring_chunk_offset("partial_rx", SLOT)
        partial_tx_ptr = arena_window.local_ptr + partial_tx_offset
        partial_rx_ptr = arena_window.local_ptr + partial_rx_offset
        return_bytes = TOKENS * PARTIAL_RECORD_BYTES
        record_module = build_partial_record_module(HIDDEN)
        source_ids = torch.arange(
            remote_start,
            remote_start + TOKENS,
            dtype=torch.int32,
            device=device,
        )
        pack_error = torch.zeros(1, dtype=torch.int32, device=device)
        record_module.launch_pack(
            node_partial.data_ptr(),
            source_ids.data_ptr(),
            partial_tx_ptr,
            TOKENS,
            GLOBAL_TOKENS,
            pack_error.data_ptr(),
            stream=stream,
        )
        cco.zero_window(partial_rx_ptr, return_bytes)
        torch.cuda.synchronize(device)
        comm.barrier()

        h2_sidecar.post_partial_return(
            dc.ptr,
            arena_window.handle,
            arena_window.local_ptr,
            remote_node,
            SLOT,
            GENERATION,
            stream=stream,
        )
        # First use the publisher only as a system-acquire CCO-ready wait. The
        # resulting dummy generation is not consumed by final combine.
        wire_ready = torch.zeros(TOKENS, dtype=torch.int64, device=device)
        h2_sidecar.module.launch_publish_received(
            arena_window.local_ptr,
            SLOT,
            GENERATION,
            wire_ready.data_ptr(),
            0,
            TOKENS,
            stream=stream,
        )

        remote_rows = torch.empty(
            (TOKENS, HIDDEN), dtype=torch.bfloat16, device=device
        )
        remote_sources = torch.empty(
            TOKENS, dtype=torch.int32, device=device
        )
        unpack_error = torch.zeros(1, dtype=torch.int32, device=device)
        record_module.launch_unpack(
            partial_rx_ptr,
            remote_rows.data_ptr(),
            remote_sources.data_ptr(),
            TOKENS,
            GLOBAL_TOKENS,
            unpack_error.data_ptr(),
            stream=stream,
        )
        torch.cuda.synchronize(device)
        expected_sources = torch.arange(
            local_start,
            local_start + TOKENS,
            dtype=torch.int32,
            device=device,
        )
        local_failures += int(pack_error.item() != 0)
        local_failures += int(unpack_error.item() != 0)
        local_failures += int(
            torch.count_nonzero(wire_ready != GENERATION).item() != 0
        )
        local_failures += int(
            torch.count_nonzero(remote_sources != expected_sources).item() != 0
        )

        # Only after unpack/source/padding validation do we publish the
        # consumer-visible remote rows used by final combine.
        remote_node_ready = torch.zeros(
            TOKENS, dtype=torch.int64, device=device
        )
        h2_sidecar.module.launch_publish_received(
            arena_window.local_ptr,
            SLOT,
            GENERATION,
            remote_node_ready.data_ptr(),
            0,
            TOKENS,
            stream=stream,
        )

        final = torch.zeros(
            (TOKENS, HIDDEN), dtype=torch.bfloat16, device=device
        )
        final_ready = torch.zeros(TOKENS, dtype=torch.int64, device=device)
        combine = compile_final_combine(
            D_HIDDEN=HIDDEN,
            local_dtype="bf16",
            remote_dtype="bf16",
            output_dtype="bf16",
        )
        combine(
            node_partial.data_ptr() + local_start * PARTIAL_ROW_BYTES,
            remote_rows.data_ptr(),
            node_ready.data_ptr() + local_start * 8,
            remote_node_ready.data_ptr(),
            final.data_ptr(),
            final_ready.data_ptr(),
            GENERATION,
            TOKENS,
            stream=stream,
        )
        torch.cuda.synchronize(device)

        expected = reference_world[
            local_start : local_start + TOKENS
        ].to(torch.bfloat16)
        final_cpu = final.cpu()
        final_diff = _logits_diff(expected, final_cpu)
        local_failures += int(final_diff >= 1.0e-2)
        local_failures += int(
            torch.count_nonzero(final_ready != GENERATION).item() != 0
        )

        h2_sidecar.return_credit(
            dc.ptr,
            arena_window.handle,
            remote_node,
            SLOT,
            GENERATION,
            stream=stream,
        )
        h2_sidecar.reclaim_partial(
            dc.ptr,
            arena_window.local_ptr,
            SLOT,
            GENERATION,
            stream=stream,
        )
        torch.cuda.synchronize(device)
        comm.barrier()
        _barrier("cascade-complete")

    return local_failures, final_diff


def main() -> int:
    dist.init_process_group(
        "gloo", timeout=timedelta(seconds=TIMEOUT_SECONDS)
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    error = ""
    local_failures = 0
    final_diff = float("inf")

    try:
        if ROUTE_MODE not in ("aligned", "fanout"):
            raise ValueError(
                "MEGAMOE_CASCADE_ROUTE_MODE must be aligned or fanout"
            )
        if world != WORLD or local_world != LOCAL_WORLD:
            raise ValueError(
                f"requires 2x8 world=16/local_world=8, got {world}/{local_world}"
            )
        if local_rank != rank % LOCAL_WORLD:
            raise ValueError(
                f"requires node-major ranks, rank={rank} local_rank={local_rank}"
            )
        torch.cuda.set_device(local_rank)
        uid = _broadcast_cco_uid(rank)
        _barrier("preflight")
        local_failures, final_diff = _run_cascade(rank, local_rank, uid)
    except Exception:
        error = traceback.format_exc()
        local_failures += 1

    # Every non-hung rank reports the same global state. The process-group
    # timeout and monitored stage barriers turn peer loss into an exception
    # instead of an unbounded Gloo wait.
    diagnostics = [None for _ in range(world)]
    try:
        dist.all_gather_object(
            diagnostics,
            {
                "rank": rank,
                "failures": local_failures,
                "final_diff": final_diff,
                "error": error,
            },
        )
        global_failures = sum(int(item["failures"]) for item in diagnostics)
    except Exception:
        global_failures = max(1, local_failures)
        diagnostics = []

    if error:
        print(error, flush=True)
    print(
        f"MEGAMOE_EP16_CASCADE_{'PASS' if global_failures == 0 else 'FAIL'} "
        f"rank={rank} local_rank={local_rank} route_mode={ROUTE_MODE} "
        f"seam=generic-sort-fanout "
        f"final_logits_diff={final_diff:.8e} local_failures={local_failures} "
        f"global_failures={global_failures}",
        flush=True,
    )
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    return 0 if global_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

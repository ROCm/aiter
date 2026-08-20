# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Production DP/TP AllGather + ready-aware MoE Stage1 backend."""

from __future__ import annotations

import atexit
import ctypes
import functools
import logging
import math
import os
from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, gpu
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from mori.cco import (
    CCODevCommRequirements,
    GDA_CONNECTION_NONE,
    Communicator,
    UniqueId,
)
from mori.cco.device import flydsl as cco_dsl
from mori.cco.device._build_flags import BUILD_CCO_SDMA
from mori.tensor_utils import from_gpu_ptr

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.kernels import buffer_ops, communication_ops_utils as comm_ops
from aiter.ops.flydsl.moe_stage1_ready import (
    PreparedStage1Input,
    Stage1ReadyPlan,
)

logger = logging.getLogger("aiter")

TP_STAGE1_AG_KID = (
    "flydsl_moe1_afp8_wfp4_bf16_t64x384x256_tpag_v2"
)
TP_STAGE1_AG_N_PARTITIONS = 2
TP_STAGE1_AG_MIN_GLOBAL_TOKENS = 2048
_QUANT_BLOCK = 64
_QUANT_GROUP = 32
_READY_GROUP_SIZE = 2
_FP8_E4M3_INV_MAX_POS_BITS = 0x3B124925


@dataclass
class _CcoWorldState:
    comm: Communicator
    dev_comm: object
    peer_lsa_ranks: torch.Tensor
    transport_stream: torch.cuda.Stream


_CCO_WORLD: _CcoWorldState | None = None


def _clear_hip_runtime_error() -> None:
    """Discard the stale HIP status left by current CCO communicator init."""

    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipGetLastError.argtypes = []
    hip.hipGetLastError.restype = ctypes.c_int
    hip.hipGetLastError()


def preinitialize_tp_stage1_cco_world() -> None:
    """Create the world CCO communicator before AITER's device communicators.

    MORI CCO and AITER's PyNCCL/custom communicator both establish peer VA
    mappings.  CCO must be first; creating it lazily from the MoE layer makes
    later RCCL collectives fail on current ROCm runtimes.
    """

    global _CCO_WORLD
    if _CCO_WORLD is not None:
        return
    if not dist.is_initialized():
        raise RuntimeError("CCO preinitialization requires torch.distributed")
    if not BUILD_CCO_SDMA:
        raise RuntimeError(
            "Stage1 AG requires an amd_mori build with BUILD_CCO_SDMA=ON"
        )
    world = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    if rank == 0:
        uid_bytes = bytes(Communicator.get_unique_id())
        uid_tensor = torch.tensor(
            list(uid_bytes), dtype=torch.uint8, device=device
        )
    else:
        uid_tensor = torch.empty(128, dtype=torch.uint8, device=device)
    dist.broadcast(uid_tensor, src=0)
    uid = UniqueId.from_bytes(bytes(uid_tensor.cpu().tolist()))
    comm = Communicator.init(world, rank, uid, per_rank_vmm=1 << 30)
    requirements = CCODevCommRequirements()
    requirements.gda_connection_type = GDA_CONNECTION_NONE
    requirements.gda_context_count = 0
    requirements.gda_signal_count = 0
    requirements.gda_counter_count = 0
    requirements.sdma_queue_count = world
    dev_comm = comm.create_dev_comm(requirements)
    if dev_comm.lsa_size != world:
        comm.destroy()
        raise RuntimeError("Stage1 CCO requires one intra-node world LSA team")
    local_lsa_rank = torch.tensor(
        [dev_comm.lsa_rank], dtype=torch.int32, device=device
    )
    peer_lsa_ranks = torch.empty(world, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(peer_lsa_ranks, local_lsa_rank)
    if sorted(peer_lsa_ranks.cpu().tolist()) != list(range(world)):
        comm.destroy()
        raise RuntimeError("CCO LSA ranks do not map one-to-one to world ranks")
    comm.barrier()
    _clear_hip_runtime_error()
    _CCO_WORLD = _CcoWorldState(
        comm=comm,
        dev_comm=dev_comm,
        peer_lsa_ranks=peer_lsa_ranks,
        transport_stream=torch.cuda.Stream(device=device),
    )


@functools.cache
def _build_chunk_major_quant(n: int, source_count: int, chunks_per_source: int):
    """Build local MXFP8 quantization directly into chunk-major AG slots."""
    scale_n = n // _QUANT_GROUP

    @flyc.kernel(
        name=(
            f"moe_stage1_ag_quant_n{n}"
            f"_s{source_count}x{chunks_per_source}_v1"
        )
    )
    def quant_kernel(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        m: fx.Int32,
        rank: fx.Int32,
    ):
        in_rsrc = buffer_ops.create_buffer_resource(x, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(y, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)
        group_id = fx.block_idx.x * fx.Int32(_QUANT_BLOCK) + fx.thread_idx.x
        if group_id < m * fx.Int32(scale_n):
            local_row = group_id // fx.Int32(scale_n)
            scale_col = group_id % fx.Int32(scale_n)
            tokens_per_chunk = m // fx.Int32(chunks_per_source)
            chunk = local_row // tokens_per_chunk
            chunk_offset = local_row % tokens_per_chunk
            physical_row = (
                (chunk * fx.Int32(source_count) + rank) * tokens_per_chunk
                + chunk_offset
            )
            physical_group = physical_row * fx.Int32(scale_n) + scale_col

            input_word = group_id * fx.Int32(_QUANT_GROUP * 2 // 4)
            values = []
            local_max = fx.Float32(1e-10)
            for chunk_i in range_constexpr(_QUANT_GROUP // 8):
                raw = buffer_ops.buffer_load(
                    in_rsrc,
                    input_word + fx.Int32(chunk_i * 4),
                    vec_width=4,
                    dtype=T.i32,
                )
                chunk_values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                local_max = local_max.maximumf(
                    fmath.absf(chunk_values).reduce(ReductionOp.MAX)
                )
                for element in range_constexpr(8):
                    values.append(chunk_values[element])

            working = (
                local_max
                * fx.Int32(_FP8_E4M3_INV_MAX_POS_BITS).bitcast(fx.Float32)
            ).bitcast(fx.Int32)
            mantissa = working & fx.Int32(0x7FFFFF)
            biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
            e8m0 = (mantissa != fx.Int32(0)).select(
                biased_exp + fx.Int32(1), biased_exp
            )
            e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
            buffer_ops.buffer_store(
                e8m0.to(fx.Uint8),
                scale_rsrc,
                physical_group,
                offset_is_bytes=True,
            )

            quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(
                fx.Float32
            )
            output_word = physical_group * fx.Int32(_QUANT_GROUP // 4)
            scaled = [
                values[index] * quant_scale
                for index in range_constexpr(_QUANT_GROUP)
            ]
            for half in range_constexpr(2):
                words = []
                for word in range_constexpr(4):
                    base = (half * 4 + word) * 4
                    packed = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        scaled[base],
                        scaled[base + 1],
                        fx.Int32(0),
                        0,
                    )
                    packed = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        scaled[base + 2],
                        scaled[base + 3],
                        packed,
                        1,
                    )
                    words.append(packed)
                buffer_ops.buffer_store(
                    fx.Vector.from_elements(words, fx.Int32),
                    out_rsrc,
                    output_word + fx.Int32(half * 4),
                )

    @flyc.jit
    def launch(
        x: fx.Tensor,
        y: fx.Tensor,
        scale: fx.Tensor,
        m: fx.Int32,
        rank: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        quant_kernel(x, y, scale, m, rank).launch(
            grid=(fx.Int64(grid_blocks), 1, 1),
            block=(_QUANT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch


def _quantize_chunk_major(
    x: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    *,
    rank: int,
    source_count: int,
    chunks_per_source: int,
) -> None:
    if x.dtype != torch.bfloat16 or x.ndim != 2 or x.shape[1] % _QUANT_GROUP:
        raise ValueError("chunk-major quant input must be BF16 [M, N]")
    m, n = map(int, x.shape)
    if not 0 <= rank < source_count or m % chunks_per_source:
        raise ValueError("chunk-major quant rank/chunk geometry is invalid")
    if y.element_size() != 1 or y.numel() != m * source_count * n:
        raise ValueError("chunk-major FP8 destination has the wrong size")
    if scale.element_size() != 1 or scale.numel() != (
        m * source_count * (n // _QUANT_GROUP)
    ):
        raise ValueError("chunk-major scale destination has the wrong size")
    if not x.is_contiguous() or not y.is_contiguous() or not scale.is_contiguous():
        raise ValueError("chunk-major quant tensors must be contiguous")
    if x.device != y.device or x.device != scale.device:
        raise ValueError("chunk-major quant tensors must share one device")
    grid_blocks = (m * (n // _QUANT_GROUP) + _QUANT_BLOCK - 1) // _QUANT_BLOCK
    _build_chunk_major_quant(n, source_count, chunks_per_source)(
        x,
        y.view(torch.uint8),
        scale.view(torch.uint8),
        m,
        rank,
        grid_blocks,
        stream=fx.Stream(torch.cuda.current_stream(x.device)),
    )


def signed_i32(value: int) -> int:
    value = int(value) & 0xFFFFFFFF
    return value if value < (1 << 31) else value - (1 << 32)


def _global_i32_ptr(addr_i64):
    return llvm.IntToPtrOp(
        llvm.PointerType.get(address_space=1), arith.unwrap(addr_i64)
    ).result


def _load_i32_system_acquire(addr_i64):
    return llvm.LoadOp(
        ir.IntegerType.get_signless(32),
        _global_i32_ptr(addr_i64),
        alignment=4,
        ordering=llvm.AtomicOrdering.acquire,
        syncscope="one-as",
    ).result


def _increment_i32_system(addr_i64):
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        _global_i32_ptr(addr_i64),
        arith.unwrap(fx.Int32(1)),
        llvm.AtomicOrdering.acq_rel,
        syncscope="one-as",
    ).res


def _wait_i32_at_least(addr_i64, expected):
    first = _load_i32_system_acquire(addr_i64)
    loop = scf.WhileOp([T.i32], [first])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = fx.Int32(before.arguments[0])
        keep_waiting = fx.Uint32(current) < fx.Uint32(expected)
        scf.ConditionOp(keep_waiting.ir_value(), [before.arguments[0]])
    with ir.InsertionPoint(after):
        fx.rocdl.s_sleep(fx.Int32(2))
        next_value = _load_i32_system_acquire(addr_i64)
        scf.YieldOp([next_value])


@functools.cache
def _build_cco_route_allgather(
    padded_rows: int,
    topk: int,
    source_count: int,
    rank: int,
    chunks_per_source: int,
):
    rows_per_chunk = padded_rows // chunks_per_source
    chunk_bytes = rows_per_chunk * topk * 4
    remote_sources = tuple(
        (rank + offset) % source_count for offset in range(1, source_count)
    )
    remote_destinations = tuple(
        (rank - offset) % source_count for offset in range(1, source_count)
    )

    @flyc.kernel(
        name=(
            f"moe_stage1_cco_route_ag_m{padded_rows}_t{topk}"
            f"_s{source_count}x{chunks_per_source}_r{rank}_ordered_v2"
        ),
        known_block_size=[64, 1, 1],
    )
    def route_allgather_kernel(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        peer_lsa_ranks: fx.Tensor,
        route_epoch_offset: fx.Int64,
        route_source_epoch_offset: fx.Int64,
        global_ids_offset: fx.Int64,
        global_weights_offset: fx.Int64,
    ):
        thread = fx.thread_idx.x
        dc = cco_dsl.DevComm(dev_comm)
        window = cco_dsl.Window(window_handle)
        local_base = window.lsa_ptr(dc.lsa_rank, fx.Int64(0))
        local_epoch = local_base + route_epoch_offset
        if thread == fx.Int32(0):
            _increment_i32_system(local_epoch)
        if thread == fx.Int32(0):
            expected = fx.Int32(_load_i32_system_acquire(local_epoch))
            peer_map = buffer_ops.create_buffer_resource(peer_lsa_ranks)
            sdma = dc.sdma()
            signal_destination = (
                route_source_epoch_offset + fx.Int64(rank * 4)
            )
            for remote_index, destination in enumerate(remote_destinations):
                peer = buffer_ops.buffer_load(
                    peer_map,
                    destination,
                    vec_width=1,
                    dtype=T.i32,
                )
                for chunk in range_constexpr(chunks_per_source):
                    source_delta = fx.Int64(chunk * chunk_bytes)
                    destination_row = padded_rows + (
                        chunk * (source_count - 1) + remote_index
                    ) * rows_per_chunk
                    destination_delta = fx.Int64(
                        destination_row * topk * 4
                    )
                    sdma.put(
                        peer,
                        window_handle,
                        global_ids_offset + destination_delta,
                        window_handle,
                        global_ids_offset + source_delta,
                        chunk_bytes,
                        destination,
                        coop=cco_dsl.CoopScope.THREAD,
                        signal=False,
                        aggregate=True,
                    )
                    sdma.put(
                        peer,
                        window_handle,
                        global_weights_offset + destination_delta,
                        window_handle,
                        global_weights_offset + source_delta,
                        chunk_bytes,
                        destination,
                        coop=cco_dsl.CoopScope.THREAD,
                        signal=False,
                        aggregate=True,
                    )
                sdma.put(
                    peer,
                    window_handle,
                    signal_destination,
                    window_handle,
                    route_epoch_offset,
                    4,
                    destination,
                    coop=cco_dsl.CoopScope.THREAD,
                    signal=False,
                    aggregate=True,
                )
                sdma.commit(
                    peer,
                    destination,
                    coop=cco_dsl.CoopScope.THREAD,
                )
            for source in remote_sources:
                source_epoch = (
                    local_base
                    + route_source_epoch_offset
                    + fx.Int64(source * 4)
                )
                _wait_i32_at_least(source_epoch, expected)

    @flyc.jit
    def run_route_allgather(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        peer_lsa_ranks: fx.Tensor,
        route_epoch_offset: fx.Int64,
        route_source_epoch_offset: fx.Int64,
        global_ids_offset: fx.Int64,
        global_weights_offset: fx.Int64,
        stream: fx.Stream,
    ):
        route_allgather_kernel(
            dev_comm,
            window_handle,
            peer_lsa_ranks,
            route_epoch_offset,
            route_source_epoch_offset,
            global_ids_offset,
            global_weights_offset,
        ).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return run_route_allgather


@functools.cache
def _build_cco_publish_epoch():
    @flyc.kernel(
        name="moe_stage1_cco_publish_epoch",
        known_block_size=[64, 1, 1],
    )
    def publish_epoch_kernel(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        epoch_offset: fx.Int64,
    ):
        if fx.thread_idx.x == fx.Int32(0):
            dc = cco_dsl.DevComm(dev_comm)
            window = cco_dsl.Window(window_handle)
            local_epoch = (
                window.lsa_ptr(dc.lsa_rank, fx.Int64(0)) + epoch_offset
            )
            _increment_i32_system(local_epoch)

    @flyc.jit
    def run_publish_epoch(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        epoch_offset: fx.Int64,
        stream: fx.Stream,
    ):
        publish_epoch_kernel(
            dev_comm, window_handle, epoch_offset
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return run_publish_epoch


@functools.cache
def _build_cco_value_transport(
    padded_rows: int,
    hidden: int,
    source_count: int,
    rank: int,
    chunks_per_source: int,
):
    value_chunk_bytes = padded_rows * hidden // chunks_per_source
    scale_chunk_bytes = (
        padded_rows * (hidden // _QUANT_GROUP) // chunks_per_source
    )
    remote_sources = tuple(
        (rank + offset) % source_count for offset in range(1, source_count)
    )
    remote_destinations = tuple(
        (rank - offset) % source_count for offset in range(1, source_count)
    )
    stages = tuple(
        tuple(
            remote_sources[group_begin : group_begin + _READY_GROUP_SIZE]
        )
        for _chunk in range(chunks_per_source)
        for group_begin in range(0, len(remote_sources), _READY_GROUP_SIZE)
    )
    groups_per_chunk = (
        len(remote_sources) + _READY_GROUP_SIZE - 1
    ) // _READY_GROUP_SIZE
    cumulative = sum(
        1 << (rank * chunks_per_source + chunk)
        for chunk in range(chunks_per_source)
    )
    ready_masks: list[int] = []
    for stage, sources in enumerate(stages):
        chunk = stage // groups_per_chunk
        for source in sources:
            cumulative |= 1 << (source * chunks_per_source + chunk)
        ready_masks.append(signed_i32(cumulative))

    @flyc.kernel(
        name=(
            f"moe_stage1_cco_sdma_values_m{padded_rows}_h{hidden}"
            f"_s{source_count}x{chunks_per_source}_r{rank}"
            f"_g{_READY_GROUP_SIZE}_push_v4"
        ),
        known_block_size=[64, 1, 1],
    )
    def value_transport_kernel(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        peer_lsa_ranks: fx.Tensor,
        value_epoch_offset: fx.Int64,
        source_epoch_offset: fx.Int64,
        values_offset: fx.Int64,
        scales_offset: fx.Int64,
        ready: fx.Tensor,
    ):
        thread = fx.thread_idx.x
        dc = cco_dsl.DevComm(dev_comm)
        window = cco_dsl.Window(window_handle)
        sdma = dc.sdma()
        local_base = window.lsa_ptr(dc.lsa_rank, fx.Int64(0))
        expected = fx.Int32(
            _load_i32_system_acquire(local_base + value_epoch_offset)
        )
        peer_map = buffer_ops.create_buffer_resource(peer_lsa_ranks)
        ready_base = fx.Int64(fx.ptrtoint(fx.get_iter(ready)))

        def issue_chunk(chunk: int):
            physical_slot = chunk * source_count + rank
            value_offset = (
                values_offset
                + fx.Int64(physical_slot * value_chunk_bytes)
            )
            scale_offset = (
                scales_offset
                + fx.Int64(physical_slot * scale_chunk_bytes)
            )
            signal_offset = (
                source_epoch_offset
                + fx.Int64((chunk * source_count + rank) * 4)
            )
            for destination in remote_destinations:
                peer = buffer_ops.buffer_load(
                    peer_map,
                    destination,
                    vec_width=1,
                    dtype=T.i32,
                )
                queue = destination
                sdma.put(
                    peer,
                    window_handle,
                    value_offset,
                    window_handle,
                    value_offset,
                    value_chunk_bytes,
                    queue,
                    coop=cco_dsl.CoopScope.THREAD,
                    signal=False,
                )
                sdma.put(
                    peer,
                    window_handle,
                    scale_offset,
                    window_handle,
                    scale_offset,
                    scale_chunk_bytes,
                    queue,
                    coop=cco_dsl.CoopScope.THREAD,
                    signal=False,
                )
                sdma.put(
                    peer,
                    window_handle,
                    signal_offset,
                    window_handle,
                    value_epoch_offset,
                    4,
                    queue,
                    coop=cco_dsl.CoopScope.THREAD,
                    signal=False,
                )

        if thread == fx.Int32(0):
            issue_chunk(0)
            issue_chunk(1)
            for stage, sources in enumerate(stages):
                chunk = stage // groups_per_chunk
                for source in sources:
                    local_signal = (
                        local_base
                        + source_epoch_offset
                        + fx.Int64((chunk * source_count + source) * 4)
                    )
                    _wait_i32_at_least(local_signal, expected)
                comm_ops.store_i32_system(
                    ready_base,
                    fx.Int32(0),
                    fx.Int32(ready_masks[stage]),
                )
                prefetch_group = (
                    groups_per_chunk // 2 + 1
                    if chunk == 0
                    else groups_per_chunk // 2
                )
                if (
                    stage % groups_per_chunk == prefetch_group
                    and chunk + 2 < chunks_per_source
                ):
                    issue_chunk(chunk + 2)

    @flyc.jit
    def run_value_transport(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        peer_lsa_ranks: fx.Tensor,
        value_epoch_offset: fx.Int64,
        source_epoch_offset: fx.Int64,
        values_offset: fx.Int64,
        scales_offset: fx.Int64,
        ready: fx.Tensor,
        stream: fx.Stream,
    ):
        value_transport_kernel(
            dev_comm,
            window_handle,
            peer_lsa_ranks,
            value_epoch_offset,
            source_epoch_offset,
            values_offset,
            scales_offset,
            ready,
        ).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
            stream=stream,
        )

    return run_value_transport


@functools.cache
def _build_cco_consumer_barrier(source_count: int, rank: int):
    @flyc.kernel(
        name=f"moe_stage1_cco_consumer_barrier_s{source_count}_r{rank}",
        known_block_size=[64, 1, 1],
    )
    def consumer_barrier_kernel(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        peer_lsa_ranks: fx.Tensor,
        consumer_epoch_offset: fx.Int64,
    ):
        thread = fx.thread_idx.x
        dc = cco_dsl.DevComm(dev_comm)
        window = cco_dsl.Window(window_handle)
        local_base = window.lsa_ptr(dc.lsa_rank, fx.Int64(0))
        local_epoch = local_base + consumer_epoch_offset
        if thread == fx.Int32(0):
            _increment_i32_system(local_epoch)
        gpu.barrier()
        expected = fx.Int32(_load_i32_system_acquire(local_epoch))
        peer_map = buffer_ops.create_buffer_resource(peer_lsa_ranks)
        if thread == fx.Int32(0):
            for source in range_constexpr(source_count):
                if source != rank:
                    peer = buffer_ops.buffer_load(
                        peer_map,
                        source,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    remote_epoch = (
                        window.lsa_ptr(peer, fx.Int64(0))
                        + consumer_epoch_offset
                    )
                    _wait_i32_at_least(remote_epoch, expected)
        gpu.barrier()

    @flyc.jit
    def run_consumer_barrier(
        dev_comm: fx.Int64,
        window_handle: fx.Int64,
        peer_lsa_ranks: fx.Tensor,
        consumer_epoch_offset: fx.Int64,
        stream: fx.Stream,
    ):
        consumer_barrier_kernel(
            dev_comm,
            window_handle,
            peer_lsa_ranks,
            consumer_epoch_offset,
        ).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return run_consumer_barrier


@triton.jit
def _bitwise_or(left, right):
    return left | right


@triton.jit
def _build_tile_source_masks_kernel(
    sorted_ids,
    masks,
    tokens,
    sorted_count,
    tokens_per_chunk,
    tokens_per_source: tl.constexpr,
    source_count: tl.constexpr,
    chunks_per_source: tl.constexpr,
    rank: tl.constexpr,
    TILE_M: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tile * TILE_M + tl.arange(0, TILE_M)
    row_valid = row < sorted_count
    fused = tl.load(sorted_ids + row, mask=row_valid, other=tokens).to(tl.int32)
    ordered_row = fused & 0xFFFFFF
    rows_per_chunk = tokens_per_source // chunks_per_source
    remote_offset = ordered_row - tokens_per_source
    chunk_span = (source_count - 1) * rows_per_chunk
    chunk = remote_offset // chunk_span
    within_chunk = remote_offset % chunk_span
    remote_index = within_chunk // rows_per_chunk
    chunk_offset = within_chunk % rows_per_chunk
    remote_source = (rank + 1 + remote_index) % source_count
    remote_row = (
        remote_source * tokens_per_source
        + chunk * rows_per_chunk
        + chunk_offset
    )
    token_id = tl.where(
        ordered_row < tokens_per_source,
        rank * tokens_per_source + ordered_row,
        remote_row,
    )
    valid = ordered_row < tokens
    restored = (fused & -16777216) | token_id
    tl.store(
        sorted_ids + row,
        tl.where(valid, restored, fused),
        mask=row_valid,
    )
    dependency = token_id // tokens_per_chunk
    dependency = tl.where(valid, dependency, 0)
    bits = tl.where(valid, 1 << dependency, 0)
    mask = tl.reduce(bits, axis=0, combine_fn=_bitwise_or)
    tl.store(masks + tile, mask)


def _build_tile_source_masks(
    sorted_ids: torch.Tensor,
    tile_count: int,
    *,
    tile_m: int,
    tokens: int,
    tokens_per_source: int,
    source_count: int,
    chunks_per_source: int,
    rank: int,
) -> torch.Tensor:
    dependency_count = source_count * chunks_per_source
    if dependency_count > 32:
        raise ValueError("tile dependency masks currently support at most 32 bits")
    if tokens_per_source % chunks_per_source:
        raise ValueError("tokens_per_source must divide evenly into chunks")
    masks = torch.empty(tile_count, dtype=torch.int32, device=sorted_ids.device)
    _build_tile_source_masks_kernel[(tile_count,)](
        sorted_ids,
        masks,
        tokens,
        sorted_ids.numel(),
        tokens_per_source // chunks_per_source,
        tokens_per_source=tokens_per_source,
        source_count=source_count,
        chunks_per_source=chunks_per_source,
        rank=rank,
        TILE_M=tile_m,
        num_warps=1,
    )
    return masks


@dataclass(frozen=True)
class TpStage1AgConfig:
    hidden: int
    experts: int
    topk: int
    chunks_per_source: int = 4
    queue_workers: int = 256
    slot_count: int = 1
    min_global_tokens: int = 32768

    def validate(self, world: int) -> None:
        if world != 8:
            raise ValueError("Stage1 AG currently requires exactly 8 ranks")
        if self.hidden != 7168 or self.experts != 384 or self.topk != 6:
            raise ValueError(
                "Stage1 AG supports only H=7168, E=384, topk=6"
            )
        if self.chunks_per_source != 4:
            raise ValueError("Stage1 CCO SDMA requires four chunks per source")
        if self.slot_count <= 0:
            raise ValueError("at least one workspace slot is required")
        if self.min_global_tokens < TP_STAGE1_AG_MIN_GLOBAL_TOKENS:
            raise ValueError(
                "Stage1 AG requires min_global_tokens >= "
                f"{TP_STAGE1_AG_MIN_GLOBAL_TOKENS} so Stage2 uses a "
                "block_m=64-compatible kernel"
            )


class _CcoArena:
    """One symmetric CCO allocation exposed as non-owning torch views."""

    def __init__(self, comm: Communicator):
        self.comm = comm
        self._size = 0
        self._specs: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
        self.offsets: dict[str, int] = {}
        self.memory = None
        self.window = None
        self.tensors: dict[str, torch.Tensor] = {}

    @staticmethod
    def _align(value: int, alignment: int = 256) -> int:
        return (value + alignment - 1) // alignment * alignment

    def reserve(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        if self.memory is not None or name in self._specs:
            raise RuntimeError("invalid CCO arena reservation")
        offset = self._align(self._size)
        element_size = torch.empty(0, dtype=dtype).element_size()
        size = math.prod(shape) * element_size
        self.offsets[name] = offset
        self._specs[name] = (offset, shape, dtype)
        self._size = offset + size

    def allocate(self) -> None:
        if self.memory is not None:
            raise RuntimeError("CCO arena is already allocated")
        size = self._align(self._size, 2 * 1024 * 1024)
        self.memory = self.comm.alloc_mem(size)
        self.window = self.comm.register_window(self.memory.ptr, size)
        self.tensors = {
            name: from_gpu_ptr(self.memory.ptr + offset, shape, dtype)
            for name, (offset, shape, dtype) in self._specs.items()
        }

    def tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class _TpStage1AgSlot:
    def __init__(
        self,
        backend: TpStage1AgBackend,
        padded_rows: int,
    ):
        self.backend = backend
        self.config = backend.config
        self.group = backend.group
        self.rank = backend.rank
        self.world = backend.world
        self.device = backend.device
        self.padded_rows = int(padded_rows)
        self.global_rows = self.padded_rows * self.world
        chunks = self.config.chunks_per_source
        if self.padded_rows % chunks:
            raise ValueError("padded rows must divide evenly into transport chunks")
        rows_per_chunk = self.padded_rows // chunks

        self.hidden_padded = torch.empty(
            (self.padded_rows, self.config.hidden),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.arena = _CcoArena(backend.cco_comm)
        self.arena.reserve(
            "values",
            (self.global_rows, self.config.hidden),
            dtypes.fp8,
        )
        self.arena.reserve(
            "scales",
            (self.global_rows, self.config.hidden // 32),
            torch.uint8,
        )
        self.arena.reserve(
            "global_topk_ids",
            (self.global_rows, self.config.topk),
            torch.int32,
        )
        self.arena.reserve(
            "global_topk_weights",
            (self.global_rows, self.config.topk),
            torch.float32,
        )
        for name in ("route_epoch", "value_epoch", "consumer_epoch"):
            self.arena.reserve(name, (1,), torch.int32)
        self.arena.reserve(
            "source_epochs", (chunks, self.world), torch.int32
        )
        self.arena.reserve(
            "route_source_epochs", (self.world,), torch.int32
        )
        self.arena.allocate()

        self.values = self.arena.tensor("values")
        self.scales = self.arena.tensor("scales")
        self.value_layout = self.values.view(
            chunks, self.world, rows_per_chunk, self.config.hidden
        )
        self.scale_layout = self.scales.view(
            chunks,
            self.world,
            rows_per_chunk,
            self.config.hidden // 32,
        )
        self.global_topk_ids = self.arena.tensor("global_topk_ids")
        self.global_topk_weights = self.arena.tensor("global_topk_weights")
        partitions = TP_STAGE1_AG_N_PARTITIONS
        max_sorted_rows = (
            self.global_rows * self.config.topk
            + self.config.experts * 64
            - self.config.topk
        )
        tile_capacity = (max_sorted_rows + 63) // 64
        self.expert_queue_state = torch.zeros(
            2 * partitions + tile_capacity * partitions,
            dtype=torch.int32,
            device=self.device,
        )
        self.expert_cursor = self.expert_queue_state[:partitions]
        self.completed_tiles = self.expert_queue_state[
            partitions : 2 * partitions
        ]
        self.tile_claimed = self.expert_queue_state[2 * partitions :].view(
            tile_capacity, partitions
        )

        local_ready_mask = sum(
            1 << (self.rank * chunks + chunk) for chunk in range(chunks)
        )
        self.local_ready_mask = signed_i32(local_ready_mask)
        self.ready = torch.full(
            (1,), self.local_ready_mask, dtype=torch.int32, device=self.device
        )
        for name in (
            "route_epoch",
            "value_epoch",
            "consumer_epoch",
            "source_epochs",
            "route_source_epochs",
        ):
            self.arena.tensor(name).zero_()
        torch.cuda.synchronize(self.device)
        backend.cco_comm.barrier()

    def begin(
        self,
        hidden_local: torch.Tensor,
        topk_weights_local: torch.Tensor,
        topk_ids_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_rows = hidden_local.shape[0]
        if hidden_local.dtype != torch.bfloat16 or hidden_local.shape[1] != self.config.hidden:
            raise TypeError("Stage1 AG requires BF16 [M, 7168] input")
        if local_rows > self.padded_rows:
            raise ValueError("local rows exceed the selected workspace bucket")
        if topk_ids_local.shape != (local_rows, self.config.topk):
            raise ValueError("local topk IDs have the wrong shape")
        if topk_weights_local.shape != (local_rows, self.config.topk):
            raise ValueError("local topk weights have the wrong shape")

        self.hidden_padded[:local_rows].copy_(hidden_local)
        if local_rows < self.padded_rows:
            # A zero MXFP8 row carries the minimum e8m0 scale and can amplify
            # otherwise harmless padding differences into Inf/NaN in the GUGU
            # epilogue.  Padding routes have exactly-zero weights, so copying a
            # finite real row keeps their GEMM numerically benign without
            # changing any logical token output.  A DPA rank can legitimately
            # receive no requests; use a finite non-zero row in that case.
            if local_rows:
                self.hidden_padded[local_rows:].copy_(
                    hidden_local[:1].expand(self.padded_rows - local_rows, -1)
                )
            else:
                self.hidden_padded.fill_(1.0)
        local_ids_slot = self.global_topk_ids[: self.padded_rows]
        local_weights_slot = self.global_topk_weights[: self.padded_rows]
        local_ids_slot.zero_()
        local_weights_slot.zero_()
        local_ids_slot[:local_rows].copy_(topk_ids_local)
        local_weights_slot[:local_rows].copy_(topk_weights_local)
        current_stream = torch.cuda.current_stream(self.device)
        transport_stream = self.backend.transport_stream
        transport_stream.wait_stream(current_stream)
        with torch.cuda.stream(transport_stream):
            _build_cco_route_allgather(
                self.padded_rows,
                self.config.topk,
                self.world,
                self.rank,
                self.config.chunks_per_source,
            )(
                self.backend.cco_dev_comm.ptr,
                self.arena.window.handle,
                self.backend.peer_lsa_ranks,
                self.arena.offsets["route_epoch"],
                self.arena.offsets["route_source_epochs"],
                self.arena.offsets["global_topk_ids"],
                self.arena.offsets["global_topk_weights"],
                stream=fx.Stream(transport_stream),
            )

        _quantize_chunk_major(
            self.hidden_padded,
            self.values,
            self.scales,
            rank=self.rank,
            source_count=self.world,
            chunks_per_source=self.config.chunks_per_source,
        )
        self.ready.fill_(self.local_ready_mask)
        self.expert_queue_state.zero_()

        # Quant overlaps the compact route AllGather. Once both producers are
        # ready, value SDMA overlaps sorting and the ready-aware GEMM.
        current_stream.wait_stream(transport_stream)
        transport_stream.wait_stream(current_stream)
        with torch.cuda.stream(transport_stream):
            self.launch_transport(transport_stream)
        return self.global_topk_weights, self.global_topk_ids

    def launch_transport(self, stream: torch.cuda.Stream) -> None:
        _build_cco_publish_epoch()(
            self.backend.cco_dev_comm.ptr,
            self.arena.window.handle,
            self.arena.offsets["value_epoch"],
            stream=fx.Stream(stream),
        )
        _build_cco_value_transport(
            self.padded_rows,
            self.config.hidden,
            self.world,
            self.rank,
            self.config.chunks_per_source,
        )(
            self.backend.cco_dev_comm.ptr,
            self.arena.window.handle,
            self.backend.peer_lsa_ranks,
            self.arena.offsets["value_epoch"],
            self.arena.offsets["source_epochs"],
            self.arena.offsets["values"],
            self.arena.offsets["scales"],
            self.ready,
            stream=fx.Stream(stream),
        )

    def prepare(
        self,
        *,
        metadata,
        sorted_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
    ) -> PreparedStage1Input:
        del sorted_weights
        if int(metadata.block_m) != 64:
            raise ValueError("Stage1 AG requires sort block_m=64")
        tile_source_masks = _build_tile_source_masks(
            sorted_ids,
            sorted_expert_ids.numel(),
            tile_m=64,
            tokens=self.global_rows,
            tokens_per_source=self.padded_rows,
            source_count=self.world,
            chunks_per_source=self.config.chunks_per_source,
            rank=self.rank,
        )
        if tile_source_masks.numel() != self.tile_claimed.shape[0]:
            raise ValueError("sorting tile capacity changed after workspace allocation")
        plan = Stage1ReadyPlan(
            ready=self.ready,
            tile_source_masks=tile_source_masks,
            expert_cursor=self.expert_cursor,
            completed_tiles=self.completed_tiles,
            tile_claimed=self.tile_claimed,
            source_count=self.world,
            chunks_per_source=self.config.chunks_per_source,
            queue_workers=self.config.queue_workers,
        )
        return PreparedStage1Input(
            values=self.values,
            scales=self.scales,
            load_sorted_ids=sorted_ids,
            ready_plan=plan,
        )

    def finish(self) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        current_stream.wait_stream(self.backend.transport_stream)
        _build_cco_consumer_barrier(self.world, self.rank)(
            self.backend.cco_dev_comm.ptr,
            self.arena.window.handle,
            self.backend.peer_lsa_ranks,
            self.arena.offsets["consumer_epoch"],
            stream=fx.Stream(current_stream),
        )

    def close(self) -> None:
        self.arena.tensors.clear()


class _TpStage1AgWorkspaceSet:
    def __init__(self, backend: TpStage1AgBackend, padded_rows: int):
        self.slots = tuple(
            _TpStage1AgSlot(backend, padded_rows)
            for _ in range(backend.config.slot_count)
        )
        self.index = 0

    def acquire(self) -> _TpStage1AgSlot:
        slot = self.slots[self.index]
        self.index = (self.index + 1) % len(self.slots)
        return slot

    def close(self) -> None:
        for slot in self.slots:
            slot.close()


class TpStage1AgBackend:
    """Shared, shape-bucketed local-to-global MoE backend for one DP group."""

    def __init__(self, group, config: TpStage1AgConfig):
        self.group = group
        self.config = config
        self.world = int(group.world_size)
        self.rank = int(group.rank_in_group)
        self.device = group.device
        config.validate(self.world)
        if _CCO_WORLD is None:
            raise RuntimeError(
                "Stage1 CCO must be enabled before distributed group creation; "
                "set AITER_MOE_STAGE1_CCO=1 before init_distributed_environment"
            )
        group_ranks = dist.get_process_group_ranks(group.device_group)
        if group_ranks != list(range(dist.get_world_size())):
            raise RuntimeError(
                "Stage1 CCO currently requires the MoE gather group to be world"
            )
        self.cco_comm = _CCO_WORLD.comm
        self.cco_dev_comm = _CCO_WORLD.dev_comm
        self.peer_lsa_ranks = _CCO_WORLD.peer_lsa_ranks
        self.transport_stream = _CCO_WORLD.transport_stream
        self._workspaces: dict[int, _TpStage1AgWorkspaceSet] = {}
        self._reported_active = False

    def padded_rows(self, local_rows: int, sizes: list[int] | None) -> int:
        max_rows = max(sizes) if sizes else local_rows
        # The arrival-ordered route layout uses 16-token Opus packs inside each
        # transport chunk. Keep one fixed physical stride per source and round
        # it to the common transport/sort alignment.
        alignment = self.config.chunks_per_source * 16
        return max(
            alignment,
            ((int(max_rows) + alignment - 1) // alignment) * alignment,
        )

    def supports(self, local_rows: int, sizes: list[int] | None = None) -> bool:
        logical_global_rows = sum(sizes) if sizes else local_rows * self.world
        return logical_global_rows >= self.config.min_global_tokens

    def _workspace(self, padded_rows: int) -> _TpStage1AgSlot:
        workspace_set = self._workspaces.get(padded_rows)
        if workspace_set is None:
            workspace_set = _TpStage1AgWorkspaceSet(self, padded_rows)
            self._workspaces[padded_rows] = workspace_set
        return workspace_set.acquire()

    @staticmethod
    def _metadata_transform(
        metadata,
        *,
        activation: ActivationType,
        hidden_pad: int,
        intermediate_pad: int,
    ):
        from aiter.fused_moe import _flydsl_stage1_wrapper

        if metadata.run_1stage or metadata.stage2 is None:
            raise ValueError("Stage1 AG requires a two-stage MoE configuration")
        return replace(
            metadata,
            stage1=functools.partial(
                _flydsl_stage1_wrapper,
                kernelName=TP_STAGE1_AG_KID,
                activation=activation,
                inter_dim_pad=intermediate_pad,
                model_dim_pad=hidden_pad,
            ),
            block_m=64,
            ksplit=1,
            run_1stage=False,
            has_bias=False,
            fuse_quant="fp8",
            prequant=True,
        )

    def apply(
        self,
        hidden_local: torch.Tensor,
        topk_weights_local: torch.Tensor,
        topk_ids_local: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        sizes: list[int] | None,
        activation: ActivationType,
        quant_type: QuantType,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        hidden_pad: int,
        intermediate_pad: int,
        swiglu_limit: float | None,
        gate_mode: str,
        beta: float | None = None,
        linear_beta: float | None = None,
    ) -> torch.Tensor:
        if quant_type != QuantType.per_1x32 or w1.dtype != dtypes.fp4x2:
            raise TypeError("Stage1 AG requires MXFP8 activation/MXFP4 weight")
        if gate_mode != "interleave":
            raise ValueError("Stage1 AG requires interleaved gate/up weights")
        if not self._reported_active:
            if self.rank == 0:
                logger.info(
                    "Using CCO SDMA AllGather + ready-aware Stage1 for DP MoE: "
                    "world=%d, H=%d, E=%d, "
                    "topk=%d, chunks/source=%d, KID=%s",
                    self.world,
                    self.config.hidden,
                    self.config.experts,
                    self.config.topk,
                    self.config.chunks_per_source,
                    TP_STAGE1_AG_KID,
                )
            self._reported_active = True
        padded_rows = self.padded_rows(hidden_local.shape[0], sizes)
        slot = self._workspace(padded_rows)
        global_weights, global_ids = slot.begin(
            hidden_local,
            topk_weights_local,
            topk_ids_local,
        )

        from aiter.fused_moe import _fused_moe_impl

        transform = functools.partial(
            self._metadata_transform,
            activation=activation,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
        )
        try:
            output = _fused_moe_impl(
                hidden_states=hidden_local,
                w1=w1,
                w2=w2,
                topk_weight=global_weights,
                topk_ids=global_ids,
                activation=activation,
                quant_type=quant_type,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                dtype=hidden_local.dtype,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                swiglu_limit=swiglu_limit,
                beta=beta,
                linear_beta=linear_beta,
                gate_mode=gate_mode,
                _q_dtype_a=dtypes.fp8,
                _metadata_transform=transform,
                _prepare_stage1=slot.prepare,
            )
        finally:
            slot.finish()
        return output

    def close(self) -> None:
        for workspace in self._workspaces.values():
            workspace.close()
        self._workspaces.clear()


_BACKENDS: dict[tuple[str, int, TpStage1AgConfig], TpStage1AgBackend] = {}


def get_tp_stage1_ag_backend(group, config: TpStage1AgConfig) -> TpStage1AgBackend:
    key = (group.unique_name, torch.cuda.current_device(), config)
    backend = _BACKENDS.get(key)
    if backend is None:
        backend = TpStage1AgBackend(group, config)
        _BACKENDS[key] = backend
    return backend


def close_tp_stage1_ag_backends() -> None:
    global _CCO_WORLD
    while _BACKENDS:
        _key, backend = _BACKENDS.popitem()
        try:
            backend.close()
        except Exception:
            logger.exception("failed to close TP Stage1 AG backend")
    if _CCO_WORLD is not None:
        try:
            _CCO_WORLD.comm.destroy()
        except Exception:
            logger.exception("failed to close Stage1 CCO communicator")
        _CCO_WORLD = None


atexit.register(close_tp_stage1_ag_backends)


__all__ = [
    "TP_STAGE1_AG_KID",
    "TpStage1AgBackend",
    "TpStage1AgConfig",
    "close_tp_stage1_ag_backends",
    "get_tp_stage1_ag_backend",
    "preinitialize_tp_stage1_cco_world",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused stage1 with low-ID dispatch producers and oversubscribed FP8xFP4 grouped-GEMM1 consumers."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_function import _jit_function_cache_key
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _preload_compiled, _run_compiled
from .dispatch import (
    DispatchSlot,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_payload,
)
from .gemm1 import _LdsF32View, build_fused_gemm1
from .gemm_util import _buffer_load, _buffer_store, _make_buffer, _make_buffer_from_addr
from .mega_moe_config import (
    FIXED_GRID_MULT_VALUES,
    INDEXED_PAYLOAD_MIN_MTPR,
    INDEXED_PAYLOAD_MIN_SBM,
    Stage1Config,
    fixed_stage1_epoch_slot,
)

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32


class _Stage1KernelSpec:
    __slots__ = ("block_x", "grid_x", "kernel", "waves_per_eu_hint")

    def __init__(self, kernel, grid_x, block_x, waves_per_eu_hint):
        self.kernel = kernel
        self.grid_x = int(grid_x)
        self.block_x = int(block_x)
        self.waves_per_eu_hint = int(waves_per_eu_hint)


def ceildiv(a, b):
    return (a + b - 1) // b


def _use_direct_fixed_slot(
    enabled, npes, experts_per_rank, max_tokens_per_rank, cap, tile_m
):
    if not enabled or tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    return npes == 8 and experts_per_rank == 48 and cap == required_cap


def _validate_dispatch_capacity(
    batch_size,
    npes,
    experts_per_rank,
    topk,
    tile_m,
    row_bytes,
    output_row_bytes,
    use_tile_resource,
):
    max_rows = npes * batch_size * topk + experts_per_rank * tile_m
    if not use_tile_resource and max_rows * row_bytes >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError(
            "MegaMoE v2 stage1 payload exceeds the 32-bit buffer-resource ABI"
        )
    if (
        not use_tile_resource
        and max_rows * output_row_bytes >= _BUFFER_OFFSET_ABI_BYTES
    ):
        raise ValueError(
            "MegaMoE v2 stage1 output exceeds the 32-bit buffer-resource ABI"
        )


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fixed_slot_dispatch: bool, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32,
    b_nt: int = -1,
    work_shards: int | None = None, payload_chunk_rows: int = 0, payload_tile_ready: bool = False,
    tile_state_stride: int = 0,
    fanout_masks: tuple[int, ...] = (),
    runtime_fanout: bool = False,
    debug_role_mode: int = 0,
    swiglu_limit: float = 0.0,
    _return_kernel_spec: bool = False,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage1 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    NUM_WAVES = int(num_waves)
    assert NUM_WAVES > 1, "planner needs one communication wave and at least one grouping wave"
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    dispatch_blocks = int(num_dispatch_cu)
    assert grid_mult in FIXED_GRID_MULT_VALUES, "grid_mult out of range"
    grid_epoch_slot = fixed_stage1_epoch_slot(grid_mult, dispatch_blocks, num_cu)
    payload_chunk_rows = int(payload_chunk_rows)
    tile_state_stride = int(tile_state_stride)
    assert 0 < dispatch_blocks < num_cu, "num_dispatch_cu must be in [1, num_cu)"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    if payload_chunk_rows:
        assert not fixed_slot_dispatch and payload_chunk_rows % sort_block_m == 0
    assert not payload_tile_ready or payload_chunk_rows > 0
    assert not payload_tile_ready or tile_state_stride > 0
    fanout_enabled = bool(fanout_masks) or runtime_fanout
    if fanout_enabled:
        if fixed_slot_dispatch or not payload_tile_ready:
            raise ValueError("fanout segments require compact tile-ready dispatch")
        if fanout_masks and len(fanout_masks) != fuse_npes:
            raise ValueError("fanout_masks must contain one mask per destination")
        valid_expert_mask = (1 << experts_per_rank) - 1
        for mask in fanout_masks:
            if mask & ~valid_expert_mask:
                raise ValueError("fanout mask references an out-of-range local expert")
            if mask and mask.bit_count() < 2:
                raise ValueError("fanout mask must contain at least two experts")
    preplanned_compact = not fixed_slot_dispatch
    ready_tile_queue = preplanned_compact
    # Assign roles by first-arrival ticket rather than block ID.  The launch
    # deliberately oversubscribes the device so producer CTAs can retire and
    # queued consumers can backfill them.  Hardware does not guarantee block
    # residency order, while tickets guarantee that every required producer
    # belongs to the first resident cohort.
    if preplanned_compact and (not payload_tile_ready or payload_chunk_rows <= 0):
        raise ValueError("compact dispatch requires preplanned tile-ready payloads")
    debug_role_mode = int(debug_role_mode)
    if debug_role_mode not in (0, 1, 2, 3, 4, 5, 6):
        raise ValueError("debug_role_mode must be in [0, 6]")
    planner_blocks = 0 if preplanned_compact else 1
    # Compact launches exactly one CTA per CU.  Low block IDs perform finite
    # payload work and then join the common GEMM work queue; the remaining
    # CTAs may wait for tile readiness without preventing any producer from
    # becoming resident.  Fixed-slot retains arrival tickets until its owner
    # epoch protocol is converted to the same bounded-grid scheme.
    consumer_ticket_base = 0 if preplanned_compact else dispatch_blocks + planner_blocks
    # Compact external-counting overwrites every source-owned histogram row
    # and publishes COUNT_DONE with the invocation generation.  That exchange
    # already provides the cross-rank release/acquire edge, so a separate
    # launch-ready round trip is redundant.  Fixed-slot and locally-counted
    # paths retain the entry handshake because they do not have that edge.
    cross_rank_entry_handshake = fixed_slot_dispatch
    # Compact uses a bounded all-resident cohort.  Fixed-slot still queues a
    # full consumer cohort behind its arrival-ticket owner/producers.
    grid_x = 1 if debug_role_mode == 5 else num_cu * grid_mult
    assert grid_x > 0, "consumer grid must remain positive"
    launch_grid_x = grid_x if preplanned_compact else consumer_ticket_base + grid_x
    assert launch_grid_x <= num_cu * 33 + 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2
    assert TILE_K_BYTES % 128 == 0
    A_K_STEP_BYTES = tile_k
    assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    WORK_SHARDS = 4 if work_shards is None and int(fuse_mtpr) >= 8192 else 8
    if work_shards is not None:
        WORK_SHARDS = int(work_shards)
    assert WORK_SHARDS in (1, 2, 4, 8)

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    if fz_npes * fz_mtpr > 1 << 24:
        raise ValueError("MegaMoE v2 source-token encoding exceeds 24 bits")
    if fz_k > 1 << 8:
        raise ValueError("MegaMoE v2 top-k slot encoding exceeds 8 bits")
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    direct_fixed_slot = _use_direct_fixed_slot(
        fixed_slot_dispatch, fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m
    )
    if fixed_slot_dispatch and not direct_fixed_slot:
        raise ValueError("fixed-slot dispatch requires the direct fixed-slot layout")
    fz_total_experts = fz_npes * fz_epr
    indexed_payload = (
        not fixed_slot_dispatch
        and fuse_mtpr >= INDEXED_PAYLOAD_MIN_MTPR
        and sort_block_m >= INDEXED_PAYLOAD_MIN_SBM
    )
    source_rows = fz_npes * fz_mtpr + 1 if indexed_payload else 0
    if indexed_payload and source_rows * model_dim >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError(
            "MegaMoE v2 indexed payload exceeds the 32-bit buffer-resource ABI"
        )
    if indexed_payload and fuse_scale_dim % 16:
        raise ValueError(
            "MegaMoE v2 indexed E8M0 rows must be a multiple of 16 bytes"
        )
    # Small batches stream B; large batches cache it across M tiles.
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fz_mtpr <= 512 else 0)
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if direct_fixed_slot and fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512
    _validate_dispatch_capacity(
        fz_mtpr, fz_npes, fz_epr, fz_k, fz_tile_m, fz_nbytes, inter_dim, use_tile_resource
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    fanout_suffix = (
        "_fov6_" + "x".join(f"{mask:x}" for mask in fanout_masks)
        if fanout_masks
        else ("_fov_runtime" if runtime_fanout else "")
    )
    WORK_BATCH = 1
    kernel_name = (
        f"megamoe_stage1_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"_cgc{grid_x}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}_ws{WORK_SHARDS}"
        f"_pc{payload_chunk_rows}"
        f"_ptr{int(payload_tile_ready)}"
        f"_tss{tile_state_stride}"
        f"_rc31_wb{WORK_BATCH}_adaptive"
        f"_erh{int(cross_rank_entry_handshake)}"
        f"_prep{int(preplanned_compact)}"
        f"_ix{int(indexed_payload)}"
        f"_rtq{int(ready_tile_queue)}"
        f"_drm{debug_role_mode}"
        f"{fanout_suffix}"
        f"{swiglu_suffix}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_pair_order_ready = _disp_ptr(DispatchSlot.PAIR_ORDER_READY)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_work_tail = _disp_ptr(DispatchSlot.WORK_TAIL)
        a_ready_tile_queue = fx.Int64(0)
        a_ready_tile_epoch = fx.Int64(0)
        a_ready_tile_tail = fx.Int64(0)
        if const_expr(ready_tile_queue):
            a_ready_tile_queue = _disp_ptr(DispatchSlot.READY_TILE_QUEUE)
            a_ready_tile_epoch = _disp_ptr(DispatchSlot.READY_TILE_EPOCH)
            a_ready_tile_tail = _disp_ptr(DispatchSlot.READY_TILE_TAIL)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
        a_payload_chunks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)
        a_launch_ready = fx.Int64(0)
        p_launch_ready = fx.Int64(0)
        if const_expr(cross_rank_entry_handshake):
            a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
            p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_payload_ready_rows = _disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)
        a_max_expert_tiles = fx.Int64(0)
        if const_expr(ready_tile_queue):
            a_max_expert_tiles = _disp_ptr(DispatchSlot.MAX_EXPERT_TILES)

        if const_expr(preplanned_compact):
            ticket = fx.block_idx.x
            generation = fx.Int64(0)
        else:
            ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
            ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
            if tid == fx.Int32(0):
                ticket64 = fx.Int64(
                    comm_ops.atomic_add_agent(
                        a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1)
                    )
                )
                fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
            fx.barrier()
            ticket64 = Vec(ticket_view.load())[0]
            generation = ticket64 // fx.Int64(launch_grid_x)
            ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        if const_expr(preplanned_compact):
            compact_owner = fx.Int32(0) == fx.Int32(1)
            compact_producer = ticket < fx.Int32(dispatch_blocks)
            producer_slot = ticket
        else:
            compact_owner = ticket == fx.Int32(0)
            compact_producer = (ticket > fx.Int32(0)) & (
                ticket <= fx.Int32(dispatch_blocks)
            )
            producer_slot = ticket - fx.Int32(1)

        if const_expr(not preplanned_compact):
            if compact_owner:
                next_parity_lane = fx.Int32(0)
                launch_epoch_lane = fx.Int32(0)
                if tid == fx.Int32(0):
                    old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                    next_parity_lane = old_parity ^ fx.Int32(1)
                    previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
                    next_expected = previous_expected + fx.Int32(fz_npes)
                    _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
                    if const_expr(cross_rank_entry_handshake):
                        launch_epoch_lane = (
                            (next_expected // fx.Int32(fz_npes)) * fx.Int32(2)
                            - next_parity_lane
                        )
                next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
                launch_epoch = fx.Int32(0)
                if const_expr(cross_rank_entry_handshake):
                    launch_epoch = fx.Int32(
                        fx.rocdl.readfirstlane(T.i32, launch_epoch_lane)
                    )
                if const_expr(payload_tile_ready):
                    if tid == fx.Int32(0):
                        comm_ops.store_i32_system(a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m))
                        comm_ops.fence_system_release()
                    fx.barrier()
                if const_expr(cross_rank_entry_handshake):  # noqa: SIM102 - preserve DSL staging
                    if tid < fx.Int32(fz_npes):
                        peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                        comm_ops.fence_system_release()
                        launch_ready_table = _make_buffer_from_addr(
                            p_launch_ready, fx.Int64
                        )
                        remote_launch_ready = _buffer_load(
                            launch_ready_table, peer, fx.Int64
                        )
                        comm_ops.store_i32_system(
                            remote_launch_ready, fx.Int32(fz_rank), launch_epoch
                        )
                        comm_ops.wait_i32_until_greater_than(
                            a_launch_ready + fx.Int64(peer) * fx.Int64(4),
                            launch_epoch - fx.Int32(1),
                        )
                        comm_ops.fence_system_acquire()
                if tid == fx.Int32(0):
                    work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                    for shard in range_constexpr(WORK_SHARDS):
                        _buffer_store(
                            work_head_rsrc,
                            fx.Int32(shard * 16),
                            fx.Int32(0),
                            fx.Int32,
                        )
                    comm_ops.store_i32_system(
                        a_work_tail, fx.Int32(0), fx.Int32(0)
                    )
                    if const_expr(ready_tile_queue and debug_role_mode != 6):
                        _buffer_store(
                            _make_buffer_from_addr(a_ready_tile_tail, fx.Int32),
                            next_parity,
                            fx.Int32(0),
                            fx.Int32,
                        )
                    group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                    for destination in range_constexpr(fz_npes):
                        _buffer_store(
                            group_done_rsrc,
                            fx.Int32(destination),
                            fx.Int32(0),
                            fx.Int32,
                        )
                if tid == fx.Int32(0):
                    fx.rocdl.s_waitcnt(0)
                    comm_ops.fence_agent_release()
                    _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                    fx.rocdl.s_waitcnt(0)
                    comm_ops.fence_agent_release()
                    comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
                fx.rocdl.s_waitcnt(0)
                fx.barrier()
            else:
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(gate_addr, gate_epoch)
                    comm_ops.fence_agent_acquire()
                fx.barrier()

        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)
        payload_epoch = (
            payload_expected // fx.Int32(fz_npes)
        ) * fx.Int32(2) - payload_parity
        tile_state_byte_offset = fx.Int64(0)
        if const_expr(payload_tile_ready):
            tile_state_byte_offset = (
                fx.Int64(payload_parity)
                * fx.Int64(tile_state_stride)
                * fx.Int64(4)
            )
            a_ready_tile_queue = a_ready_tile_queue + tile_state_byte_offset
            a_ready_tile_epoch = a_ready_tile_epoch + tile_state_byte_offset

        if compact_producer:
            if const_expr(direct_fixed_slot):
                emit_direct_fixed_slot_payload(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                    parity=payload_parity, expected=payload_expected,
                )
            else:
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(
                        a_pair_order_ready
                        + fx.Int64(payload_parity) * fx.Int64(4),
                        payload_expected,
                    )
                    comm_ops.fence_system_acquire()
                fx.barrier()
                producer_destination = producer_slot % fx.Int32(fz_npes)
                producers_per_destination = _buffer_load(
                    _make_buffer_from_addr(
                        a_payload_blocks_per_destination, fx.Int32
                    ),
                    producer_destination,
                    fx.Int32,
                )
                chunks_per_destination = _buffer_load(
                    _make_buffer_from_addr(
                        a_payload_chunks_per_destination, fx.Int32
                    ),
                    producer_destination,
                    fx.Int32,
                )
                if const_expr(debug_role_mode != 1):
                    emit_dispatch_payload(
                        num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                        fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                        fz_safe_end_i32=fz_safe_end_i32, fz_scale_n_i32=fz_scale_n_i32,
                        fz_enable_scales=fz_enable_scales, addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                        addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                        dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        producers_per_destination=producers_per_destination, payload_chunk_rows=payload_chunk_rows,
                        chunks_per_destination=chunks_per_destination, payload_tile_ready=payload_tile_ready,
                        ready_tile_queue=ready_tile_queue,
                        tile_state_stride=tile_state_stride,
                        indexed_payload=indexed_payload,
                        fanout_masks=fanout_masks,
                        runtime_fanout=runtime_fanout,
                    )
        if const_expr(direct_fixed_slot):
            if compact_owner:
                emit_direct_fixed_slot_finalize(
                    fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_tile_m=fz_tile_m, n_tiles=N_TILES, addr_disp=addr_disp, parity=payload_parity,
                    expected=payload_expected,
                )
        else:
            payload_table = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.P2P_PAYLOAD_READY)), fx.Int64)
            addr_payload_ready = _buffer_load(
                _make_buffer_from_addr(payload_table, fx.Int64), fx.Int32(fz_rank), fx.Int64
            )
            addr_tile_ready = _disp_ptr(DispatchSlot.TILE_READY)
            addr_tile_expected = _disp_ptr(DispatchSlot.TILE_EXPECTED)
            if const_expr(payload_tile_ready):
                addr_tile_ready = addr_tile_ready + tile_state_byte_offset
                addr_tile_expected = addr_tile_expected + tile_state_byte_offset
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        tib_rsrc = _make_buffer_from_addr(
            _disp_ptr(DispatchSlot.TILE_INPUT_BASE), fx.Int32
        )
        srcmap_rsrc = _make_buffer_from_addr(
            _disp_ptr(DispatchSlot.SRCMAP), fx.Int32
        )
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_nbytes = tokens * fx.Int32(inter_dim)
            out_rsrc = _make_buffer(out, fx.Int16, max_size=False, num_records_bytes=out_nbytes)
        os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

        expert_of_flat, _do_scheduled_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc,
            sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
            tib_rsrc=tib_rsrc,
            a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
            model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
            tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
            m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
            n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
            indirect_input=fanout_enabled and not indexed_payload,
            indexed_input=indexed_payload,
            row_map_rsrc=srcmap_rsrc,
            source_rows=source_rows,
            swiglu_limit=swiglu_limit,
        )

        if tid == fx.Int32(0):
            local_plan_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.PLAN_READY)), fx.Int64)
            ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.wait_i32_until_equals(
                local_plan_ready + fx.Int64(ready_index) * fx.Int64(4), payload_expected)
            # PLAN_READY is published with a system-scope release after the
            # planner rewrites TILE_EXPECTED.  Match that scope so a runtime
            # layout transition cannot reuse the previous invocation's
            # expected count and wait forever on the wrong value.
            if const_expr(preplanned_compact):
                comm_ops.fence_system_acquire()
            else:
                comm_ops.fence_agent_acquire()
        fx.barrier()

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)
        use_ready_order = fx.Int32(0) == fx.Int32(1)
        if const_expr(ready_tile_queue):
            max_expert_tiles = _buffer_load(
                _make_buffer_from_addr(a_max_expert_tiles, fx.Int32),
                fx.Int32(0),
                fx.Int32,
            )
            use_ready_order = max_expert_tiles * fx.Int32(4) >= num_m_tiles

        def _wait_tile_payload(flat):
            if const_expr(payload_tile_ready):
                tile_index = flat // fx.Int32(N_TILES)
                expected_tiles = _buffer_load(
                    _make_buffer_from_addr(addr_tile_expected, fx.Int32), tile_index, fx.Int32
                )
                if const_expr(debug_role_mode == 5):
                    tile_ready_rsrc = _make_buffer_from_addr(
                        addr_tile_ready, fx.Int32
                    )
                    observed_tiles = _buffer_load(
                        tile_ready_rsrc,
                        tile_index,
                        fx.Int32,
                        cache_modifier=_SC0_CACHE,
                    )
                    if observed_tiles != expected_tiles:
                        previous_missing = fx.Int32(
                            comm_ops.atomic_add_agent(
                                a_work_head + fx.Int64(4), fx.Int32(1)
                            )
                        )
                        if previous_missing == fx.Int32(0):
                            debug_wait_rsrc = _make_buffer_from_addr(
                                a_work_head, fx.Int32
                            )
                            _buffer_store(
                                debug_wait_rsrc,
                                fx.Int32(2),
                                tile_index,
                                fx.Int32,
                            )
                            _buffer_store(
                                debug_wait_rsrc,
                                fx.Int32(3),
                                observed_tiles,
                                fx.Int32,
                            )
                            _buffer_store(
                                debug_wait_rsrc,
                                fx.Int32(4),
                                expected_tiles,
                                fx.Int32,
                            )
                            _buffer_store(
                                debug_wait_rsrc,
                                fx.Int32(5),
                                payload_epoch,
                                fx.Int32,
                            )
                    # Debug role 5 records a missing publication and returns;
                    # it deliberately does not wait or execute GEMM.
                else:
                    comm_ops.wait_i32_until_equals(
                        addr_tile_ready + fx.Int64(tile_index) * fx.Int64(4),
                        expected_tiles,
                    )
            else:
                pe = expert_of_flat(flat)
                pe_index = payload_parity * fx.Int32(fz_epr) + pe
                comm_ops.wait_i32_until_equals(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), payload_expected
                )

        # Compact producers join the work queue after their finite payload
        # copy.  Its grid is capped at one CTA per CU, so waiting consumers
        # cannot starve an unscheduled producer.  Fixed-slot producers retain
        # the queued-consumer behavior until its owner epoch is converted.
        consumer_id = ticket - fx.Int32(consumer_ticket_base)
        consumer_base = fx.Int32(consumer_ticket_base)
        consumer_active = (ticket >= consumer_base) & (
            consumer_id < total_work
        )
        if const_expr(debug_role_mode in (1, 2)):
            consumer_active = fx.Int32(0) == fx.Int32(1)
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        work_shard = consumer_id & fx.Int32(WORK_SHARDS - 1)
        work_batch = WORK_BATCH
        assert N_TILES % work_batch == 0

        def _run_work_batch(first_work, scheduled_first):
            for batch_offset in range_constexpr(work_batch):
                if const_expr(ready_tile_queue):
                    work = use_ready_order.select(
                        scheduled_first, first_work
                    ) + fx.Int32(batch_offset)
                else:
                    work = first_work + fx.Int32(batch_offset * WORK_SHARDS)
                if work < total_work:
                    if const_expr(not ready_tile_queue):
                        if tid == fx.Int32(0):  # noqa: SIM102 - preserve DSL staging
                            if const_expr(
                                not direct_fixed_slot
                                and debug_role_mode not in (3, 4)
                            ):
                                _wait_tile_payload(work)
                        fx.barrier()
                        if const_expr(not direct_fixed_slot):
                            comm_ops.fence_system_acquire()
                    elif not use_ready_order:
                        if tid == fx.Int32(0):
                            _wait_tile_payload(work)
                        fx.barrier()
                        comm_ops.fence_system_acquire()
                    if const_expr(debug_role_mode not in (4, 5)):
                        _do_scheduled_tile(work)

        while consumer_active:
            if tid == fx.Int32(0):
                first_work = fx.Int32(0)
                if const_expr(ready_tile_queue):
                    if use_ready_order:
                        first_work = fx.Int32(
                            comm_ops.atomic_add_agent(
                                a_work_head,
                                fx.Int32(work_batch),
                            )
                        )
                    else:
                        local_work = fx.Int32(
                            comm_ops.atomic_add_agent(
                                a_work_head
                                + fx.Int64(work_shard) * fx.Int64(64),
                                fx.Int32(work_batch),
                            )
                        )
                        first_work = (
                            work_shard
                            + local_work * fx.Int32(WORK_SHARDS)
                        )
                else:
                    local_work = fx.Int32(
                        comm_ops.atomic_add_agent(
                            a_work_head
                            + fx.Int64(work_shard) * fx.Int64(64),
                            fx.Int32(work_batch),
                        )
                    )
                    first_work = (
                        work_shard + local_work * fx.Int32(WORK_SHARDS)
                    )
                fx.ptr_store(
                    Vec.from_elements([first_work], fx.Int32), work_scratch
                )
            fx.barrier()
            first_work = Vec(work_scratch_view.load())[0]
            if (  # noqa: SIM102 - preserve DSL staging
                const_expr(ready_tile_queue and debug_role_mode != 4)
                and first_work < total_work
            ):
                if use_ready_order:
                    ready_slot = first_work // fx.Int32(N_TILES)
                    first_n_tile = first_work - ready_slot * fx.Int32(N_TILES)
                    if tid == fx.Int32(0):
                        comm_ops.wait_i32_until_equals(
                            a_ready_tile_epoch
                            + fx.Int64(ready_slot) * fx.Int64(4),
                            payload_epoch,
                        )
                        comm_ops.fence_system_acquire()
                        ready_m_tile = _buffer_load(
                            _make_buffer_from_addr(a_ready_tile_queue, fx.Int32),
                            ready_slot,
                            fx.Int32,
                        )
                        scheduled_first = (
                            ready_m_tile * fx.Int32(N_TILES) + first_n_tile
                        )
                        fx.ptr_store(
                            Vec.from_elements([scheduled_first], fx.Int32),
                            work_scratch,
                        )
                    fx.barrier()
            scheduled_first = Vec(work_scratch_view.load())[0]
            if const_expr(ready_tile_queue):  # noqa: SIM102 - preserve DSL staging
                if use_ready_order:
                    comm_ops.fence_system_acquire()
            _run_work_batch(first_work, scheduled_first)
            if const_expr(debug_role_mode == 5):
                consumer_active = fx.Int32(0) == fx.Int32(1)
            else:
                consumer_active = first_work < total_work

    spec = _Stage1KernelSpec(
        kernel, launch_grid_x, TOTAL_THREADS, waves_per_eu_hint
    )
    if _return_kernel_spec:
        return spec

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, stream: fx.Stream,
    ):
        spec.kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            value_attrs={
                "rocdl.waves_per_eu": spec.waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{spec.block_x},{spec.block_x}",
            },
        ).launch(
            grid=(spec.grid_x, 1, 1),
            block=(spec.block_x, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_mega_moe_stage1_bundle(
    *,
    model_dim: int,
    inter_dim: int,
    rank: int,
    experts_per_rank: int,
    fuse_npes: int,
    fuse_topk: int,
    fuse_cap: int,
    fuse_mtpr: int,
    fuse_scale_dim: int,
    fixed_slot_dispatch: bool,
    num_cu: int,
    tile_state_stride: int,
    variants: tuple[Stage1Config, ...],
    swiglu_limit: float = 0.0,
):
    """Compile every production Stage1 variant into one profile module."""
    if not variants:
        raise ValueError("MegaMoE Stage1 bundle requires at least one variant")
    specs = tuple(
        compile_mega_moe_stage1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            rank=rank,
            experts_per_rank=experts_per_rank,
            fuse_npes=fuse_npes,
            fuse_topk=fuse_topk,
            fuse_cap=fuse_cap,
            fuse_mtpr=fuse_mtpr,
            fuse_scale_dim=fuse_scale_dim,
            fixed_slot_dispatch=fixed_slot_dispatch,
            sort_block_m=config.sort_block_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            num_waves=config.num_waves,
            grid_mult=config.grid_mult,
            pipe_weights=config.pipe_weights,
            mfma_amajor=config.mfma_amajor,
            swizzle_a=config.swizzle_a,
            async_a_copy=config.async_a_copy,
            use_tile_resource=config.use_tile_resource,
            waves_per_eu_hint=config.waves_per_eu_hint,
            num_cu=num_cu,
            num_dispatch_cu=config.num_dispatch_cu,
            b_nt=config.b_nt,
            work_shards=config.work_shards,
            payload_chunk_rows=config.payload_chunk_rows,
            payload_tile_ready=config.payload_tile_ready,
            tile_state_stride=tile_state_stride,
            runtime_fanout=not fixed_slot_dispatch,
            swiglu_limit=swiglu_limit,
            _return_kernel_spec=True,
        )
        for config in variants
    )
    # FlyDSL treats tuples as scalar closure values when it builds the JIT
    # manager key.  A tuple of KernelFunction objects therefore hashes their
    # process-local repr (including object addresses), which makes an AOT
    # artifact impossible to reuse in a service process.  Keep the executable
    # objects in a list and capture stable dependency fingerprints separately.
    kernels = [spec.kernel for spec in specs]
    kernel_cache_keys = tuple(
        _jit_function_cache_key(spec.kernel._original_func) for spec in specs
    )
    grid_xs = tuple(spec.grid_x for spec in specs)
    block_xs = tuple(spec.block_x for spec in specs)
    waves_per_eu = tuple(spec.waves_per_eu_hint for spec in specs)

    @flyc.jit
    def launch(
        out: fx.Tensor,
        x: fx.Tensor,
        w: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor,
        tokens: fx.Int32,
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64,
        addr_in_wts: fx.Int64,
        addr_in_sc: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        variant_id: fx.Int32,
        stream: fx.Stream,
    ):
        for index in range_constexpr(len(kernel_cache_keys)):
            if variant_id == fx.Int32(index):
                kernels[index](
                    out,
                    x,
                    w,
                    scale_x,
                    scale_w,
                    sorted_token_ids,
                    expert_ids,
                    num_valid_ids,
                    out_scale,
                    tokens,
                    addr_disp,
                    i32_cur_tok,
                    addr_in_tok,
                    addr_in_idx,
                    addr_in_wts,
                    addr_in_sc,
                    addr_parity,
                    addr_expected,
                    value_attrs={
                        "rocdl.waves_per_eu": waves_per_eu[index],
                        "rocdl.flat_work_group_size": (
                            f"{block_xs[index]},{block_xs[index]}"
                        ),
                    },
                ).launch(
                    grid=(grid_xs[index], 1, 1),
                    block=(block_xs[index], 1, 1),
                    stream=stream,
                )

    return launch


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, work_shards=None,
    payload_chunk_rows=0, payload_tile_ready=False, tile_state_stride=0,
    fanout_masks=(),
    runtime_fanout=False,
    debug_role_mode=0,
    swiglu_limit=0.0):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, work_shards=work_shards, payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
        tile_state_stride=tile_state_stride,
        fanout_masks=tuple(fanout_masks),
        runtime_fanout=runtime_fanout,
        debug_role_mode=debug_role_mode,
        swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, stream,
    )


def run_mega_moe_stage1_bundle(
    out,
    x,
    w,
    scale_x,
    scale_w,
    sorted_token_ids,
    expert_ids,
    num_valid_ids,
    out_scale,
    tokens,
    addr_disp,
    i32_cur_tok,
    addr_in_tok,
    addr_in_idx,
    addr_in_wts,
    addr_in_sc,
    addr_parity,
    addr_expected,
    variant_id,
    stream,
    **compile_kw,
):
    variants = tuple(compile_kw["variants"])
    if not 0 <= int(variant_id) < len(variants):
        raise ValueError(f"invalid Stage1 bundle variant_id={variant_id}")
    launch = compile_mega_moe_stage1_bundle(**compile_kw)
    _run_compiled(
        launch,
        out,
        x,
        w,
        scale_x,
        scale_w,
        sorted_token_ids,
        expert_ids,
        num_valid_ids,
        out_scale,
        tokens,
        addr_disp,
        i32_cur_tok,
        addr_in_tok,
        addr_in_idx,
        addr_in_wts,
        addr_in_sc,
        addr_parity,
        addr_expected,
        fx.Int32(variant_id),
        stream,
    )


def preload_mega_moe_stage1_bundle(
    out,
    x,
    w,
    scale_x,
    scale_w,
    sorted_token_ids,
    expert_ids,
    num_valid_ids,
    out_scale,
    tokens,
    addr_disp,
    i32_cur_tok,
    addr_in_tok,
    addr_in_idx,
    addr_in_wts,
    addr_in_sc,
    addr_parity,
    addr_expected,
    variant_id,
    stream,
    **compile_kw,
):
    """Compile and load the complete Stage1 bundle without dispatching it."""
    launch = compile_mega_moe_stage1_bundle(**compile_kw)
    return _preload_compiled(
        launch,
        out,
        x,
        w,
        scale_x,
        scale_w,
        sorted_token_ids,
        expert_ids,
        num_valid_ids,
        out_scale,
        tokens,
        addr_disp,
        i32_cur_tok,
        addr_in_tok,
        addr_in_idx,
        addr_in_wts,
        addr_in_sc,
        addr_parity,
        addr_expected,
        fx.Int32(variant_id),
        stream,
    )
# fmt: on

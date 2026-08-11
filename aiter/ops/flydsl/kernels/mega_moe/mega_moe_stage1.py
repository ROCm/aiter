# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Fused stage1 with low-ID dispatch producers and oversubscribed FP8xFP4 grouped-GEMM1 consumers."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .dispatch import (
    DispatchSlot,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_group,
    emit_dispatch_payload,
    emit_dispatch_plan,
)
from .gemm1 import _LdsF32View, build_fused_gemm1
from .gemm_util import _buffer_load, _buffer_store, _make_buffer, _make_buffer_from_addr

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32


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
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    work_shards: int | None = None, external_grouping: bool | None = None,
    external_counting: bool | None = None, payload_chunk_rows: int = 0, payload_tile_ready: bool = False,
    swiglu_limit: float = 0.0,
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
    GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    assert grid_mult in GRID_MULT_VALUES, "grid_mult out of range"
    grid_epoch_slot = GRID_MULT_VALUES.index(grid_mult)
    dispatch_blocks = int(num_dispatch_cu)
    payload_chunk_rows = int(payload_chunk_rows)
    assert 0 < dispatch_blocks < num_cu, "num_dispatch_cu must be in [1, num_cu)"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    if payload_chunk_rows:
        assert not fixed_slot_dispatch and payload_chunk_rows % sort_block_m == 0
    assert not payload_tile_ready or payload_chunk_rows > 0
    planner_blocks = 1
    # Keep the fused grid on an exact CU multiple instead of appending control/producer CTAs as a tail.
    grid_x = num_cu * grid_mult - planner_blocks - dispatch_blocks
    assert grid_x > 0, "consumer grid must remain positive"
    launch_grid_x = planner_blocks + dispatch_blocks + grid_x
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
    if external_grouping is None:
        external_grouping = fz_mtpr >= 2048 and fz_npes == 8 and fz_epr == 48
    if external_counting is None:
        external_counting = external_grouping and fz_mtpr >= 8192
    assert not external_counting or external_grouping
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    direct_fixed_slot = _use_direct_fixed_slot(
        fixed_slot_dispatch, fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m
    )
    fz_total_experts = fz_npes * fz_epr
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
    kernel_name = (
        f"megamoe_stage1_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}_ws{WORK_SHARDS}"
        f"_pc{payload_chunk_rows}"
        f"_ptr{int(payload_tile_ready)}"
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
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
        a_payload_chunks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)
        a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
        p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_payload_ready_rows = _disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1))
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid_x)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        compact_owner = ticket == fx.Int32(0)
        compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)

        if compact_owner:
            next_parity_lane = fx.Int32(0)
            launch_epoch_lane = fx.Int32(0)
            if tid == fx.Int32(0):
                old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                next_parity_lane = old_parity ^ fx.Int32(1)
                previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
                next_expected = previous_expected + fx.Int32(fz_npes)
                _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
                launch_epoch_lane = (
                    (next_expected // fx.Int32(fz_npes)) * fx.Int32(2) - next_parity_lane
                )
            next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
            launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
            if const_expr(payload_tile_ready):
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m))
                    comm_ops.fence_system_release()
                fx.barrier()
            if tid < fx.Int32(fz_npes):
                peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                comm_ops.fence_system_release()
                launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
                remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
                comm_ops.store_i32_system(remote_launch_ready, fx.Int32(fz_rank), launch_epoch)
                mori_shmem.int32_wait_until_greater_than(
                    a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
                )
                comm_ops.fence_system_acquire()
            if tid == fx.Int32(0):
                work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                for shard in range_constexpr(8):
                    _buffer_store(work_head_rsrc, fx.Int32(shard * 16), fx.Int32(0), fx.Int32)
                _buffer_store(_make_buffer_from_addr(a_work_tail, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
                if const_expr(external_grouping or direct_fixed_slot):
                    group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                    for destination in range_constexpr(fz_npes if direct_fixed_slot else 1):
                        _buffer_store(group_done_rsrc, fx.Int32(destination), fx.Int32(0), fx.Int32)
            fx.barrier()
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
                mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
                comm_ops.fence_agent_acquire()
            fx.barrier()

        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)

        if compact_owner:  # noqa: SIM102 - keep the device and compile-time branches separate.
            if const_expr(not direct_fixed_slot):
                emit_dispatch_plan(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr,
                    fz_rank=fz_rank, fz_tile_m=fz_tile_m, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, parity=payload_parity,
                    expected=payload_expected, external_grouping=external_grouping,
                    external_counting=external_counting,
                    dispatch_blocks=dispatch_blocks, payload_chunk_rows=payload_chunk_rows,
                    payload_tile_ready=payload_tile_ready,
                )

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
                if const_expr(external_grouping):
                    emit_dispatch_group(
                        num_waves=NUM_WAVES, fz_k=fz_k, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        external_counting=external_counting, adaptive_grouping=payload_tile_ready,
                    )
                else:
                    if tid == fx.Int32(0):
                        mori_shmem.int32_wait_until_equals(
                            a_pair_order_ready + fx.Int64(payload_parity) * fx.Int64(4), payload_expected)
                        comm_ops.fence_agent_acquire()
                    fx.barrier()
                producers_per_destination = fx.Int32(dispatch_blocks // fz_npes)
                chunks_per_destination = fx.Int32(1)
                if const_expr(payload_chunk_rows > 0):
                    chunks_per_destination = fx.Int32(
                        (fz_mtpr + payload_chunk_rows - 1) // payload_chunk_rows
                    )
                payload_active = fx.Int32(0) == fx.Int32(0)
                if const_expr(payload_tile_ready and dispatch_blocks > 32):
                    producer_destination = producer_slot % fx.Int32(fz_npes)
                    producer_round = producer_slot // fx.Int32(fz_npes)
                    producers_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_blocks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    chunks_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_chunks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    payload_active = producer_round < producers_per_destination
                if payload_active:
                    emit_dispatch_payload(
                        num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                        fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                        fz_safe_end_i32=fz_safe_end_i32, fz_scale_n_i32=fz_scale_n_i32,
                        fz_enable_scales=fz_enable_scales, addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                        addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        producers_per_destination=producers_per_destination, payload_chunk_rows=payload_chunk_rows,
                        chunks_per_destination=chunks_per_destination, payload_tile_ready=payload_tile_ready,
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
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
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
            a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
            model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
            tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
            m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
            n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
            swiglu_limit=swiglu_limit,
        )

        if tid == fx.Int32(0):
            local_plan_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.PLAN_READY)), fx.Int64)
            ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            mori_shmem.int32_wait_until_equals(
                local_plan_ready + fx.Int64(ready_index) * fx.Int64(4), payload_expected)
            comm_ops.fence_agent_acquire()
        fx.barrier()

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)

        def _wait_tile_payload(flat):
            if const_expr(payload_tile_ready):
                tile_index = flat // fx.Int32(N_TILES)
                expected_tiles = _buffer_load(
                    _make_buffer_from_addr(addr_tile_expected, fx.Int32), tile_index, fx.Int32
                )
                mori_shmem.int32_wait_until_equals(
                    addr_tile_ready + fx.Int64(tile_index) * fx.Int64(4), expected_tiles
                )
            else:
                pe = expert_of_flat(flat)
                pe_index = payload_parity * fx.Int32(fz_epr) + pe
                mori_shmem.int32_wait_until_equals(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), payload_expected
                )

        # Control CTAs join the work pool after dispatch.
        consumer_active = fx.Int32(1) == fx.Int32(1)
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        work_shard = ticket & fx.Int32(WORK_SHARDS - 1)
        while consumer_active:
            if tid == fx.Int32(0):
                local_work = fx.Int32(
                    comm_ops.atomic_add_agent(
                        a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)
                    )
                )
                work = work_shard + local_work * fx.Int32(WORK_SHARDS)
                fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            fx.barrier()
            work = Vec(work_scratch_view.load())[0]
            if tid == fx.Int32(0):
                has_work = (work < total_work).select(fx.Int32(1), fx.Int32(0))
                if has_work != fx.Int32(0):  # noqa: SIM102 - keep the device and compile-time branches separate.
                    if const_expr(not direct_fixed_slot):
                        _wait_tile_payload(work)
                fx.ptr_store(Vec.from_elements([has_work], fx.Int32), work_scratch)
            fx.barrier()
            has_work = Vec(work_scratch_view.load())[0]
            if has_work != fx.Int32(0):
                if const_expr(not direct_fixed_slot):
                    comm_ops.fence_system_acquire()
                _do_scheduled_tile(work)
            consumer_active = has_work != fx.Int32(0)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, work_shards=None, external_grouping=None, external_counting=None,
    payload_chunk_rows=0, payload_tile_ready=False, swiglu_limit=0.0):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, work_shards=work_shards, external_grouping=external_grouping,
        external_counting=external_counting, payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
        swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, stream,
    )
# fmt: on

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Fused Stage1 dispatch with FP8/FP4 activation and FP4-weight GEMM1."""

import dataclasses
import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.aot_launch import aot_factory

from .. import communication_ops_utils as comm_ops
from ..kernels_common import ceildiv
from ..tensor_shim import (
    _preload_compiled,
    _run_compiled,
    buf_copy_load,
    ptr_buf_tensor,
)
from .dispatch import (
    FUSED_STAGE2_CLAIM,
    FUSED_STAGE2_DONE,
    FUSED_STAGE2_HEAD_INTS,
    DispatchSlot,
    emit_direct_fixed_slot_counts,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_payload,
    fused_stage2_state_ints,
)
from .gemm1 import _LdsF32View, build_fused_gemm1
from .mega_moe_config import (
    FIXED_GRID_MULT_VALUES,
    FIXED_SLOT_EXPERTS_PER_RANK,
    INDEXED_PAYLOAD_MIN_MTPR,
    REFERENCE_EXPERTS_PER_RANK,
    Stage1Config,
    fixed_stage1_epoch_slot,
)
from .mega_moe_stage2 import Stage2Unit

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32
# MoRI's symmetric heap type, fixed for the process once MoRI initializes.
_UNCACHED_HEAP = (
    os.environ.get("MORI_SHMEM_HEAP_TYPE", "uncached").lower() == "uncached"
)


class _Stage1KernelSpec:
    __slots__ = ("block_x", "grid_x", "kernel", "waves_per_eu_hint")

    def __init__(self, kernel, grid_x, block_x, waves_per_eu_hint):
        self.kernel = kernel
        self.grid_x = int(grid_x)
        self.block_x = int(block_x)
        self.waves_per_eu_hint = int(waves_per_eu_hint)


def _validate_fixed_slot_geometry(
    npes, experts_per_rank, max_tokens_per_rank, cap, tile_m
):
    if tile_m <= 0 or max_tokens_per_rank <= 0:
        raise ValueError("fixed-slot requires positive tile and token capacities")
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    if (
        npes != 8
        or experts_per_rank not in FIXED_SLOT_EXPERTS_PER_RANK
        or cap != required_cap
    ):
        raise ValueError(
            "fixed-slot dispatch requires the direct EP8 layout with "
            f"{FIXED_SLOT_EXPERTS_PER_RANK} experts per rank"
        )


def stage1_output_nbytes(rows, inter_dim, a_dtype="fp8"):
    """Bytes of the GEMM1 output for ``rows`` routed rows and of its scale buffer."""
    out = rows * (inter_dim // 2 if a_dtype == "fp4" else inter_dim)
    prows = ceildiv(rows, 256) * 256
    pcols = ceildiv(inter_dim // 32, 8) * 8
    return out, prows * pcols + inter_dim


def stage1_output_buffers(rows, inter_dim, device, a_dtype="fp8", arena=None):
    """GEMM1 output ``[rows, inter_dim]`` (FP8, or packed FP4) and its scale buffer.

    ``arena``: (output bytes, scale bytes) uint8 tensors to view instead of
    allocating, shared by routes whose calls never overlap.
    """
    out_bytes, osd_bytes = stage1_output_nbytes(rows, inter_dim, a_dtype)
    if arena is None:
        out = torch.zeros(out_bytes, dtype=torch.uint8, device=device)
        osd = torch.zeros(osd_bytes, dtype=torch.uint8, device=device)
    else:
        if out_bytes > arena[0].numel() or osd_bytes > arena[1].numel():
            raise ValueError("the Stage1 output arena is smaller than this route needs")
        out, osd = arena[0][:out_bytes], arena[1][:osd_bytes]
    out = out.view(rows, out_bytes // rows)
    out = out.view(torch.float4_e2m1fn_x2 if a_dtype == "fp4" else torch.float8_e4m3fn)
    return out, osd


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
    work_shards: int | None = None, payload_chunk_rows: int = 0,
    tile_state_stride: int = 0, a_dtype: str = "fp8", out_dtype: str = "fp8",
    swiglu_limit: float = 0.0,
    act: str = "silu", situ_beta: float = 1.0, situ_linear_beta: float = 1.0,
    early_tiles: bool = False, light_release: bool = False, resident_grid: bool = False,
    unroll_k: bool = False, g1_split_tail: bool = False, fused_stage2: tuple | None = None,
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
    compact_dispatch = not fixed_slot_dispatch
    # Assign roles by first-arrival ticket rather than block ID.  The launch
    # deliberately oversubscribes the device so producer CTAs can retire and
    # queued consumers can backfill them.  Hardware does not guarantee block
    # residency order, while tickets guarantee that every required producer
    # belongs to the first resident cohort.
    if compact_dispatch and (payload_chunk_rows <= 0 or tile_state_stride <= 0):
        raise ValueError("compact dispatch requires preplanned tile-ready payloads")
    # Compact launches exactly one CTA per CU.  Low block IDs perform finite
    # payload work and then join the common GEMM work queue; the remaining
    # CTAs may wait for tile readiness without preventing any producer from
    # becoming resident.  Fixed-slot retains arrival tickets until its owner
    # epoch protocol is converted to the same bounded-grid scheme.
    # Fixed-slot with a resident grid launches only the consumer cohort; the
    # arrival-ticket owner and producers are its first resident CTAs and take
    # GEMM work once their role is done, so no consumer queues behind them.
    # The fixed-slot Stage1 options below are validated for the V4/V4.1 target layout (48 experts per rank)
    # only; other fixed-slot layouts (the V4.1-Flash DSpark draft MoE's 16 per rank) keep the base kernel.
    TARGET_FIXED = fixed_slot_dispatch and int(experts_per_rank) == REFERENCE_EXPERTS_PER_RANK
    RESIDENT_GRID = TARGET_FIXED and resident_grid
    consumer_ticket_base = 0 if compact_dispatch or RESIDENT_GRID else dispatch_blocks + 1
    # Compact external-counting overwrites every source-owned histogram row
    # and publishes COUNT_DONE with the invocation generation.  That exchange
    # already provides the cross-rank release/acquire edge, so a separate
    # launch-ready round trip is redundant.  Fixed-slot and locally-counted
    # paths retain the entry handshake because they do not have that edge.
    # Compact uses a bounded all-resident cohort.  Fixed-slot still queues a
    # full consumer cohort behind its arrival-ticket owner/producers.
    grid_x = num_cu * grid_mult
    assert grid_x > 0, "consumer grid must remain positive"
    launch_grid_x = grid_x if compact_dispatch else consumer_ticket_base + grid_x
    assert launch_grid_x <= num_cu * 33 + 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0
    assert a_dtype in ("fp4", "fp8")
    assert out_dtype in ("fp4", "fp8")

    assert tile_k == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    A_K_STEP_BYTES = tile_k // (2 if a_dtype == "fp4" else 1)
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    WORK_SHARDS = 4 if work_shards is None and int(fuse_mtpr) >= 8192 else 8
    if work_shards is not None:
        WORK_SHARDS = int(work_shards)
    assert WORK_SHARDS in (1, 2, 4, 8)
    NARROW_FINAL_A = (
        compact_dispatch and model_dim == 5120 and inter_dim == 2304
        and fuse_npes == 8 and experts_per_rank == 48 and fuse_topk == 6
        and fuse_mtpr >= 8192 and sort_block_m == 128 and tile_n == 512
        and NUM_WAVES == 8 and pipe_weights and async_a_copy
        and a_dtype == "fp8" and out_dtype == "fp8"
    )

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
    if fixed_slot_dispatch:
        _validate_fixed_slot_geometry(fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m)
    fz_total_experts = fz_npes * fz_epr
    # The payload buffer is sized for the source-indexed layout at the indexed-
    # capable MTPR class, so every rank there uses indexed (one row per source
    # token) regardless of sort_block_m. The receive layout must not depend on a
    # per-rank quantity: a route-major rank would need one row per (token,expert),
    # which cross-rank token skew can push past the shared payload buffer.
    indexed_payload = (
        not fixed_slot_dispatch and fuse_mtpr >= INDEXED_PAYLOAD_MIN_MTPR
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
    a_pack = 2 if a_dtype == "fp4" else 1
    out_pack = 2 if out_dtype == "fp4" else 1
    fz_nbytes = model_dim // a_pack
    fz_n_i32 = fz_nbytes // 4
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if fixed_slot_dispatch and fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    _validate_dispatch_capacity(
        fz_mtpr, fz_npes, fz_epr, fz_k, fz_tile_m, fz_nbytes, inter_dim // out_pack, use_tile_resource
    )

    # Fused GEMM2: once GEMM1 has no work left, consumers run Stage2's output
    # tiles in this kernel, each once its Stage1 M tile's GEMM1 work items are
    # done, instead of a separate Stage2 launch.  ``fused_stage2`` holds the
    # Stage2Unit settings as sorted (name, value) pairs; the unit runs from the
    # base of ``pool``.
    FUSE_G2 = TARGET_FIXED and fused_stage2 is not None
    if FUSE_G2:
        assert TOTAL_THREADS == 256, "fused GEMM2 runs Stage2's 256-thread tile"
        G2_UNIT = Stage2Unit(model_dim=model_dim, inter_dim=inter_dim, experts=fz_epr, topk=fz_k, rank=fz_rank,
            npes=fz_npes, max_tok=fz_mtpr, SBM=sort_block_m, a_dtype=out_dtype, has_pad=False,
            **dict(fused_stage2))
        assert sort_block_m % G2_UNIT.BM == 0 and model_dim % G2_UNIT.BN == 0
        G2_N_BLOCKS = model_dim // G2_UNIT.BN
        lds_pool_bytes = max(lds_pool_bytes, G2_UNIT.lds_bytes)
        u = G2_UNIT
        G2_TAG = (
            f"_fg2t{u.BM}x{u.BN}x{u.BK}nt{int(u.use_nt)}bh{int(u.g2_bhoist)}b2{int(u.g2_b2stage)}"
            f"apf{int(u.g2_ascale_pf)}da{int(u.g2_deep_a_pipeline)}_{u.p2p_quant_type}_sv{u.scatter_vec}"
            f"{'_ws' if u.wide_scatter else ''}"
        )
    # GEMM1 v2: when the work items overflow one round of consumer CTAs, the last
    # G1_V2_TAIL_ITEMS (192) of them (at most the overflow) run as two 128-column items
    # each, so the partial last round is spread over every CTA. M-tile done counts
    # are in 128-column units (a whole item counts 2).
    GEMM1_V2 = FUSE_G2 and g1_split_tail and tile_n == 256
    TAIL_N = tile_n // 2
    N_TILES_TAIL = 2 * N_TILES
    G1_M_TARGET = N_TILES_TAIL if GEMM1_V2 else N_TILES
    G1_V2_TAIL_ITEMS = 192
    if GEMM1_V2:
        G2_TAG += "_g1v2"

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
        # Block-wide broadcast slots. They must not alias ``pool``: the A tile's
        # LDS DMA and the GEMM1 C tile reuse pool while slower waves may still
        # be reading a broadcast value (a lagging wave then read A bytes as its
        # tile index and loaded weights from a garbage expert).
        bc_ticket: fx.Array[fx.Int8, 16, 16]
        bc_work: fx.Array[fx.Int8, 16, 16]
        bc_sched: fx.Array[fx.Int8, 16, 16]

    # Paired GEMM1 K loop for fixed-slot tiles with LDS-DMA A copies; compact
    # prefill is slower with it (its 8-wave tiles already hide the latency).
    PAIR_K = fixed_slot_dispatch and async_a_copy
    # Fully unrolled fixed-slot GEMM1 K loop for the wide tuned tile.
    UNROLL_K = TARGET_FIXED and async_a_copy and tile_n == 256 and unroll_k
    # Fixed-slot GEMM tiles wait for their expert's rows rather than for the
    # whole dispatch: TILE_READY[parity][local_expert * npes + source] counts
    # published rows against COUNT_MATRIX[parity][source][local_expert].
    EARLY_TILES = TARGET_FIXED and early_tiles
    # A system acquire per tile invalidates L2 under the running GEMM, so on
    # mori's uncached heap a tile drops only its CU's L1 lines.
    EARLY_L1_ONLY = EARLY_TILES and _UNCACHED_HEAP
    LIGHT_RELEASE = TARGET_FIXED and light_release and _UNCACHED_HEAP
    if EARLY_TILES:
        assert tile_state_stride > fz_epr * fz_npes and fz_npes <= 64
        assert NUM_WAVES >= 2, "early tiles send counts from wave 1 during the handshake"
        assert lds_pool_bytes >= (fz_total_experts + 2) * 4
    # Count wave LDS: histogram, then the parity/expected broadcast.
    EPOCH_LDS_OFFSET = fz_total_experts * 4
    if EARLY_TILES:
        assert lds_pool_bytes >= EPOCH_LDS_OFFSET + 8
    # GEMM1 quant epilogue columns per lane (SiluQuantEpilogue; FP4 output uses 2).
    EPI_EVEC = 8
    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    # Activation is baked into the epilogue, so it (and its baked beta/linear_beta)
    # must key the kernel or a situv2 launch would collide with the silu cache.
    act_suffix = "" if act == "silu" else (
        f"_{act}"
        f"b{str(float(situ_beta)).replace('.', 'p')}"
        f"lb{str(float(situ_linear_beta)).replace('.', 'p')}"
    )
    WORK_BATCH = 1

    kernel_name = (
        f"megamoe_stage1_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_a{a_dtype}o{out_dtype}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"_cgc{grid_x}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}_ws{WORK_SHARDS}"
        f"_pc{payload_chunk_rows}"
        f"_tss{tile_state_stride}"
        f"_rc31_wb{WORK_BATCH}_adaptive_bc"
        f"_ix{int(indexed_payload)}"
        f"{swiglu_suffix}"
        f"{act_suffix}"
        # _ev8b / _fc: the epilogue width and the compact two-row payload copy,
        # fixed now; kept so kernel names (and AOT caches) stay stable.
        f"_ev{EPI_EVEC}b"
        f"{'_fc' if compact_dispatch else ''}"
        f"{'_kp2' if PAIR_K else ''}"
        f"{'_et' if EARLY_TILES else ''}{'' if EARLY_L1_ONLY or not EARLY_TILES else 'a'}"
        f"{'_lr' if LIGHT_RELEASE else ''}"
        f"{'_rg' if RESIDENT_GRID else ''}{'_uk' if UNROLL_K else ''}"
        f"{'_ac32' if NARROW_FINAL_A else ''}"
        f"{G2_TAG if FUSE_G2 else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, addr_w2: fx.Int64, addr_w2_scale: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        disp_rsrc = ptr_buf_tensor(addr_disp, fx.Int64)
        parity_rsrc = ptr_buf_tensor(addr_parity, fx.Int32)
        expected_rsrc = ptr_buf_tensor(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return disp_rsrc[fx.Int32(int(slot))]

        if const_expr(EARLY_TILES):
            # The owner's count histogram and its parity/expected broadcast
            # live in pool, which the owner CTA does not use until GEMM.
            hist_lds_addr = fx.Int64(fx.ptrtoint(lds.pool.ptr))
            epoch_lds_addr = hist_lds_addr + fx.Int64(EPOCH_LDS_OFFSET)
            warp_id = tid // fx.Int32(64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_pair_order_ready = _disp_ptr(DispatchSlot.PAIR_ORDER_READY)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_work_tail = _disp_ptr(DispatchSlot.WORK_TAIL)
        a_ready_tile_queue = fx.Int64(0)
        a_ready_tile_epoch = fx.Int64(0)
        if const_expr(compact_dispatch):
            a_ready_tile_queue = _disp_ptr(DispatchSlot.READY_TILE_QUEUE)
            a_ready_tile_epoch = _disp_ptr(DispatchSlot.READY_TILE_EPOCH)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
        a_payload_chunks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)
        a_launch_ready = fx.Int64(0)
        p_launch_ready = fx.Int64(0)
        if const_expr(fixed_slot_dispatch):
            a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
            p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_max_expert_tiles = fx.Int64(0)
        if const_expr(compact_dispatch):
            a_max_expert_tiles = _disp_ptr(DispatchSlot.MAX_EXPERT_TILES)

        if const_expr(compact_dispatch):
            ticket = fx.block_idx.x
            generation = fx.Int64(0)
        else:
            ticket_scratch = fx.recast_iter(fx.Int64, lds.bc_ticket.ptr)
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
        if const_expr(compact_dispatch):
            is_owner = fx.Int32(0) == 1
            is_producer = ticket < fx.Int32(dispatch_blocks)
            producer_slot = ticket
        else:
            is_owner = ticket == fx.Int32(0)
            is_producer = (ticket > fx.Int32(0)) & (
                ticket <= fx.Int32(dispatch_blocks)
            )
            producer_slot = ticket - fx.Int32(1)

        if const_expr(fixed_slot_dispatch):
            if is_owner:
                next_parity_lane = fx.Int32(0)
                launch_epoch_lane = fx.Int32(0)
                if tid == fx.Int32(0):
                    old_parity = parity_rsrc[fx.Int32(0)]
                    next_parity_lane = old_parity ^ fx.Int32(1)
                    previous_expected = expected_rsrc[next_parity_lane]
                    next_expected = previous_expected + fx.Int32(fz_npes)
                    expected_rsrc[next_parity_lane] = next_expected
                    launch_epoch_lane = (
                        (next_expected // fx.Int32(fz_npes)) * fx.Int32(2)
                        - next_parity_lane
                    )
                    if const_expr(EARLY_TILES):
                        comm_ops.store_i32_lds(epoch_lds_addr, next_parity_lane)
                        comm_ops.store_i32_lds(epoch_lds_addr + fx.Int64(4), next_expected)
                if const_expr(EARLY_TILES):
                    fx.barrier()
                    if warp_id == fx.Int32(1):
                        emit_direct_fixed_slot_counts(
                            fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_rank=fz_rank,
                            fz_total_experts=fz_total_experts, addr_disp=addr_disp, addr_in_idx=addr_in_idx,
                            i32_cur_tok=i32_cur_tok, hist_lds_addr=hist_lds_addr,
                            parity=fx.Int32(comm_ops.load_i32_lds(epoch_lds_addr)),
                            expected=fx.Int32(comm_ops.load_i32_lds(epoch_lds_addr + fx.Int64(4))),
                        )
                next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
                launch_epoch = fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, launch_epoch_lane)
                )
                if tid < fx.Int32(fz_npes):
                    peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                    comm_ops.fence_system_release()
                    launch_ready_table = ptr_buf_tensor(p_launch_ready, fx.Int64)
                    remote_launch_ready = launch_ready_table[peer]
                    comm_ops.store_i32_system(
                        remote_launch_ready, fx.Int32(fz_rank), launch_epoch
                    )
                    comm_ops.wait_i32_until_greater_than(
                        a_launch_ready + fx.Int64(peer) * 4,
                        launch_epoch - 1,
                    )
                    comm_ops.fence_system_acquire()
                if tid == fx.Int32(0):
                    work_head_rsrc = ptr_buf_tensor(a_work_head, fx.Int32)
                    for shard in range_constexpr(WORK_SHARDS):
                        work_head_rsrc[fx.Int32(shard * 16)] = fx.Int32(0)
                    comm_ops.store_i32_system(
                        a_work_tail, fx.Int32(0), fx.Int32(0)
                    )
                    group_done_rsrc = ptr_buf_tensor(a_group_done, fx.Int32)
                    for destination in range_constexpr(fz_npes):
                        group_done_rsrc[fx.Int32(destination)] = fx.Int32(0)
                if tid == fx.Int32(0):
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    comm_ops.fence_agent_release()
                    parity_rsrc[fx.Int32(0)] = next_parity
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    comm_ops.fence_agent_release()
                    comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
            else:
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(gate_addr, gate_epoch)
                    comm_ops.fence_agent_acquire()
                fx.barrier()

        payload_parity = buf_copy_load(
            parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE
        )
        payload_expected = buf_copy_load(
            expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE
        )
        payload_epoch = fx.Int32(0)
        tile_state_byte_offset = fx.Int64(0)
        if const_expr(compact_dispatch):
            payload_epoch = (
                payload_expected // fx.Int32(fz_npes)
            ) * fx.Int32(2) - payload_parity
            tile_state_byte_offset = (
                fx.Int64(payload_parity)
                * fx.Int64(tile_state_stride)
                * 4
            )
            a_ready_tile_queue = a_ready_tile_queue + tile_state_byte_offset
            a_ready_tile_epoch = a_ready_tile_epoch + tile_state_byte_offset

        if is_producer:
            if const_expr(fixed_slot_dispatch):
                emit_direct_fixed_slot_payload(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                    parity=payload_parity, expected=payload_expected,
                    early_tiles=EARLY_TILES, tile_state_stride=tile_state_stride,
                    ready_lds_addr=hist_lds_addr if EARLY_TILES else None, ready_lds_bytes=lds_pool_bytes,
                    light_release=LIGHT_RELEASE,
                )
            else:
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(
                        a_pair_order_ready
                        + fx.Int64(payload_parity) * 4,
                        payload_expected,
                    )
                    comm_ops.fence_system_acquire()
                fx.barrier()
                producer_destination = producer_slot % fx.Int32(fz_npes)
                producers_per_destination = ptr_buf_tensor(
                    a_payload_blocks_per_destination,
                    fx.Int32,
                )[producer_destination]
                chunks_per_destination = ptr_buf_tensor(
                    a_payload_chunks_per_destination,
                    fx.Int32,
                )[producer_destination]
                emit_dispatch_payload(
                    num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32,
                    fz_enable_scales=fz_enable_scales, addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                    addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    dispatch_blocks=dispatch_blocks,
                    producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                    producers_per_destination=producers_per_destination, payload_chunk_rows=payload_chunk_rows,
                    chunks_per_destination=chunks_per_destination,
                    tile_state_stride=tile_state_stride,
                    indexed_payload=indexed_payload,
                )
        if const_expr(fixed_slot_dispatch):
            if is_owner:
                emit_direct_fixed_slot_finalize(
                    fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_tile_m=fz_tile_m, n_tiles=N_TILES, addr_disp=addr_disp, parity=payload_parity,
                    expected=payload_expected,
                    early_tiles=EARLY_TILES, tile_state_stride=tile_state_stride, num_waves=NUM_WAVES,
                )
        else:
            addr_tile_ready = _disp_ptr(DispatchSlot.TILE_READY)
            addr_tile_expected = _disp_ptr(DispatchSlot.TILE_EXPECTED)
            addr_tile_ready = addr_tile_ready + tile_state_byte_offset
            addr_tile_expected = addr_tile_expected + tile_state_byte_offset
        wave_id = fx.thread_idx.x // 64

        w_rsrc = ptr_buf_tensor(fx.get_iter(w), fx.Int32, unit_elems=4)
        sx_rsrc = ptr_buf_tensor(fx.get_iter(scale_x), fx.Int32, unit_elems=4)
        sw_rsrc = ptr_buf_tensor(fx.get_iter(scale_w), fx.Int32)
        trb_rsrc = ptr_buf_tensor(fx.get_iter(sorted_token_ids), fx.Int32)
        tib_rsrc = ptr_buf_tensor(
            _disp_ptr(DispatchSlot.TILE_INPUT_BASE), fx.Int32
        )
        srcmap_rsrc = ptr_buf_tensor(_disp_ptr(DispatchSlot.SRCMAP), fx.Int32)
        expert_rsrc = ptr_buf_tensor(fx.get_iter(expert_ids), fx.Int32)
        nv_rsrc = ptr_buf_tensor(fx.get_iter(num_valid_ids), fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + 8192
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_nbytes = tokens * fx.Int32(inter_dim // out_pack)
            # In the element width of the epilogue's stores (see SiluQuantEpilogue).
            if const_expr(out_dtype == "fp4"):
                out_rsrc = ptr_buf_tensor(
                    fx.get_iter(out), fx.Int8, num_records_bytes=out_nbytes
                )
            else:
                out_rsrc = ptr_buf_tensor(
                    fx.get_iter(out), fx.Int32, unit_elems=2, num_records_bytes=out_nbytes
                )
        os_rsrc = ptr_buf_tensor(
            fx.get_iter(out_scale), fx.Int8, num_records_bytes=os_nbytes
        )

        _, _do_scheduled_tile = build_fused_gemm1(
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
            pair_k=PAIR_K, evec=EPI_EVEC, unroll_k=UNROLL_K,
            indirect_input=compact_dispatch and not indexed_payload,
            indexed_input=indexed_payload,
            row_map_rsrc=srcmap_rsrc,
            source_rows=source_rows,
            a_dtype=a_dtype, out_dtype=out_dtype,
            narrow_final_a=NARROW_FINAL_A,
            swiglu_limit=swiglu_limit,
            act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
        )

        def _do_tail_tile(flat):
            # Built at the call so its per-thread setup is not live through the
            # kernel, and with the rolled K loop: a second unrolled one spills.
            _, run_tail = build_fused_gemm1(
                x_tensor=x, w_rsrc=w_rsrc,
                sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
                trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
                tib_rsrc=tib_rsrc,
                a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
                model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
                tile_n=TAIL_N, num_waves=NUM_WAVES, n_per_wave=TAIL_N // NUM_WAVES, wave_id=wave_id,
                m_repeat=M_REPEAT, num_acc_n=(TAIL_N // NUM_WAVES) // 16, a_k_step_bytes=A_K_STEP_BYTES,
                total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
                n_tiles=N_TILES_TAIL, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
                swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
                async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
                pair_k=PAIR_K, evec=EPI_EVEC, unroll_k=False,
                indirect_input=False, indexed_input=indexed_payload,
                row_map_rsrc=srcmap_rsrc,
                source_rows=source_rows,
                a_dtype=a_dtype, out_dtype=out_dtype,
                narrow_final_a=NARROW_FINAL_A,
                swiglu_limit=swiglu_limit,
                act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
            )
            run_tail(flat)

        if tid == fx.Int32(0):
            local_plan_ready = disp_rsrc[fx.Int32(int(DispatchSlot.PLAN_READY))]
            ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.wait_i32_until_equals(
                local_plan_ready + fx.Int64(ready_index) * 4, payload_expected)
            # PLAN_READY is published with a system-scope release after the
            # planner rewrites TILE_EXPECTED.  Match that scope so a runtime
            # layout transition cannot reuse the previous invocation's
            # expected count and wait forever on the wrong value.
            if const_expr(compact_dispatch):
                comm_ops.fence_system_acquire()
            else:
                comm_ops.fence_agent_acquire()
        fx.barrier()

        num_valid = nv_rsrc[fx.Int32(0)]
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)
        if const_expr(GEMM1_V2):
            g1_items = total_work
            g1_over = g1_items - fx.Int32(launch_grid_x - consumer_ticket_base)
            g1_over = (g1_over > fx.Int32(0)).select(g1_over, fx.Int32(0))
            g1_tail = (g1_over < fx.Int32(G1_V2_TAIL_ITEMS)).select(g1_over, fx.Int32(G1_V2_TAIL_ITEMS))
            g1_big = g1_items - g1_tail
            total_work = g1_big + g1_tail * fx.Int32(2)
        use_ready_order = fx.Int32(0) == 1
        if const_expr(compact_dispatch):
            max_expert_tiles = ptr_buf_tensor(a_max_expert_tiles, fx.Int32)[
                fx.Int32(0)
            ]
            use_ready_order = max_expert_tiles * fx.Int32(4) >= num_m_tiles

        if const_expr(EARLY_TILES):
            a_fixed_rows_ready = _disp_ptr(DispatchSlot.TILE_READY) + fx.Int64(
                payload_parity
            ) * fx.Int64(tile_state_stride * 4)
            fixed_source_counts = ptr_buf_tensor(
                _disp_ptr(DispatchSlot.COUNT_MATRIX), fx.Int32
            )

        def _wait_fixed_tile_rows(flat, source):
            # Lane ``source`` waits until that source has published all of its
            # rows for the tile's expert.
            local_expert = trb_rsrc[flat // fx.Int32(N_TILES)] // fx.Int32(fz_cap)
            comm_ops.wait_i32_until_equals(
                a_fixed_rows_ready
                + fx.Int64(local_expert * fx.Int32(fz_npes) + source) * 4,
                fixed_source_counts[
                    payload_parity * fx.Int32(fz_npes * fz_epr)
                    + source * fx.Int32(fz_epr)
                    + local_expert
                ],
            )

        def _wait_tile_payload(flat):
            tile_index = flat // fx.Int32(N_TILES)
            expected_tiles = ptr_buf_tensor(addr_tile_expected, fx.Int32)[tile_index]
            comm_ops.wait_i32_until_equals(
                addr_tile_ready + fx.Int64(tile_index) * 4,
                expected_tiles,
            )

        # Arrival tickets keep fixed-slot owner/producers in the first resident
        # cohort.  The stable block-ID suffix performs GEMM after any finite
        # role work, except on the two-cohort grid where arrival order has the
        # shorter tail.  Compact already uses block IDs and a bounded grid.
        consumer_ticket = ticket
        if const_expr(fixed_slot_dispatch and grid_mult != 2):
            consumer_ticket = fx.block_idx.x
        consumer_id = consumer_ticket - fx.Int32(consumer_ticket_base)
        consumer_base = fx.Int32(consumer_ticket_base)
        consumer_active = (consumer_ticket >= consumer_base) & (
            consumer_id < total_work
        )
        if const_expr(FUSE_G2):
            # Every consumer reaches the GEMM2 phase, with or without GEMM1 work.
            consumer_active = consumer_ticket >= consumer_base
        work_scratch = fx.recast_iter(fx.Int32, lds.bc_work.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        # scheduled_first gets its own slot: tid 0 writes it while other waves
        # may not have read first_work yet.
        sched_scratch = fx.recast_iter(fx.Int32, lds.bc_sched.ptr)
        sched_scratch_view = fx.make_view(sched_scratch, fx.make_layout(1, 1))
        work_shard = consumer_id & fx.Int32(WORK_SHARDS - 1)
        work_batch = WORK_BATCH
        assert N_TILES % work_batch == 0


        if const_expr(FUSE_G2):
            a_g2_state = _disp_ptr(DispatchSlot.G2_STATE)
            G2_CONSUMERS = launch_grid_x - consumer_ticket_base

            def _g2_state_addr(entry):
                return a_g2_state + fx.Int64(entry) * 4

            def _g2_m_done_addr(m_tile):
                return _g2_state_addr(fx.Int32(FUSED_STAGE2_HEAD_INTS) + m_tile)

            def _run_fused_stage2():
                g2_scratch = fx.recast_iter(fx.Int32, lds.bc_work.ptr)
                g2_scratch_view = fx.make_view(g2_scratch, fx.make_layout(1, 1))
                # Every wave has read its last GEMM1 claim from the same slot.
                fx.barrier()
                if consumer_ticket >= consumer_base:
                    lds_base = fx.Int32(fx.ptrtoint(lds.pool.ptr))
                    G2_UNIT.load_peers(lds_base, _disp_ptr(DispatchSlot.P2P_COMB_INP), tid)
                    g2_total = ceildiv(num_valid, fx.Int32(G2_UNIT.BM)) * fx.Int32(G2_N_BLOCKS)
                    # Rebuilt here rather than reusing GEMM1's srcmap view, which
                    # would stay live through the whole GEMM1 loop.
                    g2_srcmap = ptr_buf_tensor(_disp_ptr(DispatchSlot.SRCMAP), fx.Int32)
                    g2_weights = ptr_buf_tensor(_disp_ptr(DispatchSlot.WEIGHT), fx.Float32)
                    g2_lane = tid % fx.Int32(64)
                    g2_wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // fx.Int32(64)))
                    g2_active = fx.Int32(1) == 1
                    while g2_active:
                        if tid == fx.Int32(0):
                            claimed = fx.Int32(comm_ops.atomic_add_agent(
                                _g2_state_addr(fx.Int32(FUSED_STAGE2_CLAIM)), fx.Int32(1)
                            ))
                            fx.ptr_store(Vec.from_elements([claimed], fx.Int32), g2_scratch)
                        fx.barrier()
                        g2_unit = Vec(g2_scratch_view.load())[0]
                        if g2_unit < g2_total:
                            m_block = g2_unit // fx.Int32(G2_N_BLOCKS)
                            if tid == fx.Int32(0):
                                comm_ops.wait_i32_until_equals(
                                    _g2_m_done_addr(
                                        m_block * fx.Int32(G2_UNIT.BM) // fx.Int32(sort_block_m)
                                    ),
                                    fx.Int32(G1_M_TARGET),
                                )
                                comm_ops.invalidate_l1()
                            fx.barrier()
                            G2_UNIT.run(
                                lds_base, g2_unit, m_block,
                                arg_aq=_disp_ptr(DispatchSlot.S1_OUT),
                                arg_ascale=_disp_ptr(DispatchSlot.S1_OUT_SCALE),
                                arg_bq=addr_w2,
                                arg_bscale=addr_w2_scale,
                                arg_eids=_disp_ptr(DispatchSlot.SORTED_EXPERT),
                                i32_max_m_blocks=tokens // fx.Int32(G2_UNIT.BM),
                                i32_inter=fx.Int32(inter_dim), i32_hidden=fx.Int32(model_dim),
                                i32_kpad=fx.Int32(0), i32_npad=fx.Int32(0),
                                k_bytes=fx.Int32(inter_dim // (1 if out_dtype == "fp8" else 2)),
                                trb_buf=trb_rsrc, stids_buf=g2_srcmap, sweights_buf=g2_weights,
                                tx_i32=tid, lane=g2_lane, wave=g2_wave,
                            )
                        # The next tile's A stages reuse this one's CShuffle LDS.
                        fx.barrier()
                        g2_active = g2_unit < g2_total
                    # Thread 0 made all of this CTA's GEMM1 counts and claims;
                    # on the uncached heap they are performed in memory once
                    # vmcnt drains, so the last CTA done clears a state no one
                    # touches again in this launch.
                    if tid == fx.Int32(0):
                        fx.rocdl.s_waitcnt(vmcnt=0)
                        done = fx.Int32(comm_ops.atomic_add_agent(
                            _g2_state_addr(fx.Int32(FUSED_STAGE2_DONE)), fx.Int32(1)
                        ))
                        fx.ptr_store(Vec.from_elements([done], fx.Int32), g2_scratch)
                    fx.barrier()
                    if Vec(g2_scratch_view.load())[0] == fx.Int32(G2_CONSUMERS - 1):
                        for entry in range(
                            tid, fx.Int32(fused_stage2_state_ints(tile_state_stride)), TOTAL_THREADS
                        ):
                            comm_ops.store_i32_global_system_monotonic(
                                _g2_state_addr(entry), fx.Int32(0)
                            )

        def _run_work_batch(first_work, scheduled_first):
            for batch_offset in range_constexpr(work_batch):
                if const_expr(compact_dispatch):
                    work = use_ready_order.select(
                        scheduled_first, first_work
                    ) + fx.Int32(batch_offset)
                else:
                    work = first_work + fx.Int32(batch_offset * WORK_SHARDS)
                if work < total_work:
                    if const_expr(fixed_slot_dispatch):
                        if const_expr(EARLY_TILES):
                            if tid < fx.Int32(fz_npes):
                                if const_expr(GEMM1_V2):
                                    # A 64 x 128 item waits for the rows of the 64 x 256 item it is half of.
                                    _wait_fixed_tile_rows(
                                        (work < g1_big).select(work, (g1_big * fx.Int32(2) + (work - g1_big)) // fx.Int32(2)),
                                        tid,
                                    )
                                else:
                                    _wait_fixed_tile_rows(work, tid)
                                if const_expr(EARLY_L1_ONLY):
                                    comm_ops.invalidate_l1()
                                else:
                                    comm_ops.fence_system_acquire()
                            fx.barrier()
                        else:
                            fx.barrier()
                    elif not use_ready_order:
                        # Ready-order waits for the tile in the scheduler block.
                        if tid == fx.Int32(0):
                            _wait_tile_payload(work)
                        fx.barrier()
                        comm_ops.fence_system_acquire()
                    if const_expr(GEMM1_V2):
                        if work < g1_big:
                            _do_scheduled_tile(work)
                        else:
                            _do_tail_tile(g1_big * fx.Int32(2) + (work - g1_big))
                    else:
                        _do_scheduled_tile(work)
                    # Recomputed after the tile so nothing extra stays live through it.
                    if const_expr(GEMM1_V2):
                        g1_whole = work < g1_big
                        g1_m = g1_whole.select(
                            work // fx.Int32(N_TILES),
                            (g1_big * fx.Int32(2) + (work - g1_big)) // fx.Int32(N_TILES_TAIL),
                        )
                        g1_inc = g1_whole.select(fx.Int32(2), fx.Int32(1))
                    else:
                        g1_m = work // fx.Int32(N_TILES)
                        g1_inc = fx.Int32(1)
                    if const_expr(FUSE_G2):
                        # The GEMM2 tiles of this M tile read its output once
                        # every work item's stores have landed (uncached heap).
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        fx.barrier()
                        if tid == fx.Int32(0):
                            comm_ops.atomic_add_agent(_g2_m_done_addr(g1_m), g1_inc)

        steal = fx.Int32(0)
        while consumer_active:
            if tid == fx.Int32(0):
                first_work = fx.Int32(0)
                if const_expr(compact_dispatch):
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
                                + fx.Int64(work_shard) * 64,
                                fx.Int32(work_batch),
                            )
                        )
                        first_work = (
                            work_shard
                            + local_work * fx.Int32(WORK_SHARDS)
                        )
                else:
                    claim_shard = work_shard
                    if const_expr(FUSE_G2):
                        claim_shard = (work_shard + steal) & fx.Int32(WORK_SHARDS - 1)
                    local_work = fx.Int32(
                        comm_ops.atomic_add_agent(
                            a_work_head
                            + fx.Int64(claim_shard) * 64,
                            fx.Int32(work_batch),
                        )
                    )
                    first_work = (
                        claim_shard + local_work * fx.Int32(WORK_SHARDS)
                    )
                fx.ptr_store(
                    Vec.from_elements([first_work], fx.Int32), work_scratch
                )
                fx.ptr_store(
                    Vec.from_elements([first_work], fx.Int32), sched_scratch
                )
            fx.barrier()
            first_work = Vec(work_scratch_view.load())[0]
            if (  # noqa: SIM102 - preserve DSL staging
                const_expr(compact_dispatch)
                and first_work < total_work
            ):
                if use_ready_order:
                    ready_slot = first_work // fx.Int32(N_TILES)
                    first_n_tile = first_work - ready_slot * fx.Int32(N_TILES)
                    if tid == fx.Int32(0):
                        comm_ops.wait_i32_until_equals(
                            a_ready_tile_epoch
                            + fx.Int64(ready_slot) * 4,
                            payload_epoch,
                        )
                        comm_ops.fence_system_acquire()
                        ready_m_tile = ptr_buf_tensor(
                            a_ready_tile_queue, fx.Int32
                        )[ready_slot]
                        scheduled_first = (
                            ready_m_tile * fx.Int32(N_TILES) + first_n_tile
                        )
                        _wait_tile_payload(scheduled_first)
                        fx.ptr_store(
                            Vec.from_elements([scheduled_first], fx.Int32),
                            sched_scratch,
                        )
                    fx.barrier()
            scheduled_first = Vec(sched_scratch_view.load())[0]
            if const_expr(compact_dispatch):  # noqa: SIM102 - preserve DSL staging
                if use_ready_order:
                    comm_ops.fence_system_acquire()
            _run_work_batch(first_work, scheduled_first)
            if const_expr(FUSE_G2):
                # A CTA leaves for GEMM2 only once every shard is drained, so
                # each GEMM1 work item a GEMM2 tile waits for is already held
                # by a running CTA, whatever the residency order.
                steal = steal + (first_work >= total_work).select(fx.Int32(1), fx.Int32(0))
                consumer_active = steal < fx.Int32(WORK_SHARDS)
            else:
                consumer_active = first_work < total_work

        if const_expr(FUSE_G2):
            _run_fused_stage2()


    spec = _Stage1KernelSpec(
        kernel, launch_grid_x, TOTAL_THREADS, waves_per_eu_hint
    )
    if _return_kernel_spec:
        return spec

    # Close over the kernel and the launch ints themselves, not ``spec``: FlyDSL
    # keys a launcher on closed-over kernels and scalars but not on an opaque
    # object, so every Stage1 config would otherwise share one cached launcher.
    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, addr_w2: fx.Int64, addr_w2_scale: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            addr_w2, addr_w2_scale,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(
            grid=(launch_grid_x, 1, 1),
            block=(TOTAL_THREADS, 1, 1),
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
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    a_dtype: str = "fp8",
    out_dtype: str = "fp8",
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
            tile_state_stride=tile_state_stride,
            swiglu_limit=swiglu_limit,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            a_dtype=a_dtype,
            out_dtype=out_dtype,
            early_tiles=config.early_tiles,
            light_release=config.light_release,
            resident_grid=config.resident_grid,
            unroll_k=config.unroll_k,
            _return_kernel_spec=True,
        )
        for config in variants
    )
    kernels = [spec.kernel for spec in specs]
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
        for index in range_constexpr(len(kernels)):
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
                    fx.Int64(0),
                    fx.Int64(0),
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


@aot_factory("mega_moe_stage1_bundle")
@functools.cache
def _stage1_bundle_launcher(*, variants: tuple, **compile_kw):
    """compile_mega_moe_stage1_bundle's launcher, with ``variants`` given as
    the field values of each Stage1Config (_variant_fields), which the AOT
    cache can key."""
    return compile_mega_moe_stage1_bundle(
        variants=tuple(Stage1Config(*fields) for fields in variants), **compile_kw
    )


@functools.cache
def _variant_fields(variants: tuple) -> tuple:
    return tuple(
        tuple(getattr(config, f.name) for f in dataclasses.fields(config))
        for config in variants
    )


def _stage1_bundle_lazy(compile_kw: dict):
    """The launch handle of compile_mega_moe_stage1_bundle(**compile_kw)."""
    variants = _variant_fields(tuple(compile_kw["variants"]))
    return _stage1_bundle_launcher.lazy(**{**compile_kw, "variants": variants})


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, addr_w2, addr_w2_scale, stream, *, model_dim, inter_dim, rank, experts_per_rank,
    fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, work_shards=None,
    payload_chunk_rows=0, tile_state_stride=0,
    a_dtype="fp8", out_dtype="fp8",
    swiglu_limit=0.0, act="silu", situ_beta=1.0, situ_linear_beta=1.0,
    early_tiles=False, light_release=False, resident_grid=False, unroll_k=False, g1_split_tail=False,
    fused_stage2=None):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, work_shards=work_shards, payload_chunk_rows=payload_chunk_rows,
        tile_state_stride=tile_state_stride, a_dtype=a_dtype, out_dtype=out_dtype,
        swiglu_limit=swiglu_limit, act=act, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta,
        early_tiles=early_tiles, light_release=light_release, resident_grid=resident_grid,
        unroll_k=unroll_k, g1_split_tail=g1_split_tail, fused_stage2=fused_stage2,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, addr_w2, addr_w2_scale, stream,
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
    launch = _stage1_bundle_lazy(compile_kw)
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
    launch = _stage1_bundle_lazy(compile_kw)
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

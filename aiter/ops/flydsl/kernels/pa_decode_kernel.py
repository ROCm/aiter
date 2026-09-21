# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8 paged-attention tile kernel.

K/V use e4m3 (FNUZ on gfx942, OCP on gfx950); BF16/FP16 Q and probabilities P
are quantized to FP8. Q/key scales fold into QK, value scale and 1/FP8_MAX into
the epilogue; softmax max/sum stay f32. Tuned gfx950 BF16 per-token MTP3/MTP4
uses K128 MFMA instead of K32, preserving normalized Q/P and operand layouts.
Tuned gfx950 BF16 scalar decode casts Q/P directly, without normalization or
1/FP8_MAX compensation: Q must fit FP8, and small Q/probabilities may underflow.
Per-token scales retain range normalization.

Logical layouts (not preshuffled):

* ``query``        [num_seqs, num_q_heads, head_dim]  f16/bf16 (head_dim contiguous)
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (plain), by rank
* ``block_tables`` [num_seqs, max_blocks_per_seq]  int32
* ``context_lengths`` [num_seqs]  int32
* ``output``       [num_seqs, num_q_heads, head_dim]  same dtype as query
* K/V scales      [1] per-tensor or [num_blocks, num_kv_heads, block_size] per-token

Four-wave CTAs process 256-token blocks: QK splits tokens, PV splits head dim,
and P passes through LDS to transpose ownership between the MMAs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.runtime.device import get_rocm_arch

from . import dpp_utils
from .tensor_shim import buf_base_i64, buf_copy_store, ptr_buf_tensor
from .utils import rcp_f32

MFMA_MNK = 16  # M=N=16; query rows are padded to M-tiles.
# One i64 operand pack/lane covers K32, including in K128 layouts.
FP8_PACK_K = 32
WAVE = 64
# Accumulator elements/lane are independent of MFMA K.
MFMA_ACC_ELEMS = MFMA_MNK * MFMA_MNK // WAVE
LOG2E = 1.4426950408889634
KV_COMPUTE_BLOCK = 256
# Key by selected specialization, not batch/head/CU scheduling inputs.
_PA_DECODE_TILE_CACHE = {}


def compile_pa_decode_tile(
    *,
    head_dim: int,
    query_group_size: int,
    block_size: int,
    num_seqs: int,
    num_kv_heads: int,
    num_compute_units: int,
    num_partitions: int = 1,
    softmax_scale: float | None = None,
    query_dtype: str = "f16",
    per_token_kv: bool = False,
    query_length: int = 1,
    trans_v: bool = True,
    wide_kv_addressing: bool = False,
    query_splits: int | None = None,
    use_work_plan: bool = False,
    work_capacity: int | None = None,
    sliding_window: int = 0,
    use_sinks: bool = False,
    sink_dtype_str: str = "f32",
):
    """Select and cache a PA-decode kernel and launch wrapper.

    ``query_splits=None`` selects from host-known grid bounds; an explicit
    count overrides splitting and selects the matching prefetch policy.
    CTAs receive equal groups of MTP positions, flattened with GQA into 16-row
    M-tiles. Partial output stays unsplit for the shared reducer.

    Single-tile specialization requires a standard plan whose capacity proves
    one KV tile per active task; padding skips the task body. Selected plans
    map packed slots over (B, C/B), taking sequence IDs from plan records.
    Launch B must be positive, divide capacity C, and match schedule selection;
    the JIT launch does not recheck these conditions. B and C/B are not cache keys.
    Planned gfx950 BF16 D128 output uses slot-rebased buffer stores when the
    slot's byte span fits signed i32, retaining the dtype's 2-byte alignment.

    Positive ``sliding_window`` requires a plan and includes the query token.
    Plans cover the MTP window union; scores are masked per query row.
    ``use_sinks`` adds a zero-value per-head logit only to direct NP=1 output;
    partitioned/planned output adds it once in the reducer, not in partials.

    Masked V bytes must remain finite because ``0 * NaN == NaN`` in PV MFMA.
    Pages past the sequence are pinned to block 0; callers must leave the
    unwritten tail of the last owned page finite. ``wide_kv_addressing`` uses
    i64 offsets when a cache reaches 2 GiB and the i32 page product would wrap.
    """
    if sliding_window > 0 and not use_work_plan:
        raise ValueError("positive sliding_window requires work_plan")
    is_gfx950 = "gfx95" in get_rocm_arch()
    IS_BF16 = query_dtype == "bf16"
    TUNED_SHAPE = is_gfx950 and head_dim == 128 and block_size in (16, 128)
    TUNED_PER_TOKEN = TUNED_SHAPE and IS_BF16 and per_token_kv
    # Scalar scheduling is broader than the BF16 single-query numerical fast path.
    TUNED_SCALAR = TUNED_SHAPE and trans_v and not per_token_kv

    dense_workgroups = num_seqs * num_kv_heads * num_partitions
    assert query_length >= 1, f"query_length must be >= 1, got {query_length}"
    split_workgroups = dense_workgroups
    single_tile_plan = False
    if use_work_plan:
        if work_capacity is not None:
            split_workgroups = work_capacity * num_kv_heads
        if sliding_window > 0:
            # Bound the unaligned MTP window union, excluding padding CTAs.
            window_tiles = (
                sliding_window + query_length - 2 + KV_COMPUTE_BLOCK - 1
            ) // KV_COMPUTE_BLOCK + 1
            split_workgroups = min(
                split_workgroups, num_seqs * num_kv_heads * window_tiles
            )
            # For n nonempty sequences, T <= n * window_tiles and the budget
            # leaves >= n * (window_tiles - 1) extras. Prefix apportionment
            # therefore gives each visible tile its own task, even after refresh.
            single_tile_plan = (
                work_capacity is not None
                and num_partitions >= window_tiles
                and work_capacity >= num_seqs * window_tiles
            )
    if query_splits is None:
        # Single-tile page128 MTP4 tolerates more split-query CTAs per CU.
        split_weight = (
            1
            if single_tile_plan and query_length == 4 and block_size == 128 and trans_v
            else query_length
        )
        query_splits = (
            query_length
            if TUNED_PER_TOKEN
            and num_kv_heads == 1
            and query_length in (2, 4)
            and query_group_size == 16
            and split_weight * split_workgroups <= 2 * num_compute_units
            else 1
        )
    assert query_splits in (1, 2, 4), "query_splits must be one of 1, 2, 4"
    assert query_length % query_splits == 0, "query_splits must divide query_length"
    QUERIES_PER_CTA = query_length // query_splits
    TOTAL_ROWS = query_length * query_group_size

    # Prefetch single-M-tile queries; large multi-tile grids favor fewer registers.
    PER_TOKEN_M1 = (
        TUNED_PER_TOKEN
        and QUERIES_PER_CTA == 1
        and 8 <= query_group_size <= 16
        and (
            query_splits > 1
            or block_size == 16
            or not trans_v
            or dense_workgroups <= num_compute_units
            or (single_tile_plan and split_workgroups <= 2 * num_compute_units)
        )
    )
    prefetch_v = PER_TOKEN_M1 or (
        TUNED_SCALAR
        and block_size == 128
        and TOTAL_ROWS <= MFMA_MNK
        and num_compute_units < dense_workgroups <= 2 * num_compute_units
    )
    # Require an exact one-tile task budget and the small planned reducer.
    batch_first_plan_grid = (
        TUNED_PER_TOKEN
        and single_tile_plan
        and query_length == query_splits == 4
        and num_kv_heads == 1
        and query_group_size == 16
        and block_size == 128
        and 4096 <= sliding_window <= 8192
        and 4 <= num_seqs <= 24
        and num_partitions <= 64
        and work_capacity == num_seqs * window_tiles
        and split_workgroups <= 2 * num_compute_units
    )
    cache_key = (
        head_dim,
        query_group_size,
        block_size,
        num_partitions,
        softmax_scale,
        query_dtype,
        per_token_kv,
        query_length,
        trans_v,
        wide_kv_addressing,
        prefetch_v,
        query_splits,
        use_work_plan,
        single_tile_plan,
        batch_first_plan_grid,
        sliding_window,
        use_sinks,
        sink_dtype_str,
    )
    cached = _PA_DECODE_TILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    buffer_plan_output = (
        use_work_plan
        and is_gfx950
        and head_dim == 128
        and IS_BF16
        and 0 < TOTAL_ROWS * head_dim * 2 <= 0x7FFFFFFF
    )
    # Larger windows have identical visibility for int32 context lengths.
    sliding_window = min(sliding_window, 2**31 - 1)
    FP8 = fx.Float8E4M3FN if is_gfx950 else fx.Float8E4M3FNUZ
    FP8_MAX = (
        448.0 if is_gfx950 else 240.0
    )  # max representable magnitude of the format above

    assert (
        head_dim % MFMA_MNK == 0
    ), f"head_dim {head_dim} must be a multiple of {MFMA_MNK}"
    assert block_size in (
        16,
        64,
        128,
    ), f"pa_decode_tile only supports block_size in (16, 64, 128), got {block_size}"
    assert query_dtype in (
        "f16",
        "bf16",
    ), f"pa_decode_tile only supports query_dtype in ('f16', 'bf16'), got {query_dtype}"
    Q_DTYPE = fx.BFloat16 if IS_BF16 else fx.Float16

    assert (
        head_dim % 64 == 0
    ), f"pa_decode_tile only supports head_dim that's a multiple of 64, got {head_dim}"
    # Query rows flatten as (MTP, GQA).
    CTA_ROWS = QUERIES_PER_CTA * query_group_size
    M_TILES = (CTA_ROWS + MFMA_MNK - 1) // MFMA_MNK
    ROWS_PADDED = M_TILES * MFMA_MNK
    # Avoid repeated BF16 max packing/unpacking.
    Q_ABSMAX_F32 = sliding_window > 0 and TUNED_PER_TOKEN
    # Stage each token's scales once rather than once per rgroup.
    UNIQUE_SCALE_STAGING = (
        sliding_window > 0
        and TUNED_PER_TOKEN
        and block_size == 128
        and trans_v
        and query_group_size == 16
        and query_length == 1
        and single_tile_plan == prefetch_v
    )
    SCALAR_FP8_DECODE = (
        TUNED_SCALAR and IS_BF16 and query_length == 1 and query_group_size in (8, 16)
    )
    # K128 consumes four K32 packs without changing cache/LDS layouts.
    WIDE_FP8_MFMA = (
        TUNED_PER_TOKEN and query_length in (3, 4) and query_group_size == 16
    )
    MFMA_K = 128 if WIDE_FP8_MFMA else FP8_PACK_K
    PACKS_PER_MFMA = MFMA_K // FP8_PACK_K
    MTP4_FUSED = WIDE_FP8_MFMA and M_TILES == 4
    # Wide page128 addresses make carrying V across QK too register-heavy.
    MTP4_PREFETCH_V = MTP4_FUSED and (block_size == 16 or not wide_kv_addressing)
    TUNE_PAGE128 = block_size == 128 and (PER_TOKEN_M1 or TUNED_SCALAR)
    PAGE16_VPIPE = prefetch_v and block_size == 16
    REUSE_KV_PAGES = PER_TOKEN_M1
    SCALES_BEFORE_CURRENT_V = (
        WIDE_FP8_MFMA and PER_TOKEN_M1 and (block_size == 16 or not trans_v)
    )
    # Scale before masking to avoid -inf * 0 and a second mask.
    M1_SCALE_BEFORE_MASK = REUSE_KV_PAGES or SCALAR_FP8_DECODE
    P_BUFFERS = M_TILES if MTP4_FUSED else 2 if TUNE_PAGE128 and M_TILES == 3 else 1
    # PV uses V=A, P=B: output [head-dim, query-row=lane16].
    NWARP = 4  # 4 waves / CTA
    TILE_TOK = KV_COMPUTE_BLOCK
    TOK_PER_WARP = TILE_TOK // NWARP
    assert TILE_TOK == NWARP * TOK_PER_WARP, "KV tile must split evenly across warps"
    assert (
        TOK_PER_WARP == NWARP * MFMA_MNK
    ), "per-warp token ownership must match the MFMA chunk layout"
    NCHUNK = TOK_PER_WARP // MFMA_MNK  # 4
    # A warp owns 64 tokens: four page-16s, one page-64, or half a page-128.
    PAGES_PER_CHUNK = (TOK_PER_WARP + block_size - 1) // block_size
    KV_EXTENT = (1 << 42) if wide_kv_addressing else (1 << 30)
    assert (
        head_dim % (NWARP * MFMA_MNK) == 0
    ), "head_dim must split across the 4 warps for PV"

    # Four 16-element QK loads form each 64-element fetch group.
    RGROUP_QUARTERS = 4
    QK_CHUNK_ELEMS = 16
    QKHE_LOOP = head_dim // (RGROUP_QUARTERS * QK_CHUNK_ELEMS)
    assert (
        QKHE_LOOP >= 1
    ), f"head_dim {head_dim} must be at least {RGROUP_QUARTERS * QK_CHUNK_ELEMS}"
    # QK operand-pack count, not the number of MFMA instructions.
    N_SUBCHUNKS = head_dim // FP8_PACK_K
    assert N_SUBCHUNKS % PACKS_PER_MFMA == 0, "QK packs must fill whole MFMA atoms"
    assert TILE_TOK % MFMA_K == 0, "PV tokens must fill whole MFMA atoms"

    # Absmax's lane16 butterfly fixes the Q chunk count at 16.
    NQCHUNK = 16
    QCHUNK = (
        head_dim // NQCHUNK
    )  # f16 elements per lane's load chunk (8 for head_dim=128, 4 for head_dim=64)
    # Loads are at most 128 bits; larger chunks must not leave an unloaded tail.
    assert QCHUNK <= 8 or QCHUNK % 8 == 0, (
        f"head_dim {head_dim} is unsupported: head_dim//{NQCHUNK} ({QCHUNK}) must "
        f"be <= 8 or a multiple of 8"
    )
    QLOAD_UNIT = min(8, QCHUNK)
    N_QLOADS = QCHUNK // QLOAD_UNIT

    VHE_CHUNKS = head_dim // (NWARP * MFMA_MNK)  # 2 for head_dim=128, 1 for head_dim=64
    VHE_SIZE = head_dim // VHE_CHUNKS
    OP_ELEMS = MFMA_ACC_ELEMS  # PV C-fragment elements/lane/chunk
    # Eight i64 packs/lane: eight K32 or two K128 PV instructions.
    NVOPS = TILE_TOK // FP8_PACK_K
    STEPS_PER_PAGE = block_size // MFMA_MNK
    STEPS_PER_CHUNK = min(block_size, TOK_PER_WARP) // MFMA_MNK

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    NP = int(num_partitions)  # context partitions (grid.z); compile-time constant
    DIRECT_SINKS = use_sinks and NP == 1 and not use_work_plan
    SINK_DTYPE = fx.Float32
    if DIRECT_SINKS:
        SINK_DTYPE = {
            "f32": fx.Float32,
            "f16": fx.Float16,
            "bf16": fx.BFloat16,
        }[sink_dtype_str]

    BLOCK_THREADS = NWARP * WAVE  # 256

    K_SLOT, V_SLOT = 0, 1
    # Per-M-tile loop-carried state after K/V: output chunks, max, and denom.
    STATE_PER_M = VHE_CHUNKS + 2
    V_DATA_SLOT = 2 + STATE_PER_M * M_TILES

    # LDS holds Q/P, cross-wave max/sum, V page IDs and per-token scales.
    # PV output and online-softmax state remain in registers.
    f32 = 4
    sQ_bytes = ROWS_PADDED * head_dim * 1  # fp8
    # First-QK's max barrier retires all sQ reads before P overwrites it.
    # Still-live Q scales stay outside the aliased, 16-byte-aligned Q/P regions.
    sP_off = 0
    # Padding avoids 32-bank conflicts while preserving ds_read_b128 alignment.
    SP_ROW_BYTES = TILE_TOK + 16
    sP_bytes = P_BUFFERS * MFMA_MNK * SP_ROW_BYTES  # fp8, padded rows
    sQscale_off = max(sQ_bytes, sP_bytes)
    sQscale_bytes = 0 if SCALAR_FP8_DECODE else ROWS_PADDED * f32
    # Tuned rows need 16-byte vector alignment; other paths use bank padding.
    NWARP_PAD = NWARP if TUNE_PAGE128 or PER_TOKEN_M1 or MTP4_FUSED else NWARP + 1
    # Phase-split slices sLmax per M-tile so all pass-1 writes share one barrier.
    sLmax_off = sQscale_off + sQscale_bytes
    sLsum_off = sLmax_off + M_TILES * MFMA_MNK * NWARP_PAD * f32
    sVPage_off = sLsum_off + P_BUFFERS * MFMA_MNK * NWARP_PAD * f32
    sVPage_bytes = NWARP * PAGES_PER_CHUNK * 4  # i32
    # Double buffering prevents next-tile prefetch from clobbering live scales.
    KV_BUF_STRIDE = 2 * NWARP * TOK_PER_WARP * f32  # k-region + v-region, one buffer
    KV_SCALE_BUFFERS = 1 if single_tile_plan else 2
    sKScale_off = sVPage_off + sVPage_bytes
    sVScale_off = sKScale_off + NWARP * TOK_PER_WARP * f32
    sKVScale_bytes = KV_SCALE_BUFFERS * KV_BUF_STRIDE if per_token_kv else 0
    sVScaleMax_off = sKScale_off + sKVScale_bytes
    sVScaleMax_bytes = (
        NWARP_PAD * f32 if per_token_kv else 0
    )  # m-independent: one cross-warp slot
    total_bytes = sVScaleMax_off + sVScaleMax_bytes

    # All typed LDS regions are 4-byte-aligned within this i32 blob.
    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 16]

    @flyc.jit
    def _pa_decode_tile_task(
        output_ptr: fx.Pointer,  # Direct static NP=1 output.
        # Static partials: [B,H,NP,rows]; planned: [H,capacity,rows].
        pmax_ptr: fx.Pointer,  # Natural-log row max.
        psum_ptr: fx.Pointer,  # Row sum.
        pout_ptr: fx.Pointer,  # Adds head_dim; Q_DTYPE normalized O_p/l_p.
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        context_lengths_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        sinks_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_o_row: fx.Int32,
        stride_o_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        num_sequences: fx.Int32,
        planned_seq: fx.Int32,
        planned_start: fx.Int32,
        planned_end: fx.Int32,
        planned_context: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // WAVE  # 0..NWARP-1
        lane = tid - warp * WAVE  # 0..63
        seq = fx.Int32(gpu.block_id("x"))
        kv_query = fx.Int32(gpu.block_id("y"))
        kv_h = kv_query // query_splits
        query_begin = (kv_query % query_splits) * QUERIES_PER_CTA
        part = fx.Int32(gpu.block_id("z"))  # context partition handled by this CTA
        n_kv = fx.Int32(gpu.grid_dim.y) // query_splits
        if const_expr(use_work_plan):
            if const_expr(batch_first_plan_grid):
                part = fx.Int32(
                    fx.Uint32(gpu.block_id("x")) * fx.Uint32(gpu.grid_dim.z)
                    + fx.Uint32(gpu.block_id("z"))
                )
                # Physical x groups packed slots, not sequences.
                seq = planned_seq
                capacity = fx.Int32(
                    fx.Uint32(gpu.grid_dim.x) * fx.Uint32(gpu.grid_dim.z)
                )
                partial_slot = kv_h * capacity + part
            else:
                part = seq  # Packed work slot, shared by all KV heads.
                seq = planned_seq
                partial_slot = kv_h * fx.Int32(gpu.grid_dim.x) + part
        else:
            partial_slot = (seq * n_kv + kv_h) * NP + part

        output = fx.recast_iter(Q_DTYPE, output_ptr)
        pmax = fx.recast_iter(fx.Float32, pmax_ptr)
        psum = fx.recast_iter(fx.Float32, psum_ptr)
        pout = fx.recast_iter(Q_DTYPE, pout_ptr)
        if const_expr(buffer_plan_output):
            # Widen before multiplying: packed storage can exceed 2 GiB.
            if const_expr(batch_first_plan_grid):
                pout_slot = fx.Int64(kv_h) * fx.Int64(gpu.grid_dim.x) * fx.Int64(
                    gpu.grid_dim.z
                ) + fx.Int64(part)
            else:
                pout_slot = fx.Int64(kv_h) * fx.Int64(gpu.grid_dim.x) + fx.Int64(part)
            pout_slot_bytes = TOTAL_ROWS * head_dim * 2
            pout_base = buf_base_i64(pout_ptr) + pout_slot * fx.Int64(pout_slot_bytes)
            pout_buffer = ptr_buf_tensor(
                pout_base,
                Q_DTYPE,
                n=TOTAL_ROWS * head_dim,
                unit_elems=OP_ELEMS,
                # BF16 views may be only 2-byte aligned, even for 8-byte stores.
                unit_stride=1,
                num_records_bytes=pout_slot_bytes,
            )
        if const_expr(DIRECT_SINKS):
            sink_token = fx.recast_iter(SINK_DTYPE, sinks_ptr)

        # K/V use raw UniversalCopy so their optional i64 offsets remain intact.
        def _make_raw_flat_loader(tensor_ptr, elem_ty, reg_width, extent):
            copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_ty)
            reg = fx.make_rmem_tensor(fx.make_layout(reg_width, 1), elem_ty)
            flat = fx.Tensor(
                fx.make_view(
                    fx.recast_iter(elem_ty, tensor_ptr), fx.make_layout(extent, 1)
                )
            )
            tiled = fx.logical_divide(flat, fx.make_layout(1, 1))

            def _load(elem_idx):
                fx.copy(copy_atom, fx.slice(tiled, (None, elem_idx)), reg)
                return fx.Vector(fx.memref_load_vec(reg))

            return _load

        _k_load_fp8x16 = _make_raw_flat_loader(key_cache_ptr, FP8, 16, KV_EXTENT)
        _v_load_fp8x16 = _make_raw_flat_loader(value_cache_ptr, FP8, 16, KV_EXTENT)

        def _kv_addr(phys, page_elems, rest):
            # Widen before the page product reaches 2^31 FP8 elements.
            if const_expr(wide_kv_addressing):
                return fx.Int64(phys) * fx.Int64(page_elems) + fx.Int64(rest)
            return phys * page_elems + rest

        _q_copy_op = (
            fx.rocdl.BufferCopy128b() if QLOAD_UNIT == 8 else fx.rocdl.BufferCopy64b()
        )
        q_buf = ptr_buf_tensor(query_ptr, Q_DTYPE)
        q_tiled = fx.logical_divide(q_buf, fx.make_layout(1, 1))
        q_copy_atom = fx.make_copy_atom(_q_copy_op, Q_DTYPE)
        q_reg = fx.make_rmem_tensor(fx.make_layout(QLOAD_UNIT, 1), Q_DTYPE)

        def _q_load_chunk(elem_idx):
            fx.copy(q_copy_atom, fx.slice(q_tiled, (None, elem_idx)), q_reg)
            return fx.Vector(fx.memref_load_vec(q_reg))

        rgroup = lane // MFMA_MNK  # 0..3: quarter-wave (paired with warp -> query row)
        lane16 = lane - rgroup * MFMA_MNK  # 0..15: this row's head-dim chunk index
        qh_local = warp * 4 + rgroup  # 0..15: this thread's query row within an M-tile

        def _load_q_row(qi, gs_head):
            qh0 = kv_h * query_group_size + gs_head
            row_byte0 = (
                (seq * query_length + query_begin + qi) * stride_q_row
                + qh0 * stride_q_head
            ) * 2  # 16-bit float = 2B/elem
            base_elem = (
                row_byte0 + lane16 * (QCHUNK * 2)
            ) // 2  # byte offset -> element index
            return [
                _q_load_chunk(base_elem + u * QLOAD_UNIT)
                for u in range_constexpr(N_QLOADS)
            ]

        # Issue independent Q loads ahead of K/page/scale prefetch.
        q_units_prefetched = None
        if const_expr(MTP4_FUSED):
            q_units_prefetched = []
            for m in range_constexpr(M_TILES):
                flat_idx = m * MFMA_MNK + qh_local
                qi = flat_idx // query_group_size
                gs_head = flat_idx - qi * query_group_size
                q_units_prefetched.append(_load_q_row(qi, gs_head))

        def _k_load16(byte_off):
            return _k_load_fp8x16(byte_off).bitcast(fx.Int64)

        def _v_load16(byte_off):
            return _v_load_fp8x16(byte_off).bitcast(fx.Int64)

        if const_expr(use_work_plan):
            context_len = planned_context
        else:
            ctx_buf = ptr_buf_tensor(context_lengths_ptr, fx.Int32)
            ctx_tiled = fx.logical_divide(ctx_buf, fx.make_layout(1, 1))
            ctx_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            ctx_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
            fx.copy(ctx_copy_atom, fx.slice(ctx_tiled, (None, seq)), ctx_reg)
            context_len = fx.Int32(fx.Vector(fx.memref_load_vec(ctx_reg))[0])
        # A partial compute tile may read past block_tables; bounded loads
        # return page 0 for those masked tail tokens instead of faulting.
        bt_num_records_bytes = (
            fx.Int64(num_sequences) * fx.Int64(max_blocks_per_seq) * 4
        )
        # Wide loads must preserve row starts that are only int32-aligned.
        bt_buf = ptr_buf_tensor(
            block_tables_ptr,
            fx.Int32,
            unit_elems=PAGES_PER_CHUNK,
            unit_stride=1,
            num_records_bytes=bt_num_records_bytes,
        )
        if const_expr(not per_token_kv):
            key_scale_buf = ptr_buf_tensor(key_scale_ptr, fx.Float32)
            value_scale_buf = ptr_buf_tensor(value_scale_ptr, fx.Float32)
            key_scale = fx.Float32(key_scale_buf[0])
            value_scale = fx.Float32(value_scale_buf[0])

        num_pages = (
            context_len + block_size - 1
        ) // block_size  # pages this sequence really owns
        if const_expr(use_work_plan):
            part_start = planned_start
            part_end = planned_end
        else:
            num_tiles = (context_len + TILE_TOK - 1) // TILE_TOK
            tiles_per_part = (num_tiles + NP - 1) // NP
            part_start = part * tiles_per_part
            part_end_raw = part_start + tiles_per_part
            part_end = (part_end_raw < num_tiles).select(part_end_raw, num_tiles)

        # Keep an ir.Value pointer for use inside scf control flow;
        # the Python SharedStorage handle cannot cross that boundary.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.recast_iter(fx.Uint8, lds.buf.ptr)  # byte-addressed base

        def _lds_ptr(byte_off, elem_ty):
            # Restore element alignment lost by the byte-base recast.
            p = fx.add_offset(lds_base, fx.make_int_tuple(byte_off))
            ptr_ty = fx.PointerType.get(
                elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty)
            )
            return fx.recast_iter(ptr_ty, p)

        def _lds_load(byte_off, elem_ty, n):
            return fx.ptr_load(
                _lds_ptr(byte_off, elem_ty), result_type=fx.Vector.make_type(n, elem_ty)
            )

        def _lds_store(byte_off, elem_ty, vec):
            fx.ptr_store(vec, _lds_ptr(byte_off, elem_ty))

        if const_expr(per_token_kv):
            scale_load_width = (
                NCHUNK if block_size >= 64 and not UNIQUE_SCALE_STAGING else 1
            )
            scale_copy_op = (
                fx.rocdl.BufferCopy32b()
                if scale_load_width == 1
                else fx.rocdl.BufferCopy128b()
            )
            scale_copy_atom = fx.make_copy_atom(scale_copy_op, fx.Float32)
            k_scale_buf = ptr_buf_tensor(key_scale_ptr, fx.Float32)
            v_scale_buf = ptr_buf_tensor(value_scale_ptr, fx.Float32)
            k_scale_tiled = fx.logical_divide(k_scale_buf, fx.make_layout(1, 1))
            v_scale_tiled = fx.logical_divide(v_scale_buf, fx.make_layout(1, 1))
            k_scale_reg = fx.make_rmem_tensor(
                fx.make_layout(scale_load_width, 1), fx.Float32
            )
            v_scale_reg = fx.make_rmem_tensor(
                fx.make_layout(scale_load_width, 1), fx.Float32
            )

            def _k_scale_load(elem_idx):
                fx.copy(
                    scale_copy_atom,
                    fx.slice(k_scale_tiled, (None, elem_idx)),
                    k_scale_reg,
                )
                return fx.Vector(fx.memref_load_vec(k_scale_reg))

            def _v_scale_load(elem_idx):
                fx.copy(
                    scale_copy_atom,
                    fx.slice(v_scale_tiled, (None, elem_idx)),
                    v_scale_reg,
                )
                return fx.Vector(fx.memref_load_vec(v_scale_reg))

        def _load_phys_scalar(page, vec_width=1):
            # Padding may contain stale page IDs. Force out-of-context pages
            # to block 0 even when the table read itself is in bounds.
            element_offset = seq * max_blocks_per_seq + page
            if const_expr(vec_width == 1):
                result = bt_buf[element_offset]
                return (page < num_pages).select(fx.Int32(result), fx.Int32(0))
            bt_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
            frag = fx.make_fragment_like(fx.slice(bt_buf, (0, None)))
            fx.copy(
                bt_copy_atom,
                fx.slice(bt_buf, (element_offset, None)),
                frag,
            )
            loaded = fx.Vector(fx.memref_load_vec(frag))
            return fx.Vector.from_elements(
                [
                    (page + i < num_pages).select(fx.Int32(loaded[i]), fx.Int32(0))
                    for i in range_constexpr(vec_width)
                ],
                dtype=fx.Int32,
            )

        def _stage_v_page_row(phys_vec):
            if lane == 0:
                _lds_store(
                    sVPage_off + warp * (PAGES_PER_CHUNK * 4), fx.Int32, phys_vec
                )

        def _v_page_fetch_and_stage(tt_i32):
            # Warp w broadcasts the page row later read by rgroup w.
            base_page = tt_i32 * TILE_TOK // block_size  # tile start is page-aligned
            fetched = _load_phys_scalar(
                base_page + (warp * TOK_PER_WARP) // block_size, PAGES_PER_CHUNK
            )
            fetched_vec = (
                fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
                if const_expr(PAGES_PER_CHUNK == 1)
                else fx.Vector(fetched)
            )
            _stage_v_page_row(fetched_vec)
            return fetched_vec

        def _v_page_read_row():
            off = sVPage_off + rgroup * (PAGES_PER_CHUNK * 4)
            return _lds_load(off, fx.Int32, PAGES_PER_CHUNK)

        def _k_page_read_warp():
            off = sVPage_off + warp * (PAGES_PER_CHUNK * 4)
            return _lds_load(off, fx.Int32, PAGES_PER_CHUNK)

        def _kv_buf_off(tt_val):
            if const_expr(per_token_kv and not single_tile_plan):
                return (tt_val & fx.Int32(1)) * KV_BUF_STRIDE
            return 0

        def _stage_kv_scale_to_lds(phys_vec, buf_off=0):
            if const_expr(block_size >= 64):
                phys = fx.Int32(phys_vec[0])
                chunk_tok = (
                    lane if const_expr(UNIQUE_SCALE_STAGING) else lane16 * NCHUNK
                )
                page_tok = (warp * TOK_PER_WARP) % block_size + chunk_tok
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + page_tok
                k_scale_vec = _k_scale_load(scale_idx)
                v_scale_vec = _v_scale_load(scale_idx)
                slot = (warp * TOK_PER_WARP + chunk_tok) * f32
                _lds_store(sKScale_off + buf_off + slot, fx.Float32, k_scale_vec)
                _lds_store(sVScale_off + buf_off + slot, fx.Float32, v_scale_vec)
            else:
                # Each rgroup stages its own page16 sub-block.
                phys = fx.Int32(fx.Vector(phys_vec)[rgroup])
                scale_idx = phys * stride_ks_block + kv_h * stride_ks_head + lane16
                k_scale_scalar = fx.Float32(_k_scale_load(scale_idx)[0])
                v_scale_scalar = fx.Float32(_v_scale_load(scale_idx)[0])
                fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                slot = (warp * TOK_PER_WARP + rgroup * MFMA_MNK + lane16) * f32
                _lds_store(
                    sKScale_off + buf_off + slot,
                    fx.Float32,
                    fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32),
                )
                _lds_store(
                    sVScale_off + buf_off + slot,
                    fx.Float32,
                    fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32),
                )

        def _load_scale_vec(base_off, a, buf_off=0):
            slot = (warp * TOK_PER_WARP + a * MFMA_MNK + rgroup * 4) * f32
            return _lds_load(base_off + buf_off + slot, fx.Float32, 4)

        def _load_kv_scale_vecs(a, buf_off=0):
            return _load_scale_vec(sKScale_off, a, buf_off), _load_scale_vec(
                sVScale_off, a, buf_off
            )

        def _mfma_fp8(a_ops, b_ops, a_base, b_base, k_packs, acc):
            # Counts and offsets are i64 operand packs, not MFMA instructions.
            if const_expr(WIDE_FP8_MFMA):
                for inst in range_constexpr(k_packs // PACKS_PER_MFMA):
                    a_pack = fx.Vector.from_elements(
                        [
                            a_ops[a_base + inst * PACKS_PER_MFMA + i]
                            for i in range_constexpr(PACKS_PER_MFMA)
                        ],
                        dtype=fx.Int64,
                    ).bitcast(fx.Int32)
                    b_pack = fx.Vector.from_elements(
                        [
                            b_ops[b_base + inst * PACKS_PER_MFMA + i]
                            for i in range_constexpr(PACKS_PER_MFMA)
                        ],
                        dtype=fx.Int64,
                    ).bitcast(fx.Int32)
                    acc = fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [
                            a_pack,
                            b_pack,
                            acc,
                            0,
                            0,
                            0,
                            fx.Int32(0x7F7F7F7F),
                            0,
                            fx.Int32(0x7F7F7F7F),
                        ],
                    )
            else:
                for pack in range_constexpr(k_packs):
                    acc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4,
                        [a_ops[a_base + pack], b_ops[b_base + pack], acc, 0, 0, 0],
                    )
            return acc

        # K token = warp*TOK_PER_WARP + a*MFMA_MNK + lane16;
        # softmax masks and P-pack writes must use the same mapping.
        def _k_ops(phys, a):
            within_page_tok = (warp * TOK_PER_WARP + a * MFMA_MNK + lane16) % block_size
            ops = []
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                base = _kv_addr(
                    phys,
                    n_kv * (QCHUNK * block_size * QK_CHUNK_ELEMS),
                    ((kv_h * QCHUNK + he_idx) * block_size + within_page_tok)
                    * QK_CHUNK_ELEMS,
                )
                w = _k_load16(base)  # head[he_idx*16 : +16] -> two K32 operand packs
                if const_expr(block_size == 16):
                    # Overlap page16 gathers.
                    fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                ops.extend([w[0], w[1]])
            return ops  # N_SUBCHUNKS i64 operands

        def _k_ops_from_phys(phys_vec):
            flat = []
            for a in range_constexpr(NCHUNK):
                phys = fx.Int32(phys_vec[(a * MFMA_MNK) // block_size])
                flat.extend(_k_ops(phys, a))
            if const_expr(head_dim == 64):
                fx.rocdl.sched_vmem(len(flat) // 2)

            return fx.Vector.from_elements(flat, dtype=fx.Int64)

        def _k_ops_flat(tt_i32):
            base_page = tt_i32 * TILE_TOK // block_size  # tile start is page-aligned
            fetched = _load_phys_scalar(
                base_page + (warp * TOK_PER_WARP) // block_size, PAGES_PER_CHUNK
            )
            phys_vec = (
                fx.Vector.from_elements([fx.Int32(fetched)], dtype=fx.Int32)
                if const_expr(PAGES_PER_CHUNK == 1)
                else fx.Vector(fetched)
            )
            return _k_ops_from_phys(phys_vec), phys_vec

        # Empty partitions must not read K/V or block_tables.
        k_pf0 = fx.Vector.filled(NCHUNK * N_SUBCHUNKS, 0, fx.Int64)
        if part_start < part_end:
            k_pf0, phys_vec0 = _k_ops_flat(part_start)
            if const_expr(REUSE_KV_PAGES):
                # Reuse K page IDs for V's LDS broadcast.
                _stage_v_page_row(phys_vec0)
            else:
                _v_page_fetch_and_stage(part_start)
            if const_expr(per_token_kv):
                _stage_kv_scale_to_lds(phys_vec0, _kv_buf_off(fx.Int32(part_start)))
        elif lane == 0:
            _lds_store(
                sVPage_off + warp * (PAGES_PER_CHUNK * 4),
                fx.Int32,
                fx.Vector.filled(PAGES_PER_CHUNK, 0, fx.Int32),
            )

        # Per-token scales fold into logits/P rather than scalar QK/epilogue scales.
        if const_expr(per_token_kv):
            scale_qk = fx.Float32(softmax_scale * LOG2E)
        else:
            scale_qk = fx.Float32(softmax_scale * LOG2E) * fx.Float32(key_scale)
            v_scale_f = fx.Float32(value_scale)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)
        # Finite or -inf scores permit nnan's bare max instructions.
        # Do not set ninf: -inf is the mask sentinel.
        fm_nnan = arith.FastMathFlags.nnan

        def _row_off(byte_off, m_idx, width, elem_ty):
            return byte_off + m_idx * (width * dsl_size_of(elem_ty))

        def _ld1(byte_off, m_idx):
            return _lds_load(_row_off(byte_off, m_idx, 1, fx.Float32), fx.Float32, 1)[0]

        def _st1(byte_off, m_idx, val):
            _lds_store(
                _row_off(byte_off, m_idx, 1, fx.Float32),
                fx.Float32,
                fx.Vector.from_elements([val], dtype=fx.Float32),
            )

        # Cross-wave scratch: scalar writes, NWARP-wide row reads.
        def _st_lw(base_off, row, w, val):
            off = base_off + (row * NWARP_PAD + w) * 4
            _lds_store(
                off, fx.Float32, fx.Vector.from_elements([val], dtype=fx.Float32)
            )

        def _ld_lw_row(base_off, row):
            off = base_off + row * (NWARP_PAD * 4)
            return _lds_load(off, fx.Float32, NWARP)

        def _f32_to_fp8_words(vf32):
            # Use hardware FP8 conversion; arith.truncf cannot lower it.
            n = vf32.shape[0]
            words = []
            for i in range_constexpr(n // 4):
                b = i * 4
                lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b], vf32[b + 1], 0, False)
                words.append(
                    fx.rocdl.cvt_pk_fp8_f32(T.i32, vf32[b + 2], vf32[b + 3], lo, True)
                )
            return fx.Vector.from_elements(words, dtype=fx.Int32)

        def _st_words(byte_off, words):
            _lds_store(byte_off, fx.Int32, words)

        def _q_local_absmax(q_unit):
            if const_expr(Q_ABSMAX_F32):
                return fmath.absf(q_unit.to(fx.Float32)).reduce(ReductionOp.MAX)
            else:
                return fmath.absf(q_unit).reduce(ReductionOp.MAX).to(fx.Float32)

        # M-tiles quantize disjoint rows, requiring no inter-tile barrier.
        def _quant_q_row(m, q_row_off, q_units):
            if const_expr(SCALAR_FP8_DECODE):
                for u in range_constexpr(N_QLOADS):
                    _st_words(
                        q_row_off
                        + qh_local * head_dim
                        + lane16 * QCHUNK
                        + u * QLOAD_UNIT,
                        _f32_to_fp8_words(q_units[u].to(fx.Float32)),
                    )
            else:
                absmax = _q_local_absmax(q_units[0])
                for u in range_constexpr(1, N_QLOADS):
                    absmax = fx.maxnumf(
                        absmax,
                        _q_local_absmax(q_units[u]),
                    )
                for sh in (8, 4, 2, 1):
                    absmax = fx.maxnumf(absmax, dpp_utils.dpp_xor_f32(absmax, sh))

                q_scale = absmax * fx.Float32(1.0 / FP8_MAX)
                inv = fx.Float32(rcp_f32(fx.maxnumf(q_scale, fx.Float32(1e-20))))
                inv_b = fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                    QLOAD_UNIT
                )

                for u in range_constexpr(N_QLOADS):
                    q_scaled_unit = q_units[u].to(fx.Float32) * inv_b
                    _st_words(
                        q_row_off
                        + qh_local * head_dim
                        + lane16 * QCHUNK
                        + u * QLOAD_UNIT,
                        _f32_to_fp8_words(q_scaled_unit),
                    )
                if lane16 == 0:
                    # Transposed [qh][m] enables one vector read across M-tiles.
                    _st1(sQscale_off, qh_local * M_TILES + m, q_scale)

        for m in range_constexpr(M_TILES):
            flat_idx = m * MFMA_MNK + qh_local
            qi = flat_idx // query_group_size
            gs_head = flat_idx - qi * query_group_size
            q_row_off = m * MFMA_MNK * head_dim
            # Only the final M-tile can need a runtime row guard.
            if const_expr((m + 1) * MFMA_MNK <= CTA_ROWS):
                q_units = (
                    q_units_prefetched[m]
                    if const_expr(MTP4_FUSED)
                    else _load_q_row(qi, gs_head)
                )
                _quant_q_row(m, q_row_off, q_units)
            elif flat_idx < CTA_ROWS:
                _quant_q_row(m, q_row_off, _load_q_row(qi, gs_head))
            else:
                _st_words(
                    q_row_off + qh_local * head_dim + lane16 * QCHUNK,
                    fx.Vector.filled(QCHUNK // 4, 0, fx.Int32),
                )
                if const_expr(not SCALAR_FP8_DECODE) and lane16 == 0:
                    _st1(sQscale_off, qh_local * M_TILES + m, ZERO_F)

        gpu.barrier()

        v_page_pf0 = _v_page_read_row()

        # Register-resident Q (B operand) must match _k_ops' head-dim permutation.
        q_ops_all = []
        for m in range_constexpr(M_TILES):
            q_row_off = m * MFMA_MNK * head_dim
            for qkhe in range_constexpr(QKHE_LOOP):
                he_idx = qkhe * RGROUP_QUARTERS + rgroup
                chunk = _lds_load(
                    q_row_off + lane16 * head_dim + he_idx * QK_CHUNK_ELEMS, fx.Int64, 2
                )
                q_ops_all.extend([chunk[0], chunk[1]])
        _ct = [
            fx.Vector.from_elements(
                [float(a * MFMA_MNK + r) for r in range_constexpr(4)]
            )
            for a in range_constexpr(NCHUNK)
        ]

        def _score_mask(a, upper, lower):
            valid = _ct[a] < upper
            if const_expr(sliding_window > 0):
                valid = valid & (_ct[a] >= lower)
            return valid

        # Load one 16-token FP8 vector per head element; trans_v selects offsets.
        def _v_ops(phys_row, vh):
            head_group = ((vh * VHE_SIZE) // 16) + warp
            head_element = head_group * 16 + lane16
            ops = []
            for sub in range_constexpr(PAGES_PER_CHUNK):
                for step in range_constexpr(STEPS_PER_CHUNK):
                    # PV's token chunk is owned by rgroup after the LDS transpose.
                    page_step = ((rgroup * TOK_PER_WARP) % block_size) // 16 + step
                    if const_expr(trans_v):
                        base = _kv_addr(
                            phys_row[sub],
                            n_kv * (STEPS_PER_PAGE * head_dim * 16),
                            (
                                (kv_h * STEPS_PER_PAGE + page_step) * head_dim
                                + head_element
                            )
                            * 16,
                        )
                    else:
                        base = _kv_addr(
                            phys_row[sub],
                            n_kv * (head_dim * block_size),
                            (kv_h * head_dim + head_element) * block_size
                            + page_step * 16,
                        )
                    w = _v_load16(base)
                    if const_expr(block_size == 16):
                        fx.rocdl.sched_barrier(fx.rocdl.mask_vmem_rd)
                    ops.extend([w[0], w[1]])
            if const_expr(head_dim == 64):
                fx.rocdl.sched_vmem(len(ops) // 2)
            return ops  # NVOPS i64, the 64-token contiguous run for this head

        # Distance to the newest query keeps causal bounds tile-relative.
        if const_expr(QUERIES_PER_CTA == 1):
            causal_offset = [
                query_length - 1 - query_begin for _m in range_constexpr(M_TILES)
            ]
        else:
            causal_offset = [
                query_length
                - 1
                - query_begin
                - (m * MFMA_MNK + lane16) // query_group_size
                for m in range_constexpr(M_TILES)
            ]
            if const_expr(CTA_ROWS % MFMA_MNK != 0):
                # Padded rows still need nonnegative causal offsets.
                causal_offset = [
                    (offset > 0).select(offset, 0) for offset in causal_offset
                ]

        def _o_slot(m, vh):
            return 2 + STATE_PER_M * m + vh

        def _m_slot(m):
            return 2 + STATE_PER_M * m + VHE_CHUNKS

        def _l_slot(m):
            return 2 + STATE_PER_M * m + VHE_CHUNKS + 1

        o_zero = fx.Vector.filled(OP_ELEMS, 0.0, fx.Float32)
        init_state = [k_pf0, v_page_pf0]
        for _m in range_constexpr(M_TILES):
            init_state.extend([o_zero] * VHE_CHUNKS + [NEG_INF, ZERO_F])
        # Reuse carried V registers only after the final query consumes a chunk.
        if const_expr(MTP4_PREFETCH_V):
            v_pf0 = fx.Vector.filled(VHE_CHUNKS * NVOPS, 0, fx.Int64)
            if part_start < part_end:
                v_flat0 = []
                for vh in range_constexpr(VHE_CHUNKS):
                    v_flat0.extend(_v_ops(v_page_pf0, vh))
                v_pf0 = fx.Vector.from_elements(v_flat0, dtype=fx.Int64)
            init_state.append(v_pf0)
        # Single-tile plans eliminate loop/history but keep absolute tt for
        # addressing and masks; the outer guard excludes padded tasks.
        loop_start = 0 if const_expr(single_tile_plan) else part_start
        loop_end = 1 if const_expr(single_tile_plan) else part_end
        for loop_i, ostate in range(loop_start, loop_end, 1, init=init_state):
            k_cur = ostate[
                K_SLOT
            ]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
            v_page_cur = ostate[
                V_SLOT
            ]  # this tile's V pages, as one PAGES_PER_CHUNK-wide i32 vector
            tt = (
                fx.Int32(part_start)
                if const_expr(single_tile_plan)
                else fx.Int32(loop_i)
            )
            tok0 = tt * TILE_TOK
            # Interleave MFMA with VALU/LDS or page16 V loads.
            if const_expr((not per_token_kv and M_TILES > 1) or PAGE16_VPIPE):
                fx.rocdl.iglp_opt(0)

            tt1 = tt + 1

            next_state = [
                None,
                None,
            ]  # slots 0/1 (K_SLOT/V_SLOT) filled in at m==0 below

            cur_kv_buf = _kv_buf_off(tt)

            # Exclude unwritten-tail scales from FP8 normalization using the
            # context bound, not a per-query causal bound, for shared MTP scales.
            tile_valid = context_len - tok0
            window_left = None
            if const_expr(sliding_window > 0):
                # Clamp before subtracting MTP offsets to avoid int32 underflow;
                # negative left edges exclude no tile-relative token.
                window_left = tile_valid - sliding_window
                window_left = (window_left > 0).select(window_left, 0)

            if const_expr(per_token_kv):
                ctx_thr = fx.Vector.from_elements(
                    [
                        tile_valid.to(fx.Float32)
                        - fx.Int32(warp * TOK_PER_WARP + rgroup * 4).to(fx.Float32)
                    ],
                    dtype=fx.Float32,
                ).broadcast_to(4)
                window_scale_thr = None
                if const_expr(sliding_window > 0):
                    # Only the MTP window union may set the shared FP8 scale;
                    # invisible large scales could underflow visible probabilities.
                    first_visible = (window_left - (query_length - 1)).to(fx.Float32)
                    window_scale_thr = fx.Vector.from_elements(
                        [
                            first_visible
                            - fx.Int32(warp * TOK_PER_WARP + rgroup * 4).to(fx.Float32)
                        ],
                        dtype=fx.Float32,
                    ).broadcast_to(4)
                zero4_scale = fx.Vector.filled(4, 0.0, fx.Float32)

                def _mask_v_scale(
                    vec, a, thr=ctx_thr, lower=window_scale_thr, zero=zero4_scale
                ):
                    return _score_mask(a, thr, lower).select(vec, zero)

            # Independent V loads overlap QK/softmax and are shared across M-tiles.
            v_vh_shared = None
            if const_expr(MTP4_PREFETCH_V):
                v_carried = ostate[V_DATA_SLOT]
                v_vh_shared = [
                    [v_carried[vh * NVOPS + i] for i in range_constexpr(NVOPS)]
                    for vh in range_constexpr(VHE_CHUNKS)
                ]
            elif const_expr(
                (M_TILES > 1 and not MTP4_FUSED)
                or (prefetch_v and not SCALES_BEFORE_CURRENT_V)
            ):
                v_vh_shared = [
                    _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                ]

            q_scale_vec = None
            if const_expr(M_TILES > 1):
                q_scale_vec = _lds_load(
                    sQscale_off + lane16 * (M_TILES * f32), fx.Float32, M_TILES
                )

            def _lmax_off_m(m):
                return sLmax_off + m * MFMA_MNK * NWARP_PAD * f32

            # Phase A publishes all M-tiles' QK maxima with one barrier.
            if const_expr(M_TILES > 1):
                masked_chunks_saved = [None] * M_TILES

                # Reread V scales after Phase A to reduce peak register liveness.
                k_scale_shared = None
                if const_expr(per_token_kv):
                    v_scale_A = [
                        _load_scale_vec(sVScale_off, a, cur_kv_buf)
                        for a in range_constexpr(NCHUNK)
                    ]
                    pv_max = fx.Float32(0.0)
                    for a in range_constexpr(NCHUNK):
                        pv_max = fx.maxnumf(
                            pv_max,
                            _mask_v_scale(v_scale_A[a], a).reduce(ReductionOp.MAX),
                        )
                    for sh in (16, 32):
                        pv_max = fx.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
                    _st_lw(sVScaleMax_off, 0, warp, pv_max)
                    k_scale_shared = [
                        _load_scale_vec(sKScale_off, a, cur_kv_buf)
                        for a in range_constexpr(NCHUNK)
                    ]

                for m in range_constexpr(M_TILES):
                    frag_Ss = []
                    for a in range_constexpr(NCHUNK):
                        acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
                        acc = _mfma_fp8(
                            k_cur,
                            q_ops_all,
                            a * N_SUBCHUNKS,
                            m * N_SUBCHUNKS,
                            N_SUBCHUNKS,
                            acc,
                        )
                        frag_Ss.append(fx.Vector(acc))

                    scale = scale_qk * fx.Float32(q_scale_vec[m])
                    n_valid_tile = (tile_valid - causal_offset[m]).to(fx.Float32)
                    base_tok_f = fx.Int32(warp * TOK_PER_WARP + rgroup * 4).to(
                        fx.Float32
                    )
                    thr = fx.Vector.from_elements(
                        [n_valid_tile - base_tok_f], dtype=fx.Float32
                    ).broadcast_to(4)
                    window_thr = None
                    if const_expr(sliding_window > 0):
                        # Subtract before converting to avoid rounding large edges.
                        first_valid_tile = (window_left - causal_offset[m]).to(
                            fx.Float32
                        )
                        window_thr = fx.Vector.from_elements(
                            [first_valid_tile - base_tok_f], dtype=fx.Float32
                        ).broadcast_to(4)
                    neg4 = fx.Vector.filled(4, float("-inf"), fx.Float32)

                    # Scale finite logits before selecting -inf to avoid 0 * -inf
                    # for zero-Q rows. The stored row max is already scaled.
                    scale_b = fx.Vector.from_elements(
                        [scale], dtype=fx.Float32
                    ).broadcast_to(4)
                    if const_expr(per_token_kv):
                        scaled_frags = [
                            frag_Ss[a] * k_scale_shared[a] * scale_b
                            for a in range_constexpr(NCHUNK)
                        ]
                    else:
                        scaled_frags = [
                            frag_Ss[a] * scale_b for a in range_constexpr(NCHUNK)
                        ]

                    if const_expr(MTP4_FUSED):
                        # Interior tiles need no mask for any of the four queries.
                        masked_all = fx.Vector.from_elements(
                            [
                                scaled_frags[a][r]
                                for a in range_constexpr(NCHUNK)
                                for r in range_constexpr(MFMA_ACC_ELEMS)
                            ],
                            dtype=fx.Float32,
                        )
                        needs_mask = tile_valid < TILE_TOK + query_length - 1
                        if const_expr(sliding_window > 0):
                            needs_mask = needs_mask | (window_left > 0)
                        if needs_mask:
                            masked_all = fx.Vector.from_elements(
                                [
                                    _score_mask(a, thr, window_thr).select(
                                        scaled_frags[a], neg4
                                    )[r]
                                    for a in range_constexpr(NCHUNK)
                                    for r in range_constexpr(MFMA_ACC_ELEMS)
                                ],
                                dtype=fx.Float32,
                            )
                        masked_chunks = [
                            fx.Vector.from_elements(
                                [
                                    masked_all[a * MFMA_ACC_ELEMS + r]
                                    for r in range_constexpr(MFMA_ACC_ELEMS)
                                ],
                                dtype=fx.Float32,
                            )
                            for a in range_constexpr(NCHUNK)
                        ]
                    else:
                        masked_chunks = [
                            _score_mask(a, thr, window_thr).select(
                                scaled_frags[a], neg4
                            )
                            for a in range_constexpr(NCHUNK)
                        ]

                    pm = fx.Float32(float("-inf"))
                    for a in range_constexpr(NCHUNK):
                        pm = fx.maxnumf(
                            pm,
                            masked_chunks[a].reduce(ReductionOp.MAX, fastmath=fm_nnan),
                            fastmath=fm_nnan,
                        )
                    for sh in (16, 32):
                        pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE), fastmath=fm_nnan)
                    _st_lw(_lmax_off_m(m), lane16, warp, pm)

                    masked_chunks_saved[m] = masked_chunks

                # Publish next pages/scales with the Phase A barrier.
                k_next = (
                    fx.Vector.filled(NCHUNK * N_SUBCHUNKS, 0, fx.Int64)
                    if const_expr(MTP4_FUSED)
                    else k_cur
                )
                if const_expr(not single_tile_plan) and tt1 < part_end:
                    if const_expr(MTP4_FUSED):
                        # Defer K until P packing releases saved score registers.
                        phys_vec1 = _v_page_fetch_and_stage(tt1)
                        _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                    else:
                        k_next, phys_vec1 = _k_ops_flat(tt1)
                        _v_page_fetch_and_stage(tt1)
                        if const_expr(per_token_kv):
                            _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                next_state[K_SLOT] = k_next

                gpu.barrier()

                v_page_next = v_page_cur
                if const_expr(not single_tile_plan) and tt1 < part_end:
                    v_page_next = _v_page_read_row()
                next_state[V_SLOT] = v_page_next

                # Large M-tile groups reread scale chunks to bound VGPR liveness.
                v_scale_shared = None
                if const_expr(per_token_kv and M_TILES < 4):
                    v_scale_shared = [
                        _load_scale_vec(sVScale_off, a, cur_kv_buf)
                        for a in range_constexpr(NCHUNK)
                    ]

                if const_expr(MTP4_FUSED):
                    if const_expr(not MTP4_PREFETCH_V and not trans_v):
                        # Hide plain V's strided loads behind P packing.
                        v_vh_shared = [
                            _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                        ]
                    # V-scale normalization is shared by all query rows.
                    v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(ReductionOp.MAX)
                    v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                    v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                    norm_factor = fx.Float32(rcp_f32(v_max_safe))
                    norm_factor_b = fx.Vector.from_elements(
                        [norm_factor], dtype=fx.Float32
                    ).broadcast_to(4)

                    # Private P/Lsum slots share one barrier without overwrite races.
                    m_new_saved = []
                    safe_max_saved = []
                    for m in range_constexpr(M_TILES):
                        tile_max = _ld_lw_row(_lmax_off_m(m), lane16).reduce(
                            ReductionOp.MAX, fastmath=fm_nnan
                        )
                        m_new = (
                            tile_max
                            if const_expr(single_tile_plan)
                            else fx.maxnumf(
                                ostate[_m_slot(m)], tile_max, fastmath=fm_nnan
                            )
                        )
                        m_new_saved.append(m_new)
                        safe_max = (m_new > NEG_INF).select(m_new, ZERO_F)
                        safe_max_saved.append(
                            fx.Vector.from_elements(
                                [safe_max], dtype=fx.Float32
                            ).broadcast_to(4)
                        )
                    ls_saved = [ZERO_F for _ in range_constexpr(M_TILES)]
                    # Share one scale fragment across queries, not the full scale tile.
                    for a in range_constexpr(NCHUNK):
                        v_sc = _load_scale_vec(sVScale_off, a, cur_kv_buf)
                        for m in range_constexpr(M_TILES):
                            Pa = fx.Vector(
                                fx.exp2(
                                    masked_chunks_saved[m][a] - safe_max_saved[m],
                                    fastmath="fast",
                                )
                            )
                            ls_saved[m] = ls_saved[m] + Pa.reduce(ReductionOp.ADD)
                            p_scaled = Pa * v_sc * norm_factor_b
                            word = _f32_to_fp8_words(p_scaled)[0]
                            p_off = (
                                sP_off
                                + m * MFMA_MNK * SP_ROW_BYTES
                                + lane16 * SP_ROW_BYTES
                                + warp * TOK_PER_WARP
                                + rgroup * 4
                                + a * (MFMA_MNK // 4) * f32
                            )
                            _lds_store(
                                p_off,
                                fx.Int32,
                                fx.Vector.from_elements([word], dtype=fx.Int32),
                            )
                    for m in range_constexpr(M_TILES):
                        ls = ls_saved[m]
                        for sh in (16, 32):
                            ls = ls + ls.shuffle_xor(sh, WAVE)
                        if rgroup == 0:
                            _st_lw(
                                sLsum_off + m * MFMA_MNK * NWARP_PAD * f32,
                                lane16,
                                warp,
                                ls,
                            )
                    if const_expr(not single_tile_plan) and tt1 < part_end:
                        # Saved scores are dead; next K can overlap the P barrier/PV.
                        k_next = _k_ops_from_phys(_k_page_read_warp())
                    next_state[K_SLOT] = k_next

                    gpu.barrier()

                    if const_expr(not MTP4_PREFETCH_V and trans_v):
                        # Delay transposed V until saved score registers are dead.
                        v_vh_shared = [
                            _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                        ]
                    v_next_chunks = []
                    for m in range_constexpr(M_TILES):
                        p_base = sP_off + m * MFMA_MNK * SP_ROW_BYTES
                        lsum_base = sLsum_off + m * MFMA_MNK * NWARP_PAD * f32
                        o_acc = [
                            ostate[_o_slot(m, vh)] for vh in range_constexpr(VHE_CHUNKS)
                        ]
                        m_prev = ostate[_m_slot(m)]
                        l_prev = ostate[_l_slot(m)]
                        m_new = m_new_saved[m]
                        safe_max = (m_new > NEG_INF).select(m_new, ZERO_F)
                        corr_reg = (
                            ZERO_F
                            if const_expr(single_tile_plan)
                            else fx.Float32(fx.exp2(m_prev - safe_max, fastmath="fast"))
                        )
                        gsum = _ld_lw_row(lsum_base, lane16).reduce(ReductionOp.ADD)
                        l_new = (
                            gsum
                            if const_expr(single_tile_plan)
                            else l_prev * corr_reg + gsum
                        )
                        p_ops = _lds_load(
                            p_base + lane16 * SP_ROW_BYTES + rgroup * 64,
                            fx.Int64,
                            NVOPS,
                        )
                        corr_b = fx.Vector.from_elements(
                            [corr_reg], dtype=fx.Float32
                        ).broadcast_to(OP_ELEMS)
                        for vh in range_constexpr(VHE_CHUNKS):
                            v_vh = v_vh_shared[vh]
                            acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
                            acc = _mfma_fp8(v_vh, p_ops, 0, 0, NVOPS, acc)
                            op = fx.Vector(acc) * fx.Vector.from_elements(
                                [v_max_scaled], dtype=fx.Float32
                            ).broadcast_to(OP_ELEMS)
                            o_acc[vh] = (
                                op
                                if const_expr(single_tile_plan)
                                else o_acc[vh] * corr_b + op
                            )
                            if const_expr(
                                MTP4_PREFETCH_V
                                and m == M_TILES - 1
                                and not single_tile_plan
                            ):
                                # Reuse the dead V chunk while the final PV finishes.
                                v_next_chunk = fx.Vector.filled(NVOPS, 0, fx.Int64)
                                if const_expr(not single_tile_plan) and tt1 < part_end:
                                    v_next_chunk = fx.Vector.from_elements(
                                        _v_ops(v_page_next, vh), dtype=fx.Int64
                                    )
                                v_next_chunks.append(v_next_chunk)
                        next_state.extend([*o_acc, m_new, l_new])
                        if const_expr(m < M_TILES - 1):
                            fx.rocdl.sched_barrier(0)
                    if const_expr(MTP4_PREFETCH_V):
                        if const_expr(single_tile_plan):
                            # Preserve the unused next-V slot in the loop state.
                            v_next = ostate[V_DATA_SLOT]
                        else:
                            v_next = fx.Vector.from_elements(
                                [
                                    v_next_chunks[vh][i]
                                    for vh in range_constexpr(VHE_CHUNKS)
                                    for i in range_constexpr(NVOPS)
                                ],
                                dtype=fx.Int64,
                            )
                else:
                    for m in range_constexpr(M_TILES):
                        p_base = sP_off + (m % P_BUFFERS) * MFMA_MNK * SP_ROW_BYTES
                        lsum_base = (
                            sLsum_off + (m % P_BUFFERS) * MFMA_MNK * NWARP_PAD * f32
                        )
                        o_acc = [
                            ostate[_o_slot(m, vh)] for vh in range_constexpr(VHE_CHUNKS)
                        ]
                        m_prev = ostate[
                            _m_slot(m)
                        ]  # this thread's own running max, carried from last tile
                        l_prev = ostate[
                            _l_slot(m)
                        ]  # this thread's own running denom, carried from last tile

                        masked_chunks = masked_chunks_saved[m]

                        v_max_scaled = None
                        norm_factor_b = None
                        if const_expr(per_token_kv):
                            v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(
                                ReductionOp.MAX
                            )
                            v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                            v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                            norm_factor = fx.Float32(rcp_f32(v_max_safe))
                            norm_factor_b = fx.Vector.from_elements(
                                [norm_factor], dtype=fx.Float32
                            ).broadcast_to(4)

                        tile_max = _ld_lw_row(_lmax_off_m(m), lane16).reduce(
                            ReductionOp.MAX, fastmath=fm_nnan
                        )
                        m_new = (
                            tile_max
                            if const_expr(single_tile_plan)
                            else fx.maxnumf(m_prev, tile_max, fastmath=fm_nnan)
                        )
                        # Empty rows need exponent reference 0 to avoid -inf-(-inf).
                        safe_max = (m_new > NEG_INF).select(m_new, ZERO_F)
                        m_new_b = fx.Vector.from_elements(
                            [safe_max], dtype=fx.Float32
                        ).broadcast_to(4)
                        ls = fx.Float32(0.0)
                        words = []
                        for a in range_constexpr(NCHUNK):
                            Pa = fx.Vector(
                                fx.exp2(masked_chunks[a] - m_new_b, fastmath="fast")
                            )
                            ls = ls + Pa.reduce(ReductionOp.ADD)
                            if const_expr(per_token_kv):
                                v_sc = (
                                    _load_scale_vec(sVScale_off, a, cur_kv_buf)
                                    if const_expr(M_TILES >= 4)
                                    else v_scale_shared[a]
                                )
                                p_scaled = Pa * v_sc * norm_factor_b
                            else:
                                p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                            words.append(_f32_to_fp8_words(p_scaled)[0])

                        p_off0 = (
                            p_base
                            + lane16 * SP_ROW_BYTES
                            + warp * TOK_PER_WARP
                            + rgroup * 4
                        )
                        # Scatter P words in the interleave expected by PV reads.
                        for a in range_constexpr(NCHUNK):
                            _lds_store(
                                p_off0 + a * (MFMA_MNK // 4) * f32,
                                fx.Int32,
                                fx.Vector.from_elements([words[a]], dtype=fx.Int32),
                            )
                        for sh in (16, 32):
                            ls = ls + ls.shuffle_xor(sh, WAVE)
                        # Empty history must contribute exp2(-inf-safe_max)=0;
                        # replacing m_prev with 0 can overflow for negative logits.
                        corr_reg = (
                            ZERO_F
                            if const_expr(single_tile_plan)
                            else fx.Float32(fx.exp2(m_prev - safe_max, fastmath="fast"))
                        )
                        if rgroup == 0:
                            _st_lw(lsum_base, lane16, warp, ls)
                        gpu.barrier()
                        gsum = _ld_lw_row(lsum_base, lane16).reduce(ReductionOp.ADD)
                        l_new = (
                            gsum
                            if const_expr(single_tile_plan)
                            else l_prev * corr_reg + gsum
                        )

                        p_ops = _lds_load(
                            p_base + lane16 * SP_ROW_BYTES + rgroup * 64,
                            fx.Int64,
                            NVOPS,
                        )

                        corr_b = fx.Vector.from_elements(
                            [corr_reg], dtype=fx.Float32
                        ).broadcast_to(OP_ELEMS)
                        for vh in range_constexpr(VHE_CHUNKS):
                            v_vh = v_vh_shared[vh]
                            acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
                            acc = _mfma_fp8(v_vh, p_ops, 0, 0, NVOPS, acc)
                            op = fx.Vector(acc)
                            if const_expr(per_token_kv):
                                op = op * fx.Vector.from_elements(
                                    [v_max_scaled], dtype=fx.Float32
                                ).broadcast_to(OP_ELEMS)
                            o_acc[vh] = (
                                op
                                if const_expr(single_tile_plan)
                                else o_acc[vh] * corr_b + op
                            )
                        next_state.extend([*o_acc, m_new, l_new])
                        # Single P/Lsum slots need a read-retirement barrier.
                        # Alternating slots use the next write barrier; Phase A
                        # protects loop boundaries. The fence bounds live registers.
                        if const_expr(m < M_TILES - 1):
                            if const_expr(P_BUFFERS == 1):
                                gpu.barrier()
                            fx.rocdl.sched_barrier(0)
            else:
                o_acc = [ostate[_o_slot(0, vh)] for vh in range_constexpr(VHE_CHUNKS)]
                m_prev = ostate[_m_slot(0)]  # running max, carried from last tile
                l_prev = ostate[_l_slot(0)]  # running denom, carried from last tile
                frag_Ss = []
                for a in range_constexpr(NCHUNK):
                    acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
                    acc = _mfma_fp8(
                        k_cur, q_ops_all, a * N_SUBCHUNKS, 0, N_SUBCHUNKS, acc
                    )
                    frag_Ss.append(fx.Vector(acc))
                k_next = k_cur
                if const_expr(not single_tile_plan) and tt1 < part_end:
                    if const_expr(REUSE_KV_PAGES):
                        # Stage scales before K so scale waits cannot drain K;
                        # page16 defers K until the P barrier.
                        phys_vec1 = _v_page_fetch_and_stage(tt1)
                        if const_expr(per_token_kv):
                            _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                        if const_expr(not PAGE16_VPIPE):
                            k_next = _k_ops_from_phys(phys_vec1)
                    else:
                        k_next, phys_vec1 = _k_ops_flat(tt1)
                        _v_page_fetch_and_stage(tt1)
                        if const_expr(per_token_kv):
                            _stage_kv_scale_to_lds(phys_vec1, _kv_buf_off(tt1))
                if const_expr(SCALES_BEFORE_CURRENT_V):
                    # Issue V after scale staging so scale waits do not drain V.
                    v_vh_shared = [
                        _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                    ]
                scale = (
                    scale_qk
                    if const_expr(SCALAR_FP8_DECODE)
                    else scale_qk * _ld1(sQscale_off, lane16)
                )  # per-qhead positive score scale
                n_valid_tile = (tile_valid - causal_offset[0]).to(fx.Float32)
                base_tok_f = fx.Int32(warp * TOK_PER_WARP + rgroup * 4).to(fx.Float32)
                thr = fx.Vector.from_elements(
                    [n_valid_tile - base_tok_f], dtype=fx.Float32
                ).broadcast_to(4)
                window_thr = None
                if const_expr(sliding_window > 0):
                    first_valid_tile = (window_left - causal_offset[0]).to(fx.Float32)
                    window_thr = fx.Vector.from_elements(
                        [first_valid_tile - base_tok_f], dtype=fx.Float32
                    ).broadcast_to(4)
                neg4 = fx.Vector.filled(
                    4,
                    float("-inf") if M1_SCALE_BEFORE_MASK else -1e30,
                    fx.Float32,
                )
                # As in Phase A, scale before -inf masking to avoid zero-Q NaNs.
                scale_b = None
                if const_expr(M1_SCALE_BEFORE_MASK):
                    scale_b = fx.Vector.from_elements(
                        [scale], dtype=fx.Float32
                    ).broadcast_to(4)
                v_scale_vecs = None
                if const_expr(per_token_kv):
                    v_scale_vecs = []
                    scaled_frags = []
                    masked_chunks = []
                    for a in range_constexpr(NCHUNK):
                        k_scale_vec, v_scale_vec = _load_kv_scale_vecs(a, cur_kv_buf)
                        v_scale_vecs.append(v_scale_vec)
                        scaled_frag = frag_Ss[a] * k_scale_vec
                        if const_expr(M1_SCALE_BEFORE_MASK):
                            scaled_frag = scaled_frag * scale_b
                            masked_chunks.append(
                                _score_mask(a, thr, window_thr).select(
                                    scaled_frag, neg4
                                )
                            )
                        else:
                            scaled_frags.append(scaled_frag)
                else:
                    if const_expr(M1_SCALE_BEFORE_MASK):
                        masked_chunks = [
                            _score_mask(a, thr, window_thr).select(
                                frag_Ss[a] * scale_b, neg4
                            )
                            for a in range_constexpr(NCHUNK)
                        ]
                    else:
                        scaled_frags = frag_Ss
                if const_expr(not M1_SCALE_BEFORE_MASK):
                    masked_chunks = [
                        _score_mask(a, thr, window_thr).select(scaled_frags[a], neg4)
                        for a in range_constexpr(NCHUNK)
                    ]
                # pass 1: per-warp max for this qhead
                pm = fx.Float32(float("-inf"))
                for a in range_constexpr(NCHUNK):
                    if const_expr(M1_SCALE_BEFORE_MASK):
                        pm = fx.maxnumf(
                            pm,
                            masked_chunks[a].reduce(ReductionOp.MAX, fastmath=fm_nnan),
                            fastmath=fm_nnan,
                        )
                    else:
                        pm = fx.maxnumf(pm, masked_chunks[a].reduce(ReductionOp.MAX))
                for sh in (16, 32):
                    if const_expr(M1_SCALE_BEFORE_MASK):
                        pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE), fastmath=fm_nnan)
                    else:
                        pm = fx.maxnumf(pm, pm.shuffle_xor(sh, WAVE))
                _st_lw(
                    sLmax_off,
                    lane16,
                    warp,
                    pm if const_expr(M1_SCALE_BEFORE_MASK) else pm * scale,
                )  # redundant across the 4 lanes sharing this qhead
                if const_expr(per_token_kv):
                    pv_max = fx.Float32(0.0)
                    for a in range_constexpr(NCHUNK):
                        pv_max = fx.maxnumf(
                            pv_max,
                            _mask_v_scale(v_scale_vecs[a], a).reduce(ReductionOp.MAX),
                        )
                    for sh in (16, 32):
                        pv_max = fx.maxnumf(pv_max, pv_max.shuffle_xor(sh, WAVE))
                    _st_lw(sVScaleMax_off, 0, warp, pv_max)
                gpu.barrier()
                v_page_next = v_page_cur
                if const_expr(not single_tile_plan) and tt1 < part_end:
                    v_page_next = _v_page_read_row()
                next_state[V_SLOT] = v_page_next
                v_max_scaled = None
                norm_factor_b = None
                if const_expr(per_token_kv):
                    v_max_global = _ld_lw_row(sVScaleMax_off, 0).reduce(ReductionOp.MAX)
                    v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX)
                    v_max_safe = v_max_scaled + fx.Float32(1e-8 / FP8_MAX)
                    norm_factor = fx.Float32(rcp_f32(v_max_safe))
                    norm_factor_b = fx.Vector.from_elements(
                        [norm_factor], dtype=fx.Float32
                    ).broadcast_to(4)
                # pass 2: global max over warps -> exp -> fp8 P pack (-> sP) -> sum
                if const_expr(M1_SCALE_BEFORE_MASK):
                    tile_max = _ld_lw_row(sLmax_off, lane16).reduce(
                        ReductionOp.MAX, fastmath=fm_nnan
                    )
                    m_new = (
                        tile_max
                        if const_expr(single_tile_plan)
                        else fx.maxnumf(m_prev, tile_max, fastmath=fm_nnan)
                    )
                else:
                    tile_max = _ld_lw_row(sLmax_off, lane16).reduce(ReductionOp.MAX)
                    m_new = (
                        tile_max
                        if const_expr(single_tile_plan)
                        else fx.maxnumf(m_prev, tile_max)
                    )
                # Keep empty partitions' persisted max at -inf, but exponentiate
                # relative to 0 to avoid -inf-(-inf) NaNs in P and corrections.
                softmax_max = m_new
                if const_expr(M1_SCALE_BEFORE_MASK):
                    softmax_max = (m_new > NEG_INF).select(m_new, ZERO_F)
                m_new_b = fx.Vector.from_elements(
                    [softmax_max], dtype=fx.Float32
                ).broadcast_to(4)
                ls = fx.Float32(0.0)
                words = []
                if const_expr(not M1_SCALE_BEFORE_MASK):
                    zero4_p = fx.Vector.filled(4, 0.0, fx.Float32)
                for a in range_constexpr(NCHUNK):
                    if const_expr(M1_SCALE_BEFORE_MASK):
                        Pa = fx.Vector(
                            fx.exp2(masked_chunks[a] - m_new_b, fastmath="fast")
                        )
                    else:
                        # Finite mask sentinels require an explicit zero probability.
                        valid_a = masked_chunks[a] > fx.Vector.filled(
                            4, -1e29, fx.Float32
                        )
                        Pa = valid_a.select(
                            fx.Vector(
                                fx.exp2(
                                    masked_chunks[a] * scale - m_new_b,
                                    fastmath="fast",
                                )
                            ),
                            zero4_p,
                        )
                    ls = ls + Pa.reduce(ReductionOp.ADD)
                    if const_expr(per_token_kv):
                        v_scale_this = (
                            _load_scale_vec(sVScale_off, a, cur_kv_buf)
                            if const_expr(head_dim == 64)
                            else v_scale_vecs[a]
                        )
                        p_scaled = Pa * v_scale_this * norm_factor_b
                    elif const_expr(SCALAR_FP8_DECODE):
                        p_scaled = Pa
                    else:
                        p_scaled = Pa * fx.Vector.filled(4, FP8_MAX, fx.Float32)
                    words.append(_f32_to_fp8_words(p_scaled)[0])
                p_off0 = (
                    sP_off + lane16 * SP_ROW_BYTES + warp * TOK_PER_WARP + rgroup * 4
                )
                for a in range_constexpr(NCHUNK):
                    _lds_store(
                        p_off0 + a * (MFMA_MNK // 4) * f32,
                        fx.Int32,
                        fx.Vector.from_elements([words[a]], dtype=fx.Int32),
                    )
                if const_expr(head_dim == 64):
                    fx.rocdl.sched_dswr(NCHUNK)
                for sh in (16, 32):
                    ls = ls + ls.shuffle_xor(sh, WAVE)
                corr_reg = (
                    ZERO_F
                    if const_expr(single_tile_plan)
                    else fx.Float32(fx.exp2(m_prev - softmax_max, fastmath="fast"))
                )
                if rgroup == 0:
                    _st_lw(sLsum_off, lane16, warp, ls)
                gpu.barrier()
                if const_expr(PAGE16_VPIPE):  # noqa: SIM102
                    # Next K can now overlap P reads/PV.
                    if const_expr(not single_tile_plan) and tt1 < part_end:
                        k_next = _k_ops_from_phys(_k_page_read_warp())
                next_state[K_SLOT] = k_next
                gsum = _ld_lw_row(sLsum_off, lane16).reduce(ReductionOp.ADD)
                l_new = (
                    gsum if const_expr(single_tile_plan) else l_prev * corr_reg + gsum
                )
                p_ops = _lds_load(
                    sP_off + lane16 * SP_ROW_BYTES + rgroup * 64, fx.Int64, NVOPS
                )
                corr_b = fx.Vector.from_elements(
                    [corr_reg], dtype=fx.Float32
                ).broadcast_to(OP_ELEMS)
                if const_expr(prefetch_v):
                    v_vh_batch = v_vh_shared
                else:
                    v_vh_batch = [
                        _v_ops(v_page_cur, vh) for vh in range_constexpr(VHE_CHUNKS)
                    ]
                for vh in range_constexpr(VHE_CHUNKS):
                    v_vh = v_vh_batch[vh]
                    acc = fx.Vector.filled(MFMA_ACC_ELEMS, 0.0, fx.Float32)
                    acc = _mfma_fp8(v_vh, p_ops, 0, 0, NVOPS, acc)
                    op = fx.Vector(acc)
                    if const_expr(per_token_kv):
                        op = op * fx.Vector.from_elements(
                            [v_max_scaled], dtype=fx.Float32
                        ).broadcast_to(OP_ELEMS)
                    o_acc[vh] = (
                        op if const_expr(single_tile_plan) else o_acc[vh] * corr_b + op
                    )
                next_state.extend([*o_acc, m_new, l_new])
            if const_expr(MTP4_PREFETCH_V):
                next_state.append(v_next)
            results = yield next_state
        o_final = results

        # Each lane stores four head-dim values for one query row.
        inv_fp8 = fx.Float32(1.0 / FP8_MAX)
        for m in range_constexpr(M_TILES):
            row = m * MFMA_MNK + lane16  # flat (mtp, gqa) query-row for this lane
            global_row = query_begin * query_group_size + row
            qi_e = row // query_group_size
            gs_head_e = row - qi_e * query_group_size
            qh = kv_h * query_group_size + gs_head_e
            l_row = o_final[_l_slot(m)]
            safe_l = (l_row > ZERO_F).select(l_row, fx.Float32(1.0))
            inv_l = fx.Float32(rcp_f32(safe_l))
            if const_expr(DIRECT_SINKS):
                # Sinks affect only the denominator. Compare in natural-logit
                # units before LOG2E scaling to avoid overflow for finite sinks.
                sink_value = fx.Float32(sink_token[qh])
                row_max = o_final[_m_slot(m)] * fx.Float32(1.0 / LOG2E)
                total_max = fx.maxnumf(row_max, sink_value)
                safe_max = (total_max > NEG_INF).select(total_max, ZERO_F)
                kv_mass = (l_row > ZERO_F).select(
                    fx.exp2((row_max - safe_max) * fx.Float32(LOG2E), fastmath="fast"),
                    ZERO_F,
                )
                # +inf suppresses all KV output; -inf disables this head's
                # sink. Select before exp2 to avoid the +inf - +inf case.
                sink_shift = (sink_value == safe_max).select(
                    ZERO_F, sink_value - safe_max
                )
                sink_mass = fx.exp2(sink_shift * fx.Float32(LOG2E), fastmath="fast")
                denominator = l_row * kv_mass + sink_mass
                safe_denominator = (denominator > ZERO_F).select(
                    denominator, fx.Float32(1.0)
                )
                inv_l = kv_mass * fx.Float32(rcp_f32(safe_denominator))
            if const_expr(per_token_kv):
                o_scale = inv_l
            elif const_expr(SCALAR_FP8_DECODE):
                # Direct P conversion needs no compensating 1/FP8_MAX.
                o_scale = inv_l * v_scale_f
            else:
                o_scale = inv_l * (v_scale_f * inv_fp8)
            o_scale_b = fx.Vector.from_elements(
                [o_scale], dtype=fx.Float32
            ).broadcast_to(OP_ELEMS)

            def _emit(o_norm, sub, query_idx, query_head):
                if const_expr(NP == 1 and not use_work_plan):
                    out_offset = (
                        (seq * query_length + query_begin + query_idx) * stride_o_row
                        + query_head * stride_o_head
                        + sub * OP_ELEMS
                    )
                    fx.ptr_store(o_norm, fx.add_offset(output, out_offset))
                elif const_expr(buffer_plan_output):
                    # Row guards keep the full 8-byte store within this slot.
                    pout_offset = global_row * head_dim + sub * OP_ELEMS  # noqa: B023
                    buf_copy_store(
                        pout_buffer,
                        pout_offset,
                        o_norm,
                        elem=Q_DTYPE,
                        unit_elems=OP_ELEMS,
                        cache_modifier=0,
                    )
                else:
                    base = partial_slot * TOTAL_ROWS + global_row  # noqa: B023
                    pout_offset = base * head_dim + sub * OP_ELEMS
                    fx.ptr_store(
                        o_norm,
                        fx.add_offset(pout, pout_offset),
                    )

            for vh in range_constexpr(VHE_CHUNKS):
                o_slot = _o_slot(m, vh)
                o_norm = (o_final[o_slot] * o_scale_b).to(Q_DTYPE)
                head_base = (
                    vh * (NWARP * MFMA_MNK) + warp * MFMA_MNK + rgroup * OP_ELEMS
                )
                sub = head_base // OP_ELEMS
                if row < CTA_ROWS:
                    _emit(o_norm, sub, qi_e, qh)

            if const_expr(NP > 1 or use_work_plan):  # noqa: SIM102
                if warp == 0 and rgroup == 0:
                    base = partial_slot * TOTAL_ROWS + global_row
                    if row < CTA_ROWS:
                        # The shared reducer expects maxima in natural-log units.
                        pmax[base] = o_final[_m_slot(m)] * fx.Float32(1.0 / LOG2E)
                        psum[base] = l_row

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_tile_kernel(
        output_ptr: fx.Pointer,
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        context_lengths_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        sinks_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_o_row: fx.Int32,
        stride_o_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info_ptr: fx.Pointer,
        num_sequences: fx.Int32,
    ):
        def _run_task(seq, start, end, context):
            _pa_decode_tile_task(
                output_ptr,
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                context_lengths_ptr,
                key_scale_ptr,
                value_scale_ptr,
                sinks_ptr,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_o_row,
                stride_o_head,
                stride_q_row,
                stride_q_head,
                num_sequences,
                seq,
                start,
                end,
                context,
            )

        if const_expr(use_work_plan):
            # Skip cleared padding records: no scratch writes or Q loads.
            # This CTA-uniform guard keeps every task barrier convergent.
            if const_expr(batch_first_plan_grid):
                slot = fx.Int32(
                    fx.Uint32(gpu.block_id("x")) * fx.Uint32(gpu.grid_dim.z)
                    + fx.Uint32(gpu.block_id("z"))
                )
            else:
                slot = fx.Int32(gpu.block_id("x"))
            work = fx.recast_iter(fx.Int32, work_info_ptr)
            task = fx.ptr_load(
                fx.add_offset(work, slot * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            start = fx.Int32(task[1])
            end = fx.Int32(task[2])
            if start < end:
                _run_task(fx.Int32(task[0]), start, end, fx.Int32(task[3]))
        else:
            zero = fx.Int32(0)
            _run_task(zero, zero, zero, zero)

    @flyc.jit
    def pa_decode_tile_launch(
        output: fx.Pointer,
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        block_tables: fx.Pointer,
        context_lengths: fx.Pointer,
        key_scale: fx.Pointer,
        value_scale: fx.Pointer,
        sinks: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        num_seqs: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_o_row: fx.Int32,
        stride_o_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info: fx.Pointer,
        work_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # Ambient contract permits FMAs; explicit per-op fastmath still wins.
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_decode_tile_kernel(
                output,
                pmax,
                psum,
                pout,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lengths,
                key_scale,
                value_scale,
                sinks,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_o_row,
                stride_o_head,
                stride_q_row,
                stride_q_head,
                work_info,
                num_seqs,
            ).launch(
                grid=(
                    (num_seqs, num_kv_heads * query_splits, work_capacity // num_seqs)
                    if batch_first_plan_grid
                    else (
                        work_capacity if use_work_plan else num_seqs,
                        num_kv_heads * query_splits,
                        1 if use_work_plan else NP,
                    )
                ),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    compiled = {
        "launch": pa_decode_tile_launch,
        "kernel": pa_decode_tile_kernel,
    }
    return _PA_DECODE_TILE_CACHE.setdefault(cache_key, compiled)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Compile-time specialization and storage layout for paged-attention decode."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

MFMA_MNK = 16  # M=N=16; query rows are padded to M-tiles.
# One i64 operand pack/lane covers K32, including in K128 layouts.
FP8_PACK_K = 32
WAVE = 64
# Accumulator elements/lane are independent of MFMA K.
MFMA_ACC_ELEMS = MFMA_MNK * MFMA_MNK // WAVE
LOG2E = 1.4426950408889634
KV_COMPUTE_BLOCK = 256


@dataclass(frozen=True)
class PaDecodeSchedule:
    """Select the cache key without constructing the full kernel layout."""

    cache_key: tuple[object, ...]
    is_gfx950: bool
    PER_TOKEN_M1: bool

    @classmethod
    def select(
        cls,
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
        kv_buffer_u32: bool = False,
        query_splits: int | None = None,
        use_work_plan: bool = False,
        work_capacity: int | None = None,
        max_context_length: int | None = None,
        sliding_window: int = 0,
        use_sinks: bool = False,
        sink_dtype_str: str = "f32",
    ) -> PaDecodeSchedule:
        """Choose the existing policy from host-known grid and query bounds."""
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
            if sliding_window == 0 and max_context_length is not None:
                context_tiles = (
                    max_context_length + KV_COMPUTE_BLOCK - 1
                ) // KV_COMPUTE_BLOCK
                split_workgroups = min(
                    split_workgroups, num_seqs * num_kv_heads * context_tiles
                )
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
                if single_tile_plan
                and query_length == 4
                and block_size == 128
                and trans_v
                else query_length
            )
            query_splits = 1
            if (
                TUNED_PER_TOKEN
                and (num_kv_heads == 1 or (use_work_plan and sliding_window == 0))
                and query_length in (2, 4)
                and query_group_size == 16
                and split_weight * split_workgroups <= 2 * num_compute_units
            ):
                query_splits = query_length
            elif (
                TUNED_PER_TOKEN
                and use_work_plan
                and sliding_window == 0
                and query_length == 4
                and query_group_size == 8
                and block_size == 128
                and trans_v
                and 2 * split_workgroups <= num_compute_units
            ):
                query_splits = 2
        assert query_splits in (1, 2, 4), "query_splits must be one of 1, 2, 4"
        assert query_length % query_splits == 0, "query_splits must divide query_length"
        QUERIES_PER_CTA = query_length // query_splits
        TOTAL_ROWS = query_length * query_group_size
        CTA_ROWS = QUERIES_PER_CTA * query_group_size
        M_TILES = (CTA_ROWS + MFMA_MNK - 1) // MFMA_MNK
        WIDE_FP8_MFMA = (
            TUNED_PER_TOKEN and query_length in (3, 4) and query_group_size == 16
        )
        MTP4_FUSED = WIDE_FP8_MFMA and M_TILES == 4
        BUFFER_KV = (
            kv_buffer_u32 and MTP4_FUSED and wide_kv_addressing and block_size == 128
        )

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
            BUFFER_KV,
            prefetch_v,
            query_splits,
            use_work_plan,
            single_tile_plan,
            batch_first_plan_grid,
            sliding_window,
            use_sinks,
            sink_dtype_str,
        )
        return cls(
            cache_key=cache_key,
            is_gfx950=is_gfx950,
            PER_TOKEN_M1=PER_TOKEN_M1,
        )


@dataclass(frozen=True)
class PaDecodeTraits:
    """Selected PA specialization shared by the kernel's operation helpers.

    Scheduling inputs only select the policy; the cache key records that policy
    and preserves the caller's scale/window values before normalization.
    """

    cache_key: tuple[object, ...]

    # Selected specialization and query shape.
    head_dim: int
    query_group_size: int
    block_size: int
    softmax_scale: float
    per_token_kv: bool
    query_length: int
    trans_v: bool
    wide_kv_addressing: bool
    query_splits: int
    use_work_plan: bool
    sliding_window: int
    single_tile_plan: bool
    prefetch_v: bool
    batch_first_plan_grid: bool
    buffer_plan_output: bool

    # Element formats and instruction policy.
    FP8: type
    FP8_MAX: float
    Q_DTYPE: type
    SINK_DTYPE: type
    Q_ABSMAX_F32: bool
    UNIQUE_SCALE_STAGING: bool
    SCALAR_FP8_DECODE: bool
    WIDE_FP8_MFMA: bool
    PACKS_PER_MFMA: int
    MTP4_FUSED: bool
    MTP4_PREFETCH_V: bool
    BUFFER_KV: bool
    PAGE16_VPIPE: bool
    REUSE_KV_PAGES: bool
    SCALES_BEFORE_CURRENT_V: bool
    M1_SCALE_BEFORE_MASK: bool

    # Tile geometry, register operands, and loop-carried state.
    QUERIES_PER_CTA: int
    TOTAL_ROWS: int
    CTA_ROWS: int
    M_TILES: int
    P_BUFFERS: int
    NWARP: int
    TILE_TOK: int
    TOK_PER_WARP: int
    NCHUNK: int
    PAGES_PER_CHUNK: int
    KV_EXTENT: int
    RGROUP_QUARTERS: int
    QK_CHUNK_ELEMS: int
    QKHE_LOOP: int
    N_SUBCHUNKS: int
    QCHUNK: int
    QLOAD_UNIT: int
    N_QLOADS: int
    VHE_CHUNKS: int
    VHE_SIZE: int
    OP_ELEMS: int
    NVOPS: int
    STEPS_PER_PAGE: int
    STEPS_PER_CHUNK: int
    NP: int
    DIRECT_SINKS: bool
    BLOCK_THREADS: int
    K_SLOT: int
    V_SLOT: int
    STATE_PER_M: int
    V_DATA_SLOT: int

    # LDS byte offsets; Q and P intentionally share storage.
    f32: int
    sP_off: int
    SP_ROW_BYTES: int
    sQscale_off: int
    NWARP_PAD: int
    sLmax_off: int
    sLsum_off: int
    sVPage_off: int
    KV_BUF_STRIDE: int
    sKScale_off: int
    sVScale_off: int
    sVScaleMax_off: int
    total_bytes: int

    def o_slot(self, m: int, vh: int) -> int:
        return 2 + self.STATE_PER_M * m + vh

    def m_slot(self, m: int) -> int:
        return 2 + self.STATE_PER_M * m + self.VHE_CHUNKS

    def l_slot(self, m: int) -> int:
        return 2 + self.STATE_PER_M * m + self.VHE_CHUNKS + 1

    @classmethod
    def create(cls, schedule: PaDecodeSchedule) -> PaDecodeTraits:
        """Derive the immutable kernel layout after a specialization cache miss."""
        cache_key = schedule.cache_key
        (
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
            BUFFER_KV,
            prefetch_v,
            query_splits,
            use_work_plan,
            single_tile_plan,
            batch_first_plan_grid,
            sliding_window,
            use_sinks,
            sink_dtype_str,
        ) = cache_key
        is_gfx950 = schedule.is_gfx950
        PER_TOKEN_M1 = schedule.PER_TOKEN_M1
        IS_BF16 = query_dtype == "bf16"
        TUNED_SHAPE = is_gfx950 and head_dim == 128 and block_size in (16, 128)
        TUNED_PER_TOKEN = TUNED_SHAPE and IS_BF16 and per_token_kv
        TUNED_SCALAR = TUNED_SHAPE and trans_v and not per_token_kv
        QUERIES_PER_CTA = query_length // query_splits
        TOTAL_ROWS = query_length * query_group_size

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
            TUNED_SCALAR
            and IS_BF16
            and query_length == 1
            and query_group_size in (8, 16)
        )
        # K128 consumes four K32 packs without changing cache/LDS layouts.
        WIDE_FP8_MFMA = (
            TUNED_PER_TOKEN and query_length in (3, 4) and query_group_size == 16
        )
        MFMA_K = 128 if WIDE_FP8_MFMA else FP8_PACK_K
        PACKS_PER_MFMA = MFMA_K // FP8_PACK_K
        MTP4_FUSED = WIDE_FP8_MFMA and M_TILES == 4
        MTP4_PREFETCH_V = MTP4_FUSED and (
            block_size == 16 or not wide_kv_addressing or BUFFER_KV
        )
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
        assert (
            TILE_TOK == NWARP * TOK_PER_WARP
        ), "KV tile must split evenly across warps"
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

        VHE_CHUNKS = head_dim // (
            NWARP * MFMA_MNK
        )  # 2 for head_dim=128, 1 for head_dim=64
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
        KV_BUF_STRIDE = (
            2 * NWARP * TOK_PER_WARP * f32
        )  # k-region + v-region, one buffer
        KV_SCALE_BUFFERS = 1 if single_tile_plan else 2
        sKScale_off = sVPage_off + sVPage_bytes
        sVScale_off = sKScale_off + NWARP * TOK_PER_WARP * f32
        sKVScale_bytes = KV_SCALE_BUFFERS * KV_BUF_STRIDE if per_token_kv else 0
        sVScaleMax_off = sKScale_off + sKVScale_bytes
        sVScaleMax_bytes = (
            NWARP_PAD * f32 if per_token_kv else 0
        )  # m-independent: one cross-warp slot
        total_bytes = sVScaleMax_off + sVScaleMax_bytes

        return cls(
            cache_key=cache_key,
            head_dim=head_dim,
            query_group_size=query_group_size,
            block_size=block_size,
            softmax_scale=softmax_scale,
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            wide_kv_addressing=wide_kv_addressing,
            query_splits=query_splits,
            use_work_plan=use_work_plan,
            sliding_window=sliding_window,
            single_tile_plan=single_tile_plan,
            prefetch_v=prefetch_v,
            batch_first_plan_grid=batch_first_plan_grid,
            buffer_plan_output=buffer_plan_output,
            FP8=FP8,
            FP8_MAX=FP8_MAX,
            Q_DTYPE=Q_DTYPE,
            SINK_DTYPE=SINK_DTYPE,
            Q_ABSMAX_F32=Q_ABSMAX_F32,
            UNIQUE_SCALE_STAGING=UNIQUE_SCALE_STAGING,
            SCALAR_FP8_DECODE=SCALAR_FP8_DECODE,
            WIDE_FP8_MFMA=WIDE_FP8_MFMA,
            PACKS_PER_MFMA=PACKS_PER_MFMA,
            MTP4_FUSED=MTP4_FUSED,
            MTP4_PREFETCH_V=MTP4_PREFETCH_V,
            BUFFER_KV=BUFFER_KV,
            PAGE16_VPIPE=PAGE16_VPIPE,
            REUSE_KV_PAGES=REUSE_KV_PAGES,
            SCALES_BEFORE_CURRENT_V=SCALES_BEFORE_CURRENT_V,
            M1_SCALE_BEFORE_MASK=M1_SCALE_BEFORE_MASK,
            QUERIES_PER_CTA=QUERIES_PER_CTA,
            TOTAL_ROWS=TOTAL_ROWS,
            CTA_ROWS=CTA_ROWS,
            M_TILES=M_TILES,
            P_BUFFERS=P_BUFFERS,
            NWARP=NWARP,
            TILE_TOK=TILE_TOK,
            TOK_PER_WARP=TOK_PER_WARP,
            NCHUNK=NCHUNK,
            PAGES_PER_CHUNK=PAGES_PER_CHUNK,
            KV_EXTENT=KV_EXTENT,
            RGROUP_QUARTERS=RGROUP_QUARTERS,
            QK_CHUNK_ELEMS=QK_CHUNK_ELEMS,
            QKHE_LOOP=QKHE_LOOP,
            N_SUBCHUNKS=N_SUBCHUNKS,
            QCHUNK=QCHUNK,
            QLOAD_UNIT=QLOAD_UNIT,
            N_QLOADS=N_QLOADS,
            VHE_CHUNKS=VHE_CHUNKS,
            VHE_SIZE=VHE_SIZE,
            OP_ELEMS=OP_ELEMS,
            NVOPS=NVOPS,
            STEPS_PER_PAGE=STEPS_PER_PAGE,
            STEPS_PER_CHUNK=STEPS_PER_CHUNK,
            NP=NP,
            DIRECT_SINKS=DIRECT_SINKS,
            BLOCK_THREADS=BLOCK_THREADS,
            K_SLOT=K_SLOT,
            V_SLOT=V_SLOT,
            STATE_PER_M=STATE_PER_M,
            V_DATA_SLOT=V_DATA_SLOT,
            f32=f32,
            sP_off=sP_off,
            SP_ROW_BYTES=SP_ROW_BYTES,
            sQscale_off=sQscale_off,
            NWARP_PAD=NWARP_PAD,
            sLmax_off=sLmax_off,
            sLsum_off=sLsum_off,
            sVPage_off=sVPage_off,
            KV_BUF_STRIDE=KV_BUF_STRIDE,
            sKScale_off=sKScale_off,
            sVScale_off=sVScale_off,
            sVScaleMax_off=sVScaleMax_off,
            total_bytes=total_bytes,
        )

// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// V40 Gen1 buffer managers: QManager8to16bitsV1, KvManager8to16bitsV1,
// OManager16bitsV4Gen1Swizzle, OManager32bitsV4Gen1Swizzle, OManager32bitsV4Gen1SwNoStage. Moved
// out of hk_mla_buffer_managers.cuh so V32 and V40-gen1 keep their residency managers separated.
// Generic cvt/pack helpers (float_2_bf16_pair etc.) live in hk_mla_utils.cuh.

#include "hk_mla_utils.cuh"

using namespace hk_mla;

// =============================================================================
// V40 sub-tile-of-8 swizzle helpers
// =============================================================================
// Each warp's K and Q wave-tile spans 64 cols organized as 8 sub-tiles of 8.
// Storing the sub-tiles in LDS in permuted order [0,2,4,6,1,3,5,7] eliminates
// the 2-way ds_write_b128 bank conflict (Site C, write side). The QK reduction
// is unaffected because partial sums are commutative; PV inherits the
// permutation on its N-axis (=kKvLoraRank=512), un-swizzled in OManagerV3.
//
// Index a col-elem by p; bits [2:0] are intra-sub-tile (preserved), bits [5:3]
// are the sub-tile id, bits [>=6] are inter-wave-tile (preserved).
//
//   sub_d -> sub_L : L = (d >> 1) | ((d & 1) << 2)        // data -> LDS
//   sub_L -> sub_d : d = ((L >> 2) & 1) | ((L & 3) << 1)  // LDS  -> data
//
// Mapping of sub-tile fields (two equivalent views of the same permutation):
//   LDS  position : 0 1 2 3 4 5 6 7      <- walk LDS slots
//   data sub-tile : 0 2 4 6 1 3 5 7      <- find this data sub-tile there
//                                           (the user-specified order)
//   data sub-tile : 0 1 2 3 4 5 6 7      <- walk data sub-tiles
//   LDS  position : 0 4 1 5 2 6 3 7      <- store it at this LDS slot
//                                           (= inverse, used by sb8_perm)
//
__device__ __forceinline__ static constexpr uint32_t sb8_perm_col_elems(uint32_t p)
{
    // data -> LDS (forward). Operates on bits [5:3] of p.
    //   p_bit3 (LSB of sub_d) -> L_bit5 (MSB of sub_L)
    //   p_bit4,5 (high 2 bits of sub_d) -> L_bit3,4 (low 2 bits of sub_L)
    return (p & 0x7u) | (((p >> 3) & 0x1u) << 5) | (((p >> 3) & 0x6u) << 2) | (p & ~0x3Fu);
}

__device__ __forceinline__ static constexpr uint32_t sb8_inv_perm_col_elems(uint32_t L)
{
    // LDS -> data (inverse).
    //   L_bit5 -> p_bit3
    //   L_bit3,4 -> p_bit4,5
    return (L & 0x7u) | (((L >> 5) & 0x1u) << 3) | ((L & 0x18u) << 1) | (L & ~0x3Fu);
}

// V4.0 Q manager: separate FP8 NoPE + BF16 RoPE buffers. Q is split: Q[:, 0:256]
// lives pinned in VGPR after fp8->bf16 cvt+scale; Q[:, 256:512] is converted to
// bf16 and parked in a per-WG bf16 LDS region for in-loop ds_read.
//
// Per V4 spec §5.1.3 ("LDS reuse trick"):
//   * Phase 1 (warmup) -- the 64 KB Q LDS region is used as staging while
//     loading Q[:, 0:256] (vmem fp8 -> cvt+scale -> bf16 -> ds_write -> ds_read
//     into pinned q_vgpr). At end of Phase 1 the LDS region's contents become
//     dead.
//   * Phase 2 (residence) -- the SAME 64 KB region is overwritten with the
//     bf16 form of Q[:, 256:512] and stays live for the whole work-loop.
// No barrier between the two phases: each lane reads only what it wrote (no
// inter-wave LDS communication), so the per-warp regions are private.
//
// Total LDS footprint = max(Phase 1, Phase 2) = 64 KB (both halves are the same
// 128 x 256 bf16 size).
//
// V4.0 Q manager. Loads Q from packed FP8 NoPE + BF16 RoPE buffers into a
// hybrid residency: Q[:, 0:256] (= half of kQkNopeHeadDim) lives pinned in
// VGPRs after fp8->bf16 cvt+scale; Q[:, 256:512] (rest of NoPE + RoPE) lives
// in a per-WG bf16 LDS region used by QK Phase B in-loop ds_reads.
//
// Phase 1 (warmup, VGPR half):
//   Per warp, 4 chunks of 16 rows x 64 cols are staged via buffer_load_lds_b128
//   into a per-warp 1024-byte staging slot (double-buffered = 2x1024 B/warp =
//   16 KB across 8 warps). For each chunk we then ds_read_b64 + 4 cvts/iter to
//   produce 8 bf16/lane = 4 dwords/lane in mfma A-operand layout, written to
//   q_vgpr[GPR_NOPE_VGPR_START + chunk*8 + iter*4 + (0..3)].
//
// Phase 2 (residence, LDS half):
//   Per warp, 4 chunks of 16 rows x 64 cols cover NoPE[256:448] (3 fp8 chunks)
//   + RoPE[0:64] (1 bf16 chunk). NoPE chunks: fp8 -> VGPR -> cvt+scale ->
//   bank-conflict-free swapped ds_write_b128 (mirrors KvManager8to16bitsV1).
//   RoPE chunk: 2 buffer_load_lds_b128 direct vmem->LDS (no cvt).
//   Final layout = wave-major contiguous 16x32 bf16 sub-blocks:
//   sub_block_byte_offset(warp_idx, col_tile) = warp_idx*8192 + col_tile*1024.
//   Each wave owns its own 8 KB region [warp_idx*8192, (warp_idx+1)*8192).
//
// LDS reuse: Phase 1 staging (2 KB/warp = 16 KB total) lives at the FRONT of
// each wave's OWN 8 KB final region. Phase 2 then overwrites those bytes as
// part of the same region. No barrier needed -- per-wave program order
// sequences the intra-wave staging->final overwrite, and no other wave ever
// touches wave w's bytes (wave-major exclusivity).
template <typename T>
class QManager8to16bitsV1
{
    private:
    using q_nope_t = typename T::q_nope_t;
    using q_rope_t = typename T::q_rope_t;
    static_assert(std::is_same_v<q_nope_t, hk::fp8e4m3>,
                  "QManager8to16bitsV1: q_nope_t must be fp8e4m3.");
    static_assert(std::is_same_v<q_rope_t, hk::bf16>,
                  "QManager8to16bitsV1: q_rope_t must be bf16.");
    static_assert(T::kQkNopeHeadDim == 448, "QManager8to16bitsV1: NOPE width must be 448.");
    static_assert(T::kQkRopeHeadDim == 64, "QManager8to16bitsV1: ROPE width must be 64.");
    static_assert(T::kQkHeadDim == 512, "QManager8to16bitsV1: kQkHeadDim must be 512 (NOPE+ROPE).");
    static_assert(T::kBlockM == 128, "QManager8to16bitsV1: kBlockM must be 128.");
    static_assert(T::kNumWarps == 8, "QManager8to16bitsV1: requires 8 warps.");
    static_assert(T::kTileM == 16, "QManager8to16bitsV1: kTileM must be 16.");

    public:
    // Sub-block geometry (16 rows x 32 bf16 cols = 1024 B). This is the unit
    // ds_read_b128 grabs for a QK A-tile.
    static constexpr uint32_t kSubBlockRows  = 16;
    static constexpr uint32_t kSubBlockCols  = 32;
    static constexpr uint32_t kSubBlockBytes = kSubBlockRows * kSubBlockCols * sizeof(hk::bf16);

    // Q split: VGPR half = Q[:, 0:256], LDS half = Q[:, 256:512].
    // The LDS half is 192 bf16 NoPE cols (record bytes 256..448) + 64 bf16 RoPE
    // cols (= 8 col_tiles total in the LDS sub-block grid).
    static constexpr uint32_t kVgprHalfCols    = 256;
    static constexpr uint32_t kLdsHalfCols     = T::kQkHeadDim - kVgprHalfCols;     // 256
    static constexpr uint32_t kLdsHalfNopeCols = T::kQkNopeHeadDim - kVgprHalfCols; // 192
    static constexpr uint32_t kLdsHalfRopeCols = T::kQkRopeHeadDim;                 // 64
    static_assert(kLdsHalfNopeCols + kLdsHalfRopeCols == kLdsHalfCols,
                  "QManager8to16bitsV1: LDS half geometry mismatch.");

    // Phase 1 chunking: ALL NoPE (cols 0:448 = 7 chunks of 64) is relayed
    // vmem -> staging -> VGPR. NoPE is never kept in LDS for the QK loop.
    static constexpr uint32_t kP1ChunkCols = 64;
    static constexpr uint32_t kP1NumChunks = kVgprHalfCols / kP1ChunkCols;     // 4
    static constexpr uint32_t kP1MaxChunks = T::kQkNopeHeadDim / kP1ChunkCols; // 7

    static constexpr uint32_t kP1StagingBytesPerWarp =
        T::kTileM * kP1ChunkCols * sizeof(q_nope_t);   // 1024
    static constexpr uint32_t kP1NumStagingBuffers = 2; // double-buffer
    static constexpr uint32_t kP1StagingBytesPerWarpTotal =
        kP1NumStagingBuffers * kP1StagingBytesPerWarp; // 2048

    // RoPE (cols 448:512) is one 64-col chunk.
    static constexpr uint32_t kP2ChunkCols = 64;
    static_assert(kLdsHalfRopeCols == kP2ChunkCols,
                  "QManager8to16bitsV1: RoPE chunk currently assumed to be one full chunk.");

    // Phase-2 "final" LDS region: holds RoPE ONLY (all NoPE is in VGPR), as two
    // 16x32 bf16 sub-block tiles at col-tiles 0,1. RoPE reuses the staging bytes
    // (consumed to VGPR first; load_q drains lgkmcnt before the RoPE ds_write),
    // so the region is just the staging footprint -> 2 col-tiles = 16 KB (the old
    // NoPE-in-LDS layout needed all 8 tiles / 64 KB).
    static constexpr uint32_t kRopeColTileLo = 0u;
    static constexpr uint32_t kRopeColTileHi = 1u;

    static constexpr uint32_t kFinalLdsRows     = T::kBlockM;                        // 128
    static constexpr uint32_t kFinalLdsRowTiles = kFinalLdsRows / kSubBlockRows;     // 8
    static constexpr uint32_t kFinalLdsColTiles = kLdsHalfRopeCols / kSubBlockCols;  // 2 (RoPE only)
    static constexpr uint32_t kWarpFinalBytes   = kFinalLdsColTiles * kSubBlockBytes; // 2048
    static constexpr uint32_t kFinalLdsBytes    = T::kNumWarps * kWarpFinalBytes;     // 16 KB
    // Wave-major: each wave owns kWarpFinalBytes exclusively (no inter-wave barrier
    // between staging and the RoPE store). The RoPE-reuses-staging overwrite is
    // intra-wave only, sequenced by program order + the lgkmcnt drain in load_q.
    static_assert(kWarpFinalBytes >= kP1StagingBytesPerWarpTotal,
                  "QManager8to16bitsV1: per-warp Phase 1 staging must fit within the "
                  "wave's OWN Phase 2 final region (wave-major contiguous layout).");

    // Per-row record byte stride for the packed fp8 NoPE + scale + pad input.
    static constexpr uint32_t kPackedNopeStride = T::kQkPackedNopeQElems * sizeof(q_nope_t); // 512
    static constexpr uint32_t kRopeStride       = T::kQkRopeHeadDim * sizeof(q_rope_t);      // 128
    static constexpr uint32_t kScaleBaseOff     = 448u; // E8M0 scales start at byte 448 of record.

    private:
    // Sub-block byte offset inside the final region (wave-major layout). Wave w
    // owns the contiguous kWarpFinalBytes region [w*kWarpFinalBytes, ...); inside
    // that, col_tile c occupies [c*1024, (c+1)*1024). Signature takes warp_idx
    // (not row_tile) because row_tile == warp_idx everywhere this is called: each
    // warp owns one of the 8 row-tiles of the 128-row Q block.
    __device__ __forceinline__ static constexpr uint32_t sub_block_byte_offset(uint32_t warp_idx,
                                                                               uint32_t col_tile)
    { return warp_idx * kWarpFinalBytes + col_tile * kSubBlockBytes; }

    // Per-warp staging base = the wave's OWN final region (kWarpFinalBytes).
    // RoPE (Phase 2) later overwrites these same bytes; no other wave touches
    // them, so the intra-wave overwrite is sequenced by per-wave program order.
    __device__ __forceinline__ static uintptr_t p1_warp_staging_base(uintptr_t p_lds_q,
                                                                     uint32_t warp_idx)
    { return p_lds_q + warp_idx * kWarpFinalBytes; }

    // ---- Inline-asm v_cvt_scalef32_pk_bf16_fp8 with a compile-time pinned
    //      destination VGPR. The clang builtin allocates a fresh VGPR for the
    //      result, which must then be v_mov_b32'd into the caller-pinned slot;
    //      this helper emits the cvt directly into the pinned slot, eliminating
    //      8 v_mov_b32s per Phase-1 chunk. opsel=false picks the low fp8 pair
    //      (lanes 0,1 of the 4-element source dword), opsel=true picks the high
    //      pair (lanes 2,3). ----
    template <uint32_t DST_GPR, bool kOpSelHigh>
    __device__ __forceinline__ static void cvt_scalef32_pk_bf16_fp8_pinned(uint32_t fp8_dw,
                                                                           float scale_f)
    {
        static_assert(DST_GPR < 256, "Pinned dst must be a VGPR (id < 256).");
        if constexpr(kOpSelHigh)
        {
            asm volatile("v_cvt_scalef32_pk_bf16_fp8 v[%0], %1, %2 op_sel:[1,0,0]"
                         :
                         : "n"(DST_GPR), "v"(fp8_dw), "v"(scale_f));
        }
        else
        {
            asm volatile("v_cvt_scalef32_pk_bf16_fp8 v[%0], %1, %2"
                         :
                         : "n"(DST_GPR), "v"(fp8_dw), "v"(scale_f));
        }
    }

    // ---- Phase 1: vmem fp8 -> per-warp staging via buffer_load_lds_b128 ----
    // Lane T loads 16 fp8 = 16 B from row T/4, cols (T%4)*16..+16 of the chunk
    // and writes them to staging[T*16] (the buffer_load_lds_b128 destination
    // pattern is fixed: lane T writes 16 B at lds_base + i_offset + T*16).
    //
    // After this layout the staging contains row-major data: row r occupies
    // bytes [r*64, r*64+64) (since 4 lanes/row * 16 B = 64 B/row), so the
    // subsequent ds_read_b64 in p1_staging_to_vgpr_chunk() can extract
    // contiguous 8 fp8/lane straight in mfma A-operand lane order.
    //
    // The two per-row E8M0 scale bytes for this chunk are also issued here
    // (returned via s0_dw/s1_dw output params) so their vmem latency overlaps
    // with the staging dwordx4_lds; the consuming p1_staging_to_vgpr_chunk
    // just drains vmcnt and reads from the cached dwords.
    template <uint32_t kChunkIdx, uint32_t kBufIdx>
    __device__ __forceinline__ static void p1_vmem_to_staging_chunk(
        const q_nope_t* p_q_warp, const uintptr_t p_lds_warp_staging, uint32_t& s_dw)
    {
        static_assert(kChunkIdx < 7u, "p1_vmem_to_staging_chunk: bad kChunkIdx (0..6).");
        static_assert(kBufIdx < kP1NumStagingBuffers, "p1_vmem_to_staging_chunk: bad kBufIdx.");

        constexpr uint32_t kColInRecord = kChunkIdx * kP1ChunkCols; // 0,64,128,192
        constexpr int kVOffI            = static_cast<int>(kColInRecord);
        constexpr uint32_t kStagingI    = kBufIdx * kP1StagingBytesPerWarp;
        // V4 packs ONE E8M0 scale per 64-col tile, duplicated to 2 bytes for
        // 16-bit alignment. Chunk == tile (both 64 cols), so each chunk has
        // exactly ONE scale shared across its 2 mfma A-tiles (cols [0,32) and
        // [32,64) of the chunk). Tile T's dup pair lives at bytes [448+2T, +2T+1].
        constexpr uint32_t kScaleByteInRec = kScaleBaseOff + 2u * kChunkIdx; // 448 + 2*kChunkIdx

        uint32_t lane_idx    = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uint32_t row_in_warp = lane_idx >> 2; // 0..15
        const uint32_t col_quad    = lane_idx & 3u; // 0..3
        // Swizzle: 16x64 chunk tiled into 4x4 sub-tiles (4 rows x 16 cols).
        // On the sub-tile-row band selected by S = (row_in_warp>>2)&1 (rows
        // 4..7 and 12..15), swap the upper/lower pair of col sub-tiles
        // (C_phys = C_log XOR 2). Identity elsewhere. Reader must apply the
        // same XOR. Breaks the 2-way ds_read_b128 conflict at the consumer:
        // b128's non-linear cycle 0 = {0..3,12..15,20..23,24..27} pairs lanes
        // (L,L+20), and `+20` flips bit 2 of L (=S) and bit 4 (=cb bit 0)
        // together -- so any XOR of bit 0 of cb cancels. XOR of bit 1 of cb
        // doesn't (`+20` doesn't touch bit 1), so the pair lands on distinct
        // quads. LDS-write side is conflict-free regardless: the HW-fixed
        // buffer_load_dwordx4_lds destination is lane T -> T*16, independent
        // of the data permutation.
        const uint32_t S            = (lane_idx >> 4) & 1u; // = (row_in_warp>>2)&1
        const uint32_t col_quad_swz = col_quad ^ (S << 1);  // 0..3 (physical)
        const uint32_t v_off        = row_in_warp * kPackedNopeStride + col_quad_swz * 16u;
        // Scale must be loaded for the row that the CONSUMER attributes to this
        // lane (consumer uses lane & 15, NOT lane >> 2 -- see
        // p1_staging_to_vgpr_chunk). Otherwise each lane scales row R's fp8 data
        // by row (R/4)'s scale, which is silently wrong on near-uniform data and
        // catastrophic on outliers.
        const uint32_t scale_row   = lane_idx & 15u;
        const uint32_t v_off_scale = scale_row * kPackedNopeStride;

        // Call __builtin_amdgcn_raw_ptr_buffer_load_lds directly so we
        // can (a) use opus' cached_rsrc (__amdgpu_buffer_rsrc_t) without
        // any cast and (b) pass a non-zero i_off. Opus' async_load
        // wrapper hardcodes i_off=0; HK's hk::llvm_amdgcn_raw_buffer_load_lds
        // exposes i_off but takes hk::i32x4 (forcing a separate rsrc
        // setup or a reinterpret_cast that spills). The imm offset adds
        // to BOTH gmem and LDS, so pre-subtract kColInRecord from the
        // LDS dst to cancel its contribution there. Requires
        // kLdsHeadPadBytes of pad before p_lds_q so warp 0's dst stays
        // >= 0 (static_assert in kernel body).
        auto g_q_nope =
            opus::make_gmem<uint8_t>(reinterpret_cast<const uint8_t*>(p_q_warp));
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            g_q_nope.cached_rsrc,
            (hk::as3_uint32_ptr)(p_lds_warp_staging + kStagingI - kColInRecord),
            /*size=*/16,
            v_off,
            /*s_off=*/0u,
            /*i_off=*/static_cast<int>(kColInRecord),
            /*aux=*/0);

        // Scale: 1 byte/lane via opus' load<1, uint8>. Stored as uint32_t so
        // e8m0_to_f32's asm volatile consumer has the v-class operand it needs.
        // Shares cached_rsrc with the LDS load above.
        s_dw = static_cast<uint32_t>(
            g_q_nope.template load<1>(v_off_scale, /*s_os=*/kScaleByteInRec)[0]);
    }

    // ---- Phase 1: 1 fp8 chunk in staging -> 2 mfma A-tiles in VGPR ----
    // Each chunk covers 64 cols = 2 mfma A-tiles (cols [0,32) and [32,64)).
    // Per iter: 1 ds_read_b64 (8 fp8 = 2 dwords), 4 cvts -> 4 dwords land in
    // vgpr range. The 2 per-row scale bytes are issued in
    // p1_vmem_to_staging_chunk and arrive via s0_dw/s1_dw.
    //
    // Caller VGPR contract:
    //   q_vgpr[GPR_NOPE_VGPR_START + 8*kChunkIdx + 4*iter + (0..3)] holds the
    //   bf16 form of Q[:, kChunkIdx*64 + 32*iter .. +32], in mfma A layout.
    //
    // Caller MUST have called p1_vmem_to_staging_chunk<kChunkIdx, kBufIdx>
    // earlier (no waitcnt in between is fine; this helper drains vmcnt first
    // to ensure the staging bytes and scale dwords are valid before the cvt).
    template <uint32_t kChunkIdx, uint32_t kBufIdx, uint32_t GPR_NOPE_VGPR_START>
    __device__ __forceinline__ static void
    p1_staging_to_vgpr_chunk(const uintptr_t p_lds_warp_staging,
                             const uint32_t s_dw,
                             const float q_scale_log2)
    {
        static_assert(kChunkIdx < 7u, "p1_staging_to_vgpr_chunk: bad kChunkIdx (0..6).");
        static_assert(kBufIdx < kP1NumStagingBuffers, "p1_staging_to_vgpr_chunk: bad kBufIdx.");

        constexpr uint32_t kStagingI      = kBufIdx * kP1StagingBytesPerWarp;
        constexpr uint32_t kVgprChunkBase = GPR_NOPE_VGPR_START + 8u * kChunkIdx;

        uint32_t lane_idx    = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uint32_t row_in_warp = lane_idx & 15u; // 0..15 (= row in warp tile)

        // Swizzle-aware addressing (mirror of p1_vmem_to_staging_chunk writer).
        // Logical col sub-tile within chunk indexes a 16-col slot.
        // Physical col sub-tile: C_phys = C_log XOR (S<<1), where
        // S = (row_in_warp>>2) & 1 (=1 on rows 4..7 and 12..15). XOR-ing bit 1
        // of cb (not bit 0) is what breaks ds_read_b128's non-linear cycle 0
        // (L,L+20) collision pair -- see writer comment for the algebra.
        //
        // Sub-tile-of-8 perm [0,2,4,6,1,3,5,7] applies to the K-side LDS for
        // this wave-tile (Site 2 / Site 4 / Site 3 / Site 5). For QK to stay
        // lockstep, this chunk's Q VGPRs (= mfma A-tile) must hold the K-LDS
        // matching data:
        //   mfma at sub-block s of wave-tile  ->  Q lane (row r, col_band cb
        //   = lane>>4) needs Q-DATA sub-tile (2*cb + s).
        // Iter j ∈ {0,1} maps directly to sub-block s = j (iter 0 = cols 0..31
        // of chunk = sub-block 0; iter 1 = cols 32..63 = sub-block 1).
        // Therefore lane needs data col base = 16*cb + 8*j, decomposing into
        //   C_log = cb ;  byte_off = j * 8.
        // Both iters share C_phys, so the +8 byte delta from iter 0 to iter 1
        // folds into the ds_read_b64 imm offset -- no second addr VGPR.
        const uint32_t S        = (lane_idx >> 2) & 1u; // = (row_in_warp>>2)&1
        const uint32_t cb       = (lane_idx >> 4) & 3u; // 0..3
        const uint32_t C_log    = cb;                   // 0..3
        const uint32_t C_phys   = C_log ^ (S << 1);
        const uint32_t byte_off = 0u; // iter 0 base

        // kStagingI is still folded into the ds_read imm `offset:` field so
        // the two staging buffers share these per-lane address computations.
        // Both iters share C_phys; the iter1 +8-byte delta folds into the
        // ds_read_b64 imm offset (combined with kStagingI).
        const uintptr_t addr_base =
            p_lds_warp_staging + row_in_warp * kP1ChunkCols + C_phys * 16u + byte_off;

        // CALLER CONTRACT: drain vmcnt to the level appropriate for this
        // (kChunkIdx, kBufIdx) before calling. buffer_load_lds completion is
        // tracked in vmcnt only (NOT lgkmcnt); the matching scale
        // buffer_load_ubyte from p1_vmem_to_staging_chunk is also vmcnt.

        // 16 fp8/lane (both iters) via a single ds_read_b128. The two iters
        // are contiguous (offset 0 and +8 within the per-lane row chunk), so
        // they fold into one b128 load. Bank analysis: per-lane quad =
        // (addr>>4)&15 = (row&3)*4 + C_phys. With C_phys = cb ^ (S<<1) the
        // 16 lanes per non-linear b128 cycle land on distinct quads in
        // {0..15} -- conflict-free on all 4 cycles.
        const hk::u32x4 fp8 = hkm::ds_read_b128<hk::u32x4>(static_cast<uint32_t>(addr_base),
                                                           static_cast<int>(kStagingI));

        // V4 shares one E8M0 scale across the full 64-col chunk -> single
        // scale_f for both 32-col mfma A-tiles (iter0 cols [0,32), iter1
        // cols [32,64)). Fold the softmax temperature (sm_scale * log2e) into
        // the cvt's fp32 scale operand here so the QK scores arrive already
        // temperature-scaled and in log2 domain -- this removes the per-tile
        // softmax_scale_p multiply and softmax_p1's log2e multiply. Single
        // rounding (the cvt rounds once to bf16), no extra instructions.
        const float scale_f = hk_mla::e8m0_to_f32(s_dw) * q_scale_log2;

        // Drain lgkmcnt: ds_read fp8 results must be ready before cvt builtin
        // consumes them. Pair with sched_barrier(0) -- the cvt is a pure-SSA
        // intrinsic and is otherwise free to be hoisted past a bare s_waitcnt
        // (verified by ISA inspection on KvManager8to16bitsV1). NOTE: LLVM's
        // SIInsertWaitcnts does NOT auto-insert lgkmcnt for the inline-asm
        // cvt consumer here -- the asm is opaque and the dependency on the
        // ds_read result isn't visible. The manual wait is load-bearing.
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
        __builtin_amdgcn_sched_barrier(0);

        // Direct cvt into the caller-pinned VGPR slots (no v_mov trampoline).
        // Per iter: dword 0 -> bf16 dw[0,1] (cols 0..3), dword 1 -> bf16
        // dw[2,3] (cols 4..7). opsel false/true selects low/high fp8 pair.
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 0u, false>(fp8[0], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 1u, true>(fp8[0], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 2u, false>(fp8[1], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 3u, true>(fp8[1], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 4u, false>(fp8[2], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 5u, true>(fp8[2], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 6u, false>(fp8[3], scale_f);
        cvt_scalef32_pk_bf16_fp8_pinned<kVgprChunkBase + 7u, true>(fp8[3], scale_f);
    }

    // Scale one bf16x2 dword by an fp32 scalar: unpack each bf16 to fp32
    // (bf16 = high 16 bits of fp32, so element k -> bits<<16), multiply, then
    // repack with v_cvt_pk_bf16_f32 (element 0 -> low half, element 1 -> high).
    // Used by the RoPE Q load to fold sm_scale*log2e into bf16 RoPE Q.
    __device__ __forceinline__ static uint32_t scale_bf16_pair(uint32_t b, float s)
    {
        const float f0 = __builtin_bit_cast(float, (b & 0x0000ffffu) << 16) * s;
        const float f1 = __builtin_bit_cast(float, (b & 0xffff0000u)) * s;
        uint32_t out;
        asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(out) : "v"(f0), "v"(f1));
        return out;
    }

    // ---- Phase 2: RoPE bf16 chunk -> LDS (load -> VGPR, scale, ds_write) ----
    // Lane mapping (matches the row-major-within-sub-block layout the QK
    // ds_read_b128 expects): lane T writes 16 B = 8 bf16 to row T/4, cols
    // (T%4)*8..+8 of one 16x32 sub-block. Two 16-B halves cover the 64-col
    // RoPE region. Unlike NoPE (cvt-at-store from fp8), RoPE is already bf16,
    // but we must fold the softmax temperature (sm_scale*log2e) in, so this
    // can no longer be a direct vmem->LDS copy: load to VGPR, scale each bf16
    // pair, then ds_write_b128 to the same swizzled LDS slots. (The scale is
    // folded HERE, in the stage, not at the prologue VGPR read -- a scale-at-read
    // path was tried and perturbed the kernel schedule enough to re-expose a
    // latent split_out race, so it was dropped.)
    __device__ __forceinline__ static void p2_load_rope_chunk(const q_rope_t* p_q_rope_warp,
                                                              const uintptr_t p_lds_q,
                                                              const uint32_t warp_idx,
                                                              const float q_scale_log2)
    {
        constexpr uint32_t kColTileLo = kRopeColTileLo; // 0 when all NoPE in VGPR
        constexpr uint32_t kColTileHi = kRopeColTileHi; // 1 when all NoPE in VGPR

        uint32_t lane_idx    = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uint32_t row_in_warp = lane_idx >> 2; // 0..15
        const uint32_t col_quad    = lane_idx & 3u; // 0..3

        // Row-conditional half-swap (vmem-load side, RoPE): swap col_quad
        // halves (XOR bit 1) on sub-tile-rows 1 & 3 (rows 4..7 and 12..15).
        const uint32_t col_quad_swz = col_quad ^ (((row_in_warp >> 2) & 1u) << 1);

        // Sub-tile-of-8 perm [0,2,4,6,1,3,5,7] (vmem-src side). LDS sub-tile k
        // <- data sub-tile perm^{-1}(k): (sb=0, q) <- data cols 16q..+7,
        // (sb=1, q) <- data cols 16q+8..+15. Both lo & hi share v_off base =
        // row*kRopeStride + col_quad_swz*32 B; the hi half adds +16 B in the gmem
        // source. The explicit ds_write addresses the LDS dst directly, so the
        // +16 applies only to the gmem offset (s_os) and the two halves land at
        // their natural swizzled LDS slots.
        const uint32_t v_off_lo = row_in_warp * kRopeStride + col_quad_swz * 32u;

        const uint32_t lds_off = lane_idx * 16u;

        const uintptr_t p_dst_lo = p_lds_q + sub_block_byte_offset(warp_idx, kColTileLo) + lds_off;
        const uintptr_t p_dst_hi = p_lds_q + sub_block_byte_offset(warp_idx, kColTileHi) + lds_off;

        // Process each 16-B half independently (load -> scale -> ds_write) before
        // touching the next, so only one u32x4 + its fp32 temps are live at a
        // time. Byte-addressed gmem (uint8) so v_off_lo / the +16 stay in bytes;
        // load<16> pulls 16 B.
        auto g_q_rope =
            opus::make_gmem<uint8_t>(reinterpret_cast<const uint8_t*>(p_q_rope_warp));
        auto scale_and_store = [&](uint32_t s_os, uintptr_t p_dst) {
            hk::u32x4 v =
                __builtin_bit_cast(hk::u32x4, g_q_rope.template load<16>(v_off_lo, s_os));
            opus::static_for<4>(
                [&](auto i) { v[i.value] = scale_bf16_pair(v[i.value], q_scale_log2); });
            hkm::ds_write_b128(v, static_cast<uint32_t>(p_dst), 0);
        };
        scale_and_store(0u, p_dst_lo);
        scale_and_store(16u, p_dst_hi);
    }

    public:
    // Max kColInRecord subtracted from the LDS dst pointer in
    // p1_vmem_to_staging_chunk: chunks 0..3 use 0/64/128/192. The kernel MUST
    // allocate this many bytes of dummy padding BEFORE p_lds_q so warp 0's
    // staging (= p_lds_q + 0) doesn't underflow when chunk 3 subtracts 192.
    // Without the pad, m0 wraps mod 2^32 and the LDS store lands outside the
    // LDS allocation (silently dropped on warp 0, the only warp where
    // staging - kColInRecord goes negative).
    // All 7 NoPE chunks (0..6) can go through p1 now; chunk 6 pre-subtracts
    // 6*64=384 from the staging dst, so the pad must cover the max chunk col.
    static constexpr uint32_t kLdsHeadPadBytes = 6u * kP1ChunkCols; // 384

    __device__ QManager8to16bitsV1() {}

    // Total LDS footprint = the per-warp staging region, reused by RoPE. With all
    // NoPE in VGPR the final region holds RoPE only -> kFinalLdsBytes = 16 KB.
    __device__ __forceinline__ static constexpr uint32_t get_lds_size_in_byte()
    { return kFinalLdsBytes; }

    // Loads Q from VRAM into pinned VGPRs (NoPE half 0:256) and the bf16 LDS
    // region (NoPE half 256:448 + RoPE 0:64, each 192/64 cols of bf16).
    //
    //   GPR_NOPE_VGPR_START : start of the 32-vgpr range that holds Q[:, 0:256]
    //                         in bf16 (16 rows x 256 cols / 64 lanes / 2
    //                         elem-per-vgpr = 32). Slot layout:
    //                           [GPR_NOPE_VGPR_START + 8*chunk + 4*iter + i]
    //                         = bf16 mfma A-tile for QK iter (2*chunk + iter).
    //   p_lds_q             : start of the 64 KB bf16 LDS region. Phase 1 also
    //                         uses the per-warp staging, then RoPE reuses it.
    // All kP1MaxChunks NoPE chunks (cols 0:448) go to VGPR via p1; RoPE goes
    // through the (reused) staging LDS then to VGPR in the prologue.
    template <uint32_t GPR_NOPE_VGPR_START>
    __device__ __forceinline__ void load_q(const q_nope_t* p_q_buffer_nope,
                                           const q_rope_t* p_q_buffer_rope,
                                           const int32_t num_qheads,
                                           const int32_t warp_idx,
                                           const int32_t qo_start,
                                           const uintptr_t p_lds_q,
                                           const float q_scale_log2)
    {
        // Per-warp base pointers in vmem (each warp owns kTileM=16 rows).
        // Q layout: [total_q, num_qheads, kQkPackedNopeQElems] (the
        // num_qheads/kTileM x kTileM axes from gl_q_nope collapse to one
        // contiguous num_qheads axis since adjacent strides are unit).
        const q_nope_t* p_q_warp      = p_q_buffer_nope +
                                        qo_start * num_qheads * T::kQkPackedNopeQElems +
                                        warp_idx * T::kTileM * T::kQkPackedNopeQElems;
        const q_rope_t* p_q_rope_warp = p_q_buffer_rope +
                                        qo_start * num_qheads * T::kQkRopeHeadDim +
                                        warp_idx * T::kTileM * T::kQkRopeHeadDim;

        const uintptr_t p_lds_warp_staging = p1_warp_staging_base(p_lds_q, warp_idx);

        // ---- Phase 1: VGPR half (Q[:, 0:256]) ----
        // Double-buffered pipeline: prefetch chunks 0,1 in parallel; for each
        // chunk, drain to VGPR before issuing the next prefetch into its buf
        // (chunks 2,3 reuse bufs 0,1 respectively, so the prior chunk MUST be
        // consumed first). This keeps 1 prefetch in flight while the previous
        // chunk's cvt runs. The single per-chunk scale dword (V4 has one scale
        // per 64-col tile) is returned by p1_vmem_to_staging_chunk and held in
        // s_X across the ladder until p1_staging_to_vgpr_chunk consumes it.
        // Phase 1 vmcnt budget: each p1_vmem_to_staging_chunk issues 2 vmem
        // ops (dwordx4_lds + scale ubyte). buffer_load_lds completion is in
        // vmcnt only (NOT lgkmcnt). buf 0/1 are double-buffered: chunk c+2
        // overwrites buf used by chunk c, so before reusing a buf we also
        // need the prior consume's ds_read drained (lgkmcnt=0).
        uint32_t s_0, s_1, s_2, s_3;
        p1_vmem_to_staging_chunk<0, 0>(p_q_warp, p_lds_warp_staging, s_0); // out=2
        p1_vmem_to_staging_chunk<1, 1>(p_q_warp, p_lds_warp_staging, s_1); // out=4
        // Drain c0 (oldest 2 of 4 outstanding).
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/-1, /*vmcnt=*/2));
        __builtin_amdgcn_sched_barrier(0);
        p1_staging_to_vgpr_chunk<0, 0, GPR_NOPE_VGPR_START>(p_lds_warp_staging, s_0, q_scale_log2);
        // c0 ds_read must be done before c2 overwrites buf 0.
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
        __builtin_amdgcn_sched_barrier(0);
        p1_vmem_to_staging_chunk<2, 0>(p_q_warp, p_lds_warp_staging, s_2); // out=4 (c1+c2)
        // Drain c1 (oldest 2 of 4).
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/-1, /*vmcnt=*/2));
        __builtin_amdgcn_sched_barrier(0);
        p1_staging_to_vgpr_chunk<1, 1, GPR_NOPE_VGPR_START>(p_lds_warp_staging, s_1, q_scale_log2);
        // c1 ds_read must be done before c3 overwrites buf 1.
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
        __builtin_amdgcn_sched_barrier(0);
        p1_vmem_to_staging_chunk<3, 1>(p_q_warp, p_lds_warp_staging, s_3); // out=4 (c2+c3)
        // Drain c2.
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/-1, /*vmcnt=*/2));
        __builtin_amdgcn_sched_barrier(0);
        p1_staging_to_vgpr_chunk<2, 0, GPR_NOPE_VGPR_START>(p_lds_warp_staging, s_2, q_scale_log2);
        // Drain c3.
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/-1, /*vmcnt=*/0));
        __builtin_amdgcn_sched_barrier(0);
        p1_staging_to_vgpr_chunk<3, 1, GPR_NOPE_VGPR_START>(p_lds_warp_staging, s_3, q_scale_log2);

        // ---- Hi NoPE chunks 4..kP1MaxChunks-1 (Q[:,256:448]) -> VGPR via p1 ----
        // chunk c lands at GPR_NOPE_VGPR_START + 8*c. Each reuses staging buf c%2;
        // drain prior buf reads + its own vmem load before consume.
        opus::static_for<kP1MaxChunks - 4u>([&](auto cc) {
            constexpr uint32_t c   = 4u + cc.value;
            constexpr uint32_t buf = c % 2u;
            uint32_t s_c;
            __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
            __builtin_amdgcn_sched_barrier(0);
            p1_vmem_to_staging_chunk<c, buf>(p_q_warp, p_lds_warp_staging, s_c);
            __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/-1, /*vmcnt=*/0));
            __builtin_amdgcn_sched_barrier(0);
            p1_staging_to_vgpr_chunk<c, buf, GPR_NOPE_VGPR_START>(
                p_lds_warp_staging, s_c, q_scale_log2);
        });

        // ---- RoPE -> LDS (reusing the now-consumed staging region) -> VGPR ----
        // RoPE's ds_write lands in the same per-warp bytes p1 staging just used,
        // so drain the last staging ds_read (lgkmcnt=0) before overwriting them.
        const uint32_t warp_idx_u = static_cast<uint32_t>(warp_idx);
        __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
        __builtin_amdgcn_sched_barrier(0);
        p2_load_rope_chunk(p_q_rope_warp, p_lds_q, warp_idx_u, q_scale_log2);
    }

    // QK A-tile load from the bf16 final Q LDS region. Loads one 16 x 32 bf16
    // sub-block (= 4 vgprs/lane) into RT in mfma_f32_16x16x32_bf16 A layout.
    //   kColTile selects the col tile inside Q[:, 256:512] (0..7, where 0..5
    //   are NoPE Q cols 256..447, 6..7 are RoPE Q cols 448..511).
    //   warp_idx selects the wave's 8 KB final region (wave-major layout).
    // The per-wave 8 KB byte stride is dynamic (warp_idx * kWarpFinalBytes,
    // scalar) so it cannot fold into the ds_read offset immediate; the col-tile
    // bytes (kColTile * 1024) fold via the 16-bit ds_read offset:.
    template <uint32_t kColTile, hkdart::all RT>
    __device__ __forceinline__ static void
    load_q_lds_to_gpr(RT& dst, const uintptr_t p_lds_q, const uint32_t warp_idx)
    {
        static_assert(kColTile < kFinalLdsColTiles, "load_q_lds_to_gpr: kColTile out of range.");

        constexpr uint32_t kMfmaRows       = 16;
        constexpr uint32_t kMfmaElemPerThr = 8;

        uint32_t lane_idx = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uint32_t row      = lane_idx % kMfmaRows;
        const uint32_t col      = (lane_idx / kMfmaRows) * kMfmaElemPerThr;

        // Site C bank-conflict swizzle (reader side): XOR byte-in-sub-block
        // by 32 on sub-tile-rows 1 & 3 (rows 4..7 and 12..15 of the 16-row
        // sub-block). Identical pattern to KvManager8to16bitsV1 readers
        // (load_k_to_gpr, load_transposed_v_to_gpr) so both managers share
        // one bank-arithmetic invariant. Writers mirror the swap so the
        // bf16 LDS contents match what the reader pulls.
        const uint32_t swz = ((row >> 2) & 1u) << 5;
        const uint32_t in_sb_byte =
            row * (kSubBlockCols * sizeof(hk::bf16)) + (col * sizeof(hk::bf16) ^ swz);

        constexpr uint32_t kColTileBytes = kColTile * kSubBlockBytes;

        using range_type = hkdart::get_nth_range_t<typename RT::register_ranges, 0>;
        static_assert(range_type::lo + 3 == range_type::hi,
                      "ds_read_b128 requires 4 consecutive registers");

        const uintptr_t p_lds_q_lane = p_lds_q + warp_idx * kWarpFinalBytes + in_sb_byte;
        hkm::ds_read_b128<range_type::lo>(static_cast<uint32_t>(p_lds_q_lane), kColTileBytes);
    }

};

// V4.0 KV manager: per-token VMEM layout = NoPE 448 B FP8 + dup-E8M0 16 B
// + pad 112 B = 576 B in `gl_kv_nope`; RoPE is BF16 in a *separate* tensor
// `gl_kv_rope` (kQkRopeHeadDim=64 elements per token).  Per spec wave-to-tile
// map (Option 2), only waves 5 and 7 issue the RoPE buffer_load_dwordx4 lds.
//
// IMPORTANT (v4 vs v3.2): the FP8->BF16 cvt happens on the *load path*, BEFORE
// the LDS write.  LDS stores BF16 only; load_k_to_gpr/load_transposed_v_to_gpr
// are plain ds_read of bf16 (no cvt at read).  This means:
//   - The E8M0 scale needs no LDS storage -- it lives only briefly in VGPR
//     between vmem fp8 read and the cvt+scale -> ds_write bf16.
//   - No padding is needed (MI35x has 64 LDS banks, twice MI300).
//
// V4.0 KV manager: vmem fp8 (NoPE) + bf16 (RoPE)  ->  LDS bf16 (cvt at *store* time).
//
// LDS layout per pong (32 KB at kBlockN=32, kQkHeadDim=512):
//   The 32 x 512 bf16 region is viewed as 32 sub-blocks of 16 x 32 bf16 each (1024 B/sub-block),
//   stored in COLUMN-MAJOR sub-block order.  Sub-block (row_tile, col_tile) lives at byte offset
//       (col_tile * 2 + row_tile) * 1024
//   so sub-blocks are written/read as:
//       (0,0), (1,0), (0,1), (1,1), (0,2), (1,2), ..., (0,15), (1,15).
//   row_tile in {0, 1}      = which 16-row half of the 32-row tile.
//   col_tile in {0..15}     = which 32-col strip of the 512-col tile.  Strips 0..13 are NoPE,
//                             strips 14..15 (cols 448..511) are RoPE.
//
// Loading is interleaved across 4 chunks of 32 x 128 cols (spec section 5.3.1):
//     chunk c covers cols [c*128, (c+1)*128) = col_tiles {4c, 4c+1, 4c+2, 4c+3}.
//   NoPE source = packed fp8 in `kv_buf_nope` (576 B/token: 448 fp8 + 16 dup-E8M0 + 112 pad).
//   RoPE source = bf16 in `kv_buf_rope` (separate tensor, 64 bf16 = 128 B/token).
//   Chunks 0..2 are pure NoPE for all 8 waves.
//   Chunk 3 spans NoPE cols 384..447 (col_tiles 12, 13) AND RoPE cols 448..511 (col_tiles 14, 15);
//     all waves load the NoPE half, but only waves 5 & 7 load the RoPE half (which is bf16,
//     so it goes straight to LDS without cvt).
//
// Two-phase per chunk (so loads overlap with QK MFMAs):
//   1. prefetch_nope_chunk_to_vgpr<chunk>(...)   -- issue buffer_load to lane VGPRs
//                                                   for fp8 + E8M0 scale of this chunk.
//   2. cvt_store_nope_chunk_to_lds<chunk>(...)   -- s_waitcnt vmcnt, run
//                                                   __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8,
//                                                   ds_write_b128 into the bf16 LDS sub-blocks.
// For chunk 3 only:
//   3. async_load_rope_to_lds(...)               -- waves {5,7} buffer_load_dwordx4 lds: directly
//                                                   to col_tiles {14, 15} of the LDS pong.
template <typename T>
class KvManager8to16bitsV1
{
    private:
    using kv_nope_t = typename T::kv_nope_t;
    using kv_rope_t = typename T::kv_rope_t;
    static_assert(std::is_same_v<kv_nope_t, hk::fp8e4m3>,
                  "KvManager8to16bitsV1: kv_nope_t must be fp8e4m3.");
    static_assert(std::is_same_v<kv_rope_t, hk::bf16>,
                  "KvManager8to16bitsV1: kv_rope_t must be bf16.");

    static_assert(T::kBlockN == 32, "KvManager8to16bitsV1: only kBlockN=32 supported in Gen.1.");
    static_assert(T::kQkNopeHeadDim == 448, "KvManager8to16bitsV1: NOPE width must be 448.");
    static_assert(T::kQkRopeHeadDim == 64, "KvManager8to16bitsV1: ROPE width must be 64.");
    static_assert(T::kQkHeadDim == 512,
                  "KvManager8to16bitsV1: kQkHeadDim must be 512 (NOPE+ROPE).");
    static_assert(T::kNumWarps == 8, "KvManager8to16bitsV1: requires 8 warps.");

    public:
    // ---- Sub-block geometry ------------------------------------------------
    // Each LDS sub-block is 16 rows x 32 cols of bf16 = 1024 B.
    static constexpr uint32_t kSubBlockRows    = 16;
    static constexpr uint32_t kSubBlockCols    = 32;
    static constexpr uint32_t kSubBlockBytes   = kSubBlockRows * kSubBlockCols * sizeof(hk::bf16);
    static constexpr uint32_t kNumRowTiles     = T::kBlockN / kSubBlockRows;        // 2
    static constexpr uint32_t kNumColTiles     = T::kQkHeadDim / kSubBlockCols;     // 16
    static constexpr uint32_t kNumColTilesNope = T::kQkNopeHeadDim / kSubBlockCols; // 14
    static constexpr uint32_t kNumColTilesRope =
        (T::kQkHeadDim - T::kQkNopeHeadDim) / kSubBlockCols; // 2

    // ---- Tile geometry -----------------------------------------------------
    // Two 32x256 half-tiles cover the full 32x512 KV pong. Tile 0 = cols [0,256)
    // (all FP8 NoPE). Tile 1 = cols [256,512) (FP8 NoPE for waves 0..4,6 in
    // cols [256,448); BF16 RoPE for waves 5,7 in cols [448,512)).
    static constexpr uint32_t kNumTiles                = 2;
    static constexpr uint32_t kTileCols                = T::kQkHeadDim / kNumTiles; // 256
    static constexpr uint32_t kColTilesPerTile         = kTileCols / kSubBlockCols; // 8
    static constexpr uint32_t kWaveColTilesPerWaveTile = 2u; // 16x64 = 2x(16x32)
    static constexpr uint32_t kWaveTileCols = kWaveColTilesPerWaveTile * kSubBlockCols; // 64

    // Total bf16 bytes in LDS for one pong.
    __device__ __forceinline__ static constexpr uint32_t get_lds_size_in_byte()
    {
        return T::kBlockN * T::kQkHeadDim * sizeof(hk::bf16); // 32 KB
    }

    // Byte offset of LDS sub-block (row_tile, col_tile) inside one pong.
    // Col-major sub-block order: (0,0),(1,0),(0,1),(1,1),...,(0,15),(1,15).
    __device__ __forceinline__ static constexpr uint32_t
    sub_block_byte_offset(const uint32_t row_tile, const uint32_t col_tile)
    { return (col_tile * kNumRowTiles + row_tile) * kSubBlockBytes; }

    // ---- Wave -> sub-tile map (spec section 4.2 Option 2, branchless) ------
    // Per 32x256 half-tile, the 8 waves partition the 2 row-tiles x 4 col-tiles
    // grid via:
    //   row_tile = (warp_idx >> 1) & 1;
    //   col_tile = ((warp_idx >> 1) & 2) | (warp_idx & 1);
    // Waves 5 and 7 always land on col_tile == 3 (the last 16x64 sub-tile), which
    // for tile 1 is the BF16 RoPE region [448,512) and is loaded by a different
    // path. See load_kv_tile_to_lds() for the merged dispatch.
    __device__ __forceinline__ static constexpr uint32_t wave_row_tile(const uint32_t warp_idx)
    { return (warp_idx >> 1) & 1u; }
    __device__ __forceinline__ static constexpr uint32_t
    wave_col_tile_in_tile(const uint32_t warp_idx)
    { return ((warp_idx >> 1) & 2u) | (warp_idx & 1u); }

    // True for the two waves that issue RoPE buffer_loads in tile 1.
    // Wave 5 covers row_tile 0 (rows 0..15) RoPE; wave 7 covers row_tile 1 RoPE.
    // Each wave does 2 x dwordx4/lane (32 B/lane) = full 16 x 64 bf16 patch.
    __device__ __forceinline__ static constexpr bool wave_is_rope_owner(const uint32_t warp_idx)
    { return (warp_idx == 5u) || (warp_idx == 7u); }

    // ---- Public API: addressing helpers used by the kernel body ------------
    // Per-warp logical row inside the 32-row KV tile (range [0, 31]).
    // Per-lane row index in the 32-row KV tile. Maps the wave-to-tile partition
    // (row_tile = (warp_idx>>1)&1; lanes 0..63 cover 16 rows x 4 col_groups) to
    // an absolute row [0, 32) in the tile. The kernel body then adds
    // kv_tile_start to get a row index in the *flat* KV-token space.
    //
    // Each row of the 32-row tile is covered by 4 lanes (col_group 0..3 reading
    // 16 fp8 cols each = 64 cols total per wave-col-tile). row_in_warp = lane>>2
    // gives the within-wave row 0..15; row_tile*16 selects the upper-half or
    // lower-half of the 32-row tile.
    __device__ __forceinline__ static uint32_t get_kv_ld_row_base_idx(const int32_t warp_idx)
    {
        const uint32_t lane_idx    = opus::lane_id();
        const uint32_t row_tile    = (static_cast<uint32_t>(warp_idx) >> 1) & 1u; // 0 or 1
        const uint32_t row_in_warp = lane_idx >> 2;                               // 0..15
        return row_tile * 16u + row_in_warp;                                      // 0..31
    }

    // Per-lane column byte offset into the packed 576 B/token KV-NoPE record.
    __device__ __forceinline__ static uint32_t get_kv_ld_col_base(const int32_t warp_idx)
    {
        // TODO(v4.0 Phase 2).
        (void)warp_idx;
        return 0;
    }

    public:
    // Per-lane prefetch carrier for one 32x256 half-tile (NoPE branch only).
    // Lives in VGPRs across the gap between prefetch_kv_tile() and
    // cvt_and_store_kv_tile(); the gap is where the kernel body hides vmem
    // latency by issuing QK MFMAs. The RoPE branch (waves 5,7 in tile 1)
    // does NOT touch this struct -- its data is delivered by buffer_load
    // dwordx4 lds direct to LDS during prefetch.
    struct KvTilePrefetch
    {
        hk::u32x4 nope_dw; // 16 fp8 = 4 dw
        uint32_t scale_dw; // E8M0 scale byte, zero-extended to dw
    };

    // ---- Phase A: NoPE prefetch (issue VRAM loads into VGPR carrier) -------
    // 1 x buffer_load_dwordx4 (fp8 nope -> prefetch_out.nope_dw) + 1 x
    // buffer_load_ubyte (E8M0 scale -> prefetch_out.scale_dw). VGPR-landing
    // only (no LDS write), so safe to issue ahead of the cross-warp barrier.
    // On tile 1 the RoPE-owner waves (5,7) carry no NoPE data -> no-op here
    // (their RoPE half is issued by prefetch_kv_rope after the barrier).
    //
    // No s_waitcnt is issued here -- the caller chooses when to wait.
    template <uint32_t kRowOffset, uint32_t kColOffset, bool kCheckBoundary, bool kIsRopeWarp>
    __device__ __forceinline__ static void prefetch_kv_nope(const uint32_t warp_idx,
                                                            const kv_nope_t* p_kv_buf_nope,
                                                            const int32_t row_kv_ld,
                                                            KvTilePrefetch& prefetch_out)
    {
        static_assert(kRowOffset == 0u,
                      "prefetch_kv_nope: kRowOffset must be 0 -- a tile spans all 32 rows.");
        static_assert((kColOffset % kTileCols == 0u) && (kColOffset < T::kQkHeadDim),
                      "prefetch_kv_nope: kColOffset must be 0 or kTileCols (=256).");

        constexpr uint32_t kTileIdx = kColOffset / kTileCols;
        constexpr bool kIsNoPEPath = (kTileIdx == 0u) || (kIsRopeWarp == false);

        uint32_t lane_idx = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uint32_t col_group        = lane_idx & 3u; // 0..3
        const uint32_t col_tile_in_tile = wave_col_tile_in_tile(warp_idx);
        const bool in_bounds            = (kCheckBoundary == false) || (row_kv_ld >= 0);

        if constexpr(kIsNoPEPath)
        {
            // ---------------- NoPE prefetch ----------------
            constexpr uint32_t kPackedStride = T::kQkPackedNopeKvElems * sizeof(kv_nope_t); // 576

            const uint64_t as_u64 =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_kv_buf_nope));
            const hk::buffer_resource br = hk::make_buffer_resource(as_u64, 0xffffffff, 0x00020000);

            // Address split (NoPE): row_kv_ld is *per-lane* (each lane covers a
            // distinct row of the 32-row KV tile, see get_kv_ld_row_base_idx),
            // so it MUST live in v_offset -- routing it via s_offset would force
            // v_readfirstlane and collapse all lanes onto row 0.
            //   v_offset (per-lane)   = row_kv_ld * 576 + col_group_swz * 16
            //   s_offset (wave-unif)  = col_tile_in_tile * 64
            //   i_offset (compile-tm) = kTileIdx * 256
            //
            // Bank-conflict swizzle (vmem-load side, Method 2): for rows whose
            // sub-tile-row is odd (rows 4..7, 12..15) swap the 16 B chunk that
            // this lane loads with its in-pair neighbour (col_group XOR 1).
            // Pairs with the matching XOR on load_k_to_gpr/load_transposed_v_to_gpr
            // readers, and lets cvt_and_store_kv_tile keep the LDS dst address
            // straight -- same pattern QManager8to16bitsV1 ships.
            //
            // NOTE: kept on HK (hkm::buffer_load_dwordx4 + buffer_load_ubyte).
            // Opus migration measured +18% regression at b=33 c=63333 -- the
            // i_offset is load-bearing on this hot path; folding it into s_os
            // costs an SGPR add per iter and stretches the prefetch chain.
            const uint32_t col_group_swz = col_group ^ (((lane_idx >> 4) & 1u) << 1);
            const uint32_t v_off_nope =
                in_bounds ? (static_cast<uint32_t>(row_kv_ld) * kPackedStride + col_group_swz * 16u)
                          : 0u;
            const uint32_t s_off_nope = col_tile_in_tile * kWaveTileCols;
            constexpr int i_off_nope  = static_cast<int>(kTileIdx * kTileCols);

            // Address split (scale): also per-lane (each lane consumes the scale
            // for its own row in cvt_and_store_kv_tile). 1 byte zero-extended.
            //   v_offset (per-lane)   = row_kv_ld * 576 + col_tile_in_tile * 2
            //   s_offset (wave-unif)  = 0
            //   i_offset (compile-tm) = 448 + kTileIdx * 8
            // Per kTileIdx we cover kColTilesPerTile (=8) sub-block-cols of 32
            // V-cols each = 256 V-cols = 4 scale tiles. Each scale tile occupies
            // 2 bytes (duplicated), so skip 4*2 = 8 = kColTilesPerTile bytes per
            // kTileIdx (since 1 sub-block-col is half a scale tile = 1 dup byte).
            constexpr uint32_t kScaleBaseOff = 448u;
            const uint32_t v_off_scale =
                in_bounds ? static_cast<uint32_t>(row_kv_ld) * kPackedStride : 0u;
            // col_tile_in_tile is wave-uniform -> route via s_offset so it
            // doesn't sit in the per-lane v_offset (saves one v_add).
            const uint32_t s_off_scale = col_tile_in_tile * 2u;
            constexpr int i_off_scale =
                static_cast<int>(kScaleBaseOff + kTileIdx * kColTilesPerTile);

            prefetch_out.nope_dw =
                in_bounds
                    ? hkm::buffer_load_dwordx4(br, v_off_nope, /*s_off=*/s_off_nope, i_off_nope)
                    : hk::u32x4{0u, 0u, 0u, 0u};
            prefetch_out.scale_dw =
                in_bounds
                    ? hkm::buffer_load_ubyte(br, v_off_scale, /*s_off=*/s_off_scale, i_off_scale)
                    : 0u;
        }
    }

    // ---- Phase A: RoPE prefetch (vmem -> LDS direct) -----------------------
    // Only the tile-1 RoPE-owner waves (5,7) do work; every other wave (and any
    // tile-0 call) is a no-op. Writes p_lds_kv directly, so the caller MUST
    // issue this AFTER the cross-warp barrier that protects p_lds_kv.
    template <uint32_t kRowOffset, uint32_t kColOffset, bool kCheckBoundary, bool kIsRopeWarp>
    __device__ __forceinline__ static void prefetch_kv_rope(const uintptr_t p_lds_kv,
                                                            const uint32_t warp_idx,
                                                            const kv_rope_t* p_kv_buf_rope,
                                                            const int32_t row_kv_ld)
    {
        static_assert(kRowOffset == 0u,
                      "prefetch_kv_rope: kRowOffset must be 0 -- a tile spans all 32 rows.");
        static_assert((kColOffset % kTileCols == 0u) && (kColOffset < T::kQkHeadDim),
                      "prefetch_kv_rope: kColOffset must be 0 or kTileCols (=256).");

        constexpr uint32_t kTileIdx = kColOffset / kTileCols;
        constexpr bool kIsRoPEPath = (kTileIdx == 1u) && kIsRopeWarp;

        const uint32_t lane_idx  = opus::lane_id();
        const uint32_t col_group = lane_idx & 3u; // 0..3
        const bool in_bounds     = (kCheckBoundary == false) || (row_kv_ld >= 0);

        if constexpr(kIsRoPEPath)
        {
            // ---------------- RoPE prefetch (vmem -> LDS direct) ----------------
            //
            // Two buffer_load_dwordx4 lds: cover the full 16x64 bf16 RoPE patch
            // for this wave as TWO sub-blocks (row_tile, 14) and (row_tile, 15)
            // of 16 rows x 32 cols x 2 B = 1024 B each. Each call writes
            // 16 B/lane to LDS at M0 + LANE_ID*16 (the LDS per-lane stride is
            // HW-fixed). The dst pointer is the wave-uniform sub-block base
            // (no + lane_idx*16) so m0 is set from an SGPR -- no readfirstlane.
            //
            // Trick (mirrors QManager8to16bitsV1::p2_load_rope_chunk): share
            // one v_off_lo VGPR and walk to the upper half via i_off=kVStride.
            // The imm `offset:` field of buffer_load_lds advances BOTH vmem
            // (+kVStride = next 32 bf16 cols of RoPE) AND LDS (+kVStride), so
            // pre-subtract kVStride from p_dst_hi_adj to land the LDS dst at
            // sub_block_byte_offset(rt, 15) instead of sb_off(rt, 14)+kVStride.
            //
            // The prior implementation used a single shared M0 with i_off=0
            // and i_off=16, which is broken: the same M0 means Call 2 writes
            // each lane T at M0+T*16+16 = M0+(T+1)*16, overlapping Call 1's
            // lane (T+1) slot and leaving sub-block 15 entirely unwritten.
            constexpr uint32_t kRopeStride    = T::kQkRopeHeadDim * sizeof(hk::bf16); // 128
            constexpr uint32_t kRopeColTileLo = T::kQkNopeHeadDim / kSubBlockCols;    // 14
            constexpr uint32_t kRopeColTileHi = kRopeColTileLo + 1u;                  // 15
            constexpr uint32_t kVStride       = kSubBlockCols * sizeof(hk::bf16);     // 64

            const uint32_t row_tile = wave_row_tile(warp_idx);

            if(in_bounds)
            {
                // Bank-conflict swizzle (vmem-load-side, matches the LDS-dst
                // XOR-by-32 the NoPE writer applies for sub-tile-rows 1,3 of
                // each 16-row sub-block). buffer_load_lds places lane t at
                // LDS byte t*16 = (row_in_sb*64 + col_group*16), so the LDS
                // dst is HW-fixed -- we permute the vmem source instead.
                //
                // Sub-tile-of-8 perm [0,2,4,6,1,3,5,7] (vmem-src side).
                // Each LDS sub-tile k holds data sub-tile perm^{-1}(k):
                // lo (sb=14, q) <- data cols 16q..+7; hi (sb=15, q) <- data
                // cols 16q+8..+15. Base v_off_lo = row*128 + col_quad_swz*32
                // bytes; hi adds +16 bytes, routed through the buffer_load_lds
                // imm `offset:` field (adds to BOTH vmem and LDS -- we
                // pre-subtract 16 from p_dst_hi_adj to cancel on the LDS
                // side, landing the hi load at sub_block(rt, 15)).
                const uint32_t col_group_swz = col_group ^ (((lane_idx >> 4) & 1u) << 1);
                const uint32_t v_off_lo =
                    static_cast<uint32_t>(row_kv_ld) * kRopeStride + col_group_swz * 32u;

                // LDS dst is the wave-uniform sub-block base only: do NOT add
                // lane_idx*16 here. buffer_load_dwordx4 lds writes lane T at
                // m0 + T*16 automatically (HW-fixed per-lane stride), so the
                // explicit per-lane term is redundant -- and omitting it keeps
                // the address in an SGPR (m0 set via s_mov, no v_readfirstlane
                // broadcast). hi sub-block pre-subtracts kVStride (16) to cancel
                // the i_off=16 that the imm `offset:` adds to the LDS side.
                const uintptr_t p_dst_lo =
                    p_lds_kv + sub_block_byte_offset(row_tile, kRopeColTileLo);
                const uintptr_t p_dst_hi_adj =
                    p_lds_kv + sub_block_byte_offset(row_tile, kRopeColTileHi) - 16u;

                auto g_kv_rope = opus::make_gmem<uint8_t>(
                    reinterpret_cast<const uint8_t*>(p_kv_buf_rope));
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    g_kv_rope.cached_rsrc,
                    (hk::as3_uint32_ptr)(p_dst_lo),
                    /*size=*/16, v_off_lo, /*s_off=*/0u, /*i_off=*/0, /*aux=*/0);
                __builtin_amdgcn_raw_ptr_buffer_load_lds(
                    g_kv_rope.cached_rsrc,
                    (hk::as3_uint32_ptr)(p_dst_hi_adj),
                    /*size=*/16, v_off_lo, /*s_off=*/0u, /*i_off=*/16, /*aux=*/0);
            }
            else
            {
                // OOB: zero-fill both sub-blocks at the same per-lane stride
                // the in-bounds path uses (16 B/lane). Sub-blocks (rt, 14) and
                // (rt, 15) are kNumRowTiles*kSubBlockBytes = 2048 B apart --
                // fits the ds_write_b128 imm-offset field, so we reuse a
                // single addr VGPR for both writes.
                constexpr uint32_t kInterSbStride = kNumRowTiles * kSubBlockBytes; // 2048
                const hk::u32x4 zero{0u, 0u, 0u, 0u};
                const uint32_t addr_lo = static_cast<uint32_t>(
                    p_lds_kv + sub_block_byte_offset(row_tile, kRopeColTileLo) + lane_idx * 16u);
                hkm::ds_write_b128(zero, addr_lo, 0);
                hkm::ds_write_b128(zero, addr_lo, static_cast<int>(kInterSbStride));
            }
        }
    }

    // ---- Phase B: wait for prefetch loads to retire ------------------------
    // Waits only on vmcnt (drain all outstanding vmem). Pairs with
    // sched_barrier(0) because pure-SSA cvt builtins are otherwise free to be
    // hoisted past a bare `asm volatile("s_waitcnt ...")` (verified by ISA
    // inspection -- the intrinsic+sched_barrier pair is a true scheduling
    // barrier; bare inline asm is only ordered against other inline asm).
    //
    // For waves 5,7 on tile 1: no-op. Their RoPE direct vmem->LDS path is
    // synchronized later by an s_barrier (the QK consumer reads from LDS), so
    // they don't need to gate cvt+store on vmcnt here -- and their
    // cvt_and_store_kv_tile<1> is itself a no-op.
    template <uint32_t kRowOffset, uint32_t kColOffset, bool kIsRopeWarp, int32_t kVmCnt = 0>
    __device__ __forceinline__ static void wait_kv_loads(const uint32_t warp_idx)
    {
        static_assert(kRowOffset == 0u,
                      "wait_kv_loads: kRowOffset must be 0 -- a tile spans all 32 rows.");
        static_assert((kColOffset % kTileCols == 0u) && (kColOffset < T::kQkHeadDim),
                      "wait_kv_loads: kColOffset must be 0 or kTileCols (=256).");
        constexpr uint32_t kTileIdx = kColOffset / kTileCols;

        constexpr bool skip = (kTileIdx == 1u) && kIsRopeWarp;
        if constexpr(skip == false)
        {
            __builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/-1, /*vmcnt=*/kVmCnt));
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    // ---- Phase C: cvt + store (split form) ---------------------------------
    // Lets the caller interleave each cvt step + each ds_write with mfmas to
    // hide cvt latency (~2-3 VALU ops can overlap per mfma).
    //
    // Caller workflow per tile:
    //   hk::u32x4 dw;                            // single carrier reused
    //   cvt_kv_tile_step<0>(dw, prefetch, scale_f); // nope_dw[0] -> dw[0,1]
    //   cvt_kv_tile_step<1>(dw, prefetch, scale_f); // nope_dw[1] -> dw[2,3]
    //   store_kv_tile_step<R, C, 0>(p_lds_kv, warp_idx, dw); // ds_write lo
    //   cvt_kv_tile_step<2>(dw, prefetch, scale_f); // nope_dw[2] -> dw[0,1]
    //   cvt_kv_tile_step<3>(dw, prefetch, scale_f); // nope_dw[3] -> dw[2,3]
    //   store_kv_tile_step<R, C, 1>(p_lds_kv, warp_idx, dw); // ds_write hi
    //
    // dw is reused between lo and hi -- safe because the lo ds_write issues
    // the value before cvt step 2/3 overwrites it.

    // Compute the e8m0 -> fp32 scale for this tile (1 ALU op, hoist once).
    __device__ __forceinline__ static float kv_tile_scale_f(const KvTilePrefetch& prefetch_in)
    { return hk_mla::e8m0_to_f32(prefetch_in.scale_dw); }

    // kStep in [0,4): each does 2 cvts feeding 2 dwords of `dw`.
    //   kStep 0,2 -> dw[0],dw[1]   (sources nope_dw[2*(kStep&1) + 0 or 2])
    //   kStep 1,3 -> dw[2],dw[3]
    template <uint32_t kStep>
    __device__ __forceinline__ static void
    cvt_kv_tile_step(hk::u32x4& dw, const KvTilePrefetch& prefetch_in, float scale_f)
    {
        static_assert(kStep < 4u, "cvt_kv_tile_step: kStep must be 0..3");
        constexpr uint32_t kSrc   = kStep;             // nope_dw index
        constexpr uint32_t kDstLo = (kStep & 1u) * 2u; // 0 or 2
        constexpr uint32_t kDstHi = kDstLo + 1u;

        using bf16x2_v           = __attribute__((__vector_size__(4))) short;
        const hk::u32x4& nope_dw = prefetch_in.nope_dw;
        bf16x2_v r;
        r          = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(nope_dw[kSrc], scale_f, false);
        dw[kDstLo] = __builtin_bit_cast(uint32_t, r);
        r          = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(nope_dw[kSrc], scale_f, true);
        dw[kDstHi] = __builtin_bit_cast(uint32_t, r);
    }

    // kStep in {0,1}: 0 -> store lo (offset 0), 1 -> store hi (offset
    // kNumRowTiles * kSubBlockBytes = 2048).
    template <uint32_t kRowOffset, uint32_t kColOffset, uint32_t kStep, bool kIsRopeWarp>
    __device__ __forceinline__ static void
    store_kv_tile_step(const uintptr_t p_lds_kv, const uint32_t warp_idx, const hk::u32x4& dw)
    {
        static_assert(kRowOffset == 0u,
                      "store_kv_tile_step: kRowOffset must be 0 -- a tile spans all 32 rows.");
        static_assert((kColOffset % kTileCols == 0u) && (kColOffset < T::kQkHeadDim),
                      "store_kv_tile_step: kColOffset must be 0 or kTileCols (=256).");
        static_assert(kStep < 2u, "store_kv_tile_step: kStep must be 0 or 1.");
        constexpr uint32_t kTileIdx = kColOffset / kTileCols;

        constexpr bool skip = (kTileIdx == 1u) && kIsRopeWarp;
        if constexpr(skip)
        {
            return;
        }

        const uint32_t lane_idx         = opus::lane_id();
        const uint32_t row_in_tile      = lane_idx >> 2;
        const uint32_t col_group        = lane_idx & 3u;
        const uint32_t row_tile         = wave_row_tile(warp_idx);
        const uint32_t col_tile_in_tile = wave_col_tile_in_tile(warp_idx);

        const uint32_t col_tile_global_lo =
            kTileIdx * kColTilesPerTile + col_tile_in_tile * kWaveColTilesPerWaveTile;
        const uint32_t byte_in_sb = col_group << 4;

        const uintptr_t p_dst_lane = p_lds_kv +
                                     sub_block_byte_offset(row_tile, col_tile_global_lo) +
                                     row_in_tile * (kSubBlockCols * sizeof(hk::bf16)) + byte_in_sb;

        const uint32_t addr        = static_cast<uint32_t>(p_dst_lane);
        constexpr uint32_t kImmOff = kStep * (kNumRowTiles * kSubBlockBytes);
        hkm::ds_write_b128(dw, addr, kImmOff);
    }

    // ---- Convenience wrapper: non-overlapped full-pong load ----------------
    // Equivalent to: prefetch tile 0 -> prefetch tile 1 -> wait<0> -> cvt+store
    // tile 0 -> wait<1> -> cvt+store tile 1. Useful for the prologue (and for
    // any callers that don't need to interleave QK mfmas with the cvts).
    template <bool kCheckBoundary, bool kIsRopeWarp>
    __device__ __forceinline__ static void async_load_k(const uintptr_t p_lds_kv,
                                                        const uint32_t warp_idx,
                                                        const kv_nope_t* p_kv_buf_nope,
                                                        const kv_rope_t* p_kv_buf_rope,
                                                        const int32_t row_kv_ld)
    {
        KvTilePrefetch p0, p1;
        prefetch_kv_nope<0u, 0u, kCheckBoundary, kIsRopeWarp>(warp_idx, p_kv_buf_nope, row_kv_ld, p0);
        prefetch_kv_nope<0u, kTileCols, kCheckBoundary, kIsRopeWarp>(
            warp_idx, p_kv_buf_nope, row_kv_ld, p1);
        prefetch_kv_rope<0u, kTileCols, kCheckBoundary, kIsRopeWarp>(
            p_lds_kv, warp_idx, p_kv_buf_rope, row_kv_ld);

        // Full-pong cvt+store, expressed via split steps (no interleave
        // here -- the wrapper is for prologue / cold callers).
        hk::u32x4 dw;
        wait_kv_loads<0u, 0u, kIsRopeWarp, /*kVmCnt=*/2>(warp_idx);
        const float scale_f0 = kv_tile_scale_f(p0);
        cvt_kv_tile_step<0>(dw, p0, scale_f0);
        cvt_kv_tile_step<1>(dw, p0, scale_f0);
        store_kv_tile_step<0u, 0u, 0, kIsRopeWarp>(p_lds_kv, warp_idx, dw);
        cvt_kv_tile_step<2>(dw, p0, scale_f0);
        cvt_kv_tile_step<3>(dw, p0, scale_f0);
        store_kv_tile_step<0u, 0u, 1, kIsRopeWarp>(p_lds_kv, warp_idx, dw);

        wait_kv_loads<0u, kTileCols, kIsRopeWarp, /*kVmCnt=*/0>(warp_idx);
        // Tile-1 RoPE-owner waves (5,7) have no NoPE data in p1 -> skip the
        // scale+cvt+store (store already skips; the cvts would be discarded).
        if constexpr(!kIsRopeWarp)
        {
            const float scale_f1 = kv_tile_scale_f(p1);
            cvt_kv_tile_step<0>(dw, p1, scale_f1);
            cvt_kv_tile_step<1>(dw, p1, scale_f1);
            store_kv_tile_step<0u, kTileCols, 0, kIsRopeWarp>(p_lds_kv, warp_idx, dw);
            cvt_kv_tile_step<2>(dw, p1, scale_f1);
            cvt_kv_tile_step<3>(dw, p1, scale_f1);
            store_kv_tile_step<0u, kTileCols, 1, kIsRopeWarp>(p_lds_kv, warp_idx, dw);
        }
    }

    // ---- LDS -> VGPR readout for QK / PV mfmas -----------------------------
    // QK A-tile load: ds_read_b128 of one 16 x 32 bf16 sub-block into 4 vgprs.
    // (kRowOffset, kColOffset) selects which (row_tile, col_tile) of the pong;
    // the per-lane offset within the sub-block follows the mfma_f32_16x16x32_bf16
    // A-operand layout (lane = (row_in_tile, group_in_row)).
    template <uint32_t kRowOffset, uint32_t kColOffset, hkdart::all RT>
    __device__ __forceinline__ static void load_k_to_gpr(RT& dst, const uintptr_t p_lds_kv)
    {
        static_assert(kRowOffset % kSubBlockRows == 0,
                      "load_k_to_gpr: kRowOffset must be a multiple of 16.");
        static_assert(kColOffset % kSubBlockCols == 0,
                      "load_k_to_gpr: kColOffset must be a multiple of 32.");
        static_assert(kRowOffset < T::kBlockN, "load_k_to_gpr: kRowOffset out of range.");
        static_assert(kColOffset < T::kQkHeadDim, "load_k_to_gpr: kColOffset out of range.");

        // mfma_f32_16x16x32_bf16 A-operand layout: lane t holds 8 bf16 from
        //   row r = lane%16, cols [c, c+8) where c = (lane/16) * 8.
        // 8 bf16 = 4 dwords -> ds_read_b128.
        constexpr uint32_t kMfmaRows       = 16;
        constexpr uint32_t kMfmaElemPerThr = 8;

        const uint32_t lane_idx = opus::lane_id();
        const uint32_t row      = lane_idx % kMfmaRows;
        const uint32_t col      = (lane_idx / kMfmaRows) * kMfmaElemPerThr;

        // Un-swizzle: writer XORs intra-sub-block byte position by 32 on
        // sub-tile-rows 1 & 3 (rows 4..7 and 12..15) to break the 2-way bank
        // conflict; the reader applies the same XOR on the col-byte component.
        const uint32_t row_bank_swap = ((row >> 2) & 1u) << 5;
        const uint32_t in_sb_byte =
            row * (kSubBlockCols * sizeof(hk::bf16)) + ((col * sizeof(hk::bf16)) ^ row_bank_swap);

        // Constexpr sub-block selector (compiles to immediate offset).
        constexpr uint32_t kFixedOffset =
            sub_block_byte_offset(kRowOffset / kSubBlockRows, kColOffset / kSubBlockCols);

        // RT must hold a single 4-vgpr range (16 bf16 mfma A-tile = 4 vgprs/lane).
        using range_type = hkdart::get_nth_range_t<typename RT::register_ranges, 0>;
        static_assert(range_type::lo + 3 == range_type::hi,
                      "ds_read_b128 requires 4 consecutive registers");

        const uintptr_t p_lds_kv_lane = p_lds_kv + in_sb_byte;
        hkm::ds_read_b128<range_type::lo>(static_cast<uint32_t>(p_lds_kv_lane), kFixedOffset);
    }

    // PV A-tile load: ds_read_b64_tr_b16 (bf16 transpose-read) of one 16-row x 16-col
    // bf16 patch from a sub-block, results land in 2 dwords/lane in (GPR, GPR+1).
    //
    // PV math is V^T @ P^T = O^T computed via mma_ABt(oaccu, kv, p_mfma) (= kv @ p_mfma^T,
    // matching the QK convention of K^T @ Q^T = P^T). So `kv` is the A operand of
    // v_mfma_f32_16x16x32_bf16, holding V^T values reorganized into the mfma A layout.
    //
    // Within each 16-lane group, lane t's 4 bf16 (4 bf16 = 1 b64 = 2 dwords/lane) are
    // (after HW transpose):
    //   output_lane[g*16 + l] holds V[g*4+0..g*4+3, kColOffset + l]
    // for g = lane_group_idx (0..3), l = lane_in_group (0..15). I.e. each lane gets
    // 4 K-rows of one V-col. Caller stitches two row halves (kRowOffset = 0, then 16)
    // into a single mfma A operand spanning 8 K-rows (= mfma K = 0..7).
    //
    // Per-lane source address (within the selected 16x32 sub-block):
    //   in_sb_byte = (lane >> 2) * (kSubBlockCols * sizeof(bf16)) + (lane & 3) * 8
    //              = lane_row * 64 + lane_col_quad * 8
    // (row stride = 32 bf16 cols * 2 B = 64 B; each "col_quad" = 4 bf16 = 8 B.)
    //
    // Compile-time fixed_offset selects:
    //   * the sub-block (row_tile = kRowOffset/16, col_tile = kColOffset/32), and
    //   * the 16-col half within that 32-col sub-block: kColOffset%32 -> +0 or +32 B.
    //
    // Un-swizzle: writer XORs intra-sub-block byte position by 32 on rows
    // whose sub-tile-row index is odd (rows 4..7 and 12..15 within the 16-row
    // sub-block). The reader applies the same XOR. With this swizzle both
    // cycles of ds_read_b64_tr_b16 (lanes 0..31 covering rows 0..7, lanes
    // 32..63 covering rows 8..15) hit 32 distinct conflict slots -- fully
    // conflict-free. See [[v40-qlds-bank-conflict-swizzle]] for the analogous
    // QManager fix and the bank-arithmetic derivation.
    template <uint32_t kRowOffset, uint32_t kColOffset, uint32_t GPR>
    __device__ __forceinline__ static void load_transposed_v_to_gpr(const uintptr_t p_lds_v)
    {
        static_assert((kRowOffset % kSubBlockRows == 0u) && (kRowOffset < T::kBlockN),
                      "load_transposed_v_to_gpr: kRowOffset must be 0 or 16.");
        static_assert(
            (kColOffset % 16u == 0u) && (kColOffset < T::kVoHeadDim),
            "load_transposed_v_to_gpr: kColOffset must be a multiple of 16, < kVoHeadDim.");

        constexpr uint32_t kRowTile      = kRowOffset / kSubBlockRows; // 0 or 1
        constexpr uint32_t kColTile      = kColOffset / kSubBlockCols; // 0..15
        constexpr uint32_t kColInSbBytes = (kColOffset % kSubBlockCols) * sizeof(hk::bf16);
        // Bank-swizzle re-expressed as a conditional ±32 delta so that
        // (kFixedOffset + kColInSbBytes) stays fully constexpr in the
        // ds_read_b64_tr_b16 immediate offset: XOR-by-32 against a constexpr
        // value flips bit 5, equivalent to "+32 if bit was 0, else -32".
        // The sign is compile-time (from kColInSbBytes's bit 5); only the
        // boolean `is_swz` (1 bit per lane) is runtime. Avoids materialising
        // kColInSbBytes as a runtime VGPR (vs. the plain XOR formulation),
        // which freed 2 unpinned VGPRs in the audit.
        constexpr int32_t kSwzDelta     = (kColInSbBytes & 32u) ? -32 : +32;
        constexpr uint32_t kFixedOffset = sub_block_byte_offset(kRowTile, kColTile) + kColInSbBytes;

        const uint32_t lane_idx  = opus::lane_id();
        const uint32_t row_in_sb = lane_idx >> 2;
        const uint32_t is_swz    = (row_in_sb >> 2) & 1u;
        const uint32_t in_sb     = row_in_sb * (kSubBlockCols * sizeof(hk::bf16)) +
                                   (lane_idx & 3u) * 8u + is_swz * static_cast<uint32_t>(kSwzDelta);
        const uint32_t addr      = static_cast<uint32_t>(p_lds_v) + in_sb;

        hkm::ds_read_b64_tr_b16<GPR>(addr, kFixedOffset);
    }

    // bf16 ds_read_b64_tr_b16 already lands in the mfma A-operand layout -- no
    // intra-lane v_swap_b32 fixup needed (V32's fp8 path interleaved cols c and
    // c+16 into the same 2 GPRs and required a swap; the b16 transpose does not).
    // Kept as a no-op for caller parity with KvManager8bitsV3.
    template <uint32_t GPR_0, uint32_t GPR_1>
    __device__ __forceinline__ static void finalize_load_transposed_v_to_gpr()
    {
    }
};

// per call and un-swizzle on the bounce-LDS read side: lane L reads from LDS sub-tile
// `sb8_perm_subtile(L_subtile)` so that its dwordx4 VRAM destination lands at the
// straight (un-permuted) data col — adjacent lanes write to adjacent VRAM cols,
// so the buffer_store_dwordx4 wave stays coalesced and the per-iter VRAM imm offset
// reuses the existing kColOffset wiring.
//
// LDS-store side is written in permuted (LDS) order, no perm here — that matches
// oaccu's natural col axis.
template <typename T, typename out_t>
class OManager16bitsV4Gen1Swizzle
{
    private:
    static_assert(sizeof(out_t) == 2, "Output type must be 16 bits");

    static constexpr uint32_t kNumRows                = 16;
    static constexpr uint32_t kNumCols                = 64; // full wave-tile per call
    static constexpr uint32_t kNumPaddingElemPer2Rows = 4;
    static constexpr uint32_t kNumElemPerPadded2Rows =
        2 * kNumCols + kNumPaddingElemPer2Rows; // 132
    static constexpr uint32_t kVramStElemPerLane =
        4 * sizeof(uint32_t) / sizeof(out_t); // buffer_store_dwordx4 = 8 bf16
    static constexpr uint32_t kVramStLanePerRow = kNumCols / kVramStElemPerLane; // 64/8=8
    static constexpr uint32_t kVramStRowsPerRnd =
        opus::get_warp_size() / kVramStLanePerRow;                           // 64/8=8
    static constexpr uint32_t kVramStNumRnds = kNumRows / kVramStRowsPerRnd; // 16/8=2
    static constexpr uint32_t kLdsLdOffsetDeltaInBytes =
        (kVramStRowsPerRnd / 2u) * kNumElemPerPadded2Rows * sizeof(out_t); // (8/2)*132*2=1056
    static constexpr uint32_t kVramStOffsetDeltaInBytes =
        kVramStRowsPerRnd * T::kVoHeadDim * sizeof(out_t); // 8*512*2=8192

    // mfma_f32_16x16x32_bf16: per-lane (row=lane%16, cols=(lane/16)*4 + {0..3}).
    static constexpr uint32_t kMfmaRows        = 16;
    static constexpr uint32_t kMfmaCols        = 16;
    static constexpr uint32_t kMfmaElemPerLane = kMfmaRows * kMfmaCols / opus::get_warp_size(); // 4
    static constexpr uint32_t kNumMfmasPerCall = kNumCols / kMfmaCols;                          // 4

    __device__ __forceinline__ static constexpr uint32_t get_lds_size_per_warp_in_byte()
    {
        return (kNumRows / 2u) * kNumElemPerPadded2Rows * sizeof(out_t); // 8*132*2=2112
    }

    public:
    __device__ __forceinline__ static constexpr uint32_t get_lds_size_in_byte()
    {
        return T::kNumWarps * get_lds_size_per_warp_in_byte(); // 8*2112=16896
    }

    // GPR_START: starting GPR of the 16x64 wave-tile (16 fp32/lane = 16 vgprs).
    // kWaveTileColOff: element-wise col offset in the output buffer (multiple of 64).
    // See OManager16bitsV1 for the kCheckOOB contract.
    template <uint32_t GPR_START, uint32_t kWaveTileColOff, bool kCheckOOB>
    __device__ __forceinline__ void output_to_vram_pair(const out_t* p_output,
                                                        const uint32_t warp_idx,
                                                        const uint32_t qo_start,
                                                        const uint32_t qo_end,
                                                        const uintptr_t p_lds,
                                                        const uint32_t num_qheads)
    {
        static_assert((kWaveTileColOff % kNumCols) == 0,
                      "kWaveTileColOff must be a multiple of 64");
        constexpr uint32_t kOffsetInBytes = kWaveTileColOff * sizeof(out_t);

        uint32_t lane_idx    = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uintptr_t p_lds_warp = p_lds + warp_idx * get_lds_size_per_warp_in_byte();

        auto get_v_offset_lds = [&](const uint32_t row, const uint32_t col) -> uint32_t {
            return ((row / 2u) * kNumElemPerPadded2Rows + (row % 2u) * kNumCols + col) *
                   sizeof(out_t);
        };

        // ---- LDS store side ----
        const uint32_t row_lds_st      = lane_idx % kNumRows;
        const uint32_t col_lds_st_base = (lane_idx / kNumRows) * kMfmaElemPerLane; // 0/4/8/12
        const uint32_t v_offset_lds_st = ((row_lds_st / 2u) * kNumElemPerPadded2Rows +
                                          (row_lds_st % 2u) * kNumCols + col_lds_st_base) *
                                         sizeof(out_t);

        // ---- LDS read side: undo perm on sub-tile field ----
        // Lane wants VRAM col = lane_in_row*8. That data lives in LDS sub-tile
        // sb8_perm_subtile(lane_in_row). Address goes through the swizzled
        // helper so the row-half-swap done by the writer is undone here.
        const uint32_t row_lds_ld  = lane_idx / kVramStLanePerRow; // 0..7
        const uint32_t lane_in_row = lane_idx % kVramStLanePerRow; // 0..7
        const uint32_t lds_subtile =
            ((lane_in_row & 0x1u) << 2) | ((lane_in_row & 0x6u) >> 1); // perm(0..7)
        const uint32_t col_lds_ld      = lds_subtile * 8u;
        const uint32_t v_offset_lds_ld = get_v_offset_lds(row_lds_ld, col_lds_ld);

        // ---- VRAM store side: straight ----
        const uint32_t row_vram_st = row_lds_ld + warp_idx * kNumRows;
        const uint32_t col_vram_st = lane_in_row * kVramStElemPerLane;
        const uint32_t v_offset_vram_st =
            (row_vram_st * T::kVoHeadDim + col_vram_st) * sizeof(out_t);

        const uintptr_t p_output_batch =
            reinterpret_cast<uintptr_t>(p_output) +
            uint64_t(qo_start) * num_qheads * T::kVoHeadDim * sizeof(out_t);
        const uint32_t num_records =
            kCheckOOB ? ((qo_end - qo_start) * num_qheads * T::kVoHeadDim * sizeof(out_t))
                      : 0xFFFFFFFFu;
        const hk::buffer_resource out_br = hk::make_buffer_resource(
            static_cast<uint64_t>(p_output_batch), num_records, 0x00020000);

        if constexpr(std::is_same_v<out_t, hk::bf16>)
        {
            v2ui b16_pair_m0, b16_pair_m1, b16_pair_m2, b16_pair_m3;
            b16_pair_m0[0] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 0, GPR_START + 1);
            b16_pair_m0[1] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 2, GPR_START + 3);
            b16_pair_m1[0] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 4, GPR_START + 5);
            b16_pair_m1[1] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 6, GPR_START + 7);
            b16_pair_m2[0] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 8, GPR_START + 9);
            b16_pair_m2[1] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 10, GPR_START + 11);
            b16_pair_m3[0] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 12, GPR_START + 13);
            b16_pair_m3[1] = float_2_bf16_pair<T::kRoundMode>(GPR_START + 14, GPR_START + 15);

            constexpr uint32_t kMfmaByteStride = kMfmaCols * sizeof(out_t); // 32 B
            const uintptr_t addr_st            = p_lds_warp + v_offset_lds_st;
            hkm::ds_write_b64(b16_pair_m0, addr_st, 0u * kMfmaByteStride);
            hkm::ds_write_b64(b16_pair_m1, addr_st, 1u * kMfmaByteStride);
            hkm::ds_write_b64(b16_pair_m2, addr_st, 2u * kMfmaByteStride);
            hkm::ds_write_b64(b16_pair_m3, addr_st, 3u * kMfmaByteStride);
        }
        else
        {
            static_assert(false, "Unsupported output type");
        }

        // Reuse oaccu pinned VGPRs (GPR_START..GPR_START+7) as ds_read
        // destinations: after the bf16 packs + ds_writes complete the
        // GPR_START source range is dead (oaccu not read again this work_idx).
        // Pinning the read targets keeps the compiler from allocating extra
        // unpinned VGPRs that would risk leaking into pinned q_vgpr.
        //
        // Finer-grained lgkmcnt: drain reads one-at-a-time so the matching
        // buffer_store_dwordx4 can issue as soon as its data is ready.
        asm volatile("s_waitcnt lgkmcnt(0)");
        hkm::ds_read_b128<GPR_START + 0>(static_cast<uint32_t>(p_lds_warp + v_offset_lds_ld), 0);
        hkm::ds_read_b128<GPR_START + 4>(static_cast<uint32_t>(p_lds_warp + v_offset_lds_ld),
                                         static_cast<int>(kLdsLdOffsetDeltaInBytes));
        asm volatile("s_waitcnt lgkmcnt(1)");
        hkm::buffer_store_dwordx4<GPR_START + 0>(out_br, v_offset_vram_st, 0, kOffsetInBytes);
        asm volatile("s_waitcnt lgkmcnt(0)");
        hkm::buffer_store_dwordx4<GPR_START + 4>(
            out_br, v_offset_vram_st + kVramStOffsetDeltaInBytes, 0, kOffsetInBytes);
    }
};

// 32-bit (fp32 split-O) sibling of OManager16bitsV4Gen1Swizzle — same sub-tile-of-8 un-swizzle
// model: straight permuted layout into the bounce, perm-undo on the LDS read side
// so VRAM stores stay coalesced.
template <typename T, typename out_t>
class OManager32bitsV4Gen1Swizzle
{
    private:
    static_assert(sizeof(out_t) == 4, "Output type must be 32 bits");

    static constexpr uint32_t kNumRows              = 16;
    static constexpr uint32_t kNumCols              = 64;
    static constexpr uint32_t kNumPaddingElemPerRow = 4;
    static constexpr uint32_t kNumElemPerPaddedRow  = kNumCols + kNumPaddingElemPerRow; // 68
    static constexpr uint32_t kVramStElemPerLane =
        4 * sizeof(uint32_t) / sizeof(out_t);                                    // dwordx4 = 4 fp32
    static constexpr uint32_t kVramStLanePerRow = kNumCols / kVramStElemPerLane; // 64/4=16
    static constexpr uint32_t kVramStRowsPerRnd =
        opus::get_warp_size() / kVramStLanePerRow;                           // 64/16=4
    static constexpr uint32_t kVramStNumRnds = kNumRows / kVramStRowsPerRnd; // 4
    static constexpr uint32_t kLdsLdOffsetDeltaInBytes =
        kVramStRowsPerRnd * kNumElemPerPaddedRow * sizeof(out_t); // 4*68*4=1088
    static constexpr uint32_t kVramStOffsetDeltaInBytes =
        kVramStRowsPerRnd * T::kVoHeadDim * sizeof(out_t); // 4*512*4=8192

    static constexpr uint32_t kMfmaRows        = 16;
    static constexpr uint32_t kMfmaCols        = 16;
    static constexpr uint32_t kMfmaElemPerLane = kMfmaRows * kMfmaCols / opus::get_warp_size(); // 4
    static constexpr uint32_t kNumMfmasPerCall = kNumCols / kMfmaCols;                          // 4

    __device__ __forceinline__ static constexpr uint32_t get_lds_size_per_warp_in_byte()
    {
        return kNumRows * kNumElemPerPaddedRow * sizeof(out_t); // 16*68*4=4352
    }

    public:
    __device__ __forceinline__ static constexpr uint32_t get_lds_size_in_byte()
    {
        return T::kNumWarps * get_lds_size_per_warp_in_byte(); // 8*4352=34816
    }

    // See OManager16bitsV1 for the kCheckOOB contract.
    template <uint32_t GPR_START, uint32_t kWaveTileColOff, bool kCheckOOB>
    __device__ __forceinline__ void output_to_vram_pair(const out_t* p_output,
                                                        const uint32_t warp_idx,
                                                        const uint32_t qo_start,
                                                        const uint32_t qo_end,
                                                        const uintptr_t p_lds,
                                                        const uint32_t num_qheads)
    {
        static_assert((kWaveTileColOff % kNumCols) == 0,
                      "kWaveTileColOff must be a multiple of 64");
        constexpr uint32_t kOffsetInBytes = kWaveTileColOff * sizeof(out_t);

        uint32_t lane_idx    = opus::lane_id();
        asm volatile("" : "+v"(lane_idx)); // break-CSE: shorten lane-derived live ranges
        const uintptr_t p_lds_warp = p_lds + warp_idx * get_lds_size_per_warp_in_byte();

        auto get_v_offset_lds = [&](const uint32_t row, const uint32_t col) -> uint32_t {
            return (row * kNumElemPerPaddedRow + col) * sizeof(out_t);
        };

        // ---- LDS store side (straight permuted layout) ----
        const uint32_t row_lds_st      = lane_idx % kNumRows;
        const uint32_t col_lds_st_base = (lane_idx / kNumRows) * kMfmaElemPerLane; // 0/4/8/12
        const uint32_t v_offset_lds_st = get_v_offset_lds(row_lds_st, col_lds_st_base);

        // ---- LDS read side: perm-undo on sub-tile field ----
        // Lane wants VRAM col = lane_in_row*4. data sub-tile = lane_in_row >> 1;
        // LDS sub-tile = sb8_perm_subtile(data sub-tile); intra = (lane_in_row & 1)*4.
        const uint32_t row_lds_ld   = lane_idx / kVramStLanePerRow; // 0..3
        const uint32_t lane_in_row  = lane_idx % kVramStLanePerRow; // 0..15
        const uint32_t data_subtile = lane_in_row >> 1;             // 0..7
        const uint32_t lds_subtile  = ((data_subtile & 0x1u) << 2) | ((data_subtile & 0x6u) >> 1);
        const uint32_t intra_off    = (lane_in_row & 0x1u) * 4u;
        const uint32_t col_lds_ld   = lds_subtile * 8u + intra_off;
        const uint32_t v_offset_lds_ld = get_v_offset_lds(row_lds_ld, col_lds_ld);

        // ---- VRAM store: straight ----
        const uint32_t row_vram_st = row_lds_ld + warp_idx * kNumRows;
        const uint32_t col_vram_st = lane_in_row * kVramStElemPerLane;
        const uint32_t v_offset_vram_st =
            (row_vram_st * T::kVoHeadDim + col_vram_st) * sizeof(out_t);

        const uintptr_t p_output_batch =
            reinterpret_cast<uintptr_t>(p_output) +
            uint64_t(qo_start) * num_qheads * T::kVoHeadDim * sizeof(out_t);
        const uint32_t num_records =
            kCheckOOB ? ((qo_end - qo_start) * num_qheads * T::kVoHeadDim * sizeof(out_t))
                      : 0xFFFFFFFFu;
        const hk::buffer_resource out_br = hk::make_buffer_resource(
            static_cast<uint64_t>(p_output_batch), num_records, 0x00020000);

        if constexpr(std::is_same_v<out_t, float>)
        {
            // EXPERIMENT: removed the vmcnt(0) gate at the top of the call
            // (was 30k+ cycle stall on b=8 c=23333 OMgr trace, ~27% of
            // runtime across 8 calls). Should be safe -- prev call's
            // buffer_store_dwordx4 reads from THIS function's earlier
            // ds_read result, which is already drained via lgkmcnt(0) at
            // the end of each call.
            constexpr uint32_t kMfmaByteStride = kMfmaCols * sizeof(out_t); // 64 B
            hkm::ds_write_b128<GPR_START + 0>(p_lds_warp + v_offset_lds_st, 0u * kMfmaByteStride);
            hkm::ds_write_b128<GPR_START + 4>(p_lds_warp + v_offset_lds_st, 1u * kMfmaByteStride);
            hkm::ds_write_b128<GPR_START + 8>(p_lds_warp + v_offset_lds_st, 2u * kMfmaByteStride);
            hkm::ds_write_b128<GPR_START + 12>(p_lds_warp + v_offset_lds_st, 3u * kMfmaByteStride);
        }
        else
        {
            static_assert(false, "Unsupported output type");
        }

        // Reuse the oaccu pinned VGPRs (GPR_START..GPR_START+15) as ds_read
        // destinations. After the ds_writes complete, oaccu is dead (we just
        // wrote it to LDS, won't read it again until next work_idx which
        // reinitializes). Using the pinned slots as read targets prevents
        // the compiler from allocating extra unpinned VGPRs for the read
        // results -- those allocations would otherwise leak into the pinned
        // q_vgpr region under tighter scheduling.
        //
        // Finer-grained lgkmcnt: drain reads one-at-a-time so the matching
        // buffer_store_dwordx4 can issue as soon as its data is ready,
        // overlapping the remaining LDS latency with vmem store traffic.
        // LDS reads complete in issue order, so lgkmcnt(N) means N reads
        // remain in flight.
        asm volatile("s_waitcnt lgkmcnt(0)");
        hkm::ds_read_b128<GPR_START + 0>(static_cast<uint32_t>(p_lds_warp + v_offset_lds_ld), 0);
        hkm::ds_read_b128<GPR_START + 4>(static_cast<uint32_t>(p_lds_warp + v_offset_lds_ld),
                                         static_cast<int>(1u * kLdsLdOffsetDeltaInBytes));
        hkm::ds_read_b128<GPR_START + 8>(static_cast<uint32_t>(p_lds_warp + v_offset_lds_ld),
                                         static_cast<int>(2u * kLdsLdOffsetDeltaInBytes));
        hkm::ds_read_b128<GPR_START + 12>(static_cast<uint32_t>(p_lds_warp + v_offset_lds_ld),
                                          static_cast<int>(3u * kLdsLdOffsetDeltaInBytes));
        asm volatile("s_waitcnt lgkmcnt(3)");
        hkm::buffer_store_dwordx4<GPR_START + 0>(out_br, v_offset_vram_st, 0, kOffsetInBytes);
        asm volatile("s_waitcnt lgkmcnt(2)");
        hkm::buffer_store_dwordx4<GPR_START + 4>(
            out_br, v_offset_vram_st + 1u * kVramStOffsetDeltaInBytes, 0, kOffsetInBytes);
        asm volatile("s_waitcnt lgkmcnt(1)");
        hkm::buffer_store_dwordx4<GPR_START + 8>(
            out_br, v_offset_vram_st + 2u * kVramStOffsetDeltaInBytes, 0, kOffsetInBytes);
        asm volatile("s_waitcnt lgkmcnt(0)");
        hkm::buffer_store_dwordx4<GPR_START + 12>(
            out_br, v_offset_vram_st + 3u * kVramStOffsetDeltaInBytes, 0, kOffsetInBytes);
    }
};

// 32-bit O writer that bypasses the LDS bounce: each lane issues 4
// buffer_store_dwordx4 straight from its accumulator VGPRs to VRAM. The lanes
// in a wave write to non-contiguous addresses (no coalescing), but the manager
// allocates zero LDS, which removes any possibility of overlap with the
// split-O reduction region that OManager32bitsV4Gen1Swizzle (staged) competes with.
//
// Per-lane oaccu layout for one 64-col wave-tile (16 fp32 in VGPRs):
//   GPR_START + m*4 + i  =  fp32 at (row=lane%16, mfma=m, col-in-mfma = (lane/16)*4 + i)
// where mfma m covers LDS-cols [m*16 .. m*16+15] (in sb8-permuted order).
// Under sb8 perm, LDS sub-tile k holds data sub-tile sb8_inv(k) =
//   ((k & 4) >> 2) | ((k & 3) << 1).
// For lane (row, col_quad = lane/16), mfma m:
//   lds_subtile = m*2 + (col_quad >> 1)        // 0..7
//   data_subtile = sb8_inv(lds_subtile)
//   intra_half  = col_quad & 1                  // 0 or 1
//   data_col    = data_subtile*8 + intra_half*4
template <typename T, typename out_t>
class OManager32bitsV4Gen1SwNoStage
{
    private:
    static_assert(sizeof(out_t) == 4, "Output type must be 32 bits");

    static constexpr uint32_t kNumRows  = 16;
    static constexpr uint32_t kNumCols  = 64;
    static constexpr uint32_t kMfmaCols = 16;
    static constexpr uint32_t kNumMfmas = kNumCols / kMfmaCols; // 4

    public:
    __device__ __forceinline__ static constexpr uint32_t get_lds_size_in_byte() { return 0; }

    // See OManager16bitsV1 for the kCheckOOB contract.
    template <uint32_t GPR_START, uint32_t kWaveTileColOff, bool kCheckOOB>
    __device__ __forceinline__ void output_to_vram_pair(const out_t* p_output,
                                                        const uint32_t warp_idx,
                                                        const uint32_t qo_start,
                                                        const uint32_t qo_end,
                                                        const uintptr_t /*p_lds*/,
                                                        const uint32_t num_qheads)
    {
        static_assert((kWaveTileColOff % kNumCols) == 0,
                      "kWaveTileColOff must be a multiple of 64");
        constexpr uint32_t kColOffBytes = kWaveTileColOff * sizeof(out_t);

        const uint32_t lane_idx = opus::lane_id();
        const uint32_t row      = lane_idx % kNumRows; // 0..15
        const uint32_t col_quad = lane_idx / kNumRows; // 0..3

        const uint32_t vram_row       = row + warp_idx * kNumRows;
        const uint32_t row_base_bytes = vram_row * T::kVoHeadDim * sizeof(out_t);

        const uintptr_t p_output_batch =
            reinterpret_cast<uintptr_t>(p_output) +
            uint64_t(qo_start) * num_qheads * T::kVoHeadDim * sizeof(out_t);
        const uint32_t num_records =
            kCheckOOB ? ((qo_end - qo_start) * num_qheads * T::kVoHeadDim * sizeof(out_t))
                      : 0xFFFFFFFFu;
        const hk::buffer_resource out_br = hk::make_buffer_resource(
            static_cast<uint64_t>(p_output_batch), num_records, 0x00020000);

        if constexpr(std::is_same_v<out_t, float>)
        {
            asm volatile("s_waitcnt vmcnt(0)");
            opus::static_for<kNumMfmas>([&](auto im) {
                constexpr uint32_t m       = im.value;
                const uint32_t lds_subtile = m * 2u + (col_quad >> 1);
                const uint32_t data_subtile =
                    ((lds_subtile & 0x4u) >> 2) | ((lds_subtile & 0x3u) << 1);
                const uint32_t intra_half = col_quad & 0x1u;
                const uint32_t data_col   = data_subtile * 8u + intra_half * 4u;
                const uint32_t v_offset   = row_base_bytes + data_col * sizeof(out_t);
                hkm::buffer_store_dwordx4<GPR_START + m * 4u>(
                    out_br, v_offset, /*s_off=*/0, /*i_off=*/kColOffBytes);
            });
        }
        else
        {
            static_assert(false, "Unsupported output type");
        }
    }
};

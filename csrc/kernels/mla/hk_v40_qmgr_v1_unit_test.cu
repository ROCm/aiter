// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Standalone unit test entry for QManager8to16bitsV1. Runs only the Q loader
// (no QK / softmax / PV / epilogue) on a single block and dumps:
//   - the VGPR half (Q[:, 0:256]) pinned at v72..v103 into q_vgpr_out
//   - the LDS  half (Q[:, 256:512]) raw 64 KB region into q_lds_out
// Layout of both outputs is [warp_idx (0..7), head_in_warp (0..15),
// feat (0..255)] BF16. q_lds_out is decoded per the wave-major contiguous
// sub-block layout described in hk_mla_buffer_managers.cuh (see Python ref).

#include "hk/hk_mla_buffer_managers.cuh"
#include "hk/hk_mla_utils.cuh"
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "mla.h"
#include "mla_hk.h"

namespace {

// Emit `v_mov_b32 vDST, v[VGPR]` and return the value in a compiler-allocated
// VGPR. The compiler must keep its scratch in v0..v67 (enforced by
// amdgpu_num_vgpr(68) on the kernel), so the destination lands in a slot
// disjoint from the pinned source.
template <int VGPR>
__device__ __forceinline__ uint32_t read_pinned_vgpr_dw()
{
    static_assert(VGPR < 256, "VGPR id out of range");
    uint32_t out;
    asm volatile("v_mov_b32 %0, v[%1]" : "=v"(out) : "n"(VGPR));
    return out;
}

// Dump one mfma A-tile (32 cols x 16 rows, 8 bf16/lane = 4 dwords/lane) from
// v[kGprBase + kChunk*8 + kIter*4 + 0..3] to q_vgpr_out at cols
// [kChunk*64 + kIter*32 + col_group*8, ... + col_group*8 + 8).
template <int kGprBase, int kChunk, int kIter>
__device__ __forceinline__ void dump_vgpr_iter(hk::bf16*      p_out_head,
                                               const uint32_t col_group)
{
    constexpr int kSlot0 = kGprBase + kChunk * 8 + kIter * 4 + 0;
    constexpr int kSlot1 = kGprBase + kChunk * 8 + kIter * 4 + 1;
    constexpr int kSlot2 = kGprBase + kChunk * 8 + kIter * 4 + 2;
    constexpr int kSlot3 = kGprBase + kChunk * 8 + kIter * 4 + 3;

    uint32_t pack[4];
    pack[0] = read_pinned_vgpr_dw<kSlot0>();
    pack[1] = read_pinned_vgpr_dw<kSlot1>();
    pack[2] = read_pinned_vgpr_dw<kSlot2>();
    pack[3] = read_pinned_vgpr_dw<kSlot3>();

    const hk::bf16* p_pack = reinterpret_cast<const hk::bf16*>(pack);
    const int       col_base = kChunk * 64 + kIter * 32 + static_cast<int>(col_group) * 8;
#pragma unroll
    for(int e = 0; e < 8; ++e)
    {
        p_out_head[col_base + e] = p_pack[e];
    }
}

using QmgrTestTraits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,
                                               hk::bf16,
                                               hk::fp8e4m3,
                                               hk::bf16,
                                               hk::bf16,
                                               /*kBlockN=*/32,
                                               /*kNumWarps=*/8,
                                               /*kOccupancy=*/1,
                                               /*kBlockM=*/128,
                                               /*kPageSize=*/1>;

struct QmgrV1UnitTestParams
{
    typename QmgrTestTraits::gl_q_nope query;
    typename QmgrTestTraits::gl_q_rope query_rope;
    hk::bf16*                          q_vgpr_out;
    hk::bf16*                          q_lds_out;
};

constexpr int kGprNopeStart = 72;

__global__ __launch_bounds__(QmgrTestTraits::kNumThreads, QmgrTestTraits::kOccupancy)
    __attribute__((amdgpu_num_vgpr(68))) void
    kn_qmgr_v1_unit_test(QmgrV1UnitTestParams params)
{
    using T = QmgrTestTraits;

    // readfirstlane on the warp-uniform scalars: load_q derives a per-warp
    // pointer from (qo_start, warp_idx), and the inline-asm `buffer_load_ubyte`
    // path inside p1_vmem_to_staging_chunk requires srsrc in scalar regs.
    // Without these readfirstlanes the compiler cannot scalar-promote the
    // buffer_resource and the .s fails to assemble ("invalid operand").
    const int32_t warp_idx =
        __builtin_amdgcn_readfirstlane(static_cast<int32_t>(threadIdx.x) / 64);
    const int32_t qo_start = __builtin_amdgcn_readfirstlane(0); // single Q token

    extern __shared__ int32_t p_lds[];
    // Pad before p_lds_q so warp 0's P1 staging (= p_lds_q + 0) survives the
    // `staging - kColInRecord` LDS-dst-pointer arithmetic without underflow.
    const uintptr_t           p_lds_q = reinterpret_cast<uintptr_t>(p_lds)
                                        + QManager8to16bitsV1<T>::kLdsHeadPadBytes;

    QManager8to16bitsV1<T> q_manager;
    q_manager.template load_q<kGprNopeStart>(
        params.query, params.query_rope, warp_idx, qo_start, p_lds_q);

    // s_nop after the last cvt-to-pinned; required before any VALU/MFMA
    // consumes pinned vgprs (the cvt-to-pinned hazard is invisible to the
    // compiler through opaque inline asm).
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_nop 7");

    // Force the kernel's .vgpr_count to encompass v72..v103 (the pinned slots
    // load_q writes via numeric-literal `v[%0]` operands). Without these
    // clobbers the compiler reports num_vgpr = (its own scratch only) -- e.g.
    // 28 -- and the GPU allocates only that many per thread; references to
    // v72+ then alias the compiler's scratch and silently corrupt.
    // The full kernel doesn't need this because its pinned-VGPR MFMAs use "v"
    // constraint operands that already extend the live range.
#define CLOBBER_V(N) asm volatile("" ::: "v" #N)
    CLOBBER_V(72);  CLOBBER_V(73);  CLOBBER_V(74);  CLOBBER_V(75);
    CLOBBER_V(76);  CLOBBER_V(77);  CLOBBER_V(78);  CLOBBER_V(79);
    CLOBBER_V(80);  CLOBBER_V(81);  CLOBBER_V(82);  CLOBBER_V(83);
    CLOBBER_V(84);  CLOBBER_V(85);  CLOBBER_V(86);  CLOBBER_V(87);
    CLOBBER_V(88);  CLOBBER_V(89);  CLOBBER_V(90);  CLOBBER_V(91);
    CLOBBER_V(92);  CLOBBER_V(93);  CLOBBER_V(94);  CLOBBER_V(95);
    CLOBBER_V(96);  CLOBBER_V(97);  CLOBBER_V(98);  CLOBBER_V(99);
    CLOBBER_V(100); CLOBBER_V(101); CLOBBER_V(102); CLOBBER_V(103);
#undef CLOBBER_V

    // ---- Dump VGPR half (Q[:, 0:256]) ----
    const uint32_t lane        = opus::lane_id();
    const uint32_t row_in_warp = lane & 15u;       // head_in_warp 0..15
    const uint32_t col_group   = (lane >> 4) & 3u; // 0..3 -> col_base = col_group * 8

    hk::bf16* p_out_vgpr =
        params.q_vgpr_out + warp_idx * (16 * 256) + row_in_warp * 256;

    opus::static_for<4>([&](auto kChunkN) {
        opus::static_for<2>([&](auto kIterN) {
            constexpr int kChunk = decltype(kChunkN)::value;
            constexpr int kIter  = decltype(kIterN)::value;
            dump_vgpr_iter<kGprNopeStart, kChunk, kIter>(p_out_vgpr, col_group);
        });
    });

    // ---- Dump LDS half (Q[:, 256:512]) raw ----
    // Cross-warp barrier: every wave has completed Phase 2 ds_writes before we
    // read the final region.
    __syncthreads();

    const int             tid       = threadIdx.x;
    const uint32_t*       p_lds_dw  = reinterpret_cast<const uint32_t*>(p_lds_q);
    uint32_t*             p_out_dw  = reinterpret_cast<uint32_t*>(params.q_lds_out);
    constexpr int         kDwordsPerThread = (64 * 1024) / (256 * 4); // 64
#pragma unroll
    for(int i = 0; i < kDwordsPerThread; ++i)
    {
        const int linear_dw = tid * kDwordsPerThread + i;
        p_out_dw[linear_dw] = p_lds_dw[linear_dw];
    }
}

} // namespace

void hk_mla_v40_qmanager_v1_unit_test(aiter_tensor_t& query,
                                      aiter_tensor_t& query_rope,
                                      aiter_tensor_t& q_vgpr_out,
                                      aiter_tensor_t& q_lds_out)
{
    HipDeviceGuard  device_guard(q_vgpr_out.device_id);
    const std::string gfx = get_gpu_arch();

    AITER_CHECK(gfx == "gfx950",
                "hk_mla_v40_qmanager_v1_unit_test: requires gfx950, got ",
                gfx);
    AITER_CHECK(query.dtype() == AITER_DTYPE_fp8,
                "hk_mla_v40_qmanager_v1_unit_test: query must be FP8.");
    AITER_CHECK(query_rope.dtype() == AITER_DTYPE_bf16,
                "hk_mla_v40_qmanager_v1_unit_test: query_rope must be BF16.");
    AITER_CHECK(q_vgpr_out.dtype() == AITER_DTYPE_bf16,
                "hk_mla_v40_qmanager_v1_unit_test: q_vgpr_out must be BF16.");
    AITER_CHECK(q_lds_out.dtype() == AITER_DTYPE_bf16,
                "hk_mla_v40_qmanager_v1_unit_test: q_lds_out must be BF16.");

    using T                  = QmgrTestTraits;
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const auto q_nope_view = hk::make_gl<typename T::gl_q_nope>(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
        /*dim0=*/static_cast<int32_t>(1),
        /*dim1=*/static_cast<int32_t>(T::kBlockM / T::kTileM),
        /*dim2=*/static_cast<int32_t>(T::kTileM),
        /*dim3=*/static_cast<int32_t>(T::kQkPackedNopeQElems));
    const auto q_rope_view = hk::make_gl<typename T::gl_q_rope>(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query_rope.data_ptr())),
        /*dim0=*/static_cast<int32_t>(1),
        /*dim1=*/static_cast<int32_t>(T::kBlockM / T::kTileM),
        /*dim2=*/static_cast<int32_t>(T::kTileM),
        /*dim3=*/static_cast<int32_t>(T::kQkRopeHeadDim));

    QmgrV1UnitTestParams params{
        q_nope_view,
        q_rope_view,
        reinterpret_cast<hk::bf16*>(q_vgpr_out.data_ptr()),
        reinterpret_cast<hk::bf16*>(q_lds_out.data_ptr())};

    const dim3   grid(1);
    // DEBUG variant 1: launch only 1 warp instead of T::kNumThreads (= 8 warps).
    // Only warp 0 runs load_q; warps 1..7 do not exist. If warp 0's q_vgpr at
    // chunk 0 upper half still passes here, the bug requires multi-warp
    // context (cross-warp LDS/m0 race). If it FAILS, the bug is single-warp.
    const dim3   block(64);
    constexpr int kLdsBytes = 64 * 1024;

    kn_qmgr_v1_unit_test<<<grid, block, kLdsBytes, stream>>>(params);
    HIP_CALL(hipGetLastError());
}

// ============================================================================
// P1 ladder checkpoint probe.
// ============================================================================

namespace {

struct QmgrV1P1LadderProbeParams
{
    typename QmgrTestTraits::gl_q_nope query;
    typename QmgrTestTraits::gl_q_rope query_rope;  // unused; kept for parity
    hk::bf16*                          dump_out;
};

__global__ __launch_bounds__(QmgrTestTraits::kNumThreads, QmgrTestTraits::kOccupancy)
    __attribute__((amdgpu_num_vgpr(68))) void
    kn_qmgr_v1_p1_ladder_probe(QmgrV1P1LadderProbeParams params)
{
    using T = QmgrTestTraits;

    const int32_t warp_idx =
        __builtin_amdgcn_readfirstlane(static_cast<int32_t>(threadIdx.x) / 64);
    const int32_t qo_start = __builtin_amdgcn_readfirstlane(0);

    extern __shared__ int32_t p_lds[];
    // Pad before p_lds_q so warp 0's P1 staging (= p_lds_q + 0) survives the
    // V32-style LDS-pointer pre-subtraction in p1_vmem_to_staging_chunk
    // (which subtracts kColInRecord = 0/64/128/192).
    const uintptr_t           p_lds_q = reinterpret_cast<uintptr_t>(p_lds)
                                        + QManager8to16bitsV1<T>::kLdsHeadPadBytes;

    QManager8to16bitsV1<T> q_manager;
    q_manager.template probe_p1_ladder_checkpoints<kGprNopeStart>(
        params.query, warp_idx, qo_start, p_lds_q, params.dump_out);
}

} // namespace

void hk_mla_v40_qmanager_v1_p1_ladder_probe(aiter_tensor_t& query,
                                            aiter_tensor_t& query_rope,
                                            aiter_tensor_t& dump_out)
{
    HipDeviceGuard  device_guard(dump_out.device_id);
    const std::string gfx = get_gpu_arch();

    AITER_CHECK(gfx == "gfx950",
                "hk_mla_v40_qmanager_v1_p1_ladder_probe: requires gfx950, got ", gfx);
    AITER_CHECK(query.dtype() == AITER_DTYPE_fp8,
                "hk_mla_v40_qmanager_v1_p1_ladder_probe: query must be FP8.");
    AITER_CHECK(query_rope.dtype() == AITER_DTYPE_bf16,
                "hk_mla_v40_qmanager_v1_p1_ladder_probe: query_rope must be BF16.");
    AITER_CHECK(dump_out.dtype() == AITER_DTYPE_bf16,
                "hk_mla_v40_qmanager_v1_p1_ladder_probe: dump_out must be BF16.");

    using T                  = QmgrTestTraits;
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const auto q_nope_view = hk::make_gl<typename T::gl_q_nope>(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
        static_cast<int32_t>(1),
        static_cast<int32_t>(T::kBlockM / T::kTileM),
        static_cast<int32_t>(T::kTileM),
        static_cast<int32_t>(T::kQkPackedNopeQElems));
    const auto q_rope_view = hk::make_gl<typename T::gl_q_rope>(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query_rope.data_ptr())),
        static_cast<int32_t>(1),
        static_cast<int32_t>(T::kBlockM / T::kTileM),
        static_cast<int32_t>(T::kTileM),
        static_cast<int32_t>(T::kQkRopeHeadDim));

    QmgrV1P1LadderProbeParams params{
        q_nope_view,
        q_rope_view,
        reinterpret_cast<hk::bf16*>(dump_out.data_ptr())};

    const dim3   grid(1);
    const dim3   block(T::kNumThreads);
    constexpr int kLdsBytes = 64 * 1024;

    kn_qmgr_v1_p1_ladder_probe<<<grid, block, kLdsBytes, stream>>>(params);
    HIP_CALL(hipGetLastError());
}

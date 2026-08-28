// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits + kargs for the gfx1250 (MI450) a8w8 mxscale BMM with a PRESHUFFLED B.
//
//   Y[M, batch, N] = O[M, batch, K] @ wo_a[batch, N, K]^T
//   O / wo_a are fp8 (e4m3); the per-block e8m0 scales are applied by the
//   scaled WMMA (__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4).
//
// SCOPE OF THIS HEADER: it is the single source of truth for every compile-time
// constant the pipeline reads (same rule as opus_gemm_traits_a16w16_gfx1250.cuh
// -- the pipeline carries no local `constexpr`, it references `T::...`).
//
// Two things differ from the a16w16 TDM traits and both are load-bearing:
//   1. the MMA is the SCALED wmma 16x16x128, so the traits also fix the scale
//      geometry (how many e8m0 bytes one WMMA consumes, and where they live);
//   2. B is not a plain [N, K] tensor -- it is the host-side
//      shuffle_weight(w, layout=(16,16)) buffer, so its TDM window is a window
//      over 16-row BLOCKS, not over rows. See kBShuf* below.
#pragma once

#include "../opus_gemm_utils.cuh"

#include <type_traits>

namespace opus_gfx1250_bmm {
// Consumer-wave tiling (mirrors opus_gfx1250::kCtdmLayout*): the 2 consumer
// waves split exactly one dimension of the B_M x B_N tile.
constexpr int kLayoutTileN = 0;   // consumers split N -- best for small M
constexpr int kLayoutTileM = 1;   // consumers split M
}  // namespace opus_gfx1250_bmm

#ifndef OPUS_BMM_A8W8_MXSCALE_KARGS_GFX1250_DEFINED
#define OPUS_BMM_A8W8_MXSCALE_KARGS_GFX1250_DEFINED
// Kernel arguments. Deliberately shaped like opus_gemm_scale_kargs_gfx950 (the
// gfx950 BMM mxscale kargs) so the host side of the two paths reads the same:
// every tensor gets a row stride and a BATCH stride, and every stride is in
// ELEMENTS of that tensor's dtype, never bytes.
//
// Layouts the launcher validates (aiter side, see the launcher header):
//   ptr_a   fp8   O     [M, batch, K]   stride_a       = O.stride(0)
//                                       stride_a_batch = O.stride(1)   (K contiguous)
//   ptr_b   fp8   wo_a  [batch, N, K]   PRESHUFFLED, see kBShuf* below
//   ptr_c   out   Y     [M, batch, N]   stride_c       = Y.stride(0)
//   ptr_sfa e8m0  x_scale  per (row, K/GROUP_K)
//   ptr_sfb e8m0  w_scale  per (col, K/GROUP_K)
struct opus_bmm_a8w8_mxscale_kargs_gfx1250 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;     // preshuffled weights
    void*       __restrict__ ptr_c;
    const void* __restrict__ ptr_sfa;   // e8m0
    const void* __restrict__ ptr_sfb;   // e8m0
    int m;
    int n;
    int k;
    int batch;
    int split_k;                        // 1 = no split-K (scaffold: only 1 is wired)
    int stride_a;                       // A row pitch (elements)
    int stride_b;                       // B row pitch of the UNSHUFFLED weight (= K)
    int stride_c;                       // C row pitch (elements)
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    int stride_sfa;                     // A scale row pitch (e8m0 bytes)
    int stride_sfb;                     // B scale row pitch (e8m0 bytes)
    int stride_sfa_batch;
    int stride_sfb_batch;
};
#endif

// -- traits ----------------------------------------------------------------
// GROUP_K_ = the K extent one e8m0 scale covers:
//   32  -> MX (one scale per 32 K elements; a WMMA's K=128 needs 4 bytes)
//   128 -> DSV3-style 1x128 blockscale (one byte broadcast over the WMMA)
// Both land in the SAME BX32 int operand of the scaled WMMA; the only
// difference is whether the 4 bytes are distinct or a broadcast (kScaleBcast).
template<int BLOCK_SIZE_,
         int B_M_, int B_N_, int B_K_,
         int LAYOUT_,
         typename D_A_, typename D_B_, typename D_C_, typename D_ACC_,
         int GROUP_K_   = 128,
         int NUM_SLOTS_ = 3,
         int WG_PER_CU_ = 2>
struct opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250 {
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int B_M = B_M_;
    static constexpr int B_N = B_N_;
    static constexpr int B_K = B_K_;
    static constexpr int LAYOUT = LAYOUT_;

    using D_A   = D_A_;
    using D_B   = D_B_;
    using D_C   = D_C_;
    using D_ACC = D_ACC_;
    using D_SF  = opus::e8m0_t;

    using DataA   = D_A;
    using DataB   = D_B;
    using DataC   = D_C;
    using DataAcc = D_ACC;
    using DataSf  = D_SF;

    static_assert(std::is_same<D_A, D_B>::value, "A/B dtype must match");
    static_assert(sizeof(D_A) == 1, "a8w8: A/B are 8-bit (fp8_t / bf8_t)");
    static_assert(std::is_same<D_ACC, float>::value, "scaled WMMA accumulates in fp32");

    static constexpr int kBlockM = B_M;
    static constexpr int kBlockN = B_N;
    static constexpr int kBlockK = B_K;

    // -- MMA: gfx1250 scaled WMMA 16x16x128 fp8 (wave32) --------------------
    static constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 128;
    static constexpr int kWarp  = 32;                       // gfx1250 wave size
    static constexpr int kWarpRt = opus::get_warp_size();    // 32 device / 64 host
    static constexpr int kNumWaves = BLOCK_SIZE / kWarp;
    static constexpr int kNumProducerWaves = 2;              // w0 = A, w1 = B
    static constexpr int kNumConsumerWaves = kNumWaves - kNumProducerWaves;
    static_assert(kNumWaves == 4, "scaffold is locked to 4 waves (2 producer + 2 consumer)");

    static constexpr int kTileM = (LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? 2 : 1;
    static constexpr int kTileN = (LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? 1 : 2;
    static constexpr int kTileK = 1;
    static_assert(kTileM * kTileN == kNumConsumerWaves,
                  "consumer waves must equal kTileM*kTileN");

    static constexpr int kExpM = kBlockM / (kWmmaM * kTileM);   // M-tiles per consumer wave
    static constexpr int kExpN = kBlockN / (kWmmaN * kTileN);   // N-tiles per consumer wave
    static constexpr int kExpK = kBlockK / kWmmaK;              // K-tiles per LDS slot
    static_assert(kExpM >= 1 && kExpN >= 1 && kExpK >= 1, "tile too small for this layout");
    static_assert(kExpM * kWmmaM * kTileM == kBlockM, "B_M must be a multiple of kWmmaM*kTileM");
    static_assert(kExpN * kWmmaN * kTileN == kBlockN, "B_N must be a multiple of kWmmaN*kTileN");
    static_assert(kExpK * kWmmaK == kBlockK, "B_K must be a multiple of 128");

    static constexpr int VEC_A = 16 / (int)sizeof(D_A);   // 16 fp8 per b128 ds_read
    static constexpr int VEC_B = 16 / (int)sizeof(D_B);
    static constexpr int kVecA = VEC_A;
    static constexpr int kVecB = VEC_B;
    static constexpr int kVecC = 4;                       // fp32 dwordx4 store

    // WMMA register decomposition (from kWarpRt so device/host passes agree).
    static constexpr int kReptA = kWmmaM * kWmmaK / kWarpRt / kVecA;
    static constexpr int kReptB = kWmmaN * kWmmaK / kWarpRt / kVecB;
    static constexpr int kGrpKA = kWarpRt / kWmmaM;
    static constexpr int kGrpKB = kWarpRt / kWmmaN;

    // -- scale geometry -----------------------------------------------------
    // The scaled WMMA's BX32 operand is ONE int = 4 packed e8m0 bytes, and the
    // instruction consumes them across its K=128. So a GROUP_K=32 tile hands it
    // 4 distinct bytes; a GROUP_K=128 tile broadcasts one byte 4 times (this is
    // the gfx950 pack_e8m0x4 trick, unchanged).
    static constexpr int kGroupK = GROUP_K_;
    static_assert(kGroupK == 32 || kGroupK == 128,
                  "GROUP_K must be 32 (mx) or 128 (1x128 blockscale)");
    static constexpr int kScalesPerWmmaK = kWmmaK / kGroupK;         // 4 or 1
    static constexpr bool kScaleBcast    = (kScalesPerWmmaK == 1);   // broadcast one byte
    static_assert(kScalesPerWmmaK == 1 || kScalesPerWmmaK == 4,
                  "one BX32 scale word holds exactly 4 e8m0 bytes");
    static_assert(kBlockK % kGroupK == 0, "B_K must be a whole number of scale groups");
    // e8m0 bytes one row of this tile needs per K step.
    static constexpr int kSfPerRowPerStep = kBlockK / kGroupK;
    // A-scale bytes / B-scale bytes a whole tile needs per K step. Sized here so
    // the pipeline's optional LDS scale panel has one place to come from.
    static constexpr int kSfATileBytes = kBlockM * kSfPerRowPerStep;
    static constexpr int kSfBTileBytes = kBlockN * kSfPerRowPerStep;

    // -- B preshuffle geometry ----------------------------------------------
    // ptr_b is the host-side shuffle_weight(w, layout=(16,16)) buffer:
    //   off(n, k) = (n>>4)*(K>>5)*512 + (k>>5)*512 + ((k>>4)&1)*256
    //             + (n&15)*16 + (k&15)
    // Read that as a 2D tensor of 16-row BLOCKS:
    //   block row index  nb = n >> 4                (kBShufRows blocks per tile)
    //   within a block   the whole K range is CONTIGUOUS, 16 bytes per (n, k16)
    // so one block row is K*16 elements and a B_N x B_K tile is a
    //   [kBShufBlockElems x kBShufRows] window with row stride K*16.
    // That is why the B window's fast dim is NOT B_K: it is B_K*16.
    static constexpr int kBShufBlockN     = 16;
    static_assert(B_N % kBShufBlockN == 0, "B_N must be a multiple of 16 (shuffle_weight block)");
    static constexpr int kBShufRows       = B_N / kBShufBlockN;        // blocks per tile
    static constexpr int kBShufBlockElems = kBlockK * kBShufBlockN;    // elements per block per K step
    // Row stride of the shuffled buffer, in elements. Runtime K, so the pipeline
    // computes `kargs.k * kBShufBlockN`; this constant only names the factor.
    static constexpr int kBShufRowStrideFactor = kBShufBlockN;

    // -- LDS ----------------------------------------------------------------
    // A: [B_M rows][B_K + pad] with one read-vector of pad per row (the
    // conflict-free recipe). B: the shuffled blocks are already bank-friendly
    // (16 contiguous bytes per lane), so it gets NO pad -- keeping the D# pad
    // off B is also what keeps its fast dim (B_K*16) inside the 16-bit field.
    static_assert((kBlockK & (kBlockK - 1)) == 0,
                  "B_K must be a power of 2 (the D# pad interval is 8 << enc bytes)");
    static constexpr int kPadReadVecBytes = 16;
    static constexpr int kPadElems  = kPadReadVecBytes / (int)sizeof(DataA);   // 16 fp8
    static constexpr int kSmemPitchA = kBlockK + kPadElems;
    static constexpr int kSmemPitchB = kBShufBlockElems;                       // unpadded

    static constexpr int kARows = kBlockM;
    static constexpr int kBRows = kBShufRows;
    static constexpr int kSlotElemsA = kARows * kSmemPitchA;
    static constexpr int kSlotElemsB = kBRows * kSmemPitchB;

    static constexpr int kNumSlots = NUM_SLOTS_;    // prefetch depth P
    static_assert(kNumSlots == 2 || kNumSlots == 3, "prefetch depth must be 2 or 3");

    static constexpr int kWgPerCu = WG_PER_CU_;
    static_assert(kWgPerCu == 1 || kWgPerCu == 2, "kWgPerCu must be 1 or 2");

    static constexpr int kSegBytesA = kNumSlots * kSlotElemsA * (int)sizeof(DataA);
    static constexpr int kSegBytesB = kNumSlots * kSlotElemsB * (int)sizeof(DataB);
    // Scale panel: OFF in the scaffold (the consumer reads scales straight from
    // global). Flip kSfPanel to true and this reserves the LDS for it; the
    // panel fill/read is the TODO in the pipeline.
    static constexpr bool kSfPanel      = false;
    static constexpr int  kSegBytesSfA  = kSfPanel ? kNumSlots * kSfATileBytes : 0;
    static constexpr int  kSegBytesSfB  = kSfPanel ? kNumSlots * kSfBTileBytes : 0;

    static constexpr int kSegBytesAB = kSegBytesA + kSegBytesB + kSegBytesSfA + kSegBytesSfB;
    // 1-WG/CU enforcement by LDS padding (same portable trick as the a16w16
    // traits: a WG over 160 KB leaves no room for a second on the 320 KB budget).
    static constexpr int kHalfLds = 160 * 1024;
    static constexpr int kLdsTotalBytes =
        (kWgPerCu == 1 && kSegBytesAB <= kHalfLds) ? (kHalfLds + 1024) : kSegBytesAB;
    static_assert(kLdsTotalBytes <= 320 * 1024, "LDS exceeds gfx1250's 320KB");
    // A WG_PER_CU_ = 2 request is a claim about occupancy, and LDS is what
    // actually decides it: two workgroups only co-reside if each fits in half
    // the 320 KB. Without this the tile compiles, launches, and quietly runs at
    // 1 WG/CU -- the scheduler never reports it, and the only symptom is that
    // the kernel is slower than the tile it was tuned as. Make it a build error.
    static_assert(kWgPerCu == 1 || kSegBytesAB <= kHalfLds,
                  "WG_PER_CU_ = 2 but the A+B (+scale) LDS segments exceed 160KB, "
                  "so only one WG can be resident. Shrink B_K/NUM_SLOTS or pass "
                  "WG_PER_CU_ = 1.");

    // -- scheduler hints (per K-tile round) ---------------------------------
    // NOT optional: these force ds_read-before-WMMA ordering, so they must equal
    // the real per-round instruction counts (see the a16w16 traits' note --
    // undercounting surfaces as NaN at the software-pipeline tail).
    static constexpr int kSchedDsMask    = 0x100;
    static constexpr int kSchedDsCount   = kExpM * kReptA + kExpN * kReptB;
    static constexpr int kSchedWmmaMask  = 0x008;
    static constexpr int kSchedWmmaCount = kExpM * kExpN;

#if (defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)) && (__clang_major__ >= 22)
    // TDM windows. Both are [fast, slow] in D# order (index 0 = fastest).
    //   A: [B_K x B_M]              over the plain [M, K] tensor, padded LDS write
    //   B: [B_K*16 x B_N/16]        over the SHUFFLED buffer, unpadded
    using PaddingA = opus::tdm_traits::padding_auto<DataA, kBlockK, kPadReadVecBytes>;
    using PaddingB = opus::tdm_traits::padding<>;      // no pad -- see kSmemPitchB
    using WindowA  = opus::tdm<DataA, opus::seq<kBlockK, kARows>, PaddingA>;
    using WindowB  = opus::tdm<DataB, opus::seq<kBShufBlockElems, kBRows>, PaddingB>;
    // The D# pad and the pitch the consumer reads with are two spellings of one
    // layout, so they cannot be allowed to drift.
    static_assert(PaddingA::pitch_elements == kSmemPitchA,
                  "kSmemPitchA must equal the D# padded row pitch");
#endif
};

// -- the shipped tile -------------------------------------------------------
// Declared HERE, not in either .cu, and that placement is the point: the host
// TU (opus_bmm.cu) and the device instantiation TU
// (opus_bmm_a8w8_mxscale_bpreshuffle_gfx1250.cu) must name the SAME traits type,
// because the launch stub is keyed on it. Two copies that drifted by one
// template argument would compile and link and then launch a kernel nobody
// instantiated -- a silent no-op. One alias, included by both.
//
// WG_PER_CU = 1 is forced by LDS, not chosen: 3 slots x (128x272 A + 8x4096 B)
// = 198 KB, more than half the 320 KB budget, so a second WG cannot co-reside
// (the static_assert above makes that a build error rather than a silent
// occupancy loss). For a real 2-WG/CU tile use B_K = 128 with NUM_SLOTS = 2
// (68 KB) and WG_PER_CU = 2.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/128, /*B_N*/128, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1>;

// -- smem -> register read layouts -----------------------------------------
// Device-only in effect, but compiled on the host pass too so vtype_c matches.
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)

// LDS tile layouts, i.e. what the TDM write produced.
//   A is [rows][K] with the padded pitch; B is [blocks][kBShufBlockElems] flat,
//   because the shuffled buffer's own interior order is the one the WMMA wants.
template <typename T>
__device__ inline auto make_layout_sa_bmm_mx() {
    return opus::make_layout<0>(
        opus::make_tuple(opus::number<T::kARows>{}, opus::number<T::kBlockK>{}),
        opus::make_tuple(opus::number<T::kSmemPitchA>{}, 1_I));
}
template <typename T>
__device__ inline auto make_layout_sb_bmm_mx() {
    return opus::make_layout<0>(
        opus::make_tuple(opus::number<T::kBRows>{}, opus::number<T::kSmemPitchB>{}),
        opus::make_tuple(opus::number<T::kSmemPitchB>{}, 1_I));
}

// A: LDS holds [B_M rows][B_K] with pitch kSmemPitchA. wave_m selects this
// consumer's M sub-tile (TileM: 0..1; TileN: always 0).
template <typename T>
__device__ inline auto make_layout_ra_bmm_mx(int lane_id, int wave_m) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpM>{}, opus::number<T::kTileM>{}, opus::number<T::kWmmaM>{},
        opus::number<T::kExpK>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptA>{}, opus::number<T::kGrpKA>{}, opus::number<T::kVecA>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitchA, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_m, lane_id % T::kWmmaM, 0, lane_id / T::kWmmaM}));
}

// B: TODO (yours). LDS holds the shuffled tile VERBATIM -- kBShufRows blocks of
// kBShufBlockElems elements, no pad -- so the read layout is NOT the mirror of
// make_layout_ra_bmm_mx: within one 16-row block a lane's 16 K-elements sit at
//   off = (k>>5)*512 + ((k>>4)&1)*256 + (n&15)*16 + (k&15)
// with n&15 = lane % 16 and the K position coming from the WMMA B-fragment
// mapping (grpk_b = kGrpKB, rept_b = kReptB). Write it as a make_layout<0> over
// the same 8-tuple shape as A, with the strides above, once you have fixed the
// fragment mapping on hardware -- the gfx950 analogue is
// gemm_bpre_shuf_kernel's B path in
// opus_gemm_pipeline_a8w8_blockscale_bpreshuffle_gfx950.cuh:693.
template <typename T>
__device__ inline auto make_layout_rb_bmm_mx(int lane_id, int wave_n) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpN>{}, opus::number<T::kTileN>{}, opus::number<T::kWmmaN>{},
        opus::number<T::kExpK>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptB>{}, opus::number<T::kGrpKB>{}, opus::number<T::kVecB>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitchB, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_n, lane_id % T::kWmmaN, 0, lane_id / T::kWmmaN}));
}

// -- scale layouts ---------------------------------------------------------
// ptr_sfa / ptr_sfb are PLAIN row-major e8m0 tensors (not preshuffled). WMMA
// OPSEL 0 reads scale bytes from lanes 0-15, so each lane must supply the e8m0
// for row (a_row0 + lane%16) / col (b_col0 + lane%16) -- mirror gfx950
// make_layout_sfa_mxsk. Shapes use kSfPerRowPerStep (= B_K/GROUP_K) for one
// LDS-slot K range; add (ik * kScalesPerWmmaK) when loading one WMMA K=128 tile.

// Global A-scale [M, K/GROUP_K]: one row's scale bytes for this K step.
template <typename T>
__device__ inline auto make_layout_gsfa_bmm_mx(int lane_id, int wave_m, int stride_sfa) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpM>{}, opus::number<T::kTileM>{}, opus::number<T::kWmmaM>{},
        opus::number<T::kSfPerRowPerStep>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_m, lane_id % T::kWmmaM}));
}

// Global B-scale [N, K/GROUP_K]: same per-lane N rule as gsfa's M rule.
template <typename T>
__device__ inline auto make_layout_gsfb_bmm_mx(int lane_id, int wave_n, int stride_sfb) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpN>{}, opus::number<T::kTileN>{}, opus::number<T::kWmmaN>{},
        opus::number<T::kSfPerRowPerStep>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_n, lane_id % T::kWmmaN}));
}

// Optional LDS scale panel (kSfPanel): one slot holds the tile's A/B scale bytes
// for a whole K step -- [B_M][kSfPerRowPerStep] and [B_N][kSfPerRowPerStep].
template <typename T>
__device__ inline auto make_layout_ssfa_bmm_mx() {
    return opus::make_layout<0>(
        opus::make_tuple(opus::number<T::kBlockM>{}, opus::number<T::kSfPerRowPerStep>{}),
        opus::make_tuple(opus::number<T::kSfPerRowPerStep>{}, 1_I));
}
template <typename T>
__device__ inline auto make_layout_ssfb_bmm_mx() {
    return opus::make_layout<0>(
        opus::make_tuple(opus::number<T::kBlockN>{}, opus::number<T::kSfPerRowPerStep>{}),
        opus::make_tuple(opus::number<T::kSfPerRowPerStep>{}, 1_I));
}

// Consumer ds_read from the LDS panel: same (im/tile/wmma) map as gsfa/gsfb but
// compact pitch kSfPerRowPerStep instead of the global row stride.
template <typename T>
__device__ inline auto make_layout_rsfa_bmm_mx(int lane_id, int wave_m) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpM>{}, opus::number<T::kTileM>{}, opus::number<T::kWmmaM>{},
        opus::number<T::kSfPerRowPerStep>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSfPerRowPerStep, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_m, lane_id % T::kWmmaM}));
}
template <typename T>
__device__ inline auto make_layout_rsfb_bmm_mx(int lane_id, int wave_n) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpN>{}, opus::number<T::kTileN>{}, opus::number<T::kWmmaN>{},
        opus::number<T::kSfPerRowPerStep>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSfPerRowPerStep, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_n, lane_id % T::kWmmaN}));
}

// Pack kScalesPerWmmaK contiguous e8m0 bytes into the WMMA BX32 operand.
template <typename T>
__device__ inline int pack_sf_word_bmm_mx(const opus::e8m0_t* p, int row, int row_stride, int k_group) {
    if constexpr (T::kScaleBcast) {
        const unsigned b = (unsigned)__builtin_bit_cast(unsigned char, p[(size_t)row * row_stride + k_group]);
        return (int)(b * 0x01010101u);
    } else {
        unsigned w = 0;
        opus::static_for<T::kScalesPerWmmaK>([&](auto jN) __attribute__((always_inline)) {
            constexpr int j = decltype(jN)::value;
            w |= ((unsigned)__builtin_bit_cast(unsigned char,
                    p[(size_t)row * row_stride + k_group + j])) << (8 * j));
        });
        return (int)w;
    }
}

#endif  // __gfx1250__ || !__HIP_DEVICE_COMPILE__

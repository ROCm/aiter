// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits + kargs for the gfx1250 a16w16 cluster/TDM split-K pipeline that
// reduces via an fp32 WORKSPACE + a separate REDUCE kernel (no atomic_add,
// no self-clear, no semaphore). Mirrors the gfx950 flatmm-splitk ABI
// (opus_splitk_ws_handle + opus_gemm_flatmm_splitk_kargs_gfx950).
//
// This header is the SINGLE source of truth for every compile-time constant
// the pipeline needs: the pipeline file
// (opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh) does NOT
// define a KernelTraits and carries no local `constexpr` -- it references
// `T::...` exclusively. All warp-derived quantities are computed from
// opus::get_warp_size() (device = 32 / host = 64) so the device and host
// compile passes stay consistent.
//
// Reference kernel: demon_gcn/wmma_opus_rdna4/gemm_a16w16_cluster_tdm_splitk_reduce_4wave.cc
//   (NO_CLUSTER variant: one workgroup == one B_M x B_N tile).
#pragma once

#include "../opus_gemm_utils.cuh"

#include <type_traits>

// ── Consumer-wave tiling layout ─────────────────────────────────────────────
// The 4-wave pipeline is fixed at 2 producers + 2 consumers; the 2 consumers
// split ONE dimension of the B_M x B_N tile:
//   TileN: consumers split along N (kTileM=1, kTileN=2) -- best for small M.
//   TileM: consumers split along M (kTileM=2, kTileN=1) -- generalizes M.
namespace opus_gfx1250 {
constexpr int kCtdmLayoutTileN = 0;
constexpr int kCtdmLayoutTileM = 1;
}  // namespace opus_gfx1250

// log2 of a power-of-2 (B_K is always a power of 2 here).
__host__ __device__ constexpr inline int opus_ctdm_log2_i(int x) {
    int r = 0; while (x > 1) { x >>= 1; ++r; } return r;
}

#ifndef OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
#define OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
// Indirection slot for the split-K fp32 workspace pointer. Captured HIP
// graphs hold the slot address (stable), not the workspace ptr, so a
// post-capture grow + hipFree of the old buffer doesn't dangle the graph.
struct opus_splitk_ws_handle {
    void*         ptr;    // current backing workspace; null until first grow
    unsigned long bytes;  // current capacity in bytes
};
#endif

#ifndef OPUS_GEMM_CLUSTER_TDM_WS_KARGS_GFX1250_DEFINED
#define OPUS_GEMM_CLUSTER_TDM_WS_KARGS_GFX1250_DEFINED
// Kernel arguments for the gfx1250 a16w16 cluster/TDM split-K (workspace)
// pipeline. The main kernel writes fp32 partial sums into
// *ws_handle->ptr laid out as [split_k, padded_M, padded_N] (per host launch;
// batch handled by a per-batch host launch with pointer offsets). The reduce
// kernel consumes it, folds bias once, casts fp32 -> Y dtype, writes C[M, N].
//
// Field semantics mirror opus_gemm_flatmm_splitk_kargs_gfx950 so the shared
// reduce kernel ABI (ws_handle*) is reused verbatim.
struct opus_gemm_cluster_tdm_ws_kargs_gfx1250 {
    const void* __restrict__ ptr_a;          // bf16 [M, K]
    const void* __restrict__ ptr_b;          // bf16 [N, K] (A @ B^T)
    const opus_splitk_ws_handle* __restrict__ ws_handle;  // deref at kernel entry
    void*       __restrict__ ptr_c;          // bf16/fp32 [M, N] (filled by reduce kernel)
    const void* __restrict__ ptr_bias;       // consumed by reduce kernel only
    int m;
    int n;
    int k;
    int batch;                               // = 1 per launch (host loops batch)
    int split_k;                             // runtime split factor (KBatch)
    int stride_a;                            // A row pitch (typically = K; > K if row-padded)
    int stride_b;                            // B row pitch (typically = K; > K if row-padded)
    int stride_ws;                           // = padded_N
    int stride_c;                            // = N
    int stride_a_batch;                      // A batch pitch (= M * stride_a)
    int stride_b_batch;                      // B batch pitch (= N * stride_b)
    int stride_ws_batch;                     // = padded_M * padded_N (one split slice)
    int stride_c_batch;                      // = M * N
    int stride_bias_batch;                   // 0 (broadcast [N]) or N ([batch, N])
};
#endif

// ── User-facing traits = the SINGLE compile-time config the pipeline reads ──
//   D_A=D_B=bf16, D_ACC=float (WMMA fp32 acc), D_C MUST be float (main kernel
//   writes the fp32 workspace; the reduce kernel casts to the final Y dtype).
template<int BLOCK_SIZE_,
         int B_M_, int B_N_, int B_K_,
         int LAYOUT_,
         typename D_A_, typename D_B_, typename D_C_, typename D_ACC_,
         bool ENABLE_BIAS_ = false,
         int NUM_SLOTS_ = 3,
         int WG_PER_CU_ = 2,
         int CLUSTER_WG_M_ = 4,    // cluster-launch variant: WGs per cluster in M
         int CLUSTER_WG_N_ = 4>    //                         WGs per cluster in N
struct opus_cluster_tdm_splitk_ws_traits_gfx1250 {
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;   // 128 (4 waves x 32)
    static constexpr int B_M = B_M_;
    static constexpr int B_N = B_N_;
    static constexpr int B_K = B_K_;
    static constexpr int LAYOUT = LAYOUT_;

    using D_A   = D_A_;
    using D_B   = D_B_;
    using D_C   = D_C_;                                // workspace dtype
    using D_ACC = D_ACC_;
    static_assert(std::is_same<D_A, D_B>::value, "A/B dtype must match");
    static_assert(std::is_same<D_C, float>::value,
                  "cluster_tdm_splitk_ws main kernel writes an fp32 workspace; D_C must be float");

    // Aliases used by the pipeline / layout helpers.
    using DataA   = D_A;
    using DataB   = D_B;
    using DataC   = D_C;
    using DataAcc = D_ACC;

    static constexpr int VEC_A = 16 / (int)sizeof(D_A);   // 8 for bf16
    static constexpr int VEC_B = 16 / (int)sizeof(D_B);   // 8
    static constexpr int VEC_C = 8;
    static constexpr int kVecA = VEC_A;
    static constexpr int kVecB = VEC_B;
    static constexpr int kVecC = VEC_C;
    static constexpr bool ENABLE_BIAS = ENABLE_BIAS_;
    static constexpr bool kEnableBias = ENABLE_BIAS_;

    // Tile geometry.
    static constexpr int kBlockM = B_M;
    static constexpr int kBlockN = B_N;
    static constexpr int kBlockK = B_K;

    // WMMA 16x16x32 (gfx1250 bf16).
    static constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 32;
    // Consumer-wave tiling: TileN splits N (1,2); TileM splits M (2,1).
    static constexpr int kTileM = (LAYOUT == opus_gfx1250::kCtdmLayoutTileM) ? 2 : 1;
    static constexpr int kTileN = (LAYOUT == opus_gfx1250::kCtdmLayoutTileM) ? 1 : 2;
    static constexpr int kTileK = 1;
    static constexpr int kExpM = kBlockM / (kWmmaM * kTileM);   // M-tiles / consumer wave
    static constexpr int kExpN = kBlockN / (kWmmaN * kTileN);   // N-tiles / consumer wave
    static_assert(kExpM >= 1, "B_M too small for this layout (TileM needs B_M>=32)");
    static_assert(kExpN >= 1, "B_N too small for this layout (TileN needs B_N>=32)");
    static_assert(kExpM * (kWmmaM * kTileM) == kBlockM, "B_M must be a multiple of kWmmaM*kTileM");
    static_assert(kExpN * (kWmmaN * kTileN) == kBlockN, "B_N must be a multiple of kWmmaN*kTileN");

    static constexpr int kExpKHalf = 2;                          // K-tiles per ds-prefetch half
    static constexpr int kKHalfElems = kWmmaK * kExpKHalf;       // 64 (K elems per half)
    static_assert(kBlockK % kKHalfElems == 0, "B_K must be a multiple of 64");
    static constexpr int kHalvesPerSlot = kBlockK / kKHalfElems; // 64-K ds-halves per slot

    // One TDM loads a whole B_K slot (tile_dim0 = B_K fits the 16-bit field).
    static constexpr int kTdmK = kBlockK;
    static constexpr int kNumChunk = kBlockK / kTdmK;            // 1
    static_assert(kBlockK == kNumChunk * kTdmK, "B_K must equal kTdmK");

    static constexpr int kWarp = 32;                             // gfx1250 wave size (fixed geometry)
    // Runtime-pass warp (32 device / 64 host); the WMMA register decomposition
    // is derived from this so device + host passes agree on vtype_c.
    static constexpr int kWarpRt = opus::get_warp_size();
    static constexpr int kNumWaves = BLOCK_SIZE / kWarp;         // 4
    static constexpr int kNumProducerWaves = 2;
    static constexpr int kNumConsumerWaves = 2;
    static_assert(kNumWaves == kNumProducerWaves + kNumConsumerWaves,
                  "pipeline is locked to 4 waves (2 producer + 2 consumer)");
    static_assert(kTileM * kTileN == kNumConsumerWaves,
                  "consumer waves must equal kTileM*kTileN");

    // ── TDM / tdm_window pad parameters (all B_K-derived) ──────────────────
    // One +kPadElems(=8) bf16 pad per B_K row -> conflict-free 16-row b128 reads.
    // PadInterval is the DWORD interval that yields exactly one pad per row:
    //   row = B_K bf16 = B_K/2 DWORD; pad every 2^(PadInterval+1) DWORD ->
    //   PadInterval = log2(B_K/2) - 1 = log2(B_K) - 2.
    static_assert((kBlockK & (kBlockK - 1)) == 0, "B_K must be a power of 2 (PadInterval formula)");
    static constexpr int kLdsPadEn   = 1;
    static constexpr int kPadInterval = opus_ctdm_log2_i(kBlockK) - 2;
    static constexpr int kPadAmount  = 3;                        // +16B = +8 bf16
    static constexpr int kPadElems   = 8;                        // bf16 elems added per row
    static_assert(kPadElems * (int)sizeof(DataA) == 16, "kPadAmount=3 encodes +16B; kPadElems must match");
    static constexpr int kSmemPitch  = kBlockK + kPadElems;      // padded row pitch

    static constexpr int kARows = kBlockM;                       // w0 loads all A rows
    static constexpr int kBRows = kBlockN;                       // w1 loads all B rows
    static constexpr int kSlotElemsA = kARows * kSmemPitch;
    static constexpr int kSlotElemsB = kBRows * kSmemPitch;
    // Prefetch depth P (number of LDS slots). The s_barrier producer issues the
    // first P TDMs in the prologue (peak in-flight = P), then per K-step does a
    // full s_wait_tensorcnt(0) drain before the workgroup barrier and refills one
    // slot -- so P bounds the peak in-flight TDM count. Lower P reduces the
    // direct-copy TDM request count req = rows * P * (B_K/128) per operand,
    // which must stay under the hardware direct-copy limit (256 / SIMD-pair;
    // < 128 per operand for 2 WG/CU co-residency). The codegen picks P per tile
    // to satisfy this -- see _ctdm_pick_num_slots() in opus_gemm_common.py.
    static constexpr int kNumSlots = NUM_SLOTS_;                 // prefetch depth P (2 or 3)
    static_assert(kNumSlots == 2 || kNumSlots == 3,
                  "kNumSlots (prefetch depth P) must be 2 or 3");

    // Target WG/CU co-residency. When the in-flight TDM request count
    // (rows * P * B_K*2/256 per operand) would exceed the 2-WG safe budget
    // (< 128/operand) but LDS could still fit 2 WG, we MUST force 1 WG/CU so two
    // workgroups don't oversubscribe a SIMD-pair's 256-request direct-copy budget
    // (which deadlocks the TDM engine). The codegen sets this per (tile,P) -- see
    // _ctdm_pick_configs() in opus_gemm_common.py. 1 WG/CU is enforced portably
    // via LDS padding below (a WG using > 160 KB LDS leaves no room for a second
    // on the 320 KB/CU budget), avoiding fragile occupancy attributes.
    static constexpr int kWgPerCu = WG_PER_CU_;
    static_assert(kWgPerCu == 1 || kWgPerCu == 2, "kWgPerCu must be 1 or 2");

    // Cluster-launch variant geometry: a kClusterWgM x kClusterWgN grid of WGs
    // per cluster (used by the clusterlaunch pipeline's __cluster_dims__ +
    // CLUSTER_LOAD_ASYNC multicast). Ignored by the plain-grid pipeline.
    static constexpr int kClusterWgM = CLUSTER_WG_M_;
    static constexpr int kClusterWgN = CLUSTER_WG_N_;
    // TDM multicast fans out to AT MOST 5 WGs per group: A is shared by the
    // kClusterWgN WGs of a column, B by the kClusterWgM WGs of a row, so neither
    // cluster dim may exceed 5. The total cluster WG count is also capped at 16
    // (the 16-bit per-cluster workgroup_mask).
    static_assert(kClusterWgM >= 1 && kClusterWgN >= 1 &&
                  kClusterWgM <= 5 && kClusterWgN <= 5 &&
                  kClusterWgM * kClusterWgN <= 16,
                  "cluster dims must be 1..5 per side (TDM multicast <= 5 WGs) "
                  "and kClusterWgM*kClusterWgN <= 16 (16-bit workgroup_mask)");
    static constexpr int kSlotBytesA = kSlotElemsA * (int)sizeof(DataA);
    static constexpr int kSlotBytesB = kSlotElemsB * (int)sizeof(DataB);

    static constexpr int kSegBytesA = kNumSlots * kSlotElemsA * (int)sizeof(DataA);
    static constexpr int kSegBytesB = kNumSlots * kSlotElemsB * (int)sizeof(DataB);
    // Real A/B LDS footprint.
    static constexpr int kSegBytesAB = kSegBytesA + kSegBytesB;
    // 1-WG/CU enforcement via LDS padding: if this instance must run 1 WG/CU
    // (kWgPerCu==1) but its A/B LDS would otherwise fit two WGs (<= 160 KB), pad
    // the shared allocation past 160 KB so only one WG fits on the 320 KB/CU
    // budget. (The pad tail is never accessed.) When kWgPerCu==2 or the tile is
    // already > 160 KB, no pad is added.
    static constexpr int kHalfLds = 160 * 1024;
    static constexpr int kLdsTotalBytes =
        (kWgPerCu == 1 && kSegBytesAB <= kHalfLds) ? (kHalfLds + 1024) : kSegBytesAB;
    // gfx1250 LDS max ~320KB.
    static_assert(kLdsTotalBytes <= 320 * 1024, "LDS exceeds 320KB");

    // Workspace plain store: fp32 dwordx4.
    static constexpr int kCVec = 16 / (int)sizeof(DataAcc);      // 4 (fp32)

    // ── Warp-derived WMMA register-decomposition constants ───────────────────
    // (computed from kWarpRt so device/host passes match)
    static constexpr int kReptA = kWmmaM * kWmmaK / kWarpRt / kVecA;
    static constexpr int kReptB = kWmmaN * kWmmaK / kWarpRt / kVecB;
    static constexpr int kGrpKA = kWarpRt / kWmmaM;
    static constexpr int kGrpKB = kWarpRt / kWmmaN;

    // ── scheduler hints (sched_group_barrier counts per ds-prefetch half) ────
    // These counts are NOT optional perf hints: they force ds_read-before-WMMA
    // ordering, so they MUST equal the real per-half instruction counts. They
    // scale with the consumer register expansion (kExpM for A, kExpN for B,
    // kExpKHalf K-subtiles per half). Hard-coding them for the kExpM==kExpN==1
    // base (DS=8, WMMA=2) under-counts the high-expand tiles (e.g. tileM
    // B_M=64 -> kExpM=2 with B_N=64 -> kExpN=4) and the resulting mis-schedule
    // surfaces as NaN at the software-pipeline tail (per-split k_steps%3==2).
    //   ds_reads / half = (kExpM*kReptA + kExpN*kReptB) * kExpKHalf
    //   wmmas    / half =  kExpM*kExpN * kExpKHalf
    // Base (kExpM=kExpN=1, kReptA=kReptB=2, kExpKHalf=2): DS=8, WMMA=2.
    static constexpr int kSchedDsMask   = 0x100;                 // DS_READ group
    static constexpr int kSchedDsCount  = (kExpM * kReptA + kExpN * kReptB) * kExpKHalf;
    static constexpr int kSchedWmmaMask = 0x008;                 // WMMA group
    static constexpr int kSchedWmmaCount = kExpM * kExpN * kExpKHalf;

#if (defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)) && (__clang_major__ >= 22)
    // tdm_window types live only where tdm_window is available (gfx1250 device
    // pass + host pass, clang>=22). Non-gfx1250 device passes and clang-20
    // (ROCm 7.1) CI never reference them -- opus::tdm_window is gated identically.
    using WindowA = opus::tdm_window<DataA, kTdmK, kARows, 0, 0, 0,
                                       1, 0, 0, 0, 1, 0, 0, 0, 0,
                                       kLdsPadEn, kPadInterval, kPadAmount, opus::seq<>>;
    using WindowB = opus::tdm_window<DataB, kTdmK, kBRows, 0, 0, 0,
                                       1, 0, 0, 0, 1, 0, 0, 0, 0,
                                       kLdsPadEn, kPadInterval, kPadAmount, opus::seq<>>;
#endif
};

// ── smem -> register read layouts (device-only; host pass compiles them too
//    for vtype_c consistency). Reference each traits member; no local
//    constexpr beyond loop-index helpers. ────────────────────────────────────
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)

// smem TILE layout (what the TDM write produces, ra/rb read): row-major
// [rows][K] with padded row pitch.
template <typename T>
__device__ inline auto make_layout_sa_ctdm() {
    return opus::make_layout<0>(
        opus::make_tuple(opus::number<T::kARows>{}, opus::number<T::kTdmK>{}),
        opus::make_tuple(opus::number<T::kSmemPitch>{}, 1_I));
}
template <typename T>
__device__ inline auto make_layout_sb_ctdm() {
    return opus::make_layout<0>(
        opus::make_tuple(opus::number<T::kBRows>{}, opus::number<T::kTdmK>{}),
        opus::make_tuple(opus::number<T::kSmemPitch>{}, 1_I));
}

// A operand (M x K) read layout. wave_m selects this consumer's M sub-tile
// (TileM: 0..1; TileN: always 0).
template <typename T>
__device__ inline auto make_layout_ra_ctdm(int lane_id, int wave_m) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpM>{}, opus::number<T::kTileM>{}, opus::number<T::kWmmaM>{},
        opus::number<T::kExpKHalf>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptA>{}, opus::number<T::kGrpKA>{}, opus::number<T::kVecA>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitch, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_m, lane_id % T::kWmmaM, 0, lane_id / T::kWmmaM}));
}

// B operand (N x K) read layout. wave_n selects this consumer's N sub-tile
// (TileN: 0..1; TileM: always 0).
//
// N-dim decomposition order MUST be (kExpN[y] outer, kTileN[p]=wave_n inner,
// kWmmaN), mirroring the A-side (kExpM outer, kTileM inner) and the C-output
// tile layout tile_shape_c=(expd_m,tile_m,expd_n,tile_n) where expd_n is outer
// and tile_n(=wave_n) is inner. If kTileN were placed outer here (wave_n owns a
// contiguous N half) the B-read N positions would NOT match where the C-store
// scatters them whenever BOTH kTileN==2 AND kExpN>1 -> wrong values for tileN
// with B_N>32. (kExpN==1 or kTileN==1 hide the mismatch, which is why tileM
// kExpN>1 and tileN B_N==32 were already correct.)
template <typename T>
__device__ inline auto make_layout_rb_ctdm(int lane_id, int wave_n) {
    constexpr auto shape = opus::make_tuple(
        opus::number<T::kExpN>{}, opus::number<T::kTileN>{}, opus::number<T::kWmmaN>{},
        opus::number<T::kExpKHalf>{}, opus::number<T::kTileK>{},
        opus::number<T::kReptB>{}, opus::number<T::kGrpKB>{}, opus::number<T::kVecB>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<0>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::kSmemPitch, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_n, lane_id % T::kWmmaN, 0, lane_id / T::kWmmaN}));
}

#endif  // __gfx1250__ || !__HIP_DEVICE_COMPILE__

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
//
// GROUP_N_ = the N extent one B-scale byte covers. This is the SECOND half of
// the scale granularity and it applies to B only; A is always per-row (its
// granularity along M is 1, which is what "1x128" means on the A side).
//
//   1   -> per-column B scale, w_scale is [batch, N, K/GROUP_K]
//   128 -> DSV3/DSV4 128x128 block scale, w_scale is [batch, N/128, K/GROUP_K]
//
// This matters far more than it looks. gfx950's tuned DSV4 kernels are named
// ..._16x16x128_1x128x128_... -- the "1x128x128" is (A granularity 1x128,
// B granularity 128x128), so 128 is the shipping configuration and 1 is the
// generic one. At GROUP_N_ = 128 every column of a WMMA tile shares one scale
// byte, which turns the B scale fetch from a 16-way per-lane gather into a
// single wave-uniform access, and shrinks a whole workgroup's B-scale working
// set from B_N x K/GROUP_K bytes to (B_N/128 or 1) x K/GROUP_K -- 32 to 64
// bytes for the entire K range at n = 1024, k = 4096.
//
// Default is 1 so that every tile alias written before this parameter existed
// keeps its exact previous behaviour.
template<int BLOCK_SIZE_,
         int B_M_, int B_N_, int B_K_,
         int LAYOUT_,
         typename D_A_, typename D_B_, typename D_C_, typename D_ACC_,
         int GROUP_K_   = 128,
         int NUM_SLOTS_ = 3,
         int WG_PER_CU_ = 2,
         int GROUP_N_   = 1,
         bool SF_A_LDS_ = false,
         bool SF_B_LDS_ = false,
         int SF_A_TDM_KG_  = 0,
         int SF_A_TDM_PAD_ = 16>
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
    static_assert(BLOCK_SIZE % kWarp == 0,
                  "BLOCK_SIZE must be a whole number of wave32 waves");
    static_assert(kNumConsumerWaves >= 1, "need at least one consumer wave");

    // WHY BLOCK_SIZE IS WORTH RAISING ABOVE 128. The producers do not consume any
    // math: their steady-state loop is one barrier wait, one scalar address
    // advance, one TDM descriptor, one s_wait_tensorcnt. The transfer itself is
    // the TDM engine's, so the data never touches a producer VGPR and never
    // issues to a VALU or WMMA pipe. What a producer does occupy is a wave slot,
    // and therefore a SIMD.
    //
    // A CU has 4 SIMDs and waves are dispatched round-robin (wave i -> SIMD
    // i % 4). At BLOCK_SIZE = 128 that is 4 waves: w0/w1 (producers) land on
    // SIMD0/SIMD1 and w2/w3 (consumers) on SIMD2/SIMD3, so SIMD0 and SIMD1 have
    // no compute-capable wave resident for the whole kernel and their matrix
    // pipes receive zero instructions. Half the CU's WMMA throughput is
    // unreachable, and no amount of retiling recovers it.
    //
    // Note that WG_PER_CU = 2 does NOT fix this: the second workgroup numbers its
    // waves from 0 again, so SIMD0 collects two producers and SIMD2 two
    // consumers. Consumer waves double but stay on the same two SIMDs.
    //
    // Raising BLOCK_SIZE does fix it, because consumers are the waves past
    // kNumProducerWaves and they keep wrapping around all 4 SIMDs. At
    // BLOCK_SIZE = 192 (6 waves) the consumers are w2..w5 -> SIMD2, SIMD3,
    // SIMD0, SIMD1: every SIMD hosts exactly one. The producer:consumer ratio
    // also improves from 1:1 to 1:2, which is nearly free given the above.
    //
    // The consumer split stays 1-D (all-M or all-N), so kTileM * kTileN is
    // kNumConsumerWaves by construction and the pipeline's wave_m/wave_n
    // derivation is unchanged. The cost is that B_N (or B_M) must grow with the
    // wave count to keep kExpN >= 1 -- see the kid4 alias below.
    static constexpr int kTileM = (LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? kNumConsumerWaves : 1;
    static constexpr int kTileN = (LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? 1 : kNumConsumerWaves;
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

    // -- B scale granularity along N -----------------------------------------
    static constexpr int kGroupN = GROUP_N_;
    static_assert(kGroupN == 1 || kGroupN == 128,
                  "GROUP_N must be 1 (per-column B scale) or 128 (DSV3/DSV4 "
                  "128x128 block scale)");
    // Rows of w_scale this tile spans: N columns at GROUP_N == 1, N/128 blocks
    // at GROUP_N == 128. ceil, because B_N need not be a multiple of GROUP_N
    // (B_N = 32 and 64 both sit inside one 128-block).
    static constexpr int kSfBRowsPerTile = (kBlockN + kGroupN - 1) / kGroupN;

    // Is one consumer wave's whole N span inside a SINGLE scale block? A wave
    // owns kExpN * kWmmaN consecutive columns starting at a multiple of that
    // same quantity (tile_col is a multiple of kBlockN = kExpN*kWmmaN*kTileN),
    // so the span cannot straddle a block boundary exactly when the block size
    // is a whole multiple of the span.
    //
    // This is ONLY an optimisation property: permission for the B scale address
    // to drop its lane_id term, because every lane would have computed the same
    // block row anyway. It is NOT the choice of scale layout -- that is kGroupN's
    // job, and conflating the two is a silent-wrong-answer bug. The two agree for
    // every tile defined below, but they are independent: a B_N=192 tile on 4
    // consumer waves has kExpN=3, whose 48-column span does not divide 128, so
    // uniformity would be false while the tensor is still blocked.
    //
    // At kGroupN == 1 it is false by construction (a 16-column WMMA tile spans 16
    // distinct scale columns) and the per-lane gather is genuinely required.
    static constexpr bool kSfBUniformOverN = (kGroupN % (kExpN * kWmmaN) == 0);

    // Distinct B scale values one consumer wave loads per WMMA K-tile: kExpN
    // under a per-column scale, but ONE when the wave's span is inside a single
    // block (see kSfBUniformOverN).
    static constexpr int kSfBLoadsPerK = kSfBUniformOverN ? 1 : kExpN;
    // Stage A's scales in LDS via one prologue DMA instead of a per-lane global
    // gather; the panel geometry and the reasoning are down in the LDS section.
    static constexpr bool kSfALds = SF_A_LDS_;
    // Same for B. Separate flags because the two sides are not comparable: A is
    // always a per-lane gather of kExpM rows, while B is already down to ONE
    // wave-uniform load per K-tile whenever kSfBUniformOverN holds. Staging both
    // is what gfx950's PRELOAD_SF_LDS does, and A-alone is what was measured
    // here first (see the kSfALds section).
    static constexpr bool kSfBLds = SF_B_LDS_;
    // GLOBAL scale loads one wave issues per WMMA K-tile. Each side drops out
    // when staged, and that is the point of counting them here rather than
    // counting scale operands: the early/late choice below is an argument about
    // memory round trips, and an LDS panel read is not one.
    static constexpr int kSfLoadsPerK =
        (kSfALds ? 0 : kExpM) + (kSfBLds ? 0 : kSfBLoadsPerK);

    // -- where to ISSUE the scale loads within a K-tile ----------------------
    // The scale bytes come from global memory and depend on nothing the
    // producers put in LDS, so they may be issued before the ds_reads instead of
    // after s_wait_dscnt(0). Doing so replaces a fully exposed global latency
    // (the wave stalls at s_wait_loadcnt with nothing else in flight) with one
    // combined s_wait_loadcnt_dscnt that covers the whole LDS read burst.
    //
    // That is a win only while the batch is SMALL. Issued early, every scale
    // load must complete before the first WMMA; issued late they are spread
    // through the WMMA sequence and cover each other. Measured on gfx1250,
    // n=1024 k=4096, taking per-shape minima over 4 sweeps and keeping only
    // shapes whose run-to-run spread was under 2% in both builds:
    //
    //   kSfLoadsPerK  tiles                     early vs late
    //   2             kid1, kid4, kid8, kid9     1.013x .. 1.049x   all gains
    //   3             kid5, kid6                 0.999x .. 1.033x   gain/neutral
    //   4             kid10                      0.944x .. 0.955x   REGRESSION
    //   5             kid7                       0.938x .. 0.998x   REGRESSION
    //   12            kid0                       mixed, 0.824x seen
    //
    // The threshold cleanly separates every measured gain from every measured
    // regression, so it is set from that data and not from first principles.
    // Note the prefill tile (kExpM=8, kExpN=4) lands firmly on the late side,
    // which is also the conservative answer for the shipping prefill path.
    //
    // With the gate in place, re-measured the same way: the early tiles keep
    // 1.014x .. 1.053x and the late tiles are back on the baseline (kid7
    // 0.992x .. 1.015x, kid10 0.996x .. 1.004x, kid0 unchanged, and kid0/kid7
    // VGPR counts identical to before at 542 / 150, no spills anywhere).
    static constexpr bool kSfEarly = (kSfLoadsPerK <= 3);
    // Where A's fill goes. A panel read is a ds_read, so it belongs with the
    // other ds_reads where the existing s_wait_dscnt(0) covers it in the same
    // wait -- issued late it would need a second dscnt wait of its own, right
    // before the WMMA that consumes it.
    //
    // Only a scheduling choice, NOT correctness: the compiler's waitcnt pass
    // inserts whatever wait a ds_read needs regardless of where it sits, and the
    // explicit s_wait_dscnt(0) here is a scheduling device. Verified rather than
    // assumed -- forcing the panel onto the late path leaves all 17 tests green.
    static constexpr bool kSfAEarly = kSfALds || kSfEarly;
    static constexpr bool kSfBEarly = kSfBLds || kSfEarly;

    // e8m0 bytes one row of this tile needs per K step.
    static constexpr int kSfPerRowPerStep = kBlockK / kGroupK;
    // A-scale bytes / B-scale bytes a whole tile needs per K step. Sized here so
    // the pipeline's optional LDS scale panel has one place to come from.
    static constexpr int kSfATileBytes = kBlockM * kSfPerRowPerStep;
    static constexpr int kSfBTileBytes = kSfBRowsPerTile * kSfPerRowPerStep;

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

    // -- A-scale LDS panel (SF_A_LDS_) ---------------------------------------
    // A's scale is the one access in this kernel that never got fixed. It is a
    // per-lane gather: lane l reads row a_row0 + l%16, and rows are stride_sfa
    // bytes apart (g * K/GROUP_K = 256 B at b=8, k=4096), so 16 lanes touch 16
    // separate cache lines to collect 16 bytes. GROUP_N only fixed the B side.
    // With GROUP_K=128 each of those is a global_load_u8 -- one byte, broadcast
    // to the operand dword -- and the prefill tile issues kExpM = 8 of them per
    // K-tile. An ablation that replaces the A scale with a constant is worth
    // ~11% on kid9 at b=8 m=128 measured at constant occupancy.
    //
    // Staged instead as a whole-K panel in LDS, filled ONCE in the prologue.
    // What makes that work is that A's scale is small and K-contiguous: the
    // panel is kBlockM x K/GROUP_K bytes, 4 KB for the prefill tile at k=4096
    // and 512 B for a decode tile, so one fill covers the whole kernel. No
    // ring, no per-slot handshake, and no new named barrier -- ids
    // 1..3*kNumSlots already reach 9 and the binit/bjs/bjsw chains silently
    // alias anything past 9. It publishes on the barrier that already publishes
    // the barrier inits.
    //
    // HISTORY, because the first attempt is why this one looks the way it does.
    // Every ratio in this paragraph predates the measurement fix and is off by
    // roughly 9 points -- see the error-bar note further down before quoting any
    // of them. What holds up is the width TREND, not the levels.
    // The fill was a TDM 2D window, kBlockM rows x kSfAPanelKG bytes with a
    // padded pitch. It worked, and it was a REGRESSION at k=4096: 0.877 geomean
    // vs kid0, recovered to 0.969 only by matching the panel width to K and
    // moving the issue off the producer wave, and exactly neutral (1.005) at
    // k=16384 where 64 K steps amortise every fixed cost away. The cost tracked
    // the D#'s ROW GEOMETRY rather than the payload -- at fixed k=4096 the ratio
    // went 0.935 / 0.959 / 0.969 for widths 128 / 64 / 32. 128 row-transfers of
    // 32 bytes each is the wrong instrument for 4 KB.
    //
    // The re-measurement (kid17/kid18, below) keeps that conclusion but moves the
    // reason: geometry is the SECOND-order term, and the first is that a DMA fill
    // is issued by one wave while the workgroup waits, where the cooperative fill
    // is issued by all of them. Hence the shape dependence -- the penalty is 12%
    // at b=8 m=512 and 1% at m=2048, and the old "neutral at k=16384" reading is
    // the same effect seen from the far end.
    //
    // Hence the cooperative fill, which is the shape gfx950 already uses (see
    // PRELOAD_SFA_LDS in opus_bmm_pipeline_a8w8_mxscale_gfx950.cuh and
    // PRELOAD_SF_LDS in the flatmm splitk pipeline): all BLOCK_SIZE threads
    // grid-stride over the panel's flat byte count with the widest aligned
    // vector the geometry allows. At k=4096 with g=8 the widths are 32 | 256, so
    // VEC=16, and the prefill tile's 4 KB panel fills in two iterations per
    // thread.
    //
    // Two things follow from the fill being a flat vector copy, both
    // load-bearing:
    //   * the LDS row pitch tracks the RUNTIME K/GROUP_K (rounded up, plus the
    //     bank pad below) instead of a compile-time width, so a small K wastes
    //     LDS and nothing else -- which is the whole difference from the TDM
    //     version. Being a runtime value costs nothing: pack_sf already takes
    //     its row stride as a plain int.
    //   * a chunk must not span two panel rows nor land unaligned, so the vector
    //     width has to divide both K/GROUP_K and stride_sfa. Both are runtime,
    //     so that test is in the pipeline, with a 1-byte fallback.
    //
    // MEASURED, b=8/16, m=128..2048, n=1024, k=4096, per-shape minima over 12
    // INTERLEAVED sweeps, geomean over the shapes whose spread stayed under 2%:
    //
    //   kid0  -> kid13   A staged            1.068
    //   kid13 -> kid14   B staged as well    1.020
    //   kid0  -> kid14   both                1.096
    //
    // and the win GROWS with K, which is the right shape for a once-per-
    // workgroup fill paying off per K-tile (b=8, kid0 -> kid14):
    //
    //   m      k=4096   k=8192   k=16384
    //   256     1.090    1.133    1.179
    //   512     1.117    1.034    1.102
    //
    // ISA at kid0 -> kid13 -> kid14, per unrolled body of 10 K-steps:
    //   global_load_u8   120 -> 40 -> 0        every A and B scale byte gone
    //   ds_load_u16        0 -> 40 -> 60       what replaces them
    //   v_perm_b32         0 -> 80 -> 120      byte extract, replacing...
    //   v_mul_lo_u32     120 -> 46 -> 10       ...the broadcast multiply
    //   s_wait_loadcnt   120 -> 43 -> 5
    //   VGPR             542 -> 523 -> 422     no spills anywhere
    //   LDS           202752 -> 221184 -> 239760   (of 320 KB, 1 WG/CU either way)
    //
    // READ THIS BEFORE TRUSTING A RATIO IN THIS FILE. The numbers above replace
    // an earlier set that said the opposite -- kid13 recorded at 0.990, with
    // kid0 at 38.60 us for b=8 m=512 -- and here is why the new set is the one
    // to believe. Four independent things now agree:
    //
    //   * a same-kid-vs-itself control pair (0, 0) reads 1.000. Any protocol
    //     that cannot pass this is not measuring what it claims to.
    //   * two unrelated compilers agree on the ratios. Built with the container
    //     toolchain (aa451e1f) and with the host's /opt/rocm (7f910a89), the
    //     panel reads 1.077/1.022/1.099 and 1.068/1.020/1.096 respectively. The
    //     effect survives a change of code generator, so it is not a codegen
    //     accident.
    //   * sequential and interleaved sweep order agree to 0.001, so ordering
    //     bias is not carrying the result either.
    //   * the card is not in a different state than it was. An unrelated kernel
    //     with numbers recorded on this machine (inverse_rope_group_quant at
    //     h=128 g=16: 11.32 us at s=512, 237.8 us at s=16384) reproduces its
    //     history to within 1%.
    //
    // What did NOT survive: the old 38.60 us kid0. Both compilers put kid0 at
    // ~48 us today, so that reading did not come from this kid0 on this card,
    // and no configuration has been found that reproduces it. Treat the whole
    // earlier row -- ratio and absolute alike -- as measuring some
    // mid-experiment binary, and discard it rather than trying to reconcile it.
    //
    // Rules that follow, all learned the expensive way:
    //   * BUILD AND MEASURE IN THE yzhou_latest CONTAINER. The host has three
    //     toolchains on it and the SDK wheel one (HIP 7.2.0) miscompiles at
    //     least one aiter kernel into an out-of-bounds read, so a host build is
    //     not merely a different number, it can be a wrong one.
    //   * touch the TUs (or just run rb.sh) before every timing run. This ninja
    //     setup writes no depfiles, so a header-only edit leaves it reporting
    //     "no work to do" and the run silently re-measures the old binary. A
    //     ratio from a stale binary is not noisy, it is wrong, and it looks
    //     clean.
    //   * record ABSOLUTE times next to every ratio. The old table was
    //     ratio-only, which is exactly why a bogus baseline went unnoticed.
    //   * cross-check the machine against an unrelated kernel's recorded numbers
    //     before blaming the machine.
    //
    // The earlier mechanism comparison (TDM width 128 = 0.877, width 32 = 0.969,
    // cooperative fill = 0.988, +wide read = 1.000, +pad16 = 0.990) is all from
    // the bad era, and its error bar is known: that same cooperative fill reads
    // 1.077 when measured properly, so the era was off by ~9 points. Read it
    // accordingly.
    //   * SURVIVES: the width trend 0.935 / 0.959 / 0.969 for TDM widths
    //     128 / 64 / 32 at fixed k=4096. Those three share one baseline and one
    //     4 KB payload, so the trend is internal to the era's error -- the cost
    //     tracked the descriptor's row geometry, not the bytes moved. That is
    //     the mechanism argument, and it stands.
    //   * WAS NOT ESTABLISHED THEN, BUT IS NOW: "the cooperative fill beat TDM".
    //     It rested on 0.988 vs 0.969, a 1.9-point gap inside a 9-point error,
    //     which is no evidence at all. It has since been measured properly
    //     against kid17/kid18 -- see the kSfATdmKG block -- and it holds, at
    //     0.99 on the large shapes and 0.88 on b=8 m=512. Right answer, but it
    //     was not the old numbers that earned it.
    //
    // THE ERROR BAR IS BETWEEN RUNS, NOT WITHIN THEM, and this is the trap the
    // bad era fell into. This host runs other people's containers, which take
    // the GPU in bursts of a few seconds. Under load kid0 at b=16 m=2048 times
    // at 421..533 us against 345 us idle, and the pair ratio moves with it: kid13
    // vs kid17 read 1.04 / 1.04 / 1.03 in three contended runs and 0.990 / 0.994
    // in the two quiet ones. The sign of a 1% effect flips with the neighbours.
    //
    // Interleaving the sweeps does NOT save this. It was supposed to -- a burst
    // inside a sweep hits both kids of the pair -- and it demonstrably does not,
    // because those contended runs were interleaved too.
    //
    // What makes it insidious is that the contended runs look CLEAN: forty reps
    // taken back to back share one machine state, so they agree with each other
    // to 0.002 while disagreeing with the next run by 0.05. A tight spread is
    // evidence about the last thirty seconds and nothing else. The only defence
    // that worked is to record the BASELINE'S ABSOLUTE TIME with every ratio and
    // only compare runs that agree on it: the two runs that both put kid0 at
    // 345 us also both put kid13-vs-kid17 at 0.990/0.994. _sfa/bench_tdm.py does
    // this selection.
    //
    // Unresolved, and left that way: kid0's old 38.60 us at b=8 m=512 is still
    // 24% under the 47.85 us that the VERIFIED-QUIET state gives today, so
    // contention does not explain that one. It stays discarded.
    //
    // WHY IT WINS, now that it does. gfx950 sees the same effect from the same
    // change (PRELOAD_SF_LDS: ATT measured ~20% of consumer cycles stalled on
    // vmcnt for the per-K-tile scale load, +8..26% TFLOPS from staging both
    // panels). The mechanism there is that its scale loads share the vmcnt
    // budget with the A/B tile async_loads, so waiting for tile data also waits
    // for scale. gfx1250 has no such coupling -- tiles retire on tensorcnt,
    // scale on loadcnt -- and its baseline is genuinely well pipelined: kid0
    // issues all 12 of a round's scale loads under s_clause and drains them with
    // an s_wait_loadcnt staircase (0xb, 0xa, ... 0x0), so they cost roughly one
    // memory latency rather than twelve. That argument was used to predict a
    // null result, and it was wrong: one latency per K-tile, times 40 K-tiles,
    // against a fill that is paid once, is still worth ~7%, and the 120 -> 10
    // v_mul_lo_u32 and 542 -> 422 VGPR are real on top of it. "Already
    // pipelined" is not the same as "free".
    //
    // kSfALds/kSfBLds are declared up beside kSfLoadsPerK, which reads them.

    // Panel width bound in K-groups. Caps the K this tile accepts at
    // kSfAPanelKG * GROUP_K = 16384 at GROUP_K=128, enforced in the launcher.
    // Unlike the TDM version this is ONLY an allocation bound -- the layout uses
    // the runtime K/GROUP_K -- so a smaller K now wastes LDS and nothing else.
    static constexpr int kSfAPanelKG = 128;

    // TDM FILL FOR THE A PANEL, as a measurable alternative to the cooperative
    // one rather than a replacement. 0 is the cooperative fill; a positive value
    // is the descriptor's COMPILE-TIME tile width in K-groups, which therefore
    // also caps K at kSfATdmKG * GROUP_K.
    //
    // These exist because the claim "the cooperative fill beat TDM" had never
    // been established: it rested on 0.988 vs 0.969, a 1.9-point gap taken in
    // the era whose error is now known to be ~9 points. They are kept, rather
    // than deleted once the answer came back, because the answer is what makes
    // the cooperative fill's shape a decision instead of an accident, and
    // because re-deriving it costs a day.
    //
    // The comparison worth running is kSfATdmKG = K/GROUP_K with
    // kSfATdmPad = kSfPanelPad, because then the panel's pitch, its LDS layout
    // and every read instruction are IDENTICAL to the cooperative tile's -- at
    // k=4096 both are 32 + 16 = 48 -- so the pair difference is the fill and
    // nothing else. kSfATdmKG = 128 with pad 4 reproduces the original instead.
    //
    // MEASURED, and the old conclusion survives its bad numbers: the cooperative
    // fill wins. kid13 as base, k=4096, n=1024, control pair at 1.000, forty
    // reps per point with the machine verified quiet (see the note below on what
    // that means here):
    //
    //   shape          13 vs 17 (w32)   13 vs 18 (w128)
    //   b=8  m=512         0.877             0.782
    //   b=8  m=2048        0.990             0.983
    //   b=16 m=2048        0.994             0.992
    //
    // So TDM costs ~1% on the large shapes and 12% on b=8 m=512, and the width
    // axis stacks on top of that rather than rescuing it.
    //
    // The reason is the ISSUE, not the transfer. The panel is filled once in the
    // prologue, and TDM fills it from ONE wave that then sits on
    // s_wait_tensorcnt while the rest of the workgroup waits at the barrier --
    // where the cooperative fill spends all BLOCK_SIZE threads on it. That is a
    // serialisation the ring's producers can hide behind a 40-step main loop and
    // a once-paid prologue cannot, which is exactly why the penalty grows as the
    // shape gets shorter: 0.994 at b=16 m=2048, 0.877 at b=8 m=512.
    //
    // Two geometry notes, both now confirmed rather than assumed. opus.hpp's
    // padding tag says a pad interval under 128 B demotes part of the transfer
    // off the direct-copy path, and the interval here IS the panel width -- so a
    // width matched to K is 32 B at k=4096 and takes the demoted path, while a
    // width of 128 B reaches the fast path only by fetching 4x the bytes the
    // tile needs. The measurement says the over-fetch is the worse of the two
    // (kid18 below kid17 everywhere), and that the choice between them is not
    // the point anyway, because both lose to not using the engine at all.
    static constexpr int  kSfATdmKG  = SF_A_TDM_KG_;
    static constexpr int  kSfATdmPad = SF_A_TDM_PAD_;
    static constexpr bool kSfATdm    = kSfALds && (kSfATdmKG > 0);
    static_assert(kSfATdmKG == 0 || SF_A_LDS_,
                  "SF_A_TDM_KG_ > 0 without SF_A_LDS_: there is no panel to fill");
    static_assert(!kSfATdm || (kSfATdmKG * kGroupK) % kGroupK == 0, "");
    // The pad tag encodes the interval as 8 << enc BYTES, so the width must be a
    // power of two; the amount is 4 * (enc + 1) bytes, so the pad must be a
    // whole DWORD. Both are e8m0 bytes here, hence the plain element counts.
    static_assert(!kSfATdm || (kSfATdmKG & (kSfATdmKG - 1)) == 0,
                  "SF_A_TDM_KG_ must be a power of two (pad interval is 8 << enc)");
    static_assert(!kSfATdm || kSfATdmKG >= 8,
                  "SF_A_TDM_KG_ * 1 byte must be >= 8 (pad interval field floor)");
    static_assert(!kSfATdm || (kSfATdmPad >= 4 && kSfATdmPad % 4 == 0),
                  "SF_A_TDM_PAD_ must be a whole DWORD >= 4");
    // Both panels are indexed with ONE pitch in the pipeline, and the TDM one is
    // compile-time while the cooperative one tracks K. Rather than carry two,
    // keep the experiment A-only -- which is also the only side it asks about.
    static_assert(!kSfATdm || !SF_B_LDS_,
                  "SF_A_TDM_KG_ with SF_B_LDS_: the two panels would need "
                  "different row pitches; measure the A fill on its own");
    // The cooperative fill covers A only when TDM has not taken it over.
    static constexpr bool kSfACoop = kSfALds && !kSfATdm;

    // Read the panel kExpK bytes at a time instead of one, which is the only
    // part of this that removes INSTRUCTIONS rather than relocating them. A
    // row's kExpK bytes for one K-step are consecutive in the panel (kg advances
    // by one per ik), so one ds_read covers the whole step and each WMMA then
    // extracts its byte with a v_perm_b32 -- which replaces the broadcast
    // multiply the byte path was already paying, so the VALU count does not
    // move and only the LDS count does. kid13: 80 ds_read_u8 -> 40 ds_read_u16.
    //
    // Only for kScaleBcast: at kGroupK==32 one WMMA already consumes 4 bytes, so
    // pack_sf's read is wide there to begin with. kExpK in {2,4} is what the
    // 2/4-byte LDS reads can be aligned for; anything else keeps the byte path.
    static constexpr bool kSfAWideRead =
        kSfALds && kScaleBcast && (kExpK == 2 || kExpK == 4);
    static constexpr bool kSfBWideRead =
        kSfBLds && kScaleBcast && (kExpK == 2 || kExpK == 4);

    // Panel row pitch = round16(K/GROUP_K) + kSfPanelPad. The pad is not
    // optional: 16 lanes read 16 CONSECUTIVE rows at one k_step, so the pitch IS
    // the bank stride, and an unpadded panel makes the dword stride K/GROUP_K/4,
    // whose gcd with the 32 banks grows with K -- so the conflict grows with K.
    // Measured unpadded, kid13 vs kid0 at b=8 n=1024:
    //
    //   k       pitch  dword stride  gcd(.,32)  ways  ratio
    //   4096      32          8          8        4   1.000
    //   8192      64         16         16        8   0.976
    //   16384    128         32         32       16   0.963   <- one bank
    //
    // The first fix was wrong in an instructive way. +4 makes the dword stride
    // ODD, hence coprime with 32 and fully conflict-free -- but pitch = 4 mod 8
    // cannot be 16B-aligned, so it caps the fill at a 4-byte store, which is 4x
    // the fill iterations. Measured, that trade is badly negative:
    // k=4096/8192/16384 went to 0.922/0.925/0.882, against 1.000/0.976/0.963
    // unpadded. The fill's width dominates the read's bank conflict.
    // (Both rows are from the pre-correction era described in the kSfALds
    // section, so read them as an ORDERING -- pad=4 lost to no pad, which is
    // the point -- and not as levels.)
    //
    // +16 keeps both: the dword stride becomes round16(K/GROUP_K)/4 + 4, whose
    // gcd with the 32 banks is 4 rather than growing with K, so the worst case
    // drops from 16-way to 4-way AND the fill keeps its 16-byte store.
    static constexpr int kSfPanelPad   = (kSfALds || kSfBLds) ? 16 : 0;
    static constexpr int kSfFillVecMax = 16;
    // The widest row pitch either panel can ask for, given the K cap.
    static constexpr int kSfPanelPitchMax = kSfAPanelKG + kSfPanelPad;
    // A COMPILE-TIME pitch for the TDM fill, because there the pitch is the D#'s
    // padding and so cannot track K at runtime. 0 means "use the runtime pitch",
    // which is what the cooperative fill does.
    static constexpr int kSfAPitchFixed = kSfATdm ? (kSfATdmKG + kSfATdmPad) : 0;

    // B-panel rows: the kGroupN blocks this tile's N range can touch. ceil, plus
    // one for a STRADDLE -- tile_col is a multiple of kBlockN, which need not be
    // a multiple of kGroupN, so the span can start mid-block. kid10 (B_N=192,
    // GROUP_N=128) is exactly that case and is the reason for the +1.
    static constexpr int kSfBPanelRows =
        (kBlockN + kGroupN - 1) / kGroupN + 1;
    static constexpr int kSegBytesSfALds =
        kSfALds ? kBlockM * (kSfATdm ? kSfAPitchFixed : kSfPanelPitchMax) : 0;
    static constexpr int kSegBytesSfBLds =
        kSfBLds ? kSfBPanelRows * kSfPanelPitchMax : 0;

    static constexpr int kSegBytesAB =
        kSegBytesA + kSegBytesB + kSegBytesSfA + kSegBytesSfB
        + kSegBytesSfALds + kSegBytesSfBLds;
    // The A panel's LDS base is kSegBytesA + kSegBytesB and the B panel's is that
    // plus kSegBytesSfALds. The fill stores up to kSfFillVecMax bytes relative to
    // a base, and the per-row offsets only guarantee alignment WITHIN a panel, so
    // each base has to carry that alignment itself. kSegBytesSfALds is a multiple
    // of kSfPanelPitchMax (= 144), which is 16-aligned, so B's base inherits A's.
    static_assert(!(kSfALds || kSfBLds)
                      || (kSegBytesA + kSegBytesB) % kSfFillVecMax == 0,
                  "scale panel LDS base must be kSfFillVecMax aligned");
    static_assert(!kSfBLds || kSegBytesSfALds % kSfFillVecMax == 0,
                  "B panel base = A base + kSegBytesSfALds, so that must keep "
                  "the alignment too");
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
    // A-scale panel window, used only when kSfATdm; the cooperative fill needs
    // no descriptor. Both template arguments have to stay well-formed when the
    // tile does NOT ask for TDM, because naming the alias instantiates it: pad
    // (0, 0) is padding<>'s "neither", and the dummy width is never issued.
    static constexpr int kSfATdmKGOrDummy = kSfATdm ? kSfATdmKG : 8;
    using PaddingSfA = opus::tdm_traits::padding<DataSf,
                                                kSfATdm ? kSfATdmKG  : 0,
                                                kSfATdm ? kSfATdmPad : 0>;
    using WindowSfA  = opus::tdm<DataSf, opus::seq<kSfATdmKGOrDummy, kBlockM>,
                                 PaddingSfA>;
    // The D# pad and the pitch the consumer reads with are two spellings of one
    // layout, so they cannot be allowed to drift.
    static_assert(PaddingA::pitch_elements == kSmemPitchA,
                  "kSmemPitchA must equal the D# padded row pitch");
    static_assert(!kSfATdm || PaddingSfA::pitch_elements == kSfAPitchFixed,
                  "kSfAPitchFixed must equal the A-panel D# padded row pitch");
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

// -- decode tiles (kid 1..3) ------------------------------------------------
// The tile above is a prefill tile: at the DSV4 decode shapes (n=1024, k=4096,
// m<=16) its grid is ceil(m/128) x ceil(n/128) x batch = 1 x 8 x b, so b=2
// launches 16 workgroups onto 256 CUs and the measured throughput is 6.5% of
// what the memory system delivers. B_M does NOT fix that -- ceil(m/B_M) is 1
// for every B_M at m=1 -- so the parallelism has to come from B_N, and the
// per-CU rate has to come from more waves resident per CU.
//
// Three variants, each adding ONE thing to the one before, so an A/B run
// attributes the difference instead of just reporting it:
//   kid1  B_N 128 -> 32   : 4x the workgroups at fixed b (B_M 16 only trims the
//                           wasted M work, which is second-order here).
//   kid2  + WG_PER_CU 2   : the small tile needs 36.75 KB, so unlike the prefill
//                           tile it is not forced to 1 WG/CU by LDS. This is the
//                           only lever that raises the ~27 GB/s per-CU ceiling.
//   kid3  + B_K 256 -> 512: the B_K the gfx950 tuner's decode winner uses
//                           (256x16x32x512).
// LDS: kid1/kid2 = 3 x (16x272 A + 2x4096 B) = 36.75 KB; kid3 = 72.75 KB. Both
// are under the 160 KB half-budget, so WG_PER_CU = 2 is legal (the
// static_assert above turns a wrong claim here into a build error).
//
// !! kid2 AND kid3 ARE NUMERICALLY WRONG. DO NOT SHIP THEM. !!
// Measured max relative error at n=1024, k=4096: correct (2^-8, the bf16 floor)
// for every shape whose grid is <= 256 workgroups, and wildly wrong the moment
// the grid exceeds the 256-CU count -- b=2/m=128 (512 WGs) and b=8/m=128 (2048
// WGs) and b=8/m=2048 (32768 WGs) all fail, while b=8/m=1 and b=8/m=16 (exactly
// 256 WGs) pass. The error is also nondeterministic run to run (kid3 at
// b=8/m=128 gave 112.0 and then 224.0), which makes it a race and not a bad
// index. kid1, which is this same geometry at WG_PER_CU = 1, is correct at every
// shape including 32768 WGs.
//
// The threshold sitting exactly at co-residency, for precisely the two tiles
// whose LDS admits a second workgroup, points at the named barriers. They are
// addressed by COMPILE-TIME id (1..3*kNumSlots) and s_barrier_init writes what
// the a16w16 pipeline this was ported from calls "the shared barrier-unit
// state". Two workgroups co-resident on one CU therefore init, join and signal
// the SAME physical barriers, so each one's DATA[s]/FREE[s] releases the other's
// waves early -- which also explains why kid2/kid3 look fast in that regime.
//
// This makes WG_PER_CU = 2 unusable with this pipeline as written, not merely
// unprofitable: the barrier ids would have to be partitioned per co-resident
// workgroup, and there is no workgroup-slot index available to do it with. The
// supported way to put more waves on a CU is therefore to put them in ONE
// workgroup, i.e. raise BLOCK_SIZE -- which is what kid4 does.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/16, /*B_N*/32, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_wg2_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/16, /*B_N*/32, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/2>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_k512_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/16, /*B_N*/32, /*B_K*/512,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/2>;

// -- kid4: 6 waves, so every SIMD hosts a consumer ---------------------------
// The reasoning is in the kNumWaves block above: kid0..kid3 all run 4 waves, and
// 4 waves leave two of the CU's four SIMDs with no compute wave resident. This
// tile changes BLOCK_SIZE 128 -> 192, giving 2 producers + 4 consumers, and the
// consumers wrap onto SIMD2, SIMD3, SIMD0, SIMD1 -- one each.
//
// MEASURED, and the result does NOT support the reasoning above. kid4 is the
// fastest decode tile in the sweep (1.28x - 2.03x over kid1 at b in {2,8},
// m in {1,16,128}), but almost none of that is the wave count. kid5 below is
// kid4's tile at 4 waves, and kid4 beats it by only 1.02x - 1.09x. The rest of
// kid4's margin over kid1 is B_N 32 -> 64, which kid5 gets too (1.21x - 1.99x
// over kid1 on its own).
//
// So filling the two idle SIMDs is worth 2-9%, not the ~2x that "half the matrix
// pipes are unreachable" predicts. The conclusion is that those SIMDs were idle
// for lack of work rather than for lack of a resident wave: the consumers are
// not WMMA-issue-limited, so giving the CU more of them changes little. That is
// consistent with the ~98 cyc/wmma measured on kid0 (the matrix pipe running at
// roughly 9% of issue rate) and with the per-CU bandwidth ceiling near
// 27 GB/s. The binding constraint is upstream of the WMMAs.
//
// kid4 is still the tile to ship at decode -- it is the fastest and it is
// correct at every shape measured -- but the next real lever is the scale fetch,
// not the wave count. See TODO(kernel) 2 in the pipeline: the consumer reads
// e8m0 bytes from GLOBAL per WMMA, which is explicitly the slow path, and its
// traffic scales with the workgroup count. That is also the most likely reason
// B_N 32 -> 64 pays as much as it does, since halving the grid halves both the
// redundant A tile loads (B_M = 16 rows per WG at m = 1) and the scale re-reads.
//
// WG_PER_CU stays 1 to match kid1. LDS is 3 x (16x272 A + 4x4096 B) = 60.75 KB,
// far under the 160 KB half-budget, so WG_PER_CU = 2 would be legal here -- but
// do not use it: see the kid2/kid3 warning above.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/64, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1>;

// -- kid5: kid4's control, 4 waves at kid4's tile ----------------------------
// kid4 moves two things away from kid1 at once: the wave count (128 -> 192
// threads) and B_N (32 -> 64). Comparing it only against kid1 therefore cannot
// say which one paid, and "a wider tile is better here" is a perfectly good
// competing explanation for kid4's win.
//
// This tile is the control that separates them. It is kid4's geometry exactly --
// same B_M/B_N/B_K, same NUM_SLOTS, same WG_PER_CU, same 60.75 KB of LDS, same
// launched grid -- at kid1's wave count. Its 2 consumer waves therefore each own
// kExpN = 2 N-subtiles instead of kid4's 1, so the tile's total work per
// workgroup is identical and only its distribution over SIMDs differs.
//
// kid4 vs kid5 is thus a single-variable experiment on wave count, and it is the
// one that decides whether the two idle SIMDs were the problem. Measured: they
// were not. kid4 leads kid5 by 1.02x - 1.09x, while kid5 alone already carries
// 1.21x - 1.99x over kid1. Keep this tile -- it is the reason the wave-count
// claim is bounded at "a few percent" instead of being credited with kid4's
// whole margin.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w4_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/16, /*B_N*/64, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1>;

// -- kid6, kid7: the B_N ladder past kid4 -----------------------------------
// B_N 32 -> 64 (kid1 -> kid5) was worth 1.21x - 1.99x, far more than the wave
// count was, so the question is where that stops paying. These two continue the
// same axis at kid4's wave count, changing nothing else:
//
//   kid4  B_N  64, kTileN 4, kExpN 1
//   kid6  B_N 128, kTileN 4, kExpN 2
//   kid7  B_N 256, kTileN 4, kExpN 4
//
// There has to be a knee, because B_N pulls two ways at once. Total B traffic is
// independent of it -- (N/B_N) workgroups x B_N x K is N x K either way -- so
// what a wider tile actually buys is fewer workgroups over which the REDUNDANT
// traffic is duplicated: the B_M = 16 rows of A that every workgroup loads even
// when m = 1, and the per-WMMA global e8m0 scale reads. What it costs is grid:
// at n = 1024 the N axis supplies only 1024/B_N tiles, so b = 2, m = 1 goes from
// 32 workgroups at kid4 to 16 at kid6 and 8 at kid7, on 256 CUs.
//
// MEASURED, n = 1024, k = 4096, 256 CUs. The knee moves with the shape, so there
// is no single best decode tile -- best B_N by (b, m), among {32, 64, 128, 256}:
//
//     m \ b        2            8           16
//       1      64 (kid4)    64 (kid4)    64 (kid4)
//      16      64 (kid4)    64 (kid4)    64 (kid4)
//     128     128 (kid6)   256 (kid7)   B_M=128 (kid0 wins outright)
//    2048          -       B_M=128 (kid0 wins outright)
//
// The optimum is interior, not just "as wide as the grid allows". At b=2, m=1
// every candidate is grid-starved (32 / 16 / 8 workgroups for B_N 64 / 128 /
// 256) and B_N = 64 still wins, yet widening 32 -> 64 pays there too even though
// it halves an already-starved grid. So the redundant-traffic saving is worth
// roughly one halving of the grid and no more.
//
// Where the grid does not have to be given up, the gain is large: at b = 8,
// m = 128, B_N = 256 keeps a full 256 workgroups and reaches 299 TFLOP/s, which
// is 3.08x kid1, 1.89x kid4, and 1.44x even the prefill tile kid0 -- the first
// point where a decode tile beats kid0 at m = 128.
//
// The m = 128 row is really telling us to grow B_M rather than B_N: at m = 128
// the M axis supplies 8 tiles at B_M = 16, and kid0 (B_M = 128) wins outright at
// b = 16 and at m = 2048. A B_M = 32/64 variant of kid7 is the untested tile
// most likely to win the m >= 128 band, and B_M is the axis this ladder never
// moved.
//
// LDS, all at WG_PER_CU = 1: kid6 = 3 x (16x272 A + 8x4096 B) = 108.75 KB, so it
// still gets the 160 KB tail-pad to keep a second workgroup out. kid7 =
// 3 x (16x272 A + 16x4096 B) = 204.75 KB, which excludes a second workgroup on
// its own and so is passed through unpadded. Both are inside the 320 KB budget
// (kid0 already ships 198 KB), and 1 WG/CU is required regardless while the
// named barriers stay keyed on compile-time ids -- see the kid2/kid3 warning.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n128_w6_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/128, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/256, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1>;

// -- kid8, kid9: the DSV4 scale granularity (GROUP_N = 128) ------------------
// Everything above uses GROUP_N = 1, i.e. a per-column B scale, which is what
// op_tests has always built. That is NOT the shipping DSV4 configuration:
// gfx950's tuned kernels are named ..._1x128x128_..., meaning A is 1x128 and B
// is 128x128. These two tiles are the GROUP_N = 128 versions of the two winners
// of the B_N ladder (kid4 for m <= 16, kid7 for b=8/m=128), changing nothing
// else, so kid4-vs-kid8 and kid7-vs-kid9 price the scale granularity directly.
//
// The first reason for these tiles is FUNCTIONAL: every GROUP_N=1 tile requires
// the caller to materialise a 128x larger scale tensor ([batch, N, K/GROUP_K]
// instead of [batch, N/128, K/GROUP_K]); at n=1024 k=4096 that is 1 MB per batch
// of duplicated bytes instead of 8 KB.
//
// The direct traffic argument for them turned out to be nearly worthless, and it
// is worth recording why the static reasoning oversold it. The per-lane gather
// GROUP_N=128 removes touches 16 cache lines (rows 32 bytes apart) to deliver 16
// useful bytes -- 3.1% line utilisation -- but a whole batch's B scale is only
// ~8 KB, so those lines were L1-resident and the traffic was never the cost. Nor
// is it about the load being scalar: it stays a VECTOR load with a uniform
// address, not an s_load, so it still costs a VMEM slot and full latency; only
// the divergence went away. At kExpN=1 the load COUNT is identical (20
// global_load_u8 either way) and kid8-vs-kid4 measured as noise around parity.
//
// What GROUP_N=128 is actually worth runs through kSfLoadsPerK. Uniformity lets
// one load serve all kExpN in-tile N indices, so at kExpN=4 the B side collapses
// 4 -> 1 and the tile's per-K-tile scale load count goes 5 -> 2. That is what
// drops it below the kSfEarly threshold, which is where the real gain lives:
// kid9 (early) vs kid7 (late) now measures 1.037x .. 1.073x across the shapes
// whose noise is under 2%, where before the early-issue work it was ~1.02x. The
// ISA agrees -- kid9 issues 20 scale loads against kid7's 50 and needs 100 VGPRs
// against 132. So the mechanism is "fewer distinct scale values per K-tile lets
// the whole batch be issued ahead of the ds_reads", not "less scale traffic".
//
// What these tiles actually buy is the ability to consume DSV4's REAL w_scale.
// Every GROUP_N=1 tile requires the caller to materialise a 128x larger scale
// tensor ([batch, N, K/GROUP_K] instead of [batch, N/128, K/GROUP_K]); at
// n=1024 k=4096 that is 1 MB per batch of duplicated bytes instead of 8 KB.
//
// A real speedup would need the load off the VMEM path entirely: the whole
// B-scale working set for the ENTIRE K range is kSfBRowsPerTile x K/GROUP_K =
// 32 bytes (kid8) or 64 bytes (kid9), so it is fully hoistable into prologue
// SGPRs. Given the 2-4% measured here, that upside is bounded and small.
//
// NOTE these tiles require the differently shaped w_scale and the launcher
// enforces it, so they cannot be swapped in for kid4/kid7 against the same
// inputs -- that is the point of the check.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gn128_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/64, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gn128_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/256, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128>;

// -- decode twins with the scale panels (kid11/12 = A, kid15/16 = A and B) ----
// kid8 and kid9 with the panel flags and nothing else changed. Restored after
// the prefill result came out positive (see the kSfALds section): they were
// deleted when the measurement said the panel was worth nothing, which was a
// measurement error, so the deletion was on a false premise.
//
// These are a genuinely different test from kid13/kid14, not a repeat, because
// the decode tiles start from almost nothing left to save. kExpM is 1 against
// the prefill tile's 8, and GROUP_N=128 already collapsed B to ONE uniform load
// per K-tile, so kSfLoadsPerK is 2 rather than 12 -- the panels remove 2 global
// loads per K-tile instead of 12. If the win is "one fewer exposed round trip
// per K-tile" it should still show up; if it needs a BATCH of loads to be worth
// the LDS, it will not. That is the question these four answer.
//
// The B panel is also nearly free here, which is the other reason to separate
// the flags on this side: at GROUP_N=128 it is 2 rows (kid8) or 3 rows (kid9)
// of the padded pitch, 288 or 432 bytes, against 18.6 KB at GROUP_N=1.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_sfa_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/64, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128,
        /*SF_A_LDS*/true>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_sfa_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/256, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128,
        /*SF_A_LDS*/true>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_sfab_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/64, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128,
        /*SF_A_LDS*/true, /*SF_B_LDS*/true>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_sfab_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/256, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128,
        /*SF_A_LDS*/true, /*SF_B_LDS*/true>;

// -- kid10: the one tile where blocked-but-NOT-uniform actually happens -------
// B_N = 192 on 4 consumer waves gives kExpN = 3, so a wave owns 48 columns --
// which does not divide 128. Wave 2 of every workgroup straddles a scale block
// boundary (columns 96..143 span block 0 and block 1), so its B scale genuinely
// differs between lanes even though GROUP_N is 128 and the tensor is blocked.
//
// This tile exists to TEST that: it is the only configuration that separates
// kGroupN (which layout) from kSfBUniformOverN (may the lane term be dropped),
// and it is the case a layout branch keyed on uniformity would silently get
// wrong. It is not a performance winner and is not meant to be: measured, it
// lands between its ladder neighbours everywhere it competes (b=16 m=128: 56.1us
// vs kid6's 68.8 and kid7's 52.9) and never wins a shape.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n192_w6_gn128_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/192, /*B_M*/16, /*B_N*/192, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/128>;

// -- A-scale LDS panel tile (kid13) ----------------------------------------
// kid0 with SF_A_LDS flipped on and NOTHING else changed, so the kid13-vs-kid0
// pair isolates the panel exactly. GROUP_N stays 1 for the same reason.
//
// This is the tile the whole exercise is aimed at: the prefill tile has
// kExpM=8, so it issues 8 A-scale gathers per K-tile against a decode tile's 1,
// and it sits at 542 VGPRs -- 30 above the 512 that would let a third wave onto
// a SIMD. Its panel is kBlockM=128 rows x K/GROUP_K bytes, 4 KB at k=4096,
// which costs no occupancy (kid0's A+B ring is already 198 KB, so 1 WG/CU
// either way).
//
// The decode twins kid11 = kid8 + SF_A_LDS and kid12 = kid9 + SF_A_LDS are
// declared above. They are the clean GROUP_N=128 comparison -- B's scale is
// already one load per K-tile there, so A is the only per-lane scale gather left
// and the pair difference is uncontaminated -- but they are NOT yet reachable
// from the dispatcher, so nothing has been measured on them.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/128, /*B_N*/128, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/1,
        /*SF_A_LDS*/true>;

// -- both scales in LDS (kid14) ---------------------------------------------
// kid13 plus SF_B_LDS, so the kid13-vs-kid14 pair isolates the B side with
// everything else -- tile shape, GROUP_N, panel geometry, fill mechanism --
// held fixed. This is the gfx950 arrangement (its PRELOAD_SF_LDS stages both),
// and it is the only tile whose inner loop issues ZERO global scale loads:
// kid0 issues 12 per K-tile (8 A + 4 B), kid13 4, this 0.
//
// MEASURED 1.020 over kid13 and 1.096 over kid0 at k=4096, so B is worth about
// a third of what A is -- which is the expected ordering, since B was already
// down to kExpN loads against A's kExpM, and it confirms the win is "one fewer
// exposed global round trip per K-tile" rather than anything specific to A.
//
// GROUP_N stays 1, so B's panel is the full kBlockN=128 block rows -- 18.6 KB,
// which fits only because kid0 is already 1 WG/CU (198 KB ring + 36 KB panels =
// 234 KB of the 320 KB). At GROUP_N=128 the same panel would be 2 rows.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfab_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/128, /*B_N*/128, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/1,
        /*SF_A_LDS*/true, /*SF_B_LDS*/true>;

// -- A's panel filled by TDM instead of cooperatively (kid17, kid18) ---------
// The pair that settled whether the cooperative fill was actually the better
// mechanism or just the better-measured one. It was the better mechanism: kid13
// beats kid17 by ~1% at b={8,16} m=2048 and by 12% at b=8 m=512. Numbers and the
// reason are in the kSfATdmKG block. These two stay reachable so the result can
// be re-checked rather than re-derived.
//
// kid17 is the CONTROLLED one and the one to read: width 32 = k/GROUP_K at
// k=4096 and pad 16 = kSfPanelPad, so its pitch is 48, the same 48 kid13 computes
// at runtime. Same panel, same LDS layout, same read instructions -- kid13 vs
// kid17 is the fill and nothing else. It caps K at 32*128 = 4096, which is the
// benchmark point; a different K needs a different alias, because the width is a
// descriptor field and cannot follow K.
//
// kid18 reproduces the ORIGINAL geometry (width 128, pad 4) that first measured
// as a regression. It is here so the width/pad axis can be separated from the
// mechanism axis: kid18 vs kid17 is geometry at fixed mechanism, kid17 vs kid13
// is mechanism at fixed geometry. Its pad of 4 makes the pitch 132, which is not
// 16-aligned -- harmless now, because the TDM engine writes the panel and the
// 16-byte cooperative store is not involved.
template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm32_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/128, /*B_N*/128, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/1,
        /*SF_A_LDS*/true, /*SF_B_LDS*/false,
        /*SF_A_TDM_KG*/32, /*SF_A_TDM_PAD*/16>;

template <typename DataC>
using opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm128_gfx1250 =
    opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250<
        /*BLOCK_SIZE*/128, /*B_M*/128, /*B_N*/128, /*B_K*/256,
        /*LAYOUT*/opus_gfx1250_bmm::kLayoutTileN,
        /*D_A*/opus::fp8_t, /*D_B*/opus::fp8_t, /*D_C*/DataC, /*D_ACC*/float,
        /*GROUP_K*/128, /*NUM_SLOTS*/3, /*WG_PER_CU*/1, /*GROUP_N*/1,
        /*SF_A_LDS*/true, /*SF_B_LDS*/false,
        /*SF_A_TDM_KG*/128, /*SF_A_TDM_PAD*/4>;

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
                    p[(size_t)row * row_stride + k_group + j])) << (8 * j);
        });
        return (int)w;
    }
}

#endif  // __gfx1250__ || !__HIP_DEVICE_COMPILE__

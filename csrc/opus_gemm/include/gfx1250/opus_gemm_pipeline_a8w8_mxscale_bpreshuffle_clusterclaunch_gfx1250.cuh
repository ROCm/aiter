// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 (MI450) a8w8 mxscale BMM with a preshuffled B, TDM producer/consumer.
//
//   Y[M, batch, N] = O[M, batch, K] @ wo_a[batch, N, K]^T
//
// CLUSTER-LAUNCH, FUSED SPLIT-K variant: split-K reduce happens INSIDE this one
// kernel, with no separate reduce launch and no semaphore.
//
// DESIGN
//   * cluster = __cluster_dims__(SplitK, MClusterWg, 1); grid, in workgroups,
//       = ( SplitK, round_up(num_tiles_m, MClusterWg), num_tiles_n * batch ).
//     cluster.x is the k-slice; cluster.y folds MClusterWg ADJACENT M-TILES so
//     those peers, which share one N-tile and one batch, can TDM-MULTICAST the B
//     (weight) block -- one global read fans out into all their LDS. Folding M
//     and multicasting B is the mirror image of the a16w16 splitk_fuse kernel
//     (which folds N and multicasts A) and is the right way round for a BMM:
//     the weight is the larger read at decode-ish M, and it also keeps a whole
//     peer group inside one batch, which the multicast requires.
//   * Cross-WG split-K sync is a CLUSTER BARRIER (-3), not a semaphore: all
//     splits of a tile co-reside, so the barrier alone orders every non-last
//     partial's store before the last WG's read. Deterministic -- no atomics,
//     bit-identical run to run.
//   * The K tiles are split BALANCED across the SplitK WGs (the first k_rem
//     splits take one extra), never ceil-front-loaded, so no split is empty.
//   * Each NON-LAST split casts its fp32 partial to DataWs and stores it with
//     gfx12 CPOL TH_WB|SCOPE_DEV so it stays dirty-resident in GL2 and the
//     reduce read hits L2, not HBM. Workspace layout is tile-major/split-minor:
//       ((batch*num_tiles_m + gm)*num_tiles_n + gn)*(SplitK-1) + s
//     tiles of B_M*B_N DataWs elements. The LAST split keeps its own partial in
//     fp32 registers, folds bias into it, then sums the SplitK-1 published
//     partials and casts to D_OUT once at the C store.
//   * Partials use a PRIVATE lane-contiguous scratch layout, not the C fragment
//     map -- nothing outside this kernel reads them, so the store and reduce need
//     only agree with each other, and being contiguous per lane makes both sides
//     16B dwordx4. This is why an unverified C map (below) does not endanger the
//     split-K plumbing.
//
// MEASURED 2026-09-01 (_bmm_perf/bench_cc.py; control arm 1.000, 11/11 reps
// quiet). Two results, and they point opposite ways:
//
//   * mClusterWg=2 at SplitK=1 -- the B multicast across M-tile peers -- is a
//     real win on prefill: 1.034 / 1.060 / 1.053 at (b=8,m=512), (b=8,m=2048),
//     (b=16,m=2048). Small, consistent, and free of a workspace.
//   * SplitK does not pay anywhere yet. On prefill it costs about 2.5x per
//     doubling (sk2 0.379, sk4 0.209, sk8 0.108 of the plain launch), and the
//     reason is structural rather than a tuning miss: at b=8 m=512 the SplitK=1
//     grid is already 4*8*8 = 256 workgroups against 256 CUs, so a split buys
//     no parallelism, while kid0's partial tile is a full B_M x B_N fp32 --
//     (SplitK-1) x 117 MB written and read again at sk=8, against ~50 MB of A
//     and B for the entire GEMM. On decode the partials are small and SplitK is
//     free (b=2: sk1..sk8 all read 0.82), but entering the cluster path costs a
//     flat ~21% at SplitK=1 there, which nothing then wins back.
//
// So the multicast is the part of this kernel that pays today. A split-K that
// pays needs either a tile whose partial is much smaller than its output, or a
// shape whose SplitK=1 grid does not already fill the machine.
//
// STILL A SCAFFOLD IN ONE RESPECT. The data movement, the scale path and the
// whole split-K epilogue are complete, but two fragment maps are still inferred
// rather than measured, and they are marked `TODO(kernel)`:
//
//   1. frag_a / frag_b   -- the LDS -> VGPR fragment maps. The A one is a guess at
//                           the wave32 WMMA operand layout; the B one additionally
//                           has to walk the shuffle_weight(16,16) interior. Both
//                           are marked UNVERIFIED and both need the probe below.
//   2. the scale fetch   -- DONE. SF_A_LDS / SF_B_LDS stage A's and B's scales
//                           in LDS for a whole K, filled once in the prologue.
//                           Measured 1.068 / 1.096 (A / both) against the global
//                           path on the prefill tile, growing to 1.18 at
//                           k=16384. kid13 and kid14 carry the flags; the
//                           default is still global because the ladder's decode
//                           tiles have not been re-measured. See the traits'
//                           kSfALds section, INCLUDING its note on why an
//                           earlier round of measurement got the sign wrong.
//                           The older per-slot T::kSfPanel scaffold is unused
//                           and hardwired false.
//                           NOTE for split-K: both panels cover the tile's WHOLE
//                           K range on EVERY split, so each split redundantly
//                           fills bytes it will not read. That is deliberate --
//                           it keeps the panel index absolute, which is why
//                           consume_slot below takes an ABSOLUTE k_step.
//   3. store_c           -- the C fragment map, same UNVERIFIED caveat as (1).
//
// HOW TO VERIFY (1) AND (3) BEFORE TRUSTING ANY NUMBER. These maps are the only
// places this kernel can be wrong while still producing plausible output, so
// do not infer them -- measure them. Launch one workgroup on M=N=16, K=128 with
// A[m][k] = m*128+k, B = identity in the shuffled layout, all scales 0x7F, and
// print (lane, i) -> value. Anything that disagrees with the constants below is
// the hardware telling you the layout, and the layout wins. Run the probe at
// SplitK=1: it isolates the maps from the reduce.
//
// The wave split is w0 = A producer, w1 = B producer, and every wave from
// kNumProducerWaves up is a WMMA consumer. NO WAVE RETURNS EARLY -- unlike the
// non-cluster sibling, the producers fall through to the cluster barrier, and
// producer w0 stages the reduce after it. An early return anywhere takes its
// cluster's barrier quorum with it and hangs the launch.
//
// The wave count follows BLOCK_SIZE and is not assumed to be 4 anywhere below:
// the barrier member counts are expressed in kNumWaves / kNumConsumerWaves and
// the consumer's wave_m/wave_n come from kTileM/kTileN, which the traits derive
// from kNumConsumerWaves. See the kNumWaves block in the traits header for why
// going past 4 waves matters. Every compile-time value comes from the traits
// header; this file declares no geometry of its own.
#pragma once

#include "opus_bmm_traits_a8w8_mxscale_bpreshuffle_gfx1250.cuh"

#ifdef __HIP_DEVICE_COMPILE__
using namespace opus;
using opus::operator""_I;
#endif

// Shared scalar helpers. GUARDED because a single TU can now include BOTH this
// header and its cluster-launch sibling (the instantiation TU does exactly
// that), and `constexpr inline` permits one definition per TU, not two.
#ifndef OPUS_BMM_MX_SCALAR_HELPERS_DEFINED
#define OPUS_BMM_MX_SCALAR_HELPERS_DEFINED
__host__ __device__ constexpr inline int opus_bmm_mx_ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}
__host__ __device__ constexpr inline int opus_bmm_mx_min_i(int a, int b) {
    return a < b ? a : b;
}
#endif  // OPUS_BMM_MX_SCALAR_HELPERS_DEFINED

// Depth of the last-split reduce LDS ring: how many published partial tiles are
// staged at once when they do not all fit. Deliberately NOT named
// kFuseReduceRing -- the a16w16 splitk_fuse header defines a namespace-scope
// `static constexpr int kFuseReduceRing` with internal linkage, so a TU that
// included both would hit a redefinition, not a merge.
static constexpr int kBmmFuseReduceRing = 3;

// gfx12 CPOL for the partial store: TH_WB(3) | SCOPE_DEV(2 << 3) = 19. Keeps the
// partial dirty-resident in GL2 instead of write-rinsing it to HBM, published at
// device scope. Same value and same reasoning as the a16w16 sibling's
// OPUS_SKFUSE_WS_CPOL; separately named so the two can be overridden apart.
#ifndef OPUS_BMM_MXSK_WS_CPOL
#define OPUS_BMM_MXSK_WS_CPOL (/*TH_WB*/ 3 | (/*SCOPE_DEV*/ 2 << 3))
#endif

// launch_bounds comes from the traits, not a literal: the host launcher already
// sizes the block as dim3(T::BLOCK_SIZE), so a hardcoded bound here is a silent
// mismatch the moment a tile picks a wave count other than 4 (the register
// allocator would be budgeting for 128 threads while 192 launch).
template <typename UserTraits, int SplitK, typename DataWs, int MClusterWg, typename D_OUT>
__global__ __launch_bounds__(opus::remove_cvref_t<UserTraits>::BLOCK_SIZE, 1)
#if defined(__gfx1250__) || !defined(__HIP_DEVICE_COMPILE__)
    __cluster_dims__(SplitK, MClusterWg, 1)
#endif
    void gemm_a8w8_mxscale_bpreshuffle_clusterclaunch_kernel_gfx1250(
        opus_gemm_cluster_claunch_kargs_gfx1250 kargs)
{
    // TWO independent limits, and the second is not implied by the first:
    //   * the cluster holds at most 16 workgroups  -> the PRODUCT is <= 16;
    //   * each __cluster_dims__ field is 4 bits    -> each DIMENSION is <= 15.
    // A 16x1 cluster satisfies the product budget and still fails to encode
    // ("integer constant expression evaluates to value 16 that cannot be
    // represented in a 4-bit unsigned integer type"), so spell the per-dimension
    // bound out here rather than leaving that error to explain itself. Use 8x2
    // or 4x4 when you want all 16 workgroups.
    static_assert(SplitK >= 1 && MClusterWg >= 1, "cluster dimensions must be >= 1");
    static_assert(SplitK <= 15 && MClusterWg <= 15,
                  "each __cluster_dims__ dimension is a 4-bit field, so SplitK and "
                  "MClusterWg must each be <= 15 (a 16-WG cluster must be 8x2 or 4x4)");
    static_assert(SplitK * MClusterWg <= 16, "cluster WG count (SplitK*MClusterWg) must be <= 16");
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx1250__)
    // remove_cvref_t<UserTraits>, NOT ::T -- UserTraits IS the traits struct
    // (the launcher instantiates opus_bmm_a8w8_mxscale_bpreshuffle_traits_gfx1250
    // directly), and there is no wrapper with a nested ::T to unwrap.
    using T       = remove_cvref_t<UserTraits>;
    using DataA   = typename T::DataA;
    using DataB   = typename T::DataB;
    using DataC   = typename T::DataC;    // traits' declared C dtype
    using DataAcc = typename T::DataAcc;  // fp32: WMMA acc AND the reduce acc
    using DataSf  = typename T::DataSf;
    // DataWs and D_OUT are template parameters and are already in scope; a
    // `using DataWs = DataWs;` would be a self-referential (ill-formed) typedef,
    // not a re-export.
    // D_OUT, not DataC, is what the C store writes: the traits' D_C fixes a tile
    // alias's nominal output dtype, but the fused epilogue keeps its accumulator
    // in fp32 all the way to the store and casts ONCE, so the caller is free to
    // ask for fp32 C from a bf16-C tile alias. DataC survives only as the
    // documented default the launcher hands to D_OUT.
    DECLARE_NAMED_BARRIERS();   // __nbar_1..__nbar_15; we use 1..3*kNumSlots <= 9

    // -- named-barrier helpers (compile-time ids) ---------------------------
    // Barrier layout, P = kNumSlots:
    //   DATA[s]   = 1        + s   memcnt = kNumWaves            (both producers + all consumers)
    //   FREE_A[s] = 1 +   P  + s   memcnt = 1 + kNumConsumerWaves (prodA + consumers)
    //   FREE_B[s] = 1 + 2*P  + s   memcnt = 1 + kNumConsumerWaves (prodB + consumers)
    // DATA's count is kNumWaves because both producers signal it once and each
    // consumer joins it once: 2 + kNumConsumerWaves == kNumWaves. That identity
    // holds only while every producer wave actually issues a DMA -- if you ever
    // add a producer that does not, this count must become 2 + kNumConsumerWaves
    // explicitly or the barrier will never fill.
    // FREE is PER-PRODUCER on purpose: with one shared FREE the consumer's extra
    // free can substitute for a producer's signal, releasing whichever producer
    // happened to be joined and hanging the other. Do not merge them.
    auto binit = [&](auto IdN, u32_t mc) __attribute__((always_inline)) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) s_barrier_init_ptr(&__nbar_1, mc);
        else if constexpr (id == 2) s_barrier_init_ptr(&__nbar_2, mc);
        else if constexpr (id == 3) s_barrier_init_ptr(&__nbar_3, mc);
        else if constexpr (id == 4) s_barrier_init_ptr(&__nbar_4, mc);
        else if constexpr (id == 5) s_barrier_init_ptr(&__nbar_5, mc);
        else if constexpr (id == 6) s_barrier_init_ptr(&__nbar_6, mc);
        else if constexpr (id == 7) s_barrier_init_ptr(&__nbar_7, mc);
        else if constexpr (id == 8) s_barrier_init_ptr(&__nbar_8, mc);
        else                        s_barrier_init_ptr(&__nbar_9, mc);
    };
    auto bjs = [&](auto IdN) __attribute__((always_inline)) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) { __builtin_amdgcn_s_barrier_signal(1); }
        else if constexpr (id == 2) { __builtin_amdgcn_s_barrier_signal(2); }
        else if constexpr (id == 3) { __builtin_amdgcn_s_barrier_signal(3); }
        else if constexpr (id == 4) { __builtin_amdgcn_s_barrier_signal(4); }
        else if constexpr (id == 5) { __builtin_amdgcn_s_barrier_signal(5); }
        else if constexpr (id == 6) { __builtin_amdgcn_s_barrier_signal(6); }
        else if constexpr (id == 7) { __builtin_amdgcn_s_barrier_signal(7); }
        else if constexpr (id == 8) { __builtin_amdgcn_s_barrier_signal(8); }
        else                        { __builtin_amdgcn_s_barrier_signal(9); }
    };
    auto bjsw = [&](auto IdN) __attribute__((always_inline)) {
        constexpr int id = IdN.value;
        if      constexpr (id == 1) { s_barrier_join_ptr(&__nbar_1); __builtin_amdgcn_s_barrier_signal(1); __builtin_amdgcn_s_barrier_wait(1); }
        else if constexpr (id == 2) { s_barrier_join_ptr(&__nbar_2); __builtin_amdgcn_s_barrier_signal(2); __builtin_amdgcn_s_barrier_wait(2); }
        else if constexpr (id == 3) { s_barrier_join_ptr(&__nbar_3); __builtin_amdgcn_s_barrier_signal(3); __builtin_amdgcn_s_barrier_wait(3); }
        else if constexpr (id == 4) { s_barrier_join_ptr(&__nbar_4); __builtin_amdgcn_s_barrier_signal(4); __builtin_amdgcn_s_barrier_wait(4); }
        else if constexpr (id == 5) { s_barrier_join_ptr(&__nbar_5); __builtin_amdgcn_s_barrier_signal(5); __builtin_amdgcn_s_barrier_wait(5); }
        else if constexpr (id == 6) { s_barrier_join_ptr(&__nbar_6); __builtin_amdgcn_s_barrier_signal(6); __builtin_amdgcn_s_barrier_wait(6); }
        else if constexpr (id == 7) { s_barrier_join_ptr(&__nbar_7); __builtin_amdgcn_s_barrier_signal(7); __builtin_amdgcn_s_barrier_wait(7); }
        else if constexpr (id == 8) { s_barrier_join_ptr(&__nbar_8); __builtin_amdgcn_s_barrier_signal(8); __builtin_amdgcn_s_barrier_wait(8); }
        else                        { s_barrier_join_ptr(&__nbar_9); __builtin_amdgcn_s_barrier_signal(9); __builtin_amdgcn_s_barrier_wait(9); }
    };

    const int wave_id     = __builtin_amdgcn_readfirstlane((int)opus::waveid_in_workgroup());
    const int lane_id     = (int)opus::lane_id();
    const bool is_producer = wave_id < T::kNumProducerWaves;

    // -- cluster / tile / batch coordinates ---------------------------------
    // CLUSTER = __cluster_dims__(SplitK, MClusterWg, 1), so the launcher's grid,
    // counted in WORKGROUPS, is
    //     grid = ( SplitK,
    //              round_up(num_tiles_m, MClusterWg),
    //              num_tiles_n * batch )
    // and the cluster-local ids decompose as
    //     split_idx = cluster_workgroup_id_x()   0 .. SplitK-1
    //     local_m   = cluster_workgroup_id_y()   0 .. MClusterWg-1
    //     gm        = cluster_id_y()*MClusterWg + local_m     (M tile)
    //     gn, batch = cluster_id_z() split by num_tiles_n     (N tile, batch)
    //
    // M-DIRECTION folding, not N. cluster.y groups MClusterWg ADJACENT M-TILES
    // that share one N-tile and one batch, so the operand their peers hold in
    // common is B -- the preshuffled weight block -- and B is the one that gets
    // the TDM multicast below. That is the opposite assignment from the a16w16
    // splitk_fuse kernel (which folds N and multicasts A), and it is the right
    // way round HERE because this is a BMM whose A is the activation O[M,batch,K]
    // and whose B is the weight wo_a[batch,N,K]: at decode-ish M the weight is by
    // far the larger read, and folding M is also what lets a whole cluster stay
    // inside one batch (a peer group must share ptr_b's batch slice or the
    // multicast would fan one batch's weights into another's LDS).
    //
    // batch rides in cluster.z BELOW the N tile (gn fastest) so that the peers of
    // a cluster -- which differ only in cluster.y -- always agree on both gn and
    // batch. Putting batch in the fast position would still be correct but would
    // scatter consecutive N tiles of one batch across distant clusters.
    const int split_idx = (int)__builtin_amdgcn_cluster_workgroup_id_x();  // 0..SplitK-1
    const int local_m   = (int)__builtin_amdgcn_cluster_workgroup_id_y();  // 0..MClusterWg-1
    const int gm        = (int)__builtin_amdgcn_cluster_id_y() * MClusterWg + local_m;
    const int gzn       = (int)__builtin_amdgcn_cluster_id_z();
    const int gn        = gzn % kargs.num_tiles_n;
    const int batch_id  = gzn / kargs.num_tiles_n;
    const int tile_row  = gm * T::kBlockM;
    const int tile_col  = gn * T::kBlockN;
    const bool is_last  = (split_idx == SplitK - 1);

    // B multicast mask: the MClusterWg M-peers at the SAME split (same k-slice)
    // and the same (gn, batch) all read the identical B block, so one global
    // fetch fans out into all their LDS. The flat cluster WG id is x-fastest
    // (local_m*SplitK + split_idx), which is exactly the set peers_along_y names
    // over cluster dims (SplitK, MClusterWg). MClusterWg==1 folds to mask 0
    // (multicast off) inside the helper -- issuing a cluster-load from a
    // one-WG peer group is what the fold exists to prevent.
    const auto mask_b = opus::tdm_traits::peers_along_y<SplitK, MClusterWg>();

    // A cluster is rounded UP to MClusterWg in M, so the last cluster can carry
    // peers whose M tile does not exist. They must NOT return: every WG of the
    // cluster has to reach the -3 barrier below or the ones that do hang. They
    // run the whole pipeline against a fully out-of-range window (a zero-extent
    // DMA that touches no memory), and their C store is dropped by the row guard.
    const bool m_oob = (gm >= kargs.num_tiles_m);

    // -- BALANCED split of the K tiles across the SplitK WGs ----------------
    // Not ceil front-loading: that starves the trailing splits and can leave them
    // with ZERO tiles, and a WG with no tiles still has to publish a (zero)
    // partial and still has to reach the barrier. Here the first k_rem splits
    // take one extra B_K tile, so every split gets >= 1 tile whenever
    // SplitK <= k_steps_tot -- which the launcher enforces. The single partial
    // B_K tile at the global K tail (K % B_K != 0) belongs to whichever split
    // contains it and is clamped by the producer window's extent, so the tail is
    // handled by TDM rather than by emptying a WG.
    const int k_steps_tot = opus_bmm_mx_ceil_div_i(kargs.k, T::kBlockK);
    const int k_base      = k_steps_tot / SplitK;
    const int k_rem       = k_steps_tot - k_base * SplitK;   // k_steps_tot % SplitK
    const int k_step_beg  = split_idx * k_base + opus_bmm_mx_min_i(split_idx, k_rem);
    const int k_steps     = k_base + (split_idx < k_rem ? 1 : 0);

    // The ONLY early return allowed before the cluster barrier is one every WG in
    // the cluster takes together. k_steps_tot depends on nothing WG-local, so an
    // empty K exits the whole cluster uniformly; a per-split `k_steps <= 0` test
    // here would let some peers leave and hang the rest, which is why the
    // balanced split above (not the guard) is what keeps splits non-empty.
    if (k_steps_tot <= 0) return;

    // Per-batch bases. Strides are in elements of each tensor's own dtype.
    const DataA* ptr_a  = reinterpret_cast<const DataA*>(kargs.ptr_a)  + (size_t)batch_id * kargs.stride_a_batch;
    const DataB* ptr_b  = reinterpret_cast<const DataB*>(kargs.ptr_b)  + (size_t)batch_id * kargs.stride_b_batch;
    D_OUT*       ptr_c  = reinterpret_cast<D_OUT*>(kargs.ptr_c)        + (size_t)batch_id * kargs.stride_c_batch;
    const DataSf* ptr_sfa = reinterpret_cast<const DataSf*>(kargs.ptr_sfa) + (size_t)batch_id * kargs.stride_sfa_batch;
    const DataSf* ptr_sfb = reinterpret_cast<const DataSf*>(kargs.ptr_sfb) + (size_t)batch_id * kargs.stride_sfb_batch;

    __shared__ char lds_buf[T::kLdsTotalBytes];
    DataA* smem_a = reinterpret_cast<DataA*>(lds_buf);
    DataB* smem_b = reinterpret_cast<DataB*>(lds_buf + T::kSegBytesA);
    // Whole-K scale panels: A is [kBlockM rows x sf_pitch], B is
    // [kSfBPanelRows x sf_pitch]. Both pitches track the RUNTIME K-group count
    // (plus a bank pad) rather than a compile-time width, so a small K wastes
    // only LDS. They sit after both ring segments, A then B, and are zero-sized
    // when the tile does not ask for them.
    DataSf* smem_sfa =
        reinterpret_cast<DataSf*>(lds_buf + T::kSegBytesA + T::kSegBytesB);
    DataSf* smem_sfb = reinterpret_cast<DataSf*>(
        lds_buf + T::kSegBytesA + T::kSegBytesB + T::kSegBytesSfALds);
    // Row length in K-groups, shared by both panels because both are indexed by
    // the same kg. ceil, not floor: a K that does not fill its last group still
    // needs that group's byte, and the read side's kg reaches it.
    const int sf_kg = (T::kSfALds || T::kSfBLds)
                          ? opus_bmm_mx_ceil_div_i(kargs.k, T::kGroupK)
                          : 0;
    // LDS row pitch: round up to the pad, then add it. See kSfPanelPad -- the
    // pad is what keeps 16 lanes reading 16 consecutive rows off one bank, and
    // rounding to 16 is what lets the fill keep its 16-byte store.
    //
    // Under kSfATdm the pitch is the D#'s own padding, so it is compile-time and
    // the runtime expression would disagree with the layout the engine wrote.
    // Only A can be on TDM (asserted in the traits), so there is no ambiguity
    // about which panel this pitch belongs to.
    const int sf_pitch =
        T::kSfATdm
            ? T::kSfAPitchFixed
            : ((T::kSfALds || T::kSfBLds)
                   ? (((sf_kg + T::kSfPanelPad - 1) & ~(T::kSfPanelPad - 1))
                      + T::kSfPanelPad)
                   : 0);
    // First kGroupN block this tile's N range touches; the B panel is indexed
    // relative to it. Blocks, not columns: tile_col need not be a multiple of
    // kGroupN, which is what kSfBPanelRows' +1 covers.
    const int sfb_nb_base = T::kSfBLds ? (tile_col / T::kGroupN) : 0;
    constexpr int slot_a = T::kSlotElemsA;
    constexpr int slot_b = T::kSlotElemsB;

    using WindowA = typename T::WindowA;
    using WindowB = typename T::WindowB;

    // One wave inits the per-slot barriers; the workgroup barrier publishes them.
    if (wave_id == T::kNumProducerWaves) {
        constexpr u32_t kFreeMemCnt = 1 + T::kNumConsumerWaves;
        opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
            constexpr int s = decltype(sN)::value;
            binit(opus::number<1 + s>{}, T::kNumWaves);
            binit(opus::number<1 + T::kNumSlots + s>{}, kFreeMemCnt);
            binit(opus::number<1 + 2 * T::kNumSlots + s>{}, kFreeMemCnt);
        });
    }

    // Scale panels: one cooperative fill by ALL threads, before the
    // producer/consumer split, published by the barrier below -- the same
    // barrier that publishes the binits above, so this costs no new sync.
    // Whole-K, so it happens ONCE per workgroup rather than per K-tile, which is
    // what makes it cheap enough to be free at the ring's expense in LDS only.
    //
    // Grid-stride over each panel's flat byte count with the widest vector the
    // geometry allows; see fill_panel for the width rule. At k=4096 with g=8
    // that is 32 | 256, so 16 bytes, and the prefill tile's 4 KB A panel fills
    // in two iterations per thread.
    //
    // OOB is handled by the buffer bound, not by clamping the row: a buffer load
    // past num_records returns zero rather than faulting, and the rows it
    // affects are the partial-M / partial-N tail, whose C store is dropped
    // anyway. Nothing reduces across rows or columns, so a zero exponent in a
    // dropped row cannot reach a live one.
    // A by DMA instead: one wave programs a 2D window over [K-groups, M] and the
    // engine writes the padded panel itself. Issued from the first CONSUMER
    // wave, because the producers' steady loop is descriptor-bound and the
    // historical TDM fill only reached its best number once the issue moved off
    // them. tensorcnt is per-wave, so this wave drains its own transfer and the
    // s_barrier below is what publishes the panel to everyone else -- the same
    // barrier that publishes the binits, so again no new sync.
    //
    // Extents are the tensor's, not the tile's: shape1 = m with origin_1 =
    // tile_row gives the partial-M tail a zero-extent DMA instead of a fault,
    // exactly as the A-tile window gets it.
    if constexpr (T::kSfATdm) {
        if (wave_id == T::kNumProducerWaves) {
            auto w = opus::make_tdm<typename T::WindowSfA>(
                (u32_t)reinterpret_cast<u64_t>(smem_sfa), ptr_sfa,
                (u32_t)sf_kg, (u32_t)kargs.m, (u64_t)kargs.stride_sfa,
                (u32_t)0, (u32_t)tile_row);
            w.async_load((u32_t)0);
            opus::s_wait_tensorcnt<0>();
        }
    }
    if constexpr (T::kSfACoop || T::kSfBLds) {
        const int tid = (int)opus::thread_id_x();
        // Flat index over the SOURCE row length; the destination re-applies
        // sf_pitch. VEC divides sf_kg on the wide rung, so kt is a multiple of
        // VEC, and sf_pitch is a multiple of kSfPanelPad (16) -- so the store
        // keeps the alignment the load has, which is the whole reason the pad is
        // a power of two rather than the 4 that would minimise bank conflicts.
        auto fill_panel = [&](const DataSf* src, int src_pitch, DataSf* dst,
                              int rows, unsigned bound)
            __attribute__((always_inline)) {
            auto g = opus::make_gmem(src, bound);
            auto s = opus::make_smem(dst);
            const int total = rows * sf_kg;
            auto run = [&](auto VecN) __attribute__((always_inline)) {
                constexpr int VEC = decltype(VecN)::value;
                for (int idx = tid * VEC; idx < total; idx += T::BLOCK_SIZE * VEC) {
                    const int r  = idx / sf_kg;
                    const int kt = idx - r * sf_kg;
                    opus::store<VEC>(s, opus::load<VEC>(g, r * src_pitch + kt),
                                     r * sf_pitch + kt);
                }
            };
            // A chunk must not span two source rows nor land unaligned, so the
            // width has to divide both the row length and the source row stride.
            // Runtime, hence a ladder rather than a compile-time VEC.
            const int widths = sf_kg | src_pitch;
            if ((widths & (T::kSfFillVecMax - 1)) == 0)
                run(opus::number<T::kSfFillVecMax>{});
            else
                run(opus::number<1>{});
        };
        // A: rows are tile_row .. tile_row + kBlockM - 1, and the tail past m is
        // covered by the buffer bound rather than a clamp.
        if constexpr (T::kSfACoop) {
            const int rows_avail = kargs.m - tile_row;      // > 0: tile_row < m
            fill_panel(ptr_sfa + (size_t)tile_row * kargs.stride_sfa,
                       kargs.stride_sfa, smem_sfa, T::kBlockM,
                       (unsigned)((rows_avail - 1) * kargs.stride_sfa + sf_kg));
        }
        // B: rows are kGroupN BLOCKS, sfb_nb_base .. + kSfBPanelRows - 1. The
        // last row of the panel is past the tensor whenever the tile does not
        // straddle, so the bound is doing real work here on every launch, not
        // just on a tail.
        if constexpr (T::kSfBLds) {
            const int nb_max = (kargs.n + T::kGroupN - 1) / T::kGroupN - 1;
            fill_panel(ptr_sfb + (size_t)sfb_nb_base * kargs.stride_sfb,
                       kargs.stride_sfb, smem_sfb, T::kSfBPanelRows,
                       (unsigned)((nb_max - sfb_nb_base) * kargs.stride_sfb + sf_kg));
        }
        // s_barrier retires neither counter on its own: loadcnt for the global
        // reads feeding the panels, dscnt for the ds_writes that publish them.
        opus::s_wait_loadcnt<0>();
        opus::s_wait_dscnt<0>();
    }
    __builtin_amdgcn_s_barrier();

    // ---------------------------------------------------------------------
    // Shared consumer geometry and the fp32 accumulator, HOISTED above the
    // producer/consumer split because the fused epilogue below runs after the
    // branches rejoin and both halves index into it: consumers own `acc`, and
    // producer wave 0 owns the reduce staging that feeds it.
    // ---------------------------------------------------------------------
    // TileN: consumers split N (wave_n = wave_split, wave_m = 0).
    // TileM: consumers split M (wave_m = wave_split, wave_n = 0).
    // A producer has no C fragment; it is pinned to 0 rather than left negative
    // so that the workspace offsets below stay in range for every wave, even
    // where they are computed and then not used.
    const int wave_split = is_producer ? 0 : (wave_id - T::kNumProducerWaves);
    const int wave_m = (T::LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? wave_split : 0;
    const int wave_n = (T::LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? 0 : wave_split;
    // First C column this lane owns. Hoisted with wave_m/wave_n because BOTH
    // post-rejoin users need it -- the bias fold and the C store -- and they sit
    // on opposite sides of the consumer branch's closing brace.
    const int col_base = tile_col + wave_n * (T::kExpN * T::kWmmaN) + (lane_id % T::kWmmaN);

    using Mma = opus::wmma<DataA, DataB, DataAcc, T::kWmmaM, T::kWmmaN, T::kWmmaK>;
    using FragA = opus::vector_t<opus::i32_t, 16>;   // 64 fp8 per lane
    using FragB = opus::vector_t<opus::i32_t, 16>;
    using FragC = typename Mma::vtype_c;             // 8 fp32 per lane
    constexpr int kFragC = (int)opus::size<FragC>(); // 8

    // fp32 all the way to the store: the last split keeps its OWN partial in
    // fp32 (no DataWs round-trip) and sums the published ones into it, so the
    // only rounding is the single cast at the C store. A DataWs accumulator
    // would round once per partial.
    FragC acc[T::kExpM][T::kExpN];
    opus::static_for<T::kExpM>([&](auto im) __attribute__((always_inline)) {
        opus::static_for<T::kExpN>([&](auto in) __attribute__((always_inline)) {
            clear(acc[decltype(im)::value][decltype(in)::value]);
        });
    });

    // -- workspace addressing ----------------------------------------------
    // LANE-CONTIGUOUS scratch layout, NOT the (M,N) map the C store uses. The
    // workspace is private to this kernel: nothing outside reads it, so the only
    // requirement is that the store side and the reduce side agree. Giving each
    // (wave_split, lane) a contiguous run makes both sides fully coalesced 16B
    // dwordx4 traffic, which the real C fragment map (one element per lane along
    // N, see store_c) could never be.
    //
    //   lane run   = kExpM * kExpN * kFragC elements at
    //                (wave_split*kWarp + lane_id) * kPerLane
    //   frag (im,in) occupies the kFragC elements at (im*kExpN + in)*kFragC
    //
    // The runs tile the B_M x B_N partial exactly:
    //   kNumConsumerWaves * kWarp * kExpM*kExpN*kFragC
    //     == (kTileM*kExpM*kWmmaM) * (kTileN*kExpN*kWmmaN) == B_M * B_N.
    constexpr int kPerLane = T::kExpM * T::kExpN * kFragC;
    constexpr int kWsChunk = 16 / (int)sizeof(DataWs);   // dwordx4: 8 bf16 / 4 fp32
    constexpr size_t kWsTileElems = (size_t)T::kBlockM * (size_t)T::kBlockN;
    static_assert((size_t)T::kNumConsumerWaves * T::kWarp * kPerLane == kWsTileElems,
                  "workspace lane runs must tile B_M x B_N exactly");
    static_assert(kFragC % kWsChunk == 0 || kWsChunk % kFragC == 0,
                  "dwordx4 workspace traffic needs kFragC and kWsChunk commensurate");
    static_assert(kFragC % kWsChunk == 0,
                  "a C fragment must be a whole number of dwordx4 chunks");
    const int ws_lane_base = (wave_split * (int)T::kWarp + lane_id) * kPerLane;
    // Tile-major, split-minor. Only SplitK-1 partials are ever stored: the last
    // split keeps its own in registers, which is what makes the workspace
    // (SplitK-1) tiles rather than SplitK.
    const size_t ws_tile_idx =
        ((size_t)batch_id * (size_t)kargs.num_tiles_m + (size_t)gm) * (size_t)kargs.num_tiles_n
        + (size_t)gn;

    // ---------------------------------------------------------------------
    // Producers. w0 streams A tiles, w1 streams B tiles, into a kNumSlots ring.
    // Unchanged from the a16w16 pipeline except for the window construction, and
    // that difference is the whole point of the preshuffled B -- see below.
    // ---------------------------------------------------------------------
    if (is_producer) {
        constexpr auto KStepA = opus::number<T::kBlockK>{};
        // B's fast dim is kBShufBlockElems (= B_K*16), so ITS K step is that many
        // elements, not B_K. Getting this wrong reads the right number of bytes
        // from the wrong place and produces plausible garbage.
        constexpr auto KStepB = opus::number<T::kBShufBlockElems>{};

        auto produce = [&](auto& w, int slot_elems, auto KStepN, auto FreeBaseN) __attribute__((always_inline)) {
            constexpr int kFreeBase = FreeBaseN.value;
            auto load_slot = [&](auto SlotN, auto AdvanceN) __attribute__((always_inline)) {
                if constexpr (AdvanceN.value) w.move(KStepN);
                w.async_load((u32_t)(decltype(SlotN)::value * slot_elems));
            };
            // Steady step: wait FREE[s], issue slot s, keep 2 TDMs in flight, then
            // signal the LAGGED DATA[s-2] (that one has provably landed).
            auto step_slot = [&](auto sN) __attribute__((always_inline)) {
                constexpr int s     = decltype(sN)::value;
                constexpr int prev2 = (s - 2 + T::kNumSlots) % T::kNumSlots;
                bjsw(opus::number<kFreeBase + s>{});
                load_slot(sN, opus::number<1>{});
                opus::s_wait_tensorcnt<2>();
                bjs(opus::number<1 + prev2>{});
            };
            if (k_steps >= T::kNumSlots) {
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    load_slot(sN, opus::number<(decltype(sN)::value > 0) ? 1 : 0>{});
                });
                opus::static_for<T::kNumSlots - 2>([&](auto jN) __attribute__((always_inline)) {
                    constexpr int j = decltype(jN)::value;
                    opus::s_wait_tensorcnt<T::kNumSlots - 1 - j>();
                    bjs(opus::number<1 + j>{});
                });
                // Steady state: full-group main loop + once-run tail. step_slot signals
                // the LAGGED DATA (slot s-2), keeping 2 TDMs in flight per step.
                int k = T::kNumSlots;
                for (; k + T::kNumSlots <= k_steps; k += T::kNumSlots)
                    opus::static_for<T::kNumSlots>(step_slot);
                const int rem = k_steps - k;
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value < rem) step_slot(sN);
                });
                // Drain the last two in-flight loads and signal their pending DATA.
                opus::s_wait_tensorcnt<0>();
                const int last2_slot = (k_steps - 2) % T::kNumSlots;
                const int last_slot  = (k_steps - 1) % T::kNumSlots;
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value == last2_slot) bjs(opus::number<1 + decltype(sN)::value>{});
                });
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value == last_slot) bjs(opus::number<1 + decltype(sN)::value>{});
                });
            } else {
                const int nload = k_steps;
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value < nload)
                        load_slot(sN, opus::number<(decltype(sN)::value > 0) ? 1 : 0>{});
                });
                opus::s_wait_tensorcnt<0>();
                opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                    if ((int)decltype(sN)::value < nload) bjs(opus::number<1 + decltype(sN)::value>{});
                });
            }
        };

        // This split's K origin, in each operand's own fast-axis units. A counts
        // in elements of K; B counts in whole 16-row shuffled BLOCKS of K (see
        // KStepB), so the same tile index scales by a different stride. Feeding
        // A's origin to B is the classic version of this bug: it reads the right
        // number of bytes from the wrong place and produces plausible garbage.
        const u32_t gk0_a = (u32_t)((size_t)k_step_beg * T::kBlockK);
        const u32_t gk0_b = (u32_t)((size_t)k_step_beg * T::kBShufBlockElems);

        if (wave_id == 0) {
            // A is a plain [M, K] tensor: extents (K, M), row stride = stride_a.
            // Out-of-range M rows and the K tail clamp to a zero-extent DMA, which
            // is where the free OOB handling comes from -- do not add an M guard.
            // The extents stay the WHOLE tensor's and only the origin moves, so a
            // split whose range runs past K clamps instead of faulting.
            auto w = opus::make_tdm<WindowA>((u32_t)reinterpret_cast<u64_t>(smem_a), ptr_a,
                                             (u32_t)kargs.k, (u32_t)kargs.m, (u64_t)kargs.stride_a,
                                             gk0_a, (u32_t)tile_row);
            produce(w, T::kSlotElemsA, KStepA, opus::number<1 + T::kNumSlots>{});
        } else {
            // B is the shuffle_weight(16,16) buffer, read as a tensor of 16-row
            // BLOCKS: fast extent = K*16 elements (the whole K range of one block,
            // contiguous), slow extent = N/16 blocks, slow stride = K*16.
            const u32_t b_fast_extent = (u32_t)((size_t)kargs.k * T::kBShufBlockN);
            const u32_t b_rows        = (u32_t)(kargs.n / T::kBShufBlockN);
            const u64_t b_row_stride  = (u64_t)((size_t)kargs.k * T::kBShufBlockN);
            auto w = opus::make_tdm<WindowB>((u32_t)reinterpret_cast<u64_t>(smem_b), ptr_b,
                                             b_fast_extent, b_rows, b_row_stride,
                                             gk0_b, (u32_t)(tile_col / T::kBShufBlockN));
            // CLUSTER_LOAD_ASYNC multicast of B across the MClusterWg M-peers.
            // They share (gn, batch, split), so this exact block is what each of
            // them would otherwise fetch on its own.
            w.set_workgroup_mask(mask_b);
            produce(w, T::kSlotElemsB, KStepB, opus::number<1 + 2 * T::kNumSlots>{});
        }
        // NO `return` here, unlike the non-split sibling. The fused epilogue
        // needs every wave of every WG to reach the cluster barrier below, and it
        // needs producer wave 0 alive AFTER that barrier to TDM-stage the
        // published partials for the reduce. A producer that exits early takes
        // its cluster's barrier quorum with it and hangs the launch.
    } else {

    // ---------------------------------------------------------------------
    // Consumers (w[kNumProducerWaves] .. w[kNumWaves-1]). wmma accumulates the
    // result of the matrix multiplication.
    // ---------------------------------------------------------------------
    // The A-scale panel was already filled and published in the prologue, above
    // the producer/consumer split. It covers the tile's whole K range, so the
    // entire K loop reads it and nothing ever overwrites it -- no ring slot, no
    // FREE/DATA handshake, and no barrier id past the 9 already in use (the
    // binit/bjs/bjsw chains silently alias anything above 9 to __nbar_9).

    // Mma, FragA/FragB/FragC and `acc` are declared above the producer/consumer
    // split; only the instance is local to the consumers.
    Mma mma;
    //auto mma = make_tiled_mma<DataA, DataB, DataAcc>(
    //    seq<T::kExpM, T::kExpN, T::kExpKHalf>{},
    //    seq<T::kTileM, T::kTileN, T::kTileK>{},
    //    seq<T::kWmmaM, T::kWmmaN, T::kWmmaK>{}, wmma_adaptor_swap_ab{});
    //auto u_ra = make_layout_ra_ctdm<T>(lane_id, wave_m);
    //auto u_rb = make_layout_rb_ctdm<T>(lane_id, wave_n);
//
    //// WMMA source regs: 3-deep ring so a round's ds_load never overwrites VGPRs a
    //// still-running (multi-cycle) WMMA of a recent round reads (WMMA-source WAR,
    //// MI400 SPG 4.6.12.1).
    //typename decltype(mma)::vtype_a v_a[3];
    //typename decltype(mma)::vtype_b v_b[3];
    //typename decltype(mma)::vtype_c reg_c;
    //clear(reg_c);
    // -- TODO(kernel) 1a: A fragment map. UNVERIFIED. -----------------------
    // Assumed wave32 WMMA 16x16x128 A layout: lane l holds row m = l % 16 and the
    // 64 contiguous K elements starting at (l / 16) * 64. Read as 4 x b128.
    // If the probe disagrees, only this lambda changes.
    auto frag_a = [&](int slot, int m_tile, int k_tile) __attribute__((always_inline)) -> FragA {
        const int row = m_tile * T::kWmmaM + wave_m * (T::kExpM * T::kWmmaM) + (lane_id % T::kWmmaM);
        const int k0  = k_tile * T::kWmmaK + (lane_id / T::kWmmaM) * (T::kWmmaK / 2);
        const DataA* p = smem_a + slot * slot_a + (size_t)row * T::kSmemPitchA + k0;
        return *reinterpret_cast<const FragA*>(p);
    };

    // -- TODO(kernel) 1b: B fragment map. UNVERIFIED, and the harder of the two.
    // LDS holds the shuffled tile VERBATIM, so within the 16-row block that owns
    // column n the element (n, k) sits at
    //     (k>>5)*512 + ((k>>4)&1)*256 + (n&15)*16 + (k&15)
    // relative to the block's base for this K step. With the same assumed operand
    // layout as A (lane l -> column n = l%16, K half (l/16)), a lane's 64 elements
    // are NOT contiguous: they are four 16-byte runs at k = k0 + {0,16,32,48}+... .
    // The read below spells that out one 16-byte run at a time; collapsing it into
    // wider reads is an optimisation to make only after the probe confirms the map.
    auto frag_b = [&](int slot, int n_tile, int k_tile) __attribute__((always_inline)) -> FragB {
        const int n_in_tile = n_tile * T::kWmmaN + wave_n * (T::kExpN * T::kWmmaN);
        const int blk       = n_in_tile / T::kBShufBlockN;      // which 16-col block
        const int n_lo      = lane_id % T::kWmmaN;              // n & 15
        const int k0        = k_tile * T::kWmmaK + (lane_id / T::kWmmaN) * (T::kWmmaK / 2);
        const DataB* base   = smem_b + slot * slot_b + (size_t)blk * T::kSmemPitchB;
        FragB v;
        opus::static_for<4>([&](auto rN) __attribute__((always_inline)) {
            constexpr int r = decltype(rN)::value;
            const int k     = k0 + r * 16;
            const int off   = (k >> 5) * 512 + ((k >> 4) & 1) * 256 + n_lo * 16 + (k & 15);
            const auto w4   = *reinterpret_cast<const opus::vector_t<opus::i32_t, 4>*>(base + off);
            v[r * 4 + 0] = w4[0]; v[r * 4 + 1] = w4[1];
            v[r * 4 + 2] = w4[2]; v[r * 4 + 3] = w4[3];
        });
        return v;
    };

    // -- the scale fetch ----------------------------------------------------
    // Pack the e8m0 bytes this WMMA needs into the BX32 int operand. One int =
    // 4 bytes = the instruction's whole K=128. At kGroupK==32 those are 4 distinct
    // groups; at kGroupK==128 the one byte is broadcast (gfx950's pack_e8m0x4).
    //
    // Parameterised by (pointer, row, row pitch) so it serves all three sources
    // this pipeline now has: A from global, A from the LDS panel (T::kSfALds),
    // and B from global. Keeping ONE packer is deliberate -- the three differ
    // only in where the bytes live, never in how they are assembled.
    //
    // NOTE the OPSEL argument to the WMMA below is 0, meaning "scale comes from
    // lanes 0-15". Every lane computing the same value makes that safe; if you
    // move to a per-lane packing you must revisit a_scale_sel/b_scale_sel.
    auto pack_sf = [&](const DataSf* p, int row, int row_stride, int k_group) __attribute__((always_inline)) -> int {
        if constexpr (T::kScaleBcast) {
            const unsigned b = (unsigned)__builtin_bit_cast(unsigned char, p[(size_t)row * row_stride + k_group]);
            return (int)(b * 0x01010101u);
        } else {
            unsigned w = 0;
            opus::static_for<4>([&](auto jN) __attribute__((always_inline)) {
                constexpr int j = decltype(jN)::value;
                w |= ((unsigned)__builtin_bit_cast(unsigned char,
                        p[(size_t)row * row_stride + k_group + j])) << (8 * j);
            });
            return (int)w;
        }
    };

    // Per K-step consumer. DATA[s] gates on BOTH producers having landed slot s;
    // FREE_A[s]/FREE_B[s] release it afterwards. The s_wait_dscnt(0) before the
    // frees is what stops the producer's reload racing these ds-reads.
    auto consume_slot = [&](auto Sn, int k_step) __attribute__((always_inline)) {
        constexpr int s = Sn.value;
        bjsw(opus::number<1 + s>{});
        asm volatile("" ::: "memory");

        // Which kGroupN block row of w_scale span slot `in` reads, ABSOLUTE (the
        // B panel subtracts sfb_nb_base). One expression for both B scale
        // layouts, and it must stay that way: which ROW to read is decided by
        // kGroupN alone (the tensor has ceil(n/kGroupN) rows), while whether the
        // lane term can be dropped is a separate, weaker property,
        // kSfBUniformOverN. Branching the LAYOUT on the uniformity flag would be
        // a latent silent-wrong-answer bug: they agree for every tile defined
        // today, but a B_N=192 tile on 4 consumer waves has kExpN=3, so its
        // 48-column wave span does not divide 128, uniformity goes false while
        // kGroupN stays 128, and a blocked tensor would get indexed as if it had
        // n rows. The launcher's size(1) check cannot catch that -- the shape is
        // right. Clamped because the tile's span can run past n on the last
        // tile; those columns have their C store dropped anyway.
        //
        // Under uniformity the body ignores `in` entirely, which is what
        // collapses kSfBLoadsPerK to 1 -- the compiler will not prove that
        // itself, it would need tile_col's alignment. Measurably it does not:
        // ISA load counts followed 5*kExpK*(kExpM+kExpN) exactly.
        const int b_col_base = tile_col + wave_n * (T::kExpN * T::kWmmaN);
        const int nb_max     = (kargs.n + T::kGroupN - 1) / T::kGroupN - 1;
        auto sfb_nb = [&](int in) __attribute__((always_inline)) -> int {
            const int off = T::kSfBUniformOverN
                                ? 0
                                : (in * T::kWmmaN + (lane_id % T::kWmmaN));
            return opus_bmm_mx_min_i((b_col_base + off) / T::kGroupN, nb_max);
        };

        // ONE wide panel read per row for the whole K-step; each ik below takes
        // its own byte out of it. Hoisted to here, out of the ik loop, because
        // that is the entire point: kExpK WMMAs share one ds_read. A row's kExpK
        // bytes are consecutive in the panel (kg advances by one per ik), and
        // sf_pitch is 16-aligned while k_step*kExpK is kExpK-aligned, so the
        // 2- or 4-byte read is always aligned.
        //
        // It is an LDS read like the fragment reads, so it sits with them under
        // the ik loop's s_wait_dscnt(0). The panels are outside the ring, so
        // unlike those they race nothing.
        auto wide_read = [&](const DataSf* p) __attribute__((always_inline)) {
            if constexpr (T::kExpK == 2) {
                return (unsigned)*reinterpret_cast<const unsigned short*>(p);
            } else {
                return *reinterpret_cast<const unsigned*>(p);
            }
        };
        unsigned sa_w[T::kExpM];
        if constexpr (T::kSfAWideRead) {
            opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                constexpr int im = decltype(imN)::value;
                const int r = wave_m * (T::kExpM * T::kWmmaM) + im * T::kWmmaM
                              + (lane_id % T::kWmmaM);
                sa_w[im] = wide_read(smem_sfa + (size_t)r * sf_pitch
                                     + (size_t)k_step * T::kExpK);
            });
        }
        // kSfBLoadsPerK entries, not kExpN: under uniformity the whole wave span
        // sits in one block row, so there is one distinct read for the K-step.
        unsigned sb_w[T::kSfBWideRead ? T::kSfBLoadsPerK : 1];
        if constexpr (T::kSfBWideRead) {
            opus::static_for<T::kSfBLoadsPerK>([&](auto inN) __attribute__((always_inline)) {
                constexpr int in = decltype(inN)::value;
                const int nb = sfb_nb(in) - sfb_nb_base;
                sb_w[in] = wide_read(smem_sfb + (size_t)nb * sf_pitch
                                     + (size_t)k_step * T::kExpK);
            });
        }

        opus::static_for<T::kExpK>([&](auto ikN) __attribute__((always_inline)) {
            constexpr int ik = decltype(ikN)::value;
            // K group index of this WMMA's first scale byte.
            const int kg = (k_step * T::kBlockK + ik * T::kWmmaK) / T::kGroupK;

            // The GLOBAL scale bytes depend on nothing the producers put in the
            // ring, so they can be issued either before the
            // ds_reads (overlapping the LDS round trip) or after s_wait_dscnt(0)
            // (spread through the WMMA sequence, covering each other). Which one
            // wins depends on how many there are; T::kSfEarly carries the measured
            // threshold and the data behind it. Defined once here, issued at
            // whichever of the two points that flag selects.
            //
            // The scale operand is PER LANE, not per WMMA tile: the instruction
            // takes row m's 4 e8m0 bytes from lane m (opus.hpp:3339 "per-lane E8M0
            // exponent values", and OPSEL 0 = read them from lanes 0-15). So lane
            // l must supply row a_row + l%16 -- exactly what the gfx950 sibling's
            // make_layout_sfa_mxsk does with its `lane_id % T::W_M` coordinate. A
            // uniform per-tile scale would be in-bounds, silent, and would apply
            // row a_row's exponent to all 16 rows; with per-128K checkpoints that
            // is wrong on every row but one (DS V4 uses GROUP_K=128 + broadcast).
            // Lanes 16-31 duplicate 0-15 and are discarded by OPSEL 0.
            //
            // Both a_row and the B block row are clamped because the tile's span
            // can run past m / n on the last tile; the rows and columns they clamp
            // onto have their C store dropped anyway.
            // One row's A scale, from the LDS panel when the tile has one and
            // straight from global otherwise. Same pack_sf either way -- it is
            // already parameterised by pointer, row and row pitch, and the panel
            // is just a second (row, pitch) pair whose pointer lives in LDS.
            auto pack_sfa = [&](int im) __attribute__((always_inline)) -> int {
                const int r = wave_m * (T::kExpM * T::kWmmaM) + im * T::kWmmaM
                              + (lane_id % T::kWmmaM);
                if constexpr (T::kSfAWideRead) {
                    // Byte ik of this step's wide read, replicated across the
                    // BX32 operand. V_PERM_B32 permutes {src0:src1} with src1 as
                    // the low dword, so a selector of ik in all four bytes picks
                    // byte ik of sa_w four times. ONE instruction, and it stands
                    // in for the broadcast multiply the byte path already paid --
                    // so this widening spends no extra VALU, it only deletes
                    // ds_reads. No LDS access here at all.
                    return (int)__builtin_amdgcn_perm(sa_w[im], sa_w[im],
                                                      (unsigned)ik * 0x01010101u);
                } else if constexpr (T::kSfALds) {
                    // TILE-local row, and deliberately unclamped: r < kBlockM by
                    // construction, and a row past m read zero from the buffer
                    // bound during the fill. That is safe for the same reason the
                    // global path's clamp is: the scale operand is per lane, so
                    // row r's byte only ever reaches row r's accumulators, and an
                    // out-of-range row's C store is dropped. A zero exponent
                    // there cannot reach a valid row -- it is not a reduction.
                    return pack_sf(smem_sfa, r, sf_pitch, kg);
                } else {
                    return pack_sf(ptr_sfa,
                                   opus_bmm_mx_min_i(tile_row + r, kargs.m - 1),
                                   kargs.stride_sfa, kg);
                }
            };

            int sa_v[T::kExpM];
            auto fill_sa = [&]() __attribute__((always_inline)) {
                opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                    sa_v[decltype(imN)::value] = pack_sfa(decltype(imN)::value);
                });
            };

            // One B scale value per span slot, from the LDS panel when the tile
            // has one and straight from global otherwise. sfb_nb (hoisted to the
            // K-step, where the wide read shares it) owns the row expression;
            // this only chooses which (pointer, pitch) pair to apply it to.
            int sb_v[T::kExpN];
            auto fill_sb = [&]() __attribute__((always_inline)) {
              opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                constexpr int in = decltype(inN)::value;
                // Under uniformity every slot reads the SAME value, so index the
                // one entry that exists rather than kExpN copies of it.
                constexpr int iw = T::kSfBUniformOverN ? 0 : in;
                if constexpr (T::kSfBWideRead) {
                    sb_v[in] = (int)__builtin_amdgcn_perm(sb_w[iw], sb_w[iw],
                                                          (unsigned)ik * 0x01010101u);
                } else if constexpr (T::kSfBLds) {
                    // PANEL-local block row. Unclamped subtraction is safe:
                    // sfb_nb returns at least tile_col/kGroupN = sfb_nb_base,
                    // and at most the block of the span's last column, which is
                    // inside kSfBPanelRows by construction.
                    sb_v[in] = pack_sf(smem_sfb, sfb_nb(in) - sfb_nb_base,
                                       sf_pitch, kg);
                } else {
                    sb_v[in] = pack_sf(ptr_sfb, sfb_nb(in), kargs.stride_sfb, kg);
                }
              });
            };

            // kSfAEarly / kSfBEarly, not kSfEarly: a panel read is a ds_read and
            // wants to sit with the other ds_reads, under the one
            // s_wait_dscnt(0) below.
            if constexpr (T::kSfAEarly) { fill_sa(); }
            if constexpr (T::kSfBEarly) { fill_sb(); }

            FragA va[T::kExpM];
            FragB vb[T::kExpN];
            opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                va[decltype(imN)::value] = frag_a(s, decltype(imN)::value, ik);
            });
            opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                vb[decltype(inN)::value] = frag_b(s, decltype(inN)::value, ik);
            });
            opus::s_wait_dscnt(opus::number<0>{});

            if constexpr (!T::kSfBEarly) { fill_sb(); }

            opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                constexpr int im = decltype(imN)::value;
                // On the late path each row's A scale is issued here rather than
                // in one batch up front, so its latency is covered by the WMMAs of
                // the previous im iteration -- that staggering is exactly what the
                // high-kSfLoadsPerK tiles lose if they go early.
                if constexpr (!T::kSfAEarly) sa_v[im] = pack_sfa(im);
                opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                    constexpr int in = decltype(inN)::value;
                    acc[im][in] = mma(va[im], vb[in], acc[im][in], sa_v[im], sb_v[in],
                                      opus::number<0>{}, opus::number<0>{});
                });
            });
            __builtin_amdgcn_sched_barrier(0);
        });

        asm volatile("" ::: "memory");
        bjs(opus::number<1 + T::kNumSlots + s>{});      // FREE_A[s]
        bjs(opus::number<1 + 2 * T::kNumSlots + s>{});  // FREE_B[s]
    };

    // Slots are consumed in the same compile-time order the producer signals them,
    // which is what keeps the per-slot signal counts symmetric (asymmetry hangs).
    //
    // consume_slot's k_step is ABSOLUTE (k_step_beg + local), not split-relative.
    // It has to be: the only thing it indexes with it is the scale panel, and
    // both panels are filled over the tile's WHOLE K range by every split, so a
    // split-relative step would read split 0's exponents on every split.
    {
        int k = 0;
        for (; k + T::kNumSlots <= k_steps; k += T::kNumSlots) {
            const int kb = k_step_beg + k;
            opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                consume_slot(sN, kb + (int)decltype(sN)::value);
            });
        }
        const int rem = k_steps - k;
        const int kb  = k_step_beg + k;
        opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
            if ((int)decltype(sN)::value < rem) consume_slot(sN, kb + (int)decltype(sN)::value);
        });
    }

    // -- epilogue A (still inside the consumer branch) ----------------------
    // Two jobs, split by whether this WG owns the last k-slice:
    //   NOT last -- publish the fp32 partial as DataWs to the global workspace.
    //   last     -- fold bias into its own fp32 partial and keep it in registers.
    // Bias is folded HERE and only here, by the last split: folding it into a
    // published partial instead would let SplitK copies of it survive the sum.
    // The C write and the reduce both wait for the cluster barrier below.
    if constexpr (SplitK > 1) {
        if (!is_last && !m_oob) {
            DataWs* ws_ptr = reinterpret_cast<DataWs*>(kargs.ptr_ws);
            const size_t ws_base =
                (ws_tile_idx * (size_t)(SplitK - 1) + (size_t)split_idx) * kWsTileElems;
            auto g_ws = opus::make_gmem<DataWs>(
                ws_ptr + ws_base, (unsigned int)(kWsTileElems * sizeof(DataWs)));
            // gfx12 CPOL TH_WB(3) | SCOPE_DEV(2<<3): keep the partial DIRTY-RESIDENT
            // in GL2 rather than write-rinsing it to HBM, published device-wide, so
            // the last WG's reduce read hits L2. The peers that read it co-reside on
            // the same device by construction (they are cluster peers), which is what
            // makes device scope sufficient and a system-scope rinse pure cost.
            opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                constexpr int im = decltype(imN)::value;
                opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                    constexpr int in   = decltype(inN)::value;
                    const int frag_off = ws_lane_base + (im * T::kExpN + in) * kFragC;
                    auto reg_ws        = opus::cast<DataWs>(acc[im][in]);
                    opus::static_for<kFragC / kWsChunk>([&](auto cN) __attribute__((always_inline)) {
                        constexpr int c = decltype(cN)::value;
                        g_ws.template store<kWsChunk>(
                            opus::slice(reg_ws,
                                        opus::number<c * kWsChunk>{},
                                        opus::number<c * kWsChunk + kWsChunk>{}),
                            frag_off + c * kWsChunk,
                            0,
                            opus::number<OPUS_BMM_MXSK_WS_CPOL>{});
                    });
                });
            });
            opus::s_wait_storecnt(opus::number<0>{});
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        }
    }

    if (is_last && kargs.ptr_bias) {
        // bf16 [N], broadcast over M. Every element of a C fragment sits in the
        // SAME column (the map below walks rows with i, columns with in), so one
        // scalar load per (im, in) covers all kFragC of them -- do not lift this
        // into the i loop.
        const opus::bf16_t* pb = reinterpret_cast<const opus::bf16_t*>(kargs.ptr_bias)
                                 + (size_t)batch_id * kargs.stride_bias_batch;
        opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
            constexpr int im = decltype(imN)::value;
            opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                constexpr int in = decltype(inN)::value;
                const int col    = col_base + in * T::kWmmaN;
                // Columns past n have their C store dropped anyway; reading 0
                // rather than clamping keeps a tail column from inheriting the
                // last real column's bias in any future vectorised store.
                const float b = (col < kargs.n) ? (float)pb[col] : 0.0f;
                opus::static_for<kFragC>([&](auto iN) __attribute__((always_inline)) {
                    acc[im][in][decltype(iN)::value] += b;
                });
            });
        });
    }
    }   // end of the consumer branch (producers skipped straight to here)

    // -- CONVERGED cross-WG sync -------------------------------------------
    // WG s_barrier first (aligns this WG's own waves and retires their memory
    // state), then the CLUSTER barrier (-3) signalled by wave 0, which aligns all
    // SplitK*MClusterWg WGs of the cluster. A cluster barrier, not a semaphore:
    // every split of a tile CO-RESIDES by construction, so the barrier alone is
    // enough to guarantee each non-last partial is stored before the last WG
    // reads it, and it is deterministic (no atomics, run-to-run bit-identical).
    //
    // Skipped entirely at SplitK == 1, where no WG reads another's data and the
    // cluster is a single workgroup.
    // The WG-local barrier is UNCONDITIONAL: it is the same producer/consumer
    // rendezvous the non-split sibling ran before its C store, and it is what
    // retires this WG's own memory state before the cluster barrier reads across
    // WGs. Only the cross-WG half is SplitK-gated.
    __builtin_amdgcn_s_barrier();
    if constexpr (SplitK > 1) {
        if (wave_id == 0)
            __builtin_amdgcn_s_barrier_signal(-3);
        __builtin_amdgcn_s_barrier_wait(-3);
    }

    // -- reduce: the last split sums the SplitK-1 published partials into acc --
    // Two strategies, picked at compile time by whether a partial tile even fits
    // the LDS this tile allocated. The a8w8 tiles are byte-typed, so their
    // kLdsTotalBytes is sized for 1-byte A/B rings and a bf16 B_M x B_N partial
    // can be several times larger than the whole ring -- which is why this cannot
    // be the a16w16 sibling's unconditional static_assert on a 3-deep ring.
    //
    //   kRing >= 1 : producer w0 TDM bulk-stages partial tiles into LDS (one
    //                coalesced global read, deep MLP) and the consumers read
    //                their lane runs back as 16B dwordx4 out of fast LDS.
    //   kRing == 0 : no tile fits; consumers read their lane runs straight from
    //                the workspace. Still fully coalesced and still an L2 hit
    //                (the partials are dirty-resident there), just without the
    //                staging's latency hiding.
    if constexpr (SplitK > 1) {
        constexpr size_t kWsTileBytes = kWsTileElems * sizeof(DataWs);
        constexpr int kFitTiles       = (int)((size_t)T::kLdsTotalBytes / kWsTileBytes);
        constexpr int kRing =
            (kFitTiles >= kBmmFuseReduceRing) ? kBmmFuseReduceRing : kFitTiles;

        if (is_last && !m_oob) {
            DataWs* ws_ptr = reinterpret_cast<DataWs*>(kargs.ptr_ws);
            auto ws_base_of = [&](int sp) __attribute__((always_inline)) -> size_t {
                return (ws_tile_idx * (size_t)(SplitK - 1) + (size_t)sp) * kWsTileElems;
            };
            // Add one staged partial into acc, reading through `src` -- either an
            // LDS slot or the global tile. The (im, in, c) walk is the SAME one the
            // store used, which is the whole contract between the two sides.
            auto accumulate = [&](auto& src) __attribute__((always_inline)) {
                opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                    constexpr int im = decltype(imN)::value;
                    opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                        constexpr int in   = decltype(inN)::value;
                        const int frag_off = ws_lane_base + (im * T::kExpN + in) * kFragC;
                        opus::static_for<kFragC / kWsChunk>(
                            [&](auto cN) __attribute__((always_inline)) {
                                constexpr int c = decltype(cN)::value;
                                auto vp = src.template load<kWsChunk>(frag_off + c * kWsChunk);
#pragma unroll
                                for (int i = 0; i < kWsChunk; ++i)
                                    acc[im][in][c * kWsChunk + i] += (float)vp[i];
                            });
                    });
                });
            };

            if constexpr (kRing >= 1) {
                // Contiguous B_M x B_N DataWs tile: dim0 (N) is the fast axis, the
                // window is the whole extent, so no LDS padding and no multicast.
                using WindowWs = opus::tdm<DataWs, opus::seq<T::kBlockN, T::kBlockM>>;
                DataWs* lds_ws = reinterpret_cast<DataWs*>(lds_buf);
                auto stage     = [&](int slot, int sp) __attribute__((always_inline)) {
                    auto w = opus::make_tdm<WindowWs>((u32_t) reinterpret_cast<u64_t>(lds_ws),
                                                      ws_ptr + ws_base_of(sp),
                                                      (u32_t)T::kBlockN,
                                                      (u32_t)T::kBlockM,
                                                      (u64_t)T::kBlockN);
                    w.async_load((u32_t)((size_t)slot * kWsTileElems));
                };
                // Ring-bounded chunks. Two WG barriers per chunk: the first
                // publishes the staged tiles, the second keeps the next chunk's
                // DMA from overwriting slots the consumers are still reading.
                // When the ring covers every partial this runs once, which is the
                // stage-all case (max MLP, one barrier pair) without a second
                // code path for it.
#pragma unroll 1
                for (int base = 0; base < SplitK - 1; base += kRing) {
                    const int chunk = opus_bmm_mx_min_i(kRing, (SplitK - 1) - base);
                    if (is_producer && wave_id == 0) {
#pragma unroll 1
                        for (int j = 0; j < chunk; ++j)
                            stage(j, base + j);
                        opus::s_wait_tensorcnt<0>();
                    }
                    __builtin_amdgcn_s_barrier();
                    if (!is_producer) {
#pragma unroll 1
                        for (int j = 0; j < chunk; ++j) {
                            auto s_ws = opus::make_smem(lds_ws + (size_t)j * kWsTileElems);
                            accumulate(s_ws);
                        }
                    }
                    __builtin_amdgcn_s_barrier();
                }
            } else {
                if (!is_producer) {
#pragma unroll 1
                    for (int sp = 0; sp < SplitK - 1; ++sp) {
                        auto g_ws = opus::make_gmem<DataWs>(
                            ws_ptr + ws_base_of(sp),
                            (unsigned int)(kWsTileElems * sizeof(DataWs)));
                        accumulate(g_ws);
                    }
                }
            }
        }
    }

    // -- TODO(kernel) 3: C fragment map. UNVERIFIED. ------------------------
    // Assumed wave32 WMMA 16x16 C layout: lane l holds column n = l % 16 and rows
    // m = (l / 16) * 8 + i for i in 0..7. Scalar stores, because that map is one
    // element per (lane, i) along N -- vectorise only after the probe.
    //
    // Only the LAST split writes C, and only its consumer waves: `acc` now holds
    // the full-K sum in fp32 and is cast to D_OUT exactly once, here.
    //
    // This is the one map the workspace path does NOT depend on -- the partial
    // store/reduce uses the private lane-contiguous layout above -- so a probe
    // that corrects this lambda does not invalidate the split-K plumbing.
    if (is_last && !is_producer) {
        const int row_half = (lane_id / T::kWmmaN) * (T::kWmmaM / 2);
        opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
            constexpr int im = decltype(imN)::value;
            opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                constexpr int in = decltype(inN)::value;
                const int col = col_base + in * T::kWmmaN;
                opus::static_for<T::kWmmaM / 2>([&](auto iN) __attribute__((always_inline)) {
                    constexpr int i = decltype(iN)::value;
                    const int row = tile_row + wave_m * (T::kExpM * T::kWmmaM)
                                  + im * T::kWmmaM + row_half + i;
                    if (row < kargs.m && col < kargs.n)
                        ptr_c[(size_t)row * kargs.stride_c + col] =
                            (D_OUT)acc[im][in][i];
                });
            });
        });
    }
#else
    (void)kargs;   // non-gfx1250 device pass: empty stub (multi-arch wheel safety)
#endif // __gfx1250__
#endif // __HIP_DEVICE_COMPILE__
}

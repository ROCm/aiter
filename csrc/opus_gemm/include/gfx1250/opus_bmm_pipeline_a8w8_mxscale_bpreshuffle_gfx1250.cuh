// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 (MI450) a8w8 mxscale BMM with a preshuffled B, TDM producer/consumer.
//
//   Y[M, batch, N] = O[M, batch, K] @ wo_a[batch, N, K]^T
//
// This file is a SCAFFOLD. The data movement is complete and is a direct port of
// opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh (same 4-wave shape,
// same per-slot named-barrier protocol, same TDM run-ahead). What is left for you
// is marked `TODO(kernel)` and is exactly three things:
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
//   3. store_c           -- the C fragment map, same UNVERIFIED caveat as (1).
//
// HOW TO VERIFY (1) AND (3) BEFORE TRUSTING ANY NUMBER. These three maps are the
// only places this kernel can be wrong while still producing plausible output, so
// do not infer them -- measure them. Launch one workgroup on M=N=16, K=128 with
// A[m][k] = m*128+k, B = identity in the shuffled layout, all scales 0x7F, and
// print (lane, i) -> value. Anything that disagrees with the constants below is
// the hardware telling you the layout, and the layout wins.
//
// The wave split is w0 = A producer, w1 = B producer, and every wave from
// kNumProducerWaves up is a WMMA consumer. The wave count follows BLOCK_SIZE and
// is not assumed to be 4 anywhere below: the barrier member counts are expressed
// in kNumWaves / kNumConsumerWaves and the consumer's wave_m/wave_n come from
// kTileM/kTileN, which the traits derive from kNumConsumerWaves. See the
// kNumWaves block in the traits header for why going past 4 waves matters.
// Every compile-time value comes from the traits header; this file declares no
// geometry of its own.
#pragma once

#include "opus_bmm_traits_a8w8_mxscale_bpreshuffle_gfx1250.cuh"

#ifdef __HIP_DEVICE_COMPILE__
using namespace opus;
using opus::operator""_I;
#endif

__host__ __device__ constexpr inline int opus_bmm_mx_ceil_div_i(int a, int b) {
    return (a + b - 1) / b;
}
__host__ __device__ constexpr inline int opus_bmm_mx_min_i(int a, int b) {
    return a < b ? a : b;
}

// launch_bounds comes from the traits, not a literal: the host launcher already
// sizes the block as dim3(T::BLOCK_SIZE), so a hardcoded bound here is a silent
// mismatch the moment a tile picks a wave count other than 4 (the register
// allocator would be budgeting for 128 threads while 192 launch).
template <typename UserTraits>
__global__ __launch_bounds__(opus::remove_cvref_t<UserTraits>::BLOCK_SIZE, 1)
void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250(opus_bmm_a8w8_mxscale_kargs_gfx1250 kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx1250__)
    using T       = remove_cvref_t<UserTraits>;
    using DataA   = typename T::DataA;
    using DataB   = typename T::DataB;
    using DataC   = typename T::DataC;
    using DataAcc = typename T::DataAcc;
    using DataSf  = typename T::DataSf;
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

    // -- tile / batch coordinates -------------------------------------------
    // grid = (M/B_M, N/B_N, batch). Split-K is NOT wired in this scaffold: there
    // is no fp32 workspace and no reduce kernel, so one WG owns the whole K range
    // of its tile and stores C directly. Add them the way the a16w16 pipeline does
    // (ptr_ws + a separate reduce launch) if you need it.
    const int tile_row = (int)__builtin_amdgcn_workgroup_id_x() * T::kBlockM;
    const int tile_col = (int)__builtin_amdgcn_workgroup_id_y() * T::kBlockN;
    const int batch_id = (int)__builtin_amdgcn_workgroup_id_z();

    const int k_steps = opus_bmm_mx_ceil_div_i(kargs.k, T::kBlockK);
    if (k_steps <= 0) return;

    // Per-batch bases. Strides are in elements of each tensor's own dtype.
    const DataA* ptr_a  = reinterpret_cast<const DataA*>(kargs.ptr_a)  + (size_t)batch_id * kargs.stride_a_batch;
    const DataB* ptr_b  = reinterpret_cast<const DataB*>(kargs.ptr_b)  + (size_t)batch_id * kargs.stride_b_batch;
    DataC*       ptr_c  = reinterpret_cast<DataC*>(kargs.ptr_c)        + (size_t)batch_id * kargs.stride_c_batch;
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

        if (wave_id == 0) {
            // A is a plain [M, K] tensor: extents (K, M), row stride = stride_a.
            // Out-of-range M rows and the K tail clamp to a zero-extent DMA, which
            // is where the free OOB handling comes from -- do not add an M guard.
            auto w = opus::make_tdm<WindowA>((u32_t)reinterpret_cast<u64_t>(smem_a), ptr_a,
                                             (u32_t)kargs.k, (u32_t)kargs.m, (u64_t)kargs.stride_a,
                                             (u32_t)0, (u32_t)tile_row);
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
                                             (u32_t)0, (u32_t)(tile_col / T::kBShufBlockN));
            produce(w, T::kSlotElemsB, KStepB, opus::number<1 + 2 * T::kNumSlots>{});
        }
        __builtin_amdgcn_s_barrier();   // rendezvous; never "signal then exit"
        return;
    }

    // ---------------------------------------------------------------------
    // Consumers (w[kNumProducerWaves] .. w[kNumWaves-1]). wmma accumulates the
    // result of the matrix multiplication.
    // ---------------------------------------------------------------------
    const int wave_split = wave_id - T::kNumProducerWaves;
    // TileN: consumers split N (wave_n = wave_split, wave_m = 0).
    // TileM: consumers split M (wave_m = wave_split, wave_n = 0).
    const int wave_m = (T::LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? wave_split : 0;
    const int wave_n = (T::LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? 0 : wave_split;

    // The A-scale panel was already filled and published in the prologue, above
    // the producer/consumer split. It covers the tile's whole K range, so the
    // entire K loop reads it and nothing ever overwrites it -- no ring slot, no
    // FREE/DATA handshake, and no barrier id past the 9 already in use (the
    // binit/bjs/bjsw chains silently alias anything above 9 to __nbar_9).

    using Mma = opus::wmma<DataA, DataB, DataAcc, T::kWmmaM, T::kWmmaN, T::kWmmaK>;
    Mma mma;
    using FragA = opus::vector_t<opus::i32_t, 16>;   // 64 fp8 per lane
    using FragB = opus::vector_t<opus::i32_t, 16>;
    using FragC = typename Mma::vtype_c;             // 8 fp32 per lane

    FragC acc[T::kExpM][T::kExpN];
    opus::static_for<T::kExpM>([&](auto im) __attribute__((always_inline)) {
        opus::static_for<T::kExpN>([&](auto in) __attribute__((always_inline)) {
            clear(acc[decltype(im)::value][decltype(in)::value]);
        });
    });
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
    {
        int k = 0;
        for (; k + T::kNumSlots <= k_steps; k += T::kNumSlots) {
            const int kb = k;
            opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
                consume_slot(sN, kb + (int)decltype(sN)::value);
            });
        }
        const int rem = k_steps - k;
        const int kb  = k;
        opus::static_for<T::kNumSlots>([&](auto sN) __attribute__((always_inline)) {
            if ((int)decltype(sN)::value < rem) consume_slot(sN, kb + (int)decltype(sN)::value);
        });
    }

    // -- TODO(kernel) 3: C fragment map. UNVERIFIED. ------------------------
    // Assumed wave32 WMMA 16x16 C layout: lane l holds column n = l % 16 and rows
    // m = (l / 16) * 8 + i for i in 0..7. Scalar stores, because that map is one
    // element per (lane, i) along N -- vectorise only after the probe.
    __builtin_amdgcn_s_barrier();
    {
        const int col_base = tile_col + wave_n * (T::kExpN * T::kWmmaN) + (lane_id % T::kWmmaN);
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
                            (DataC)acc[im][in][i];
                });
            });
        });
    }
#else
    (void)kargs;   // non-gfx1250 device pass: empty stub (multi-arch wheel safety)
#endif // __gfx1250__
#endif // __HIP_DEVICE_COMPILE__
}

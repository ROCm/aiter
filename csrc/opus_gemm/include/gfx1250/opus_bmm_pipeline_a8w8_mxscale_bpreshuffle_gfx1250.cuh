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
//   2. the scale fetch   -- currently plain per-tile global loads of e8m0 bytes.
//                           Correct but not fast; the LDS scale panel is the
//                           optimisation and T::kSfPanel reserves the room for it.
//   3. store_c           -- the C fragment map, same UNVERIFIED caveat as (1).
//
// HOW TO VERIFY (1) AND (3) BEFORE TRUSTING ANY NUMBER. These three maps are the
// only places this kernel can be wrong while still producing plausible output, so
// do not infer them -- measure them. Launch one workgroup on M=N=16, K=128 with
// A[m][k] = m*128+k, B = identity in the shuffled layout, all scales 0x7F, and
// print (lane, i) -> value. Anything that disagrees with the constants below is
// the hardware telling you the layout, and the layout wins.
//
// The 4-wave split is w0 = A producer, w1 = B producer, w2/w3 = WMMA consumers.
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

template <typename UserTraits>
__global__ __launch_bounds__(128, 1)
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
    //   DATA[s]   = 1        + s   memcnt = kNumWaves            (both producers + both consumers)
    //   FREE_A[s] = 1 +   P  + s   memcnt = 1 + kNumConsumerWaves (prodA + consumers)
    //   FREE_B[s] = 1 + 2*P  + s   memcnt = 1 + kNumConsumerWaves (prodB + consumers)
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
    // Consumers (w2, w3). wmma accumulates the result of the matrix multiplication.
    // ---------------------------------------------------------------------
    const int wave_split = wave_id - T::kNumProducerWaves;
    // TileN: consumers split N (wave_n = wave_split, wave_m = 0).
    // TileM: consumers split M (wave_m = wave_split, wave_n = 0).
    const int wave_m = (T::LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? wave_split : 0;
    const int wave_n = (T::LAYOUT == opus_gfx1250_bmm::kLayoutTileM) ? 0 : wave_split;

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

    // -- TODO(kernel) 2: the scale fetch. -----------------------------------
    // Simple and correct: read the e8m0 bytes this WMMA needs straight from global
    // and pack them into the BX32 int operand. One int = 4 bytes = the instruction's
    // whole K=128. At kGroupK==32 those are 4 distinct groups; at kGroupK==128 the
    // one byte is broadcast (this is gfx950's pack_e8m0x4, unchanged).
    //
    // This is the slow path on purpose -- it keeps the scaffold's correctness
    // argument short. The optimisation is to stage the tile's scales in LDS once
    // per K step (T::kSfPanel reserves kSegBytesSfA/B for exactly that) and read
    // them with ds_read, which is what the gfx950 mxscale pipeline does.
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

        opus::static_for<T::kExpK>([&](auto ikN) __attribute__((always_inline)) {
            constexpr int ik = decltype(ikN)::value;
            // K group index of this WMMA's first scale byte.
            const int kg = (k_step * T::kBlockK + ik * T::kWmmaK) / T::kGroupK;

            FragA va[T::kExpM];
            FragB vb[T::kExpN];
            opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                va[decltype(imN)::value] = frag_a(s, decltype(imN)::value, ik);
            });
            opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                vb[decltype(inN)::value] = frag_b(s, decltype(inN)::value, ik);
            });
            opus::s_wait_dscnt(opus::number<0>{});

            opus::static_for<T::kExpM>([&](auto imN) __attribute__((always_inline)) {
                constexpr int im = decltype(imN)::value;
                // The scale operand is PER LANE, not per WMMA tile: the
                // instruction takes row m's 4 e8m0 bytes from lane m (opus.hpp:3339
                // "per-lane E8M0 exponent values", and OPSEL 0 = read them from
                // lanes 0-15). So lane l must supply row a_row + l%16 -- exactly
                // what the gfx950 sibling's make_layout_sfa_mxsk does with its
                // `lane_id % T::W_M` coordinate. A uniform per-tile scale would be
                // in-bounds, silent, and would apply row a_row's exponent to all 16
                // rows; with per-128K checkpoints a uniform per-tile scale would
                // be wrong on every row but one (DS V4 uses GROUP_K=128 + broadcast).
                // Lanes 16-31 duplicate 0-15 and are discarded by OPSEL 0.
                // Clamped because a_row + 15 can exceed m on the last tile; the
                // rows it clamps onto have their C store dropped anyway.
                const int a_row0 = tile_row + wave_m * (T::kExpM * T::kWmmaM) + im * T::kWmmaM;
                const int a_row  = opus_bmm_mx_min_i(a_row0 + (lane_id % T::kWmmaM), kargs.m - 1);
                const int sa     = pack_sf(ptr_sfa, a_row, kargs.stride_sfa, kg);
                opus::static_for<T::kExpN>([&](auto inN) __attribute__((always_inline)) {
                    constexpr int in = decltype(inN)::value;
                    // Same per-lane rule on the B side: lane l supplies column
                    // b_col + l%16. B's scale is NOT preshuffled -- w_scale is the
                    // plain [batch, N, K/GROUP_K] tensor, so this indexes N
                    // directly even though the weights themselves are shuffled.
                    const int b_col0 = tile_col + wave_n * (T::kExpN * T::kWmmaN) + in * T::kWmmaN;
                    const int b_col  = opus_bmm_mx_min_i(b_col0 + (lane_id % T::kWmmaN), kargs.n - 1);
                    const int sb     = pack_sf(ptr_sfb, b_col, kargs.stride_sfb, kg);
                    acc[im][in] = mma(va[im], vb[in], acc[im][in], sa, sb,
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

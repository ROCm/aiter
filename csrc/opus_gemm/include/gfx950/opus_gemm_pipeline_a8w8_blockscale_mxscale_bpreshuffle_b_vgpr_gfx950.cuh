// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_overlap_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_prefetch_b_vgpr_detail {

// Read an A fragment directly from LDS into its assigned AGPRs.
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
    constexpr int physical_reg = decltype(reg)::value;
    static_assert(physical_reg >= 0 && physical_reg < 128);
    opus::array<Chunk, 2> values;

    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        [[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
        values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
    });

    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
    [[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}

// Pack adjacent MFMA output fragments into contiguous BF16 stores.
// Use the known wave64 lane directly; opus::shfl adds lane-id instructions here.
template<class T, bool ContiguousWaveN = false, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(
    const Accumulators& c, Gmem gc, int lane, int wm, int wn, int stride_c) {
    using namespace opus;
    static_assert(T::W_M == 16 && T::W_N == 16);
    const int side = (lane / 16) % 2;
    static_for<T::E_M * T::E_N / 2>([&](auto pair) {
        constexpr int ci = decltype(pair)::value * 2;
        const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci + 1]));
        const unsigned own0 = side ? second[0] : first[0];
        const unsigned own1 = side ? second[1] : first[1];
        const unsigned src0 = side ? first[0] : second[0];
        const unsigned src1 = side ? first[1] : second[1];
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src0);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src1);
        vector_t<unsigned, 4> packed;
        packed[0] = side ? other0 : own0;
        packed[1] = side ? other1 : own1;
        packed[2] = side ? own0 : other0;
        packed[3] = side ? own1 : other1;
        const int m = (ci + side) / T::E_N;
        const int n = (ci + side) % T::E_N;
        const int rr = (m * T::T_M + wm) * 16 + lane % 16;
        const int nn = (ContiguousWaveN ? n + wn * T::E_N : n * T::T_N + wn) * 16 + (lane / 32) * 8;
        store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed),
            rr * stride_c + nn, 0, number<T::STORE_CACHE>{});
    });
    if constexpr (T::E_M * T::E_N % 2 != 0) {
        constexpr int ci = T::E_M * T::E_N - 1;
        const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[0]);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[1]);
        if (side == 0) {
            vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
            const int rr = ((ci / T::E_N) * T::T_M + wm) * 16 + lane % 16;
            const int nn =
                (ContiguousWaveN ? (ci % T::E_N) + wn * T::E_N : (ci % T::E_N) * T::T_N + wn) * 16 + (lane / 32) * 8;
            store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed),
                rr * stride_c + nn, 0, number<T::STORE_CACHE>{});
        }
    }
}

// Choose an output stripe that fits the LDS allocation reused by the epilogue.
__device__ inline constexpr int output_rows(int bm, int bn, int pipeline_bytes) {
    for (int rows = bm; rows >= 16; rows -= 16)
        if (bm % rows == 0 && rows * (bn + 8) * 2 <= pipeline_bytes)
            return rows;
    return 16;
}

// Counts follow the actual prologue: B, first A, scales, remaining A.
template<class T>
__device__ inline constexpr int wait_a(int t) {
    int issued = T::INITIAL_B * T::B_ISSUES;
    int ready = -1;
    issued += T::A_COPIES;
    if (t == 0)
        ready = issued;
    issued += T::SCALE_ISSUES;
    // K0 consumes the resident scales as well as A0. The first barrier
    // must publish those async LDS writes before any scale LDS reads.
    if (t == 0)
        ready = issued;
    for (int stage = 1; stage < T::INITIAL_A; ++stage) {
        issued += T::A_COPIES;
        if (stage == t)
            ready = issued;
    }
    for (int step = 0; step < t; ++step) {
        if (step + T::B_AHEAD < T::PANELS)
            issued += T::B_ISSUES;
        if (step + T::STAGES - 1 < T::PANELS) {
            issued += T::A_COPIES;
            if (step + T::STAGES - 1 == t)
                ready = issued;
        }
    }
    return issued - ready;
}

template<class T>
__device__ inline constexpr int wait_b(int t) {
    int issued = 0;
    int ready = -1;
    for (int set = 0; set < T::INITIAL_B; ++set) {
        issued += T::B_ISSUES;
        if (set == t)
            ready = issued;
    }
    issued += T::INITIAL_A * T::A_COPIES + T::SCALE_ISSUES;
    for (int step = 0; step <= t; ++step) {
        if (step + T::B_AHEAD < T::PANELS) {
            issued += T::B_ISSUES;
            if (step + T::B_AHEAD == t)
                ready = issued;
        }
        if (step + T::STAGES - 1 < T::PANELS)
            issued += T::A_COPIES;
    }
    return issued - ready;
}

// Fixed-shape entries use contiguous matrix and scale strides.
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
__device__ __forceinline__ void apply_fixed_args(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950& a) {
    a.m = OPUS_BP_FIXED_M;
    a.n = OPUS_BP_FIXED_N;
    a.k = OPUS_BP_FIXED_K;
    a.batch = 1;
    a.stride_a = OPUS_BP_FIXED_K;
    a.stride_b = OPUS_BP_FIXED_K;
    a.stride_c = OPUS_BP_FIXED_N;
    a.stride_a_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_K);
    a.stride_b_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_N) * OPUS_BP_FIXED_K);
    a.stride_c_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_N);
    a.stride_sfa = OPUS_BP_FIXED_M;
    a.stride_sfb = OPUS_BP_FIXED_K / 128;
    a.stride_sfa_batch = OPUS_BP_FIXED_M * (OPUS_BP_FIXED_K / 128);
    a.stride_sfb_batch = (OPUS_BP_FIXED_N / 128) * (OPUS_BP_FIXED_K / 128);
}
#endif

// Read the current A panel, prefetch the future panel, then consume
// current A/B operands. B uses a register ring and A uses an LDS ring.
template<class Traits>
__device__ __forceinline__ void bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
    opus_gemm_prefetch_b_vgpr_detail::apply_fixed_args(args);
#endif
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using Index = unsigned;
    constexpr int WaveColumns = T::T_N;

    // Tile geometry and runtime strides.
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int NB = T::STAGES;
    constexpr int BD = T::B_AHEAD;
    constexpr int BS = T::B_SETS;
    constexpr int P = T::PANELS;
    constexpr int MR = T::MR;
    constexpr int NR = T::NR;
    constexpr int K = T::K;
    constexpr int SC = T::SCALE_COLS;
    constexpr int BK = T::B_K;
    constexpr int KS = BK / 128;

    const int M = args.m;
    const int N = args.n;
    const int stride_a = T::CONTIGUOUS ? K : args.stride_a;
    const int stride_b = T::CONTIGUOUS ? K : args.stride_b;
    const int stride_c = args.stride_c;

    static_assert((BK == 128 || BK == 256) && SC == K / 128);
    static_assert(T::T_M == 1 && T::T_N * 64 == T::BLOCK_SIZE);
    static_assert(T::MR * 16 == T::B_M && T::NR * T::T_N * 16 == T::B_N);

    // Tile and thread coordinates.
    const Index tid = thread_id_x();
    const Index lane = tid % 64;
    const Index lane16 = lane % 16;
    const Index kg = lane / 16;
    const Index wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const Index wn = wave;
    const unsigned pid = block_id_x();
    const unsigned gm = (M + BM - 1) / BM;
    const unsigned gn = N / BN;
    const unsigned total = gm * gn;
    unsigned tile = T::XCD_ORDER ? pid % 8 * ((total + 7) / 8) + pid / 8 : pid;
    if constexpr (T::XCD_ORDER)
        if (tile >= total)
            return;
    if constexpr (T::GROUPED_M > 0) {
        const unsigned first = tile / (T::GROUPED_M * gn) * T::GROUPED_M;
        const unsigned rows = gm - first < T::GROUPED_M ? gm - first : T::GROUPED_M;
        const unsigned local = tile % (T::GROUPED_M * gn);
        tile = (first + local % rows) * gn + local / rows;
    }
    const int row = M <= BM ? 0 : (tile / gn) * BM;
    const int col = M <= BM ? tile * BN : (tile % gn) * BN;
    const int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));

    // Matrix and scale global-memory views, bounded for the final partial M tile.
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + row * stride_a,
        static_cast<unsigned>(valid_rows * stride_a));
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * stride_b,
        static_cast<unsigned>(BN * stride_b));
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
        static_cast<unsigned>((SC - 1) * args.stride_sfa + M));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
        static_cast<unsigned>((N / 128 - 1) * args.stride_sfb + SC));
    auto gc = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + static_cast<int64_t>(row) * stride_c + col,
        static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * 2));

    // A and scale LDS; the allocation is reused for the BF16 epilogue.
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto sa = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto ss = make_smem(arena);

    // MMA accumulators and the B register ring.
    using Base = decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> c{};
    array<array<array<typename Base::vtype_b, NR>, KS>, BS> bregs;

    // B global memory -> VGPR ring.
    auto issue_b = [&](auto bi, int kt) {
        constexpr int set = decltype(bi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int br = wn * (BN / WaveColumns) + n * 16;
                const int off = ks * 2048 + kg * 256 + lane16 * 16;
                set_slice(bregs[set][ks][n],
                    load<16>(gb, off, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 0_I, 16_I);
                set_slice(bregs[set][ks][n],
                    load<16>(gb, off + 1024, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 16_I, 32_I);
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    // A global memory -> LDS ring. Keep the async-LDS builtin and asyncmark paired;
    // opus::async_load uses buffer_load_lds with different compiler wait tracking.
    auto issue_a = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int lin = (i * T::BLOCK_SIZE + tid) * 16;
            const int r = lin / BK;
            const int kc = (lin % BK) ^ ((r % (BK / 16)) * 16);
            auto* dst = reinterpret_cast<OPUS_LDS_ADDR void*>(
                reinterpret_cast<__UINTPTR_TYPE__>(
                    arena + slot * T::STAGE_BYTES + (i * T::BLOCK_SIZE + wave * 64) * 16));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
                ga.cached_rsrc, dst, 16, r * stride_a + kc, kt * BK, 0, T::CACHECTL_A);
        });
        __builtin_amdgcn_asyncmark();
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue =====
    // Prefetch B and A, load resident scales, then fill the A ring.
    static_for<T::INITIAL_B>([&](auto i) { issue_b(i, decltype(i)::value); });
    issue_a(0, 0);
    static_for<T::SA_COPIES>([&](auto ii) {
        constexpr int i = decltype(ii)::value;
        const int index = (i * T::BLOCK_SIZE + tid) * 16;
        const int sr = index % T::SA_ROWS;
        const int scol = index / T::SA_ROWS;
        const int sc = scol < SC ? scol : SC - 1;
        int source = sr < BM && row + sr < M ? sc * args.stride_sfa + row + sr : (SC - 1) * args.stride_sfa + M;
        asm volatile("" : "+v"(source));
        auto* dst = reinterpret_cast<OPUS_LDS_ADDR void*>(
            reinterpret_cast<__UINTPTR_TYPE__>(arena + T::OFF_SA + (i * T::BLOCK_SIZE + wave * 64) * 16));
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc, dst, 16, source, 0, 0, 0);
    });
    auto* sbdst = reinterpret_cast<OPUS_LDS_ADDR void*>(
        reinterpret_cast<__UINTPTR_TYPE__>(arena + T::OFF_SB + wave * 64 * 4));
    if constexpr (T::SB_SLOTS) {
        const int index = tid % (T::SB_ROWS * SC);
        const int sr = index / SC;
        const int sc = index % SC;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            gsb.cached_rsrc, sbdst, 1, (col / 128 + sr) * args.stride_sfb + sc, 0, 0, 0);
    } else {
        const int index = tid % (T::SB_ROWS * (SC / 4));
        const int sr = index / (SC / 4);
        const int word = index % (SC / 4);
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            gsb.cached_rsrc, sbdst, 4, (col / 128 + sr) * args.stride_sfb + word * 4, 0, 0, 0);
    }
    static_for<T::INITIAL_A - 1>([&](auto i) { issue_a(decltype(i)::value + 1, decltype(i)::value + 1); });

    // ===== Main loop =====
    // Wait for the A slot, read operands, refill, and compute.
    auto step = [&](int t, int slot, auto bi, auto wai, auto wbi, auto fai, auto fbi) {
        constexpr int set = decltype(bi)::value;
        constexpr int wa = decltype(wai)::value;
        constexpr int wb = decltype(wbi)::value;
        constexpr bool future_a = decltype(fai)::value;
        constexpr bool future_b = decltype(fbi)::value;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        opus::sync_threads();
        if constexpr (future_b)
            issue_b(number<(set + BD) % BS>{}, t + BD);
        array<array<typename Base::vtype_a, MR>, KS> af;
        array<array<unsigned, MR>, KS> as;
        array<array<unsigned, NR>, KS> bs;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            const int scale_col = t * KS + ks;
            static_for<NR>([&](auto ni) {
                constexpr int nr = decltype(ni)::value;
                const int group_n = (wn * (BN / WaveColumns) + nr * 16) / 128;
                bs[ks][nr] = load<1>(ss, T::OFF_SB + (group_n * SC + scale_col) * (T::SB_SLOTS ? 4 : 1))[0];
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int ar = m * 16 + lane16;
                const int lo = slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg) ^ (ar % (BK / 16))) * 16);
                const int hi = slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg + 4) ^ (ar % (BK / 16))) * 16);
                if constexpr (T::PIN_A) {
                    array<int, 2> offsets;
                    offsets[0] = lo;
                    offsets[1] = hi;
                    opus_gemm_prefetch_b_vgpr_detail::load_operand_fragment_pinned<16, 0>(
                        number<(ks * MR + m) * 8>{}, af[ks][m], sa, offsets);
                } else {
                    set_slice(af[ks][m], load<16>(sa, lo), 0_I, 16_I);
                    set_slice(af[ks][m], load<16>(sa, hi), 16_I, 32_I);
                }
                as[ks][m] = load<1>(ss, T::OFF_SA + scale_col * T::SA_ROWS + ar)[0];
            });
        });
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (future_a)
            issue_a(slot == 0 ? NB - 1 : slot - 1, t + NB - 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wb > 63 ? 63 : wb)>{});
        s_waitcnt_lgkmcnt(0_I);

        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                static_for<NR>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    constexpr int ci = m * NR + n;
                    if constexpr (T::PIN_C) {
                        // Pin C/D to the same AGPRs for in-place accumulation.
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                        c[ci] = c[ci];
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                        c[ci] = Base{}(af[ks][m], bregs[set][ks][n], c[ci], static_cast<int>(as[ks][m]), static_cast<int>(bs[ks][n]), 0_I, 0_I);
                    } else {
                        c[ci] = Base{}(af[ks][m], bregs[set][ks][n], c[ci], static_cast<int>(as[ks][m]), static_cast<int>(bs[ks][n]), 0_I, 0_I);
                    }
                });
            });
        });
        if constexpr (!T::PIN_C)
            static_for<MR * NR>([&](auto ci) {
                auto& v = c[decltype(ci)::value];
                asm volatile("" : "+v"(v));
            });
        __builtin_amdgcn_sched_barrier(0);
    };

    // Run the initial steps, complete ring periods, then drain the remaining panels.
    constexpr int START = T::STEADY < NB - 1 ? T::STEADY : NB - 1;
    static_for<START>([&](auto ti) {
        constexpr int t = decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<wait_a<T>(t)>{}, number<wait_b<T>(t)>{},
            number<true>{}, number<(t + BD < P)>{});
    });
    constexpr int PHASES = []() constexpr {
        int a = NB;
        int b = BS;
        while (b) {
            int r = a % b;
            a = b;
            b = r;
        }
        return NB / a * BS;
    }();
    constexpr int GROUPS = (T::STEADY - START) / PHASES;
    constexpr int REGULAR_END = START + GROUPS * PHASES;
    if constexpr (GROUPS > 0) {
#pragma clang loop unroll(disable)
        for (int group = 0; group < GROUPS; ++group) {
            static_for<PHASES>([&](auto phase) {
                constexpr int ph = decltype(phase)::value;
                constexpr int t0 = START + ph;
                step(START + group * PHASES + ph, t0 % NB, number<t0 % BS>{},
                    number<wait_a<T>(t0)>{}, number<wait_b<T>(t0)>{}, number<true>{}, number<true>{});
            });
        }
    }
    // ===== Epilogue =====
    static_for<P - REGULAR_END>([&](auto ti) {
        constexpr int t = REGULAR_END + decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<wait_a<T>(t)>{}, number<wait_b<T>(t)>{}, number<(t + NB - 1 < P)>{},
            number<(t + BD < P)>{});
    });

    // ===== Output writeback =====
    // Pack BF16 output directly or stage contiguous stripes through LDS.
    if constexpr (T::OUTPUT == 2) {
        opus_gemm_prefetch_b_vgpr_detail::store_mfma16_packed<T, true>(c, gc, lane, 0, wn, stride_c);
    } else {
        auto sc = make_smem(reinterpret_cast<bf16_t*>(arena));
        constexpr int MAX_ROWS = BM;
        constexpr int C_ROWS = output_rows(BM, BN, T::PIPELINE_BYTES);
        static_for<(MAX_ROWS + C_ROWS - 1) / C_ROWS>([&](auto parti) {
            constexpr int part = decltype(parti)::value;
            constexpr int PART_ROWS = MAX_ROWS - part * C_ROWS < C_ROWS ? MAX_ROWS - part * C_ROWS : C_ROWS;
            s_waitcnt_lgkmcnt(0_I);
            opus::sync_threads();
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                if constexpr (m * 16 / C_ROWS == part && m * 16 < MAX_ROWS)
                    static_for<NR>([&](auto ni) {
                        constexpr int n = decltype(ni)::value;
                        constexpr int ci = m * NR + n;
                        const int rr = m * 16 + lane16;
                        const int nn = wn * (BN / WaveColumns) + n * 16 + kg * 4;
                        store<4>(sc, cast<bf16_t>(c[ci]), (rr - part * C_ROWS) * (BN + 8) + nn);
                    });
            });
            s_waitcnt_lgkmcnt(0_I);
            opus::sync_threads();
            constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            array<vector_t<bf16_t, 8>, CHUNKS> output;
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid;
                const int rr = index / (BN / 8);
                const int nn = index % (BN / 8) * 8;
                if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                    output[i] = load<8>(sc, rr * (BN + 8) + nn);
                else if (rr < PART_ROWS)
                    output[i] = load<8>(sc, rr * (BN + 8) + nn);
            });
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid;
                const int rr = index / (BN / 8);
                const int nn = index % (BN / 8) * 8;
                if (rr < PART_ROWS && part * C_ROWS + rr < valid_rows)
                    store<8>(gc, output[i], (part * C_ROWS + rr) * stride_c + nn, 0, number<T::STORE_CACHE>{});
            });
        });
    }
    s_waitcnt_vmcnt(0_I);
}

// Retain A operands across K iterations; publish and replace fragments
// after their last MFMA consumer. PublishM selects a legal publication point.
template<class Traits>
__device__ __forceinline__ void bulk_panel_overlap_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
    opus_gemm_prefetch_b_vgpr_detail::apply_fixed_args(args);
#endif
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using Index = unsigned;
    constexpr int WaveColumns = T::T_N;

    // Tile geometry and runtime strides.
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int NB = T::STAGES;
    constexpr int BD = T::B_AHEAD;
    constexpr int BS = T::B_SETS;
    constexpr int P = T::PANELS;
    constexpr int MR = T::MR;
    constexpr int NR = T::NR;
    constexpr int K = T::K;
    constexpr int SC = T::SCALE_COLS;
    constexpr int BK = T::B_K;
    constexpr int KS = BK / 128;

    const int M = args.m;
    const int N = args.n;
    const int stride_a = T::CONTIGUOUS ? K : args.stride_a;
    const int stride_b = T::CONTIGUOUS ? K : args.stride_b;
    const int stride_c = args.stride_c;

    static_assert((BK == 128 || BK == 256) && SC == K / 128);
    static_assert(T::T_M == 1 && T::T_N * 64 == T::BLOCK_SIZE);
    static_assert(T::MR * 16 == T::B_M && T::NR * T::T_N * 16 == T::B_N);

    // Tile and thread coordinates.
    const Index tid = thread_id_x();
    const Index lane = tid % 64;
    const Index lane16 = lane % 16;
    const Index kg = lane / 16;
    const Index wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const Index wn = wave;
    const unsigned pid = block_id_x();
    const unsigned gm = (M + BM - 1) / BM;
    const unsigned gn = N / BN;
    const unsigned total = gm * gn;
    unsigned tile = T::XCD_ORDER ? pid % 8 * ((total + 7) / 8) + pid / 8 : pid;
    if constexpr (T::XCD_ORDER)
        if (tile >= total)
            return;
    if constexpr (T::GROUPED_M > 0) {
        const unsigned first = tile / (T::GROUPED_M * gn) * T::GROUPED_M;
        const unsigned rows = gm - first < T::GROUPED_M ? gm - first : T::GROUPED_M;
        const unsigned local = tile % (T::GROUPED_M * gn);
        tile = (first + local % rows) * gn + local / rows;
    }
    const int row = M <= BM ? 0 : (tile / gn) * BM;
    const int col = M <= BM ? tile * BN : (tile % gn) * BN;
    const int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));

    // Matrix and scale global-memory views, bounded for the final partial M tile.
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + row * stride_a,
        static_cast<unsigned>(valid_rows * stride_a));
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * stride_b,
        static_cast<unsigned>(BN * stride_b));
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
        static_cast<unsigned>((SC - 1) * args.stride_sfa + M));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
        static_cast<unsigned>((N / 128 - 1) * args.stride_sfb + SC));
    auto gc = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + static_cast<int64_t>(row) * stride_c + col,
        static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * 2));

    // A and scale LDS; the allocation is reused for the BF16 epilogue.
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto sa = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto ss = make_smem(arena);

    // MMA accumulators and the B register ring.
    using Base = decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> c{};
    array<array<array<typename Base::vtype_b, NR>, KS>, BS> bregs;

    // B global memory -> VGPR ring.
    auto issue_b = [&](auto bi, int kt) {
        constexpr int set = decltype(bi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int br = wn * (BN / WaveColumns) + n * 16;
                const int off = ks * 2048 + kg * 256 + lane16 * 16;
                set_slice(bregs[set][ks][n],
                    load<16>(gb, off, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 0_I, 16_I);
                set_slice(bregs[set][ks][n],
                    load<16>(gb, off + 1024, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 16_I, 32_I);
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    // A global memory -> LDS ring. Keep the async-LDS builtin and asyncmark paired;
    // opus::async_load uses buffer_load_lds with different compiler wait tracking.
    auto issue_a = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int lin = (i * T::BLOCK_SIZE + tid) * 16;
            const int r = lin / BK;
            const int kc = (lin % BK) ^ ((r % (BK / 16)) * 16);
            auto* dst = reinterpret_cast<OPUS_LDS_ADDR void*>(
                reinterpret_cast<__UINTPTR_TYPE__>(
                    arena + slot * T::STAGE_BYTES + (i * T::BLOCK_SIZE + wave * 64) * 16));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
                ga.cached_rsrc, dst, 16, r * stride_a + kc, kt * BK, 0, T::CACHECTL_A);
        });
        __builtin_amdgcn_asyncmark();
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue =====
    // Prefetch B and A, load resident scales, then fill the A ring.
    static_for<T::INITIAL_B>([&](auto i) { issue_b(i, decltype(i)::value); });
    issue_a(0, 0);
    static_for<T::SA_COPIES>([&](auto ii) {
        constexpr int i = decltype(ii)::value;
        const int index = (i * T::BLOCK_SIZE + tid) * 16;
        const int sr = index % T::SA_ROWS;
        const int scol = index / T::SA_ROWS;
        const int sc = scol < SC ? scol : SC - 1;
        int source = sr < BM && row + sr < M ? sc * args.stride_sfa + row + sr : (SC - 1) * args.stride_sfa + M;
        asm volatile("" : "+v"(source));
        auto* dst = reinterpret_cast<OPUS_LDS_ADDR void*>(
            reinterpret_cast<__UINTPTR_TYPE__>(arena + T::OFF_SA + (i * T::BLOCK_SIZE + wave * 64) * 16));
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc, dst, 16, source, 0, 0, 0);
    });
    auto* sbdst = reinterpret_cast<OPUS_LDS_ADDR void*>(
        reinterpret_cast<__UINTPTR_TYPE__>(arena + T::OFF_SB + wave * 64 * 4));
    if constexpr (T::SB_SLOTS) {
        const int index = tid % (T::SB_ROWS * SC);
        const int sr = index / SC;
        const int sc = index % SC;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            gsb.cached_rsrc, sbdst, 1, (col / 128 + sr) * args.stride_sfb + sc, 0, 0, 0);
    } else {
        const int index = tid % (T::SB_ROWS * (SC / 4));
        const int sr = index / (SC / 4);
        const int word = index % (SC / 4);
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            gsb.cached_rsrc, sbdst, 4, (col / 128 + sr) * args.stride_sfb + word * 4, 0, 0, 0);
    }
    static_for<T::INITIAL_A - 1>([&](auto i) { issue_a(decltype(i)::value + 1, decltype(i)::value + 1); });

    // Retain current A operands while progressively replacing them with K+1.
    array<array<typename Base::vtype_a, MR>, KS> af;
    array<array<unsigned, MR>, KS> as;
    array<array<unsigned, NR>, KS> bs;
    auto read_a = [&](int t, int slot, auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            const int ar = m * 16 + lane16;
            const int lo = slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg) ^ (ar % (BK / 16))) * 16);
            const int hi = slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg + 4) ^ (ar % (BK / 16))) * 16);
            if constexpr (T::PIN_A) {
                array<int, 2> offsets;
                offsets[0] = lo;
                offsets[1] = hi;
                opus_gemm_prefetch_b_vgpr_detail::load_operand_fragment_pinned<16, 0>(
                    number<(ks * MR + m) * 8>{}, af[ks][m], sa, offsets);
            } else {
                set_slice(af[ks][m], load<16>(sa, lo), 0_I, 16_I);
                set_slice(af[ks][m], load<16>(sa, hi), 16_I, 32_I);
            }
            as[ks][m] = load<1>(ss, T::OFF_SA + (t * KS + ks) * T::SA_ROWS + ar)[0];
        });
    };
    auto read_bs = [&](int t) {
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int group_n = (wn * (BN / WaveColumns) + n * 16) / 128;
                bs[ks][n] = load<1>(ss, T::OFF_SB + (group_n * SC + t * KS + ks) * (T::SB_SLOTS ? 4 : 1))[0];
            });
        });
    };
    s_waitcnt_vmcnt(number<(wait_a<T>(0) > 63 ? 63 : wait_a<T>(0))>{});
    s_waitcnt_lgkmcnt(0_I);
    opus::sync_threads();
    static_for<MR>([&](auto mi) { read_a(0, 0, mi); });
    read_bs(0);
    s_waitcnt_lgkmcnt(0_I);

    // ===== Main loop =====
    // Consume each M repeat, publish K+1, then replace consumed operands.
    auto step = [&](int t, int slot, auto bi, auto /*wai*/, auto wbi, auto fai, auto fbi, auto next_ai) {
        constexpr int set = decltype(bi)::value;
        constexpr bool future_a = decltype(fai)::value;
        constexpr bool future_b = decltype(fbi)::value;
        // wait_a/wait_b include the next async A copy. This schedule issues
        // that copy only after publishing K+1, so subtract it at both waits.
        constexpr int skipped_a = future_a ? T::A_COPIES : 0;
        constexpr int wb = decltype(wbi)::value - skipped_a;
        constexpr int wnxt = decltype(next_ai)::value - skipped_a;
        if constexpr (future_b)
            issue_b(number<(set + BD) % BS>{}, t + BD);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wb < 0 ? 0 : wb > 63 ? 63 : wb)>{});
        s_waitcnt_lgkmcnt(0_I);

        static_for<MR>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<KS>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                static_for<NR>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    constexpr int ci = m * NR + n;
                    if constexpr (T::PIN_C) {
                        // Pin C/D to the same AGPRs for in-place accumulation.
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                        c[ci] = c[ci];
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                        c[ci] = Base{}(af[ks][m], bregs[set][ks][n], c[ci], static_cast<int>(as[ks][m]), static_cast<int>(bs[ks][n]), 0_I, 0_I);
                    } else {
                        c[ci] = Base{}(af[ks][m], bregs[set][ks][n], c[ci], static_cast<int>(as[ks][m]), static_cast<int>(bs[ks][n]), 0_I, 0_I);
                    }
                });
            });
            if (t + 1 < P) {
                if constexpr (m == T::PUBLISH_M) {
                    __builtin_amdgcn_sched_barrier(0);
                    s_waitcnt_vmcnt(number<(wnxt < 0 ? 0 : wnxt > 63 ? 63 : wnxt)>{});
                    s_waitcnt_lgkmcnt(0_I);
                    opus::sync_threads();
                    if constexpr (future_a)
                        issue_a(slot == 0 ? NB - 1 : slot - 1, t + NB - 1);
                    static_for<T::PUBLISH_M>([&](auto earlier) { read_a(t + 1, (slot + 1) % NB, earlier); });
                }
                if constexpr (m >= T::PUBLISH_M)
                    read_a(t + 1, (slot + 1) % NB, mi);
            }
        });
        if (t + 1 < P)
            read_bs(t + 1);
        s_waitcnt_lgkmcnt(0_I);
        if constexpr (!T::PIN_C)
            static_for<MR * NR>([&](auto ci) {
                auto& v = c[decltype(ci)::value];
                asm volatile("" : "+v"(v));
            });
        __builtin_amdgcn_sched_barrier(0);
    };

    // Run the initial steps, complete ring periods, then drain the remaining panels.
    constexpr int START = T::STEADY < NB - 1 ? T::STEADY : NB - 1;
    static_for<START>([&](auto ti) {
        constexpr int t = decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<wait_a<T>(t)>{}, number<wait_b<T>(t)>{},
            number<true>{}, number<(t + BD < P)>{},
            number<wait_a<T>(t + 1)>{});
    });
    constexpr int PHASES = []() constexpr {
        int a = NB;
        int b = BS;
        while (b) {
            int r = a % b;
            a = b;
            b = r;
        }
        return NB / a * BS;
    }();
    constexpr int GROUPS = (T::STEADY - START) / PHASES;
    constexpr int REGULAR_END = START + GROUPS * PHASES;
    if constexpr (GROUPS > 0) {
#pragma clang loop unroll(disable)
        for (int group = 0; group < GROUPS; ++group) {
            static_for<PHASES>([&](auto phase) {
                constexpr int ph = decltype(phase)::value;
                constexpr int t0 = START + ph;
                step(START + group * PHASES + ph, t0 % NB, number<t0 % BS>{},
                    number<wait_a<T>(t0)>{}, number<wait_b<T>(t0)>{},
                    number<true>{}, number<true>{}, number<wait_a<T>(t0 + 1)>{});
            });
        }
    }
    // ===== Epilogue =====
    static_for<P - REGULAR_END>([&](auto ti) {
        constexpr int t = REGULAR_END + decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<wait_a<T>(t)>{}, number<wait_b<T>(t)>{}, number<(t + NB - 1 < P)>{},
            number<(t + BD < P)>{}, number<wait_a<T>(t + 1)>{});
    });

    // ===== Output writeback =====
    // Pack BF16 output directly or stage contiguous stripes through LDS.
    if constexpr (T::OUTPUT == 2) {
        opus_gemm_prefetch_b_vgpr_detail::store_mfma16_packed<T, true>(c, gc, lane, 0, wn, stride_c);
    } else {
        auto sc = make_smem(reinterpret_cast<bf16_t*>(arena));
        constexpr int MAX_ROWS = BM;
        constexpr int C_ROWS = output_rows(BM, BN, T::PIPELINE_BYTES);
        static_for<(MAX_ROWS + C_ROWS - 1) / C_ROWS>([&](auto parti) {
            constexpr int part = decltype(parti)::value;
            constexpr int PART_ROWS = MAX_ROWS - part * C_ROWS < C_ROWS ? MAX_ROWS - part * C_ROWS : C_ROWS;
            s_waitcnt_lgkmcnt(0_I);
            opus::sync_threads();
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                if constexpr (m * 16 / C_ROWS == part && m * 16 < MAX_ROWS)
                    static_for<NR>([&](auto ni) {
                        constexpr int n = decltype(ni)::value;
                        constexpr int ci = m * NR + n;
                        const int rr = m * 16 + lane16;
                        const int nn = wn * (BN / WaveColumns) + n * 16 + kg * 4;
                        store<4>(sc, cast<bf16_t>(c[ci]), (rr - part * C_ROWS) * (BN + 8) + nn);
                    });
            });
            s_waitcnt_lgkmcnt(0_I);
            opus::sync_threads();
            constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            array<vector_t<bf16_t, 8>, CHUNKS> output;
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid;
                const int rr = index / (BN / 8);
                const int nn = index % (BN / 8) * 8;
                if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                    output[i] = load<8>(sc, rr * (BN + 8) + nn);
                else if (rr < PART_ROWS)
                    output[i] = load<8>(sc, rr * (BN + 8) + nn);
            });
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid;
                const int rr = index / (BN / 8);
                const int nn = index % (BN / 8) * 8;
                if (rr < PART_ROWS && part * C_ROWS + rr < valid_rows)
                    store<8>(gc, output[i], (part * C_ROWS + rr) * stride_c + nn, 0, number<T::STORE_CACHE>{});
            });
        });
    }
    s_waitcnt_vmcnt(0_I);
}

} // namespace opus_gemm_prefetch_b_vgpr_detail
#endif

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_body<Traits>(args);
#endif
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_prefetch_b_vgpr_detail::bulk_panel_overlap_body<Traits>(args);
#endif
}

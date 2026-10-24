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

// Stream individual B fragments across N, reusing each fragment for all M repeats.
template<class Traits, int PublishM = -1>
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
    array<typename Base::vtype_b, BS> bregs;

    auto issue_b = [&](auto slot_i, auto fragment_i, int kt) {
        constexpr int set = decltype(slot_i)::value;
        constexpr int fragment = decltype(fragment_i)::value;
        constexpr int ks = fragment / NR;
        constexpr int n = fragment % NR;
        const int br = wn * (BN / WaveColumns) + n * 16;
        const int off = ks * 2048 + kg * 256 + lane16 * 16;
        set_slice(bregs[set],
            load<16>(gb, off, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 0_I, 16_I);
        set_slice(bregs[set],
            load<16>(gb, off + 1024, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 16_I, 32_I);
    };
    auto issue_a = [&](int slot, int kt) {
        __builtin_amdgcn_sched_barrier(0);
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int lin = (i * T::BLOCK_SIZE + tid) * 16;
            const int r = lin / BK;
            const int kc = (lin % BK) ^ ((r % (BK / 16)) * 16);
            async_load<16>(ga, arena + slot * T::STAGE_BYTES + (i * T::BLOCK_SIZE + wave * 64) * 16,
                r * stride_a + kc, kt * BK, 0_I, number<T::CACHECTL_A>{});
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue =====
    static_for<T::SA_COPIES>([&](auto ii) {
        constexpr int i = decltype(ii)::value;
        const int index = (i * T::BLOCK_SIZE + tid) * 16;
        const int sr = index % T::SA_ROWS;
        const int scol = index / T::SA_ROWS;
        const int sc = scol < SC ? scol : SC - 1;
        int source = sr < BM && row + sr < M ? sc * args.stride_sfa + row + sr : (SC - 1) * args.stride_sfa + M;
        async_load<16>(gsa, arena + T::OFF_SA + (i * T::BLOCK_SIZE + wave * 64) * 16, source);
    });
    auto* sbdst = arena + T::OFF_SB + wave * 64 * 4;
    if constexpr (T::SB_SLOTS) {
        const int index = tid % (T::SB_ROWS * SC);
        const int sr = index / SC;
        const int sc = index % SC;
        async_load<1>(gsb, sbdst, (col / 128 + sr) * args.stride_sfb + sc);
    } else {
        const int index = tid % (T::SB_ROWS * (SC / 4));
        const int sr = index / (SC / 4);
        const int word = index % (SC / 4);
        async_load<4>(gsb, sbdst, (col / 128 + sr) * args.stride_sfb + word * 4);
    }
    static_for<T::INITIAL_A>([&](auto i) { issue_a(decltype(i)::value, decltype(i)::value); });

    constexpr int F = KS * NR;
    static_for<BD>([&](auto i) {
        constexpr int fragment = decltype(i)::value;
        issue_b(number<fragment % BS>{}, number<fragment % F>{}, fragment / F);
    });

    // ===== Main loop =====
    auto consume_panel = [&](int t, auto phase_i) {
        constexpr int phase = decltype(phase_i)::value;
        constexpr bool future_a = phase + T::INITIAL_A < P;
        constexpr int issued_b = (phase * F + BD < P * F ? phase * F + BD : P * F);
        constexpr int before_a = BD + (phase - T::INITIAL_A) * F + 1;
        constexpr int newer_b_a = phase < T::INITIAL_A ? issued_b : (issued_b > before_a ? issued_b - before_a : 0);
        constexpr int newer_a = phase < T::INITIAL_A ? T::INITIAL_A - phase - 1 : (P - phase - 1 < T::INITIAL_A - 1 ? P - phase - 1 : T::INITIAL_A - 1);
        constexpr int wa = newer_b_a * 2 + newer_a * T::A_COPIES;
        const int slot = T::ROTATE_B ? t % NB : phase % NB;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        opus::sync_threads();

        array<array<typename Base::vtype_a, MR>, T::ROTATE_B ? 1 : KS> af;
        array<array<unsigned, (MR + 3) / 4>, KS> as{};
        array<unsigned, KS> bs{};
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            const int scale_col = t * KS + ks;
            static_for<(BN / WaveColumns + 127) / 128>([&](auto group_i) {
                constexpr int group = decltype(group_i)::value;
                const int group_n = wn * (BN / WaveColumns) / 128 + group;
                const unsigned scale = load<1>(ss, T::OFF_SB + (group_n * SC + scale_col) * (T::SB_SLOTS ? 4 : 1))[0];
                bs[ks] |= scale << (group * 8);
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int ar = m * 16 + lane16;
                const unsigned scale = load<1>(ss, T::OFF_SA + scale_col * T::SA_ROWS + ar)[0];
                as[ks][m / 4] |= scale << ((m % 4) * 8);
            });
        });
        auto read_a = [&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int ar = m * 16 + lane16;
                const int lo = slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg) ^ (ar % (BK / 16))) * 16);
                const int hi = slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg + 4) ^ (ar % (BK / 16))) * 16);
                if constexpr (T::PIN_A) {
                    array<int, 2> offsets;
                    offsets[0] = lo;
                    offsets[1] = hi;
                    load_operand_fragment_pinned<16, 0>(number<(ks * MR + m) * 8>{}, af[T::ROTATE_B ? 0 : ks][m], sa, offsets);
                } else {
                    set_slice(af[T::ROTATE_B ? 0 : ks][m], load<16>(sa, lo), 0_I, 16_I);
                    set_slice(af[T::ROTATE_B ? 0 : ks][m], load<16>(sa, hi), 16_I, 32_I);
                }
            });
        };
        if constexpr (!T::ROTATE_B)
            static_for<KS>([&](auto ki) { read_a(ki); });
        static_for<KS * NR>([&](auto fragment_i) {
            constexpr int fragment = decltype(fragment_i)::value;
            constexpr int ks = fragment / NR;
            constexpr int n = fragment % NR;
            constexpr int absolute = phase * F + fragment;
            if constexpr (T::ROTATE_B && n == 0)
                read_a(number<ks>{});
            if constexpr (absolute + BD < P * F)
                issue_b(number<((T::ROTATE_B ? fragment : absolute) + BD) % BS>{}, number<(fragment + BD) % F>{}, t + (fragment + BD) / F);
            constexpr int newer_b = P * F - absolute - 1 < BD ? P * F - absolute - 1 : BD;
            constexpr int first_a = absolute < BD ? 0 : (absolute - BD + F - 1) / F;
            constexpr int last_a = (phase - (fragment == 0 ? 1 : 0) < P - T::INITIAL_A - 1)
                ? phase - (fragment == 0 ? 1 : 0) : P - T::INITIAL_A - 1;
            constexpr int pending_a = last_a >= first_a ? (last_a - first_a + 1) * T::A_COPIES : 0;
            constexpr int wait = newer_b * 2 + pending_a;
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_vmcnt(number<(wait > 63 ? 63 : wait)>{});
            if constexpr (fragment == 0 || (T::ROTATE_B && n == 0))
                s_waitcnt_lgkmcnt(0_I);
            if constexpr (future_a && fragment == 0 && PublishM < 0)
                issue_a((T::ROTATE_B ? t + NB - 1 : phase + NB - 1) % NB, t + NB - 1);

            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                constexpr int ci = m * NR + n;
                if constexpr (T::PIN_C) {
                    [[clang::amdgpu_pin_agpr(128 + ci * 4)]] c[ci] = c[ci];
                    [[clang::amdgpu_pin_agpr(128 + ci * 4)]]
                    c[ci] = Base{}(af[T::ROTATE_B ? 0 : ks][m], bregs[(T::ROTATE_B ? fragment : phase * F + fragment) % BS], c[ci],
                        static_cast<int>(as[ks][m / 4]), static_cast<int>(bs[ks]), number<m % 4>{}, number<n / 8>{});
                } else {
                    c[ci] = Base{}(af[T::ROTATE_B ? 0 : ks][m], bregs[(T::ROTATE_B ? fragment : phase * F + fragment) % BS], c[ci],
                        static_cast<int>(as[ks][m / 4]), static_cast<int>(bs[ks]), number<m % 4>{}, number<n / 8>{});
                }
                if constexpr (future_a && fragment == 0 && m == PublishM)
                    issue_a((T::ROTATE_B ? t + NB - 1 : phase + NB - 1) % NB, t + NB - 1);
            });
        });
        if constexpr (T::ROTATE_B) {
            // Advance the bounded queue without expanding its ring period.
            auto next_b = bregs;
            static_for<BS>([&](auto i) {
                constexpr int slot_b = decltype(i)::value;
                next_b[slot_b] = bregs[(slot_b + F) % BS];
            });
            bregs = next_b;
        }
    };
    constexpr int STEADY = P - T::INITIAL_A;
    constexpr int START = T::INITIAL_A < STEADY ? T::INITIAL_A : STEADY;
    static_for<START>([&](auto i) {
        constexpr int t = decltype(i)::value;
        consume_panel(t, number<t>{});
    });
    constexpr int PERIOD = []() constexpr {
        if constexpr (T::ROTATE_B) return 1;
        int a = NB, b = BS;
        while (b) { int r = a % b; a = b; b = r; }
        return NB / a * BS;
    }();
    constexpr int GROUPS = (STEADY - START) / PERIOD;
    constexpr int END = START + GROUPS * PERIOD;
    if constexpr (GROUPS > 0) {
#pragma nounroll
        for (int group = 0; group < GROUPS; ++group) {
            static_for<PERIOD>([&](auto i) {
                constexpr int phase = START + decltype(i)::value;
                consume_panel(phase + group * PERIOD, number<phase>{});
            });
        }
    }
    static_for<P - END>([&](auto i) {
        constexpr int t = END + decltype(i)::value;
        consume_panel(t, number<t>{});
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

template<class Traits>
__device__ __forceinline__ void bulk_panel_overlap_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    bulk_panel_body<Traits, Traits::PUBLISH_M>(args);
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

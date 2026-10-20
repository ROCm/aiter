// SPDX-License-Identifier: Apache-2.0
// Layouts originate from the FlyDSL 8-wave adaptation; the OPUS main loop
// streams N fragments with uniform block barriers and two B register slots.
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_interleave_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_bpreshuffle_wave_interleave_layout {

// The XOR exchanges 16-byte K vectors inside each group of 16 A rows.
__device__ inline int a_offset(int row, int col) {
    const int offset = row * 128 + col;
    return offset ^ (((offset % 2048) >> 8) << 4);
}

__device__ inline int b_offset(int row, int col) {
    return (row / 8) * 1024 + (row % 8) * 16 + (col / 16) * 128;
}

template<bool Preshuffled, class Fragment, class Mem>
__device__ inline Fragment read_operand_fragment(Mem& smem, int base, int row, int lane) {
    const int col = (lane / 16) * 16;
    const int lo = Preshuffled ? b_offset(row, col) : a_offset(row, col);
    const int hi = Preshuffled ? b_offset(row, col + 64) : a_offset(row, col + 64);
    Fragment result;
    opus::set_slice(result, opus::load<16>(smem, base + lo), opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, opus::load<16>(smem, base + hi), opus::number<16>{}, opus::number<32>{});
    return result;
}

} // namespace opus_gemm_bpreshuffle_wave_interleave_layout

template<class Traits>
__device__ __forceinline__ void opus_gemm_bpreshuffle_wave_interleave_body(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using namespace opus_gemm_bpreshuffle_wave_interleave_layout;
    using T = Traits;
    using D_A = fp8_t;
    using D_C = bf16_t;

    if constexpr (T::FIXED_K > 0) {
        kargs.k = T::FIXED_K;
        kargs.stride_a = T::FIXED_K;
        kargs.stride_b = T::FIXED_K;
        kargs.stride_sfb = T::FIXED_K / 128;
    }

    // Tile and thread coordinates.
    const int lane = thread_id_x() % 64;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wave_m = wave / T::T_N;
    const int wave_n = wave % T::T_N;
    const int gn = kargs.n / T::B_N;
    const int gm = (kargs.m + T::B_M - 1) / T::B_M;
    int block_m = block_id_y();
    int block_n = block_id_x();
    if constexpr (T::GROUP_M > 0) {
        int linear = block_m * gn + block_n;
        if constexpr (T::XCD_ORDER) {
            const int xcd = linear % 8;
            const int intra = linear / 8;
            const int base = gm * gn / 8;
            const int extra = gm * gn % 8;
            linear = xcd * base + (xcd < extra ? xcd : extra) + intra;
        }
        const int first = linear / (T::GROUP_M * gn) * T::GROUP_M;
        const int rows = gm - first < T::GROUP_M ? gm - first : T::GROUP_M;
        const int local = linear % (T::GROUP_M * gn);
        block_m = first + local % rows;
        block_n = local / rows;
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch = block_id_z();
    const int loops = (T::FIXED_K ? T::FIXED_K : kargs.k) / T::K_STAGE;
    const int valid_rows = kargs.m - row < T::B_M ? kargs.m - row : T::B_M;

    // Matrix and scale global-memory views, bounded for the final partial M tile.
    auto g_a = make_gmem(
        reinterpret_cast<const D_A*>(kargs.ptr_a) + batch * kargs.stride_a_batch + row * kargs.stride_a,
        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_b) +
        batch * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(
        reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(batch) * kargs.stride_c_batch +
            static_cast<int64_t>(row) * kargs.stride_c + col,
        static_cast<unsigned>((static_cast<int64_t>(valid_rows - 1) * kargs.stride_c + T::B_N) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa) + batch * kargs.stride_sfa_batch + row,
        static_cast<unsigned>(kargs.stride_sfa_batch - row));
    auto g_sfb = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb) +
        batch * kargs.stride_sfb_batch + (col / 128) * kargs.stride_sfb);

    // Matrix and scale LDS; matrix storage is reused for the BF16 epilogue.
    alignas(16) __shared__ char storage[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(storage));
    auto s_b = make_smem(reinterpret_cast<D_A*>(storage + 4 * T::A_SLOT));
    auto s_sfa = make_smem(reinterpret_cast<unsigned char*>(storage + T::MATRIX_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<unsigned short*>(storage + T::MATRIX_BYTES + T::SFA_BYTES));
    auto s_c = make_smem(reinterpret_cast<D_C*>(storage));

    // Matrix global -> LDS offsets. Each wave copies contiguous 16-byte slots.
    array<int, T::A_STEPS> ga;
    array<int, T::B_STEPS> gb;
    static_for<T::A_STEPS>([&](auto i) {
        const int r = lane / 8 + wave * 8 + decltype(i)::value * (T::BLOCK_SIZE / 8);
        const int c = a_offset(r, lane % 8 * 16) % 128;
        ga[decltype(i)::value] = r * kargs.stride_a + c;
    });
    static_for<T::B_STEPS>([&](auto i) {
        const int r = lane % 8 + wave * 8 + decltype(i)::value * (T::BLOCK_SIZE / 8);
        const int c = lane / 8 * 16;
        gb[decltype(i)::value] = (r / 16) * kargs.stride_b * 16 + (r % 16) * 16 +
            (c / 64) * 1024 + ((c % 64) / 16) * 256;
    });
    auto load_a = [&](auto slot_i, int k) {
        constexpr int slot = decltype(slot_i)::value;
        static_for<T::A_STEPS>([&](auto i) {
            constexpr int step = decltype(i)::value;
            async_load<16>(g_a, s_a.ptr + slot * T::A_SLOT + wave * 1024 + step * (T::BLOCK_SIZE * 16),
                ga[step], (slot % 2) * T::HALF_M * kargs.stride_a + k * 128, number<0>{}, number<T::CACHE_A>{});
        });
    };
    auto load_b = [&](auto slot_i, int k) {
        constexpr int slot = decltype(slot_i)::value;
        static_for<T::B_STEPS>([&](auto i) {
            constexpr int step = decltype(i)::value;
            async_load<16>(g_b, s_b.ptr + slot * T::B_SLOT + wave * 1024 + step * (T::BLOCK_SIZE * 16),
                gb[step], (slot % 2) * T::HALF_N * kargs.stride_b + k * 2048, number<0>{}, number<T::CACHE_B>{});
        });
    };

    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_A, fp32_t>(
        seq<T::E_M, T::E_N, 1>{},
        seq<1, 1, 1>{},
        seq<16, 16, 128>{},
        mfma_adaptor{});
    using MMA = typename decltype(mma)::MMA;
    array<array<typename MMA::vtype_a, T::E_M>, 2> v_a;
    array<typename MMA::vtype_b, 2 * T::E_N> v_b;
    array<typename MMA::vtype_c, 4 * T::E_M * T::E_N> v_c{};
    array<unsigned, 2> first_scale_a{};
    array<unsigned, 2> scale_a{};
    unsigned scale_b = 0;

    // Keep K=0 scales in registers. The remaining 127 A columns plus packed
    // B scales fit exactly beside the matrix slots at the maximum K=16384.
    static_for<2>([&](auto half_i) {
        constexpr int half = decltype(half_i)::value;
        static_for<T::E_M>([&](auto i) {
            const int r = half * T::HALF_M + wave_m * T::E_M * 16 + decltype(i)::value * 16 + lane % 16;
            first_scale_a[half] |= static_cast<unsigned>(load<1>(g_sfa, r)[0]) << (decltype(i)::value * 8);
        });
    });
    constexpr int bytes_per_copy = T::BLOCK_SIZE * 16;
    static_for<(T::SFA_BYTES + bytes_per_copy - 1) / bytes_per_copy>([&](auto i) {
        constexpr int begin = decltype(i)::value * bytes_per_copy;
        const int linear = begin + (wave * 64 + lane) * 16;
        const int group = linear / T::B_M;
        if (group + 1 < loops && group < T::SCALE_GROUPS - 1)
            async_load<16>(g_sfa, s_sfa.ptr + begin + wave * 1024,
                (group + 1) * kargs.stride_sfa + linear % T::B_M);
    });
    const int sf_group = wave * 64 + lane;
    if (sf_group < loops) {
        const unsigned lo = load<1>(g_sfb, sf_group)[0];
        const unsigned hi = load<1>(g_sfb, kargs.stride_sfb + sf_group)[0];
        store<1>(s_sfb, vector_t<unsigned short, 1>{static_cast<unsigned short>(lo | (hi << 8))}, sf_group);
    }
    if constexpr (T::PACKED_SCALE) {
        // Repack four M16 scale bytes into a dword once, before accumulators
        // become live. A main-loop scale read then feeds four MFMAs directly.
        constexpr int words_per_group = T::B_M / 4;
        constexpr int rounds = (T::SCALE_GROUPS - 1) * words_per_group / T::BLOCK_SIZE + 1;
        array<unsigned, rounds> packed{};
        s_waitcnt_vmcnt(number<0>{});
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
        static_for<rounds>([&](auto i) {
            constexpr int step = decltype(i)::value;
            const int word = step * T::BLOCK_SIZE + wave * 64 + lane;
            const int group = word / words_per_group;
            const int local = word % words_per_group;
            const int r = local / 16 * 64 + local % 16;
            if (group + 1 < loops && group < T::SCALE_GROUPS - 1)
                static_for<4>([&](auto b) {
                    packed[step] |=
                        static_cast<unsigned>(load<1>(s_sfa, group * T::B_M + r + decltype(b)::value * 16)[0]) <<
                        (decltype(b)::value * 8);
                });
        });
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
        auto s_packed = make_smem(reinterpret_cast<unsigned*>(s_sfa.ptr));
        static_for<rounds>([&](auto i) {
            constexpr int step = decltype(i)::value;
            const int word = step * T::BLOCK_SIZE + wave * 64 + lane;
            const int group = word / words_per_group;
            if (group + 1 < loops && group < T::SCALE_GROUPS - 1)
                store<1>(s_packed, vector_t<unsigned, 1>{packed[step]}, word);
        });
        s_waitcnt_lgkmcnt(number<0>{});
        sync_threads();
    }

    auto read_scale_a = [&](int k, auto half_i) {
        constexpr int half = decltype(half_i)::value;
        unsigned result = 0;
        if (k == 0)
            return first_scale_a[half];
        if constexpr (T::PACKED_SCALE) {
            const int r = half * T::HALF_M + wave_m * T::E_M * 16 + lane % 16;
            auto s_packed = make_smem(reinterpret_cast<unsigned*>(s_sfa.ptr));
            result = load<1>(s_packed, (k - 1) * (T::B_M / 4) + r / 64 * 16 + r % 16)[0];
        } else {
            static_for<T::E_M>([&](auto i) {
                const int r = half * T::HALF_M + wave_m * T::E_M * 16 + decltype(i)::value * 16 + lane % 16;
                result |= static_cast<unsigned>(load<1>(s_sfa, (k - 1) * T::B_M + r)[0]) << (decltype(i)::value * 8);
            });
        }
        return result;
    };
    auto issue_tile = [&](auto stage_i, int k) {
        constexpr int first = decltype(stage_i)::value * 2;
        load_a(number<first>{}, k);
        load_a(number<first + 1>{}, k);
        load_b(number<first>{}, k);
        load_b(number<first + 1>{}, k);
    };
    auto read_b = [&](auto stage_i, auto n_i) {
        constexpr int n = decltype(n_i)::value;
        constexpr int half = n / T::E_N;
        constexpr int slot = decltype(stage_i)::value * 2 + half;
        const int r = (wave_n * T::E_N + n % T::E_N) * 16 + lane % 16;
        v_b[n] = read_operand_fragment<true, typename MMA::vtype_b>(
            s_b, slot * T::B_SLOT, r, lane);
    };
    auto consume_tile = [&](auto stage_i, int k) {
        constexpr int stage = decltype(stage_i)::value;
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if (k + 1 < loops)
            issue_tile(number<1 - stage>{}, k + 1);

        static_for<2>([&](auto half_i) {
            constexpr int half = decltype(half_i)::value;
            scale_a[half] = read_scale_a(k, half_i);
            static_for<T::E_M>([&](auto m_i) {
                constexpr int m = decltype(m_i)::value;
                const int r = (wave_m * T::E_M + m) * 16 + lane % 16;
                v_a[half][m] = read_operand_fragment<false, typename MMA::vtype_a>(
                    s_a, (stage * 2 + half) * T::A_SLOT, r, lane);
            });
        });
        scale_b = load<1>(s_sfb, k)[0];
        static_for<2 * T::E_N>([&](auto n_i) { read_b(stage_i, n_i); });
        s_waitcnt_lgkmcnt(0_I);

        static_for<2 * T::E_N>([&](auto n_i) {
            constexpr int n = decltype(n_i)::value;
            constexpr int half_n = n / T::E_N;
            static_for<2>([&](auto half_i) {
                constexpr int half_m = decltype(half_i)::value;
                static_for<T::E_M>([&](auto m_i) {
                    constexpr int m = decltype(m_i)::value;
                    constexpr int ci = ((half_m * 2 + half_n) * T::E_M + m) * T::E_N + n % T::E_N;
                    v_c[ci] = MMA{}(v_a[half_m][m], v_b[n], v_c[ci],
                        static_cast<int>(scale_a[half_m]), static_cast<int>(scale_b), m_i, number<half_n>{});
                });
            });

        });
    };

    // ===== Prologue =====
    issue_tile(0_I, 0);

    // ===== Main loop =====
    // All waves publish together; each B fragment is reused across both M halves.
#pragma nounroll
    for (int k = 0; k < loops; k += 2) {
        consume_tile(0_I, k);
        if (k + 1 < loops)
            consume_tile(1_I, k + 1);
    }

    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    sync_threads();

    // ===== Output writeback =====
    // Stage the MFMA's row fragments, then write contiguous BF16 vectors.
    static_for<4 * T::E_M * T::E_N>([&](auto ci) {
        constexpr int index = decltype(ci)::value;
        constexpr int quadrant = index / (T::E_M * T::E_N);
        constexpr int mr = index / T::E_N % T::E_M;
        constexpr int nr = index % T::E_N;
        const int r = quadrant / 2 * T::HALF_M + wave_m * T::E_M * 16 + mr * 16 + lane / 16 * 4;
        const int c = quadrant % 2 * T::HALF_N + wave_n * T::E_N * 16 + nr * 16 + lane % 16;
        const auto converted = cast<D_C>(v_c[index]);
        static_for<4>([&](auto e) {
            const vector_t<D_C, 1> value{converted[decltype(e)::value]};
            store<1>(s_c, value, (r + decltype(e)::value) * T::C_STRIDE + c);
        });
    });

    s_waitcnt_lgkmcnt(number<0>{});
    sync_threads();
    static_for<T::OUTPUT_PASSES>([&](auto i) {
        const int linear = (wave * 64 + lane) * 8 + decltype(i)::value * T::BLOCK_SIZE * 8;
        const int r = linear / T::B_N;
        const int c = linear % T::B_N;
        store<8>(g_c, load<8>(s_c, r * T::C_STRIDE + c), r * kargs.stride_c + c, 0, number<2>{});
    });
}
#endif

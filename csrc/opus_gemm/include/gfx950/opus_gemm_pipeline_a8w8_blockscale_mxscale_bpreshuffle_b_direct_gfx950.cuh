// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <type_traits>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_direct_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_b_direct_detail {
using opus::operator""_I;

// The ring pipelines count DMA groups, rather than individual VMEM operations.
// Keep the grouped instruction behind the same view/layout API as opus::gmem;
// the ordinary gmem transport is not tracked by asyncmark on gfx950.
template<class D>
struct async_gmem : opus::gmem<D>
{
    using Base = opus::gmem<D>;
    using Base::Base;

    template<opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void _async_load(OPUS_LDS_ADDR void* destination,
                                                int offset,
                                                int scalar_offset       = 0,
                                                opus::number<Immediate> = {},
                                                opus::number<Cache>     = {})
    {
        using Value = typename Base::template vector_type<Vec>;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(this->cached_rsrc,
                                                       destination,
                                                       sizeof(Value),
                                                       offset * sizeof(D),
                                                       scalar_offset * sizeof(D),
                                                       Immediate,
                                                       Cache);
    }

    template<opus::index_t Vec, opus::index_t Immediate = 0, opus::index_t Cache = 0>
    __device__ __forceinline__ void async_load(void* destination,
                                               int offset,
                                               int scalar_offset       = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache>     = {})
    {
        auto* target =
            reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(destination));
        _async_load<Vec>(
            target, offset, scalar_offset, opus::number<Immediate>{}, opus::number<Cache>{});
    }

    template<opus::index_t Vec,
              class GlobalLayout,
              class SharedLayout,
              opus::index_t Immediate = 0,
              opus::index_t Cache     = 0,
              std::enable_if_t<opus::is_layout_v<GlobalLayout> && opus::is_layout_v<SharedLayout>,
                               bool>  = true>
    __device__ __forceinline__ void async_load(OPUS_LDS_ADDR void* destination,
                                               const GlobalLayout& global_layout,
                                               const SharedLayout& shared_layout,
                                               int scalar_offset       = 0,
                                               opus::number<Immediate> = {},
                                               opus::number<Cache>     = {})
    {
        using Load                = opus::layout_load_traits<GlobalLayout, Vec>;
        const auto global_offsets = opus::layout_to_offsets<Vec>(global_layout);
        const auto shared_offsets = opus::layout_to_offsets<Vec>(shared_layout);
        // Preserve the OPUS shared view's address space through layout offsets.
        // A generic-pointer round trip adds a null-address select at each DMA.
        auto* target = reinterpret_cast<OPUS_LDS_ADDR D*>(destination);
        opus::static_for<Load::r_elem.value>([&](auto i) {
            _async_load<Vec>(target + shared_offsets[i],
                             global_offsets[i],
                             scalar_offset,
                             opus::number<Immediate>{},
                             opus::number<Cache>{});
        });
    }
};

template<class D>
__device__ __forceinline__ auto make_async_gmem(const D* ptr, unsigned int bytes)
{
    return async_gmem<D>(ptr, bytes);
}

__device__ __forceinline__ void commit_async_loads() { __builtin_amdgcn_asyncmark(); }

// Some scale-address policies require a VGPR offset even when all active rows
// happen to share a scalar address. Preserve that requirement at the transport.
template<class Layout>
__device__ __forceinline__ auto vectorize_layout_offsets(Layout layout)
{
#if defined(__AMDGCN__)
    for (int i = 0; i < Layout::num_issues; ++i)
    {
        asm volatile("" : "+v"(layout.offsets[i]));
    }
#endif
    return layout;
}

// Assemble the two MFMA chunks directly, as in the pinned-AGPR pipeline.
// Static chunk ownership avoids a register-array index in the LDS load path.
template<class Fragment, class Memory, class Layout>
__device__ __forceinline__ Fragment load_operand_fragment(Memory& memory, const Layout& layout)
{
    using Load = opus::layout_load_traits<Layout, 16>;
    static_assert(Load::r_elem.value == 2);
    const auto offsets = opus::layout_to_offsets<16>(layout);
    using Chunk        = decltype(opus::load<16>(memory, offsets[0]));
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) { values[i] = opus::load<16>(memory, offsets[i]); });
    Fragment result;
    opus::set_slice(result, values[0], 0_I, 16_I);
    opus::set_slice(result, values[1], 16_I, 32_I);
    return result;
}

template<class T, bool ContiguousWaveN = false, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(const Accumulators &c, Gmem gc, int lane,
                                                    int wm, int wn, int stride_c) {
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
        const int m = (ci + side) / T::E_N, n = (ci + side) % T::E_N;
        const int rr = (m * T::T_M + wm) * 16 + lane % 16;
        const int nn = (ContiguousWaveN ? n + wn * T::E_N : n * T::T_N + wn) * 16 + (lane / 32) * 8;
        store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                 number<T::STORE_CACHE>{});
    });
    if constexpr (T::E_M * T::E_N % 2 != 0) {
        constexpr int ci = T::E_M * T::E_N - 1;
        const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[0]);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[1]);
        if (side == 0) {
            vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
            const int rr = ((ci / T::E_N) * T::T_M + wm) * 16 + lane % 16;
            const int nn = (ContiguousWaveN ? (ci % T::E_N) + wn * T::E_N : (ci % T::E_N) * T::T_N + wn) * 16 + (lane / 32) * 8;
            store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                     number<T::STORE_CACHE>{});
        }
    }
}

#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
template<class Arguments>
__device__ __forceinline__ void
apply_fixed_args(Arguments &a) {
    if constexpr (requires { a.m; })
        a.m = OPUS_BP_FIXED_M;
    if constexpr (requires { a.n; })
        a.n = OPUS_BP_FIXED_N;
    if constexpr (requires { a.k; })
        a.k = OPUS_BP_FIXED_K;
    if constexpr (requires { a.batch; })
        a.batch = 1;
    if constexpr (requires { a.stride_a; })
        a.stride_a = OPUS_BP_FIXED_K;
    if constexpr (requires { a.stride_b; })
        a.stride_b = OPUS_BP_FIXED_K;
    if constexpr (requires { a.stride_c; })
        a.stride_c = OPUS_BP_FIXED_N;
    if constexpr (requires { a.stride_a_batch; })
        a.stride_a_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_K);
    if constexpr (requires { a.stride_b_batch; })
        a.stride_b_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_N) * OPUS_BP_FIXED_K);
    if constexpr (requires { a.stride_c_batch; })
        a.stride_c_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_N);
    if constexpr (requires { a.stride_sfa; })
        a.stride_sfa = OPUS_BP_FIXED_M;
    if constexpr (requires { a.stride_sfb; })
        a.stride_sfb = OPUS_BP_FIXED_K / 128;
    if constexpr (requires { a.stride_sfa_batch; })
        a.stride_sfa_batch = OPUS_BP_FIXED_M * (OPUS_BP_FIXED_K / 128);
    if constexpr (requires { a.stride_sfb_batch; })
        a.stride_sfb_batch = (OPUS_BP_FIXED_N / 128) * (OPUS_BP_FIXED_K / 128);
}
#endif

using namespace opus;

template<class T, int Vec, class Copy>
__device__ inline constexpr auto make_layout_shared_copy(Copy copy, int wave_id, int base, int copies)
{
    constexpr int lane_stride = Vec < 4 ? 4 : Vec;
    const auto shared_block_shape =
        opus::make_tuple(copies, opus::number<T::BLOCK_SIZE / 64>{}, 64_I, opus::number<Vec>{});
    constexpr auto shared_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(
               shared_block_shape,
               opus::unfold_x_stride(shared_block_dim,
                                     shared_block_shape,
                                     opus::tuple{opus::number<lane_stride>{}, 1_I}),
               opus::unfold_p_coord(shared_block_dim, opus::tuple{int(copy), wave_id, 0_I})) +
           base;
}

// A: global copy, LDS destination, and MFMA operand.
template<class T, class Copy>
__device__ inline constexpr auto make_layout_ga_scale(Copy copy, int lane_id, int wave_id, int stride_a)
{
    constexpr int threads_k       = T::B_K / 16;
    constexpr int rows_per_wave   = 64 / threads_k;
    constexpr int waves           = T::BLOCK_SIZE / 64;
    constexpr int copies          = (T::B_M * T::B_K + T::BLOCK_SIZE * 16 - 1) / (T::BLOCK_SIZE * 16);
    constexpr auto ga_block_shape = opus::make_tuple(opus::number<copies>{},
                                                     opus::number<waves>{},
                                                     opus::number<rows_per_wave>{},
                                                     opus::number<threads_k>{},
                                                     16_I);
    constexpr auto ga_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const auto logical_coord =
        opus::tuple{int(copy), wave_id, lane_id / threads_k, lane_id % threads_k};
    // XOR permutes K-vector ownership using the row coordinate. Compute that
    // coordinate from the same tile decomposition, independently of its pitch.
    const auto row_layout = opus::make_layout<-1>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{1_I, 0_I}),
        opus::unfold_p_coord(ga_block_dim, logical_coord));
    const int row_swizzle = row_layout(0_I) % threads_k;
    return opus::make_layout<16>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(
            ga_block_dim,
            opus::tuple{
                int(copy), wave_id, lane_id / threads_k, (lane_id % threads_k) ^ row_swizzle}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_sa_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, 16>(copy, wave_id, base, T::A_COPIES);
}

template<class T, bool VectorOffsets = false>
__device__ inline constexpr auto make_layout_ra_scale(int row, int lane_id, int k_slice, int base = 0)
{
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::B_M / 16>{}, 16_I, opus::number<T::B_K / 128>{}, 2_I, 4_I, 16_I);
    constexpr auto ra_block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}),
                                                   opus::make_tuple(opus::p_dim{}),
                                                   opus::make_tuple(opus::y_dim{}));
    const int swizzle           = row % (T::B_K / 16);
    // XOR flips independent K-slice/lane coordinates. A flipped half axis
    // starts at half one and walks backwards, preserving MFMA operand order.
    const int half_stride = swizzle & 4 ? -64 : 64;
    auto layout =
        opus::make_layout<16>(
            ra_block_shape,
            opus::unfold_x_stride(
                ra_block_dim,
                ra_block_shape,
                opus::tuple{opus::number<T::B_K>{}, 128_I, half_stride, 16_I, 1_I}),
            opus::unfold_p_coord(
                ra_block_dim,
                opus::tuple{
                    row / 16, row % 16, k_slice ^ (swizzle / 8), (lane_id / 16) ^ (swizzle % 4)})) +
        base + (swizzle & 4) * 16;
    if constexpr (VectorOffsets)
        return opus_gemm_b_direct_detail::vectorize_layout_offsets(
            layout);
    else
        return layout;
}

// B operand layouts.
template<class T>
__device__ inline constexpr auto
make_layout_rb_scale(int group_n, int lane_id, int k_slice, int stride_b, int base = 0)
{
    const auto rb_block_shape =
        opus::make_tuple(opus::number<T::B_N / 16>{}, stride_b / 128, 2_I, 64_I, 16_I);
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<16>(
               rb_block_shape,
               opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{16 * stride_b, 1_I}),
               opus::unfold_p_coord(rb_block_dim, opus::tuple{group_n, k_slice, lane_id})) +
           base;
}

// A scale: global copy, LDS destination, and MFMA scale.
template<class T, class Copy>
__device__ inline auto make_layout_gsfa_scale(Copy copy, int tid, int row, int m, int stride_sfa)
{
    constexpr int Vec        = T::BYTE_SCALE ? 1 : 16;
    const int copy_lane      = int(copy) * T::BLOCK_SIZE + tid;
    const int local_row      = copy_lane % (T::SA_ROWS / Vec) * Vec;
    const int column         = copy_lane / (T::SA_ROWS / Vec);
    const int bounded_column = column < T::SCALE_COLS ? column : T::SCALE_COLS - 1;
    const bool valid_row     = local_row < T::B_M && row + local_row < m;
    const bool valid         = valid_row && (!T::BYTE_SCALE || column < T::SCALE_COLS);
    const int scale_column   = valid ? bounded_column : T::SCALE_COLS - 1;
    const int scale_row      = valid ? row + local_row : m;

    const auto block_shape =
        opus::make_tuple(opus::number<T::SCALE_COLS>{}, (m + Vec - 1) / Vec, opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    auto layout              = opus::make_layout<Vec>(
                      block_shape,
                      opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
                      opus::unfold_p_coord(block_dim, opus::tuple{scale_column, scale_row / Vec})) +
                  scale_row % Vec;
    return opus_gemm_b_direct_detail::vectorize_layout_offsets(layout);
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_ssfa_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, T::BYTE_SCALE ? 1 : 16>(copy, wave_id, base, T::SA_COPIES);
}

template<class T>
__device__ inline constexpr auto
make_layout_rsfa_scale(int row, int k_slice, int block_m, int columns, int base = 0)
{
    constexpr int spacing       = T::SA_SCALE;
    const auto rsfa_block_shape = opus::make_tuple(columns, block_m / 16, 16_I, 1_I);
    constexpr auto rsfa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               rsfa_block_shape,
               opus::unfold_x_stride(rsfa_block_dim,
                                     rsfa_block_shape,
                                     opus::tuple{block_m * spacing, opus::number<spacing>{}}),
               opus::unfold_p_coord(rsfa_block_dim, opus::tuple{k_slice, row / 16, row % 16})) +
           base;
}

// B scale: global copy, LDS destination, and MFMA scale.
template<class T>
__device__ inline constexpr auto
make_layout_gsfb_scale(int tid, int column, int groups, int stride_sfb)
{
    constexpr int Vec      = T::SB_SLOTS ? 1 : 4;
    constexpr int columns  = T::SCALE_COLS / Vec;
    const int copy_lane    = tid % (T::SB_ROWS * columns);
    const auto block_shape = opus::make_tuple(groups, opus::number<columns>{}, opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<Vec>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(block_dim,
                             opus::tuple{column / 128 + copy_lane / columns, copy_lane % columns}));
}

template<class T, class Copy>
__device__ inline constexpr auto make_layout_ssfb_scale(Copy copy, int wave_id, int base)
{
    return make_layout_shared_copy<T, T::SB_SLOTS ? 1 : 4>(copy, wave_id, base, 1);
}

template<class T>
__device__ inline constexpr auto make_layout_rsfb_scale(int group_n, int k_column)
{
    constexpr int spacing = T::SB_SLOTS ? 4 : 1;
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SB_ROWS>{}, opus::number<T::SCALE_COLS>{}, 1_I);
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
               block_shape,
               opus::unfold_x_stride(
                   block_dim,
                   block_shape,
                   opus::tuple{opus::number<T::SCALE_COLS * spacing>{}, opus::number<spacing>{}}),
               opus::unfold_p_coord(block_dim, opus::tuple{group_n, k_column})) +
           T::OFF_SB;
}

// C register and output layouts.
template<class T, bool Normal = false>
__device__ inline constexpr auto make_layout_rc_scale(int row, int column, int lane_id, int stride_c)
{
    if constexpr (Normal)
    {
        constexpr auto rc_block_shape = opus::make_tuple(
            opus::number<T::B_M / 16>{}, 4_I, 4_I, opus::number<T::B_N / 16>{}, 16_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}));
        return opus::make_layout<1>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id / 16, column / 16, lane_id % 16}));
    }
    else
    {
        constexpr auto rc_block_shape = opus::make_tuple(
            opus::number<T::B_M / 16>{}, 16_I, opus::number<T::B_N / 16>{}, 4_I, 4_I);
        constexpr auto rc_block_dim =
            opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                             opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
        return opus::make_layout<4>(
            rc_block_shape,
            opus::unfold_x_stride(rc_block_dim, rc_block_shape, opus::tuple{stride_c, 1_I}),
            opus::unfold_p_coord(rc_block_dim,
                                 opus::tuple{row / 16, lane_id % 16, column / 16, lane_id / 16}));
    }
}

template<int Vec, int Rows, int Columns>
__device__ inline constexpr auto make_layout_matrix_vector(int row, int column, int stride)
{
    constexpr auto matrix_block_shape =
        opus::make_tuple(opus::number<Rows>{}, opus::number<Columns / Vec>{}, opus::number<Vec>{});
    constexpr auto matrix_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<Vec>(
        matrix_block_shape,
        opus::unfold_x_stride(matrix_block_dim, matrix_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(matrix_block_dim, opus::tuple{row, column / Vec}));
}

template<class T, int Rows = T::B_M>
__device__ inline constexpr auto make_layout_sc(int row, int column, int stride)
{
    return make_layout_matrix_vector<8, Rows, T::B_N>(row, column, stride);
}

template<class T>
__device__ inline constexpr auto make_layout_gc(int row, int column, int stride)
{
    return make_layout_matrix_vector<8, T::B_M, T::B_N>(row, column, stride);
}

} // namespace opus_gemm_b_direct_detail

namespace opus {
template<class D>
struct is_gmem<opus_gemm_b_direct_detail::async_gmem<D>> : true_type {};
} // namespace opus


template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
    opus_gemm_b_direct_detail::apply_fixed_args(kargs);
#endif
    using namespace opus;
    using namespace opus_gemm_b_direct_detail;
    namespace layout = opus_gemm_b_direct_detail;
    using T          = opus::remove_cvref_t<Traits>;
    using Index      = unsigned;

    // Prepare tile views, operand rings, scales, and accumulators.
    constexpr int BM = T::B_M, BN = T::B_N, NB = T::STAGES, BD = T::B_AHEAD, BS = T::B_SETS,
                  P  = T::PANELS;
    constexpr int MR = T::MR, NR = T::NR, K = T::K, SC = T::SCALE_COLS, BK = T::B_K, KS = BK / 128;
    const int M = kargs.m, N = kargs.n;
    const int split    = T::SPLIT_K > 1 ? block_id_z() : 0;
    using D_C          = std::conditional_t<T::SPLIT_K == 1, bf16_t, float>;
    const int stride_a = T::CONTIGUOUS ? T::TOTAL_K : kargs.stride_a,
              stride_b = T::CONTIGUOUS ? T::TOTAL_K : kargs.stride_b, stride_c = kargs.stride_c;
    static_assert((BK == 128 || BK == 256) && SC == K / 128);
    static_assert(T::T_M == 1 && T::T_N * 64 == T::BLOCK_SIZE);
    static_assert(T::MR * 16 == T::B_M && T::NR * T::T_N * 16 == T::B_N);
    const Index tid = thread_id_x(), lane_id = tid % 64, lane_id_m = lane_id % 16;
    const Index wave_id = __builtin_amdgcn_readfirstlane(tid / 64), wave_id_n = wave_id;
    if constexpr (requires { T::THREAD_RANGE_KNOWN; })
    {
        __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
        __builtin_assume(static_cast<unsigned>(wave_id) < T::BLOCK_SIZE / 64);
    }
    const unsigned pid = block_id_x(), gm = (M + BM - 1) / BM, gn = N / BN, total = gm * gn;
    const unsigned tile = T::XCD_ORDER ? pid % 8 * ((total + 7) / 8) + pid / 8 : pid;
    if constexpr (T::XCD_ORDER)
        if (tile >= total)
            return;
    const int row  = T::GRID_2D ? block_id_y() * BM : (M <= BM ? 0 : (tile / gn) * BM);
    const int col  = T::GRID_2D ? block_id_x() * BN : (M <= BM ? tile * BN : (tile % gn) * BN);
    int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));
    if constexpr (requires { T::SIMPLE_TAIL_ROWS; })
        valid_rows = M - row < BM ? M - row : BM;
    auto g_a =
        make_async_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_a) + row * stride_a + split * K,
                        static_cast<unsigned>(valid_rows * stride_a - split * K));
    auto g_b =
        make_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) + col * stride_b + split * K * 16,
                  static_cast<unsigned>(BN * stride_b - split * K * 16));
    auto g_sfa = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa) +
                                     split * SC * kargs.stride_sfa,
                                 static_cast<unsigned>((SC - 1) * kargs.stride_sfa + M));
    auto g_sfb = make_async_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb) + split * SC,
                                 static_cast<unsigned>((N / 128 - 1) * kargs.stride_sfb + SC));
    auto g_c =
        make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<int64_t>(split) * M * stride_c +
                      static_cast<int64_t>(row) * stride_c + col,
                  static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * sizeof(D_C)));
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto s_a  = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto s_sf = make_smem(arena);
    using Base =
        decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> c{};
    array<array<array<typename Base::vtype_b, NR>, KS>, BS> b_regs;
    constexpr int VecScaleA  = T::BYTE_SCALE ? 1 : 16;
    constexpr int VecScaleB  = T::SB_SLOTS ? 1 : 4;
    auto issue_b             = [&](auto bi, int kt) {
        constexpr int set = decltype(bi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const auto u_rb =
                    layout::make_layout_rb_scale<T>(wave_id_n * NR + n, lane_id, ks, stride_b);
                b_regs[set][ks][n] = load<16>(g_b, u_rb, kt * (BK * 16), number<T::B_CACHE>{});
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };
    auto issue_a = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            const auto u_ga =
                layout::make_layout_ga_scale<T>(ii, lane_id, wave_id, stride_a);
            const auto u_sa = layout::make_layout_sa_scale<T>(ii, wave_id, slot * T::STAGE_BYTES);
            async_load<16>(g_a, s_a.ptr, u_ga, u_sa, kt * BK, 0_I, number<T::CACHECTL_A>{});
        });
        commit_async_loads();
        __builtin_amdgcn_sched_barrier(0);
    };
    // Retire one A/B pair with the original VMEM ledger and pinning order.
    auto step = [&](int t, int slot, auto bi, auto wai, auto wbi, auto fai, auto fbi) {
        constexpr int set = decltype(bi)::value, wa = decltype(wai)::value,
                      wb        = decltype(wbi)::value;
        constexpr bool future_a = decltype(fai)::value, future_b = decltype(fbi)::value;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        sync_threads();
        if constexpr (future_b)
            issue_b(number<(set + BD) % BS>{}, t + BD);
        array<array<typename Base::vtype_a, MR>, KS> af;
        array<array<unsigned, MR>, KS> as;
        array<array<unsigned, NR>, KS> bs;
        static_for<KS>([&](auto ki) {
            constexpr int ks    = decltype(ki)::value;
            const int scale_col = t * KS + ks;
            static_for<NR>([&](auto ni) {
                constexpr int nr  = decltype(ni)::value;
                const int group_n = (wave_id_n * NR * 16 + nr * 16) / 128;
                const auto u_rsfb = layout::make_layout_rsfb_scale<T>(group_n, scale_col);
                bs[ks][nr]        = load<1>(s_sf, u_rsfb)[0];
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int ar    = m * 16 + lane_id_m;
                const auto u_ra =
                    layout::make_layout_ra_scale<T>(ar, lane_id, ks, slot * T::STAGE_BYTES);
                {
                    af[ks][m] = load_operand_fragment<typename Base::vtype_a>(s_a, u_ra);
                }
                const auto u_rsfa =
                    layout::make_layout_rsfa_scale<T>(ar, scale_col, T::SA_ROWS, SC, T::OFF_SA);
                as[ks][m] = load<1>(s_sf, u_rsfa)[0];
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
                    constexpr int n = decltype(ni)::value, ci = m * NR + n;
                    {
                        c[ci] = Base{}(af[ks][m],
                                       b_regs[set][ks][n],
                                       c[ci],
                                       static_cast<int>(as[ks][m]),
                                       static_cast<int>(bs[ks][n]),
                                       0_I,
                                       0_I);
                    }
                });
            });
        });
        if constexpr (!T::C_AGPR && !T::PIN_C)
            static_for<MR * NR>([&](auto ci) {
                auto& v = c[decltype(ci)::value];
                asm volatile("" : "+v"(v));
            });
        __builtin_amdgcn_sched_barrier(0);
    };

    // ===== Prologue =====
    // Issue B, the first A stage, both scale panels, then the remaining A stages.
    static_for<T::INITIAL_B>([&](auto i) { issue_b(i, decltype(i)::value); });
    issue_a(0, 0);
    static_for<T::SA_COPIES>([&](auto ii) {
        const auto u_gsfa = layout::make_layout_gsfa_scale<T>(ii, tid, row, M, kargs.stride_sfa);
        const auto u_ssfa =
            layout::make_layout_ssfa_scale<T>(ii, wave_id, T::OFF_SA);
        async_load<VecScaleA>(g_sfa, s_sf.ptr, u_gsfa, u_ssfa);
    });
    const auto u_gsfb = layout::make_layout_gsfb_scale<T>(tid, col, N / 128, kargs.stride_sfb);
    const auto u_ssfb =
        layout::make_layout_ssfb_scale<T>(0_I, wave_id, T::OFF_SB);
    async_load<VecScaleB>(g_sfb, s_sf.ptr, u_gsfb, u_ssfb);
    static_for<T::INITIAL_A - 1>(
        [&](auto i) { issue_a(decltype(i)::value + 1, decltype(i)::value + 1); });

    // ===== Main loop =====
    constexpr int START = T::STEADY < NB - 1 ? T::STEADY : NB - 1;
    static_for<START>([&](auto ti) {
        constexpr int t = decltype(ti)::value;
        step(t,
             t % NB,
             number<t % BS>{},
             number<T::wait_a(t)>{},
             number<T::wait_b(t)>{},
             number<true>{},
             number<(t + BD < P)>{});
    });
    constexpr int PHASES = []() constexpr {
        int a = NB, b = BS;
        while(b)
        {
            int r = a % b;
            a     = b;
            b     = r;
        }
        return NB / a * BS;
    }();
    constexpr int GROUPS = (T::STEADY - START) / PHASES, REGULAR_END = START + GROUPS * PHASES;
    if constexpr (GROUPS > 0)
    {
#pragma clang loop unroll(disable)
        for (int group = 0; group < GROUPS; ++group)
        {
            static_for<PHASES>([&](auto phase) {
                constexpr int ph = decltype(phase)::value, t0 = START + ph;
                step(START + group * PHASES + ph,
                     t0 % NB,
                     number<t0 % BS>{},
                     number<T::wait_a(t0)>{},
                     number<T::wait_b(t0)>{},
                     number<true>{},
                     number<true>{});
            });
        }
    }

    // ===== Epilogue =====
    // Retire the final panels with their existing refill conditions and fixed waits.
    static_for<P - REGULAR_END>([&](auto ti) {
        constexpr int t = REGULAR_END + decltype(ti)::value;
        step(t,
             t % NB,
             number<t % BS>{},
             number<T::wait_a(t)>{},
             number<T::wait_b(t)>{},
             number<(t + NB - 1 < P)>{},
             number<(t + BD < P)>{});
    });

    // ===== Write back =====
    if constexpr (T::SPLIT_K > 1)
    {
        static_for<MR>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * NR + n;
                const auto u_gc = layout::make_layout_rc_scale<T>(
                    m * 16, wave_id_n * NR * 16 + n * 16, lane_id, stride_c);
                store<4>(g_c, c[ci], u_gc, 0, number<T::STORE_CACHE>{});
            });
        });
    }
    else if constexpr (T::OUTPUT == 2)
    {
        opus_gemm_b_direct_detail::store_mfma16_packed<T, true>(c, g_c, lane_id, 0, wave_id_n, stride_c);
    }
    else
    {
        // Stage C in LDS and gather contiguous BF16 stores.
        auto s_c               = make_smem(reinterpret_cast<bf16_t*>(arena));
        constexpr int MAX_ROWS = BM;
        static_for<(MAX_ROWS + T::C_ROWS - 1) / T::C_ROWS>([&](auto parti) {
            constexpr int part = decltype(parti)::value;
            constexpr int PART_ROWS =
                MAX_ROWS - part * T::C_ROWS < T::C_ROWS ? MAX_ROWS - part * T::C_ROWS : T::C_ROWS;
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                if constexpr (m * 16 / T::C_ROWS == part && m * 16 < MAX_ROWS)
                    static_for<NR>([&](auto ni) {
                        constexpr int n = decltype(ni)::value, ci = m * NR + n;
                        const auto u_sc =
                            layout::make_layout_rc_scale<T>(m * 16 - part * T::C_ROWS,
                                                                  wave_id_n * NR * 16 + n * 16,
                                                                  lane_id,
                                                                  BN + 8);
                        store<4>(s_c, cast<bf16_t>(c[ci]), u_sc);
                    });
            });
            s_waitcnt_lgkmcnt(0_I);
            sync_threads();
            constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
            array<vector_t<bf16_t, 8>, CHUNKS> output;
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid, rr = index / (BN / 8),
                          nn = index % (BN / 8) * 8;
                const auto u_sc =
                    layout::make_layout_sc<T, PART_ROWS>(rr, nn, BN + 8);
                if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                    output[i] = load<8>(s_c, u_sc);
                else if (rr < PART_ROWS)
                    output[i] = load<8>(s_c, u_sc);
            });
            __builtin_amdgcn_sched_barrier(0);
            s_waitcnt_lgkmcnt(0_I);
            static_for<CHUNKS>([&](auto ii) {
                constexpr int i = decltype(ii)::value;
                const int index = i * T::BLOCK_SIZE + tid, rr = index / (BN / 8),
                          nn = index % (BN / 8) * 8;
                if (rr < PART_ROWS && part * T::C_ROWS + rr < valid_rows)
                    store<8>(g_c,
                             output[i],
                             layout::make_layout_gc<T>(part * T::C_ROWS + rr, nn, stride_c),
                             0,
                             number<T::STORE_CACHE>{});
            });
        });
    }
    s_waitcnt_vmcnt(0_I);
}

// Existing generated CO stubs retain their registered compute entry.
namespace opus_gemm_b_direct_detail {
template<class Traits>
__device__ __forceinline__ void
bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
    opus_gemm_blockscale_mxscale_bpreshuffle_b_direct<Traits>(kargs);
}
} // namespace opus_gemm_b_direct_detail
#endif

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU) void
opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_bulk_panel_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_blockscale_mxscale_bpreshuffle_b_direct<Traits>(kargs);
#endif
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 fp8/e8m0 flatmm split-K pipeline for decode-oriented BMM.
//
// This is the first B_K=128 version: one K iteration maps to one DSv4
// checkpoint scale block, and one consumer-wave M tile maps to one per-row A
// scale. The host launcher keeps this v1 on divisible decode shapes.
#pragma once

#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"
// opus_bmm_splitk_reduce_kernel: the split-K > 1 path's fp32->Y reduce. Lives
// in the shared reduce header so the codegen'd BMM launcher (which #includes
// this pipeline header on the non-fused device pass) can launch it.
#include "splitk_reduce_gfx950.cuh"

#ifdef __HIP_DEVICE_COMPILE__

// ============================================================================
// Layout helpers. Suffixed with _mxsk to avoid ODR collisions with the bf16
// flatmm splitK helpers when both headers are included in a build.
// ============================================================================

template<typename T, int WAVES>
inline __device__ auto make_layout_gmem_group_load_mxsk(int lane_id, int wave_id, int stride) {
    constexpr int threads_k = T::LOAD_GROUP_K / T::VEC_A;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto g_block_shape = opus::make_tuple(
        opus::number<interlanegroup_m>{},
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto g_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        g_block_shape,
        opus::unfold_x_stride(g_block_dim, g_block_shape, opus::tuple{stride, 1_I}),
        opus::unfold_p_coord(g_block_dim,
            opus::tuple{lane_id / threads_k / T::LOAD_GROUP_M_LANE,
                        wave_id % WAVES,
                        (lane_id / threads_k) % T::LOAD_GROUP_M_LANE,
                        lane_id % threads_k}));
}

template<typename T, int WAVES>
inline __device__ auto make_layout_smem_group_load_mxsk(int lane_id, int wave_id) {
    constexpr int repeat_m = T::slots / WAVES;

    constexpr auto s_block_shape = opus::make_tuple(
        opus::number<repeat_m>{},
        opus::number<WAVES>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto s_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<0>(
        s_block_shape,
        opus::unfold_x_stride(s_block_dim, s_block_shape,
            opus::tuple{T::smem_linear_wave_per_async_load + T::smem_padding, 1_I}),
        opus::unfold_p_coord(s_block_dim, opus::tuple{wave_id % WAVES, lane_id}));
}

template<typename T>
inline __device__ auto make_layout_ra_mxsk(int lane_id, int wave_id_m) {
    constexpr int threads_k = opus::get_warp_size() / T::W_M;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;
    constexpr int interlanegroup_m = threads_m_per_wave / T::LOAD_GROUP_M_LANE;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int m_block_stride = T::NUM_LOAD_GROUPS_PER_BK * per_block_load;

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::COM_REP_M>{},
        opus::number<T::slots>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::T_M>{},
        opus::number<interlanegroup_m / T::slots>{},
        opus::number<T::LOAD_GROUP_M_LANE>{},
        opus::number<2>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{},
                         opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<0>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<m_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(ra_block_dim,
            opus::tuple{lane_id_m % T::slots,
                        wave_id_m,
                        lane_id_m / T::slots,
                        lane_id_m % T::LOAD_GROUP_M_LANE,
                        lane_id / T::W_M}));
}

template<typename T>
inline __device__ auto make_layout_rb_mxsk(int lane_id) {
    constexpr int grpk_b = opus::get_warp_size() / T::W_N;
    constexpr int interlanegroup_n = T::W_N / T::LOAD_GROUP_N_LANE;
    constexpr int loops_b = interlanegroup_n / T::slots;
    constexpr int tiles_per_block_n = T::LOAD_GROUP_N / T::W_N;
    constexpr int num_blocks_n = T::COM_REP_N / tiles_per_block_n;
    constexpr int per_block_load = T::slots * (T::smem_linear_wave_per_async_load + T::smem_padding);
    constexpr int n_block_stride = T::NUM_LOAD_GROUPS_PER_BK * per_block_load;
    constexpr int n_intra_stride = T::LOAD_GROUP_N_LANE * 2 * grpk_b * T::VEC_B;

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<num_blocks_n>{},
        opus::number<T::slots>{},
        opus::number<tiles_per_block_n>{},
        opus::number<loops_b>{},
        opus::number<T::NUM_LOAD_GROUPS_PER_BK>{},
        opus::number<T::LOAD_GROUP_N_LANE>{},
        opus::number<2>{},
        opus::number<grpk_b>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout<0>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<n_block_stride>{},
                        opus::number<T::smem_linear_wave_per_async_load + T::smem_padding>{},
                        opus::number<n_intra_stride>{},
                        opus::number<per_block_load>{},
                        1_I}),
        opus::unfold_p_coord(rb_block_dim,
            opus::tuple{lane_id_n % T::slots,
                        lane_id_n / T::slots,
                        lane_id_n % T::LOAD_GROUP_N_LANE,
                        lane_id / T::W_N}));
}

template<typename T>
inline __device__ auto make_layout_sfa_mxsk(int lane_id, int wave_id_m, int stride_sfa) {
    constexpr auto sfa_block_shape = opus::make_tuple(
        opus::number<T::COM_REP_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::B_K / T::GROUP_K>{});

    constexpr auto sfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sfa_block_shape,
        opus::unfold_x_stride(sfa_block_dim, sfa_block_shape,
            opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(sfa_block_dim,
            opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// pack_e8m0x4 (broadcast e8m0 -> x4 word) is shared via opus_gemm_utils.cuh.

// Per-subtile scaled-MFMA loop -- the shared "else" body used whenever the
// register tile spans more than one MX scale group. The MMA issue pattern is
// identical for every scale layout; only where each subtile's packed scale comes
// from differs, so that is injected via scale_a_of(im, ik) / scale_b_of(in, ik).
// Providers receive compile-time (opus::number<>) subtile indices and return the
// packed-int32 e8m0x4 scale. This lets plain block-scale and shuffled/preloaded
// block-scale reuse one implementation.
// OPSEL == false (default): scale_a_of/scale_b_of return a broadcast-packed x4
//   e8m0 word (all 4 bytes equal) and every MFMA selects byte 0 -- legacy
//   scalar behavior.
// OPSEL == true: scale_a_of/scale_b_of return a K-packed word holding the
//   COM_REP_K distinct K-group e8m0 bytes (byte ik == the ik-th K group) and are
//   K-independent; each MFMA selects its own byte through the compile-time
//   scale_op_sel == ik. This drops the per-subtile broadcast pack and shrinks the
//   K-direction scale register footprint to one word per M / N-scale group.
template<typename T, typename Mma, bool OPSEL = false,
         typename VA, typename VB, typename VC,
         typename ScaleAOf, typename ScaleBOf>
OPUS_D void mma_mxscale_subtile_loop(const VA& v_a, const VB& v_b, VC& v_c,
                                     ScaleAOf&& scale_a_of, ScaleBOf&& scale_b_of) {
    using MMA = typename Mma::MMA;
    constexpr int a_len = Mma::mma_a_len;
    constexpr int b_len = Mma::mma_b_len;
    constexpr int c_len = Mma::mma_c_len;
    opus::static_for<T::COM_REP_M>([&](auto im_c) {
        constexpr int im = decltype(im_c)::value;
        opus::static_for<T::COM_REP_N>([&](auto in_c) {
            constexpr int in = decltype(in_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                const int scale_a = scale_a_of(im_c, ik_c);
                const int scale_b = scale_b_of(in_c, ik_c);
                constexpr int i_tile_a = (im * T::COM_REP_K + ik);
                constexpr int i_tile_b = (in * T::COM_REP_K + ik);
                constexpr int i_tile_c = im * T::COM_REP_N + in;
                auto s_a = opus::slice(v_a,
                    opus::number<i_tile_a * a_len>{},
                    opus::number<i_tile_a * a_len + a_len>{});
                auto s_b = opus::slice(v_b,
                    opus::number<i_tile_b * b_len>{},
                    opus::number<i_tile_b * b_len + b_len>{});
                auto s_c = opus::slice(v_c,
                    opus::number<i_tile_c * c_len>{},
                    opus::number<i_tile_c * c_len + c_len>{});
                if constexpr (OPSEL)
                    s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, ik_c, ik_c);
                else
                    s_c = MMA{}(s_a, s_b, s_c, scale_a, scale_b, 0_I, 0_I);
                opus::set_slice(v_c, s_c,
                    opus::number<i_tile_c * c_len>{},
                    opus::number<i_tile_c * c_len + c_len>{});
            });
        });
    });
}

// How the per-subtile scales are materialized in the multi-scale-group path:
//   preload   -- pack every distinct scale once up front, then index the packed
//                registers in the loop (fewer pack ops when a scale is reused
//                across COM_REP_N / COM_REP_M).
//   on_demand -- pack each subtile's scale inline (lower register pressure).
//   opsel     -- pack the COM_REP_K K-group scales into one word per M / N-scale
//                group (native e8m0x4, no broadcast) and select the K byte per
//                MFMA via the hardware scale_op_sel immediate. Fewest scale ALU
//                ops + smallest K-direction scale register footprint.
enum class mxscale_pack { preload, on_demand, opsel };

template<typename T, mxscale_pack MODE = mxscale_pack::preload,
         typename Mma, typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_tiled(Mma& mma, const VA& v_a, const VB& v_b,
                              const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    static_assert(std::is_same_v<typename T::D_SF, unsigned char>);
    static_assert((T::COM_REP_M == 1 || T::COM_REP_M == 2 || T::COM_REP_M == 4)
                  && (T::COM_REP_K == 1 || T::COM_REP_K == 2 || T::COM_REP_K == 4));
    static_assert(T::B_K % T::GROUP_K == 0);
    constexpr int rep_n_per_scale = T::GROUP_N / (T::W_N * T::T_N);
    static_assert(rep_n_per_scale > 0 && T::GROUP_N % (T::W_N * T::T_N) == 0);
    // Whole register tile in a single scale group -> one (scale_a, scale_b) pair
    // -> a single tiled-mma call covers the tile.
    if constexpr (T::COM_REP_M == 1 && T::COM_REP_N <= rep_n_per_scale && T::COM_REP_K == 1) {
        const int scale_a = pack_e8m0x4(v_sfa[0]);
        const int scale_b = pack_e8m0x4(v_sfb[0]);
        v_c = mma(v_a, v_b, v_c, scale_a, scale_b, 0_I, 0_I);
    } else if constexpr (MODE == mxscale_pack::opsel) {
        // One word per M-subtile / N-scale-group holding the COM_REP_K K-group
        // e8m0 bytes; the subtile loop picks byte ik via scale_op_sel == ik.
        // NOTE: reference path only. With the vec-wide (dword) scale load below,
        // the shift/or here folds away, but op_sel packing still measures on par
        // with or slightly slower than preload's broadcast pack across the tuned
        // shapes, so preload stays the default. Kept for experimentation.
        opus::vector_t<int, T::COM_REP_M> packed_sfa;
        opus::vector_t<int, T::N_SCALE_GROUPS> packed_sfb;
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            int w = 0;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                w |= (static_cast<int>(v_sfa[im * T::SCALES_PER_BK + ik]) & 0xFF) << (8 * ik);
            });
            packed_sfa[im] = w;
        });
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            int w = 0;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                w |= (static_cast<int>(v_sfb[ng * T::SCALES_PER_BK + ik]) & 0xFF) << (8 * ik);
            });
            packed_sfb[ng] = w;
        });
        mma_mxscale_subtile_loop<T, Mma, /*OPSEL=*/true>(v_a, v_b, v_c,
            [&](auto im_c, auto) {
                return packed_sfa[decltype(im_c)::value];
            },
            [&](auto in_c, auto) {
                return packed_sfb[decltype(in_c)::value / rep_n_per_scale];
            });
    } else if constexpr (MODE == mxscale_pack::preload) {
        opus::vector_t<int, T::COM_REP_M * T::COM_REP_K> packed_sfa;
        opus::vector_t<int, T::N_SCALE_GROUPS * T::COM_REP_K> packed_sfb;
        opus::static_for<T::COM_REP_M>([&](auto im_c) {
            constexpr int im = decltype(im_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                packed_sfa[im * T::COM_REP_K + ik] =
                    pack_e8m0x4(v_sfa[im * T::SCALES_PER_BK + ik]);
            });
        });
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            opus::static_for<T::COM_REP_K>([&](auto ik_c) {
                constexpr int ik = decltype(ik_c)::value;
                packed_sfb[ng * T::COM_REP_K + ik] =
                    pack_e8m0x4(v_sfb[ng * T::SCALES_PER_BK + ik]);
            });
        });
        mma_mxscale_subtile_loop<T, Mma>(v_a, v_b, v_c,
            [&](auto im_c, auto ik_c) {
                return packed_sfa[decltype(im_c)::value * T::COM_REP_K + decltype(ik_c)::value];
            },
            [&](auto in_c, auto ik_c) {
                return packed_sfb[(decltype(in_c)::value / rep_n_per_scale) * T::COM_REP_K
                                  + decltype(ik_c)::value];
            });
    } else {
        mma_mxscale_subtile_loop<T, Mma>(v_a, v_b, v_c,
            [&](auto im_c, auto ik_c) {
                return pack_e8m0x4(
                    v_sfa[decltype(im_c)::value * T::SCALES_PER_BK + decltype(ik_c)::value]);
            },
            [&](auto in_c, auto ik_c) {
                return pack_e8m0x4(
                    v_sfb[(decltype(in_c)::value / rep_n_per_scale) * T::SCALES_PER_BK
                          + decltype(ik_c)::value]);
            });
    }
}

// Scale-packing strategy for the default multi-scale-group accum path. Flip
// between preload and opsel here to A/B the hardware scale_op_sel byte-select.
inline constexpr mxscale_pack MXSCALE_ACCUM_MODE = mxscale_pack::preload;

// Thin wrappers preserving the original entry points / call sites.
template<typename T, typename Mma, typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_flatmm_accum(Mma& mma, const VA& v_a, const VB& v_b,
                                     const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    mma_mxscale_tiled<T, MXSCALE_ACCUM_MODE>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
}

template<typename T, typename Mma, typename VA, typename VB, typename VSFA, typename VSFB, typename VC>
OPUS_D void mma_mxscale_flatmm_accum_on_demand(Mma& mma, const VA& v_a, const VB& v_b,
                                               const VSFA& v_sfa, const VSFB& v_sfb, VC& v_c) {
    mma_mxscale_tiled<T, mxscale_pack::on_demand>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
}

#endif // __HIP_DEVICE_COMPILE__

// ============================================================================
// Main kernel: 4-wave flatmm splitK, fp32 workspace output.
// ============================================================================

template<typename Traits, typename D_OUT = void, bool DIRECT_ONLY = false, bool PREFETCH_SCALE = false,
         bool PRELOAD_SF_LDS = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_splitk_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(std::is_same_v<D_C, fp32_t>, "flatmm splitK main writes fp32 workspace");
    static_assert(!DIRECT_ONLY || !std::is_void_v<D_OUT>,
                  "DIRECT_ONLY requires an output dtype for direct Y stores");
    // tileN (T_M=1, T_N=2 consumer N-split) is only wired through the non-DIRECT
    // producer/consumer path (barrier-synced; split_k==1 still stores Y directly
    // via the branch below). The persistent DIRECT_ONLY schedule keeps its
    // original tileM-only consumer mapping, so do not instantiate it for tileN.
    static_assert(!(DIRECT_ONLY && T::IS_TILE_N),
                  "tileN is not supported by the DIRECT_ONLY persistent kernel");
    // PRELOAD_SF_LDS stages the scale panels for the barrier-synced
    // producer/consumer schedule; it is not wired into the DIRECT_ONLY
    // consumer-self-load path.
    static_assert(!(PRELOAD_SF_LDS && DIRECT_ONLY),
                  "PRELOAD_SF_LDS is only supported by the non-DIRECT_ONLY schedule");

    int wgid_full = opus::block_id_x();
    int split_id  = 0;
    int wgid      = wgid_full;
    if constexpr (!DIRECT_ONLY) {
        split_id = wgid_full % kargs.split_k;
        wgid = wgid_full / kargs.split_k;
    }
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();

    const int total_iters = ceil_div(kargs.k, T::B_K);
    int my_loops = total_iters;
    int k_start = 0;
    int sf_start = 0;
    if constexpr (!DIRECT_ONLY) {
        const int iters_full = ceil_div(total_iters, kargs.split_k);
        my_loops = (split_id < kargs.split_k - 1)
                 ? iters_full
                 : (total_iters - (kargs.split_k - 1) * iters_full);
        k_start = split_id * iters_full * T::B_K;
        sf_start = split_id * iters_full * (T::B_K / T::GROUP_K);
    }
    if (my_loops < T::prefetch_k_iter) return;

    // OOB masking for partial M tiles: bound the A / sfa / C buffers to the
    // valid row window so lanes mapping to rows >= M read 0 and their stores are
    // dropped by the buffer's num_records bound. This lets any M run on a B_M
    // tile without requiring M % B_M == 0 (N/K stay divisible so B and the K
    // axis need no bound).
    //
    // Each WG owns exactly one B_M row tile and the buffer base is already at
    // `row`, so the bound only needs to cover this tile's own rows -- clamp to
    // B_M. Using the full (M - row) span would set num_records to
    // rows_avail*stride_a, and with batch-in-the-middle stride_a = batch*K, so a
    // large-M / high-batch shape would overflow the 32-bit buffer-descriptor
    // num_records (4 GiB) field and silently wrap, corrupting the OOB bound.
    // min(rows_avail, B_M) still masks the partial-M tail correctly.
    // rows_avail >= 1 always (row < M by construction).
    const int rows_left = kargs.m - row;
    const int rows_avail = rows_left < T::B_M ? rows_left : T::B_M;
    const unsigned int a_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_a * sizeof(D_A);
    // 64-bit base offsets: batch_id*stride_*_batch (= M*K for a batch-in-the-
    // middle A layout) overflows int32 for large M well before the 4 GiB buffer
    // limit, so cast the batch/row products to size_t to keep the base exact.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch + (size_t)row * kargs.stride_a + k_start,
                         a_bytes);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch + (size_t)col * kargs.stride_b + k_start);
    const bool direct_store = DIRECT_ONLY || (!std::is_void_v<D_OUT> && kargs.split_k == 1);
    const int stride_c_main = direct_store ? kargs.stride_c : kargs.stride_ws;
    const unsigned int sfa_bytes =
        (unsigned int)rows_avail * (unsigned int)kargs.stride_sfa * sizeof(D_SF);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa + sf_start,
                           sfa_bytes);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb + sf_start);

    int role = ((wave_id & 1) ^ ((wgid >> 8) & 1));

    constexpr int smem_slot_factor = DIRECT_ONLY ? 2 : 1;
    __shared__ char smem_a[smem_slot_factor * T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[smem_slot_factor * T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    // PRELOAD_SF_LDS (kid324): stage the A per-token scale (SFA) and B block
    // scale (SFB) panels for this split's whole K range into LDS once, then read
    // them from LDS (ds_read/lgkmcnt) in the consumer's per-K-tile scale fetch
    // instead of a per-tile global buffer_load (vmcnt) that gates every MMA. The
    // panels are compact byte tiles: SFA is [B_M/GROUP_M rows][loops*SCALES_PER_BK]
    // and SFB is [N_SCALE_GROUPS rows][loops*SCALES_PER_BK], both row-major with a
    // runtime per-row stride == the packed K-scale count. The LDS buffer is sized
    // for a compile-time K upper bound (SFA_K_MAX); the actual packed count is a
    // runtime value so any K<=SFA_K_MAX (K%B_K==0) works. SFA_K_MAX=8192 keeps the
    // combined panel <=~4.2 KiB, well inside the WG_PER_CU=2 LDS headroom.
    constexpr int SFA_K_MAX        = 8192;
    constexpr int SFA_K_TILES_MAX  = PRELOAD_SF_LDS ? (SFA_K_MAX / T::B_K) : 1;
    constexpr int SF_SCALES_MAX    = SFA_K_TILES_MAX * T::SCALES_PER_BK;
    constexpr int SFA_ROWS         = T::B_M / T::GROUP_M;
    constexpr int SF_LDS_ELEMS     =
        PRELOAD_SF_LDS ? ((SFA_ROWS + T::N_SCALE_GROUPS) * SF_SCALES_MAX) : 1;
    // 16B-aligned so the panel fill below can land ds_write_b128; a byte array is
    // only byte-aligned as far as the language is concerned.
    __shared__ alignas(16) D_SF smem_sf[SF_LDS_ELEMS];

    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };
    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    const int loops = my_loops;
    // Runtime packed K-scale count per SFA/SFB row (== loops * SCALES_PER_BK) and
    // the two LDS panel base pointers. SFB is packed immediately after SFA using
    // the runtime SFA size so both stay compact regardless of K.
    const int sf_k_scales = loops * T::SCALES_PER_BK;
    D_SF* s_sfa_ptr = smem_sf;
    D_SF* s_sfb_ptr = smem_sf + SFA_ROWS * sf_k_scales;
    constexpr int mb_a = T::a_buffer_load_insts;
    constexpr int mb_b = T::b_buffer_load_insts;
    constexpr int mb = mb_a + mb_b;

    if constexpr (DIRECT_ONLY) {
        __shared__ int b_ready[T::prefetch_k_iter];
        if (opus::thread_id_x() < T::prefetch_k_iter) {
            b_ready[opus::thread_id_x()] = -1;
        }
        s_waitcnt_lgkmcnt(0_I);  // retire the init writes before other waves read them
        __builtin_amdgcn_s_barrier();
        if ((wave_id & 1) == 0) return;

        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 1>(lane_id, 0, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 1>(lane_id, 0);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 1>(lane_id, 0, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 1>(lane_id, 0);
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);
        auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a;
        typename decltype(mma)::vtype_b v_b;
        typename decltype(mma)::vtype_c v_c;
        clear(v_c);

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

        auto issue_a_tile = [&](int loop_k) {
            const int slot = wave_id_m * T::prefetch_k_iter + (loop_k % T::prefetch_k_iter);
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(loop_k, m, kg));
                });
            });
        };

        auto issue_b_tile = [&](int loop_k) {
            const int slot = loop_k % T::prefetch_k_iter;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(loop_k, n, kg));
                });
            });
        };

        auto load_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
            const int scale_base = loop_k * T::SCALES_PER_BK;
            v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
            opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                constexpr int ng = decltype(ng_c)::value;
                auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                });
            });
            s_waitcnt_vmcnt(0_I);
        };

        auto do_mma = [&](const auto& va, const auto& vb,
                          const vtype_sfa& v_sfa, const vtype_sfb& v_sfb) {
            __builtin_amdgcn_s_setprio(1);
            mma_mxscale_flatmm_accum<T>(mma, va, vb, v_sfa, v_sfb, v_c);
            __builtin_amdgcn_s_setprio(0);
        };

        issue_a_tile(0);
        if (wave_id_m == 0) {
            issue_b_tile(0);
        }
        for (int k = 0; k < loops; ++k) {
            const int a_slot = wave_id_m * T::prefetch_k_iter + (k % T::prefetch_k_iter);
            const int b_slot = k % T::prefetch_k_iter;
            s_waitcnt_vmcnt(0_I);
            if (wave_id_m == 0) {
                reinterpret_cast<volatile int*>(b_ready)[b_slot] = k;
            } else {
                volatile int* ready = reinterpret_cast<volatile int*>(b_ready);
                while (ready[b_slot] != k) {
                }
            }

            auto sa = make_smem(smem_a_at(a_slot, 0, 0));
            auto sb = make_smem(smem_b_at(b_slot, 0, 0));
            v_a = load<T::VEC_A>(sa, u_ra);
            v_b = load<T::VEC_B>(sb, u_rb);
            s_waitcnt_lgkmcnt(0_I);

            vtype_sfa v_sfa;
            vtype_sfb v_sfb;
            load_scales(k, v_sfa, v_sfb);
            if (k + 1 < loops) {
                issue_a_tile(k + 1);
                if (wave_id_m == 0) {
                    issue_b_tile(k + 1);
                }
            }
            do_mma(v_a, v_b, v_sfa, v_sfb);
        }

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
        D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                       + (size_t)batch_id * kargs.stride_c_batch
                       + (size_t)row * kargs.stride_c
                       + (size_t)col;
        auto g_out = make_gmem(out_ptr,
            (unsigned int)rows_avail * (unsigned int)kargs.stride_c * sizeof(D_OUT));
        store<T::VEC_C>(g_out, v_c, u_gc, 0);
        return;
    }

    // PRELOAD_SF_LDS: cooperative one-shot fill of the SFA + SFB panels into LDS.
    // Executed by all BLOCK_SIZE threads (both producer and consumer waves) before
    // the producer/consumer role split, with a barrier publishing the panels for
    // the consumer reads below. Grid-stride over the compact scalar byte counts.
    // OOB rows (partial-M tail) read 0 via g_sfa's num_records bound and are
    // never consumed. Bail if this split's K exceeds the compile-time LDS bound
    // (the dispatch never selects kid324 for such K, so this only guards misuse).
    if constexpr (PRELOAD_SF_LDS) {
        if (loops > SFA_K_TILES_MAX) return;
        const int tid = opus::thread_id_x();
        auto sm_sfa = make_smem(s_sfa_ptr);
        auto sm_sfb = make_smem(s_sfb_ptr);
        const int sfa_total = SFA_ROWS * sf_k_scales;
        const int sfb_total = T::N_SCALE_GROUPS * sf_k_scales;
        // Copy the widest chunk the panel geometry allows. Byte-at-a-time is 16
        // grid-stride iterations per thread for the 128-row SFA panel at K=4096
        // (kid325/326), and the fill sits in front of a barrier, so its latency is
        // exposed rather than overlapped. A chunk must not span two panel rows and
        // its source offset must stay naturally aligned, so the width has to divide
        // both sf_k_scales and the row stride; hence the short-K fallbacks.
        auto fill = [&](auto vec_c, auto sm, auto g, int stride, int total) {
            constexpr int VEC = decltype(vec_c)::value;
            for (int idx = tid * VEC; idx < total; idx += T::BLOCK_SIZE * VEC) {
                const int r  = idx / sf_k_scales;
                const int kt = idx - r * sf_k_scales;
                sm.template store<VEC>(load<VEC>(g, r * stride + kt), idx);
            }
        };
        auto fill_panel = [&](auto sm, auto g, int stride, int total) {
            const int widths = sf_k_scales | stride;
            if      ((widths & 15) == 0) fill(number<16>{}, sm, g, stride, total);
            else if ((widths & 3) == 0)  fill(number<4>{},  sm, g, stride, total);
            else                         fill(number<1>{},  sm, g, stride, total);
        };
        fill_panel(sm_sfa, g_sfa, kargs.stride_sfa, sfa_total);
        fill_panel(sm_sfb, g_sfb, kargs.stride_sfb, sfb_total);
        // vmcnt retires the global reads feeding the panel; lgkmcnt retires the
        // ds_writes that actually publish it. s_barrier does neither on its own.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);

        opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
            constexpr int p = decltype(p_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                    constexpr int m = decltype(m_c)::value;
                    async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                });
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(p, n, kg), u_gb, u_sb, b_offset(p, n, kg));
                });
            });
        });

        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
            s_waitcnt_vmcnt(number<mb * p>{});
            __builtin_amdgcn_s_barrier();
        });

        if constexpr (T::prefetch_k_iter == 3) {
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                int issue_k = i + 1;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                    });
                });
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        } else {
            for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                int issue_k = i + 2;
                int slot = issue_k % T::prefetch_k_iter;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                    });
                });
                s_waitcnt_vmcnt(number<2 * mb>{});
                __builtin_amdgcn_s_barrier();
            }
            s_waitcnt_vmcnt(number<mb>{});
            __builtin_amdgcn_s_barrier();
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    } else {
        // Consumer waves. tileM: two waves split M (wave_id_m in {0,1}, single
        // N column-block). tileN: two waves split N (single M-wave, wave_id_n
        // in {0,1}); each consumer reads its own 16-col B group from smem, and
        // the C partition / rb layout follow via T_N=2. wave_id_n_cons is 0 for
        // tileM, so the shared `n_block = wave_id_n_cons` smem-B base below is
        // bit-identical for the existing tileM kids.
        int wave_id_m = T::IS_TILE_N ? 0 : (wave_id / 2);
        int wave_id_n_cons = T::IS_TILE_N ? (wave_id / 2) : 0;
        // Consumer B smem N-group base. Each consumer N-wave owns COM_REP_N
        // contiguous 16-col load groups (rb reads num_blocks_n=COM_REP_N from
        // this base). tileM: wave_id_n_cons=0 -> nbc=0 (bit-identical).
        const int nbc = wave_id_n_cons * T::COM_REP_N;
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);
        auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);
        // LDS read layout for the preloaded SFA panel: same lane/wave mapping as
        // u_sfa but with the compact per-row K-scale count as the row stride.
        auto u_sfa_lds = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, sf_k_scales);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a0, v_a1;
        typename decltype(mma)::vtype_b v_b0, v_b1;
        typename decltype(mma)::vtype_c v_c;
        clear(v_c);

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;
        constexpr int ds_read_insts = T::a_ds_read_insts + T::b_ds_read_insts;

        auto load_scale_regs = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
            const int scale_base = loop_k * T::SCALES_PER_BK;
            if constexpr (PRELOAD_SF_LDS) {
                // Read this K-tile's scales from the preloaded LDS panels
                // (ds_read / lgkmcnt) instead of a per-tile global buffer_load.
                // Vec = SCALES_PER_BK so the contiguous per-M-row K bytes come in
                // as one dword (ds_read_b32) instead of SCALES_PER_BK byte reads.
                auto sm_a = make_smem(s_sfa_ptr + scale_base);
                v_sfa = load<T::SCALES_PER_BK>(sm_a, u_sfa_lds);
                opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    auto sm_b = make_smem(s_sfb_ptr + ng * sf_k_scales + scale_base);
                    auto sfb = load<T::SCALES_PER_BK>(sm_b, 0);
                    opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                    });
                });
            } else {
                // Vec = SCALES_PER_BK: the contiguous per-M-row K-scale bytes are
                // read as one dword (buffer_load_b32) rather than SCALES_PER_BK
                // separate buffer_load_ubyte. SFB already loads b32 the same way.
                v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
                opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                    opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                    });
                });
            }
        };

        auto do_scaled_mma = [&](const auto& va, const auto& vb,
                                 const vtype_sfa& v_sfa, const vtype_sfb& v_sfb) {
            if constexpr (PRELOAD_SF_LDS) {
                // Scales come from LDS now, so wait on lgkmcnt rather than vmcnt.
                // This also drains the just-issued next-buffer A/B ds_reads, still
                // a win against the global scale-load stall it replaces.
                s_waitcnt_lgkmcnt(0_I);
            } else {
                s_waitcnt_vmcnt(0_I);
            }
            __builtin_amdgcn_s_setprio(1);
            mma_mxscale_flatmm_accum<T>(mma, va, vb, v_sfa, v_sfb, v_c);
            __builtin_amdgcn_s_setprio(0);
        };

        auto scaled_mma = [&](const auto& va, const auto& vb, int loop_k) {
            vtype_sfa v_sfa;
            vtype_sfb v_sfb;
            load_scale_regs(loop_k, v_sfa, v_sfb);
            do_scaled_mma(va, vb, v_sfa, v_sfb);
        };

        auto wait_lgkm_then_scaled_mma =
            [&](const auto& va, const auto& vb, int loop_k, auto lgkm_cnt) {
                if constexpr (PREFETCH_SCALE) {
                    vtype_sfa v_sfa;
                    vtype_sfb v_sfb;
                    load_scale_regs(loop_k, v_sfa, v_sfb);
                    s_waitcnt_lgkmcnt(lgkm_cnt);
                    do_scaled_mma(va, vb, v_sfa, v_sfb);
                } else {
                    s_waitcnt_lgkmcnt(lgkm_cnt);
                    scaled_mma(va, vb, loop_k);
                }
            };

        __builtin_amdgcn_s_barrier();
        {
            auto sa0 = make_smem(smem_a_at(0, 0, 0));
            auto sb0 = make_smem(smem_b_at(0, nbc, 0));
            v_a0 = load<T::VEC_A>(sa0, u_ra);
            v_b0 = load<T::VEC_B>(sb0, u_rb);
        }

        opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
            constexpr int p = decltype(i_c)::value + 1;
            constexpr int cur = (p - 1) & 1;
            constexpr int nxt = p & 1;
            __builtin_amdgcn_s_barrier();
            auto sa_p = make_smem(smem_a_at(p, 0, 0));
            auto sb_p = make_smem(smem_b_at(p, nbc, 0));
            if constexpr (nxt == 0) {
                v_a0 = load<T::VEC_A>(sa_p, u_ra);
                v_b0 = load<T::VEC_B>(sb_p, u_rb);
            } else {
                v_a1 = load<T::VEC_A>(sa_p, u_ra);
                v_b1 = load<T::VEC_B>(sb_p, u_rb);
            }
            if constexpr (cur == 0) {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, p - 1, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, p - 1, number<ds_read_insts>{});
            }
        });

        constexpr int L = (T::prefetch_k_iter - 2) & 1;
        int k = T::prefetch_k_iter - 1;
        for (; k + 1 < loops - 1; k += 2) {
            __builtin_amdgcn_s_barrier();
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, nbc, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            if constexpr (L == 0) {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, k - 1, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, k - 1, number<ds_read_insts>{});
            }

            __builtin_amdgcn_s_barrier();
            {
                int slot = (k + 1) % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, nbc, 0));
                if constexpr (L == 0) {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            if constexpr (L == 0) {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, k, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, k, number<ds_read_insts>{});
            }
        }

        bool last_in_buf1 = (L != 0);
        if (k < loops - 1) {
            __builtin_amdgcn_s_barrier();
            {
                int slot = k % T::prefetch_k_iter;
                auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                auto sb_k = make_smem(smem_b_at(slot, nbc, 0));
                if constexpr (L == 0) {
                    v_a1 = load<T::VEC_A>(sa_k, u_ra);
                    v_b1 = load<T::VEC_B>(sb_k, u_rb);
                } else {
                    v_a0 = load<T::VEC_A>(sa_k, u_ra);
                    v_b0 = load<T::VEC_B>(sb_k, u_rb);
                }
            }
            if constexpr (L == 0) {
                wait_lgkm_then_scaled_mma(v_a0, v_b0, k - 1, number<ds_read_insts>{});
            } else {
                wait_lgkm_then_scaled_mma(v_a1, v_b1, k - 1, number<ds_read_insts>{});
            }
            last_in_buf1 = (L == 0);
            k++;
        }

        __builtin_amdgcn_s_barrier();
        int last_slot = (loops - 1) % T::prefetch_k_iter;
        auto sa_last = make_smem(smem_a_at(last_slot, 0, 0));
        auto sb_last = make_smem(smem_b_at(last_slot, nbc, 0));
        if (last_in_buf1) {
            v_a0 = load<T::VEC_A>(sa_last, u_ra);
            v_b0 = load<T::VEC_B>(sb_last, u_rb);
            wait_lgkm_then_scaled_mma(v_a1, v_b1, loops - 2, number<ds_read_insts>{});
            wait_lgkm_then_scaled_mma(v_a0, v_b0, loops - 1, 0_I);
        } else {
            v_a1 = load<T::VEC_A>(sa_last, u_ra);
            v_b1 = load<T::VEC_B>(sb_last, u_rb);
            wait_lgkm_then_scaled_mma(v_a0, v_b0, loops - 2, number<ds_read_insts>{});
            wait_lgkm_then_scaled_mma(v_a1, v_b1, loops - 1, 0_I);
        }

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(stride_c_main, 1_I), p_coord_c);
        // tileN with COM_REP_N>1: two consumer waves split B_N and each replicates
        // COM_REP_N N-repeats. The generic swap_ab C partition nests the register
        // N-repeat (expd_n) OUTSIDE the consumer-wave tile (tile_n), which transposes
        // the (wave, n-rep) -> column map (each wave writes a strided, interleaved
        // column set) whenever BOTH T_N>1 and COM_REP_N>1 -- the accumulators are
        // correct but land in swapped output columns. Consumer wave w computes n-rep
        // j from B column-group (w*COM_REP_N + j) (see nbc = wave_id_n_cons*COM_REP_N
        // + the num_blocks_n rb read), so store each n-rep slice to that contiguous
        // column group with a single-N-tile (E_N=1,T_N=1) layout and a scalar column
        // offset. tileM (T_N=1) and tileN COM_REP_N==1 keep the original single store
        // (SPLIT_N_STORE=false), so they stay bit-identical.
        constexpr bool SPLIT_N_STORE = T::IS_TILE_N && (T::COM_REP_N > 1);
        constexpr int C_LEN = decltype(mma)::mma_c_len;
        auto mma_c1 = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, 1, T::COM_REP_K>{}, seq<T::T_M, 1, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
        auto p_coord_c1 = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                           0, lane_id / mma.grpn_c);
        auto u_gc1 = partition_layout_c<T::VEC_C>(mma_c1,
            opus::make_tuple(stride_c_main, 1_I), p_coord_c1);
        auto store_c = [&](auto& g) {
            if constexpr (SPLIT_N_STORE) {
                opus::static_for<T::COM_REP_N>([&](auto j_c) {
                    constexpr int j = decltype(j_c)::value;
                    auto vj = opus::slice(v_c, opus::number<j * C_LEN>{},
                                          opus::number<j * C_LEN + C_LEN>{});
                    store<T::VEC_C>(g, vj, u_gc1,
                                    (wave_id_n_cons * T::COM_REP_N + j) * T::W_N);
                });
            } else {
                store<T::VEC_C>(g, v_c, u_gc, 0);
            }
        };
        if constexpr (!std::is_void_v<D_OUT>) {
            if (kargs.split_k == 1) {
                D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                               + (size_t)batch_id * kargs.stride_c_batch
                               + (size_t)row * kargs.stride_c
                               + (size_t)col;
                auto g_out = make_gmem(out_ptr,
                    (unsigned int)rows_avail * (unsigned int)kargs.stride_c * sizeof(D_OUT));
                store_c(g_out);
            } else {
                D_C* ws_c_ptr = reinterpret_cast<D_C*>(kargs.ws_handle->ptr)
                              + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                              + (size_t)batch_id * kargs.stride_ws_batch
                              + (size_t)row * kargs.stride_ws
                              + (size_t)col;
                auto g_c = make_gmem(ws_c_ptr);
                store_c(g_c);
            }
        } else {
            D_C* ws_c_ptr = reinterpret_cast<D_C*>(kargs.ws_handle->ptr)
                          + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                          + (size_t)batch_id * kargs.stride_ws_batch
                          + (size_t)row * kargs.stride_ws
                          + (size_t)col;
            auto g_c = make_gmem(ws_c_ptr);
            store_c(g_c);
        }
    }

    if constexpr (!std::is_void_v<D_OUT>) {
        if (kargs.split_k == 1) return;

        __shared__ int fused_do_reduce;
        if (opus::thread_id_x() == 0) {
            fused_do_reduce = 0;
        }
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __builtin_amdgcn_s_barrier();

        int* counters = reinterpret_cast<int*>(
            reinterpret_cast<char*>(kargs.ws_handle->ptr) + kargs.counter_offset_bytes);
        const int num_tiles = num_tiles_m * ceil_div(kargs.n, T::B_N);
        const int tile_id = batch_id * num_tiles + wgid;
        if (opus::thread_id_x() == 0) {
            const int old = __atomic_fetch_add(counters + tile_id, 1, __ATOMIC_ACQ_REL);
            fused_do_reduce = (old == kargs.split_k - 1);
        }
        // Every thread branches on this below, so lane 0's write has to be retired
        // (not merely issued) before the barrier lets the rest read it.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        if (fused_do_reduce) {
            const D_C* ws_base = reinterpret_cast<const D_C*>(kargs.ws_handle->ptr);
            D_OUT* out = reinterpret_cast<D_OUT*>(kargs.ptr_c);
            const size_t split_stride = (size_t)kargs.batch * (size_t)kargs.stride_ws_batch;
            for (int i = int(opus::thread_id_x()); i < T::B_M * T::B_N; i += T::BLOCK_SIZE) {
                const int mi = i / T::B_N;
                const int ni = i - mi * T::B_N;
                if (row + mi >= kargs.m) continue;  // skip OOB rows of a partial M tile
                float acc = 0.0f;
                const size_t base = (size_t)batch_id * (size_t)kargs.stride_ws_batch
                                  + (size_t)(row + mi) * (size_t)kargs.stride_ws
                                  + (size_t)(col + ni);
                for (int s = 0; s < kargs.split_k; ++s) {
                    acc += static_cast<float>(ws_base[(size_t)s * split_stride + base]);
                }
                const size_t out_idx = (size_t)batch_id * (size_t)kargs.stride_c_batch
                                     + (size_t)(row + mi) * (size_t)kargs.stride_c
                                     + (size_t)(col + ni);
                out[out_idx] = static_cast<D_OUT>(acc);
            }
            __builtin_amdgcn_s_barrier();
            if (opus::thread_id_x() == 0) {
                counters[tile_id] = 0;
            }
        }
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// Direct-store persistent M-outer kernel. Each WG owns one N tile and a small
// run of M tiles, reusing the same B tile stream across the outer loop without
// increasing the per-tile accumulator footprint.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_splitk_mouter_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;

    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int m_per_wg = kargs.split_k;
    int bid = opus::block_id_x();
    constexpr int NUM_XCD = 8;
    int xcd_id = __builtin_amdgcn_readfirstlane(bid % NUM_XCD);
    int pos_xcd = __builtin_amdgcn_readfirstlane(bid / NUM_XCD);
    int tile_n_id = __builtin_amdgcn_readfirstlane(pos_xcd % num_tiles_n);
    int m_grp_local = __builtin_amdgcn_readfirstlane(pos_xcd / num_tiles_n);
    int m_grp = __builtin_amdgcn_readfirstlane(xcd_id * kargs.stride_ws_batch + m_grp_local);
    if (m_grp >= kargs.stride_ws) return;
    int tile_m_lo = m_grp * m_per_wg;
    int tile_m_hi = tile_m_lo + m_per_wg;
    int col = tile_n_id * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    int role = ((wave_id & 1) ^ ((bid >> 8) & 1));

    const int loops = kargs.k / T::B_K;
    if (loops < T::prefetch_k_iter) return;

    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch + (size_t)col * kargs.stride_b);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[T::prefetch_k_iter * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    constexpr int mb_a = T::a_buffer_load_insts;
    constexpr int mb_b = T::b_buffer_load_insts;
    constexpr int mb = mb_a + mb_b;

    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);

        for (int tile_m = tile_m_lo; tile_m < tile_m_hi && tile_m < num_tiles_m; ++tile_m) {
            int row = tile_m * T::B_M;
            auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                                 + (size_t)batch_id * kargs.stride_a_batch
                                 + (size_t)row * kargs.stride_a);
            auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
                return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
                     + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
            };

            opus::static_for<T::prefetch_k_iter>([&](auto p_c) {
                constexpr int p = decltype(p_c)::value;
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(p, m, kg), u_ga, u_sa, a_offset(p, m, kg));
                    });
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                        constexpr int n = decltype(n_c)::value;
                        async_load<T::VEC_B>(g_b, smem_b_at(p, n, kg), u_gb, u_sb, b_offset(p, n, kg));
                    });
                });
            });

            opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
                constexpr int p = T::prefetch_k_iter - 1 - decltype(i_c)::value;
                s_waitcnt_vmcnt(number<mb * p>{});
                __builtin_amdgcn_s_barrier();
            });

            if constexpr (T::prefetch_k_iter == 3) {
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
                for (int i = T::prefetch_k_iter - 1; i < loops - 1; i++) {
                    int issue_k = i + 1;
                    int slot = issue_k % T::prefetch_k_iter;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                            constexpr int m = decltype(m_c)::value;
                            async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                        });
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                            constexpr int n = decltype(n_c)::value;
                            async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                        });
                    });
                    s_waitcnt_vmcnt(number<mb>{});
                    __builtin_amdgcn_s_barrier();
                }
                s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            } else {
                for (int i = T::prefetch_k_iter - 2; i < loops - 2; i++) {
                    int issue_k = i + 2;
                    int slot = issue_k % T::prefetch_k_iter;
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                            constexpr int m = decltype(m_c)::value;
                            async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(issue_k, m, kg));
                        });
                        opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                            constexpr int n = decltype(n_c)::value;
                            async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(issue_k, n, kg));
                        });
                    });
                    s_waitcnt_vmcnt(number<2 * mb>{});
                    __builtin_amdgcn_s_barrier();
                }
                s_waitcnt_vmcnt(number<mb>{});
                __builtin_amdgcn_s_barrier();
                s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            }
            __builtin_amdgcn_s_barrier();
        }
    } else {
        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a0, v_a1;
        typename decltype(mma)::vtype_b v_b0, v_b1;
        typename decltype(mma)::vtype_c v_c;

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;
        constexpr int ds_read_insts = T::a_ds_read_insts + T::b_ds_read_insts;

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);

        for (int tile_m = tile_m_lo; tile_m < tile_m_hi && tile_m < num_tiles_m; ++tile_m) {
            int row = tile_m * T::B_M;
            auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                                   + (size_t)batch_id * kargs.stride_sfa_batch
                                   + (size_t)row * kargs.stride_sfa);
            clear(v_c);

            auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);
            auto scaled_mma = [&](const auto& va, const auto& vb, int loop_k) {
                const int scale_base = loop_k * T::SCALES_PER_BK;
                vtype_sfa v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
                vtype_sfb v_sfb;
                opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                    constexpr int ng = decltype(ng_c)::value;
                    auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                    opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                        constexpr int kg = decltype(kg_c)::value;
                        v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                    });
                });
                if constexpr (!SKIP_SCALE_WAIT) {
                    s_waitcnt_vmcnt(0_I);
                }
                __builtin_amdgcn_s_setprio(1);
                mma_mxscale_flatmm_accum<T>(mma, va, vb, v_sfa, v_sfb, v_c);
                __builtin_amdgcn_s_setprio(0);
            };

            __builtin_amdgcn_s_barrier();
            {
                auto sa0 = make_smem(smem_a_at(0, 0, 0));
                auto sb0 = make_smem(smem_b_at(0, 0, 0));
                v_a0 = load<T::VEC_A>(sa0, u_ra);
                v_b0 = load<T::VEC_B>(sb0, u_rb);
            }

            opus::static_for<T::prefetch_k_iter - 2>([&](auto i_c) {
                constexpr int p = decltype(i_c)::value + 1;
                constexpr int cur = (p - 1) & 1;
                constexpr int nxt = p & 1;
                __builtin_amdgcn_s_barrier();
                auto sa_p = make_smem(smem_a_at(p, 0, 0));
                auto sb_p = make_smem(smem_b_at(p, 0, 0));
                if constexpr (nxt == 0) {
                    v_a0 = load<T::VEC_A>(sa_p, u_ra);
                    v_b0 = load<T::VEC_B>(sb_p, u_rb);
                } else {
                    v_a1 = load<T::VEC_A>(sa_p, u_ra);
                    v_b1 = load<T::VEC_B>(sb_p, u_rb);
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (cur == 0) scaled_mma(v_a0, v_b0, p - 1);
                else                    scaled_mma(v_a1, v_b1, p - 1);
            });

            constexpr int L = (T::prefetch_k_iter - 2) & 1;
            int k = T::prefetch_k_iter - 1;
            for (; k + 1 < loops - 1; k += 2) {
                __builtin_amdgcn_s_barrier();
                {
                    int slot = k % T::prefetch_k_iter;
                    auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                    auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                    if constexpr (L == 0) {
                        v_a1 = load<T::VEC_A>(sa_k, u_ra);
                        v_b1 = load<T::VEC_B>(sb_k, u_rb);
                    } else {
                        v_a0 = load<T::VEC_A>(sa_k, u_ra);
                        v_b0 = load<T::VEC_B>(sb_k, u_rb);
                    }
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (L == 0) scaled_mma(v_a0, v_b0, k - 1);
                else                  scaled_mma(v_a1, v_b1, k - 1);

                __builtin_amdgcn_s_barrier();
                {
                    int slot = (k + 1) % T::prefetch_k_iter;
                    auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                    auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                    if constexpr (L == 0) {
                        v_a0 = load<T::VEC_A>(sa_k, u_ra);
                        v_b0 = load<T::VEC_B>(sb_k, u_rb);
                    } else {
                        v_a1 = load<T::VEC_A>(sa_k, u_ra);
                        v_b1 = load<T::VEC_B>(sb_k, u_rb);
                    }
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (L == 0) scaled_mma(v_a1, v_b1, k);
                else                  scaled_mma(v_a0, v_b0, k);
            }

            bool last_in_buf1 = (L != 0);
            if (k < loops - 1) {
                __builtin_amdgcn_s_barrier();
                {
                    int slot = k % T::prefetch_k_iter;
                    auto sa_k = make_smem(smem_a_at(slot, 0, 0));
                    auto sb_k = make_smem(smem_b_at(slot, 0, 0));
                    if constexpr (L == 0) {
                        v_a1 = load<T::VEC_A>(sa_k, u_ra);
                        v_b1 = load<T::VEC_B>(sb_k, u_rb);
                    } else {
                        v_a0 = load<T::VEC_A>(sa_k, u_ra);
                        v_b0 = load<T::VEC_B>(sb_k, u_rb);
                    }
                }
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                if constexpr (L == 0) scaled_mma(v_a0, v_b0, k - 1);
                else                  scaled_mma(v_a1, v_b1, k - 1);
                last_in_buf1 = (L == 0);
                k++;
            }

            __builtin_amdgcn_s_barrier();
            int last_slot = (loops - 1) % T::prefetch_k_iter;
            auto sa_last = make_smem(smem_a_at(last_slot, 0, 0));
            auto sb_last = make_smem(smem_b_at(last_slot, 0, 0));
            if (last_in_buf1) {
                v_a0 = load<T::VEC_A>(sa_last, u_ra);
                v_b0 = load<T::VEC_B>(sb_last, u_rb);
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                scaled_mma(v_a1, v_b1, loops - 2);
                s_waitcnt_lgkmcnt(0_I);
                scaled_mma(v_a0, v_b0, loops - 1);
            } else {
                v_a1 = load<T::VEC_A>(sa_last, u_ra);
                v_b1 = load<T::VEC_B>(sb_last, u_rb);
                s_waitcnt_lgkmcnt(number<ds_read_insts>{});
                scaled_mma(v_a0, v_b0, loops - 2);
                s_waitcnt_lgkmcnt(0_I);
                scaled_mma(v_a1, v_b1, loops - 1);
            }

            D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                           + (size_t)batch_id * kargs.stride_c_batch
                           + (size_t)row * kargs.stride_c
                           + (size_t)col;
            auto g_out = make_gmem(out_ptr);
            store<T::VEC_C>(g_out, v_c, u_gc, 0);
            __builtin_amdgcn_s_barrier();
        }
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// -- M-tile interleaved direct-store kernel (correctness-first v1) ------------
//
// Each WG owns one N tile and MI=2 consecutive M tiles that share the SAME B
// tile stream. Both M tiles' A operands are resident in LDS simultaneously
// (smem_a0 / smem_a1) while B is loaded once (smem_b). Per K iteration the
// consumer issues MFMA for tile0 then tile1 back-to-back into two independent
// accumulators, so the MFMA instruction stream is ~MIx longer across a single
// prologue/epilogue, and B global traffic is halved.
//
// v1 is single-buffered (load-then-compute, barrier-bracketed) for provable
// deadlock-freedom and correctness; the software-pipelined perf version is a
// follow-up. Producer waves (role 0) load; consumer waves (role 1) MFMA.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void gemm_a8w8_mxscale_flatmm_minterleave_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;

    constexpr int MI = 2;                 // M tiles interleaved per WG
    constexpr int SB = 2;                 // double LDS buffer per operand (prefetch k+1)

    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    int bid = opus::block_id_x();
    constexpr int NUM_XCD = 8;
    int xcd_id = __builtin_amdgcn_readfirstlane(bid % NUM_XCD);
    int pos_xcd = __builtin_amdgcn_readfirstlane(bid / NUM_XCD);
    int tile_n_id = __builtin_amdgcn_readfirstlane(pos_xcd % num_tiles_n);
    int m_grp_local = __builtin_amdgcn_readfirstlane(pos_xcd / num_tiles_n);
    int m_grp = __builtin_amdgcn_readfirstlane(xcd_id * kargs.stride_ws_batch + m_grp_local);
    if (m_grp >= kargs.stride_ws) return;
    int tile_m0 = m_grp * MI;
    if (tile_m0 + MI > num_tiles_m) return;  // host guarantees M % (MI*B_M) == 0
    int col = tile_n_id * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    int role = (wave_id & 1);

    const int loops = kargs.k / T::B_K;
    if (loops < T::prefetch_k_iter) return;

    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch + (size_t)col * kargs.stride_b);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[MI * SB * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[SB * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int mi, int slot_k, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + (((mi * SB + slot_k) * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot_k, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot_k * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    if (role == 0) {
        int wave_id_prod = wave_id / 2;
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_prod, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_prod);

        auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
            return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
                 + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        };

        const D_A* base_a = reinterpret_cast<const D_A*>(kargs.ptr_a)
                            + (size_t)batch_id * kargs.stride_a_batch;

        auto issue_loads = [&](int k, int slot) {
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                opus::static_for<MI>([&](auto mi_c) {
                    constexpr int mi = decltype(mi_c)::value;
                    auto g_a = make_gmem(base_a + (size_t)(tile_m0 + mi) * T::B_M * kargs.stride_a);
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(mi, slot, m, kg), u_ga, u_sa, a_offset(k, m, kg));
                    });
                });
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(k, n, kg));
                });
            });
        };

        // Prologue: preload slot 0 (K-tile 0), then stream while consumers MFMA.
        issue_loads(0, 0);
        s_waitcnt_vmcnt(0_I);
        for (int k = 0; k < loops; ++k) {
            __builtin_amdgcn_s_barrier();   // R(k): slot k ready
            if (k + 1 < loops) {
                issue_loads(k + 1, (k + 1) % SB);
                s_waitcnt_vmcnt(0_I);       // slot k+1 fully loaded before releasing slot k
            }
            __builtin_amdgcn_s_barrier();   // F(k): consumer done reading slot k
        }
    } else {
        int wave_id_m = wave_id / 2;
        int wave_id_n_cons = 0;
        auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
        auto u_rb = make_layout_rb_mxsk<T>(lane_id);

        auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
            seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
            seq<T::T_M, T::T_N, T::T_K>{},
            seq<T::W_M, T::W_N, T::W_K>{},
            mfma_adaptor_swap_ab{});

        typename decltype(mma)::vtype_a v_a[MI];
        typename decltype(mma)::vtype_b v_b;
        typename decltype(mma)::vtype_c v_c[MI];

        using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
        using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

        auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                          wave_id_n_cons, lane_id / mma.grpn_c);
        auto u_gc = partition_layout_c<T::VEC_C>(mma,
            opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);

        // Per-tile A-scale gmem + layout.
        auto g_sfa = [&](int mi) {
            return make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                             + (size_t)batch_id * kargs.stride_sfa_batch
                             + (size_t)(tile_m0 + mi) * T::B_M * kargs.stride_sfa);
        };
        auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

        opus::static_for<MI>([&](auto mi_c) { clear(v_c[decltype(mi_c)::value]); });

        for (int k = 0; k < loops; ++k) {
            __builtin_amdgcn_s_barrier();   // R(k): wait producer data ready
            const int slot = k % SB;
            auto sb0 = make_smem(smem_b_at(slot, 0, 0));
            v_b = load<T::VEC_B>(sb0, u_rb);
            opus::static_for<MI>([&](auto mi_c) {
                constexpr int mi = decltype(mi_c)::value;
                auto sa = make_smem(smem_a_at(mi, slot, 0, 0));
                v_a[mi] = load<T::VEC_A>(sa, u_ra);
            });
            s_waitcnt_lgkmcnt(0_I);

            const int scale_base = k * T::SCALES_PER_BK;
            vtype_sfb v_sfb;
            opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
                constexpr int ng = decltype(ng_c)::value;
                auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
                opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                    constexpr int kg = decltype(kg_c)::value;
                    v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
                });
            });
            opus::static_for<MI>([&](auto mi_c) {
                constexpr int mi = decltype(mi_c)::value;
                auto gsfa = g_sfa(mi);
                vtype_sfa v_sfa = load<T::SCALES_PER_BK>(gsfa, u_sfa, scale_base);
                if constexpr (!SKIP_SCALE_WAIT) s_waitcnt_vmcnt(0_I);
                __builtin_amdgcn_s_setprio(1);
                mma_mxscale_flatmm_accum<T>(mma, v_a[mi], v_b, v_sfa, v_sfb, v_c[mi]);
                __builtin_amdgcn_s_setprio(0);
            });
            __builtin_amdgcn_s_barrier();   // (C2) done reading slot
        }

        opus::static_for<MI>([&](auto mi_c) {
            constexpr int mi = decltype(mi_c)::value;
            int row = (tile_m0 + mi) * T::B_M;
            D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                           + (size_t)batch_id * kargs.stride_c_batch
                           + (size_t)row * kargs.stride_c
                           + (size_t)col;
            auto g_out = make_gmem(out_ptr);
            store<T::VEC_C>(g_out, v_c[mi], u_gc, 0);
        });
    }
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// 8-wave split-accumulator direct-store kernel.
//
// Logical tile: 128x256x128. Internally this is two independent 128x128
// accumulator groups computed by consumer waves {4,5} and {6,7}. Producer
// waves {0,1} load shared A and B phase 0, producer waves {2,3} load B phase 1.
// This keeps each consumer's v_c identical to the proven 128x128 WG1 kernel
// while halving the logical N workgroup count.
template<typename Traits, typename D_OUT>
__global__ __launch_bounds__(512, 1)
void gemm_a8w8_mxscale_flatmm_splitk_wave8n2_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(T::B_M == 128 && T::B_N == 128 && T::B_K == 128,
                  "wave8n2 builds a logical 128x256 tile from 128x128 traits");

    constexpr int N_PHASES = 2;
    int wgid = opus::block_id_x();
    const int num_tiles_m = ceil_div(kargs.m, T::B_M);
    int row = (wgid % num_tiles_m) * T::B_M;
    int col_base = (wgid / num_tiles_m) * (T::B_N * N_PHASES);
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int loops = kargs.k / T::B_K;
    if (loops < 1) return;

    constexpr int WAVE8N2_PREFETCH_SLOTS = 2;
    __shared__ char smem_a[WAVE8N2_PREFETCH_SLOTS * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[WAVE8N2_PREFETCH_SLOTS * N_PHASES * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + ((slot * T::NUM_LOAD_GROUPS_PER_BM + m_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot, int phase, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + (((slot * N_PHASES + phase) * T::NUM_LOAD_GROUPS_PER_BN + n_block)
               * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    if (wave_id < 4) {
        const int phase = wave_id / 2;
        const int prod_wave = wave_id & 1;
        auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                             + (size_t)batch_id * kargs.stride_a_batch
                             + (size_t)row * kargs.stride_a);
        auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                             + (size_t)batch_id * kargs.stride_b_batch
                             + (size_t)(col_base + phase * T::B_N) * kargs.stride_b);
        auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, prod_wave, kargs.stride_a);
        auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, prod_wave);
        auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, prod_wave, kargs.stride_b);
        auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, prod_wave);

        auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
            return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
                 + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        };
        auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
            return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
                 + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
        };

        auto issue_tile = [&](int loop_k) {
            const int slot = loop_k & 1;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                if (phase == 0) {
                    opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                        constexpr int m = decltype(m_c)::value;
                        async_load<T::VEC_A>(g_a, smem_a_at(slot, m, kg), u_ga, u_sa, a_offset(loop_k, m, kg));
                    });
                }
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, phase, n, kg), u_gb, u_sb, b_offset(loop_k, n, kg));
                });
            });
        };

        issue_tile(0);
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        for (int k = 0; k < loops; ++k) {
            if (k + 1 < loops) {
                issue_tile(k + 1);
            }
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        return;
    }

    const int consumer = wave_id - 4;
    const int phase = consumer / 2;
    const int wave_id_m = consumer & 1;
    const int col = col_base + phase * T::B_N;

    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
    auto u_rb = make_layout_rb_mxsk<T>(lane_id);
    auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

    __builtin_amdgcn_s_barrier();
    {
        auto sa = make_smem(smem_a_at(0, 0, 0));
        auto sb = make_smem(smem_b_at(0, phase, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
        v_b = load<T::VEC_B>(sb, u_rb);
        s_waitcnt_lgkmcnt(0_I);
    }

    for (int k = 0; k < loops; ++k) {
        const int scale_base = k * T::SCALES_PER_BK;
        vtype_sfa v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
        vtype_sfb v_sfb;
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
        s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_s_setprio(1);
        mma_mxscale_flatmm_accum<T>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        if (k + 1 < loops) {
            const int slot = (k + 1) & 1;
            auto sa = make_smem(smem_a_at(slot, 0, 0));
            auto sb = make_smem(smem_b_at(slot, phase, 0, 0));
            v_a = load<T::VEC_A>(sa, u_ra);
            v_b = load<T::VEC_B>(sb, u_rb);
            s_waitcnt_lgkmcnt(0_I);
        }
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      0, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma,
        opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                   + (size_t)batch_id * kargs.stride_c_batch
                   + (size_t)row * kargs.stride_c
                   + (size_t)col;
    auto g_out = make_gmem(out_ptr);
    store<T::VEC_C>(g_out, v_c, u_gc, 0);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

// 4-wave self-load split-accumulator direct-store kernel with M reuse.
//
// Logical tile: 256x128x128. Two independent 128x128 accumulator groups cover
// adjacent M tiles and share one B tile. This targets large-M shapes where B
// reuse matters more than reducing N workgroups.
template<typename Traits, typename D_OUT, bool SKIP_SCALE_WAIT = false,
         bool PACK_SCALE_ON_DEMAND = false>
__global__ __launch_bounds__(256, 1)
void gemm_a8w8_mxscale_flatmm_splitk_wave4m2_selfload_kernel(opus_gemm_scale_splitk_kargs_gfx950 kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_ACC = typename T::D_ACC;
    using D_SF = typename T::D_SF;
    static_assert(T::B_M == 128 && T::B_N == 128 && T::B_K == 128,
                  "wave4m2 selfload builds a logical 256x128 tile from 128x128 traits");

    constexpr int M_PHASES = 2;
    constexpr int PREFETCH_SLOTS = 2;
    int wgid = opus::block_id_x();
    const int logical_b_m = T::B_M * M_PHASES;
    const int num_tiles_m = ceil_div(kargs.m, logical_b_m);
    int row_base = (wgid % num_tiles_m) * logical_b_m;
    int col = (wgid / num_tiles_m) * T::B_N;
    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    const int loops = kargs.k / T::B_K;
    if (loops < 1) return;

    const int m_phase = wave_id / 2;
    const int wave_id_m = wave_id & 1;
    const int row = row_base + m_phase * T::B_M;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a)
                         + (size_t)batch_id * kargs.stride_a_batch
                         + (size_t)row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b)
                         + (size_t)batch_id * kargs.stride_b_batch
                         + (size_t)col * kargs.stride_b);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa)
                           + (size_t)batch_id * kargs.stride_sfa_batch
                           + (size_t)row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb)
                           + (size_t)batch_id * kargs.stride_sfb_batch
                           + (size_t)(col / T::GROUP_N) * kargs.stride_sfb);

    __shared__ char smem_a[PREFETCH_SLOTS * M_PHASES * T::NUM_LOAD_GROUPS_PER_BM
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];
    __shared__ char smem_b[PREFETCH_SLOTS * T::NUM_LOAD_GROUPS_PER_BN
                           * T::NUM_LOAD_GROUPS_PER_BK * T::smem_per_group_load_size];

    auto smem_a_at = [&](int slot, int phase, int m_block, int k_group) -> D_A* {
        return reinterpret_cast<D_A*>(smem_a
            + (((slot * M_PHASES + phase) * T::NUM_LOAD_GROUPS_PER_BM + m_block)
               * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };
    auto smem_b_at = [&](int slot, int n_block, int k_group) -> D_B* {
        return reinterpret_cast<D_B*>(smem_b
            + ((slot * T::NUM_LOAD_GROUPS_PER_BN + n_block) * T::NUM_LOAD_GROUPS_PER_BK + k_group)
              * T::smem_per_group_load_size);
    };

    auto a_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_M * kargs.stride_a
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };
    auto b_offset = [&](int loop_k_idx, int group_load_idx, int k_group) {
        return group_load_idx * T::LOAD_GROUP_N * kargs.stride_b
             + (loop_k_idx * T::NUM_LOAD_GROUPS_PER_BK + k_group) * T::LOAD_GROUP_K;
    };

    auto u_ga = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_m, kargs.stride_a);
    auto u_sa = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_m);
    auto u_gb = make_layout_gmem_group_load_mxsk<T, 2>(lane_id, wave_id_m, kargs.stride_b);
    auto u_sb = make_layout_smem_group_load_mxsk<T, 2>(lane_id, wave_id_m);
    auto u_ra = make_layout_ra_mxsk<T>(lane_id, wave_id_m);
    auto u_rb = make_layout_rb_mxsk<T>(lane_id);
    auto u_sfa = make_layout_sfa_mxsk<T>(lane_id, wave_id_m, kargs.stride_sfa);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::COM_REP_M, T::COM_REP_N, T::COM_REP_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c;
    clear(v_c);
    using vtype_sfa = vector_t<D_SF, T::COM_REP_M * T::SCALES_PER_BK>;
    using vtype_sfb = vector_t<D_SF, T::N_SCALE_GROUPS * T::SCALES_PER_BK>;

    auto issue_tile = [&](int loop_k) {
        const int slot = loop_k & 1;
        opus::static_for<T::NUM_LOAD_GROUPS_PER_BK>([&](auto kg_c) {
            constexpr int kg = decltype(kg_c)::value;
            opus::static_for<T::NUM_LOAD_GROUPS_PER_BM>([&](auto m_c) {
                constexpr int m = decltype(m_c)::value;
                async_load<T::VEC_A>(g_a, smem_a_at(slot, m_phase, m, kg), u_ga, u_sa, a_offset(loop_k, m, kg));
            });
            if (m_phase == 0) {
                opus::static_for<T::NUM_LOAD_GROUPS_PER_BN>([&](auto n_c) {
                    constexpr int n = decltype(n_c)::value;
                    async_load<T::VEC_B>(g_b, smem_b_at(slot, n, kg), u_gb, u_sb, b_offset(loop_k, n, kg));
                });
            }
        });
    };

    auto load_scales = [&](int loop_k, vtype_sfa& v_sfa, vtype_sfb& v_sfb) {
        const int scale_base = loop_k * T::SCALES_PER_BK;
        v_sfa = load<T::SCALES_PER_BK>(g_sfa, u_sfa, scale_base);
        opus::static_for<T::N_SCALE_GROUPS>([&](auto ng_c) {
            constexpr int ng = decltype(ng_c)::value;
            auto sfb = load<T::SCALES_PER_BK>(g_sfb, ng * kargs.stride_sfb + scale_base);
            opus::static_for<T::SCALES_PER_BK>([&](auto kg_c) {
                constexpr int kg = decltype(kg_c)::value;
                v_sfb[ng * T::SCALES_PER_BK + kg] = sfb[kg];
            });
        });
        if constexpr (!SKIP_SCALE_WAIT) {
            s_waitcnt_vmcnt(0_I);
        }
    };

    issue_tile(0);
    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    {
        auto sa = make_smem(smem_a_at(0, m_phase, 0, 0));
        auto sb = make_smem(smem_b_at(0, 0, 0));
        v_a = load<T::VEC_A>(sa, u_ra);
        v_b = load<T::VEC_B>(sb, u_rb);
        s_waitcnt_lgkmcnt(0_I);
    }

    for (int k = 0; k < loops; ++k) {
        vtype_sfa v_sfa;
        vtype_sfb v_sfb;
        load_scales(k, v_sfa, v_sfb);
        if (k + 1 < loops) {
            issue_tile(k + 1);
        }
        __builtin_amdgcn_s_setprio(1);
        if constexpr (PACK_SCALE_ON_DEMAND) {
            mma_mxscale_flatmm_accum_on_demand<T>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
        } else {
            mma_mxscale_flatmm_accum<T>(mma, v_a, v_b, v_sfa, v_sfb, v_c);
        }
        __builtin_amdgcn_s_setprio(0);
        if (k + 1 < loops) {
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            const int slot = (k + 1) & 1;
            auto sa = make_smem(smem_a_at(slot, m_phase, 0, 0));
            auto sb = make_smem(smem_b_at(slot, 0, 0));
            v_a = load<T::VEC_A>(sa, u_ra);
            v_b = load<T::VEC_B>(sb, u_rb);
            s_waitcnt_lgkmcnt(0_I);
        }
    }

    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      0, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(mma,
        opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    D_OUT* out_ptr = reinterpret_cast<D_OUT*>(kargs.ptr_c)
                   + (size_t)batch_id * kargs.stride_c_batch
                   + (size_t)row * kargs.stride_c
                   + (size_t)col;
    auto g_out = make_gmem(out_ptr);
    store<T::VEC_C>(g_out, v_c, u_gc, 0);
#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
}

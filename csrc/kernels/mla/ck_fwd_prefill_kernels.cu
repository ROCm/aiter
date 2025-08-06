// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/core/tensor/tile_scatter_gather.hpp>
#include "ck_fwd_kernels_params.hpp"
#include "ck_fwd_prefill_kernels_pipelines.hpp"

// =====================================================================================================================
// Kernel Functions
//

template <typename Traits>
CK_TILE_DEVICE static auto GetTileIndex(const int32_t num_splits)
{
    const auto f = [](int32_t dividend, int32_t divisor) {
        int32_t quotient = dividend / divisor;
        int32_t modulus  = dividend - quotient * divisor;
        return ck_tile::make_tuple(quotient, modulus);
    };

    const auto [mid, split_id] = f(blockIdx.x, num_splits);
    const int32_t hid          = blockIdx.y;
    const int32_t bid          = blockIdx.z;

    return ck_tile::make_tuple(mid, split_id, hid, bid);
}

template <typename Policy, int32_t HiddenDim, typename scalar_t = typename Policy::InOutType>
CK_TILE_DEVICE static auto MakeQDram(const scalar_t* p_data,
                                     const int32_t size_s,
                                     const int32_t stride_s,
                                     const int32_t hq_hk_ratio,
                                     const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto q_dram_naive = [&] {
        if constexpr(Traits::kEnableXqa)
        {
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s, hq_hk_ratio, HiddenDim),
                ck_tile::make_tuple(stride_s, stride_h, 1),
                ck_tile::number<Policy::GetAlignmentQ()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s, hq_hk_ratio)),
                    ck_tile::make_pass_through_transform(HiddenDim)),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s, HiddenDim),
                ck_tile::make_tuple(stride_s, 1),
                ck_tile::number<Policy::GetAlignmentQ()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        q_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockK0>{}),
        ck_tile::sequence<false, Traits::kPadHeadDimQ>{});
}

template <typename Policy, int32_t HiddenDim, typename scalar_t = typename Policy::InOutType>
CK_TILE_DEVICE static auto MakeKDram(
    const scalar_t* p_data,
    const int32_t   height,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    const auto k_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data, // will update this pointer if using paged-kvcache
        ck_tile::make_tuple(height, HiddenDim),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentK()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        k_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{}),
        ck_tile::sequence<true, Traits::kPadHeadDimQ>{});
}

template <typename Policy, typename scalar_t = typename Policy::InOutType>
CK_TILE_DEVICE static auto MakeVDram(
    const scalar_t* p_data,
    const int32_t   length,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    // Assuming Value is row-major just like Key.
    const auto v_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data, // will update this pointer if using paged-kvcache
        ck_tile::make_tuple(length, Traits::kSizeDV),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentV()>{},
        ck_tile::number<1>{});

    const auto v_dram_transposed = ck_tile::transform_tensor_view(
        v_dram_naive,
        ck_tile::make_tuple(ck_tile::make_pass_through_transform(Traits::kSizeDV),
                            ck_tile::make_pass_through_transform(length)),
        ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

    return ck_tile::pad_tensor_view(
        v_dram_transposed,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN1>{},
                            ck_tile::number<Traits::kBlockK1>{}),
        ck_tile::sequence<Traits::kPadHeadDimV, Traits::kPadSeqLenK>{});
}

template <typename Policy, typename Lengths, typename scalar_t>
CK_TILE_DEVICE static auto MakeLseAccDram(scalar_t* p_data,
                                          const Lengths& window_lengths,
                                          const int32_t size_s,
                                          const int32_t hq_hk_ratio,
                                          const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto lse_acc_dram_naive = [&] {
        if constexpr(Traits::kEnableXqa)
        {
            // transpose + merge: (hq_hk_ratio, seqlen_q) -> (seqlenq * hq_hk_ratio)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(hq_hk_ratio, size_s),
                ck_tile::make_tuple(stride_h, 1),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s, hq_hk_ratio))),
                ck_tile::make_tuple(ck_tile::sequence<1, 0>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s),
                ck_tile::make_tuple(1),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        lse_acc_dram_naive,
        window_lengths,
        ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutAccDram(scalar_t* p_data,
                                          const int32_t size_s,
                                          const int32_t stride_s,
                                          const int32_t hq_hk_ratio,
                                          const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto o_acc_dram_naive = [&] {
        if constexpr(Traits::kEnableXqa)
        {
            // merge: (seqlen_q, hq_hk_ratio, headdim) -> (seqlen_q*hq_hk_ratio, headdim)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s, hq_hk_ratio, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, stride_h, 1),
                ck_tile::number<Policy::GetAlignmentOacc()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s, hq_hk_ratio)),
                    ck_tile::make_pass_through_transform(Traits::kSizeDV)),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, 1),
                ck_tile::number<Policy::GetAlignmentOacc()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        o_acc_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <typename Policy, typename Lengths, typename scalar_t>
CK_TILE_DEVICE static auto MakeLseDram(scalar_t* p_data,
                                       const Lengths& window_lenghts,
                                       const int32_t size_s,
                                       const int32_t hq_hk_ratio,
                                       const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto lse_dram_naive = [&] {
        if constexpr(Traits::kEnableXqa)
        {
            // transpose + merge: (hq_hk_ratio, seqlen_q) -> (seqlenq * hq_hk_ratio)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(hq_hk_ratio, size_s),
                ck_tile::make_tuple(stride_h, 1),
                ck_tile::number<Policy::GetAlignmentLse()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s, hq_hk_ratio))),
                ck_tile::make_tuple(ck_tile::sequence<1, 0>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}));
        }
        else
        {

            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s),
                ck_tile::make_tuple(1),
                ck_tile::number<Policy::GetAlignmentLse()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        lse_dram_naive, window_lenghts, ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutDram(scalar_t* p_data,
                                       const int32_t size_s,
                                       const int32_t stride_s,
                                       const int32_t hq_hk_ratio,
                                       const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto o_dram_naive = [&] {
        if constexpr(Traits::kEnableXqa)
        {
            // merge: (seqlen_q, hq_hk_ratio, headdim) -> (seqlen_q * hq_hk_ratio, headdim)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s, hq_hk_ratio, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, stride_h, 1),
                ck_tile::number<Policy::GetAlignmentO()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s, hq_hk_ratio)),
                    ck_tile::make_pass_through_transform(Traits::kSizeDV)),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, 1),
                ck_tile::number<Policy::GetAlignmentO()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        o_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}
// =====================================================================================================================
// Kernel Entry
//

template <typename Traits, typename scalar_t, typename acc_t, typename out_t, bool kIsCausal, bool kIsRopeSeparate, bool kDoSplit>
__launch_bounds__(Traits::kNumThreads, Traits::kWaveOccupancy)
__global__ void kn_ck_mla_fwd_splictkv_prefill(
    const CkMlaPrefillFwdParams params)
{
    using Policy = CkMlaPrefillPolicy<Traits, scalar_t, acc_t>;
    constexpr auto HiddenDimSize = kIsRopeSeparate ? Traits::kSizeNope : Traits::kSizeD;

    // allocate LDS
    __shared__ uint8_t p_smem[Policy::GetSmemSize()];

    const auto [tile_m_id, split_id, hqid, bid] =
        kDoSplit ? GetTileIndex<Traits>(params.num_splits) : GetTileIndex<Traits>(1);
    const auto    hqid_xqa = Traits::kEnableXqa ? hqid * params.hq_hk_ratio : hqid;
    const auto    hkid     = hqid_xqa / params.hq_hk_ratio;
    const int32_t mid      = __builtin_amdgcn_readfirstlane(tile_m_id * Traits::kBlockM);

    const int32_t seqlen_qo    = params.p_seqlens_qo[bid];
    const int32_t seqlen_qo_pk = Traits::kEnableXqa ? seqlen_qo * params.hq_hk_ratio : seqlen_qo;
    const int32_t seqlen_kv    = params.p_seqlens_kv[bid];

    if (mid >= seqlen_qo_pk)
    {
        return;
    }

    // Define causal mask
    using Mask = std::conditional_t<Traits::kEnableXqa,
                                    ck_tile::SimplifiedRatioAttentionMask<kIsCausal>,
                                    ck_tile::SimplifiedGenericAttentionMask<kIsCausal>>;
    Mask mask = [&] {
        if constexpr(kIsCausal)
        {
            if constexpr(Traits::kEnableXqa)
            {
                return Mask{seqlen_qo,
                            seqlen_kv - seqlen_qo + 1,
                            seqlen_qo_pk,
                            seqlen_kv,
                            params.mask_y_ratio_mdiv};
            }
            else
            {
                return Mask{seqlen_qo,
                            seqlen_kv - seqlen_qo + 1,
                            seqlen_qo,
                            seqlen_kv};
            }
        }
        else
        {
            return Mask{seqlen_qo_pk, seqlen_kv};
        }
    }();

    constexpr auto dram_nope_window_length_k = Traits::kKVLoadOnce
                                                   ? ck_tile::number<Traits::kSizeNope>{}
                                                   : ck_tile::number<Traits::kBlockK0>{};
    constexpr auto dram_rope_window_length_k = Traits::kKVLoadOnce
                                                   ? ck_tile::number<Traits::kSizeRope>{}
                                                   : ck_tile::number<Traits::kBlockK0>{};

    constexpr auto q_nope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, dram_nope_window_length_k);
    constexpr auto q_rope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, dram_rope_window_length_k);
    constexpr auto k_nope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, dram_nope_window_length_k);
    constexpr auto k_rope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, dram_rope_window_length_k);
    constexpr auto v_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN1>{}, ck_tile::number<Traits::kBlockK1>{});

    const scalar_t* p_query_nope = reinterpret_cast<const scalar_t*>(params.p_query_nope) +
                              int64_t(hqid_xqa) * params.stride_h_q_nope +   // head offset
                              int64_t(bid) * params.stride_b_q_nope;     // batch offset
    const scalar_t* p_key_nope   = reinterpret_cast<const scalar_t*>(params.p_key_nope) +
                              int64_t(hkid) * params.stride_h_k_nope;    // head offset
    const scalar_t* p_value = reinterpret_cast<const scalar_t*>(params.p_value) +
                              int64_t(hkid) * params.stride_h_v;    // head offset
    const int32_t*  p_block_table = params.p_block_table +
                                    int64_t(bid) * params.block_table_batch_stride; // batch offset

    const int32_t kv_cache_width = params.num_page_blocks * params.page_block_size;

    const auto q_dram_nope = MakeQDram<Policy, HiddenDimSize>(
        p_query_nope, seqlen_qo, params.stride_s_q_nope, params.hq_hk_ratio, params.stride_h_q_nope);

    const auto k_dram_nope = MakeKDram<Policy, HiddenDimSize>(p_key_nope,   kv_cache_width, params.stride_s_k_nope);
    const auto v_dram = MakeVDram<Policy>(p_value, kv_cache_width, params.stride_s_v);    

    auto q_dram_window_nope = ck_tile::make_tile_window(q_dram_nope, q_nope_dram_window_lengths, {mid, 0});
    auto q_dram_window_rope = [&] {
        if constexpr(kIsRopeSeparate)
        {
            const scalar_t* p_query_rope =
                reinterpret_cast<const scalar_t*>(params.p_query_rope) +
                int64_t(hqid_xqa) * params.stride_h_q_rope + // head offset
                int64_t(bid) * params.stride_b_q_rope;       // batch offset
            const auto q_dram_rope = MakeQDram<Policy, Traits::kSizeRope>(p_query_rope,
                                                                          seqlen_qo,
                                                                          params.stride_s_q_rope,
                                                                          params.hq_hk_ratio,
                                                                          params.stride_h_q_rope);
            return ck_tile::make_tile_window(q_dram_rope, q_rope_dram_window_lengths, {mid, 0});
        }
        else
        {
            return ck_tile::make_tile_window(q_dram_nope, q_rope_dram_window_lengths, {mid, Traits::kSizeNope});
        }
    }();

    auto k_dram_window_nope = ck_tile::make_tile_window(k_dram_nope, k_nope_dram_window_lengths, {0, 0});
    auto k_dram_window_rope = [&] {
        if constexpr(kIsRopeSeparate)
        {
            const scalar_t* p_key_rope = reinterpret_cast<const scalar_t*>(params.p_key_rope) +
                                         int64_t(hkid) * params.stride_h_k_rope; // head offset
            const auto k_dram_rope = MakeKDram<Policy, Traits::kSizeRope>(
                p_key_rope, kv_cache_width, params.stride_s_k_rope);
            return ck_tile::make_tile_window(k_dram_rope, k_rope_dram_window_lengths, {0, 0});
        }
        else
        {
            return ck_tile::make_tile_window(k_dram_nope, k_rope_dram_window_lengths, {0, Traits::kSizeNope});
        }
    }();

    auto v_dram_window = ck_tile::make_tile_window(v_dram, v_dram_window_lengths, {0, 0});

    const auto real_stride_s_k_rope = kIsRopeSeparate ? params.stride_s_k_rope : params.stride_s_k_nope;
    if constexpr (kDoSplit)
    {
        acc_t* p_lse_acc = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
                           int64_t(hqid_xqa) * params.stride_h_lseacc +     // head offset
                           int64_t(bid) * params.stride_b_lseacc +      // batch offset
                           int64_t(split_id) * params.stride_sp_lseacc; // split offset
        out_t* p_out_acc = reinterpret_cast<out_t*>(params.p_output_accum) +
                           int64_t(hqid_xqa) * params.stride_h_oacc +      // head offset
                           int64_t(bid) * params.stride_b_oacc +       // batch offset
                           int64_t(split_id) * params.stride_sp_oacc;  // split offset

        auto lse_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});

        const auto lse_acc_dram = MakeLseAccDram<Policy>(p_lse_acc,
                                                         lse_acc_dram_window_lengths,
                                                         seqlen_qo,
                                                         params.hq_hk_ratio,
                                                         params.stride_h_lseacc);
        const auto out_acc_dram = MakeOutAccDram<Policy>(p_out_acc,
                                                         seqlen_qo,
                                                         params.stride_s_oacc,
                                                         params.hq_hk_ratio,
                                                         params.stride_h_oacc);

        auto lse_acc_dram_window =
            ck_tile::make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {mid});
        auto out_acc_dram_window =
            ck_tile::make_tile_window(out_acc_dram, out_acc_dram_window_lengths, {mid, 0});


        if constexpr (!Traits::kKVLoadOnce) {
            kn_ck_mla_fwd_splitkv_prefill_tile<Traits, scalar_t, acc_t, out_t, kIsRopeSeparate>(
                q_dram_window_nope,
                q_dram_window_rope,
                k_dram_window_nope,
                k_dram_window_rope,
                v_dram_window,
                lse_acc_dram_window,
                out_acc_dram_window,
                p_block_table,
                __builtin_amdgcn_readfirstlane(params.page_block_size),
                __builtin_amdgcn_readfirstlane(params.stride_s_k_nope),
                __builtin_amdgcn_readfirstlane(real_stride_s_k_rope),
                __builtin_amdgcn_readfirstlane(params.stride_s_v),
                seqlen_kv,
                params.num_splits,
                split_id,
                mask,
#if CK_MLA_FWD_FAST_EXP2
                static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
                params.scale_softmax,
#endif
                p_smem);
        }
        else
        {
            kn_ck_mla_fwd_splitkv_prefill_load_once_tile<Traits, scalar_t, acc_t, out_t>(
                q_dram_window_nope,
                q_dram_window_rope,
                k_dram_window_nope,
                k_dram_window_rope,
                lse_acc_dram_window,
                out_acc_dram_window,
                p_block_table,
                __builtin_amdgcn_readfirstlane(params.page_block_size),
                __builtin_amdgcn_readfirstlane(params.stride_s_k_nope),
                __builtin_amdgcn_readfirstlane(real_stride_s_k_rope),
                __builtin_amdgcn_readfirstlane(params.stride_s_v),
                seqlen_kv,
                params.num_splits,
                split_id,
                mask,
#if CK_MLA_FWD_FAST_EXP2
                static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
                params.scale_softmax,
#endif
                p_smem);
        }
    }
    else
    {
        // Assuming lse is in shape [b, h, s] and is contiguous
        acc_t* p_lse =
            reinterpret_cast<acc_t*>(params.p_softmax_lse) +
            (int64_t(bid) * params.size_h_ori + hqid_xqa) * params.max_size_s_ori; // batch+head offset
        out_t* p_out = reinterpret_cast<out_t*>(params.p_output) +
                       int64_t(hqid_xqa) * params.stride_h_o + // head offset
                       int64_t(bid) * params.stride_b_o;       // batch offset

        auto lse_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});

        const auto lse_dram = MakeLseDram<Policy>(p_lse,
                                                  lse_dram_window_lengths,
                                                  seqlen_qo,
                                                  params.hq_hk_ratio,
                                                  params.stride_h_lse);
        const auto out_dram = MakeOutDram<Policy>(
            p_out, seqlen_qo, params.stride_s_o, params.hq_hk_ratio, params.stride_h_o);

        auto lse_dram_window =
            ck_tile::make_tile_window(lse_dram, lse_dram_window_lengths, {mid});
        auto out_dram_window =
            ck_tile::make_tile_window(out_dram, out_dram_window_lengths, {mid, 0});

        if constexpr (!Traits::kKVLoadOnce)
        {
            kn_ck_mla_fwd_splitkv_prefill_tile<Traits, scalar_t, acc_t, out_t, kIsRopeSeparate>(
                q_dram_window_nope,
                q_dram_window_rope,
                k_dram_window_nope,
                k_dram_window_rope,
                v_dram_window,
                lse_dram_window,
                out_dram_window,
                p_block_table,
                __builtin_amdgcn_readfirstlane(params.page_block_size),
                __builtin_amdgcn_readfirstlane(params.stride_s_k_nope),
                __builtin_amdgcn_readfirstlane(real_stride_s_k_rope),
                __builtin_amdgcn_readfirstlane(params.stride_s_v),
                seqlen_kv,
                1, // num_splits
                0, // split_id
                mask,
#if CK_MLA_FWD_FAST_EXP2
                static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
                params.scale_softmax,
#endif
                p_smem);
        }
        else
        {
            kn_ck_mla_fwd_splitkv_prefill_load_once_tile<Traits, scalar_t, acc_t, out_t>(
                q_dram_window_nope,
                q_dram_window_rope,
                k_dram_window_nope,
                k_dram_window_rope,
                lse_dram_window,
                out_dram_window,
                p_block_table,
                __builtin_amdgcn_readfirstlane(params.page_block_size),
                __builtin_amdgcn_readfirstlane(params.stride_s_k_nope),
                __builtin_amdgcn_readfirstlane(real_stride_s_k_rope),
                __builtin_amdgcn_readfirstlane(params.stride_s_v),
                seqlen_kv,
                1, // num_splits
                0, // split_id
                mask,
#if CK_MLA_FWD_FAST_EXP2
                static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
                params.scale_softmax,
#endif
                p_smem);
        }
    }
}

template <typename Traits, int32_t kMaxSplits, typename out_t, typename in_t>
__global__ void kn_ck_mla_fwd_splictkv_prefill_combine(
    const CkMlaPrefillFwdParams params)
{
    using Policy  = CkMlaCombineKernelPolicy<Traits, out_t, in_t>;
    using index_t = int64_t;

    __shared__ in_t lds_lse_scale[kMaxSplits];

    const int32_t bidx = blockIdx.z;

    const int32_t num_splits   = params.num_splits;
    const int32_t split_offset = bidx * params.num_splits;
    assert((num_splits > 1) && (num_splits <= kMaxSplits));

    const int32_t lane_id          = ck_tile::get_lane_id();
    const int32_t hidx             = blockIdx.y;
    const int32_t sidx             = blockIdx.x;
    const int32_t hsidx            = hidx * params.max_size_s_ori + sidx;
    const int32_t shidx            = hidx + sidx * params.size_h_ori;
    const int32_t size_hs          = params.size_h_ori * params.max_size_s_ori;
    const index_t offset_lse_accum = split_offset * size_hs + hsidx; // offset to split 0
    const index_t offset_lse       = bidx * size_hs + hsidx;

    if (ck_tile::get_warp_id() == 0)
    {
        const in_t* p_lse_accum = reinterpret_cast<in_t*>(params.p_softmax_lseaccum) + offset_lse_accum;
        in_t* p_lse             = reinterpret_cast<in_t*>(params.p_softmax_lse) + offset_lse;

        constexpr int32_t kNumLsePerThr = ck_tile::integer_divide_ceil(kMaxSplits, ck_tile::get_warp_size());
        in_t local_lse[kNumLsePerThr];

        // Load thread local LSE and get local max LSE
        in_t max_lse = -ck_tile::numeric<in_t>::infinity();
        #pragma unroll
        for (int32_t i = 0; i < kNumLsePerThr; ++i)
        {
            const int32_t split_idx = i * ck_tile::get_warp_size() + lane_id;
            const in_t lse =
                (split_idx < num_splits) ? p_lse_accum[split_idx * size_hs] : -ck_tile::numeric<in_t>::infinity();
            local_lse[i] = lse;
            max_lse = ck_tile::max(max_lse, lse);
        }

        // Get global max LSE
        #pragma unroll
        for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
        {
            max_lse = ck_tile::max(max_lse, __shfl_xor(max_lse, offset));
        }

        // Get sum of LSE
        in_t sum_lse = 0.f;
        #pragma unroll
        for (int32_t i = 0; i < kNumLsePerThr; ++i)
        {
#ifdef CK_MLA_FWD_FAST_EXP2
            static_assert(0, "have not figured out if need exp2 here");
#endif
            sum_lse += ck_tile::exp(local_lse[i] - max_lse);
        }
        #pragma unroll
        for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
        {
            sum_lse += __shfl_xor(sum_lse, offset);
        }

        // Get global LSE
        const auto [global_lse, output_lse] = [&]() {
            if ((sum_lse == 0.f) || (sum_lse != sum_lse))
            {
                return ck_tile::make_tuple(ck_tile::numeric<in_t>::infinity(), -ck_tile::numeric<in_t>::infinity());
            }
            else
            {
                const in_t lse = ck_tile::log(sum_lse) + max_lse;
                return ck_tile::make_tuple(lse, lse);
            }
        } ();

        if (lane_id == 0)
        {
            *p_lse = output_lse;
        }

        // Write LSE to LDS
        #pragma unroll
        for (int32_t i = 0; i < kNumLsePerThr; ++i)
        {
            const int32_t split_idx = i * ck_tile::get_warp_size() + lane_id;
            if (split_idx < num_splits)
            {
                lds_lse_scale[split_idx] = ck_tile::exp(local_lse[i] - global_lse);
            }
        }
    }

    __builtin_amdgcn_sched_barrier(0);
    ck_tile::block_sync_lds();

    static_assert(Traits::kSizeDV % Traits::kNumThreadsCombine == 0);

    auto oaccu_window =
        Policy::MakeOaccuTileWindow(params.p_output_accum, shidx, size_hs, split_offset, num_splits);

    auto reg_out = ck_tile::make_static_distributed_tensor<in_t>(
        decltype(ck_tile::load_tile(oaccu_window))::get_tile_distribution());
    ck_tile::set_tile(reg_out, 0.f);

    for (int32_t split_idx = 0; split_idx < num_splits; ++split_idx)
    {
        const in_t lse_scale = lds_lse_scale[split_idx];
        auto oaccu = ck_tile::load_tile(oaccu_window);
        ck_tile::sweep_tile(oaccu, [&](auto idx) {
            reg_out(idx) += lse_scale * oaccu(idx);
        });
        ck_tile::move_tile_window(oaccu_window, {size_hs, 0});
    }

    auto dram_out = Policy::MakeOutputTileWindow(
        static_cast<out_t*>(params.p_output) +
        bidx * params.stride_b_o + hidx * params.stride_h_o + sidx * params.stride_s_o);
    ck_tile::store_tile(dram_out, ck_tile::cast_tile<out_t>(reg_out));
}

// =====================================================================================================================
// Dispatch
//

template <typename Traits, typename scalar_t, typename acc_t, typename out_t, bool kIsCausal, bool kIsRopeSeparate>
void dispatch_ck_mla_fwd_splictkv_prefill(
    const CkMlaPrefillFwdParams& params)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int32_t num_blk =
        ck_tile::integer_divide_ceil(params.max_size_s_pk, Traits::kBlockM) * params.num_splits;
    const dim3 grid_attn = dim3(num_blk, params.size_h_pk, params.size_b);
    const dim3 grid_comb = dim3(params.max_size_s_ori, params.size_h_ori, params.size_b);


    if (params.num_splits > 1)
    {
        // out_t is not take into consideration when doing splits because combine shader is always expected to do
        // the final output type conversion.
        auto kn_attn = &kn_ck_mla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, acc_t, kIsCausal, kIsRopeSeparate, true>;
        auto kn_comb =
            (params.num_splits <= 32)  ? &kn_ck_mla_fwd_splictkv_prefill_combine<Traits, 32,  scalar_t, acc_t> :
            // (params.num_splits <= 64)  ? &kn_ck_mla_fwd_splictkv_prefill_combine<Traits, 64,  scalar_t, acc_t> :
            // (params.num_splits <= 96)  ? &kn_ck_mla_fwd_splictkv_prefill_combine<Traits, 96,  scalar_t, acc_t> :
            // (params.num_splits <= 128) ? &kn_ck_mla_fwd_splictkv_prefill_combine<Traits, 128, scalar_t, acc_t> :
            static_cast<decltype(kn_ck_mla_fwd_splictkv_prefill_combine<Traits, 32, scalar_t, acc_t>)*>(nullptr);
        TORCH_CHECK(kn_comb != nullptr, "num_splits is larger than expected (<=128) !");
        kn_attn<<<grid_attn, Traits::kNumThreads, 0, stream>>>(params);
        kn_comb<<<grid_comb, Traits::kNumThreadsCombine, 0, stream>>>(params);
    }
    else
    {
        auto kn_attn = &kn_ck_mla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, kIsCausal, kIsRopeSeparate, false>;
        kn_attn<<<grid_attn, Traits::kNumThreads, 0, stream>>>(params);
    }
}

// =====================================================================================================================
// Interfaces
//
#define CK_MLA_CASE(IS_CAUSAL, IS_ROPE_SEPARATE, ...)                  \
    if(is_causal == IS_CAUSAL && is_rope_separate == IS_ROPE_SEPARATE) \
    {                                                                  \
        constexpr bool Is_causal        = IS_CAUSAL;                   \
        constexpr bool Is_rope_separate = IS_ROPE_SEPARATE;            \
        __VA_ARGS__;                                                   \
    }

#define DISPATCH_CK_MLA_TYPES(TYPE, NAME, ...)                                   \
    switch((TYPE))                                                               \
    {                                                                            \
    case at::ScalarType::BFloat16: {                                             \
        using scalar_t = ck_tile::bf16_t;                                        \
        using out_t    = std::conditional_t<kForceOutAcc, acc_t, scalar_t>;      \
        CK_MLA_CASE(true, true, __VA_ARGS__)                                     \
        CK_MLA_CASE(true, false, __VA_ARGS__)                                    \
        CK_MLA_CASE(false, true, __VA_ARGS__)                                    \
        CK_MLA_CASE(false, false, __VA_ARGS__)                                   \
        break;                                                                   \
    }                                                                            \
    case at::ScalarType::Half: {                                                 \
        using scalar_t = ck_tile::fp16_t;                                        \
        using out_t    = std::conditional_t<kForceOutAcc, acc_t, scalar_t>;      \
        CK_MLA_CASE(true, true, __VA_ARGS__)                                     \
        CK_MLA_CASE(true, false, __VA_ARGS__)                                    \
        CK_MLA_CASE(false, true, __VA_ARGS__)                                    \
        CK_MLA_CASE(false, false, __VA_ARGS__)                                   \
        break;                                                                   \
    }                                                                            \
    default: TORCH_CHECK(false, NAME " does't support ", toString((TYPE)), "."); \
    }

int num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    int32_t result = 1;

    if (batch_nhead_mblocks < 0.8f * num_SMs)
    {
        max_splits = std::min(max_splits, std::min(num_SMs, num_n_blocks));
        float max_efficiency = 0.f;
        std::vector<float> efficiency;
        efficiency.reserve(max_splits);

        // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
        // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
        // (i.e. it's 11 splits anyway).
        // So we check if the number of blocks per split is the same as the previous num_splits.
        auto is_split_eligible = [&num_n_blocks](int num_splits) {
            return (num_splits == 1) ||
                (ck_tile::integer_divide_ceil(num_n_blocks, num_splits) !=
                 ck_tile::integer_divide_ceil(num_n_blocks, num_splits - 1));
        };

        for(int num_splits = 1; num_splits <= max_splits; num_splits++)
        {
            if(!is_split_eligible(num_splits))
            {
                efficiency.push_back(0.f);
            }
            else
            {
                float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
                float eff     = n_waves / ceil(n_waves);
                if(eff > max_efficiency)
                {
                    max_efficiency = eff;
                }
                efficiency.push_back(eff);
            }
        }

        for(int num_splits = 1; num_splits <= max_splits; num_splits++)
        {
            if(!is_split_eligible(num_splits))
            {
                continue;
            }

            if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
            {
                result = num_splits;
                break;
            }
        }
    }

    return result;
}

template <typename Traits>
int32_t calculate_num_splits(
    const int32_t size_b,
    const int32_t size_h,
    const int32_t size_s)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    ck_tile::hip_check_error(hipGetDevice(&dev));
    ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
    const int32_t cu_count = dev_prop.multiProcessorCount;

    const int32_t num_m_blocks = ck_tile::integer_divide_ceil(size_s, Traits::kBlockM);
    const int32_t num_n_blocks = ck_tile::integer_divide_ceil(Traits::kSizeDV, Traits::kBlockN1);

    return num_splits_heuristic(size_b * size_h * num_m_blocks, cu_count * Traits::kCuReuse, num_n_blocks, 128);
}

std::vector<torch::Tensor>
ck_mla_fwd_prefill_with_kvcache_impl(
    torch::Tensor& query_nope,
    const torch::Tensor& key_nope_cache,
    const torch::Tensor& value_cache,
    const int32_t head_size_v,
    const torch::Tensor& seqlens_qo,
    const torch::Tensor& seqlens_kv,
    const torch::Tensor& block_table,
    const float softmax_scale,
    const bool is_causal,
    std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope_cache)
{
    const bool is_rope_separate = query_rope.has_value() && key_rope_cache.has_value();

    constexpr bool kKVLoadOnce = false;
    constexpr bool kEnableXqa  = true;
    //TODO:
    // cases need maintenance:
    //     warp4 + load_once=false + occ=2
    //     warp8 + load_once=true  + occ=1
    // targe case:
    // warp4 + load_once=true + occ=1
    //                             dqk  dv   m0  n0  n1   #warp  wave_occu
    using Traits = std::conditional_t<kKVLoadOnce,
        CkMlaPrefillKernelTrait<576, 512, 64, 16, 512, 8,     1,   kKVLoadOnce, kEnableXqa>,
        CkMlaPrefillKernelTrait<576, 512, 64, 64, 256, 4,     2,   kKVLoadOnce, kEnableXqa>>;
    constexpr bool kForceOutAcc = false;
    using acc_t                 = float;

    torch::Tensor vcache = value_cache.data_ptr() ? value_cache : key_nope_cache;

    auto opts = query_nope.options();
    static_assert(std::is_same_v<acc_t, float>);
    auto opts_acc = opts.dtype(torch::kFloat32);

    const int32_t batch_size        = query_nope.size(0);
    const int32_t max_seqlen_qo_ori = query_nope.size(1);
    const int32_t num_heads_q_ori   = query_nope.size(2);
    int32_t max_seqlen_qo           = max_seqlen_qo_ori;
    int32_t num_heads_q             = num_heads_q_ori;

    const int32_t head_size_nope = query_nope.size(3);
    const int32_t head_size_rope = is_rope_separate ? query_rope.value().size(3) : 0;
    const int32_t head_size = head_size_nope + head_size_rope;
    TORCH_CHECK((head_size == 576) && (head_size_v == 512), "Only support QK head dim 576 and V head dim 512!");

    const int32_t num_blocks      = key_nope_cache.size(0);
    const int32_t page_block_size = key_nope_cache.size(1);
    const int32_t num_heads_k     = key_nope_cache.size(2);

    TORCH_CHECK(num_heads_q % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    const int32_t hq_hk_ratio = num_heads_q_ori / num_heads_k;
    int32_t mask_y_ratio = 1;

    if constexpr(kEnableXqa)
    {
        max_seqlen_qo = max_seqlen_qo_ori * hq_hk_ratio;
        num_heads_q   = num_heads_k;
        mask_y_ratio  = hq_hk_ratio;
    }

    const int32_t num_splits = calculate_num_splits<Traits>(batch_size, num_heads_q, max_seqlen_qo);
    const bool    do_splits = num_splits > 1;

    // Combine shader, which only exists when num_splits > 1, will conduct type convert by default and force.
    // Thus, kForceOutAcc doesn't work in this case.
    auto output = torch::empty({batch_size, max_seqlen_qo_ori, num_heads_q_ori, head_size_v},
                               (kForceOutAcc && !do_splits) ? opts_acc : opts);
    auto softmax_lse = torch::empty({batch_size, num_heads_q_ori, max_seqlen_qo_ori}, opts_acc);

    CkMlaPrefillFwdParams params = {};

    params.num_splits    = num_splits;
    params.p_seqlens_qo  = seqlens_qo.data_ptr<int32_t>();
    params.p_seqlens_kv  = seqlens_kv.data_ptr<int32_t>();
    params.p_block_table = block_table.data_ptr<int32_t>();

    params.p_query_nope  = query_nope.data_ptr();
    params.p_key_nope    = key_nope_cache.data_ptr();
    params.p_value       = vcache.data_ptr();
    params.p_output      = output.data_ptr();
    params.p_softmax_lse = softmax_lse.data_ptr();

    params.size_b                   = batch_size;
    params.max_size_s_pk            = max_seqlen_qo;
    params.max_size_s_ori           = max_seqlen_qo_ori;
    params.size_h_pk                = num_heads_q;
    params.size_h_ori               = num_heads_q_ori;
    params.hq_hk_ratio              = hq_hk_ratio;
    params.block_table_batch_stride = block_table.stride(0);
    params.num_page_blocks          = num_blocks;
    params.page_block_size          = page_block_size;
    params.scale_softmax            = softmax_scale;

    params.mask_y_ratio_mdiv = ck_tile::mdiv{static_cast<uint32_t>(mask_y_ratio)};

    params.stride_b_q_nope = query_nope.stride(0);
    params.stride_s_q_nope = query_nope.stride(1);
    params.stride_h_q_nope = query_nope.stride(2);
    params.stride_b_k_nope = key_nope_cache.stride(0);
    params.stride_s_k_nope = key_nope_cache.stride(1); // size_hk * size_d
    params.stride_h_k_nope = key_nope_cache.stride(2);
    params.stride_b_v      = vcache.stride(0);
    params.stride_s_v      = vcache.stride(1); // size_hk * size_d
    params.stride_h_v      = vcache.stride(2);
    params.stride_b_o      = output.stride(0);
    params.stride_s_o      = output.stride(1);
    params.stride_h_o      = output.stride(2);
    params.stride_h_lse    = softmax_lse.stride(1);
    if (is_rope_separate)
    {
        params.p_query_rope    = query_rope.value().data_ptr();
        params.p_key_rope      = key_rope_cache.value().data_ptr();
        params.stride_b_q_rope = query_rope.value().stride(0);
        params.stride_s_q_rope = query_rope.value().stride(1);
        params.stride_h_q_rope = query_rope.value().stride(2);
        params.stride_b_k_rope = key_rope_cache.value().stride(0);
        params.stride_s_k_rope = key_rope_cache.value().stride(1); // size_hk * size_d
        params.stride_h_k_rope = key_rope_cache.value().stride(2);
    }

    if(num_splits > 1)
    {
        auto output_accum =
            torch::empty({batch_size, num_splits, max_seqlen_qo_ori, num_heads_q_ori, head_size_v}, opts_acc);
        auto softmax_lseaccum =
            torch::empty({batch_size, num_splits, num_heads_q_ori, max_seqlen_qo_ori}, opts_acc);

        params.p_softmax_lseaccum = softmax_lseaccum.data_ptr();
        params.p_output_accum     = output_accum.data_ptr();
        params.stride_b_oacc      = output_accum.stride(0);
        params.stride_h_oacc      = output_accum.stride(3);
        params.stride_sp_oacc     = output_accum.stride(1);
        params.stride_s_oacc      = output_accum.stride(2);
        params.stride_b_lseacc    = softmax_lseaccum.stride(0);
        params.stride_h_lseacc    = softmax_lseaccum.stride(2);
        params.stride_sp_lseacc   = softmax_lseaccum.stride(1);
    }

    DISPATCH_CK_MLA_TYPES(
        query_nope.scalar_type(),
        "ck_mla_fwd",
        [&](){
            dispatch_ck_mla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, Is_causal, Is_rope_separate>(params);
        }();
    );
    // assert(is_causal == false);
    // assert(query_nope.scalar_type() == at::ScalarType::BFloat16);
    // assert(query_rope.value().scalar_type() == at::ScalarType::BFloat16);
    // assert(is_rope_separate == false);
    // using scalar_t = ck_tile::bf16_t;
    // using out_t = std::conditional_t<kForceOutAcc, acc_t, scalar_t>;
    // dispatch_ck_mla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, false, false>(params);


    return {output.to(opts), softmax_lse};
}

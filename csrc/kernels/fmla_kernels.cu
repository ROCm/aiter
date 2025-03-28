// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/gemm.hpp>
// #include <ck_tile/ops/fmha/block/block_masking.hpp>
#include <ck_tile/ops/reduce/block/block_reduce.hpp>

// =====================================================================================================================
// Definitions and helper structures
//

template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN_,
          int32_t kNumWarps_>
struct FlashMlaKernelTrait
{
    static constexpr int32_t kSizeD                  = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                 = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kNumWarps               = kNumWarps_;
    static constexpr int32_t kNumThreads             = kNumWarps * warpSize;
    static constexpr int32_t kNumWarpsSoftmax        = 4;
    static constexpr int32_t kNumThreadsSoftmax      = kNumWarpsSoftmax * warpSize;
    static constexpr int32_t kBlockM                 = kBlockM_;
    static constexpr int32_t kBlockN                 = kBlockN_;
    static constexpr int32_t kFixedOverheadNumBlocks = 5;
    static constexpr int32_t kMaxBatchSize           = 4096;

    static constexpr int32_t kLdsOffsetP        = 2 * kBlockN * kSizeD;
    static constexpr int32_t kLdsOffsetScale    = kLdsOffsetP + kNumThreadsSoftmax;
    static constexpr int32_t kLdsOffsetMax      = kLdsOffsetScale + kNumThreadsSoftmax;
    static constexpr int32_t kLdsOffsetSum      = kLdsOffsetMax + kNumThreadsSoftmax;

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);

    using Gemm0BlockWarps = ck_tile::sequence<2, 1, 1>;
    using Gemm1BlockWarps = ck_tile::sequence<2, 1, 1>;
    using Gemm0WarpTile = ck_tile::sequence<16, 16, 16>;
    using Gemm1WarpTile = ck_tile::sequence<16, 16, 16>;

    static constexpr int32_t kNumGemm0Warps = 2;
    static constexpr int32_t kNumGemm1Warps = 4;
};

using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<576, 512, 64, 64, 4>;

union TileSchedulerMetaData
{
    struct Core
    {
        int32_t begin_batch_idx;
        int32_t begin_seqlen_idx;
        int32_t end_batch_idx;
        int32_t end_seqlen_idx;
        int32_t begin_n_split_idx;
    };
    uint32_t data[8];
};
constexpr size_t TileSchedulerMetaDataSizeInDw = sizeof(TileSchedulerMetaData) / sizeof(int32_t);

struct FlashMlaFwdParams
{
    int32_t* __restrict__ p_cu_seqlens_k;
    int32_t* __restrict__ p_block_table;
    int32_t* __restrict__ p_tile_scheduler_metadata;
    int32_t* __restrict__ p_num_splits;
    
    void* __restrict__ p_query;
    void* __restrict__ p_key;
    void* __restrict__ p_value;
    void* __restrict__ p_output;
    void* __restrict__ p_softmax_lse;
    void* __restrict__ p_softmax_lseaccum;
    void* __restrict__ p_output_accum;

    int32_t size_b;
    int32_t size_s;
    int32_t size_h;
    int32_t hq_hk_ratio;
    int32_t num_groups;
    int32_t num_cu_parts;
    int64_t block_table_batch_stride;
    int32_t page_block_size;
    float   scale_softmax;
    float   scale_softmax_log2;
    bool    is_causal;

    // Use int64_t if there is int32 overflow case. For now, just use int32 to save sgpr and prevent using
    // spill table.
    using index_t = int32_t;

    index_t stride_b_q;     // stride in batch of query
    index_t stride_s_q;     //    ... in sequence ...
    index_t stride_h_q;     //    ... in head ...
    index_t stride_b_k;     // stride in batch of key
    index_t stride_s_k;     //    ... in sequence ...
    index_t stride_h_k;     //    ... in head ...
    index_t stride_b_v;     // stride in batch of value
    index_t stride_s_v;     //    ... in sequence ...
    index_t stride_h_v;     //    ... in head ...
    index_t stride_b_o;     // stride in batch of output
    index_t stride_s_o;     //    ... in sequence ...
    index_t stride_h_o;     //    ... in head ...
};

// =====================================================================================================================
// Kernel Entries
//

template <typename Traits, typename scalar_t, typename acc_t>
__device__ constexpr auto get_qk_block_gemm()
{
    using GemmProblem = ck_tile::BlockGemmProblem<scalar_t,
                                                  scalar_t,
                                                  acc_t,
                                                  Traits::kNumGemm0Warps * ck_tile::get_warp_size(), 
                                                  ck_tile::TileGemmShape<
                                                      ck_tile::sequence<Traits::kBlockM,
                                                                        Traits::kBlockN,
                                                                        Traits::kSizeD>,
                                                      typename Traits::Gemm0BlockWarps,
                                                      typename Traits::Gemm0WarpTile>>;
    constexpr auto warp_gemm = []() {
        constexpr int32_t WarpGemmM = Traits::Gemm0WarpTile::at(ck_tile::number<0>{});
        if constexpr(std::is_same_v<scalar_t, ck_tile::half_t> && 
                     std::is_same_v<acc_t, float>)
        {
            if constexpr(WarpGemmM == 32)
                return ck_tile::WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
            else if constexpr(WarpGemmM == 16)
                return ck_tile::WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
            else // WarpGemmM == 4
                return ck_tile::WarpGemmMfmaF16F16F32M4N64K16{};
        }
        else if constexpr(std::is_same_v<scalar_t, ck_tile::bf16_t> &&
                          std::is_same_v<acc_t, float>)
        {
            if constexpr(WarpGemmM == 32)
                return ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
            else if constexpr(WarpGemmM == 16)
                return ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
            else // WarpGemmM == 4
                return ck_tile::WarpGemmMfmaBf16Bf16F32M4N64K16{};
        }
    }();

    using BlockGemmPolicy =
        ck_tile::BlockGemmARegBSmemCRegV2CustomPolicy<scalar_t,
                                                      scalar_t,
                                                      acc_t,
                                                      typename Traits::Gemm0BlockWarps,
                                                      decltype(warp_gemm)>;
    if constexpr(1 < Traits::kNumGemm0Warps)
        return ck_tile::BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
    else
        return ck_tile::BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
}

template <typename Traits, typename scalar_t, typename acc_t>
__device__ constexpr auto get_kv_block_gemm()
{
    using GemmProblem = ck_tile::BlockGemmProblem<scalar_t,
                                                  scalar_t,
                                                  acc_t,
                                                  Traits::kNumGemm1Warps * ck_tile::get_warp_size(),
                                                  ck_tile::TileGemmShape<ck_tile::sequence<Traits::kBlockM,
																						   Traits::kSizeDV,
																						   Traits::kBlockN>,
												  typename Traits::Gemm1BlockWarps,
												  typename Traits::Gemm1WarpTile>>;

    auto warp_gemm = [&]() {
        return ck_tile::WarpGemmMfmaDispatcher<
            scalar_t,
            scalar_t,
            acc_t,
            Traits::Gemm1WarpTile::at(ck_tile::number<0>{}),
            Traits::Gemm1WarpTile::at(ck_tile::number<1>{}),
            Traits::Gemm1WarpTile::at(ck_tile::number<2>{}),
            true>{};
    }();

    using BlockGemmPolicy =
        ck_tile::BlockGemmARegBSmemCRegV2CustomPolicy<scalar_t,
                                             scalar_t,
                                             acc_t,
                                             typename Traits::Gemm1BlockWarps,
                                             decltype(warp_gemm)>;
    return ck_tile::BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
}

template <typename Traits, typename scalar_t, typename acc_t, bool Is_causal>
__global__ void flash_fwd_splitkv_mla_kernel(
    const FlashMlaFwdParams params)
{
    constexpr int32_t kSizeD             = Traits::kSizeD; 
    constexpr int32_t kSizeDV            = Traits::kSizeDV; 
    constexpr int32_t kNumThreads        = Traits::kNumThreads;
    constexpr int32_t kNumThreadsSoftmax = Traits::kNumThreadsSoftmax;
    constexpr int32_t kBlockM            = Traits::kBlockM;
    constexpr int32_t kBlockN            = Traits::kBlockN;
    constexpr int32_t kLdsOffsetP        = 2 * kBlockN * kSizeD;
    constexpr int32_t kLdsOffsetScale    = kLdsOffsetP + kNumThreadsSoftmax;
    constexpr int32_t kLdsOffsetMax      = kLdsOffsetScale + kNumThreadsSoftmax;
    constexpr int32_t kLdsOffsetSum      = kLdsOffsetMax + kNumThreadsSoftmax;

    const int32_t i_block_m   = blockIdx.x;
    const int32_t i_nhead     = blockIdx.y;
    const int32_t i_nhead_k   = i_nhead / params.hq_hk_ratio;
    const int32_t i_partition = blockIdx.z;

    const ck_tile::index_t i_m0 = __builtin_amdgcn_readfirstlane(i_block_m * kBlockM);
    constexpr int32_t kKPack = 16 / sizeof(scalar_t);

    constexpr auto I0 = ck_tile::number<0>{};
    constexpr auto I1 = ck_tile::number<1>{};

    extern __shared__ char shared_memory[];

    const int32_t tidx = threadIdx.x; 

    const auto make_q_dram = [&](const scalar_t* data) {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            data,
            ck_tile::make_tuple(params.size_s, kSizeD),
            ck_tile::make_tuple(params.stride_s_q, I1),
            kKPack,
            I1);
    }();

    const auto q_dram = make_naive_tensor_view<ck_tile::address_space_enum::global>(
        params.p_query,
        ck_tile::make_tuple(params.size_s, kSizeD),
        ck_tile::make_tuple(params.stride_s_q, I1),
        ck_tile::number<kKPack>{},
        I1);

    const auto make_k_dram = [&](const scalar_t* data, int32_t height) {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            data,
            ck_tile::make_tuple(height, kSizeD),
            ck_tile::make_tuple(params.stride_s_k, I1),
			ck_tile::number<kKPack>{},
            I1);
    };

    // const auto make_v_dram = [&](const scalar_t* data, int32_t length) {
    //     const auto v_dram_naive = make_naive_tensor_view<ck_tile::address_space_enum::global>(
    //         data, // will update this pointer if using paged-kvcache
    //         ck_tile::make_tuple(length, ck_tile::kSizeDV),
    //         ck_tile::make_tuple(params.stride_s_v, I1),
    //         ck_tile::kKPack,
    //         I1);
    //
    //     //TODO: hack this, tride with v is hdim_q
    //     return transform_tensor_view(v_dram_naive,
    //                                  ck_tile::make_tuple(make_pass_through_transform(ck_tile::kSizeDV),
    //                                             make_pass_through_transform(length)),
    //                                  ck_tile::make_tuple(sequence<1>{}, sequence<0>{}),
    //                                  ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));
    // };

    auto k_page_block_navigator_func = [&, i_nhead_k_ = i_nhead_k](int32_t i_batch_, int32_t seqlen_kv_, auto& k_dram_) {
        const auto* block_indices =
            reinterpret_cast<const int32_t*>(params.p_block_table) +
            i_batch_ * params.block_table_batch_stride;
        const int32_t num_blocks =
            ck_tile::integer_divide_ceil(seqlen_kv_, params.page_block_size);

        const int64_t fixed_offset =
            static_cast<int64_t>(i_nhead_k_) * params.stride_h_k;

        return make_page_block_navigator<const scalar_t, 0>(
            params.p_key,
            params.stride_b_k, // kcache page-block stride/size
            fixed_offset,
            block_indices,
            num_blocks,
            params.page_block_size,
            k_dram_,
            make_k_dram(nullptr,
                        (seqlen_kv_ - (num_blocks - 1) * params.page_block_size)));
    };

    // auto v_page_block_navigator_func = [&](int i_batch_, int seqlen_kv_, auto& v_dram) {
    //     const auto* block_indices =
    //         reinterpret_cast<const int32_t*>(params.p_block_table) +
    //         i_batch_ * params.block_table_batch_stride;
    //     const index_t num_blocks =
    //         ck_tile::integer_divide_ceil(seqlen_k_, params.page_block_size);
    //
    //     const long_index_t fixed_offset =
    //         static_cast<long_index_t>(i_nhead_k) * params.nhead_stride_v;
    //
    //     return make_page_block_navigator<const VDataType, 1>(
    //         params.v_ptr,
    //         params.batch_stride_v, // vcache page-block stride/size
    //         fixed_offset,
    //         block_indices,
    //         num_blocks,
    //         params.page_block_size,
    //         v_dram,
    //         make_v_dram(nullptr,
    //                     seqlen_kv_ - (num_blocks - 1) * params.page_block_size)));
    // };

	constexpr auto lse_acc_dram_window_lengths = ck_tile::make_tuple(kBlockM);
    auto make_lse_acc_dram_window = [&, i_nhead_ = i_nhead, i_block_m_ = i_block_m](int i_batch_, int i_split_) {
        const int split_offset = params.p_num_splits[i_batch_];
        acc_t* lse_acc_ptr = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
                                 ((split_offset + i_split_) * params.size_h + i_nhead_) *
                                 params.size_s + i_block_m_ * kBlockM;

        const auto lse_acc_dram = [&] {
            return make_naive_tensor_view<ck_tile::address_space_enum::global>(
                lse_acc_ptr,
                ck_tile::make_tuple(params.size_s),
                ck_tile::make_tuple(1),
				ck_tile::number<kKPack>{},
                I1);
        }();

        return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {0});
    };

    auto lse_dram = [&] {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            params.p_softmax_lse,
            ck_tile::make_tuple(params.size_s),
            ck_tile::make_tuple(1),
			ck_tile::number<kKPack>{},
            I1);
    }();
    auto lse_dram_window = make_tile_window(lse_dram, lse_acc_dram_window_lengths, {i_m0});

    constexpr auto o_dram_window_lengths = ck_tile::make_tuple(kBlockM, kBlockN);
    auto make_o_acc_dram_window = [&, i_nhead_ = i_nhead, i_block_m_ = i_block_m](int i_batch_, int i_split_) {
        const int split_offset = params.p_num_splits[i_batch_];
        acc_t* o_acc_ptr = reinterpret_cast<acc_t*>(params.p_output_accum) +
                               (((split_offset + i_split_) * params.size_h + i_nhead_) *
                               params.size_s + i_block_m_ * kBlockM) *
                               kSizeDV;

        const auto o_acc_dram = [&] {
            return make_naive_tensor_view<ck_tile::address_space_enum::global>(
                o_acc_ptr,
                ck_tile::make_tuple(params.size_s, kSizeDV),
                ck_tile::make_tuple(kSizeDV, 1),
				ck_tile::number<kKPack>{},
                I1);
		}();

        return make_tile_window(o_acc_dram, o_dram_window_lengths, {0, 0});
    };

    auto o_dram = [&] {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            params.p_output,
            ck_tile::make_tuple(params.size_s, kSizeDV),
            ck_tile::make_tuple(params.stride_s_o, 1),
			ck_tile::number<kKPack>{},
            I1);
    }();
    auto o_dram_window = make_tile_window(o_dram, o_dram_window_lengths, {i_m0, 0});

    auto q_dram_window_lengths = ck_tile::make_tuple(kBlockM, kSizeD);

    auto k_dram_window_lengths = ck_tile::make_tuple(kBlockN, kSizeD);
    // auto v_dram_window_lengths =
    //     ck_tile::make_tuple(kBlockN, kSizeDV);

    auto k_dram_tile_distribution = [&]() {
        constexpr int32_t kNPerBlock = kBlockM;
        constexpr int32_t kKPerBlock = kBlockN;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();
        constexpr int32_t NumWarps   = Traits::kNumGemm0Warps;
        constexpr int32_t kCopyBlockSize = NumWarps * warpSize;

        constexpr int32_t KVector = kKPack; // this is for global load

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr int32_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr int32_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr int32_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kCopyBlockSize * KVector));

        constexpr int32_t N0 = NumIssues;
        constexpr int32_t N1 = LaneGroups;
        constexpr int32_t N2 = NumWarps;
        constexpr int32_t K0 = LanesPerK;
        constexpr int32_t K1 = KVector;

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
				ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K0, K1>>,
				ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
				ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
				ck_tile::sequence<1, 2>,
				ck_tile::sequence<0, 1>>{});
    }();

    auto k_lds_block_descriptor = [&]() {
        constexpr auto k_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(kBlockN, kSizeD / kKPack, kKPack),
            ck_tile::make_tuple((kSizeD / kKPack + 1) * kKPack, kKPack, I1),
			ck_tile::number<kKPack>{},
            I1);

        return k_lds_block_desc;
    }();

    auto v_lds_block_descriptor = [&]() {
        constexpr auto v_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(kBlockN, kSizeDV / kKPack, kKPack),
            ck_tile::make_tuple((kSizeD / kKPack + 1) * kKPack, kKPack, I1),
			ck_tile::number<kKPack>{},
            I1);

        constexpr auto v_lds_block_desc_transpose = ck_tile::transform_tensor_view(
            v_lds_block_desc,
            ck_tile::make_tuple(ck_tile::make_pass_through_transform(kBlockN),
				ck_tile::make_merge_transform(ck_tile::make_tuple(kSizeDV / kKPack, kKPack)),
				ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
				ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{})));
        return v_lds_block_desc_transpose;
    }();

    auto p_lds_block_descriptor = [&]() {
        constexpr auto p_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(kBlockM, kBlockN),
            ck_tile::make_tuple(kBlockN, I1),
			ck_tile::number<kKPack>{},
            I1);

        return p_lds_block_desc;
    }();

    auto row_lds_block_descriptor = [&]() {
        constexpr auto scale_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(kNumThreadsSoftmax),
            ck_tile::make_tuple(I1),
			ck_tile::number<kKPack>{},
            I1);
        return scale_lds_block_desc;
    }();

    //   auto ck_tile::make_generic_attention_mask_from_lr_window<ck_tile::GenericAttentionMask<true, false>>(
        // params.window_size_left,
        // params.window_size_right,
        // params.size_s,
        // params.seqlen_k,
        // ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);


	scalar_t* kv_lds_ptr = static_cast<scalar_t*>(shared_memory);
	auto k_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
		kv_lds_ptr, k_lds_block_descriptor);
	auto k_lds_window =
		make_tile_window(k_lds, ck_tile::make_tuple(kBlockN, kSizeD), {0, 0});

	auto v_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
		kv_lds_ptr, v_lds_block_descriptor);
	auto v_lds_window =
		make_tile_window(v_lds, ck_tile::make_tuple(kSizeDV, kBlockN), {0, 0});

	scalar_t* p_lds_ptr = static_cast<scalar_t*>(
		static_cast<char*>(shared_memory) + kLdsOffsetP);
	auto p_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
		p_lds_ptr, p_lds_block_descriptor);
	auto p_lds_window =
		make_tile_window(p_lds, ck_tile::make_tuple(kBlockM, kBlockN), {0, 0});

	scalar_t* scale_lds_ptr = static_cast<scalar_t*>(
		static_cast<char*>(shared_memory) + kLdsOffsetScale);
	auto scale_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
		scale_lds_ptr, row_lds_block_descriptor);
	auto scale_lds_window =
		make_tile_window(scale_lds, ck_tile::make_tuple(kNumThreadsSoftmax), {0, 0});

	scalar_t* m_lds_ptr = static_cast<scalar_t*>(static_cast<void*>(
		static_cast<char*>(shared_memory) + kLdsOffsetMax));
	auto m_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
		m_lds_ptr, row_lds_block_descriptor);
	auto m_lds_window =
		make_tile_window(m_lds_ptr, ck_tile::make_tuple(kNumThreadsSoftmax), {0, 0});

	scalar_t* l_lds_ptr = static_cast<scalar_t*>(static_cast<void*>(
		static_cast<char*>(shared_memory) + kLdsOffsetSum));
	auto l_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
		l_lds_ptr, row_lds_block_descriptor);
	auto l_lds_window =
		make_tile_window(l_lds_ptr, ck_tile::make_tuple(kNumThreadsSoftmax), {0, 0});

    TileSchedulerMetaData::Core metadata;
    reinterpret_cast<TileSchedulerMetaData*>(&metadata) = reinterpret_cast<TileSchedulerMetaData*>(
		reinterpret_cast<char*>(params.p_tile_scheduler_metadata) + i_partition * TileSchedulerMetaDataSizeInDw);

    int32_t begin_batch_idx   = metadata.begin_batch_idx;
    int32_t begin_seqlen_idx  = metadata.begin_seqlen_idx;
    int32_t end_batch_idx     = metadata.end_batch_idx;
    int32_t end_seqlen_idx    = metadata.end_seqlen_idx;
    int32_t begin_n_split_idx = metadata.begin_n_split_idx;

    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    for (int32_t i_batch = begin_batch_idx; i_batch <= end_batch_idx; ++i_batch) {
        const int32_t i_split = i_batch == begin_batch_idx ? begin_n_split_idx : 0;
        const int32_t seqlen_k    = params.p_cu_seqlens_k[i_batch];
        const int32_t n_block_min = i_batch == begin_batch_idx ? begin_seqlen_idx / kBlockN : 0;
        const int32_t n_block_max = i_batch == end_batch_idx ? ck_tile::integer_divide_ceil(end_seqlen_idx, kBlockN) : ck_tile::integer_divide_ceil(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == ck_tile::integer_divide_ceil(seqlen_k, kBlockN);
        int32_t i_block_n = n_block_max - 1;

        const int32_t total_seqlen_kv = end_seqlen_idx - begin_seqlen_idx;

        if (i_batch > begin_batch_idx) {
            __syncthreads();  // Barrier between two tiles.
        }

        auto k_dram = make_k_dram(nullptr, total_seqlen_kv);
        // auto v_dram = make_v_dram(nullptr, total_seqlen_kv);

        auto k_page_block_navigator = k_page_block_navigator_func(i_batch, total_seqlen_kv, k_dram);
        // auto v_page_block_navigator = v_page_block_navigator_func(i_batch, total_seqlen_kv, v_dram);

        auto gemm_kv = get_kv_block_gemm<Traits, scalar_t, acc_t>();
        using OaccBlockType = decltype(gemm_kv.MakeCBlockTile());
		using MLBlockTileType = decltype(ck_tile::block_tile_reduce<acc_t>(
			OaccBlockType{}, ck_tile::sequence<1>{}, f_max, acc_t{0}));
        auto o_acc = OaccBlockType{};
        clear_tile(o_acc);
		
		auto m_block = MLBlockTileType{};
		auto l_block = MLBlockTileType{};

        const int32_t wave_group_id = tidx / kNumThreadsSoftmax;
        if (wave_group_id == 1)
        {
            auto gemm_qk = get_qk_block_gemm<Traits, scalar_t, acc_t>();
            // const index_t row_offset_q = i_batch * params.q_batch_stride + i_block_m * kBlockM * params.q_row_stride + i_nhead * params.q_head_stride;
            auto q_dram_window = make_tile_window(
                // make_q_dram(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q)),
                q_dram,
                q_dram_window_lengths,
                {i_m0, 0},
                ck_tile::remove_cvref_t<decltype(gemm_qk)>::template MakeABlockTileDistribution<
                    Traits::kBlockM,
                    Traits::kSizeD>());

            auto q = load_tile(q_dram_window);

			auto s_acc              = gemm_qk.MakeCBlockTile();
			clear_tile(s_acc); // initialize C
            using SBlockTileType = decltype(ck_tile::cast_tile<acc_t>(s_acc));
            using MLBlockTileType = decltype(ck_tile::block_tile_reduce<acc_t>(
                SBlockTileType{}, ck_tile::sequence<1>{}, f_max, acc_t{0}));

            auto m = MLBlockTileType{};
            auto l = MLBlockTileType{};

            if (i_block_n % 2 == 1)
            {
                move_tile_window(k_lds_window, {kBlockN, 0});
                move_tile_window(v_lds_window, {kSizeDV, 0});
            }
            constexpr int n_masking_steps = !Is_causal ? 1 : ck_tile::integer_divide_ceil(kBlockM, kBlockN) + 1;
            for (int masking_step = n_masking_steps; i_block_n >= n_block_min; --masking_step, --i_block_n) {
                ck_tile::block_sync_lds();
                gemm_0(s_acc,
                       q,
                       k_lds_window);
                ck_tile::block_sync_lds();
                const bool is_masking_step = masking_step > 0;
                const bool is_first_masking_step = masking_step == n_masking_steps;

                //TODO: masking
                if (is_masking_step) {
					constexpr auto sacc_spans = decltype(s_acc)::get_distributed_spans();
					sweep_tile_span(sacc_spans[I0], [&](auto idx0) {
						constexpr auto i_idx = ck_tile::make_tuple(idx0);
						sweep_tile_span(sacc_spans[I1], [&](auto idx1) {
							constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                            if constexpr (!Is_causal)
                            {
                                if (idx1 >= int(seqlen_k - i_block_n * kBlockN))
                                    s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
                            }
                            else
                            {
                                int col_limit_right = seqlen_k - 1 - i_block_n * kBlockN - (params.size_s - 1 - (i_block_m * kBlockM + idx0)) / params.num_groups;
                                if (idx1 >= col_limit_right)
                                    s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
                            }
						});
					});
                }

                const auto s = ck_tile::cast_tile<acc_t>(s_acc);
                auto m_local = block_tile_reduce<acc_t>(
                    s_acc,
                    ck_tile::sequence<1>{},
                    f_max,
                    -ck_tile::numeric<acc_t>::infinity());
                block_tile_reduce_sync(m_local, f_max, ck_tile::bool_constant<false>{});

                const auto m_old = m;
                tile_elementwise_inout(
                    [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);

                auto p_compute = make_static_distributed_tensor<acc_t>(
                    s_acc.get_tile_distribution());

                constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
                sweep_tile_span(p_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    auto row_max = params.scale_softmax_log2 * m[i_idx];
                    sweep_tile_span(p_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        p_compute(i_j_idx) = exp2(params.scale_softmax_log2 * s_acc[i_j_idx] - row_max);
                    });
                });

                auto rowsum_p = block_tile_reduce<acc_t>(
                    p_compute, ck_tile::sequence<1>{}, f_sum, acc_t{0});
                ck_tile::block_tile_reduce_sync(rowsum_p, f_sum, ck_tile::bool_constant<false>{});

                const auto scale_o = make_static_distributed_tensor<scalar_t>(m_block.get_tile_distribution());
                constexpr auto m_spans = decltype(m)::get_distributed_spans();
                sweep_tile_span(m_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    [&](const auto& v_) {
                        auto row_max = params.scale_softmax_log2 * m[i_idx];
                        scale_o(i_idx) = exp2(params.scale_softmax_log2 * m_old[i_idx] - row_max);
                    }();
                    l(i_idx) = scale_o[i_idx] * l[i_idx] + rowsum_p[i_idx];
                });
                
                // TODO: !!!!!!this p is distribution in wave group 1, gemm_kv use p with distribution in block;
                auto p = ck_tile::cast_tile<scalar_t>(p_compute);
                store_tile(p_lds_window, p);
                store_tile(scale_lds_window, scale_o);

                // l{j}, Oacc{j}
                constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    sweep_tile_span(o_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= scale_o[i_idx];
                    });
                });

                gemm_kv(o_acc,
                        p,
                        v_lds_window);
                ck_tile::block_sync_lds();

                auto k_lds_offset = i_block_n % 2 == 0 ? 0 : kBlockN;
                auto v_lds_offset = i_block_n % 2 == 0 ? 0 : kSizeDV;
                move_tile_window(k_lds_window, {k_lds_offset, 0});
                move_tile_window(v_lds_window, {v_lds_offset, 0});
            }
            store_tile(m_lds_window, m);
            store_tile(l_lds_window, l);
        }
        else
        {
            auto [i_page_block_k, k_dram_block_window] = k_page_block_navigator.make_tile_window(
                k_dram_window_lengths, {begin_seqlen_idx, 0});
			
            // auto [i_page_block_v, v_dram_window] = v_page_block_navigator.make_tile_window(
            //     v_dram_block_window_lengths,
            //     {0, aligned_physical_seqlen_k_start},
            //     Policy::template MakeVDramTileDistribution<Problem>());

            auto k_dram_window = make_tile_window(
                k_dram_block_window,
                k_dram_tile_distribution); // K DRAM tile window for

            if (i_block_n % 2 == 1)
            {
                move_tile_window(k_lds_window, {kBlockN, 0});
                move_tile_window(v_lds_window, {kSizeDV, 0});
            }
            auto k_block_tile = load_tile(k_dram_window);
            move_tile_window(k_dram_window, {kBlockN, 0});
            store_tile(k_lds_window, k_block_tile);

            for (; i_block_n >= n_block_min; --i_block_n)
            {
                __syncthreads();
                if (i_block_n - 1 >= n_block_min)
                {
                    auto k_lds_offset = i_block_n % 2 == 0 ? 0 : kBlockN;
                    move_tile_window(k_lds_window, {k_lds_offset, 0});

                    auto k_block_tile = load_tile(k_dram_window);
                    move_tile_window(k_dram_window, {kBlockN, 0});
                    store_tile(k_lds_window, k_block_tile);
                    ck_tile::block_sync_lds();
                }
                auto p = load_tile(p_lds_window);
                auto scale_o = load_tile(scale_lds_window);
                // tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, acc);

                auto o_acc = OaccBlockType{};

                constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    sweep_tile_span(o_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= scale_o[i_idx];
                    });
                });

                ck_tile::block_sync_lds();
                gemm_kv(o_acc,
                       p,
                       v_lds_window);
                ck_tile::block_sync_lds();

                auto v_lds_offset = i_block_n % 2 == 0 ? 0 : kSizeDV;
                move_tile_window(v_lds_window, {v_lds_offset, 0});
            }
        }

		m_block = load_tile(m_lds_window);
		l_block = load_tile(l_lds_window);
        // Epilogue
        auto lse_acc = make_static_distributed_tensor<scalar_t>(m_block.get_tile_distribution());
        constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
        sweep_tile_span(lse_acc_spans[I0], [&, m_ = m_block, l_ = l_block](auto idx0) {
            constexpr auto i_idx = ck_tile::make_tuple(idx0);
            lse_acc(i_idx) = m_[i_idx] * params.scale_softmax + log(l_[i_idx]);
        });

        if (NoSplit)
            store_tile(lse_dram_window, lse_acc);
        else
            store_tile(make_lse_acc_dram_window(i_batch, i_split),  lse_acc);

        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = ck_tile::make_tuple(idx0);
            const auto tmp = [&]() {
                    return l_block[i_idx] == 0.f ? 0.f : 1 / l_block[i_idx];
            }();
            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });
        if (NoSplit)
            store_tile(o_dram_window, o_acc);
        else
            store_tile(make_o_acc_dram_window, o_acc);
    }
}

template <typename Traits, typename scalar_t, typename acc_t, bool Is_causal>
void dispatch_fmla_fwd_splictkv(
    const FlashMlaFwdParams& params)
{
    // assert(params.page_block_size == Traits::kBlockN);
    const uint32_t num_m_block = static_cast<uint32_t>(ck_tile::integer_divide_ceil(params.size_s, Traits::kBlockM));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid {
        num_m_block,
        static_cast<uint32_t>(params.size_h),
        static_cast<uint32_t>(params.num_cu_parts)
    };

    constexpr int64_t smem_size = Traits::kLdsOffsetSum + Traits::kNumThreadsSoftmax;
    auto kernel = &flash_fwd_splitkv_mla_kernel<Traits, scalar_t, acc_t, Is_causal>;
    kernel<<<grid, Traits::kNumThreads, smem_size, stream>>>(params);
}

#define DISPATCH_FMLA_TYPES(TYPE, IS_CAUSAL, NAME, ...) \
    switch ((TYPE))                                     \
    {                                                   \
        case at::ScalarType::BFloat16:                  \
        {                                               \
            using scalar_t = ck_tile::bf16_t;           \
            if ((IS_CAUSAL))                            \
            {                                           \
                constexpr bool Is_causal = true;        \
                __VA_ARGS__;                            \
            }                                           \
            else                                        \
            {                                           \
                constexpr bool Is_causal = false;       \
                __VA_ARGS__;                            \
            }                                           \
            break;                                      \
        }                                               \
        case at::ScalarType::Half:                      \
        {                                               \
            using scalar_t = ck_tile::fp16_t;           \
            if ((IS_CAUSAL))                            \
            {                                           \
                constexpr bool Is_causal = true;        \
                __VA_ARGS__;                            \
            }                                           \
            else                                        \
            {                                           \
                constexpr bool Is_causal = false;       \
                __VA_ARGS__;                            \
            }                                           \
            break;                                      \
        }                                               \
        default:                                        \
            TORCH_CHECK(false, NAME " does't support ", \
                        toString((TYPE)), ".");         \
    }

std::vector<torch::Tensor> flash_mla_fwd_with_kvcache_impl(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const int32_t        head_size_v,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const float          softmax_scale,
    const bool           is_causal,
    const torch::Tensor& tile_scheduler_metadata,
    const torch::Tensor& num_splits)
{
    using Traits = FlashMlaKernelTraitsInstance;

    torch::Tensor vcache = value_cache.data_ptr() ? value_cache : key_cache;

    auto opts = query.options();

    const int32_t batch_size = query.size(0);
    const int32_t seqlen_q_ori = query.size(1);
    const int32_t num_heads_q_ori = query.size(2);

    const int32_t head_size = query.size(3);


    const int32_t num_blocks = key_cache.size(0);
    const int32_t page_block_size = key_cache.size(1);
    const int32_t num_heads_k = key_cache.size(2);

    const int32_t num_groups = num_heads_q_ori / num_heads_k;
    const int32_t seqlen_q = seqlen_q_ori * num_groups;
    const int32_t num_heads = num_heads_k;
    const int32_t num_cu_parts = tile_scheduler_metadata.size(0);

    query = query.reshape({batch_size, seqlen_q_ori, num_heads_k, num_groups, head_size}).transpose(2, 3)
                .reshape({batch_size, seqlen_q, num_heads, head_size});

    // CHECK_SHAPE(query, batch_size, seqlen_q, num_heads, head_size);
    // CHECK_SHAPE(key_cache, num_blocks, page_block_size, num_heads, head_size);

    auto output = torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));

    auto softmax_lseaccum = torch::empty({batch_size + num_cu_parts, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));
    auto output_accum = torch::empty({batch_size + num_cu_parts, num_heads, seqlen_q, head_size_v}, opts.dtype(torch::kFloat32));


    FlashMlaFwdParams params = {};
    params.p_cu_seqlens_k            = cache_seqlens.data_ptr<int32_t>();
    params.p_block_table             = block_table.data_ptr<int32_t>();
    params.p_tile_scheduler_metadata = tile_scheduler_metadata.data_ptr<int32_t>();
    params.p_num_splits              = num_splits.data_ptr<int32_t>();

    params.p_query            = query.data_ptr();
    params.p_key              = key_cache.data_ptr();
    params.p_value            = vcache.data_ptr();
    params.p_output           = output.data_ptr();
    params.p_softmax_lse      = softmax_lse.data_ptr();
    params.p_softmax_lseaccum = softmax_lseaccum.data_ptr();
    params.p_output_accum     = output_accum.data_ptr();

    params.size_b                   = batch_size;
    params.size_s                   = seqlen_q;
    params.size_h                   = num_heads;
    params.hq_hk_ratio              = num_heads / num_heads_k;
    params.num_groups               = num_groups;
    params.num_cu_parts             = tile_scheduler_metadata.size(0);
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size          = page_block_size;
    params.scale_softmax            = softmax_scale;
    params.scale_softmax_log2       = float(softmax_scale * M_LOG2E);
    params.is_causal                = is_causal;

    params.stride_b_q = query.stride(0);
    params.stride_s_q = query.stride(1);
    params.stride_h_q = query.stride(2);
    params.stride_b_k = key_cache.stride(0);
    params.stride_s_k = key_cache.stride(1);
    params.stride_h_k = key_cache.stride(2);
    params.stride_b_v = vcache.stride(0);
    params.stride_s_v = vcache.stride(1);
    params.stride_h_v = vcache.stride(2);
    params.stride_b_o = output.stride(0);
    params.stride_s_o = output.stride(1);
    params.stride_h_o = output.stride(2);

	using acc_t = float;

    DISPATCH_FMLA_TYPES(
        query.scalar_type(),
        is_causal,
        "fmla_fwd",
        [&](){
            dispatch_fmla_fwd_splictkv<Traits, scalar_t, acc_t, Is_causal>(params);
        }(););

    return {output, softmax_lse};
}

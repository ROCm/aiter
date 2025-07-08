// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <ck_tile/ops/reduce/block/block_reduce.hpp>
#include <ck_tile/ops/fmha/block/page_block_navigator.hpp>

#define ZZDebug
#define DEBUG_TID 255

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
    static constexpr int32_t kNumWarpsSoftmax        = 2;
    static constexpr int32_t kNumThreadsSoftmax      = kNumWarpsSoftmax * warpSize;
    static constexpr int32_t kBlockM                 = kBlockM_;
    static constexpr int32_t kBlockN                 = kBlockN_;
    static constexpr int32_t kFixedOverheadNumBlocks = 5;
    static constexpr int32_t kMaxBatchSize           = 4096;

    static constexpr int32_t kLdsOffsetP        = 2 * kBlockN * kSizeD * 2;
    static constexpr int32_t kLdsOffsetScale    = kLdsOffsetP + kBlockN * kBlockM * 2;
    static constexpr int32_t kLdsOffsetMax      = kLdsOffsetScale + kNumThreadsSoftmax * 4;
    static constexpr int32_t kLdsOffsetSum      = kLdsOffsetMax + kNumThreadsSoftmax * 4;

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);

    using Gemm0BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm1BlockWarps = ck_tile::sequence<4, 1, 1>;
    using Gemm0WarpTile = ck_tile::sequence<16, 16, 16>;
    using Gemm1WarpTile = ck_tile::sequence<16, 16, 16>;

    static constexpr int32_t kNumGemm0Warps = kNumWarps_;
    static constexpr int32_t kNumGemm1Warps = kNumWarps_;
    static constexpr int32_t kBlockSize = kNumWarps * warpSize;
    static constexpr int32_t kKPack = 8;
};

// using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<576, 512, 64, 64, 4>;
using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<576, 512, 64, 16, 4>;

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
    Core core;
};
constexpr size_t TileSchedulerMetaDataSizeInDw = sizeof(TileSchedulerMetaData) / sizeof(int32_t);
constexpr size_t TileSchedulerMetaDataSizeInInt4 = sizeof(TileSchedulerMetaData) / sizeof(int4);

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

#define PRINT_INT(value) printf("[zan_debug]"); printf(#value); printf(" :\t %d \n", value);
#define PRINT_FLOAT(value) printf("[zan_debug]"); printf(#value); printf(" :\t %f \n", value);
#define PRINT_HALF(value) printf("[zan_debug]"); printf(#value); printf(" :\t %f \n", __half2float(value));
#define PRINT_BFLOAT16(value) printf("[zan_debug]"); printf(#value); printf(" :\t  %f \n", __bfloat162float(value));


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
__device__ constexpr auto get_qk_block_gemm_qr_kr()
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
            false>{};
    }();

    using BlockGemmPolicy =
        ck_tile::BlockGemmARegBRegCRegV1CustomPolicy<scalar_t,
                                             scalar_t,
                                             acc_t,
                                             typename Traits::Gemm1BlockWarps,
                                             decltype(warp_gemm)>;
    return ck_tile::BlockGemmARegBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
}


// template <typename Traits, typename scalar_t>
// __device__ __inline__ static constexpr auto GetAlignmentV()
// {
// 	constexpr int32_t kBlockSize   = Traits::kBlockSize; // 256
// 	constexpr int32_t kNPerBlock   = Traits::kSizeDV; // 512
// 	constexpr int32_t kKPerBlock   = Traits::kBlockN; // 16
// 	constexpr int32_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize; // 32 
// 	constexpr int32_t kMaxVecLoad =
// 		min(total_pixels, static_cast<int32_t>(16 / sizeof(scalar_t))); // 8
// 	constexpr int32_t kMinVecLoad = 4 / sizeof(scalar_t); // 2
//
//     // total_pixels / kMaxVecLoad = 32 / 8 = 4
// 	constexpr int32_t kVecLoad = ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
// 									 ? kMaxVecLoad
// 									 : (total_pixels / kMinVecLoad);
// 	return kVecLoad;
// }
// template <typename Traits, typename scalar_t>
// CK_TILE_DEVICE static constexpr auto MakeVLdsTileDistribution()
// {
//
// 	constexpr index_t kBlockSize = Traits::kBlockSize;
// 	constexpr index_t kNPerBlock = Traits::kSizeDV;
// 	constexpr index_t kKPerBlock = Traits::kBlockN;
//
//     constexpr index_t N1 = GetAlignmentV<Traits, scalar_t>(); // 8
//     constexpr index_t N0 = kNPerBlock / N1; // 512 / 8 = 64
//
//     constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize; // 32
//
//     static_assert(total_pixels % N1 == 0);
//
//     constexpr index_t K3     = total_pixels / N1; // 32 / 8 = 4
//     constexpr index_t kKPack = Traits::kKPack; // 8
//     static_assert(kKPack % K3 == 0);
//     constexpr index_t K2 = kKPack / K3; // 8 / 4 = 2
//     if constexpr(get_warp_size() % (K2 * N0) == 0)
//     {
//         constexpr index_t K1 = get_warp_size() / (K2 * N0);
//         constexpr index_t K0 = kBlockSize / get_warp_size();
//         static_assert(kKPerBlock == K0 * K1 * K2 * K3);
//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
//                                        tuple<sequence<2>, sequence<2, 1, 2>>,
//                                        tuple<sequence<0>, sequence<1, 0, 2>>,
//                                        sequence<1, 2>,
//                                        sequence<1, 3>>{});
//     }
//     else
//     {
//         constexpr index_t K1   = (K2 * N0) / get_warp_size(); // (2 * 64) / 64 = 2
//         constexpr index_t K2_m = K2 / K1; // 2 / 2 = 1
//         constexpr index_t K0   = kBlockSize / get_warp_size() / K1; // 256 / 64 / 2 = 2
//         static_assert(kKPerBlock == K0 * K1 * K2_m * K3); // 2 * 2 * 1 * 4 = 16
//         return make_static_tile_distribution(
//             tile_distribution_encoding<sequence<1>,
//                                        tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
//                                        tuple<sequence<2, 2>, sequence<1, 2>>,
//                                        tuple<sequence<0, 1>, sequence<0, 2>>,
//                                        sequence<1, 2>,
//                                        sequence<1, 3>>{});
//     }
// }

template <typename Traits, typename scalar_t, typename acc_t>
__device__ constexpr auto get_kv_block_gemm_backup()
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

/*
		template <typename Traits, typename scalar_t, typename acc_t, bool Is_causal>
		__global__ void flash_fwd_splitkv_mla_two_group_kernel(
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
			constexpr int32_t kPackScalar = 16 / sizeof(scalar_t);
			constexpr int32_t kPackAcc = 16 / sizeof(scalar_t);
			constexpr int32_t kKPack = kPackScalar;

			constexpr auto I0 = ck_tile::number<0>{};
			constexpr auto I1 = ck_tile::number<1>{};
			constexpr auto IBlockM = ck_tile::number<kBlockM>{};
			constexpr auto IBlockN = ck_tile::number<kBlockN>{};
			constexpr auto IPack = ck_tile::number<kKPack>{};

			const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
			const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

			extern __shared__ char shared_memory[];
			char *shared_ptr = (char *)(((size_t)shared_memory + 255) & ~255);

			const int32_t tidx = threadIdx.x; 

			const auto make_q_dram = [&](const scalar_t* data) {
				return make_naive_tensor_view<ck_tile::address_space_enum::global>(
					data,
					ck_tile::make_tuple(params.size_s, kSizeD),
					ck_tile::make_tuple(params.stride_s_q, 1),
					IPack,
					I1);
			};

			const auto q_dram = make_naive_tensor_view<ck_tile::address_space_enum::global>(
				reinterpret_cast<const scalar_t*>(params.p_query),
				ck_tile::make_tuple(params.size_s, kSizeD),
				ck_tile::make_tuple(params.stride_s_q, 1),
				IPack,
				I1);

			const auto make_k_dram = [&](const scalar_t* data, int32_t height) {
				return make_naive_tensor_view<ck_tile::address_space_enum::global>(
					data,
					ck_tile::make_tuple(height, kSizeD),
					ck_tile::make_tuple(params.stride_s_k, 1),
					IPack,
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

			auto k_page_block_navigator_func = [&, i_nhead_k_ = i_nhead_k](int32_t i_batch_, int32_t seqlen_kv_end_, auto& k_dram_) {
				const auto* block_indices =
					reinterpret_cast<const int32_t*>(params.p_block_table) +
					i_batch_ * params.block_table_batch_stride;
				const int32_t num_blocks =
					ck_tile::integer_divide_ceil(seqlen_kv_end_, params.page_block_size);

				const int64_t fixed_offset =
					static_cast<int64_t>(i_nhead_k_) * params.stride_h_k;

				return ck_tile::make_page_block_navigator<const scalar_t, 0>(
					reinterpret_cast<scalar_t*>(params.p_key),
					params.stride_b_k, // kcache page-block stride/size
					fixed_offset,
					block_indices,
					num_blocks,
					params.page_block_size, // page_size
					k_dram_,
					make_k_dram(nullptr,
								(seqlen_kv_end_ - (num_blocks - 1) * params.page_block_size)));
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
			//                     seqlen_kv_ - (num_blocks - 1) * params.page_block_size));
			// };

			// TODO: add lds distribution
			constexpr auto lse_acc_dram_window_lengths = ck_tile::make_tuple(kBlockM);
			auto make_lse_acc_dram_window = [&, i_nhead_ = i_nhead, i_block_m_ = i_block_m](int32_t i_batch_, int32_t i_split_) {
				const int32_t split_offset = params.p_num_splits[i_batch_];
				acc_t* lse_acc_ptr = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
										 ((split_offset + i_split_) * params.size_h + i_nhead_) *
										 params.size_s + i_block_m_ * kBlockM;

				const auto lse_acc_dram = [&] {
					return make_naive_tensor_view<ck_tile::address_space_enum::global>(
						lse_acc_ptr,
						ck_tile::make_tuple(params.size_s),
						ck_tile::make_tuple(1),
						IPack,
						I1);
				}();

				return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {0});
			};

			auto lse_dram = make_naive_tensor_view<ck_tile::address_space_enum::global>(
					reinterpret_cast<acc_t*>(params.p_softmax_lse),
					ck_tile::make_tuple(params.size_s),
					ck_tile::make_tuple(1),
					IPack,
					I1);
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
						IPack,
						I1);
				}();

				return make_tile_window(o_acc_dram, o_dram_window_lengths, {0, 0});
			};

			auto o_dram = [&] {
				return make_naive_tensor_view<ck_tile::address_space_enum::global>(
					reinterpret_cast<scalar_t*>(params.p_output),
					ck_tile::make_tuple(params.size_s, kSizeDV),
					ck_tile::make_tuple(params.stride_s_o, 1),
					IPack,
					I1);
			}();
			auto o_dram_window = make_tile_window(o_dram, o_dram_window_lengths, {i_m0, 0});

			auto q_dram_window_lengths = ck_tile::make_tuple(IBlockM, kSizeD);

			auto k_dram_window_lengths = ck_tile::make_tuple(kBlockN, kSizeD);
#ifdef ZZDebug
			auto debug_k_dram_window_lengths = ck_tile::make_tuple(kBlockN, kBlockM);
#endif

			// auto v_dram_window_lengths =
			//     ck_tile::make_tuple(kBlockN, kSizeDV);

				// constexpr int32_t kNPerBlock = kBlockN; //16
				// constexpr int32_t kKPerBlock = kSizeD; //576
				// constexpr int32_t warpSize   = ck_tile::get_warp_size(); //64
				// constexpr int32_t NumWarps   = Traits::kNumGemm0Warps; // 2
				// constexpr int32_t kCopyBlockSize = NumWarps * warpSize; // 128
				//
				// constexpr int32_t KVector = kKPack; // 8
				//
				// // static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
				//
				// constexpr int32_t NumIssuesK = 9;
				// constexpr int32_t LanesPerK  = kKPerBlock / (KVector * NumIssuesK); // within a wave // 576 / 72 = 8 
				//
				// constexpr int32_t LaneGroups = warpSize / LanesPerK; // within a wave // 64 / 8 = 8;
				// constexpr int32_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);  // 16 / (8 * 2) = 1;
				//
				// // static_assert(NumIssues == kNPerBlock * kKPerBlock / (kCopyBlockSize * KVector));
				//
				// constexpr int32_t N0 = NumIssues;
				// constexpr int32_t N1 = LaneGroups;
				// constexpr int32_t N2 = NumWarps;
				//
				// constexpr int32_t K0 = NumIssuesK;
				// constexpr int32_t K1 = LanesPerK;
				// constexpr int32_t K2 = KVector;

#ifdef ZZDebug
			constexpr auto debug_k_dram_tile_distribution = make_static_tile_distribution(
				ck_tile::tile_distribution_encoding<
					ck_tile::sequence<>,
					ck_tile::tuple<ck_tile::sequence<1, 2, 8, 1>, ck_tile::sequence<1, 1, 8, 8>>,
					ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
					ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
					ck_tile::sequence<1, 1, 2, 2>,
					ck_tile::sequence<0, 3, 0, 3>>{});
#endif

			constexpr auto k_dram_tile_distribution = make_static_tile_distribution(
				ck_tile::tile_distribution_encoding<
					ck_tile::sequence<>,
					ck_tile::tuple<ck_tile::sequence<1, 2, 8, 1>, ck_tile::sequence<9, 1, 8, 8>>,
					ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
					ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
					ck_tile::sequence<1, 1, 2, 2>,
					ck_tile::sequence<0, 3, 0, 3>>{});

			constexpr auto debug_k_lds_tile_distribution = make_static_tile_distribution(
				ck_tile::tile_distribution_encoding<
					ck_tile::sequence<>,
					ck_tile::tuple<ck_tile::sequence<1, 2, 8, 1>, ck_tile::sequence<1, 1, 8, 8>>,
					ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
					ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
					ck_tile::sequence<1, 1, 2, 2>,
					ck_tile::sequence<0, 3, 0, 3>>{});

			// auto k_dram_tile_distribution = [&]() {
			//     constexpr index_t kBlockSize = Problem::kBlockSize;
			//     constexpr index_t kNPerBlock = kBlockM;
			//     constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;
			//
			//     constexpr index_t K1 = 16 / sizeof(KDataType);
			//     constexpr index_t K0 = kKPerBlock / K1;
			//     constexpr index_t N2 = get_warp_size() / K0;
			//     constexpr index_t N1 = kBlockSize / get_warp_size();
			//     constexpr index_t N0 = kNPerBlock / (N2 * N1);
			//
			//     return make_static_tile_distribution(
			//         tile_distribution_encoding<sequence<1>,
			//                                    tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
			//                                    tuple<sequence<1>, sequence<1, 2>>,
			//                                    tuple<sequence<1>, sequence<2, 0>>,
			//                                    sequence<1, 2>,
			//                                    sequence<0, 1>>{});
			// }();

			auto k_lds_block_descriptor = [&]() {
				constexpr auto k_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
					ck_tile::make_tuple(ck_tile::number<2 * kBlockN>{},
										ck_tile::number<kSizeD / kKPack>{},
										IPack),
					ck_tile::make_tuple(ck_tile::number<(kSizeD / kKPack + 1) * kKPack>{}, IPack, I1),
					IPack,
					I1);
				constexpr auto k_lds_block_desc_merge = ck_tile::transform_tensor_descriptor(
					k_lds_block_desc,
					ck_tile::make_tuple(
							ck_tile::make_pass_through_transform(ck_tile::number<2 * kBlockN>{}),
							ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kSizeDV / kKPack>{},
														  IPack))),
					ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
					ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

				return k_lds_block_desc_merge;
			}();

			auto v_lds_block_descriptor = [&]() {
				constexpr auto v_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
					ck_tile::make_tuple(ck_tile::number<2 * kBlockN>{},
										ck_tile::number<kSizeDV / kKPack>{},
										IPack),
					ck_tile::make_tuple(ck_tile::number<(kSizeD / kKPack + 1) * kKPack>{},
										IPack,
										I1),
					IPack,
					I1);

				constexpr auto v_lds_block_desc_transpose = ck_tile::transform_tensor_descriptor(
					v_lds_block_desc,
					ck_tile::make_tuple(
							ck_tile::make_pass_through_transform(ck_tile::number<2 * kBlockN>{}),
							ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kSizeDV / kKPack>{},
														  IPack))),
					ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
					ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{}));
				return v_lds_block_desc_transpose;
			}();

			auto p_lds_block_descriptor = [&]() {
				constexpr auto p_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
					ck_tile::make_tuple(kBlockM, kBlockN / kKPack, kKPack),
					ck_tile::make_tuple(kBlockN, kKPack, I1),
					IPack,
					I1);

				constexpr auto p_lds_block_desc_merge = ck_tile::transform_tensor_descriptor(
					p_lds_block_desc,
					ck_tile::make_tuple(
							ck_tile::make_pass_through_transform(IBlockM),
							ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kBlockN / kKPack>{},
														  IPack))),
					ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
					ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

				return p_lds_block_desc_merge;
			}();

			auto row_lds_block_descriptor = [&]() {
				constexpr auto row_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
					ck_tile::make_tuple(ck_tile::number<kBlockN / kKPack>{}, IPack),
					ck_tile::make_tuple(IPack, I1),
					IPack,
					I1);

				constexpr auto row_lds_block_desc_merge = ck_tile::transform_tensor_descriptor(
					row_lds_block_desc,
					ck_tile::make_tuple(
							ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kBlockN / kKPack>{},
														  IPack))),
					ck_tile::make_tuple(ck_tile::sequence<0, 1>{}),
					ck_tile::make_tuple(ck_tile::sequence<0>{}));
				return row_lds_block_desc_merge;
			}();

			//   auto ck_tile::make_generic_attention_mask_from_lr_window<ck_tile::GenericAttentionMask<true, false>>(
				// params.window_size_left,
				// params.window_size_right,
				// params.size_s,
				// params.seqlen_k,
				// ck_tile::GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
			auto gemm_1 = get_kv_block_gemm<Traits, scalar_t, acc_t>();
			auto o_acc = gemm_1.MakeCBlockTile();

			using OaccBlockType = decltype(ck_tile::cast_tile<acc_t>(o_acc));
			using MLBlockTileType = decltype(ck_tile::block_tile_reduce<acc_t>(
				OaccBlockType{}, ck_tile::sequence<1>{}, f_max, acc_t{0}));
			auto m_block = MLBlockTileType{};
			auto l_block = MLBlockTileType{};

			scalar_t* kv_lds_ptr = reinterpret_cast<scalar_t*>(shared_ptr);
			auto k_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
				kv_lds_ptr, k_lds_block_descriptor);

			auto k_lds_window =
				make_tile_window(k_lds,
					ck_tile::make_tuple(ck_tile::number<kBlockN>{},
										ck_tile::number<kSizeD>{}), {0, 0} );
			auto debug_k_ld_lds_window =
				make_tile_window(k_lds,
					ck_tile::make_tuple(ck_tile::number<kBlockN>{},
										ck_tile::number<kBlockM>{}), {0, 0}, debug_k_lds_tile_distribution);

			auto debug_k_ld_lds_window_1 =
				make_tile_window(k_lds,
					ck_tile::make_tuple(ck_tile::number<kBlockN>{},
										ck_tile::number<kBlockM>{}), {0, 0}, debug_k_lds_tile_distribution);

			auto k_st_lds_window =
				make_tile_window(k_lds,
					ck_tile::make_tuple(ck_tile::number<2 * kBlockN>{},
										ck_tile::number<kSizeD>{}), {0, 0});

			auto v_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
				kv_lds_ptr, v_lds_block_descriptor);
			auto v_lds_window =
				make_tile_window(v_lds,
					ck_tile::make_tuple(ck_tile::number<kSizeDV>{},
										IBlockN), {0, 0});

			scalar_t* p_lds_ptr = reinterpret_cast<scalar_t*>(
				reinterpret_cast<char*>(shared_ptr) + kLdsOffsetP);
			auto p_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
				p_lds_ptr, p_lds_block_descriptor);
			auto p_lds_ld_window = 
				make_tile_window(p_lds, ck_tile::make_tuple(kBlockM, kBlockN), {0, 0},
					ck_tile::remove_cvref_t<decltype(gemm_1)>::template MakeABlockTileDistribution<kBlockM, kBlockN>());
			auto p_lds_st_window = 
				make_tile_window(p_lds, ck_tile::make_tuple(kBlockM, kBlockN), {0, 0});

			acc_t* scale_lds_ptr = reinterpret_cast<acc_t*>(
				reinterpret_cast<char*>(shared_ptr) + kLdsOffsetScale);
			auto scale_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
				scale_lds_ptr, row_lds_block_descriptor);
			auto scale_lds_st_window =
				make_tile_window(scale_lds, ck_tile::make_tuple(kBlockM), {0});
			auto scale_lds_ld_window =
				make_tile_window(scale_lds, ck_tile::make_tuple(kBlockM), {0}, m_block.get_tile_distribution());


			acc_t* m_lds_ptr = reinterpret_cast<acc_t*>(reinterpret_cast<void*>(
				reinterpret_cast<char*>(shared_ptr) + kLdsOffsetMax));
			auto m_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
				m_lds_ptr, row_lds_block_descriptor);
			auto m_lds_st_window =
				make_tile_window(m_lds, ck_tile::make_tuple(kBlockM), {0});
			auto m_lds_ld_window =
				make_tile_window(m_lds, ck_tile::make_tuple(kBlockM), {0}, m_block.get_tile_distribution());

			acc_t* l_lds_ptr = reinterpret_cast<acc_t*>(reinterpret_cast<void*>(
				reinterpret_cast<char*>(shared_ptr) + kLdsOffsetSum));
			auto l_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
				l_lds_ptr, row_lds_block_descriptor);
			auto l_lds_st_window =
				make_tile_window(l_lds, ck_tile::make_tuple(kBlockM), {0});
			auto l_lds_ld_window =
				make_tile_window(l_lds, ck_tile::make_tuple(kBlockM), {0}, m_block.get_tile_distribution());

			TileSchedulerMetaData metadata;
			reinterpret_cast<int4*>(&(metadata.data))[0] = reinterpret_cast<int4*>(
				params.p_tile_scheduler_metadata)[i_partition * TileSchedulerMetaDataSizeInInt4];
			reinterpret_cast<int4*>(&(metadata.data))[1] = reinterpret_cast<int4*>(
				params.p_tile_scheduler_metadata)[i_partition * TileSchedulerMetaDataSizeInInt4 + 1];

			int32_t begin_batch_idx   = metadata.core.begin_batch_idx;
			int32_t begin_seqlen_idx  = metadata.core.begin_seqlen_idx;
			int32_t end_batch_idx     = metadata.core.end_batch_idx;
			int32_t end_seqlen_idx    = metadata.core.end_seqlen_idx;
			int32_t begin_n_split_idx = metadata.core.begin_n_split_idx;

			if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
			{
				PRINT_INT(begin_batch_idx);
				PRINT_INT(begin_seqlen_idx);
				PRINT_INT(end_batch_idx);
				PRINT_INT(end_seqlen_idx);
				PRINT_INT(begin_n_split_idx);
			}

			for (int32_t i_batch = begin_batch_idx; i_batch <= end_batch_idx; ++i_batch) {
				const int32_t i_split = i_batch == begin_batch_idx ? begin_n_split_idx : 0;
				const int32_t seqlen_k    = params.p_cu_seqlens_k[i_batch];
				const int32_t n_block_min = i_batch == begin_batch_idx ? begin_seqlen_idx / kBlockN : 0;
				const int32_t n_block_max = i_batch == end_batch_idx ? ck_tile::integer_divide_ceil(end_seqlen_idx, kBlockN) : ck_tile::integer_divide_ceil(seqlen_k, kBlockN);
				const bool NoSplit = n_block_min == 0 && n_block_max == ck_tile::integer_divide_ceil(seqlen_k, kBlockN);
				int32_t i_block_n = n_block_max - 1;

				const int32_t total_seqlen_kv = (n_block_max - n_block_min) * kBlockN;

				if (i_batch > begin_batch_idx) {
					__syncthreads();  // Barrier between two tiles.
				}
				ck_tile::clear_tile(o_acc);
				ck_tile::clear_tile(m_block);
				ck_tile::clear_tile(l_block);

				auto k_dram = make_k_dram(nullptr, total_seqlen_kv);
				// auto v_dram = make_v_dram(nullptr, total_seqlen_kv);

				auto k_page_block_navigator = k_page_block_navigator_func(i_batch, n_block_max * kBlockN, k_dram);
				// auto v_page_block_navigator = v_page_block_navigator_func(i_batch, total_seqlen_kv, v_dram);

				const int32_t wave_group_id = tidx / kNumThreadsSoftmax;
				if (wave_group_id == 0)
				{
					auto gemm_0 = get_qk_block_gemm<Traits, scalar_t, acc_t>();

					const int32_t row_offset_q = i_batch * params.stride_b_q + i_block_m * kBlockM * params.stride_s_q + i_nhead * params.stride_h_q;

					auto q_dram_window = make_tile_window(
						make_q_dram(reinterpret_cast<scalar_t*>(params.p_query) + row_offset_q),
						// q_dram,
						q_dram_window_lengths,
						{0, 0},
						ck_tile::remove_cvref_t<decltype(gemm_0)>::template MakeABlockTileDistribution<
							Traits::kBlockM,
							Traits::kSizeD>());

					auto q = load_tile(q_dram_window);

					auto s_acc = gemm_0.MakeCBlockTile();
					ck_tile::clear_tile(s_acc); // initialize C

					using SBlockTileType = decltype(ck_tile::cast_tile<acc_t>(s_acc));
					using MLWarpGroupTileType = decltype(ck_tile::block_tile_reduce<acc_t>(
						SBlockTileType{}, ck_tile::sequence<1>{}, f_max, acc_t{0}));

					auto m = MLWarpGroupTileType{};
					auto l = MLWarpGroupTileType{};

					if (i_block_n % 2 == 1)
					{
						move_tile_window(k_lds_window, {kBlockN, 0});
#ifdef ZZDebug
						move_tile_window(debug_k_ld_lds_window, {kBlockN, 0});
#endif
						move_tile_window(v_lds_window, {kSizeDV, 0});
					}
		// #ifdef ZZDebug
		//             if (tidx == 0 && blockIdx.x == 0 && blockIdx.y == 0 && (blockIdx.z == 0))
		//             {
		//                 const auto span_q2d = decltype(q)::get_distributed_spans();
		//                 sweep_tile_span(span_q2d[I0], [&](auto idx0) {
		//                     sweep_tile_span(span_q2d[I1], [&](auto idx1) {
		//                         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                         const auto tile_idx = get_x_indices_from_distributed_indices(
		//                             q.get_tile_distribution(), i_j_idx);
		//
		//                         auto row_id = tile_idx.at(I0);
		//                         auto col_id = tile_idx.at(I1);
		//
		//                         printf("q index [%d, %d] ", row_id, col_id);
		//                         printf("%f ,", ck_tile::type_convert<float>(q[i_j_idx]));
		//                     });
		//                     printf("\n");
		//                 });
		//             }
		// #endif

		// #ifdef ZZDebug
					// ck_tile::block_sync_lds();
					//
					// gemm_0(s_acc,
					//         q,
					//         k_lds_window);
		//             if (tidx == 0 && blockIdx.x == 0 && blockIdx.y == 0 && (blockIdx.z == 0))
		//             {
		//                 // auto k_tile = load_tile(debug_k_ld_lds_window);
		//                 // const auto span_q2d = decltype(k_tile)::get_distributed_spans();
		//                 // sweep_tile_span(span_q2d[I0], [&](auto idx0) {
		//                 //     sweep_tile_span(span_q2d[I1], [&](auto idx1) {
		//                 //         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                 //         const auto tile_idx = get_x_indices_from_distributed_indices(
		//                 //             k_tile.get_tile_distribution(), i_j_idx);
		//                 //
		//                 //         auto row_id = tile_idx.at(I0);
		//                 //         auto col_id = tile_idx.at(I1);
		//                 //
		//                 //         printf("load_lds_warp_1 index [%d, %d] ", row_id, col_id);
		//                 //         printf("%f ,", ck_tile::type_convert<float>(k_tile[i_j_idx]));
		//                 //     });
		//                 //     printf("\n");
		//                 // });
		//
		//                 // __syncthreads();
		//                 // const auto span_2d = decltype(s_acc)::get_distributed_spans();
		//                 // sweep_tile_span(span_2d[I0], [&](auto idx0) {
		//                 //     sweep_tile_span(span_2d[I1], [&](auto idx1) {
		//                 //         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                 //         const auto tile_idx = get_x_indices_from_distributed_indices(
		//                 //             s_acc.get_tile_distribution(), i_j_idx);
		//                 //
		//                 //         auto row_id = tile_idx.at(I0);
		//                 //         auto col_id = tile_idx.at(I1);
		//                 //
		//                 //         printf("s_acc [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(s_acc[i_j_idx]));
		//                 //     });
		//                 //     printf("\n");
		//                 // });
		//             }
		// #endif

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
								// constexpr auto i_idx = ck_tile::make_tuple(idx0);
								sweep_tile_span(sacc_spans[I1], [&](auto idx1) {
									constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
									const auto tile_idx = get_x_indices_from_distributed_indices(
										s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
									auto row_id = tile_idx.at(ck_tile::number<0>{});
									auto col_id = tile_idx.at(ck_tile::number<1>{});
									if constexpr (!Is_causal)
									{
										if (col_id >= int(seqlen_k - i_block_n * kBlockN))
											s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
									}
									else
									{
										int32_t col_limit_right = seqlen_k - 1 - i_block_n * kBlockN -
											(params.size_s - 1 - (i_block_m * kBlockM + row_id)) / params.num_groups;
										if (col_id >= col_limit_right)
											s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
									}
								});
							});
						}
		//
		//                 const auto s = ck_tile::cast_tile<acc_t>(s_acc);
		//                 auto m_local = block_tile_reduce<acc_t>(
		//                     s_acc,
		//                     ck_tile::sequence<1>{},
		//                     f_max,
		//                     -ck_tile::numeric<acc_t>::infinity());
		//                 block_tile_reduce_sync(m_local, f_max, ck_tile::bool_constant<false>{});
		//
		//                 const auto m_old = m;
		//                 tile_elementwise_inout(
		//                     [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);
		//
		//                 auto p_compute = make_static_distributed_tensor<acc_t>(
		//                     s_acc.get_tile_distribution());
		//
		//                 constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
		//                 sweep_tile_span(p_spans[I0], [&](auto idx0) {
		//                     constexpr auto i_idx = ck_tile::make_tuple(idx0);
		//                     auto row_max = params.scale_softmax_log2 * m[i_idx];
		//                     sweep_tile_span(p_spans[I1], [&](auto idx1) {
		//                         constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                         p_compute(i_j_idx) = exp2(params.scale_softmax_log2 * s_acc[i_j_idx] - row_max);
		//                     });
		//                 });
		//
		//                 auto rowsum_p = block_tile_reduce<acc_t>(
		//                     p_compute, ck_tile::sequence<1>{}, f_sum, acc_t{0});
		//                 ck_tile::block_tile_reduce_sync(rowsum_p, f_sum, ck_tile::bool_constant<false>{});
		//
		//                 auto scale_o = make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
		//                 constexpr auto m_spans = decltype(m)::get_distributed_spans();
		//                 sweep_tile_span(m_spans[I0], [&](auto idx0) {
		//                     constexpr auto i_idx = ck_tile::make_tuple(idx0);
		//                     auto row_max = params.scale_softmax_log2 * m[i_idx];
		//                     scale_o(i_idx) = exp2(params.scale_softmax_log2 * m_old[i_idx] - row_max);
		//                     l(i_idx) = scale_o[i_idx] * l[i_idx] + rowsum_p[i_idx];
		//                 });
		//                 
		//                 // TODO: !!!!!!this p is distribution in wave group 1, gemm_1 use p with distribution in block;
		//                 auto p = ck_tile::cast_tile<scalar_t>(p_compute);
		//                 store_tile(p_lds_st_window, p);
		//                 store_tile(scale_lds_st_window, scale_o);
		//
		//                 // l{j}, Oacc{j}
		//                 auto scale_o_block = load_tile(scale_lds_ld_window);
		//                 constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
		//                 sweep_tile_span(o_spans[I0], [&](auto idx0) {
		//                     constexpr auto i_idx = ck_tile::make_tuple(idx0);
		//                     sweep_tile_span(o_spans[I1], [&](auto idx1) {
		//                         constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                         o_acc(i_j_idx) *= scale_o_block[i_idx];
		//                     });
		//                 });
		//
		//                 auto p_block = load_tile(p_lds_ld_window);
		//                 gemm_1(o_acc,
		//                         p_block,
		//                         v_lds_window);
		//                 ck_tile::block_sync_lds();
		//
		//                 auto k_lds_offset = i_block_n % 2 == 0 ? 0 : kBlockN;
		//                 auto v_lds_offset = i_block_n % 2 == 0 ? 0 : kSizeDV;
		//                 move_tile_window(k_lds_window, {k_lds_offset, 0});
		//                 move_tile_window(v_lds_window, {v_lds_offset, 0});
					}
		//             store_tile(m_lds_st_window, m);
		//             store_tile(l_lds_st_window, l);
				}
				else
				{
					// move_tile_window(k_lds_window, {-kBlockN, 0});
					auto [i_page_block_k, k_dram_window] = k_page_block_navigator.make_tile_window(
						k_dram_window_lengths, {(n_block_min - 1) * kBlockN, 0}, k_dram_tile_distribution);
		// #ifdef ZZDebug
		//             auto [i_page_block_k, k_dram_window] = k_page_block_navigator.make_tile_window(
		//                 debug_k_dram_window_lengths, {(n_block_min - 1) * kBlockN, 0}, debug_k_dram_tile_distribution);
		// #endif
					
					// auto [i_page_block_v, v_dram_window] = v_page_block_navigator.make_tile_window(
					//     v_dram_block_window_lengths,
					//     {0, aligned_physical_seqlen_k_start},
					//     Policy::template MakeVDramTileDistribution<Problem>());

					// auto k_dram_window = make_tile_window(
					//     make_k_dram(reinterpret_cast<scalar_t*>(params.p_key), total_seqlen_kv),
					//     k_dram_window_lengths,
					//     {0, 0},
					//     k_dram_tile_distribution); // K DRAM tile window for

					if (i_block_n % 2 == 1)
					{
						move_tile_window(k_st_lds_window, {kBlockN, 0});
						move_tile_window(debug_k_ld_lds_window_1, {kBlockN, 0});
						move_tile_window(v_lds_window, {kSizeDV, 0});
					}

					auto k_block_tile = load_tile(k_dram_window);
		// #ifdef ZZDebug 
		//             if (threadIdx.x == 128 && blockIdx.x == 0 && blockIdx.y == 0 && (blockIdx.z == 0))
		//             {
		//
		//                 auto block_idx = ck_tile::integer_divide_floor(n_block_min * kBlockN, params.page_block_size);
		//                 const int32_t length              = n_block_min * kBlockN;
		//                 const int32_t num_complete_blocks = ck_tile::integer_divide_floor(length, params.page_block_size);
		//
		//                 PRINT_INT(n_block_min);
		//                 PRINT_INT(block_idx);
		//                 PRINT_INT(block_idx);
		//                 PRINT_INT(num_complete_blocks);
		//                 PRINT_INT(length - params.page_block_size * num_complete_blocks);
		//
		//
		//                 const auto span_2d = decltype(k_block_tile)::get_distributed_spans();
		//                 sweep_tile_span(span_2d[I0], [&](auto idx0) {
		//                     sweep_tile_span(span_2d[I1], [&](auto idx1) {
		//                         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                         const auto tile_idx = get_x_indices_from_distributed_indices(
		//                             k_block_tile.get_tile_distribution(), i_j_idx);
		//
		//                         auto row_id = tile_idx.at(I0);
		//                         auto col_id = tile_idx.at(I1);
		//
		//                         printf("k_block_tile index [%d, %d] ", row_id, col_id);
		//                         printf("%f ,", ck_tile::type_convert<float>(k_block_tile[i_j_idx]));
		//                     });
		//                     printf("\n");
		//                 });
		//             }
		// #endif
					__syncthreads();

					i_page_block_k = k_page_block_navigator.move_tile_window(i_page_block_k, k_dram_window, {kBlockN, 0});
					// move_tile_window(k_dram_window, {kBlockN, 0});

					store_tile(k_st_lds_window, k_block_tile);
					ck_tile::block_sync_lds();

#ifdef ZZDebug 
					if (tidx == 128 && blockIdx.x == 0 && blockIdx.y == 0 && (blockIdx.z == 0))
					{
						auto k_tile = load_tile(debug_k_ld_lds_window_1);

						const auto span_q2d = decltype(k_tile)::get_distributed_spans();
						sweep_tile_span(span_q2d[I0], [&](auto idx0) {
							sweep_tile_span(span_q2d[I1], [&](auto idx1) {
								const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
								const auto tile_idx = get_x_indices_from_distributed_indices(
									k_tile.get_tile_distribution(), i_j_idx);

								auto row_id = tile_idx.at(I0);
								auto col_id = tile_idx.at(I1);

								printf("index lds_wrap 2 [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(k_tile[i_j_idx]));
								// printf("%f ,", ck_tile::type_convert<float>(k_tile[i_j_idx]));
							});
							printf("\n");
						});
					}
#endif

					ck_tile::block_sync_lds();

		//
		//             for (; i_block_n >= n_block_min; --i_block_n)
		//             {
		//                 __syncthreads();
		//                 if (i_block_n - 1 >= n_block_min)
		//                 {
		//                     auto k_lds_offset = i_block_n % 2 == 0 ? 0 : kBlockN;
		//                     move_tile_window(k_lds_window, {k_lds_offset, 0});
		//
		//                     k_block_tile = load_tile(k_dram_window);
		//                     move_tile_window(k_dram_window, {kBlockN, 0});
		//                     store_tile(k_lds_window, k_block_tile);
		//                     ck_tile::block_sync_lds();
		//                 }
		//                 auto p_block = load_tile(p_lds_ld_window);
		//                 auto scale_o_block = load_tile(scale_lds_ld_window );
		//                 // tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, acc);
		//
		//                 auto o_acc = OaccBlockType{};
		//
		//                 constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
		//                 sweep_tile_span(o_spans[I0], [&](auto idx0) {
		//                     constexpr auto i_idx = ck_tile::make_tuple(idx0);
		//                     sweep_tile_span(o_spans[I1], [&](auto idx1) {
		//                         constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                         o_acc(i_j_idx) *= scale_o_block[i_idx];
		//                     });
		//                 });
		//
		//                 ck_tile::block_sync_lds();
		//                 gemm_1(o_acc,
		//                         p_block,
		//                         v_lds_window);
		//                 ck_tile::block_sync_lds();
		//
		//                 auto v_lds_offset = i_block_n % 2 == 0 ? 0 : kSizeDV;
		//                 move_tile_window(v_lds_window, {v_lds_offset, 0});
		//             }
				}
		//
		//         load_tile(m_block, m_lds_ld_window);
		//         load_tile(l_block, l_lds_ld_window);
		//         // Epilogue
		//         auto lse_acc = make_static_distributed_tensor<acc_t>(m_block.get_tile_distribution());
		//         constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
		//         sweep_tile_span(lse_acc_spans[I0], [&](auto idx0) {
		//             constexpr auto i_idx = ck_tile::make_tuple(idx0);
		//             lse_acc(i_idx) = m_block[i_idx] * params.scale_softmax + log(l_block[i_idx]);
		//         });
		//
		//         if (NoSplit)
		//         {
		//             store_tile(lse_dram_window, lse_acc);
		//         }
		//         else
		//         {
		//             auto lse_acc_dram_window = make_lse_acc_dram_window(i_batch, i_split);
		//             store_tile(lse_acc_dram_window, lse_acc);
		//         }
		//
		//         constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
		//         sweep_tile_span(o_spans[I0], [&](auto idx0) {
		//             constexpr auto i_idx = ck_tile::make_tuple(idx0);
		//             const auto tmp = [&]() {
		//                     return l_block[i_idx] == 0.f ? 0.f : 1 / l_block[i_idx];
		//             }();
		//             sweep_tile_span(o_spans[I1], [&](auto idx1) {
		//                 constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
		//                 o_acc(i_j_idx) *= tmp;
		//             });
		//         });
		//         if (NoSplit)
		//         {
		//             store_tile(o_dram_window, ck_tile::cast_tile<scalar_t>(o_acc));
		//         }
		//         else
		//         {
		//             auto o_acc_dram_window = make_o_acc_dram_window(i_batch, i_split);
		//             store_tile(o_acc_dram_window, o_acc);
		//         }
			}
		}
*/


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

    constexpr int32_t kPackScalar = 16 / sizeof(scalar_t);
    constexpr int32_t kPackAcc = 16 / sizeof(scalar_t);
    constexpr int32_t kKPack = kPackScalar;

    constexpr auto I0 = ck_tile::number<0>{};
    constexpr auto I1 = ck_tile::number<1>{};
    constexpr auto IBlockM = ck_tile::number<kBlockM>{};
    constexpr auto IBlockN = ck_tile::number<kBlockN>{};
    constexpr auto IPack = ck_tile::number<kKPack>{};

    const int32_t i_block_m   = blockIdx.x;
    const int32_t i_nhead     = blockIdx.y;
    const int32_t i_nhead_k   = i_nhead / params.hq_hk_ratio;
    const int32_t i_partition = blockIdx.z;

    const ck_tile::index_t i_m0 = __builtin_amdgcn_readfirstlane(i_block_m * kBlockM);

    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    extern __shared__ char shared_memory[];
	char *shared_ptr = (char *)(((size_t)shared_memory + 255) & ~255);

    const int32_t tidx = threadIdx.x; 

    const auto make_q_dram = [&](const scalar_t* data) {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            data,
            ck_tile::make_tuple(params.size_s, kSizeD),
            ck_tile::make_tuple(params.stride_s_q, 1),
            IPack,
            I1);
    };

    const auto make_k_dram = [&](const scalar_t* data, int32_t height) {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            data,
            ck_tile::make_tuple(height, kSizeD),
            ck_tile::make_tuple(params.stride_s_k, 1),
            IPack,
            I1);
    };

    auto k_page_block_navigator_func = [&, i_nhead_k_ = i_nhead_k](int32_t i_batch_, int32_t seqlen_kv_end_, auto& k_dram_) {
        const auto* block_indices =
            reinterpret_cast<const int32_t*>(params.p_block_table) +
            i_batch_ * params.block_table_batch_stride;
        const int32_t num_blocks =
            ck_tile::integer_divide_ceil(seqlen_kv_end_, params.page_block_size);

        const int64_t fixed_offset =
            static_cast<int64_t>(i_nhead_k_) * params.stride_h_k;

        return ck_tile::make_page_block_navigator<const scalar_t, 0>(
            reinterpret_cast<scalar_t*>(params.p_key),
            params.stride_b_k, // kcache page-block stride/size
            fixed_offset,
            block_indices,
            num_blocks,
            params.page_block_size, // page_size
            k_dram_,
            make_k_dram(nullptr,
                        (seqlen_kv_end_ - (num_blocks - 1) * params.page_block_size)));
    };

    // TODO: add lds distribution
    constexpr auto lse_acc_dram_window_lengths = ck_tile::make_tuple(kBlockM);
    auto make_lse_acc_dram_window = [&, i_nhead_ = i_nhead, i_block_m_ = i_block_m](int32_t i_batch_, int32_t i_split_) {
        const int32_t split_offset = params.p_num_splits[i_batch_];
        acc_t* lse_acc_ptr = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
                                 ((split_offset + i_split_) * params.size_h + i_nhead_) *
                                 params.size_s + i_block_m_ * kBlockM;

        const auto lse_acc_dram = [&] {
            return make_naive_tensor_view<ck_tile::address_space_enum::global>(
                lse_acc_ptr,
                ck_tile::make_tuple(params.size_s),
                ck_tile::make_tuple(1),
                IPack,
                I1);
        }();

        return make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {0});
    };

    auto lse_dram = make_naive_tensor_view<ck_tile::address_space_enum::global>(
            reinterpret_cast<acc_t*>(params.p_softmax_lse),
            ck_tile::make_tuple(params.size_s),
            ck_tile::make_tuple(1),
            IPack,
            I1);
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
                IPack,
                I1);
        }();

        return make_tile_window(o_acc_dram, o_dram_window_lengths, {0, 0});
    };

    auto o_dram = [&] {
        return make_naive_tensor_view<ck_tile::address_space_enum::global>(
            reinterpret_cast<scalar_t*>(params.p_output),
            ck_tile::make_tuple(params.size_s, kSizeDV),
            ck_tile::make_tuple(params.stride_s_o, 1),
            IPack,
            I1);
    }();
    auto o_dram_window = make_tile_window(o_dram, o_dram_window_lengths, {i_m0, 0});

    auto q_dram_window_lengths = ck_tile::make_tuple(IBlockM, kSizeD);

    auto k_dram_window_lengths = ck_tile::make_tuple(kBlockN, kSizeD);

#ifdef ZZDebug
    auto debug_q_dram_window_lengths = ck_tile::make_tuple(IBlockM, kBlockN);
    auto debug_k_dram_window_lengths = ck_tile::make_tuple(kBlockN, kBlockM);

    constexpr auto debug_k_dram_tile_distribution = make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<1, 2, 8, 1>, ck_tile::sequence<1, 2, 8, 4>>,
            ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
            ck_tile::sequence<1, 1, 2, 2>,
            ck_tile::sequence<0, 3, 0, 3>>{});

    constexpr auto debug_k_lds_tile_distribution = make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<1, 2, 8, 1>, ck_tile::sequence<1, 2, 8, 4>>,
            ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
            ck_tile::sequence<1, 1, 2, 2>,
            ck_tile::sequence<0, 3, 0, 3>>{});

    constexpr auto debug_v_lds_reg_block_distribution = [&]() {
        constexpr auto v_lds_shuffle_tile_distribution = make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<64, 1>, ck_tile::sequence<2, 2, 1, 4>>,
                ck_tile::tuple<ck_tile::sequence<2, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<0, 2>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<1, 3>>{});
        return v_lds_shuffle_tile_distribution;
    }();
#endif

    constexpr auto k_dram_tile_distribution = make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<1, 2, 8, 1>, ck_tile::sequence<9, 2, 8, 4>>,
            ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
            ck_tile::sequence<1, 1, 2, 2>,
            ck_tile::sequence<0, 3, 0, 3>>{});


    auto k_lds_block_descriptor = [&]() {
        constexpr auto k_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<2 * kBlockN>{},
                                ck_tile::number<kSizeD / kKPack>{},
                                IPack),
            ck_tile::make_tuple(ck_tile::number<(kSizeD / kKPack + 1) * kKPack>{}, IPack, I1),
            IPack,
            I1);
        constexpr auto k_lds_block_desc_merge = ck_tile::transform_tensor_descriptor(
            k_lds_block_desc,
            ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ck_tile::number<2 * kBlockN>{}),
                    ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kSizeDV / kKPack>{},
                                                  IPack))),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return k_lds_block_desc_merge;
    }();

    auto v_lds_block_descriptor = [&]() {
        constexpr auto v_lds_block_desc = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<2 * kBlockN>{},
                                ck_tile::number<kSizeDV / kKPack>{},
                                IPack),
            ck_tile::make_tuple(ck_tile::number<(kSizeD / kKPack + 1) * kKPack>{},
                                IPack,
                                I1),
            IPack,
            I1);

        constexpr auto v_lds_block_desc_transpose = ck_tile::transform_tensor_descriptor(
            v_lds_block_desc,
            ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ck_tile::number<2 * kBlockN>{}),
                    ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kSizeDV / kKPack>{},
                                                  IPack))),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
            ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{}));
        return v_lds_block_desc_transpose;
    }();

    auto gemm_0 = get_qk_block_gemm<Traits, scalar_t, acc_t>();
    auto s_acc = gemm_0.MakeCBlockTile();
    using SBlockTileType = decltype(ck_tile::cast_tile<acc_t>(s_acc));
    using MLBlockTileType = decltype(ck_tile::block_tile_reduce<acc_t>(
        SBlockTileType{}, ck_tile::sequence<1>{}, f_max, acc_t{0}));
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};

    auto gemm_1 = get_kv_block_gemm<Traits, scalar_t, acc_t>();
    auto o_acc = gemm_1.MakeCBlockTile();

    scalar_t* kv_lds_ptr = reinterpret_cast<scalar_t*>(shared_ptr);
    auto k_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
        kv_lds_ptr, k_lds_block_descriptor);

    auto k_st_lds_window =
        make_tile_window(k_lds,
            ck_tile::make_tuple(ck_tile::number<kBlockN>{},
                                ck_tile::number<kSizeD>{}), {0, 0});
    auto k_ld_lds_window =
        make_tile_window(k_lds,
            ck_tile::make_tuple(ck_tile::number<kBlockN>{},
                                ck_tile::number<kSizeD>{}), {0, 0});

#ifdef ZZDebug
    auto debug_k_ld_lds_window =
        make_tile_window(k_lds,
            ck_tile::make_tuple(ck_tile::number<kBlockN>{},
                                ck_tile::number<kSizeD>{}),
            {0, 0},
            debug_k_lds_tile_distribution);
#endif

    auto v_lds = make_tensor_view<ck_tile::address_space_enum::lds>(
        kv_lds_ptr, v_lds_block_descriptor);
    constexpr auto v_lds_reg_block_distribution = [&]() {
		using GemmProblem = ck_tile::BlockGemmProblem<scalar_t,
													  scalar_t,
													  acc_t,
													  Traits::kNumGemm1Warps * ck_tile::get_warp_size(),
													  ck_tile::TileGemmShape<ck_tile::sequence<Traits::kBlockM,
																							   Traits::kSizeDV,
																							   Traits::kBlockN>,
														  typename Traits::Gemm1BlockWarps,
														  typename Traits::Gemm1WarpTile>>;

        constexpr auto config = decltype(gemm_1)::Policy::template GetWarpGemmMWarpNWarp<GemmProblem>();
        using WarpGemm        = ck_tile::remove_cvref_t<decltype(config.template at<0>())>;
        constexpr int32_t MWarp = Traits::Gemm1BlockWarps::at(ck_tile::number<0>{});
        constexpr int32_t NWarp = Traits::Gemm1BlockWarps::at(ck_tile::number<1>{});

        constexpr int32_t kNPerBlock = Traits::kSizeDV;
        constexpr int32_t kKPerBlock = Traits::kBlockN;

        constexpr int32_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr int32_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // constexpr auto vt_lds_shuffle_tile_encode =
        //     ck_tile::tile_distribution_encoding<
        //         ck_tile::sequence<>,
        //         ck_tile::tuple<ck_tile::sequence<64, 8>, ck_tile::sequence<2, 2, 1, 4>>,
        //         ck_tile::tuple<ck_tile::sequence<2, 2>, ck_tile::sequence<1, 2>>,
        //         ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<0, 2>>,
        //         ck_tile::sequence<1, 2>,
        //         ck_tile::sequence<1, 3>>{};
        constexpr auto vt_lds_outer_encode =
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<MWarp>,
                ck_tile::tuple<ck_tile::sequence<NIterPerWarp, NWarp>, ck_tile::sequence<KIterPerWarp>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 0>>{};

        constexpr auto vt_lds_shuffle_block_dstr_encode = ck_tile::detail::make_embed_tile_distribution_encoding(
            vt_lds_outer_encode, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto vt_lds_block_dstr = make_static_tile_distribution(vt_lds_shuffle_block_dstr_encode);
        return vt_lds_block_dstr;
    }();

    auto v_ld_lds_window =
        make_tile_window(v_lds,
            ck_tile::make_tuple(ck_tile::number<kSizeD>{},
                                ck_tile::number<kBlockN>{}),
            {0, 0},
            v_lds_reg_block_distribution);

#ifdef ZZDebug
    auto debug_v_ld_lds_window =
        make_tile_window(v_lds,
            ck_tile::make_tuple(ck_tile::number<kBlockN>{},
                                ck_tile::number<kSizeD>{}),
            {0, 0},
            debug_v_lds_reg_block_distribution);
#endif

    auto p_shuffle_distribution = [&]() {
        auto p_encoding = s_acc.get_tile_distribution().get_static_tile_distribution_encoding();
		constexpr auto N = p_encoding.hs_lengthss_.at(I0);
		constexpr auto K = p_encoding.hs_lengthss_.at(I1);
		constexpr auto N0 = N[0];
		constexpr auto N1 = N[1];
		constexpr auto N2 = N[2];
		constexpr auto K0 = K[0];
		constexpr auto K1 = K[1];
		constexpr auto K2 = K[2];
		constexpr auto K3 = K[3];
		constexpr auto K4 = K[4];

        return make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<ck_tile::sequence<K1>,
                                       ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K2, K3, K4>>,
                                       ck_tile::tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<2, 1>>,
                                       ck_tile::tuple<ck_tile::sequence<1, 0>, ck_tile::sequence<1, 2>>,
                                       ck_tile::sequence<1, 2, 2>,
                                       ck_tile::sequence<0, 0, 2>>{});
    }();

    TileSchedulerMetaData metadata;
    reinterpret_cast<int4*>(&(metadata.data))[0] = reinterpret_cast<int4*>(
        params.p_tile_scheduler_metadata)[i_partition * TileSchedulerMetaDataSizeInInt4];
    reinterpret_cast<int4*>(&(metadata.data))[1] = reinterpret_cast<int4*>(
        params.p_tile_scheduler_metadata)[i_partition * TileSchedulerMetaDataSizeInInt4 + 1];

    const int32_t begin_batch_idx   = metadata.core.begin_batch_idx;
    const int32_t begin_seqlen_idx  = metadata.core.begin_seqlen_idx;
    const int32_t end_batch_idx     = metadata.core.end_batch_idx;
    const int32_t end_seqlen_idx    = metadata.core.end_seqlen_idx;
    const int32_t begin_n_split_idx = metadata.core.begin_n_split_idx;

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
    //    {
    //        PRINT_INT(begin_batch_idx);
    //        PRINT_INT(begin_seqlen_idx);
    //        PRINT_INT(end_batch_idx);
    //        PRINT_INT(end_seqlen_idx);
    //        PRINT_INT(begin_n_split_idx);
    //    }

    for (int32_t i_batch = begin_batch_idx; i_batch <= begin_batch_idx; ++i_batch)
    {
        const int32_t i_split = i_batch == begin_batch_idx ? begin_n_split_idx : 0;
        const int32_t seqlen_k    = params.p_cu_seqlens_k[i_batch];
        const int32_t n_block_min = i_batch == begin_batch_idx ? begin_seqlen_idx / kBlockN : 0;
        const int32_t n_block_max = i_batch == end_batch_idx ? ck_tile::integer_divide_ceil(end_seqlen_idx, kBlockN) : ck_tile::integer_divide_ceil(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == ck_tile::integer_divide_ceil(seqlen_k, kBlockN);
        int32_t i_block_n = n_block_max - 1;

        const int32_t total_seqlen_kv = (n_block_max - n_block_min) * kBlockN;

        if (i_batch > begin_batch_idx)
        {
            __syncthreads();
        }

        ck_tile::clear_tile(o_acc);
        ck_tile::clear_tile(m);
        ck_tile::clear_tile(l);
        ck_tile::clear_tile(s_acc);

        const int32_t row_offset_q = i_batch * params.stride_b_q +
                                     i_block_m * kBlockM * params.stride_s_q +
                                     i_nhead * params.stride_h_q;
        auto q_dram_window = make_tile_window(
            make_q_dram(reinterpret_cast<scalar_t*>(params.p_query) + row_offset_q),
            q_dram_window_lengths,
            {0, 0},
            ck_tile::remove_cvref_t<decltype(gemm_0)>::template MakeABlockTileDistribution<
                Traits::kBlockM,
                Traits::kSizeD>());
        auto q = load_tile(q_dram_window);


        auto k_dram = make_k_dram(nullptr, total_seqlen_kv);
        auto k_page_block_navigator = k_page_block_navigator_func(i_batch, n_block_max * kBlockN, k_dram);
        auto [i_page_block_k, k_dram_window] = k_page_block_navigator.make_tile_window(
            k_dram_window_lengths, {(n_block_max - 1) * kBlockN, 0}, k_dram_tile_distribution);

// #ifdef ZZDebug
//         if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && (blockIdx.z == 0))
//         {
//             // // TODO: query dram ready
//             // auto debug_q_dram_window = make_tile_window(
//             //     make_q_dram(reinterpret_cast<scalar_t*>(params.p_query) + row_offset_q),
//             //     debug_q_dram_window_lengths,
//             //     {0, 0},
//             //     ck_tile::remove_cvref_t<decltype(gemm_0)>::template MakeABlockTileDistribution<
//             //         Traits::kBlockM,
//             //         Traits::kBlockN>());
//             // auto debug_q = load_tile(debug_q_dram_window);
//             // const auto span_q2d = decltype(debug_q)::get_distributed_spans();
//             // sweep_tile_span(span_q2d[I0], [&](auto idx0) {
//             //     sweep_tile_span(span_q2d[I1], [&](auto idx1) {
//             //         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
//             //         const auto tile_idx = get_x_indices_from_distributed_indices(
//             //             debug_q.get_tile_distribution(), i_j_idx);
//             //         auto row_id = tile_idx.at(I0);
//             //         auto col_id = tile_idx.at(I1);
//             //         printf("debug_q index [%d, %d] %f ", row_id, col_id, ck_tile::type_convert<float>(debug_q[i_j_idx]));
//             //     });
//             //     printf("\n");
//             // });
//
//             // // TODO: key dram ready
//             // auto debug_k_page_block_navigator = k_page_block_navigator_func(i_batch, n_block_max * kBlockN, k_dram);
//             // auto [debug_i_page_block_k, debug_k_dram_window] = debug_k_page_block_navigator.make_tile_window(
//             //     debug_k_dram_window_lengths, {(n_block_max - 1) * kBlockN, 0}, debug_k_dram_tile_distribution);
//             // auto debug_k = load_tile(debug_k_dram_window);
//             // const auto span_k2d = decltype(debug_k)::get_distributed_spans();
//             // sweep_tile_span(span_k2d[I0], [&](auto idx0) {
//             //     sweep_tile_span(span_k2d[I1], [&](auto idx1) {
//             //         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
//             //         const auto tile_idx = get_x_indices_from_distributed_indices(
//             //             debug_k.get_tile_distribution(), i_j_idx);
//             //         auto row_id = tile_idx.at(I0);
//             //         auto col_id = tile_idx.at(I1);
//             //         printf("debug_k index [%d, %d] %f ", row_id, col_id, ck_tile::type_convert<float>(debug_k[i_j_idx]));
//             //     });
//             //     printf("\n");
//             // });
//         }
// #endif

        int32_t k_st_lds_offset = kBlockN;
        int32_t k_ld_lds_offset = kBlockN;

//         if (i_block_n % 2 == 1)
//         {
//             move_tile_window(k_st_lds_window, {kBlockN, 0});
//             move_tile_window(k_ld_lds_window, {kBlockN, 0});
//
// #ifdef ZZDebug
//             move_tile_window(debug_k_ld_lds_window, {kBlockN, 0});
//             move_tile_window(debug_v_ld_lds_window, {kSizeDV, 0});
// #endif
//
//             move_tile_window(v_ld_lds_window, {kSizeDV, 0});
//             k_st_lds_offset = -k_st_lds_offset;
//             k_ld_lds_offset = -k_st_lds_offset;
//         }

        auto k_block_tile = load_tile(k_dram_window);
        i_page_block_k = k_page_block_navigator.move_tile_window(i_page_block_k, k_dram_window, {-kBlockN, 0});

        store_tile(k_st_lds_window, k_block_tile);
        move_tile_window(k_st_lds_window, {k_st_lds_offset, 0});
        k_st_lds_offset = -k_st_lds_offset;

// #ifdef ZZDebug
//         ck_tile::block_sync_lds();
//         if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && (blockIdx.z == 0))
//         {
//             auto debug_k_lds = load_tile(debug_k_ld_lds_window);
//             const auto span_k2d = decltype(debug_k_lds)::get_distributed_spans();
//             sweep_tile_span(span_k2d[I0], [&](auto idx0) {
//                 sweep_tile_span(span_k2d[I1], [&](auto idx1) {
//                     const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
//                     const auto tile_idx = get_x_indices_from_distributed_indices(
//                         debug_k_lds.get_tile_distribution(), i_j_idx);
//                     auto row_id = tile_idx.at(I0);
//                     auto col_id = tile_idx.at(I1);
//                     printf("debug_k_lds index [%d, %d] %f ", row_id, col_id, ck_tile::type_convert<float>(debug_k_lds[i_j_idx]));
//                 });
//                 printf("\n");
//             });
//         }
// #endif

        constexpr int n_masking_steps = !Is_causal ? 1 : ck_tile::integer_divide_ceil(kBlockM, kBlockN) + 1;
        int masking_step = n_masking_steps;
        for (; i_block_n > n_block_min; --masking_step, --i_block_n)
        {
            ck_tile::block_sync_lds();
            gemm_0(s_acc,
                   q,
                   k_ld_lds_window);

#ifdef ZZDebug
            //TODO: s_acc is ready
            if (tidx == DEBUG_TID && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
            {
                __syncthreads();
                const auto span_2d = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(span_2d[I0], [&](auto idx0) {
                    sweep_tile_span(span_2d[I1], [&](auto idx1) {
                        const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), i_j_idx);

                        auto row_id = tile_idx.at(I0);
                        auto col_id = tile_idx.at(I1);

                        printf("s_acc [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(s_acc[i_j_idx]));
                    });
                    printf("\n");
                });
            }
#endif

            auto k_block_tile = load_tile(k_dram_window);
            i_page_block_k = k_page_block_navigator.move_tile_window(i_page_block_k, k_dram_window, {-kBlockN, 0});
            move_tile_window(k_st_lds_window, {k_st_lds_offset, 0});

            const bool is_masking_step = masking_step > 0;
            const bool is_first_masking_step = masking_step == n_masking_steps;


            // if seq_len == 1, never need to add mask to s
            if (is_masking_step) {
                constexpr auto sacc_spans = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(sacc_spans[I0], [&](auto idx0) {
                    // constexpr auto i_idx = ck_tile::make_tuple(idx0);
                    sweep_tile_span(sacc_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                        auto row_id = tile_idx.at(ck_tile::number<0>{});
                        auto col_id = tile_idx.at(ck_tile::number<1>{});
                        if constexpr (!Is_causal)
                        {
                            if (col_id >= int(seqlen_k - i_block_n * kBlockN))
                                s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
// #ifdef ZZDebug
//                             if (tidx == DEBUG_TID && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
//                                 printf("s_acc [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(s_acc[i_j_idx]));
// #endif
                        }
                        else
                        {
                            int32_t col_limit_right = seqlen_k - 1 - i_block_n * kBlockN -
                                (params.size_s - 1 - (i_block_m * kBlockM + row_id)) / params.num_groups;
                            if (col_id > col_limit_right)
                                s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
#ifdef ZZDebug
                            if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
                                printf("s_acc [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(s_acc[i_j_idx]));
#endif
                        }
                    });
                });
            }

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
#ifdef ZZDebug
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto row_id = tile_idx.at(ck_tile::number<0>{});
                    auto col_id = tile_idx.at(ck_tile::number<1>{});
                    if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
                        printf("p_compute [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(p_compute[i_j_idx]));
#endif
                });
#ifdef ZZDebug
                if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
                    printf("\n");
#endif
            });

            auto rowsum_p = block_tile_reduce<acc_t>(
                p_compute, ck_tile::sequence<1>{}, f_sum, acc_t{0});
            ck_tile::block_tile_reduce_sync(rowsum_p, f_sum, ck_tile::bool_constant<false>{});

            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = ck_tile::make_tuple(idx0);
                const auto tmp = exp2(params.scale_softmax_log2 * m_old[i_idx] - params.scale_softmax_log2 * m[i_idx]);
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];

#ifdef ZZDebug
                const auto tile_idx = get_x_indices_from_distributed_indices(
                    l.get_tile_distribution(), make_tuple(idx0));
                auto row_id = tile_idx.at(ck_tile::number<0>{});
                if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
                {
                    printf("l [%d]: %f", row_id, ck_tile::type_convert<float>(l[i_idx]));
                    printf("\n");
                    printf("m [%d]: %f", row_id, ck_tile::type_convert<float>(m[i_idx]));
                    printf("\n");
                    printf("m_local [%d]: %f", row_id, ck_tile::type_convert<float>(m_local[i_idx]));
                    printf("\n");
                }
#endif
                sweep_tile_span(o_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
#ifdef ZZDebug

					// const auto tile_idx = get_x_indices_from_distributed_indices(o_acc.get_tile_distribution(), make_tuple(i_j_idx));
					// auto row_id = tile_idx.at(ck_tile::number<0>{});
					// auto col_id = tile_idx.at(ck_tile::number<1>{});
					//
					// if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n >= n_block_max - 2)
     //                    printf("o_acc [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(o_acc[i_j_idx]));
#endif
                    o_acc(i_j_idx) *= tmp;
                });
            });


// #ifdef ZZDebug
//             if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
//             {
//                 auto v_tile = load_tile(debug_v_ld_lds_window);
//                 const auto span_2d = decltype(v_tile)::get_distributed_spans();
//                 sweep_tile_span(span_2d[I0], [&](auto idx0) {
//                     sweep_tile_span(span_2d[I1], [&](auto idx1) {
//                         const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
//                         const auto tile_idx = get_x_indices_from_distributed_indices(
//                             v_tile.get_tile_distribution(), i_j_idx);
//
//                         auto row_id = tile_idx.at(I0);
//                         auto col_id = tile_idx.at(I1);
//
//                         printf("v_tile [%d, %d]: %f", row_id, col_id, ck_tile::type_convert<float>(v_tile[i_j_idx]));
//                     });
//                     printf("\n");
//                 });
//             }
// #endif

        // auto shuffled_k_block_tile = make_static_distributed_tensor<KDataType>(
        //     Policy::template MakeShuffledKRegWriteBlockDescriptor<Problem>());

            auto p = cast_tile<scalar_t>(p_compute);
			auto p_gemm_in = make_static_distributed_tensor<scalar_t>(p_shuffle_distribution, p.get_thread_buffer());

			auto v_tile = load_tile(v_ld_lds_window);
            gemm_1(o_acc,
                   p_gemm_in,
                   v_tile);
            ck_tile::block_sync_lds();

#ifdef ZZDebug
            if ((tidx == DEBUG_TID || tidx == 0) && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && i_block_n == n_block_max - 1)
            {
                __syncthreads();
                const auto span_2d = decltype(p_gemm_in)::get_distributed_spans();
                sweep_tile_span(span_2d[I0], [&](auto idx0) {
                    sweep_tile_span(span_2d[I1], [&](auto idx1) {
                        const auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                        const auto tile_idx_p = get_x_indices_from_distributed_indices(
                            p_gemm_in.get_tile_distribution(), i_j_idx);
                        const auto tile_idx_v = get_x_indices_from_distributed_indices(
                            v_tile.get_tile_distribution(), i_j_idx);
                        // const auto tile_idx_o = get_x_indices_from_distributed_indices(
                        //     o_acc.get_tile_distribution(), i_j_idx);

                        printf("p_gemm_in [%d, %d]: %f", tile_idx_p.at(I1), tile_idx_p.at(I0), ck_tile::type_convert<float>(p_gemm_in[i_j_idx]));
                        printf("v_tile [%d, %d]: %f", tile_idx_v.at(I1), tile_idx_v.at(I0), ck_tile::type_convert<float>(v_tile[i_j_idx]));
                        // printf("o_acc [%d, %d]: %f", tile_idx_o.at(I1), tile_idx_o.at(I0), ck_tile::type_convert<float>(o_acc[i_j_idx]));
						printf("\n");
                    });
                    printf("\n");
                });
            }
#endif

            move_tile_window(k_ld_lds_window, {k_ld_lds_offset, 0});
            move_tile_window(v_ld_lds_window, {k_ld_lds_offset, 0});
            k_ld_lds_offset = -k_ld_lds_offset;
        }

        // tail block
        // {
        //     ck_tile::block_sync_lds();
        //     gemm_0(s_acc,
        //            q,
        //            k_ld_lds_window);
        //
        //     const bool is_masking_step = masking_step > 0;
        //     const bool is_first_masking_step = masking_step == n_masking_steps;
        //
        //     //TODO: masking
        //     if (is_masking_step) {
        //         constexpr auto sacc_spans = decltype(s_acc)::get_distributed_spans();
        //         sweep_tile_span(sacc_spans[I0], [&](auto idx0) {
        //             // constexpr auto i_idx = ck_tile::make_tuple(idx0);
        //             sweep_tile_span(sacc_spans[I1], [&](auto idx1) {
        //                 constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
        //                 const auto tile_idx = get_x_indices_from_distributed_indices(
        //                     s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
        //                 auto row_id = tile_idx.at(ck_tile::number<0>{});
        //                 auto col_id = tile_idx.at(ck_tile::number<1>{});
        //                 if constexpr (!Is_causal)
        //                 {
        //                     if (col_id >= int(seqlen_k - i_block_n * kBlockN))
        //                         s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
        //                 }
        //                 else
        //                 {
        //                     int32_t col_limit_right = seqlen_k - 1 - i_block_n * kBlockN -
        //                         (params.size_s - 1 - (i_block_m * kBlockM + row_id)) / params.num_groups;
        //                     if (col_id >= col_limit_right)
        //                         s_acc(i_j_idx) = -ck_tile::numeric<acc_t>::infinity();
        //                 }
        //             });
        //         });
        //     }
        //
        //     const auto s = ck_tile::cast_tile<acc_t>(s_acc);
        //     auto m_local = block_tile_reduce<acc_t>(
        //         s_acc,
        //         ck_tile::sequence<1>{},
        //         f_max,
        //         -ck_tile::numeric<acc_t>::infinity());
        //     block_tile_reduce_sync(m_local, f_max, ck_tile::bool_constant<false>{});
        //
        //     const auto m_old = m;
        //     tile_elementwise_inout(
        //         [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local);
        //
        //     auto p_compute = make_static_distributed_tensor<acc_t>(
        //         s_acc.get_tile_distribution());
        //
        //     constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
        //     sweep_tile_span(p_spans[I0], [&](auto idx0) {
        //         constexpr auto i_idx = ck_tile::make_tuple(idx0);
        //         auto row_max = params.scale_softmax_log2 * m[i_idx];
        //         sweep_tile_span(p_spans[I1], [&](auto idx1) {
        //             constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
        //             p_compute(i_j_idx) = exp2(params.scale_softmax_log2 * s_acc[i_j_idx] - row_max);
        //         });
        //     });
        //
        //     auto rowsum_p = block_tile_reduce<acc_t>(
        //         p_compute, ck_tile::sequence<1>{}, f_sum, acc_t{0});
        //     ck_tile::block_tile_reduce_sync(rowsum_p, f_sum, ck_tile::bool_constant<false>{});
        //
        //     // l{j}, Oacc{j}
        //     constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
        //     sweep_tile_span(o_spans[I0], [&](auto idx0) {
        //         constexpr auto i_idx = ck_tile::make_tuple(idx0);
        //         const auto tmp = exp2(scale_s * m_old[i_idx] - params.scale_softmax_log2 * m[i_idx]);
        //         l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
        //         sweep_tile_span(o_spans[I1], [&](auto idx1) {
        //             constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
        //             o_acc(i_j_idx) *= tmp;
        //         });
        //     });
        //
        //     auto p = cast_tile<scalar_t>(p_compute);
        //     gemm_1(o_acc,
        //            ck_tile::get_slice_tile(
        //                p, ck_tile::sequence<0, 0>{}, ck_tile::sequence<kBlockM, kBlockN>{}),
        //            v_lds_window);
        //     ck_tile::block_sync_lds();
        // }

        // Epilogue
        auto lse_acc = make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
        constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
        sweep_tile_span(lse_acc_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = ck_tile::make_tuple(idx0);
            lse_acc(i_idx) = m[i_idx] * params.scale_softmax + log(l[i_idx]);
        });

        if (NoSplit)
        {
            store_tile(lse_dram_window, lse_acc);
        }
        else
        {
            auto lse_acc_dram_window = make_lse_acc_dram_window(i_batch, i_split);
            store_tile(lse_acc_dram_window, lse_acc);
        }

        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = ck_tile::make_tuple(idx0);
            const auto tmp = [&]() {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = ck_tile::make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });
        if (NoSplit)
        {
            store_tile(o_dram_window, ck_tile::cast_tile<scalar_t>(o_acc));
        }
        else
        {
            auto o_acc_dram_window = make_o_acc_dram_window(i_batch, i_split);
            store_tile(o_acc_dram_window, o_acc);
        }
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

    constexpr int64_t smem_size = Traits::kLdsOffsetSum + Traits::kNumThreadsSoftmax * 4;
    auto kernel = &flash_fwd_splitkv_mla_kernel<Traits, scalar_t, acc_t, Is_causal>;
    kernel<<<grid, Traits::kNumThreads, smem_size, stream>>>(params);
}

#define DISPATCH_FMLA_TYPES(TYPE, IS_CAUSAL, NAME, ...) \
    switch ((TYPE))                                     \
    {                                                   \
        case at::ScalarType::BFloat16:                           \
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
        case at::ScalarType::Half:                               \
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

std::vector<torch::Tensor> flash_mla_fwd_decode_with_kvcache_impl(
    torch::Tensor& query,
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

    dispatch_fmla_fwd_splictkv<Traits, ck_tile::fp16_t, float, true>(params);
    // DISPATCH_FMLA_TYPES(
    //     query.scalar_type(),
    //     is_causal,
    //     "fmla_fwd",
    //     [&](){
    //         dispatch_fmla_fwd_splictkv<Traits, scalar_t, acc_t, Is_causal>(params);
    //     }();
    // );

    return {output, softmax_lse};
}

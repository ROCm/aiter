// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/gemm.hpp>
#include "fmla_a16w16_qh16_m16x4_n16x1_coex0_mask1.hpp"

// =====================================================================================================================
// Utils
//
#define enable_inline
#define FMLA_FWD_FAST_EXP2 1
#define DEBUG_ONE_KERNEL 0

CK_TILE_DEVICE bool IsDebugThreadBlock(const int x = 0, const int y = 0, const int z = 0)
{
    return blockIdx.x == x && blockIdx.y == y && blockIdx.z == z;
}

// Returns count of warps which don't contain any idle thread.
template <int32_t NumWarps, int32_t M, int32_t N>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
{
    static_assert((NumWarps > 0) && ((NumWarps & (NumWarps - 1)) == 0)); // NumWarps should be power of 2
    constexpr int32_t ElemPerThread = (M * N) / (NumWarps * ck_tile::get_warp_size());
    if constexpr(0 < ElemPerThread)
    {
        return NumWarps;
    }
    else
    {
        return GetMaxNumWarpsForTile<NumWarps / 2, M, N>();
    }
}

// Returns vector size for given warp count for handing the specified matrix.
template <int32_t NumWarps, int32_t M, int32_t N, typename DataType>
CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
{
    constexpr int32_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N>();
    constexpr int32_t ElemPerThread = (M * N) / (MaxNumWarps * ck_tile::get_warp_size());
    constexpr int32_t MaxNPerThread = 16 / sizeof(DataType);
    return ck_tile::min(MaxNPerThread, ElemPerThread);
}

// =====================================================================================================================
// Definitions and helper structures
//

/// TODO: combine it with decode trait.
template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN0_,
          int32_t kBlockN1_,
          int32_t kNumWarps_>
struct FlashMlaPrefillKernelTrait
{
    static constexpr int32_t kSizeD                     = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                    = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kSizeNope                  = kSizeDV;
    static constexpr int32_t kSizeRope                  = kSizeD - kSizeNope;
    static constexpr int32_t kNumWarps                  = kNumWarps_;
    static constexpr int32_t kWaveOccupancy             = 1;
    static constexpr int32_t kNumWarpsSoftmax           = 4;
    static constexpr int32_t kNumThreadsSoftmax         = kNumWarpsSoftmax * ck_tile::get_warp_size();
    static constexpr int32_t kNumWarpsCombine           = 4;
    static constexpr int32_t kNumThreadsCombine         = kNumWarpsCombine * ck_tile::get_warp_size();
    static constexpr int32_t kBlockM                    = kBlockM_;
    static constexpr int32_t kBlockN0                   = kBlockN0_;
    static constexpr int32_t kBlockK0                   = 32;
    static constexpr int32_t kBlockN1                   = kBlockN1_;
    static constexpr int32_t kBlockK1                   = ck_tile::min(16, kBlockN0);
    static constexpr int32_t kNumThreads                = kNumWarps * ck_tile::get_warp_size();
};

template<typename Traits_, typename scalar_t, typename acc_t>
struct FlashMlaPrefillPolicy
{
public:
    using Traits = Traits_;
    using InOutType = scalar_t;
    using AccType   = acc_t;

#ifdef enable_inline
    CK_TILE_HOST_DEVICE static constexpr auto MakeKInlineLdsBlockDescriptor()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr ck_tile::index_t kNPerBlock = Traits::kBlockN0;
        constexpr ck_tile::index_t kKPerBlock = 64;
        constexpr ck_tile::index_t kRepeatK = 9;
        constexpr ck_tile::index_t kBlockSize = Traits::kNumThreads;
        constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
        constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

        constexpr ck_tile::index_t KVector = 4  / sizeof(scalar_t);
        constexpr ck_tile::index_t kPad = kKPerBlock;

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr ck_tile::index_t LanesPerK = kKPerBlock / KVector;   // 64 / 2 = 32
        constexpr ck_tile::index_t LaneGroups = warpSize / LanesPerK;  // 64 / 32 = 2
        constexpr ck_tile::index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

		constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
			ck_tile::make_tuple(ck_tile::number<NumIssues>{},  // n0
					   ck_tile::number<NumWarps>{},   // n2
					   ck_tile::number<LaneGroups>{}, // n1
					   ck_tile::number<kRepeatK>{},  // krepeatK
					   ck_tile::number<LanesPerK>{},  // k0
					   ck_tile::number<KVector>{}),   // k1
			ck_tile::make_tuple(ck_tile::number<(2 * kKPerBlock) * kRepeatK * LaneGroups * NumWarps>{},
					   ck_tile::number<(2 * kKPerBlock) * kRepeatK * LaneGroups>{},
					   ck_tile::number<(2 * kKPerBlock) * kRepeatK>{},
					   ck_tile::number<2 * kKPerBlock>{},
					   ck_tile::number<KVector>{},
					   ck_tile::number<1>{}),
			ck_tile::number<KVector>{},
			ck_tile::number<1>{});

		// TODO this layout is hard coded, and will be used in async copy buffer view load
		// in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
		constexpr auto k_lds_block_desc_issues_warps_lanes = ck_tile::transform_tensor_descriptor(
			k_lds_block_desc_0,
			ck_tile::make_tuple(make_pass_through_transform(ck_tile::number<NumIssues>{}),
					   ck_tile::make_pass_through_transform(ck_tile::number<NumWarps>{}),
					   ck_tile::make_merge_transform(ck_tile::make_tuple(
						   ck_tile::number<LaneGroups>{},
						   ck_tile::number<kRepeatK>{},
						   ck_tile::number<LanesPerK>{}, 
						   ck_tile::number<KVector>{}))),
			ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2, 3, 4, 5>{}),
			ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2>{}));

		return k_lds_block_desc_issues_warps_lanes;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKInlineLoadLdsBlockDescriptor()
    {
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kSizeD;
        constexpr int32_t kKPack     = 4 / sizeof(scalar_t);
        constexpr int32_t kKPackPerRepeat = 32;
        constexpr int32_t kKPacks = kKPerBlock / (kKPack * kKPackPerRepeat);

        constexpr auto k_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kNPerBlock>{},
                                ck_tile::number<kKPacks>{},
                                ck_tile::number<kKPackPerRepeat>{},
                                ck_tile::number<kKPack>{}),
            // ck_tile::make_tuple(ck_tile::number<kKPerBlock * 2>{},
            //                     ck_tile::number<kKPackPerRepeat * kKPack * 2>{},
            //                     ck_tile::number<kKPack>{},
            //                     ck_tile::number<1>{}),
            ck_tile::make_tuple(ck_tile::number<kKPerBlock * 1>{},
                                ck_tile::number<kKPackPerRepeat * kKPack * 1>{},
                                ck_tile::number<kKPack>{},
                                ck_tile::number<1>{}),
            ck_tile::number<kKPack>{},
            ck_tile::number<1>{});

        constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
            k_lds_block_desc_0,
            ck_tile::make_tuple(
                ck_tile::make_pass_through_transform(ck_tile::number<kNPerBlock>{}),
                ck_tile::make_merge_transform(make_tuple(ck_tile::number<kKPacks>{}, 
                                                         ck_tile::number<kKPackPerRepeat>{},
                                                         ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2, 3>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return k_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKAllInlineDramTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<2, 1, 8, 1>,
                               ck_tile::sequence<9, 4, 8, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKInlineDramTileDistribution()
    {
        // return ck_tile::make_static_tile_distribution(
        //     ck_tile::tile_distribution_encoding<
        //         ck_tile::sequence<>,
        //         ck_tile::tuple<ck_tile::sequence<2, 1, 8, 1>,
        //                        ck_tile::sequence<9, 4, 8, 2>>,
        //         ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
        //         ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
        //         ck_tile::sequence<1, 1, 2, 2>,
        //         ck_tile::sequence<0, 3, 0, 3>>{});
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr ck_tile::index_t kKPerBlock = 64;
        constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
        constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

        constexpr ck_tile::index_t KVector = 4 / sizeof(scalar_t);

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr ck_tile::index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr ck_tile::index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr ck_tile::index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr ck_tile::index_t N0 = NumIssues;
        constexpr ck_tile::index_t N1 = LaneGroups;
        constexpr ck_tile::index_t N2 = NumWarps;
        constexpr ck_tile::index_t K0 = LanesPerK;
        constexpr ck_tile::index_t K1 = KVector;

        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                       ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K0, K1>>,
                                       ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                                       ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
                                       ck_tile::sequence<1, 2>,
                                       ck_tile::sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeOInlineCTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                    // ck_tile::sequence<1, 1, 8, 2>,
                    // ck_tile::sequence<8, 4, 8, 2>,
                ck_tile::tuple<ck_tile::sequence<1, 4, 1,  4>,
                               ck_tile::sequence<1, 1, 64, 4>>, // 4 * 16 * 2
                ck_tile::tuple<ck_tile::sequence<2, 1>, ck_tile::sequence<2, 1>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<1, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeOInlineCShuffleTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<ck_tile::sequence<1, 4, 1, 4>,
                               ck_tile::sequence<1, 1, 64, 4>>, // 4 * 16 * 2
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<1, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeOLdsStoreBlockDescriptor()
    {
        constexpr int32_t kNPerBlock = 32;
        constexpr int32_t kKPerBlock = 256;
        constexpr int32_t kKPack     = 32 / sizeof(acc_t);

        constexpr auto k_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kNPerBlock>{},
                                ck_tile::number<kKPerBlock / kKPack>{},
                                ck_tile::number<kKPack>{}),
            ck_tile::make_tuple(ck_tile::number<kKPerBlock + kKPack>{},
                                ck_tile::number<kKPack>{},
                                ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});

        constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
            k_lds_block_desc_0,
            ck_tile::make_tuple(
                ck_tile::make_pass_through_transform(ck_tile::number<kNPerBlock>{}),
                ck_tile::make_merge_transform(make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1, 2>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return k_lds_block_desc;
    }


    CK_TILE_HOST_DEVICE static constexpr auto MakeOLdsLoadBlockDescriptor()
    {
        constexpr int32_t kNPerBlock = 32;
        constexpr int32_t kKPerBlock = 64;
        constexpr int32_t kNRpeat = 4;
        constexpr int32_t kKPack     = 32 / sizeof(scalar_t);

        constexpr auto k_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kNPerBlock>{},
                                ck_tile::number<kNRpeat>{},
                                ck_tile::number<kKPerBlock>{}),
            ck_tile::make_tuple(ck_tile::number<(kNRpeat * kNPerBlock) + kKPack>{},
                                ck_tile::number<kNPerBlock>{},
                                ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});

        constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
            k_lds_block_desc_0,
            ck_tile::make_tuple(
                ck_tile::make_merge_transform(make_tuple(ck_tile::number<kNPerBlock>{},
                                                         ck_tile::number<kNRpeat>{})),
                ck_tile::make_pass_through_transform(ck_tile::number<kKPerBlock>{})),
            ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return k_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeOInlineCopyOutSplitDramTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                    // ck_tile::sequence<1, 1, 8, 2>,
                    // ck_tile::sequence<8, 4, 8, 2>,
                ck_tile::tuple<ck_tile::sequence<4, 4, 4, 1>,
                               ck_tile::sequence<1, 1, 16, 2>>, // 4 * 16 * 2
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<1, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeOInlineCopyOutDramTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                    // ck_tile::sequence<1, 1, 8, 2>,
                    // ck_tile::sequence<8, 4, 8, 2>,
                ck_tile::tuple<ck_tile::sequence<4, 4, 4, 1>,
                               ck_tile::sequence<1, 1, 16, 4>>, // 4 * 16 * 2
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<1, 3, 0, 3>>{});
    }

#endif
};

template <typename Traits, typename scalar_t, typename acc_t>
struct FlashMlaCombineKernelPolicy
{
private:
    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
    {
        constexpr int32_t kVectorN     = GetVectorSizeForTile<Traits::kNumWarpsCombine, 1, Traits::kSizeDV, DataType>();
        constexpr int32_t kThrPerWarpN = ck_tile::get_warp_size();
        constexpr int32_t kNumWarpN    = Traits::kNumWarpsCombine;

        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,    // no replicate
                ck_tile::tuple<ck_tile::sequence<1>,
                               ck_tile::sequence<kNumWarpN, kThrPerWarpN, kVectorN>>,
                ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2>>,
                ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 2>>{});
    }

public:


    template <typename data_type_t>
    CK_TILE_DEVICE static auto MakeOaccuTileWindow(
        void* p_output_accum,
        const int32_t size_hs,
        const int32_t split_offset,
        const int32_t num_splits)
    {
        const int32_t offset_oaccum = (split_offset * 4 / sizeof(data_type_t) + size_hs) * Traits::kSizeDV;

        // Shape of tensor for a block: [num_splits, Traits::kSizeDV]
        const auto naive_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                reinterpret_cast<data_type_t*>(p_output_accum) + offset_oaccum,
                ck_tile::make_tuple(num_splits * 4 / sizeof(data_type_t), Traits::kSizeDV), // lengths
                ck_tile::make_tuple(Traits::kSizeDV * 16, 1),                    // strides
                ck_tile::number<Traits::kSizeDV>{},                         // last dim alignment
                ck_tile::number<1>{});                                      // last dim stride

        // Each thread group handles tile whose shape is [1, Traits::kSizeDV]
        const auto tile_window = ck_tile::make_tile_window(
            naive_view,
            ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                                ck_tile::number<Traits::kSizeDV>{}),
            {0, 0});                          // origin

        return ck_tile::make_tile_window(tile_window, MakeOutputTileDistribution<data_type_t>());
    }

    CK_TILE_DEVICE static auto MakeOutputTileWindow(
        scalar_t* p_output)
    {
        const auto naive_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_output,
                ck_tile::make_tuple(1, Traits::kSizeDV),    // lengths
                ck_tile::make_tuple(Traits::kSizeDV, 1),    // strides
                ck_tile::number<Traits::kSizeDV>{},         // last dim alignment
                ck_tile::number<1>{});                      // last dim stride

        const auto tile_window = ck_tile::make_tile_window(
            naive_view,
            ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                                ck_tile::number<Traits::kSizeDV>{}),
            {0, 0});                                                // origin

        return ck_tile::make_tile_window(tile_window, MakeOutputTileDistribution<scalar_t>());
    }
};
// =====================================================================================================================
// Kernel Entry
//

//bool kIsCausal = True, bool kDoSplit
template <typename Traits, typename scalar_t, typename acc_t, typename out_t, bool IsRopeSep>
__launch_bounds__(Traits::kNumThreads, Traits::kWaveOccupancy)
__global__ void kn_fmla_fwd_splictkv_prefill_inline(
    const ck_tile::FlashMlaInlineFwdParams params)
{
    __shared__ uint8_t smem[65535];
    auto o_lds_ptr   = reinterpret_cast<acc_t*>(smem);

    const int32_t mid = __builtin_amdgcn_readfirstlane(blockIdx.x * Traits::kBlockM);

    ck_tile::Fmla_gfx9_a16w16_qh16_m16x4_n16x1_coex0_mask1_total<Traits, scalar_t, acc_t, IsRopeSep> fmla_inline{};

    fmla_inline(params, smem);
}

template <typename Traits, int32_t kMaxSplits, typename out_t, typename in_t>
__global__ void kn_fmla_fwd_splictkv_prefill_combine(
    const ck_tile::FlashMlaInlineFwdParams params)
{
#ifdef DO_COMBINE_IN_HIP
    using Policy  = FlashMlaCombineKernelPolicy<Traits, out_t, in_t>;
    int32_t cur_batch = blockIdx.x;
    int32_t cur_head  = blockIdx.y;

    int32_t cur_qo_start = params.p_qo_indptr[cur_batch];
    int32_t cur_qo_end   = params.p_qo_indptr[cur_batch + 1];

    int32_t cur_split_start   = params.p_num_kv_splits_indptr[cur_batch];
    int32_t cur_split_end     = params.p_num_kv_splits_indptr[cur_batch + 1];
    int32_t num_max_kv_splits = params.p_num_kv_splits_indptr[params.size_b];
    int32_t cur_kv_seq_len    = params.p_seqlens_k[cur_batch + 1] - params.p_seqlens_k[cur_batch];

    int32_t offs_oacc = cur_head * params.stride_h_lseacc;

    int32_t num_valid_kv_splits = ck_tile::min(
        cur_split_end - cur_split_start, ck_tile::integer_divide_ceil(cur_kv_seq_len, 16)
    );

    bool FINAL_OUT = true && num_max_kv_splits == params.size_b; 

    auto oaccu_window_bf16 =
        Policy::template MakeOaccuTileWindow<ck_tile::bf16_t>(params.p_output,
                                                              cur_head,
                                                              cur_qo_start * params.num_splits * params.size_h,
                                                              params.num_splits * (cur_qo_end - cur_qo_start));

    auto oaccu_window =
        Policy::template MakeOaccuTileWindow<float>(params.p_output,
                                                    cur_head,
                                                    cur_qo_start * params.num_splits * params.size_h,
                                                    params.num_splits * (cur_qo_end - cur_qo_start));

    auto reg_out = ck_tile::make_static_distributed_tensor<in_t>(
        decltype(ck_tile::load_tile(oaccu_window))::get_tile_distribution());
    ck_tile::set_tile(reg_out, 0.f);

    float e_sum = 0.0;
    float e_max = -ck_tile::numeric<in_t>::infinity();
    for (int32_t cur_qo = cur_qo_start; cur_qo < cur_qo_end; ++cur_qo)
    {
		auto dram_out = Policy::MakeOutputTileWindow(
			static_cast<out_t*>(params.p_output_com) +
			cur_head * params.stride_h_o + cur_qo * params.stride_s_o);
        if (num_valid_kv_splits == 1) 
        {
            auto oaccu = ck_tile::load_tile(oaccu_window_bf16);
            ck_tile::store_tile(dram_out, oaccu);
            ck_tile::move_tile_window(oaccu_window_bf16, {2 * params.num_splits, 0});
        }
        else
        {
            for (int32_t split_idx = 0; split_idx < num_valid_kv_splits; ++split_idx)
            {
                float tlogic = reinterpret_cast<float*>(params.p_softmax_lse)[
                    cur_qo * params.stride_b_lseacc +
                    offs_oacc +
                    split_idx * params.stride_sp_lseacc
                ];
                float n_e_max = ck_tile::max(tlogic, e_max);
                auto oaccu = ck_tile::load_tile(oaccu_window);

                float old_scale = ck_tile::exp(e_max - n_e_max);
                float exp_logic = ck_tile::exp(tlogic - n_e_max);
                ck_tile::sweep_tile(oaccu, [&](auto idx) {
                    reg_out(idx) *= old_scale;
                    reg_out(idx) += exp_logic * oaccu(idx);
                });

                e_sum = e_sum * old_scale + exp_logic;
                e_max = n_e_max;
                ck_tile::move_tile_window(oaccu_window, {1, 0});
            }
            ck_tile::sweep_tile(reg_out, [&](auto idx) {
                reg_out(idx) /= e_sum;
            });
            ck_tile::store_tile(dram_out, ck_tile::cast_tile<out_t>(reg_out));
            ck_tile::set_tile(reg_out, 0.f);
            e_sum = 0.f;
            ck_tile::move_tile_window(oaccu_window, {params.num_splits - num_valid_kv_splits, 0});
        }
    }
#endif
}

// =====================================================================================================================
// Dispatch
//

template <typename Traits, typename scalar_t, typename acc_t, typename out_t, bool IsRopeSep>
void dispatch_fmla_fwd_splictkv_prefill_inline(
    const ck_tile::FlashMlaInlineFwdParams& params)
{
    int sub_Q = 64;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int32_t num_blk   = ck_tile::integer_divide_ceil(params.size_s, sub_Q);
    // const dim3    grid_attn = dim3(num_blk, params.size_b, params.num_splits);
    const dim3    grid_attn = dim3(num_blk, 1, params.cu_nums);
    const dim3    grid_comb = dim3(params.size_b, params.size_h, 1);
    // const dim3    grid_comb = dim3(1, 1, 1);
    // const dim3    grid_comb = dim3(params.size_b, 1, 1);

    auto kn_attn = &kn_fmla_fwd_splictkv_prefill_inline<Traits, scalar_t, acc_t, out_t, IsRopeSep>;
    kn_attn<<<grid_attn, Traits::kNumThreads, 0, stream>>>(params);
    // auto kn_comb = &kn_fmla_fwd_splictkv_prefill_combine<Traits, 32,  scalar_t, acc_t>;
    // kn_comb<<<grid_comb, Traits::kNumThreadsCombine, 0, stream>>>(params);
}

std::vector<torch::Tensor> flash_mla_fwd_inline_impl(
    torch::Tensor&       query,
    const torch::Tensor& key_cache,
    const torch::Tensor& qo_indptr,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_last_page_lens,
    const torch::Tensor& num_kv_splits_indptr,

    const int            max_seqlen_q,
    const float          softmax_scale,

    torch::Tensor&       split_data,
    torch::Tensor&       split_lse,
    std::optional<torch::Tensor>&       query_rope,
    const std::optional<torch::Tensor>& key_rope_cache,
    const std::optional<torch::Tensor>& batch_split_table,
    const std::optional<torch::Tensor>& split_table,
    std::optional<torch::Tensor>&       out,
    const int num_splits_ = 1)
{
    //                                        dqk  dv   m0  n0  n1  #warp
    using Traits = FlashMlaPrefillKernelTrait<576, 512, 64, 16, 512, 4>;

    using acc_t = float;
    constexpr int32_t head_size_v = 512;

    auto opts = query.options();
    static_assert(std::is_same_v<acc_t, float>);
    auto opts_acc = opts.dtype(torch::kFloat32);

    const int32_t seqlen_q_ori = max_seqlen_q;
    const int32_t pack_batch_seq_q = query.size(0);
    const int32_t batch_size = pack_batch_seq_q / seqlen_q_ori;
    const int32_t num_heads_q_ori = query.size(-2);
    int32_t seqlen_q = seqlen_q_ori;
    int32_t num_heads_q = num_heads_q_ori;

    const int32_t head_size = query.size(-1);

    const int32_t num_blocks = key_cache.size(0);
    const int32_t page_block_size = key_cache.size(1);
    const int32_t num_heads_k = key_cache.size(2);

    TORCH_CHECK(num_heads_q % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    const int32_t hq_hk_ratio = num_heads_q / num_heads_k;

    const int32_t num_splits = split_data.size(1);

    auto output = !out.has_value() ? 
            torch::empty({pack_batch_seq_q, hq_hk_ratio, head_size_v}, opts) :
            out.value().data_ptr() ? out.value() : 
                torch::empty({pack_batch_seq_q, hq_hk_ratio, head_size_v}, opts);

    ck_tile::FlashMlaInlineFwdParams params = {};

    params.cu_nums                = num_splits_;
    params.num_splits             = num_splits;
    params.p_qo_indptr            = qo_indptr.data_ptr<int32_t>();
    params.p_seqlens_k            = cache_seqlens.data_ptr<int32_t>();
    params.p_block_table          = block_table.data_ptr<int32_t>();
    params.p_num_kv_splits_indptr = num_kv_splits_indptr.data_ptr<int32_t>();
    params.p_batch_split_table    = batch_split_table.value().data_ptr<int32_t>();
    params.p_split_table          = split_table.value().data_ptr<int32_t>();

    params.p_query       = query.data_ptr();
    params.p_key         = key_cache.data_ptr();
    params.p_output_com  = output.data_ptr();
    params.p_output      = split_data.data_ptr();
    params.p_softmax_lse = split_lse.data_ptr();
    params.p_query_rope  = query_rope.has_value() ? query_rope.value().data_ptr() : nullptr;
    params.p_key_rope    = key_rope_cache.has_value()? key_rope_cache.value().data_ptr() : nullptr;

    params.max_seqlen_q     = max_seqlen_q;
    params.stride_q_b       = query_rope.has_value() ? 
                                  Traits::kSizeD * num_heads_q * query.itemsize() * max_seqlen_q :
                                  query.stride(0) * query.itemsize() * max_seqlen_q;
    params.stride_page      = key_rope_cache.has_value() ?
                                  Traits::kSizeNope * key_cache.itemsize() :
                                  Traits::kSizeD * key_cache.itemsize();
    params.stride_page_rope = key_rope_cache.has_value() ?
                                  Traits::kSizeRope * key_cache.itemsize() :
                                  Traits::kSizeD * key_cache.itemsize();
    params.size_b           = pack_batch_seq_q / params.max_seqlen_q;
    params.size_s           = seqlen_q;
    params.size_h           = num_heads_q;
    params.hq_hk_ratio      = num_heads_q / num_heads_k;
    params.num_page_blocks  = num_blocks;
    params.page_block_size  = page_block_size;
    params.scale_softmax    = softmax_scale;

    params.stride_s_o = output.stride(0);
    params.stride_h_o = output.stride(1);

#ifdef DO_COMBINE_IN_HIP
    if(num_splits > 1)
    {
        params.stride_b_oacc      = split_data.stride(0);
        params.stride_h_oacc      = split_data.stride(2);
        params.stride_sp_oacc     = split_data.stride(1);
        params.stride_b_lseacc    = split_lse.stride(0);
        params.stride_h_lseacc    = split_lse.stride(2);
        params.stride_sp_lseacc   = split_lse.stride(1);
    }
#endif

    if (query_rope.has_value())
    {
        dispatch_fmla_fwd_splictkv_prefill_inline<Traits, ck_tile::bf16_t, float, ck_tile::bf16_t, true>(params);
    }
    else
    {
        dispatch_fmla_fwd_splictkv_prefill_inline<Traits, ck_tile::bf16_t, float, ck_tile::bf16_t, false>(params);
    }
    return {output};
}

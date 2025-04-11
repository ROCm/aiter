// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>

// =====================================================================================================================
// Definitions and helper structures
//

// TODO: combine it with decode trait.
template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN0_,
          int32_t kBlockN1_,
          int32_t kNumWarps_,
          typename Mask_>
struct FlashMlaPrefillKernelTrait
{
    using Mask = Mask_;

    static constexpr int32_t kSizeD                     = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                    = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kNumWarps                  = kNumWarps_;
    static constexpr int32_t kNumThreads                = kNumWarps * warpSize;
    static constexpr int32_t kNumWarpsSoftmax           = 4;
    static constexpr int32_t kNumThreadsSoftmax         = kNumWarpsSoftmax * warpSize;
    static constexpr int32_t kBlockM                    = kBlockM_;
    static constexpr int32_t kBlockN0                   = kBlockN0_;
    static constexpr int32_t kBlockN1                   = kBlockN1_;
    static constexpr int32_t kBlockK0                   = 32;
    static constexpr int32_t kBlockK1                   = 32;
    static constexpr int32_t kFixedOverheadNumBlocks    = 5;
    static constexpr int32_t kMaxBatchSize              = 4096;
    static constexpr int32_t kCuReuse                   = 2;
    static constexpr int32_t kMaxSplits                 = 128;
    static constexpr bool    kPadHeadDimQ               = true;
    static constexpr bool    kPadHeadDimV               = true;
    static constexpr bool    kPadSeqLenQ                = true;
    static constexpr bool    kPadSeqLenK                = true;

    // For QS+QR mixed implementation, VGPR always store 256 elements in row/along K0.
    // So the rest are stored in SMEM.
    static constexpr int32_t kK0InReg  = 256;
    static constexpr int32_t kK0InSmem = kSizeD - kK0InReg;
    static constexpr int32_t kNumPrefetchKV = 1;

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);
};

template<typename Traits_, bool kIsSameKV_, typename scalar_t, typename acc_t>
struct FlashMlaPrefillPolicy
{
public:
    using Traits = Traits_;

    static constexpr bool kIsSameKV = kIsSameKV_;

private:
    template <index_t NumWarps, index_t M, index_t N, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
    {
        static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);

        constexpr index_t ElemPerThread = (M * N) / (NumWarps * ck_tile::get_warp_size());
        if constexpr(0 < ElemPerThread)
        {
            return NumWarps;
        }
        else
        { // try dividing tile by smaller # of warps
            return GetMaxNumWarpsForTile<NumWarps / 2, M, N, DataType>();
        }
    }

    template <index_t NumWarps, index_t M, index_t N, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
    {
        constexpr index_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N, DataType>();

        constexpr index_t ElemPerThread = (M * N) / (MaxNumWarps * ck_tile::get_warp_size());

        constexpr index_t MaxNPerThread = 16 / sizeof(DataType);
        return min(MaxNPerThread, ElemPerThread);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSizeQ()
    {
        constexpr int32_t lds_alignment = 16; // optional
        constexpr int32_t q_smem_size =
            ck_tile::integer_divide_ceil(
                sizeof(scalar_t) * MakeQLdsBlockDescriptor().get_element_space_size(),
                lds_alignment) *
            lds_alignment;
        return q_smem_size;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSizeSingleKV()
    {
        constexpr int32_t SingleKSize = MakeKLdsBlockDescriptor().get_element_space_size();
        constexpr int32_t SingleVSize =[&]() {
            // TODO: Check correctness
            constexpr index_t Banks        = 32; // TODO: need change based on arch
            constexpr index_t PixelsPerRow = Banks * 4 / sizeof(scalar_t);
            constexpr index_t kKPack       = 16 / sizeof(scalar_t);
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr index_t NPerRow    = PixelsPerRow / kKPack;
            constexpr index_t kNPerBlock = Traits::kBlockN1;
            constexpr index_t kKPerBlock = Traits::kBlockK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return ck_tile::max(SingleKSize, SingleVSize) * sizeof(scalar_t):
    }

public:
    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentQ()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kMPerBlock = Traits::kBlockM;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(scalar_t);

        // this should align with MakeQDramTileDistribution()
        constexpr int32_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return ck_tile::min(ElemPerThread, MaxVectorSize);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentK()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(scalar_t);
        constexpr int32_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        return ck_tile::min(MaxVectorSize, ElemPerThread);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentV()
    {
        // TODO: Assuming Value is row-major just like Key.
        constexpr int32_t kBlockSize   = Traits::kNumThreads;
        constexpr int32_t kNPerBlock   = Traits::kBlockN1;
        constexpr int32_t kKPerBlock   = Traits::kBlockK1;
        constexpr int32_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr int32_t kMaxVecLoad =
            ck_tile::min(total_pixels, static_cast<int32_t>(16 / sizeof(scalar_t)));
        constexpr int32_t kMinVecLoad = 4 / sizeof(scalar_t);

        constexpr int32_t kVecLoad =
            ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                ? kMaxVecLoad
                : (total_pixels / kMinVecLoad);

        return kVecLoad;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentOacc()
    {
        int32_t result = 1;

        if constexpr (Traits::kPadHeadDimV == false)
        {
            constexpr int32_t kBlockSize = Traits::kNumThreads;
            constexpr int32_t kMPerBlock = Traits::kBlockM0;
            constexpr int32_t kNPerBlock = Traits::kBlockN1;

            constexpr int32_t M1 = kBlockSize / ck_tile::get_warp_size();
            constexpr int32_t M2 = ck_tile::min(kMPerBlock / M1, ck_tile::get_warp_size());
            constexpr int32_t N0 = ck_tile::get_warp_size() / M2;
            constexpr int32_t N1 = kNPerBlock / N0;

            // Each thread cannot handle more than 16 bytes
            result = ck_tile::min(N1, static_cast<int32_t>(16 / sizeof(scalar_t)));
        }

        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentO()
    {
        int32_t result = 1;

        if constexpr (Traits::kPadHeadDimV == false)
        {
            constexpr int32_t kBlockSize = Traits::kNumThreads;
            constexpr int32_t kMPerBlock = Traits::kBlockM0;
            constexpr int32_t kNPerBlock = Traits::kBlockN1;

            constexpr int32_t M1 = kBlockSize / ck_tile::get_warp_size();
            constexpr int32_t M2 = ck_tile::min(kMPerBlock / M1, ck_tile::get_warp_size());
            constexpr int32_t N0 = ck_tile::get_warp_size() / M2;
            constexpr int32_t N1 = kNPerBlock / N0;

            // Each thread cannot handle more than 16 bytes
            result = ck_tile::min(N1, static_cast<int32_t>(16 / sizeof(acc_t)));
        }

        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentLSE()
    {
        return GetVectorSizeForTile<Traits::kNumWarps,
                                    Traits::kMaxSplits,
                                    Traits::kBlockM0,
                                    acc_t>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr int32_t kMPerBlock = Traits::kBlockM;
        // #elements store in SMEM along K0 for query.
        constexpr int32_t kKPerBlock = Traits::kK0InSmem;
        constexpr int32_t kKPack     = 16 / sizeof(scalar_t);

        constexpr auto q_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
            ck_tile::make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto q_lds_block_desc = transform_tensor_descriptor(
            q_lds_block_desc_0,
            ck_tile::make_tuple(ck_tile::make_pass_through_transform(kMPerBlock),
                                ck_tile::make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            ck_tile::make_tuple(sequence<1>{}, sequence<0, 2>{}),
            ck_tile::make_tuple(sequence<0>{}, sequence<1>{}));

        return q_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;
        constexpr int32_t kKPack     = 16 / sizeof(scalar_t);

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kNPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kNPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kNPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSize()
    {
        return GetSmemSizeQ() + Traits::kNumPrefetchKV * GetSmemSizeSingleKV();
    }
}

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

struct FlashMlaPrefillFwdParams
{
    int32_t* __restrict__ p_cu_seqlens_k;   // [b+1]
    int32_t* __restrict__ p_block_table;    // [b, max_seqlen_pad // block_size]
    
    void* __restrict__ p_query;
    void* __restrict__ p_key;
    void* __restrict__ p_value;
    void* __restrict__ p_output;
    void* __restrict__ p_softmax_lse;
    void* __restrict__ p_softmax_lseaccum;
    void* __restrict__ p_output_accum;

    int32_t size_b;         // batch count
    int32_t size_s;         // seqlen of q
    int32_t size_h;         // head count of q
    int32_t hq_hk_ratio;    // head count of q / head count of kv
    int64_t block_table_batch_stride;
    int32_t page_block_size;
    float   scale_softmax;
    float   scale_softmax_log2;

    // Use int64_t if there is int32 overflow case. For now, just use int32 to save sgpr and prevent using
    // spill table.
    using index_t = int32_t;

    index_t stride_b_q;         // stride in batch of query
    index_t stride_s_q;         //    ... in sequence ...
    index_t stride_h_q;         //    ... in head ...
    index_t stride_b_k;         // stride in batch of key
    index_t stride_s_k;         //    ... in sequence ...
    index_t stride_h_k;         //    ... in head ...
    index_t stride_b_v;         // stride in batch of value
    index_t stride_s_v;         //    ... in sequence ...
    index_t stride_h_v;         //    ... in head ...
    index_t stride_b_o;         // stride in batch of output
    index_t stride_s_o;         //    ... in sequence ...
    index_t stride_h_o;         //    ... in head ...
    index_t stride_b_lseacc;
    index_t stride_h_lseacc;
    index_t stride_sp_lseacc;   //    ... in split ...
    index_t stride_b_oacc;
    index_t stride_h_oacc;
    index_t stride_sp_oacc;     //    ... in split ...
    index_t stride_s_oacc;
};

// =====================================================================================================================
// Kernel Functions
//

template <typename Traits>
CK_TILE_DEVICE static auto GetTileIndex()
{
    constexpr int32_t num_tile_n1 = ck_tile::integer_divide_ceil(Traits::kSizeDV, Traits::kBlockN1);

    const auto f = [](index_t dividend, index_t divisor) {
        index_t quotient = dividend / divisor;
        index_t modulus  = dividend - quotient * divisor;
        return ck_tile::make_tuple(quotient, modulus);
    };

    const auto [mnid, split_id] = f(blockIdx.x, kargs.num_splits);
    const auto [mid, nid]       = f(mnid, num_tile_n1);
    const index_t hid           = blockIdx.y;
    const index_t bid           = blockIdx.z;

    return ck_tile::make_tuple(mid, nid, split_id, hid, bid);
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeQDram(
    const scalar_t* p_data,
    const int32_t   height,
    const int32_t   stride_s)
{
    using Traits = Policy::Traits;

    const auto q_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_query,
        ck_tile::make_tuple(height, Traits::kSizeD),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentQ()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        q_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockK0>{}),
        ck_tile::sequence<false, Traits::kPadHeadDimQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeKDram(
    const scalar_t* p_data,
    const int32_t   height,
    const int32_t   stride_s)
{
    using Traits = Policy::Traits;

    const auto k_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        data, // will update this pointer if using paged-kvcache
        ck_tile::make_tuple(height, Traits::kSizeD),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentK()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        k_dram_naive,
        ck_tile::make_tuple(Traits::number<Traits::::kBlockN0>{}, Traits::number<Traits::::kBlockK0>{}),
        ck_tile::sequence<false, Traits::kPadHeadDimQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeVDram(
    const scalar_t* p_data,
    const int32_t   length,
    const int32_t   stride_s)
{
    using Traits = Policy::Traits;

    // TODO: Assuming Value is row-major just like Key.
    const auto v_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        data, // will update this pointer if using paged-kvcache
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
CK_TILE_DEVICE static auto MakeLseAccDram(
    const scalar_t* p_data,
    const Lengths&  window_lengths,
    const int32_t   size_s)
{
    using Traits = Policy::Traits;

    const auto lse_acc_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s),
        ck_tile::make_tuple(1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        lse_acc_dram_naive,
        window_lengths,
        ck_tile::sequence<Traits::kPadSeqLenQ>);
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutAccDram(
    const scalar_t* p_data,
    const int32_t   size_s,
    const int32_t   stride_s)
{
    using Traits = Policy::Traits;

    const auto o_acc_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s, Traits::kSizeDV),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentOacc()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        o_acc_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <typename Policy, typename Lengths, typename scalar_t>
CK_TILE_DEVICE static auto MakeLseDram(
    const scalar_t* p_data,
    const Lengths&  window_lenghts,
    const int32_t   size_s)
{
    using Traits = Policy::Traits;

    const auto lse_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s),
        ck_tile::make_tuple(1),
        ck_tile::number<Policy::GetAlignmentLse()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        lse_dram_naive, window_lenghts, ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutDram(
    const scalar_t* p_data,
    const int32_t   size_s,
    const int32_t   stride_s)
{
    using Traits = Policy::Traits;

    const auto o_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        o_ptr,
        ck_tile::make_tuple(size_s, Traits::kSizeDV),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentO()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        o_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <int32_t VirtualDim, typename scalar_t, typename Dram>
CK_TILE_DEVICE static auto MakePageBlockNavigator(
    const scalar_t* p_data,
    const Dram&     dram_complete,
    const Dram&     dram_last,
    const int32_t   bid,
    const int32_t   hid,
    const int32_t   seqlen_k,
    const int32_t   stride_b,
    const int32_t   stride_h,
    const int32_t*  p_block_table,
    const int32_t   stride_b_block_table,
    const int32_t   page_block_size)
{
    const auto* p_block_indices = p_block_table + bid * stride_b_block_table;
    const int32_t num_blocks = ck_tile::integer_divide_ceil(seqlen_k, page_block_size);

    const int64_t fixed_offset = static_cast<int64_t>(hid) * stride_h;

    return ck_tile::make_page_block_navigator<const scalar_t, VirtualDim>(
        p_data,
        stride_b, // vcache page-block stride/size
        fixed_offset,
        p_block_indices,
        num_blocks,
        page_block_size,
        dram_complete,
        dram_last);
}

template<typename Traits,
         typename QDramBlockWindow,
         typename LseDramBlockWindow,
         typename KPageBlockNavigator,
         typename VPageBlockNavigator,
         typename Mask>
CK_TILE_DEVICE static auto kn_fmla_fwd_splitkv_prefill_tile(
    const QDramBlockWindow&    q_dram_req_window_,
    const QDramBlockWindow&    q_dram_smem_window_,
    const KPageBlockNavigator& k_page_block_navigator,
    const VPageBlockNavigator& v_page_block_navigator,
    LseDramBlockWindow&        lse_dram_window_,
    int32_t                    num_splits,
    int32_t                    split_id,
    Mask                       mask,
    float                      scale_s,
    float                      scale_log2_s,
    uint8_t*                   p_smem)
{
    using Policy = FlashMlaPrefillPolicy<Traits, kIsSameKV, scalar_t, acc_t>;

    auto k_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{});
    auto v_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN1>{}, ck_tile::number<Traits::kBlockK1>{});
}

// =====================================================================================================================
// Kernel Entry
//

template <typename Traits, typename scalar_t, typename acc_t, bool kIsCausal, bool kIsSameKV, bool kDoSplit>
__global__ kn_fmla_fwd_splictkv_prefill(
    const FlashMlaPrefillFwdParams params)
{
    using Policy = FlashMlaPrefillPolicy<Traits, kIsSameKV, scalar_t, acc_t>;

    // allocate LDS
    __shared__ uint8_t p_smem[Policy::GetSmemSize()];

    const auto [mid, nid, split_id, hqid, bid] = GetTileIndex();
    const auto hkid = hqid / params.hq_hk_ratio;
    const int32_t offset_m = __builtin_amdgcn_readfirstlane(mid * Traits::kBlockM);
    const int32_t offset_n = __builtin_amdgcn_readfirstlane(nid * Traits::kBlockN1);

    // TODO: replace p_cu_seqlens_k with p_seqlens_k whose shape is [b] if key_start_seq is not used.
    const int32_t key_start_seq       = params.p_cu_seqlens_k[bid];
    const int32_t seqlen_k            = params.p_cu_seqlens_k[bid+1] - key_start_seq;
    const int32_t num_block           = ck_tile::integer_divide_ceil(seqlen_k, params.page_block_size);
    const int32_t last_block_size     = seqlen_k - (num_blocks - 1) * params.page_block_size;
    const int64_t batch_offset_lseacc = bid * params.stride_b_o;

    auto q_dram_reg_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kK0InReg>{});
    auto q_dram_smem_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kK0InSmem>{});


    const scalar_t* p_query = reinterpret_cast<const scalar_t*>(params.p_query) +
                              int64_t(hqid * params.stride_h_q) +   // head offset
                              int64_t(bid * params.stride_b_q);     // batch offset

    const auto q_dram_complete = MakeQDram<Policy>(p_query, params.size_s,          params.stride_s_q);
    const auto k_dram_complete = MakeKDram<Policy>(nullptr, params.page_block_size, params.stride_s_k);
    const auto k_dram_last     = MakeKDram<Policy>(nullptr, last_block_size,        params.stride_s_k);
    const auto v_dram_complete = MakeVDram<Policy>(nullptr, params.page_block_size, params.stride_s_v);
    const auto v_dram_last     = MakeVDram<Policy>(nullptr, last_block_size,        params.stride_s_v);         

    auto q_dram_req_window =
        ck_tile::make_tile_window(q_dram_complete, q_dram_reg_window_lengths, {mid, 0});
    auto q_dram_smem_window =
        ck_tile::make_tile_window(q_dram_complete, q_dram_smem_window_lengths, {mid, Traits::kK0InReg});

    auto k_page_block_navigator = MakePageBlockNavigator<0>(
        params.p_key,   k_dram_complete, k_dram_last, bid, hkid, seqlen_k, params.stride_b_k, params.stride_h_k,
        params.p_block_table, params.block_table_batch_stride, params.page_block_size);
    auto v_page_block_navigator = MakePageBlockNavigator<1>(
        params.p_value, v_dram_complete, v_dram_last, bid, hkid, seqlen_k, params.stride_b_v, params.stride_h_v,
        params.p_block_table, params.block_table_batch_stride, params.page_block_size);
    
    using Mask = typename Traits::Mask;
    Mask mask = kIsCausal ?
                ck_tile::make_generic_attention_mask_from_lr_window<Mask>(-1, -1, params.size_s, seqlen_k, true) :
                Mask{params.size_s, seqlen_k};

    if constexpr (kDoSplit)
    {
        acc_t* p_lse_acc = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
                           int64_t(hqid) * params.stride_h_lseacc +     // head offset
                           int64_t(bid) * params.stride_b_lseacc +      // batch offset
                           int64_t(split_id) * params.stride_sp_lseacc; // split offset
        scalar_t* p_out_acc = reinterpret_cast<scalar_t*>(params.p_output_accum) +
                              int64_t(hqid) * params.stride_h_oacc +      // head offset
                              int64_t(bid) * params.stride_b_oacc +       // batch offset
                              int64_t(split_id) * params.stride_sp_oacc;  // split offset

        auto lse_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});

        const auto lse_acc_dram = MakeLseAccDram<Policy>(p_lse_acc, lse_acc_dram_window_lengths, params.size_s);
        const auto out_acc_dram = MakeOutAccDram<Policy>(p_out_acc, params.size_s, params.stride_s_oacc);

        auto lse_acc_dram_window =
            ck_tile::make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {mid});
        auto out_acc_dram_window =
            ck_tile::make_tile_window(out_acc_dram, out_acc_dram_window_lengths, {mid, nid});

        auto o_acc_tile = kn_fmla_fwd_splitkv_prefill_tile<Traits>(
            q_dram_req_window,
            q_dram_smem_window,
            k_page_block_navigator,
            v_page_block_navigator,
            lse_acc_dram_window,
            params.num_splits,
            split_id,
            mask,
            params.scale_softmax,
            params.scale_softmax_log2,
            p_smem);

        ck_tile::store_tile(out_acc_dram_window, ck_tile::cast_tile<scalar_t>(o_acc_tile));
    }
    else
    {
        // TODO: Assuming lse is in shape [b, h, s] and is contiguous
        acc_t* p_lse = reinterpret_cast<acc_t*>(params.p_softmax_lse) +
                       (int64_t(bid) * params.size_h + hqid) * params.size_s; // batch+head offset
        scalar_t* p_out = reinterpret_cast<scalar_t*>(params.p_output) +
                          int64_t(hqid) * params.stride_h_o +   // head offset
                          int64_t(bid) * params.stride_b_o;     // batch offset

        auto lse_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlcokN1>{});
        
        const auto lse_dram = MakeLseDram<Policy>();
        const auto out_dram = MakeOutDram<Policy>();

        auto lse_dram_window =
            ck_tile::make_tile_window(lse_dram, lse_dram_window_lengths, {mid});
        auto out_dram_window =
            ck_tile::make_tile_window(out_dram, out_dram_window_lengths, {mid, nid});

        auto o_acc_tile = kn_fmla_fwd_splitkv_prefill_tile<Traits>(
            q_dram_req_window,
            q_dram_smem_window,
            k_page_block_navigator,
            v_page_block_navigator,
            lse_dram_window,
            params.num_splits,
            split_id,
            mask,
            params.scale_softmax,
            params.scale_softmax_log2,
            p_smem);

        ck_tile::store_tile(out_dram_window, ck_tile::cast_tile<scalar_t>(o_acc_tile));
    }    
}

// =====================================================================================================================
// Dispatch
//

template <typename Traits, typename scalar_t, typename acc_t, bool kIsCausal, bool kIsSameKv>
void dispatch_fmla_fwd_splictkv_prefill(
    const FlashMlaPrefillFwdParams& params,
    const int32_t                   num_splits)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int32_t num_blk  = ck_tile::integer_divide_ceil(params.size_s,  Traits::kBlockM) *
                             ck_tile::integer_divide_ceil(params.size_dv, Traits::kBlockN1) *
                             num_splits;
    const dim3 grid = dim3(num_blk, params.size_h, params.size_b);

    auto kernel = (num_splits > 1) ? kn_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, kIsCausal, kIsSameKv, true> :
                                     kn_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, kIsCausal, kIsSameKv, false>;
    kernel<<<grid, Traits::kNumThreads, 0, stream>>>(params);
}

// =====================================================================================================================
// Interfaces
//

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

int num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int num_n_blocks)
{
    int32_t result = 1;

    if (batch_nhead_mblocks < 0.8f * num_SMs)
    {
        int32_t max_splits = std::min({max_splits, num_SMs, num_n_blocks});
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

    return num_splits_heuristic(size_b * size_h * num_m_blocks, cu_count * Traits::kCuReuse, num_n_blocks);
}

std::vector<torch::Tensor> flash_mla_fwd_prefill_with_kvcache_impl(
    torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const int32_t        head_size_v,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const float          softmax_scale,
    const bool           is_causal)
{
    using Traits = FlashMlaPrefillKernelTrait<576, 512, 32, 32, 32, 4>;

    torch::Tensor vcache = value_cache.data_ptr() ? value_cache : key_cache;

    auto opts = query.options();

    const int32_t batch_size = query.size(0);
    const int32_t seqlen_q_ori = query.size(1);
    const int32_t num_heads_q = query.size(2);

    const int32_t head_size = query.size(3);


    const int32_t num_blocks = key_cache.size(0);
    const int32_t page_block_size = key_cache.size(1);
    const int32_t num_heads_k = key_cache.size(2);

    const int32_t num_groups = num_heads_q / num_heads_k;
    const int32_t seqlen_q = seqlen_q_ori * num_groups;

    query = query.reshape({batch_size, seqlen_q_ori, num_heads_k, num_groups, head_size}).transpose(2, 3)
                .reshape({batch_size, seqlen_q_ori, num_heads_q, head_size});

    // CHECK_SHAPE(query, batch_size, seqlen_q, num_heads, head_size);
    // CHECK_SHAPE(key_cache, num_blocks, page_block_size, num_heads, head_size);

    // TODO: transpose before return!!! But it may not be required.
    auto output = torch::empty({batch_size, seqlen_q, num_heads_q, head_size_v}, opts);
    auto softmax_lse = torch::empty({batch_size, num_heads_q, seqlen_q}, opts.dtype(torch::kFloat32));

    torch::Tensor softmax_lseaccum;
    torch::Tensor output_accum;
    int32_t num_splits = calculate_num_splits<Traits>(batch_size, num_heads, seqlen_q);
    if (num_splits > 1)
    {
        softmax_lseaccum = torch::empty({batch_size, num_heads, num_splits, seqlen_q}, opts.dtype(torch::kFloat32));
        output_accum = torch::empty({batch_size, num_heads, num_splits, seqlen_q, head_size_v}, opts.dtype(torch::kFloat32));
    }

    FlashMlaPrefillFwdParams params = {};
    params.p_cu_seqlens_k            = cache_seqlens.data_ptr<int32_t>();
    params.p_block_table             = block_table.data_ptr<int32_t>();

    params.p_query            = query.data_ptr();
    params.p_key              = key_cache.data_ptr();
    params.p_value            = vcache.data_ptr();
    params.p_output           = output.data_ptr();
    params.p_softmax_lse      = softmax_lse.data_ptr();
    params.p_softmax_lseaccum = softmax_lseaccum.data_ptr();
    params.p_output_accum     = output_accum.data_ptr();

    params.size_b                   = batch_size;
    params.size_s                   = seqlen_q;
    params.size_h                   = num_heads_q_ori;
    params.hq_hk_ratio              = num_heads_q_ori / num_heads_k;
    params.block_table_batch_stride = block_table.stride(0);
    params.page_block_size          = page_block_size;
    params.scale_softmax            = softmax_scale;
    params.scale_softmax_log2       = float(softmax_scale * M_LOG2E);

    params.stride_b_q = query.stride(0);
    params.stride_s_q = query.stride(1);
    params.stride_h_q = query.stride(2);
    params.stride_b_k = key_cache.stride(0);
    params.stride_s_k = key_cache.stride(1); // size_hk * size_d
    params.stride_h_k = key_cache.stride(2);
    params.stride_b_v = vcache.stride(0);
    params.stride_s_v = vcache.stride(1);    // size_hk * size_d
    params.stride_h_v = vcache.stride(2);
    params.stride_b_o = output.stride(0);
    params.stride_s_o = output.stride(1);
    params.stride_h_o = output.stride(2);
    // TODO: not all of the following params are required?
    params.stride_b_lseacc = softmax_lseaccum.stride(0);
    params.stride_h_lseacc = softmax_lseaccum.stride(1);
    params.stride_sp_lseacc = softmax_lseaccum.stride(2);
    params.stride_b_oacc = output_accum.stride(0);
    params.stride_h_oacc = output_accum.stride(1);
    params.stride_sp_oacc = output_accum.stride(2);
    params.stride_s_oacc = output_accum.stride(3);

    TORCH_CHECK(params.p_key == params.p_value, "Key and value are expected as the same thing for now.");

    // TODO: Replace the follow code with DISPATCH_FMLA_TYPES!
    TORCH_CHECK(query.scalar_type() == at::ScalarType::Half, "Only support fp16 inputs!")
    dispatch_fmla_fwd_splictkv_prefill<Traits, ck_tile::fp16_t, float, Is_causal, true>(params, num_splits);

	// using acc_t = float;
    // DISPATCH_FMLA_TYPES(
    //     query.scalar_type(),
    //     is_causal,
    //     "fmla_fwd",
    //     [&](){
    //         dispatch_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, Is_causal, true>(params, num_splits);
    //     }();
    // );

    // TODO: transpose before return!!!
    return {output, softmax_lse};
}

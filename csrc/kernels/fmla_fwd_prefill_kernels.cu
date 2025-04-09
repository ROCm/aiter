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
          int32_t kNumWarps_>
struct FlashMlaPrefillKernelTrait
{
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

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);
};

template<typename Traits, bool kIsSameKV_, typename scalar_t, typename acc_t>
struct FlashMlaPrefillPolicy
{
    static constexpr bool kIsSameKV = kIsSameKV_;

    // For QS+QR mixed implementation, VGPR always store 256 elements in row/along M.
    // So the rest are stored in SMEM.
    static constexpr uint32_t kMPerReg = 256;
    static constexpr uint32_t kNumPrefetchKV = 1;

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Traits::kNumThreads;
        constexpr index_t kMPerBlock = Traits::kBlockM0;
        constexpr index_t kKPerBlock = Traits::kBlockK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(scalar_t);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr uint32_t kMPerBlock = Traits::kBlockM;
        // #elements store in SMEM along K0 for query.
        constexpr uint32_t kKPerBlock = Traits::kSizeD - kMPerReg;
        constexpr uint32_t kKPack     = 16 / sizeof(scalar_t);

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
        constexpr uint32_t kNPerBlock = Traits::kBlockN0;
        constexpr uint32_t kKPerBlock = Traits::kBlockK0;
        constexpr uint32_t kKPack     = 16 / sizeof(scalar_t);

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

    CK_TILE_HOST_DEVICE static constexpr uint32_t GetSmemSizeQ()
    {
        constexpr uint32_t lds_alignment = 16; // optional
        constexpr uint32_t q_smem_size =
            ck_tile::integer_divide_ceil(
                sizeof(scalar_t) * MakeQLdsBlockDescriptor().get_element_space_size(),
                lds_alignment) *
            lds_alignment;
        return q_smem_size;
    }

    CK_TILE_HOST_DEVICE static constexpr uint32_t GetSmemSizeSingleKV()
    {
        constexpr uint32_t SingleKSize = MakeKLdsBlockDescriptor().get_element_space_size();
        constexpr uint32_t SingleVSize =[&]() {
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

    CK_TILE_HOST_DEVICE static constexpr uint32_t GetSmemSize()
    {
        return GetSmemSizeQ() + kNumPrefetchKV * GetSmemSizeSingleKV();
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

    int32_t size_b;
    int32_t size_s;
    int32_t size_h;
    int32_t hq_hk_ratio;
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
    constexpr uint32_t num_tile_n1 = ck_tile::integer_divide_ceil(Traits::kSizeDV, Traits::kBlockN1);

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

// =====================================================================================================================
// Kernel Entry
//

template <typename Traits, typename scalar_t, typename acc_t, bool kIsCausal, bool kIsSameKV, bool kDoSplit>
__global__ kn_fmla_fwd_splictkv_prefill(
    const FlashMlaPrefillFwdParams params)
{
    using Policy = FlashMlaPrefillPolicy<Traits, kIsSameKV, scalar_t, acc_t>;

    // allocate LDS
    __shared__ uint8_t smem_ptr[Policy::GetSmemSize()];

    const auto [mid, nid, split_id, hqid, bid] = GetTileIndex();
    const auto hkid = hqid / params.hq_hk_ratio;
    const uint32_t offset_m = __builtin_amdgcn_readfirstlane(mid * Traits::kBlockM0);
    const uint32_t offset_n = __builtin_amdgcn_readfirstlane(nid * Traits::kBlockN1);

    const uint64_t key_start_seq       = params.p_cu_seqlens_k[bid];
    const uint64_t seqlen_k            = params.p_cu_seqlens_k[bid+1] - key_start_seq;
    const uint64_t batch_offset_q      = bid * params.stride_b_q;
    const uint64_t batch_offset_k      = key_start_seq * params.stride_s_k;
    // TODO: this is row major code. May not currect
    const uint64_t batch_offset_v      = key_start_seq * params.stride_s_v;
    const uint64_t batch_offset_lseacc = bid * params.stride_b_o;
    const uint64_t batch_offset_oacc   = bid * params.stride_b_oacc;

    const scalar_t* p_query = reinterpret_cast<const scalar_t*>(params.p_query) +
                              hqid * params.stride_h_q +
                              batch_offset_q;
    const scalar_t* p_key   = reinterpret_cast<const scalar_t*>(params.p_key) +
                              hkid * params.stride_h_k +
                              batch_offset_k;
    const scalar_t* p_value = reinterpret_cast<const scalar_t*>(params.p_value) +
                              hkid * params.stride_h_v +
                              batch_offset_v;
    scalar_t* p_oacc        = reinterpret_cast<scalar_t*>(params.p_output_accum) +
                              hqid * params.stride_h_oacc +
                              batch_offset_oacc +
                              split_id * params.stride_sp_oacc;

    const auto q_dram = [&] {
        return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            p_query,
            ck_tile::make_tuple(params.size_s, Traits::kSizeD),
            ck_tile::make_tuple(params.stride_s_q, 1),
            ck_tile::number<Policy::GetAlignmentQ()>{},
            ck_tile::number<1>{});
    }();
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

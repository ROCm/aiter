// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <sstream>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "mla.h"

template <int32_t kSizeDV_,
          int32_t kNumHeadQ_,
          int32_t kMaxSplits_,
          bool    kOutputLse_>
struct MlaReduceKernelV1Traits
{
    static constexpr int32_t kSizeDV      = kSizeDV_;       // hidden dimension size of value/output
    static constexpr int32_t kNumHeadQ    = kNumHeadQ_;     // head count of q
    static constexpr int32_t kMaxSplits   = kMaxSplits_;
    static constexpr int32_t kNumWarps    = 2;
    static constexpr int32_t kNumThreads  = kNumWarps * ck_tile::get_warp_size();
    static constexpr bool    kOutputLse   = kOutputLse_;
};

struct MlaReduceKernelV1Params
{
    const int32_t*            p_reduce_indptr;
    const MlaPartialTileInfo* p_reduce_final_map;
    const int32_t*            p_reduce_partial_map;

    void* __restrict__ p_final_lse;
    void* __restrict__ p_final_output;
    void* __restrict__ p_partial_lse;
    void* __restrict__ p_partial_output;

    int32_t stride_s_o;
    int32_t stride_h_o;
};

// Returns count of warps which don't contain any idle thread.
template <int32_t NumWarps, int32_t M, int32_t N>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
{
    static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);
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
template <int32_t NumWarps, int32_t M, int32_t N, typename scalar_t>
CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
{
    constexpr int32_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N>();
    constexpr int32_t ElemPerThread = (M * N) / (MaxNumWarps * ck_tile::get_warp_size());
    constexpr int32_t MaxNPerThread = 16 / sizeof(scalar_t);
    return ck_tile::min(MaxNPerThread, ElemPerThread);
}

template <typename Traits, typename scalar_t>
CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
{
    constexpr int32_t kVectorN     = GetVectorSizeForTile<Traits::kNumWarps, 1, Traits::kSizeDV, scalar_t>();
    constexpr int32_t kThrPerWarpN = ck_tile::get_warp_size();
    constexpr int32_t kNumWarpN    = Traits::kNumWarps;
    constexpr int32_t kNumRepeat   = ck_tile::max(1, Traits::kSizeDV / kThrPerWarpN / kNumWarpN / kVectorN);

    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,    // no replicate
            ck_tile::tuple<ck_tile::sequence<1>,
                           ck_tile::sequence<kNumRepeat, kNumWarpN, kThrPerWarpN, kVectorN>>,
            ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2>>,
            ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2>>,
            ck_tile::sequence<2, 1, 2>,
            ck_tile::sequence<0, 0, 3>>{});
}

template <typename Traits, typename scalar_t>
CK_TILE_DEVICE static auto MakeTileWindow(
    scalar_t* p_tile)
{
    const auto naive_view =
        ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            p_tile,
            ck_tile::make_tuple(1, Traits::kSizeDV),    // lengths
            ck_tile::make_tuple(Traits::kSizeDV, 1),    // strides
            ck_tile::number<Traits::kSizeDV>{},         // last dim alignment
            ck_tile::number<1>{});                      // last dim stride

    const auto tile_window = ck_tile::make_tile_window(
        naive_view,
        ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                            ck_tile::number<Traits::kSizeDV>{}),
        {0, 0});                                                // origin

    return tile_window;
}

template <typename Traits, typename lse_t, typename out_t>
__global__ void kn_mla_reduce_v1(
    const MlaReduceKernelV1Params params)
{
    __shared__ float lds_lse_scale[Traits::kMaxSplits];

    const int32_t lane_idx = ck_tile::get_lane_id();
    const int32_t work_idx = blockIdx.x;
    const int32_t head_idx = blockIdx.y;

    const int32_t reduce_tile_start = params.p_reduce_indptr[work_idx];
    const int32_t reduce_tile_end = params.p_reduce_indptr[work_idx + 1];

    if (reduce_tile_start < reduce_tile_end)
    {
        const MlaPartialTileInfo final_loc = params.p_reduce_final_map[work_idx];

        // Assuming that the layout of LSE final output is in [bs, h].
        // Thus, stride of head is 1 and stride of b/s is #heads.
        lse_t* p_final_lse_base = reinterpret_cast<lse_t*>(params.p_final_lse) + head_idx;
        const float* p_partial_lse_base =
            reinterpret_cast<const float*>(params.p_partial_lse) + head_idx;

        // Assuming that the layout of partial output is in [bs, h, d].
        // Thus, stride of hidden dim is 1, head is Traits::kSizeDV and b/s is Traits::kSizeDV * #heads
        // while the strides are 1, params.stride_h_o and params.stride_s_o for final output.
        out_t* p_final_out_base = reinterpret_cast<out_t*>(params.p_final_output) + head_idx * params.stride_h_o;
        const float* p_partial_output_base =
            reinterpret_cast<float*>(params.p_partial_output) + head_idx * Traits::kSizeDV;

        for (int32_t seq_idx = final_loc.q_start; seq_idx < final_loc.q_end; ++seq_idx)
        {
            const int32_t local_seqlen_idx = seq_idx - final_loc.q_start;
            const float* p_partial_lse_seq_base = p_partial_lse_base + local_seqlen_idx * Traits::kNumHeadQ;
            const float* p_partial_output_seq_base =
                p_partial_output_base + local_seqlen_idx * Traits::kNumHeadQ * Traits::kSizeDV;

            if (ck_tile::get_warp_id() == 0)
            {
                constexpr int32_t kNumLsePerThr =
                    ck_tile::integer_divide_ceil(Traits::kMaxSplits, ck_tile::get_warp_size());
                float local_lse[kNumLsePerThr];

                // Load thread local LSE and get local max LSE
                float max_lse = -INFINITY;

                #pragma unroll
                for (int32_t i = 0; i < kNumLsePerThr; ++i)
                {
                    const int32_t split_idx = i * ck_tile::get_warp_size() + lane_idx;
                    const int32_t tile_idx = reduce_tile_start + split_idx;
                    if (tile_idx < reduce_tile_end)
                    {
                        const int64_t reduce_tile_pos =
                            params.p_reduce_partial_map[tile_idx] * int64_t(Traits::kNumHeadQ);
                        const float lse = p_partial_lse_seq_base[reduce_tile_pos];
                        local_lse[i] = lse;
                        max_lse = ck_tile::max(max_lse, lse);
                    }
                    else
                    {
                        local_lse[i] = -INFINITY;
                    }
                }

                // Get global max LSE
                #pragma unroll
                for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
                {
                    max_lse = ck_tile::max(max_lse, __shfl_xor(max_lse, offset));
                }

                // Get sum of LSE
                float sum_lse = 0.f;
                #pragma unroll
                for (int32_t i = 0; i < kNumLsePerThr; ++i)
                {
                    sum_lse += expf(local_lse[i] - max_lse);
                }
                #pragma unroll
                for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
                {
                    sum_lse += __shfl_xor(sum_lse, offset);
                }

                // Get global LSE
                float global_lse = ((sum_lse == 0.f) || (sum_lse != sum_lse)) ? INFINITY : (logf(sum_lse) + max_lse);
                if constexpr (Traits::kOutputLse)
                {
                    if (lane_idx == 0)
                    {
                        lse_t* p_final_lse = p_final_lse_base + seq_idx * Traits::kNumHeadQ;
                        *p_final_lse = ck_tile::type_convert<lse_t>(global_lse);
                    }
                }

                // Write LSE to LDS
                #pragma unroll
                for (int32_t i = 0; i < kNumLsePerThr; ++i)
                {
                    const int32_t split_idx = i * ck_tile::get_warp_size() + lane_idx;
                    if ((reduce_tile_start + split_idx) < reduce_tile_end)
                    {
                        lds_lse_scale[split_idx] = expf(local_lse[i] - global_lse);
                    }
                }
            }

            __builtin_amdgcn_sched_barrier(0);
            ck_tile::block_sync_lds();

            auto oaccu_window = ck_tile::make_tile_window(MakeTileWindow<Traits, const float>(nullptr),
                                                        MakeOutputTileDistribution<Traits, const float>());
            auto reg_out = ck_tile::make_static_distributed_tensor<float>(
                decltype(ck_tile::load_tile(oaccu_window))::get_tile_distribution());
            ck_tile::set_tile(reg_out, 0.f);

            for (int32_t tile_idx = reduce_tile_start; tile_idx < reduce_tile_end; ++tile_idx)
            {
                const int32_t split_idx = tile_idx - reduce_tile_start;

                const int64_t reduce_tile_pos =
                    params.p_reduce_partial_map[tile_idx] * int64_t(Traits::kNumHeadQ * Traits::kSizeDV);
                const float* p_partial_output = p_partial_output_seq_base + reduce_tile_pos;
                oaccu_window.set_bottom_tensor_view_data_ptr(p_partial_output);

                const float lse_scale = lds_lse_scale[split_idx];
                auto oaccu = ck_tile::load_tile(oaccu_window);
                ck_tile::sweep_tile(oaccu, [&](auto idx) {
                    reg_out(idx) += lse_scale * oaccu(idx);
                });
            }

            out_t* p_final_out = p_final_out_base + seq_idx * params.stride_s_o;
            auto dram_out = MakeTileWindow<Traits, out_t>(p_final_out);
            ck_tile::store_tile(dram_out, ck_tile::cast_tile<out_t>(reg_out));
        }
    }
}

#define MLA_MERGE_CASE(NUM_HEAD_C, NUM_CU_C, OUTPUT_LSE_C, ...)                                             \
    constexpr int32_t NumHeads  = (NUM_HEAD_C);                                                             \
    constexpr int32_t NumCUs    = (NUM_CU_C);                                                               \
    constexpr bool    OutputLse = (OUTPUT_LSE_C);                                                           \
    using Traits = MlaReduceKernelV1Traits<512, NumHeads, NumCUs, OutputLse>;                               \
    __VA_ARGS__;

#define MLA_MERGE_CASE_IF(NUM_HEAD, NUM_HEAD_C, NUM_CU, NUM_CU_C, OUTPUT_LSE, OUTPUT_LSE_C, ...)            \
    if (((NUM_HEAD) == (NUM_HEAD_C)) && ((NUM_CU) == (NUM_CU_C)) && ((OUTPUT_LSE) == (OUTPUT_LSE_C)))       \
    {                                                                                                       \
        MLA_MERGE_CASE(NUM_HEAD_C, NUM_CU_C, OUTPUT_LSE_C, __VA_ARGS__)                                     \
    }

#define MLA_MERGE_CASE_EF(NUM_HEAD, NUM_HEAD_C, NUM_CU, NUM_CU_C, OUTPUT_LSE, OUTPUT_LSE_C, ...)            \
    else if (((NUM_HEAD) == (NUM_HEAD_C)) && ((NUM_CU) == (NUM_CU_C)) && ((OUTPUT_LSE) == (OUTPUT_LSE_C)))  \
    {                                                                                                       \
        MLA_MERGE_CASE(NUM_HEAD_C, NUM_CU_C, OUTPUT_LSE_C, __VA_ARGS__)                                     \
    }

#define MLA_MERGE_ERROR(NUM_HEAD, NUM_CU, OUTPUT_LSE, NAME)                                                 \
    {                                                                                                       \
        std::stringstream ss;                                                                               \
        ss << "#heads: " << (NUM_HEAD) << ", #CUs: " << (NUM_CU) << ", Output LSE: " << (OUTPUT_LSE);       \
        TORCH_CHECK(false, NAME " doesn't support the specified settings: ", ss.str().c_str(), ".");        \ 
    }

#define DISPATCH_MLA_MERGE_KERNEL(LSE_TYPE, OUT_TYPE, NUM_HEAD, NUM_CU, OUTPUT_LSE, NAME, ...)              \
    switch ((LSE_TYPE))                                                                                     \
    {                                                                                                       \
        case at::ScalarType::Float:                                                                         \
        {                                                                                                   \
            using lse_t = float;                                                                            \
            switch ((OUT_TYPE))                                                                             \
            {                                                                                               \
                case at::ScalarType::BFloat16:                                                              \
                {                                                                                           \
                    using out_t = ck_tile::bf16_t;                                                          \
                    MLA_MERGE_CASE_IF(NUM_HEAD,  16, NUM_CU, 80, OUTPUT_LSE, true,  __VA_ARGS__)            \
                    MLA_MERGE_CASE_EF(NUM_HEAD,  16, NUM_CU, 80, OUTPUT_LSE, false, __VA_ARGS__)            \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, NUM_CU, 80, OUTPUT_LSE, true,  __VA_ARGS__)            \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, NUM_CU, 80, OUTPUT_LSE, false, __VA_ARGS__)            \
                    else MLA_MERGE_ERROR(NUM_HEAD, NUM_CU, OUTPUT_LSE, NAME);                               \
                }                                                                                           \
                break;                                                                                      \
                case at::ScalarType::Half:                                                                  \
                {                                                                                           \
                    using out_t = ck_tile::fp16_t;                                                          \
                    MLA_MERGE_CASE_IF(NUM_HEAD,  16, NUM_CU, 80, OUTPUT_LSE, true,  __VA_ARGS__)            \
                    MLA_MERGE_CASE_EF(NUM_HEAD,  16, NUM_CU, 80, OUTPUT_LSE, false, __VA_ARGS__)            \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, NUM_CU, 80, OUTPUT_LSE, true,  __VA_ARGS__)            \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, NUM_CU, 80, OUTPUT_LSE, false, __VA_ARGS__)            \
                    else MLA_MERGE_ERROR(NUM_HEAD, NUM_CU, OUTPUT_LSE, NAME);                               \
                }                                                                                           \
                break;                                                                                      \
                default:                                                                                    \
                    TORCH_CHECK(false, NAME " doesn't support output type ", toString((OUT_TYPE)), ".");    \
            }                                                                                               \
        }                                                                                                   \
        break;                                                                                              \
        default:                                                                                            \
            TORCH_CHECK(false, NAME " doesn't support LSE type ", toString((LSE_TYPE)), ".");               \
    }

template <typename Traits, typename lse_t, typename out_t>
void dispatch_mla_reduce_v1(
    const MlaReduceKernelV1Params& params,
    const int32_t num_reduce_tile,
    const cudaStream_t& stream)
{
    const dim3 grid = dim3(num_reduce_tile, Traits::kNumHeadQ);
    kn_mla_reduce_v1<Traits, lse_t, out_t><<<grid, Traits::kNumThreads, 0, stream>>>(params);
}

void mla_reduce_v1(
    const torch::Tensor&          partial_output,        // contiguous [max(reduce_partial_map)+s, h, dv]
    const torch::Tensor&          partial_lse,           // contiguous [max(reduce_partial_map)+s, h]
    const torch::Tensor&          reduce_indptr,         // contiguous [#work + 1]
    const torch::Tensor&          reduce_final_map,      // contiguous [#work, 2]
    const torch::Tensor&          reduce_partial_map,    // contiguous [reduce_indptr[-1]]
    torch::Tensor&                final_output,          //            [bs, h, dv]
    std::optional<torch::Tensor>& final_lse)             // contiguous [bs, h]
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(final_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const bool output_lse = final_lse.has_value();
    const int32_t num_reduce_tile = reduce_indptr.size(0) - 1;
    const int32_t num_heads = partial_output.size(-2);
    TORCH_CHECK(num_reduce_tile == reduce_final_map.size(0),
                __func__, ": Invalid size of reduce_indptr or reduce_final_map!");

    if (num_reduce_tile > 0)
    {
        MlaReduceKernelV1Params params = {};
        params.p_reduce_indptr = reduce_indptr.data_ptr<int32_t>();
        params.p_reduce_final_map = reinterpret_cast<const MlaPartialTileInfo*>(reduce_final_map.data_ptr());
        params.p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();
        params.p_final_lse = output_lse ? final_lse.value().data_ptr() : nullptr;
        params.p_final_output = final_output.data_ptr();
        params.p_partial_lse = partial_lse.data_ptr();
        params.p_partial_output = partial_output.data_ptr();
        params.stride_s_o = final_output.stride(-3);
        params.stride_h_o = final_output.stride(-2);

        DISPATCH_MLA_MERGE_KERNEL(
            output_lse ? final_lse.value().scalar_type() : at::ScalarType::Float,
            final_output.scalar_type(),
            num_heads,
            dev_prop.multiProcessorCount,
            output_lse,
            "kn_mla_reduce_v1",
            dispatch_mla_reduce_v1<Traits, lse_t, out_t>(params, num_reduce_tile, stream)
        );
    }
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <sstream>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "mla.h"

template <int32_t kSizeDV_,
          int32_t kNumHeadQ_,
          bool    kOutputLse_>
struct MlaReduceKernelV1Traits
{
    static constexpr int32_t kSizeDV          = kSizeDV_;       // hidden dimension size of value/output
    static constexpr int32_t kNumHeadQ        = kNumHeadQ_;     // head count of q
    static constexpr int32_t kNumWarps        = 2;
    static constexpr int32_t kNumThreads      = kNumWarps * ck_tile::get_warp_size();
    static constexpr int32_t kMaxVgprLocalLse = 16;             // scratch buffer will be used with larger value
    static constexpr bool    kOutputLse       = kOutputLse_;
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
    int32_t max_splits;
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

template <typename T>
class LocalLseLds
{
public:
    CK_TILE_DEVICE LocalLseLds(T* p_local_lse, const int32_t group_size, const int32_t idx_in_group) :
        p_local_lse_(p_local_lse), group_size_(group_size), idx_in_group_(idx_in_group) {}
    CK_TILE_DEVICE T& operator[](int32_t idx) { return p_local_lse_[idx * group_size_ + idx_in_group_]; }
    CK_TILE_DEVICE T operator[](int32_t idx) const { return p_local_lse_[idx * group_size_ + idx_in_group_]; }

private:
    T* p_local_lse_;
    int32_t group_size_;
    int32_t idx_in_group_;
};

template <typename Traits,
          bool kFastMode,
          typename LocalLse,
          typename lse_t>
CK_TILE_DEVICE void reduce_lse(
    const MlaReduceKernelV1Params& params,
    const int32_t                  seq_idx,
    const int32_t                  reduce_tile_start,
    const int32_t                  reduce_tile_end,
    const int32_t                  num_lse_per_thr,
    const int32_t                  q_len,
    const float*                   p_partial_lse_seq_base,
    LocalLse&                      local_lse,
    float*                         p_lds_lse_scale,
    lse_t*                         p_final_lse_base)
{
    if (ck_tile::get_warp_id() == 0)
    {
        const int32_t lane_idx = ck_tile::get_lane_id();

        // Load thread local LSE and get local max LSE
        float max_lse = -INFINITY;

        #pragma unroll 2
        for (int32_t i = 0; i < num_lse_per_thr; ++i)
        {
            const int32_t split_idx = i * ck_tile::get_warp_size() + lane_idx;
            const int32_t tile_idx = reduce_tile_start + split_idx;
            if (tile_idx < reduce_tile_end)
            {
                const int32_t q_loc = [&]() {
                    if constexpr (kFastMode)
                    {
                        return tile_idx * q_len;
                    }
                    else
                    {
                        return params.p_reduce_partial_map[tile_idx];
                    }
                }();
                const int64_t reduce_tile_pos = q_loc * int64_t(Traits::kNumHeadQ);
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
            const int32_t srd_lane = (offset ^ ck_tile::get_warp_size()) ^ ck_tile::get_lane_id();
            max_lse = ck_tile::max(max_lse, ck_tile::warp_shuffle(max_lse, srd_lane));
        }

        // Get sum of LSE
        float sum_lse = 0.f;
        #pragma unroll 2
        for (int32_t i = 0; i < num_lse_per_thr; ++i)
        {
            sum_lse += expf(local_lse[i] - max_lse);
        }
        #pragma unroll
        for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
        {
            const int32_t srd_lane = (offset ^ ck_tile::get_warp_size()) ^ ck_tile::get_lane_id();
            sum_lse += ck_tile::warp_shuffle(sum_lse, srd_lane);
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
        #pragma unroll 2
        for (int32_t i = 0; i < num_lse_per_thr; ++i)
        {
            const int32_t split_idx = i * ck_tile::get_warp_size() + lane_idx;
            if ((reduce_tile_start + split_idx) < reduce_tile_end)
            {
                p_lds_lse_scale[split_idx] = expf(local_lse[i] - global_lse);
            }
        }
    }
}

template <typename Traits,
          bool kFastMode,
          typename out_t>
CK_TILE_DEVICE void reduce_output(
    const MlaReduceKernelV1Params& params,
    const int32_t                  seq_idx,
    const int32_t                  reduce_tile_start,
    const int32_t                  reduce_tile_end,
    const int32_t                  q_len,
    const float*                   p_lds_lse_scale,
    const float*                   p_partial_output_seq_base,
    out_t*                         p_final_out_base)
{
    auto oaccu_window = ck_tile::make_tile_window(MakeTileWindow<Traits, const float>(nullptr),
                                                  MakeOutputTileDistribution<Traits, const float>());
    auto reg_out = ck_tile::make_static_distributed_tensor<float>(
        decltype(ck_tile::load_tile(oaccu_window))::get_tile_distribution());
    ck_tile::set_tile(reg_out, 0.f);

    for (int32_t tile_idx = reduce_tile_start; tile_idx < reduce_tile_end; ++tile_idx)
    {
        const int32_t split_idx = tile_idx - reduce_tile_start;
        const int32_t q_loc = [&]() {
            if constexpr (kFastMode)
            {
                return tile_idx * q_len;
            }
            else
            {
                return params.p_reduce_partial_map[tile_idx];
            }
        }();
        const int64_t reduce_tile_pos = q_loc * int64_t(Traits::kNumHeadQ * Traits::kSizeDV);
        const float* p_partial_output = p_partial_output_seq_base + reduce_tile_pos;
        oaccu_window.set_bottom_tensor_view_data_ptr(p_partial_output);

        const float lse_scale = p_lds_lse_scale[split_idx];
        auto oaccu = ck_tile::load_tile(oaccu_window);
        ck_tile::sweep_tile(oaccu, [&](auto idx) {
            reg_out(idx) += lse_scale * oaccu(idx);
        });
    }

    out_t* p_final_out = p_final_out_base + seq_idx * params.stride_s_o;
    auto dram_out = MakeTileWindow<Traits, out_t>(p_final_out);
    ck_tile::store_tile(dram_out, ck_tile::cast_tile<out_t>(reg_out));
}

template <typename Traits, typename lse_t, typename out_t>
__global__ void kn_mla_reduce_v1(
    const MlaReduceKernelV1Params params)
{
    extern __shared__ float p_lds_lse_scale[];

    const int32_t head_idx = blockIdx.x;
    const int32_t work_idx = blockIdx.y;

    const int32_t reduce_tile_start = params.p_reduce_indptr[work_idx];
    const int32_t reduce_tile_end = params.p_reduce_indptr[work_idx + 1];


    if (reduce_tile_start < reduce_tile_end)
    {
        int32_t q_len = 0;
        MlaPartialTileInfo final_loc{};
        const int32_t fast_mode = params.p_reduce_partial_map[0];
        if (fast_mode == -1)
        {
            q_len = params.p_reduce_partial_map[1];
            final_loc.q_start = q_len * work_idx;
            final_loc.q_end   = final_loc.q_start + q_len;
        }
        else
        {
            final_loc = params.p_reduce_final_map[work_idx];
        }

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

        const int32_t num_lse_per_thr =
            ck_tile::integer_divide_ceil(params.max_splits, ck_tile::get_warp_size());

        if (fast_mode == -1)
        {
            for (int32_t seq_idx = final_loc.q_start; seq_idx < final_loc.q_end; ++seq_idx)
            {
                const int32_t local_seqlen_idx = seq_idx - final_loc.q_start;
                const float* p_partial_lse_seq_base = p_partial_lse_base + local_seqlen_idx * Traits::kNumHeadQ;
                const float* p_partial_output_seq_base =
                    p_partial_output_base + local_seqlen_idx * Traits::kNumHeadQ * Traits::kSizeDV;

                float* p_local_lse = p_lds_lse_scale + params.max_splits;
                LocalLseLds<float> local_lse(p_local_lse, ck_tile::get_warp_size(), ck_tile::get_lane_id());
                reduce_lse<Traits, true>(
                    params,
                    seq_idx,
                    reduce_tile_start,
                    reduce_tile_end,
                    num_lse_per_thr,
                    q_len,
                    p_partial_lse_seq_base,
                    local_lse,
                    p_lds_lse_scale,
                    p_final_lse_base);

                __builtin_amdgcn_sched_barrier(0);
                ck_tile::block_sync_lds();

                reduce_output<Traits, true>(
                    params,
                    seq_idx,
                    reduce_tile_start,
                    reduce_tile_end,
                    q_len,
                    p_lds_lse_scale,
                    p_partial_output_seq_base,
                    p_final_out_base);
            }
        }
        else
        {
            for (int32_t seq_idx = final_loc.q_start; seq_idx < final_loc.q_end; ++seq_idx)
            {
                const int32_t local_seqlen_idx = seq_idx - final_loc.q_start;
                const float* p_partial_lse_seq_base = p_partial_lse_base + local_seqlen_idx * Traits::kNumHeadQ;
                const float* p_partial_output_seq_base =
                    p_partial_output_base + local_seqlen_idx * Traits::kNumHeadQ * Traits::kSizeDV;

                float* p_local_lse = p_lds_lse_scale + params.max_splits;
                LocalLseLds<float> local_lse(p_local_lse, ck_tile::get_warp_size(), ck_tile::get_lane_id());
                reduce_lse<Traits, false>(
                    params,
                    seq_idx,
                    reduce_tile_start,
                    reduce_tile_end,
                    num_lse_per_thr,
                    q_len,
                    p_partial_lse_seq_base,
                    local_lse,
                    p_lds_lse_scale,
                    p_final_lse_base);

                __builtin_amdgcn_sched_barrier(0);
                ck_tile::block_sync_lds();

                reduce_output<Traits, false>(
                    params,
                    seq_idx,
                    reduce_tile_start,
                    reduce_tile_end,
                    q_len,
                    p_lds_lse_scale,
                    p_partial_output_seq_base,
                    p_final_out_base);
            }
        }
    }
}

#define MLA_MERGE_CASE(NUM_HEAD_C, OUTPUT_LSE_C, NAME, ...)                                                 \
    constexpr int32_t NumHeads  = (NUM_HEAD_C);                                                             \
    constexpr bool    OutputLse = (OUTPUT_LSE_C);                                                           \
    using Traits = MlaReduceKernelV1Traits<512, NumHeads, OutputLse>;                                       \
    __VA_ARGS__;

#define MLA_MERGE_CASE_IF(NUM_HEAD, NUM_HEAD_C, OUTPUT_LSE, OUTPUT_LSE_C, NAME, ...)                        \
    if (((NUM_HEAD) == (NUM_HEAD_C)) && ((OUTPUT_LSE) == (OUTPUT_LSE_C)))                                   \
    {                                                                                                       \
        MLA_MERGE_CASE(NUM_HEAD_C, OUTPUT_LSE_C, NAME, __VA_ARGS__)                                         \
    }

#define MLA_MERGE_CASE_EF(NUM_HEAD, NUM_HEAD_C, OUTPUT_LSE, OUTPUT_LSE_C, NAME, ...)                        \
    else if (((NUM_HEAD) == (NUM_HEAD_C)) && ((OUTPUT_LSE) == (OUTPUT_LSE_C)))                              \
    {                                                                                                       \
        MLA_MERGE_CASE(NUM_HEAD_C, OUTPUT_LSE_C, NAME, __VA_ARGS__)                                         \
    }

#define MLA_MERGE_ERROR(NUM_HEAD, OUTPUT_LSE, NAME)                                                         \
    {                                                                                                       \
        std::stringstream ss;                                                                               \
        ss << "#heads: " << (NUM_HEAD) << ", Output LSE: " << (OUTPUT_LSE);                                 \
        TORCH_CHECK(false, NAME " doesn't support the specified settings: ", ss.str().c_str(), ".");        \
    }

#define DISPATCH_MLA_MERGE_KERNEL(LSE_TYPE, OUT_TYPE, NUM_HEAD, OUTPUT_LSE, NAME, ...)                      \
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
                    MLA_MERGE_CASE_IF(NUM_HEAD,  16, OUTPUT_LSE, true,  NAME, __VA_ARGS__)                  \
                    MLA_MERGE_CASE_EF(NUM_HEAD,  16, OUTPUT_LSE, false, NAME, __VA_ARGS__)                  \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, OUTPUT_LSE, true,  NAME, __VA_ARGS__)                  \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, OUTPUT_LSE, false, NAME, __VA_ARGS__)                  \
                    else MLA_MERGE_ERROR(NUM_HEAD, OUTPUT_LSE, NAME);                                       \
                }                                                                                           \
                break;                                                                                      \
                case at::ScalarType::Half:                                                                  \
                {                                                                                           \
                    using out_t = ck_tile::fp16_t;                                                          \
                    MLA_MERGE_CASE_IF(NUM_HEAD,  16, OUTPUT_LSE, true,  NAME, __VA_ARGS__)                  \
                    MLA_MERGE_CASE_EF(NUM_HEAD,  16, OUTPUT_LSE, false, NAME, __VA_ARGS__)                  \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, OUTPUT_LSE, true,  NAME, __VA_ARGS__)                  \
                    MLA_MERGE_CASE_EF(NUM_HEAD, 128, OUTPUT_LSE, false, NAME, __VA_ARGS__)                  \
                    else MLA_MERGE_ERROR(NUM_HEAD, OUTPUT_LSE, NAME);                                       \
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
    const int32_t                  num_reduce_tile,
    const cudaStream_t&            stream)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t lds_size = params.max_splits * sizeof(float) * 2;
    if (lds_size <= dev_prop.maxSharedMemoryPerMultiProcessor)
    {
        const dim3 grid = dim3(Traits::kNumHeadQ, num_reduce_tile);
        kn_mla_reduce_v1<Traits, lse_t, out_t><<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
    }
    else
    {
        TORCH_CHECK(false, "kn_mla_reduce_v1: There are too much splits. We cannot handle them.");
    }
}

void mla_reduce_v1(
    const torch::Tensor& partial_output,        // contiguous [max(reduce_partial_map)+s, h, dv]
    const torch::Tensor& partial_lse,           // contiguous [max(reduce_partial_map)+s, h]
    const torch::Tensor& reduce_indptr,         // contiguous [#work + 1]
    const torch::Tensor& reduce_final_map,      // contiguous [#work, 2]
    const torch::Tensor& reduce_partial_map,    // contiguous [reduce_indptr[-1]]
    torch::Tensor&       final_output,          //            [bs, h, dv]
    std::optional<torch::Tensor>&       final_lse)             // contiguous [bs, h]
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

    if (num_reduce_tile > 0)
    {
        MlaReduceKernelV1Params params = {};
        params.p_reduce_indptr = reduce_indptr.data_ptr<int32_t>();
        params.p_reduce_final_map =
            reinterpret_cast<const MlaPartialTileInfo*>(reduce_final_map.data_ptr());
        params.p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();
        params.p_final_lse = output_lse ? final_lse.value().data_ptr() : nullptr;
        params.p_final_output = final_output.data_ptr();
        params.p_partial_lse = partial_lse.data_ptr();
        params.p_partial_output = partial_output.data_ptr();
        params.stride_s_o = final_output.stride(-3);
        params.stride_h_o = final_output.stride(-2);
        params.max_splits = dev_prop.multiProcessorCount;

        DISPATCH_MLA_MERGE_KERNEL(
            output_lse ? final_lse.value().scalar_type() : at::ScalarType::Float,
            final_output.scalar_type(),
            num_heads,
            output_lse,
            "kn_mla_reduce_v1",
            dispatch_mla_reduce_v1<Traits, lse_t, out_t>(params, num_reduce_tile, stream)
        );
    }
}

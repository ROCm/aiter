// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "kittens.cuh"
#include "mla.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>

namespace hk     = kittens;
namespace hkdart = hk::ducks::art;
namespace hkm    = hk::macros;
namespace ckt    = ck_tile;

// ================================================================
// Temp Helper functions
// ================================================================
union FUI
{
    uint32_t ui;
    hk::fp8e4m3_4 fp8_4;
    struct
    {
        ckt::fp8_t x;
        ckt::fp8_t y;
        ckt::fp8_t z;
        ckt::fp8_t w;
    };
};
__device__ float4 convert_fp8x4_to_float4(FUI in)
{
    static constexpr __hip_fp8_interpretation_t interpret =
#if defined(__gfx950__)
        __HIP_E4M3;
#else
        __HIP_E4M3_FNUZ;
#endif
    float4 r;
    r.x = static_cast<float>(__half(__hip_cvt_fp8_to_halfraw(in.x, interpret)));
    r.y = static_cast<float>(__half(__hip_cvt_fp8_to_halfraw(in.y, interpret)));
    r.z = static_cast<float>(__half(__hip_cvt_fp8_to_halfraw(in.z, interpret)));
    r.w = static_cast<float>(__half(__hip_cvt_fp8_to_halfraw(in.w, interpret)));
    return r;
}

template <int GPR>
__device__ constexpr int reg_2_loc_q()
{
    constexpr int off = GPR - 92;
    return (off % 2) * 4 + (off / 2) * 32;
}

// ================================================================
// Main part
// ================================================================

template <typename q_t_, typename kv_t_, typename out_t_, int32_t kQoNumHead_>
struct HkMlaDecodeFwdTraits
{
    static constexpr int32_t kQoNumHead     = kQoNumHead_;
    static constexpr int32_t kKvNumHead     = 1;
    static constexpr int32_t kKvLoraRank    = 512;
    static constexpr int32_t kQkNopeHeadDim = kKvLoraRank;
    static constexpr int32_t kQkRopeHeadDim = 64;
    static constexpr int32_t kQkHeadDim     = kQkNopeHeadDim + kQkRopeHeadDim;
    static constexpr int32_t kVoHeadDim     = kKvLoraRank;
    static constexpr int32_t kPageSize      = 1;
    static constexpr int32_t kNumWarps      = 8;
    static constexpr int32_t kNumThreads    = kNumWarps * ckt::get_warp_size();
    static constexpr int32_t kOccupancy     = 1;
    static constexpr int32_t kBlockM        = 128; // Block=ThreadBlock
    static constexpr int32_t kBlockN        = 16;
    static constexpr int32_t kTileM         = kBlockM / kNumWarps; // Tile=ThreadWarp
    static constexpr int32_t kNumTilesM     = kBlockM / kTileM;

    static_assert(kBlockM == kQoNumHead, "Only supports nhead=128!");

    // base types
    using q_t   = q_t_;
    using kv_t  = kv_t_;
    using out_t = out_t_;
    // global memory tiles
    using gl_q = hk::gl<q_t, -1, kNumTilesM, kTileM, kQkHeadDim>; // [#batch*#seqlen, #warp, #head /
                                                                  // #warp, 576]
    using gl_kv =
        hk::gl<kv_t, -1, kPageSize, kKvNumHead, kQkHeadDim>; // [#page, page_size, #head_kv, 576]
    using gl_o    = hk::gl<out_t, 1, -1, kQoNumHead, kVoHeadDim>; // [1, #batch*#seqlen, #head, 512]
    using gl_so   = hk::gl<float, 1, -1, kQoNumHead, kVoHeadDim>; // [1, #partial_slots, #head, 512]
    using gl_slse = hk::gl<float, 1, -1, kQoNumHead, 1>;          // [1, #partial_slots, #head, 1]
    // lds tiles
    static_assert(std::is_same_v<kv_t, hk::bf16> || std::is_same_v<kv_t, hk::fp8e4m3>);
    using st_kv = std::conditional_t<std::is_same_v<kv_t, hk::fp8e4m3>,
                                     hk::st_fp8e4m3<kBlockN, kKvLoraRank, hk::st_16x16_s>,
                                     hk::st_bf<kBlockN, kKvLoraRank, hk::st_16x16_s>>;
};

template <typename Traits>
struct HkMlaDecodeFwdParams
{
    // inputs
    Traits::gl_q query;
    Traits::gl_kv kv_buffer;

    // outputs
    Traits::gl_o final_output;
    Traits::gl_so split_output;
    Traits::gl_slse split_lse;

    // metadata
    const int32_t* p_work_indptr;
    const int32_t* p_work_info_set;
};

template <typename T>
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy)
    __attribute__((amdgpu_num_vgpr(64))) void kn_ml_decode_fwd_n128(HkMlaDecodeFwdParams<T> params)
{
    using q_t    = T::q_t;
    using kv_t   = T::kv_t;
    using out_t  = T::out_t;
    using comp_t = float;

    using G = hk::group<T::kNumWarps>;

    const int32_t worker_idx     = blockIdx.x;
    const int32_t work_start_idx = __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx]);
    const int32_t work_end_idx =
        __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx + 1]);
    if(work_start_idx >= work_end_idx)
    {
        return;
    }

    // LDS tiles
    extern __shared__ int32_t p_lds[];
    hk::shared_allocator al(p_lds);
    auto lds_k = al.allocate<typename T::st_kv>();

    // Reg tiles
    using q_nope_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<92, 123>>, 2>; // 32 agprs
    using q_rope_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<124, 127>>, 2>; // 4 agprs
    using o_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<128, 255>>, 4>; // 128 vgprs
    hkdart::clobber<q_nope_ranges>();
    hkdart::clobber<q_rope_ranges>();
    hkdart::clobber<o_ranges>();

    hk::art<q_t, T::kTileM, T::kQkNopeHeadDim, hk::row_l, hk::rt_16x32_s, q_nope_ranges> q_nope;
    hk::art<q_t, T::kTileM, T::kQkRopeHeadDim, hk::row_l, hk::rt_16x32_s, q_rope_ranges> q_rope;
    hk::art<comp_t, T::kTileM, T::kVoHeadDim, hk::row_l, hk::rt_16x16_s, o_ranges> oaccu;

    // Runtime constants
    const int32_t warp_idx = ckt::get_warp_id();

    for(int32_t work_idx = work_start_idx; work_idx < work_end_idx; ++work_idx)
    {
        const int32_t partial_qo_loc = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 1]);
        const int32_t qo_start = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 2]);
        const int32_t qo_end = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 3]);
        const int32_t kv_start = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 4]);
        const int32_t kv_end = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 5]);

        ///
        /// Load Q from VRAM to GPRs
        ///
        hk::load<2>(q_nope, params.query, {qo_start, 0, 0, 0}, {0, warp_idx, 0, 0});
        hk::load<2>(
            q_rope, params.query, {qo_start, 0, 0, 0}, {0, warp_idx, 0, 0}, T::kQkNopeHeadDim);

        if((threadIdx.x % 64) < 3)
        {
            float4 nope_92  = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<92>()});
            float4 nope_93  = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<93>()});
            float4 nope_96  = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<96>()});
            float4 nope_100 = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<100>()});
            float4 rope_124 = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<124>()});
            float4 rope_125 = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<125>()});
            float4 rope_126 = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<126>()});
            float4 rope_127 = convert_fp8x4_to_float4(FUI{hkm::v_get_gpr<127>()});

            printf("[mla-dbg][%d, %d] %d=(%f,%f,%f,%f), %d=(%f,%f,%f,%f), %d=(%f,%f,%f,%f), "
                   "%d=(%f,%f,%f,%f)\n",
                   blockIdx.x,
                   threadIdx.x,
                   reg_2_loc_q<92>(),
                   nope_92.x,
                   nope_92.y,
                   nope_92.z,
                   nope_92.w,
                   reg_2_loc_q<93>(),
                   nope_93.x,
                   nope_93.y,
                   nope_93.z,
                   nope_93.w,
                   reg_2_loc_q<96>(),
                   nope_96.x,
                   nope_96.y,
                   nope_96.z,
                   nope_96.w,
                   reg_2_loc_q<100>(),
                   nope_100.x,
                   nope_100.y,
                   nope_100.z,
                   nope_100.w);
            printf("[mla-dbg][%d, %d] %d=(%f,%f,%f,%f), %d=(%f,%f,%f,%f), %d=(%f,%f,%f,%f), "
                   "%d=(%f,%f,%f,%f)\n",
                   blockIdx.x,
                   threadIdx.x,
                   reg_2_loc_q<124>(),
                   rope_124.x,
                   rope_124.y,
                   rope_124.z,
                   rope_124.w,
                   reg_2_loc_q<125>(),
                   rope_125.x,
                   rope_125.y,
                   rope_125.z,
                   rope_125.w,
                   reg_2_loc_q<126>(),
                   rope_126.x,
                   rope_126.y,
                   rope_126.z,
                   rope_126.w,
                   reg_2_loc_q<127>(),
                   rope_127.x,
                   rope_127.y,
                   rope_127.z,
                   rope_127.w);
        }

        ///
        /// Outputs
        ///
        if(partial_qo_loc < 0) {}
        else {}
    }
}

template <typename Traits>
void dispatch_mla_decode_fwd_n128(torch::Tensor& query,
                                  torch::Tensor& kv_buffer,
                                  const torch::Tensor& qo_indptr,
                                  const torch::Tensor& kv_indptr,
                                  const torch::Tensor& kv_page_indices,
                                  const torch::Tensor& kv_last_page_lens,
                                  const torch::Tensor& work_indptr,
                                  const torch::Tensor& work_info_set,
                                  const int max_seqlen_q,
                                  const float softmax_scale,
                                  torch::Tensor& split_output,
                                  torch::Tensor& split_lse,
                                  torch::Tensor& final_output)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    HkMlaDecodeFwdParams<Traits> params = {
        hk::make_gl<typename Traits::gl_q>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
            query.size(0),
            Traits::kNumTilesM,
            Traits::kTileM,
            Traits::kQkHeadDim),
        hk::make_gl<typename Traits::gl_kv>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer.data_ptr())),
            kv_buffer.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkHeadDim),
        hk::make_gl<typename Traits::gl_o>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(final_output.data_ptr())),
            1,
            final_output.size(0),
            Traits::kQoNumHead,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_so>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_output.data_ptr())),
            1,
            split_output.size(0),
            Traits::kQoNumHead,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_slse>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_lse.data_ptr())),
            1,
            split_lse.size(0),
            Traits::kQoNumHead,
            1),
        // metadata
        work_indptr.data_ptr<int32_t>(),
        work_info_set.data_ptr<int32_t>()};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_ml_decode_fwd_n128<Traits><<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi35x_mla_decode_fwd_n128(torch::Tensor& query,
                                  torch::Tensor& kv_buffer,
                                  const torch::Tensor& qo_indptr,
                                  const torch::Tensor& kv_indptr,
                                  const torch::Tensor& kv_page_indices,
                                  const torch::Tensor& kv_last_page_lens,
                                  const torch::Tensor& work_indptr,
                                  const torch::Tensor& work_info_set,
                                  const int max_seqlen_q,
                                  const float softmax_scale,
                                  torch::Tensor& split_output,
                                  torch::Tensor& split_lse,
                                  torch::Tensor& final_output)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(final_output));

    const bool q_is_fp8 = (query.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                          (query.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool kv_is_fp8 = (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                           (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool q_is_bf16  = (query.scalar_type() == at::ScalarType::BFloat16);
    const bool kv_is_bf16 = (kv_buffer.scalar_type() == at::ScalarType::BFloat16);

    if(q_is_fp8 && kv_is_fp8)
    {
        using Traits = HkMlaDecodeFwdTraits<hk::fp8e4m3, hk::fp8e4m3, hk::bf16, 128>;
        dispatch_mla_decode_fwd_n128<Traits>(query,
                                             kv_buffer,
                                             qo_indptr,
                                             kv_indptr,
                                             kv_page_indices,
                                             kv_last_page_lens,
                                             work_indptr,
                                             work_info_set,
                                             max_seqlen_q,
                                             softmax_scale,
                                             split_output,
                                             split_lse,
                                             final_output);
    }
    // else if(q_is_bf16 && kv_is_bf16)
    // {
    //     using Traits = HkMlaDecodeFwdTraits<hk::bf16, hk::bf16, hk::bf16, 128>;
    //     dispatch_mla_decode_fwd_n128<Traits>(query,
    //                                          kv_buffer,
    //                                          qo_indptr,
    //                                          kv_indptr,
    //                                          kv_page_indices,
    //                                          kv_last_page_lens,
    //                                          work_indptr,
    //                                          work_info_set,
    //                                          max_seqlen_q,
    //                                          softmax_scale,
    //                                          split_output,
    //                                          split_lse,
    //                                          final_output);
    // }
    else
    {
        TORCH_CHECK(false,
                    "hk_mi35x_mla_decode_fwd_n128 doesn't support q type ",
                    toString(query.scalar_type()),
                    " and kv type",
                    toString(kv_buffer.scalar_type()),
                    ".");
    }
}

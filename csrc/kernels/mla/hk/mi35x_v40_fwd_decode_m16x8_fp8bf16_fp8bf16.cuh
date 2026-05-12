// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "hk_mla_buffer_managers.cuh"
#include "hk_mla_softmax.cuh"
#include "mla.h"
#include <assert.h>

using namespace hk_mla;

// V4.0 mi35x m16x8 decode kernel: separate FP8 NOPE + BF16 ROPE buffers for
// both Q and KV. Device code intentionally left empty -- this file currently
// provides only the host wiring (Traits/Params/launcher/dispatcher) so the
// Python entry point `aiter.hk_mla_v40_decode_fwd` can be exercised end-to-end
// against the silver reference while the kernel body is being developed.
template <typename T>
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy) void
kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(HkMlaV40DecodeFwdParams<T> params)
{
    // TODO(v4.0): implement device code (QK gemm with FP8 NOPE + BF16 ROPE,
    // softmax, PV gemm, split-lse output).
    (void)params;
    assert(false);
}

template <typename Traits>
void mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(aiter_tensor_t& query,
                                                    aiter_tensor_t& query_rope,
                                                    aiter_tensor_t& kv_buffer,
                                                    aiter_tensor_t& kv_buffer_rope,
                                                    const aiter_tensor_t& qo_indptr,
                                                    const aiter_tensor_t& kv_indptr,
                                                    const aiter_tensor_t& kv_page_indices,
                                                    const aiter_tensor_t& kv_last_page_lens,
                                                    const aiter_tensor_t& work_indptr,
                                                    const aiter_tensor_t& work_info_set,
                                                    const int max_seqlen_q,
                                                    const float softmax_scale,
                                                    aiter_tensor_t& split_output,
                                                    aiter_tensor_t& split_lse,
                                                    aiter_tensor_t& final_output)
{
    const int32_t num_qheads = query.size(1);
    AITER_CHECK((num_qheads & (num_qheads - 1)) == 0 && num_qheads >= 16 && num_qheads <= 128,
                "num_qheads must be a power of 2 in [16, 128], got ",
                num_qheads);
    AITER_CHECK(num_qheads * max_seqlen_q == Traits::kBlockM,
                "num_qheads * max_seqlen_q must equal ",
                Traits::kBlockM,
                ", got ",
                num_qheads,
                " * ",
                max_seqlen_q,
                " = ",
                num_qheads * max_seqlen_q);
    const int32_t log2_num_qheads = __builtin_ctz(num_qheads);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = aiter::getCurrentHIPStream();

    HkMlaV40DecodeFwdParams<Traits> params = {
        hk::make_gl<typename Traits::gl_q_nope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
            query.size(0),
            num_qheads / Traits::kTileM,
            Traits::kTileM,
            Traits::kQkPackedNopeQElems),
        hk::make_gl<typename Traits::gl_q_rope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query_rope.data_ptr())),
            query_rope.size(0),
            num_qheads / Traits::kTileM,
            Traits::kTileM,
            Traits::kQkRopeHeadDim),
        hk::make_gl<typename Traits::gl_kv_nope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer.data_ptr())),
            kv_buffer.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkPackedNopeKvElems),
        hk::make_gl<typename Traits::gl_kv_rope>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer_rope.data_ptr())),
            kv_buffer_rope.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkRopeHeadDim),
        // kv_indices
        reinterpret_cast<int32_t*>(kv_page_indices.data_ptr()),
        // kv_last_page_lens (only read by kernel when kPageSize > 1)
        reinterpret_cast<int32_t*>(kv_last_page_lens.data_ptr()),
        // metadata
        reinterpret_cast<int32_t*>(work_indptr.data_ptr()),
        reinterpret_cast<int32_t*>(work_info_set.data_ptr()),
        hk::make_gl<typename Traits::gl_o>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(final_output.data_ptr())),
            1,
            final_output.size(0),
            Traits::kBlockM,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_so>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_output.data_ptr())),
            1,
            split_output.size(0),
            Traits::kBlockM,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_slse>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_lse.data_ptr())),
            1,
            split_lse.size(0),
            Traits::kBlockM,
            1),
        // parameters
        softmax_scale,
        log2_num_qheads};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16<Traits>
        <<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16(aiter_tensor_t& query,
                                                       aiter_tensor_t& query_rope,
                                                       aiter_tensor_t& kv_buffer,
                                                       aiter_tensor_t& kv_buffer_rope,
                                                       const aiter_tensor_t& qo_indptr,
                                                       const aiter_tensor_t& kv_indptr,
                                                       const aiter_tensor_t& kv_page_indices,
                                                       const aiter_tensor_t& kv_last_page_lens,
                                                       const aiter_tensor_t& work_indptr,
                                                       const aiter_tensor_t& work_info_set,
                                                       const int max_seqlen_q,
                                                       const float softmax_scale,
                                                       aiter_tensor_t& split_output,
                                                       aiter_tensor_t& split_lse,
                                                       aiter_tensor_t& final_output)
{
    HipDeviceGuard device_guard(final_output.device_id);

    const bool q_nope_is_fp8  = (query.dtype() == AITER_DTYPE_fp8);
    const bool kv_nope_is_fp8 = (kv_buffer.dtype() == AITER_DTYPE_fp8);
    const bool q_rope_is_bf16  = (query_rope.dtype() == AITER_DTYPE_bf16);
    const bool kv_rope_is_bf16 = (kv_buffer_rope.dtype() == AITER_DTYPE_bf16);

    AITER_CHECK(q_nope_is_fp8 && kv_nope_is_fp8,
                "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16 requires FP8 NOPE; got q=",
                AiterDtype_to_str(query.dtype()),
                ", kv=",
                AiterDtype_to_str(kv_buffer.dtype()));
    AITER_CHECK(q_rope_is_bf16 && kv_rope_is_bf16,
                "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16 requires BF16 ROPE; got q_rope=",
                AiterDtype_to_str(query_rope.dtype()),
                ", kv_rope=",
                AiterDtype_to_str(kv_buffer_rope.dtype()));

    const int32_t page_size = kv_buffer.size(1);

#define DISPATCH_PAGE_SIZE(PageSize)                                                 \
    case PageSize: {                                                                 \
        using Traits = HkMlaV40DecodeFwdTraits<hk::fp8e4m3,                          \
                                               hk::bf16,                             \
                                               hk::fp8e4m3,                          \
                                               hk::bf16,                             \
                                               hk::bf16,                             \
                                               /*kBlockN_=*/64,                      \
                                               /*kNumWarps_=*/8,                     \
                                               /*kOccupancy_=*/1,                    \
                                               /*kBlockM_=*/128,                     \
                                               /*kPageSize_=*/PageSize>;             \
        mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16<Traits>(query,                \
                                                              query_rope,            \
                                                              kv_buffer,             \
                                                              kv_buffer_rope,        \
                                                              qo_indptr,             \
                                                              kv_indptr,             \
                                                              kv_page_indices,       \
                                                              kv_last_page_lens,     \
                                                              work_indptr,           \
                                                              work_info_set,         \
                                                              max_seqlen_q,          \
                                                              softmax_scale,         \
                                                              split_output,          \
                                                              split_lse,             \
                                                              final_output);         \
        break;                                                                       \
    }

    // Only page_size in {1, 64} are instantiated -- same pattern as v32.
    switch(page_size)
    {
        DISPATCH_PAGE_SIZE(1)
        DISPATCH_PAGE_SIZE(64)
    default:
        AITER_CHECK(false,
                    "hk_mi35x_mla_v40_fwd_decode_m16x8_fp8bf16_fp8bf16: unsupported page_size ",
                    page_size,
                    " (supported: 1, 64).");
    }

#undef DISPATCH_PAGE_SIZE
}

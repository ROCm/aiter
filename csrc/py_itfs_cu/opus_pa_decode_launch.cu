// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host launcher for Opus PA decode — maps aiter_tensor_t to pa_decode_kargs and launches
// pa_decode_kernel<pa_default_traits>. Pattern: opus_gdn_chunk_prepare_launch.cu (PR #4016).
#include "opus_pa_decode.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "opus_pa/pa_decode_defs.h"

#include <cmath>

// Definition in opus_pa_decode.cu (separate TU avoids ODR issues).
template<typename Traits>
__global__ void pa_decode_kernel(pa_decode_kargs kargs);

namespace {

const float kLog2E = log2f(expf(1.f));

pa_decode_kargs make_pa_decode_kargs(int batch,
                                   int kv_nheads,
                                   int gqa,
                                   int head_dim,
                                   int max_blks_per_seq,
                                   void* dev_o,
                                   void* dev_q,
                                   void* dev_k,
                                   void* dev_v,
                                   void* dev_bt,
                                   void* dev_cl,
                                   void* dev_kq,
                                   void* dev_vq,
                                   void* dev_qtp,
                                   int stride_q_bytes,
                                   int stride_blk_bytes,
                                   int stride_kvhead_bytes,
                                   float scale_log2e)
{
    pa_decode_kargs args{};
    args.ptr_O          = dev_o;
    args.ptr_Q          = dev_q;
    args.ptr_K          = dev_k;
    args.ptr_V          = dev_v;
    args.ptr_BT         = dev_bt;
    args.ptr_CL         = dev_cl;
    args.ptr_KQ         = dev_kq;
    args.ptr_VQ         = dev_vq;
    args.ptr_QTP        = dev_qtp;
    args.ptr_DBG        = nullptr;
    args.scale_log2e    = scale_log2e;
    args.max_blks       = static_cast<uint32_t>(max_blks_per_seq);
    args.kv_nheads      = static_cast<uint32_t>(kv_nheads);
    args.stride_Q       = static_cast<uint32_t>(stride_q_bytes);
    args.stride_blk     = static_cast<uint32_t>(stride_blk_bytes);
    args.stride_kvhead  = static_cast<uint32_t>(stride_kvhead_bytes);
    args.mtp            = 0;
    args.gqa_ratio      = static_cast<uint32_t>(gqa);
    (void)batch;
    (void)head_dim;
    return args;
}

bool pa_opus_config_supported(AiterDtype q_dtype, AiterDtype kv_dtype, int head_size, int block_size, int gqa)
{
    if(head_size != 128 || block_size != 16)
        return false;
    if(q_dtype != AITER_DTYPE_bf16)
        return false;
    if(kv_dtype != AITER_DTYPE_fp8)
        return false;
    if(gqa != 8 && gqa != 1)
        return false;
    return true;
}

} // namespace

void pa_opus_fwd(aiter_tensor_t& Q,
                 aiter_tensor_t& K,
                 aiter_tensor_t& V,
                 aiter_tensor_t& block_tables,
                 aiter_tensor_t& context_lens,
                 int block_tables_stride0,
                 int max_qlen,
                 std::optional<aiter_tensor_t> K_QScale,
                 std::optional<aiter_tensor_t> V_QScale,
                 std::optional<aiter_tensor_t> out_,
                 std::optional<aiter_tensor_t> qo_indptr,
                 int high_precision)
{
    (void)max_qlen;
    (void)high_precision;

    AITER_CHECK(out_.has_value(), "pa_opus_fwd: out_ is required");
    aiter_tensor_t& out_ref = out_.value();

    const int batch        = static_cast<int>(context_lens.size(0));
    const int num_heads    = static_cast<int>(Q.size(1));
    const int head_size    = static_cast<int>(Q.size(2));
    const int num_kv_heads = static_cast<int>(K.size(1));
    const int block_size   = static_cast<int>(K.size(3));
    const int gqa_ratio    = num_heads / num_kv_heads;

    AITER_CHECK(num_heads % num_kv_heads == 0,
                "pa_opus_fwd: num_heads must be divisible by num_kv_heads");
    AITER_CHECK(out_ref.dtype() == Q.dtype(), "pa_opus_fwd: out dtype must match Q");
    AITER_CHECK(pa_opus_config_supported(Q.dtype(), K.dtype(), head_size, block_size, gqa_ratio),
                "pa_opus_fwd: unsupported config (need bf16 Q, fp8 KV, head=128, block=16, gqa=1|8)");

    const int stride_Q       = static_cast<int>(Q.stride(0) * Q.element_size());
    const int stride_KV_head = static_cast<int>(K.stride(1) * K.element_size());
    const int stride_KV_blk  = static_cast<int>(K.stride(0) * K.element_size());
    const float scale_log2e  = static_cast<float>(
        static_cast<double>(kLog2E) / std::sqrt(static_cast<double>(head_size)));

    pa_decode_kargs kargs = make_pa_decode_kargs(
        batch,
        num_kv_heads,
        gqa_ratio,
        head_size,
        block_tables_stride0,
        out_ref.data_ptr(),
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        block_tables.data_ptr(),
        context_lens.data_ptr(),
        K_QScale.has_value() ? K_QScale.value().data_ptr() : nullptr,
        V_QScale.has_value() ? V_QScale.value().data_ptr() : nullptr,
        qo_indptr.has_value() ? qo_indptr.value().data_ptr() : nullptr,
        stride_Q,
        stride_KV_blk,
        stride_KV_head,
        scale_log2e);

    HipDeviceGuard device_guard(Q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    constexpr int kBlockThreads = pa_default_traits::BLOCK_THREADS;
    dim3 grid(static_cast<unsigned>(num_kv_heads), static_cast<unsigned>(batch), 1u);
    dim3 block(kBlockThreads, 1u, 1u);

    hipLaunchKernelGGL(HIP_KERNEL_NAME((pa_decode_kernel<pa_default_traits>)),
                       grid,
                       block,
                       0,
                       stream,
                       kargs);
}

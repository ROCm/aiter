// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Host entry points for the gfx950 HipKittens-style MLA target-verify kernel
// (QLEN 8, 12 heads, latent 576 / value 512, FP8 e4m3 KV, page size 1).

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "hk/mi35x_k3_mla_verify_hk.cuh"
#include "mla_hk.h"

namespace {

void check_out(const aiter_tensor_t& out)
{
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
    AITER_CHECK(out.ndim == 3 && out.size(1) == k3hk::kH && out.size(2) == k3hk::kDV &&
                    out.stride(2) == 1,
                "out must be [bs*8, 12, 512] with a contiguous last dim");
    AITER_CHECK(out.size(0) % k3hk::kQLen == 0, "token count must be a multiple of 8");
}

void check_parts(const aiter_tensor_t& o_part,
                 const aiter_tensor_t& lse_part,
                 int64_t bs,
                 int64_t nsplit)
{
    AITER_CHECK(o_part.dtype() == AITER_DTYPE_bf16 && o_part.is_contiguous() &&
                    o_part.numel() >= static_cast<size_t>(bs * nsplit * k3hk::kRows * k3hk::kDV),
                "o_part must be contiguous bf16 [bs, nsplit, 96, 512]");
    AITER_CHECK(lse_part.dtype() == AITER_DTYPE_fp32 && lse_part.is_contiguous() &&
                    lse_part.numel() >= static_cast<size_t>(bs * nsplit * k3hk::kRows),
                "lse_part must be contiguous fp32 [bs, nsplit, 96]");
}

} // namespace

void hk_mla_verify_fwd(const aiter_tensor_t& q,
                       const aiter_tensor_t& kv,
                       const aiter_tensor_t& kv_indptr,
                       const aiter_tensor_t& kv_indices,
                       const aiter_tensor_t& kv_scale,
                       aiter_tensor_t& o_part,
                       aiter_tensor_t& lse_part,
                       aiter_tensor_t& out,
                       const int nsplit,
                       const int min_chunk,
                       const float sm_scale_log2,
                       const int part8_min)
{
    check_out(out);
    const int64_t ntok = q.size(0);
    const int64_t bs   = ntok / k3hk::kQLen;
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && q.ndim == 3 && q.size(1) == k3hk::kH &&
                    q.size(2) == k3hk::kD && q.stride(2) == 1,
                "q must be bf16 [bs*8, 12, 576] with a contiguous last dim");
    AITER_CHECK(ntok == out.size(0), "q and out token counts differ");
    AITER_CHECK(kv.size(-1) == k3hk::kD && kv.stride(-1) == 1 && kv.element_size() == 1,
                "kv must be fp8 [N, ..., 576] with a contiguous last dim");
    AITER_CHECK(kv_indptr.dtype() == AITER_DTYPE_i32 && kv_indices.dtype() == AITER_DTYPE_i32,
                "kv_indptr / kv_indices must be int32");
    AITER_CHECK(kv_indptr.numel() >= static_cast<size_t>(bs + 1), "kv_indptr must hold bs + 1");
    AITER_CHECK(kv_scale.dtype() == AITER_DTYPE_fp32 && kv_scale.numel() >= 1,
                "kv_scale must be fp32 [1]");
    AITER_CHECK(nsplit >= 1 && min_chunk > 0 && min_chunk % k3hk::kBN == 0, "bad split config");
    if(nsplit > 1)
        check_parts(o_part, lse_part, bs, nsplit);
    if(bs == 0)
        return;

    k3hk::Params p{};
    p.q            = static_cast<const uint16_t*>(q.data_ptr());
    p.kv           = static_cast<const uint8_t*>(kv.data_ptr());
    p.kv_indptr    = static_cast<const int32_t*>(kv_indptr.data_ptr());
    p.kv_indices   = static_cast<const int32_t*>(kv_indices.data_ptr());
    p.kv_scale     = static_cast<const float*>(kv_scale.data_ptr());
    p.o_part       = static_cast<uint16_t*>(o_part.data_ptr());
    p.lse_part     = static_cast<float*>(lse_part.data_ptr());
    p.o_final      = static_cast<uint16_t*>(out.data_ptr());
    p.stride_q_tok = q.stride(0);
    p.stride_q_h   = q.stride(1);
    p.stride_o_tok = out.stride(0);
    p.stride_o_h   = out.stride(1);
    const int64_t kv_rows   = kv.size(0);
    const int64_t kv_stride = kv.stride(0);
    const int64_t kv_bytes  = (kv_rows - 1) * kv_stride + k3hk::kD;
    AITER_CHECK(kv_bytes <= 0xFFFFFFFFll, "kv pool > 4 GiB is not supported by the buffer path");
    p.kv_stride     = static_cast<int32_t>(kv_stride);
    p.kv_bytes      = static_cast<uint32_t>(kv_bytes);
    p.nsplit        = nsplit;
    p.min_chunk     = min_chunk;
    p.sm_scale_log2 = sm_scale_log2;
    p.q_lds = (q.stride(1) == k3hk::kD && q.stride(0) == k3hk::kH * k3hk::kD) ? 1 : 0;
    p.bs        = static_cast<int32_t>(bs);
    p.part8_min = part8_min;

    HipDeviceGuard guard(q.device_id);
    hipLaunchKernelGGL(k3hk::k3_mla_verify_hk_kernel,
                       dim3(static_cast<uint32_t>(nsplit * bs)),
                       dim3(k3hk::kThreads),
                       0,
                       aiter::getCurrentHIPStream(),
                       p);
}

void hk_mla_verify_reduce(const aiter_tensor_t& o_part,
                          const aiter_tensor_t& lse_part,
                          const aiter_tensor_t& kv_indptr,
                          const aiter_tensor_t& kv_scale,
                          aiter_tensor_t& out,
                          const int nsplit,
                          const int min_chunk,
                          const int part8_min)
{
    check_out(out);
    const int64_t bs = out.size(0) / k3hk::kQLen;
    AITER_CHECK(nsplit >= 1 && nsplit <= 256, "nsplit must be in [1, 256]");
    AITER_CHECK(min_chunk > 0 && min_chunk % k3hk::kBN == 0, "bad split config");
    AITER_CHECK(kv_indptr.dtype() == AITER_DTYPE_i32 &&
                    kv_indptr.numel() >= static_cast<size_t>(bs + 1),
                "kv_indptr must be int32 [bs + 1]");
    AITER_CHECK(kv_scale.dtype() == AITER_DTYPE_fp32 && kv_scale.numel() >= 1,
                "kv_scale must be fp32 [1]");
    check_parts(o_part, lse_part, bs, nsplit);
    if(bs == 0)
        return;

    k3hk::Params p{};
    p.kv_indptr    = static_cast<const int32_t*>(kv_indptr.data_ptr());
    p.o_part       = static_cast<uint16_t*>(o_part.data_ptr());
    p.lse_part     = static_cast<float*>(lse_part.data_ptr());
    p.o_final      = static_cast<uint16_t*>(out.data_ptr());
    p.kv_scale     = static_cast<const float*>(kv_scale.data_ptr());
    p.stride_o_tok = out.stride(0);
    p.stride_o_h   = out.stride(1);
    p.nsplit       = nsplit;
    p.min_chunk    = min_chunk;
    p.bs           = static_cast<int32_t>(bs);
    p.part8_min    = part8_min;

    HipDeviceGuard guard(out.device_id);
    k3hk::launch_reduce1(p, static_cast<int>(bs), aiter::getCurrentHIPStream());
}

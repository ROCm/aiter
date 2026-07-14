// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based GQA flash attention (D=128). Host launcher + grid/kargs setup on
// top of the device kernel template in `fmha_fwd_hd128_bf16_opus.h` (single-header, IMPL-guarded).

#define FMHA_FWD_HD128_BF16_OPUS_IMPL
#include "fmha_fwd_hd128_bf16_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

#include <cmath>

void fmha_fwd_hd128_bf16_opus_fwd(aiter_tensor_t& q,
                  aiter_tensor_t& k,
                  aiter_tensor_t& v,
                  aiter_tensor_t& out,
                  bool causal,
                  float softmax_scale)
{
    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, N, H, D], got ndim=", q.dim());
    AITER_CHECK(k.dim() == 4, "k must be 4-D [B, N, H_KV, D], got ndim=", k.dim());
    AITER_CHECK(v.dim() == 4, "v must be 4-D [B, N, H_KV, D], got ndim=", v.dim());
    AITER_CHECK(out.dim() == 4, "out must be 4-D [B, N, H, D], got ndim=", out.dim());

    AITER_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype() &&
                    q.dtype() == out.dtype(),
                "q/k/v/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_hd128_bf16_opus_fwd only supports bf16 (D=128 kernel)");

    const int B    = static_cast<int>(q.size(0));
    const int N    = static_cast<int>(q.size(1));
    const int H    = static_cast<int>(q.size(2));
    const int D    = static_cast<int>(q.size(3));
    const int H_KV = static_cast<int>(k.size(2));

    AITER_CHECK(D == 128, "fmha_fwd_hd128_bf16_opus_fwd only compiles D=128, got D=", D);
    AITER_CHECK(k.size(0) == B && v.size(0) == B, "k/v batch must equal q batch B");
    AITER_CHECK(k.size(1) == N && v.size(1) == N, "k/v seqlen must equal q seqlen N");
    AITER_CHECK(v.size(2) == H_KV, "k/v must share H_KV");
    AITER_CHECK(k.size(3) == D && v.size(3) == D, "k/v head dim must equal D=128");
    AITER_CHECK(H_KV > 0 && (H % H_KV) == 0, "H must be divisible by H_KV (GQA group)");
    AITER_CHECK(out.size(0) == B && out.size(1) == N && out.size(2) == H && out.size(3) == D,
                "out shape must match q [B, N, H, D]");

    // Row-major contiguous along the head dim D is required by the kernel layouts.
    AITER_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 && out.stride(3) == 1,
                "q/k/v/out must be contiguous along the head dim D");

    if (B == 0 || N == 0 || H == 0) return;

    // ---- Build kernel args -----------------------------------------------
    opus_gqa_kargs kargs{};
    kargs.ptr_q = q.data_ptr();
    kargs.ptr_k = k.data_ptr();
    kargs.ptr_v = v.data_ptr();
    kargs.ptr_o = out.data_ptr();
    kargs.B     = B;
    kargs.N     = N;
    kargs.H     = H;
    kargs.H_KV  = H_KV;
    kargs.D     = D;
    kargs.stride_q_b  = static_cast<int>(q.stride(0));
    kargs.stride_q_n  = static_cast<int>(q.stride(1));
    kargs.stride_q_h  = static_cast<int>(q.stride(2));
    kargs.stride_kv_b = static_cast<int>(k.stride(0));
    kargs.stride_kv_n = static_cast<int>(k.stride(1));
    kargs.stride_kv_h = static_cast<int>(k.stride(2));

    if (softmax_scale <= 0.0f) {
        softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    }
    kargs.softmax_scale = softmax_scale;  // kernel applies scale * log2(e) to Q

    // ---- Launch ----------------------------------------------------------
    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    using TraitsCausal    = opus_gqa_traits<32, 64, 128, 8, true>;
    using TraitsNonCausal = opus_gqa_traits<32, 64, 128, 8, false>;

    auto launch = [&](auto traits_tag) {
        using Traits          = decltype(traits_tag);
        const int q_block     = Traits::NUM_WARPS * Traits::Q_TILE_SIZE;  // 256
        const int num_q_tiles = ceil_div(N, Traits::Q_TILE_SIZE);
        const int num_q_blk   = ceil_div(num_q_tiles, Traits::NUM_WARPS);
        (void)q_block;
        dim3 grid(H, num_q_blk, B);
        dim3 block(Traits::BLOCK_SIZE);
        gqa_d128_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    if (causal) {
        launch(TraitsCausal{});
    } else {
        launch(TraitsNonCausal{});
    }
}

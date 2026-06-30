// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host launcher for gdn_chunk_prepare — the fused intra-chunk GDN prefill prep
// (chunk_local_cumsum + chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd).
// Target: gfx942 (MI300X) / gfx950 (MI350).
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>
#include <cstring>

#include "opus_gdn/gdn_chunk_prepare_defs.h"

static inline int gdn_cp_ceil_div(int a, int b) { return (a + b - 1) / b; }

// gfx950 uses the LDS C⁻¹ path (OCC=2); gfx942 the register-cached path (OCC=3).
// Host can't see __gfx950__, so detect at runtime (JIT build is per-machine).
static inline bool gdn_cp_is_gfx950() {
    static int cached = -1;
    if (cached < 0) {
        hipDeviceProp_t p;
        cached = (hipGetDeviceProperties(&p, 0) == hipSuccess &&
                  std::strstr(p.gcnArchName, "gfx950") != nullptr) ? 1 : 0;
    }
    return cached != 0;
}

// Definition lives in opus_gdn_chunk_prepare.cu (separate TU to avoid ODR issues).
template<typename Traits>
__global__ void gdn_chunk_prepare_kernel(gdn_chunk_prepare_kargs kargs);

// Inputs:  k[B,T,H,K] bf16, v[B,T,H,V] bf16, g[B,T,H] fp32, beta[B,T,H] fp32
// Outputs: g_cumsum[B,T,H] fp32, w_bar[B,T,H,K] bf16, u_bar[B,T,H,V] bf16
void gdn_chunk_prepare_fwd(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor w_bar,
    torch::Tensor u_bar,
    torch::Tensor g_cumsum)
{
    using K1Traits = gdn_chunk_prepare_traits<64, 128, 128, 4>;
    constexpr int BT = K1Traits::BT;

    const int B = k.size(0);
    const int T = k.size(1);
    const int H = k.size(2);
    const int K = k.size(3);
    const int V = v.size(3);
    const int NT = gdn_cp_ceil_div(T, BT);

    gdn_chunk_prepare_kargs kargs{};
    kargs.ptr_k        = k.data_ptr();
    kargs.ptr_v        = v.data_ptr();
    kargs.ptr_beta     = beta.data_ptr();
    kargs.ptr_g        = g.data_ptr();
    kargs.ptr_w_bar    = w_bar.data_ptr();
    kargs.ptr_u_bar    = u_bar.data_ptr();
    kargs.ptr_g_cumsum = g_cumsum.data_ptr();
    kargs.B = B; kargs.T = T; kargs.H = H; kargs.K = K; kargs.V = V;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(k));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    dim3 grid(NT, B * H);
    dim3 block(K1Traits::BLOCK_SIZE);
    size_t smem = gdn_cp_is_gfx950()
                      ? K1Traits::smem_size_bytes_cinv_lds()
                      : K1Traits::smem_size_bytes();

    gdn_chunk_prepare_kernel<K1Traits><<<grid, block, smem, stream>>>(kargs);
}

// GDN Prefill K2-SCAN — PURE HIP, single-warp, register-resident H.
//
// Drop-in replacement for gdn_k2_scan_kernel that bypasses the opus
// tiled_gemm_mfma template entirely: raw __builtin_amdgcn_mfma_f32_16x16x16bf16_1k,
// H kept in VGPR accumulators and fed straight back as the MFMA B-operand
// (no LDS staging, no barriers, no cross-warp reduction). One wavefront (64
// lanes) owns one (head, v-tile) and runs the whole serial chunk scan.
//
// Validated layout (C = A @ B^T):
//   operand elem[i] = MAT[outer=lane&15, inner=(lane>>4)*4+i]
//   output  d[i]    = OUT[outer=(lane>>4)*4+i, inner=lane&15]
// -> H accumulator (d-layout) == B-operand for retrieve (W@H);
//    v_new (retrieve d-layout) == B-operand for h-update (k_gated^T@v_new).
//
// per chunk t:  snapshot H_{t-1};  retrieve=W@H;  v_new=u_bar-retrieve (ungated);
//               H = exp(g_last)*H + k_gated^T @ v_new,  k_gated[s,j]=k[s,j]*exp(g_last-g[s])
// Grid: (cdiv(V,BV), B*H)   Block: 64 (one warp).  BV<=32 (H = K/16 * BV/16 acc tiles).
#pragma once
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"

template<typename Traits>
__global__ void __launch_bounds__(64, 8)
gdn_k2_scan_hip_kernel(gdn_k2_kargs kargs) {
    using D_ATTN = typename Traits::D_ATTN;   // bf16_t
    using bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
    using f32x4  = float  __attribute__((ext_vector_type(4)));

    constexpr int BT  = Traits::BT;     // 64
    constexpr int BV  = Traits::BV;     // 32
    constexpr int W16 = 16;
    constexpr int KT  = 128 / W16;      // K tiles (K=128) = 8
    constexpr int MT  = BT  / W16;      // BT tiles = 4
    constexpr int NV  = BV  / W16;      // BV tiles = 2
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int NT = kargs.NT;
    const int T  = kargs.T;

    const int i_v  = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_n  = i_nh / H;
    const int i_h  = i_nh % H;
    const int lane = threadIdx.x;       // 0..63
    const int lo   = lane & 15;
    const int hi   = (lane >> 4) * 4;
    const int v_off = i_v * BV;
    const int bos   = i_n * T;
    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    const __bf16* w_hbm = reinterpret_cast<const __bf16*>(kargs.ptr_w_bar) + (bos * H + i_h) * K;
    const __bf16* u_hbm = reinterpret_cast<const __bf16*>(kargs.ptr_u_bar) + (bos * H + i_h) * V;
    const __bf16* k_hbm = reinterpret_cast<const __bf16*>(kargs.ptr_k)     + (bos * H + i_h) * K;
    const float*  g_hbm = reinterpret_cast<const float*>(kargs.ptr_g_cumsum) + bos * H + i_h;
    __bf16* vn_hbm = reinterpret_cast<__bf16*>(kargs.ptr_v_new) + (bos * H + i_h) * V;
    __bf16* hsnap  = reinterpret_cast<__bf16*>(kargs.ptr_h_snap)
                     + ((int64_t)i_n * NT * H + i_h) * V * K;   // + i_t*H*V*K per chunk

    f32x4 Hacc[KT][NV];
    #pragma unroll
    for (int kt = 0; kt < KT; kt++)
        #pragma unroll
        for (int nt = 0; nt < NV; nt++) Hacc[kt][nt] = f32x4{0,0,0,0};

    // initial state h0 [V,K] (fp32): h0[(v_off+n)*K + kk]
    if (kargs.ptr_h0 != nullptr) {
        const float* h0 = reinterpret_cast<const float*>(kargs.ptr_h0) + (i_n * H + i_h) * V * K;
        #pragma unroll
        for (int kt = 0; kt < KT; kt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++)
                for (int i = 0; i < 4; i++) {
                    int kk = kt*W16 + hi + i, n = nt*W16 + lo;
                    if (v_off + n < V) Hacc[kt][nt][i] = h0[(int64_t)(v_off + n) * K + kk];
                }
    }

    for (int it = 0; it < NT; it++) {
        const int t0 = it * BT;
        const int T_rem = T - t0;
        const bool full = (T_rem >= BT);
        const __bf16* w_ch = w_hbm + (int64_t)t0 * stride_k;
        const __bf16* u_ch = u_hbm + (int64_t)t0 * stride_v;
        const __bf16* k_ch = k_hbm + (int64_t)t0 * stride_k;
        const float*  g_ch = g_hbm + (int64_t)t0 * stride_g;
        __bf16* vn_ch = vn_hbm + (int64_t)t0 * stride_v;
        __bf16* hs_ch = hsnap  + (int64_t)it * H * V * K;
        int last = full ? (BT - 1) : (T_rem - 1);
        float g_last = g_ch[last * stride_g];

        // (a) snapshot H_{t-1} -> h_snap [BV,K]
        #pragma unroll
        for (int kt = 0; kt < KT; kt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++)
                for (int i = 0; i < 4; i++) {
                    int kk = kt*W16 + hi + i, n = nt*W16 + lo;
                    if (v_off + n < V) hs_ch[(int64_t)(v_off + n) * K + kk] = (__bf16)Hacc[kt][nt][i];
                }

        // (b) retrieve R[BT,BV] = W @ H
        f32x4 R[MT][NV];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++) {
                f32x4 acc = f32x4{0,0,0,0};
                #pragma unroll
                for (int kt = 0; kt < KT; kt++) {
                    bf16x4 a;
                    int row = mt*W16 + lo;
                    bool rok = full || row < T_rem;
                    for (int i = 0; i < 4; i++)
                        a[i] = rok ? w_ch[(int64_t)row * stride_k + kt*W16 + hi + i] : (__bf16)0;
                    bf16x4 b;
                    for (int i = 0; i < 4; i++) b[i] = (__bf16)Hacc[kt][nt][i];
                    acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc, 0, 0, 0);
                }
                R[mt][nt] = acc;
            }

        // (c) v_new = u_bar - R  (ungated); store + keep
        f32x4 Vn[MT][NV];
        #pragma unroll
        for (int mt = 0; mt < MT; mt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++)
                for (int i = 0; i < 4; i++) {
                    int t = mt*W16 + hi + i, v = nt*W16 + lo;
                    bool ok = (full || t < T_rem) && (v_off + v < V);
                    float u = ok ? (float)u_ch[(int64_t)t * stride_v + v_off + v] : 0.0f;
                    float vv = u - R[mt][nt][i];
                    Vn[mt][nt][i] = vv;
                    if (ok) vn_ch[(int64_t)t * stride_v + v_off + v] = (__bf16)vv;
                }

        // (d) decay H; H += k_gated^T @ v_new
        float decay = __expf(g_last);
        #pragma unroll
        for (int kt = 0; kt < KT; kt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++)
                for (int i = 0; i < 4; i++) Hacc[kt][nt][i] *= decay;

        #pragma unroll
        for (int kt = 0; kt < KT; kt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++) {
                #pragma unroll
                for (int tt = 0; tt < MT; tt++) {
                    bf16x4 a;
                    for (int i = 0; i < 4; i++) {
                        int t = tt*W16 + hi + i;
                        bool ok = full || t < T_rem;
                        float gate = ok ? __expf(g_last - g_ch[t * stride_g]) : 0.0f;
                        a[i] = ok ? (__bf16)((float)k_ch[(int64_t)t * stride_k + kt*W16 + lo] * gate) : (__bf16)0;
                    }
                    bf16x4 b;
                    for (int i = 0; i < 4; i++) b[i] = (__bf16)Vn[tt][nt][i];
                    Hacc[kt][nt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, Hacc[kt][nt], 0, 0, 0);
                }
            }
    }

    // epilogue: final state ht [V,K] fp32
    if (kargs.ptr_ht != nullptr) {
        float* ht = reinterpret_cast<float*>(kargs.ptr_ht) + (i_n * H + i_h) * V * K;
        #pragma unroll
        for (int kt = 0; kt < KT; kt++)
            #pragma unroll
            for (int nt = 0; nt < NV; nt++)
                for (int i = 0; i < 4; i++) {
                    int kk = kt*W16 + hi + i, n = nt*W16 + lo;
                    if (v_off + n < V) ht[(int64_t)(v_off + n) * K + kk] = Hacc[kt][nt][i];
                }
    }
}

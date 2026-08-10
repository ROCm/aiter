// GDN Prefill K2-SCAN — faithful triton fwd_h port: 32x32x16 MFMA, nw=2,
// num_stages=2 software pipeline (cross-chunk register prefetch of W+k).
//
// 32x32x16 MFMA (v_mfma_f32_32x32x16_bf16) = 4x output/instr vs 16x16.
// Latency hiding: each chunk's W,k are prefetched into registers during the
// PREVIOUS chunk's compute (num_stages=2), then installed into LDS — this is
// what lets triton hit ~418us at only ~0.5 WG/CU (grid-starved).
//
// Layout (validated, C=A@B^T): a[i]=A[m=lane%32,k=(lane/32)*8+i] (i<8);
//   d[i]=C[m=(i/4)*8+(lane/32)*4+(i%4), n=lane%32] (i<16).
// 2 warps/(head,v-tile). warp w owns H K-tiles {2w,2w+1}. H reg-resident;
// LDS for the acc->operand transpose (s_hT), v_new transpose (s_vT), and the
// staged W/k operands (s_W/s_k).  BV=32, K=128, BT=64.
#pragma once
#include <hip/hip_runtime.h>
#include "opus_gdn/gdn_defs.h"

template<typename Traits>
__global__ void __launch_bounds__(128, 2)
gdn_k2_scan32_kernel(gdn_k2_kargs kargs) {
// 32x32x16 bf16 MFMA needs gfx950-insts; stub on other arches so the TU still
// compiles (this is research-only, env-gated via OPUS_GDN_SCAN32, never default).
#if defined(__gfx950__)
    using b8   = __bf16 __attribute__((ext_vector_type(8)));
    using v8   = __bf16 __attribute__((ext_vector_type(8)));
    using f16v = float  __attribute__((ext_vector_type(16)));
    constexpr int BT = 64, BV = 32, K = 128;
    constexpr int KP = K + 8;                 // padded K stride (LDS bank conflicts)
    const int H = kargs.H, V = kargs.V, NT = kargs.NT, T = kargs.T;

    const int i_v  = blockIdx.x, i_nh = blockIdx.y;
    const int i_n  = i_nh / H, i_h = i_nh % H;
    const int tid  = threadIdx.x;
    const int warp = tid >> 6, lane = tid & 63;
    const int o    = lane & 31, hi = lane >> 5;
    const int v_off = i_v * BV;
    const int bos   = i_n * T;
    const int stride_k = H * K, stride_v = H * V, stride_g = H;

    const __bf16* w_hbm = reinterpret_cast<const __bf16*>(kargs.ptr_w_bar) + (bos * H + i_h) * K;
    const __bf16* u_hbm = reinterpret_cast<const __bf16*>(kargs.ptr_u_bar) + (bos * H + i_h) * V;
    const __bf16* k_hbm = reinterpret_cast<const __bf16*>(kargs.ptr_k)     + (bos * H + i_h) * K;
    const float*  g_hbm = reinterpret_cast<const float*>(kargs.ptr_g_cumsum) + bos * H + i_h;
    __bf16* vn_hbm = reinterpret_cast<__bf16*>(kargs.ptr_v_new) + (bos * H + i_h) * V;
    __bf16* hsnap  = reinterpret_cast<__bf16*>(kargs.ptr_h_snap) + ((int64_t)i_n * NT * H + i_h) * V * K;

    __shared__ __bf16 s_W[BT * KP];           // staged W[t][k]
    __shared__ __bf16 s_k[BT * KP];           // staged k[t][k]
    __shared__ __bf16 s_hT[BV * K];           // H^T [bv][k]
    __shared__ __bf16 s_vT[BV * BT];          // v_new^T [bv][t]
    __shared__ float  s_g[BT];

    // each of 128 threads streams BT*K/128 = 64 bf16 = 8 v8 of W and of k
    constexpr int NVEC = (BT * K) / (128 * 8);   // 8 v8-loads per thread
    v8 pf_w[NVEC], pf_k[NVEC];

    auto load_chunk = [&](int t0, int trem, bool full) {
        #pragma unroll
        for (int j = 0; j < NVEC; j++) {
            int e = (tid + j * 128) * 8;       // element offset in [BT*K]
            int row = e / K, col = e % K;
            v8 zw{}, zk{};
            if (full || row < trem) {
                zw = *reinterpret_cast<const v8*>(&w_hbm[(int64_t)(t0 + row) * stride_k + col]);
                zk = *reinterpret_cast<const v8*>(&k_hbm[(int64_t)(t0 + row) * stride_k + col]);
            }
            pf_w[j] = zw; pf_k[j] = zk;
        }
    };
    auto install = [&]() {
        #pragma unroll
        for (int j = 0; j < NVEC; j++) {
            int e = (tid + j * 128) * 8; int row = e / K, col = e % K;
            *reinterpret_cast<v8*>(&s_W[row * KP + col]) = pf_w[j];
            *reinterpret_cast<v8*>(&s_k[row * KP + col]) = pf_k[j];
        }
    };

    f16v Hacc[2];
    for (int j = 0; j < 2; j++) for (int i = 0; i < 16; i++) Hacc[j][i] = 0.0f;
    if (kargs.ptr_h0 != nullptr) {
        const float* h0 = reinterpret_cast<const float*>(kargs.ptr_h0) + (i_n * H + i_h) * V * K;
        for (int j = 0; j < 2; j++) { int kt = warp*2 + j;
            for (int i = 0; i < 16; i++) { int kk = kt*32+(i/4)*8+hi*4+(i%4); int bv=v_off+o;
                if (bv < V) Hacc[j][i] = h0[(int64_t)bv*K + kk]; } }
    }

    // prologue: prefetch chunk 0
    { int trem = T; bool full = (T >= BT); load_chunk(0, trem, full); }

    for (int it = 0; it < NT; it++) {
        const int t0 = it * BT, T_rem = T - t0;
        const bool full = (T_rem >= BT);
        __bf16* vn_ch = vn_hbm + (int64_t)t0 * stride_v;
        __bf16* hs_ch = hsnap + (int64_t)it * H * V * K;
        const float* g_ch = g_hbm + (int64_t)t0 * stride_g;

        for (int x = tid; x < BT; x += 128) s_g[x] = (x < T_rem) ? g_ch[(int64_t)x * stride_g] : 0.0f;
        install();                              // pf (this chunk) -> s_W/s_k
        __syncthreads();
        float g_last = s_g[full ? (BT-1) : (T_rem-1)];

        // prefetch NEXT chunk (overlaps this chunk's compute) — num_stages=2
        if (it + 1 < NT) { int n_t0=(it+1)*BT, n_rem=T-n_t0; load_chunk(n_t0, n_rem, n_rem>=BT); }

        // (a) snapshot + spill H -> s_hT
        for (int j = 0; j < 2; j++) { int kt = warp*2 + j;
            for (int i = 0; i < 16; i++) { int kk=kt*32+(i/4)*8+hi*4+(i%4); int bv=o;
                s_hT[bv*K + kk] = (__bf16)Hacc[j][i];
                if (v_off+bv < V) hs_ch[(int64_t)(v_off+bv)*K + kk] = (__bf16)Hacc[j][i]; } }
        __syncthreads();

        // (b) retrieve R[BT,BV]=W@H : warp w does BT M-tile w
        f16v R; for (int i=0;i<16;i++) R[i]=0.0f;
        { int mrow0 = warp*32;
          for (int kt2=0; kt2<K/16; kt2++) { int kb=kt2*16+hi*8;
            b8 a,b; for (int i=0;i<8;i++){ a[i]=s_W[(mrow0+o)*KP + kb+i]; b[i]=s_hT[o*K + kb+i]; }
            R = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a,b,R,0,0,0); } }
        // (c) v_new = u_bar - R ; store + spill s_vT
        { int mrow0 = warp*32;
          for (int i=0;i<16;i++){ int t=mrow0+(i/4)*8+hi*4+(i%4); int bv=o;
            bool ok=(full||t<T_rem)&&(v_off+bv<V);
            float u = ok ? (float)u_hbm[(int64_t)(t0+t)*stride_v + v_off+bv] : 0.0f;
            float vv = u - R[i];
            s_vT[bv*BT + t] = (__bf16)vv;
            if (ok) vn_ch[(int64_t)t*stride_v + v_off+bv] = (__bf16)vv; } }
        __syncthreads();

        // (d) decay; H += k_gated^T @ v_new : warp w its 2 K-tiles
        float decay = __expf(g_last);
        for (int j=0;j<2;j++) for (int i=0;i<16;i++) Hacc[j][i]*=decay;
        for (int j=0;j<2;j++){ int kt=warp*2+j;
          for (int tt=0; tt<BT/16; tt++){
            b8 a,b;
            for (int i=0;i<8;i++){ int t=tt*16+hi*8+i; int kk=kt*32+o;
              float gate = (full||t<T_rem) ? __expf(g_last - s_g[t]) : 0.0f;
              a[i]=(__bf16)((float)s_k[t*KP + kk]*gate);
              b[i]=s_vT[o*BT + t]; }
            Hacc[j]=__builtin_amdgcn_mfma_f32_32x32x16_bf16(a,b,Hacc[j],0,0,0); } }
        __syncthreads();
    }

    if (kargs.ptr_ht != nullptr) {
        float* ht = reinterpret_cast<float*>(kargs.ptr_ht) + (i_n*H + i_h)*V*K;
        for (int j=0;j<2;j++){ int kt=warp*2+j;
          for (int i=0;i<16;i++){ int kk=kt*32+(i/4)*8+hi*4+(i%4); int bv=v_off+o;
            if (bv<V) ht[(int64_t)bv*K + kk]=Hacc[j][i]; } }
    }
#else
    (void)kargs;  // 32x32x16 bf16 MFMA unavailable off gfx950
#endif
}

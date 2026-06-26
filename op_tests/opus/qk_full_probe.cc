// SPDX-License-Identifier: MIT
// Validate the full QK first-GEMM in isolation:
//   S[m,n] = NOPE(fp8 16x16x32 MFMA + per-64-tile sw scale)  +  ROPE(bf16 16x16x32 MFMA)
// vs a dequant CPU reference. This is the complete first GEMM the fused attention
// kernel will use ("mxfp8 nope, then bf16 mfma accumulate the rope part").
//
// Build (docker, gfx950):
//   hipcc -std=c++20 --offload-arch=gfx950 -O2 -I csrc/include op_tests/opus/qk_full_probe.cc -o /tmp/qkf && /tmp/qkf
#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#endif

using opus::fp8_t;
using opus::bf16_t;
using opus::fp32_t;
using fp8x8_t  = opus::fp8x8_t;
using bf16x8_t = opus::bf16x8_t;
using fp32x4_t = opus::vector_t<fp32_t, 4>;

constexpr int M = 16, N = 16, KD = 448, TILE = 64, NTILE = 7;
constexpr int RD = 64;  // rope dim (bf16)

__global__ void qk_full(const fp8_t*  __restrict__ q8,    // [M, KD] nope fp8
                        const fp8_t*  __restrict__ k8,    // [N, KD] nope fp8
                        const float*  __restrict__ sq,    // [M, NTILE]
                        const float*  __restrict__ sk,    // [N, NTILE]
                        const bf16_t* __restrict__ qr,    // [M, RD] rope bf16
                        const bf16_t* __restrict__ kr,    // [N, RD] rope bf16
                        float* __restrict__ S)            // [M, N]
{
#if defined(__gfx950__)
    using namespace opus;
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int m    = lane % 16;
    int kblk = lane / 16;     // 0..3
    int n    = lane % 16;
    float s[4] = {0.f, 0.f, 0.f, 0.f};

    // --- NOPE: fp8 MFMA + software per-64-tile scale ---
    for (int t = 0; t < NTILE; ++t) {
        fp32x4_t vc{0.f, 0.f, 0.f, 0.f};
        for (int kk = 0; kk < 2; ++kk) {
            int kbase = t * TILE + kk * 32;
            fp8x8_t a_reg, b_reg;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                a_reg[j] = q8[m * KD + kbase + kblk * 8 + j];
                b_reg[j] = k8[n * KD + kbase + kblk * 8 + j];
            }
            vc = mfma<fp8_t, fp8_t, fp32_t, 16, 16, 32>{}(a_reg, b_reg, vc);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
            s[i] += vc[i] * sq[(kblk * 4 + i) * NTILE + t] * sk[n * NTILE + t];
    }

    // --- ROPE: bf16 MFMA (no scale), accumulate into same S ---
    {
        fp32x4_t vc{0.f, 0.f, 0.f, 0.f};
        for (int kk = 0; kk < 2; ++kk) {       // 2 x 32 = 64 rope
            int kbase = kk * 32;
            bf16x8_t a_reg, b_reg;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                a_reg[j] = qr[m * RD + kbase + kblk * 8 + j];
                b_reg[j] = kr[n * RD + kbase + kblk * 8 + j];
            }
            vc = mfma<bf16_t, bf16_t, fp32_t, 16, 16, 32>{}(a_reg, b_reg, vc);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] += vc[i];
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) S[(kblk * 4 + i) * N + n] = s[i];
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
static float e4m3_decode(unsigned char b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, mant = b & 0x7;
    float v;
    if (e == 0) v = (float)mant / 8.0f * 0.015625f;
    else        v = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, e - 7);
    return s ? -v : v;
}
static unsigned char rand_fp8() {
    for (;;) {
        unsigned char b = (unsigned char)(rand() & 0x3F);
        int e = (b >> 3) & 0xF, mant = b & 0x7;
        if (e == 15 && mant == 7) continue;
        return b;
    }
}
static unsigned short to_bf16(float f) {
    unsigned int x; __builtin_memcpy(&x, &f, 4);
    unsigned int round = ((x >> 16) & 1u) + 0x7FFFu;
    return (unsigned short)(((x + round) >> 16) & 0xFFFFu);
}
static float from_bf16(unsigned short h) {
    unsigned int x = ((unsigned int)h) << 16; float f; __builtin_memcpy(&f, &x, 4); return f;
}

int main() {
    srand(1234);
    const int qn=M*KD, kn=N*KD, sqn=M*NTILE, skn=N*NTILE, qrn=M*RD, krn=N*RD;
    unsigned char *hq=new unsigned char[qn], *hk=new unsigned char[kn];
    float *hsq=new float[sqn], *hsk=new float[skn];
    unsigned short *hqr=new unsigned short[qrn], *hkr=new unsigned short[krn];
    for (int i=0;i<qn;++i) hq[i]=rand_fp8();
    for (int i=0;i<kn;++i) hk[i]=rand_fp8();
    for (int i=0;i<sqn;++i) hsq[i]=ldexpf(1.f,(rand()%5)-2);
    for (int i=0;i<skn;++i) hsk[i]=ldexpf(1.f,(rand()%5)-2);
    for (int i=0;i<qrn;++i) hqr[i]=to_bf16(((rand()%2001)-1000)/1000.f);
    for (int i=0;i<krn;++i) hkr[i]=to_bf16(((rand()%2001)-1000)/1000.f);

    float* ref=new float[M*N];
    for (int m=0;m<M;++m) for (int n=0;n<N;++n) {
        double acc=0;
        for (int t=0;t<NTILE;++t) {
            double part=0;
            for (int d=t*TILE; d<(t+1)*TILE; ++d)
                part += (double)e4m3_decode(hq[m*KD+d]) * (double)e4m3_decode(hk[n*KD+d]);
            acc += part * (double)hsq[m*NTILE+t] * (double)hsk[n*NTILE+t];
        }
        for (int d=0; d<RD; ++d)
            acc += (double)from_bf16(hqr[m*RD+d]) * (double)from_bf16(hkr[n*RD+d]);
        ref[m*N+n]=(float)acc;
    }

    void *dq,*dk,*dqr,*dkr; float *dsq,*dsk,*dS;
    hipMalloc(&dq,qn); hipMalloc(&dk,kn);
    hipMalloc(&dqr,qrn*2); hipMalloc(&dkr,krn*2);
    hipMalloc(&dsq,sqn*4); hipMalloc(&dsk,skn*4); hipMalloc(&dS,M*N*4);
    hipMemcpy(dq,hq,qn,hipMemcpyHostToDevice); hipMemcpy(dk,hk,kn,hipMemcpyHostToDevice);
    hipMemcpy(dqr,hqr,qrn*2,hipMemcpyHostToDevice); hipMemcpy(dkr,hkr,krn*2,hipMemcpyHostToDevice);
    hipMemcpy(dsq,hsq,sqn*4,hipMemcpyHostToDevice); hipMemcpy(dsk,hsk,skn*4,hipMemcpyHostToDevice);
    hipLaunchKernelGGL(qk_full, dim3(1), 64, 0, 0,
        (const fp8_t*)dq,(const fp8_t*)dk,dsq,dsk,(const bf16_t*)dqr,(const bf16_t*)dkr,dS);
    hipDeviceSynchronize();
    float* hS=new float[M*N]; hipMemcpy(hS,dS,M*N*4,hipMemcpyDeviceToHost);

    double max_abs=0,max_rel=0;
    for (int i=0;i<M*N;++i){ double d=fabs((double)hS[i]-(double)ref[i]);
        max_abs=fmax(max_abs,d); max_rel=fmax(max_rel,d/(fabs((double)ref[i])+1e-6)); }
    printf("QK-full (fp8 nope sw-scale + bf16 rope) vs ref:  max_abs=%.5f max_rel=%.5f\n", max_abs, max_rel);
    printf("S[0][0..3] gpu=[%.3f %.3f %.3f %.3f] ref=[%.3f %.3f %.3f %.3f]\n",
           hS[0],hS[1],hS[2],hS[3], ref[0],ref[1],ref[2],ref[3]);
    printf("%s\n",(max_rel<2e-3)?"PASS":"FAIL");
    return (max_rel<2e-3)?0:1;
}
#endif

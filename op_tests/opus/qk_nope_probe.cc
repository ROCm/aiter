// SPDX-License-Identifier: MIT
// Validate the fused QK-nope primitive in isolation:
//   S[m,n] = sum over 7 tiles of  sq[m,t]*sk[n,t] * (sum_{d in tile t} q8[m,d]*k8[n,d])
// using the PROVEN non-scaled fp8 16x16x32 MFMA (opus::mfma) + SOFTWARE per-64-tile
// scaling (the opus_gemm a8w8_scale pattern). This is exactly how the attention
// kernel's QK nope path will work. Compared against a dequant CPU reference.
//
// 16x16x32 fp8 fragment layout (standard CDNA, same 16x16 C-layout as test_mxfp.cu):
//   A[16,32]: a_reg[j] = A[m=lane%16][k=(lane/16)*8 + j], j=0..7
//   B[32,16]: for QK we want S=Q@K^T, so B[d][n]=K[n][d]; b_reg[j]=k8[n*K + kbase + (lane/16)*8 + j]
//   C[16,16]: c_reg[i] = C[m=(lane/16)*4 + i][n=lane%16], i=0..3
//
// Build (docker, gfx950):
//   hipcc -std=c++20 --offload-arch=gfx950 -O2 -I csrc/include op_tests/opus/qk_nope_probe.cc -o /tmp/qk && /tmp/qk
#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#endif

using opus::fp8_t;
using opus::fp32_t;
using fp8x8_t  = opus::fp8x8_t;
using fp32x4_t = opus::vector_t<fp32_t, 4>;

constexpr int M = 16, N = 16, KD = 448, TILE = 64, NTILE = 7;

__global__ void qk_nope(const fp8_t* __restrict__ q8,   // [M, KD]
                        const fp8_t* __restrict__ k8,   // [N, KD]
                        const float* __restrict__ sq,   // [M, NTILE]
                        const float* __restrict__ sk,   // [N, NTILE]
                        float* __restrict__ S)          // [M, N]
{
#if defined(__gfx950__)
    using namespace opus;
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int m    = lane % 16;     // A-load row
    int kblk = lane / 16;     // 0..3, which 8-wide K sub-block this lane loads
    int n    = lane % 16;     // B/output column
    float s[4] = {0.f, 0.f, 0.f, 0.f};

    for (int t = 0; t < NTILE; ++t) {
        fp32x4_t vc{0.f, 0.f, 0.f, 0.f};
        for (int kk = 0; kk < 2; ++kk) {       // 2 x 32 = 64 (one tile)
            int kbase = t * TILE + kk * 32;
            fp8x8_t a_reg, b_reg;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                a_reg[j] = q8[m * KD + kbase + kblk * 8 + j];
                b_reg[j] = k8[n * KD + kbase + kblk * 8 + j];
            }
            vc = mfma<fp8_t, fp8_t, fp32_t, 16, 16, 32>{}(a_reg, b_reg, vc);
        }
        // vc[i] = partial Q.K over this 64-tile for C[m_out=kblk*4+i][n]
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int mo = kblk * 4 + i;
            s[i] += vc[i] * sq[mo * NTILE + t] * sk[n * NTILE + t];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) S[(kblk * 4 + i) * N + n] = s[i];
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
static float e4m3_decode(unsigned char b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, mant = b & 0x7;
    float v;
    if (e == 0) v = (float)mant / 8.0f * 0.015625f;            // 2^-6 subnormal
    else        v = (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, e - 7);
    return s ? -v : v;
}
static unsigned char rand_fp8() {
    // Positive-only (clear sign) so QK sums don't catastrophically cancel; this
    // isolates the computation logic from fp32-vs-double rounding on near-zero
    // results. Also cap exponent to keep magnitudes ~O(1) (realistic attention).
    for (;;) {
        unsigned char b = (unsigned char)(rand() & 0x3F); // sign=0, exp<=7
        int e = (b >> 3) & 0xF, mant = b & 0x7;
        if (e == 15 && mant == 7) continue; // skip NaN (e4m3fn)
        return b;
    }
}

int main() {
    srand(1234);
    const int qn = M * KD, kn = N * KD, sqn = M * NTILE, skn = N * NTILE;
    unsigned char *hq = new unsigned char[qn], *hk = new unsigned char[kn];
    float *hsq = new float[sqn], *hsk = new float[skn];
    for (int i = 0; i < qn; ++i) hq[i] = rand_fp8();
    for (int i = 0; i < kn; ++i) hk[i] = rand_fp8();
    // scales: random powers of two 2^{-2..2}
    for (int i = 0; i < sqn; ++i) hsq[i] = ldexpf(1.0f, (rand() % 5) - 2);
    for (int i = 0; i < skn; ++i) hsk[i] = ldexpf(1.0f, (rand() % 5) - 2);

    // CPU reference: dequant then Q@K^T
    float* ref = new float[M * N];
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int t = 0; t < NTILE; ++t) {
                double part = 0;
                for (int d = t * TILE; d < (t + 1) * TILE; ++d)
                    part += (double)e4m3_decode(hq[m * KD + d]) * (double)e4m3_decode(hk[n * KD + d]);
                acc += part * (double)hsq[m * NTILE + t] * (double)hsk[n * NTILE + t];
            }
            ref[m * N + n] = (float)acc;
        }

    void *dq, *dk; float *dsq, *dsk, *dS;
    hipMalloc(&dq, qn); hipMalloc(&dk, kn);
    hipMalloc(&dsq, sqn * 4); hipMalloc(&dsk, skn * 4); hipMalloc(&dS, M * N * 4);
    hipMemcpy(dq, hq, qn, hipMemcpyHostToDevice);
    hipMemcpy(dk, hk, kn, hipMemcpyHostToDevice);
    hipMemcpy(dsq, hsq, sqn * 4, hipMemcpyHostToDevice);
    hipMemcpy(dsk, hsk, skn * 4, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(qk_nope, dim3(1), 64, 0, 0,
                       (const fp8_t*)dq, (const fp8_t*)dk, dsq, dsk, dS);
    hipDeviceSynchronize();
    float* hS = new float[M * N];
    hipMemcpy(hS, dS, M * N * 4, hipMemcpyDeviceToHost);

    double max_abs = 0, max_rel = 0;
    for (int i = 0; i < M * N; ++i) {
        double d = fabs((double)hS[i] - (double)ref[i]);
        max_abs = fmax(max_abs, d);
        max_rel = fmax(max_rel, d / (fabs((double)ref[i]) + 1e-6));
    }
    printf("QK-nope fp8-MFMA + sw-scale vs dequant ref:  max_abs=%.5f  max_rel=%.5f\n", max_abs, max_rel);
    printf("sample S[0][0..3] gpu=[%.3f %.3f %.3f %.3f] ref=[%.3f %.3f %.3f %.3f]\n",
           hS[0], hS[1], hS[2], hS[3], ref[0], ref[1], ref[2], ref[3]);
    printf("%s\n", (max_rel < 1e-3) ? "PASS" : "FAIL");
    return (max_rel < 1e-3) ? 0 : 1;
}
#endif

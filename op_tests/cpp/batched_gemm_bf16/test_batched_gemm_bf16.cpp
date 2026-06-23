// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free correctness test for libbatched_gemm_bf16.so.
//
// Allocates bf16 A/B/Y entirely with hipMalloc (no torch), runs the GEMM via
// the C-ABI entry point, and compares against an fp32 CPU reference computed
// from the same bf16-rounded inputs. Prints PASS/FAIL.
//
//   E[b, m, n] = sum_k A[b, m, k] * B[b, n, k]      (B is the col-major operand)
//
// Usage: ./test_bgemm.exe [B M N K]   (defaults: 2 512 512 512)
#include "batched_gemm_bf16.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(cmd)                                                                  \
    do                                                                                  \
    {                                                                                   \
        hipError_t e = (cmd);                                                           \
        if(e != hipSuccess)                                                             \
        {                                                                               \
            fprintf(stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(e),          \
                    __FILE__, __LINE__);                                                \
            exit(1);                                                                    \
        }                                                                               \
    } while(0)

// round-to-nearest-even float -> bf16
static inline uint16_t f2b(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t lsb      = (x >> 16) & 1u;
    uint32_t rounding = 0x7fffu + lsb;
    x += rounding;
    return static_cast<uint16_t>(x >> 16);
}

static inline float b2f(uint16_t b)
{
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

int main(int argc, char** argv)
{
    int B = 2, M = 512, N = 512, K = 512;
    if(argc >= 5)
    {
        B = atoi(argv[1]);
        M = atoi(argv[2]);
        N = atoi(argv[3]);
        K = atoi(argv[4]);
    }

    const size_t aN = static_cast<size_t>(B) * M * K;
    const size_t bN = static_cast<size_t>(B) * N * K;
    const size_t eN = static_cast<size_t>(B) * M * N;

    std::vector<float> Af(aN), Bf(bN);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for(auto& x : Af)
        x = dist(rng);
    for(auto& x : Bf)
        x = dist(rng);

    // bf16-rounded host copies (used both on device and in the reference so the
    // comparison isolates kernel error, not input-rounding error).
    std::vector<uint16_t> Ab(aN), Bb(bN), Eb(eN);
    for(size_t i = 0; i < aN; i++)
        Ab[i] = f2b(Af[i]);
    for(size_t i = 0; i < bN; i++)
        Bb[i] = f2b(Bf[i]);

    // fp32 CPU reference.
    std::vector<float> Eref(eN);
    for(int b = 0; b < B; b++)
        for(int m = 0; m < M; m++)
            for(int n = 0; n < N; n++)
            {
                float acc          = 0.f;
                const uint16_t* ar = &Ab[(static_cast<size_t>(b) * M + m) * K];
                const uint16_t* br = &Bb[(static_cast<size_t>(b) * N + n) * K];
                for(int k = 0; k < K; k++)
                    acc += b2f(ar[k]) * b2f(br[k]);
                Eref[(static_cast<size_t>(b) * M + m) * N + n] = acc;
            }

    void *dA = nullptr, *dB = nullptr, *dE = nullptr;
    HIP_CHECK(hipMalloc(&dA, aN * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&dB, bN * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&dE, eN * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(dA, Ab.data(), aN * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, Bb.data(), bN * sizeof(uint16_t), hipMemcpyHostToDevice));

    batched_gemm_bf16_args args{};
    args.a_ptr     = dA;
    args.b_ptr     = dB;
    args.e_ptr     = dE;
    args.bias_ptr  = nullptr;
    args.B         = B;
    args.M         = M;
    args.N         = N;
    args.K         = K;
    args.kbatch    = 1;
    args.device_id = 0;
    args.stream    = nullptr;

    int rc = aiter_batched_gemm_bf16(&args);
    if(rc != 0)
    {
        fprintf(stderr, "aiter_batched_gemm_bf16 returned %d\n", rc);
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(Eb.data(), dE, eN * sizeof(uint16_t), hipMemcpyDeviceToHost));

    // Mixed absolute/relative tolerance suited to bf16 output + fp32 accumulate.
    const float atol = 1e-2f;
    const float rtol = 2e-2f;
    double max_abs = 0.0, max_rel = 0.0;
    size_t n_bad = 0;
    for(size_t i = 0; i < eN; i++)
    {
        float got = b2f(Eb[i]);
        float ref = Eref[i];
        float ae  = std::fabs(got - ref);
        float re  = ae / (std::fabs(ref) + 1e-6f);
        if(ae > max_abs)
            max_abs = ae;
        if(re > max_rel)
            max_rel = re;
        if(ae > atol + rtol * std::fabs(ref))
            n_bad++;
    }

    printf("B=%d M=%d N=%d K=%d  elems=%zu  max_abs=%.4f  max_rel=%.4f  mismatches=%zu\n",
           B, M, N, K, eN, max_abs, max_rel, n_bad);

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dE));

    bool pass = (n_bad == 0);
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

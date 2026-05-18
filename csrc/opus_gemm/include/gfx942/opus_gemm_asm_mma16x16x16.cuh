// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 inline-asm primitives for v_mfma_f32_16x16x16_bf16 pipelines.
//
// Types:  float4_acc (AGPR-resident), short4_ab (MFMA input), i32x4_t (ds_read)
// Atoms:  mfma_agpr, extract_a/b, compute_lds_addrs, scatter_ds_read
// Phases: phase_mma8_dsread2, phase_mma8_dsread4, phase_dsread6_mma8,
//         phase_mma8_pure, kstep_mma2
// AGPR:   agpr_to_vgpr
//
// All functions are __device__ __forceinline__ and require __HIP_DEVICE_COMPILE__.
#pragma once

#ifdef __HIP_DEVICE_COMPILE__

// ============================================================================
// Types
// ============================================================================

using float4_acc = float __attribute__((ext_vector_type(4)));
using short4_ab  = short __attribute__((ext_vector_type(4)));
using i32x4_t = int __attribute__((ext_vector_type(4)));

// ============================================================================
// Atom: single MFMA with AGPR-resident accumulator
// ============================================================================

__device__ __forceinline__
void mfma_agpr(short4_ab src0, short4_ab src1, float4_acc& acc) {
    asm("v_mfma_f32_16x16x16_bf16 %0, %1, %2, %0"
        : "+a"(acc) : "v"(src0), "v"(src1));
}

// ============================================================================
// Operand extraction / LDS address helpers
// ============================================================================

template<typename T, typename V>
__device__ __forceinline__
short4_ab extract_a(const V& v, int i_m, int i_k) {
    return reinterpret_cast<const short4_ab*>(&v)[i_m * T::E_K + i_k];
}

template<typename T, typename V>
__device__ __forceinline__
short4_ab extract_b(const V& v, int i_n, int i_k) {
    return reinterpret_cast<const short4_ab*>(&v)[i_n * T::E_K + i_k];
}

template<typename Smem, int N>
__device__ __forceinline__
void compute_lds_addrs(unsigned* addrs, const Smem& s,
                       const opus::array<opus::index_t, N>& offsets) {
    const unsigned base = static_cast<unsigned>(
        reinterpret_cast<__UINTPTR_TYPE__>(s.ptr));
    using scalar = typename Smem::scalar_type;
    #pragma unroll
    for (int i = 0; i < N; i++)
        addrs[i] = base + static_cast<unsigned>(offsets[i]) * sizeof(scalar);
}

template<typename VT>
__device__ __forceinline__
void scatter_ds_read(VT& dst, const i32x4_t& chunk, int idx) {
    auto* p = reinterpret_cast<i32x4_t*>(&dst);
    p[idx] = chunk;
}

// ============================================================================
// Phase helpers: interleaved MMA + ds_read sequences
// ============================================================================

// Phase 1: 8 MFMAs (4 ksteps x E_M=2) + 2 ds_reads for v_b_next.
// PENDING: outstanding lgkmcnt ops from previous phase's ds_reads.
template<typename T, int PENDING, typename VA, typename VB, typename VB_OUT>
__device__ __forceinline__
void phase_mma8_dsread2(
    const VA& v_a, const VB& v_b, float4_acc* acc,
    VB_OUT& v_b_out, const unsigned* lds_b_addrs)
{
    auto b0 = extract_b<T>(v_b, 0, 0);
    auto b1 = extract_b<T>(v_b, 0, 1);
    auto b2 = extract_b<T>(v_b, 0, 2);
    auto b3 = extract_b<T>(v_b, 0, 3);
    auto a0m0 = extract_a<T>(v_a, 0, 0);
    auto a0m1 = extract_a<T>(v_a, 1, 0);
    auto a1m0 = extract_a<T>(v_a, 0, 1);
    auto a1m1 = extract_a<T>(v_a, 1, 1);
    auto a2m0 = extract_a<T>(v_a, 0, 2);
    auto a2m1 = extract_a<T>(v_a, 1, 2);
    auto a3m0 = extract_a<T>(v_a, 0, 3);
    auto a3m1 = extract_a<T>(v_a, 1, 3);

    i32x4_t rd0, rd1;
    asm volatile(
        "s_waitcnt lgkmcnt(%[w0])\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b0], %[a0m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b0], %[a0m1], %[c1]\n"
        "ds_read_b128 %[rd0], %[addr0]\n"
        "s_waitcnt lgkmcnt(%[w1])\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b1], %[a1m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b1], %[a1m1], %[c1]\n"
        "ds_read_b128 %[rd1], %[addr1]\n"
        "s_waitcnt lgkmcnt(%[w2])\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b2], %[a2m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b2], %[a2m1], %[c1]\n"
        "s_waitcnt lgkmcnt(%[w3])\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b3], %[a3m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b3], %[a3m1], %[c1]\n"
        : [c0]"+a"(acc[0]), [c1]"+a"(acc[1]),
          [rd0]"=&v"(rd0), [rd1]"=&v"(rd1)
        : [b0]"v"(b0), [b1]"v"(b1), [b2]"v"(b2), [b3]"v"(b3),
          [a0m0]"v"(a0m0), [a0m1]"v"(a0m1),
          [a1m0]"v"(a1m0), [a1m1]"v"(a1m1),
          [a2m0]"v"(a2m0), [a2m1]"v"(a2m1),
          [a3m0]"v"(a3m0), [a3m1]"v"(a3m1),
          [addr0]"v"(lds_b_addrs[0]), [addr1]"v"(lds_b_addrs[1]),
          [w0]"n"(PENDING - 3),
          [w1]"n"(PENDING - 3),
          [w2]"n"(PENDING - 3),
          [w3]"n"(PENDING - 4)
        : "memory"
    );
    scatter_ds_read(v_b_out, rd0, 0);
    scatter_ds_read(v_b_out, rd1, 1);
}

// Phase 2: 8 MFMAs + 4 ds_reads for v_a_next.
// PENDING: outstanding lgkmcnt ops from Phase 1's ds_reads.
template<typename T, int PENDING, typename VA, typename VB, typename VA_OUT>
__device__ __forceinline__
void phase_mma8_dsread4(
    const VA& v_a, const VB& v_b, float4_acc* acc,
    VA_OUT& v_a_out, const unsigned* lds_a_addrs)
{
    auto b0 = extract_b<T>(v_b, 0, 0);
    auto b1 = extract_b<T>(v_b, 0, 1);
    auto b2 = extract_b<T>(v_b, 0, 2);
    auto b3 = extract_b<T>(v_b, 0, 3);
    auto a0m0 = extract_a<T>(v_a, 0, 0);
    auto a0m1 = extract_a<T>(v_a, 1, 0);
    auto a1m0 = extract_a<T>(v_a, 0, 1);
    auto a1m1 = extract_a<T>(v_a, 1, 1);
    auto a2m0 = extract_a<T>(v_a, 0, 2);
    auto a2m1 = extract_a<T>(v_a, 1, 2);
    auto a3m0 = extract_a<T>(v_a, 0, 3);
    auto a3m1 = extract_a<T>(v_a, 1, 3);

    i32x4_t rd0, rd1, rd2, rd3;
    asm volatile(
        "s_waitcnt lgkmcnt(%[w0])\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b0], %[a0m0], %[c0]\n"
        "ds_read_b128 %[rd0], %[addr0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b0], %[a0m1], %[c1]\n"
        "ds_read_b128 %[rd1], %[addr1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b1], %[a1m0], %[c0]\n"
        "ds_read_b128 %[rd2], %[addr2]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b1], %[a1m1], %[c1]\n"
        "ds_read_b128 %[rd3], %[addr3]\n"
        "s_waitcnt lgkmcnt(4)\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b2], %[a2m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b2], %[a2m1], %[c1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b3], %[a3m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b3], %[a3m1], %[c1]\n"
        : [c0]"+a"(acc[0]), [c1]"+a"(acc[1]),
          [rd0]"=&v"(rd0), [rd1]"=&v"(rd1), [rd2]"=&v"(rd2), [rd3]"=&v"(rd3)
        : [b0]"v"(b0), [b1]"v"(b1), [b2]"v"(b2), [b3]"v"(b3),
          [a0m0]"v"(a0m0), [a0m1]"v"(a0m1),
          [a1m0]"v"(a1m0), [a1m1]"v"(a1m1),
          [a2m0]"v"(a2m0), [a2m1]"v"(a2m1),
          [a3m0]"v"(a3m0), [a3m1]"v"(a3m1),
          [addr0]"v"(lds_a_addrs[0]), [addr1]"v"(lds_a_addrs[1]),
          [addr2]"v"(lds_a_addrs[2]), [addr3]"v"(lds_a_addrs[3]),
          [w0]"n"(PENDING - 1)
        : "memory"
    );
    scatter_ds_read(v_a_out, rd0, 0);
    scatter_ds_read(v_a_out, rd1, 1);
    scatter_ds_read(v_a_out, rd2, 2);
    scatter_ds_read(v_a_out, rd3, 3);
}

// Phase 4: 8 MFMAs interleaved with 6 ds_reads (2 v_b_new + 4 v_a_new).
template<typename T, typename VA, typename VB, typename VB_OUT, typename VA_OUT>
__device__ __forceinline__
void phase_dsread6_mma8(
    const VA& v_a, const VB& v_b, float4_acc* acc,
    VB_OUT& v_b_out, const unsigned* lds_b_addrs,
    VA_OUT& v_a_out, const unsigned* lds_a_addrs)
{
    auto b0 = extract_b<T>(v_b, 0, 0);
    auto b1 = extract_b<T>(v_b, 0, 1);
    auto b2 = extract_b<T>(v_b, 0, 2);
    auto b3 = extract_b<T>(v_b, 0, 3);
    auto a0m0 = extract_a<T>(v_a, 0, 0);
    auto a0m1 = extract_a<T>(v_a, 1, 0);
    auto a1m0 = extract_a<T>(v_a, 0, 1);
    auto a1m1 = extract_a<T>(v_a, 1, 1);
    auto a2m0 = extract_a<T>(v_a, 0, 2);
    auto a2m1 = extract_a<T>(v_a, 1, 2);
    auto a3m0 = extract_a<T>(v_a, 0, 3);
    auto a3m1 = extract_a<T>(v_a, 1, 3);

    i32x4_t brd0, brd1, ard0, ard1, ard2, ard3;
    asm volatile(
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b0], %[a0m0], %[c0]\n"
        "ds_read_b128 %[brd0], %[baddr0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b0], %[a0m1], %[c1]\n"
        "ds_read_b128 %[brd1], %[baddr1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b1], %[a1m0], %[c0]\n"
        "ds_read_b128 %[ard0], %[aaddr0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b1], %[a1m1], %[c1]\n"
        "ds_read_b128 %[ard1], %[aaddr1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b2], %[a2m0], %[c0]\n"
        "ds_read_b128 %[ard2], %[aaddr2]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b2], %[a2m1], %[c1]\n"
        "ds_read_b128 %[ard3], %[aaddr3]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b3], %[a3m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b3], %[a3m1], %[c1]\n"
        : [c0]"+a"(acc[0]), [c1]"+a"(acc[1]),
          [brd0]"=&v"(brd0), [brd1]"=&v"(brd1),
          [ard0]"=&v"(ard0), [ard1]"=&v"(ard1), [ard2]"=&v"(ard2), [ard3]"=&v"(ard3)
        : [b0]"v"(b0), [b1]"v"(b1), [b2]"v"(b2), [b3]"v"(b3),
          [a0m0]"v"(a0m0), [a0m1]"v"(a0m1),
          [a1m0]"v"(a1m0), [a1m1]"v"(a1m1),
          [a2m0]"v"(a2m0), [a2m1]"v"(a2m1),
          [a3m0]"v"(a3m0), [a3m1]"v"(a3m1),
          [baddr0]"v"(lds_b_addrs[0]), [baddr1]"v"(lds_b_addrs[1]),
          [aaddr0]"v"(lds_a_addrs[0]), [aaddr1]"v"(lds_a_addrs[1]),
          [aaddr2]"v"(lds_a_addrs[2]), [aaddr3]"v"(lds_a_addrs[3])
        : "memory"
    );
    scatter_ds_read(v_b_out, brd0, 0);
    scatter_ds_read(v_b_out, brd1, 1);
    scatter_ds_read(v_a_out, ard0, 0);
    scatter_ds_read(v_a_out, ard1, 1);
    scatter_ds_read(v_a_out, ard2, 2);
    scatter_ds_read(v_a_out, ard3, 3);
}

// Pure 8 MFMAs, no ds_read (epilogue drain).
template<typename T, typename VA, typename VB>
__device__ __forceinline__
void phase_mma8_pure(const VA& v_a, const VB& v_b, float4_acc* acc) {
    auto b0 = extract_b<T>(v_b, 0, 0);
    auto b1 = extract_b<T>(v_b, 0, 1);
    auto b2 = extract_b<T>(v_b, 0, 2);
    auto b3 = extract_b<T>(v_b, 0, 3);
    auto a0m0 = extract_a<T>(v_a, 0, 0);
    auto a0m1 = extract_a<T>(v_a, 1, 0);
    auto a1m0 = extract_a<T>(v_a, 0, 1);
    auto a1m1 = extract_a<T>(v_a, 1, 1);
    auto a2m0 = extract_a<T>(v_a, 0, 2);
    auto a2m1 = extract_a<T>(v_a, 1, 2);
    auto a3m0 = extract_a<T>(v_a, 0, 3);
    auto a3m1 = extract_a<T>(v_a, 1, 3);

    asm volatile(
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b0], %[a0m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b0], %[a0m1], %[c1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b1], %[a1m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b1], %[a1m1], %[c1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b2], %[a2m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b2], %[a2m1], %[c1]\n"
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b3], %[a3m0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b3], %[a3m1], %[c1]\n"
        : [c0]"+a"(acc[0]), [c1]"+a"(acc[1])
        : [b0]"v"(b0), [b1]"v"(b1), [b2]"v"(b2), [b3]"v"(b3),
          [a0m0]"v"(a0m0), [a0m1]"v"(a0m1),
          [a1m0]"v"(a1m0), [a1m1]"v"(a1m1),
          [a2m0]"v"(a2m0), [a2m1]"v"(a2m1),
          [a3m0]"v"(a3m0), [a3m1]"v"(a3m1)
    );
}

// Single k-step: 2 MFMAs (E_M=2, E_N=1).
template<typename T, int I_K, typename VA, typename VB>
__device__ __forceinline__
void kstep_mma2(const VA& v_a, const VB& v_b, float4_acc* acc) {
    auto b = extract_b<T>(v_b, 0, I_K);
    auto am0 = extract_a<T>(v_a, 0, I_K);
    auto am1 = extract_a<T>(v_a, 1, I_K);
    asm volatile(
        "v_mfma_f32_16x16x16_bf16 %[c0], %[b], %[am0], %[c0]\n"
        "v_mfma_f32_16x16x16_bf16 %[c1], %[b], %[am1], %[c1]\n"
        : [c0]"+a"(acc[0]), [c1]"+a"(acc[1])
        : [b]"v"(b), [am0]"v"(am0), [am1]"v"(am1)
    );
}

// ============================================================================
// AGPR -> VGPR readback for store epilogue
// ============================================================================

template<int N_SUB>
__device__ __forceinline__
auto agpr_to_vgpr(const float4_acc* acc) {
    using VC = float __attribute__((ext_vector_type(N_SUB * 4)));
    VC result;
    float* p = reinterpret_cast<float*>(&result);
    #pragma unroll
    for (int i = 0; i < N_SUB; i++) {
        float4_acc tmp = acc[i];
        asm volatile("v_accvgpr_read_b32 %0, %1" : "=v"(p[i*4+0]) : "a"(tmp[0]));
        asm volatile("v_accvgpr_read_b32 %0, %1" : "=v"(p[i*4+1]) : "a"(tmp[1]));
        asm volatile("v_accvgpr_read_b32 %0, %1" : "=v"(p[i*4+2]) : "a"(tmp[2]));
        asm volatile("v_accvgpr_read_b32 %0, %1" : "=v"(p[i*4+3]) : "a"(tmp[3]));
    }
    return result;
}

#endif // __HIP_DEVICE_COMPILE__

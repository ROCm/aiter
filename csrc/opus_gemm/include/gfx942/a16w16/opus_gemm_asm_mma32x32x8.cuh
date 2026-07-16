// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 inline-asm primitives for v_mfma_f32_32x32x8_bf16 pipelines.
#pragma once

#ifdef __HIP_DEVICE_COMPILE__

namespace opus_quad_mfma32_gfx942 {

using float16_acc = float __attribute__((ext_vector_type(16)));
using short4_ab   = short __attribute__((ext_vector_type(4)));
using i32x4_t     = int __attribute__((ext_vector_type(4)));

template<int N>
struct frag_b128 {
    i32x4_t chunk[N];
};

template<typename T, int N>
__device__ __forceinline__
short4_ab extract_a(const frag_b128<N>& v, int i_m, int i_k) {
    return reinterpret_cast<const short4_ab*>(v.chunk)[i_m * T::E_K + i_k];
}

template<typename T, int N>
__device__ __forceinline__
short4_ab extract_b(const frag_b128<N>& v, int i_n, int i_k) {
    return reinterpret_cast<const short4_ab*>(v.chunk)[i_n * T::E_K + i_k];
}

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

template<typename Smem, int N, int STRIDE_ELEM = 512>
__device__ __forceinline__
void compute_lds_addrs_x1b(unsigned* addrs, const Smem& s,
                           const opus::array<opus::index_t, N>& offsets) {
    const unsigned base = static_cast<unsigned>(
        reinterpret_cast<__UINTPTR_TYPE__>(s.ptr));
    using scalar = typename Smem::scalar_type;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        unsigned uv = static_cast<unsigned>(offsets[i]);
        unsigned row = uv / STRIDE_ELEM;
        unsigned col = uv - row * STRIDE_ELEM;
        unsigned xor_col = ((row ^ (col >> 5)) & 7u) << 3;
        unsigned sw = row * STRIDE_ELEM + (col ^ xor_col);
        addrs[i] = base + sw * static_cast<unsigned>(sizeof(scalar));
    }
}

template<int N>
__device__ __forceinline__
void ds_read_b128_frag(frag_b128<N>& dst, const unsigned* lds_addrs, int idx) {
    asm volatile(
        "ds_read_b128 %[rd], %[addr]\n"
        : [rd]"=&v"(dst.chunk[idx])
        : [addr]"v"(lds_addrs[idx])
        : "memory"
    );
}

template<typename T, int I_M, int I_N, int I_K, typename VA, typename VB>
__device__ __forceinline__
void mfma32_accumulate_one(const VA& v_a, const VB& v_b, float16_acc& acc) {
    static_assert(T::E_M == 2 && T::E_N == 2 && T::E_K == 4);
    auto b = extract_b<T>(v_b, I_N, I_K);
    auto a = extract_a<T>(v_a, I_M, I_K);
    asm volatile(
        "v_mfma_f32_32x32x8_bf16 %[c0], %[b], %[a], %[c0]\n"
        : [c0]"+a"(acc)
        : [b]"v"(b), [a]"v"(a)
    );
}

template<typename T, int I_K, typename VA, typename VB>
__device__ __forceinline__
void kstep_compute_2x2(const VA& v_a, const VB& v_b, float16_acc* acc) {
    static_assert(T::E_M == 2 && T::E_N == 2 && T::E_K == 4);
    mfma32_accumulate_one<T, 0, 0, I_K>(v_a, v_b, acc[0]);
    mfma32_accumulate_one<T, 0, 1, I_K>(v_a, v_b, acc[1]);
    mfma32_accumulate_one<T, 1, 0, I_K>(v_a, v_b, acc[2]);
    mfma32_accumulate_one<T, 1, 1, I_K>(v_a, v_b, acc[3]);
}

template<typename T, typename VA, typename VB>
__device__ __forceinline__
void phase_compute_2x2(const VA& v_a, const VB& v_b, float16_acc* acc) {
    kstep_compute_2x2<T, 0>(v_a, v_b, acc);
    kstep_compute_2x2<T, 1>(v_a, v_b, acc);
    kstep_compute_2x2<T, 2>(v_a, v_b, acc);
    kstep_compute_2x2<T, 3>(v_a, v_b, acc);
}

template<typename T, int I_K, typename VA, typename VB>
__device__ __forceinline__
void mfma32_prefetch_step_2x2(
    const VA& v_a, const VB& v_b, float16_acc* acc,
    frag_b128<T::b_ds_read_insts>& v_b0_out, const unsigned* lds_b0_addrs,
    frag_b128<T::a_ds_read_insts>& v_a0_out, const unsigned* lds_a0_addrs,
    frag_b128<T::b_ds_read_insts>& v_b1_out, const unsigned* lds_b1_addrs,
    frag_b128<T::a_ds_read_insts>& v_a1_out, const unsigned* lds_a1_addrs)
{
    kstep_compute_2x2<T, I_K>(v_a, v_b, acc);
    ds_read_b128_frag(v_b0_out, lds_b0_addrs, I_K);
    ds_read_b128_frag(v_a0_out, lds_a0_addrs, I_K);
    ds_read_b128_frag(v_b1_out, lds_b1_addrs, I_K);
    ds_read_b128_frag(v_a1_out, lds_a1_addrs, I_K);
}

// 256x256 phase: compute the final quadrant while prefetching the entire next
// tile. The ds_read outputs are written directly into the next-tile fragments
// and intentionally left pending. The read order matches the LDS overwrite
// order in the 256x256 pipeline: B0/A0 first, then B1/A1 progressively. The
// caller is responsible for partial lgkmcnt waits before consuming or
// overwriting those fragments.
template<typename T, typename VA, typename VB>
__device__ __forceinline__
void phase_ab_prefetch_all_quadrants_2x2_ordered_nowait(
    const VA& v_a, const VB& v_b, float16_acc* acc,
    frag_b128<T::b_ds_read_insts>& v_b0_out, const unsigned* lds_b0_addrs,
    frag_b128<T::a_ds_read_insts>& v_a0_out, const unsigned* lds_a0_addrs,
    frag_b128<T::b_ds_read_insts>& v_b1_out, const unsigned* lds_b1_addrs,
    frag_b128<T::a_ds_read_insts>& v_a1_out, const unsigned* lds_a1_addrs)
{
    static_assert(T::E_M == 2 && T::E_N == 2 && T::E_K == 4);
    auto b00 = extract_b<T>(v_b, 0, 0);
    auto b10 = extract_b<T>(v_b, 1, 0);
    auto a00 = extract_a<T>(v_a, 0, 0);
    auto a10 = extract_a<T>(v_a, 1, 0);
    auto b01 = extract_b<T>(v_b, 0, 1);
    auto b11 = extract_b<T>(v_b, 1, 1);
    auto a01 = extract_a<T>(v_a, 0, 1);
    auto a11 = extract_a<T>(v_a, 1, 1);
    auto b02 = extract_b<T>(v_b, 0, 2);
    auto b12 = extract_b<T>(v_b, 1, 2);
    auto a02 = extract_a<T>(v_a, 0, 2);
    auto a12 = extract_a<T>(v_a, 1, 2);
    auto b03 = extract_b<T>(v_b, 0, 3);
    auto b13 = extract_b<T>(v_b, 1, 3);
    auto a03 = extract_a<T>(v_a, 0, 3);
    auto a13 = extract_a<T>(v_a, 1, 3);

    asm volatile(
        "v_mfma_f32_32x32x8_bf16 %[c0], %[b00], %[a00], %[c0]\n"
        "ds_read_b128 %[b0rd0], %[b0addr0]\n"
        "v_mfma_f32_32x32x8_bf16 %[c1], %[b10], %[a00], %[c1]\n"
        "ds_read_b128 %[a0rd0], %[a0addr0]\n"
        "v_mfma_f32_32x32x8_bf16 %[c2], %[b00], %[a10], %[c2]\n"
        "ds_read_b128 %[b0rd1], %[b0addr1]\n"
        "v_mfma_f32_32x32x8_bf16 %[c3], %[b10], %[a10], %[c3]\n"
        "ds_read_b128 %[a0rd1], %[a0addr1]\n"

        "v_mfma_f32_32x32x8_bf16 %[c0], %[b01], %[a01], %[c0]\n"
        "ds_read_b128 %[b0rd2], %[b0addr2]\n"
        "v_mfma_f32_32x32x8_bf16 %[c1], %[b11], %[a01], %[c1]\n"
        "ds_read_b128 %[a0rd2], %[a0addr2]\n"
        "v_mfma_f32_32x32x8_bf16 %[c2], %[b01], %[a11], %[c2]\n"
        "ds_read_b128 %[b0rd3], %[b0addr3]\n"
        "v_mfma_f32_32x32x8_bf16 %[c3], %[b11], %[a11], %[c3]\n"
        "ds_read_b128 %[a0rd3], %[a0addr3]\n"

        "v_mfma_f32_32x32x8_bf16 %[c0], %[b02], %[a02], %[c0]\n"
        "ds_read_b128 %[b1rd0], %[b1addr0]\n"
        "v_mfma_f32_32x32x8_bf16 %[c1], %[b12], %[a02], %[c1]\n"
        "ds_read_b128 %[a1rd0], %[a1addr0]\n"
        "v_mfma_f32_32x32x8_bf16 %[c2], %[b02], %[a12], %[c2]\n"
        "ds_read_b128 %[b1rd1], %[b1addr1]\n"
        "v_mfma_f32_32x32x8_bf16 %[c3], %[b12], %[a12], %[c3]\n"
        "ds_read_b128 %[a1rd1], %[a1addr1]\n"

        "v_mfma_f32_32x32x8_bf16 %[c0], %[b03], %[a03], %[c0]\n"
        "ds_read_b128 %[b1rd2], %[b1addr2]\n"
        "v_mfma_f32_32x32x8_bf16 %[c1], %[b13], %[a03], %[c1]\n"
        "ds_read_b128 %[a1rd2], %[a1addr2]\n"
        "v_mfma_f32_32x32x8_bf16 %[c2], %[b03], %[a13], %[c2]\n"
        "ds_read_b128 %[b1rd3], %[b1addr3]\n"
        "v_mfma_f32_32x32x8_bf16 %[c3], %[b13], %[a13], %[c3]\n"
        "ds_read_b128 %[a1rd3], %[a1addr3]\n"
        : [c0]"+a"(acc[0]), [c1]"+a"(acc[1]), [c2]"+a"(acc[2]), [c3]"+a"(acc[3]),
          [b0rd0]"=&v"(v_b0_out.chunk[0]), [a0rd0]"=&v"(v_a0_out.chunk[0]),
          [b0rd1]"=&v"(v_b0_out.chunk[1]), [a0rd1]"=&v"(v_a0_out.chunk[1]),
          [b0rd2]"=&v"(v_b0_out.chunk[2]), [a0rd2]"=&v"(v_a0_out.chunk[2]),
          [b0rd3]"=&v"(v_b0_out.chunk[3]), [a0rd3]"=&v"(v_a0_out.chunk[3]),
          [b1rd0]"=&v"(v_b1_out.chunk[0]), [a1rd0]"=&v"(v_a1_out.chunk[0]),
          [b1rd1]"=&v"(v_b1_out.chunk[1]), [a1rd1]"=&v"(v_a1_out.chunk[1]),
          [b1rd2]"=&v"(v_b1_out.chunk[2]), [a1rd2]"=&v"(v_a1_out.chunk[2]),
          [b1rd3]"=&v"(v_b1_out.chunk[3]), [a1rd3]"=&v"(v_a1_out.chunk[3])
        : [b00]"v"(b00), [b10]"v"(b10), [a00]"v"(a00), [a10]"v"(a10),
          [b01]"v"(b01), [b11]"v"(b11), [a01]"v"(a01), [a11]"v"(a11),
          [b02]"v"(b02), [b12]"v"(b12), [a02]"v"(a02), [a12]"v"(a12),
          [b03]"v"(b03), [b13]"v"(b13), [a03]"v"(a03), [a13]"v"(a13),
          [b0addr0]"v"(lds_b0_addrs[0]), [a0addr0]"v"(lds_a0_addrs[0]),
          [b0addr1]"v"(lds_b0_addrs[1]), [a0addr1]"v"(lds_a0_addrs[1]),
          [b0addr2]"v"(lds_b0_addrs[2]), [a0addr2]"v"(lds_a0_addrs[2]),
          [b0addr3]"v"(lds_b0_addrs[3]), [a0addr3]"v"(lds_a0_addrs[3]),
          [b1addr0]"v"(lds_b1_addrs[0]), [a1addr0]"v"(lds_a1_addrs[0]),
          [b1addr1]"v"(lds_b1_addrs[1]), [a1addr1]"v"(lds_a1_addrs[1]),
          [b1addr2]"v"(lds_b1_addrs[2]), [a1addr2]"v"(lds_a1_addrs[2]),
          [b1addr3]"v"(lds_b1_addrs[3]), [a1addr3]"v"(lds_a1_addrs[3])
        : "memory"
    );
}

template<int N_SUB>
__device__ __forceinline__
auto agpr_to_vgpr(const float16_acc* acc) {
    using VC = float __attribute__((ext_vector_type(N_SUB * 16)));
    VC result;
    float* p = reinterpret_cast<float*>(&result);
    #pragma unroll
    for (int i = 0; i < N_SUB; i++) {
        float16_acc tmp = acc[i];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            asm volatile("v_accvgpr_read_b32 %0, %1" : "=v"(p[i * 16 + j]) : "a"(tmp[j]));
        }
    }
    return result;
}

template<int N_SUB>
__device__ __forceinline__
auto agpr_to_bf16_vgpr_trunc(const float16_acc* acc) {
    using VC = __bf16 __attribute__((ext_vector_type(N_SUB * 16)));
    VC result;
    __bf16* p = reinterpret_cast<__bf16*>(&result);
    #pragma unroll
    for (int i = 0; i < N_SUB; i++) {
        float16_acc tmp = acc[i];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            float f;
            asm volatile("v_accvgpr_read_b32 %0, %1" : "=v"(f) : "a"(tmp[j]));
            unsigned bits = __builtin_bit_cast(unsigned, f);
            p[i * 16 + j] = __builtin_bit_cast(__bf16, static_cast<unsigned short>(bits >> 16));
        }
    }
    return result;
}

} // namespace opus_quad_mfma32_gfx942

#endif // __HIP_DEVICE_COMPILE__

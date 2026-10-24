// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Standalone gfx950 probe; no Python binding or production ABI is added.
// hipcc -std=c++20 -O3 --offload-arch=gfx950 -Icsrc/include \
//   op_tests/test_opus_moe_backward_primitives.hip -o /tmp/opus_primitives
#include <hip/hip_runtime.h>
#define OPUS_GFX950_NO_INLINE_ASM_TR_LOAD 1
#include "../csrc/opus_moe/moe_backward/include/gfx950/bf16/opus_moe_down_bwd_pipeline_gfx950.cuh"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

using probe_bf16x4 = __bf16 __attribute__((ext_vector_type(4)));
using probe_u32x2 = uint32_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ probe_bf16x4 native_tr_load_bf16x4(
    const __bf16* ptr, int byte_offset)
{
    using i32x2 = int32_t __attribute__((ext_vector_type(2)));
    i32x2 raw;
    const uint32_t addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(ptr) + byte_offset);
    asm volatile("ds_read_b64_tr_b16 %0, %1"
                 : "=v"(raw)
                 : "v"(addr)
                 : "memory");
    return __builtin_bit_cast(probe_bf16x4, raw);
}

static void check(hipError_t status)
{
    if(status != hipSuccess)
    {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(2);
    }
}

__global__ void probe(const uint32_t* bits, uint32_t* packed,
                      uint32_t* lane_outputs, uint32_t* transpose_outputs,
                      int n)
{
#ifdef __HIP_DEVICE_COMPILE__
    __shared__ __bf16 transpose_tile[16 * 16];
    const int lane = threadIdx.x;
    for(int i = lane; i < 16 * 16; i += blockDim.x)
        transpose_tile[i] = __builtin_bit_cast(__bf16,
                                               static_cast<uint16_t>(i + 1));
    __syncthreads();
    for(int i = lane; i < n; i += blockDim.x)
        packed[i] = opus_moe_backward::cvt_pk_bf16_f32(
            __builtin_bit_cast(float, bits[i]),
            __builtin_bit_cast(float, bits[(i + 1) % n]));
    uint32_t lo = 1000 + lane, hi = 2000 + lane;
    // This isolated probe has no MFMA/LDS scheduling between the seed VALU
    // and cross-lane read. Pin the inputs, then satisfy the instruction's
    // producer/read wait states explicitly; production scheduling is untouched.
    asm volatile("s_nop 1" : "+v"(lo), "+v"(hi));
    opus_moe_backward::gfx950::down_bwd_permlane32_swap(lo, hi);
    lane_outputs[lane] = lo;
    lane_outputs[64 + lane] = hi;

    const int byte_offset = ((lane >> 2) * 16 + (lane & 3) * 4) * 2;
    auto smem_tile = opus::make_smem(transpose_tile);
    const auto fallback = smem_tile.template _tr_load<4>(byte_offset);
    const auto native = native_tr_load_bf16x4(transpose_tile, byte_offset);
    opus::s_waitcnt_lgkmcnt(opus::number<0>{});
    const auto fallback_bits = __builtin_bit_cast(probe_u32x2, fallback);
    const auto native_bits = __builtin_bit_cast(probe_u32x2, native);
    transpose_outputs[lane * 4] = fallback_bits[0];
    transpose_outputs[lane * 4 + 1] = fallback_bits[1];
    transpose_outputs[lane * 4 + 2] = native_bits[0];
    transpose_outputs[lane * 4 + 3] = native_bits[1];
#endif
}

// Integer round-to-nearest-even oracle independent of HIP conversion/asm.
static uint16_t bf16_rne(uint32_t bits)
{
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static bool same_bf16(uint16_t got, uint32_t bits)
{
    if((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
        // NaN payload canonicalization is not a numeric equality contract.
        return (got & 0x7f80u) == 0x7f80u && (got & 0x007fu);
    return got == bf16_rne(bits);
}

int main()
{
    std::vector<uint32_t> bits = {
        0u, 0x80000000u, 0x00000001u, 0x80000001u,
        0x007fffffu, 0x807fffffu, 0x00800000u, 0x80800000u,
        0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
        0x7f800001u, 0xff800001u, 0x7f7fffffu, 0xff7fffffu,
    };
    // Both signs, even/odd BF16 mantissas, just below/at/above halfway.
    for(uint32_t sign : {0u, 0x80000000u})
        for(uint32_t exponent : {1u, 63u, 126u, 127u, 128u, 200u, 254u})
            for(uint32_t mantissa = 0; mantissa < 128; ++mantissa)
                for(uint32_t tail : {0x7fffu, 0x8000u, 0x8001u})
                    bits.push_back(sign | (exponent << 23) | (mantissa << 16) | tail);
    uint32_t *input, *output, *lanes, *transpose;
    check(hipMalloc(&input, bits.size() * sizeof(uint32_t)));
    check(hipMalloc(&output, bits.size() * sizeof(uint32_t)));
    check(hipMalloc(&lanes, 128 * sizeof(uint32_t)));
    check(hipMalloc(&transpose, 64 * 4 * sizeof(uint32_t)));
    check(hipMemcpy(input, bits.data(), bits.size() * sizeof(uint32_t), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, input, output, lanes,
                       transpose, static_cast<int>(bits.size()));
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    std::vector<uint32_t> packed(bits.size());
    std::array<uint32_t, 128> layout;
    std::array<uint32_t, 64 * 4> transpose_layout;
    check(hipMemcpy(packed.data(), output, packed.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
    check(hipMemcpy(layout.data(), lanes, sizeof(layout), hipMemcpyDeviceToHost));
    check(hipMemcpy(transpose_layout.data(), transpose,
                    sizeof(transpose_layout), hipMemcpyDeviceToHost));
    int errors = 0;
    for(size_t i = 0; i < bits.size(); ++i)
        if(!same_bf16(packed[i] & 0xffffu, bits[i]) ||
           !same_bf16(packed[i] >> 16, bits[(i + 1) % bits.size()]))
        {
            if(errors++ < 8)
                std::fprintf(stderr, "conversion %zu: %08x %08x -> %08x\n",
                             i, bits[i], bits[(i + 1) % bits.size()], packed[i]);
        }
    // permlane32_swap joins lo/hi fragments into contiguous 32-lane halves.
    for(int lane = 0; lane < 64; ++lane)
    {
        const uint32_t expected_lo = lane < 32 ? 1000 + lane : 2000 + lane - 32;
        const uint32_t expected_hi = lane < 32 ? 1000 + lane + 32 : 2000 + lane;
        if(layout[lane] != expected_lo || layout[64 + lane] != expected_hi)
        {
            if(errors++ < 8)
                std::fprintf(stderr, "lane %d: %u %u expected %u %u\n", lane,
                             layout[lane], layout[64 + lane], expected_lo, expected_hi);
        }
    }
    for(int lane = 0; lane < 64; ++lane)
    {
        const uint32_t fallback0 = transpose_layout[lane * 4];
        const uint32_t fallback1 = transpose_layout[lane * 4 + 1];
        const uint32_t native0 = transpose_layout[lane * 4 + 2];
        const uint32_t native1 = transpose_layout[lane * 4 + 3];
        if(fallback0 != native0 || fallback1 != native1)
        {
            if(errors++ < 8)
                std::fprintf(stderr,
                             "transpose lane %d: fallback %08x %08x native %08x %08x\n",
                             lane, fallback0, fallback1, native0, native1);
        }
    }
    check(hipFree(input));
    check(hipFree(output));
    check(hipFree(lanes));
    check(hipFree(transpose));
    std::printf("%zu packed conversion pairs, 64 lane pairs, 64 transpose lanes: %d errors\n",
                bits.size(), errors);
    return errors ? 1 : 0;
}

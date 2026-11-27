// SPDX-License-Identifier: MIT
#pragma once
#ifndef SM_LANE_MODE
#define SM_LANE_MODE 0
#endif
// Preserve native xor16 then xor32 arithmetic order. Gluon uses these
// opcodes but its reduction tree/FP8 arithmetic contract is different.
template <bool Max>
__device__ __forceinline__ float softmax_four_groups(float x)
{
    uint32_t seed16, bits16 = __builtin_bit_cast(uint32_t, x);
    asm("v_mov_b32_e32 %0, %1" : "=v"(seed16) : "v"(bits16));
    asm("s_nop 2\n\tv_permlane16_swap_b32 %0, %1" : "+v"(seed16), "+v"(bits16));
    float l16 = __builtin_bit_cast(float, seed16), r16 = __builtin_bit_cast(float, bits16);
    x = Max ? fmaxf(l16, r16) : l16 + r16;
    uint32_t seed32, bits32 = __builtin_bit_cast(uint32_t, x);
    asm("v_mov_b32_e32 %0, %1" : "=v"(seed32) : "v"(bits32));
    asm("s_nop 2\n\tv_permlane32_swap_b32 %0, %1" : "+v"(seed32), "+v"(bits32));
    float l32 = __builtin_bit_cast(float, seed32), r32 = __builtin_bit_cast(float, bits32);
    return Max ? fmaxf(l32, r32) : l32 + r32;
}

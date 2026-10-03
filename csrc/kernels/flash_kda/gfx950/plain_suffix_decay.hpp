// gfx950-private BT64 suffix-decay cache helpers.
//
// Keep the arithmetic tree in one forced-inline function: the uncached plain
// scan and the fused post-preparation publisher must form bit-identical FP32
// values.  The cache uses a segment-major [D, 4] layout so one lane publishes
// and consumes all four suffixes with a naturally aligned 16-byte access.
#pragma once

#include "../mfma.hpp"

namespace flashkda_hip::gfx950 {

constexpr int kPlainSuffixDecayD = 128;
constexpr int kPlainSuffixDecayCount = 4;

__device__ __forceinline__ f32x4 plain_bt64_suffix_decay(
        const f32x4& g) {
    // These parentheses spell the existing left-associated scan AST exactly:
    // (((g0 + g1) + g2) + g3), ((g1 + g2) + g3), (g2 + g3), g3.
    return f32x4{
        ex2(((g[0] + g[1]) + g[2]) + g[3]),
        ex2((g[1] + g[2]) + g[3]),
        ex2(g[2] + g[3]),
        ex2(g[3])};
}

__device__ __forceinline__ void store_plain_bt64_suffix_decay(
        float* __restrict__ cache,
        int64_t segment,
        int d,
        const f32x4& decay) {
    reinterpret_cast<f32x4*>(
        cache + segment * kPlainSuffixDecayD * kPlainSuffixDecayCount)[d] =
        decay;
}

__device__ __forceinline__ f32x4 load_plain_bt64_suffix_decay(
        const float* __restrict__ cache,
        int64_t segment,
        int d) {
    return reinterpret_cast<const f32x4*>(
        cache + segment * kPlainSuffixDecayD * kPlainSuffixDecayCount)[d];
}

}  // namespace flashkda_hip::gfx950

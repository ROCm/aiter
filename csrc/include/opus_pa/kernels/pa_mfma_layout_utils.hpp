// MFMA 16x16 FP8 output lane mapping + scatter/gather helpers (CDNA3 gfx942).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/kernels/pa_gemm0_utils.hpp"

namespace pa_decode {

// acc[k] at MFMA 16x16 tile: row = 4*(lane/16)+k, col = lane%16 (within one 64-lane wave).
__device__ __forceinline__ void mfma16_lane_to_mn(int lane, int k, int& row, int& col) {
    col = lane & 15;
    row = ((lane >> 4) << 2) + k;
}

// asm: 4 waves × 16 cols = 64-wide N tile (R_write_out / GEMM1 head slice).
__device__ __forceinline__ int mfma_wave_n_offset(int wave) { return wave << 4; }

// Scatter one wave's MFMA GEMM0 slice (16x16) into S[query_base + row][kv_base + wave*16 + col].
template<int GQA, int SUB_KV>
__device__ __forceinline__ void mfma_scatter_scores_slice(const mfma_acc4& acc,
                                                          int lane,
                                                          int wave,
                                                          int query_base,
                                                          int kv_base,
                                                          int tile_kv,
                                                          float s_out[GQA][SUB_KV]) {
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int row = 0;
        int col = 0;
        mfma16_lane_to_mn(lane, k, row, col);
        const int qi = query_base + row;
        const int ki = kv_base + mfma_wave_n_offset(wave) + col;
        if (qi < GQA && ki < tile_kv) {
            s_out[qi][ki] = acc[k];
        }
    }
#else
    (void)acc;
    (void)lane;
    (void)wave;
    (void)query_base;
    (void)kv_base;
    (void)tile_kv;
    (void)s_out;
#endif
}

// Gather MFMA GEMM1 16x16 head slice into O[query_base + row][head_base + wave*16 + col].
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void mfma_gather_o_slice(const mfma_acc4& acc,
                                                    int lane,
                                                    int wave,
                                                    int query_base,
                                                    int head_base,
                                                    float o_out[GQA][HEAD_DIM]) {
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int row = 0;
        int col = 0;
        mfma16_lane_to_mn(lane, k, row, col);
        const int qi = query_base + row;
        const int di = head_base + mfma_wave_n_offset(wave) + col;
        if (qi < GQA && di < HEAD_DIM) {
            o_out[qi][di] = acc[k];
        }
    }
#else
    (void)acc;
    (void)lane;
    (void)wave;
    (void)query_base;
    (void)head_base;
    (void)o_out;
#endif
}

// Accumulating variant for multi-pi GEMM1.
template<int GQA, int HEAD_DIM>
__device__ __forceinline__ void mfma_gather_o_slice_accum(const mfma_acc4& acc,
                                                          int lane,
                                                          int wave,
                                                          int query_base,
                                                          int head_base,
                                                          float o_out[GQA][HEAD_DIM]) {
#if defined(__gfx942__) || defined(__gfx950__)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int row = 0;
        int col = 0;
        mfma16_lane_to_mn(lane, k, row, col);
        const int qi = query_base + row;
        const int di = head_base + mfma_wave_n_offset(wave) + col;
        if (qi < GQA && di < HEAD_DIM) {
            o_out[qi][di] += acc[k];
        }
    }
#else
    (void)acc;
    (void)lane;
    (void)wave;
    (void)query_base;
    (void)head_base;
    (void)o_out;
#endif
}

}  // namespace pa_decode

// sp3 pa_fuse_alu step0: compact S via row_shr:[8] (DPP simulation).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "opus_pa/kernels/pa_decode_device_utils.hpp"
#include "opus_pa/kernels/pa_mfma_layout_utils.hpp"

namespace pa_decode {

static constexpr int kSHalfRegSize = 8;  // _v_S_half_reg_size
static constexpr int kSRegSize = 16;     // _v_S_reg_size

__device__ __forceinline__ float dpp_row_shr8(float v) {
#if defined(__gfx942__) || defined(__gfx950__)
    const int bits = __float_as_int(v);
    const int sh = __builtin_amdgcn_mov_dpp(bits, 0x118, 0xf, 0xf, true);
    return __int_as_float(sh);
#else
    (void)v;
    return 0.f;
#endif
}

__device__ __forceinline__ float scores_compact_merge_row_shr8(float lo, float hi) {
    return lo + dpp_row_shr8(hi);
}

// sp3 pa_fuse step0 compact on _v_S[0..15] -> dense _v_S[0..7].
__device__ __forceinline__ void scores_compact_lane_regs(float s_lane[kSRegSize]) {
    float upper[kSHalfRegSize];
    for (int k = 0; k < kSHalfRegSize; ++k) {
        upper[k] = s_lane[kSHalfRegSize + k];
    }
    for (int k = 0; k < kSHalfRegSize; ++k) {
        s_lane[k] = scores_compact_merge_row_shr8(s_lane[k], upper[k]);
    }
}

// Gather this lane's MFMA-sparse S into 16 virtual VGPRs (inverse of mfma_scatter_scores_slice).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_sparse_to_lane_regs(const float s_sparse[GQA][SUB_KV],
                                                           int pi,
                                                           int query_base,
                                                           int kv_offset,
                                                           int tile_kv,
                                                           float s_lane[kSRegSize]) {
    constexpr int kNumJ = SUB_KV / 64;
    const int kv_pi_base = kv_offset + pi * (SUB_KV / 2);
    const int lane = lane_id();
    const int wave = wave_id();

    for (int k = 0; k < kSRegSize; ++k) {
        s_lane[k] = 0.f;
    }

    for (int j = 0; j < kNumJ; ++j) {
        const int kv_base = kv_pi_base + j * 64;
        for (int k = 0; k < 4; ++k) {
            int row = 0;
            int col = 0;
            mfma16_lane_to_mn(lane, k, row, col);
            const int qi = query_base + row;
            const int ki = kv_base + mfma_wave_n_offset(wave) + col;
            if (qi < GQA && ki < tile_kv) {
                s_lane[j * 4 + k] = s_sparse[qi][ki];
            }
        }
    }
}

// Write compacted lane regs into dense S for one pi half-tile.
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compact_regs_to_dense_pi(const float s_lane[kSHalfRegSize],
                                                                int pi,
                                                                int query_base,
                                                                int kv_offset,
                                                                int tile_kv,
                                                                float s_dense[GQA][SUB_KV]) {
    const int lane = lane_id();
    const int wave = wave_id();
    const int col = lane & 15;
    const int row_group = lane >> 4;
    const int kv_pi_base = kv_offset + pi * (SUB_KV / 2);
    constexpr int kRowsPerGroup = 4;

    if (row_group >= (GQA + kRowsPerGroup - 1) / kRowsPerGroup) {
        return;
    }

    for (int reg_k = 0; reg_k < kSHalfRegSize; ++reg_k) {
        const int j = reg_k >> 2;
        const int inner_k = reg_k & 3;
        const int qi = query_base + row_group * kRowsPerGroup + inner_k;
        const int ki = kv_pi_base + j * 64 + mfma_wave_n_offset(wave) + col;
        if (qi < GQA && ki < tile_kv) {
            s_dense[qi][ki] = s_lane[reg_k];
        }
    }
}

// One pi slice: sparse -> row_shr:[8] compact -> dense [pi*(SUB_KV/2), (pi+1)*(SUB_KV/2)).
template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compact_mfma_pi(const float s_sparse[GQA][SUB_KV],
                                                       int pi,
                                                       int kv_offset,
                                                       int tile_kv,
                                                       float s_dense[GQA][SUB_KV]) {
    float s_lane[kSRegSize];
    scores_sparse_to_lane_regs<GQA, SUB_KV>(s_sparse, pi, 0, kv_offset, tile_kv, s_lane);
    scores_compact_lane_regs(s_lane);
    scores_compact_regs_to_dense_pi<GQA, SUB_KV>(s_lane, pi, 0, kv_offset, tile_kv, s_dense);
    __syncthreads();
}

template<int GQA, int SUB_KV>
__device__ __forceinline__ void scores_compact_mfma_tile(const float s_sparse[GQA][SUB_KV],
                                                         int kv_offset,
                                                         int tile_kv,
                                                         float s_dense[GQA][SUB_KV],
                                                         bool clear_dense = true) {
    if (clear_dense) {
        for (int idx = threadIdx.x; idx < GQA * SUB_KV; idx += blockDim.x) {
            s_dense[idx / SUB_KV][idx % SUB_KV] = 0.f;
        }
        __syncthreads();
    }
    scores_compact_mfma_pi<GQA, SUB_KV>(s_sparse, 0, kv_offset, tile_kv, s_dense);
    scores_compact_mfma_pi<GQA, SUB_KV>(s_sparse, 1, kv_offset, tile_kv, s_dense);
}

}  // namespace pa_decode

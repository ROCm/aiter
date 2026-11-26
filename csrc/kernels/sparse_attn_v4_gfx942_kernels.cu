// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Sparse 2-source paged attention (CSA + SWA, and SWA-only) for DeepSeek-V4.1
// on gfx942 (MI308X), BF16 Q/KV/O, head_dim = 512, optional fused GPT-J
// inverse RoPE (rope_dim = 64) in the output epilogue.
//
// Design:
//   - 4 waves (256 threads) per workgroup split D = 512 (128 dims/wave, 8 BF16
//     16x16x16 MFMA tiles/wave) with 20 KB LDS (3 WGs/CU = 12 waves/CU on
//     MI308X, 0 scratch spills).
//   - All 4 waves cooperatively load 16 full 512-dim KV rows using 1024-byte
//     scalar-address coalesced uint4 bursts (1 full row per wave load),
//     software-pipelined across QK^T + softmax + PV into XOR-swizzled LDS
//     (zero bank conflicts).
//   - Transposed CDNA3 MFMA layout (S^T = K*Q^T, O^T = V^T*P^T) keeps P in
//     registers across online softmax (2 XOR shuffles, zero LDS transpose).
//   - Path A (Decode / short batches, CSA+SWA & SWA-only): L2-cached split-KV
//     stage1 + 32-WG parallel combine (16 dims/WG, fully unrolled, zero
//     scratch) with optional fused inverse RoPE on tiles 28..31 (D = 448..511).
//   - Path C (Prefill): unsplit 4-wave cooperative kernel with optional fused
//     inverse RoPE on wave 3 tiles 4..7 (zero workspace).
//
// Standalone build:
//   hipcc -O3 --offload-arch=gfx942 -std=c++17 -shared -fPIC \
//     -Wno-hip-deprecated-builtins -o sparse_attn_v4_gfx942.so \
//     csrc/kernels/sparse_attn_v4_gfx942_kernels.cu

#include <hip/hip_runtime.h>
#include <cstdint>

namespace {

constexpr int kWave = 64;
constexpr int kHeadTile = 16;
constexpr int kHeadDim = 512;
constexpr int kRopeDim = 64;
constexpr int kMaxSplits = 32;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kNegLarge = -3.4028234663852886e38f;

using bf16x4_t = short __attribute__((ext_vector_type(4)));
using fp32x4_t = float __attribute__((ext_vector_type(4)));
using u32x4_t = uint32_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint2 pack_bf16x4(float f0, float f1, float f2, float f3) {
    uint32_t lo = static_cast<uint32_t>(f32_to_bf16_rne(f0)) | (static_cast<uint32_t>(f32_to_bf16_rne(f1)) << 16);
    uint32_t hi = static_cast<uint32_t>(f32_to_bf16_rne(f2)) | (static_cast<uint32_t>(f32_to_bf16_rne(f3)) << 16);
    return make_uint2(lo, hi);
}
__device__ __forceinline__ bf16x4_t pack_bf16x4_vec(float f0, float f1, float f2, float f3) {
    bf16x4_t v;
    v[0] = static_cast<short>(f32_to_bf16_rne(f0));
    v[1] = static_cast<short>(f32_to_bf16_rne(f1));
    v[2] = static_cast<short>(f32_to_bf16_rne(f2));
    v[3] = static_cast<short>(f32_to_bf16_rne(f3));
    return v;
}
__device__ __forceinline__ fp32x4_t mfma_16x16x16_bf16(bf16x4_t a, bf16x4_t b, fp32x4_t c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}

struct Smem4W {
    alignas(16) uint16_t kv[16][kHeadDim];   // 16 KB
    alignas(16) fp32x4_t s_red[4][kWave];    // 4 KB -> 20 KB total = 3 WGs/CU!
};

template <int DimTiles>
__device__ __forceinline__ void load_q_slice(
    const uint16_t* __restrict__ q_ptr, int64_t q_base, int64_t q_stride_h,
    int H, int h_base, int w, int lane, bf16x4_t q_reg[DimTiles]) {
    constexpr int WaveDim = DimTiles * 16;
    const int r = lane & 15, g = lane >> 4, h = h_base + r;
    if (h < H) {
        const uint16_t* row = q_ptr + q_base + static_cast<int64_t>(h) * q_stride_h + w * WaveDim + 4 * g;
        #pragma unroll
        for (int dt = 0; dt < DimTiles; ++dt) {
            const uint2 raw = *reinterpret_cast<const uint2*>(row + dt * 16);
            q_reg[dt][0] = static_cast<short>(raw.x & 0xffffu);
            q_reg[dt][1] = static_cast<short>(raw.x >> 16);
            q_reg[dt][2] = static_cast<short>(raw.y & 0xffffu);
            q_reg[dt][3] = static_cast<short>(raw.y >> 16);
        }
    } else {
        #pragma unroll
        for (int dt = 0; dt < DimTiles; ++dt) q_reg[dt] = {0, 0, 0, 0};
    }
}

struct Pref4 {
    u32x4_t r0, r1, r2, r3;
};

__device__ __forceinline__ uint32_t prefetch_coop_4w(
    const uint16_t* __restrict__ kv_base, int64_t kv_stride_n,
    const int* __restrict__ indices_ptr, int idx_base,
    int k_start, int k_hi, int w, int lane, Pref4& pref) {
    const int my_k = k_start + lane;
    const int my_slot = (lane < 16 && my_k < k_hi) ? indices_ptr[idx_base + my_k] : -1;
    const uint32_t valid_mask = static_cast<uint32_t>(__ballot(my_slot >= 0) & 0xffffu);
    const u32x4_t zero = {0u, 0u, 0u, 0u};
    pref.r0 = zero; pref.r1 = zero; pref.r2 = zero; pref.r3 = zero;
    if (valid_mask == 0u) return 0u;

    const int d_off = lane * 8;
    const int s0 = __builtin_amdgcn_readfirstlane(__shfl(my_slot, w * 4 + 0));
    const int s1 = __builtin_amdgcn_readfirstlane(__shfl(my_slot, w * 4 + 1));
    const int s2 = __builtin_amdgcn_readfirstlane(__shfl(my_slot, w * 4 + 2));
    const int s3 = __builtin_amdgcn_readfirstlane(__shfl(my_slot, w * 4 + 3));
    if (s0 >= 0) pref.r0 = *reinterpret_cast<const u32x4_t*>(kv_base + static_cast<int64_t>(s0) * kv_stride_n + d_off);
    if (s1 >= 0) pref.r1 = *reinterpret_cast<const u32x4_t*>(kv_base + static_cast<int64_t>(s1) * kv_stride_n + d_off);
    if (s2 >= 0) pref.r2 = *reinterpret_cast<const u32x4_t*>(kv_base + static_cast<int64_t>(s2) * kv_stride_n + d_off);
    if (s3 >= 0) pref.r3 = *reinterpret_cast<const u32x4_t*>(kv_base + static_cast<int64_t>(s3) * kv_stride_n + d_off);
    return valid_mask;
}

__device__ __forceinline__ void walk_region_4w(
    Smem4W& smem, const uint16_t* __restrict__ kv_base, int64_t kv_stride_n,
    const int* __restrict__ indices_ptr, int idx_base, int k_lo, int k_hi,
    int w, int lane, bool head_valid, float qk_scale,
    const bf16x4_t q_reg[8], float& m_i, float& l_i, fp32x4_t o_acc[8]) {
    if (k_lo >= k_hi) return;
    const int r = lane & 15, g = lane >> 4;
    const int g_half = g & 1, g_hi = g >> 1;
    const int r_hi = r >> 3, r_lo = r & 7;
    const int c8_wave = w * 16;
    const int k0 = w * 4 + 0, k1 = w * 4 + 1, k2 = w * 4 + 2, k3 = w * 4 + 3;

    Pref4 pref;
    uint32_t valid_mask = prefetch_coop_4w(kv_base, kv_stride_n, indices_ptr, idx_base, k_lo, k_hi, w, lane, pref);

    for (int k_start = k_lo; k_start < k_hi; k_start += 16) {
        const uint32_t cur_mask = valid_mask;
        if (cur_mask != 0u) {
            *reinterpret_cast<u32x4_t*>(&smem.kv[k0][(lane ^ k0) * 8]) = pref.r0;
            *reinterpret_cast<u32x4_t*>(&smem.kv[k1][(lane ^ k1) * 8]) = pref.r1;
            *reinterpret_cast<u32x4_t*>(&smem.kv[k2][(lane ^ k2) * 8]) = pref.r2;
            *reinterpret_cast<u32x4_t*>(&smem.kv[k3][(lane ^ k3) * 8]) = pref.r3;
        }
        __syncthreads();

        // Overlap NEXT tile's 1024B full-row global loads with THIS tile's QK^T + softmax + PV!
        const int next_k = k_start + 16;
        if (next_k < k_hi) {
            valid_mask = prefetch_coop_4w(kv_base, kv_stride_n, indices_ptr, idx_base, next_k, k_hi, w, lane, pref);
        }

        if (cur_mask != 0u) {
            fp32x4_t s_acc = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                const int c8 = c8_wave + dt * 2 + g_hi;
                const uint2 raw = *reinterpret_cast<const uint2*>(&smem.kv[r][(c8 ^ r) * 8 + g_half * 4]);
                bf16x4_t k_op;
                k_op[0] = static_cast<short>(raw.x & 0xffffu); k_op[1] = static_cast<short>(raw.x >> 16);
                k_op[2] = static_cast<short>(raw.y & 0xffffu); k_op[3] = static_cast<short>(raw.y >> 16);
                s_acc = mfma_16x16x16_bf16(k_op, q_reg[dt], s_acc);
            }
            smem.s_red[w][lane] = s_acc;
            __syncthreads();

            fp32x4_t s = smem.s_red[0][lane] + smem.s_red[1][lane] + smem.s_red[2][lane] + smem.s_red[3][lane];
            const uint32_t lv = (cur_mask >> (4 * g)) & 0xfu;
            const bool ok0 = head_valid && ((lv >> 0) & 1u), ok1 = head_valid && ((lv >> 1) & 1u);
            const bool ok2 = head_valid && ((lv >> 2) & 1u), ok3 = head_valid && ((lv >> 3) & 1u);
            const float sv0 = ok0 ? (s[0] * qk_scale) : kNegLarge, sv1 = ok1 ? (s[1] * qk_scale) : kNegLarge;
            const float sv2 = ok2 ? (s[2] * qk_scale) : kNegLarge, sv3 = ok3 ? (s[3] * qk_scale) : kNegLarge;
            float m_loc = fmaxf(fmaxf(sv0, sv1), fmaxf(sv2, sv3));
            m_loc = fmaxf(m_loc, __shfl_xor(m_loc, 16));
            m_loc = fmaxf(m_loc, __shfl_xor(m_loc, 32));
            const bool has_v = (m_loc > kNegLarge);
            const float m_new = has_v ? fmaxf(m_i, m_loc) : m_i;
            const float alpha = !has_v ? 1.0f : ((m_i == kNegLarge) ? 0.0f : exp2f(m_i - m_new));
            const float pv0 = ok0 ? exp2f(sv0 - m_new) : 0.0f, pv1 = ok1 ? exp2f(sv1 - m_new) : 0.0f;
            const float pv2 = ok2 ? exp2f(sv2 - m_new) : 0.0f, pv3 = ok3 ? exp2f(sv3 - m_new) : 0.0f;
            float l_loc = (pv0 + pv1) + (pv2 + pv3);
            l_loc += __shfl_xor(l_loc, 16); l_loc += __shfl_xor(l_loc, 32);
            l_i = l_i * alpha + l_loc; m_i = m_new;
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) o_acc[dt] *= alpha;
            const bf16x4_t p_op = pack_bf16x4_vec(pv0, pv1, pv2, pv3);

            const int kb0 = 4 * g;
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                const int c8 = c8_wave + dt * 2 + r_hi;
                bf16x4_t v_op;
                v_op[0] = static_cast<short>(smem.kv[kb0 + 0][(c8 ^ (kb0 + 0)) * 8 + r_lo]);
                v_op[1] = static_cast<short>(smem.kv[kb0 + 1][(c8 ^ (kb0 + 1)) * 8 + r_lo]);
                v_op[2] = static_cast<short>(smem.kv[kb0 + 2][(c8 ^ (kb0 + 2)) * 8 + r_lo]);
                v_op[3] = static_cast<short>(smem.kv[kb0 + 3][(c8 ^ (kb0 + 3)) * 8 + r_lo]);
                o_acc[dt] = mfma_16x16x16_bf16(v_op, p_op, o_acc[dt]);
            }
            __syncthreads();
        }
    }
}

template <bool FuseInvRope>
__device__ __forceinline__ void store_output_4w(
    uint16_t* __restrict__ out_ptr, int64_t out_base, int64_t out_stride_h,
    int H, int h_base, int w, int lane, float scale_out, const fp32x4_t o_acc[8],
    const uint16_t* __restrict__ cos_row, const uint16_t* __restrict__ sin_row) {
    const int r = lane & 15, g = lane >> 4, h = h_base + r;
    if (h >= H) return;
    uint16_t* row = out_ptr + out_base + static_cast<int64_t>(h) * out_stride_h + w * 128 + 4 * g;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        float f0 = o_acc[dt][0] * scale_out, f1 = o_acc[dt][1] * scale_out;
        float f2 = o_acc[dt][2] * scale_out, f3 = o_acc[dt][3] * scale_out;
        if constexpr (FuseInvRope) {
            if (w == 3 && dt >= 4) {
                f0 = bf16_to_f32(f32_to_bf16_rne(f0)); f1 = bf16_to_f32(f32_to_bf16_rne(f1));
                f2 = bf16_to_f32(f32_to_bf16_rne(f2)); f3 = bf16_to_f32(f32_to_bf16_rne(f3));
                const int p0 = (dt - 4) * 8 + 2 * g, p1 = p0 + 1;
                const float c0 = bf16_to_f32(cos_row[p0]), s0 = bf16_to_f32(sin_row[p0]);
                const float c1 = bf16_to_f32(cos_row[p1]), s1 = bf16_to_f32(sin_row[p1]);
                const float r0 = f0 * c0 + f1 * s0, r1 = -f0 * s0 + f1 * c0;
                const float r2 = f2 * c1 + f3 * s1, r3 = -f2 * s1 + f3 * c1;
                f0 = r0; f1 = r1; f2 = r2; f3 = r3;
            }
        }
        *reinterpret_cast<uint2*>(row + dt * 16) = pack_bf16x4(f0, f1, f2, f3);
    }
}

template <bool FuseInvRope>
__global__ void __launch_bounds__(256, 3)
sparse_attn_v4_unsplit_4w_kernel(
    const uint16_t* __restrict__ q_ptr, const uint16_t* __restrict__ unified_kv_ptr,
    const int* __restrict__ kv_indices_prefix, const int* __restrict__ kv_indptr_prefix,
    const uint16_t* __restrict__ kv_ptr, const int* __restrict__ kv_indices_extend,
    const int* __restrict__ kv_indptr_extend, const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out_ptr,
    int64_t q_stride_t, int64_t q_stride_h, int64_t pkv_stride_n, int64_t ekv_stride_n,
    int64_t out_stride_t, int64_t out_stride_h, int H, float softmax_scale,
    const uint16_t* __restrict__ cos_cache, const uint16_t* __restrict__ sin_cache,
    const int64_t* __restrict__ positions, int64_t cos_stride_s) {
    __shared__ Smem4W smem;
    const int t = blockIdx.x, hb = blockIdx.y, h_base = hb * kHeadTile;
    const int w = threadIdx.x >> 6, lane = threadIdx.x & 63, h = h_base + (lane & 15);
    const bool head_valid = (h < H);
    bf16x4_t q_reg[8];
    load_q_slice<8>(q_ptr, static_cast<int64_t>(t) * q_stride_t, q_stride_h, H, h_base, w, lane, q_reg);
    fp32x4_t o_acc[8];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) o_acc[dt] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_i = kNegLarge, l_i = 0.0f;
    const float qk_scale = softmax_scale * kLog2E;

    const int p_start = kv_indptr_prefix[t], p_len = kv_indptr_prefix[t + 1] - p_start;
    if (p_len > 0) walk_region_4w(smem, unified_kv_ptr, pkv_stride_n, kv_indices_prefix, p_start, 0, p_len, w, lane, head_valid, qk_scale, q_reg, m_i, l_i, o_acc);
    const int e_start = kv_indptr_extend[t], e_len = kv_indptr_extend[t + 1] - e_start;
    if (e_len > 0) walk_region_4w(smem, kv_ptr, ekv_stride_n, kv_indices_extend, e_start, 0, e_len, w, lane, head_valid, qk_scale, q_reg, m_i, l_i, o_acc);

    const float sink_log2 = head_valid ? (attn_sink[h] * kLog2E) : kNegLarge;
    const float m_final = fmaxf(m_i, sink_log2);
    const float alpha = (m_i == kNegLarge) ? 0.0f : exp2f(m_i - m_final);
    const float l_final = l_i * alpha + ((sink_log2 == kNegLarge) ? 0.0f : exp2f(sink_log2 - m_final));
    const float scale_out = (l_final > 0.0f) ? (alpha / fmaxf(l_final, 1.0e-30f)) : 0.0f;

    const uint16_t* cos_row = nullptr; const uint16_t* sin_row = nullptr;
    if constexpr (FuseInvRope) {
        if (w == 3) {
            const int64_t pos = positions[t];
            cos_row = cos_cache + pos * cos_stride_s;
            sin_row = sin_cache + pos * cos_stride_s;
        }
    }
    store_output_4w<FuseInvRope>(out_ptr, static_cast<int64_t>(t) * out_stride_t, out_stride_h, H, h_base, w, lane, scale_out, o_acc, cos_row, sin_row);
}

__global__ void __launch_bounds__(256, 3)
sparse_attn_v4_split_stage_4w_kernel(
    const uint16_t* __restrict__ q_ptr, const uint16_t* __restrict__ unified_kv_ptr,
    const int* __restrict__ kv_indices_prefix, const int* __restrict__ kv_indptr_prefix,
    const uint16_t* __restrict__ kv_ptr, const int* __restrict__ kv_indices_extend,
    const int* __restrict__ kv_indptr_extend,
    int64_t q_stride_t, int64_t q_stride_h, int64_t pkv_stride_n, int64_t ekv_stride_n,
    int H, int H_blocks, int num_splits, float softmax_scale,
    float2* __restrict__ part_ml, fp32x4_t* __restrict__ part_acc) {
    __shared__ Smem4W smem;
    const int t = blockIdx.x, hb = blockIdx.y, s = blockIdx.z;
    const int h_base = hb * kHeadTile, w = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int r = lane & 15, g = lane >> 4, h = h_base + r;
    const bool head_valid = (h < H);
    const int p_start = kv_indptr_prefix[t], p_len = kv_indptr_prefix[t + 1] - p_start;
    const int e_start = kv_indptr_extend[t], e_len = kv_indptr_extend[t + 1] - e_start;
    const int total = p_len + e_len;
    const int chunk = (((total + num_splits - 1) / num_splits) + 15) & ~15;
    const int lo = s * chunk, hi = (lo + chunk < total) ? (lo + chunk) : total;

    fp32x4_t o_acc[8];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) o_acc[dt] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_i = kNegLarge, l_i = 0.0f;

    if (lo < hi) {
        bf16x4_t q_reg[8];
        load_q_slice<8>(q_ptr, static_cast<int64_t>(t) * q_stride_t, q_stride_h, H, h_base, w, lane, q_reg);
        const float qk_scale = softmax_scale * kLog2E;
        const int p_hi = (hi < p_len) ? hi : p_len;
        if (lo < p_hi) walk_region_4w(smem, unified_kv_ptr, pkv_stride_n, kv_indices_prefix, p_start, lo, p_hi, w, lane, head_valid, qk_scale, q_reg, m_i, l_i, o_acc);
        const int e_lo = ((lo > p_len) ? lo : p_len) - p_len, e_hi = hi - p_len;
        if (e_lo < e_hi) walk_region_4w(smem, kv_ptr, ekv_stride_n, kv_indices_extend, e_start, e_lo, e_hi, w, lane, head_valid, qk_scale, q_reg, m_i, l_i, o_acc);
    }

    const int group = t * H_blocks + hb;
    if (w == 0 && g == 0) {
        part_ml[(static_cast<int64_t>(group) * num_splits + s) * kHeadTile + r] = make_float2(m_i, l_i);
    }
    // part_acc layout: [group, tile_d (32), split (num_splits), lane (64)]
    // so each of the 32 combine WGs reads 1 contiguous float4 per split!
    const int64_t grp_off = static_cast<int64_t>(group) * 32 * num_splits;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        const int td = w * 8 + dt;
        part_acc[(grp_off + static_cast<int64_t>(td) * num_splits + s) * kWave + lane] = o_acc[dt];
    }
}

// 32 parallel 64-thread WGs per (t, hb), 16 dims (1 float4/lane) per WG!
// Full unroll of up to 20 splits uses only ~90 VGPRs -> ZERO scratch, 100% pipelined L2 reads!
template <bool FuseInvRope, int NumSplitsConst>
__global__ void __launch_bounds__(kWave, 4)
sparse_attn_v4_combine_32wg_kernel(
    const float2* __restrict__ part_ml, const fp32x4_t* __restrict__ part_acc,
    const float* __restrict__ attn_sink, uint16_t* __restrict__ out_ptr,
    int64_t out_stride_t, int64_t out_stride_h, int H, int H_blocks, int num_splits_dyn,
    const uint16_t* __restrict__ cos_cache, const uint16_t* __restrict__ sin_cache,
    const int64_t* __restrict__ positions, int64_t cos_stride_s) {
    const int t = blockIdx.x, hb = blockIdx.y, td = blockIdx.z;
    const int lane = threadIdx.x, r = lane & 15, g = lane >> 4;
    const int h_base = hb * kHeadTile, h = h_base + r;
    if (h >= H) return;

    const int ns = (NumSplitsConst > 0) ? NumSplitsConst : num_splits_dyn;
    const int group = t * H_blocks + hb;
    const int64_t ml_base = static_cast<int64_t>(group) * ns;
    const float sink_log2 = attn_sink[h] * kLog2E;
    float m_final = sink_log2;
    float l_final = 0.0f;
    fp32x4_t acc = {0.f, 0.f, 0.f, 0.f};
    const int64_t acc_base = (static_cast<int64_t>(group) * 32 + td) * ns;

    if constexpr (NumSplitsConst > 0) {
        float2 mls[NumSplitsConst];
        #pragma unroll
        for (int k = 0; k < NumSplitsConst; ++k) mls[k] = part_ml[(ml_base + k) * kHeadTile + r];
        #pragma unroll
        for (int k = 0; k < NumSplitsConst; ++k) m_final = fmaxf(m_final, mls[k].x);
        l_final = (sink_log2 == kNegLarge) ? 0.0f : exp2f(sink_log2 - m_final);
        #pragma unroll
        for (int k = 0; k < NumSplitsConst; ++k) {
            const float w = (mls[k].x == kNegLarge) ? 0.0f : exp2f(mls[k].x - m_final);
            l_final += mls[k].y * w;
            acc += part_acc[(acc_base + k) * kWave + lane] * w;
        }
    } else {
        for (int k = 0; k < ns; ++k) m_final = fmaxf(m_final, part_ml[(ml_base + k) * kHeadTile + r].x);
        l_final = (sink_log2 == kNegLarge) ? 0.0f : exp2f(sink_log2 - m_final);
        for (int k = 0; k < ns; ++k) {
            const float2 ml = part_ml[(ml_base + k) * kHeadTile + r];
            const float w = (ml.x == kNegLarge) ? 0.0f : exp2f(ml.x - m_final);
            l_final += ml.y * w;
            acc += part_acc[(acc_base + k) * kWave + lane] * w;
        }
    }

    const float scale_out = (l_final > 0.0f) ? (1.0f / fmaxf(l_final, 1.0e-30f)) : 0.0f;
    float f0 = acc[0] * scale_out, f1 = acc[1] * scale_out;
    float f2 = acc[2] * scale_out, f3 = acc[3] * scale_out;
    if constexpr (FuseInvRope) {
        if (td >= 28) {
            const int64_t pos = positions[t];
            const uint16_t* cos_row = cos_cache + pos * cos_stride_s;
            const uint16_t* sin_row = sin_cache + pos * cos_stride_s;
            f0 = bf16_to_f32(f32_to_bf16_rne(f0)); f1 = bf16_to_f32(f32_to_bf16_rne(f1));
            f2 = bf16_to_f32(f32_to_bf16_rne(f2)); f3 = bf16_to_f32(f32_to_bf16_rne(f3));
            const int p0 = (td - 28) * 8 + 2 * g, p1 = p0 + 1;
            const float c0 = bf16_to_f32(cos_row[p0]), s0 = bf16_to_f32(sin_row[p0]);
            const float c1 = bf16_to_f32(cos_row[p1]), s1 = bf16_to_f32(sin_row[p1]);
            const float r0 = f0 * c0 + f1 * s0, r1 = -f0 * s0 + f1 * c0;
            const float r2 = f2 * c1 + f3 * s1, r3 = -f2 * s1 + f3 * c1;
            f0 = r0; f1 = r1; f2 = r2; f3 = r3;
        }
    }
    uint16_t* dst = out_ptr + static_cast<int64_t>(t) * out_stride_t +
                    static_cast<int64_t>(h) * out_stride_h + td * 16 + 4 * g;
    *reinterpret_cast<uint2*>(dst) = pack_bf16x4(f0, f1, f2, f3);
}

inline int64_t align256(int64_t n) { return (n + 255) & ~int64_t(255); }
}  // namespace

extern "C" {

int sparse_attn_v4_gfx942_splits(int num_tokens, int num_heads, int avg_keys, int num_cus) {
    if (num_tokens <= 0 || num_heads <= 0 || avg_keys <= 16) return 1;
    const int h_blocks = (num_heads + kHeadTile - 1) / kHeadTile;
    const int base_ctas = num_tokens * h_blocks;
    if (num_cus <= 0) num_cus = 80;
    // 4-wave 20 KB LDS kernel runs 3 WGs/CU = 240 CTAs on MI308X.
    if (base_ctas >= 3 * num_cus) return 1;
    if (avg_keys <= 128) {
        // SWA-only / short: split for small M (s=8 at M=1..4, s=4 at M=8..16, s=3 at M=24),
        // stay unsplit once base_ctas >= 32 (M=48..64).
        if (base_ctas >= 32) return 1;
        int want = (base_ctas <= 4) ? 8 : (base_ctas <= 16 ? 4 : 3);
        int cap = avg_keys / 16;
        int s = (want < cap) ? want : cap;
        return (s < 1) ? 1 : s;
    }
    // CSA+SWA: fill 1 wave (80 CTAs) for M<=8 (M=1,4 -> s=20; M=8 -> s=10),
    // and 3 WGs/CU (240 CTAs) for M=16..64 (M=16,24 -> s=10; M=48 -> s=5; M=64 -> s=3).
    int want = (base_ctas <= 8) ? (num_cus / base_ctas) : ((3 * num_cus) / base_ctas);
    if (base_ctas >= 16 && want > 10) want = 10;
    if (want > 20) want = 20;
    int cap = avg_keys / 32;
    int s = (want < cap) ? want : cap;
    if (s > kMaxSplits) s = kMaxSplits;
    if (s < 1) s = 1;
    return s;
}

long long sparse_attn_v4_gfx942_workspace_bytes(int num_tokens, int num_heads, int num_splits) {
    if (num_splits <= 1 || num_tokens <= 0 || num_heads <= 0) return 0;
    const int h_blocks = (num_heads + kHeadTile - 1) / kHeadTile;
    const int64_t groups = static_cast<int64_t>(num_tokens) * h_blocks;
    const int64_t ml_bytes = align256(groups * num_splits * kHeadTile * static_cast<int64_t>(sizeof(float2)));
    const int64_t acc_bytes = align256(groups * 8 * num_splits * 4 * kWave * static_cast<int64_t>(sizeof(fp32x4_t)));
    return static_cast<long long>(ml_bytes + acc_bytes);
}

int sparse_attn_v4_gfx942_launch(
    const void* q_ptr, const void* unified_kv_ptr,
    const void* kv_indices_prefix, const void* kv_indptr_prefix,
    const void* kv_ptr, const void* kv_indices_extend, const void* kv_indptr_extend,
    const void* attn_sink, void* out_ptr,
    long long q_stride_t, long long q_stride_h, long long pkv_stride_n, long long ekv_stride_n,
    long long out_stride_t, long long out_stride_h,
    int num_tokens, int num_heads, int num_splits, float softmax_scale, void* workspace,
    const void* cos_cache, const void* sin_cache, const void* positions,
    long long cos_stride_s, int rope_dim, void* stream_ptr) {
    if (num_tokens <= 0 || num_heads <= 0) return 0;
    if (num_splits < 1 || num_splits > kMaxSplits) return -1;
    const bool fuse_irope = (cos_cache != nullptr && sin_cache != nullptr && positions != nullptr && rope_dim == kRopeDim);
    const int h_blocks = (num_heads + kHeadTile - 1) / kHeadTile;
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);

    if (num_splits == 1) {
        dim3 grid(static_cast<unsigned>(num_tokens), static_cast<unsigned>(h_blocks), 1);
        auto fn = fuse_irope ? sparse_attn_v4_unsplit_4w_kernel<true> : sparse_attn_v4_unsplit_4w_kernel<false>;
        hipLaunchKernelGGL(
            fn, grid, dim3(256, 1, 1), 0, stream,
            static_cast<const uint16_t*>(q_ptr), static_cast<const uint16_t*>(unified_kv_ptr),
            static_cast<const int*>(kv_indices_prefix), static_cast<const int*>(kv_indptr_prefix),
            static_cast<const uint16_t*>(kv_ptr), static_cast<const int*>(kv_indices_extend),
            static_cast<const int*>(kv_indptr_extend), static_cast<const float*>(attn_sink),
            static_cast<uint16_t*>(out_ptr), q_stride_t, q_stride_h, pkv_stride_n, ekv_stride_n,
            out_stride_t, out_stride_h, num_heads, softmax_scale,
            static_cast<const uint16_t*>(cos_cache), static_cast<const uint16_t*>(sin_cache),
            static_cast<const int64_t*>(positions), cos_stride_s);
        return static_cast<int>(hipGetLastError());
    }

    if (workspace == nullptr) return -2;
    const int64_t groups = static_cast<int64_t>(num_tokens) * h_blocks;
    const int64_t ml_bytes = align256(groups * num_splits * kHeadTile * static_cast<int64_t>(sizeof(float2)));
    auto* base = static_cast<uint8_t*>(workspace);
    auto* part_ml = reinterpret_cast<float2*>(base);
    auto* part_acc = reinterpret_cast<fp32x4_t*>(base + ml_bytes);

    dim3 grid1(static_cast<unsigned>(num_tokens), static_cast<unsigned>(h_blocks), static_cast<unsigned>(num_splits));
    hipLaunchKernelGGL(
        sparse_attn_v4_split_stage_4w_kernel, grid1, dim3(256, 1, 1), 0, stream,
        static_cast<const uint16_t*>(q_ptr), static_cast<const uint16_t*>(unified_kv_ptr),
        static_cast<const int*>(kv_indices_prefix), static_cast<const int*>(kv_indptr_prefix),
        static_cast<const uint16_t*>(kv_ptr), static_cast<const int*>(kv_indices_extend),
        static_cast<const int*>(kv_indptr_extend),
        q_stride_t, q_stride_h, pkv_stride_n, ekv_stride_n,
        num_heads, h_blocks, num_splits, softmax_scale, part_ml, part_acc);

    dim3 grid2(static_cast<unsigned>(num_tokens), static_cast<unsigned>(h_blocks), 32);
    #define LAUNCH_COMB(NS) \
        do { \
            auto fn2 = fuse_irope ? sparse_attn_v4_combine_32wg_kernel<true, NS> \
                                  : sparse_attn_v4_combine_32wg_kernel<false, NS>; \
            hipLaunchKernelGGL( \
                fn2, grid2, dim3(kWave, 1, 1), 0, stream, \
                part_ml, part_acc, static_cast<const float*>(attn_sink), static_cast<uint16_t*>(out_ptr), \
                out_stride_t, out_stride_h, num_heads, h_blocks, num_splits, \
                static_cast<const uint16_t*>(cos_cache), static_cast<const uint16_t*>(sin_cache), \
                static_cast<const int64_t*>(positions), cos_stride_s); \
        } while (0)
    switch (num_splits) {
        case 2: LAUNCH_COMB(2); break;
        case 3: LAUNCH_COMB(3); break;
        case 4: LAUNCH_COMB(4); break;
        case 5: LAUNCH_COMB(5); break;
        case 6: LAUNCH_COMB(6); break;
        case 8: LAUNCH_COMB(8); break;
        case 10: LAUNCH_COMB(10); break;
        case 16: LAUNCH_COMB(16); break;
        case 20: LAUNCH_COMB(20); break;
        default: LAUNCH_COMB(0); break;
    }
    #undef LAUNCH_COMB
    return static_cast<int>(hipGetLastError());
}

}  // extern "C"

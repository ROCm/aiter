// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1201 Sol sparse attention, Phase A: zeroth-order proxy (pooled INT8 keys, summed FP8 values) over the
// key blocks outside each workgroup's exact union, emitted as the initial online-softmax state of the ASM core.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x8_t = int __attribute__((ext_vector_type(8)));
using float32x8_t = float __attribute__((ext_vector_type(8)));

static constexpr int WARP = 32;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WMMA_LANE_K = 8;
static constexpr int BLOCK_M = 512;
static constexpr int BLOCK_N = 32;
static constexpr int HEAD_DIM = 128;
static constexpr int RG = 2;                              // 16-row WMMA groups per wave
static constexpr int NUM_WAVES = BLOCK_M / (16 * RG);     // 16
static constexpr int BLOCK_SIZE = NUM_WAVES * WARP;       // 512
static constexpr int K_STEPS_QK = HEAD_DIM / WMMA_K;      // 8
static constexpr int D_CHUNKS = HEAD_DIM / WMMA_N;        // 8
static constexpr int NUM_S_VALS = 16;
static constexpr int LDS_PAD = 8;
static constexpr int K_STRIDE = HEAD_DIM + LDS_PAD;       // 136
static constexpr int V_STRIDE = BLOCK_N + LDS_PAD;        // 40
static constexpr int LDS_K_TILE = BLOCK_N * K_STRIDE;
static constexpr int LDS_V_TILE = HEAD_DIM * V_STRIDE;
static constexpr int LDS_BUF = LDS_K_TILE + LDS_V_TILE;
static constexpr float FP8_P_OFFSET = 8.807f;
static constexpr float NEG_BIG = -3.0e38f;

__device__ __forceinline__ int32x2_t pack_i8x8(const int8_t* p) {
  int32x2_t out;
  out[0] = *reinterpret_cast<const int32_t*>(p);
  out[1] = *reinterpret_cast<const int32_t*>(p + 4);
  return out;
}
__device__ __forceinline__ int32x8_t wmma_qk_i8(int32x2_t a, int32x2_t b, int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
__device__ __forceinline__ float32x8_t wmma_pv_fp8(int32x2_t a, int32x2_t b, float32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
__device__ __forceinline__ int32x2_t pack_fp8_prob(const float p[8]) {
  int32_t lo = 0, hi = 0;
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[0], p[1], lo, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[2], p[3], lo, true);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[4], p[5], hi, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[6], p[7], hi, true);
  int32x2_t out;
  out[0] = lo;
  out[1] = hi;
  return out;
}

struct SolParams {
  const int8_t* Q; const int8_t* K; const int8_t* V;
  const float* QScale; const float* KScale; const float* VScale;
  const int8_t* KB; const int8_t* VB; const float* KBScale; const float* VBScale; const float* Cnt;
  const int32_t* TileList; const int32_t* TileCount; const uint32_t* Mask;
  float* State;
  int padded_seq_len, valid_seq_len, num_heads, nkb, nkbp, list_stride;
};

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 2)
void gfx1201_sol_phase_a(SolParams p) {
  __shared__ int8_t lds_all[2 * LDS_BUF];
  int8_t* lds = lds_all;
  const int tid = (int)threadIdx.x;
  const int wave_id = tid / WARP;
  const int lane = tid % WARP;
  const int lane16 = lane % 16;
  const int klane = lane / 16;
  const int H = p.num_heads;
  const int nqt = (p.valid_seq_len + BLOCK_M - 1) / BLOCK_M;  // ASM order: wg = head * nqt + qtile
  const int head = (int)blockIdx.x / nqt;
  const int qt = (int)blockIdx.x % nqt;
  const int seq = p.padded_seq_len;
  const long long wg = (long long)blockIdx.x;

  int q_row[RG];
  bool q_ok[RG];
  #pragma unroll
  for (int rg = 0; rg < RG; ++rg) {
    q_row[rg] = qt * BLOCK_M + wave_id * 32 + rg * 16 + lane16;
    q_ok[rg] = q_row[rg] < p.valid_seq_len;
  }
  int32x2_t q_frags[RG][K_STEPS_QK];
  #pragma unroll
  for (int rg = 0; rg < RG; ++rg)
    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks) {
      int8_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (q_ok[rg])
        *reinterpret_cast<int64_t*>(tmp) = *reinterpret_cast<const int64_t*>(
            p.Q + (((long long)q_row[rg] * H + head) * HEAD_DIM + ks * WMMA_K + klane * WMMA_LANE_K));
      q_frags[rg][ks] = pack_i8x8(tmp);
    }
  const int row0 = qt * BLOCK_M + wave_id * 32;  // both groups share one 32-row Q scale
  const float q_scale = p.QScale[(long long)head * (seq / 32) + (row0 < p.valid_seq_len ? row0 / 32 : 0)];

  float m_run[RG], l_run[RG];
  float32x8_t o_acc[RG][D_CHUNKS];
  #pragma unroll
  for (int rg = 0; rg < RG; ++rg) {
    m_run[rg] = -1.0e30f;
    l_run[rg] = 0.f;
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) o_acc[rg][dc] = (float32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
  }

  // Thread t < 256 moves one int4 of the K tile, t >= 256 one int4 of the V^T tile.
  const bool is_k = tid < 256;
  const int k_r = (tid & 255) / 8, k_c = ((tid & 255) % 8) * 16;
  const int v_d = (tid & 255) / 2, v_off = ((tid & 255) % 2) * 16;
  auto fetch = [&](const int8_t* Kt, const int8_t* Vt, int tok, int k_rows, int v_cols) -> int4 {
    if (tok >= k_rows) return make_int4(0, 0, 0, 0);
    return is_k ? *reinterpret_cast<const int4*>(Kt + (((long long)(tok + k_r) * H + head) * HEAD_DIM + k_c))
                : *reinterpret_cast<const int4*>(Vt + (((long long)head * HEAD_DIM + v_d) * v_cols + tok + v_off));
  };
  auto commit = [&](int buf, int4 val) {
    int8_t* b = lds_all + buf * LDS_BUF;
    if (is_k) *reinterpret_cast<int4*>(b + k_r * K_STRIDE + k_c) = val;
    else *reinterpret_cast<int4*>(b + LDS_K_TILE + v_d * V_STRIDE + v_off) = val;
  };

  auto qk_tile = [&](float raw[RG][NUM_S_VALS]) {
    int32x8_t s[RG][2];
    #pragma unroll
    for (int rg = 0; rg < RG; ++rg) s[rg][0] = s[rg][1] = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks) {
      const int k_col = ks * WMMA_K + klane * WMMA_LANE_K;
      int32x2_t a = pack_i8x8(lds + lane16 * K_STRIDE + k_col);
      int32x2_t b = pack_i8x8(lds + (lane16 + 16) * K_STRIDE + k_col);
      #pragma unroll
      for (int rg = 0; rg < RG; ++rg) {
        s[rg][0] = wmma_qk_i8(a, q_frags[rg][ks], s[rg][0]);
        s[rg][1] = wmma_qk_i8(b, q_frags[rg][ks], s[rg][1]);
      }
    }
    #pragma unroll
    for (int rg = 0; rg < RG; ++rg)
      #pragma unroll
      for (int i = 0; i < 8; ++i) { raw[rg][i] = (float)s[rg][0][i]; raw[rg][8 + i] = (float)s[rg][1][i]; }
  };

  auto softmax_pv = [&](float raw[RG][NUM_S_VALS], float score_scale, const float* weight) {
    int32x2_t pf[RG][2];
    #pragma unroll
    for (int rg = 0; rg < RG; ++rg) {
      float local_max = raw[rg][0];
      #pragma unroll
      for (int i = 1; i < NUM_S_VALS; ++i) local_max = fmaxf(local_max, raw[rg][i]);
      local_max = fmaxf(local_max, __shfl_xor(local_max, 16, WARP));
      const float row_max = local_max <= NEG_BIG ? -1.0e30f : fmaf(local_max, score_scale, -FP8_P_OFFSET);
      const float m_new = fmaxf(m_run[rg], row_max);
      const float corr = __builtin_amdgcn_exp2f(m_run[rg] - m_new);
      float pv[NUM_S_VALS];
      float local_sum = 0.f;
      #pragma unroll
      for (int i = 0; i < NUM_S_VALS; ++i) {
        const float e = raw[rg][i] <= NEG_BIG ? 0.f : __builtin_amdgcn_exp2f(fmaf(raw[rg][i], score_scale, -m_new));
        pv[i] = e;
        local_sum += weight ? e * weight[i] : e;
      }
      l_run[rg] = corr * l_run[rg] + local_sum + __shfl_xor(local_sum, 16, WARP);
      m_run[rg] = m_new;
      if (corr != 1.f) {
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; ++dc)
          #pragma unroll
          for (int i = 0; i < 8; ++i) o_acc[rg][dc][i] *= corr;
      }
      pf[rg][0] = pack_fp8_prob(&pv[0]);
      pf[rg][1] = pack_fp8_prob(&pv[8]);
    }
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) {
      const int d_pos = dc * WMMA_N + lane16;
      int32x2_t v0 = pack_i8x8(lds + LDS_K_TILE + d_pos * V_STRIDE + klane * WMMA_LANE_K);
      int32x2_t v1 = pack_i8x8(lds + LDS_K_TILE + d_pos * V_STRIDE + 16 + klane * WMMA_LANE_K);
      #pragma unroll
      for (int rg = 0; rg < RG; ++rg) {
        o_acc[rg][dc] = wmma_pv_fp8(v0, pf[rg][0], o_acc[rg][dc]);
        o_acc[rg][dc] = wmma_pv_fp8(v1, pf[rg][1], o_acc[rg][dc]);
      }
    }
  };

  const uint32_t* mask_row = p.Mask + wg * (p.nkbp / 32);
  const int n_pool = p.nkbp / 32;
  bool any_approx = false;
  for (int t = 0; t < n_pool && !any_approx; ++t) any_approx = mask_row[t] != 0xffffffffu;

  // Phase A: proxy correction over 64-key blocks outside this workgroup's exact union.
  if (any_approx) {
    int4 reg = fetch(p.KB, p.VB, 0, p.nkbp, p.nkbp);
    for (int t = 0; t < n_pool; ++t) {
      __syncthreads();
      commit(t & 1, reg);
      if (t + 1 < n_pool) reg = fetch(p.KB, p.VB, (t + 1) * 32, p.nkbp, p.nkbp);
      __syncthreads();
      lds = lds_all + (t & 1) * LDS_BUF;
      const uint32_t bits = mask_row[t];
      if (bits == 0xffffffffu) continue;
      float raw[RG][NUM_S_VALS], w[NUM_S_VALS];
      qk_tile(raw);
      #pragma unroll
      for (int i = 0; i < NUM_S_VALS; ++i) {
        const int j = (i / 8) * 16 + klane * 8 + (i % 8);
        const int blk = t * 32 + j;
        const bool use = blk < p.nkb && !((bits >> j) & 1u);
        w[i] = use ? p.Cnt[blk] : 0.f;
        if (!use) {
          #pragma unroll
          for (int rg = 0; rg < RG; ++rg) raw[rg][i] = NEG_BIG;
        }
      }
      softmax_pv(raw, q_scale * p.KBScale[(long long)head * n_pool + t], w);
    }
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int c = dc * WMMA_N + klane * 8 + i;
        const float vs = p.VScale[(long long)head * HEAD_DIM + c];
        const float ratio = vs > 0.f ? p.VBScale[(long long)head * HEAD_DIM + c] / vs : 0.f;
        #pragma unroll
        for (int rg = 0; rg < RG; ++rg) o_acc[rg][dc][i] *= ratio;
      }
  }

  // Emit the online-softmax state in the ASM lane-native layout:
  // [wg][wave][chunk 0..32][lane][4]; reg = (1-rg)*64 + (7-dc)*8 + i, chunk = reg/4;
  // chunk 32 = (l RG1, l RG0, m RG1, m RG0). m in exp2 units incl. the FP8 offset.
  float* st = p.State + ((wg * NUM_WAVES + wave_id) * 33) * 128 + lane * 4;
  #pragma unroll
  for (int rg = 0; rg < RG; ++rg)
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      #pragma unroll
      for (int i4 = 0; i4 < 2; ++i4) {
        const int reg = (1 - rg) * 64 + (7 - dc) * 8 + i4 * 4;
        float4 val = make_float4(o_acc[rg][dc][i4 * 4], o_acc[rg][dc][i4 * 4 + 1], o_acc[rg][dc][i4 * 4 + 2],
                                 o_acc[rg][dc][i4 * 4 + 3]);
        *reinterpret_cast<float4*>(st + (reg / 4) * 128) = val;
      }
  *reinterpret_cast<float4*>(st + 32 * 128) =
      any_approx ? make_float4(l_run[1], l_run[0], m_run[1], m_run[0]) : make_float4(0.f, 0.f, -INFINITY, -INFINITY);
}

void launch_gfx1201_sol_phase_a(int64_t Q, int64_t QScale, int64_t VScale, int64_t KB, int64_t VB, int64_t KBScale,
                        int64_t VBScale, int64_t Cnt, int64_t Mask, int64_t State, int64_t padded_seq_len,
                        int64_t valid_seq_len, int64_t num_heads, int64_t nkb, int64_t nkbp, int64_t n_wg,
                        int64_t stream) {
  SolParams p{};
  p.Q = (const int8_t*)Q; p.QScale = (const float*)QScale; p.VScale = (const float*)VScale;
  p.KB = (const int8_t*)KB; p.VB = (const int8_t*)VB; p.KBScale = (const float*)KBScale;
  p.VBScale = (const float*)VBScale; p.Cnt = (const float*)Cnt; p.Mask = (const uint32_t*)Mask;
  p.State = (float*)State;
  p.padded_seq_len = (int)padded_seq_len; p.valid_seq_len = (int)valid_seq_len; p.num_heads = (int)num_heads;
  p.nkb = (int)nkb; p.nkbp = (int)nkbp;
  hipLaunchKernelGGL(gfx1201_sol_phase_a, dim3((int)n_wg), dim3(BLOCK_SIZE), 0, (hipStream_t)stream, p);
}

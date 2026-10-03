// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

using int32x2_t = int __attribute__((ext_vector_type(2)));
using int32x8_t = int __attribute__((ext_vector_type(8)));
using float32x8_t = float __attribute__((ext_vector_type(8)));

#if defined(SAGE_HIP_USE_ASM_WAIT)
#define SAGE_WAIT_LOAD() asm volatile("s_wait_loadcnt 0x0" ::: "memory")
#define SAGE_WAIT_DSCNT() asm volatile("s_wait_dscnt 0x0" ::: "memory")
#else
#define SAGE_WAIT_LOAD() ((void)0)
#define SAGE_WAIT_DSCNT() ((void)0)
#endif

static constexpr int WARP = 32;
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;
static constexpr int WMMA_LANE_K = 8;
static constexpr int BLOCK_M = 512;
static constexpr int BLOCK_N = 32;
static constexpr int HEAD_DIM = 128;
static constexpr int ROWS_PER_WAVE = 32;
static constexpr int ROW_GROUPS = ROWS_PER_WAVE / WMMA_M;  // 1
static constexpr int NUM_WAVES = BLOCK_M / ROWS_PER_WAVE;  // 8
static constexpr int BLOCK_SIZE = NUM_WAVES * WARP;        // 256
static constexpr int K_THREADS_PER_ROW = HEAD_DIM / 16;    // 8
static constexpr int K_LOAD_ITEMS = BLOCK_N * K_THREADS_PER_ROW;  // 256
static constexpr int K_LOAD_BATCHES = (K_LOAD_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 2
static constexpr int V_CHUNKS_PER_ROW = BLOCK_N / 16;      // 2
static constexpr int V_LOAD_ITEMS = HEAD_DIM * V_CHUNKS_PER_ROW;  // 256
static constexpr int V_LOAD_BATCHES = (V_LOAD_ITEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 2
static constexpr int K_SUB_N = 32;
static constexpr int N_SUB_TILES = BLOCK_N / K_SUB_N;      // 1
static constexpr int NUM_S_ACCS = N_SUB_TILES * 2;          // 2
static constexpr int NUM_S_VALS = NUM_S_ACCS * 8;           // 16
static constexpr int K_STEPS_QK = HEAD_DIM / WMMA_K;       // 8
static constexpr int D_CHUNKS = HEAD_DIM / WMMA_N;         // 8
static constexpr int PV_K_STEPS = K_SUB_N / WMMA_K;        // 2
static constexpr int Q_SCALE_ROWS = 32;
static constexpr int VEC_WIDTH = 16;
static constexpr int LDS_PAD = 8;
static constexpr int K_STRIDE = HEAD_DIM + LDS_PAD;        // 136
static constexpr int V_STRIDE = BLOCK_N + LDS_PAD;         // 40
static constexpr int LDS_K_TILE = BLOCK_N * K_STRIDE;      // 4352
static constexpr int LDS_V_TILE = HEAD_DIM * V_STRIDE;     // 5120
static constexpr int LDS_BUFFERS = 2;
static constexpr int LDS_V_BASE = LDS_BUFFERS * LDS_K_TILE;
static constexpr float FP8_P_OFFSET = 8.807f;

__device__ __forceinline__ int32x2_t pack_i8x8(const int8_t* p) {
  int32x2_t out;
  out[0] = *reinterpret_cast<const int32_t*>(p);
  out[1] = *reinterpret_cast<const int32_t*>(p + 4);
  return out;
}

__device__ __forceinline__ int32x8_t wmma_qk_i8(int32x2_t a, int32x2_t b, int32x8_t c) {
#if defined(__gfx1201__) || defined(__gfx12__)
  // LLVM: (A_signed, A, B_signed, B, C, clamp) — signed to match FlyDSL.
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
      true, a, true, b, c, false);
#else
  (void)a; (void)b; return c;
#endif
}

__device__ __forceinline__ float32x8_t wmma_pv_fp8(int32x2_t a, int32x2_t b, float32x8_t c) {
#if defined(__gfx1201__) || defined(__gfx12__)
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
  (void)a; (void)b; return c;
#endif
}

__device__ __forceinline__ int32x2_t pack_fp8_prob(const float p[8]) {
  // Pack 8 f32 probabilities into 8 e4m3 bytes (two i32).
  int32_t lo = 0, hi = 0;
#if defined(__gfx1201__) || defined(__gfx12__)
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[0], p[1], lo, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(p[2], p[3], lo, true);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[4], p[5], hi, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(p[6], p[7], hi, true);
#else
  (void)p;
#endif
  int32x2_t out;
  out[0] = lo;
  out[1] = hi;
  return out;
}

__device__ __forceinline__ float shfl_xor_f(float v, int mask) {
  return __shfl_xor(v, mask, WARP);
}

__device__ __forceinline__ float exp2f_fast(float x) {
  return __builtin_amdgcn_exp2f(x);
}

// Grid: (batch * q_tiles * num_heads). Block: 512 = 16 waves x 32 rows.
// Two LDS buffers: compute reads buf[i%2] while the next tile sits in regs
// and is published to buf[(i+1)%2] behind a single barrier (prod e1 pipeline).
extern "C" __global__
__launch_bounds__(BLOCK_SIZE, 2)
void sage_hip_attn_bm128_bn32(
    const int8_t* __restrict__ Q,
    const int8_t* __restrict__ K,
    const int8_t* __restrict__ V,
    const float* __restrict__ QScale,
    const float* __restrict__ KScale,
    const float* __restrict__ VScale,
    __hip_bfloat16* __restrict__ O,
    int batch_size,
    int padded_seq_len,
    int valid_seq_len,
    int num_heads
) {
  __shared__ int8_t lds[LDS_BUFFERS * (LDS_K_TILE + LDS_V_TILE)];

  const int tid = (int)threadIdx.x;
  const int wave_id = tid / WARP;
  const int lane = tid % WARP;
  const int lane16 = lane % 16;
  const int klane = lane / 16;

  const int q_tiles = (valid_seq_len + BLOCK_M - 1) / BLOCK_M;
  const int block_id = (int)blockIdx.x;
  const int head_idx = block_id % num_heads;
  const int batch_q_tile = block_id / num_heads;
  const int q_tile_idx = batch_q_tile % q_tiles;
  const int batch_idx = batch_q_tile / q_tiles;
  const int q_start = q_tile_idx * BLOCK_M;
  const int seq = padded_seq_len;

  int q_row[ROW_GROUPS];
  bool q_ok[ROW_GROUPS];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    q_row[rg] = q_start + wave_id * ROWS_PER_WAVE + rg * WMMA_M + lane16;
    q_ok[rg] = q_row[rg] < valid_seq_len;
  }

  auto qk_idx = [&](int token, int col) -> long long {
    return ((((long long)batch_idx * seq + token) * num_heads + head_idx) * HEAD_DIM + col);
  };
  auto v_idx = [&](int d, int token) -> long long {
    return ((((long long)batch_idx * num_heads + head_idx) * HEAD_DIM + d) * seq + token);
  };

  int32x2_t q_frags[ROW_GROUPS][K_STEPS_QK];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    const int q_safe = q_ok[rg] ? q_row[rg] : 0;
    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks) {
      const int q_col = ks * WMMA_K + klane * WMMA_LANE_K;
      int8_t tmp[8] = {0,0,0,0,0,0,0,0};
      if (q_ok[rg]) {
        *reinterpret_cast<int64_t*>(tmp) =
            *reinterpret_cast<const int64_t*>(Q + qk_idx(q_safe, q_col));
      }
      q_frags[rg][ks] = pack_i8x8(tmp);
    }
  }

  const int q_scale_blocks = seq / Q_SCALE_ROWS;
  const int q_safe0 = q_ok[0] ? q_row[0] : 0;
  const float q_scale = QScale[((batch_idx * num_heads + head_idx) * q_scale_blocks)
                               + (q_safe0 / Q_SCALE_ROWS)];

  float m_running[ROW_GROUPS], l_running[ROW_GROUPS];
  float32x8_t o_acc[ROW_GROUPS][D_CHUNKS];
  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    m_running[rg] = -INFINITY;
    l_running[rg] = 0.f;
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc)
      o_acc[rg][dc] = (float32x8_t){0,0,0,0,0,0,0,0};
  }

  int4 k_pref[K_LOAD_BATCHES];
  int4 v_pref[V_LOAD_BATCHES];
  auto prefetch_k = [&](int tile) {
    #pragma unroll
    for (int b = 0; b < K_LOAD_BATCHES; ++b) {
      const int item = tid + b * BLOCK_SIZE;
      if (item >= K_LOAD_ITEMS) continue;
      const int k_row = item / K_THREADS_PER_ROW;
      const int k_col = (item % K_THREADS_PER_ROW) * VEC_WIDTH;
      k_pref[b] = *reinterpret_cast<const int4*>(K + qk_idx(tile + k_row, k_col));
    }
  };
  auto prefetch_v = [&](int tile) {
    #pragma unroll
    for (int b = 0; b < V_LOAD_BATCHES; ++b) {
      const int item = tid + b * BLOCK_SIZE;
      if (item >= V_LOAD_ITEMS) continue;
      const int d_row = item / V_CHUNKS_PER_ROW;
      const int token_off = (item % V_CHUNKS_PER_ROW) * VEC_WIDTH;
      v_pref[b] = *reinterpret_cast<const int4*>(V + v_idx(d_row, tile + token_off));
    }
  };
  auto commit_k = [&](int buf) {
    #pragma unroll
    for (int b = 0; b < K_LOAD_BATCHES; ++b) {
      const int item = tid + b * BLOCK_SIZE;
      if (item >= K_LOAD_ITEMS) continue;
      const int k_row = item / K_THREADS_PER_ROW;
      const int k_col = (item % K_THREADS_PER_ROW) * VEC_WIDTH;
      *reinterpret_cast<int4*>(lds + buf * LDS_K_TILE + k_row * K_STRIDE + k_col) = k_pref[b];
    }
  };
  auto commit_v = [&](int buf) {
    #pragma unroll
    for (int b = 0; b < V_LOAD_BATCHES; ++b) {
      const int item = tid + b * BLOCK_SIZE;
      if (item >= V_LOAD_ITEMS) continue;
      const int d_row = item / V_CHUNKS_PER_ROW;
      const int token_off = (item % V_CHUNKS_PER_ROW) * VEC_WIDTH;
      *reinterpret_cast<int4*>(
          lds + LDS_V_BASE + buf * LDS_V_TILE + d_row * V_STRIDE + token_off) = v_pref[b];
    }
  };
  auto safe_tile = [&](int tile) -> int { return tile < seq ? tile : 0; };

  // Tile 0 into buf 0; tile 1 stays in regs until the first loop barrier.
  prefetch_k(0);
  prefetch_v(0);
  commit_k(0);
  commit_v(0);
  prefetch_k(safe_tile(BLOCK_N));
  prefetch_v(safe_tile(BLOCK_N));

  auto process_tile = [&](int kv_start, auto tail_tag) {
    constexpr bool MASK_TAIL = decltype(tail_tag)::value;
    const int cur = (kv_start / BLOCK_N) & 1;
    const int nxt = cur ^ 1;
    __syncthreads();
    // Publish the prefetched tile into the idle buffer, then issue the
    // tile two ahead so those global loads overlap this iteration's WMMA.
    commit_k(nxt);
    commit_v(nxt);
    prefetch_k(safe_tile(kv_start + 2 * BLOCK_N));
    prefetch_v(safe_tile(kv_start + 2 * BLOCK_N));
    const int8_t* k_lds = lds + cur * LDS_K_TILE;
    const int8_t* v_lds = lds + LDS_V_BASE + cur * LDS_V_TILE;

    int32x8_t s_acc[ROW_GROUPS][NUM_S_ACCS];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg)
      #pragma unroll
      for (int i = 0; i < NUM_S_ACCS; ++i)
        s_acc[rg][i] = (int32x8_t){0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int ks = 0; ks < K_STEPS_QK; ++ks) {
      const int k_col = ks * WMMA_K + klane * WMMA_LANE_K;
      int8_t ka[8], kb[8];
      *reinterpret_cast<int64_t*>(ka) =
          *reinterpret_cast<const int64_t*>(k_lds + lane16 * K_STRIDE + k_col);
      *reinterpret_cast<int64_t*>(kb) =
          *reinterpret_cast<const int64_t*>(k_lds + (lane16 + 16) * K_STRIDE + k_col);
      int32x2_t pk_a = pack_i8x8(ka);
      int32x2_t pk_b = pack_i8x8(kb);
      #pragma unroll
      for (int rg = 0; rg < ROW_GROUPS; ++rg) {
        s_acc[rg][0] = wmma_qk_i8(pk_a, q_frags[rg][ks], s_acc[rg][0]);
        s_acc[rg][1] = wmma_qk_i8(pk_b, q_frags[rg][ks], s_acc[rg][1]);
      }
    }

    const int k_scale_blocks = seq / BLOCK_N;
    const float k_scale = KScale[((batch_idx * num_heads + head_idx) * k_scale_blocks)
                                 + (kv_start / BLOCK_N)];
    const float score_scale = q_scale * k_scale;

    int32x2_t p_frags[ROW_GROUPS][PV_K_STEPS];
    #pragma unroll
    for (int rg = 0; rg < ROW_GROUPS; ++rg) {
      float raw[NUM_S_VALS];
      #pragma unroll
      for (int st = 0; st < NUM_S_ACCS; ++st)
        #pragma unroll
        for (int item = 0; item < 8; ++item)
          raw[st * 8 + item] = (float)s_acc[rg][st][item];
      if constexpr (MASK_TAIL) {
        #pragma unroll
        for (int item = 0; item < NUM_S_VALS; ++item)
          if (kv_start + (item / 8) * 16 + klane * 8 + item % 8 >= valid_seq_len)
            raw[item] = -3.402823466e+38F;
      }

      float local_max = raw[0];
      #pragma unroll
      for (int i = 1; i < NUM_S_VALS; ++i) local_max = fmaxf(local_max, raw[i]);
      local_max = fmaxf(local_max, shfl_xor_f(local_max, 16));
      const float row_max = fmaf(local_max, score_scale, -FP8_P_OFFSET);
      const float m_new = fmaxf(m_running[rg], row_max);
      const float corr = exp2f_fast(m_running[rg] - m_new);

      float p_vals[NUM_S_VALS];
      float local_sum = 0.f;
      const float neg_m = -m_new;
      if constexpr (!MASK_TAIL) {
        #pragma unroll
        for (int i = 0; i < NUM_S_VALS; ++i) {
          const float p = exp2f_fast(fmaf(raw[i], score_scale, neg_m));
          p_vals[i] = p;
          local_sum += p;
        }
      } else {
        #pragma unroll
        for (int item = 0; item < NUM_S_VALS; ++item) {
          const bool key_valid = kv_start + (item / 8) * 16 + klane * 8 + item % 8 < valid_seq_len;
          const float probability = key_valid ? exp2f_fast(fmaf(raw[item], score_scale, neg_m)) : 0.f;
          p_vals[item] = probability;
          local_sum += probability;
        }
      }
      const float tile_sum = local_sum + shfl_xor_f(local_sum, 16);
      l_running[rg] = corr * l_running[rg] + tile_sum;
      m_running[rg] = m_new;
      if (corr != 1.f) {
        #pragma unroll
        for (int dc = 0; dc < D_CHUNKS; ++dc)
          #pragma unroll
          for (int i = 0; i < 8; ++i) o_acc[rg][dc][i] *= corr;
      }

      #pragma unroll
      for (int pks = 0; pks < PV_K_STEPS; ++pks)
        p_frags[rg][pks] = pack_fp8_prob(&p_vals[pks * 8]);
    }

    #pragma unroll
    for (int pks = 0; pks < PV_K_STEPS; ++pks) {
      #pragma unroll
      for (int dc = 0; dc < D_CHUNKS; ++dc) {
        const int d_pos = dc * WMMA_N + lane16;
        const int token_pos = pks * WMMA_K + klane * WMMA_LANE_K;
        int8_t vb[8];
        *reinterpret_cast<int64_t*>(vb) = *reinterpret_cast<const int64_t*>(
            v_lds + d_pos * V_STRIDE + token_pos);
        int32x2_t pv = pack_i8x8(vb);
        #pragma unroll
        for (int rg = 0; rg < ROW_GROUPS; ++rg)
          o_acc[rg][dc] = wmma_pv_fp8(pv, p_frags[rg][pks], o_acc[rg][dc]);
      }
    }
  };
  const int full_end = (valid_seq_len / BLOCK_N) * BLOCK_N;
  for (int tile_start = 0; tile_start < full_end; tile_start += BLOCK_N)
    process_tile(tile_start, std::false_type{});
  if (full_end < valid_seq_len)
    process_tile(full_end, std::true_type{});

  #pragma unroll
  for (int rg = 0; rg < ROW_GROUPS; ++rg) {
    if (!q_ok[rg]) continue;
    const float inv_l = 1.f / l_running[rg];
    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) {
      const int d_col = dc * WMMA_N + klane * 8;
      const long long scale_index =
          ((long long)batch_idx * num_heads + head_idx) * HEAD_DIM + d_col;
      float vs[8];
      *reinterpret_cast<float4*>(vs) =
          *reinterpret_cast<const float4*>(VScale + scale_index);
      *reinterpret_cast<float4*>(vs + 4) =
          *reinterpret_cast<const float4*>(VScale + scale_index + 4);
      __hip_bfloat16 outv[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        outv[i] = __float2bfloat16(o_acc[rg][dc][i] * inv_l * vs[i]);
      *reinterpret_cast<int4*>(O + qk_idx(q_row[rg], d_col)) =
          *reinterpret_cast<int4*>(outv);
    }
  }
}

void launch_gfx1201_sage_attention(
    int64_t Q, int64_t K, int64_t V,
    int64_t QScale, int64_t KScale, int64_t VScale,
    int64_t O,
    int64_t batch_size, int64_t padded_seq_len, int64_t valid_seq_len,
    int64_t num_heads, int64_t stream
) {
  const int q_tiles = ((int)valid_seq_len + BLOCK_M - 1) / BLOCK_M;
  const dim3 grid((int)batch_size * q_tiles * (int)num_heads);
  const dim3 block(BLOCK_SIZE);
  hipLaunchKernelGGL(
      sage_hip_attn_bm128_bn32, grid, block, 0, (hipStream_t)stream,
      (const int8_t*)Q, (const int8_t*)K, (const int8_t*)V,
      (const float*)QScale, (const float*)KScale, (const float*)VScale,
      (__hip_bfloat16*)O,
      (int)batch_size, (int)padded_seq_len, (int)valid_seq_len,
      (int)num_heads);
}

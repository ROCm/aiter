// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_tt_sparse_prefill.h"
#include <cstdint>

namespace m3_tt {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) void* lds_ptr_t;
typedef __attribute__((address_space(1))) void* g_ptr_t;

constexpr int NW = 12;     // consumer waves, one query token each
constexpr int NPROD = 4;   // DMA producer waves, one LDS stage each
constexpr int NSTAGE = NPROD;
constexpr int TQ = NW;     // query tokens per CTA
constexpr int NTHR = 64 * (NW + NPROD);
constexpr int HG = 16;
constexpr int HD = 128;
constexpr int TOPK = 16;
constexpr int MAXW = 128;  // union bitmap words: block ids < 4096 (524,288 tokens)
constexpr int MAXU = TQ * TOPK;
// an LDS stage holds one block: K (swizzled), V, then the fp32 K and V scales of its 128 keys
constexpr int K_OFF = 0;
constexpr int V_OFF = 16384;
constexpr int KS_OFF = 32768;
constexpr int VS_OFF = 33280;
constexpr int STAGE_BYTES = 33792;
constexpr int LDS_BYTES = NSTAGE * STAGE_BYTES;

// s_waitcnt vmcnt(n), the other counters at their maximum
#define WAIT_VM(n) __builtin_amdgcn_s_waitcnt(((n) & 15) | (7 << 4) | (15 << 8) | (((n) >> 4) << 14))

struct Params {
  const uint16_t* q;
  const uint8_t* k;
  const uint8_t* v;
  const float* ks;
  const float* vs;
  const int* topk;
  const int* bt;
  const int* req;
  const int* kv_len;
  uint16_t* out;
  int T, H, hkv, bt_stride;
  int ntiles;
  const int* tile_t0;
  const int* tile_n;
  float sm_log2e;
};

#if defined(__gfx950__)
__device__ __forceinline__ uint16_t f_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float xor16_max(float x) {
  auto a = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return fmaxf(__uint_as_float(a[0]), __uint_as_float(a[1]));
}
__device__ __forceinline__ float xor32_max(float x) {
  auto b = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return fmaxf(__uint_as_float(b[0]), __uint_as_float(b[1]));
}
__device__ __forceinline__ float sum4l(float x) {
  auto a = __builtin_amdgcn_permlane16_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  x = __uint_as_float(a[0]) + __uint_as_float(a[1]);
  auto b = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return __uint_as_float(b[0]) + __uint_as_float(b[1]);
}
#endif

// A CTA = TQ consecutive tokens of one request x the 16 query heads of one KV head, walking the
// union of the 128-token blocks its tokens selected; producers DMA each block into LDS once.
__global__ __launch_bounds__(NTHR, 1) void m3_tt_sparse_prefill_kernel(Params p) {
#if defined(__gfx950__)
  __shared__ uint32_t s_bitmap[MAXW];
  __shared__ int s_wordoff[MAXW];
  __shared__ int s_blk[MAXU];
  __shared__ int s_page[MAXU];
  __shared__ uint32_t s_mask[MAXU];
  __shared__ int s_U;
  __shared__ int s_ready[NSTAGE];
  __shared__ int s_readyV[NSTAGE];
  __shared__ int s_doneV[NSTAGE];
  __shared__ uint8_t s_list[TQ * TOPK];  // per token: its union slots, ascending block id
  __shared__ int s_cnt[TQ];
  __shared__ int s_done[NSTAGE];
  extern __shared__ __attribute__((aligned(16))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
  const int hh = lane & 15, qg = lane >> 4;

  // consecutive CTAs go round-robin over the 8 XCDs: give each XCD a contiguous run of tiles
  const int ncta = gridDim.x;
  int bid = blockIdx.x;
  {
    const int per = ncta / 8;
    if (bid < per * 8) bid = (bid & 7) * per + (bid >> 3);
  }
  const int h = bid / p.ntiles;
  const int tile = bid - h * p.ntiles;
  const int t0 = p.tile_t0[tile];
  const int ntok = p.tile_n[tile];
  if (ntok <= 0) return;  // padding tile (whole CTA, before any barrier)
  const int rq = p.req[t0];

  // ---- union of the tile's selected blocks
  for (int i = tid; i < MAXW; i += NTHR) s_bitmap[i] = 0u;
  for (int i = tid; i < MAXU; i += NTHR) s_mask[i] = 0u;
  if (tid < NSTAGE) { s_ready[tid] = -1; s_done[tid] = 0; s_readyV[tid] = -1; s_doneV[tid] = 0; }
  __syncthreads();
  int my_b = -1, my_t = 0;
  if (tid < MAXU) {
    my_t = tid / TOPK;
    const int s = tid % TOPK;
    if (my_t < ntok) my_b = p.topk[((size_t)h * p.T + t0 + my_t) * TOPK + s];
    if (my_b >= MAXW * 32) my_b = -1;
    if (my_b >= 0) atomicOr(&s_bitmap[my_b >> 5], 1u << (my_b & 31));
  }
  __syncthreads();
  if (tid < 64) {
    int c0 = __popc(s_bitmap[2 * tid]), c1 = __popc(s_bitmap[2 * tid + 1]);
    int x = c0 + c1, incl = x;
#pragma unroll
    for (int o = 1; o < 64; o <<= 1) {
      int y = __shfl_up(incl, o);
      if (lane >= o) incl += y;
    }
    const int excl = incl - x;
    s_wordoff[2 * tid] = excl;
    s_wordoff[2 * tid + 1] = excl + c0;
    if (tid == 63) s_U = incl;
  }
  __syncthreads();
  if (my_b >= 0) {
    const uint32_t word = s_bitmap[my_b >> 5];
    const int slot = s_wordoff[my_b >> 5] + __popc(word & ((1u << (my_b & 31)) - 1u));
    s_blk[slot] = my_b;
    s_page[slot] = p.bt[(size_t)rq * p.bt_stride + my_b];
    atomicOr(&s_mask[slot], 1u << my_t);
  }
  {  // each token's selected union slots in ascending block order (= union order)
    int rank = 0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int ob = __shfl(my_b, (lane & ~15) | k);
      rank += (ob >= 0 && ob < my_b) ? 1 : 0;
    }
    const uint64_t bal = __ballot(my_b >= 0);
    if (my_b >= 0) {
      const uint32_t word = s_bitmap[my_b >> 5];
      const int slot = s_wordoff[my_b >> 5] + __popc(word & ((1u << (my_b & 31)) - 1u));
      s_list[my_t * TOPK + rank] = (uint8_t)slot;
    }
    if (tid < MAXU && (tid & 15) == 0) s_cnt[my_t] = __popcll((bal >> (lane & 48)) & 0xffffull);
  }
  __syncthreads();
  const int U = __builtin_amdgcn_readfirstlane(s_U);

  // ---- Q of this wave's token -> FP8 B operand with a 448 / amax scale (lane: head hh, d [32qg, +32))
  const int t = w;
  const int kvl = (t < ntok) ? p.kv_len[t0 + t] : 0;
  v8i qf;
  float cqk, m_t, l_t;
  v4f acc[8];
  {
    const int tt = min(t, ntok - 1);
    const uint16_t* qp = p.q + ((size_t)(t0 + tt) * p.H + h * HG + hh) * HD + qg * 32;
    float f[32];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const v4i raw = *reinterpret_cast<const v4i*>(qp + 8 * k);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t u = (uint32_t)raw[i];
        f[8 * k + 2 * i] = __uint_as_float(u << 16);
        f[8 * k + 2 * i + 1] = __uint_as_float(u & 0xffff0000u);
      }
    }
    float amax = 1e-6f;
#pragma unroll
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(f[k]));
    amax = xor32_max(xor16_max(amax));
    const float sc = __builtin_amdgcn_rcpf(amax) * 448.0f;
    cqk = __builtin_amdgcn_rcpf(sc) * p.sm_log2e;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      int pk = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * k] * sc, f[4 * k + 1] * sc, 0, false);
      pk = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * k + 2] * sc, f[4 * k + 3] * sc, pk, true);
      qf[k] = pk;
    }
    m_t = -INFINITY;
    l_t = 0.f;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) acc[dt] = (v4f){0.f, 0.f, 0.f, 0.f};
  }

  const int hkv = p.hkv;
  // K A-operand addresses (per subtile i): key = 32*(m>>2) + 4i + (m&3), m = hh
  const int kq = hh >> 2, kr = hh & 3;

  if (w >= NW) {
    // producer pi owns stage pi: it reloads K (+ scales) once the previous block's consumers
    // finished their softmax, and V once they finished their PV
    const int pi = __builtin_amdgcn_readfirstlane(w - NW);
    uint8_t* dst = smem + pi * STAGE_BYTES;
    for (int u = pi; u < U; u += NSTAGE) {
      const int page = __builtin_amdgcn_readfirstlane(s_page[u]);
      const size_t rbase = (size_t)page * 8 * hkv + h;
      const int need = (u >= NSTAGE) ? __popc(__builtin_amdgcn_readfirstlane((int)s_mask[u - NSTAGE])) : 0;
      while (__hip_atomic_load(&s_done[pi], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < need)
        __builtin_amdgcn_s_sleep(1);
      __hip_atomic_store(&s_done[pi], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
      asm volatile("" ::: "memory");
#pragma unroll
      for (int I = 0; I < 16; ++I) {  // 16 x 1 KB of K (swizzled)
        const int sl = I * 64 + lane;
        const int pg = sl >> 7, c = sl & 127, dc = c >> 4;
        const int tok = (c & 15) ^ (4 * ((pg >> 1) & 3));
        const size_t rec = rbase + (size_t)pg * hkv;
        __builtin_amdgcn_global_load_lds((g_ptr_t)(p.k + rec * 2048 + dc * 256 + tok * 16),
                                         (lds_ptr_t)(dst + K_OFF + I * 1024), 16, 0, 0);
      }
      {  // ks | vs: 1 KB, lane -> 4 floats (page pg, tokens 4qq .. +3)
        const int pg = (lane & 31) >> 2, qq = lane & 3;
        const float* src = ((lane < 32) ? p.ks : p.vs) + (rbase + (size_t)pg * hkv) * 16 + qq * 4;
        __builtin_amdgcn_global_load_lds((g_ptr_t)src, (lds_ptr_t)(dst + KS_OFF), 16, 0, 0);
      }
      // V region: issue now if the previous block's PVs are done, else after publishing K
      bool v_issued = false;
      auto issue_v = [&]() {
        __hip_atomic_store(&s_doneV[pi], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        asm volatile("" ::: "memory");
#pragma unroll
        for (int I = 0; I < 16; ++I) {
          const int sl = I * 64 + lane;
          const int pg = sl >> 7, c = sl & 127;
          const size_t rec = rbase + (size_t)pg * hkv;
          __builtin_amdgcn_global_load_lds((g_ptr_t)(p.v + rec * 2048 + (c << 4)),
                                           (lds_ptr_t)(dst + V_OFF + I * 1024), 16, 0, 0);
        }
      };
      if (__hip_atomic_load(&s_doneV[pi], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) >= need) {
        issue_v();
        v_issued = true;
        WAIT_VM(16);
      } else {
        WAIT_VM(0);
      }
      asm volatile("" ::: "memory");
      __hip_atomic_store(&s_ready[pi], u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
      if (!v_issued) {
        while (__hip_atomic_load(&s_doneV[pi], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < need)
          __builtin_amdgcn_s_sleep(1);
        issue_v();
      }
      WAIT_VM(0);
      asm volatile("" ::: "memory");
      __hip_atomic_store(&s_readyV[pi], u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
    }
    WAIT_VM(0);
    return;
  }

  // ---- consumer wave: the blocks its token selected, in union order
  const int n_sel = (w < ntok) ? __builtin_amdgcn_readfirstlane(s_cnt[w]) : 0;
  for (int ii = 0; ii < n_sel; ++ii) {
    const int u = __builtin_amdgcn_readfirstlane((int)s_list[w * TOPK + ii]);
    const int stg = u % NSTAGE;
    while (__hip_atomic_load(&s_ready[stg], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != u)
      __builtin_amdgcn_s_sleep(1);
    const int b = __builtin_amdgcn_readfirstlane(s_blk[u]);
    const uint8_t* st = smem + stg * STAGE_BYTES;
    bool k_released = false;
    const int kpos0 = b * 128 + 32 * qg;  // key of (i = 0, r = 0) for this lane
    // ---- S^T = K Q^T over the 128 keys: 8 subtiles; lane -> head hh, keys 32qg + 4i + r
    v4f S[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int pk = 2 * kq + (i >> 2);
      const int tok = 4 * (i & 3) + kr;
      const int sw = tok ^ (4 * ((pk >> 1) & 3));
      const uint8_t* a = st + K_OFF + pk * 2048 + (2 * qg) * 256 + sw * 16;
      const v4i a0 = *reinterpret_cast<const v4i*>(a);
      const v4i a1 = *reinterpret_cast<const v4i*>(a + 256);
      const v8i kf = (v8i){a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]};
      S[i] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(kf, qf, (v4f){0.f, 0.f, 0.f, 0.f}, 0, 0, 0, 127, 0,
                                                             127);
    }
    // ---- scores * k_scale, causal tail mask, running max
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const v4f kk = *reinterpret_cast<const v4f*>(st + KS_OFF + (32 * qg + 4 * i) * 4);
      S[i] *= kk;
    }
    if (b * 128 + 128 > kvl) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          if (kpos0 + 4 * i + r >= kvl) S[i][r] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) mx = fmaxf(mx, fmaxf(fmaxf(S[i][0], S[i][1]), fmaxf(S[i][2], S[i][3])));
    mx = xor32_max(xor16_max(mx));
    const float c = cqk;
    const float m_new = fmaxf(m_t, mx * c);
    if (m_new != -INFINITY) {
      const float alpha = __builtin_amdgcn_exp2f(m_t - m_new);
      // 4 independent partial sums / maxima (r): shorter dependency chains than one serial chain
      float lsv[4] = {0.f, 0.f, 0.f, 0.f}, amv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const v4f vv = *reinterpret_cast<const v4f*>(st + VS_OFF + (32 * qg + 4 * i) * 4);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float pr = __builtin_amdgcn_exp2f(__builtin_fmaf(S[i][r], c, -m_new));
          lsv[r] += pr;
          S[i][r] = pr * vv[r];  // P' = P * v_scale
          amv[r] = fmaxf(amv[r], S[i][r]);
        }
      }
      const float ls = (lsv[0] + lsv[1]) + (lsv[2] + lsv[3]);
      float am = fmaxf(fmaxf(amv[0], amv[1]), fmaxf(amv[2], amv[3]));
      l_t = l_t * alpha + ls;
      m_t = m_new;
      // P' -> FP8 with an exact 448 / amax scale per (head, 128 keys), as the ASM kernel
      am = fmaxf(xor32_max(xor16_max(am)), 1e-6f);
      const float sc = __builtin_amdgcn_rcpf(am) * 448.0f;
      const float inv = __builtin_amdgcn_rcpf(sc);
      v8i pf;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int pk = __builtin_amdgcn_cvt_pk_fp8_f32(S[i][0] * sc, S[i][1] * sc, 0, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(S[i][2] * sc, S[i][3] * sc, pk, true);
        pf[i] = pk;
      }
      // K + scales read: hand the K region back, then wait for this block's V
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      if (lane == 0) __hip_atomic_fetch_add(&s_done[stg], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
      k_released = true;
      while (__hip_atomic_load(&s_readyV[stg], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != u)
        __builtin_amdgcn_s_sleep(1);
      // lazy rescale: only when some head's running max moved
      if (__builtin_amdgcn_ballot_w64(alpha != 1.0f)) {
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) acc[dt] *= alpha;
      }
      // ---- O^T += inv * (V^T P_q^T): A = V^T (row d = 16dt + hh, keys 32qg .. +31), B = P_q
#pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
        const uint8_t* vp = st + V_OFF + (2 * qg) * 2048 + (16 * dt + hh) * 16;
        const v4i v0 = *reinterpret_cast<const v4i*>(vp);
        const v4i v1 = *reinterpret_cast<const v4i*>(vp + 2048);
        const v8i vf = (v8i){v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3]};
        const v4f tmp = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(vf, pf, (v4f){0.f, 0.f, 0.f, 0.f}, 0, 0, 0,
                                                                        127, 0, 127);
        acc[dt] += tmp * inv;
      }
    }
    // all LDS reads of this stage done -> release what is still held (V always after its ready)
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    if (!k_released) {
      if (lane == 0) __hip_atomic_fetch_add(&s_done[stg], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
      while (__hip_atomic_load(&s_readyV[stg], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) != u)
        __builtin_amdgcn_s_sleep(1);
    }
    if (lane == 0) __hip_atomic_fetch_add(&s_doneV[stg], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
  }

  // ---- epilogue: O = acc / l (lane: head hh, d = 16dt + 4qg + r)
  const float L = sum4l(l_t);
  const float inv = (L > 0.f) ? __builtin_amdgcn_rcpf(L) : 0.f;
  if (t < ntok) {
    uint16_t* op = p.out + ((size_t)(t0 + t) * p.H + h * HG + hh) * HD + 4 * qg;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      const uint32_t lo = (uint32_t)f_to_bf16(acc[dt][0] * inv) | ((uint32_t)f_to_bf16(acc[dt][1] * inv) << 16);
      const uint32_t hi = (uint32_t)f_to_bf16(acc[dt][2] * inv) | ((uint32_t)f_to_bf16(acc[dt][3] * inv) << 16);
      *reinterpret_cast<uint2*>(op + 16 * dt) = make_uint2(lo, hi);
    }
  }
#else
  // gfx950 only (scaled f8f6f4 MFMA, permlane swaps, 16-byte LDS DMA): the wrapper never launches it elsewhere
  __builtin_trap();
#endif
}

}  // namespace m3_tt

namespace aiter {

void m3_tt_sparse_prefill(const aiter_tensor_t& q,
                          const aiter_tensor_t& k_cache,
                          const aiter_tensor_t& v_cache,
                          const aiter_tensor_t& k_scale,
                          const aiter_tensor_t& v_scale,
                          const aiter_tensor_t& topk_idx,
                          const aiter_tensor_t& block_table,
                          const aiter_tensor_t& row_req,
                          const aiter_tensor_t& row_kv_len,
                          const aiter_tensor_t& tile_t0,
                          const aiter_tensor_t& tile_n,
                          int64_t ntiles,
                          float sm_scale,
                          aiter_tensor_t& out) {
  using namespace m3_tt;
  // a failed argument check raises a Python RuntimeError instead of aborting
  aiter_detail::AiterThrowGuard throw_guard;
  AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && q.dim() == 3 && q.size(2) == HD && q.is_contiguous(),
              "q must be a contiguous bf16 [T, H, 128]");
  const int64_t T = q.size(0), H = q.size(1);
  AITER_CHECK(topk_idx.dtype() == AITER_DTYPE_i32 && topk_idx.dim() == 3 && topk_idx.size(1) == T &&
                  topk_idx.size(2) == TOPK && topk_idx.is_contiguous(),
              "topk_idx must be a contiguous int32 [hkv, T, 16]");
  const int64_t hkv = topk_idx.size(0);
  AITER_CHECK(H == HG * hkv, "q must carry 16 query heads per KV head, got ", H, " for ", hkv);
  const int64_t pages = k_cache.size(0);
  AITER_CHECK(k_cache.dtype() == AITER_DTYPE_fp8 && k_cache.dim() == 5 && k_cache.size(1) == hkv &&
                  k_cache.size(2) == 8 && k_cache.size(3) == 16 && k_cache.size(4) == 16 && k_cache.is_contiguous(),
              "k_cache must be the contiguous FP8 page-16 SHUFFLE view [pages, hkv, 8, 16, 16]");
  AITER_CHECK(v_cache.dtype() == AITER_DTYPE_fp8 && v_cache.dim() == 5 && v_cache.size(0) == pages &&
                  v_cache.size(1) == hkv && v_cache.size(2) == 1 && v_cache.size(3) == HD && v_cache.size(4) == 16 &&
                  v_cache.is_contiguous(),
              "v_cache must be the contiguous FP8 page-16 SHUFFLE view [pages, hkv, 1, 128, 16]");
  for (const aiter_tensor_t* s : {&k_scale, &v_scale})
    AITER_CHECK(s->dtype() == AITER_DTYPE_fp32 && s->dim() == 3 && s->size(0) == pages && s->size(1) == hkv &&
                    s->size(2) == 16 && s->is_contiguous(),
                "k_scale / v_scale must be contiguous fp32 [pages, hkv, 16]");
  AITER_CHECK(block_table.dtype() == AITER_DTYPE_i32 && block_table.dim() == 2 && block_table.stride(1) == 1,
              "block_table must be int32 [batch, blocks] with unit column stride");
  for (const aiter_tensor_t* r : {&row_req, &row_kv_len})
    AITER_CHECK(r->dtype() == AITER_DTYPE_i32 && r->numel() == (size_t)T && r->is_contiguous(),
                "row_req / row_kv_len must be contiguous int32 [T]");
  for (const aiter_tensor_t* r : {&tile_t0, &tile_n})
    AITER_CHECK(r->dtype() == AITER_DTYPE_i32 && r->numel() == (size_t)ntiles && r->is_contiguous(),
                "tile_t0 / tile_n must be contiguous int32 [ntiles]");
  AITER_CHECK(out.dtype() == AITER_DTYPE_bf16 && out.numel() == (size_t)(T * H * HD) && out.is_contiguous(),
              "out must be a contiguous bf16 [T, H, 128]");
  if (T == 0 || ntiles == 0) return;

  Params p;
  p.q = static_cast<const uint16_t*>(q.data_ptr());
  p.k = static_cast<const uint8_t*>(k_cache.data_ptr());
  p.v = static_cast<const uint8_t*>(v_cache.data_ptr());
  p.ks = static_cast<const float*>(k_scale.data_ptr());
  p.vs = static_cast<const float*>(v_scale.data_ptr());
  p.topk = static_cast<const int*>(topk_idx.data_ptr());
  p.bt = static_cast<const int*>(block_table.data_ptr());
  p.req = static_cast<const int*>(row_req.data_ptr());
  p.kv_len = static_cast<const int*>(row_kv_len.data_ptr());
  p.out = static_cast<uint16_t*>(out.data_ptr());
  p.T = static_cast<int>(T);
  p.H = static_cast<int>(H);
  p.hkv = static_cast<int>(hkv);
  p.bt_stride = static_cast<int>(block_table.stride(0));
  p.ntiles = static_cast<int>(ntiles);
  p.tile_t0 = static_cast<const int*>(tile_t0.data_ptr());
  p.tile_n = static_cast<const int*>(tile_n.data_ptr());
  p.sm_log2e = sm_scale * 1.4426950408889634f;

  const HipDeviceGuard device_guard(q.device_id);
  const hipStream_t stream = aiter::getCurrentHIPStream();
  HIP_CALL(hipFuncSetAttribute(reinterpret_cast<const void*>(m3_tt_sparse_prefill_kernel),
                               hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
  m3_tt_sparse_prefill_kernel<<<dim3(ntiles * hkv), dim3(NTHR), LDS_BYTES, stream>>>(p);
  HIP_CALL_LAUNCH(hipGetLastError());
}

}  // namespace aiter

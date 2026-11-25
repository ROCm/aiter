// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_tt_sparse_prefill.h"
#include <algorithm>
#include <cstdint>

namespace m3_tt {

typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) void* lds_ptr_t;
typedef __attribute__((address_space(1))) void* g_ptr_t;

constexpr int NW = 12;     // consumer waves, one query token each
constexpr int NPROD = 4;   // DMA producer waves, one LDS stage each
constexpr int NSTAGE = NPROD;
constexpr int NBUF = 3;    // tile unions in flight
constexpr int TQ = NW;     // query tokens per tile
constexpr int NTHR = 64 * (NW + NPROD);
constexpr int HG = 16;
constexpr int HD = 128;
constexpr int TOPK = 16;
constexpr int MAXW = 128;  // union bitmap words: block ids < 4096 (524,288 tokens)
constexpr int MAXU = TQ * TOPK;
constexpr int XBAL = 16;   // consecutive tiles an XCD takes from its share
// an LDS stage holds one block: K (swizzled), V, then the fp32 K and V scales of its 128 keys
constexpr int K_OFF = 0;
constexpr int V_OFF = 16384;
constexpr int KS_OFF = 32768;
constexpr int VS_OFF = 33280;
constexpr int STAGE_BYTES = 33792;
constexpr int LDS_BYTES = NSTAGE * STAGE_BYTES;

// s_waitcnt vmcnt(n), the other counters at their maximum
#define WAIT_VM(n) __builtin_amdgcn_s_waitcnt(((n) & 15) | (7 << 4) | (15 << 8) | (((n) >> 4) << 14))
#define LDS_LOAD(x) __hip_atomic_load(&(x), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP)
#define LDS_STORE(x, v) __hip_atomic_store(&(x), (v), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP)
#define LDS_ZERO(x) __hip_atomic_store(&(x), 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)
#define LDS_ADD(x, v) __hip_atomic_fetch_add(&(x), (v), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP)
#define SPIN_UNTIL(cond) \
  while (!(cond)) __builtin_amdgcn_s_sleep(1)

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
  int ntiles;  // tiles a KV head (the tile map's length)
  int total;   // ntiles * hkv
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
// lane N of x's 16-lane row, to every lane of the row (DPP row_newbcast)
template <int N>
__device__ __forceinline__ float rbcast(float x) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), 0x150 + N, 0xf, 0xf, false));
}

// The k-th tile of CTA c, which runs on XCD c % 8: each XCD takes every 8th run of XBAL tiles of its
// eighth, so a chunk's cheap first tiles spread over all XCDs. -> (KV head, tile)
__device__ __forceinline__ void tile_of(const Params& p, int c, int k, int& h, int& tile) {
  int bid = c + k * (int)gridDim.x;
  const int per = p.total / 8;
  if (bid < per * 8) {
    const int x = bid & 7, L = bid >> 3;
    const int perB = (per / XBAL) * XBAL;
    if (L < perB) {
      const int g = L / XBAL;
      bid = (g * 8 + x) * XBAL + (L - g * XBAL);
    } else {
      bid = 8 * perB + x * (per - perB) + (L - perB);
    }
  }
  h = bid / p.ntiles;
  tile = bid - h * p.ntiles;
}
#endif

struct Meta {  // one tile's block union: slots in ascending block id, and each token's slots
  int blk[MAXU];
  int page[MAXU];
  uint32_t mask[MAXU];
  uint8_t list[TQ * TOPK];
  int cnt[TQ];
  int U, h, t0, ntok;
};

// One CTA a CU walks its tiles (TQ tokens of one request x one KV head's 16 query heads) over their
// selected blocks' union; DMA waves load each block into LDS once, in a 4-stage ring across tiles.
__global__ __launch_bounds__(NTHR, 1) void m3_tt_sparse_prefill_kernel(Params p) {
#if defined(__gfx950__)
  __shared__ Meta s_meta[NBUF];
  __shared__ uint32_t s_bitmap[NPROD][MAXW];  // per-builder scratch
  __shared__ int s_wordoff[NPROD][MAXW];
  __shared__ int s_seq[NBUF];        // tile k a buffer holds, -1 none
  __shared__ int s_cons_done[NBUF];  // consumer waves done with the buffer's tile
  __shared__ int s_prod_done[NBUF];  // DMA waves done with the buffer's tile
  __shared__ int s_ready[NSTAGE];
  __shared__ int s_readyV[NSTAGE];
  __shared__ int s_done[NSTAGE];
  __shared__ int s_doneV[NSTAGE];
  extern __shared__ __attribute__((aligned(16))) uint8_t smem[];

  const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
  const int hh = lane & 15, qg = lane >> 4;
  const int c = blockIdx.x;
  const int nt = (p.total - c + (int)gridDim.x - 1) / (int)gridDim.x;  // tiles of this CTA
  if (tid < NBUF) { s_seq[tid] = -1; s_cons_done[tid] = 0; s_prod_done[tid] = 0; }
  if (tid < NSTAGE) { s_ready[tid] = -1; s_done[tid] = 0; s_readyV[tid] = -1; s_doneV[tid] = 0; }
  __syncthreads();
  const int hkv = p.hkv;

  if (w >= NW) {
    const int pi = __builtin_amdgcn_readfirstlane(w - NW);
    // this wave builds tile m's block union into buffer m % NBUF
    auto build = [&](int m) {
      Meta& M = s_meta[m % NBUF];
      uint32_t* bm = s_bitmap[pi];
      int* wo = s_wordoff[pi];
      int h, tile;
      tile_of(p, c, m, h, tile);
      const int t0 = p.tile_t0[tile];
      const int ntok = p.tile_n[tile];
      bm[2 * lane] = 0u;
      bm[2 * lane + 1] = 0u;
#pragma unroll
      for (int j = 0; j < 3; ++j) M.mask[lane + 64 * j] = 0u;
      int bsel[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int e = lane + 64 * j, t = e >> 4, s = e & 15;
        int b = -1;
        if (t < ntok) b = p.topk[((size_t)h * p.T + t0 + t) * TOPK + s];
        if (b >= MAXW * 32) b = -1;
        bsel[j] = b;
      }
      const int rq = (ntok > 0) ? p.req[t0] : 0;
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#pragma unroll
      for (int j = 0; j < 3; ++j)
        if (bsel[j] >= 0) atomicOr(&bm[bsel[j] >> 5], 1u << (bsel[j] & 31));
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      const int c0 = __popc(bm[2 * lane]), c1 = __popc(bm[2 * lane + 1]);
      int x = c0 + c1, incl = x;
#pragma unroll
      for (int o = 1; o < 64; o <<= 1) {
        int y = __shfl_up(incl, o);
        if (lane >= o) incl += y;
      }
      const int excl = incl - x;
      wo[2 * lane] = excl;
      wo[2 * lane + 1] = excl + c0;
      const int U = __builtin_amdgcn_readlane(incl, 63);
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int e = lane + 64 * j, t = e >> 4;
        const int b = bsel[j];
        int slot = 0;
        if (b >= 0) {
          const uint32_t word = bm[b >> 5];
          slot = wo[b >> 5] + __popc(word & ((1u << (b & 31)) - 1u));
          M.blk[slot] = b;
          M.page[slot] = p.bt[(size_t)rq * p.bt_stride + b];
          atomicOr(&M.mask[slot], 1u << t);
        }
        // the token's selected slots in ascending block order (= union order)
        int rank = 0;
#pragma unroll
        for (int k2 = 0; k2 < TOPK; ++k2) {
          const int ob = __shfl(b, (lane & ~15) | k2);
          rank += (ob >= 0 && ob < b) ? 1 : 0;
        }
        const uint64_t bal = __ballot(b >= 0);
        if (b >= 0) M.list[t * TOPK + rank] = (uint8_t)slot;
        if ((lane & 15) == 0) M.cnt[t] = __popcll((bal >> (lane & 48)) & 0xffffull);
      }
      if (lane == 0) { M.U = U; M.h = h; M.t0 = t0; M.ntok = ntok; }
      WAIT_VM(0);
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      if (lane == 0) LDS_STORE(s_seq[m % NBUF], m);
    };
    if (pi < NBUF && pi < nt) build(pi);

    // producer pi owns stage pi: block g (CTA-wide sequence number) goes to stage g % NSTAGE once the
    // previous block's consumers there finished their softmax (K) and their PV (V)
    int G = 0, prev_need = 0;
    uint8_t* dst = smem + pi * STAGE_BYTES;
    for (int k = 0; k < nt; ++k) {
      const int bi = k % NBUF;
      SPIN_UNTIL(LDS_LOAD(s_seq[bi]) == k);
      const Meta& M = s_meta[bi];
      const int U = __builtin_amdgcn_readfirstlane(M.U);
      const int h = __builtin_amdgcn_readfirstlane(M.h);
      // first slot u of this tile with (G + u) % NSTAGE == pi
      int u = (pi - (G % NSTAGE) + NSTAGE) % NSTAGE;
      for (; u < U; u += NSTAGE) {
        const int g = G + u;
        const int page = __builtin_amdgcn_readfirstlane(M.page[u]);
        const int need_this = __popc(__builtin_amdgcn_readfirstlane((int)M.mask[u]));
        const size_t rbase = (size_t)page * 8 * hkv + h;
        const int need = prev_need;
        SPIN_UNTIL(LDS_LOAD(s_done[pi]) >= need);
        LDS_ZERO(s_done[pi]);
        asm volatile("" ::: "memory");
#pragma unroll
        for (int I = 0; I < 16; ++I) {  // 16 x 1 KB of K (swizzled)
          const int sl = I * 64 + lane;
          const int pg = sl >> 7, cc = sl & 127, dc = cc >> 4;
          const int tok = (cc & 15) ^ (4 * ((pg >> 1) & 3));
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
          LDS_ZERO(s_doneV[pi]);
          asm volatile("" ::: "memory");
#pragma unroll
          for (int I = 0; I < 16; ++I) {
            const int sl = I * 64 + lane;
            const int pg = sl >> 7, cc = sl & 127;
            const size_t rec = rbase + (size_t)pg * hkv;
            __builtin_amdgcn_global_load_lds((g_ptr_t)(p.v + rec * 2048 + (cc << 4)),
                                             (lds_ptr_t)(dst + V_OFF + I * 1024), 16, 0, 0);
          }
        };
        if (LDS_LOAD(s_doneV[pi]) >= need) {
          issue_v();
          v_issued = true;
          WAIT_VM(16);
        } else {
          WAIT_VM(0);
        }
        asm volatile("" ::: "memory");
        LDS_STORE(s_ready[pi], g);
        if (!v_issued) {
          SPIN_UNTIL(LDS_LOAD(s_doneV[pi]) >= need);
          issue_v();
        }
        WAIT_VM(0);
        asm volatile("" ::: "memory");
        LDS_STORE(s_readyV[pi], g);
        prev_need = need_this;
      }
      G += U;
      if (lane == 0) LDS_ADD(s_prod_done[bi], 1);
      // union m = k + 2 (m >= NBUF) is built by DMA wave m % NPROD once it has issued tile k
      const int m = k + 2;
      if (m >= NBUF && m < nt && (m % NPROD) == pi) {
        const int ob = m % NBUF;  // holds tile m - NBUF
        SPIN_UNTIL(LDS_LOAD(s_cons_done[ob]) >= NW && LDS_LOAD(s_prod_done[ob]) >= NPROD);
        LDS_ZERO(s_cons_done[ob]);
        LDS_ZERO(s_prod_done[ob]);
        build(m);
      }
    }
    WAIT_VM(0);
    return;
  }

  // ---- consumer wave w: token w of each of the CTA's tiles
  // K A-operand addresses (per subtile i): key = 32*(m>>2) + 4i + (m&3), m = hh
  const int kq = hh >> 2, kr = hh & 3;
  int G = 0;
  for (int k = 0; k < nt; ++k) {
    const int bi = k % NBUF;
    SPIN_UNTIL(LDS_LOAD(s_seq[bi]) == k);
    const Meta& M = s_meta[bi];
    const int U = __builtin_amdgcn_readfirstlane(M.U);
    const int h = __builtin_amdgcn_readfirstlane(M.h);
    const int t0 = __builtin_amdgcn_readfirstlane(M.t0);
    const int ntok = __builtin_amdgcn_readfirstlane(M.ntok);
    const int n_sel = (w < ntok) ? __builtin_amdgcn_readfirstlane(M.cnt[w]) : 0;
    if (w < ntok) {
      // Q of this wave's token -> FP8 B operand with a 448 / amax scale (lane: head hh, d [32qg, +32))
      v8i qf;
      float cqk, m_t, l_t;
      v4f acc[8];
      const int kvl = p.kv_len[t0 + w];
      {
        const uint16_t* qp = p.q + ((size_t)(t0 + w) * p.H + h * HG + hh) * HD + qg * 32;
        float f[32];
#pragma unroll
        for (int k4 = 0; k4 < 4; ++k4) {
          const v4i raw = *reinterpret_cast<const v4i*>(qp + 8 * k4);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint32_t uu = (uint32_t)raw[i];
            f[8 * k4 + 2 * i] = __uint_as_float(uu << 16);
            f[8 * k4 + 2 * i + 1] = __uint_as_float(uu & 0xffff0000u);
          }
        }
        float amax = 1e-6f;
#pragma unroll
        for (int k2 = 0; k2 < 32; ++k2) amax = fmaxf(amax, fabsf(f[k2]));
        amax = xor32_max(xor16_max(amax));
        const float sc = __builtin_amdgcn_rcpf(amax) * 448.0f;
        cqk = __builtin_amdgcn_rcpf(sc) * p.sm_log2e;
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
          int pk = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * k2] * sc, f[4 * k2 + 1] * sc, 0, false);
          pk = __builtin_amdgcn_cvt_pk_fp8_f32(f[4 * k2 + 2] * sc, f[4 * k2 + 3] * sc, pk, true);
          qf[k2] = pk;
        }
        m_t = -INFINITY;
        l_t = 0.f;
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) acc[dt] = (v4f){0.f, 0.f, 0.f, 0.f};
      }
      // the blocks this token selected, in union order
      for (int ii = 0; ii < n_sel; ++ii) {
        const int u = __builtin_amdgcn_readfirstlane((int)M.list[w * TOPK + ii]);
        const int g = G + u;
        const int stg = g % NSTAGE;
        SPIN_UNTIL(LDS_LOAD(s_ready[stg]) == g);
        const int b = __builtin_amdgcn_readfirstlane(M.blk[u]);
        const uint8_t* st = smem + stg * STAGE_BYTES;
        bool k_released = false;
        const int kpos0 = b * 128 + 32 * qg;  // key of (i = 0, r = 0) for this lane
        // ---- S^T = K Q^T over the 128 keys: 8 subtiles, operands prefetched 2 deep
        v4f S[8];
        __builtin_amdgcn_s_setprio(2);
        // this lane's K and V scales, broadcast to their rows below
        const v2f ksv = *reinterpret_cast<const v2f*>(st + KS_OFF + (32 * qg + 2 * hh) * 4);
        const v2f vsv = *reinterpret_cast<const v2f*>(st + VS_OFF + (32 * qg + 2 * hh) * 4);
        {
          v4i a0[2], a1[2];
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int pk = 2 * kq + (i >> 2);
            const int tok = 4 * (i & 3) + kr;
            const int sw = tok ^ (4 * ((pk >> 1) & 3));
            const uint8_t* a = st + K_OFF + pk * 2048 + (2 * qg) * 256 + sw * 16;
            a0[i] = *reinterpret_cast<const v4i*>(a);
            a1[i] = *reinterpret_cast<const v4i*>(a + 256);
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const v8i kf = (v8i){a0[i & 1][0], a0[i & 1][1], a0[i & 1][2], a0[i & 1][3],
                                 a1[i & 1][0], a1[i & 1][1], a1[i & 1][2], a1[i & 1][3]};
            S[i] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(kf, qf, (v4f){0.f, 0.f, 0.f, 0.f}, 0, 0, 0,
                                                                   127, 0, 127);
            if (i + 2 < 8) {
              const int i2 = i + 2;
              const int pk = 2 * kq + (i2 >> 2);
              const int tok = 4 * (i2 & 3) + kr;
              const int sw = tok ^ (4 * ((pk >> 1) & 3));
              const uint8_t* a = st + K_OFF + pk * 2048 + (2 * qg) * 256 + sw * 16;
              a0[i & 1] = *reinterpret_cast<const v4i*>(a);
              a1[i & 1] = *reinterpret_cast<const v4i*>(a + 256);
            }
          }
        }
        __builtin_amdgcn_s_setprio(0);
        // ---- scores * k_scale, causal tail mask, running max
#define KMUL(i)                           \
  S[i][0] *= rbcast<2 * (i)>(ksv[0]);     \
  S[i][1] *= rbcast<2 * (i)>(ksv[1]);     \
  S[i][2] *= rbcast<2 * (i) + 1>(ksv[0]); \
  S[i][3] *= rbcast<2 * (i) + 1>(ksv[1]);
        KMUL(0) KMUL(1) KMUL(2) KMUL(3) KMUL(4) KMUL(5) KMUL(6) KMUL(7)
#undef KMUL
        if (b * 128 + 128 > kvl) {
#pragma unroll
          for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int r = 0; r < 4; ++r)
              if (kpos0 + 4 * i + r >= kvl) S[i][r] = -INFINITY;
        }
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) mx = fmaxf(mx, fmaxf(fmaxf(S[i][0], S[i][1]), fmaxf(S[i][2], S[i][3])));
        mx = xor32_max(xor16_max(mx));
        const float cc = cqk;
        const float m_new = fmaxf(m_t, mx * cc);
        if (m_new != -INFINITY) {
          const float alpha = __builtin_amdgcn_exp2f(m_t - m_new);
          // 4 independent partial sums / maxima (r): shorter dependency chains than one serial chain
          float lsv[4] = {0.f, 0.f, 0.f, 0.f}, amv[4] = {0.f, 0.f, 0.f, 0.f};
#define PMUL(i)                                                                   \
  {                                                                               \
    const float p0 = __builtin_amdgcn_exp2f(__builtin_fmaf(S[i][0], cc, -m_new)); \
    lsv[0] += p0;                                                                 \
    S[i][0] = p0 * rbcast<2 * (i)>(vsv[0]);                                       \
    amv[0] = fmaxf(amv[0], S[i][0]);                                              \
    const float p1 = __builtin_amdgcn_exp2f(__builtin_fmaf(S[i][1], cc, -m_new)); \
    lsv[1] += p1;                                                                 \
    S[i][1] = p1 * rbcast<2 * (i)>(vsv[1]);                                       \
    amv[1] = fmaxf(amv[1], S[i][1]);                                              \
    const float p2 = __builtin_amdgcn_exp2f(__builtin_fmaf(S[i][2], cc, -m_new)); \
    lsv[2] += p2;                                                                 \
    S[i][2] = p2 * rbcast<2 * (i) + 1>(vsv[0]);                                   \
    amv[2] = fmaxf(amv[2], S[i][2]);                                              \
    const float p3 = __builtin_amdgcn_exp2f(__builtin_fmaf(S[i][3], cc, -m_new)); \
    lsv[3] += p3;                                                                 \
    S[i][3] = p3 * rbcast<2 * (i) + 1>(vsv[1]);                                   \
    amv[3] = fmaxf(amv[3], S[i][3]);                                              \
  }
          PMUL(0) PMUL(1) PMUL(2) PMUL(3) PMUL(4) PMUL(5) PMUL(6) PMUL(7)
#undef PMUL
          const float ls = (lsv[0] + lsv[1]) + (lsv[2] + lsv[3]);
          float am = fmaxf(fmaxf(amv[0], amv[1]), fmaxf(amv[2], amv[3]));
          l_t = l_t * alpha + ls;
          m_t = m_new;
          // P' (= P * v_scale) -> FP8 with an exact 448 / amax scale per (head, 128 keys), as the ASM kernel
          am = fmaxf(xor32_max(xor16_max(am)), 1e-6f);
          const float sc = __builtin_amdgcn_rcpf(am) * 448.0f;
          const float inv = __builtin_amdgcn_rcpf(sc);
          v8i pf;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            int pk = __builtin_amdgcn_cvt_pk_fp8_f32(S[i][0] * sc, S[i][1] * sc, __float_as_int(S[i][2]), false);
            pk = __builtin_amdgcn_cvt_pk_fp8_f32(S[i][2] * sc, S[i][3] * sc, pk, true);
            pf[i] = pk;
          }
          // K + scales read: hand the K region back, then wait for this block's V
          asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
          if (lane == 0) LDS_ADD(s_done[stg], 1);
          k_released = true;
          SPIN_UNTIL(LDS_LOAD(s_readyV[stg]) == g);
          // lazy rescale: only when some head's running max moved
          if (__builtin_amdgcn_ballot_w64(alpha != 1.0f)) {
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) acc[dt] *= alpha;
          }
          __builtin_amdgcn_s_setprio(2);
          // ---- O^T += inv * (V^T P_q^T): A = V^T (row d = 16dt + hh, keys 32qg .. +31), B = P_q
          {
            v4i b0[2], b1[2];
#pragma unroll
            for (int dt = 0; dt < 2; ++dt) {
              const uint8_t* vp = st + V_OFF + (2 * qg) * 2048 + (16 * dt + hh) * 16;
              b0[dt] = *reinterpret_cast<const v4i*>(vp);
              b1[dt] = *reinterpret_cast<const v4i*>(vp + 2048);
            }
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
              const v8i vf = (v8i){b0[dt & 1][0], b0[dt & 1][1], b0[dt & 1][2], b0[dt & 1][3],
                                   b1[dt & 1][0], b1[dt & 1][1], b1[dt & 1][2], b1[dt & 1][3]};
              const v4f tmp = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(vf, pf, (v4f){0.f, 0.f, 0.f, 0.f}, 0,
                                                                              0, 0, 127, 0, 127);
              acc[dt] += tmp * inv;
              if (dt + 2 < 8) {
                const uint8_t* vp = st + V_OFF + (2 * qg) * 2048 + (16 * (dt + 2) + hh) * 16;
                b0[dt & 1] = *reinterpret_cast<const v4i*>(vp);
                b1[dt & 1] = *reinterpret_cast<const v4i*>(vp + 2048);
              }
            }
          }
        }
        __builtin_amdgcn_s_setprio(0);
        // all LDS reads of this stage done -> release what is still held (V always after its ready)
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        if (!k_released) {
          if (lane == 0) LDS_ADD(s_done[stg], 1);
          SPIN_UNTIL(LDS_LOAD(s_readyV[stg]) == g);
        }
        if (lane == 0) LDS_ADD(s_doneV[stg], 1);
      }

      // ---- epilogue: O = acc / l (lane: head hh, d = 16dt + 4qg + r)
      const float L = sum4l(l_t);
      const float inv = (L > 0.f) ? __builtin_amdgcn_rcpf(L) : 0.f;
      uint16_t* op = p.out + ((size_t)(t0 + w) * p.H + h * HG + hh) * HD + 4 * qg;
#pragma unroll
      for (int dt = 0; dt < 8; ++dt) {
        const uint32_t lo = (uint32_t)f_to_bf16(acc[dt][0] * inv) | ((uint32_t)f_to_bf16(acc[dt][1] * inv) << 16);
        const uint32_t hi = (uint32_t)f_to_bf16(acc[dt][2] * inv) | ((uint32_t)f_to_bf16(acc[dt][3] * inv) << 16);
        *reinterpret_cast<uint2*>(op + 16 * dt) = make_uint2(lo, hi);
      }
    }
    G += U;
    // this wave no longer reads the tile's union
    if (lane == 0) LDS_ADD(s_cons_done[bi], 1);
  }
#else
  // gfx950 only (scaled f8f6f4 MFMA, permlane swaps, DPP row broadcasts, 16-byte LDS DMA): the
  // wrapper never launches it elsewhere
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
  p.total = static_cast<int>(ntiles * hkv);
  p.tile_t0 = static_cast<const int*>(tile_t0.data_ptr());
  p.tile_n = static_cast<const int*>(tile_n.data_ptr());
  p.sm_log2e = sm_scale * 1.4426950408889634f;

  const HipDeviceGuard device_guard(q.device_id);
  const hipStream_t stream = aiter::getCurrentHIPStream();
  int num_cus = 0;
  HIP_CALL(hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, q.device_id));
  // one CTA a CU, in whole rounds of the 8 XCDs that the tile order maps each CTA's tiles to
  const int grid = std::max(1, std::min(num_cus / 8 * 8, p.total));
  HIP_CALL(hipFuncSetAttribute(reinterpret_cast<const void*>(m3_tt_sparse_prefill_kernel),
                               hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES));
  m3_tt_sparse_prefill_kernel<<<dim3(grid), dim3(NTHR), LDS_BYTES, stream>>>(p);
  HIP_CALL_LAUNCH(hipGetLastError());
}

}  // namespace aiter

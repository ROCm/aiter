// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// kpool cache writers for the GLM-5.3-Flash sparse-attention indexer: prefill
// insert, decode update and the mixed step (insert + tail seed + decode in one
// launch). Byte-identical to the vLLM Triton path (torch gather +
// `_kpool_softmax_rotate_write_cache_kernel`, `kpool_seed_tail_cache`,
// `kpool_decode_update_and_maybe_write_cache_batched`).
//
// Semantics: every token t with slot_mapping[t] >= 0 and t >= POOL-1 closes
// the pool made of tokens [t-POOL+1, t]. For that pool, per head dim d:
//   score_s = f32(gate[t-POOL+1+s][d]) + ape[s][d]
//   prob_s  = exp(score_s - max_s score_s)          (exp = exp2(x * log2e))
//   x       = (sum_s k_s * prob_s) / (sum_s prob_s)   -> bf16 -> f32
//   x       = Hadamard128(x) * (1/sqrt(128))          -> bf16 -> f32
//   absmax  = max(max_d |x_d|, 1e-4)
//   scale   = round_scale ? exp2(ceil(log2(absmax/448))) : absmax/448
//   q       = clamp(x / scale, -448, 448) -> fp8 e4m3fn
// q (128 B, the MFMA 16x16 preshuffled layout of indexer_k_quant_and_cache
// when page_size > 1) and scale (fp32) go to kv_cache at slot_mapping[t]:
// kv_cache is [num_blocks, page_size, HEAD_DIM + 4] uint8, each page holding
// page_size * 128 B of keys followed by page_size fp32 scales.
//
// Mapping: one window of POOL consecutive tokens per group of LANES lanes,
// one wave (64 lanes) per workgroup. LANES = 64 (1 pool / wave, 2 dims per
// lane), 32 (2 pools / wave) or 16 (4 pools / wave, 16 B loads); picked from
// the token count unless the caller forces it. K / gate are read in place
// (no [n, POOL, 128] gather) and only pools that really close are compressed.
// Decode: one wave per request, the request's metadata in SGPRs, all K / gate
// rows in one round trip, completions and stashes in token order (same result
// as the ordered walk, deterministic).
//
// This translation unit is torch-free (AITER_NO_TORCH_TYPES): everything
// crosses the ABI as aiter_tensor_t.

#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "kpool_compress.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace {

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));

constexpr int HEAD_DIM = 128;
constexpr float FP8_MAX = 448.0f;
// Triton: 1.0 / FP8_MAX is a Python double folded into an fp32 constant.
constexpr float FP8_MAX_INV = (float)(1.0 / 448.0);
// Triton: x * 0.08838834764831845 (python float -> fp32 constant).
constexpr float HAD_SCALE = (float)0.08838834764831845;
// Triton AMD lowers tl.exp(x) to exp2(x * log2e) with log2e as fp32.
constexpr float LOG2E = (float)1.4426950408889634;

__device__ __forceinline__ float bf16lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float bf16at(const uint32_t* w, int d) {
  return (d & 1) ? bf16hi(w[d >> 1]) : bf16lo(w[d >> 1]);
}
// Round-to-nearest-even to bf16, returned as fp32 (finite inputs).
__device__ __forceinline__ float round_bf16(float x) {
  uint32_t u = __float_as_uint(x);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xFFFF0000u);
}
__device__ __forceinline__ float fast_exp2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ float fast_log2(float x) { return __builtin_amdgcn_logf(x); }

// Lane exchange: v from lane (lane ^ MASK), VALU only (no LDS traffic):
//   1, 2  DPP quad_perm
//   4     DPP row_shl:4 / row_shr:4 + select on lane bit 2
//   8     DPP row_ror:8 (rotate by 8 inside a 16-lane row == xor 8)
//   16    v_permlane16_swap (gfx950)
//   32    v_permlane32_swap (gfx950)
// -DKPOOL_LDS_XCHG falls back to ds_swizzle / ds_bpermute for 4..32; selected
// automatically for every device pass that is not gfx950 (permlane*_swap
// exist on gfx950 only).
#if !defined(KPOOL_LDS_XCHG) && !defined(__gfx950__)
#define KPOOL_LDS_XCHG
#endif
template <int MASK>
__device__ __forceinline__ float xor_lane(float v) {
  int iv = __float_as_int(v);
  int r;
  if constexpr (MASK == 1) {
    r = __builtin_amdgcn_update_dpp(0, iv, 0xB1, 0xF, 0xF, true);  // quad_perm(1,0,3,2)
  } else if constexpr (MASK == 2) {
    r = __builtin_amdgcn_update_dpp(0, iv, 0x4E, 0xF, 0xF, true);  // quad_perm(2,3,0,1)
#if !defined(KPOOL_LDS_XCHG)
  } else if constexpr (MASK == 4) {
    const int up = __builtin_amdgcn_update_dpp(iv, iv, 0x104, 0xF, 0xF, false);  // row_shl:4
    const int dn = __builtin_amdgcn_update_dpp(iv, iv, 0x114, 0xF, 0xF, false);  // row_shr:4
    r = (__lane_id() & 4) ? dn : up;
  } else if constexpr (MASK == 8) {
    r = __builtin_amdgcn_update_dpp(iv, iv, 0x128, 0xF, 0xF, false);  // row_ror:8
  } else if constexpr (MASK == 16) {
    auto p = __builtin_amdgcn_permlane16_swap((unsigned)iv, (unsigned)iv, false, false);
    r = (__lane_id() & 16) ? (int)p[0] : (int)p[1];
  } else if constexpr (MASK == 32) {
    auto p = __builtin_amdgcn_permlane32_swap((unsigned)iv, (unsigned)iv, false, false);
    r = (__lane_id() & 32) ? (int)p[0] : (int)p[1];
#else
  } else if constexpr (MASK < 32) {
    r = __builtin_amdgcn_ds_swizzle(iv, (MASK << 10) | 0x1F);  // bitmode xor MASK
  } else {
    r = __shfl_xor(iv, MASK, 64);
#endif
  }
  return __int_as_float(r);
}

// One cross-lane Hadamard stage: pair (lane, lane ^ MASK); the lower lane
// keeps a + b, the upper one a - b (same op order as the Triton butterfly).
template <int MASK, int DPL>
__device__ __forceinline__ void had_stage_xlane(float (&x)[DPL], int sub) {
  const bool upper = (sub & MASK) != 0;
#pragma unroll
  for (int d = 0; d < DPL; ++d) {
    float o = xor_lane<MASK>(x[d]);
    x[d] = upper ? (o - x[d]) : (x[d] + o);
  }
}

template <int DPL>
__device__ __forceinline__ void had_stages_inlane(float (&x)[DPL]) {
#pragma unroll
  for (int s = 1; s < DPL; s <<= 1) {
#pragma unroll
    for (int i = 0; i < DPL; ++i) {
      if ((i & s) == 0) {
        float a = x[i], b = x[i + s];
        x[i] = a + b;
        x[i + s] = a - b;
      }
    }
  }
}

// DPL consecutive bf16 of one row into DPL/2 dwords. Rows are read exactly
// once, so the vector loads are non-temporal.
template <int DPL, bool VEC>
__device__ __forceinline__ void load_row(const uint16_t* __restrict__ p, uint32_t (&w)[DPL / 2]) {
  if constexpr (VEC && DPL >= 8) {
#pragma unroll
    for (int c = 0; c < DPL / 8; ++c) {
      u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(p) + c);
      w[c * 4 + 0] = v[0];
      w[c * 4 + 1] = v[1];
      w[c * 4 + 2] = v[2];
      w[c * 4 + 3] = v[3];
    }
  } else if constexpr (VEC && DPL == 4) {
    u32x2 v = __builtin_nontemporal_load(reinterpret_cast<const u32x2*>(p));
    w[0] = v[0];
    w[1] = v[1];
  } else if constexpr (VEC && DPL == 2) {
    w[0] = __builtin_nontemporal_load(reinterpret_cast<const uint32_t*>(p));
  } else {
#pragma unroll
    for (int i = 0; i < DPL / 2; ++i) {
      w[i] = (uint32_t)p[2 * i] | ((uint32_t)p[2 * i + 1] << 16);
    }
  }
}

template <int POOL, int DPL, bool VEC>
__device__ __forceinline__ void load_rows(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ gate, int64_t first,
    int64_t k_stride, int64_t gate_stride, int d0,
    uint32_t (&gw)[POOL][DPL / 2], uint32_t (&kw)[POOL][DPL / 2]) {
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
    load_row<DPL, VEC>(gate + (first + s) * gate_stride + d0, gw[s]);
    load_row<DPL, VEC>(k + (first + s) * k_stride + d0, kw[s]);
  }
}

// Pool math for one pool spread over LANES lanes; this lane holds dims
// [sub*DPL, sub*DPL + DPL) of every row. Same instructions and op order as
// the Triton kernel (bit-exact).
template <int POOL, int LANES>
__device__ __forceinline__ void compress_pool(
    const uint32_t (&gw)[POOL][HEAD_DIM / LANES / 2],
    const uint32_t (&kw)[POOL][HEAD_DIM / LANES / 2],
    const float (&ape)[POOL][HEAD_DIM / LANES],
    uint8_t* __restrict__ cache,
    int64_t loc,
    int page_size,
    int page_shift,  // log2(page_size), or -1 when page_size is not a power of two
    int64_t page_bytes,
    bool preshuffle,
    bool round_scale,
    int sub) {
  constexpr int DPL = HEAD_DIM / LANES;
  const int d0 = sub * DPL;

  // --- pass 1: per-dim max of score = gate + ape ---
  float mx[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) mx[d] = -INFINITY;
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
#pragma unroll
    for (int d = 0; d < DPL; ++d) mx[d] = fmaxf(mx[d], bf16at(gw[s], d) + ape[s][d]);
  }

  // --- pass 2: softmax-weighted sum of k (denom += p; acc = fma(k, p, acc)) ---
  float acc[DPL], den[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) { acc[d] = 0.0f; den[d] = 0.0f; }
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
      float score = bf16at(gw[s], d) + ape[s][d];
      float p = fast_exp2((score - mx[d]) * LOG2E);
      den[d] += p;
      acc[d] = __builtin_fmaf(bf16at(kw[s], d), p, acc[d]);
    }
  }

  // --- pooled vector -> bf16 -> Hadamard-128 -> bf16 ---
  float x[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) x[d] = round_bf16(acc[d] / den[d]);

  had_stages_inlane<DPL>(x);
  if constexpr (LANES >= 2) had_stage_xlane<1, DPL>(x, sub);
  if constexpr (LANES >= 4) had_stage_xlane<2, DPL>(x, sub);
  if constexpr (LANES >= 8) had_stage_xlane<4, DPL>(x, sub);
  if constexpr (LANES >= 16) had_stage_xlane<8, DPL>(x, sub);
  if constexpr (LANES >= 32) had_stage_xlane<16, DPL>(x, sub);
  if constexpr (LANES >= 64) had_stage_xlane<32, DPL>(x, sub);
#pragma unroll
  for (int d = 0; d < DPL; ++d) x[d] = round_bf16(x[d] * HAD_SCALE);

  // --- per-vector absmax fp8 quant ---
  float am = 0.0f;
#pragma unroll
  for (int d = 0; d < DPL; ++d) am = fmaxf(am, fabsf(x[d]));
  if constexpr (LANES >= 2) am = fmaxf(am, xor_lane<1>(am));
  if constexpr (LANES >= 4) am = fmaxf(am, xor_lane<2>(am));
  if constexpr (LANES >= 8) am = fmaxf(am, xor_lane<4>(am));
  if constexpr (LANES >= 16) am = fmaxf(am, xor_lane<8>(am));
  if constexpr (LANES >= 32) am = fmaxf(am, xor_lane<16>(am));
  if constexpr (LANES >= 64) am = fmaxf(am, xor_lane<32>(am));
  am = fmaxf(am, 1e-4f);
  float scale;
  if (round_scale) {
    scale = fast_exp2(__builtin_ceilf(fast_log2(am * FP8_MAX_INV)));
  } else {
    scale = am * FP8_MAX_INV;
  }
  constexpr int NQW = DPL >= 4 ? DPL / 4 : 1;
  uint32_t qw[NQW];
  if constexpr (DPL == 2) {
    float q0 = fminf(fmaxf(x[0] / scale, -FP8_MAX), FP8_MAX);
    float q1 = fminf(fmaxf(x[1] / scale, -FP8_MAX), FP8_MAX);
    qw[0] = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q0, q1, 0, false);
  }
#pragma unroll
  for (int c = 0; c < DPL / 4; ++c) {
    float q0 = fminf(fmaxf(x[c * 4 + 0] / scale, -FP8_MAX), FP8_MAX);
    float q1 = fminf(fmaxf(x[c * 4 + 1] / scale, -FP8_MAX), FP8_MAX);
    float q2 = fminf(fmaxf(x[c * 4 + 2] / scale, -FP8_MAX), FP8_MAX);
    float q3 = fminf(fmaxf(x[c * 4 + 3] / scale, -FP8_MAX), FP8_MAX);
    uint32_t w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q0, q1, 0, false);
    w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q2, q3, (int)w, true);
    qw[c] = w;
  }

  // --- store ---
  int64_t page;
  int tok;
  if (page_shift >= 0) {
    page = loc >> page_shift;
    tok = (int)(loc & (page_size - 1));
  } else {
    page = loc / page_size;
    tok = (int)(loc % page_size);
  }
  uint8_t* pbase = cache + page * page_bytes;
  if (preshuffle) {
    // byte offset of (tok, dim) = (tok/16)*16*128 + (dim/16)*256 + (tok%16)*16 + dim%16
    uint8_t* kp = pbase + (int64_t)(tok >> 4) * (16 * HEAD_DIM) + (d0 >> 4) * 256 + (tok & 15) * 16 + (d0 & 15);
    if constexpr (DPL == 16) {
      *reinterpret_cast<uint4*>(kp) = make_uint4(qw[0], qw[1], qw[2], qw[3]);
    } else if constexpr (DPL == 8) {
      *reinterpret_cast<uint2*>(kp) = make_uint2(qw[0], qw[1]);
    } else if constexpr (DPL == 2) {
      // lane pairs (2k, 2k+1) hold dims 4k..4k+3: the even lane stores one
      // dword (2-byte stores per lane are far slower in the memory pipeline)
      const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0])));
      if ((sub & 1) == 0) *reinterpret_cast<uint32_t*>(kp) = (qw[0] & 0xFFFFu) | (other << 16);
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) reinterpret_cast<uint32_t*>(kp)[c] = qw[c];
    }
  } else {
    uint8_t* kp = pbase + (int64_t)tok * HEAD_DIM + d0;
    if constexpr (DPL == 2) {
      const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0])));
      if ((sub & 1) == 0) *reinterpret_cast<uint32_t*>(kp) = (qw[0] & 0xFFFFu) | (other << 16);
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) reinterpret_cast<uint32_t*>(kp)[c] = qw[c];
    }
  }
  if (sub == 0) {
    *reinterpret_cast<float*>(pbase + (int64_t)page_size * HEAD_DIM + (int64_t)tok * 4) = scale;
  }
}

// Workgroups are dispatched round-robin over the 8 XCDs (each with its own
// L2). Remapped, each XCD gets a contiguous range of windows, so the rows a
// misaligned pool re-reads from the previous window were fetched by a
// neighbour on the same XCD (L2 hit). Bijective for any grid size.
template <int XCD>
__device__ __forceinline__ int xcd_remap(int b, int nb) {
  if constexpr (XCD <= 1) return b;
  const int q = nb / XCD, r = nb % XCD;
  const int x = b % XCD, i = b / XCD;
  return x * q + min(x, r) + i;
}

// ===========================================================================
// Prefill insert window (+ optional tail seeding), shared by the insert
// kernel and the mixed kernel.
// ===========================================================================

struct TailArgs {
  uint16_t* tail;        // [num_blocks, 2, ring, 128] bf16 (strides below, in elements)
  int64_t block_elems;   // stride(0)
  int64_t head_elems;    // stride(1)
  int ring;              // shape[2]
  int ring_shift;        // log2(ring) or -1
};

__device__ __forceinline__ int ring_div(int x, const TailArgs& ta) {
  return ta.ring_shift >= 0 ? (x >> ta.ring_shift) : (x / ta.ring);
}
__device__ __forceinline__ int ring_mod(int x, const TailArgs& ta) {
  return ta.ring_shift >= 0 ? (x & (ta.ring - 1)) : (x % ta.ring);
}

template <int POOL, int LANES, bool VEC, typename SlotT>
__device__ __forceinline__ void insert_window(
    int64_t win,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ gate,
    const float* __restrict__ ape,
    const void* __restrict__ slot_mapping,
    uint8_t* __restrict__ cache,
    int64_t n_tokens,
    int64_t n_windows,
    int64_t k_stride,
    int64_t gate_stride,
    int64_t ape_stride,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    int flags,  // bit0: slot int64, bit1: preshuffle, bit2: round_scale, bit3: seed tail
    const int32_t* __restrict__ seed_tslot,
    TailArgs ta,
    int grp,
    int sub) {
  constexpr int DPL = HEAD_DIM / LANES;
  static_assert(POOL <= LANES, "one lane per slot in the window");
  const bool preshuffle = flags & 2;
  const bool round_scale = flags & 4;
  const bool seed = flags & 8;
  const int d0 = sub * DPL;
  const int64_t win_first = win * POOL;

  // Slot read first, unconditional (clamped index) so it is one vector load
  // with no branch; K / gate / ape loads below are issued right behind it and
  // pinned so the compiler cannot sink them under the slot-dependent branch.
  const int64_t my_tok = win_first + (sub % POOL);
  const int64_t tok_c = my_tok < n_tokens ? my_tok : n_tokens - 1;
  const int64_t my_slot = (int64_t)reinterpret_cast<const SlotT*>(slot_mapping)[tok_c];

  float apev[POOL][DPL];
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
    const float* ap = ape + s * ape_stride + d0;
    if constexpr (DPL == 2) {
      float2 a = *reinterpret_cast<const float2*>(ap);
      apev[s][0] = a.x;
      apev[s][1] = a.y;
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) {
        float4 a = reinterpret_cast<const float4*>(ap)[c];
        apev[s][c * 4 + 0] = a.x;
        apev[s][c * 4 + 1] = a.y;
        apev[s][c * 4 + 2] = a.z;
        apev[s][c * 4 + 3] = a.w;
      }
    }
  }
  uint32_t gw[POOL][DPL / 2];
  uint32_t kw[POOL][DPL / 2];
  bool have_window = win < n_windows && win_first + POOL <= n_tokens;
  if (have_window) {
    load_rows<POOL, DPL, VEC>(k, gate, win_first, k_stride, gate_stride, d0, gw, kw);
  }
  // tail-seed metadata, independent of everything else: issued up front
  int seed_ts[POOL], seed_ahead[POOL];
  if (seed && win < n_windows) {
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      const int64_t tok = win_first + s;
      seed_ts[s] = tok < n_tokens ? seed_tslot[tok] : -1;
      seed_ahead[s] = (tok + POOL < n_tokens) ? seed_tslot[tok + POOL] : -1;
    }
  }

  const int closes = (win < n_windows && sub < POOL && my_tok < n_tokens && my_tok >= POOL - 1 && my_slot >= 0) ? 1 : 0;
  const uint64_t ballot = __ballot(closes);
  uint64_t bits = (ballot >> (grp * LANES)) & ((LANES == 64) ? ~0ull : ((1ull << LANES) - 1));
  // Pin the preloaded rows here (after the ballot, which must only wait for
  // the slot) so the compiler neither sinks the loads under the
  // slot-dependent branch nor waits for them before the ballot.
#pragma unroll
  for (int s2 = 0; s2 < POOL; ++s2) {
#pragma unroll
    for (int c = 0; c < DPL / 2; ++c) { asm volatile("" : "+v"(gw[s2][c])); asm volatile("" : "+v"(kw[s2][c])); }
#pragma unroll
    for (int d = 0; d < DPL; ++d) asm volatile("" : "+v"(apev[s2][d]));
  }

  while (bits) {
    const int i = __builtin_ctzll(bits);
    bits &= bits - 1;
    const int64_t last = win_first + i;
    const int64_t loc = (int64_t)reinterpret_cast<const SlotT*>(slot_mapping)[last];
    if (!(have_window && i == POOL - 1)) {
      load_rows<POOL, DPL, VEC>(k, gate, last - (POOL - 1), k_stride, gate_stride, d0, gw, kw);
      have_window = false;
    }
    compress_pool<POOL, LANES>(gw, kw, apev, cache, loc, page_size, page_shift, page_bytes, preshuffle,
                               round_scale, sub);
  }

  // Tail seeding (kpool_seed_tail_cache): token i is among its request's last
  // POOL tokens if the token POOL ahead has another tail block (or is past
  // the batch / has slot < 0). Its raw K / gate rows go to
  // tail[blk, {0, 1}, ts % ring]. The rows are already in registers when the
  // window was preloaded and not clobbered.
  if (seed && win < n_windows) {
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      const int64_t tok = win_first + s;
      if (tok >= n_tokens) break;
      const int ts = seed_ts[s];
      if (ts < 0) continue;
      const int blk = ring_div(ts, ta);
      const int ahead = seed_ahead[s];
      if (ahead >= 0 && ring_div(ahead, ta) == blk) continue;
      uint32_t rk[DPL / 2], rs[DPL / 2];
      if (have_window) {
#pragma unroll
        for (int c = 0; c < DPL / 2; ++c) { rk[c] = kw[s][c]; rs[c] = gw[s][c]; }
      } else {
        load_row<DPL, VEC>(k + tok * k_stride + d0, rk);
        load_row<DPL, VEC>(gate + tok * gate_stride + d0, rs);
      }
      const int64_t dst = (int64_t)blk * ta.block_elems + (int64_t)ring_mod(ts, ta) * HEAD_DIM + d0;
      uint32_t* pk = reinterpret_cast<uint32_t*>(ta.tail + dst);
      uint32_t* ps = reinterpret_cast<uint32_t*>(ta.tail + ta.head_elems + dst);
#pragma unroll
      for (int c = 0; c < DPL / 2; ++c) { pk[c] = rk[c]; ps[c] = rs[c]; }
    }
  }
}


// ---------------------------------------------------------------------------
// Prefill insert: one wave per workgroup, 64 / LANES windows per wave.
// ---------------------------------------------------------------------------
template <int POOL, int LANES, bool VEC, typename SlotT>
__global__ void __launch_bounds__(64) kpool_insert_kernel(
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ gate,
    const float* __restrict__ ape,
    const void* __restrict__ slot_mapping,
    uint8_t* __restrict__ cache,
    int64_t n_tokens,
    int64_t n_windows,
    int64_t k_stride,
    int64_t gate_stride,
    int64_t ape_stride,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    int flags) {
  constexpr int GROUPS_PER_WAVE = 64 / LANES;
  const int lane = threadIdx.x;
  const int grp = GROUPS_PER_WAVE == 1 ? 0 : lane / LANES;
  const int sub = lane % LANES;
  const int64_t win = (int64_t)xcd_remap<8>(blockIdx.x, gridDim.x) * GROUPS_PER_WAVE + grp;
  TailArgs ta{nullptr, 0, 0, 1, 0};
  insert_window<POOL, LANES, VEC, SlotT>(win, k, gate, ape, slot_mapping, cache, n_tokens, n_windows, k_stride,
                                         gate_stride, ape_stride, page_size, page_shift, page_bytes, flags & 7,
                                         nullptr, ta, grp, sub);
}

// ===========================================================================
// Decode update: one wave (64 lanes, 2 dims per lane) per request.
// ===========================================================================

constexpr int DEC_LANES = 64;
constexpr int DEC_DPL = HEAD_DIM / DEC_LANES;  // 2
constexpr int DEC_W = DEC_DPL / 2;             // dwords per row per lane
constexpr int DEC_BLOCK_WAVES = 4;             // requests per workgroup in the standalone decode kernel

struct DecodeArgs {
  const uint16_t* key;     // [B, next_n, 128] bf16
  const uint16_t* score;   // [B, next_n, 128] bf16
  int64_t key_sb, key_st, ss_sb, ss_st;
  const int32_t* tail_slot;  // [B, next_n]
  const int32_t* slot_map;   // [B, next_n]
  const int32_t* positions;  // [B, next_n]
  int next_n;
  int num_requests;
};

// DEC_DPL bf16 = DEC_W dwords per lane per row
template <bool VEC>
__device__ __forceinline__ void load_rowd(const uint16_t* __restrict__ p, uint32_t (&w)[DEC_W]) {
  static_assert(DEC_W == 1, "decode lanes hold one dword per row");
  if constexpr (VEC) {
    w[0] = *reinterpret_cast<const uint32_t*>(p);
  } else {
#pragma unroll
    for (int i = 0; i < DEC_W; ++i) w[i] = (uint32_t)p[2 * i] | ((uint32_t)p[2 * i + 1] << 16);
  }
}

template <int NMAX>
__device__ __forceinline__ void pick_row(const uint32_t (&arr)[NMAX][DEC_W], int idx, uint32_t (&out)[DEC_W]) {
#pragma unroll
  for (int c = 0; c < DEC_W; ++c) out[c] = arr[0][c];
#pragma unroll
  for (int i = 1; i < NMAX; ++i) {
    if (i == idx) {
#pragma unroll
      for (int c = 0; c < DEC_W; ++c) out[c] = arr[i][c];
    }
  }
}

// Per-token metadata lives in clang ext vectors (SSA values, never an
// alloca): a select chain over array elements would be folded by LLVM into a
// dynamically indexed load, and the array would then land in LDS or scratch.
template <int N> struct Vec;
template <> struct Vec<1> { typedef int i __attribute__((ext_vector_type(1))); typedef unsigned u __attribute__((ext_vector_type(1))); };
template <> struct Vec<2> { typedef int i __attribute__((ext_vector_type(2))); typedef unsigned u __attribute__((ext_vector_type(2))); };
template <> struct Vec<4> { typedef int i __attribute__((ext_vector_type(4))); typedef unsigned u __attribute__((ext_vector_type(4))); };
template <> struct Vec<8> { typedef int i __attribute__((ext_vector_type(8))); typedef unsigned u __attribute__((ext_vector_type(8))); };
template <> struct Vec<16> { typedef int i __attribute__((ext_vector_type(16))); typedef unsigned u __attribute__((ext_vector_type(16))); };

// v[idx] for a runtime idx as a flat chain of constant-index selects
template <int N, typename V>
__device__ __forceinline__ auto vpick(const V& v, int idx) {
  auto out = v[0];
#pragma unroll
  for (int i = 1; i < N; ++i) out = (idx == i) ? v[i] : out;
  return out;
}

// Same result as the ordered walk (`_kpool_decode_update_batched_kernel`):
// a completion at token tc reads, for each pool slot, the latest token
// t' < tc of this launch that stashed into (block, phys) if any, else the
// ring. One wave per request: the request's metadata (3 x NMAX int32) is
// wave-uniform, so it lives in scalar registers and the completion / source
// / stash logic runs on the scalar unit; each lane holds its 2 dims of every
// K / gate row. Everything is fetched in ONE round trip; the ring rows of the
// first completion are fetched before any pool math; later completions
// normally read only this launch's tokens (registers). Completions and
// stashes are issued in token order, so a later writer of the same cache /
// ring slot wins as in the walk (same lane, same address: program order).
template <int POOL, int NMAX, bool VEC>
__device__ __forceinline__ void decode_request(
    int req,  // wave-uniform
    const DecodeArgs& da,
    const float* __restrict__ ape,
    int64_t ape_stride,
    TailArgs ta,
    uint8_t* __restrict__ cache,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    bool preshuffle,
    bool round_scale,
    int sub) {  // lane, 0..63
  typedef typename Vec<NMAX>::i ivec;
  typedef typename Vec<NMAX>::u uvec;
  const int n = da.next_n;  // 1 <= n <= NMAX
  const int64_t base = (int64_t)req * n;
  const int d0 = sub * DEC_DPL;

  // --- one round trip: metadata (scalar), ape, all K / gate rows ---
  ivec loc, pos, ts;
  uvec rk[DEC_W], rs[DEC_W];
  const uint16_t* kbase = da.key + (int64_t)req * da.key_sb + d0;
  const uint16_t* sbase = da.score + (int64_t)req * da.ss_sb + d0;
  const int32_t* mloc = da.slot_map + base;
  const int32_t* mpos = da.positions + base;
  const int32_t* mts = da.tail_slot + base;
  const int kst = (int)da.key_st, sst = (int)da.ss_st;  // row strides fit in 32 bits
#pragma unroll
  for (int i = 0; i < NMAX; ++i) {
    const int j = min(i, n - 1);
    loc[i] = mloc[j];
    pos[i] = mpos[j];
    ts[i] = mts[j];
    uint32_t wk[DEC_W], ws[DEC_W];
    load_rowd<VEC>(kbase + j * kst, wk);
    load_rowd<VEC>(sbase + j * sst, ws);
#pragma unroll
    for (int c = 0; c < DEC_W; ++c) { rk[c][i] = wk[c]; rs[c][i] = ws[c]; }
  }
  float apev[POOL][DEC_DPL];
#pragma unroll
  for (int s = 0; s < POOL; ++s) {
    if constexpr (DEC_DPL == 2) {
      float2 a = *reinterpret_cast<const float2*>(ape + s * ape_stride + d0);
      apev[s][0] = a.x; apev[s][1] = a.y;
    } else {
      float4 a = *reinterpret_cast<const float4*>(ape + s * ape_stride + d0);
      apev[s][0] = a.x; apev[s][1] = a.y; apev[s][2] = a.z; apev[s][3] = a.w;
    }
  }

  // --- derived per token (wave-uniform) ---
  ivec phys, blk;
  uint32_t smask = 0, cmask = 0;
#pragma unroll
  for (int i = 0; i < NMAX; ++i) {
    const bool valid = i < n;
    const int safe = max(pos[i], 0);
    phys[i] = ring_mod(safe, ta);
    blk[i] = ring_div(max(ts[i], 0), ta);
    if (valid && pos[i] >= 0 && ts[i] >= 0) smask |= 1u << i;
    if (valid && loc[i] >= 0 && pos[i] >= 0 && (safe % POOL == POOL - 1)) cmask |= 1u << i;
  }

  // source token (or -1 = ring) of pool slot s of the completion at token i
  auto src_of = [&](int i, int s, int& phys_s) {
    const int safe = max(vpick<NMAX>(pos, i), 0);
    const int slot = safe % POOL;
    phys_s = ring_mod(safe - slot + s, ta);
    if (s == slot) return i;
    const int blk_i = vpick<NMAX>(blk, i);
    int src = -1;
#pragma unroll
    for (int t2 = 0; t2 < NMAX; ++t2) {
      if (t2 < i && ((smask >> t2) & 1u) && blk[t2] == blk_i && phys[t2] == phys_s) src = t2;
    }
    return src;
  };

  // --- ring rows of the first completion, fetched before any pool math ---
  const int first = cmask ? __builtin_ctz(cmask) : -1;
  uint32_t fk[POOL][DEC_W], fs[POOL][DEC_W];
  if (first >= 0) {
    const int64_t bbase = (int64_t)vpick<NMAX>(blk, first) * ta.block_elems + d0;
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      int phys_s;
      (void)src_of(first, s, phys_s);
      const int64_t off = bbase + (int64_t)phys_s * HEAD_DIM;
      load_rowd<VEC>(ta.tail + off, fk[s]);
      load_rowd<VEC>(ta.tail + ta.head_elems + off, fs[s]);
    }
  }

  // --- completions, in token order: one compress body, runtime loop ---
  while (cmask) {
    const int i = __builtin_ctz(cmask);
    cmask &= cmask - 1;
    uint32_t gw[POOL][DEC_W], kw[POOL][DEC_W];
    const int64_t bbase = (int64_t)vpick<NMAX>(blk, i) * ta.block_elems + d0;
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      int phys_s;
      const int src = src_of(i, s, phys_s);
      if (src >= 0) {
#pragma unroll
        for (int c = 0; c < DEC_W; ++c) { kw[s][c] = vpick<NMAX>(rk[c], src); gw[s][c] = vpick<NMAX>(rs[c], src); }
      } else if (i == first) {
#pragma unroll
        for (int c = 0; c < DEC_W; ++c) { kw[s][c] = fk[s][c]; gw[s][c] = fs[s][c]; }
      } else {
        const int64_t off = bbase + (int64_t)phys_s * HEAD_DIM;
        load_rowd<VEC>(ta.tail + off, kw[s]);
        load_rowd<VEC>(ta.tail + ta.head_elems + off, gw[s]);
      }
    }
    compress_pool<POOL, DEC_LANES>(gw, kw, apev, cache, (int64_t)vpick<NMAX>(loc, i), page_size, page_shift,
                                   page_bytes, preshuffle, round_scale, sub);
  }

  // --- stash, in token order (every ring read above was consumed already) ---
#pragma unroll
  for (int i = 0; i < NMAX; ++i) {
    if (!((smask >> i) & 1u)) continue;
    const int64_t dst = (int64_t)blk[i] * ta.block_elems + (int64_t)phys[i] * HEAD_DIM + d0;
#pragma unroll
    for (int c = 0; c < DEC_W; ++c) {
      reinterpret_cast<uint32_t*>(ta.tail + dst)[c] = rk[c][i];
      reinterpret_cast<uint32_t*>(ta.tail + ta.head_elems + dst)[c] = rs[c][i];
    }
  }
}


template <int POOL, int NMAX, bool VEC>
__global__ void __launch_bounds__(64 * DEC_BLOCK_WAVES) kpool_decode_kernel(
    DecodeArgs da,
    const float* __restrict__ ape,
    int64_t ape_stride,
    TailArgs ta,
    uint8_t* __restrict__ cache,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    int flags) {  // bit1: preshuffle, bit2: round_scale
  const int lane = threadIdx.x & 63;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int req = blockIdx.x * DEC_BLOCK_WAVES + wave;
  if (req >= da.num_requests) return;
  decode_request<POOL, NMAX, VEC>(req, da, ape, ape_stride, ta, cache, page_size, page_shift, page_bytes,
                                  (flags & 2) != 0, (flags & 4) != 0, lane);
}

// ===========================================================================
// Mixed step, one wave per workgroup: blocks [0, num_requests) run one decode
// request each, the rest run prefill windows (+ tail seeding). The roles
// touch disjoint state (own tail blocks, own cache slots).
// ===========================================================================

template <int POOL, int NMAX, int LANES, bool VEC, typename SlotT>
__global__ void __launch_bounds__(64) kpool_mixed_kernel(
    DecodeArgs da,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ gate,
    const void* __restrict__ slot_mapping,
    int64_t n_tokens,
    int64_t n_windows,
    int64_t k_stride,
    int64_t gate_stride,
    const int32_t* __restrict__ seed_tslot,
    const float* __restrict__ ape,
    int64_t ape_stride,
    TailArgs ta,
    uint8_t* __restrict__ cache,
    int page_size,
    int page_shift,
    int64_t page_bytes,
    int flags) {
  const int lane = threadIdx.x;
  const int n_dec = da.num_requests;
  if ((int)blockIdx.x < n_dec) {
    decode_request<POOL, NMAX, VEC>(blockIdx.x, da, ape, ape_stride, ta, cache, page_size, page_shift, page_bytes,
                                    (flags & 2) != 0, (flags & 4) != 0, lane);
  } else {
    constexpr int GROUPS_PER_WAVE = 64 / LANES;
    const int grp = GROUPS_PER_WAVE == 1 ? 0 : lane / LANES;
    const int sub = lane % LANES;
    const int pb = xcd_remap<8>((int)blockIdx.x - n_dec, (int)gridDim.x - n_dec);
    const int64_t win = (int64_t)pb * GROUPS_PER_WAVE + grp;
    insert_window<POOL, LANES, VEC, SlotT>(win, k, gate, ape, slot_mapping, cache, n_tokens, n_windows, k_stride,
                                           gate_stride, ape_stride, page_size, page_shift, page_bytes, flags,
                                           seed_tslot, ta, grp, sub);
  }
}

// ===========================================================================
// Launchers
// ===========================================================================

inline int page_shift_of(int page_size) {
  return (page_size > 0 && (page_size & (page_size - 1)) == 0) ? __builtin_ctz((unsigned)page_size) : -1;
}

template <int POOL, int LANES, bool VEC, typename SlotT>
int launch_insert_t(const void* k, const void* gate, const void* ape, const void* slot, void* cache,
                    int64_t n_tokens, int64_t k_stride, int64_t gate_stride, int64_t ape_stride, int page_size,
                    int64_t page_bytes, int flags, hipStream_t stream) {
  constexpr int GROUPS_PER_WAVE = 64 / LANES;
  const int64_t n_windows = (n_tokens + POOL - 1) / POOL;
  const int64_t grid = (n_windows + GROUPS_PER_WAVE - 1) / GROUPS_PER_WAVE;
  if (grid == 0) return 0;
  hipLaunchKernelGGL((kpool_insert_kernel<POOL, LANES, VEC, SlotT>), dim3((unsigned)grid), dim3(64), 0, stream,
                     (const uint16_t*)k, (const uint16_t*)gate, (const float*)ape, slot, (uint8_t*)cache, n_tokens,
                     n_windows, k_stride, gate_stride, ape_stride, page_size, page_shift_of(page_size), page_bytes,
                     flags);
  return (int)hipGetLastError();
}

template <int POOL, int LANES>
int launch_insert_l(bool vec, bool i64, const void* k, const void* gate, const void* ape, const void* slot,
                    void* cache, int64_t n_tokens, int64_t k_stride, int64_t gate_stride, int64_t ape_stride,
                    int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
#define KPOOL_I(V, S)                                                                                          \
  return launch_insert_t<POOL, LANES, V, S>(k, gate, ape, slot, cache, n_tokens, k_stride, gate_stride, ape_stride, \
                                            page_size, page_bytes, flags, stream);
  if (vec && i64) { KPOOL_I(true, int64_t) }
  if (vec) { KPOOL_I(true, int32_t) }
  if (i64) { KPOOL_I(false, int64_t) }
  KPOOL_I(false, int32_t)
#undef KPOOL_I
}

template <int POOL>
int launch_insert(int lanes, bool vec, bool i64, const void* k, const void* gate, const void* ape, const void* slot,
                  void* cache, int64_t n_tokens, int64_t k_stride, int64_t gate_stride, int64_t ape_stride,
                  int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
#define KPOOL_L(L)                                                                                              \
  if (lanes == L)                                                                                               \
    return launch_insert_l<POOL, L>(vec, i64, k, gate, ape, slot, cache, n_tokens, k_stride, gate_stride,       \
                                    ape_stride, page_size, page_bytes, flags, stream);
  KPOOL_L(64) KPOOL_L(32) KPOOL_L(16)
#undef KPOOL_L
  return -2;
}

template <int POOL, int NMAX>
int launch_decode_n(bool vec, const DecodeArgs& da, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                    int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
  const int grid = (da.num_requests + DEC_BLOCK_WAVES - 1) / DEC_BLOCK_WAVES;
  if (grid == 0) return 0;
  if (vec)
    hipLaunchKernelGGL((kpool_decode_kernel<POOL, NMAX, true>), dim3(grid), dim3(64 * DEC_BLOCK_WAVES), 0, stream,
                       da, (const float*)ape, ape_stride, ta, (uint8_t*)cache, page_size, page_shift_of(page_size),
                       page_bytes, flags);
  else
    hipLaunchKernelGGL((kpool_decode_kernel<POOL, NMAX, false>), dim3(grid), dim3(64 * DEC_BLOCK_WAVES), 0, stream,
                       da, (const float*)ape, ape_stride, ta, (uint8_t*)cache, page_size, page_shift_of(page_size),
                       page_bytes, flags);
  return (int)hipGetLastError();
}

template <int POOL>
int launch_decode(bool vec, const DecodeArgs& da, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                  int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
  const int n = da.next_n;
#define KPOOL_N(N)                                                                                            \
  if (n <= N) return launch_decode_n<POOL, N>(vec, da, ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
  KPOOL_N(1) KPOOL_N(2) KPOOL_N(4) KPOOL_N(8) KPOOL_N(16)
#undef KPOOL_N
  return -4;
}

template <int POOL, int NMAX, int LANES, bool VEC, typename SlotT>
int launch_mixed_t(const DecodeArgs& da, const void* k, const void* gate, const void* slot, int64_t n_tokens,
                   int64_t k_stride, int64_t gate_stride, const int32_t* seed_tslot, const void* ape,
                   int64_t ape_stride, TailArgs ta, void* cache, int page_size, int64_t page_bytes, int flags,
                   hipStream_t stream) {
  constexpr int GROUPS_PER_WAVE = 64 / LANES;
  const int64_t n_windows = (n_tokens + POOL - 1) / POOL;
  const int64_t n_prefill_blocks = (n_windows + GROUPS_PER_WAVE - 1) / GROUPS_PER_WAVE;
  const int64_t grid = da.num_requests + n_prefill_blocks;
  if (grid == 0) return 0;
  hipLaunchKernelGGL((kpool_mixed_kernel<POOL, NMAX, LANES, VEC, SlotT>), dim3((unsigned)grid), dim3(64), 0, stream,
                     da, (const uint16_t*)k, (const uint16_t*)gate, slot, n_tokens, n_windows, k_stride,
                     gate_stride, seed_tslot, (const float*)ape, ape_stride, ta, (uint8_t*)cache, page_size,
                     page_shift_of(page_size), page_bytes, flags);
  return (int)hipGetLastError();
}

template <int POOL, int NMAX, int LANES>
int launch_mixed_l(bool vec, bool i64, const DecodeArgs& da, const void* k, const void* gate, const void* slot,
                   int64_t n_tokens, int64_t k_stride, int64_t gate_stride, const int32_t* seed_tslot,
                   const void* ape, int64_t ape_stride, TailArgs ta, void* cache, int page_size, int64_t page_bytes,
                   int flags, hipStream_t stream) {
#define KPOOL_M(V, S)                                                                                           \
  return launch_mixed_t<POOL, NMAX, LANES, V, S>(da, k, gate, slot, n_tokens, k_stride, gate_stride, seed_tslot, \
                                                 ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
  if (vec && i64) { KPOOL_M(true, int64_t) }
  if (vec) { KPOOL_M(true, int32_t) }
  if (i64) { KPOOL_M(false, int64_t) }
  KPOOL_M(false, int32_t)
#undef KPOOL_M
}

template <int POOL, int NMAX>
int launch_mixed_n(int lanes, bool vec, bool i64, const DecodeArgs& da, const void* k, const void* gate,
                   const void* slot, int64_t n_tokens, int64_t k_stride, int64_t gate_stride,
                   const int32_t* seed_tslot, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                   int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
#define KPOOL_L(L)                                                                                              \
  if (lanes == L)                                                                                               \
    return launch_mixed_l<POOL, NMAX, L>(vec, i64, da, k, gate, slot, n_tokens, k_stride, gate_stride, seed_tslot, \
                                         ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
  KPOOL_L(64) KPOOL_L(32) KPOOL_L(16)
#undef KPOOL_L
  return -2;
}

template <int POOL>
int launch_mixed(int lanes, bool vec, bool i64, const DecodeArgs& da, const void* k, const void* gate,
                 const void* slot, int64_t n_tokens, int64_t k_stride, int64_t gate_stride,
                 const int32_t* seed_tslot, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                 int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
  const int n = da.next_n;
#define KPOOL_N(N)                                                                                            \
  if (n <= N)                                                                                                 \
    return launch_mixed_n<POOL, N>(lanes, vec, i64, da, k, gate, slot, n_tokens, k_stride, gate_stride,       \
                                   seed_tslot, ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
  KPOOL_N(1) KPOOL_N(2) KPOOL_N(4) KPOOL_N(8) KPOOL_N(16)
#undef KPOOL_N
  return -4;
}

}  // namespace

// Launcher return codes: 0 ok, -2 bad lane count, -3 unsupported pool size, -4 next_n > 16, else hipError_t.
#define KPOOL_POOL_SWITCH(EXPR)                      \
  switch (pool_size) {                               \
    case 2: { constexpr int P = 2; EXPR }            \
    case 3: { constexpr int P = 3; EXPR }            \
    case 4: { constexpr int P = 4; EXPR }            \
    case 8: { constexpr int P = 8; EXPR }            \
    case 16: { constexpr int P = 16; EXPR }          \
    default: rc = -3; break;                         \
  }

// ===========================================================================
// aiter entry points
// ===========================================================================

namespace aiter {

namespace {

constexpr int KPOOL_HEAD_DIM = 128;

int kpool_pick_lanes(int64_t n_tokens, int pool_size)
{
    // Chunked-prefill sizes are latency bound: one pool per wave (64 lanes,
    // 2 dims per lane) gives the most workgroups and the shortest per-lane
    // chain. Large batches are bandwidth bound: 16 lanes (8 dims, one 16 B
    // load per row per lane). Tuned on MI355X, pool_size 4.
    const int64_t n_pools = n_tokens / pool_size;
    if(n_pools <= 8192)
        return 64;
    if(n_pools <= 65536)
        return 32;
    return 16;
}

bool aligned(const void* p, int64_t stride_elems, int64_t elem_bytes, int64_t align)
{
    return (reinterpret_cast<uintptr_t>(p) % align) == 0 && ((stride_elems * elem_bytes) % align) == 0;
}

void check_rc(int rc, const char* what)
{
    AITER_CHECK(rc != -2, what, ": lanes_per_pool must be 16, 32 or 64 and >= pool_size");
    AITER_CHECK(rc != -3, what, ": pool_size must be 2, 3, 4, 8 or 16");
    AITER_CHECK(rc != -4, what, ": next_n must be <= 16");
    AITER_CHECK(rc == 0, what, ": ", hipGetErrorString(static_cast<hipError_t>(rc)));
}

void check_cache(const aiter_tensor_t& kv_cache)
{
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_u8 || kv_cache.dtype() == AITER_DTYPE_fp8,
                "kv_cache must be uint8 / fp8, got ", AiterDtype_to_str(kv_cache.dtype()));
    AITER_CHECK(kv_cache.dim() == 3 && kv_cache.size(2) == KPOOL_HEAD_DIM + 4,
                "kv_cache must be [num_blocks, page_size, 132]");
    AITER_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
    const int page_size = static_cast<int>(kv_cache.size(1));
    AITER_CHECK(page_size == 1 || page_size % 16 == 0,
                "page_size must be 1 or a multiple of 16 (preshuffled tiles), got ", page_size);
}

void check_ape(const aiter_tensor_t& ape, int pool_size)
{
    AITER_CHECK(ape.dtype() == AITER_DTYPE_fp32 && ape.dim() == 2 && ape.size(0) == pool_size &&
                    ape.size(1) == KPOOL_HEAD_DIM && ape.stride(1) == 1,
                "ape must be fp32 [pool_size, 128] with unit inner stride");
}

void check_rows(const aiter_tensor_t& t, const char* name)
{
    AITER_CHECK(t.dtype() == AITER_DTYPE_bf16, name, " must be bf16");
    AITER_CHECK(t.dim() == 2 && t.size(1) == KPOOL_HEAD_DIM && t.stride(1) == 1,
                name, " must be [n_tokens, 128] with unit inner stride");
}

bool slot_is_i64(const aiter_tensor_t& slot_mapping)
{
    AITER_CHECK(slot_mapping.dtype() == AITER_DTYPE_i32 || slot_mapping.dtype() == AITER_DTYPE_i64,
                "slot_mapping must be int32 or int64, got ", AiterDtype_to_str(slot_mapping.dtype()));
    AITER_CHECK(slot_mapping.dim() == 1 && slot_mapping.stride(0) == 1, "slot_mapping must be a contiguous 1-D tensor");
    return slot_mapping.dtype() == AITER_DTYPE_i64;
}

void check_i32_map(const aiter_tensor_t& t, int64_t B, int64_t next_n, const char* name)
{
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32 && t.dim() == 2 && t.size(0) == B && t.size(1) == next_n &&
                    t.is_contiguous(),
                name, " must be a contiguous int32 [num_requests, next_n] tensor");
}

struct DecodeChecked
{
    DecodeArgs da;
    bool vec;
};

DecodeChecked check_decode(const aiter_tensor_t& tail_kv_cache,
                           const aiter_tensor_t& tail_slot_mapping,
                           const aiter_tensor_t& key,
                           const aiter_tensor_t& slot_score,
                           const aiter_tensor_t& slot_mapping,
                           const aiter_tensor_t& positions,
                           int pool_size)
{
    AITER_CHECK(key.dtype() == AITER_DTYPE_bf16 && key.dim() == 3 && key.size(2) == KPOOL_HEAD_DIM &&
                    key.stride(2) == 1,
                "key must be bf16 [num_requests, next_n, 128] with unit inner stride");
    AITER_CHECK(slot_score.dtype() == AITER_DTYPE_bf16 && slot_score.dim() == 3 &&
                    slot_score.size(0) == key.size(0) && slot_score.size(1) == key.size(1) &&
                    slot_score.size(2) == KPOOL_HEAD_DIM && slot_score.stride(2) == 1,
                "slot_score must be bf16 [num_requests, next_n, 128] with unit inner stride");
    const int64_t B = key.size(0), next_n = key.size(1);
    AITER_CHECK(next_n >= 1 && next_n <= 16, "next_n must be in [1, 16], got ", next_n);
    check_i32_map(tail_slot_mapping, B, next_n, "tail_slot_mapping");
    check_i32_map(slot_mapping, B, next_n, "slot_mapping");
    check_i32_map(positions, B, next_n, "positions");
    AITER_CHECK(tail_kv_cache.dtype() == AITER_DTYPE_bf16 && tail_kv_cache.dim() == 4 &&
                    tail_kv_cache.size(1) == 2 && tail_kv_cache.size(3) == KPOOL_HEAD_DIM &&
                    tail_kv_cache.stride(3) == 1 && tail_kv_cache.stride(2) == KPOOL_HEAD_DIM,
                "tail_kv_cache must be bf16 [num_blocks, 2, ring, 128] with contiguous rows");
    const int ring = static_cast<int>(tail_kv_cache.size(2));
    AITER_CHECK(ring >= pool_size && ring % pool_size == 0, "tail ring (", ring,
                ") must be a multiple of pool_size (", pool_size, ")");
    DecodeChecked r;
    r.da = DecodeArgs{static_cast<const uint16_t*>(key.data_ptr()),
                      static_cast<const uint16_t*>(slot_score.data_ptr()),
                      key.stride(0),
                      key.stride(1),
                      slot_score.stride(0),
                      slot_score.stride(1),
                      static_cast<const int32_t*>(tail_slot_mapping.data_ptr()),
                      static_cast<const int32_t*>(slot_mapping.data_ptr()),
                      static_cast<const int32_t*>(positions.data_ptr()),
                      static_cast<int>(next_n),
                      static_cast<int>(B)};
    r.vec = aligned(key.data_ptr(), key.stride(1), 2, 4) && aligned(slot_score.data_ptr(), slot_score.stride(1), 2, 4) &&
            aligned(tail_kv_cache.data_ptr(), tail_kv_cache.stride(0), 2, 4) &&
            aligned(tail_kv_cache.data_ptr(), tail_kv_cache.stride(1), 2, 4);
    return r;
}

TailArgs tail_args(const aiter_tensor_t& tail_kv_cache)
{
    const int ring = static_cast<int>(tail_kv_cache.size(2));
    return TailArgs{static_cast<uint16_t*>(tail_kv_cache.data_ptr()),
                    tail_kv_cache.stride(0),
                    tail_kv_cache.stride(1),
                    ring,
                    (ring > 0 && (ring & (ring - 1)) == 0) ? __builtin_ctz(static_cast<unsigned>(ring)) : -1};
}

} // namespace

void kpool_compress_insert(aiter_tensor_t& kv_cache,
                           const aiter_tensor_t& k,
                           const aiter_tensor_t& gate_score,
                           const aiter_tensor_t& ape,
                           const aiter_tensor_t& slot_mapping,
                           int pool_size,
                           bool round_scale,
                           int lanes_per_pool)
{
    check_cache(kv_cache);
    check_rows(k, "k");
    check_rows(gate_score, "gate_score");
    AITER_CHECK(gate_score.size(0) == k.size(0), "k and gate_score must have the same number of rows");
    check_ape(ape, pool_size);
    const bool i64 = slot_is_i64(slot_mapping);
    AITER_CHECK(slot_mapping.size(0) == k.size(0), "slot_mapping must have one entry per token");
    const int64_t n_tokens = k.size(0);
    if(n_tokens < pool_size)
        return;
    const int lanes = lanes_per_pool > 0 ? lanes_per_pool : kpool_pick_lanes(n_tokens, pool_size);
    const int page_size = static_cast<int>(kv_cache.size(1));
    const bool vec = aligned(k.data_ptr(), k.stride(0), 2, 16) && aligned(gate_score.data_ptr(), gate_score.stride(0), 2, 16);
    const int flags = (i64 ? 1 : 0) | (page_size > 1 ? 2 : 0) | (round_scale ? 4 : 0);

    HipDeviceGuard device_guard(kv_cache.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    int rc = -3;
    KPOOL_POOL_SWITCH(rc = launch_insert<P>(lanes, vec, i64, k.data_ptr(), gate_score.data_ptr(), ape.data_ptr(),
                                            slot_mapping.data_ptr(), kv_cache.data_ptr(), n_tokens, k.stride(0),
                                            gate_score.stride(0), ape.stride(0), page_size, kv_cache.stride(0),
                                            flags, stream); break;)
    check_rc(rc, "kpool_compress_insert");
}

void kpool_decode_update(aiter_tensor_t& kv_cache,
                         aiter_tensor_t& tail_kv_cache,
                         const aiter_tensor_t& tail_slot_mapping,
                         const aiter_tensor_t& key,
                         const aiter_tensor_t& slot_score,
                         const aiter_tensor_t& ape,
                         const aiter_tensor_t& slot_mapping,
                         const aiter_tensor_t& positions,
                         int pool_size,
                         bool round_scale)
{
    check_cache(kv_cache);
    check_ape(ape, pool_size);
    if(key.size(0) == 0 || key.size(1) == 0)
        return;
    DecodeChecked d = check_decode(tail_kv_cache, tail_slot_mapping, key, slot_score, slot_mapping, positions, pool_size);
    const TailArgs ta = tail_args(tail_kv_cache);
    const int page_size = static_cast<int>(kv_cache.size(1));
    const int flags = (page_size > 1 ? 2 : 0) | (round_scale ? 4 : 0);

    HipDeviceGuard device_guard(kv_cache.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    int rc = -3;
    KPOOL_POOL_SWITCH(rc = launch_decode<P>(d.vec, d.da, ape.data_ptr(), ape.stride(0), ta, kv_cache.data_ptr(),
                                            page_size, kv_cache.stride(0), flags, stream); break;)
    check_rc(rc, "kpool_decode_update");
}

void kpool_mixed_write(aiter_tensor_t& kv_cache,
                       const aiter_tensor_t& ape,
                       int pool_size,
                       bool round_scale,
                       int lanes_per_pool,
                       std::optional<aiter_tensor_t> prefill_k,
                       std::optional<aiter_tensor_t> prefill_gate,
                       std::optional<aiter_tensor_t> prefill_slot_mapping,
                       std::optional<aiter_tensor_t> prefill_tail_slot_mapping,
                       std::optional<aiter_tensor_t> tail_kv_cache,
                       std::optional<aiter_tensor_t> decode_tail_slot_mapping,
                       std::optional<aiter_tensor_t> decode_key,
                       std::optional<aiter_tensor_t> decode_gate,
                       std::optional<aiter_tensor_t> decode_slot_mapping,
                       std::optional<aiter_tensor_t> decode_positions)
{
    check_cache(kv_cache);
    check_ape(ape, pool_size);
    const int page_size = static_cast<int>(kv_cache.size(1));

    const int64_t n_tokens = prefill_k ? prefill_k->size(0) : 0;
    const bool seed        = n_tokens > 0 && prefill_tail_slot_mapping.has_value();
    const bool has_prefill = n_tokens >= pool_size || seed;
    const int64_t B        = decode_key ? decode_key->size(0) : 0;
    const int64_t next_n   = decode_key ? decode_key->size(1) : 0;
    const bool has_decode  = B > 0 && next_n > 0;
    if(!(has_prefill || has_decode))
        return;

    if(has_decode)
    {
        AITER_CHECK(tail_kv_cache && decode_tail_slot_mapping && decode_gate && decode_slot_mapping && decode_positions,
                    "decode_* tensors and tail_kv_cache are required together");
        if(!has_prefill)
        {
            kpool_decode_update(kv_cache, *tail_kv_cache, *decode_tail_slot_mapping, *decode_key, *decode_gate, ape,
                                *decode_slot_mapping, *decode_positions, pool_size, round_scale);
            return;
        }
    }

    TailArgs ta{nullptr, 0, 0, pool_size, -1};
    bool tail_vec = true;
    if(has_decode || seed)
    {
        AITER_CHECK(tail_kv_cache.has_value(), "tail_kv_cache is required for tail seeding and decode");
        AITER_CHECK(tail_kv_cache->dtype() == AITER_DTYPE_bf16 && tail_kv_cache->dim() == 4 &&
                        tail_kv_cache->size(1) == 2 && tail_kv_cache->size(3) == KPOOL_HEAD_DIM &&
                        tail_kv_cache->stride(3) == 1 && tail_kv_cache->stride(2) == KPOOL_HEAD_DIM,
                    "tail_kv_cache must be bf16 [num_blocks, 2, ring, 128] with contiguous rows");
        const int ring = static_cast<int>(tail_kv_cache->size(2));
        AITER_CHECK(ring >= pool_size && ring % pool_size == 0, "tail ring must be a multiple of pool_size");
        ta       = tail_args(*tail_kv_cache);
        tail_vec = aligned(tail_kv_cache->data_ptr(), tail_kv_cache->stride(0), 2, 4) &&
                   aligned(tail_kv_cache->data_ptr(), tail_kv_cache->stride(1), 2, 4);
    }

    DecodeArgs da{nullptr, nullptr, 0, 0, 0, 0, nullptr, nullptr, nullptr, 0, 0};
    bool dec_vec = true;
    if(has_decode)
    {
        DecodeChecked d = check_decode(*tail_kv_cache, *decode_tail_slot_mapping, *decode_key, *decode_gate,
                                       *decode_slot_mapping, *decode_positions, pool_size);
        da      = d.da;
        dec_vec = d.vec;
    }

    AITER_CHECK(prefill_gate && prefill_slot_mapping, "prefill_gate and prefill_slot_mapping are required with prefill_k");
    check_rows(*prefill_k, "prefill_k");
    check_rows(*prefill_gate, "prefill_gate");
    AITER_CHECK(prefill_gate->size(0) == n_tokens, "prefill_k and prefill_gate must have the same number of rows");
    const bool i64 = slot_is_i64(*prefill_slot_mapping);
    AITER_CHECK(prefill_slot_mapping->size(0) == n_tokens, "prefill_slot_mapping must have one entry per token");
    const int32_t* seed_tslot = nullptr;
    if(seed)
    {
        AITER_CHECK(prefill_tail_slot_mapping->dtype() == AITER_DTYPE_i32 && prefill_tail_slot_mapping->dim() == 1 &&
                        prefill_tail_slot_mapping->size(0) == n_tokens && prefill_tail_slot_mapping->stride(0) == 1,
                    "prefill_tail_slot_mapping must be a contiguous int32 [n_tokens] tensor");
        seed_tslot = static_cast<const int32_t*>(prefill_tail_slot_mapping->data_ptr());
    }
    const int lanes = lanes_per_pool > 0 ? lanes_per_pool : kpool_pick_lanes(n_tokens, pool_size);
    const bool vec  = aligned(prefill_k->data_ptr(), prefill_k->stride(0), 2, 16) &&
                     aligned(prefill_gate->data_ptr(), prefill_gate->stride(0), 2, 16) && dec_vec && tail_vec;
    const int flags = (i64 ? 1 : 0) | (page_size > 1 ? 2 : 0) | (round_scale ? 4 : 0) | (seed ? 8 : 0);

    HipDeviceGuard device_guard(kv_cache.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    int rc = -3;
    KPOOL_POOL_SWITCH(rc = launch_mixed<P>(lanes, vec, i64, da, prefill_k->data_ptr(), prefill_gate->data_ptr(),
                                           prefill_slot_mapping->data_ptr(), n_tokens, prefill_k->stride(0),
                                           prefill_gate->stride(0), seed_tslot, ape.data_ptr(), ape.stride(0), ta,
                                           kv_cache.data_ptr(), page_size, kv_cache.stride(0), flags, stream); break;)
    check_rc(rc, "kpool_mixed_write");
}

} // namespace aiter

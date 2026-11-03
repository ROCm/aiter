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
//             (k_s * prob_s rounded, then added: Triton >= 3.8 does not
//             contract this into an FMA; -DKPOOL_ACC_FMA restores the FMA
//             that Triton 3.7 emitted)
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

// global-address-space pointer (explicit, so pointers that went through an
// asm pin still produce global_* instead of flat_* accesses)
template <typename T> using gptr = T __attribute__((address_space(1)))*;
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef float f32x2 __attribute__((ext_vector_type(2)));

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
// The same for a pair: one v_cvt_pk_bf16_f32 (RNE on gfx950, denormals
// preserved) + 2 unpack ops instead of 2 x 4 integer ops. Bit-identical to
// round_bf16 for every finite input (checked byte for byte against vLLM).
#ifndef KPOOL_NO_CVT_BF16
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void round_bf16x2(float& a, float& b) {
  const f32x2 v = {a, b};
  const uint32_t w = __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2_t));
  a = __uint_as_float(w << 16);
  b = __uint_as_float(w & 0xFFFF0000u);
}
#else
__device__ __forceinline__ void round_bf16x2(float& a, float& b) {
  a = round_bf16(a);
  b = round_bf16(b);
}
#endif
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
__device__ __forceinline__ float xor_lane(float v, int lane) {
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
    r = (lane & 4) ? dn : up;
  } else if constexpr (MASK == 8) {
    r = __builtin_amdgcn_update_dpp(iv, iv, 0x128, 0xF, 0xF, false);  // row_ror:8
  } else if constexpr (MASK == 16) {
    auto p = __builtin_amdgcn_permlane16_swap((unsigned)iv, (unsigned)iv, false, false);
    r = (lane & 16) ? (int)p[0] : (int)p[1];
  } else if constexpr (MASK == 32) {
    auto p = __builtin_amdgcn_permlane32_swap((unsigned)iv, (unsigned)iv, false, false);
    r = (lane & 32) ? (int)p[0] : (int)p[1];
#else
  } else if constexpr (MASK < 32) {
    r = __builtin_amdgcn_ds_swizzle(iv, (MASK << 10) | 0x1F);  // bitmode xor MASK
  } else {
    r = __shfl_xor(iv, MASK, 64);
#endif
  }
  return __int_as_float(r);
}

// v_permlane{16,32}_swap with both operands = v: returns {lo, hi} where, for
// MASK 16, lo = v of the lower half of each 32-lane pair replicated to both
// halves and hi = v of the upper half replicated (MASK 32: halves of the wave).
// So in every lane {lo, hi} = {own, partner} below the mask bit and
// {partner, own} above it: a stage can use both without keeping its own v.
#if !defined(KPOOL_LDS_XCHG)
template <int MASK>
__device__ __forceinline__ void swap_pair(float v, float& lo, float& hi) {
  static_assert(MASK == 16 || MASK == 32, "permlane swap stages");
  const unsigned iv = (unsigned)__float_as_int(v);
  if constexpr (MASK == 16) {
    auto p = __builtin_amdgcn_permlane16_swap(iv, iv, false, false);
    lo = __int_as_float((int)p[0]);
    hi = __int_as_float((int)p[1]);
  } else {
    auto p = __builtin_amdgcn_permlane32_swap(iv, iv, false, false);
    lo = __int_as_float((int)p[0]);
    hi = __int_as_float((int)p[1]);
  }
}
#endif

// One cross-lane Hadamard stage: pair (lane, lane ^ MASK); the lower lane
// keeps a + b, the upper one a - b (same op order as the Triton butterfly).
template <int MASK, int DPL>
__device__ __forceinline__ void had_stage_xlane(float (&x)[DPL], int sub) {
  const bool upper = (sub & MASK) != 0;
#pragma unroll
  for (int d = 0; d < DPL; ++d) {
#if !defined(KPOOL_LDS_XCHG)
    if constexpr (MASK >= 16) {
      // lower lane: lo = x, hi = o -> x + o; upper lane: lo = o, hi = x -> o - x
      float lo, hi;
      swap_pair<MASK>(x[d], lo, hi);
      x[d] = upper ? (lo - hi) : (lo + hi);
      continue;
    }
#endif
    float o = xor_lane<MASK>(x[d], sub);
    x[d] = upper ? (o - x[d]) : (x[d] + o);
  }
}

// max of two non-negative finite floats == unsigned max of their bit patterns;
// skips the IEEE-mode canonicalization the compiler puts in front of v_max_f32
__device__ __forceinline__ float max_nonneg(float a, float b) {
  return __uint_as_float(max(__float_as_uint(a), __float_as_uint(b)));
}
// absmax stage over the pair (lane, lane ^ MASK) for non-negative values
template <int MASK>
__device__ __forceinline__ float absmax_xlane(float am, int sub) {
#if !defined(KPOOL_LDS_XCHG)
  if constexpr (MASK >= 16) {
    float lo, hi;
    swap_pair<MASK>(am, lo, hi);
    return max_nonneg(lo, hi);
  }
#endif
  return max_nonneg(am, xor_lane<MASK>(am, sub));
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
struct NoMid {
  __device__ __forceinline__ void operator()(float) const {}
};
// ALL_LANE_STORE (decode, DPL == 2): branch-free store section, every lane
// stores (lane pairs their shared dword, all lanes the scale). Insert keeps
// the exec-masked single-lane stores: with many waves per CU the same-address
// stores cost it ~7 % (mixed 11.3 -> 12.0 us), while a one-shot decode wave
// loses more to the taken branches (instruction-cache misses).
template <int POOL, int LANES, bool ALL_LANE_STORE = false, typename Mid = NoMid>
__device__ __forceinline__ void compress_pool(
    const uint32_t (&gw)[POOL][HEAD_DIM / LANES / 2],
    const uint32_t (&kw)[POOL][HEAD_DIM / LANES / 2],
    const float (&ape)[POOL][HEAD_DIM / LANES],
    gptr<uint8_t> pbase,          // page of the destination slot (cache + page * page_bytes)
    int tok,                      // slot inside the page
    uint32_t scale_off,           // byte offset of the fp32 scale inside the page: page_size * 128 + tok * 4
    bool preshuffle,
    bool round_scale,
    int sub,
    Mid mid = Mid{}) {  // mid(dep): called once every input row has been consumed, dep depends on all of them
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

  // --- pass 2: softmax-weighted sum of k (denom += p; acc += k * p) ---
  // Triton >= 3.8 (vLLM image) keeps `acc += k * prob` as a rounded multiply
  // followed by an add; Triton 3.7 contracted it into an FMA. The two differ
  // in ~11 % of the fp32 sums, which flips the bf16 rounding of x in ~1 per
  // 10^5 elements. Default: match Triton >= 3.8. -DKPOOL_ACC_FMA: Triton 3.7.
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
#ifdef KPOOL_ACC_FMA
      acc[d] = __builtin_fmaf(bf16at(kw[s], d), p, acc[d]);
#else
      {
#pragma clang fp contract(off)  // hipcc contracts even __fmul_rn/__fadd_rn
        acc[d] = acc[d] + bf16at(kw[s], d) * p;
      }
#endif
    }
  }

  mid(acc[0]);

  // --- pooled vector -> bf16 -> Hadamard-128 -> bf16 ---
  float x[DPL];
#pragma unroll
  for (int d = 0; d < DPL; ++d) x[d] = acc[d] / den[d];  // exact division (rcp * mul is not bit-exact)
#pragma unroll
  for (int d = 0; d < DPL; d += 2) round_bf16x2(x[d], x[d + 1]);

  had_stages_inlane<DPL>(x);
  if constexpr (LANES >= 2) had_stage_xlane<1, DPL>(x, sub);
  if constexpr (LANES >= 4) had_stage_xlane<2, DPL>(x, sub);
  if constexpr (LANES >= 8) had_stage_xlane<4, DPL>(x, sub);
  if constexpr (LANES >= 16) had_stage_xlane<8, DPL>(x, sub);
  if constexpr (LANES >= 32) had_stage_xlane<16, DPL>(x, sub);
  if constexpr (LANES >= 64) had_stage_xlane<32, DPL>(x, sub);
#pragma unroll
  for (int d = 0; d < DPL; ++d) x[d] = x[d] * HAD_SCALE;
#pragma unroll
  for (int d = 0; d < DPL; d += 2) round_bf16x2(x[d], x[d + 1]);

  // --- per-vector absmax fp8 quant (all values finite and >= 0: unsigned
  // integer max is the exact float max, without canonicalization) ---
  float am = 0.0f;
#pragma unroll
  for (int d = 0; d < DPL; ++d) am = max_nonneg(am, fabsf(x[d]));
  if constexpr (LANES >= 2) am = absmax_xlane<1>(am, sub);
  if constexpr (LANES >= 4) am = absmax_xlane<2>(am, sub);
  if constexpr (LANES >= 8) am = absmax_xlane<4>(am, sub);
  if constexpr (LANES >= 16) am = absmax_xlane<8>(am, sub);
  if constexpr (LANES >= 32) am = absmax_xlane<16>(am, sub);
  if constexpr (LANES >= 64) am = absmax_xlane<32>(am, sub);
  am = max_nonneg(am, 1e-4f);
  // x / scale: under round_scale the scale is 2^e, so the division is the
  // exact ldexp(x, -e) (no rcp, no division sequence); otherwise exact division.
  float scale;
  float q[DPL];
  if (round_scale) {
    const float e = __builtin_ceilf(fast_log2(am * FP8_MAX_INV));
    scale = fast_exp2(e);
#ifndef KPOOL_NO_LDEXP
    const int ne = -(int)e;
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = __builtin_amdgcn_ldexpf(x[d], ne);
#else
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = x[d] / scale;
#endif
  } else {
    scale = am * FP8_MAX_INV;
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = x[d] / scale;
  }
#pragma unroll
  for (int d = 0; d < DPL; ++d) q[d] = __builtin_amdgcn_fmed3f(q[d], -FP8_MAX, FP8_MAX);  // == min(max(q, -448), 448)
  constexpr int NQW = DPL >= 4 ? DPL / 4 : 1;
  uint32_t qw[NQW];
  if constexpr (DPL == 2) {
    qw[0] = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
  }
#pragma unroll
  for (int c = 0; c < DPL / 4; ++c) {
    uint32_t w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q[c * 4 + 0], q[c * 4 + 1], 0, false);
    w = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(q[c * 4 + 2], q[c * 4 + 3], (int)w, true);
    qw[c] = w;
  }

  // --- store: a 32-bit lane offset on the page base, so a wave-uniform page
  // (decode) gives the saddr form (SGPR base + VGPR offset) ---
  if constexpr (DPL == 2 && ALL_LANE_STORE) {
    // Branch-free (every taken branch on this one-shot path costs an
    // instruction-cache miss, ~500 clocks on the ATT ledger): the layout is
    // a select, lane pairs (2k, 2k+1) hold dims 4k..4k+3 and both lanes
    // store the same dword to the same address (2-byte stores per lane are
    // far slower in the memory pipeline), and every lane stores the
    // wave-uniform scale to its one address.
    const int d0e = d0 & ~3;  // first dim of the pair
    const uint32_t ps = (uint32_t)((tok >> 4) * (16 * HEAD_DIM) + (d0e >> 4) * 256 + (tok & 15) * 16 + (d0e & 15));
    const uint32_t lin = (uint32_t)(tok * HEAD_DIM + d0e);
    const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0]), sub));
#ifdef KPOOL_MASKED_STORE
    if ((sub & 1) == 0) *(gptr<uint32_t>)(pbase + (preshuffle ? ps : lin)) = (qw[0] & 0xFFFFu) | (other << 16);
    if (sub == 0) *(gptr<float>)(pbase + scale_off) = scale;
#else
    const bool odd = (sub & 1) != 0;
    const uint32_t lo = odd ? other : qw[0], hi = odd ? qw[0] : other;
    *(gptr<uint32_t>)(pbase + (preshuffle ? ps : lin)) = (lo & 0xFFFFu) | (hi << 16);
    *(gptr<float>)(pbase + scale_off) = scale;
#endif
    return;
  }
  if (preshuffle) {
    // byte offset of (tok, dim) = (tok/16)*16*128 + (dim/16)*256 + (tok%16)*16 + dim%16
    gptr<uint8_t> kp = pbase + (uint32_t)((tok >> 4) * (16 * HEAD_DIM) + (d0 >> 4) * 256 + (tok & 15) * 16 + (d0 & 15));
    if constexpr (DPL == 16) {
      *(gptr<u32x4>)kp = u32x4{qw[0], qw[1], qw[2], qw[3]};
    } else if constexpr (DPL == 8) {
      *(gptr<u32x2>)kp = u32x2{qw[0], qw[1]};
    } else if constexpr (DPL == 2) {
      // lane pairs (2k, 2k+1) hold dims 4k..4k+3: the even lane stores one
      // dword (2-byte stores per lane are far slower in the memory pipeline)
      const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0]), sub));
      if ((sub & 1) == 0) *(gptr<uint32_t>)kp = (qw[0] & 0xFFFFu) | (other << 16);
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) ((gptr<uint32_t>)kp)[c] = qw[c];
    }
  } else {
    gptr<uint8_t> kp = pbase + (uint32_t)(tok * HEAD_DIM + d0);
    if constexpr (DPL == 2) {
      const uint32_t other = __float_as_uint(xor_lane<1>(__uint_as_float(qw[0]), sub));
      if ((sub & 1) == 0) *(gptr<uint32_t>)kp = (qw[0] & 0xFFFFu) | (other << 16);
    } else {
#pragma unroll
      for (int c = 0; c < DPL / 4; ++c) ((gptr<uint32_t>)kp)[c] = qw[c];
    }
  }
  if (sub == 0) {
    *(gptr<float>)(pbase + scale_off) = scale;
  }
}

// page / token-in-page of a cache slot (page_shift = log2(page_size), or -1
// when page_size is not a power of two)
template <typename LocT>
__device__ __forceinline__ LocT page_of(LocT loc, int page_size, int page_shift, int& tok) {
  // (the power-of-two case is the fall-through: a taken branch over the
  // division sequence costs a one-shot wave an instruction-cache miss)
  if (__builtin_expect(page_shift >= 0, 1)) {
    tok = (int)(loc & (page_size - 1));
    return loc >> page_shift;
  }
  tok = (int)(loc % page_size);
  return loc / page_size;
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
  int block_elems;       // stride(0) < 2^31 (checked); vLLM pads the tail page to the KV
  int head_elems;        // stride(1)  page, so block * stride(0) is formed in 64 bits
  int ring_mask;         // ring - 1; ring = shape[2] is a power of two (checked by the entry points)
  int ring_shift;        // log2(ring)
};

// Wave-uniform `x >> 31` kept as an SALU shift: the compiler turns it into
// s_cmp + s_cselect_b64 lane masks (and branches on them through a v_cndmask
// / v_cmp round trip), which doubles the SALU chain in front of the ring loads.
__device__ __forceinline__ uint32_t s_bit31(uint32_t x) {
  uint32_t r;
  // readfirstlane: free for a value already in an SGPR, and keeps the "s"
  // constraint satisfiable when the compiler cannot prove x uniform (with
  // KPOOL_META_SMEM it handed the asm a VGPR and failed to assemble)
  asm("s_lshr_b32 %0, %1, 31" : "=s"(r) : "s"(__builtin_amdgcn_readfirstlane(x)));
  return r;
}

// (a < b) ? x : y for wave-uniform pointers in two SALU instructions
// (s_cmp + s_cselect_b64). Written as C, the compiler materializes the
// condition as a 64-bit lane mask and spends ~5 instructions per select.
// (readfirstlane on the inputs: free for values already in SGPRs, and keeps
// the "s" constraints satisfiable when the compiler cannot prove uniformity)
__device__ __forceinline__ uint64_t s_uniform64(uint64_t v) {
  const uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t)v);
  const uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t)(v >> 32));
  return ((uint64_t)hi << 32) | lo;
}
template <typename T>
__device__ __forceinline__ gptr<T> s_sel_lt(int a, int b, gptr<T> x, gptr<T> y) {
  uintptr_t r;
  const uint64_t xv = s_uniform64((uint64_t)(uintptr_t)x), yv = s_uniform64((uint64_t)(uintptr_t)y);
  asm("s_cmp_lt_i32 %1, %2\n\ts_cselect_b64 %0, %3, %4"
      : "=s"(r)
      : "s"(__builtin_amdgcn_readfirstlane(a)), "s"(__builtin_amdgcn_readfirstlane(b)), "s"(xv), "s"(yv)
      : "scc");
  return (gptr<T>)r;
}

// ring is a power of two (AITER_CHECK in tail_args): shift / mask only, so the
// metadata loads of a decode request are not split by a data-dependent branch.
__device__ __forceinline__ int ring_div(int x, const TailArgs& ta) { return x >> ta.ring_shift; }
__device__ __forceinline__ int ring_mod(int x, const TailArgs& ta) { return x & ta.ring_mask; }

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
    const SlotT* __restrict__ seed_tslot,
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
      seed_ts[s] = tok < n_tokens ? (int)seed_tslot[tok] : -1;
      seed_ahead[s] = (tok + POOL < n_tokens) ? (int)seed_tslot[tok + POOL] : -1;
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
    int tok;
    const int64_t page = page_of<int64_t>(loc, page_size, page_shift, tok);
    compress_pool<POOL, LANES>(gw, kw, apev, (gptr<uint8_t>)(cache + page * page_bytes), tok,
                               (uint32_t)(page_size * HEAD_DIM + tok * 4), preshuffle, round_scale, sub);
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
      const uint64_t dst = (uint64_t)(uint32_t)blk * (uint32_t)ta.block_elems + (uint32_t)(ring_mod(ts, ta) * HEAD_DIM + d0);
      uint32_t* pk = reinterpret_cast<uint32_t*>(ta.tail + dst);
      uint32_t* ps = reinterpret_cast<uint32_t*>(ta.tail + (uint32_t)ta.head_elems + dst);
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
  TailArgs ta{nullptr, 0, 0, 0, 0};
  insert_window<POOL, LANES, VEC, SlotT>(win, k, gate, ape, slot_mapping, cache, n_tokens, n_windows, k_stride,
                                         gate_stride, ape_stride, page_size, page_shift, page_bytes, flags & 7,
                                         nullptr, ta, grp, sub);
}

// ===========================================================================
// Decode update: one wave (64 lanes, 2 dims per lane) per request.
// ===========================================================================

constexpr int DEC_LANES = 64;
constexpr int DEC_DPL = HEAD_DIM / DEC_LANES;  // 2 dims = one dword per lane per row
#ifndef KPOOL_DEC_BLOCK_WAVES
#define KPOOL_DEC_BLOCK_WAVES 4
#endif
constexpr int DEC_BLOCK_WAVES = KPOOL_DEC_BLOCK_WAVES;  // requests per workgroup in the standalone decode kernel

struct DecodeArgs {
  const uint16_t* key;       // [B, next_n, 128] bf16
  const uint16_t* score;     // [B, next_n, 128] bf16
  const void* tail_slot;     // [B, next_n] int32 or int64 (IdxT)
  const void* slot_map;      // [B, next_n] int32 or int64 (IdxT)
  const void* positions;     // [B, next_n] int32 or int64 (IdxT)
  int key_sb, key_st, ss_sb, ss_st;  // strides in elements (checked to fit int32)
  int next_n;
  int num_requests;
};

// All decode kernel arguments in one 128 B, 16 B aligned block: two
// s_load_dwordx16 fetch everything (the compiler splits a long parameter list
// into many misaligned s_loads).
struct __attribute__((aligned(16))) DecodeKArgs {
  const uint16_t* key;       // [B, next_n, 128] bf16
  const uint16_t* score;     // [B, next_n, 128] bf16
  const void* tail_slot;     // [B, next_n] int32 or int64 (IdxT)
  const void* slot_map;      // [B, next_n] int32 or int64 (IdxT)
  const void* positions;     // [B, next_n] int32 or int64 (IdxT)
  const float* ape;          // [POOL, 128] fp32, contiguous
  uint16_t* tail;            // [num_blocks, 2, ring, 128] bf16
  uint8_t* cache;            // [num_blocks, page, 132] u8
  int key_sb, key_st, ss_sb, ss_st;  // strides in elements (checked to fit int32)
  int next_n, num_requests;
  int block_elems, head_elems;       // tail strides (elements, each < 2^31; block offsets are 64-bit)
  int ring_mask, ring_shift;         // ring - 1, log2(ring)
  int page_size, page_shift;         // page_shift = log2(page_size) or -1
  uint32_t page_bytes;               // page_size * 132
  int flags;                         // bit1: preshuffle, bit2: round_scale
  int pad0, pad1;
};
static_assert(sizeof(DecodeKArgs) == 128, "two s_load_dwordx16");

// Kernarg preload (the aiter JIT passes -mllvm --amdgpu-kernarg-preload-count=32
// unless AITER_DISABLE_KERNARG_PRELOAD; honoured by hipcc >= 7.2 on gfx950,
// ignored by 7.0): the first 14 dwords of the kernargs arrive in
// user SGPRs with the wave, so the kernel issues its first loads without an
// SMEM round trip. The pass preloads whole arguments only, so the standalone
// decode kernel takes the fields needed before the first loads as scalars
// (5 pointers + 4 ints = 14 dwords) and the rest, needed only after the
// metadata round trip, as one 64 B struct fetched by s_load in its shadow.
struct __attribute__((aligned(16))) DecodeKRest {
  const float* ape;
  uint16_t* tail;
  uint8_t* cache;
  int key_st, ss_st;
  int block_elems, head_elems, ring_mask, ring_shift;
  int page_size, page_shift;
  uint32_t page_bytes;
  int flags;
};
static_assert(sizeof(DecodeKRest) == 64, "one s_load_dwordx16");

__host__ __device__ __forceinline__ DecodeKArgs make_kargs(const DecodeArgs& da, const float* ape, TailArgs ta,
                                                           uint8_t* cache, int page_size, int page_shift,
                                                           uint32_t page_bytes, int flags) {
  DecodeKArgs k;
  k.key = da.key; k.score = da.score; k.tail_slot = da.tail_slot; k.slot_map = da.slot_map;
  k.positions = da.positions; k.ape = ape; k.tail = ta.tail; k.cache = cache;
  k.key_sb = da.key_sb; k.key_st = da.key_st; k.ss_sb = da.ss_sb; k.ss_st = da.ss_st;
  k.next_n = da.next_n; k.num_requests = da.num_requests;
  k.block_elems = ta.block_elems; k.head_elems = ta.head_elems; k.ring_mask = ta.ring_mask; k.ring_shift = ta.ring_shift;
  k.page_size = page_size; k.page_shift = page_shift; k.page_bytes = page_bytes; k.flags = flags;
  k.pad0 = k.pad1 = 0;
  return k;
}

// Empty asm that requires x in SGPRs: forces a wave-uniform value to be
// materialized there (kernargs: fetched before this point; derived pointers:
// kept apart from the per-lane offset, so loads take the saddr form).
#define KPOOL_PIN_S(x) asm("" : "+s"(x))
// Pointers are pinned as integers and come back in the global address space
// (the asm hides their provenance; a generic pointer would turn every access
// into a flat_load that also waits on lgkmcnt).
template <typename T>
__device__ __forceinline__ gptr<T> pin_global(T* p) {
  uintptr_t v = reinterpret_cast<uintptr_t>(p);
  KPOOL_PIN_S(v);
  return (gptr<T>)v;
}
template <typename T>
__device__ __forceinline__ gptr<T> pin_global(gptr<T> p) {
  uintptr_t v = (uintptr_t)p;
  KPOOL_PIN_S(v);
  return (gptr<T>)v;
}

// This lane's dword (2 bf16) of a 128-dim row whose address is wave-uniform:
// SGPR base + 32-bit lane offset (saddr form, no 64-bit VALU address math).
template <bool VEC>
__device__ __forceinline__ uint32_t load_dw(gptr<const uint16_t> row, int sub) {
  if constexpr (VEC) {
    return ((gptr<const uint32_t>)row)[(uint32_t)sub];
  } else {
    gptr<const uint16_t> p = row + 2 * (uint32_t)sub;
    return (uint32_t)p[0] | ((uint32_t)p[1] << 16);
  }
}
__device__ __forceinline__ void store_dw(gptr<uint16_t> row, int sub, uint32_t v) {
  ((gptr<uint32_t>)row)[(uint32_t)sub] = v;
}

// Per-token metadata lives in clang ext vectors (SSA values, never an
// alloca): a select chain over array elements would be folded by LLVM into a
// dynamically indexed load, and the array would then land in LDS or scratch.
template <int N> struct Vec {
  typedef int i __attribute__((ext_vector_type(N)));
  typedef unsigned u __attribute__((ext_vector_type(N)));
};

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
// K / gate row. Memory chain: kernargs -> K / gate / APE / metadata (one
// round trip) -> ring rows of the first completion -> math -> stores.
// Completions and stashes are issued in token order, so a later writer of the
// same cache / ring slot wins as in the walk (same lane, same address:
// program order).
template <int POOL, int NMAX, bool VEC, typename IdxT>
__device__ __forceinline__ void decode_request(
    int req,  // wave-uniform
    const DecodeKArgs& d,
    int sub,    // lane, 0..63
    uint32_t vmask = ~0u) {  // wave-uniform; 0 = padding wave (loads only, no stores)
  typedef typename Vec<NMAX>::i ivec;
  typedef typename Vec<NMAX>::u uvec;
  // IdxT: int32 or int64 metadata (vLLM keeps its slot mappings in int64);
  // the low dword is loaded either way, so the two instantiations differ
  // only in the element stride.
  gptr<const IdxT> m_ts = (gptr<const IdxT>)d.tail_slot, m_loc = (gptr<const IdxT>)d.slot_map,
                   m_pos = (gptr<const IdxT>)d.positions;
  const gptr<const float> ape_g = (gptr<const float>)d.ape;
  const gptr<uint16_t> tail = (gptr<uint16_t>)d.tail;
  const gptr<uint8_t> cache_g = (gptr<uint8_t>)d.cache;
  const TailArgs ta{d.tail, d.block_elems, d.head_elems, d.ring_mask, d.ring_shift};
  const int page_size = d.page_size, page_shift = d.page_shift, flags = d.flags;
  const uint32_t page_bytes = d.page_bytes;
  const bool preshuffle = (flags & 2) != 0, round_scale = (flags & 4) != 0;
  const int n = d.next_n;  // 1 <= n <= NMAX
  __builtin_assume(n >= 1 && n <= NMAX);
  const int base = req * n;

  // --- phase 1: K / gate of the request's tokens (one dword per lane), then
  // the metadata (scalar loads); no wait in between. With NMAX >= POOL the
  // regular path below reads the token rows through the virtual rows
  // instead, so the early loads would be duplicates ---
  constexpr bool EARLY_ROWS = NMAX < POOL;
  uvec rk, rs;
  // B * next_n * 128 < 2^31 (checked): element offsets fit in 32 bits
  uintptr_t kb = reinterpret_cast<uintptr_t>(d.key + (uint32_t)(req * d.key_sb));
  uintptr_t sb = reinterpret_cast<uintptr_t>(d.score + (uint32_t)(req * d.ss_sb));
  // The first loads "depend" on every other kernarg (empty asm): the compiler
  // fetches all kernargs in one SMEM batch with one wait, instead of one lazy
  // batch and one wait per first use along the chain.
#ifdef KPOOL_PIN_ALL_KARGS
  asm("" : "+s"(kb), "+s"(sb), "+s"(m_ts), "+s"(m_loc), "+s"(m_pos)
         : "s"(ape_g), "s"(tail), "s"(ta.block_elems),
           "s"(ta.head_elems), "s"(ta.ring_mask), "s"(ta.ring_shift), "s"(cache_g), "s"(page_size), "s"(page_shift),
           "s"(page_bytes), "s"(flags), "s"(d.key_st), "s"(d.ss_st));
#endif
  const gptr<const uint16_t> kbase = (gptr<const uint16_t>)kb;
  const gptr<const uint16_t> sbase = (gptr<const uint16_t>)sb;
  auto load_rows = [&]() {
#pragma unroll
    for (int i = 0; i < NMAX; ++i) {
      // (NMAX == 1: a literal 0, or the compiler keeps n - 1 and fetches the
      // stride before the first load)
      const int j = NMAX == 1 ? 0 : min(i, n - 1);
      rk[i] = load_dw<VEC>(kbase + (uint32_t)(j * d.key_st), sub);
      rs[i] = load_dw<VEC>(sbase + (uint32_t)(j * d.ss_st), sub);
    }
  };
  if constexpr (EARLY_ROWS) load_rows();
  ivec loc, pos, ts;
  // Metadata: the three words per token wait ~1100 clocks on the ATT ledger
  // either way (three s_load_dword in one batch, or one global_load_dword
  // with a per-lane pointer select + v_readlane). Measured on the remote
  // MI355X (B=1024): next_n=1 closing wave 4542 clocks with the scalar
  // loads vs 4868 with the vector load (fewer VALU, no readlane); next_n=4
  // 7.2 vs 7.9 us with the vector load (12 scalar loads otherwise). So the
  // one-token kernel uses the scalar loads, NMAX > 1 the vector load.
  // KPOOL_META_SMEM / KPOOL_META_VMEM force one path for every NMAX.
#if defined(KPOOL_META_SMEM)
  constexpr bool META_SMEM = true;
#elif defined(KPOOL_META_VMEM)
  constexpr bool META_SMEM = false;
#else
  constexpr bool META_SMEM = NMAX == 1;
#endif
  // vector-load path: lane l holds the three words of token min(l, n - 1)
  // (three loads, SGPR base + lane offset: issued a few instructions after
  // the wave starts; a lane-wise pointer select would cost ~20 VALU first)
  int32_t v_ts = 0, v_loc = 0, v_pos = 0;
  const int v_tok = min(sub, n - 1);
  if constexpr (META_SMEM && NMAX > 1 && NMAX <= 8 && sizeof(IdxT) == 4) {
    // next_n == NMAX (exact instantiations up to 8): the request's NMAX words
    // of each array are contiguous, one s_load_dwordxN per array (scalar
    // loads need dword alignment only)
    loc = *(gptr<const ivec>)(m_loc + (uint32_t)base);
    pos = *(gptr<const ivec>)(m_pos + (uint32_t)base);
    ts = *(gptr<const ivec>)(m_ts + (uint32_t)base);
  } else if constexpr (META_SMEM) {
#pragma unroll
    for (int i = 0; i < NMAX; ++i) {
      const uint32_t j = (uint32_t)(base + (NMAX == 1 ? 0 : min(i, n - 1)));
      loc[i] = (int32_t)m_loc[j];
      pos[i] = (int32_t)m_pos[j];
      ts[i] = (int32_t)m_ts[j];
    }
  } else {
    const uint32_t jl = (uint32_t)(base + v_tok);
    v_ts = (int32_t)m_ts[jl];
    v_loc = (int32_t)m_loc[jl];
    v_pos = (int32_t)m_pos[jl];
  }
  // every token's metadata as scalars (the one-token path)
  auto read_meta = [&]() {
    if constexpr (!META_SMEM) {
#pragma unroll
      for (int i = 0; i < NMAX; ++i) {
        ts[i] = __builtin_amdgcn_readlane(v_ts, i);
        loc[i] = __builtin_amdgcn_readlane(v_loc, i);
        pos[i] = __builtin_amdgcn_readlane(v_pos, i);
      }
    }
  };
  // APE rows: loaded only when a pool closes (one step in POOL), together
  // with the ring rows; the row pointer is pinned uniform so the lane offset
  // stays a 32-bit saddr offset.
  float apev[POOL][DEC_DPL];
  auto load_ape = [&]() {
    // ape is contiguous (checked by the entry point): row s at s * HEAD_DIM, immediate offsets
#pragma unroll
    for (int s = 0; s < POOL; ++s) {
      const f32x2 a = ((gptr<const f32x2>)(ape_g + s * HEAD_DIM))[(uint32_t)sub];
      apev[s][0] = a[0];
      apev[s][1] = a[1];
    }
  };
  // K row (block b, ring slot ph) of the tail; the gate row is ta.head_elems further
  auto tail_row = [&](int b, int ph) -> gptr<uint16_t> {
    return tail + (uint64_t)(uint32_t)b * (uint32_t)ta.block_elems + ph * HEAD_DIM;
  };
  auto write_pool = [&](const uint32_t (&gw)[POOL][1], const uint32_t (&kw)[POOL][1], int loc_i, auto mid) {
    int tok;
    const int page = page_of<int>(loc_i, page_size, page_shift, tok);
    compress_pool<POOL, DEC_LANES, true>(gw, kw, apev, cache_g + (uint64_t)(uint32_t)page * page_bytes, tok,
                                         (uint32_t)(page_size * HEAD_DIM + tok * 4), preshuffle, round_scale, sub, mid);
  };

  if constexpr (NMAX > 1) {
    // ===== regular request (the production shape): next_n == NMAX tokens
    // with contiguous rows, every token stashes, the positions are
    // consecutive, all in one tail block, and the tokens do not wrap onto
    // the ring rows a pool still reads (r + NMAX <= ring). Then the pools
    // have a static layout over the "virtual rows"
    //   V[j], j = 0 .. NMAX + POOL - 2:  ring row pstart + j  (j < r),
    //                                    token j - r          (r <= j < r + NMAX),
    // with r = pos0 % POOL and pstart the ring row of slot 0 of token 0's
    // pool: pool c is V[c*POOL .. c*POOL + POOL-1] and closes at its last row
    // when that token exists and has a cache slot. No per-slot source search
    // (the ordered walk's "latest earlier stash" is always the token itself,
    // because a token's ring row is never read by a later pool here), no
    // runtime register indexing: the shift by r is absorbed in two row
    // bases (ring rows from pstart, token rows from token -r), every row is
    // base + j * 256 B (an immediate), the rest is straight-line code. =====
    constexpr int NV = NMAX + POOL - 1;
    constexpr int NC = NV / POOL;  // pools that can close
    int ts0, pos0;
    if constexpr (META_SMEM) {
      ts0 = ts[0];
      pos0 = pos[0];
    } else {
      ts0 = __builtin_amdgcn_readlane(v_ts, 0);
      pos0 = __builtin_amdgcn_readlane(v_pos, 0);
    }
    const int safe0 = max(pos0, 0);
    const int r = safe0 % POOL;  // pool slot of token 0
    // one flag, no short-circuit: a single branch, so the fast path is laid
    // out as the fall-through
    uint32_t regular;
    if constexpr (META_SMEM) {
      regular = 1u;
#pragma unroll
      for (int i = 1; i < NMAX; ++i) {
        const int e_ts = (ts0 & ~ta.ring_mask) | ((ts0 + i) & ta.ring_mask);
        regular &= (uint32_t)(pos[i] == pos0 + i) & (uint32_t)(ts[i] == e_ts);
      }
    } else {
      // one compare per lane: does lane (token) hold the expected words?
      const int e_ts = (ts0 & ~ta.ring_mask) | ((ts0 + v_tok) & ta.ring_mask);
      const int e_pos = pos0 + v_tok;
      regular = (uint32_t)(__builtin_amdgcn_ballot_w64((v_ts == e_ts) & (v_pos == e_pos)) == ~0ull);
    }
    regular &= s_bit31(~(uint32_t)(pos0 | ts0)) & (uint32_t)(r + NMAX <= ta.ring_mask + 1) & (uint32_t)(n == NMAX) &
               (uint32_t)(d.key_st == HEAD_DIM) & (uint32_t)(d.ss_st == HEAD_DIM);
    if (__builtin_expect(regular != 0u, 1)) {
      const int blk0 = ring_div(ts0, ta);
      const int pstart = ring_mod(safe0, ta) - r;  // >= 0: ring % POOL == 0
      // row bases: virtual row j is ring row tkp + j (K; the gate row is
      // head_elems further) and, when live, token row kbr + j. The ring
      // rows read (j < r) never wrap; the token rows stash into
      // (pstart + j) mod ring: rows j >= wrap use the base one ring lower.
      const gptr<uint16_t> tkp = tail + (uint64_t)(uint32_t)blk0 * (uint32_t)ta.block_elems + pstart * HEAD_DIM;
      const gptr<uint16_t> tgp = tkp + (uint32_t)ta.head_elems;
      const int ring = ta.ring_mask + 1;
      const int wrap = ring - pstart;
      const gptr<uint16_t> tkw = tkp + (ring * -HEAD_DIM);
      const gptr<uint16_t> tgw = tgp + (ring * -HEAD_DIM);
      const gptr<const uint16_t> kbr = kbase + (r * -HEAD_DIM);
      const gptr<const uint16_t> sbr = sbase + (r * -HEAD_DIM);
      // pools that close: pool c at token POOL-1 + c*POOL - r
      uint32_t amask = 0;
      int cloc[NC];
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        const int tj = POOL - 1 + c * POOL - r;  // >= 0
        if constexpr (META_SMEM) {
          cloc[c] = vpick<NMAX>(loc, tj);
        } else {
          cloc[c] = __builtin_amdgcn_readlane(v_loc, min(tj, NMAX - 1));
        }
        const bool exists = (POOL - 1 + c * POOL - NMAX) < r;  // token tj < NMAX
        amask |= ((uint32_t)exists & s_bit31(~(uint32_t)cloc[c])) << c;
      }
      amask &= vmask;
      const bool do_stash = vmask != 0;  // padding wave: no stores at all
      // the stash: with early token rows by token, else by live virtual row;
      // the stored words "depend" on dep (the pooled accumulator) so the
      // stores cannot be hoisted above the use of the last pool row
      uint32_t vk[NV], vg[NV];
      auto stash_all = [&](float dep) {
        if constexpr (EARLY_ROWS) {
          // token t is virtual row r + t
          const gptr<uint16_t> k0 = tkp + (uint32_t)(r * HEAD_DIM), k0w = tkw + (uint32_t)(r * HEAD_DIM);
          const gptr<uint16_t> g0 = tgp + (uint32_t)(r * HEAD_DIM), g0w = tgw + (uint32_t)(r * HEAD_DIM);
#pragma unroll
          for (int t = 0; t < NMAX; ++t) {
            uint32_t k = rk[t], g = rs[t];
            asm volatile("" : "+v"(k), "+v"(g) : "v"(dep));
            // the selected bases are pinned to SGPRs: otherwise the compiler
            // builds one 64-bit VGPR address per row (and spills them)
            const gptr<uint16_t> bk = s_sel_lt(r + t, wrap, k0, k0w), bg = s_sel_lt(r + t, wrap, g0, g0w);
            store_dw(bk + t * HEAD_DIM, sub, k);
            store_dw(bg + t * HEAD_DIM, sub, g);
          }
        } else {
          // pstart and wrap are multiples of POOL: the rows of one pool group
          // share a base (one SALU select per group; a C select here became a
          // 64-bit VGPR address per row, spilled)
#pragma unroll
          for (int c = 0; c < (NV + POOL - 1) / POOL; ++c) {
            const gptr<uint16_t> bk = s_sel_lt(c * POOL, wrap, tkp, tkw), bg = s_sel_lt(c * POOL, wrap, tgp, tgw);
#pragma unroll
            for (int j = c * POOL; j < min(c * POOL + POOL, NV); ++j) {
              // live: r <= j < r + NMAX; rows POOL-1 .. NMAX-1 always are
              const bool live = j < POOL - 1 ? j >= r : (j < NMAX ? true : (j - NMAX) < r);
              if (live) {
                uint32_t k = vk[j], g = vg[j];
                asm volatile("" : "+v"(k), "+v"(g) : "v"(dep));
                store_dw(bk + j * HEAD_DIM, sub, k);
                store_dw(bg + j * HEAD_DIM, sub, g);
              }
            }
          }
        }
      };
      auto load_virtual = [&]() {
        // gate rows and APE first (the softmax needs only them), K rows after.
        // Rows j >= NMAX are dead unless j - NMAX < r: a dead one re-reads row
        // j - NMAX (always a token row then) instead of running past the
        // request's rows.
        gptr<const uint16_t> gsrc[NV], ksrc[NV];
        // row NMAX + i is live iff i < r (then token row kbr + NMAX + i); a dead
        // one re-reads token row i through the base kbr: same immediate, one
        // base select
        const gptr<const uint16_t> kbr2 = kbr + NMAX * HEAD_DIM, sbr2 = sbr + NMAX * HEAD_DIM;
#pragma unroll
        for (int j = 0; j < NV; ++j) {
          if (j < POOL - 1) {
            gsrc[j] = s_sel_lt(j, r, (gptr<const uint16_t>)tgp, sbr) + j * HEAD_DIM;
            ksrc[j] = s_sel_lt(j, r, (gptr<const uint16_t>)tkp, kbr) + j * HEAD_DIM;
          } else if (j < NMAX) {
            gsrc[j] = sbr + j * HEAD_DIM;
            ksrc[j] = kbr + j * HEAD_DIM;
          } else {
            gsrc[j] = s_sel_lt(j - NMAX, r, sbr2, sbr) + (j - NMAX) * HEAD_DIM;
            ksrc[j] = s_sel_lt(j - NMAX, r, kbr2, kbr) + (j - NMAX) * HEAD_DIM;
          }
        }
#pragma unroll
        for (int j = 0; j < NV; ++j) vg[j] = load_dw<VEC>(gsrc[j], sub);
        if (amask) load_ape();
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < NV; ++j) vk[j] = load_dw<VEC>(ksrc[j], sub);
      };
      if constexpr (EARLY_ROWS) {
        if (amask == 0) {
          if (do_stash) stash_all(0.0f);
          return;
        }
      }
      load_virtual();
      // completions in token order; the stash is issued inside the last one
      // (its store acks drain under the Hadamard and the quantization)
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        if ((amask >> c) & 1u) {
          const bool last = (amask >> (c + 1)) == 0u;
          uint32_t gw[POOL][1], kw[POOL][1];
#pragma unroll
          for (int s = 0; s < POOL; ++s) {
            gw[s][0] = vg[c * POOL + s];
            kw[s][0] = vk[c * POOL + s];
          }
          write_pool(gw, kw, cloc[c], [&](float dep) {
            if (last && do_stash) stash_all(dep);
          });
        }
      }
      if (amask == 0 && do_stash) stash_all(0.0f);
      return;
    }
    // ===== irregular request (padding, negative slots, gaps, a ring equal
    // to the pool, ...): the ordered walk, one completion at a time. Token
    // t's metadata lives in lane t; the source of a pool slot (the latest
    // earlier token that stashed into the same ring row, else the ring) is
    // a ballot over the lanes; each completion loads its own rows (one
    // round trip each). Correctness matters here, not speed. =====
    if (vmask == 0) return;
    int l_ts, l_loc, l_pos;  // this lane's token (min(sub, n - 1))
    if constexpr (META_SMEM) {
      const uint32_t jm = (uint32_t)(base + v_tok);
      l_ts = m_ts[jm];
      l_loc = m_loc[jm];
      l_pos = m_pos[jm];
    } else {
      l_ts = v_ts;
      l_loc = v_loc;
      l_pos = v_pos;
    }
    load_rows();  // rk / rs by token, for the stash
    const int l_safe = max(l_pos, 0);
    const int l_phys = ring_mod(l_safe, ta);
    const int l_blk = ring_div(max(l_ts, 0), ta);
    const bool l_ok = sub < n;
    const bool l_stash = l_ok && l_pos >= 0 && l_ts >= 0;
    const bool l_close = l_ok && l_pos >= 0 && l_loc >= 0 && l_safe % POOL == POOL - 1;
    const uint64_t smask = __builtin_amdgcn_ballot_w64(l_stash);
    uint64_t cmask = __builtin_amdgcn_ballot_w64(l_close);
    if (cmask) load_ape();
    while (cmask) {
      const int i = __builtin_ctzll(cmask);
      cmask &= cmask - 1;
      const int pos_i = __builtin_amdgcn_readlane(l_pos, i);
      const int loc_i = __builtin_amdgcn_readlane(l_loc, i);
      const int blk_i = __builtin_amdgcn_readlane(l_blk, i);
      const int slot = pos_i % POOL;
      const int pbase = pos_i - slot;  // position of pool slot 0
      const uint64_t earlier = smask & ((1ull << i) - 1ull);  // tokens before i that stash
      const bool blk_same = l_blk == blk_i;
      uint32_t gw[POOL][1], kw[POOL][1];
#pragma unroll
      for (int s = 0; s < POOL; ++s) {
        const int phys_s = ring_mod(pbase + s, ta);
        const uint64_t cand = __builtin_amdgcn_ballot_w64(blk_same && l_phys == phys_s) & earlier;
        const int latest = cand ? 63 - __builtin_clzll(cand) : -1;
        const int src = s == slot ? i : latest;  // the token itself for its own slot
        gptr<const uint16_t> krow, grow;
        if (src >= 0) {
          krow = kbase + (uint32_t)(src * d.key_st);
          grow = sbase + (uint32_t)(src * d.ss_st);
        } else {
          krow = tail_row(blk_i, phys_s);
          grow = krow + (uint32_t)ta.head_elems;
        }
        gw[s][0] = load_dw<VEC>(grow, sub);
        kw[s][0] = load_dw<VEC>(krow, sub);
      }
      write_pool(gw, kw, loc_i, NoMid{});
    }
    // stash in token order (every ring read above was consumed already)
#pragma unroll
    for (int t = 0; t < NMAX; ++t) {
      if ((smask >> t) & 1ull) {
        const gptr<uint16_t> dst =
            tail_row(__builtin_amdgcn_readlane(l_blk, t), __builtin_amdgcn_readlane(l_phys, t));
        store_dw(dst, sub, rk[t]);
        store_dw(dst + (uint32_t)ta.head_elems, sub, rs[t]);
      }
    }
    return;
  }

  // ===== one token (NMAX == 1) =====
  read_meta();
  // --- derived per token (wave-uniform) ---
  ivec phys, blk;
  uint32_t smask = 0, cmask = 0;
#pragma unroll
  for (int i = 0; i < NMAX; ++i) {
    const uint32_t tok_ok = i == 0 ? 1u : s_bit31((uint32_t)(i - n));  // i < n: token i exists
    const int safe = max(pos[i], 0);
    phys[i] = ring_mod(safe, ta);
    blk[i] = ring_div(max(ts[i], 0), ta);
    // sign tricks instead of compare + select chains: bit 31 of ~(a | b) is
    // set iff both are >= 0; bit 31 of (POOL-1 - safe % POOL) - 1 iff the
    // token is the last slot of its pool
    const uint32_t stash_ok = s_bit31(~(uint32_t)(pos[i] | ts[i]));
    const uint32_t close_ok = s_bit31(~(uint32_t)(pos[i] | loc[i])) & s_bit31((uint32_t)(POOL - 1 - safe % POOL) - 1u);
    smask |= (tok_ok & stash_ok) << i;
    cmask |= (tok_ok & close_ok) << i;
  }
  smask &= vmask;
  cmask &= vmask;
  if constexpr (NMAX == 1) {
    // One token: it closes a pool only as slot POOL-1 (pos % POOL == POOL-1),
    // so the other POOL-1 slots are the ring rows phys0 .. phys0 + POOL-2
    // (ring is a multiple of POOL): no slot selects, no load of the token's
    // own slot. Gate rows and APE first (the softmax needs only them), K rows
    // after. The stash (its ring slot is never read in this step) is issued
    // once the last K row has been consumed: earlier, the compiler's vmcnt
    // waits for the K rows would also cover the store acks; later, after the
    // pooled-key stores, its own drain would end the wave. Here the acks
    // drain under the Hadamard and the quantization.
    auto stash = [&](uint32_t k0, uint32_t s0) {
      const gptr<uint16_t> dst = tail_row(blk[0], phys[0]);
      store_dw(dst, sub, k0);
      store_dw(dst + (uint32_t)ta.head_elems, sub, s0);
    };
    // Layout: the non-closing path (3 steps in 4) is the fall-through right
    // after the branch and the closing path is the out-of-line block. Without
    // the hint the compiler puts the stash block behind the closing path, and
    // every non-closing wave pays an instruction-cache miss (~550 clocks on
    // the ATT ledger) to reach it.
    if (__builtin_expect(cmask != 0, 0)) {
      const gptr<const uint16_t> krow = tail_row(blk[0], phys[0] - (POOL - 1));
      const gptr<const uint16_t> grow = krow + (uint32_t)ta.head_elems;
      uint32_t gw[POOL][1], kw[POOL][1];
#pragma unroll
      for (int s = 0; s < POOL - 1; ++s) gw[s][0] = load_dw<VEC>(grow + s * HEAD_DIM, sub);
      load_ape();
      __builtin_amdgcn_sched_barrier(0);  // keep the gate / APE loads ahead of the K loads
#pragma unroll
      for (int s = 0; s < POOL - 1; ++s) kw[s][0] = load_dw<VEC>(krow + s * HEAD_DIM, sub);
      gw[POOL - 1][0] = rs[0];
      kw[POOL - 1][0] = rk[0];
      const bool do_stash = smask != 0;
      write_pool(gw, kw, loc[0], [&](float dep) {
        // the stored words "depend" on the pooled accumulator, so the stores
        // cannot be hoisted above the use of the last K row
        uint32_t k0 = rk[0], s0 = rs[0];
        asm volatile("" : "+v"(k0), "+v"(s0) : "v"(dep));
        if (do_stash) stash(k0, s0);
      });
    } else if (smask) {
      stash(rk[0], rs[0]);
    }
  }
}


template <int POOL, int NMAX, bool VEC, typename IdxT>
// waves_per_eu(8, 8): keep occupancy 8 (<= 64 VGPRs) for every NMAX
__global__ void __launch_bounds__(64 * DEC_BLOCK_WAVES) __attribute__((amdgpu_waves_per_eu(8, 8))) kpool_decode_kernel(
    const uint16_t* key, const uint16_t* score, const void* tail_slot, const void* slot_map,
    const void* positions, int key_sb, int ss_sb, int next_n, int num_requests, DecodeKRest r) {
  DecodeKArgs d;
  d.key = key; d.score = score; d.tail_slot = tail_slot; d.slot_map = slot_map; d.positions = positions;
  d.key_sb = key_sb; d.ss_sb = ss_sb; d.next_n = next_n; d.num_requests = num_requests;
  d.ape = r.ape; d.tail = r.tail; d.cache = r.cache; d.key_st = r.key_st; d.ss_st = r.ss_st;
  d.block_elems = r.block_elems; d.head_elems = r.head_elems; d.ring_mask = r.ring_mask; d.ring_shift = r.ring_shift;
  d.page_size = r.page_size; d.page_shift = r.page_shift; d.page_bytes = r.page_bytes; d.flags = r.flags;
  d.pad0 = d.pad1 = 0;
  const int lane = threadIdx.x & 63;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  // No early return: the padding waves of the last workgroup clamp to the
  // last request (its loads are in bounds) and store nothing, so every
  // kernarg is fetched in one SMEM batch instead of num_requests first.
  const int req_raw = blockIdx.x * DEC_BLOCK_WAVES + wave;
  const bool valid = req_raw < d.num_requests;
  const int req = valid ? req_raw : d.num_requests - 1;
  const uint32_t vmask = 0u - s_bit31((uint32_t)(req_raw - d.num_requests));  // ~0 iff valid
  decode_request<POOL, NMAX, VEC, IdxT>(req, d, lane, vmask);
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
    const SlotT* __restrict__ seed_tslot,
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
    const DecodeKArgs d = make_kargs(da, ape, ta, cache, page_size, page_shift, (uint32_t)page_bytes, flags & 6);
    decode_request<POOL, NMAX, VEC, SlotT>(blockIdx.x, d, lane);
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
#ifdef KPOOL_ISA_QUICK
  KPOOL_L(64)
#else
  KPOOL_L(64) KPOOL_L(32) KPOOL_L(16)
#endif
#undef KPOOL_L
  return -2;
}

template <int POOL, int NMAX>
int launch_decode_n(bool vec, bool i64, const DecodeArgs& da, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                    int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
  const int grid = (da.num_requests + DEC_BLOCK_WAVES - 1) / DEC_BLOCK_WAVES;
  if (grid == 0) return 0;
  (void)ape_stride;  // == HEAD_DIM, checked by the entry point
  const DecodeKArgs d = make_kargs(da, (const float*)ape, ta, (uint8_t*)cache, page_size, page_shift_of(page_size),
                                   (uint32_t)page_bytes, flags);
  DecodeKRest r;
  r.ape = d.ape; r.tail = d.tail; r.cache = d.cache; r.key_st = d.key_st; r.ss_st = d.ss_st;
  r.block_elems = d.block_elems; r.head_elems = d.head_elems; r.ring_mask = d.ring_mask; r.ring_shift = d.ring_shift;
  r.page_size = d.page_size; r.page_shift = d.page_shift; r.page_bytes = d.page_bytes; r.flags = d.flags;
#define KPOOL_D(V, I)                                                                                         \
  hipLaunchKernelGGL((kpool_decode_kernel<POOL, NMAX, V, I>), dim3(grid), dim3(64 * DEC_BLOCK_WAVES), 0, stream, \
                     d.key, d.score, d.tail_slot, d.slot_map, d.positions, d.key_sb, d.ss_sb, d.next_n,          \
                     d.num_requests, r);
  if (vec && i64) { KPOOL_D(true, int64_t) }
  else if (vec) { KPOOL_D(true, int32_t) }
  else if (i64) { KPOOL_D(false, int64_t) }
  else { KPOOL_D(false, int32_t) }
#undef KPOOL_D
  return (int)hipGetLastError();
}

template <int POOL>
int launch_decode(bool vec, bool i64, const DecodeArgs& da, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                  int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
  const int n = da.next_n;
#define KPOOL_N(N)                                                                                            \
  if (n <= N) return launch_decode_n<POOL, N>(vec, i64, da, ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
#ifdef KPOOL_ISA_QUICK
  KPOOL_N(1) KPOOL_N(2) KPOOL_N(4) KPOOL_N(8)
#else
  KPOOL_N(1) KPOOL_N(2) KPOOL_N(3) KPOOL_N(4) KPOOL_N(5) KPOOL_N(6) KPOOL_N(7) KPOOL_N(8) KPOOL_N(16)
#endif
#undef KPOOL_N
  return -4;
}

template <int POOL, int NMAX, int LANES, bool VEC, typename SlotT>
int launch_mixed_t(const DecodeArgs& da, const void* k, const void* gate, const void* slot, int64_t n_tokens,
                   int64_t k_stride, int64_t gate_stride, const void* seed_tslot, const void* ape,
                   int64_t ape_stride, TailArgs ta, void* cache, int page_size, int64_t page_bytes, int flags,
                   hipStream_t stream) {
  constexpr int GROUPS_PER_WAVE = 64 / LANES;
  const int64_t n_windows = (n_tokens + POOL - 1) / POOL;
  const int64_t n_prefill_blocks = (n_windows + GROUPS_PER_WAVE - 1) / GROUPS_PER_WAVE;
  const int64_t grid = da.num_requests + n_prefill_blocks;
  if (grid == 0) return 0;
  hipLaunchKernelGGL((kpool_mixed_kernel<POOL, NMAX, LANES, VEC, SlotT>), dim3((unsigned)grid), dim3(64), 0, stream,
                     da, (const uint16_t*)k, (const uint16_t*)gate, slot, n_tokens, n_windows, k_stride,
                     gate_stride, (const SlotT*)seed_tslot, (const float*)ape, ape_stride, ta, (uint8_t*)cache, page_size,
                     page_shift_of(page_size), page_bytes, flags);
  return (int)hipGetLastError();
}

template <int POOL, int NMAX, int LANES>
int launch_mixed_l(bool vec, bool i64, const DecodeArgs& da, const void* k, const void* gate, const void* slot,
                   int64_t n_tokens, int64_t k_stride, int64_t gate_stride, const void* seed_tslot,
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
                   const void* seed_tslot, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                   int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
#define KPOOL_L(L)                                                                                              \
  if (lanes == L)                                                                                               \
    return launch_mixed_l<POOL, NMAX, L>(vec, i64, da, k, gate, slot, n_tokens, k_stride, gate_stride, seed_tslot, \
                                         ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
#ifdef KPOOL_ISA_QUICK
  KPOOL_L(64)
#else
  KPOOL_L(64) KPOOL_L(32) KPOOL_L(16)
#endif
#undef KPOOL_L
  return -2;
}

template <int POOL>
int launch_mixed(int lanes, bool vec, bool i64, const DecodeArgs& da, const void* k, const void* gate,
                 const void* slot, int64_t n_tokens, int64_t k_stride, int64_t gate_stride,
                 const void* seed_tslot, const void* ape, int64_t ape_stride, TailArgs ta, void* cache,
                 int page_size, int64_t page_bytes, int flags, hipStream_t stream) {
  const int n = da.next_n;
#define KPOOL_N(N)                                                                                            \
  if (n <= N)                                                                                                 \
    return launch_mixed_n<POOL, N>(lanes, vec, i64, da, k, gate, slot, n_tokens, k_stride, gate_stride,       \
                                   seed_tslot, ape, ape_stride, ta, cache, page_size, page_bytes, flags, stream);
#ifdef KPOOL_ISA_QUICK
  KPOOL_N(1)
#else
  KPOOL_N(1) KPOOL_N(2) KPOOL_N(3) KPOOL_N(4) KPOOL_N(5) KPOOL_N(6) KPOOL_N(7) KPOOL_N(8) KPOOL_N(16)
#endif
#undef KPOOL_N
  return -4;
}

}  // namespace

// Launcher return codes: 0 ok, -2 bad lane count, -3 unsupported pool size, -4 next_n > 16, else hipError_t.
#ifdef KPOOL_ISA_QUICK
// -DKPOOL_ISA_QUICK: instantiate only the production shapes (pool 4, decode
// next_n <= 1 / 4, 64 lanes) for fast ISA listings; never used for a real build.
#define KPOOL_POOL_SWITCH(EXPR)                      \
  switch (pool_size) {                               \
    case 4: { constexpr int P = 4; EXPR }            \
    default: rc = -3; break;                         \
  }
#else
#define KPOOL_POOL_SWITCH(EXPR)                      \
  switch (pool_size) {                               \
    case 2: { constexpr int P = 2; EXPR }            \
    case 3: { constexpr int P = 3; EXPR }            \
    case 4: { constexpr int P = 4; EXPR }            \
    case 8: { constexpr int P = 8; EXPR }            \
    case 16: { constexpr int P = 16; EXPR }          \
    default: rc = -3; break;                         \
  }
#endif

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
    AITER_CHECK(page_size <= (1 << 20), "page_size too large, got ", page_size);
}

void check_ape(const aiter_tensor_t& ape, int pool_size)
{
    AITER_CHECK(ape.dtype() == AITER_DTYPE_fp32 && ape.dim() == 2 && ape.size(0) == pool_size &&
                    ape.size(1) == KPOOL_HEAD_DIM && ape.stride(1) == 1 && ape.stride(0) == KPOOL_HEAD_DIM,
                "ape must be a contiguous fp32 [pool_size, 128] tensor");
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

void check_map(const aiter_tensor_t& t, int64_t B, int64_t next_n, bool i64, const char* name)
{
    AITER_CHECK(t.dtype() == (i64 ? AITER_DTYPE_i64 : AITER_DTYPE_i32) && t.dim() == 2 && t.size(0) == B &&
                    t.size(1) == next_n && t.is_contiguous(),
                name, " must be a contiguous ", i64 ? "int64" : "int32",
                " [num_requests, next_n] tensor (same dtype as tail_slot_mapping)");
}

struct DecodeChecked
{
    DecodeArgs da;
    bool vec;
    bool i64;  // metadata dtype (tail_slot_mapping, slot_mapping, positions all int32 or all int64)
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
    AITER_CHECK(tail_slot_mapping.dtype() == AITER_DTYPE_i32 || tail_slot_mapping.dtype() == AITER_DTYPE_i64,
                "tail_slot_mapping must be int32 or int64, got ", AiterDtype_to_str(tail_slot_mapping.dtype()));
    const bool i64 = tail_slot_mapping.dtype() == AITER_DTYPE_i64;
    check_map(tail_slot_mapping, B, next_n, i64, "tail_slot_mapping");
    check_map(slot_mapping, B, next_n, i64, "slot_mapping");
    check_map(positions, B, next_n, i64, "positions");
    AITER_CHECK(tail_kv_cache.dtype() == AITER_DTYPE_bf16 && tail_kv_cache.dim() == 4 &&
                    tail_kv_cache.size(1) == 2 && tail_kv_cache.size(3) == KPOOL_HEAD_DIM &&
                    tail_kv_cache.stride(3) == 1 && tail_kv_cache.stride(2) == KPOOL_HEAD_DIM,
                "tail_kv_cache must be bf16 [num_blocks, 2, ring, 128] with contiguous rows");
    const int ring = static_cast<int>(tail_kv_cache.size(2));
    AITER_CHECK(ring >= pool_size && ring % pool_size == 0, "tail ring (", ring,
                ") must be a multiple of pool_size (", pool_size, ")");
    AITER_CHECK(B * key.stride(0) < (1LL << 31) && B * slot_score.stride(0) < (1LL << 31) &&
                    B * next_n < (1LL << 31),
                "decode key / slot_score must have < 2^31 elements");
    DecodeChecked r;
    r.da = DecodeArgs{static_cast<const uint16_t*>(key.data_ptr()),
                      static_cast<const uint16_t*>(slot_score.data_ptr()),
                      tail_slot_mapping.data_ptr(),
                      slot_mapping.data_ptr(),
                      positions.data_ptr(),
                      static_cast<int>(key.stride(0)),
                      static_cast<int>(key.stride(1)),
                      static_cast<int>(slot_score.stride(0)),
                      static_cast<int>(slot_score.stride(1)),
                      static_cast<int>(next_n),
                      static_cast<int>(B)};
    r.vec = aligned(key.data_ptr(), key.stride(1), 2, 4) && aligned(slot_score.data_ptr(), slot_score.stride(1), 2, 4) &&
            aligned(tail_kv_cache.data_ptr(), tail_kv_cache.stride(0), 2, 4) &&
            aligned(tail_kv_cache.data_ptr(), tail_kv_cache.stride(1), 2, 4);
    r.i64 = i64;
    return r;
}

TailArgs tail_args(const aiter_tensor_t& tail_kv_cache)
{
    const int ring = static_cast<int>(tail_kv_cache.size(2));
    AITER_CHECK(ring > 0 && (ring & (ring - 1)) == 0, "tail ring (", ring, ") must be a power of two");
    // vLLM pads every tail page to the KV-cache page (stride(0) >> 2 * ring * 128) and the
    // whole tail can exceed 2^31 elements: block offsets are 64-bit, the strides stay 32-bit.
    AITER_CHECK(tail_kv_cache.stride(0) < (1LL << 31) && tail_kv_cache.stride(1) < (1LL << 31),
                "tail_kv_cache strides must fit in 32 bits, got ", tail_kv_cache.stride(0), ", ",
                tail_kv_cache.stride(1));
    AITER_CHECK(tail_kv_cache.stride(2) == 128 && tail_kv_cache.stride(3) == 1,
                "tail_kv_cache rows must be contiguous [ring, 128]");
    return TailArgs{static_cast<uint16_t*>(tail_kv_cache.data_ptr()),
                    static_cast<int>(tail_kv_cache.stride(0)),
                    static_cast<int>(tail_kv_cache.stride(1)),
                    ring - 1,
                    __builtin_ctz(static_cast<unsigned>(ring))};
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
    KPOOL_POOL_SWITCH(rc = launch_decode<P>(d.vec, d.i64, d.da, ape.data_ptr(), ape.stride(0), ta, kv_cache.data_ptr(),
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

    TailArgs ta{nullptr, 0, 0, 0, 0};  // unused when neither decode nor seeding runs
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

    DecodeArgs da{nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0};
    bool dec_vec = true;
    bool dec_i64 = false;
    if(has_decode)
    {
        DecodeChecked d = check_decode(*tail_kv_cache, *decode_tail_slot_mapping, *decode_key, *decode_gate,
                                       *decode_slot_mapping, *decode_positions, pool_size);
        da      = d.da;
        dec_vec = d.vec;
        dec_i64 = d.i64;
    }

    AITER_CHECK(prefill_gate && prefill_slot_mapping, "prefill_gate and prefill_slot_mapping are required with prefill_k");
    check_rows(*prefill_k, "prefill_k");
    check_rows(*prefill_gate, "prefill_gate");
    AITER_CHECK(prefill_gate->size(0) == n_tokens, "prefill_k and prefill_gate must have the same number of rows");
    const bool i64 = slot_is_i64(*prefill_slot_mapping);
    AITER_CHECK(prefill_slot_mapping->size(0) == n_tokens, "prefill_slot_mapping must have one entry per token");
    // one index type per launch: the decode metadata and the seed slots must
    // match the prefill slot_mapping (int32 or int64)
    AITER_CHECK(!has_decode || dec_i64 == i64,
                "decode tail_slot_mapping / slot_mapping / positions must have the dtype of prefill_slot_mapping");
    const void* seed_tslot = nullptr;
    if(seed)
    {
        AITER_CHECK(slot_is_i64(*prefill_tail_slot_mapping) == i64 && prefill_tail_slot_mapping->size(0) == n_tokens,
                    "prefill_tail_slot_mapping must be a contiguous [n_tokens] tensor with the dtype of prefill_slot_mapping");
        seed_tslot = prefill_tail_slot_mapping->data_ptr();
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

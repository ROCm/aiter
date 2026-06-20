// rocWMMA needs float<->__half conversions that torch disables via
// -D__HIP_NO_HALF_CONVERSIONS__. Undef before any HIP/half header is pulled in,
// include rocWMMA, then restore the macros for the torch/ATen headers.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#ifndef __HIP_NO_HALF_OPERATORS__
#define __HIP_NO_HALF_OPERATORS__ 1
#endif
#ifndef __HIP_NO_HALF_CONVERSIONS__
#define __HIP_NO_HALF_CONVERSIONS__ 1
#endif
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPException.h>
#include <torch/extension.h>

#include "hip_float8.h"

// FWHT butterfly: default to the single-FMA sign form. It folds the per-element
// sub+add+cndmask of the select form into one fmaf, removing the v_cndmask that
// dominates the VALU-bound FWHT (measured: ~373 -> ~50 v_cndmask in the hpw8
// prefill kernel, up to 1.56x faster at 32k prefill, no decode regression) and
// is bit-equivalent (single-rounded). Define NO_FWHT_FMA_SIGN to fall back to
// the select form.
#if !defined(FWHT_FMA_SIGN) && !defined(NO_FWHT_FMA_SIGN)
#define FWHT_FMA_SIGN 1
#endif

namespace aiter {

constexpr int kHeadDim = 128;
constexpr int kWarpSize = 32;
constexpr int kSubWarpSize = 16;
constexpr int kWarpsPerBlock = 4;
constexpr int kHeadsPerWarp = 2;

#define CHECK_HIP_TENSOR(x) TORCH_CHECK((x).is_cuda(), #x " must be a HIP/CUDA tensor")
#define CHECK_DTYPE(x, st) TORCH_CHECK((x).scalar_type() == (st), #x " has dtype ", (x).scalar_type(), ", expected ", st)

enum ScalarKind : int {
  kFloat32 = 0,
  kBFloat16 = 1,
  kFloat16 = 2,
};

int scalar_kind(const torch::Tensor& t) {
  if (t.scalar_type() == at::kFloat) {
    return kFloat32;
  }
  if (t.scalar_type() == at::kBFloat16) {
    return kBFloat16;
  }
  if (t.scalar_type() == at::kHalf) {
    return kFloat16;
  }
  TORCH_CHECK(false, "unsupported scalar dtype: ", t.scalar_type());
}

__device__ __forceinline__ float load_scalar_f32(const void* ptr, int kind, int64_t offset) {
  if (kind == kFloat32) {
    return reinterpret_cast<const float*>(ptr)[offset];
  }
  if (kind == kBFloat16) {
    return static_cast<float>(reinterpret_cast<const __hip_bfloat16*>(ptr)[offset]);
  }
  return __half2float(reinterpret_cast<const __half*>(ptr)[offset]);
}

__device__ __forceinline__ float bf16_round_f32(float x) {
  return static_cast<float>(__hip_bfloat16(x));
}

// ==== Default prefill formulation (verified -12% on MI300 32k prefill via
// rocprofv3 hardware timestamps + eager event timing; dynamic SQ_INSTS_VALU
// -20%). Two changes that only pay off TOGETHER:
//   1) SWZ_USE_SWIZZLE: route every FWHT/reduction cross-lane through ds_swizzle
//      (the LDS port) instead of DPP. The kernel is VALU-issue bound, so keeping
//      lane-exchange OFF the saturated VALU port is strictly better as long as
//      LDS latency stays hidden by the HPW*4-way ILP (it does: SQ_WAIT_INST_LDS
//      is a small fraction of the wait).
//   2) FWHT_FMA_SIGN: express each butterfly as one sign-FMA, removing the
//      per-element v_cndmask. This only helps once (1) is on -- with the DPP
//      path it instead breaks the v_*_dpp fusion and regresses.
// Define SWZ_USE_SHFL / SWZ_FORCE_DPP / FWHT_NO_FMA_SIGN to A/B the old path.
#if !defined(SWZ_USE_SHFL) && !defined(SWZ_FORCE_DPP) && !defined(SWZ_USE_SWIZZLE)
#define SWZ_USE_SWIZZLE
#endif
#if !defined(FWHT_NO_FMA_SIGN) && !defined(FWHT_FMA_SIGN)
#define FWHT_FMA_SIGN
#endif

// Cross-lane xor-permute via ds_swizzle instead of __shfl_xor. On gfx942,
// __shfl_xor lowers to ds_bpermute plus VALU address math (__lane_id, a clamp
// compare, and a <<2 shift) on every call. ds_swizzle encodes the xor pattern
// in an immediate, so it is a single LDS op with no address setup -- fewer VALU
// instructions and lower-latency lane exchange, which is the bottleneck in the
// FWHT/reduction dependency chains. The immediate (MASK<<10)|0x1F implements
// laneid^MASK within each 32-lane group (verified vs __shfl_xor for MASK in
// {1,2,4,8,16}). MASK<16 never crosses the 16-lane boundary, so the same op
// serves both the width-32 (warp) and width-16 (subwarp) reductions.
template <int MASK>
__device__ __forceinline__ float swz_xor(float v) {
#if defined(SWZ_USE_SHFL)
  // A/B baseline: original __shfl_xor (ds_bpermute + lane-address VALU).
  return __shfl_xor(v, MASK, kWarpSize);
#elif defined(SWZ_USE_SWIZZLE)
  // A/B baseline: ds_swizzle for every mask. NOTE: on the gfx942 ROCm 7.2
  // toolchain the ds_swizzle builtin actually lowers to ds_bpermute, so this is
  // pure-LDS cross-lane for all 5 stages.
  return __int_as_float(
      __builtin_amdgcn_ds_swizzle(__float_as_int(v), (MASK << 10) | 0x1F));
#else
  // DEFAULT: DPP quad_perm (pure-VALU, no LDS round-trip) for the within-quad
  // masks 1/2 -- the compiler fuses the permute straight into the consuming
  // v_add_f32 / v_max_f32 (v_*_f32_dpp), so each of those two butterfly stages
  // costs a single VALU op instead of an LDS bpermute + dependent ALU. The
  // wider masks 4/8/16 cross the 4-lane quad, which DPP can't xor on gfx942
  // (no dpp8 / no row_xor), so they stay on ds_swizzle (-> ds_bpermute). The
  // quad_perm immediates encode laneid^MASK within each 4-lane group: 0xB1 =
  // [1,0,3,2] (xor 1), 0x4E = [2,3,0,1] (xor 2). Quads never straddle the
  // 32-lane reduction boundary (lanes 0-31 -> quads 0-7), so the per-32
  // segmented reduction is preserved on wave64.
  if (MASK == 1)
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, false));
  if (MASK == 2)
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, false));
#if defined(SWZ_DPP_ROW8)
  // mask 8 stays inside each 16-lane DPP row (lane^8 == (lane+-8)%16 for the
  // low/high half), so a row-rotate-by-8 realizes the xor partner without an
  // LDS round-trip. ctrl 0x128 = row_ror:8. bound_ctrl irrelevant (rotate
  // wraps, no OOB lanes). Experimental: must verify it fuses (v_*_dpp) and is
  // bit-correct vs ds_bpermute on this toolchain.
  if (MASK == 8)
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x128, 0xF, 0xF, false));
#endif
  return __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (MASK << 10) | 0x1F));
#endif
}

__device__ __forceinline__ float warp_sum(float v) {
  v += swz_xor<16>(v);
  v += swz_xor<8>(v);
  v += swz_xor<4>(v);
  v += swz_xor<2>(v);
  v += swz_xor<1>(v);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  v = fmaxf(v, swz_xor<16>(v));
  v = fmaxf(v, swz_xor<8>(v));
  v = fmaxf(v, swz_xor<4>(v));
  v = fmaxf(v, swz_xor<2>(v));
  v = fmaxf(v, swz_xor<1>(v));
  return v;
}

__device__ __forceinline__ float subwarp_sum(float v) {
  v += swz_xor<8>(v);
  v += swz_xor<4>(v);
  v += swz_xor<2>(v);
  v += swz_xor<1>(v);
  return v;
}

__device__ __forceinline__ float subwarp_max(float v) {
  v = fmaxf(v, swz_xor<8>(v));
  v = fmaxf(v, swz_xor<4>(v));
  v = fmaxf(v, swz_xor<2>(v));
  v = fmaxf(v, swz_xor<1>(v));
  return v;
}

// One radix-2 FWHT butterfly stage across lanes for an array of N regs.
// FMA-sign form: result = fma(own, sgn, partner) where sgn=+1 (low lane,
// own+partner) or -1 (high lane, partner-own). This collapses the per-element
// sub+add+cndmask (the v_cndmask_b32 that dominates the VALU-bound FWHT) into a
// single FMA, computing the per-stage sign just once. Verified bit-equivalent
// to the select form (single-rounded, if anything slightly more accurate).
template <int MASK, int N>
__device__ __forceinline__ void fwht_bfly(float* x, int lane) {
#if defined(FWHT_FMA_SIGN)
  const float sgn = (lane & MASK) ? -1.0f : 1.0f;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float y = swz_xor<MASK>(x[i]);
    x[i] = fmaf(x[i], sgn, y);
  }
#else
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float y = swz_xor<MASK>(x[i]);
    x[i] = (lane & MASK) ? (y - x[i]) : (x[i] + y);
  }
#endif
}

__device__ __forceinline__ int64_t find_req_for_row(const int32_t* q_index,
                                                    int64_t num_req,
                                                    int64_t row) {
  int64_t lo = 0;
  int64_t hi = num_req;
  while (lo + 1 < hi) {
    int64_t mid = (lo + hi) >> 1;
    int32_t start = q_index[mid];
    if (start <= row) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ uint8_t float_to_fp8_byte(float x, float fp8_max) {
  x = fminf(fmaxf(x, -fp8_max), fp8_max);
  return hip_fp8(x).data;
}

// Pack TWO fp32 -> two e4m3fnuz bytes in ONE v_cvt_pk_fp8_f32 (byte0=a, byte1=b),
// matching to_fp8_from_fp32's internal +-240 fmed3 saturation. Halves the
// conversion VALU vs two scalar hip_fp8() calls (each wastes the packed cvt on a
// single element). Mirrors the NVIDIA cvt.rn.satfinite.e4m3x2.f32 path.
__device__ __forceinline__ uint32_t float2_to_fp8x2(float a, float b) {
  // Explicit fmed3 saturation: unlike NVIDIA's free F2FP.SATFINITE, the AMD
  // v_cvt_pk_fp8_f32 has NO clamp/saturate modifier (CDNA3 ISA: only op_sel), so
  // out-of-range inputs are not clamped by the cvt. Kept unconditionally for
  // safety -- dropping it on the amax-normalized path saves ~1.5% but risks a
  // NaN in the KV cache at the rcp-rounding boundary, not worth it.
  a = __builtin_amdgcn_fmed3f(a, 240.0f, -240.0f);
  b = __builtin_amdgcn_fmed3f(b, 240.0f, -240.0f);
  return __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);  // WORD0 -> bytes 0,1
}

// Fast hardware reciprocal (v_rcp_f32, ~2.5 ulp). The FP8 quant scales feed a
// 3-mantissa-bit format and a stored fp32 scale, so full IEEE-754 division
// (v_div_scale/v_div_fmas/v_div_fixup + cndmask, ~8 VALU each) is wasted
// precision. These scales are uniform per (warp,head); with HPW=8 the compiler
// otherwise replicates ~56 of those full divisions across 64 lanes -- a large
// slice of the VALU-issue-bound prefill budget. rcp+mul collapses each to ~2
// ops. Verified within the abkit fp8 tolerance vs the Triton reference.
__device__ __forceinline__ float fast_rcp(float x) {
  return __builtin_amdgcn_rcpf(x);
}

// Divide/modulo for the KV-cache addressing. block_size / k_cache_x / k_scale_l
// are ALWAYS powers of two in vLLM, so x/rt = x>>log2(rt) and x%rt = x&(rt-1) --
// a runtime shift/and (the divisor is a uniform kernel arg, so the compiler
// hoists the ctz out of the per-element loops). This kills the magic-number
// runtime division (the dominant VALU cost per rocprofv3) WITHOUT any
// per-block_size template specialization -- one kernel instance serves every
// power-of-two block_size. The C template arg is kept for the rare compile-time
// constant call but is no longer needed for speed.
template <int C>
__device__ __forceinline__ int idiv(int x, int rt) {
  if constexpr (C != 0) return x / C; else return x >> __builtin_ctz(static_cast<unsigned>(rt));
}
template <int C>
__device__ __forceinline__ int imod(int x, int rt) {
  if constexpr (C != 0) return x % C; else return x & (rt - 1);
}

__device__ __forceinline__ void fwht128_inplace(float* x, int lane) {
  // Walsh-Hadamard from the same recursive construction as _get_hadamard().
#pragma unroll
  for (int stride = 1; stride < kHeadDim; stride <<= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      if ((d & stride) == 0) {
        const float a = x[d];
        const float b = x[d + stride];
        x[d] = a + b;
        x[d + stride] = a - b;
      }
    }
    __syncwarp();
  }
}

__device__ __forceinline__ void fwht128_regs(float x[4], int lane) {
  fwht_bfly<1, 4>(x, lane);
  fwht_bfly<2, 4>(x, lane);
  fwht_bfly<4, 4>(x, lane);
  fwht_bfly<8, 4>(x, lane);
  fwht_bfly<16, 4>(x, lane);

  float a = x[0];
  float b = x[1];
  x[0] = a + b;
  x[1] = a - b;
  a = x[2];
  b = x[3];
  x[2] = a + b;
  x[3] = a - b;

  a = x[0];
  b = x[2];
  x[0] = a + b;
  x[2] = a - b;
  a = x[1];
  b = x[3];
  x[1] = a + b;
  x[3] = a - b;
}

// ---- Multi-head variants: process NH independent (row,head) lanes' worth of
// work in lockstep so the cross-lane (ds_swizzle) latency of one head is hidden
// by the independent work of the others. Each stage issues NH*4 (FWHT) or NH
// (reduction) independent swizzles before any dependent consumer, giving the
// scheduler NH-way extra ILP -- the lever that fills the ~40% VALU idle the
// 1-head kernel leaves on the table at large batch.
template <int MASK, int NH>
__device__ __forceinline__ void fwht_bfly_multi(float x[NH][4], int lane) {
#if defined(FWHT_FMA_SIGN)
  const float sgn = (lane & MASK) ? -1.0f : 1.0f;
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float y = swz_xor<MASK>(x[h][i]);
      x[h][i] = fmaf(x[h][i], sgn, y);
    }
#else
#pragma unroll
  for (int h = 0; h < NH; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float y = swz_xor<MASK>(x[h][i]);
      x[h][i] = (lane & MASK) ? (y - x[h][i]) : (x[h][i] + y);
    }
#endif
}

template <int NH>
__device__ __forceinline__ void fwht128_regs_multi(float x[NH][4], int lane) {
  fwht_bfly_multi<1, NH>(x, lane);
  fwht_bfly_multi<2, NH>(x, lane);
  fwht_bfly_multi<4, NH>(x, lane);
  fwht_bfly_multi<8, NH>(x, lane);
  fwht_bfly_multi<16, NH>(x, lane);
#pragma unroll
  for (int h = 0; h < NH; ++h) {
    float a = x[h][0], b = x[h][1];
    x[h][0] = a + b; x[h][1] = a - b;
    a = x[h][2]; b = x[h][3];
    x[h][2] = a + b; x[h][3] = a - b;
    a = x[h][0]; b = x[h][2];
    x[h][0] = a + b; x[h][2] = a - b;
    a = x[h][1]; b = x[h][3];
    x[h][1] = a + b; x[h][3] = a - b;
  }
}

template <int NH>
__device__ __forceinline__ void warp_sum_multi(float (&v)[NH]) {
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] += swz_xor<16>(v[h]);
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] += swz_xor<8>(v[h]);
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] += swz_xor<4>(v[h]);
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] += swz_xor<2>(v[h]);
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] += swz_xor<1>(v[h]);
}

template <int NH>
__device__ __forceinline__ void warp_max_multi(float (&v)[NH]) {
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] = fmaxf(v[h], swz_xor<16>(v[h]));
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] = fmaxf(v[h], swz_xor<8>(v[h]));
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] = fmaxf(v[h], swz_xor<4>(v[h]));
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] = fmaxf(v[h], swz_xor<2>(v[h]));
#pragma unroll
  for (int h = 0; h < NH; ++h) v[h] = fmaxf(v[h], swz_xor<1>(v[h]));
}

__device__ __forceinline__ void fwht128_regs_halfwarp(float x[8], int lane) {
  fwht_bfly<1, 8>(x, lane);
  fwht_bfly<2, 8>(x, lane);
  fwht_bfly<4, 8>(x, lane);
  fwht_bfly<8, 8>(x, lane);

#pragma unroll
  for (int base = 0; base < 8; base += 2) {
    const float a = x[base];
    const float b = x[base + 1];
    x[base] = a + b;
    x[base + 1] = a - b;
  }

#pragma unroll
  for (int base = 0; base < 8; base += 4) {
    float a = x[base];
    float b = x[base + 2];
    x[base] = a + b;
    x[base + 2] = a - b;
    a = x[base + 1];
    b = x[base + 3];
    x[base + 1] = a + b;
    x[base + 3] = a - b;
  }

#pragma unroll
  for (int base = 0; base < 4; ++base) {
    const float a = x[base];
    const float b = x[base + 4];
    x[base] = a + b;
    x[base + 4] = a - b;
  }
}

__device__ __forceinline__ float hadamard_scale_128() {
  // H is stored as bf16 in the Triton path; keep the same rounded scale.
  return static_cast<float>(__hip_bfloat16(0.08838834764831845f));
}

__global__ void compute_row_meta_kernel(
    const int32_t* __restrict__ q_index,
    const int32_t* __restrict__ num_seqlen_per_req,
    const int32_t* __restrict__ kvcache_indices,
    int64_t num_rows,
    int64_t num_req,
    int64_t total_num_kv_cache_tokens,
    int64_t stride_kvi_r,
    int64_t stride_kvi_b,
    int64_t block_size,
    int32_t* __restrict__ row_token_pos,
    int64_t* __restrict__ row_slot) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= num_rows) {
    return;
  }
  const int64_t qlen_guess = (num_req > 0) ? (num_rows / num_req) : 1;
  int64_t guess = (qlen_guess > 0) ? (row / qlen_guess) : 0;
  if (guess >= num_req) {
    guess = num_req - 1;
  }
  int64_t req;
  int32_t req_start = q_index[guess];
  int32_t req_end = q_index[guess + 1];
  if (row >= req_start && row < req_end) {
    req = guess;
  } else {
    req = find_req_for_row(q_index, num_req, row);
    req_start = q_index[req];
    req_end = q_index[req + 1];
  }
  const int32_t seq_len = num_seqlen_per_req[req];
  const int32_t token_pos =
      (seq_len > 0) ? static_cast<int32_t>(row + seq_len - req_end) : 0;
  int64_t slot = -1;
  if (seq_len > 0 && req_end > req_start) {
    const int bs_log = __builtin_ctz(static_cast<unsigned>(block_size));
    const int64_t block_idx = token_pos >> bs_log;
    const int32_t block_row = static_cast<int32_t>(token_pos & (block_size - 1));
    const int64_t phys_block = static_cast<int64_t>(
        kvcache_indices[req * stride_kvi_r + block_idx * stride_kvi_b]);
    const int64_t s = phys_block * block_size + block_row;
    if (s >= 0 && s < total_num_kv_cache_tokens) {
      slot = s;
    }
  }
  row_token_pos[row] = token_pos;
  row_slot[row] = slot;
}

template <bool CosSinBF16, bool WeightBF16, bool HadamardBF16, bool DecodeOneToken,
          int BSIZE = 0, int KCX = 0, int KSL = 0, bool CONTIG = false,
          bool USE_META = false>
__global__ void rope_norm_store_kv_fp8_fused_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const void* __restrict__ cos_sin,
    const int32_t* __restrict__ q_index,
    const int32_t* __restrict__ num_seqlen_per_req,
    const int32_t* __restrict__ kvcache_indices,
    const void* __restrict__ q_norm_weight,
    const void* __restrict__ k_norm_weight,
    const void* __restrict__ hadamard,
    float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    uint8_t* __restrict__ out_q,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    float* __restrict__ q_scale_out,
    int64_t num_rows,
    int64_t num_req,
    int64_t total_num_kv_cache_tokens,
    float eps,
    float fp8_max,
    int64_t stride_qkv_t,
    int64_t stride_qkv_d,
    int64_t stride_cos_t,
    int64_t stride_cos_d,
    int64_t stride_kvi_r,
    int64_t stride_kvi_b,
    int64_t stride_out_q_t,
    int64_t stride_out_q_h,
    int64_t stride_out_q_d,
    int64_t stride_kc_b,
    int64_t stride_kc_h,
    int64_t stride_kc_g,
    int64_t stride_kc_t,
    int64_t stride_kc_x,
    int64_t stride_vc_b,
    int64_t stride_vc_h,
    int64_t stride_vc_d,
    int64_t stride_vc_t,
    int64_t stride_ks_b,
    int64_t stride_ks_r,
    int64_t stride_ks_h,
    int64_t stride_ks_l,
    int64_t stride_qs_t,
    int64_t stride_qs_h,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t v_head_dim,
    int64_t block_size,
    int64_t k_scale_l,
    int64_t k_cache_x,
    const int32_t* __restrict__ row_token_pos,
    const int64_t* __restrict__ row_slot_meta) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t global_warp = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  const int64_t row = global_warp / num_q_heads;
  const int64_t hq = global_warp - row * num_q_heads;
  if (row >= num_rows) {
    return;
  }

  int32_t token_pos;
  int64_t phys_block = 0;
  int32_t block_row = 0;
  bool valid_row = false;
  if constexpr (USE_META) {
    // Prefill path: token_pos and the packed slot (= phys_block*block_size +
    // block_row, or -1 when invalid) are precomputed once per row by
    // compute_row_meta_kernel. This removes the per-warp req lookup -- including
    // the two runtime 64-bit divisions in the guess -- which was otherwise
    // recomputed redundantly by every one of the num_q_heads warps of a row.
    token_pos = row_token_pos[row];
    const int64_t slot = row_slot_meta[row];
    valid_row = slot >= 0;
    if constexpr (BSIZE != 0) {
      constexpr int kBsLog = __builtin_ctz(BSIZE);
      phys_block = slot >> kBsLog;
      block_row = static_cast<int32_t>(slot & (BSIZE - 1));
    } else {
      const int bs_log = __builtin_ctz(static_cast<unsigned>(block_size));
      phys_block = slot >> bs_log;
      block_row = static_cast<int32_t>(slot & (block_size - 1));
    }
  } else {
    int64_t req;
    int32_t req_start;
    int32_t req_end;
    int32_t seq_len;
    if constexpr (DecodeOneToken) {
      req = row;
      req_start = static_cast<int32_t>(row);
      req_end = static_cast<int32_t>(row + 1);
      seq_len = num_seqlen_per_req[req];
      token_pos = (seq_len > 0) ? (seq_len - 1) : 0;
    } else {
      // Decode batches almost always have a uniform query length per request, so
      // guess req = row / qlen directly (two adjacent q_index loads to verify)
      // instead of a log2(num_req) chain of dependent loads. Fall back to the
      // binary search only when the guess misses (ragged / mixed-qlen batch), so
      // the result is always exact.
      const int64_t qlen_guess = (num_req > 0) ? (num_rows / num_req) : 1;
      int64_t guess = (qlen_guess > 0) ? (row / qlen_guess) : 0;
      if (guess >= num_req) {
        guess = num_req - 1;
      }
      req_start = q_index[guess];
      req_end = q_index[guess + 1];
      if (row >= req_start && row < req_end) {
        req = guess;
      } else {
        req = find_req_for_row(q_index, num_req, row);
        req_start = q_index[req];
        req_end = q_index[req + 1];
      }
      seq_len = num_seqlen_per_req[req];
      token_pos = (seq_len > 0) ? static_cast<int32_t>(row + seq_len - req_end) : 0;
    }

    if (seq_len > 0 && req_end > req_start) {
      const int block_idx = idiv<BSIZE>(token_pos, static_cast<int>(block_size));
      block_row = static_cast<int32_t>(imod<BSIZE>(token_pos, static_cast<int>(block_size)));
      phys_block = static_cast<int64_t>(
          kvcache_indices[req * stride_kvi_r + block_idx * stride_kvi_b]);
      const int64_t slot = phys_block * block_size + block_row;
      valid_row = slot >= 0 && slot < total_num_kv_cache_tokens;
    }
  }

  const int cos_kind = CosSinBF16 ? kBFloat16 : kFloat32;
  const int weight_kind = WeightBF16 ? kBFloat16 : kFloat32;
  const float had_scale = hadamard_scale_128();
  const float fp8_max_inv = fast_rcp(fp8_max);
  (void)hadamard;

  // Effective innermost strides: for the standard contiguous vLLM layout these
  // are all 1, so the launcher selects CONTIG=true and the compiler drops the
  // runtime stride multiplies in every load/store address (a big chunk of the
  // remaining v_mul_lo/v_cndmask). CONTIG=false keeps the fully general path.
  const int64_t es_qkv_d = CONTIG ? 1 : stride_qkv_d;
  const int64_t es_cos_d = CONTIG ? 1 : stride_cos_d;
  const int64_t es_oq_d = CONTIG ? 1 : stride_out_q_d;
  const int64_t es_kc_x = CONTIG ? 1 : stride_kc_x;
  const int64_t es_vc_t = CONTIG ? 1 : stride_vc_t;

  // 2+2 register<->head_dim layout (see tiled kernel for the full rationale):
  //   d(i) = lane*2 + (i&1) + 64*(i>>1)
  // keeps RoPE rotate-half in-register (i<->i+2) while making (i0,i1)/(i2,i3)
  // contiguous so qkv loads coalesce as bf16x2 and fp8 stores emit one b16/pair.
  float q_vals[4];
  float local_sq = 0.0f;
  const int64_t q_hb = row * stride_qkv_t + hq * kHeadDim * es_qkv_d;
  if constexpr (CONTIG) {
    const __hip_bfloat162 p0 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[q_hb + lane * 2]);
    const __hip_bfloat162 p1 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[q_hb + lane * 2 + 64]);
    q_vals[0] = static_cast<float>(p0.x); q_vals[1] = static_cast<float>(p0.y);
    q_vals[2] = static_cast<float>(p1.x); q_vals[3] = static_cast<float>(p1.y);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
      q_vals[i] = static_cast<float>(qkv[q_hb + d * es_qkv_d]);
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) local_sq += q_vals[i] * q_vals[i];
  const float q_rms_inv = rsqrtf(warp_sum(local_sq) / static_cast<float>(kHeadDim) + eps);

  // Defer the uniform 1/rms factor: it is linear through RoPE + Hadamard, so it
  // cancels in the quantized output and only re-enters the stored scale. This
  // removes a per-element divide from the hot path.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
    const float w = load_scalar_f32(q_norm_weight, weight_kind, d);
    q_vals[i] = q_vals[i] * w;
  }

  float rope_cos[4];
  float rope_sin[4];
  float q_hadamard[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
    // NeoX rotate-half partner is x[d +/- 64], i.e. register i+/-2 in the same
    // lane (the (i>>1) term in d is exactly the +64 half-offset). lane in [0,31]
    // => (d<64) == (i<2) is a compile-time fact for each unrolled i. Spelling it
    // as (i<2) keeps the partner index a constant so the compiler avoids a
    // dynamic-VGPR select tree (the source of ~170 v_cndmask in the old build).
    const float rotated = (i < 2) ? -q_vals[i + 2] : q_vals[i - 2];
    const int d_mod = d & 63;
    const int64_t cos_off = token_pos * stride_cos_t + d_mod * es_cos_d;
    rope_cos[i] = load_scalar_f32(cos_sin, cos_kind, cos_off);
    rope_sin[i] = load_scalar_f32(cos_sin, cos_kind, cos_off + 64 * es_cos_d);
    // had_scale is deferred: it is a uniform scalar and FWHT is linear, so it
    // factors out of every element and re-enters only the stored scale. Saves a
    // per-element multiply on Q and K.
    q_hadamard[i] = q_vals[i] * rope_cos[i] + rotated * rope_sin[i];
  }
  fwht128_regs(q_hadamard, lane);

  // q_rms_inv and had_scale are both uniform/linear through FWHT -> fold into one
  // factor used for both the stored scale and the dequant divisor.
  const float q_rms_had = q_rms_inv * had_scale;
  float q_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    q_abs = fmaxf(q_abs, fabsf(q_hadamard[i]));
  }
  const float q_amax = warp_max(q_abs) * q_rms_had;
  const float q_scale = fmaxf(q_amax * fp8_max_inv, 1.0e-12f);
  if (lane == 0) {
    q_scale_out[row * stride_qs_t + hq * stride_qs_h] = q_scale;
  }
  const float q_inv_scale = q_rms_had * fast_rcp(q_scale);
  const int64_t oq_h = row * stride_out_q_t + hq * stride_out_q_h;
  const uint32_t qp01 = float2_to_fp8x2(q_hadamard[0] * q_inv_scale, q_hadamard[1] * q_inv_scale);
  const uint32_t qp23 = float2_to_fp8x2(q_hadamard[2] * q_inv_scale, q_hadamard[3] * q_inv_scale);
  if constexpr (CONTIG) {
    *reinterpret_cast<uint16_t*>(&out_q[oq_h + lane * 2]) = (uint16_t)qp01;
    *reinterpret_cast<uint16_t*>(&out_q[oq_h + lane * 2 + 64]) = (uint16_t)qp23;
  } else {
    out_q[oq_h + (lane * 2) * es_oq_d] = (uint8_t)(qp01);
    out_q[oq_h + (lane * 2 + 1) * es_oq_d] = (uint8_t)(qp01 >> 8);
    out_q[oq_h + (lane * 2 + 64) * es_oq_d] = (uint8_t)(qp23);
    out_q[oq_h + (lane * 2 + 65) * es_oq_d] = (uint8_t)(qp23 >> 8);
  }

  if (hq >= num_kv_heads) {
    return;
  }

  const int64_t q_dim = num_q_heads * kHeadDim;
  const int64_t k_off_base = q_dim + hq * kHeadDim;
  float k_vals[4];
  const int64_t k_hb = row * stride_qkv_t + k_off_base * es_qkv_d;
  if constexpr (CONTIG) {
    const __hip_bfloat162 p0 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[k_hb + lane * 2]);
    const __hip_bfloat162 p1 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[k_hb + lane * 2 + 64]);
    k_vals[0] = static_cast<float>(p0.x); k_vals[1] = static_cast<float>(p0.y);
    k_vals[2] = static_cast<float>(p1.x); k_vals[3] = static_cast<float>(p1.y);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
      k_vals[i] = static_cast<float>(qkv[k_hb + d * es_qkv_d]);
    }
  }
  local_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) local_sq += k_vals[i] * k_vals[i];
  const float k_rms_inv = rsqrtf(warp_sum(local_sq) / static_cast<float>(kHeadDim) + eps);

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
    const float w = load_scalar_f32(k_norm_weight, weight_kind, d);
    k_vals[i] = k_vals[i] * w;
  }

  float k_hadamard[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float rotated = (i < 2) ? -k_vals[i + 2] : k_vals[i - 2];
    k_hadamard[i] = k_vals[i] * rope_cos[i] + rotated * rope_sin[i];
  }
  fwht128_regs(k_hadamard, lane);

  const float k_rms_had = k_rms_inv * had_scale;
  float k_abs = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    k_abs = fmaxf(k_abs, fabsf(k_hadamard[i]));
  }

  const float k_amax = warp_max(k_abs) * k_rms_had;
  const float k_scale_dyn = fmaxf(k_amax * fp8_max_inv, 1.0e-12f);
  const float k_inv_scale = k_rms_had * fast_rcp(k_scale_dyn);
  if (lane == 0 && valid_row) {
    const int r_idx = idiv<KSL>(block_row, static_cast<int>(k_scale_l));
    const int l_idx = imod<KSL>(block_row, static_cast<int>(k_scale_l));
    const int64_t ks_off =
        phys_block * stride_ks_b + r_idx * stride_ks_r + hq * stride_ks_h + l_idx * stride_ks_l;
    k_scale[ks_off] = k_scale_dyn;
  }

  const uint32_t kp01 = float2_to_fp8x2(k_hadamard[0] * k_inv_scale, k_hadamard[1] * k_inv_scale);
  const uint32_t kp23 = float2_to_fp8x2(k_hadamard[2] * k_inv_scale, k_hadamard[3] * k_inv_scale);
  const uint8_t kb[4] = {(uint8_t)(kp01), (uint8_t)(kp01 >> 8),
                         (uint8_t)(kp23), (uint8_t)(kp23 >> 8)};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
    if (valid_row) {
      const int k_group = idiv<KCX>(d, static_cast<int>(k_cache_x));
      const int k_x = imod<KCX>(d, static_cast<int>(k_cache_x));
      const int64_t kc_off = phys_block * stride_kc_b + hq * stride_kc_h +
                             k_group * stride_kc_g + block_row * stride_kc_t +
                             k_x * es_kc_x;
      key_cache[kc_off] = kb[i];
    }
  }

  const int64_t k_dim = num_kv_heads * kHeadDim;
  const int64_t v_off_base = q_dim + k_dim + hq * v_head_dim;
  const float v_scale_inv = fast_rcp(v_scale[hq]);
  const int v_iters = static_cast<int>((v_head_dim + kWarpSize - 1) / kWarpSize);
  for (int i = 0; i < v_iters; ++i) {
    const int d = lane + i * kWarpSize;
    if (d < v_head_dim && valid_row) {
      const int64_t v_in_off = row * stride_qkv_t + (v_off_base + d) * es_qkv_d;
      const float v = static_cast<float>(qkv[v_in_off]) * v_scale_inv;
      const int64_t vc_off =
          phys_block * stride_vc_b + hq * stride_vc_h + d * stride_vc_d + block_row * es_vc_t;
      value_cache[vc_off] = float_to_fp8_byte(v, fp8_max);
    }
  }
}

// ===================== Multi-head tiled shuffle variant =====================
struct RowMeta {
  int32_t token_pos;
  int64_t phys_block;
  int32_t block_row;
  bool valid;
};

template <bool DecodeOneToken>
__device__ __forceinline__ RowMeta compute_row_meta(
    int64_t row, const int32_t* __restrict__ q_index,
    const int32_t* __restrict__ num_seqlen_per_req,
    const int32_t* __restrict__ kvcache_indices, int64_t num_req, int64_t num_rows,
    int64_t total_num_kv_cache_tokens, int64_t stride_kvi_r, int64_t stride_kvi_b,
    int64_t block_size) {
  RowMeta m{0, 0, 0, false};
  int64_t req;
  int32_t req_start, req_end, seq_len;
  if constexpr (DecodeOneToken) {
    req = row;
    seq_len = num_seqlen_per_req[req];
    m.token_pos = (seq_len > 0) ? (seq_len - 1) : 0;
    req_start = static_cast<int32_t>(row);
    req_end = static_cast<int32_t>(row + 1);
  } else {
    const int64_t qlen_guess = (num_req > 0) ? (num_rows / num_req) : 1;
    int64_t guess = (qlen_guess > 0) ? (row / qlen_guess) : 0;
    if (guess >= num_req) guess = num_req - 1;
    req_start = q_index[guess];
    req_end = q_index[guess + 1];
    if (row >= req_start && row < req_end) {
      req = guess;
    } else {
      req = find_req_for_row(q_index, num_req, row);
      req_start = q_index[req];
      req_end = q_index[req + 1];
    }
    seq_len = num_seqlen_per_req[req];
    m.token_pos = (seq_len > 0) ? static_cast<int32_t>(row + seq_len - req_end) : 0;
  }
  if (seq_len > 0 && req_end > req_start) {
    const int bs_log = __builtin_ctz(static_cast<unsigned>(block_size));
    const int64_t block_idx = m.token_pos >> bs_log;
    m.block_row = static_cast<int32_t>(m.token_pos & (block_size - 1));
    m.phys_block = static_cast<int64_t>(
        kvcache_indices[req * stride_kvi_r + block_idx * stride_kvi_b]);
    const int64_t slot = m.phys_block * block_size + m.block_row;
    m.valid = slot >= 0 && slot < total_num_kv_cache_tokens;
  }
  return m;
}

// 1 warp processes HPW q-heads of the same row. Row metadata, cos/sin and norm
// weights are computed once and shared across the heads (amortizes the heavy
// SALU/addressing), and the HPW heads' FWHT + reductions run interleaved
// (HPW*4-way ILP) to hide the ds_swizzle latency. Requires num_q_heads % HPW
// == 0 and num_kv_heads % HPW == 0 so a head-group is uniformly K-active.
#ifndef TILED_LB_MINW
#define TILED_LB_MINW 0
#endif
#if TILED_LB_MINW > 0
#define TILED_LB __launch_bounds__(128, TILED_LB_MINW)
#else
#define TILED_LB
#endif
template <bool CosSinBF16, bool WeightBF16, bool HadamardBF16, bool DecodeOneToken, int HPW,
          int BSIZE = 0, int KCX = 0, int KSL = 0, bool CONTIG = false, bool USE_META = false>
__global__ void TILED_LB rope_norm_store_kv_fp8_fused_tiled_kernel(
    const __hip_bfloat16* __restrict__ qkv,
    const void* __restrict__ cos_sin,
    const int32_t* __restrict__ q_index,
    const int32_t* __restrict__ num_seqlen_per_req,
    const int32_t* __restrict__ kvcache_indices,
    const void* __restrict__ q_norm_weight,
    const void* __restrict__ k_norm_weight,
    const void* __restrict__ hadamard,
    float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    uint8_t* __restrict__ out_q,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    float* __restrict__ q_scale_out,
    int64_t num_rows, int64_t num_req, int64_t total_num_kv_cache_tokens,
    float eps, float fp8_max,
    int64_t stride_qkv_t, int64_t stride_qkv_d,
    int64_t stride_cos_t, int64_t stride_cos_d,
    int64_t stride_kvi_r, int64_t stride_kvi_b,
    int64_t stride_out_q_t, int64_t stride_out_q_h, int64_t stride_out_q_d,
    int64_t stride_kc_b, int64_t stride_kc_h, int64_t stride_kc_g, int64_t stride_kc_t, int64_t stride_kc_x,
    int64_t stride_vc_b, int64_t stride_vc_h, int64_t stride_vc_d, int64_t stride_vc_t,
    int64_t stride_ks_b, int64_t stride_ks_r, int64_t stride_ks_h, int64_t stride_ks_l,
    int64_t stride_qs_t, int64_t stride_qs_h,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t v_head_dim,
    int64_t block_size, int64_t k_scale_l, int64_t k_cache_x,
    const int32_t* __restrict__ row_token_pos,
    const int64_t* __restrict__ row_slot_meta) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int64_t head_groups = num_q_heads / HPW;
  const int64_t global_warp = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  const int64_t row = global_warp / head_groups;
  const int64_t hq_base = (global_warp - row * head_groups) * HPW;
  if (row >= num_rows) {
    return;
  }

  RowMeta meta{0, 0, 0, false};
  if constexpr (USE_META) {
    // Prefill: per-row (token_pos, packed slot) precomputed once by
    // compute_row_meta_kernel -> the head-group warps just load them.
    meta.token_pos = row_token_pos[row];
    const int64_t slot = row_slot_meta[row];
    meta.valid = slot >= 0;
    if constexpr (BSIZE != 0) {
      constexpr int kBsLog = __builtin_ctz(BSIZE);
      meta.phys_block = slot >> kBsLog;
      meta.block_row = static_cast<int32_t>(slot & (BSIZE - 1));
    } else {
      const int bs_log = __builtin_ctz(static_cast<unsigned>(block_size));
      meta.phys_block = slot >> bs_log;
      meta.block_row = static_cast<int32_t>(slot & (block_size - 1));
    }
  } else {
    meta = compute_row_meta<DecodeOneToken>(
        row, q_index, num_seqlen_per_req, kvcache_indices, num_req, num_rows,
        total_num_kv_cache_tokens, stride_kvi_r, stride_kvi_b, block_size);
  }
  const int32_t token_pos = meta.token_pos;

  const int cos_kind = CosSinBF16 ? kBFloat16 : kFloat32;
  const int weight_kind = WeightBF16 ? kBFloat16 : kFloat32;
  const float had_scale = hadamard_scale_128();
  const float fp8_max_inv = fast_rcp(fp8_max);  // one rcp, reused by every head
  (void)hadamard;

  const int64_t es_qkv_d = CONTIG ? 1 : stride_qkv_d;
  const int64_t es_cos_d = CONTIG ? 1 : stride_cos_d;
  const int64_t es_oq_d = CONTIG ? 1 : stride_out_q_d;
  const int64_t es_kc_x = CONTIG ? 1 : stride_kc_x;
  const int64_t es_vc_t = CONTIG ? 1 : stride_vc_t;

  // cos/sin + q norm weight are identical for every head of this row.
  // 2+2 register<->head_dim layout: reg i owns head_dim index
  //   d(i) = lane*2 + (i&1) + 64*(i>>1)
  // This keeps RoPE rotate-half (j <-> j+64) inside registers (i <-> i+2, since
  // the (i>>1) term is exactly the +64 half-offset) AND makes the pair (i0,i1)
  // and (i2,i3) CONTIGUOUS in head_dim, so qkv loads coalesce as bf16x2 dwords
  // and fp8 stores emit one b16 per pair (the packed v_cvt_pk_fp8_f32 output).
  // The FWHT is bit-separable so this permutation is bit-exact vs the strided
  // lane+32*i layout (intra-lane combines output bits {0,6}, cross-lane bits
  // 1..5; verified by the cmp_ref / abkit correctness checks).
  float rope_cos[4], rope_sin[4], qw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
    const int dm = d & 63;
    const int64_t coff = token_pos * stride_cos_t + dm * es_cos_d;
    rope_cos[i] = load_scalar_f32(cos_sin, cos_kind, coff);
    rope_sin[i] = load_scalar_f32(cos_sin, cos_kind, coff + 64 * es_cos_d);
    qw[i] = load_scalar_f32(q_norm_weight, weight_kind, d);
  }
  // Fold the (uniform-per-row) RMSNorm weight into the RoPE coefficients ONCE
  // per row, so the per-head loop drops its `qv *= qw` multiply (amortized over
  // HPW heads). qh[i] = qv[i]*qw[i]*cos[i] +/- qv[i±2]*qw[i±2]*sin[i]
  //         = qv[i]*qwc[i] +/- qv[i±2]*qws[i], with the rotate using RAW qv.
  // qws carries the PARTNER's weight: partner(i) = (i+2)&3. Algebraically exact.
  float qwc[4], qws[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    qwc[i] = qw[i] * rope_cos[i];
    qws[i] = qw[(i + 2) & 3] * rope_sin[i];
  }

  // ===== Q (HPW heads interleaved) =====
  float qv[HPW][4];
  float qsq[HPW];
#pragma unroll
  for (int h = 0; h < HPW; ++h) {
    const int64_t hb = row * stride_qkv_t + (hq_base + h) * kHeadDim * es_qkv_d;
    if constexpr (CONTIG) {
      // contiguous bf16x2 loads of the two RoPE-pair dwords (d=lane*2, lane*2+64)
      const __hip_bfloat162 p0 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[hb + lane * 2]);
      const __hip_bfloat162 p1 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[hb + lane * 2 + 64]);
      qv[h][0] = static_cast<float>(p0.x); qv[h][1] = static_cast<float>(p0.y);
      qv[h][2] = static_cast<float>(p1.x); qv[h][3] = static_cast<float>(p1.y);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
        qv[h][i] = static_cast<float>(qkv[hb + d * es_qkv_d]);
      }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) s += qv[h][i] * qv[h][i];
    qsq[h] = s;
  }
  warp_sum_multi<HPW>(qsq);
  float q_rms_inv[HPW];
#pragma unroll
  for (int h = 0; h < HPW; ++h) q_rms_inv[h] = rsqrtf(qsq[h] / static_cast<float>(kHeadDim) + eps);

  float qh[HPW][4];
#pragma unroll
  for (int h = 0; h < HPW; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // RAW qv for both the direct and rotate terms; weight is folded into
      // qwc/qws above. had_scale deferred into the per-head scale below.
      const float rot = (i < 2) ? -qv[h][i + 2] : qv[h][i - 2];
      qh[h][i] = qv[h][i] * qwc[i] + rot * qws[i];
    }
  }
  fwht128_regs_multi<HPW>(qh, lane);

  float qabs[HPW];
#pragma unroll
  for (int h = 0; h < HPW; ++h) {
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) a = fmaxf(a, fabsf(qh[h][i]));
    qabs[h] = a;
  }
  warp_max_multi<HPW>(qabs);
#pragma unroll
  for (int h = 0; h < HPW; ++h) {
    const int64_t hq = hq_base + h;
    const float q_rms_had = q_rms_inv[h] * had_scale;
    const float q_amax = qabs[h] * q_rms_had;
    const float q_scale = fmaxf(q_amax * fp8_max_inv, 1.0e-12f);
    if (lane == 0) q_scale_out[row * stride_qs_t + hq * stride_qs_h] = q_scale;
    const float q_inv = q_rms_had * fast_rcp(q_scale);
    const int64_t oq_h = row * stride_out_q_t + hq * stride_out_q_h;
    const uint32_t qp01 = float2_to_fp8x2(qh[h][0] * q_inv, qh[h][1] * q_inv);
    const uint32_t qp23 = float2_to_fp8x2(qh[h][2] * q_inv, qh[h][3] * q_inv);
    if constexpr (CONTIG) {
      *reinterpret_cast<uint16_t*>(&out_q[oq_h + lane * 2]) = (uint16_t)qp01;
      *reinterpret_cast<uint16_t*>(&out_q[oq_h + lane * 2 + 64]) = (uint16_t)qp23;
    } else {
      out_q[oq_h + (lane * 2) * es_oq_d] = (uint8_t)(qp01);
      out_q[oq_h + (lane * 2 + 1) * es_oq_d] = (uint8_t)(qp01 >> 8);
      out_q[oq_h + (lane * 2 + 64) * es_oq_d] = (uint8_t)(qp23);
      out_q[oq_h + (lane * 2 + 65) * es_oq_d] = (uint8_t)(qp23 >> 8);
    }
  }

  // ===== K + V =====
  // Only the kv-heads that fall inside this head-group. Done per-head (scalar
  // reductions) so it stays correct for GQA where num_kv_heads < HPW (e.g.
  // num_kv_heads == 1): a head-group then owns at most one real kv-head. K/V is
  // ~1/num_q_heads of the work, so not tiling it costs little while letting Q
  // use any HPW.
  if (hq_base >= num_kv_heads) {
    return;
  }
  const int64_t q_dim = num_q_heads * kHeadDim;
  const int64_t k_dim = num_kv_heads * kHeadDim;
  float kw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) kw[i] = load_scalar_f32(k_norm_weight, weight_kind, lane * 2 + (i & 1) + 64 * (i >> 1));

#pragma unroll
  for (int h = 0; h < HPW; ++h) {
    const int64_t hq = hq_base + h;
    if (hq >= num_kv_heads) {
      continue;
    }
    float kv[4];
    const int64_t kb_in = row * stride_qkv_t + (q_dim + hq * kHeadDim) * es_qkv_d;
    if constexpr (CONTIG) {
      const __hip_bfloat162 p0 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[kb_in + lane * 2]);
      const __hip_bfloat162 p1 = *reinterpret_cast<const __hip_bfloat162*>(&qkv[kb_in + lane * 2 + 64]);
      kv[0] = static_cast<float>(p0.x); kv[1] = static_cast<float>(p0.y);
      kv[2] = static_cast<float>(p1.x); kv[3] = static_cast<float>(p1.y);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
        kv[i] = static_cast<float>(qkv[kb_in + d * es_qkv_d]);
      }
    }
    float ksq = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) ksq += kv[i] * kv[i];
    const float k_rms_inv = rsqrtf(warp_sum(ksq) / static_cast<float>(kHeadDim) + eps);
    float kh[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) kv[i] *= kw[i];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float rot = (i < 2) ? -kv[i + 2] : kv[i - 2];
      kh[i] = kv[i] * rope_cos[i] + rot * rope_sin[i];
    }
    fwht128_regs(kh, lane);
    float kabs = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) kabs = fmaxf(kabs, fabsf(kh[i]));
    const float k_rms_had = k_rms_inv * had_scale;
    const float k_amax = warp_max(kabs) * k_rms_had;
    const float k_scale_dyn = fmaxf(k_amax * fp8_max_inv, 1.0e-12f);
    const float k_inv = k_rms_had * fast_rcp(k_scale_dyn);
    if (lane == 0 && meta.valid) {
      const int r_idx = idiv<KSL>(meta.block_row, static_cast<int>(k_scale_l));
      const int l_idx = imod<KSL>(meta.block_row, static_cast<int>(k_scale_l));
      k_scale[meta.phys_block * stride_ks_b + r_idx * stride_ks_r + hq * stride_ks_h +
              l_idx * stride_ks_l] = k_scale_dyn;
    }
    const uint32_t kp01 = float2_to_fp8x2(kh[0] * k_inv, kh[1] * k_inv);
    const uint32_t kp23 = float2_to_fp8x2(kh[2] * k_inv, kh[3] * k_inv);
    const uint8_t kb[4] = {(uint8_t)(kp01), (uint8_t)(kp01 >> 8),
                           (uint8_t)(kp23), (uint8_t)(kp23 >> 8)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane * 2 + (i & 1) + 64 * (i >> 1);
      if (meta.valid) {
        const int kg = idiv<KCX>(d, static_cast<int>(k_cache_x));
        const int kx = imod<KCX>(d, static_cast<int>(k_cache_x));
        key_cache[meta.phys_block * stride_kc_b + hq * stride_kc_h + kg * stride_kc_g +
                  meta.block_row * stride_kc_t + kx * es_kc_x] = kb[i];
      }
    }
    // V (no Hadamard)
    const int64_t v_off_base = q_dim + k_dim + hq * v_head_dim;
    const float v_scale_inv = fast_rcp(v_scale[hq]);
    const int v_iters = static_cast<int>((v_head_dim + kWarpSize - 1) / kWarpSize);
    for (int it = 0; it < v_iters; ++it) {
      const int d = lane + it * kWarpSize;
      if (d < v_head_dim && meta.valid) {
        const float v = static_cast<float>(
            qkv[row * stride_qkv_t + (v_off_base + d) * es_qkv_d]) * v_scale_inv;
        value_cache[meta.phys_block * stride_vc_b + hq * stride_vc_h + d * stride_vc_d +
                    meta.block_row * es_vc_t] = float_to_fp8_byte(v, fp8_max);
      }
    }
  }
}


template <bool WeightBF16, bool HadamardBF16>
void launch_cos_dispatch(const torch::Tensor& qkv,
                         const torch::Tensor& cos_sin,
                         const torch::Tensor& q_index,
                         const torch::Tensor& num_seqlen_per_req,
                         const torch::Tensor& kvcache_indices,
                         const torch::Tensor& q_norm_weight,
                         const torch::Tensor& k_norm_weight,
                         const torch::Tensor& hadamard,
                         const torch::Tensor& k_scale,
                         const torch::Tensor& v_scale,
                         const torch::Tensor& out_q,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const torch::Tensor& q_scale_out,
                         float eps,
                         float fp8_max,
                         bool decode_one_token) {
  const int64_t num_rows = qkv.size(0);
  const int64_t num_req = num_seqlen_per_req.size(0);
  const int64_t num_q_heads = out_q.size(1);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t v_head_dim = value_cache.size(2);
  const int64_t block_size = key_cache.size(3);
  const int64_t k_cache_x = key_cache.size(4);
  const int64_t k_scale_l = k_scale.size(3);
  const int64_t total_tokens = key_cache.size(0) * block_size;
  const int64_t total_warps = num_rows * num_q_heads;
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const size_t smem = 0;
  const auto stream = c10::hip::getCurrentHIPStream();

  // Prefill (!decode_one_token, many rows): precompute per-row (token_pos,
  // packed slot) once with a tiny fully-parallel kernel, so the fused kernel
  // skips the per-warp req lookup. The extra kernel only pays off when there
  // are enough rows to amortize its launch; small/decode-shaped batches keep
  // the inline path (no second launch).
  const bool use_meta = !decode_one_token && num_rows >= 1024;
  const int32_t* row_token_pos_ptr = nullptr;
  const int64_t* row_slot_ptr = nullptr;
  torch::Tensor row_token_pos_t;
  torch::Tensor row_slot_t;
  if (use_meta) {
    row_token_pos_t = torch::empty({num_rows}, q_index.options().dtype(torch::kInt32));
    row_slot_t = torch::empty({num_rows}, q_index.options().dtype(torch::kInt64));
    row_token_pos_ptr = row_token_pos_t.data_ptr<int32_t>();
    row_slot_ptr = row_slot_t.data_ptr<int64_t>();
    const int meta_threads = 256;
    const dim3 meta_grid((num_rows + meta_threads - 1) / meta_threads);
    hipLaunchKernelGGL(compute_row_meta_kernel, meta_grid, dim3(meta_threads), 0, stream,
                       q_index.data_ptr<int32_t>(), num_seqlen_per_req.data_ptr<int32_t>(),
                       kvcache_indices.data_ptr<int32_t>(), num_rows, num_req, total_tokens,
                       kvcache_indices.stride(0), kvcache_indices.stride(1), block_size,
                       row_token_pos_t.data_ptr<int32_t>(), row_slot_t.data_ptr<int64_t>());
  }

#define DECODE_ARGS                                                                                \
  reinterpret_cast<const __hip_bfloat16*>(qkv.data_ptr()), cos_sin.data_ptr(),                     \
      q_index.data_ptr<int32_t>(), num_seqlen_per_req.data_ptr<int32_t>(),                         \
      kvcache_indices.data_ptr<int32_t>(), q_norm_weight.data_ptr(), k_norm_weight.data_ptr(),     \
      hadamard.data_ptr(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),                   \
      reinterpret_cast<uint8_t*>(out_q.data_ptr()),                                                \
      reinterpret_cast<uint8_t*>(key_cache.data_ptr()),                                            \
      reinterpret_cast<uint8_t*>(value_cache.data_ptr()), q_scale_out.data_ptr<float>(), num_rows, \
      num_req, total_tokens, eps, fp8_max, qkv.stride(0), qkv.stride(1), cos_sin.stride(0),         \
      cos_sin.stride(1), kvcache_indices.stride(0), kvcache_indices.stride(1), out_q.stride(0),     \
      out_q.stride(1), out_q.stride(2), key_cache.stride(0), key_cache.stride(1),                   \
      key_cache.stride(2), key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),         \
      value_cache.stride(1), value_cache.stride(2), value_cache.stride(3), k_scale.stride(0),       \
      k_scale.stride(1), k_scale.stride(2), k_scale.stride(3), q_scale_out.stride(0),               \
      q_scale_out.stride(1), num_q_heads, num_kv_heads, v_head_dim, block_size, k_scale_l,          \
      k_cache_x, row_token_pos_ptr, row_slot_ptr
#define LAUNCH_DECODE(CB, DEC, BS, KCX, KSL, CTG, META)                                            \
  hipLaunchKernelGGL(                                                                              \
      (rope_norm_store_kv_fp8_fused_kernel<CB, WeightBF16, HadamardBF16, DEC, BS, KCX, KSL, CTG, META>), \
      grid, block, smem, stream, DECODE_ARGS)
  // Standard contiguous vLLM layout -> innermost strides are 1.
  const bool contig = qkv.stride(1) == 1 && cos_sin.stride(1) == 1 && out_q.stride(2) == 1 &&
                      key_cache.stride(4) == 1 && value_cache.stride(3) == 1;
  // Only CONTIG (innermost-stride==1) needs a compile-time branch: it selects the
  // vectorized bf16x2/b16 load-store path. The KV-cache div/mod are now runtime
  // power-of-two shifts (block-size agnostic), so we do NOT specialize on
  // block_size/k_cache_x/k_scale_l. META mirrors use_meta (prefill = !decode).
#define LAUNCH_DECODE_DIV(CB, DEC, META)                                                            \
  do {                                                                                             \
    if (contig)                                                                                    \
      LAUNCH_DECODE(CB, DEC, 0, 0, 0, true, META);                                                 \
    else                                                                                           \
      LAUNCH_DECODE(CB, DEC, 0, 0, 0, false, META);                                                \
  } while (0)

  // Prefill bool path picks the precompute (META=true) or inline (false)
  // instantiation at runtime based on use_meta.
#define LAUNCH_PREFILL(CB)                                                                          \
  do {                                                                                             \
    if (use_meta)                                                                                  \
      LAUNCH_DECODE_DIV(CB, false, true);                                                          \
    else                                                                                           \
      LAUNCH_DECODE_DIV(CB, false, false);                                                         \
  } while (0)

  if (cos_sin.scalar_type() == at::kBFloat16) {
    if (decode_one_token)
      LAUNCH_DECODE_DIV(true, true, false);
    else
      LAUNCH_PREFILL(true);
  } else {
    if (decode_one_token)
      LAUNCH_DECODE_DIV(false, true, false);
    else
      LAUNCH_PREFILL(false);
  }
#undef LAUNCH_PREFILL
#undef LAUNCH_DECODE_DIV
#undef LAUNCH_DECODE
#undef DECODE_ARGS
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

template <bool WeightBF16, bool HadamardBF16, int HPW>
void launch_cos_dispatch_tiled(const torch::Tensor& qkv,
                               const torch::Tensor& cos_sin,
                               const torch::Tensor& q_index,
                               const torch::Tensor& num_seqlen_per_req,
                               const torch::Tensor& kvcache_indices,
                               const torch::Tensor& q_norm_weight,
                               const torch::Tensor& k_norm_weight,
                               const torch::Tensor& hadamard,
                               const torch::Tensor& k_scale,
                               const torch::Tensor& v_scale,
                               const torch::Tensor& out_q,
                               const torch::Tensor& key_cache,
                               const torch::Tensor& value_cache,
                               const torch::Tensor& q_scale_out,
                               float eps,
                               float fp8_max,
                               bool decode_one_token) {
  const int64_t num_rows = qkv.size(0);
  const int64_t num_req = num_seqlen_per_req.size(0);
  const int64_t num_q_heads = out_q.size(1);
  const int64_t num_kv_heads = key_cache.size(1);
  const int64_t v_head_dim = value_cache.size(2);
  const int64_t block_size = key_cache.size(3);
  const int64_t k_cache_x = key_cache.size(4);
  const int64_t k_scale_l = k_scale.size(3);
  const int64_t total_tokens = key_cache.size(0) * block_size;
  const int64_t total_warps = num_rows * (num_q_heads / HPW);
  const dim3 block(kWarpsPerBlock * kWarpSize);
  const dim3 grid(static_cast<unsigned>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock));
  const size_t smem = 0;
  const auto stream = c10::hip::getCurrentHIPStream();

  const bool use_meta = !decode_one_token && num_rows >= 1024;
  const int32_t* row_token_pos_ptr = nullptr;
  const int64_t* row_slot_ptr = nullptr;
  torch::Tensor row_token_pos_t;
  torch::Tensor row_slot_t;
  if (use_meta) {
    row_token_pos_t = torch::empty({num_rows}, q_index.options().dtype(torch::kInt32));
    row_slot_t = torch::empty({num_rows}, q_index.options().dtype(torch::kInt64));
    row_token_pos_ptr = row_token_pos_t.data_ptr<int32_t>();
    row_slot_ptr = row_slot_t.data_ptr<int64_t>();
    const int meta_threads = 256;
    const dim3 meta_grid((num_rows + meta_threads - 1) / meta_threads);
    hipLaunchKernelGGL(compute_row_meta_kernel, meta_grid, dim3(meta_threads), 0, stream,
                       q_index.data_ptr<int32_t>(), num_seqlen_per_req.data_ptr<int32_t>(),
                       kvcache_indices.data_ptr<int32_t>(), num_rows, num_req, total_tokens,
                       kvcache_indices.stride(0), kvcache_indices.stride(1), block_size,
                       row_token_pos_t.data_ptr<int32_t>(), row_slot_t.data_ptr<int64_t>());
  }
#define TILED_ARGS                                                                                 \
  reinterpret_cast<const __hip_bfloat16*>(qkv.data_ptr()), cos_sin.data_ptr(),                     \
      q_index.data_ptr<int32_t>(), num_seqlen_per_req.data_ptr<int32_t>(),                         \
      kvcache_indices.data_ptr<int32_t>(), q_norm_weight.data_ptr(), k_norm_weight.data_ptr(),     \
      hadamard.data_ptr(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),                   \
      reinterpret_cast<uint8_t*>(out_q.data_ptr()),                                                \
      reinterpret_cast<uint8_t*>(key_cache.data_ptr()),                                            \
      reinterpret_cast<uint8_t*>(value_cache.data_ptr()), q_scale_out.data_ptr<float>(), num_rows, \
      num_req, total_tokens, eps, fp8_max, qkv.stride(0), qkv.stride(1), cos_sin.stride(0),         \
      cos_sin.stride(1), kvcache_indices.stride(0), kvcache_indices.stride(1), out_q.stride(0),     \
      out_q.stride(1), out_q.stride(2), key_cache.stride(0), key_cache.stride(1),                   \
      key_cache.stride(2), key_cache.stride(3), key_cache.stride(4), value_cache.stride(0),         \
      value_cache.stride(1), value_cache.stride(2), value_cache.stride(3), k_scale.stride(0),       \
      k_scale.stride(1), k_scale.stride(2), k_scale.stride(3), q_scale_out.stride(0),               \
      q_scale_out.stride(1), num_q_heads, num_kv_heads, v_head_dim, block_size, k_scale_l,          \
      k_cache_x, row_token_pos_ptr, row_slot_ptr
  const bool contig = qkv.stride(1) == 1 && cos_sin.stride(1) == 1 && out_q.stride(2) == 1 &&
                      key_cache.stride(4) == 1 && value_cache.stride(3) == 1;
#define LAUNCH_TILED_K(CB, DEC, BS, KCX, KSL, CTG, META)                                           \
  hipLaunchKernelGGL(                                                                              \
      (rope_norm_store_kv_fp8_fused_tiled_kernel<CB, WeightBF16, HadamardBF16, DEC, HPW, BS, \
                                                        KCX, KSL, CTG, META>),                      \
      grid, block, smem, stream, TILED_ARGS)
  // Only CONTIG needs a compile-time branch (vectorized load/store path). The
  // KV-cache div/mod are runtime power-of-two shifts now, so no block_size
  // specialization -- one tiled instance serves every power-of-two block_size.
#define LAUNCH_TILED_DIV(CB, DEC, META)                                                            \
  do {                                                                                             \
    if (contig)                                                                                    \
      LAUNCH_TILED_K(CB, DEC, 0, 0, 0, true, META);                                                \
    else                                                                                           \
      LAUNCH_TILED_K(CB, DEC, 0, 0, 0, false, META);                                               \
  } while (0)
#define LAUNCH_TILED_PREFILL(CB)                                                                   \
  do {                                                                                             \
    if (use_meta)                                                                                  \
      LAUNCH_TILED_DIV(CB, false, true);                                                           \
    else                                                                                           \
      LAUNCH_TILED_DIV(CB, false, false);                                                          \
  } while (0)
  if (cos_sin.scalar_type() == at::kBFloat16) {
    if (decode_one_token)
      LAUNCH_TILED_DIV(true, true, false);
    else
      LAUNCH_TILED_PREFILL(true);
  } else {
    if (decode_one_token)
      LAUNCH_TILED_DIV(false, true, false);
    else
      LAUNCH_TILED_PREFILL(false);
  }
#undef LAUNCH_TILED_PREFILL
#undef LAUNCH_TILED_DIV
#undef LAUNCH_TILED_K
#undef TILED_ARGS
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

template <bool WeightBF16>
void launch_hadamard_dispatch(const torch::Tensor& qkv,
                              const torch::Tensor& cos_sin,
                              const torch::Tensor& q_index,
                              const torch::Tensor& num_seqlen_per_req,
                              const torch::Tensor& kvcache_indices,
                              const torch::Tensor& q_norm_weight,
                              const torch::Tensor& k_norm_weight,
                              const torch::Tensor& hadamard,
                              const torch::Tensor& k_scale,
                              const torch::Tensor& v_scale,
                              const torch::Tensor& out_q,
                              const torch::Tensor& key_cache,
                              const torch::Tensor& value_cache,
                              const torch::Tensor& q_scale_out,
                              float eps,
                              float fp8_max,
                              bool decode_one_token,
                              int64_t tile_hpw) {
#define LAUNCH_TILED(HADBF, HPW)                                                                   \
  launch_cos_dispatch_tiled<WeightBF16, HADBF, HPW>(                                               \
      qkv, cos_sin, q_index, num_seqlen_per_req, kvcache_indices, q_norm_weight, k_norm_weight,    \
      hadamard, k_scale, v_scale, out_q, key_cache, value_cache, q_scale_out, eps, fp8_max,        \
      decode_one_token)
#define DISPATCH_TILED(HADBF)                                                                      \
  do {                                                                                             \
    if (tile_hpw >= 8)                                                                             \
      LAUNCH_TILED(HADBF, 8);                                                                       \
    else if (tile_hpw == 4)                                                                        \
      LAUNCH_TILED(HADBF, 4);                                                                       \
    else                                                                                           \
      LAUNCH_TILED(HADBF, 2);                                                                       \
  } while (0)
  if (hadamard.scalar_type() == at::kBFloat16) {
    if (tile_hpw > 1)
      DISPATCH_TILED(true);
    else
      launch_cos_dispatch<WeightBF16, true>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                            kvcache_indices, q_norm_weight, k_norm_weight,
                                            hadamard, k_scale, v_scale, out_q, key_cache,
                                            value_cache, q_scale_out, eps, fp8_max,
                                            decode_one_token);
  } else {
    if (tile_hpw > 1)
      DISPATCH_TILED(false);
    else
      launch_cos_dispatch<WeightBF16, false>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                             kvcache_indices, q_norm_weight, k_norm_weight,
                                             hadamard, k_scale, v_scale, out_q, key_cache,
                                             value_cache, q_scale_out, eps, fp8_max,
                                             decode_one_token);
  }
#undef DISPATCH_TILED
}

}  // namespace aiter

using namespace aiter;

void rope_norm_store_kv_fp8_fused_hip(torch::Tensor qkv,
                                             torch::Tensor cos_sin,
                                             torch::Tensor q_index,
                                             torch::Tensor num_seqlen_per_req,
                                             torch::Tensor kvcache_indices,
                                             torch::Tensor q_norm_weight,
                                             torch::Tensor k_norm_weight,
                                             torch::Tensor hadamard,
                                             torch::Tensor k_scale,
                                             torch::Tensor v_scale,
                                             torch::Tensor out_q,
                                             torch::Tensor key_cache,
                                             torch::Tensor value_cache,
                                             torch::Tensor q_scale_out,
                                             double eps,
                                             double fp8_max,
                                             bool assume_decode_one_token,
                                             int64_t tile_hpw) {
  CHECK_HIP_TENSOR(qkv);
  CHECK_HIP_TENSOR(cos_sin);
  CHECK_HIP_TENSOR(q_index);
  CHECK_HIP_TENSOR(num_seqlen_per_req);
  CHECK_HIP_TENSOR(kvcache_indices);
  CHECK_HIP_TENSOR(q_norm_weight);
  CHECK_HIP_TENSOR(k_norm_weight);
  CHECK_HIP_TENSOR(hadamard);
  CHECK_HIP_TENSOR(k_scale);
  CHECK_HIP_TENSOR(v_scale);
  CHECK_HIP_TENSOR(out_q);
  CHECK_HIP_TENSOR(key_cache);
  CHECK_HIP_TENSOR(value_cache);
  CHECK_HIP_TENSOR(q_scale_out);

  CHECK_DTYPE(qkv, at::kBFloat16);
  CHECK_DTYPE(q_index, at::kInt);
  CHECK_DTYPE(num_seqlen_per_req, at::kInt);
  CHECK_DTYPE(kvcache_indices, at::kInt);
  CHECK_DTYPE(k_scale, at::kFloat);
  CHECK_DTYPE(v_scale, at::kFloat);
  CHECK_DTYPE(q_scale_out, at::kFloat);
  TORCH_CHECK(cos_sin.scalar_type() == at::kBFloat16 || cos_sin.scalar_type() == at::kFloat,
              "cos_sin must be bfloat16 or float32");
  TORCH_CHECK(q_norm_weight.scalar_type() == k_norm_weight.scalar_type(),
              "q_norm_weight/k_norm_weight dtype must match");
  TORCH_CHECK(q_norm_weight.scalar_type() == at::kBFloat16 ||
                  q_norm_weight.scalar_type() == at::kFloat,
              "norm weights must be bfloat16 or float32");
  TORCH_CHECK(hadamard.scalar_type() == at::kBFloat16 || hadamard.scalar_type() == at::kFloat,
              "hadamard must be bfloat16 or float32");

  TORCH_CHECK(qkv.dim() == 2, "qkv must be [num_rows, hidden]");
  TORCH_CHECK(out_q.dim() == 3, "out_q must be [num_rows, num_q_heads, 128]");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must be [num_blocks, kv_heads, D/x, block, x]");
  TORCH_CHECK(value_cache.dim() == 4, "value_cache must be [num_blocks, kv_heads, V, block]");
  TORCH_CHECK(out_q.size(2) == kHeadDim, "this HIP prototype supports qk_head_dim == 128 only");
  TORCH_CHECK(key_cache.size(2) * key_cache.size(4) == kHeadDim,
              "key_cache implies qk_head_dim != 128");
  TORCH_CHECK(hadamard.numel() == kHeadDim * kHeadDim, "hadamard must be [128, 128]");
  TORCH_CHECK(q_scale_out.dim() == 2, "fused HIP path currently supports decode q_scale [rows, heads]");
  TORCH_CHECK(k_scale.dim() == 4, "dynamic k_scale must be [num_blocks, R, kv_heads, L]");
  TORCH_CHECK(v_scale.numel() == key_cache.size(1), "v_scale must be per KV head");
  TORCH_CHECK(out_q.element_size() == 1 && key_cache.element_size() == 1 &&
                  value_cache.element_size() == 1,
              "out_q/key_cache/value_cache must be byte-sized FP8 tensors");

  // The multi-head tiled kernel tiles Q by HPW; K/V is handled per-kv-head so
  // num_kv_heads need not be divisible by HPW (supports GQA, e.g. nkv==1). Only
  // num_q_heads must be divisible by the tile width.
  int64_t eff_tile_hpw = tile_hpw;
  if (eff_tile_hpw > 1) {
    const int64_t nqh = out_q.size(1);
    if (nqh % eff_tile_hpw != 0) {
      eff_tile_hpw = 1;
    }
  }
  if (q_norm_weight.scalar_type() == at::kBFloat16) {
    launch_hadamard_dispatch<true>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                   kvcache_indices, q_norm_weight, k_norm_weight, hadamard,
                                   k_scale, v_scale, out_q, key_cache, value_cache,
                                   q_scale_out, static_cast<float>(eps),
                                   static_cast<float>(fp8_max),
                                   assume_decode_one_token, eff_tile_hpw);
  } else {
    launch_hadamard_dispatch<false>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                    kvcache_indices, q_norm_weight, k_norm_weight, hadamard,
                                    k_scale, v_scale, out_q, key_cache, value_cache,
                                    q_scale_out, static_cast<float>(eps),
                                    static_cast<float>(fp8_max),
                                    assume_decode_one_token, eff_tile_hpw);
  }
}

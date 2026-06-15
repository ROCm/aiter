// rocWMMA needs float<->__half conversions that torch disables via
// -D__HIP_NO_HALF_CONVERSIONS__. Undef before any HIP/half header is pulled in,
// include rocWMMA, then restore the macros for the torch/ATen headers.
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#ifndef __HIP_NO_HALF_OPERATORS__
#define __HIP_NO_HALF_OPERATORS__ 1
#endif
#ifndef __HIP_NO_HALF_CONVERSIONS__
#define __HIP_NO_HALF_CONVERSIONS__ 1
#endif
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <torch/extension.h>

#include "hip_float8.h"

namespace aiter_rope {

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

// Compile-time divide/modulo: when the divisor C is a known constant (the
// common block_size=32 / k_cache_x=16 / k_scale_l=32 case) the compiler turns
// these into a shift / and; C==0 selects the runtime divisor. The whole reason
// for this is that prefill is VALU-bound almost entirely on the integer
// div/mod magic-number sequences these addressing ops generate, so killing the
// runtime division is the single biggest lever (verified via rocprofv3:
// VALUBusy~100%, dominated by v_mul_lo/v_mul_hi/v_cndmask from these divides).
template <int C>
__device__ __forceinline__ int idiv(int x, int rt) {
  if constexpr (C != 0) return x / C; else return x / rt;
}
template <int C>
__device__ __forceinline__ int imod(int x, int rt) {
  if constexpr (C != 0) return x % C; else return x % rt;
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

__global__ void compute_pos_slot_kernel_hip(const int32_t* __restrict__ q_index,
                                        const int32_t* __restrict__ num_seqlen_per_req,
                                        const int32_t* __restrict__ kvcache_indices,
                                        int32_t* __restrict__ positions,
                                        int64_t* __restrict__ slot_indices,
                                        int32_t* __restrict__ req_ids,
                                        int32_t* __restrict__ local_idx,
                                        int64_t stride_kvi_r,
                                        int64_t stride_kvi_b,
                                        int64_t block_size) {
  const int req = blockIdx.x;
  const int32_t start = q_index[req];
  const int32_t end = q_index[req + 1];
  const int32_t seq_len = num_seqlen_per_req[req];
  const int32_t num_rows_req = end - start;

  if (seq_len <= 0 || num_rows_req <= 0) {
    return;
  }

  const int32_t pos_offset = seq_len - end;
  for (int32_t row_local = threadIdx.x; row_local < num_rows_req; row_local += blockDim.x) {
    const int32_t row = start + row_local;
    const int32_t token_pos = row + pos_offset;
    const int64_t block_idx = token_pos / block_size;
    const int32_t block_row = token_pos - block_idx * block_size;
    const int64_t phys_block =
        static_cast<int64_t>(kvcache_indices[req * stride_kvi_r + block_idx * stride_kvi_b]);
    positions[row] = token_pos;
    slot_indices[row] = phys_block * block_size + block_row;
    req_ids[row] = req;
    local_idx[row] = row_local;
  }
}

// One thread per row: resolve req -> (token_pos, packed slot) exactly once, so
// the heavy fused kernel's num_q_heads warps-per-row can just load these values
// instead of each re-running the req guess (two runtime 64-bit divisions) and
// the kvcache_indices lookup. slot = phys_block*block_size + block_row, or -1
// when the row maps to no valid KV slot. Cheap (a few ops/row) and fully
// parallel over rows, so it adds negligible time to prefill.
__global__ void compute_row_meta_kernel_hip(
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
    const int64_t block_idx = token_pos / block_size;
    const int32_t block_row = static_cast<int32_t>(token_pos - block_idx * block_size);
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
__global__ void rope_norm_store_kv_fp8_fused_kernel_hip(
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
    const int64_t* __restrict__ row_slot_meta,
    int64_t max_pos) {
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
    // compute_row_meta_kernel_hip. This removes the per-warp req lookup -- including
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
      phys_block = (block_size > 0) ? (slot / block_size) : 0;
      block_row = static_cast<int32_t>(slot - phys_block * block_size);
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

  float q_vals[4];
  float local_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const int64_t off = row * stride_qkv_t + (hq * kHeadDim + d) * es_qkv_d;
    const float x = static_cast<float>(qkv[off]);
    q_vals[i] = x;
    local_sq += x * x;
  }
  const float q_rms = sqrtf(warp_sum(local_sq) / static_cast<float>(kHeadDim) + eps);
  const float q_rms_inv = 1.0f / q_rms;

  // Defer the uniform 1/rms factor: it is linear through RoPE + Hadamard, so it
  // cancels in the quantized output and only re-enters the stored scale. This
  // removes a per-element divide from the hot path.
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const float w = load_scalar_f32(q_norm_weight, weight_kind, d);
    q_vals[i] = q_vals[i] * w;
  }

  // Defensive clamp (parity with the Triton OOB fix 8bb3ead9): a malformed or
  // padded row could carry token_pos >= cos table length. Reading row 0
  // (cos=1, sin=0) makes rope a no-op and avoids an out-of-bounds cos/sin read.
  const int32_t cos_pos = (token_pos >= 0 && token_pos < max_pos) ? token_pos : 0;
  float rope_cos[4];
  float rope_sin[4];
  float q_hadamard[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    // NeoX rotate-half partner is x[d +/- 64], i.e. register i+/-2 in the same
    // lane. d = lane + i*32 with lane in [0,31] => (d<64) == (i<2) is a
    // compile-time fact for each unrolled i. Spelling it as (i<2) keeps the
    // partner index a constant so the compiler avoids a dynamic-VGPR select
    // tree (the source of ~170 v_cndmask in the old build).
    const float rotated = (i < 2) ? -q_vals[i + 2] : q_vals[i - 2];
    const int d_mod = d & 63;
    const int64_t cos_off = cos_pos * stride_cos_t + d_mod * es_cos_d;
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
  const float q_scale = fmaxf(q_amax / fp8_max, 1.0e-12f);
  if (lane == 0) {
    q_scale_out[row * stride_qs_t + hq * stride_qs_h] = q_scale;
  }
  const float q_inv_scale = q_rms_had / q_scale;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const int64_t off = row * stride_out_q_t + hq * stride_out_q_h + d * es_oq_d;
    out_q[off] = float_to_fp8_byte(q_hadamard[i] * q_inv_scale, fp8_max);
  }

  if (hq >= num_kv_heads) {
    return;
  }

  const int64_t q_dim = num_q_heads * kHeadDim;
  const int64_t k_off_base = q_dim + hq * kHeadDim;
  float k_vals[4];
  local_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const int64_t off = row * stride_qkv_t + (k_off_base + d) * es_qkv_d;
    const float x = static_cast<float>(qkv[off]);
    k_vals[i] = x;
    local_sq += x * x;
  }
  const float k_rms = sqrtf(warp_sum(local_sq) / static_cast<float>(kHeadDim) + eps);
  const float k_rms_inv = 1.0f / k_rms;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
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
  const float k_scale_dyn = fmaxf(k_amax / fp8_max, 1.0e-12f);
  const float k_inv_scale = k_rms_had / k_scale_dyn;
  if (lane == 0 && valid_row) {
    const int r_idx = idiv<KSL>(block_row, static_cast<int>(k_scale_l));
    const int l_idx = imod<KSL>(block_row, static_cast<int>(k_scale_l));
    const int64_t ks_off =
        phys_block * stride_ks_b + r_idx * stride_ks_r + hq * stride_ks_h + l_idx * stride_ks_l;
    k_scale[ks_off] = k_scale_dyn;
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    if (valid_row) {
      const int k_group = idiv<KCX>(d, static_cast<int>(k_cache_x));
      const int k_x = imod<KCX>(d, static_cast<int>(k_cache_x));
      const int64_t kc_off = phys_block * stride_kc_b + hq * stride_kc_h +
                             k_group * stride_kc_g + block_row * stride_kc_t +
                             k_x * es_kc_x;
      key_cache[kc_off] = float_to_fp8_byte(k_hadamard[i] * k_inv_scale, fp8_max);
    }
  }

  const int64_t k_dim = num_kv_heads * kHeadDim;
  const int64_t v_off_base = q_dim + k_dim + hq * v_head_dim;
  const float v_scale_inv = 1.0f / v_scale[hq];
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
    const int64_t block_idx = m.token_pos / block_size;
    m.block_row = static_cast<int32_t>(m.token_pos - block_idx * block_size);
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
template <bool CosSinBF16, bool WeightBF16, bool HadamardBF16, bool DecodeOneToken, int HPW,
          int BSIZE = 0, int KCX = 0, int KSL = 0, bool CONTIG = false, bool USE_META = false>
__global__ void rope_norm_store_kv_fp8_fused_tiled_kernel_hip(
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
    const int64_t* __restrict__ row_slot_meta,
    int64_t max_pos) {
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
    // compute_row_meta_kernel_hip -> the head-group warps just load them.
    meta.token_pos = row_token_pos[row];
    const int64_t slot = row_slot_meta[row];
    meta.valid = slot >= 0;
    if constexpr (BSIZE != 0) {
      constexpr int kBsLog = __builtin_ctz(BSIZE);
      meta.phys_block = slot >> kBsLog;
      meta.block_row = static_cast<int32_t>(slot & (BSIZE - 1));
    } else {
      meta.phys_block = (block_size > 0) ? (slot / block_size) : 0;
      meta.block_row = static_cast<int32_t>(slot - meta.phys_block * block_size);
    }
  } else {
    meta = compute_row_meta<DecodeOneToken>(
        row, q_index, num_seqlen_per_req, kvcache_indices, num_req, num_rows,
        total_num_kv_cache_tokens, stride_kvi_r, stride_kvi_b, block_size);
  }
  // Defensive clamp (parity with the Triton OOB fix 8bb3ead9): out-of-range
  // token_pos reads cos row 0 (cos=1, sin=0) -> rope no-op, never OOB.
  const int32_t token_pos =
      (meta.token_pos >= 0 && meta.token_pos < max_pos) ? meta.token_pos : 0;

  const int cos_kind = CosSinBF16 ? kBFloat16 : kFloat32;
  const int weight_kind = WeightBF16 ? kBFloat16 : kFloat32;
  const float had_scale = hadamard_scale_128();
  (void)hadamard;

  const int64_t es_qkv_d = CONTIG ? 1 : stride_qkv_d;
  const int64_t es_cos_d = CONTIG ? 1 : stride_cos_d;
  const int64_t es_oq_d = CONTIG ? 1 : stride_out_q_d;
  const int64_t es_kc_x = CONTIG ? 1 : stride_kc_x;
  const int64_t es_vc_t = CONTIG ? 1 : stride_vc_t;

  // cos/sin + q norm weight are identical for every head of this row.
  float rope_cos[4], rope_sin[4], qw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const int dm = d & 63;
    const int64_t coff = token_pos * stride_cos_t + dm * es_cos_d;
    rope_cos[i] = load_scalar_f32(cos_sin, cos_kind, coff);
    rope_sin[i] = load_scalar_f32(cos_sin, cos_kind, coff + 64 * es_cos_d);
    qw[i] = load_scalar_f32(q_norm_weight, weight_kind, d);
  }

  // ===== Q (HPW heads interleaved) =====
  float qv[HPW][4];
  float qsq[HPW];
#pragma unroll
  for (int h = 0; h < HPW; ++h) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      const float x = static_cast<float>(
          qkv[row * stride_qkv_t + ((hq_base + h) * kHeadDim + d) * es_qkv_d]);
      qv[h][i] = x;
      s += x * x;
    }
    qsq[h] = s;
  }
  warp_sum_multi<HPW>(qsq);
  float q_rms_inv[HPW];
#pragma unroll
  for (int h = 0; h < HPW; ++h) q_rms_inv[h] = 1.0f / sqrtf(qsq[h] / static_cast<float>(kHeadDim) + eps);

  float qh[HPW][4];
#pragma unroll
  for (int h = 0; h < HPW; ++h) {
#pragma unroll
    for (int i = 0; i < 4; ++i) qv[h][i] *= qw[i];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float rot = (i < 2) ? -qv[h][i + 2] : qv[h][i - 2];
      // had_scale deferred into the per-head scale below.
      qh[h][i] = qv[h][i] * rope_cos[i] + rot * rope_sin[i];
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
    const float q_scale = fmaxf(q_amax / fp8_max, 1.0e-12f);
    if (lane == 0) q_scale_out[row * stride_qs_t + hq * stride_qs_h] = q_scale;
    const float q_inv = q_rms_had / q_scale;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      out_q[row * stride_out_q_t + hq * stride_out_q_h + d * es_oq_d] =
          float_to_fp8_byte(qh[h][i] * q_inv, fp8_max);
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
  for (int i = 0; i < 4; ++i) kw[i] = load_scalar_f32(k_norm_weight, weight_kind, lane + i * kWarpSize);

#pragma unroll
  for (int h = 0; h < HPW; ++h) {
    const int64_t hq = hq_base + h;
    if (hq >= num_kv_heads) {
      continue;
    }
    float kv[4];
    float ksq = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      const float x = static_cast<float>(
          qkv[row * stride_qkv_t + (q_dim + hq * kHeadDim + d) * es_qkv_d]);
      kv[i] = x;
      ksq += x * x;
    }
    const float k_rms_inv = 1.0f / sqrtf(warp_sum(ksq) / static_cast<float>(kHeadDim) + eps);
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
    const float k_scale_dyn = fmaxf(k_amax / fp8_max, 1.0e-12f);
    const float k_inv = k_rms_had / k_scale_dyn;
    if (lane == 0 && meta.valid) {
      const int r_idx = idiv<KSL>(meta.block_row, static_cast<int>(k_scale_l));
      const int l_idx = imod<KSL>(meta.block_row, static_cast<int>(k_scale_l));
      k_scale[meta.phys_block * stride_ks_b + r_idx * stride_ks_r + hq * stride_ks_h +
              l_idx * stride_ks_l] = k_scale_dyn;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      if (meta.valid) {
        const int kg = idiv<KCX>(d, static_cast<int>(k_cache_x));
        const int kx = imod<KCX>(d, static_cast<int>(k_cache_x));
        key_cache[meta.phys_block * stride_kc_b + hq * stride_kc_h + kg * stride_kc_g +
                  meta.block_row * stride_kc_t + kx * es_kc_x] =
            float_to_fp8_byte(kh[i] * k_inv, fp8_max);
      }
    }
    // V (no Hadamard)
    const int64_t v_off_base = q_dim + k_dim + hq * v_head_dim;
    const float v_scale_inv = 1.0f / v_scale[hq];
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

// ===================== MFMA Hadamard variant =====================
// Block = 512 threads (16 warps of 32 = 8 wavefronts of 64); processes a
// BLOCK_T=16 row tile of ONE q-head. Phase split:
//   (1) per-row (warp32) RMSNorm + RoPE -> bf16 tile in LDS
//   (2) 8 wavefronts do the [16,128]@[128,128] Hadamard via MFMA (1 N-tile/wave)
//   (3) per-row (warp32) amax + FP8 quant + store
// RoPE/norm/quant logic mirrors the shuffle kernel; only the Hadamard moves to
// the matrix cores. RMSNorm 1/rms is deferred into the final scale (same trick).
template <bool CosSinBF16, bool WeightBF16, bool HadamardBF16, bool DecodeOneToken>
__global__ __launch_bounds__(512) void rope_norm_store_kv_fp8_fused_mfma_kernel_hip(
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
    int64_t stride_qkv_t, int64_t stride_qkv_d,
    int64_t stride_cos_t, int64_t stride_cos_d,
    int64_t stride_kvi_r, int64_t stride_kvi_b,
    int64_t stride_out_q_t, int64_t stride_out_q_h, int64_t stride_out_q_d,
    int64_t stride_kc_b, int64_t stride_kc_h, int64_t stride_kc_g, int64_t stride_kc_t, int64_t stride_kc_x,
    int64_t stride_vc_b, int64_t stride_vc_h, int64_t stride_vc_d, int64_t stride_vc_t,
    int64_t stride_ks_b, int64_t stride_ks_r, int64_t stride_ks_h, int64_t stride_ks_l,
    int64_t stride_qs_t, int64_t stride_qs_h,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t v_head_dim,
    int64_t block_size, int64_t k_scale_l, int64_t k_cache_x) {
  using namespace rocwmma;
  constexpr int BT = 16;
  const int warp32 = threadIdx.x >> 5;   // 0..15 -> row in tile
  const int lane = threadIdx.x & 31;
  const int64_t hq = blockIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * BT + warp32;
  const bool row_in_range = row < num_rows;

  __shared__ __hip_bfloat16 qS[BT * kHeadDim];   // RoPE'd tile (reused for K)
  __shared__ float qO[BT * kHeadDim];            // Hadamard output (reused for K)
  __shared__ __hip_bfloat16 Hs[kHeadDim * kHeadDim];
  __shared__ float rms_inv_s[BT];
  __shared__ int64_t phys_s[BT];
  __shared__ int32_t brow_s[BT];
  __shared__ int32_t valid_s[BT];

  for (int i = threadIdx.x; i < kHeadDim * kHeadDim; i += blockDim.x)
    Hs[i] = reinterpret_cast<const __hip_bfloat16*>(hadamard)[i];

  const int cos_kind = CosSinBF16 ? kBFloat16 : kFloat32;
  const int weight_kind = WeightBF16 ? kBFloat16 : kFloat32;

  // ---- per-row metadata (every lane of the warp computes the same values) ----
  int32_t token_pos = 0;
  int64_t phys_block = 0;
  int32_t block_row = 0;
  bool valid_row = false;
  if (row_in_range) {
    int64_t req;
    int32_t req_start, req_end, seq_len;
    if constexpr (DecodeOneToken) {
      req = row;
      seq_len = num_seqlen_per_req[req];
      token_pos = (seq_len > 0) ? (seq_len - 1) : 0;
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
      token_pos = (seq_len > 0) ? static_cast<int32_t>(row + seq_len - req_end) : 0;
    }
    if (seq_len > 0 && req_end > req_start) {
      const int64_t block_idx = token_pos / block_size;
      block_row = static_cast<int32_t>(token_pos - block_idx * block_size);
      phys_block = static_cast<int64_t>(kvcache_indices[req * stride_kvi_r + block_idx * stride_kvi_b]);
      const int64_t slot = phys_block * block_size + block_row;
      valid_row = slot >= 0 && slot < total_num_kv_cache_tokens;
    }
  }
  if (lane == 0) {
    phys_s[warp32] = phys_block;
    brow_s[warp32] = block_row;
    valid_s[warp32] = valid_row ? 1 : 0;
  }

  // ===================== Q =====================
  float vals[4];
  float local_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    float x = 0.0f;
    if (row_in_range) x = static_cast<float>(qkv[row * stride_qkv_t + (hq * kHeadDim + d) * stride_qkv_d]);
    vals[i] = x;
    local_sq += x * x;
  }
  const float q_rms_inv = 1.0f / sqrtf(warp_sum(local_sq) / static_cast<float>(kHeadDim) + eps);
  if (lane == 0) rms_inv_s[warp32] = q_rms_inv;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    vals[i] *= load_scalar_f32(q_norm_weight, weight_kind, d);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const float rot = (i < 2) ? -vals[i + 2] : vals[i - 2];
    const int dm = d & 63;
    const int64_t coff = token_pos * stride_cos_t + dm * stride_cos_d;
    const float c = load_scalar_f32(cos_sin, cos_kind, coff);
    const float s = load_scalar_f32(cos_sin, cos_kind, coff + 64 * stride_cos_d);
    qS[warp32 * kHeadDim + d] = static_cast<__hip_bfloat16>(vals[i] * c + rot * s);
  }
  __syncthreads();
  {
    const int wv = threadIdx.x >> 6;  // 0..7 -> output N-tile
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);
    const bfloat16_t* A = reinterpret_cast<const bfloat16_t*>(qS);
    const bfloat16_t* B = reinterpret_cast<const bfloat16_t*>(Hs);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major> a;
      fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major> b;
      load_matrix_sync(a, A + k * 16, kHeadDim);
      load_matrix_sync(b, B + (k * 16) * kHeadDim + wv * 16, kHeadDim);
      mma_sync(acc, a, b, acc);
    }
    store_matrix_sync(qO + wv * 16, acc, kHeadDim, mem_row_major);
  }
  __syncthreads();
  if (row_in_range) {
    float h[4];
    float q_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      h[i] = qO[warp32 * kHeadDim + d];
      q_abs = fmaxf(q_abs, fabsf(h[i]));
    }
    const float q_amax = warp_max(q_abs) * q_rms_inv;
    const float q_scale = fmaxf(q_amax / fp8_max, 1.0e-12f);
    if (lane == 0) q_scale_out[row * stride_qs_t + hq * stride_qs_h] = q_scale;
    const float q_inv = q_rms_inv / q_scale;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      const int64_t off = row * stride_out_q_t + hq * stride_out_q_h + d * stride_out_q_d;
      out_q[off] = float_to_fp8_byte(h[i] * q_inv, fp8_max);
    }
  }

  if (hq >= num_kv_heads) return;
  __syncthreads();  // reuse qS/qO for K

  // ===================== K =====================
  const int64_t q_dim = num_q_heads * kHeadDim;
  const int64_t k_off_base = q_dim + hq * kHeadDim;
  local_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    float x = 0.0f;
    if (row_in_range) x = static_cast<float>(qkv[row * stride_qkv_t + (k_off_base + d) * stride_qkv_d]);
    vals[i] = x;
    local_sq += x * x;
  }
  const float k_rms_inv = 1.0f / sqrtf(warp_sum(local_sq) / static_cast<float>(kHeadDim) + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    vals[i] *= load_scalar_f32(k_norm_weight, weight_kind, d);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    const float rot = (i < 2) ? -vals[i + 2] : vals[i - 2];
    const int dm = d & 63;
    const int64_t coff = token_pos * stride_cos_t + dm * stride_cos_d;
    const float c = load_scalar_f32(cos_sin, cos_kind, coff);
    const float s = load_scalar_f32(cos_sin, cos_kind, coff + 64 * stride_cos_d);
    qS[warp32 * kHeadDim + d] = static_cast<__hip_bfloat16>(vals[i] * c + rot * s);
  }
  __syncthreads();
  {
    const int wv = threadIdx.x >> 6;
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);
    const bfloat16_t* A = reinterpret_cast<const bfloat16_t*>(qS);
    const bfloat16_t* B = reinterpret_cast<const bfloat16_t*>(Hs);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major> a;
      fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major> b;
      load_matrix_sync(a, A + k * 16, kHeadDim);
      load_matrix_sync(b, B + (k * 16) * kHeadDim + wv * 16, kHeadDim);
      mma_sync(acc, a, b, acc);
    }
    store_matrix_sync(qO + wv * 16, acc, kHeadDim, mem_row_major);
  }
  __syncthreads();
  if (row_in_range) {
    const bool valid = valid_s[warp32] != 0;
    const int64_t pb = phys_s[warp32];
    const int32_t br = brow_s[warp32];
    float h[4];
    float k_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      h[i] = qO[warp32 * kHeadDim + d];
      k_abs = fmaxf(k_abs, fabsf(h[i]));
    }
    const float k_amax = warp_max(k_abs) * k_rms_inv;
    const float k_scale_dyn = fmaxf(k_amax / fp8_max, 1.0e-12f);
    const float k_inv = k_rms_inv / k_scale_dyn;
    if (lane == 0 && valid) {
      const int64_t r_idx = br / k_scale_l;
      const int64_t l_idx = br - r_idx * k_scale_l;
      k_scale[pb * stride_ks_b + r_idx * stride_ks_r + hq * stride_ks_h + l_idx * stride_ks_l] = k_scale_dyn;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      if (valid) {
        const int64_t kg = d / k_cache_x;
        const int64_t kx = d - kg * k_cache_x;
        const int64_t kc = pb * stride_kc_b + hq * stride_kc_h + kg * stride_kc_g +
                           br * stride_kc_t + kx * stride_kc_x;
        key_cache[kc] = float_to_fp8_byte(h[i] * k_inv, fp8_max);
      }
    }
  }

  // ===================== V (no Hadamard) =====================
  if (row_in_range) {
    const bool valid = valid_s[warp32] != 0;
    const int64_t pb = phys_s[warp32];
    const int32_t br = brow_s[warp32];
    const int64_t k_dim = num_kv_heads * kHeadDim;
    const int64_t v_off_base = q_dim + k_dim + hq * v_head_dim;
    const float v_scale_inv = 1.0f / v_scale[hq];
    const int v_iters = static_cast<int>((v_head_dim + kWarpSize - 1) / kWarpSize);
    for (int i = 0; i < v_iters; ++i) {
      const int d = lane + i * kWarpSize;
      if (d < v_head_dim && valid) {
        const float v = static_cast<float>(qkv[row * stride_qkv_t + (v_off_base + d) * stride_qkv_d]) * v_scale_inv;
        const int64_t vc = pb * stride_vc_b + hq * stride_vc_h + d * stride_vc_d + br * stride_vc_t;
        value_cache[vc] = float_to_fp8_byte(v, fp8_max);
      }
    }
  }
}

// ===================== MFMA Hadamard variant v2 =====================
// Fixes the v1 inefficiencies (M=16 cooperative GEMM with 3 block syncs per 16
// rows + per-block Hs LDS reload):
//   - each wave is fully independent and owns RPW=16 rows (its own MFMA M-tile),
//     so there is no inter-wave cooperation on a tile;
//   - a block packs WPB waves => WPB*16 rows, amortizing setup;
//   - H is read straight from global (32KB, L2-resident, reused by every block)
//     so no per-block Hs staging;
//   - RoPE uses 4 lanes/row (quad) so the RMSNorm / q-amax reductions are pure
//     DPP quad_perm.
// Math mirrors the v1 kernel exactly (RMS variance on raw qkv, weight then RoPE,
// rms_inv folded into the final dynamic quant scale).
#if defined(MFMA_V2)
__device__ __forceinline__ float quad_reduce_sum(float v) {
  v += swz_xor<1>(v);
  v += swz_xor<2>(v);
  return v;
}
__device__ __forceinline__ float quad_reduce_max(float v) {
  v = fmaxf(v, swz_xor<1>(v));
  v = fmaxf(v, swz_xor<2>(v));
  return v;
}

template <bool CosSinBF16, bool WeightBF16, bool HadamardBF16, bool DecodeOneToken>
__global__ __launch_bounds__(256) void rope_norm_store_kv_fp8_fused_mfma2_kernel_hip(
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
    int64_t stride_qkv_t, int64_t stride_qkv_d,
    int64_t stride_cos_t, int64_t stride_cos_d,
    int64_t stride_kvi_r, int64_t stride_kvi_b,
    int64_t stride_out_q_t, int64_t stride_out_q_h, int64_t stride_out_q_d,
    int64_t stride_kc_b, int64_t stride_kc_h, int64_t stride_kc_g, int64_t stride_kc_t, int64_t stride_kc_x,
    int64_t stride_vc_b, int64_t stride_vc_h, int64_t stride_vc_d, int64_t stride_vc_t,
    int64_t stride_ks_b, int64_t stride_ks_r, int64_t stride_ks_h, int64_t stride_ks_l,
    int64_t stride_qs_t, int64_t stride_qs_h,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t v_head_dim,
    int64_t block_size, int64_t k_scale_l, int64_t k_cache_x) {
  using namespace rocwmma;
  constexpr int RPW = 16;            // rows per wave (= MFMA M tile)
#ifndef MFMA2_WPB
#define MFMA2_WPB 4
#endif
  constexpr int WPB = MFMA2_WPB;     // waves per block
  const int wave = threadIdx.x >> 6;       // 0..WPB-1
  const int lane = threadIdx.x & 63;       // 0..63
  const int rowInWave = lane >> 2;         // 0..15
  const int sub = lane & 3;                // 0..3 (4 lanes/row)
  const int64_t hq = blockIdx.y;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * (WPB * RPW) + wave * RPW + rowInWave;
  const bool row_in_range = row < num_rows;

  __shared__ __hip_bfloat16 qS[WPB * RPW * kHeadDim];  // RoPE'd tile (reused K)
  __shared__ float qO[WPB * RPW * kHeadDim];           // Hadamard out (reused K)

  const int cos_kind = CosSinBF16 ? kBFloat16 : kFloat32;
  const int weight_kind = WeightBF16 ? kBFloat16 : kFloat32;

  // ---- per-row metadata (redundant across the 4 lanes of a row) ----
  int32_t token_pos = 0;
  int64_t phys_block = 0;
  int32_t block_row = 0;
  bool valid_row = false;
  if (row_in_range) {
    int64_t req;
    int32_t req_start, req_end, seq_len;
    if constexpr (DecodeOneToken) {
      req = row;
      seq_len = num_seqlen_per_req[req];
      token_pos = (seq_len > 0) ? (seq_len - 1) : 0;
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
      token_pos = (seq_len > 0) ? static_cast<int32_t>(row + seq_len - req_end) : 0;
    }
    if (seq_len > 0 && req_end > req_start) {
      const int64_t block_idx = token_pos / block_size;
      block_row = static_cast<int32_t>(token_pos - block_idx * block_size);
      phys_block = static_cast<int64_t>(kvcache_indices[req * stride_kvi_r + block_idx * stride_kvi_b]);
      const int64_t slot = phys_block * block_size + block_row;
      valid_row = slot >= 0 && slot < total_num_kv_cache_tokens;
    }
  }

  const bfloat16_t* Hg = reinterpret_cast<const bfloat16_t*>(hadamard);
  const int64_t q_dim = num_q_heads * kHeadDim;

  // ============ helper lambda: one head's RoPE -> qS, MFMA -> qO ============
  // Returns rms_inv for the row. weight_ptr selects q/k norm weight; in_base is
  // the qkv column offset of this head.
  auto rope_to_qS = [&](const void* weight_ptr, int64_t in_base) -> float {
    float xl[16], xh[16];
    float local_sq = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int dl = sub * 16 + j;        // 0..63
      const int dh = 64 + sub * 16 + j;   // 64..127
      float a = 0.0f, b = 0.0f;
      if (row_in_range) {
        a = static_cast<float>(qkv[row * stride_qkv_t + (in_base + dl) * stride_qkv_d]);
        b = static_cast<float>(qkv[row * stride_qkv_t + (in_base + dh) * stride_qkv_d]);
      }
      xl[j] = a; xh[j] = b;
      local_sq += a * a + b * b;
    }
    const float sq = quad_reduce_sum(local_sq);
    const float rms_inv = 1.0f / sqrtf(sq / static_cast<float>(kHeadDim) + eps);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int dl = sub * 16 + j;
      const int dh = 64 + sub * 16 + j;
      const float wl = load_scalar_f32(weight_ptr, weight_kind, dl);
      const float wh = load_scalar_f32(weight_ptr, weight_kind, dh);
      float vl = xl[j] * wl;
      float vh = xh[j] * wh;
      const int64_t coff = token_pos * stride_cos_t + dl * stride_cos_d;  // dl<64
      const float c = load_scalar_f32(cos_sin, cos_kind, coff);
      const float s = load_scalar_f32(cos_sin, cos_kind, coff + 64 * stride_cos_d);
      const int base = wave * RPW * kHeadDim + rowInWave * kHeadDim;
      qS[base + dl] = static_cast<__hip_bfloat16>(vl * c - vh * s);
      qS[base + dh] = static_cast<__hip_bfloat16>(vh * c + vl * s);
    }
    return rms_inv;
  };

  // MFMA: this wave's 16 rows (qS region) @ H[128,128] -> qO region.
  auto hadamard_mfma = [&]() {
    const bfloat16_t* A = reinterpret_cast<const bfloat16_t*>(qS) + wave * RPW * kHeadDim;
    float* O = qO + wave * RPW * kHeadDim;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      fragment<accumulator, 16, 16, 16, float> acc;
      fill_fragment(acc, 0.0f);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major> a;
        fragment<matrix_b, 16, 16, 16, bfloat16_t, row_major> b;
        load_matrix_sync(a, A + k * 16, kHeadDim);
        load_matrix_sync(b, Hg + (k * 16) * kHeadDim + n * 16, kHeadDim);
        mma_sync(acc, a, b, acc);
      }
      store_matrix_sync(O + n * 16, acc, kHeadDim, mem_row_major);
    }
  };

  // ===================== Q =====================
  const float q_rms_inv = rope_to_qS(q_norm_weight, hq * kHeadDim);
  __syncthreads();
  hadamard_mfma();
  __syncthreads();
  if (row_in_range) {
    const int base = wave * RPW * kHeadDim + rowInWave * kHeadDim;
    float h[32];
    float q_abs = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      h[j] = qO[base + sub * 16 + j];
      h[16 + j] = qO[base + 64 + sub * 16 + j];
      q_abs = fmaxf(q_abs, fmaxf(fabsf(h[j]), fabsf(h[16 + j])));
    }
    const float q_amax = quad_reduce_max(q_abs) * q_rms_inv;
    const float q_scale = fmaxf(q_amax / fp8_max, 1.0e-12f);
    if (sub == 0) q_scale_out[row * stride_qs_t + hq * stride_qs_h] = q_scale;
    const float q_inv = q_rms_inv / q_scale;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int dl = sub * 16 + j;
      const int dh = 64 + sub * 16 + j;
      out_q[row * stride_out_q_t + hq * stride_out_q_h + dl * stride_out_q_d] =
          float_to_fp8_byte(h[j] * q_inv, fp8_max);
      out_q[row * stride_out_q_t + hq * stride_out_q_h + dh * stride_out_q_d] =
          float_to_fp8_byte(h[16 + j] * q_inv, fp8_max);
    }
  }

  if (hq >= num_kv_heads) return;

  // ===================== K =====================
  __syncthreads();
  const float k_rms_inv = rope_to_qS(k_norm_weight, q_dim + hq * kHeadDim);
  __syncthreads();
  hadamard_mfma();
  __syncthreads();
  if (row_in_range && valid_row) {
    const int base = wave * RPW * kHeadDim + rowInWave * kHeadDim;
    float h[32];
    float k_abs = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      h[j] = qO[base + sub * 16 + j];
      h[16 + j] = qO[base + 64 + sub * 16 + j];
      k_abs = fmaxf(k_abs, fmaxf(fabsf(h[j]), fabsf(h[16 + j])));
    }
    const float k_amax = quad_reduce_max(k_abs) * k_rms_inv;
    const float k_scale_dyn = fmaxf(k_amax / fp8_max, 1.0e-12f);
    const float k_inv = k_rms_inv / k_scale_dyn;
    if (sub == 0) {
      const int64_t r_idx = block_row / k_scale_l;
      const int64_t l_idx = block_row - r_idx * k_scale_l;
      k_scale[phys_block * stride_ks_b + r_idx * stride_ks_r + hq * stride_ks_h +
              l_idx * stride_ks_l] = k_scale_dyn;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int dl = sub * 16 + j;
      const int dh = 64 + sub * 16 + j;
#pragma unroll
      for (int two = 0; two < 2; ++two) {
        const int d = two == 0 ? dl : dh;
        const float val = two == 0 ? h[j] : h[16 + j];
        const int64_t kg = d / k_cache_x;
        const int64_t kx = d - kg * k_cache_x;
        key_cache[phys_block * stride_kc_b + hq * stride_kc_h + kg * stride_kc_g +
                  block_row * stride_kc_t + kx * stride_kc_x] =
            float_to_fp8_byte(val * k_inv, fp8_max);
      }
    }
  }

  // ===================== V (no Hadamard) =====================
  if (row_in_range && valid_row) {
    const int64_t k_dim = num_kv_heads * kHeadDim;
    const int64_t v_off_base = q_dim + k_dim + hq * v_head_dim;
    const float v_scale_inv = 1.0f / v_scale[hq];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
      for (int two = 0; two < 2; ++two) {
        const int d = two == 0 ? (sub * 16 + j) : (64 + sub * 16 + j);
        if (d < v_head_dim) {
          const float v = static_cast<float>(
              qkv[row * stride_qkv_t + (v_off_base + d) * stride_qkv_d]) * v_scale_inv;
          value_cache[phys_block * stride_vc_b + hq * stride_vc_h + d * stride_vc_d +
                      block_row * stride_vc_t] = float_to_fp8_byte(v, fp8_max);
        }
      }
    }
  }
}
#endif  // MFMA_V2

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
  const int64_t max_pos = cos_sin.size(0);
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
    hipLaunchKernelGGL(compute_row_meta_kernel_hip, meta_grid, dim3(meta_threads), 0, stream,
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
      k_cache_x, row_token_pos_ptr, row_slot_ptr, max_pos
#define LAUNCH_DECODE(CB, DEC, BS, KCX, KSL, CTG, META)                                            \
  hipLaunchKernelGGL(                                                                              \
      (rope_norm_store_kv_fp8_fused_kernel_hip<CB, WeightBF16, HadamardBF16, DEC, BS, KCX, KSL, CTG, META>), \
      grid, block, smem, stream, DECODE_ARGS)
  // Standard contiguous vLLM layout -> innermost strides are 1.
  const bool contig = qkv.stride(1) == 1 && cos_sin.stride(1) == 1 && out_q.stride(2) == 1 &&
                      key_cache.stride(4) == 1 && value_cache.stride(3) == 1;
  // Specialize the common (block_size=32, k_cache_x=16, k_scale_l=32) + fully
  // contiguous layout so addressing div/mod fold to shift/and and the inner
  // stride multiplies vanish; fall back to the fully-runtime instantiation
  // for any other shape/layout. META mirrors use_meta (prefill = !decode).
#define LAUNCH_DECODE_DIV(CB, DEC, META)                                                            \
  do {                                                                                             \
    if (block_size == 32 && k_cache_x == 16 && k_scale_l == 32 && contig)                          \
      LAUNCH_DECODE(CB, DEC, 32, 16, 32, true, META);                                              \
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

template <bool WeightBF16, bool HadamardBF16>
void launch_cos_dispatch_mfma(const torch::Tensor& qkv,
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
#if defined(MFMA_V2)
#ifndef MFMA2_WPB
#define MFMA2_WPB 4
#endif
  const int rows_per_block = MFMA2_WPB * 16;
  const dim3 block(MFMA2_WPB * 64);
  const dim3 grid(static_cast<unsigned>((num_rows + rows_per_block - 1) / rows_per_block),
                  static_cast<unsigned>(num_q_heads));
#else
  const dim3 block(512);
  const dim3 grid(static_cast<unsigned>((num_rows + 15) / 16), static_cast<unsigned>(num_q_heads));
#endif
  const size_t smem = 0;
  const auto stream = c10::hip::getCurrentHIPStream();
#if defined(MFMA_V2)
#define MFMA_KERNEL rope_norm_store_kv_fp8_fused_mfma2_kernel_hip
#else
#define MFMA_KERNEL rope_norm_store_kv_fp8_fused_mfma_kernel_hip
#endif
#define MFMA_ARGS                                                                                  \
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
      q_scale_out.stride(1), num_q_heads, num_kv_heads, v_head_dim, block_size, k_scale_l, k_cache_x
  if (cos_sin.scalar_type() == at::kBFloat16) {
    if (decode_one_token)
      hipLaunchKernelGGL((MFMA_KERNEL<true, WeightBF16, HadamardBF16, true>),
                         grid, block, smem, stream, MFMA_ARGS);
    else
      hipLaunchKernelGGL((MFMA_KERNEL<true, WeightBF16, HadamardBF16, false>),
                         grid, block, smem, stream, MFMA_ARGS);
  } else {
    if (decode_one_token)
      hipLaunchKernelGGL((MFMA_KERNEL<false, WeightBF16, HadamardBF16, true>),
                         grid, block, smem, stream, MFMA_ARGS);
    else
      hipLaunchKernelGGL((MFMA_KERNEL<false, WeightBF16, HadamardBF16, false>),
                         grid, block, smem, stream, MFMA_ARGS);
  }
#undef MFMA_ARGS
#undef MFMA_KERNEL
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
  const int64_t max_pos = cos_sin.size(0);
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
    hipLaunchKernelGGL(compute_row_meta_kernel_hip, meta_grid, dim3(meta_threads), 0, stream,
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
      k_cache_x, row_token_pos_ptr, row_slot_ptr, max_pos
  const bool contig = qkv.stride(1) == 1 && cos_sin.stride(1) == 1 && out_q.stride(2) == 1 &&
                      key_cache.stride(4) == 1 && value_cache.stride(3) == 1;
#define LAUNCH_TILED_K(CB, DEC, BS, KCX, KSL, CTG, META)                                           \
  hipLaunchKernelGGL(                                                                              \
      (rope_norm_store_kv_fp8_fused_tiled_kernel_hip<CB, WeightBF16, HadamardBF16, DEC, HPW, BS, \
                                                        KCX, KSL, CTG, META>),                      \
      grid, block, smem, stream, TILED_ARGS)
#define LAUNCH_TILED_DIV(CB, DEC, META)                                                            \
  do {                                                                                             \
    if (block_size == 32 && k_cache_x == 16 && k_scale_l == 32 && contig)                          \
      LAUNCH_TILED_K(CB, DEC, 32, 16, 32, true, META);                                             \
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
                              bool use_mfma,
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
    if (use_mfma)
      launch_cos_dispatch_mfma<WeightBF16, true>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                                 kvcache_indices, q_norm_weight, k_norm_weight,
                                                 hadamard, k_scale, v_scale, out_q, key_cache,
                                                 value_cache, q_scale_out, eps, fp8_max,
                                                 decode_one_token);
    else if (tile_hpw > 1)
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

}  // namespace aiter_rope

using namespace aiter_rope;

void compute_pos_slot_hip(torch::Tensor q_index,
                          torch::Tensor num_seqlen_per_req,
                          torch::Tensor kvcache_indices,
                          torch::Tensor positions,
                          torch::Tensor slot_indices,
                          torch::Tensor req_ids,
                          torch::Tensor local_idx,
                          int64_t block_size) {
  CHECK_HIP_TENSOR(q_index);
  CHECK_HIP_TENSOR(num_seqlen_per_req);
  CHECK_HIP_TENSOR(kvcache_indices);
  CHECK_HIP_TENSOR(positions);
  CHECK_HIP_TENSOR(slot_indices);
  CHECK_HIP_TENSOR(req_ids);
  CHECK_HIP_TENSOR(local_idx);
  CHECK_DTYPE(q_index, at::kInt);
  CHECK_DTYPE(num_seqlen_per_req, at::kInt);
  CHECK_DTYPE(kvcache_indices, at::kInt);
  CHECK_DTYPE(positions, at::kInt);
  CHECK_DTYPE(slot_indices, at::kLong);
  CHECK_DTYPE(req_ids, at::kInt);
  CHECK_DTYPE(local_idx, at::kInt);

  const int64_t num_req = num_seqlen_per_req.size(0);
  const dim3 block(128);
  const dim3 grid(num_req);
  const auto stream = c10::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(compute_pos_slot_kernel_hip, grid, block, 0, stream,
                     q_index.data_ptr<int32_t>(), num_seqlen_per_req.data_ptr<int32_t>(),
                     kvcache_indices.data_ptr<int32_t>(), positions.data_ptr<int32_t>(),
                     slot_indices.data_ptr<int64_t>(), req_ids.data_ptr<int32_t>(),
                     local_idx.data_ptr<int32_t>(), kvcache_indices.stride(0),
                     kvcache_indices.stride(1), block_size);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

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
                                             bool use_mfma,
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

  // Instantiate the few dtype combinations needed by the current ROCm path.
  if (use_mfma) {
    TORCH_CHECK(hadamard.scalar_type() == at::kBFloat16,
                "use_mfma requires a bfloat16 hadamard matrix");
  }
  // The multi-head tiled kernel tiles Q by HPW; K/V is handled per-kv-head so
  // num_kv_heads need not be divisible by HPW (supports GQA, e.g. nkv==1). Only
  // num_q_heads must be divisible by the tile width.
  int64_t eff_tile_hpw = tile_hpw;
  if (eff_tile_hpw > 1) {
    const int64_t nqh = out_q.size(1);
    if (use_mfma || (nqh % eff_tile_hpw != 0)) {
      eff_tile_hpw = 1;
    }
  }
  if (q_norm_weight.scalar_type() == at::kBFloat16) {
    launch_hadamard_dispatch<true>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                   kvcache_indices, q_norm_weight, k_norm_weight, hadamard,
                                   k_scale, v_scale, out_q, key_cache, value_cache,
                                   q_scale_out, static_cast<float>(eps),
                                   static_cast<float>(fp8_max),
                                   assume_decode_one_token, use_mfma, eff_tile_hpw);
  } else {
    launch_hadamard_dispatch<false>(qkv, cos_sin, q_index, num_seqlen_per_req,
                                    kvcache_indices, q_norm_weight, k_norm_weight, hadamard,
                                    k_scale, v_scale, out_q, key_cache, value_cache,
                                    q_scale_out, static_cast<float>(eps),
                                    static_cast<float>(fp8_max),
                                    assume_decode_one_token, use_mfma, eff_tile_hpw);
  }
}

// SPDX-License-Identifier: MIT
// Standalone tiled per-head FP8 V quant, additive to /opt/aiter.
//
// Adds one public symbol:
//   aiter::v_2way_per_head_fp8_quant(at::Tensor& v0, at::Tensor& v1)
//   in  : v0/v1 bf16/half [B, T0/T1, H, D=128] contiguous
//   out : (v_fp8 e4m3fnuz [B, T, H, D], v_descale fp32 [B, H])
//
// Algorithm (3-pass for high CU occupancy on MI300):
//   A  amax_tiled  — per-(B, H, tile_of_T) max(|v|) → atomic_fmax_pos to [B, H]
//   B  finalize    — amax → amax/240
//   C  quant_tiled — per-(B, H, tile_of_T) v_fp8 = v / descale
//
// Verbatim port of aiter-main lines 2945-3019 + 4290-4349, with two tiny helpers
// inlined here (atomic_fmax_pos, KernelElementType) so we depend ONLY on
// /opt/aiter's existing headers — no new headers under csrc/include/.
//
// Source: /workspace/aiter-main/csrc/kernels/fused_qk_norm_rope_cache_quant.cu
//   lines 2945-3019  (device kernels)
//   lines 4290-4349  (host wrapper)

#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include "rope/rope_common.h"  // for mrope_utils::fp8e4m3fnuz

namespace aiter {

// ---------- inlined helpers (kept out of any header) ----------

template <typename T_at>
struct KernelElementType {
    using type = T_at;
};
template <>
struct KernelElementType<c10::BFloat16> {
    using type = __hip_bfloat16;
};
template <>
struct KernelElementType<c10::Half> {
    using type = _Float16;
};

// Atomic fmax for non-negative floats. Uses bitwise-int atomicMax,
// which works because IEEE 754 ordering matches integer ordering for >=0 floats.
__device__ __forceinline__ void atomic_fmax_pos(float* addr, float val)
{
    int* iaddr = reinterpret_cast<int*>(addr);
    int ival   = __float_as_int(val);
    atomicMax(iaddr, ival);
}

// ---------- Phase A: tiled amax ----------
template <typename T, int TILE_T, int HEAD_SIZE>
__global__ void __launch_bounds__(256) v_per_head_amax_tiled_kernel(
    const T* __restrict__ v_,
    int num_tokens,
    int num_heads,
    float* __restrict__ v_amax)
{
    constexpr int BT = 256;
    int b    = blockIdx.z;
    int h    = blockIdx.y;
    int tile = blockIdx.x;
    int t_start = tile * TILE_T;
    int t_end   = min(t_start + TILE_T, num_tokens);
    int slab_h_stride = num_heads * HEAD_SIZE;
    int64_t base = ((int64_t)b * num_tokens + t_start) * slab_h_stride + (int64_t)h * HEAD_SIZE;
    int tile_elems = (t_end - t_start) * HEAD_SIZE;
    float local    = 0.0f;
    for(int idx = threadIdx.x; idx < tile_elems; idx += BT)
    {
        int dt  = idx / HEAD_SIZE;
        int d   = idx % HEAD_SIZE;
        float v = (float)v_[base + (int64_t)dt * slab_h_stride + d];
        local   = fmaxf(local, fabsf(v));
    }
    __shared__ float sm[BT];
    sm[threadIdx.x] = local;
    __syncthreads();
#pragma unroll
    for(int s = BT / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if(threadIdx.x == 0) atomic_fmax_pos(v_amax + b * num_heads + h, sm[0]);
}

// ---------- Phase B: amax -> descale ----------
__global__ void v_amax_to_descale_kernel(const float* __restrict__ v_amax,
                                         int num_heads,
                                         float* __restrict__ v_descale)
{
    int b   = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_heads) return;
    constexpr float fp8_max = 240.0f;
    v_descale[b * num_heads + idx] =
        fmaxf(v_amax[b * num_heads + idx], 1e-8f) / fp8_max;
}

// ---------- Phase C: tiled quant ----------
template <typename T, int TILE_T, int HEAD_SIZE>
__global__ void __launch_bounds__(256) v_per_head_quant_tiled_kernel(
    const T* __restrict__ v_,
    int num_tokens,
    int num_heads,
    mrope_utils::fp8e4m3fnuz* __restrict__ v_fp8_,
    const float* __restrict__ v_descale)
{
    constexpr int BT = 256;
    int b    = blockIdx.z;
    int h    = blockIdx.y;
    int tile = blockIdx.x;
    int t_start = tile * TILE_T;
    int t_end   = min(t_start + TILE_T, num_tokens);
    int slab_h_stride = num_heads * HEAD_SIZE;
    int64_t base = ((int64_t)b * num_tokens + t_start) * slab_h_stride + (int64_t)h * HEAD_SIZE;
    int tile_elems = (t_end - t_start) * HEAD_SIZE;
    float inv = 1.0f / v_descale[b * num_heads + h];
    for(int idx = threadIdx.x; idx < tile_elems; idx += BT)
    {
        int dt  = idx / HEAD_SIZE;
        int d   = idx % HEAD_SIZE;
        int64_t off = base + (int64_t)dt * slab_h_stride + d;
        v_fp8_[off] = mrope_utils::fp8e4m3fnuz((float)v_[off] * inv);
    }
}

template <typename T, int TILE_T, int HEAD_SIZE>
__global__ void __launch_bounds__(256) v_2way_per_head_amax_tiled_kernel(
    const T* __restrict__ v0_,
    const T* __restrict__ v1_,
    int num_tokens0,
    int num_tokens1,
    int num_heads,
    float* __restrict__ v_amax)
{
    constexpr int BT = 256;
    int b    = blockIdx.z;
    int h    = blockIdx.y;
    int tile = blockIdx.x;
    int total_tokens = num_tokens0 + num_tokens1;
    int t_start = tile * TILE_T;
    int t_end   = min(t_start + TILE_T, total_tokens);
    int slab_h_stride = num_heads * HEAD_SIZE;
    float local = 0.0f;

    for(int idx = threadIdx.x; idx < (t_end - t_start) * HEAD_SIZE; idx += BT)
    {
        int local_t = idx / HEAD_SIZE;
        int d       = idx % HEAD_SIZE;
        int t       = t_start + local_t;
        int64_t off;
        float val;
        if(t < num_tokens0)
        {
            off = ((int64_t)b * num_tokens0 + t) * slab_h_stride +
                  (int64_t)h * HEAD_SIZE + d;
            val = (float)v0_[off];
        }
        else
        {
            int t1 = t - num_tokens0;
            off = ((int64_t)b * num_tokens1 + t1) * slab_h_stride +
                  (int64_t)h * HEAD_SIZE + d;
            val = (float)v1_[off];
        }
        local = fmaxf(local, fabsf(val));
    }

    __shared__ float sm[BT];
    sm[threadIdx.x] = local;
    __syncthreads();
#pragma unroll
    for(int s = BT / 2; s > 0; s >>= 1)
    {
        if(threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if(threadIdx.x == 0) atomic_fmax_pos(v_amax + b * num_heads + h, sm[0]);
}

template <typename T, int TILE_T, int HEAD_SIZE>
__global__ void __launch_bounds__(256) v_2way_per_head_quant_tiled_kernel(
    const T* __restrict__ v0_,
    const T* __restrict__ v1_,
    int num_tokens0,
    int num_tokens1,
    int num_heads,
    mrope_utils::fp8e4m3fnuz* __restrict__ v_fp8_,
    const float* __restrict__ v_descale)
{
    constexpr int BT = 256;
    int b    = blockIdx.z;
    int h    = blockIdx.y;
    int tile = blockIdx.x;
    int total_tokens = num_tokens0 + num_tokens1;
    int t_start = tile * TILE_T;
    int t_end   = min(t_start + TILE_T, total_tokens);
    int slab_h_stride = num_heads * HEAD_SIZE;
    float inv = 1.0f / v_descale[b * num_heads + h];

    for(int idx = threadIdx.x; idx < (t_end - t_start) * HEAD_SIZE; idx += BT)
    {
        int local_t = idx / HEAD_SIZE;
        int d       = idx % HEAD_SIZE;
        int t       = t_start + local_t;
        int64_t in_off;
        int64_t out_off = ((int64_t)b * total_tokens + t) * slab_h_stride +
                          (int64_t)h * HEAD_SIZE + d;
        float val;
        if(t < num_tokens0)
        {
            in_off = ((int64_t)b * num_tokens0 + t) * slab_h_stride +
                     (int64_t)h * HEAD_SIZE + d;
            val = (float)v0_[in_off];
        }
        else
        {
            int t1 = t - num_tokens0;
            in_off = ((int64_t)b * num_tokens1 + t1) * slab_h_stride +
                     (int64_t)h * HEAD_SIZE + d;
            val = (float)v1_[in_off];
        }
        v_fp8_[out_off] = mrope_utils::fp8e4m3fnuz(val * inv);
    }
}

std::tuple<at::Tensor, at::Tensor> v_2way_per_head_fp8_quant(at::Tensor& v0,
                                                             at::Tensor& v1)
{
    TORCH_CHECK(v0.is_contiguous() && v1.is_contiguous(), "v0/v1 must be contiguous");
    TORCH_CHECK(v0.dim() == 4 && v1.dim() == 4, "v0/v1 must be 4D [B, T, H, D]");
    int64_t B  = v0.size(0);
    int64_t T0 = v0.size(1);
    int64_t T1 = v1.size(1);
    int64_t H  = v0.size(2);
    int64_t D  = v0.size(3);
    TORCH_CHECK(v1.size(0) == B && v1.size(2) == H && v1.size(3) == D,
                "v0/v1 must share B/H/D");
    TORCH_CHECK(D == 128, "v_2way_per_head_fp8_quant currently only supports head_size=128");
    TORCH_CHECK(v0.scalar_type() == v1.scalar_type(), "v0/v1 dtype must match");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(v0));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();

    at::Tensor v_fp8 =
        at::empty({B, T0 + T1, H, D}, v0.options().dtype(at::ScalarType::Float8_e4m3fnuz));
    at::Tensor v_descale = at::empty({B, H}, v0.options().dtype(at::ScalarType::Float));
    at::Tensor v_amax    = at::zeros({B, H}, v0.options().dtype(at::ScalarType::Float));

    constexpr int TILE_T    = 128;
    constexpr int HEAD_SIZE = 128;
    int num_tiles = (int)((T0 + T1 + TILE_T - 1) / TILE_T);
    dim3 grid((unsigned)num_tiles, (unsigned)H, (unsigned)B);
    dim3 block(256);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, v0.scalar_type(), "v_2way_per_head_amax_tiled", [&] {
            using T_ = KernelElementType<scalar_t>::type;
            v_2way_per_head_amax_tiled_kernel<T_, TILE_T, HEAD_SIZE>
                <<<grid, block, 0, stream>>>(
                    (T_*)v0.data_ptr<scalar_t>(),
                    (T_*)v1.data_ptr<scalar_t>(),
                    (int)T0,
                    (int)T1,
                    (int)H,
                    v_amax.data_ptr<float>());
        });

    {
        dim3 fg((unsigned)((H + 31) / 32), (unsigned)B);
        dim3 fb(32);
        v_amax_to_descale_kernel<<<fg, fb, 0, stream>>>(
            v_amax.data_ptr<float>(), (int)H, v_descale.data_ptr<float>());
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, v0.scalar_type(), "v_2way_per_head_quant_tiled", [&] {
            using T_ = KernelElementType<scalar_t>::type;
            v_2way_per_head_quant_tiled_kernel<T_, TILE_T, HEAD_SIZE>
                <<<grid, block, 0, stream>>>(
                    (T_*)v0.data_ptr<scalar_t>(),
                    (T_*)v1.data_ptr<scalar_t>(),
                    (int)T0,
                    (int)T1,
                    (int)H,
                    (mrope_utils::fp8e4m3fnuz*)v_fp8.data_ptr(),
                    v_descale.data_ptr<float>());
        });
    return {v_fp8, v_descale};
}

} // namespace aiter

// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// BF16-only, delayed Mega-mHC for wave64 CDNA GPUs. Derived from the
// BF16 path of mhc_fused_post_pre_gemm_sqrsum_kernel in mhc_kernels.cu:
// 4 waves own the 4 residual streams, 16 tokens per workgroup, 16x16x32
// BF16 MFMA, and the same mhc_shuffle_fn hi/lo packed FP32 weights.
//
// One launch: optional post-mix -> BF16 residual -> projection/square sums
// + collapse with the INCOMING pre gate -> next gates/Sinkhorn + RMSNorm.
// The newly computed pre gate is returned, NEVER used for this collapse.
// Split-K uses the last arriving workgroup as finisher, without spinning
// or requiring workgroups to be co-resident. No second reduction launch.
// FP32 accumulators/gates are retained; no FP16/FP8 or non-delayed path.
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "mega_mhc.h"
#include "opus/opus.hpp"
#include <limits>

namespace aiter {
namespace {
constexpr int BM      = 16;
constexpr int BK      = 32;
constexpr int THREADS = 256;
constexpr int SLOTS   = 32; // 24 projection outputs, residual sqsum, collapse sqsum
using bf16            = opus::bf16_t;
using bf16x8          = opus::vector_t<bf16, 8>;
using f32x4           = opus::vector_t<float, 4>;

__device__ __forceinline__ float sum_k_lanes(float x)
{
    x += __shfl_xor(x, 16, 64);
    x += __shfl_xor(x, 32, 64);
    return x;
}

template <bool HAS_POST, bool IDENTITY_PRE>
__global__ __launch_bounds__(THREADS, 1) void mega_mhc_kernel(bf16* residual_out,
                                                              float* post_out,
                                                              float* comb_out,
                                                              bf16* layer_input,
                                                              float* next_pre,
                                                              const bf16* residual,
                                                              const bf16* fn_packed,
                                                              const float* hc_scale,
                                                              const float* hc_base,
                                                              const float* pre_mix,
                                                              const bf16* sublayer,
                                                              const float* post_mix,
                                                              const float* comb_mix,
                                                              const bf16* norm_weight,
                                                              float* partials,
                                                              unsigned int* counters,
                                                              int tokens,
                                                              int hidden,
                                                              int split_k,
                                                              float rms_eps,
                                                              float pre_eps,
                                                              float sinkhorn_eps,
                                                              float post_mult,
                                                              float norm_eps,
                                                              int sinkhorn_repeat)
{
#if defined(__gfx950__) || defined(__gfx942__)
    const int tid            = threadIdx.x;
    const int wave           = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane           = tid % 64;
    const int row            = lane % 16;
    const int k_lane         = lane / 16 * 8;
    const int token          = blockIdx.x * BM + row;
    const int split          = blockIdx.y;
    const int hidden_part    = hidden / split_k;
    const int begin          = split * hidden_part;
    const int nblocks        = (tokens + BM - 1) / BM;
    const int64_t token_base = static_cast<int64_t>(token) * 4 * hidden;

    const int tile_token    = blockIdx.x * BM;
    const int tile_rows     = min(BM, tokens - tile_token);
    const int64_t tile_base = static_cast<int64_t>(tile_token) * hidden;
    auto g_res              = opus::make_gmem<bf16>(const_cast<bf16*>(residual) + tile_base * 4,
                                       tile_rows * 4 * hidden * sizeof(bf16));
    auto g_out =
        opus::make_gmem<bf16>(residual_out + tile_base * 4, tile_rows * 4 * hidden * sizeof(bf16));
    auto g_sub  = opus::make_gmem<bf16>(const_cast<bf16*>(sublayer) + tile_base,
                                       tile_rows * hidden * sizeof(bf16));
    auto g_norm = opus::make_gmem<bf16>(layer_input + tile_base, tile_rows * hidden * sizeof(bf16));
    auto g_weight = opus::make_gmem<bf16>(const_cast<bf16*>(norm_weight), hidden * sizeof(bf16));
    auto g_fn =
        opus::make_gmem<bf16>(const_cast<bf16*>(fn_packed), 24 * 4 * hidden * sizeof(float));

    // Separate old/new tiles keep post mixing and collapse independent. In the
    // no-post identity specialization both arrays are optimized away.
    __shared__ bf16 old_r[4][BM][BK];
    __shared__ bf16 new_r[4][BM][BK];
    __shared__ float reduction[4][BM][SLOTS];
    __shared__ float total[BM][SLOTS];
    __shared__ unsigned int last;

    float post = 0.0f, comb[4] = {}, pre[4] = {};
    if(token < tokens)
    {
        if constexpr(HAS_POST)
        {
            post = post_mix[token * 4 + wave];
#pragma unroll
            for(int h = 0; h < 4; ++h)
                comb[h] = comb_mix[token * 16 + h * 4 + wave];
        }
        if constexpr(!IDENTITY_PRE)
        {
#pragma unroll
            for(int h = 0; h < 4; ++h)
                pre[h] = pre_mix[token * 4 + h];
        }
    }
    f32x4 acc[2] = {};
    float sq_r = 0.0f, sq_x = 0.0f;
    struct Tile
    {
        bf16x8 r, x, hi[2], lo[2];
    };
    // Keep the stream assignment unchanged. Buffer addressing helps the
    // no-post mixed-pre path; post/identity retain the measured global path.
    constexpr bool BUFFER_LOADS = !HAS_POST && !IDENTITY_PRE;
    auto load_tile              = [&](int k) {
        Tile tile{};
        if constexpr(BUFFER_LOADS)
        {
            tile.r = load_vector_nbytes<bf16, 8, 16>(g_res,
                                                     row * 4 * hidden + wave * hidden + k + k_lane);
            if constexpr(HAS_POST)
                tile.x = load_vector_nbytes<bf16, 8, 16>(g_sub, row * hidden + k + k_lane);
#pragma unroll
            for(int n = 0; n < 2; ++n)
            {
                const int gate = lane % 16 + n * 16;
                const int off  = (gate * 4 * hidden + wave * hidden + k + k_lane) * 2;
                tile.hi[n]     = load_vector_nbytes<bf16, 8, 16>(g_fn, off);
                tile.lo[n]     = load_vector_nbytes<bf16, 8, 16>(g_fn, off + 8);
            }
        }
        else
        {
            if(token < tokens)
            {
                tile.r = *reinterpret_cast<const bf16x8*>(residual + token_base + wave * hidden +
                                                          k + k_lane);
                if constexpr(HAS_POST)
                    tile.x = *reinterpret_cast<const bf16x8*>(
                        sublayer + static_cast<int64_t>(token) * hidden + k + k_lane);
            }
#pragma unroll
            for(int n = 0; n < 2; ++n)
            {
                const int gate = lane % 16 + n * 16;
                if(gate < 24)
                {
                    const int64_t off =
                        (static_cast<int64_t>(gate) * 4 * hidden + wave * hidden + k + k_lane) * 2;
                    tile.hi[n] = *reinterpret_cast<const bf16x8*>(fn_packed + off);
                    tile.lo[n] = *reinterpret_cast<const bf16x8*>(fn_packed + off + 8);
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        return tile;
    };
    Tile current = load_tile(begin);
    for(int k = begin; k < begin + hidden_part; k += BK)
    {
        Tile next{};
        if(k + BK < begin + hidden_part)
            next = load_tile(k + BK);
        bf16x8 r = current.r;
        if constexpr(HAS_POST)
        {
            *reinterpret_cast<bf16x8*>(&old_r[wave][row][k_lane]) = r;
            __syncthreads();
            // This rendezvous also protects the previous iteration's new_r
            // readers: all of them finished before reaching this barrier.
            bf16x8 old_regs[4];
#pragma unroll
            for(int h = 0; h < 4; ++h)
                old_regs[h] = *reinterpret_cast<const bf16x8*>(&old_r[h][row][k_lane]);
            const bf16x8& x = current.x;
#pragma unroll
            for(int e = 0; e < 8; ++e)
            {
                float v = static_cast<float>(x[e]) * post;
#pragma unroll
                for(int h = 0; h < 4; ++h)
                    v = fmaf(static_cast<float>(old_regs[h][e]), comb[h], v);
                // Delayed semantics: all consumers use the rounded residual.
                r[e] = opus::fp32_to_bf16(v);
            }
            if(token < tokens)
                *reinterpret_cast<bf16x8*>(residual_out + token_base + wave * hidden + k + k_lane) =
                    r;
        }
        if constexpr(HAS_POST || !IDENTITY_PRE)
            *reinterpret_cast<bf16x8*>(&new_r[wave][row][k_lane]) = r;
#pragma unroll
        for(int e = 0; e < 8; ++e)
        {
            float v = static_cast<float>(r[e]);
            sq_r    = fmaf(v, v, sq_r);
        }
        // Publish new_r before the MFMA: collapse does not depend on it.
        // This also proves all old_r readers are done before old_r is reused.
        if constexpr(HAS_POST || !IDENTITY_PRE)
            __syncthreads();
        if(wave == 0)
        {
            bf16x8 collapsed;
            bf16x8 new_regs[4];
            if constexpr(!HAS_POST && IDENTITY_PRE)
            {
                // Wave 0 already owns stream 0: identity collapse is lane-local.
                new_regs[0] = r;
            }
            else
            {
#pragma unroll
                for(int h = 0; h < (IDENTITY_PRE ? 1 : 4); ++h)
                    new_regs[h] = *reinterpret_cast<const bf16x8*>(&new_r[h][row][k_lane]);
            }
#pragma unroll
            for(int e = 0; e < 8; ++e)
            {
                float v;
                if constexpr(IDENTITY_PRE)
                {
                    v = static_cast<float>(new_regs[0][e]);
                }
                else
                {
                    v = 0.0f;
#pragma unroll
                    for(int h = 0; h < 4; ++h)
                        v = fmaf(static_cast<float>(new_regs[h][e]), pre[h], v);
                }
                collapsed[e] = opus::fp32_to_bf16(v);
                v            = static_cast<float>(collapsed[e]);
                sq_x         = fmaf(v, v, sq_x);
            }
            if constexpr(BUFFER_LOADS)
            {
                store_vector_nbytes<bf16, bf16, 8, 16>(
                    g_norm, collapsed, row * hidden + k + k_lane);
            }
            else if(token < tokens)
            {
                *reinterpret_cast<bf16x8*>(layer_input + static_cast<int64_t>(token) * hidden + k +
                                           k_lane) = collapsed;
            }
        }
#pragma unroll
        for(int n = 0; n < 2; ++n)
        {
            acc[n] = opus::mfma_f32_16x16x32_bf16{}(current.hi[n], r, acc[n]);
            acc[n] = opus::mfma_f32_16x16x32_bf16{}(current.lo[n], r, acc[n]);
        }
        // Post: the next old_r barrier protects this iteration's new_r readers.
        // No-post mixed-pre: explicitly wait before overwriting new_r.
        // No-post identity: no cross-wave data is exchanged inside the loop.
        if constexpr(!HAS_POST && !IDENTITY_PRE)
            __syncthreads();
        current = next;
    }

#pragma unroll
    for(int n = 0; n < 2; ++n)
#pragma unroll
        for(int e = 0; e < 4; ++e)
            reduction[wave][row][n * 16 + lane / 16 * 4 + e] = acc[n][e];
    sq_r = sum_k_lanes(sq_r);
    sq_x = sum_k_lanes(sq_x);
    __syncthreads();
    // The last eight MFMA columns are padding; reuse them for the two sums.
    if(lane < 16)
    {
        reduction[wave][row][24] = sq_r;
        reduction[wave][row][25] = wave == 0 ? sq_x : 0.0f;
    }
    __syncthreads();
    for(int i = tid; i < BM * SLOTS; i += THREADS)
    {
        const int r = i / SLOTS, c = i % SLOTS;
        float v = 0;
#pragma unroll
        for(int h = 0; h < 4; ++h)
            v += reduction[h][r][c];
        total[r][c] = v;
        if(split_k > 1)
            partials[((static_cast<int64_t>(split) * nblocks + blockIdx.x) * BM * SLOTS) + i] = v;
    }
    if(split_k > 1)
    {
        // Each writer publishes its partials AND BF16 collapse stores before
        // thread 0 announces this block. Other blocks can retire immediately.
        __threadfence();
        __syncthreads();
        if(tid == 0)
            last = __hip_atomic_fetch_add(
                       counters + blockIdx.x, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) ==
                   static_cast<unsigned>(split_k - 1);
        __syncthreads();
        if(!last)
            return;
        // The last RMW acquires all preceding releases. Propagate the acquire
        // to every wave, invalidating its cache once instead of making every
        // partial/collapse load volatile (which serializes those loads).
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        for(int i = tid * 4; i < BM * SLOTS; i += THREADS * 4)
        {
            f32x4 v = {};
            for(int s = 0; s < split_k; s += 4)
            {
                f32x4 p[4] = {};
#pragma unroll
                for(int u = 0; u < 4; ++u)
                    if(s + u < split_k)
                        p[u] = *reinterpret_cast<const f32x4*>(
                            partials +
                            (static_cast<int64_t>(s + u) * nblocks + blockIdx.x) * BM * SLOTS + i);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(int u = 0; u < 4; ++u)
                    v += p[u];
            }
            *reinterpret_cast<f32x4*>(&total[i / SLOTS][i % SLOTS]) = v;
        }
    }
    __syncthreads();
    if(tid < BM)
    {
        total[tid][24] = rsqrtf(total[tid][24] / (4 * hidden) + rms_eps);
        total[tid][25] = rsqrtf(total[tid][25] / hidden + norm_eps);
    }
    __syncthreads();

    // One 16-lane subgroup per token: Sinkhorn's 4x4 matrix resides in lanes.
    const int gate_row   = tid / 16;
    const int gate_lane  = tid % 16;
    const int gate_token = blockIdx.x * BM + gate_row;
    const float inv      = total[gate_row][24];
    if(gate_lane < 4 && gate_token < tokens)
    {
        const float p = total[gate_row][gate_lane] * inv * hc_scale[0] + hc_base[gate_lane];
        const float q = total[gate_row][4 + gate_lane] * inv * hc_scale[1] + hc_base[4 + gate_lane];
        next_pre[gate_token * 4 + gate_lane] = 1.0f / (1.0f + __expf(-p)) + pre_eps;
        post_out[gate_token * 4 + gate_lane] = post_mult / (1.0f + __expf(-q));
    }
    float c   = total[gate_row][8 + gate_lane] * inv * hc_scale[2] + hc_base[8 + gate_lane];
    float mx  = fmaxf(c, __shfl_xor(c, 1, 16));
    mx        = fmaxf(mx, __shfl_xor(mx, 2, 16));
    c         = __expf(c - mx);
    float sum = c + __shfl_xor(c, 1, 16);
    sum += __shfl_xor(sum, 2, 16);
    c   = c / sum + sinkhorn_eps;
    sum = c + __shfl_xor(c, 4, 16);
    sum += __shfl_xor(sum, 8, 16);
    c /= sum + sinkhorn_eps;
    for(int it = 1; it < sinkhorn_repeat; ++it)
    {
        sum = c + __shfl_xor(c, 1, 16);
        sum += __shfl_xor(sum, 2, 16);
        c /= sum + sinkhorn_eps;
        sum = c + __shfl_xor(c, 4, 16);
        sum += __shfl_xor(sum, 8, 16);
        c /= sum + sinkhorn_eps;
    }
    if(gate_token < tokens)
        comb_out[gate_token * 16 + gate_lane] = c;

    // Reload the rounded collapse, including columns produced by other splits.
    // The acquire above (or the same-block barrier for split_k=1) makes the
    // rounded collapse visible to these ordinary vector loads.
    // Batch four independent vectors so normalization does not serialize a
    // load/wait/store for every eight elements. Buffer bounds cover tail rows.
    for(int base = 0; base < tile_rows * hidden; base += THREADS * 8 * 4)
    {
        bf16x8 x[4], w[4];
#pragma unroll
        for(int u = 0; u < 4; ++u)
        {
            const int i = base + tid * 8 + u * THREADS * 8;
            x[u]        = load_vector_nbytes<bf16, 8, 16>(g_norm, i);
            w[u]        = load_vector_nbytes<bf16, 8, 16>(g_weight, i % hidden);
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(int u = 0; u < 4; ++u)
        {
            const int i = base + tid * 8 + u * THREADS * 8;
            // Clamp padded rows for the LDS read; the buffer store discards them.
            const int r = min(i / hidden, tile_rows - 1);
            bf16x8 out;
#pragma unroll
            for(int e = 0; e < 8; ++e)
                out[e] = opus::fp32_to_bf16(static_cast<float>(x[u][e]) * total[r][25] *
                                            static_cast<float>(w[u][e]));
            store_vector_nbytes<bf16, bf16, 8, 16>(g_norm, out, i);
        }
    }
    // No other split can still publish: all of them incremented the counter.
    // The stream orders the next launch after this finisher has completed.
    if(split_k > 1)
    {
        __syncthreads();
        if(tid == 0)
            atomicExch(counters + blockIdx.x, 0u);
    }
#endif
}

void check_tensor(const aiter_tensor_t& t,
                  const aiter_tensor_t& ref,
                  AiterDtype dtype,
                  size_t numel,
                  const char* name)
{
    AITER_CHECK(
        t.is_gpu() && t.device_id == ref.device_id, name, " must be on the residual device");
    AITER_CHECK(t.dtype() == dtype && t.is_contiguous() && t.numel() == numel,
                name,
                " has an invalid dtype, size or layout");
    AITER_CHECK(numel == 0 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                name,
                " must be 16-byte aligned");
}
} // namespace

void mega_mhc_bf16(aiter_tensor_t& residual_out,
                   aiter_tensor_t& post_out,
                   aiter_tensor_t& comb_out,
                   aiter_tensor_t& layer_input,
                   aiter_tensor_t& next_pre,
                   aiter_tensor_t& residual,
                   aiter_tensor_t& fn_packed,
                   aiter_tensor_t& hc_scale,
                   aiter_tensor_t& hc_base,
                   aiter_tensor_t& pre_mix,
                   aiter_tensor_t& sublayer_out,
                   aiter_tensor_t& post_mix,
                   aiter_tensor_t& comb_mix,
                   aiter_tensor_t& norm_weight,
                   aiter_tensor_t& partials,
                   aiter_tensor_t& counters,
                   float rms_eps,
                   float hc_pre_eps,
                   float hc_sinkhorn_eps,
                   float hc_post_mult,
                   float norm_eps,
                   int sinkhorn_repeat,
                   int split_k,
                   bool has_post,
                   bool identity_pre)
{
    AITER_CHECK(residual.is_gpu() && residual.dim() == 3 && residual.size(1) == 4,
                "residual must be a GPU tensor of shape (T, 4, H)");
    const int64_t t = residual.size(0), h = residual.size(2);
    AITER_CHECK(t <= std::numeric_limits<int>::max() / 32 && h > 0 && h <= 65536,
                "invalid T or H (H must be <= 65536)");
    AITER_CHECK(split_k > 0 && split_k <= 32 && h % (BK * split_k) == 0,
                "split_k must be in [1, 32] and H divisible by 32*split_k");
    AITER_CHECK(rms_eps > 0 && norm_eps > 0 && hc_sinkhorn_eps > 0 && sinkhorn_repeat > 0,
                "RMS/Sinkhorn eps and sinkhorn_repeat must be positive");
    check_tensor(residual, residual, AITER_DTYPE_bf16, t * 4 * h, "residual");
    check_tensor(residual_out, residual, AITER_DTYPE_bf16, t * 4 * h, "residual_out");
    check_tensor(layer_input, residual, AITER_DTYPE_bf16, t * h, "layer_input");
    check_tensor(fn_packed, residual, AITER_DTYPE_i32, 24 * 4 * h, "fn_packed");
    check_tensor(hc_scale, residual, AITER_DTYPE_fp32, 3, "hc_scale");
    check_tensor(hc_base, residual, AITER_DTYPE_fp32, 24, "hc_base");
    check_tensor(norm_weight, residual, AITER_DTYPE_bf16, h, "norm_weight");
    check_tensor(next_pre, residual, AITER_DTYPE_fp32, t * 4, "next_pre");
    check_tensor(post_out, residual, AITER_DTYPE_fp32, t * 4, "post_out");
    check_tensor(comb_out, residual, AITER_DTYPE_fp32, t * 16, "comb_out");
    if(!identity_pre)
        check_tensor(pre_mix, residual, AITER_DTYPE_fp32, t * 4, "pre_mix");
    if(has_post)
    {
        check_tensor(sublayer_out, residual, AITER_DTYPE_bf16, t * h, "sublayer_out");
        check_tensor(post_mix, residual, AITER_DTYPE_fp32, t * 4, "post_mix");
        check_tensor(comb_mix, residual, AITER_DTYPE_fp32, t * 16, "comb_mix");
        AITER_CHECK(t == 0 || residual_out.data_ptr() != residual.data_ptr(),
                    "residual_out must not alias residual");
    }
    else
    {
        AITER_CHECK(residual_out.data_ptr() == residual.data_ptr(),
                    "no_post returns the original residual");
    }
    const int blocks = (t + BM - 1) / BM;
    check_tensor(partials,
                 residual,
                 AITER_DTYPE_fp32,
                 static_cast<size_t>(split_k) * blocks * BM * SLOTS,
                 "partials");
    check_tensor(counters, residual, AITER_DTYPE_i32, blocks, "counters");
    const HipDeviceGuard guard(residual.device_id);
    const auto arch = get_gpu_arch();
    AITER_CHECK(arch == "gfx950" || arch == "gfx942", "mega_mhc requires gfx950 or gfx942");
    if(t == 0)
        return;
    const hipStream_t stream = getCurrentHIPStream();
#define LAUNCH(POST, IDENTITY)                                                      \
    mega_mhc_kernel<POST, IDENTITY><<<dim3(blocks, split_k), THREADS, 0, stream>>>( \
        static_cast<bf16*>(residual_out.data_ptr()),                                \
        static_cast<float*>(post_out.data_ptr()),                                   \
        static_cast<float*>(comb_out.data_ptr()),                                   \
        static_cast<bf16*>(layer_input.data_ptr()),                                 \
        static_cast<float*>(next_pre.data_ptr()),                                   \
        static_cast<const bf16*>(residual.data_ptr()),                              \
        static_cast<const bf16*>(fn_packed.data_ptr()),                             \
        static_cast<const float*>(hc_scale.data_ptr()),                             \
        static_cast<const float*>(hc_base.data_ptr()),                              \
        static_cast<const float*>(pre_mix.data_ptr()),                              \
        static_cast<const bf16*>(sublayer_out.data_ptr()),                          \
        static_cast<const float*>(post_mix.data_ptr()),                             \
        static_cast<const float*>(comb_mix.data_ptr()),                             \
        static_cast<const bf16*>(norm_weight.data_ptr()),                           \
        static_cast<float*>(partials.data_ptr()),                                   \
        static_cast<unsigned int*>(counters.data_ptr()),                            \
        t,                                                                          \
        h,                                                                          \
        split_k,                                                                    \
        rms_eps,                                                                    \
        hc_pre_eps,                                                                 \
        hc_sinkhorn_eps,                                                            \
        hc_post_mult,                                                               \
        norm_eps,                                                                   \
        sinkhorn_repeat)
    if(has_post)
    {
        if(identity_pre)
        {
            LAUNCH(true, true);
        }
        else
        {
            LAUNCH(true, false);
        }
    }
    else
    {
        if(identity_pre)
        {
            LAUNCH(false, true);
        }
        else
        {
            LAUNCH(false, false);
        }
    }
#undef LAUNCH
    HIP_CALL(hipGetLastError());
}
} // namespace aiter

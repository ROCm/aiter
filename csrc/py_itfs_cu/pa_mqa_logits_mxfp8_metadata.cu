// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP8 paged MQA logits (gfx950) — persistent-grid schedule + varqlen windows.
//
// C++ HIP port of the FlyDSL device schedule (pa_mqa_logits_fp4_prefill.py:
// compute_prefill_schedule / _prefill_cta_info_kernel and compute_varqlen_windows /
// _varqlen_windows_kernel). Fully device-side and cudagraph-safe:
//   * all inputs/outputs (incl. the `scratch` workspace) are caller-allocated
//     device buffers — no hipMalloc / no host<->device sync at build time;
//   * launch grids are derived from static shapes (total_tokens / parallel_unit_num /
//     total_q), never from window/context values.
//
// The emitted cta_info[P, 6] matches the kernel ABI exactly:
//   {row_id, batch_id, chunk_start, chunk_count, local_start, local_end}.

#include "pa_mqa_logits_mxfp8_opus.h"  // public API decls + CTA_INFO_WIDTH (no IMPL)

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

namespace {

constexpr int SCHED_BLOCK = 256;

__device__ inline int ceil_div_dev(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// Prep: chunks_per_row -> safe (smallest split factor) -> inclusive prefix sum
// of per-row CTA counts. Single workgroup (grid=1) so the prefix sum stays a
// simple in-kernel scan; O(s_max * T) safe search mirrors FlyDSL's ctas_per_r_s.
//   incl[T]         (out, scratch) : inclusive cumsum of ceil(chunks_r / safe)
//   scalars[0]      (out)          : safe
//   scalars[1]      (out)          : total_splits
// ---------------------------------------------------------------------------
__global__ void mqa_logits_prefill_prep_kernel(const int* __restrict__ local_ends,
                                               int* __restrict__ incl,
                                               int* __restrict__ scalars,
                                               int T, int P, int block_k, int s_max)
{
    const int tid = threadIdx.x;
    __shared__ long long sred[SCHED_BLOCK];
    __shared__ int s_safe;
    __shared__ int s_max_chunks;

    // ---- step 1: chunks_r[r] = ceil(le / block_k); stash into incl[] ----
    int local_max = 0;
    for(int r = tid; r < T; r += SCHED_BLOCK)
    {
        int le = local_ends[r];
        int c  = (le <= 0) ? 0 : ceil_div_dev(le, block_k);
        incl[r] = c;
        if(c > local_max)
            local_max = c;
    }
    // block-max -> max_chunks (fallback safe when no split factor is feasible)
    sred[tid] = local_max;
    __syncthreads();
    for(int s = SCHED_BLOCK / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sred[tid] = (sred[tid] > sred[tid + s]) ? sred[tid] : sred[tid + s];
        __syncthreads();
    }
    if(tid == 0)
        s_max_chunks = (int)(sred[0] < 1 ? 1 : sred[0]);
    __syncthreads();

    // ---- step 2: smallest s in [1, s_max] with sum_r ceil(chunks_r / s) <= P ----
    int found_safe = -1;
    for(int s = 1; s <= s_max; ++s)
    {
        long long local_sum = 0;
        for(int r = tid; r < T; r += SCHED_BLOCK)
            local_sum += ceil_div_dev(incl[r], s);
        sred[tid] = local_sum;
        __syncthreads();
        for(int w = SCHED_BLOCK / 2; w > 0; w >>= 1)
        {
            if(tid < w)
                sred[tid] += sred[tid + w];
            __syncthreads();
        }
        if(sred[0] <= (long long)P)
        {
            found_safe = s;
            break;
        }
        __syncthreads();
    }
    if(tid == 0)
        s_safe = (found_safe > 0) ? found_safe : s_max_chunks;
    __syncthreads();
    const int safe = s_safe;

    // ---- step 3: ctas_r = ceil(chunks_r / safe); inclusive prefix sum (serial) ----
    if(tid == 0)
    {
        int running = 0;
        for(int r = 0; r < T; ++r)
        {
            int chunks_r = incl[r];   // still chunks_r from step 1
            int ctas_r   = ceil_div_dev(chunks_r, safe);   // 0 for empty rows
            running += ctas_r;
            incl[r] = running;        // inclusive prefix sum
        }
        scalars[0] = safe;
        scalars[1] = running;         // total_splits
    }
}

// ---------------------------------------------------------------------------
// Per-slot emit: searchsorted(incl, slot, right=True) -> row, then cta_info.
// Mirrors _prefill_cta_info_kernel field-for-field. chunks_row / excl are
// recovered from incl + local_ends (no extra scratch needed).
// ---------------------------------------------------------------------------
__global__ void mqa_logits_prefill_cta_info_kernel(const int* __restrict__ incl,
                                                   const int* __restrict__ row_to_batch,
                                                   const int* __restrict__ local_starts,
                                                   const int* __restrict__ local_ends,
                                                   const int* __restrict__ scalars,
                                                   int T, int P, int block_k,
                                                   int* __restrict__ cta_info)
{
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if(slot >= P)
        return;

    int* ci = cta_info + (size_t)slot * CTA_INFO_WIDTH;
    if(T <= 0)
    {   // no rows: emit an inert, well-formed entry (chunk_count >= 1)
        ci[0] = ci[1] = ci[2] = ci[4] = ci[5] = 0;
        ci[3] = 1;
        return;
    }

    const int safe         = scalars[0];
    const int total_splits = scalars[1];

    // count(incl <= slot): per-slot binary search over incl[T] (~log2(T) iters).
    int lo = 0, hi = T;
    while(lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if(incl[mid] <= slot)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int row      = (lo < (T - 1)) ? lo : (T - 1);   // clamp for gather
    const int excl_row = (row > 0) ? incl[row - 1] : 0;   // exclusive prefix

    const int le       = local_ends[row];
    const int chunks_r = (le <= 0) ? 0 : ceil_div_dev(le, block_k);

    const bool valid   = slot < total_splits;
    const int  vi      = valid ? 1 : 0;
    const int split_within = slot - excl_row;
    int start = split_within * safe;
    int count = chunks_r - start;
    count = count < safe ? count : safe;   // min(safe, chunks - start)
    count = count > 0 ? count : 0;         // max(., 0)

    ci[0] = row * vi;                       // row_id
    ci[1] = row_to_batch[row] * vi;         // batch_id
    ci[2] = start * vi;                     // chunk_start
    ci[3] = valid ? count : 1;              // chunk_count (>=1 so the loop is well-formed)
    ci[4] = local_starts[row] * vi;         // local_start
    ci[5] = le * vi;                        // local_end
}

// ---------------------------------------------------------------------------
// varqlen (per-batch MTP) window build. Port of _varqlen_windows_kernel.
//   MTP tail-causal: batch b's n-th query token (n in [0, qlen)) sees
//   [0, context_len[b] - (qlen - 1 - n)); rows past cu[B] get an empty window.
// ---------------------------------------------------------------------------
__global__ void mqa_logits_varqlen_windows_kernel(const int* __restrict__ cu,
                                                  const int* __restrict__ ctx,
                                                  int* __restrict__ row_to_batch,
                                                  int* __restrict__ local_starts,
                                                  int* __restrict__ local_ends,
                                                  int total_q, int B)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if(r >= total_q)
        return;

    // searchsorted(cu[1:], r, right=True) = count(cu[1..B] <= r).
    int lo = 0, hi = B;
    while(lo < hi)
    {
        int mid    = (lo + hi) >> 1;
        int cu_mid = cu[1 + (mid < (B - 1) ? mid : (B - 1))];
        if(cu_mid <= r)
            lo = mid + 1;
        else
            hi = mid;
    }
    const int b = (lo < (B - 1)) ? lo : (B - 1);

    const int cu_b  = cu[b];
    const int cu_b1 = cu[b + 1];
    const int ctx_b = ctx[b];
    const int n     = r - cu_b;
    const int qlen  = cu_b1 - cu_b;
    int le = ctx_b - qlen + n + 1;
    le = le > 0 ? le : 0;
    // Rows beyond the real total (cu[B]) are flat tail-padding -> empty window.
    const int real_total = cu[B];
    if(r >= real_total)
        le = 0;

    row_to_batch[r] = b;
    local_starts[r] = 0;
    local_ends[r]   = le;
}

} // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------
void pa_mqa_logits_mxfp8_prefill_schedule(aiter_tensor_t& row_to_batch,
                                       aiter_tensor_t& local_starts,
                                       aiter_tensor_t& local_ends,
                                       aiter_tensor_t& scratch,
                                       aiter_tensor_t& cta_info,
                                       int total_tokens,
                                       int parallel_unit_num,
                                       int block_k,
                                       int max_seq_len)
{
    const int T = total_tokens;
    const int P = parallel_unit_num;
    AITER_CHECK(P >= T,
                "parallel_unit_num=", P, " < total_tokens=", T,
                " would drop rows (their logits stay at the caller's pre-fill).");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.dim() == 2 &&
                    cta_info.size(1) == CTA_INFO_WIDTH,
                "cta_info must be int32 [P, 6]");
    AITER_CHECK(scratch.dtype() == AITER_DTYPE_i32,
                "scratch must be int32 (>= total_tokens + 2 elements)");
    AITER_CHECK((int)scratch.numel() >= T + 2,
                "scratch too small: need >= total_tokens + 2 int32 elements");

    HipDeviceGuard guard(cta_info.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    int* incl    = reinterpret_cast<int*>(scratch.data_ptr());
    int* scalars = incl + T;   // [safe, total_splits]
    const int* le = reinterpret_cast<const int*>(local_ends.data_ptr());
    const int* rb = reinterpret_cast<const int*>(row_to_batch.data_ptr());
    const int* ls = reinterpret_cast<const int*>(local_starts.data_ptr());
    int* ci       = reinterpret_cast<int*>(cta_info.data_ptr());

    const int s_max = (max_seq_len + block_k - 1) / block_k;
    const int s_max_c = s_max > 1 ? s_max : 1;

    if(T > 0)
        mqa_logits_prefill_prep_kernel<<<1, SCHED_BLOCK, 0, stream>>>(
            le, incl, scalars, T, P, block_k, s_max_c);

    const int grid = (P + SCHED_BLOCK - 1) / SCHED_BLOCK;
    mqa_logits_prefill_cta_info_kernel<<<grid, SCHED_BLOCK, 0, stream>>>(
        incl, rb, ls, le, scalars, T, P, block_k, ci);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void pa_mqa_logits_mxfp8_varqlen_windows(aiter_tensor_t& cu_seq_q,
                                      aiter_tensor_t& context_lens,
                                      aiter_tensor_t& row_to_batch,
                                      aiter_tensor_t& local_starts,
                                      aiter_tensor_t& local_ends,
                                      int total_q)
{
    const int B = static_cast<int>(context_lens.size(0));
    AITER_CHECK(cu_seq_q.dtype() == AITER_DTYPE_i32 && context_lens.dtype() == AITER_DTYPE_i32,
                "cu_seq_q / context_lens must be int32");
    AITER_CHECK(cu_seq_q.size(0) == B + 1, "cu_seq_q must have length B+1");

    if(total_q <= 0 || B <= 0)
        return;

    HipDeviceGuard guard(context_lens.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const int grid = (total_q + SCHED_BLOCK - 1) / SCHED_BLOCK;
    mqa_logits_varqlen_windows_kernel<<<grid, SCHED_BLOCK, 0, stream>>>(
        reinterpret_cast<const int*>(cu_seq_q.data_ptr()),
        reinterpret_cast<const int*>(context_lens.data_ptr()),
        reinterpret_cast<int*>(row_to_batch.data_ptr()),
        reinterpret_cast<int*>(local_starts.data_ptr()),
        reinterpret_cast<int*>(local_ends.data_ptr()),
        total_q, B);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SparseKV swap-in for GLM-5.2 DSA decode. Keeps the full KV in a pinned host
// cold pool and a fixed GPU hot buffer per request; each decode step, per layer,
// miss-detects the indexer top-k against the resident hot set, evicts the
// least-recently-used slots for the misses, gathers the missing tokens from the
// cold pool over PCIe/XGMI, and translates the top-k into hot-buffer rows.
//
// One wavefront64 copies one token's item_size_bytes word-wise (mirrors SGLang
// transfer_item_warp, ROCm branch of sparsekv.cuh). The cold pool pointer must
// be the device-mapped pointer from sparsekv_host_get_device_pointer (xnack-).

#include "sparsekv_swap.h"

#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int WARP_SIZE = 64;  // wavefront64 on gfx950
constexpr int BLOCK     = 256;
// Block target for the planned gather: gfx950 has 256 CUs, so a few thousand
// blocks keeps every CU fed even when the decode batch is small.
constexpr int GATHER_TARGET_BLOCKS = 2048;
constexpr int EMPTY     = -1;

// Word-wise warp copy of one item between two mapped addresses (host<->device).
__device__ __forceinline__ void transfer_item_warp(int lane_id,
                                                    const void* __restrict__ src_addr,
                                                    void* __restrict__ dst_addr,
                                                    int64_t item_size_bytes)
{
    const auto* src = static_cast<const char*>(src_addr);
    auto* dst       = static_cast<char*>(dst_addr);
    const int64_t word_count = item_size_bytes / (int64_t)sizeof(uint64_t);
    const auto* src_words = reinterpret_cast<const uint64_t*>(src);
    auto* dst_words       = reinterpret_cast<uint64_t*>(dst);
    for (int64_t i = lane_id; i < word_count; i += WARP_SIZE) {
        dst_words[i] = src_words[i];
    }
    const int64_t tail = word_count * (int64_t)sizeof(uint64_t);
    for (int64_t i = tail + lane_id; i < item_size_bytes; i += WARP_SIZE) {
        dst[i] = src[i];
    }
}

// Row counts of the two cold pools, published once by the coordinator. The
// translation tables are the only thing that says where a token lives, so a
// stale or corrupt entry is indistinguishable from a real row — and because the
// pools are exact-sized, dereferencing one past the end reaches memory the agent
// has no mapping for and kills the whole process with a memory access fault.
// Bounding here turns that into a skipped token. Defaults are permissive so a
// caller that never publishes them behaves as before.
__device__ int64_t g_sparsekv_cold_rows     = INT64_MAX;
__device__ int64_t g_sparsekv_gpu_cold_rows = INT64_MAX;
// Out-of-range rows skipped so far. Skipping keeps the process alive but makes
// the affected token read a stale hot slot, so the count has to be visible —
// a silently wrong answer is worse to debug than the crash it replaced.
__device__ unsigned long long g_sparsekv_oob_rows = 0ULL;

// Resolve a request's logical token to its absolute cold-pool row. With a paged
// host pool (host_cache_locs != nullptr) the row is read from the per-request
// translation table (req_to_host_pool[r][tok]); otherwise the dense layout maps
// logical token tok of request r to row r*cold_depth + tok.
__device__ __forceinline__ int64_t cold_row_bounded(const int32_t* __restrict__ locs,
                                                    int stride, int r, int cold_depth,
                                                    int tok, int64_t max_rows)
{
    if (locs) {
        const int64_t row = (int64_t)locs[(int64_t)r * stride + tok];
        if (row >= max_rows) {
            atomicAdd(&g_sparsekv_oob_rows, 1ULL);
            return (int64_t)-1;
        }
        return row;
    }
    return (int64_t)r * cold_depth + tok;
}

__device__ __forceinline__ int64_t cold_row_of(const int32_t* __restrict__ host_cache_locs,
                                               int host_stride, int r, int cold_depth,
                                               int tok)
{
    if (host_cache_locs) {
        const int64_t row = (int64_t)host_cache_locs[(int64_t)r * host_stride + tok];
        // Reports out-of-range as unbacked so every caller's existing `< 0`
        // guard skips it; negatives pass through unchanged.
        if (row >= g_sparsekv_cold_rows) {
            atomicAdd(&g_sparsekv_oob_rows, 1ULL);
            return (int64_t)-1;
        }
        return row;
    }
    return (int64_t)r * cold_depth + tok;
}

// Count hot slots for this request whose recency tick is <= tau (block-wide).
__device__ int block_count_le(const int64_t* __restrict__ lu_base,
                              int hot_slots, int64_t tau, int* s_count)
{
    if (threadIdx.x == 0) *s_count = 0;
    __syncthreads();
    int loc = 0;
    for (int s = threadIdx.x; s < hot_slots; s += blockDim.x) {
        if (lu_base[s] <= tau) loc++;
    }
    atomicAdd(s_count, loc);
    __syncthreads();
    int c = *s_count;
    __syncthreads();
    return c;
}

// Plain gather: one warp per miss, host cold pool -> device hot buffer.
__global__ void sparsekv_gather_kernel(const char* __restrict__ host_cache,
                                       char* __restrict__ device_buffer,
                                       const int32_t* __restrict__ src_locs,
                                       const int32_t* __restrict__ dst_locs,
                                       int num_misses, int64_t item_size_bytes)
{
    const int warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id   = threadIdx.x % WARP_SIZE;
    const int num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;
    for (int m = warp_id; m < num_misses; m += num_warps) {
        const int32_t src_row = src_locs[m];
        // Unbacked, or past the end of the pool (see g_sparsekv_cold_rows).
        if (src_row < 0) continue;
        if ((int64_t)src_row >= g_sparsekv_cold_rows) {
            atomicAdd(&g_sparsekv_oob_rows, 1ULL);
            continue;
        }
        const int64_t src = (int64_t)src_row * item_size_bytes;
        const int64_t dst = (int64_t)dst_locs[m] * item_size_bytes;
        transfer_item_warp(lane_id, host_cache + src, device_buffer + dst,
                           item_size_bytes);
    }
}

// Fused per-layer hot path. One block per decode query token. Requires each
// request to own at most one query token in the batch (non-MTP decode); the
// caller keeps MTP on the reference path. Dynamic shared memory holds two int
// arrays of length `topk`: the miss token list and the victim slot list.
__global__ void sparsekv_swap_and_translate_kernel(
    const char* __restrict__ cold_pool,   // device-mapped host cold pool, this layer
    char* __restrict__ hot_buffer,        // GPU hot buffer, this layer
    const int32_t* __restrict__ topk_logical,  // [n, topk]
    const int32_t* __restrict__ indptr,        // [n+1]
    const int32_t* __restrict__ req_slots,     // [n]
    int32_t* __restrict__ slot_token,          // [R, hot_slots]
    int64_t* __restrict__ last_used,           // [R, hot_slots]
    int32_t* __restrict__ token_to_slot,       // [R, cold_depth]
    int64_t* __restrict__ recency,             // [R]
    int32_t* __restrict__ out_translated,      // [total_topk]
    int32_t* __restrict__ plan_miss_tok,       // [n, topk] or nullptr
    int32_t* __restrict__ plan_miss_slot,      // [n, topk] or nullptr
    int32_t* __restrict__ plan_miss_count,     // [n] or nullptr
    int32_t* __restrict__ plan_miss_home,      // [n, topk] or nullptr (0=host,1=gpu)
    int record_plan,
    const int32_t* __restrict__ host_cache_locs,  // [R, host_stride] or nullptr
    int host_stride,
    const int32_t* __restrict__ gpu_cache_locs,   // [R, gpu_stride] or nullptr
    int gpu_stride,
    int skip_gather,                              // 1 = detect+translate only (no gather)
    int64_t item_size_bytes, int hot_slots, int cold_depth, int topk)
{
    extern __shared__ int smem[];
    int* miss_tok = smem;            // topk ints
    int* vic      = smem + topk;     // topk ints

    __shared__ int s_miss_count;
    __shared__ int s_scratch;
    __shared__ int64_t s_tick;

    const int q = blockIdx.x;
    const int start = indptr[q];
    const int end   = indptr[q + 1];
    const int runlen = end - start;
    if (runlen <= 0) {
        if (record_plan && threadIdx.x == 0) plan_miss_count[q] = 0;
        return;  // padding / inactive query token
    }

    const int r = req_slots[q];
    int32_t* st_base       = slot_token + (int64_t)r * hot_slots;
    int64_t* lu_base       = last_used + (int64_t)r * hot_slots;
    int32_t* tts_base      = token_to_slot + (int64_t)r * cold_depth;
    const int32_t* topk_q  = topk_logical + (int64_t)q * topk;

    if (threadIdx.x == 0) {
        s_miss_count = 0;
        s_tick = atomicAdd((unsigned long long*)&recency[r], 1ULL) + 1;
    }
    __syncthreads();
    const int64_t tick = s_tick;

    // Miss detect: refresh hits, collect misses. The indexer guarantees the
    // top-k ids are unique, so a token appears in miss_tok at most once (a
    // duplicate would resolve to two slots — self-healing, never wrong output).
    for (int k = threadIdx.x; k < runlen; k += blockDim.x) {
        int tok = topk_q[k];
        if (tok < 0 || tok >= cold_depth) continue;
        // Skip logical positions with no backing cold-pool row. A token's home is
        // in exactly one tier: req_to_host_pool[r][tok] >= 0 (host-home) or, with
        // the GPU cold tier active, req_to_gpu_pool[r][tok] >= 0 (gpu-home). Both
        // == -1 means the position is outside the request's allocated range;
        // gathering it would dereference a cold pool at row -1 -> GPU memory fault.
        // Such a token has no KV to fetch, so treat it as padding: never make it
        // resident (the translate loop maps any unresolved top-k entry to slot 0).
        // With a dense pool (host_cache_locs == nullptr) every in-range token is
        // host-backed, unchanged from Phase 0.
        bool host_home = host_cache_locs
                             ? (host_cache_locs[(int64_t)r * host_stride + tok] >= 0)
                             : true;
        bool gpu_home = gpu_cache_locs &&
                        (gpu_cache_locs[(int64_t)r * gpu_stride + tok] >= 0);
        if (!host_home && !gpu_home) continue;
        int s = tts_base[tok];
        if (s >= 0) {
            lu_base[s] = tick;  // hit: most recent
        } else {
            int idx = atomicAdd(&s_miss_count, 1);
            // miss_tok holds topk ints; runlen is not otherwise bounded by it,
            // and overrunning it corrupts vic[] (the next shared array).
            if (idx < topk) miss_tok[idx] = tok;
        }
    }
    __syncthreads();
    const int m = s_miss_count;
    if (record_plan && threadIdx.x == 0) plan_miss_count[q] = m;

    if (m > 0) {
        // Find tau* = min tick with count(last_used <= tau) >= m. Hits are at
        // `tick` (the max) so victims (all < tick) never include them.
        //
        int64_t lo = -1, hi = tick;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo) / 2;
            int c = block_count_le(lu_base, hot_slots, mid, &s_scratch);
            if (c >= m) hi = mid; else lo = mid + 1;
        }
        const int64_t tau = lo;

        // Victims: all slots strictly below tau, then fill the remainder from
        // slots exactly at tau (arbitrary order among equals — eviction policy
        // does not affect correctness, only future miss rate).
        if (threadIdx.x == 0) s_scratch = 0;
        __syncthreads();
        for (int s = threadIdx.x; s < hot_slots; s += blockDim.x) {
            if (lu_base[s] < tau) {
                int idx = atomicAdd(&s_scratch, 1);
                // Bounded by construction (count(< tau) < m <= topk) only while
                // the tau search converges; don't let a bad tau scribble past.
                if (idx < topk) vic[idx] = s;
            }
        }
        __syncthreads();
        const int c_lt = s_scratch;
        const int extra = m - c_lt;
        if (threadIdx.x == 0) s_scratch = 0;
        __syncthreads();
        if (extra > 0) {
            for (int s = threadIdx.x; s < hot_slots; s += blockDim.x) {
                if (lu_base[s] == tau) {
                    int idx = atomicAdd(&s_scratch, 1);
                    if (idx < extra) vic[c_lt + idx] = s;
                }
            }
        }
        __syncthreads();

        // Assign each miss token to a distinct victim slot. Miss tokens are not
        // resident and evicted tokens are resident, so the two sets are disjoint
        // and every write below targets a distinct location.
        for (int i = threadIdx.x; i < m; i += blockDim.x) {
            int tok = miss_tok[i];
            int v   = vic[i];
            int evicted = st_base[v];
            if (evicted >= 0) tts_base[evicted] = EMPTY;
            st_base[v] = tok;
            lu_base[v] = tick;
            tts_base[tok] = v;
            if (record_plan) {
                plan_miss_tok[(int64_t)q * topk + i]  = tok;
                plan_miss_slot[(int64_t)q * topk + i] = v;
                // home: gpu-home iff the GPU table backs this token, else host.
                // A miss token is backed (unbacked tokens were skipped above), so
                // exactly one tier claims it.
                int home = (gpu_cache_locs &&
                            gpu_cache_locs[(int64_t)r * gpu_stride + tok] >= 0)
                               ? 1
                               : 0;
                plan_miss_home[(int64_t)q * topk + i] = home;
            }
        }
        __syncthreads();

        // Gather: one warp per miss token. Skipped in dual-source mode
        // (skip_gather=1): the coordinator instead replays the recorded plan with
        // per-home gather passes so gpu-home tokens read the GPU cold pool (their
        // host_cache_locs row is -1 and would fault through this single-base path).
        if (!skip_gather) {
            const int warp_id = threadIdx.x / WARP_SIZE;
            const int lane_id = threadIdx.x % WARP_SIZE;
            const int warps_per_block = blockDim.x / WARP_SIZE;
            for (int i = warp_id; i < m; i += warps_per_block) {
                int tok = miss_tok[i];
                int v   = vic[i];
                const int64_t cold_row =
                    cold_row_of(host_cache_locs, host_stride, r, cold_depth, tok);
                if (cold_row < 0) continue;  // unbacked / out of range
                const int64_t src = cold_row * item_size_bytes;
                const int64_t dst = ((int64_t)r * hot_slots + v) * item_size_bytes;
                transfer_item_warp(lane_id, cold_pool + src, hot_buffer + dst,
                                   item_size_bytes);
            }
        }
        __syncthreads();
    }

    // Translate: every top-k entry now maps to a resident hot-buffer row.
    for (int k = threadIdx.x; k < runlen; k += blockDim.x) {
        int tok = topk_q[k];
        int s = (tok >= 0 && tok < cold_depth) ? tts_base[tok] : 0;
        if (s < 0) s = 0;  // defensive; should not happen after swap
        out_translated[start + k] = (int32_t)((int64_t)r * hot_slots + s);
    }
}

// Backup a freshly generated token's KV into cold pool + a fresh hot slot.
// One block per decode query token.
__global__ void sparsekv_backup_kernel(
    char* __restrict__ cold_pool,          // device-mapped host cold pool, layer
    char* __restrict__ gpu_cold_pool,      // GPU cold tier pool, this layer or nullptr
    char* __restrict__ hot_buffer,         // GPU hot buffer, this layer
    const char* __restrict__ layer_kv,     // GPU layer KV cache (flat rows)
    const int32_t* __restrict__ src_slots, // [n] physical row in layer_kv
    const int32_t* __restrict__ req_slots, // [n]
    const int32_t* __restrict__ logical_pos, // [n]
    int32_t* __restrict__ slot_token,      // [R, hot_slots]
    int64_t* __restrict__ last_used,       // [R, hot_slots]
    int32_t* __restrict__ token_to_slot,   // [R, cold_depth]
    int64_t* __restrict__ recency,         // [R]
    const int32_t* __restrict__ host_cache_locs,  // [R, host_stride] or nullptr
    int host_stride,
    const int32_t* __restrict__ gpu_cache_locs,   // [R, gpu_stride] or nullptr
    int gpu_stride,
    int64_t item_size_bytes, int hot_slots, int cold_depth)
{
    __shared__ int64_t s_min;
    __shared__ int s_argmin;
    __shared__ int64_t s_tick;

    const int q = blockIdx.x;
    const int pos = logical_pos[q];
    const int src = src_slots[q];
    if (pos < 0 || pos >= cold_depth || src < 0) return;

    const int r = req_slots[q];
    int32_t* st_base  = slot_token + (int64_t)r * hot_slots;
    int64_t* lu_base  = last_used + (int64_t)r * hot_slots;
    int32_t* tts_base = token_to_slot + (int64_t)r * cold_depth;

    if (threadIdx.x == 0) {
        s_min = INT64_MAX;
        s_argmin = INT32_MAX;
        s_tick = atomicAdd((unsigned long long*)&recency[r], 1ULL) + 1;
    }
    __syncthreads();

    // Argmin last_used -> victim slot (lowest index breaks ties via atomicMin).
    int64_t loc_min = INT64_MAX;
    int loc_arg = 0;
    for (int s = threadIdx.x; s < hot_slots; s += blockDim.x) {
        int64_t v = lu_base[s];
        if (v < loc_min) { loc_min = v; loc_arg = s; }
    }
    atomicMin((long long*)&s_min, (long long)loc_min);
    __syncthreads();
    if (loc_min == s_min) atomicMin(&s_argmin, loc_arg);
    __syncthreads();
    const int v = s_argmin;
    const int64_t tick = s_tick;

    if (threadIdx.x == 0) {
        int evicted = st_base[v];
        if (evicted >= 0) tts_base[evicted] = EMPTY;
        st_base[v] = pos;
        lu_base[v] = tick;
        tts_base[pos] = v;
    }
    __syncthreads();

    // Copy the new token's KV: layer_kv[src] -> its home cold pool[pos] and
    // hot[r*H1+v]. A new token's home is whichever tier grow_cold_for_new_token
    // backed it in: gpu-home (req_to_gpu_pool[r][pos] >= 0) writes the GPU cold
    // tier, else host-home writes the pinned host cold pool.
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (threadIdx.x < WARP_SIZE) {
        const int64_t kv_off = (int64_t)src * item_size_bytes;
        int64_t gpu_row = -1;
        if (gpu_cold_pool && gpu_cache_locs) {
            gpu_row = (int64_t)gpu_cache_locs[(int64_t)r * gpu_stride + pos];
            if (gpu_row >= g_sparsekv_gpu_cold_rows) {
                atomicAdd(&g_sparsekv_oob_rows, 1ULL);
                gpu_row = -1;  // fall back to the host tier
            }
        }
        if (gpu_row >= 0) {
            const int64_t cold_off = gpu_row * item_size_bytes;
            transfer_item_warp(lane_id, layer_kv + kv_off, gpu_cold_pool + cold_off,
                               item_size_bytes);
        } else {
            const int64_t cold_row =
                cold_row_of(host_cache_locs, host_stride, r, cold_depth, pos);
            if (cold_row >= 0) {  // pos is normally backed; guard against a fault
                const int64_t cold_off = cold_row * item_size_bytes;
                transfer_item_warp(lane_id, layer_kv + kv_off, cold_pool + cold_off,
                                   item_size_bytes);
            }
        }
    } else if (threadIdx.x < 2 * WARP_SIZE) {
        const int64_t kv_off  = (int64_t)src * item_size_bytes;
        const int64_t hot_off = ((int64_t)r * hot_slots + v) * item_size_bytes;
        transfer_item_warp(lane_id, layer_kv + kv_off, hot_buffer + hot_off,
                           item_size_bytes);
    }
}

// Replay a recorded miss plan (from an anchor layer's swap+translate) into a
// shared-index layer's buffers. Pure IO: no miss-detect, no LRU, no state
// writes. One block per decode query token; the hot-slot assignments come from
// the anchor, so the shared layer's slot table stays in lockstep by construction.
__global__ void sparsekv_copy_planned_kernel(
    const char* __restrict__ cold_pool,        // device-mapped host cold pool, layer
    char* __restrict__ hot_buffer,             // GPU hot buffer, this layer
    const int32_t* __restrict__ req_slots,     // [n]
    const int32_t* __restrict__ plan_miss_tok, // [n, topk]
    const int32_t* __restrict__ plan_miss_slot,// [n, topk]
    const int32_t* __restrict__ plan_miss_count,// [n]
    const int32_t* __restrict__ host_cache_locs,  // [R, host_stride] or nullptr
    int host_stride,
    int64_t item_size_bytes, int hot_slots, int cold_depth, int topk)
{
    const int q = blockIdx.x;
    const int m = plan_miss_count[q];
    if (m <= 0) return;
    const int r = req_slots[q];
    const int32_t* tok_q  = plan_miss_tok + (int64_t)q * topk;
    const int32_t* slot_q = plan_miss_slot + (int64_t)q * topk;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    for (int i = warp_id; i < m; i += warps_per_block) {
        const int tok = tok_q[i];
        const int v   = slot_q[i];
        if (tok < 0 || tok >= cold_depth || v < 0 || v >= hot_slots) continue;
        const int64_t cold_row =
            cold_row_of(host_cache_locs, host_stride, r, cold_depth, tok);
        if (cold_row < 0) continue;  // unbacked position: no KV to replay
        const int64_t src = cold_row * item_size_bytes;
        const int64_t dst = ((int64_t)r * hot_slots + v) * item_size_bytes;
        transfer_item_warp(lane_id, cold_pool + src, hot_buffer + dst,
                           item_size_bytes);
    }
}

// Replay one home's share of a recorded miss plan (Design Y dual-source swap).
// Same fixed launch shape and pure-IO semantics as sparsekv_copy_planned_kernel,
// but only gathers misses whose recorded home matches target_home, indirecting
// through that home's translation table (host req_to_host_pool with the
// device-mapped host cold pool as base, or gpu req_to_gpu_pool with the GPU cold
// pool as base). The coordinator issues two calls per layer (host then gpu) so a
// mixed-home top-k lands entirely in the hot buffer. One block per query token.
// Both homes in one pass. The per-home kernel below walks the whole miss list
// and copies only the entries matching its target, so running it twice scans the
// list twice and leaves roughly half the warps in each launch idle on a skip.
// Here every warp copies, and each picks its source from the home the plan
// already recorded for that miss. Destinations are per-miss hot slots, so the
// two homes never write the same row and merging them is safe.
__global__ void sparsekv_gather_planned_dual_kernel(
    const char* __restrict__ host_base,         // pinned host cold pool, this layer
    const char* __restrict__ gpu_base,          // GPU cold tier, this layer (or null)
    char* __restrict__ hot_buffer,              // GPU hot buffer, this layer
    const int32_t* __restrict__ req_slots,      // [n]
    const int32_t* __restrict__ plan_miss_tok,  // [n, topk]
    const int32_t* __restrict__ plan_miss_slot, // [n, topk]
    const int32_t* __restrict__ plan_miss_count,// [n]
    const int32_t* __restrict__ plan_miss_home, // [n, topk] (0=host, 1=gpu)
    const int32_t* __restrict__ host_locs,      // [R, host_stride] or nullptr
    int host_stride,
    const int32_t* __restrict__ gpu_locs,       // [R, gpu_stride] or nullptr
    int gpu_stride,
    int64_t item_size_bytes, int hot_slots, int cold_depth, int topk)
{
    const int q = blockIdx.x;
    const int m = plan_miss_count[q];
    if (m <= 0) return;
    const int r = req_slots[q];
    const int32_t* tok_q  = plan_miss_tok + (int64_t)q * topk;
    const int32_t* slot_q = plan_miss_slot + (int64_t)q * topk;
    const int32_t* home_q = plan_miss_home + (int64_t)q * topk;

    // gridDim.y spreads one query's miss list across many blocks. With one
    // block per query the whole gather ran on `n` blocks — 16 of 256 CUs at a
    // decode batch of 16, each warp walking a hundred rows in series — so the
    // kernel was parallelism-bound, not bandwidth-bound.
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int stride_i = gridDim.y * warps_per_block;
    for (int i = blockIdx.y * warps_per_block + warp_id; i < m; i += stride_i) {
        const int tok = tok_q[i];
        const int v   = slot_q[i];
        if (tok < 0 || tok >= cold_depth || v < 0 || v >= hot_slots) continue;
        const bool gpu_home = (home_q[i] != 0);
        const char* base = gpu_home ? gpu_base : host_base;
        if (!base) continue;
        const int64_t cold_row =
            gpu_home ? cold_row_bounded(gpu_locs, gpu_stride, r, cold_depth, tok,
                                        g_sparsekv_gpu_cold_rows)
                     : cold_row_bounded(host_locs, host_stride, r, cold_depth, tok,
                                        g_sparsekv_cold_rows);
        if (cold_row < 0) continue;  // unbacked in its home: nothing to gather
        const int64_t src = cold_row * item_size_bytes;
        const int64_t dst = ((int64_t)r * hot_slots + v) * item_size_bytes;
        transfer_item_warp(lane_id, base + src, hot_buffer + dst, item_size_bytes);
    }
}

__global__ void sparsekv_gather_planned_kernel(
    const char* __restrict__ base_ptr,          // this home's cold pool base, layer
    char* __restrict__ hot_buffer,              // GPU hot buffer, this layer
    const int32_t* __restrict__ req_slots,      // [n]
    const int32_t* __restrict__ plan_miss_tok,  // [n, topk]
    const int32_t* __restrict__ plan_miss_slot, // [n, topk]
    const int32_t* __restrict__ plan_miss_count,// [n]
    const int32_t* __restrict__ plan_miss_home, // [n, topk] (0=host, 1=gpu)
    int target_home,                            // gather only misses with this home
    const int32_t* __restrict__ cache_locs,     // this home's [R, cache_stride] table
    int cache_stride,
    int64_t item_size_bytes, int hot_slots, int cold_depth, int topk)
{
    const int q = blockIdx.x;
    const int m = plan_miss_count[q];
    if (m <= 0) return;
    const int r = req_slots[q];
    const int32_t* tok_q  = plan_miss_tok + (int64_t)q * topk;
    const int32_t* slot_q = plan_miss_slot + (int64_t)q * topk;
    const int32_t* home_q = plan_miss_home + (int64_t)q * topk;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    for (int i = warp_id; i < m; i += warps_per_block) {
        if (home_q[i] != target_home) continue;  // other home's gather pass owns it
        const int tok = tok_q[i];
        const int v   = slot_q[i];
        if (tok < 0 || tok >= cold_depth || v < 0 || v >= hot_slots) continue;
        const int64_t cold_row =
            cold_row_of(cache_locs, cache_stride, r, cold_depth, tok);
        if (cold_row < 0) continue;  // unbacked in this home: nothing to gather
        const int64_t src = cold_row * item_size_bytes;
        const int64_t dst = ((int64_t)r * hot_slots + v) * item_size_bytes;
        transfer_item_warp(lane_id, base_ptr + src, hot_buffer + dst,
                           item_size_bytes);
    }
}

// Backup a shared-index layer's freshly generated token into the hot slot the
// anchor already assigned to it (token_to_slot is the ANCHOR's table). Data
// only: no LRU/recency writes, so the group's shared slot table is untouched.
// One block per decode query token.
__global__ void sparsekv_backup_into_assigned_kernel(
    char* __restrict__ cold_pool,          // device-mapped host cold pool, layer
    char* __restrict__ gpu_cold_pool,      // GPU cold tier pool, this layer or nullptr
    char* __restrict__ hot_buffer,         // GPU hot buffer, this layer
    const char* __restrict__ layer_kv,     // GPU layer KV cache (flat rows)
    const int32_t* __restrict__ src_slots, // [n] physical row in layer_kv
    const int32_t* __restrict__ req_slots, // [n]
    const int32_t* __restrict__ logical_pos, // [n]
    const int32_t* __restrict__ token_to_slot, // [R, cold_depth] (anchor's)
    const int32_t* __restrict__ host_cache_locs,  // [R, host_stride] or nullptr
    int host_stride,
    const int32_t* __restrict__ gpu_cache_locs,   // [R, gpu_stride] or nullptr
    int gpu_stride,
    int64_t item_size_bytes, int hot_slots, int cold_depth)
{
    const int q = blockIdx.x;
    const int pos = logical_pos[q];
    const int src = src_slots[q];
    if (pos < 0 || pos >= cold_depth || src < 0) return;
    const int r = req_slots[q];
    const int v = token_to_slot[(int64_t)r * cold_depth + pos];
    if (v < 0 || v >= hot_slots) return;  // anchor must have made pos resident

    const int lane_id = threadIdx.x % WARP_SIZE;
    if (threadIdx.x < WARP_SIZE) {
        // Write this layer's KV to the token's home tier (same home the anchor's
        // grow_cold_for_new_token chose): gpu-home -> GPU cold tier, else host.
        const int64_t kv_off = (int64_t)src * item_size_bytes;
        int64_t gpu_row = -1;
        if (gpu_cold_pool && gpu_cache_locs) {
            gpu_row = (int64_t)gpu_cache_locs[(int64_t)r * gpu_stride + pos];
            if (gpu_row >= g_sparsekv_gpu_cold_rows) {
                atomicAdd(&g_sparsekv_oob_rows, 1ULL);
                gpu_row = -1;  // fall back to the host tier
            }
        }
        if (gpu_row >= 0) {
            const int64_t cold_off = gpu_row * item_size_bytes;
            transfer_item_warp(lane_id, layer_kv + kv_off, gpu_cold_pool + cold_off,
                               item_size_bytes);
        } else {
            const int64_t cold_row =
                cold_row_of(host_cache_locs, host_stride, r, cold_depth, pos);
            if (cold_row >= 0) {  // pos is normally backed; guard against a fault
                const int64_t cold_off = cold_row * item_size_bytes;
                transfer_item_warp(lane_id, layer_kv + kv_off, cold_pool + cold_off,
                                   item_size_bytes);
            }
        }
    } else if (threadIdx.x < 2 * WARP_SIZE) {
        const int64_t kv_off  = (int64_t)src * item_size_bytes;
        const int64_t hot_off = ((int64_t)r * hot_slots + v) * item_size_bytes;
        transfer_item_warp(lane_id, layer_kv + kv_off, hot_buffer + hot_off,
                           item_size_bytes);
    }
}

}  // namespace

// Resolve the optional paged-host translation table to a device pointer. Empty /
// undefined tensor or host_stride <= 0 means "dense cold pool" -> nullptr, and the
// kernels fall back to r*cold_depth+tok addressing (byte-identical to pre-paging).
static const int32_t* sparsekv_host_cache_locs_ptr(at::Tensor host_cache_locs,
                                                   int64_t host_stride)
{
    if (host_stride <= 0 || !host_cache_locs.defined() ||
        host_cache_locs.numel() == 0) {
        return nullptr;
    }
    TORCH_CHECK(host_cache_locs.is_cuda(), "host_cache_locs must be CUDA");
    TORCH_CHECK(host_cache_locs.scalar_type() == at::kInt,
                "host_cache_locs must be int32");
    return host_cache_locs.data_ptr<int32_t>();
}

void sparsekv_set_pool_rows(int64_t cold_rows, int64_t gpu_cold_rows)
{
    const int64_t host_rows = cold_rows > 0 ? cold_rows : INT64_MAX;
    const int64_t gpu_rows  = gpu_cold_rows > 0 ? gpu_cold_rows : INT64_MAX;
    hipError_t e = hipMemcpyToSymbol(HIP_SYMBOL(g_sparsekv_cold_rows), &host_rows,
                                     sizeof(int64_t));
    TORCH_CHECK(e == hipSuccess,
                "sparsekv_set_pool_rows(cold_rows) failed: ", hipGetErrorString(e));
    e = hipMemcpyToSymbol(HIP_SYMBOL(g_sparsekv_gpu_cold_rows), &gpu_rows,
                          sizeof(int64_t));
    TORCH_CHECK(e == hipSuccess,
                "sparsekv_set_pool_rows(gpu_cold_rows) failed: ",
                hipGetErrorString(e));
}

int64_t sparsekv_take_oob_row_count()
{
    unsigned long long count = 0ULL;
    hipError_t e = hipMemcpyFromSymbol(&count, HIP_SYMBOL(g_sparsekv_oob_rows),
                                       sizeof(unsigned long long));
    TORCH_CHECK(e == hipSuccess,
                "sparsekv_take_oob_row_count read failed: ", hipGetErrorString(e));
    if (count != 0ULL) {
        const unsigned long long zero = 0ULL;
        e = hipMemcpyToSymbol(HIP_SYMBOL(g_sparsekv_oob_rows), &zero,
                              sizeof(unsigned long long));
        TORCH_CHECK(e == hipSuccess,
                    "sparsekv_take_oob_row_count reset failed: ",
                    hipGetErrorString(e));
    }
    return (int64_t)count;
}

int64_t sparsekv_host_get_device_pointer(at::Tensor pinned_host_tensor)
{
    TORCH_CHECK(pinned_host_tensor.is_pinned(),
                "sparsekv_host_get_device_pointer: tensor must be pinned host memory");
    void* host_ptr = pinned_host_tensor.data_ptr();
    void* dev_ptr = nullptr;
    hipError_t e = hipHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    TORCH_CHECK(e == hipSuccess,
                "hipHostGetDevicePointer failed: ", hipGetErrorString(e));
    return reinterpret_cast<int64_t>(dev_ptr);
}

void sparsekv_swap_in(int64_t cold_pool_dev_ptr, at::Tensor hot_buffer,
                      at::Tensor src_locs, at::Tensor dst_locs,
                      int64_t item_size_bytes)
{
    const int num_misses = (int)src_locs.numel();
    if (num_misses == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda(), "hot_buffer must be CUDA");
    TORCH_CHECK(src_locs.is_cuda() && dst_locs.is_cuda(),
                "src_locs/dst_locs must be CUDA");
    TORCH_CHECK(src_locs.scalar_type() == at::kInt &&
                    dst_locs.scalar_type() == at::kInt,
                "src_locs/dst_locs must be int32");
    TORCH_CHECK(dst_locs.numel() == num_misses, "src/dst length mismatch");

    const char* host_cache = reinterpret_cast<const char*>(cold_pool_dev_ptr);
    char* dev_buf = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const int grid = (num_misses * WARP_SIZE + BLOCK - 1) / BLOCK;
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_gather_kernel<<<dim3(grid), dim3(BLOCK), 0, stream>>>(
        host_cache, dev_buf, src_locs.data_ptr<int32_t>(),
        dst_locs.data_ptr<int32_t>(), num_misses, item_size_bytes);
}

void sparsekv_swap_and_translate(int64_t cold_pool_dev_ptr, at::Tensor hot_buffer,
                                 at::Tensor topk_logical, at::Tensor indptr,
                                 at::Tensor req_slots, at::Tensor slot_token,
                                 at::Tensor last_used, at::Tensor token_to_slot,
                                 at::Tensor recency, at::Tensor out_translated,
                                 at::Tensor host_cache_locs, int64_t host_stride,
                                 at::Tensor gpu_cache_locs, int64_t gpu_stride,
                                 int64_t skip_gather,
                                 int64_t item_size_bytes, int64_t hot_slots,
                                 int64_t cold_depth, int64_t topk)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    // The non-record entry point records no plan, so skipping the inline gather
    // would leave the assigned slots unfilled with nothing to replay them. Dual-
    // source callers must use sparsekv_swap_and_translate_record instead.
    TORCH_CHECK(skip_gather == 0,
                "sparsekv_swap_and_translate cannot skip_gather (no plan recorded "
                "to replay); use sparsekv_swap_and_translate_record for dual-source");
    TORCH_CHECK(hot_buffer.is_cuda() && topk_logical.is_cuda() &&
                    indptr.is_cuda() && req_slots.is_cuda() &&
                    slot_token.is_cuda() && last_used.is_cuda() &&
                    token_to_slot.is_cuda() && recency.is_cuda() &&
                    out_translated.is_cuda(),
                "all sparsekv_swap_and_translate tensors must be CUDA");
    TORCH_CHECK(last_used.scalar_type() == at::kLong &&
                    recency.scalar_type() == at::kLong,
                "last_used/recency must be int64");

    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* cold = reinterpret_cast<const char*>(cold_pool_dev_ptr);
    const int32_t* hcl = sparsekv_host_cache_locs_ptr(host_cache_locs, host_stride);
    const int32_t* gcl = sparsekv_host_cache_locs_ptr(gpu_cache_locs, gpu_stride);
    const size_t shmem = (size_t)(2 * topk) * sizeof(int);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_swap_and_translate_kernel<<<dim3(n), dim3(BLOCK), shmem, stream>>>(
        cold, hot, topk_logical.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        req_slots.data_ptr<int32_t>(), slot_token.data_ptr<int32_t>(),
        last_used.data_ptr<int64_t>(), token_to_slot.data_ptr<int32_t>(),
        recency.data_ptr<int64_t>(), out_translated.data_ptr<int32_t>(),
        nullptr, nullptr, nullptr, nullptr, 0, hcl, (int)host_stride,
        gcl, (int)gpu_stride, (int)skip_gather,
        item_size_bytes, (int)hot_slots, (int)cold_depth, (int)topk);
}

void sparsekv_swap_and_translate_record(
    int64_t cold_pool_dev_ptr, at::Tensor hot_buffer, at::Tensor topk_logical,
    at::Tensor indptr, at::Tensor req_slots, at::Tensor slot_token,
    at::Tensor last_used, at::Tensor token_to_slot, at::Tensor recency,
    at::Tensor out_translated, at::Tensor plan_miss_tok, at::Tensor plan_miss_slot,
    at::Tensor plan_miss_count, at::Tensor plan_miss_home,
    at::Tensor host_cache_locs, int64_t host_stride,
    at::Tensor gpu_cache_locs, int64_t gpu_stride, int64_t skip_gather,
    int64_t item_size_bytes, int64_t hot_slots,
    int64_t cold_depth, int64_t topk)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda() && topk_logical.is_cuda() &&
                    indptr.is_cuda() && req_slots.is_cuda() &&
                    slot_token.is_cuda() && last_used.is_cuda() &&
                    token_to_slot.is_cuda() && recency.is_cuda() &&
                    out_translated.is_cuda() && plan_miss_tok.is_cuda() &&
                    plan_miss_slot.is_cuda() && plan_miss_count.is_cuda() &&
                    plan_miss_home.is_cuda(),
                "all sparsekv_swap_and_translate_record tensors must be CUDA");
    TORCH_CHECK(last_used.scalar_type() == at::kLong &&
                    recency.scalar_type() == at::kLong,
                "last_used/recency must be int64");

    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* cold = reinterpret_cast<const char*>(cold_pool_dev_ptr);
    const int32_t* hcl = sparsekv_host_cache_locs_ptr(host_cache_locs, host_stride);
    const int32_t* gcl = sparsekv_host_cache_locs_ptr(gpu_cache_locs, gpu_stride);
    const size_t shmem = (size_t)(2 * topk) * sizeof(int);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_swap_and_translate_kernel<<<dim3(n), dim3(BLOCK), shmem, stream>>>(
        cold, hot, topk_logical.data_ptr<int32_t>(), indptr.data_ptr<int32_t>(),
        req_slots.data_ptr<int32_t>(), slot_token.data_ptr<int32_t>(),
        last_used.data_ptr<int64_t>(), token_to_slot.data_ptr<int32_t>(),
        recency.data_ptr<int64_t>(), out_translated.data_ptr<int32_t>(),
        plan_miss_tok.data_ptr<int32_t>(), plan_miss_slot.data_ptr<int32_t>(),
        plan_miss_count.data_ptr<int32_t>(), plan_miss_home.data_ptr<int32_t>(),
        1, hcl, (int)host_stride, gcl, (int)gpu_stride, (int)skip_gather,
        item_size_bytes, (int)hot_slots, (int)cold_depth, (int)topk);
}

void sparsekv_copy_planned(int64_t cold_pool_dev_ptr, at::Tensor hot_buffer,
                           at::Tensor req_slots, at::Tensor plan_miss_tok,
                           at::Tensor plan_miss_slot, at::Tensor plan_miss_count,
                           at::Tensor host_cache_locs, int64_t host_stride,
                           int64_t item_size_bytes, int64_t hot_slots,
                           int64_t cold_depth, int64_t topk)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda() && req_slots.is_cuda() &&
                    plan_miss_tok.is_cuda() && plan_miss_slot.is_cuda() &&
                    plan_miss_count.is_cuda(),
                "all sparsekv_copy_planned tensors must be CUDA");
    TORCH_CHECK(req_slots.scalar_type() == at::kInt &&
                    plan_miss_tok.scalar_type() == at::kInt &&
                    plan_miss_slot.scalar_type() == at::kInt &&
                    plan_miss_count.scalar_type() == at::kInt,
                "sparsekv_copy_planned req_slots/plan tensors must be int32");

    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* cold = reinterpret_cast<const char*>(cold_pool_dev_ptr);
    const int32_t* hcl = sparsekv_host_cache_locs_ptr(host_cache_locs, host_stride);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_copy_planned_kernel<<<dim3(n), dim3(BLOCK), 0, stream>>>(
        cold, hot, req_slots.data_ptr<int32_t>(),
        plan_miss_tok.data_ptr<int32_t>(), plan_miss_slot.data_ptr<int32_t>(),
        plan_miss_count.data_ptr<int32_t>(), hcl, (int)host_stride,
        item_size_bytes, (int)hot_slots, (int)cold_depth, (int)topk);
}

void sparsekv_gather_planned_dual(int64_t host_base_ptr, int64_t gpu_base_ptr,
                                  at::Tensor hot_buffer, at::Tensor req_slots,
                                  at::Tensor plan_miss_tok, at::Tensor plan_miss_slot,
                                  at::Tensor plan_miss_count, at::Tensor plan_miss_home,
                                  at::Tensor host_cache_locs, int64_t host_stride,
                                  at::Tensor gpu_cache_locs, int64_t gpu_stride,
                                  int64_t item_size_bytes, int64_t hot_slots,
                                  int64_t cold_depth, int64_t topk)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda() && req_slots.is_cuda() &&
                    plan_miss_tok.is_cuda() && plan_miss_slot.is_cuda() &&
                    plan_miss_count.is_cuda() && plan_miss_home.is_cuda(),
                "all sparsekv_gather_planned_dual tensors must be CUDA");
    TORCH_CHECK(req_slots.scalar_type() == at::kInt &&
                    plan_miss_tok.scalar_type() == at::kInt &&
                    plan_miss_slot.scalar_type() == at::kInt &&
                    plan_miss_count.scalar_type() == at::kInt &&
                    plan_miss_home.scalar_type() == at::kInt,
                "sparsekv_gather_planned_dual req_slots/plan tensors must be int32");

    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* host_base = reinterpret_cast<const char*>(host_base_ptr);
    const char* gpu_base  = reinterpret_cast<const char*>(gpu_base_ptr);
    const int32_t* hl = sparsekv_host_cache_locs_ptr(host_cache_locs, host_stride);
    const int32_t* gl = sparsekv_host_cache_locs_ptr(gpu_cache_locs, gpu_stride);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    // Enough blocks to keep the device busy; capped by the miss list so a small
    // top-k does not launch blocks that can only exit.
    constexpr int WPB = BLOCK / WARP_SIZE;
    const int max_chunks = (int)((topk + WPB - 1) / WPB);
    const int gy = std::max(1, std::min(max_chunks, (GATHER_TARGET_BLOCKS + n - 1) / n));
    sparsekv_gather_planned_dual_kernel<<<dim3(n, gy), dim3(BLOCK), 0, stream>>>(
        host_base, gpu_base, hot, req_slots.data_ptr<int32_t>(),
        plan_miss_tok.data_ptr<int32_t>(), plan_miss_slot.data_ptr<int32_t>(),
        plan_miss_count.data_ptr<int32_t>(), plan_miss_home.data_ptr<int32_t>(),
        hl, (int)host_stride, gl, (int)gpu_stride,
        item_size_bytes, (int)hot_slots, (int)cold_depth, (int)topk);
}

void sparsekv_gather_planned(int64_t base_dev_ptr, at::Tensor hot_buffer,
                             at::Tensor req_slots, at::Tensor plan_miss_tok,
                             at::Tensor plan_miss_slot, at::Tensor plan_miss_count,
                             at::Tensor plan_miss_home, int64_t target_home,
                             at::Tensor cache_locs, int64_t cache_stride,
                             int64_t item_size_bytes, int64_t hot_slots,
                             int64_t cold_depth, int64_t topk)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda() && req_slots.is_cuda() &&
                    plan_miss_tok.is_cuda() && plan_miss_slot.is_cuda() &&
                    plan_miss_count.is_cuda() && plan_miss_home.is_cuda(),
                "all sparsekv_gather_planned tensors must be CUDA");
    TORCH_CHECK(req_slots.scalar_type() == at::kInt &&
                    plan_miss_tok.scalar_type() == at::kInt &&
                    plan_miss_slot.scalar_type() == at::kInt &&
                    plan_miss_count.scalar_type() == at::kInt &&
                    plan_miss_home.scalar_type() == at::kInt,
                "sparsekv_gather_planned req_slots/plan tensors must be int32");

    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* base = reinterpret_cast<const char*>(base_dev_ptr);
    const int32_t* cl = sparsekv_host_cache_locs_ptr(cache_locs, cache_stride);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_gather_planned_kernel<<<dim3(n), dim3(BLOCK), 0, stream>>>(
        base, hot, req_slots.data_ptr<int32_t>(),
        plan_miss_tok.data_ptr<int32_t>(), plan_miss_slot.data_ptr<int32_t>(),
        plan_miss_count.data_ptr<int32_t>(), plan_miss_home.data_ptr<int32_t>(),
        (int)target_home, cl, (int)cache_stride,
        item_size_bytes, (int)hot_slots, (int)cold_depth, (int)topk);
}

void sparsekv_backup_into_assigned(int64_t cold_pool_dev_ptr,
                                   int64_t gpu_cold_pool_ptr, at::Tensor hot_buffer,
                                   at::Tensor layer_kv, at::Tensor src_slots,
                                   at::Tensor req_slots, at::Tensor logical_pos,
                                   at::Tensor token_to_slot,
                                   at::Tensor host_cache_locs, int64_t host_stride,
                                   at::Tensor gpu_cache_locs, int64_t gpu_stride,
                                   int64_t item_size_bytes,
                                   int64_t hot_slots, int64_t cold_depth)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda() && layer_kv.is_cuda() &&
                    token_to_slot.is_cuda(),
                "hot_buffer/layer_kv/token_to_slot must be CUDA");
    TORCH_CHECK(src_slots.scalar_type() == at::kInt &&
                    req_slots.scalar_type() == at::kInt &&
                    logical_pos.scalar_type() == at::kInt &&
                    token_to_slot.scalar_type() == at::kInt,
                "sparsekv_backup_into_assigned int32 index tensors required");
    char* cold = reinterpret_cast<char*>(cold_pool_dev_ptr);
    char* gpu_cold = reinterpret_cast<char*>(gpu_cold_pool_ptr);
    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* kv = reinterpret_cast<const char*>(layer_kv.data_ptr());
    const int32_t* hcl = sparsekv_host_cache_locs_ptr(host_cache_locs, host_stride);
    const int32_t* gcl = sparsekv_host_cache_locs_ptr(gpu_cache_locs, gpu_stride);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_backup_into_assigned_kernel<<<dim3(n), dim3(BLOCK), 0, stream>>>(
        cold, gpu_cold, hot, kv, src_slots.data_ptr<int32_t>(),
        req_slots.data_ptr<int32_t>(), logical_pos.data_ptr<int32_t>(),
        token_to_slot.data_ptr<int32_t>(), hcl, (int)host_stride,
        gcl, (int)gpu_stride,
        item_size_bytes, (int)hot_slots, (int)cold_depth);
}

void sparsekv_backup_new_token(int64_t cold_pool_dev_ptr, int64_t gpu_cold_pool_ptr,
                               at::Tensor hot_buffer,
                               at::Tensor layer_kv, at::Tensor src_slots,
                               at::Tensor req_slots, at::Tensor logical_pos,
                               at::Tensor slot_token, at::Tensor last_used,
                               at::Tensor token_to_slot, at::Tensor recency,
                               at::Tensor host_cache_locs, int64_t host_stride,
                               at::Tensor gpu_cache_locs, int64_t gpu_stride,
                               int64_t item_size_bytes, int64_t hot_slots,
                               int64_t cold_depth)
{
    const int n = (int)req_slots.numel();
    if (n == 0) return;
    TORCH_CHECK(item_size_bytes % 8 == 0,
                "item_size_bytes must be a multiple of 8 (uint64 word copy)");
    TORCH_CHECK(hot_buffer.is_cuda() && layer_kv.is_cuda(),
                "hot_buffer/layer_kv must be CUDA");
    char* cold = reinterpret_cast<char*>(cold_pool_dev_ptr);
    char* gpu_cold = reinterpret_cast<char*>(gpu_cold_pool_ptr);
    char* hot = reinterpret_cast<char*>(hot_buffer.data_ptr());
    const char* kv = reinterpret_cast<const char*>(layer_kv.data_ptr());
    const int32_t* hcl = sparsekv_host_cache_locs_ptr(host_cache_locs, host_stride);
    const int32_t* gcl = sparsekv_host_cache_locs_ptr(gpu_cache_locs, gpu_stride);
    hipStream_t stream = at::hip::getCurrentHIPStream();
    sparsekv_backup_kernel<<<dim3(n), dim3(BLOCK), 0, stream>>>(
        cold, gpu_cold, hot, kv, src_slots.data_ptr<int32_t>(),
        req_slots.data_ptr<int32_t>(), logical_pos.data_ptr<int32_t>(),
        slot_token.data_ptr<int32_t>(), last_used.data_ptr<int64_t>(),
        token_to_slot.data_ptr<int32_t>(), recency.data_ptr<int64_t>(),
        hcl, (int)host_stride, gcl, (int)gpu_stride,
        item_size_bytes, (int)hot_slots, (int)cold_depth);
}

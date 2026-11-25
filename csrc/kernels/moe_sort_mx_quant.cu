// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused MoE sort + MXFP8 activation quant, one launch and no grid barrier.
//
// Replaces moe_sorting followed by fused_dynamic_mx_quant_moe_sort, which
// re-quantizes every token once per route (or, at larger M, a per-token quant
// plus a separate scale sort).
//
// top-k ids arrive as input, so the routing map is known before launch and no
// block has to wait for another. Every block rebuilds the expert histogram and
// prefix scan from topk_ids (a few KB of L2 reads), plus a count of the routes
// that precede its own tokens. That fixes the sorted position of each of its
// routes exactly -- ascending routed index within an expert -- so a block can
// quantize its tokens once and write each e8m0 scale straight into the
// swizzled layout at every sorted row the token occupies.
//
// The kernel is latency bound, so every global load is issued at entry and
// there are three barriers in all. Per block:
//   issue topk_ids, the block's own routes and its token rows
//   histogram over all routes; prefix count over routes before its tokens
//   wave 0: scan -> per-expert padded offsets, rank the block's own routes ->
//     sorted_ids / sorted_weights
//   meanwhile the other waves zero their slice of moe_buf
//   quantize its tokens -> out[token], scales -> out_scale[sorted row]
//   expert-partitioned: padding rows + sorted_expert_ids
//   grid-stride: tail past num_valid
// Global stores come after the last wait on a load, since vmcnt retires in
// order and would make that wait drain them.

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h" // load_vector_nbytes, store_vector
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "hip_reduce.h" // multithread_reduce
#include "moe_sort_mx_quant.h"
#include "mx_quant_utils.h"
#include "opus/opus.hpp"
#include "quant_amax_floor.h"
#include "warp_sort.h" // aiter::mov_dpp_

namespace aiter {
namespace msq {

static constexpr int kWaveSize = 64;
// bf16 elements each thread quantizes per token row: cols == kThreads * kTD.
static constexpr int kTD        = 16;
static constexpr int kGroupSize = 32;
// A block's own routes are ranked by one wave, one route per lane.
static constexpr int kMaxRoutesPerBlock = kWaveSize;
// Bounds E at kThreads * this (LDS histogram size).
static constexpr int kMaxExpertsPerThread = 4;
// Wave 0 scans E experts as 64 contiguous runs of up to kEpl each.
static constexpr int kSmallEpl = 8;
static constexpr int kLargeEpl = 32;
// Token rows held in registers at once; the first group is issued at entry.
static constexpr int kRowPf = 4;

__device__ __forceinline__ int pack_id(int token_id, int topk_id)
{
    return static_cast<int>((static_cast<uint32_t>(token_id) & 0x00ffffffu) |
                            ((static_cast<uint32_t>(topk_id) & 0xffu) << 24));
}

// Inclusive add-scan over a wave's 64 lanes.
__device__ __forceinline__ int wave_scan_incl(int v, int lane_id)
{
    v += aiter::mov_dpp_(v, opus::number<0x111>{}); // row_shr:1
    v += aiter::mov_dpp_(v, opus::number<0x112>{}); // row_shr:2
    v += aiter::mov_dpp_(v, opus::number<0x114>{}); // row_shr:4
    v += aiter::mov_dpp_(v, opus::number<0x118>{}); // row_shr:8
    const int r0 = __builtin_amdgcn_readlane(v, 15);
    const int r1 = __builtin_amdgcn_readlane(v, 31);
    const int r2 = __builtin_amdgcn_readlane(v, 47);
    return v + (lane_id >= 16 ? r0 : 0) + (lane_id >= 32 ? r1 : 0) + (lane_id >= 48 ? r2 : 0);
}

template <int kThreads, int kEpl, typename DTYPE_O>
__global__ void __launch_bounds__(kThreads)
    moe_sort_mx_quant_kernel(const opus::bf16_t* __restrict__ hidden, // [M, cols]
                             const int* __restrict__ topk_ids,        // [M, topk]
                             const float* __restrict__ topk_weights,  // [M, topk]
                             int* __restrict__ sorted_ids,            // [max_rows]
                             float* __restrict__ sorted_weights,      // [max_rows]
                             int* __restrict__ sorted_expert_ids,     // [max_units]
                             int* __restrict__ num_valid_ids,         // [2]
                             DTYPE_O* __restrict__ out,               // [M, cols]
                             uint8_t* __restrict__ out_scale,         // swizzled e8m0
                             opus::bf16_t* __restrict__ moe_buf,      // [M, cols] or nullptr
                             int64_t moe_buf_elems,
                             int M,
                             int E,
                             int topk,
                             int log2u,
                             int tokens_per_block,
                             int max_rows,
                             int max_units)
{
    static_assert(std::is_same_v<DTYPE_O, opus::fp8_t>, "only MXFP8 output is supported");
    constexpr int cols       = kThreads * kTD;
    constexpr int kNumWaves  = kThreads / kWaveSize;
    static_assert(kNumWaves >= 2, "wave 0 builds metadata while the rest zero moe_buf");
    constexpr int scaleN     = cols / kGroupSize;
    constexpr int scaleN_pad = (scaleN + 7) / 8 * 8;
    constexpr int kThrPerGrp = kGroupSize / kTD;
#if defined(__gfx942__)
    constexpr aiter::MxDtype kHwFp8Dtype = aiter::MxDtype::FP8_E4M3_FNUZ;
#else
    constexpr aiter::MxDtype kHwFp8Dtype = aiter::MxDtype::FP8_E4M3;
#endif

    __shared__ int s_cnt[kThreads * kMaxExpertsPerThread];
    __shared__ int s_pre[kThreads * kMaxExpertsPerThread];
    // Experts past these bounds write to the spare last entry.
    constexpr int kSpareExpert = kThreads * kMaxExpertsPerThread;
    __shared__ int s_off[kSpareExpert + 1];
    __shared__ int s_pos[kMaxRoutesPerBlock];
    __shared__ __attribute__((aligned(16))) int s_own[kMaxRoutesPerBlock];
    __shared__ int s_num_valid;

    const int tid     = threadIdx.x;
    const int lane_id = tid & (kWaveSize - 1);
    const int wave_id = tid / kWaveSize;
    const int unit    = 1 << log2u;

    const int t_begin     = blockIdx.x * tokens_per_block;
    const int t_end       = min(M, t_begin + tokens_per_block);
    const int route_begin = t_begin * topk;
    const int num_routes  = M * topk;

    using vec_i = opus::vector_t<opus::bf16_t, kTD>;
    using vec_f = opus::vector_t<float, kTD>;

    auto load_row = [&](int t) {
        auto buf =
            opus::make_gmem<opus::bf16_t>(hidden + (int64_t)t * cols, cols * sizeof(opus::bf16_t));
        return load_vector_nbytes<opus::bf16_t, kTD, 16, RT>(buf, tid * kTD);
    };

    // Loads are clamped rather than predicated so the counts stay static and a
    // wait covers exactly the loads it needs; duplicates are discarded. The
    // histogram ids go first (pinned there) since the histogram waits on
    // nothing else.
    constexpr int kBatch = 16;
    // Batch slots and rows past what the block needs are skipped by uniform
    // branches.
    int ids[kBatch];
#pragma unroll
    for(int u = 0; u < kBatch; ++u)
        if(u == 0 || u * kThreads < num_routes)
            ids[u] = topk_ids[min(tid + u * kThreads, num_routes - 1)];
    __builtin_amdgcn_sched_barrier(0);
    // Every wave loads the block's own routes to keep the counts uniform; wave
    // 0 ranks them.
    const int own_routes = (t_end - t_begin) * topk;
    const int own_r      = route_begin + min(lane_id, own_routes - 1);
    const int own_e      = topk_ids[own_r];
    const float own_w    = topk_weights[own_r];
    vec_i rows[kRowPf];
#pragma unroll
    for(int i = 0; i < kRowPf; ++i)
        if(i == 0 || t_begin + i < t_end)
            rows[i] = load_row(t_begin + i);

    for(int e = tid; e < E; e += kThreads)
    {
        s_cnt[e] = 0;
        s_pre[e] = 0;
    }
    __syncthreads();

    for(int base = tid;;)
    {
#pragma unroll
        for(int u = 0; u < kBatch; ++u)
        {
            if(u > 0 && base - tid + u * kThreads >= num_routes)
                break;
            const int i = base + u * kThreads;
            const int e = ids[u];
            if(i < num_routes && e >= 0 && e < E)
            {
                atomicAdd(&s_cnt[e], 1);
                if(i < route_begin)
                    atomicAdd(&s_pre[e], 1);
            }
        }
        base += kBatch * kThreads;
        if(base >= num_routes)
            break;
#pragma unroll
        for(int u = 0; u < kBatch; ++u)
            ids[u] = topk_ids[min(base + u * kThreads, num_routes - 1)];
    }
    __syncthreads();
    // Land every load issued at entry before the first global store. No later
    // wait on a load then drains stores (vmcnt retires in order), including
    // the waits the compiler adds before reusing a row prefetch's registers.
    __builtin_amdgcn_s_waitcnt(0x0f70); // vmcnt(0)

    // Experts split into 64 contiguous runs, one per lane, so one wave scan
    // orders them all. The unrolled loops stay branch-free so their LDS
    // accesses pipeline: slots j >= epl count as empty and write to a spare
    // entry.
    const int epl     = (E + kWaveSize - 1) / kWaveSize;
    const int e_first = lane_id * epl;
    auto load_counts  = [&](int (&cnt)[kEpl]) {
#pragma unroll
        for(int j = 0; j < kEpl; ++j)
        {
            const int e  = e_first + j;
            const bool v = j < epl && e < E;
            cnt[j]       = s_cnt[v ? e : 0];
            cnt[j]       = v ? cnt[j] : 0;
        }
    };

    const int scale_k = tid / kThrPerGrp;
    // Quantizes the tokens of one row group and stores them to out; returns
    // the e8m0 scale bytes of this thread's group, token i in byte i.
    auto quant_rows = [&](int t0) {
        uint32_t bytes = 0;
#pragma unroll
        for(int i = 0; i < kRowPf; ++i)
        {
            const int t = t0 + i;
            if(t >= t_end)
                break;
            vec_f vf;
            float amax = aiter::kAmaxFloor;
#pragma unroll
            for(int j = 0; j < kTD; ++j)
            {
                vf[j] = static_cast<float>(rows[i][j]);
                amax  = max(amax, abs(vf[j]));
            }
            amax = multithread_reduce(amax, aiter::Max(), kThrPerGrp);
            const float row_scale =
                aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, kHwFp8Dtype>(amax);
            auto obuf =
                opus::make_gmem<DTYPE_O>(out + (int64_t)t * cols, cols * sizeof(DTYPE_O));
            store_vector<DTYPE_O, float, kTD, RT, false, WARP_SIZE, 1, DTYPE_O>(
                obuf, vf, tid * kTD, __builtin_amdgcn_rcpf(row_scale));
            bytes |= ((__builtin_bit_cast(uint32_t, row_scale) >> 23) & 0xFF) << (8 * i);
        }
        return bytes;
    };
    // Writes each scale byte at every sorted row its token occupies. The
    // group's threads all hold the scale, so they split the routes.
    auto store_scales = [&](int t0, uint32_t bytes) {
#pragma unroll
        for(int i = 0; i < kRowPf; ++i)
        {
            const int t = t0 + i;
            if(t >= t_end)
                break;
            const int* pos = s_pos + (t - t_begin) * topk;
            for(int k = tid % kThrPerGrp; k < topk; k += kThrPerGrp)
            {
                const int row = pos[k];
                if(row >= 0)
                    out_scale[aiter::mx_scale_shuffle_idx(scaleN_pad, row, scale_k)] =
                        (bytes >> (8 * i)) & 0xFF;
            }
        }
    };

    if(wave_id == 0)
    {
        // Own routes: rank = earlier routes of the same expert, from earlier
        // blocks (s_pre) and from earlier lanes here.
        const int r       = route_begin + lane_id;
        const bool valid  = lane_id < own_routes;
        const int own     = valid ? own_e : -1;
        const bool routed = valid && own >= 0 && own < E;
        // Every lane compares against all earlier routes, four per broadcast
        // LDS read. LDS accesses of one wave complete in order.
        s_own[lane_id] = own;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        int rank = 0;
#pragma unroll
        for(int c = 0; c < kMaxRoutesPerBlock; c += 4)
        {
            if(c >= own_routes)
                break;
            const int4 v = *reinterpret_cast<const int4*>(&s_own[c]);
            rank += (c + 0 < lane_id && v.x == own) ? 1 : 0;
            rank += (c + 1 < lane_id && v.y == own) ? 1 : 0;
            rank += (c + 2 < lane_id && v.z == own) ? 1 : 0;
            rank += (c + 3 < lane_id && v.w == own) ? 1 : 0;
        }

        // Exclusive scan of padded row counts.
        int cnt[kEpl];
        load_counts(cnt);
        int my_rows = 0;
#pragma unroll
        for(int j = 0; j < kEpl; ++j)
            my_rows += ((cnt[j] + unit - 1) >> log2u) << log2u;
        const int rows_incl = wave_scan_incl(my_rows, lane_id);
        {
            int off = rows_incl - my_rows;
#pragma unroll
            for(int j = 0; j < kEpl; ++j)
            {
                s_off[j < epl ? e_first + j : kSpareExpert] = off;
                off += ((cnt[j] + unit - 1) >> log2u) << log2u;
            }
        }

        // s_off of other lanes.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        int pos = -1;
        if(routed)
        {
            pos                 = s_off[own] + s_pre[own] + rank;
            const int t         = r / topk;
            sorted_ids[pos]     = pack_id(t, r - t * topk);
            sorted_weights[pos] = own_w;
        }
        s_pos[lane_id] = pos;

        const int num_valid = __builtin_amdgcn_readlane(rows_incl, kWaveSize - 1);
        if(lane_id == 0)
        {
            s_num_valid = num_valid;
            if(blockIdx.x == 0)
            {
                num_valid_ids[0] = num_valid;
                num_valid_ids[1] = M;
            }
        }
    }
    else if(moe_buf != nullptr)
    {
        constexpr int kZeroThreads = kThreads - kWaveSize;
        constexpr int VEC          = 8; // 8 x bf16 = one dwordx4 store
        using vec_t                = __attribute__((__vector_size__(VEC * sizeof(short)))) short;
        const int64_t nvec         = moe_buf_elems / VEC;
        vec_t z{};
        vec_t* vbuf = reinterpret_cast<vec_t*>(moe_buf);
        for(int64_t i = (int64_t)blockIdx.x * kZeroThreads + (tid - kWaveSize); i < nvec;
            i += (int64_t)gridDim.x * kZeroThreads)
            vbuf[i] = z;
    }
    __syncthreads();

    const int num_valid = s_num_valid;

    store_scales(t_begin, quant_rows(t_begin));
    // Rows past the first group are waited for right away (draining this
    // wave's stores), so no load is ever pending after this loop.
    for(int t0 = t_begin + kRowPf; t0 < t_end; t0 += kRowPf)
    {
#pragma unroll
        for(int i = 0; i < kRowPf; ++i)
            rows[i] = load_row(min(t0 + i, t_end - 1));
        __builtin_amdgcn_s_waitcnt(0x0f70); // vmcnt(0)
        store_scales(t0, quant_rows(t0));
    }

    // Padding rows and per-unit expert ids. Experts are dealt to waves across
    // the grid in chunks of up to one per lane; the lanes read their experts'
    // counts in parallel, then the wave visits only the non-empty ones and
    // writes each one's rows lane-parallel.
    const int sentinel = pack_id(M, topk);
    const int nwaves   = gridDim.x * kNumWaves;
    const int epw      = min(kWaveSize, (E + nwaves - 1) / nwaves);
    for(int e0 = (blockIdx.x * kNumWaves + wave_id) * epw; e0 < E; e0 += nwaves * epw)
    {
        const int e = e0 + lane_id;
        int start = 0, cnt = 0, padded = 0;
        if(lane_id < epw && e < E)
        {
            start  = s_off[e];
            cnt    = s_cnt[e];
            padded = ((cnt + unit - 1) >> log2u) << log2u;
        }
        for(uint64_t todo = __ballot(padded > 0); todo != 0; todo &= todo - 1)
        {
            const int j  = __builtin_ctzll(todo);
            const int sj = __builtin_amdgcn_readlane(start, j);
            const int cj = __builtin_amdgcn_readlane(cnt, j);
            const int pj = __builtin_amdgcn_readlane(padded, j);
            for(int r = cj + lane_id; r < pj; r += kWaveSize)
            {
                sorted_ids[sj + r]     = sentinel;
                sorted_weights[sj + r] = 0.0f;
            }
            for(int u = lane_id; u < (pj >> log2u); u += kWaveSize)
                sorted_expert_ids[(sj >> log2u) + u] = e0 + j;
        }
    }

    const int gtid    = blockIdx.x * kThreads + tid;
    const int gstride = gridDim.x * kThreads;

    // GEMM1 reads an id and a token for every launched tile before it checks
    // num_valid_ids, so the region past num_valid must hold valid values.
    for(int row = num_valid + gtid; row < max_rows; row += gstride)
    {
        sorted_ids[row]     = sentinel;
        sorted_weights[row] = 0.0f;
    }
    for(int u = (num_valid >> log2u) + gtid; u < max_units; u += gstride)
        sorted_expert_ids[u] = 0;
}

template <typename T>
T* typed_ptr(const aiter_tensor_t& t, AiterDtype expected, const char* name)
{
    AITER_CHECK(t.dtype() == expected,
                "moe_sort_mx_quant_fwd: ",
                name,
                " must be ",
                AiterDtype_to_str(expected),
                ", got ",
                AiterDtype_to_str(t.dtype()));
    return reinterpret_cast<T*>(t.data_ptr());
}

} // namespace msq

void moe_sort_mx_quant_fwd(aiter_tensor_t& hidden,
                           aiter_tensor_t& topk_ids,
                           aiter_tensor_t& topk_weights,
                           aiter_tensor_t& sorted_ids,
                           aiter_tensor_t& sorted_weights,
                           aiter_tensor_t& sorted_expert_ids,
                           aiter_tensor_t& num_valid_ids,
                           aiter_tensor_t& out,
                           aiter_tensor_t& out_scale,
                           int64_t num_experts,
                           int64_t unit_size,
                           std::optional<aiter_tensor_t> moe_buf)
{
    // pybind path: make AITER_CHECK throw (-> Python RuntimeError) instead of abort().
    aiter_detail::g_aiter_can_throw = true;
    using namespace aiter::msq;

    AITER_CHECK(hidden.dim() == 2 && topk_ids.dim() == 2,
                "moe_sort_mx_quant_fwd: hidden and topk_ids must be 2-D");
    const int M    = hidden.size(0);
    const int cols = hidden.size(1);
    const int topk = topk_ids.size(1);
    const int E    = num_experts;
    AITER_CHECK(M > 0 && topk_ids.size(0) == M,
                "moe_sort_mx_quant_fwd: topk_ids has ",
                topk_ids.size(0),
                " rows, hidden has ",
                M);
    AITER_CHECK(topk_weights.numel() == topk_ids.numel(),
                "moe_sort_mx_quant_fwd: topk_weights must match topk_ids");
    AITER_CHECK(topk >= 1 && topk <= kMaxRoutesPerBlock,
                "moe_sort_mx_quant_fwd: topk must be in 1..",
                kMaxRoutesPerBlock,
                ", got ",
                topk);
    AITER_CHECK(unit_size > 0 && (unit_size & (unit_size - 1)) == 0,
                "moe_sort_mx_quant_fwd: unit_size must be a power of two, got ",
                unit_size);
    AITER_CHECK(out.dtype() == AITER_DTYPE_fp8,
                "moe_sort_mx_quant_fwd: only MXFP8 output is supported, got ",
                AiterDtype_to_str(out.dtype()));
    AITER_CHECK(out_scale.dtype() == AITER_DTYPE_fp8_e8m0 || out_scale.dtype() == AITER_DTYPE_u8,
                "moe_sort_mx_quant_fwd: out_scale must hold e8m0 bytes");

    for(const auto& p : {std::make_pair(&hidden, "hidden"),
                         std::make_pair(&topk_ids, "topk_ids"),
                         std::make_pair(&topk_weights, "topk_weights"),
                         std::make_pair(&sorted_ids, "sorted_ids"),
                         std::make_pair(&sorted_weights, "sorted_weights"),
                         std::make_pair(&sorted_expert_ids, "sorted_expert_ids"),
                         std::make_pair(&num_valid_ids, "num_valid_ids"),
                         std::make_pair(&out, "out"),
                         std::make_pair(&out_scale, "out_scale")})
    {
        AITER_CHECK(
            p.first->is_contiguous(), "moe_sort_mx_quant_fwd: ", p.second, " must be contiguous");
        AITER_CHECK(p.first->device_id == hidden.device_id,
                    "moe_sort_mx_quant_fwd: ",
                    p.second,
                    " is on device ",
                    p.first->device_id,
                    ", hidden on ",
                    hidden.device_id);
    }

    opus::bf16_t* moe_buf_ptr = nullptr;
    int64_t moe_buf_elems     = 0;
    if(moe_buf.has_value() && moe_buf->numel() > 0)
    {
        AITER_CHECK(moe_buf->is_contiguous() && moe_buf->device_id == hidden.device_id,
                    "moe_sort_mx_quant_fwd: moe_buf must be contiguous and on hidden's device");
        moe_buf_ptr   = typed_ptr<opus::bf16_t>(*moe_buf, AITER_DTYPE_bf16, "moe_buf");
        moe_buf_elems = moe_buf->numel();
        AITER_CHECK(moe_buf_elems % 8 == 0,
                    "moe_sort_mx_quant_fwd: moe_buf element count must be a multiple of 8");
    }

    // Sorted outputs are written through the computed offsets, not through
    // these lengths, so an undersized buffer is an out-of-bounds write. Each
    // expert that receives a route pads to a unit, and at most M * topk do.
    const int64_t routes = (int64_t)M * topk;
    const int64_t active = std::min<int64_t>(E, routes);
    const int64_t need_rows =
        (routes + active * (unit_size - 1) + unit_size - 1) / unit_size * unit_size;
    const int64_t max_rows  = sorted_ids.numel();
    const int64_t max_units = sorted_expert_ids.numel();
    AITER_CHECK(need_rows < (1 << 20),
                "moe_sort_mx_quant_fwd: too many sorted rows: ",
                need_rows);
    AITER_CHECK(max_rows >= need_rows && (int64_t)sorted_weights.numel() >= max_rows &&
                    max_units * unit_size >= need_rows,
                "moe_sort_mx_quant_fwd: sorted buffers too small: rows ",
                max_rows,
                ", units ",
                max_units,
                ", need ",
                need_rows,
                " rows");
    const int64_t scaleN_pad = (cols / kGroupSize + 7) / 8 * 8;
    AITER_CHECK((int64_t)out_scale.numel() >= (max_rows + 31) / 32 * 32 * scaleN_pad,
                "moe_sort_mx_quant_fwd: out_scale too small");
    AITER_CHECK((int64_t)out.numel() >= (int64_t)M * cols, "moe_sort_mx_quant_fwd: out too small");
    AITER_CHECK(num_valid_ids.numel() >= 2, "moe_sort_mx_quant_fwd: num_valid_ids needs 2 slots");

    const HipDeviceGuard device_guard(hidden.device_id);
    const hipStream_t stream = getCurrentHIPStream();

    const int num_cu           = get_num_cu_func();
    const int tokens_per_block = (M + num_cu - 1) / num_cu;
    AITER_CHECK(tokens_per_block * topk <= kMaxRoutesPerBlock,
                "moe_sort_mx_quant_fwd: M=",
                M,
                " x topk=",
                topk,
                " exceeds the per-block route limit (",
                kMaxRoutesPerBlock,
                " x num_cu)");
    const int grid  = (M + tokens_per_block - 1) / tokens_per_block;
    const int log2u = __builtin_ctz(static_cast<uint32_t>(unit_size));

    const auto* h   = typed_ptr<opus::bf16_t>(hidden, AITER_DTYPE_bf16, "hidden");
    const auto* ids = typed_ptr<int>(topk_ids, AITER_DTYPE_i32, "topk_ids");
    const auto* w   = typed_ptr<float>(topk_weights, AITER_DTYPE_fp32, "topk_weights");
    auto* s_ids     = typed_ptr<int>(sorted_ids, AITER_DTYPE_i32, "sorted_ids");
    auto* s_w       = typed_ptr<float>(sorted_weights, AITER_DTYPE_fp32, "sorted_weights");
    auto* s_eids    = typed_ptr<int>(sorted_expert_ids, AITER_DTYPE_i32, "sorted_expert_ids");
    auto* n_valid   = typed_ptr<int>(num_valid_ids, AITER_DTYPE_i32, "num_valid_ids");
    auto* o         = reinterpret_cast<opus::fp8_t*>(out.data_ptr());
    auto* o_scale   = reinterpret_cast<uint8_t*>(out_scale.data_ptr());

#define MSQ_LAUNCH_EPL(THREADS, EPL)                                  \
    {                                                                 \
        moe_sort_mx_quant_kernel<(THREADS), (EPL), opus::fp8_t>       \
            <<<grid, (THREADS), 0, stream>>>(h,                       \
                                             ids,                     \
                                             w,                       \
                                             s_ids,                   \
                                             s_w,                     \
                                             s_eids,                  \
                                             n_valid,                 \
                                             o,                       \
                                             o_scale,                 \
                                             moe_buf_ptr,             \
                                             moe_buf_elems,           \
                                             M,                       \
                                             E,                       \
                                             topk,                    \
                                             log2u,                   \
                                             tokens_per_block,        \
                                             (int)max_rows,           \
                                             (int)max_units);         \
    }
#define MSQ_LAUNCH(THREADS)                                           \
    {                                                                 \
        AITER_CHECK(E <= (THREADS) * kMaxExpertsPerThread,            \
                    "moe_sort_mx_quant_fwd: num_experts must be <= ", \
                    (THREADS) * kMaxExpertsPerThread,                 \
                    ", got ",                                         \
                    E);                                               \
        if(E <= kWaveSize * kSmallEpl)                                \
            MSQ_LAUNCH_EPL(THREADS, kSmallEpl)                        \
        else                                                          \
            MSQ_LAUNCH_EPL(THREADS, kLargeEpl)                        \
    }
    static_assert(448 * kMaxExpertsPerThread <= kWaveSize * kLargeEpl);
    switch(cols)
    {
    case 256 * kTD: MSQ_LAUNCH(256) break;
    case 320 * kTD: MSQ_LAUNCH(320) break;
    case 448 * kTD: MSQ_LAUNCH(448) break;
    default:
        AITER_CHECK(false,
                    "moe_sort_mx_quant_fwd: hidden size must be one of 4096, 5120, 7168, got ",
                    cols);
    }
#undef MSQ_LAUNCH
#undef MSQ_LAUNCH_EPL
}

} // namespace aiter

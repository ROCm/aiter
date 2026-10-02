// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based paged-attention decode.
// Hosts launcher + validation on top of the device kernel template in
// `pa_decode_opus.h` (single-header, IMPL-guarded).

#define PA_DECODE_OPUS_IMPL
#include "pa_decode_opus.h"
// Smallest run of KV tiles a split may own. On a short context this, not the CU
// count, is what caps the split count. Tune with op_tests/sweep_min_tiles_per_split.sh.
#ifndef PA_DECODE_MIN_TILES_PER_SPLIT
#define PA_DECODE_MIN_TILES_PER_SPLIT 4
#endif

// Upper bound on `Traits::WGS_PER_CU`, which is what the split heuristic actually
// reads. The value each instantiation wants depends on how much KV a workgroup
// streams, so it lives in the traits; this one only has to cover the largest of them,
// because it sizes the split scratch and the arrival counters, and those are shared
// by every instantiation on the stream -- today that is the 64-dim head.
#ifndef PA_DECODE_WGS_PER_CU
#define PA_DECODE_WGS_PER_CU 32
#endif

// Optional long-KV fill ceiling. 0 = max(16/8, CUs/base_wgs) so B=1 fills
// every CU (NP=256 on gfx950) without oversplitting B=200. Override with
// PA_DECODE_OPUS_MAX_FILL_SPLITS.
#ifndef PA_DECODE_OPUS_MAX_FILL_SPLITS
#define PA_DECODE_OPUS_MAX_FILL_SPLITS 0
#endif

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <unordered_map>

// Split-KV scratch, allocated once per stream and kept for the process: at small
// batch the kernel runs in a few microseconds, so a per-call allocate/free pair
// would sit on the critical path of every decode step.
//
// A fixed buffer works because the slot count is bounded -- splits are chosen so
// that base_wgs * num_splits <= num_cu * PA_DECODE_WGS_PER_CU -- which caps the
// scratch at a few MB and leaves no allocation to trip over during graph capture.
// Keyed by stream: concurrent streams must not share one buffer.
namespace {
struct SplitScratchRegistry
{
    std::mutex mu;
    std::unordered_map<hipStream_t, void*> map;
};
SplitScratchRegistry& split_scratch_registry()
{
    static SplitScratchRegistry r;
    return r;
}
} // namespace

static void* split_scratch_get(hipStream_t stream, size_t bytes)
{
    auto& reg = split_scratch_registry();
    std::lock_guard<std::mutex> lock(reg.mu);

    auto it = reg.map.find(stream);
    if(it != reg.map.end()) return it->second;

    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CALL(hipStreamIsCapturing(stream, &capture));
    AITER_CHECK(capture == hipStreamCaptureStatusNone,
                "pa_decode_opus split-KV scratch cannot be created during HIP graph "
                "capture (hipMalloc is capture-illegal). Run one split-KV decode "
                "eagerly on this stream before capturing.");

    void* ptr = nullptr;
            // Uncached, so the partials are visible device-wide as soon as the stores
            // retire. This is the one place the split workgroups hand data to each
            // other, and a multi-XCD part's L2s are not coherent; the alternative is
            // a __threadfence() per split, which writes back the whole L2.
    HIP_CALL(hipExtMallocWithFlags(&ptr, bytes, hipDeviceMallocUncached));
        // The arrival counters at the tail start at zero and are cleared only here:
        // the kernel's last split resets its own counter after every launch.
    HIP_CALL(hipMemset(ptr, 0, bytes));
    reg.map[stream] = ptr;
    return ptr;
}

// Shared by both entry points; they differ only in the operand dtype (which fixes
// Traits, and with it the cache's pack factor x) and in the dequant constants.
// qk_dequant/v_dequant are 1.0f on the bf16 path.
static void bind_kv_scale_maps(pa_decode_kargs& kargs,
                               aiter_tensor_t* k_scale_map,
                               aiter_tensor_t* v_scale_map,
                               int num_blocks,
                               int num_kv_heads,
                               int page_size)
{
    if(k_scale_map == nullptr)
    {
        kargs.k_scale_map = nullptr;
        kargs.v_scale_map = nullptr;
        return;
    }
    AITER_CHECK(v_scale_map != nullptr, "v_scale_map is required with k_scale_map");
    auto check = [&](aiter_tensor_t& m, const char* name) {
        AITER_CHECK(m.dtype() == AITER_DTYPE_fp32, name, " must be fp32");
        AITER_CHECK(m.dim() == 4 && m.size(3) == 1,
                    name, " must be [num_blocks, num_kv_heads, PAGE, 1]");
        AITER_CHECK(m.size(0) == num_blocks && m.size(1) == num_kv_heads
                        && m.size(2) == page_size,
                    name, " shape mismatch");
        AITER_CHECK(m.stride(2) == 1, name, " page dim must be contiguous");
    };
    check(*k_scale_map, "k_scale_map");
    check(*v_scale_map, "v_scale_map");
    kargs.k_scale_map   = reinterpret_cast<const float*>(k_scale_map->data_ptr());
    kargs.v_scale_map   = reinterpret_cast<const float*>(v_scale_map->data_ptr());
    kargs.stride_ks_blk = static_cast<int>(k_scale_map->stride(0));
    kargs.stride_ks_h   = static_cast<int>(k_scale_map->stride(1));
    kargs.stride_vs_blk = static_cast<int>(v_scale_map->stride(0));
    kargs.stride_vs_h   = static_cast<int>(v_scale_map->stride(1));
}

template<class Traits>
static void pa_decode_opus_launch(aiter_tensor_t& q,
                                  aiter_tensor_t& k_cache,
                                  aiter_tensor_t& v_cache,
                                  aiter_tensor_t& block_tables,
                                  aiter_tensor_t& context_lens,
                                  aiter_tensor_t& out,
                                  float softmax_scale,
                                  float qk_dequant,
                                  float v_dequant,
                                  aiter_tensor_t* sink = nullptr,
                                  aiter_tensor_t* k_scale_map = nullptr,
                                  aiter_tensor_t* v_scale_map = nullptr)
{
    // ---- Shape / dtype validation -----------------------------------------
    // MTP slots a token dim between batch and heads. Rather than branch on it, the
    // shape reads below index off TD, which is zero on the decode instantiations.
    constexpr int TD = Traits::HAS_MTP ? 1 : 0;
    AITER_CHECK(q.dim() == 3 + TD,
                "q must be ", 3 + TD, "-D [batch, ", (TD ? "qlen, " : ""),
                "num_heads, D], got ndim=", q.dim());
    AITER_CHECK(out.dim() == 3 + TD, "out must be ", 3 + TD, "-D, got ndim=", out.dim());
    AITER_CHECK(k_cache.dim() == 5,
                "k_cache must be 5-D [num_blocks, num_kv_heads, D/x, page, x], got ndim=",
                k_cache.dim());
    AITER_CHECK(v_cache.dim() == (Traits::V_SHUFFLED ? 5 : 4),
                "v_cache must match the selected 4-D or shuffled 5-D layout, got ndim=",
                v_cache.dim());
    AITER_CHECK(block_tables.dim() == 2,
                "block_tables must be 2-D [batch, max_blocks_per_batch_row]");
    AITER_CHECK(context_lens.dim() == 1, "context_lens must be 1-D [batch]");

    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(context_lens.dtype() == AITER_DTYPE_i32, "context_lens must be int32");

    const int batch        = static_cast<int>(q.size(0));
    const int qlen         = Traits::HAS_MTP ? static_cast<int>(q.size(1)) : 1;
    const int num_heads    = static_cast<int>(q.size(1 + TD));
    const int head_size    = static_cast<int>(q.size(2 + TD)); //head_dim
    const int num_kv_heads = static_cast<int>(k_cache.size(1));
    const int page_size    = Traits::V_SHUFFLED
                                ? static_cast<int>(v_cache.size(2) * v_cache.size(4))
                                : static_cast<int>(v_cache.size(3));

    AITER_CHECK(head_size == Traits::D_HEAD,
                "Only head_size=", Traits::D_HEAD, " is compiled, got ", head_size);
    AITER_CHECK(page_size == Traits::PAGE_SIZE,
                "Only page size=", Traits::PAGE_SIZE, " is compiled, got ", page_size);
    AITER_CHECK(static_cast<int>(k_cache.size(4)) == Traits::K_PACK,
                "k_cache pack factor x must be ", Traits::K_PACK, " for this dtype");
    AITER_CHECK(static_cast<int>(k_cache.size(2)) == Traits::D_HEAD / Traits::K_PACK,
                "k_cache dim-group count must be D/x");
    AITER_CHECK(static_cast<int>(k_cache.size(3)) == Traits::PAGE_SIZE,
                "k_cache page dim mismatch");
    if constexpr(Traits::V_SHUFFLED)
    {
        AITER_CHECK(v_cache.size(2) == Traits::PAGE_SIZE / Traits::K_PACK
                    && v_cache.size(3) == Traits::D_HEAD && v_cache.size(4) == Traits::K_PACK,
                    "transposed v_cache must be [num_blocks, num_kv_heads, PAGE/x, D, x]");
        AITER_CHECK(v_cache.size(0) == k_cache.size(0), "K/V page counts must match");
        if constexpr(Traits::HAS_SINK)
            AITER_CHECK(num_heads == 64 && num_kv_heads == 4,
                        "shuffled V requires Q64/KV4");
    }
    else
        AITER_CHECK(static_cast<int>(v_cache.size(2)) == Traits::D_HEAD, "v_cache head dim mismatch");
    AITER_CHECK(static_cast<int>(v_cache.size(1)) == num_kv_heads,
                "k_cache/v_cache must agree on num_kv_heads");

    AITER_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    const int gqa_ratio = num_heads / num_kv_heads;
    // A tile row is (token, head) under MTP and just a head otherwise, so the same
    // Q_TILE bound covers both once qlen is folded in.
    AITER_CHECK(qlen >= 1, "qlen must be >= 1, got ", qlen);
    if constexpr(Traits::MTP_Q_LOOP)
    {
        AITER_CHECK(gqa_ratio == Traits::Q_TILE,
                    "token-loop MTP requires GQA == ", Traits::Q_TILE, ", got ", gqa_ratio);
        AITER_CHECK(qlen >= 2 && qlen <= Traits::MTP_Q_MAX,
                    "token-loop MTP requires 2 <= qlen <= ", Traits::MTP_Q_MAX, ", got ", qlen);
    }
    else if constexpr(Traits::MTP_Q_SPLIT)
    {
        AITER_CHECK(gqa_ratio == Traits::Q_TILE,
                    "query-split MTP requires GQA == ", Traits::Q_TILE, ", got ", gqa_ratio);
        AITER_CHECK(qlen >= 2 && qlen <= Traits::MTP_Q_MAX,
                    "query-split MTP requires 2 <= qlen <= ", Traits::MTP_Q_MAX, ", got ", qlen);
    }
    else
    {
        AITER_CHECK(qlen * gqa_ratio <= Traits::Q_TILE,
                    "qlen * GQA ratio must be <= ", Traits::Q_TILE, ", got ", qlen, " * ",
                    gqa_ratio);
    }

    AITER_CHECK(out.size(0) == batch && out.size(1 + TD) == num_heads &&
                    out.size(2 + TD) == head_size && (!TD || out.size(1) == qlen),
                "out shape must match q");
    AITER_CHECK(block_tables.size(0) == batch, "block_tables first dim must be batch");
    AITER_CHECK(context_lens.size(0) == batch, "context_lens length must be batch");

    AITER_CHECK(q.stride(2 + TD) == 1 && out.stride(2 + TD) == 1,
                "q/out must be contiguous along the head dim");
    AITER_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),
                "k_cache/v_cache must be contiguous");
    AITER_CHECK(block_tables.is_contiguous() && context_lens.is_contiguous(),
                "block_tables/context_lens must be contiguous");

    if(batch == 0) return;

    // ---- Build kernel args -------------------------------------------------
    pa_decode_kargs kargs{};
    kargs.q_ptr                    = q.data_ptr();
    kargs.k_ptr                    = k_cache.data_ptr();
    kargs.v_ptr                    = v_cache.data_ptr();
    kargs.out_ptr                  = out.data_ptr();
    kargs.block_tables             = reinterpret_cast<const int*>(block_tables.data_ptr());
    kargs.context_lens             = reinterpret_cast<const int*>(context_lens.data_ptr());
    kargs.batch                    = batch;
    kargs.num_heads                = num_heads;
    kargs.num_kv_heads             = num_kv_heads;
    kargs.gqa_ratio                = gqa_ratio;
    kargs.max_blocks_per_batch_row = static_cast<int>(block_tables.size(1));
    kargs.stride_q_b               = static_cast<int>(q.stride(0));
    kargs.stride_q_h               = static_cast<int>(q.stride(1 + TD));
    kargs.stride_o_b               = static_cast<int>(out.stride(0));
    kargs.stride_o_h               = static_cast<int>(out.stride(1 + TD));
    kargs.qlen                     = qlen;
    kargs.stride_q_t               = TD ? static_cast<int>(q.stride(1)) : 0;
    kargs.stride_o_t               = TD ? static_cast<int>(out.stride(1)) : 0;
    kargs.stride_k_blk             = static_cast<int>(k_cache.stride(0));
    kargs.stride_k_h               = static_cast<int>(k_cache.stride(1));
    kargs.stride_v_blk             = static_cast<int>(v_cache.stride(0));
    kargs.stride_v_h               = static_cast<int>(v_cache.stride(1));
    kargs.softmax_scale            = softmax_scale;
    kargs.qk_dequant               = qk_dequant;
    kargs.v_dequant                = v_dequant;
    kargs.sink = sink ? reinterpret_cast<const float*>(sink->data_ptr()) : nullptr;
    AITER_CHECK(!Traits::HAS_SINK || kargs.sink != nullptr,
                "this instantiation reads a sink; one must be supplied");
    if constexpr(Traits::PER_TOKEN_SCALE)
        AITER_CHECK(k_scale_map != nullptr && v_scale_map != nullptr,
                    "per-token instantiation requires k_scale_map and v_scale_map");
    bind_kv_scale_maps(kargs, k_scale_map, v_scale_map,
                       static_cast<int>(k_cache.size(0)), num_kv_heads, page_size);

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // ---- Pick a split count --------------------------------------------------
    //
    // One workgroup per (kv head, batch) leaves the GPU mostly idle at small batch,
    // so the KV axis is split and the per-split partials merged afterwards. Large
    // batches already fill the machine and keep the single-pass path; short contexts
    // are kept off it by requiring min_tiles_per_split tiles per split.
    constexpr int min_tiles_per_split = PA_DECODE_MIN_TILES_PER_SPLIT;
    int num_cu = 0;
    HIP_CALL(hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, q.device_id));

    static_assert(Traits::WGS_PER_CU <= PA_DECODE_WGS_PER_CU,
                  "raise PA_DECODE_WGS_PER_CU: it bounds the shared split scratch");
    const int grid_y    = Traits::MTP_Q_SPLIT ? batch * qlen : batch;
    const int base_wgs  = num_kv_heads * grid_y; //base workgroups
    int wg_budget = num_cu * Traits::WGS_PER_CU;
    const int max_tiles = (kargs.max_blocks_per_batch_row * Traits::PAGE_SIZE + Traits::KV_TILE - 1)
                            / Traits::KV_TILE;
    if constexpr(PA_DECODE_OPUS_A8W8_LONG_SPLITS && Traits::HAS_SINK && !Traits::HAS_MTP
                 && Traits::IS_FP8 && !Traits::QUANT_Q && Traits::D_HEAD == 128
                 && Traits::PAGE_SIZE == 128 && Traits::KV_TILE == 128)
    {
        static_assert(PA_DECODE_WGS_PER_CU >= 2,
                      "long A8W8 splits require scratch for two workgroups per CU");
        if(num_kv_heads == 4 && kargs.gqa_ratio == 16 && max_tiles >= 64
           && base_wgs >= (num_cu + 3) / 4 && base_wgs <= num_cu)
        {
            kargs.short_seq_splits = num_cu / base_wgs;
            wg_budget = num_cu * 2;
        }
    }
    int num_splits = 1;
    if(base_wgs < wg_budget && max_tiles >= 2 * min_tiles_per_split)
    {
        num_splits = wg_budget / base_wgs;
        if(num_splits > max_tiles / min_tiles_per_split) num_splits = max_tiles / min_tiles_per_split;
        if(num_splits > Traits::MAX_SPLITS) num_splits = Traits::MAX_SPLITS;
        if(num_splits < 1) num_splits = 1;
    }
    // Long KV: fill idle CUs at small batch (B=1 → NP=256 on a 256-CU gfx950)
    // and keep the measured-good 16/8 oversubscribe when the grid already covers
    // the machine (B=200 → NP=16). Override with PA_DECODE_OPUS_MAX_FILL_SPLITS.
    if constexpr(Traits::QUANT_Q && !Traits::HAS_SINK && Traits::IS_FP8 && Traits::D_HEAD == 128)
    {
        if(max_tiles >= 256 && base_wgs > 0)
        {
            int fill_cap = Traits::MTP_Q_LOOP ? 16 : 8;
            const int cu_fill = num_cu / base_wgs;
            if(cu_fill > fill_cap) fill_cap = cu_fill;
            if constexpr(PA_DECODE_OPUS_MAX_FILL_SPLITS > 0)
                fill_cap = PA_DECODE_OPUS_MAX_FILL_SPLITS;
            if(const char* env = std::getenv("PA_DECODE_OPUS_MAX_FILL_SPLITS"))
            {
                fill_cap = std::atoi(env);
                if(fill_cap < 1) fill_cap = 1;
            }
            if(fill_cap > Traits::MAX_SPLITS) fill_cap = Traits::MAX_SPLITS;
            int target = fill_cap;
            const int scratch_fit = (num_cu * PA_DECODE_WGS_PER_CU) / base_wgs;
            if(target > scratch_fit) target = scratch_fit;
            if(target > max_tiles / min_tiles_per_split)
                target = max_tiles / min_tiles_per_split;
            if(target > Traits::MAX_SPLITS) target = Traits::MAX_SPLITS;
            if(target < 1) target = 1;
            num_splits = target;
        }
    }
    kargs.num_splits = num_splits;
    if(kargs.short_seq_splits > num_splits) kargs.short_seq_splits = num_splits;
    if(std::getenv("PA_DECODE_OPUS_PRINT_SPLITS"))
    {
        static int last_printed = -1;
        if(num_splits != last_printed)
        {
            last_printed = num_splits;
            std::fprintf(stderr,
                         "[opus] batch=%d nkv=%d qlen=%d splits=%d tiles=%d cu=%d qsplit=%d\n",
                         batch,
                         num_kv_heads,
                         qlen,
                         num_splits,
                         max_tiles,
                         num_cu,
                         (int)Traits::MTP_Q_SPLIT);
        }
    }

    // ---- Launch --------------------------------------------------------------
    dim3 block(Traits::BLOCK_SIZE);
    auto launch = [&]() {
        const dim3 grid(num_kv_heads, grid_y, num_splits);
        if constexpr(PA_DECODE_OPUS_A8W8_D128_BT_SCALAR && Traits::IS_FP8 && !Traits::QUANT_Q
                     && Traits::HAS_SINK && !Traits::HAS_MTP && Traits::D_HEAD == 128
                     && Traits::PAGE_SIZE == 128 && Traits::KV_TILE == 128)
        {
            const int max_tiles_per_split = (max_tiles + num_splits - 1) / num_splits;
            if(num_kv_heads == 4 && gqa_ratio == 16 && max_tiles_per_split >= 8)
            {
                pa_decode_opus_kernel<pa_decode_smem_bt_traits<Traits>>
                    <<<grid, block, 0, stream>>>(kargs);
                HIP_CALL_LAUNCH(hipGetLastError());
                return;
            }
        }
        pa_decode_opus_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    if(num_splits == 1)
    {
        launch();
        return;
    }

    // One slot holds a split's partial O plus its (m, l) pair. Never resized, so it
    // is sized for the widest launch the heuristic can ask for: num_splits is capped
    // at wg_budget / base_wgs, which bounds the slot count by wg_budget.
    //
    // The bound uses the widest head dim any instantiation compiles, not this one's.
    // One buffer is cached per stream and shared by every instantiation, so sizing it
    // to a 64-dim head would leave a 128-dim launch writing past the end -- and the
    // arrival counters, which must survive across calls, have to land at the same
    // offset whichever kernel got there first.
    // Not wg_budget: that is now per-instantiation, and the scratch is shared, so it
    // has to be sized for the most any of them can ask for.
    const size_t slot_max  = static_cast<size_t>(num_cu) * PA_DECODE_WGS_PER_CU;
    const int q_pack       = Traits::MTP_Q_LOOP ? Traits::MTP_Q_MAX : 1;
    // Cap is independent of this launch: the per-stream scratch and the arrival
    // counters must land at a fixed offset whichever instantiation allocated first.
    const int q_pack_cap   = Traits::MTP_Q_MAX;
    const size_t slots     = static_cast<size_t>(grid_y) * num_kv_heads * num_splits;
    const size_t slot_max_q = slot_max * static_cast<size_t>(q_pack_cap);
    const size_t o_bytes   = slots * static_cast<size_t>(q_pack) * Traits::Q_TILE * Traits::D_HEAD
                           * sizeof(float);
    const size_t ml_bytes  = slot_max_q * Traits::Q_TILE * 2 * sizeof(float);
    const size_t o_max = slot_max_q * Traits::Q_TILE * PA_DECODE_MAX_D_HEAD * sizeof(float);
    // One counter per (batch, kv-head); those are bounded by the slot count too.
    const size_t ctr_bytes = slot_max * sizeof(unsigned int);
    const size_t bytes_max = o_max + ml_bytes + ctr_bytes;
    static_assert(Traits::D_HEAD <= PA_DECODE_MAX_D_HEAD, "raise PA_DECODE_MAX_D_HEAD");
    AITER_CHECK(slots <= slot_max, "split-KV slot count ", slots, " exceeds the CU bound ", slot_max);

    char* scratch          = static_cast<char*>(split_scratch_get(stream, bytes_max));
    kargs.partial_o        = reinterpret_cast<float*>(scratch);
    kargs.partial_ml       = reinterpret_cast<float*>(scratch + o_bytes);
    kargs.split_counters   = reinterpret_cast<unsigned int*>(scratch + o_max + ml_bytes);

        // One launch: the split that arrives last merges the partials in place,
        // rather than a second grid being stood up to do it. A16W8 instead
        // launches a per-row reduce so NP=256 can keep every CU busy.
    launch();
    if constexpr(Traits::SEPARATE_SPLIT_REDUCE)
    {
        const int reduce_rows =
            (Traits::MTP_Q_LOOP || (Traits::HAS_MTP && !Traits::MTP_Q_SPLIT))
                ? qlen * gqa_ratio
                : gqa_ratio;
        const dim3 rgrid(num_kv_heads, grid_y, reduce_rows);
        pa_decode_opus_split_reduce_kernel<Traits><<<rgrid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    }
}

void pa_decode_opus_fwd(aiter_tensor_t& q,
                        aiter_tensor_t& k_cache,
                        aiter_tensor_t& v_cache,
                        aiter_tensor_t& block_tables,
                        aiter_tensor_t& context_lens,
                        aiter_tensor_t& out,
                        float softmax_scale)
{
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && out.dtype() == AITER_DTYPE_bf16,
                "pa_decode_opus_fwd takes bf16 q/out");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_bf16 && v_cache.dtype() == AITER_DTYPE_bf16,
                "k_cache/v_cache must be bf16 (A16W16); use pa_decode_opus_fp8_fwd for fp8");

    // A 4-D q carries a token dim and asks for the tail-causal MTP mask. qlen == 1 is
    // plain decode either way, and the caller squeezes it back to 3-D before it gets
    // here, so that case keeps the decode kernel rather than paying MTP's extra
    // peeled tile for a mask that would never fire.
    if(q.dim() == 4)
    {
        pa_decode_opus_launch<pa_decode_traits_d128_mtp>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, 1.0f, 1.0f);
        return;
    }
    pa_decode_opus_launch<pa_decode_traits_d128>(
        q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, 1.0f, 1.0f);
}

// Persistent variant. Takes the work queue and partial buffers the shared PA metadata
// and reduce kernels use, so the caller owns their lifetime and can keep the metadata
// build off the decode's critical path when the schedule is reusable.
//
// The grid is one workgroup per CU: the work queue, not the grid, carries the problem
// size. That is what lets a batch of mixed context lengths balance, where the
// grid-per-split form has to pick one split count for the whole batch.
template<class Traits>
static void pa_decode_opus_ps_launch(aiter_tensor_t& q,
                                     aiter_tensor_t& k_cache,
                                     aiter_tensor_t& v_cache,
                                     aiter_tensor_t& kv_indptr,
                                     aiter_tensor_t& kv_indices,
                                     aiter_tensor_t& context_lens,
                                     aiter_tensor_t& out,
                                     aiter_tensor_t& work_indptr,
                                     aiter_tensor_t& work_info,
                                     aiter_tensor_t& split_o,
                                     aiter_tensor_t& split_lse,
                                     float softmax_scale,
                                     float qk_dequant,
                                     float v_dequant,
                                     aiter_tensor_t* sink = nullptr)
{
    AITER_CHECK(q.dim() == 3, "q must be 3-D [batch, num_heads, D], got ndim=", q.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [batch, num_heads, D], got ndim=", out.dim());
    AITER_CHECK(k_cache.dim() == 5, "k_cache must be 5-D, got ndim=", k_cache.dim());
    AITER_CHECK(v_cache.dim() == 4, "v_cache must be 4-D, got ndim=", v_cache.dim());
    AITER_CHECK(kv_indptr.dtype() == AITER_DTYPE_i32 && kv_indices.dtype() == AITER_DTYPE_i32,
                "kv_indptr/kv_indices must be int32");
    AITER_CHECK(work_indptr.dtype() == AITER_DTYPE_i32 && work_info.dtype() == AITER_DTYPE_i32,
                "work_indptr/work_info must be int32");
    AITER_CHECK(context_lens.dtype() == AITER_DTYPE_i32, "context_lens must be int32");
    AITER_CHECK(split_o.dtype() == AITER_DTYPE_fp32 && split_lse.dtype() == AITER_DTYPE_fp32,
                "split_o/split_lse must be fp32");
    AITER_CHECK(work_info.dim() == 2 && work_info.size(1) == 8,
                "work_info must be [num_works, 8]");

    const int batch        = static_cast<int>(q.size(0));
    const int num_heads    = static_cast<int>(q.size(1));
    const int head_size    = static_cast<int>(q.size(2));
    const int num_kv_heads = static_cast<int>(k_cache.size(1));

    AITER_CHECK(head_size == Traits::D_HEAD,
                "Only head_size=", Traits::D_HEAD, " is compiled, got ", head_size);
    AITER_CHECK(static_cast<int>(v_cache.size(3)) == Traits::PAGE_SIZE,
                "Only page size=", Traits::PAGE_SIZE, " is compiled, got ", v_cache.size(3));
    AITER_CHECK(static_cast<int>(k_cache.size(4)) == Traits::K_PACK,
                "k_cache pack factor x must be ", Traits::K_PACK, " for this dtype");
    AITER_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    const int gqa_ratio = num_heads / num_kv_heads;
    AITER_CHECK(gqa_ratio <= Traits::Q_TILE,
                "GQA ratio must be <= ", Traits::Q_TILE, ", got ", gqa_ratio);
    AITER_CHECK(kv_indptr.size(0) == batch + 1, "kv_indptr must be [batch + 1]");

    if(batch == 0) return;

    pa_decode_kargs kargs{};
    kargs.q_ptr                    = q.data_ptr();
    kargs.k_ptr                    = k_cache.data_ptr();
    kargs.v_ptr                    = v_cache.data_ptr();
    kargs.out_ptr                  = out.data_ptr();
    kargs.context_lens             = reinterpret_cast<const int*>(context_lens.data_ptr());
    kargs.kv_indptr                = reinterpret_cast<const int*>(kv_indptr.data_ptr());
    kargs.kv_indices               = reinterpret_cast<const int*>(kv_indices.data_ptr());
    kargs.work_indptr              = reinterpret_cast<const int*>(work_indptr.data_ptr());
    kargs.work_info                = reinterpret_cast<const int*>(work_info.data_ptr());
    kargs.split_o                  = reinterpret_cast<float*>(split_o.data_ptr());
    kargs.split_lse                = reinterpret_cast<float*>(split_lse.data_ptr());
    kargs.batch                    = batch;
    kargs.num_heads                = num_heads;
    kargs.num_kv_heads             = num_kv_heads;
    kargs.gqa_ratio                = gqa_ratio;
    kargs.stride_q_b               = static_cast<int>(q.stride(0));
    kargs.stride_q_h               = static_cast<int>(q.stride(1));
    kargs.stride_o_b               = static_cast<int>(out.stride(0));
    kargs.stride_o_h               = static_cast<int>(out.stride(1));
    kargs.stride_k_blk             = static_cast<int>(k_cache.stride(0));
    kargs.stride_k_h               = static_cast<int>(k_cache.stride(1));
    kargs.stride_v_blk             = static_cast<int>(v_cache.stride(0));
    kargs.stride_v_h               = static_cast<int>(v_cache.stride(1));
    kargs.softmax_scale            = softmax_scale;
    kargs.qk_dequant               = qk_dequant;
    kargs.v_dequant                = v_dequant;
    kargs.sink = sink ? reinterpret_cast<const float*>(sink->data_ptr()) : nullptr;
    AITER_CHECK(!Traits::HAS_SINK || kargs.sink != nullptr,
                "this instantiation reads a sink; one must be supplied");

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // work_indptr is sized [num_cu + 1] by the metadata kernel, so it, not a device
    // query, is what fixes the grid -- the two must agree or workgroups read past it.
    const int num_work_slots = static_cast<int>(work_indptr.size(0)) - 1;
    AITER_CHECK(num_work_slots > 0, "work_indptr must be [num_cu + 1]");

    pa_decode_opus_ps_kernel<Traits>
        <<<dim3(num_work_slots, 1, 1), dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void pa_decode_opus_fp8_ps_fwd(aiter_tensor_t& q,
                               aiter_tensor_t& k_cache,
                               aiter_tensor_t& v_cache,
                               aiter_tensor_t& kv_indptr,
                               aiter_tensor_t& kv_indices,
                               aiter_tensor_t& context_lens,
                               aiter_tensor_t& out,
                               aiter_tensor_t& work_indptr,
                               aiter_tensor_t& work_info,
                               aiter_tensor_t& split_o,
                               aiter_tensor_t& split_lse,
                               float softmax_scale,
                               float q_scale,
                               float k_scale,
                               float v_scale)
{
    AITER_CHECK(q.dtype() == AITER_DTYPE_fp8, "pa_decode_opus_fp8_ps_fwd takes fp8 q");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_fp8 && v_cache.dtype() == AITER_DTYPE_fp8,
                "k_cache/v_cache must be fp8 (A8W8)");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out stays bf16 on the fp8 path");
    AITER_CHECK(q_scale > 0.0f && k_scale > 0.0f && v_scale > 0.0f,
                "q/k/v scales must be positive");

    pa_decode_opus_ps_launch<pa_decode_traits_d128_fp8>(q,
                                                        k_cache,
                                                        v_cache,
                                                        kv_indptr,
                                                        kv_indices,
                                                        context_lens,
                                                        out,
                                                        work_indptr,
                                                        work_info,
                                                        split_o,
                                                        split_lse,
                                                        softmax_scale,
                                                        q_scale * k_scale,
                                                        v_scale);
}

void pa_decode_opus_fp8_fwd(aiter_tensor_t& q,
                            aiter_tensor_t& k_cache,
                            aiter_tensor_t& v_cache,
                            aiter_tensor_t& block_tables,
                            aiter_tensor_t& context_lens,
                            aiter_tensor_t& out,
                            float softmax_scale,
                            float q_scale,
                            float k_scale,
                            float v_scale)
{
    AITER_CHECK(q.dtype() == AITER_DTYPE_fp8, "pa_decode_opus_fp8_fwd takes fp8 q");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_fp8 && v_cache.dtype() == AITER_DTYPE_fp8,
                "k_cache/v_cache must be fp8 (A8W8)");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out stays bf16 on the fp8 path");
    AITER_CHECK(q_scale > 0.0f && k_scale > 0.0f && v_scale > 0.0f,
                "q/k/v scales must be positive");

    // GEMM0's two descales commute with everything up to the softmax, so they arrive
    // as one number; v_scale stays separate because it applies after GEMM1.
    pa_decode_opus_launch<pa_decode_traits_d128_fp8>(q,
                                                     k_cache,
                                                     v_cache,
                                                     block_tables,
                                                     context_lens,
                                                     out,
                                                     softmax_scale,
                                                     q_scale * k_scale,
                                                     v_scale);
}

// gpt-oss shapes: 64- or 128-dim heads with a learned per-head sink. Share every launcher
// above; only the traits and the sink pointer differ.
template<class Traits>
static void pa_decode_opus_gptoss_check(aiter_tensor_t& q,
                                        aiter_tensor_t& k_cache,
                                        aiter_tensor_t& v_cache,
                                        aiter_tensor_t& out,
                                        aiter_tensor_t& sink,
                                        bool fp8_q)
{
    constexpr int TD = Traits::HAS_MTP ? 1 : 0;
    AITER_CHECK(q.dtype() == (fp8_q ? AITER_DTYPE_fp8 : AITER_DTYPE_bf16),
                "q dtype must match the requested activation width");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_fp8 && v_cache.dtype() == AITER_DTYPE_fp8,
                "k_cache/v_cache must be fp8");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out is bf16");
    AITER_CHECK(sink.dtype() == AITER_DTYPE_fp32, "sink must be fp32");
    AITER_CHECK(sink.dim() == 1 && sink.size(0) == q.size(1 + TD),
                "sink must be [num_heads], one scaled logit per query head");
    AITER_CHECK(static_cast<int>(q.size(2 + TD)) == Traits::D_HEAD,
                "this entry point is compiled for head_dim=", Traits::D_HEAD);
}

// gpt-oss supports several page sizes, so each is compiled and
// the cache's own shape picks. MTP is a separate instantiation: it peels an extra
// tile for the tail-causal mask, which decode should not pay.
#define PA_GPTOSS_DISPATCH_DIM(D_HEAD, FP8_Q, PAGE, MTP, LAUNCH, ...)                    \
    do                                                                                   \
    {                                                                                    \
        if((MTP) && (FP8_Q) && (PAGE) == 16)                                             \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_fp8_sink_mtp>(      \
                q, k_cache, v_cache, out, sink, true);                                   \
            LAUNCH<pa_decode_traits_d##D_HEAD##_fp8_sink_mtp>(__VA_ARGS__);              \
        }                                                                                \
        else if((MTP) && (FP8_Q) && (PAGE) == 128)                                       \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_fp8_sink_p128_mtp>( \
                q, k_cache, v_cache, out, sink, true);                                   \
            LAUNCH<pa_decode_traits_d##D_HEAD##_fp8_sink_p128_mtp>(__VA_ARGS__);         \
        }                                                                                \
        else if((MTP) && (FP8_Q) && (PAGE) == 256)                                       \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_fp8_sink_p256_mtp>( \
                q, k_cache, v_cache, out, sink, true);                                   \
            LAUNCH<pa_decode_traits_d##D_HEAD##_fp8_sink_p256_mtp>(__VA_ARGS__);         \
        }                                                                                \
        else if((MTP) && (PAGE) == 16)                                                   \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_a16w8_sink_mtp>(    \
                q, k_cache, v_cache, out, sink, false);                                  \
            LAUNCH<pa_decode_traits_d##D_HEAD##_a16w8_sink_mtp>(__VA_ARGS__);            \
        }                                                                                \
        else if((MTP) && (PAGE) == 128)                                                  \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_a16w8_sink_p128_mtp>( \
                q, k_cache, v_cache, out, sink, false);                                  \
            LAUNCH<pa_decode_traits_d##D_HEAD##_a16w8_sink_p128_mtp>(__VA_ARGS__);       \
        }                                                                                \
        else if((MTP) && (PAGE) == 256)                                                  \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_a16w8_sink_p256_mtp>( \
                q, k_cache, v_cache, out, sink, false);                                  \
            LAUNCH<pa_decode_traits_d##D_HEAD##_a16w8_sink_p256_mtp>(__VA_ARGS__);       \
        }                                                                                \
        else if((FP8_Q) && (PAGE) == 16)                                                 \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_fp8_sink>(          \
                q, k_cache, v_cache, out, sink, true);                                   \
            LAUNCH<pa_decode_traits_d##D_HEAD##_fp8_sink>(__VA_ARGS__);                  \
        }                                                                                \
        else if((FP8_Q) && (PAGE) == 128)                                                \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_fp8_sink_p128>(     \
                q, k_cache, v_cache, out, sink, true);                                   \
            LAUNCH<pa_decode_traits_d##D_HEAD##_fp8_sink_p128>(__VA_ARGS__);             \
        }                                                                                \
        else if((FP8_Q) && (PAGE) == 256)                                                \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_fp8_sink_p256>(     \
                q, k_cache, v_cache, out, sink, true);                                   \
            LAUNCH<pa_decode_traits_d##D_HEAD##_fp8_sink_p256>(__VA_ARGS__);             \
        }                                                                                \
        else if((PAGE) == 16)                                                            \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_a16w8_sink>(        \
                q, k_cache, v_cache, out, sink, false);                                  \
            LAUNCH<pa_decode_traits_d##D_HEAD##_a16w8_sink>(__VA_ARGS__);                \
        }                                                                                \
        else if((PAGE) == 128)                                                           \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_a16w8_sink_p128>(   \
                q, k_cache, v_cache, out, sink, false);                                  \
            LAUNCH<pa_decode_traits_d##D_HEAD##_a16w8_sink_p128>(__VA_ARGS__);           \
        }                                                                                \
        else if((PAGE) == 256)                                                           \
        {                                                                                \
            pa_decode_opus_gptoss_check<pa_decode_traits_d##D_HEAD##_a16w8_sink_p256>(   \
                q, k_cache, v_cache, out, sink, false);                                  \
            LAUNCH<pa_decode_traits_d##D_HEAD##_a16w8_sink_p256>(__VA_ARGS__);           \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            AITER_CHECK(false, "gpt-oss decode compiles page size 16, 128 or 256, got ", \
                        (PAGE));                                                         \
        }                                                                                \
    } while(0)

#define PA_GPTOSS_DISPATCH(FP8_Q, PAGE, MTP, LAUNCH, ...)                                  \
    do                                                                                   \
    {                                                                                    \
        const int head_dim = static_cast<int>(q.size(q.dim() - 1));                       \
        if(head_dim == 64)                                                               \
        {                                                                                \
            PA_GPTOSS_DISPATCH_DIM(64, FP8_Q, PAGE, MTP, LAUNCH, __VA_ARGS__);             \
        }                                                                                \
        else if(head_dim == 128)                                                         \
        {                                                                                \
            PA_GPTOSS_DISPATCH_DIM(128, FP8_Q, PAGE, MTP, LAUNCH, __VA_ARGS__);            \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            AITER_CHECK(false, "gpt-oss decode compiles head_dim 64 or 128, got ", head_dim); \
        }                                                                                \
    } while(0)

void pa_decode_opus_gptoss_fwd(aiter_tensor_t& q,
                               aiter_tensor_t& k_cache,
                               aiter_tensor_t& v_cache,
                               aiter_tensor_t& block_tables,
                               aiter_tensor_t& context_lens,
                               aiter_tensor_t& out,
                               aiter_tensor_t& sink,
                               float softmax_scale,
                               float q_scale,
                               float k_scale,
                               float v_scale)
{
    const bool fp8_q = q.dtype() == AITER_DTYPE_fp8;
    if(v_cache.dim() == 5)
    {
        AITER_CHECK(fp8_q && k_cache.dtype() == AITER_DTYPE_fp8
                    && v_cache.dtype() == AITER_DTYPE_fp8,
                    "shuffled V requires FP8 Q/K/V");
        AITER_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 128,
                    "shuffled V requires single-token Q [batch, 64, 128]");
        AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bf16");
        AITER_CHECK(sink.dtype() == AITER_DTYPE_fp32 && sink.dim() == 1 && sink.size(0) == 64,
                    "sink must be fp32 [64]");
        AITER_CHECK(q_scale > 0.0f && k_scale > 0.0f && v_scale > 0.0f,
                    "q/k/v scales must be positive");
        pa_decode_opus_launch<pa_decode_shuffled_v_traits<pa_decode_traits_d128_fp8_sink_p128>>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale,
            q_scale * k_scale, v_scale, &sink);
        return;
    }
    const int page   = static_cast<int>(v_cache.size(3));
    // Without a pre-quantized Q there is no q_scale to fold; the kernel derives one
    // per query row instead.
    const float qk = fp8_q ? q_scale * k_scale : k_scale;
    // 4-D q is MTP. qlen == 1 is squeezed to 3-D in Python so decode keeps the
    // single peeled tile.
    const bool mtp = q.dim() == 4;

    PA_GPTOSS_DISPATCH(fp8_q, page, mtp, pa_decode_opus_launch, q, k_cache, v_cache,
                       block_tables, context_lens, out, softmax_scale, qk, v_scale, &sink);
}

void pa_decode_opus_gptoss_ps_fwd(aiter_tensor_t& q,
                                  aiter_tensor_t& k_cache,
                                  aiter_tensor_t& v_cache,
                                  aiter_tensor_t& kv_indptr,
                                  aiter_tensor_t& kv_indices,
                                  aiter_tensor_t& context_lens,
                                  aiter_tensor_t& out,
                                  aiter_tensor_t& sink,
                                  aiter_tensor_t& work_indptr,
                                  aiter_tensor_t& work_info,
                                  aiter_tensor_t& split_o,
                                  aiter_tensor_t& split_lse,
                                  float softmax_scale,
                                  float q_scale,
                                  float k_scale,
                                  float v_scale)
{
    const bool fp8_q = q.dtype() == AITER_DTYPE_fp8;
    const int page   = static_cast<int>(v_cache.size(3));
    const float qk   = fp8_q ? q_scale * k_scale : k_scale;
    AITER_CHECK(q.dim() == 3, "persistent gpt-oss decode does not support MTP (4-D q)");

    PA_GPTOSS_DISPATCH(fp8_q, page, false, pa_decode_opus_ps_launch, q, k_cache, v_cache,
                       kv_indptr, kv_indices, context_lens, out, work_indptr, work_info,
                       split_o, split_lse, softmax_scale, qk, v_scale, &sink);
}

template<class W>
static void pa_decode_opus_a16w8_maybe_q_split(bool q_split,
                                               aiter_tensor_t& q,
                                               aiter_tensor_t& k_cache,
                                               aiter_tensor_t& v_cache,
                                               aiter_tensor_t& block_tables,
                                               aiter_tensor_t& context_lens,
                                               aiter_tensor_t& out,
                                               float softmax_scale,
                                               float qk_dequant,
                                               float v_dequant,
                                               aiter_tensor_t* k_scale_map,
                                               aiter_tensor_t* v_scale_map)
{
    if constexpr(W::MTP_Q_LOOP)
    {
        if(q_split)
            pa_decode_opus_launch<pa_decode_q_split_traits<W>>(
                q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk_dequant,
                v_dequant, nullptr, k_scale_map, v_scale_map);
        else
            pa_decode_opus_launch<W>(
                q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk_dequant,
                v_dequant, nullptr, k_scale_map, v_scale_map);
    }
    else
        pa_decode_opus_launch<W>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk_dequant,
            v_dequant, nullptr, k_scale_map, v_scale_map);
}

template<class Base>
static int pa_decode_opus_a16w8_fused_np(int batch, int nkv, int max_tiles, int num_cu)
{
    // Partition count with query-split disabled, used by the query-split gate:
    // qlen * B * nkv * NP <= 2 * CU.
    const int min_tiles = PA_DECODE_MIN_TILES_PER_SPLIT;
    const int base      = batch * nkv;
    if(base <= 0) return 1;
    int wg_budget  = num_cu * Base::WGS_PER_CU;
    int num_splits = 1;
    if(base < wg_budget && max_tiles >= 2 * min_tiles)
    {
        num_splits = wg_budget / base;
        if(num_splits > max_tiles / min_tiles) num_splits = max_tiles / min_tiles;
        if(num_splits > Base::MAX_SPLITS) num_splits = Base::MAX_SPLITS;
        if(num_splits < 1) num_splits = 1;
    }
    if constexpr(Base::QUANT_Q && !Base::HAS_SINK && Base::IS_FP8 && Base::D_HEAD == 128)
    {
        if(max_tiles >= 256)
        {
            int fill_cap = Base::MTP_Q_LOOP ? 16 : 8;
            const int cu_fill = num_cu / base;
            if(cu_fill > fill_cap) fill_cap = cu_fill;
            if constexpr(PA_DECODE_OPUS_MAX_FILL_SPLITS > 0)
                fill_cap = PA_DECODE_OPUS_MAX_FILL_SPLITS;
            if(const char* env = std::getenv("PA_DECODE_OPUS_MAX_FILL_SPLITS"))
            {
                fill_cap = std::atoi(env);
                if(fill_cap < 1) fill_cap = 1;
            }
            if(fill_cap > Base::MAX_SPLITS) fill_cap = Base::MAX_SPLITS;
            int target            = fill_cap;
            const int scratch_fit = (num_cu * PA_DECODE_WGS_PER_CU) / base;
            if(target > scratch_fit) target = scratch_fit;
            if(target > max_tiles / min_tiles) target = max_tiles / min_tiles;
            if(target > Base::MAX_SPLITS) target = Base::MAX_SPLITS;
            if(target < 1) target = 1;
            num_splits = target;
        }
    }
    return num_splits;
}

template<class Base>
static void pa_decode_opus_a16w8_launch(aiter_tensor_t& q,
                                        aiter_tensor_t& k_cache,
                                        aiter_tensor_t& v_cache,
                                        aiter_tensor_t& block_tables,
                                        aiter_tensor_t& context_lens,
                                        aiter_tensor_t& out,
                                        float softmax_scale,
                                        float qk_dequant,
                                        float v_dequant,
                                        bool trans_v,
                                        aiter_tensor_t* k_scale_map,
                                        aiter_tensor_t* v_scale_map)
{
    const bool per_token = k_scale_map != nullptr;
    int num_cu           = 0;
    HIP_CALL(hipDeviceGetAttribute(
        &num_cu, hipDeviceAttributeMultiprocessorCount, q.device_id));
    const int batch = static_cast<int>(q.size(0));
    const int nkv   = static_cast<int>(k_cache.size(1));
    const int qlen  = q.dim() == 4 ? static_cast<int>(q.size(1)) : 1;
    const int max_tiles =
        (static_cast<int>(block_tables.size(1)) * Base::PAGE_SIZE + Base::KV_TILE - 1)
        / Base::KV_TILE;
    // Split MTP2/4 across CTAs only while
    // qlen * B * nkv * NP <= 2 * CU. NP=256 on B=1 Q4 is 1024 > 512, so that
    // launch fuses the four tokens and keeps one WG per CU.
    const int np_gate = pa_decode_opus_a16w8_fused_np<Base>(batch, nkv, max_tiles, num_cu);
    const bool q_split = Base::MTP_Q_LOOP && (qlen == 2 || qlen == 4) && num_cu > 0
                         && static_cast<int64_t>(qlen) * batch * nkv * np_gate
                                <= static_cast<int64_t>(2) * num_cu;
    if(trans_v && per_token)
        pa_decode_opus_a16w8_maybe_q_split<
            pa_decode_per_token_traits<pa_decode_v_trans_traits<Base>>>(
            q_split, q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale,
            qk_dequant, v_dequant, k_scale_map, v_scale_map);
    else if(trans_v)
        pa_decode_opus_a16w8_maybe_q_split<pa_decode_v_trans_traits<Base>>(
            q_split, q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale,
            qk_dequant, v_dequant, k_scale_map, v_scale_map);
    else if(per_token)
        pa_decode_opus_a16w8_maybe_q_split<pa_decode_per_token_traits<Base>>(
            q_split, q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale,
            qk_dequant, v_dequant, k_scale_map, v_scale_map);
    else
        pa_decode_opus_a16w8_maybe_q_split<Base>(
            q_split, q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale,
            qk_dequant, v_dequant, k_scale_map, v_scale_map);
}

void pa_decode_opus_a16w8_fwd(aiter_tensor_t& q,
                              aiter_tensor_t& k_cache,
                              aiter_tensor_t& v_cache,
                              aiter_tensor_t& block_tables,
                              aiter_tensor_t& context_lens,
                              aiter_tensor_t& out,
                              float softmax_scale,
                              float k_scale,
                              float v_scale,
                              std::optional<aiter_tensor_t> k_scale_map,
                              std::optional<aiter_tensor_t> v_scale_map)
{
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16, "pa_decode_opus_a16w8_fwd takes bf16 q");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_fp8 && v_cache.dtype() == AITER_DTYPE_fp8,
                "k_cache/v_cache must be fp8 (A16W8)");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out is bf16");
    AITER_CHECK(k_scale > 0.0f && v_scale > 0.0f, "k/v scales must be positive");

    const bool trans_v = v_cache.dim() == 5;
    const int page     = trans_v ? static_cast<int>(v_cache.size(2) * v_cache.size(4))
                                 : static_cast<int>(v_cache.size(3));
    AITER_CHECK(k_scale_map.has_value() == v_scale_map.has_value(),
                "k_scale_map and v_scale_map must be supplied together");
    aiter_tensor_t* ks = k_scale_map ? &*k_scale_map : nullptr;
    aiter_tensor_t* vs = v_scale_map ? &*v_scale_map : nullptr;
    const float qk     = ks ? 1.0f : k_scale;
    const float vd     = vs ? 1.0f : v_scale;

    // No q_scale to fold: the kernel derives one per query row and applies it itself.
    // Python squeezes qlen == 1 back to 3-D, so a 4-D q here is real MTP.
    const bool mtp    = q.dim() == 4;
    const int qlen    = mtp ? static_cast<int>(q.size(1)) : 1;
    const int gqa     = static_cast<int>(q.size(mtp ? 2 : 1)) / static_cast<int>(k_cache.size(1));
    const bool q_loop = mtp && qlen * gqa > 16;

    if(q_loop && page == 16)
        pa_decode_opus_a16w8_launch<pa_decode_traits_d128_a16w8_mtp_loop>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk, vd,
            trans_v, ks, vs);
    else if(q_loop && page == 128)
        pa_decode_opus_a16w8_launch<pa_decode_traits_d128_a16w8_p128_mtp_loop>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk, vd,
            trans_v, ks, vs);
    else if(mtp && page == 16)
        pa_decode_opus_a16w8_launch<pa_decode_traits_d128_a16w8_mtp>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk, vd,
            trans_v, ks, vs);
    else if(mtp && page == 128)
        pa_decode_opus_a16w8_launch<pa_decode_traits_d128_a16w8_p128_mtp>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk, vd,
            trans_v, ks, vs);
    else if(page == 16)
        pa_decode_opus_a16w8_launch<pa_decode_traits_d128_a16w8>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk, vd,
            trans_v, ks, vs);
    else if(page == 128)
        pa_decode_opus_a16w8_launch<pa_decode_traits_d128_a16w8_p128>(
            q, k_cache, v_cache, block_tables, context_lens, out, softmax_scale, qk, vd,
            trans_v, ks, vs);
    else
        AITER_CHECK(false, "A16W8 decode compiles page size 16 or 128, got ", page);
}

void pa_decode_opus_a16w8_ps_fwd(aiter_tensor_t& q,
                                 aiter_tensor_t& k_cache,
                                 aiter_tensor_t& v_cache,
                                 aiter_tensor_t& kv_indptr,
                                 aiter_tensor_t& kv_indices,
                                 aiter_tensor_t& context_lens,
                                 aiter_tensor_t& out,
                                 aiter_tensor_t& work_indptr,
                                 aiter_tensor_t& work_info,
                                 aiter_tensor_t& split_o,
                                 aiter_tensor_t& split_lse,
                                 float softmax_scale,
                                 float k_scale,
                                 float v_scale)
{
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16, "pa_decode_opus_a16w8_ps_fwd takes bf16 q");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_fp8 && v_cache.dtype() == AITER_DTYPE_fp8,
                "k_cache/v_cache must be fp8 (A16W8)");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out is bf16");
    AITER_CHECK(k_scale > 0.0f && v_scale > 0.0f, "k/v scales must be positive");
    AITER_CHECK(q.dim() == 3, "persistent A16W8 decode does not support MTP (4-D q)");

    const int page = static_cast<int>(v_cache.size(3));
    if(page == 16)
        pa_decode_opus_ps_launch<pa_decode_traits_d128_a16w8>(q,
                                                             k_cache,
                                                             v_cache,
                                                             kv_indptr,
                                                             kv_indices,
                                                             context_lens,
                                                             out,
                                                             work_indptr,
                                                             work_info,
                                                             split_o,
                                                             split_lse,
                                                             softmax_scale,
                                                             k_scale,
                                                             v_scale);
    else if(page == 128)
        pa_decode_opus_ps_launch<pa_decode_traits_d128_a16w8_p128>(q,
                                                                  k_cache,
                                                                  v_cache,
                                                                  kv_indptr,
                                                                  kv_indices,
                                                                  context_lens,
                                                                  out,
                                                                  work_indptr,
                                                                  work_info,
                                                                  split_o,
                                                                  split_lse,
                                                                  softmax_scale,
                                                                  k_scale,
                                                                  v_scale);
    else
        AITER_CHECK(false, "A16W8 decode compiles page size 16 or 128, got ", page);
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS-based paged-attention decode.
// Hosts launcher + validation on top of the device kernel template in
// `pa_decode_opus.h` (single-header, IMPL-guarded).

#define PA_DECODE_OPUS_IMPL
#include "pa_decode_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

#include <mutex>
#include <unordered_map>

// Split-KV scratch, allocated once per stream and kept for the process.
//
// A per-call hipMallocAsync/hipFreeAsync pair is not affordable here. At small
// batch the kernel itself takes only a few microseconds, so the launch path --
// two stream-ordered allocator calls on top of the launch itself -- is what the
// benchmark actually measures. Even against a primed pool, the two API calls
// stay on the critical path of every decode step.
//
// A fixed buffer works because the slot count is structurally bounded: splits
// are chosen so that base_wgs * num_splits <= num_cu, so no batch shape can ask
// for more slots than the device has CUs. That caps the scratch at a couple of
// MB, which is worth holding onto permanently. Since it never grows, there is
// also no allocation to trip over during HIP graph capture.
//
// Keyed by stream, like the opus_gemm splitk workspace: concurrent streams must
// not share one buffer or their partials would overwrite each other.
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
    // retire. This buffer is the one place the split workgroups hand data to
    // each other, and on a multi-XCD part the L2s are not coherent between them,
    // so the alternative is a __threadfence() in every split -- which writes back
    // the whole L2 and measured at 28us of the 41us kernel. Bypassing the cache
    // for these few KB is far cheaper than flushing it for everything else.
    HIP_CALL(hipExtMallocWithFlags(&ptr, bytes, hipDeviceMallocUncached));
    // The arrival counters at the tail have to start at zero. This is the only
    // time they are cleared: the kernel's last split resets its own counter, so
    // the buffer comes back clean after every launch.
    HIP_CALL(hipMemset(ptr, 0, bytes));
    reg.map[stream] = ptr;
    return ptr;
}

void pa_decode_opus_fwd(aiter_tensor_t& q,
                        aiter_tensor_t& k_cache,
                        aiter_tensor_t& v_cache,
                        aiter_tensor_t& block_tables,
                        aiter_tensor_t& context_lens,
                        aiter_tensor_t& out,
                        float softmax_scale)
{
    using Traits = pa_decode_traits_d128;

    // ---- Shape / dtype validation -----------------------------------------
    AITER_CHECK(q.dim() == 3, "q must be 3-D [batch, num_heads, D], got ndim=", q.dim());
    AITER_CHECK(out.dim() == 3, "out must be 3-D [batch, num_heads, D], got ndim=", out.dim());
    AITER_CHECK(k_cache.dim() == 5,
                "k_cache must be 5-D [num_blocks, num_kv_heads, D/x, page, x], got ndim=",
                k_cache.dim());
    AITER_CHECK(v_cache.dim() == 4,
                "v_cache must be 4-D [num_blocks, num_kv_heads, D, page], got ndim=",
                v_cache.dim());
    AITER_CHECK(block_tables.dim() == 2,
                "block_tables must be 2-D [batch, max_blocks_per_batch_row]");
    AITER_CHECK(context_lens.dim() == 1, "context_lens must be 1-D [batch]");

    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16 && out.dtype() == AITER_DTYPE_bf16,
                "pa_decode_opus_fwd currently compiles bf16 only");
    AITER_CHECK(k_cache.dtype() == AITER_DTYPE_bf16 && v_cache.dtype() == AITER_DTYPE_bf16,
                "k_cache/v_cache must be bf16 (A16W16)");
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32, "block_tables must be int32");
    AITER_CHECK(context_lens.dtype() == AITER_DTYPE_i32, "context_lens must be int32");

    const int batch        = static_cast<int>(q.size(0));
    const int num_heads    = static_cast<int>(q.size(1));
    const int head_size    = static_cast<int>(q.size(2)); //head_dim
    const int num_kv_heads = static_cast<int>(k_cache.size(1));
    const int page_size    = static_cast<int>(v_cache.size(3));

    AITER_CHECK(head_size == Traits::D_HEAD,
                "Only head_size=", Traits::D_HEAD, " is compiled, got ", head_size);
    AITER_CHECK(page_size == Traits::PAGE_SIZE,
                "Only page size=", Traits::PAGE_SIZE, " is compiled, got ", page_size);
    AITER_CHECK(static_cast<int>(k_cache.size(4)) == Traits::K_PACK,
                "k_cache pack factor x must be ", Traits::K_PACK, " for bf16");
    AITER_CHECK(static_cast<int>(k_cache.size(2)) == Traits::D_HEAD / Traits::K_PACK,
                "k_cache dim-group count must be D/x");
    AITER_CHECK(static_cast<int>(k_cache.size(3)) == Traits::PAGE_SIZE,
                "k_cache page dim mismatch");
    AITER_CHECK(static_cast<int>(v_cache.size(2)) == Traits::D_HEAD, "v_cache head dim mismatch");
    AITER_CHECK(static_cast<int>(v_cache.size(1)) == num_kv_heads,
                "k_cache/v_cache must agree on num_kv_heads");

    AITER_CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0,
                "num_heads must be divisible by num_kv_heads");
    const int gqa_ratio = num_heads / num_kv_heads;
    AITER_CHECK(gqa_ratio <= Traits::Q_TILE,
                "GQA ratio must be <= ", Traits::Q_TILE, ", got ", gqa_ratio);

    AITER_CHECK(out.size(0) == batch && out.size(1) == num_heads && out.size(2) == head_size,
                "out shape must match q");
    AITER_CHECK(block_tables.size(0) == batch, "block_tables first dim must be batch");
    AITER_CHECK(context_lens.size(0) == batch, "context_lens length must be batch");

    AITER_CHECK(q.stride(2) == 1 && out.stride(2) == 1,
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
    kargs.stride_q_h               = static_cast<int>(q.stride(1));
    kargs.stride_o_b               = static_cast<int>(out.stride(0));
    kargs.stride_o_h               = static_cast<int>(out.stride(1));
    kargs.stride_k_blk             = static_cast<int>(k_cache.stride(0));
    kargs.stride_k_h               = static_cast<int>(k_cache.stride(1));
    kargs.stride_v_blk             = static_cast<int>(v_cache.stride(0));
    kargs.stride_v_h               = static_cast<int>(v_cache.stride(1));
    kargs.softmax_scale            = softmax_scale;

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    // ---- Pick a split count --------------------------------------------------
    //
    // The natural grid is one workgroup per (kv head, batch), which leaves the
    // GPU mostly idle at small batch: a single sequence with 8 kv heads occupies 8
    // of 256 CUs. Splitting the KV axis buys back that parallelism, at the cost of
    // a second pass to merge the per-split softmax partials. Large batches already
    // fill the machine, so they keep the single-pass path.
    //
    // The second pass is not free: it costs a workspace allocation, an extra launch
    // and a round trip through global memory, which measures at roughly 2us. Short
    // contexts have to be kept off this path or the merge outweighs the parallelism
    // it buys, so every split is required to carry a few tiles of its own.
    constexpr int min_tiles_per_split = 4;
    int num_cu = 0;
    HIP_CALL(hipDeviceGetAttribute(&num_cu, hipDeviceAttributeMultiprocessorCount, q.device_id));

    const int base_wgs  = num_kv_heads * batch; //base workgroups
    const int max_tiles = (kargs.max_blocks_per_batch_row * Traits::PAGE_SIZE + Traits::KV_TILE - 1)
                            / Traits::KV_TILE;
    int num_splits = 1;
    if(base_wgs < num_cu && max_tiles >= 2 * min_tiles_per_split)
    {
        num_splits = num_cu / base_wgs;
        if(num_splits > max_tiles / min_tiles_per_split) num_splits = max_tiles / min_tiles_per_split;
        if(num_splits > Traits::MAX_SPLITS) num_splits = Traits::MAX_SPLITS;
        if(num_splits < 1) num_splits = 1;
    }
    kargs.num_splits = num_splits;

    // ---- Launch --------------------------------------------------------------
    dim3 block(Traits::BLOCK_SIZE);

    if(num_splits == 1)
    {
        pa_decode_opus_kernel<Traits><<<dim3(num_kv_heads, batch, 1), block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
        return;
    }

    // One slot holds a split's partial O plus its (m, l) pair. The buffer is
    // cached per stream and never resized, so it is sized for the widest launch
    // the heuristic can ask for rather than for this one: num_splits is capped at
    // num_cu / base_wgs, which bounds the slot count by num_cu.
    const size_t slot_max  = static_cast<size_t>(num_cu);
    const size_t slots     = static_cast<size_t>(batch) * num_kv_heads * num_splits;
    const size_t o_bytes   = slots * Traits::Q_TILE * Traits::D_HEAD * sizeof(float);
    const size_t ml_bytes  = slot_max * Traits::Q_TILE * 2 * sizeof(float);
    const size_t o_max     = slot_max * Traits::Q_TILE * Traits::D_HEAD * sizeof(float);
    // One counter per (batch, kv-head); those are bounded by the slot count too.
    const size_t ctr_bytes = slot_max * sizeof(unsigned int);
    const size_t bytes_max = o_max + ml_bytes + ctr_bytes;
    AITER_CHECK(slots <= slot_max, "split-KV slot count ", slots, " exceeds the CU bound ", slot_max);

    char* scratch          = static_cast<char*>(split_scratch_get(stream, bytes_max));
    kargs.partial_o        = reinterpret_cast<float*>(scratch);
    kargs.partial_ml       = reinterpret_cast<float*>(scratch + o_bytes);
    kargs.split_counters   = reinterpret_cast<unsigned int*>(scratch + o_max + ml_bytes);

    // One launch: the split that arrives last merges the partials in place,
    // rather than a second grid being stood up to do it. That pass measured
    // 4-5us on every shape it ran for, essentially all of it dispatch cost.
    pa_decode_opus_kernel<Traits>
        <<<dim3(num_kv_heads, batch, num_splits), block, 0, stream>>>(kargs);
    HIP_CALL_LAUNCH(hipGetLastError());
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Stage 1 of the opus MLA decode kernels. These launch ONLY the decode kernel:
// work partitioning comes from aiter's metadata (work_indptr / work_info_set)
// and the split-KV merge is left to mla_reduce_v1. Per-split partials go to
// logits/attn_lse (== aiter split_output/split_lse); work items that own a whole
// request (partial_slot < 0) write the final output directly. gfx950 only.
//
// Two entry points, both in this translation unit:
//   opus_mla_decode_fwd        merged d=576 Q/KV, dispatched on q's dtype:
//                                bf16 -- prebuilt code objects under
//                                        hsa/<arch>/mla_opus/, because they need a
//                                        toolchain aiter does not ship.
//                                fp8  -- per-tensor scalar descales (q_scale /
//                                        kv_scale); routed by nhead x max_seqlen_q
//                                        in opus_a8w8_route.
//   opus_mla_decode_mxfp8_fwd  fp8 NoPE with per-block E8M0 scales + bf16 RoPE,
//                              so Q and KV are three tensors each.
//
// Bound to Python through the torch-free ctypes C ABI (aiter_tensor_t* +
// trailing hipStream_t), so this .so carries no pybind/libtorch dependency.

#include "aiter_tensor.h"

#include "aiter_ctypes_error.h"

#include "aiter_hip_common.h"
#include "opus/mla_decode_a8w8_16mx1_32nx4.hpp"
#include "opus/mla_decode_kargs.h"
#include "opus/mla_decode_mxfp8_16mx8_32nx1.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <string>

AITER_CTYPES_ERROR_DEF

namespace {

// fp8 32mx4 / 64nx1 (csrc/kernels/mla/opus/mla_decode_a8w8_32mx4_64nx1.hpp): 4 waves x 32 packed
// query rows, i.e. the 128-row work items aiter's metadata packs by default for fp8. Prebuilt
// only: its register map (Q, half of O and the K/V ring in AGPRs, the other half of O and both
// score buffers in VGPRs) needs the pin toolchain, and a stock ROCm clang forces every MFMA into
// AGPR form, which cannot fit. One entry per specialization.
constexpr const char* kA8W8_32mx4_co = "mla_opus/opus_mla_decode_a8w8_32mx4_64nx1.co";
// fp8 32mx2 / 32nx2 (mla_decode_a8w8_32mx2_32nx2.hpp): two wave pairs of 32 packed query rows,
// 32x32x64 for both GEMMs, prebuilt the same way (mla_decode_a8w8_32mx2_32nx2.cu). 64 packed
// query rows per workgroup, so a work item must not carry more than 64 rows -- hand it a
// 128-row item and the rows past 64 are never computed -- which the metadata guarantees for
// exactly the shapes opus_a8w8_route sends here (64 rows or fewer per request, and
// kPackedQoLenPerWg 64 for nhead 32 x 3).
constexpr const char* kA8W8_32mx2_co = "mla_opus/opus_mla_decode_a8w8_32mx2_32nx2.co";
constexpr int kCoBlockSize           = 256;

constexpr int kHeadDimQk = 576;
constexpr int kHeadDimVo = 512;

using MxFp8Traits = opus_mla_decode_mxfp8_16mx8_32nx1_traits<16, 32, 8, fp8_t, bf16_t, bf16_t>;

// --- fp8 (16, 1): 4 waves of 32 KV tokens, Q shared through LDS ---------------------
// The row-tiled builds give every wave its own query rows, so they need nhead * max_seqlen_q
// rows to fill a workgroup. At nhead 16 with one query token there are only 16 rows to go
// round, so this build tiles the KV tokens across the waves instead: four waves of 32
// tokens, ping-pong over two LDS slots.
//
// page_size 1 only -- the rope DMA deals a 16-token line, so the within-page token offset
// would be per-lane above 1. The 16nx8 geometry this replaced is the only fp8 build that
// ever addressed a real block table; it still sits in opus/ but nothing routes to it, and
// the python gate now keeps page_size > 1 on the asm path.
template <bool CAUSAL, bool LARGE_KV = false>
using OpusTraits16mx1x32nx4C =
    opus_mla_decode_a8w8_16mx1_32nx4_traits<16, 32, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 2>;
inline bool opus_env_flag(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// Shape routing for the merged-buffer fp8 path, mirroring the asm persistent fp8 heuristic
// (csrc/py_itfs_cu/asm_mla.cu): what asm runs on its qh32 / qseqlen4 kernel (128 packed rows
// per workgroup) goes to 32mx4, what it runs on qh16 / qseqlen4 (64 rows) to 32mx2. nhead
// here is nhead per KV head (MLA has one), after mla.py's fold of the non-native head counts.
// aiter/mla.py keeps the same table so a shape with no build stays on the asm path.
enum class OpusA8W8Route
{
    k32mx4,
    k32mx2,
    k16mx1,
    kNone,
};

inline OpusA8W8Route opus_a8w8_route(int nhead, int max_seqlen_q)
{
    const int q = max_seqlen_q;
    if((nhead == 32 && q >= 4) || (nhead == 64 && q >= 2) || nhead == 128 ||
       (nhead == 96 && q <= 6) || (nhead == 16 && q > 4) || (nhead == 12 && nhead * q <= 128))
        return OpusA8W8Route::k32mx4;
    if((nhead == 16 && (q == 3 || q == 4)) || (nhead == 32 && (q == 2 || q == 3)) ||
       (nhead == 64 && q == 1))
        return OpusA8W8Route::k32mx2;
    if(nhead == 16 && q == 1)
        return OpusA8W8Route::k16mx1;
    return OpusA8W8Route::kNone;
}

struct OpusLaunch16mx1x32nx4
{
    template <class Traits>
    static void
    launch(int num_workers, hipStream_t stream, const opus_mla_decode_a8w8_kargs& kargs)
    {
        opus_mla_decode_a8w8_16mx1_32nx4_kernel<Traits>
            <<<dim3(num_workers, 1, 1), dim3(Traits::BLOCK_SIZE), 0, stream>>>(kargs);
    }
};

// causal x large_kv fan-out for the traits-templated in-tree build (16mx1).
template <template <bool, bool> class TraitsC, class Launcher>
void launch_opus_16mx1(bool causal,
                       int max_seqlen_q,
                       bool large_kv,
                       int num_workers,
                       hipStream_t stream,
                       const opus_mla_decode_a8w8_kargs& kargs)
{
    const bool use_causal = causal && max_seqlen_q > 1;
    auto launch           = [&](auto traits) {
        Launcher::template launch<decltype(traits)>(num_workers, stream, kargs);
    };
    if(use_causal && large_kv)
        launch(TraitsC<true, true>{});
    else if(use_causal)
        launch(TraitsC<true, false>{});
    else if(large_kv)
        launch(TraitsC<false, true>{});
    else
        launch(TraitsC<false, false>{});
}

constexpr const char* kFwdName = "opus_mla_decode_fwd";

// bf16 Q/KV: the prebuilt a16w16 code objects, picked by head count. These builds have no
// causal specialization, so max_seqlen_q and causal do not reach them.
void opus_mla_decode_bf16(opus_mla_decode_kargs kargs, int num_workers, hipStream_t stream)
{
    const int H     = kargs.H;
    size_t arg_size = sizeof(kargs);
    auto launch_co  = [&](AiterAsmKernel& impl, int heads_per_block) {
        const int num_h_blocks = (H + heads_per_block - 1) / heads_per_block;
        impl.launch_kernel(
            {&kargs, &arg_size, num_workers, num_h_blocks, 1, kCoBlockSize, 1, 1, stream});
    };
#define OPUS_A16W16_CO_LAUNCH(name, co, heads_per_block)                                \
    do                                                                                  \
    {                                                                                   \
        static AiterAsmKernel impl("opus_mla_decode_a16w16_" name "_kernel",            \
                                   "mla_opus/opus_mla_decode_a16w16_" co ".co");       \
        launch_co(impl, heads_per_block);                                               \
    } while(0)
    if(H <= 32)
        OPUS_A16W16_CO_LAUNCH("32mx1_16nx4", "32mx1_16nx4", 32);
    else if(H % 128 == 0)
        OPUS_A16W16_CO_LAUNCH("32mx4_32nx1", "32mxt_32nx1", 128);
    else if(H % 96 == 0)
        OPUS_A16W16_CO_LAUNCH("32mx3_32nx1", "32mxt_32nx1", 96);
    else
        OPUS_A16W16_CO_LAUNCH("16mx4_64nx1", "16mx4_64nx1", 64);
#undef OPUS_A16W16_CO_LAUNCH
}

// fp8 NoPE and RoPE merged into one d=576 buffer, with per-tensor scalar descales. No build
// here addresses a block table: the 16mx1 shape's rope DMA deals a 16-token line and so
// cannot carry a per-lane within-page offset, so page_size > 1 belongs to the asm path and
// the python gate keeps it there.
void opus_mla_decode_a8w8(const opus_mla_decode_kargs& base,
                         const aiter_tensor_t* q_scale,
                         const aiter_tensor_t* kv_scale,
                         int max_seqlen_q,
                         bool causal,
                         int num_workers,
                         hipStream_t stream)
{
    AITER_CHECK(max_seqlen_q >= 1, kFwdName, ": fp8 needs max_seqlen_q, got ", max_seqlen_q);
    const OpusA8W8Route route = opus_a8w8_route(base.H, max_seqlen_q);
    AITER_CHECK(route != OpusA8W8Route::kNone,
                kFwdName,
                ": no opus fp8 build for nhead ",
                base.H,
                " x max_seqlen_q ",
                max_seqlen_q,
                " (the python gate keeps such shapes on the asm path)");
    AITER_CHECK(q_scale != nullptr && q_scale->dtype() == AITER_DTYPE_fp32 && q_scale->numel() >= 1,
                kFwdName,
                ": q_scale must be a float scalar tensor");
    AITER_CHECK(kv_scale != nullptr && kv_scale->dtype() == AITER_DTYPE_fp32 &&
                    kv_scale->numel() >= 1,
                kFwdName,
                ": kv_scale must be a float scalar tensor");

    opus_mla_decode_a8w8_kargs kargs{base, q_scale->data_ptr(), kv_scale->data_ptr()};
    const int total_tokens = kargs.total_tokens;

    // A buffer descriptor's num_records is 32 bits, so it cannot span a KV cache of 4 GiB
    // or more; past that the bound wraps and every load beyond it silently returns zero.
    // Unlike the contiguous fmha case there is no way to rebase the descriptor per tile
    // here: page_size is 1, so one KV tile's 32 tokens sit at unrelated, per-lane offsets
    // while a descriptor base is wave-uniform. The large path addresses KV with a flat
    // 64-bit pointer (global_load_lds) instead, which costs ~1-2%, hence the gate.
    const int64_t kv_bytes = static_cast<int64_t>(total_tokens) *
                             static_cast<int64_t>(kargs.stride_kv_page) *
                             static_cast<int64_t>(sizeof(fp8_t));
    const bool large_kv = kv_bytes >= (int64_t{1} << 32);

    // The prebuilt builds: entry <name>_{nc,causal}[_large][_cprr] of one code object each,
    // every entry loaded once on first use.
    const bool use_causal = causal && max_seqlen_q > 1;
    // With one query token every local KV token is in its causal past, CP or not.
    const bool use_cprr = use_causal && kargs.cp_world_size > 1;
    size_t arg_size       = sizeof(kargs);
    auto launch_co        = [&](AiterAsmKernel& impl) {
        impl.launch_kernel({&kargs, &arg_size, num_workers, 1, 1, kCoBlockSize, 1, 1, stream});
    };
#define OPUS_A8W8_CO_ENTRY(name, entry, co)                                            \
    do                                                                                 \
    {                                                                                  \
        static AiterAsmKernel impl("opus_mla_decode_a8w8_" name "_" entry, co);        \
        launch_co(impl);                                                               \
    } while(0)
#define OPUS_A8W8_CO_LAUNCH(name, co)                                                  \
    do                                                                                 \
    {                                                                                  \
        static bool traced = false; /* first call only: the test loops the kernel */   \
        if(!traced && opus_env_flag("AITER_MLA_OPUS_TRACE") && (traced = true))        \
            fprintf(stderr,                                                            \
                    "[opus_mla_decode_a8w8] " name " co causal=%d large_kv=%d"         \
                    " cprr=%d\n",                                                      \
                    use_causal,                                                        \
                    large_kv,                                                          \
                    use_cprr);                                                         \
        if(large_kv && use_cprr)                                                       \
            OPUS_A8W8_CO_ENTRY(name, "causal_large_cprr", co);                         \
        else if(use_cprr)                                                              \
            OPUS_A8W8_CO_ENTRY(name, "causal_cprr", co);                               \
        else if(large_kv && use_causal)                                                \
            OPUS_A8W8_CO_ENTRY(name, "causal_large", co);                              \
        else if(large_kv)                                                              \
            OPUS_A8W8_CO_ENTRY(name, "nc_large", co);                                  \
        else if(use_causal)                                                            \
            OPUS_A8W8_CO_ENTRY(name, "causal", co);                                    \
        else                                                                           \
            OPUS_A8W8_CO_ENTRY(name, "nc", co);                                        \
    } while(0)

    // 128 packed query rows per workgroup on 4 waves.
    if(route == OpusA8W8Route::k32mx4)
    {
        OPUS_A8W8_CO_LAUNCH("32mx4_64nx1", kA8W8_32mx4_co);
        return;
    }

    // 64 packed query rows per workgroup on 4 waves (two pairs of 32).
    if(route == OpusA8W8Route::k32mx2)
    {
        OPUS_A8W8_CO_LAUNCH("32mx2_32nx2", kA8W8_32mx2_co);
        return;
    }
#undef OPUS_A8W8_CO_LAUNCH
#undef OPUS_A8W8_CO_ENTRY

    // route == k16mx1.
    launch_opus_16mx1<OpusTraits16mx1x32nx4C, OpusLaunch16mx1x32nx4>(
        causal, max_seqlen_q, large_kv, num_workers, stream, kargs);
}

} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_fwd,
    (aiter_tensor_t * q,            // [total_q, H, 576] bf16 or fp8 (merged nope+rope)
     aiter_tensor_t* kv,            // [total_tokens, ..., 576], same dtype as q, page_size == 1
     aiter_tensor_t* qo_indptr,     // [B+1]
     aiter_tensor_t* kv_indptr,     // [B+1]
     aiter_tensor_t* kv_indices,    // [num_page_used]
     aiter_tensor_t* work_indptr,   // metadata
     aiter_tensor_t* work_info_set, // metadata
     int page_size,
     float softmax_scale,
     aiter_tensor_t* logits,    // split_output [num_partials, 1, H, 512] fp32
     aiter_tensor_t* attn_lse,  // split_lse    [num_partials, 1, H, 1]   fp32
     aiter_tensor_t* out,       // final [total_q, H, 512] bf16
     aiter_tensor_t* final_lse, // [total_q, H] fp32 (nullable)
     // Everything below was added when the fp8 entry point folded into this one; the bf16
     // arguments above keep their old positions.
     aiter_tensor_t* q_scale,   // fp8: float[1] per-tensor descale; bf16: unused (nullable)
     aiter_tensor_t* kv_scale,  // fp8: float[1] per-tensor descale; bf16: unused (nullable)
     int max_seqlen_q,          // fp8: required, picks the build; bf16: unused
     bool causal,               // fp8: apply the causal mask across the query tokens
     aiter_tensor_t* g_kv_indptr, // fp8 round-robin CP: GLOBAL per-request kv_indptr (nullable)
     int cp_world_size,           // fp8 round-robin CP: ranks; 1 = no CP
     int cp_rank,                 // fp8 round-robin CP: this rank, global pos = j * W + rank
     hipStream_t stream),
    (q,
     kv,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info_set,
     page_size,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     q_scale,
     kv_scale,
     max_seqlen_q,
     causal,
     g_kv_indptr,
     cp_world_size,
     cp_rank,
     stream))
{
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported, got ", page_size);
    AITER_CHECK(q->dtype() == kv->dtype(), __func__, ": q and kv must have the same dtype.");
    AITER_CHECK(q->size(-1) == kHeadDimQk, __func__, ": q last dim must be ", kHeadDimQk, ".");
    AITER_CHECK(kv->size(-1) == kHeadDimQk, __func__, ": kv last dim must be ", kHeadDimQk, ".");
    AITER_CHECK(out->size(-1) == kHeadDimVo, __func__, ": out last dim must be ", kHeadDimVo, ".");

    const int H           = q->size(1);
    const int num_workers = work_indptr->size(0) - 1;
    const HipDeviceGuard device_guard(q->device_id);

    // Merged d=576 buffers: one row per (token, head); rope is the +512 slice.
    opus_mla_decode_kargs kargs{};
    kargs.q_ptr     = q->data_ptr();
    kargs.kv_ptr    = kv->data_ptr();
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H              = H;
    kargs.total_tokens   = static_cast<int>(kv->size(0));
    kargs.softmax_scale  = softmax_scale;
    kargs.stride_q_b     = H * kHeadDimQk;
    kargs.stride_q_h     = kHeadDimQk;
    kargs.stride_o_b     = H * kHeadDimVo;
    kargs.stride_o_h     = kHeadDimVo;
    kargs.stride_kv_page = kHeadDimQk;
    kargs.cp_world_size  = cp_world_size > 1 ? cp_world_size : 1;
    if(kargs.cp_world_size > 1)
    {
        AITER_CHECK(q->dtype() == AITER_DTYPE_fp8,
                    __func__,
                    ": round-robin CP is supported by the fp8 builds only.");
        AITER_CHECK(g_kv_indptr != nullptr && g_kv_indptr->numel() > 0,
                    __func__,
                    ": round-robin CP needs g_kv_indptr.");
        AITER_CHECK(cp_rank >= 0 && cp_rank < cp_world_size,
                    __func__,
                    ": cp_rank ",
                    cp_rank,
                    " out of range for cp_world_size ",
                    cp_world_size);
        kargs.g_kv_indptr = static_cast<const int*>(g_kv_indptr->data_ptr());
        kargs.cp_rank     = cp_rank;
    }

    if(q->dtype() == AITER_DTYPE_bf16)
        opus_mla_decode_bf16(kargs, num_workers, stream);
    else if(q->dtype() == AITER_DTYPE_fp8)
        opus_mla_decode_a8w8(
            kargs, q_scale, kv_scale, max_seqlen_q, causal, num_workers, stream);
    else
        AITER_CHECK(false, __func__, ": q/kv must be bf16 or fp8.");
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    opus_mla_decode_mxfp8_fwd,
    (aiter_tensor_t * q_nope,   // [total_q, H, 512]         fp8
     aiter_tensor_t* q_scale,   // [total_q, H, D_SCALE]     uint8 (E8M0)
     aiter_tensor_t* q_rope,    // [total_q, H, 64]          bf16
     aiter_tensor_t* kv_nope,   // [total_tokens, 512]       fp8
     aiter_tensor_t* kv_scale,  // [total_tokens, D_SCALE]   uint8 (E8M0)
     aiter_tensor_t* kv_rope,   // [total_tokens, 64]        bf16
     aiter_tensor_t* qo_indptr,
     aiter_tensor_t* kv_indptr,
     aiter_tensor_t* kv_indices,
     aiter_tensor_t* work_indptr,
     aiter_tensor_t* work_info_set,
     int page_size,
     float softmax_scale,
     aiter_tensor_t* logits,
     aiter_tensor_t* attn_lse,
     aiter_tensor_t* out,
     aiter_tensor_t* final_lse,
     hipStream_t stream),
    (q_nope,
     q_scale,
     q_rope,
     kv_nope,
     kv_scale,
     kv_rope,
     qo_indptr,
     kv_indptr,
     kv_indices,
     work_indptr,
     work_info_set,
     page_size,
     softmax_scale,
     logits,
     attn_lse,
     out,
     final_lse,
     stream))
{
    using T = MxFp8Traits;
    const std::string gfx = get_gpu_arch();
    AITER_CHECK(
        gfx == "gfx950", __func__, ": unsupported GPU arch '", gfx, "' (supported: gfx950).");
    AITER_CHECK(page_size == 1, __func__, ": only page_size == 1 is supported.");
    AITER_CHECK(q_nope->dtype() == AITER_DTYPE_fp8 && kv_nope->dtype() == AITER_DTYPE_fp8,
                __func__,
                ": q_nope/kv_nope must be fp8.");
    AITER_CHECK(q_rope->dtype() == AITER_DTYPE_bf16 && kv_rope->dtype() == AITER_DTYPE_bf16,
                __func__,
                ": q_rope/kv_rope must be bf16.");
    // The kernel reads the scales as bit_cast<float>(e8m0 << 23); fp32 factors must be
    // converted on the host first.
    AITER_CHECK(q_scale->dtype() == AITER_DTYPE_u8 && kv_scale->dtype() == AITER_DTYPE_u8,
                __func__,
                ": q_scale/kv_scale must be E8M0 uint8.");
    AITER_CHECK(kv_scale->size(-1) == T::D_SCALE_SIZE,
                __func__,
                ": kv_scale last dim must be ",
                T::D_SCALE_SIZE,
                ".");

    const int H           = q_nope->size(1);
    const int total_tokens = kv_nope->size(0);
    const int num_workers = work_indptr->size(0) - 1;

    const HipDeviceGuard device_guard(q_nope->device_id);

    opus_mla_decode_mxfp8_kargs kargs{};
    kargs.q_nope_ptr    = q_nope->data_ptr();
    kargs.q_scale_ptr   = q_scale->data_ptr();
    kargs.q_rope_ptr    = q_rope->data_ptr();
    kargs.kv_nope_ptr   = kv_nope->data_ptr();
    kargs.kv_scale_ptr  = kv_scale->data_ptr();
    kargs.kv_rope_ptr   = kv_rope->data_ptr();
    kargs.out_ptr   = out->data_ptr();
    kargs.lse_ptr   = (final_lse && final_lse->numel() > 0) ? final_lse->data_ptr() : nullptr;
    kargs.o_accum   = logits->data_ptr();
    kargs.lse_accum = attn_lse->data_ptr();
    kargs.q_indptr      = static_cast<const int*>(qo_indptr->data_ptr());
    kargs.kv_indptr     = static_cast<const int*>(kv_indptr->data_ptr());
    kargs.kv_indices    = static_cast<const int*>(kv_indices->data_ptr());
    kargs.work_indptr   = static_cast<const int*>(work_indptr->data_ptr());
    kargs.work_info_set =
        static_cast<const opus_mla_decode_work_info*>(work_info_set->data_ptr());
    kargs.H             = H;
    kargs.total_tokens  = total_tokens;
    kargs.softmax_scale = softmax_scale;
    kargs.stride_q_nope_b      = H * T::D_NOPE_SIZE;
    kargs.stride_q_nope_h      = T::D_NOPE_SIZE;
    kargs.stride_q_scale_b     = H * T::D_SCALE_SIZE;
    kargs.stride_q_scale_h     = T::D_SCALE_SIZE;
    kargs.stride_q_rope_b      = H * T::D_ROPE_SIZE;
    kargs.stride_q_rope_h      = T::D_ROPE_SIZE;
    kargs.stride_o_b           = H * T::D_NOPE_SIZE;
    kargs.stride_o_h           = T::D_NOPE_SIZE;
    kargs.stride_kv_nope_page  = T::D_NOPE_SIZE;
    kargs.stride_kv_scale_page = T::D_SCALE_SIZE;
    kargs.stride_kv_rope_page  = T::D_ROPE_SIZE;

    opus_mla_decode_mxfp8_16mx8_32nx1_kernel<T>
        <<<dim3(num_workers, 1, 1), dim3(T::BLOCK_SIZE), 0, stream>>>(kargs);
}

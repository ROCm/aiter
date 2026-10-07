// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// v4 MLA decode dispatcher
//
// This is a peer of csrc/py_itfs_cu/asm_mla.cu but targets the v4 18-slot
// kernarg ABI which is *binary incompatible* with v3's 14-slot layout:
//
//   slot 8  = raw gqa_ratio          (not s_MQA = gqa_ratio*max_seqlen_q)
//   slot 9  = num_kv_splits          (kernel "passes")
//   slot 10 = total_kv = kv_seq_lens * num_seqs
//   slot 11 = stride_page = page_size * dim_qk_packed (bytes)
//   slot 14 = ptr_STP (split_indptr) -- NEW
//   slot 15 = out_16_nosplit         -- NEW
//   slot 16 = ptr_QROPE              -- NEW
//   slot 17 = ptr_KVROPE             -- NEW
//
// scalar is hardcoded to 1/sqrt(kV4DimNope + kV4DimRope) = 1/sqrt(512),
// independent of head_size (the dispatcher's softmax_scale arg is kept for
// API parity but the kernel itself ignores it).
//
// All kernel selection is driven by hsa/gfx950/mla_v4/mla_v4_asm.csv via
// hsa/codegen.py -m mla_v4 -> asm_mla_v4_configs.hpp -> cfg_mla_v4_asm.

#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mla_v4_configs.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <sys/stat.h>

// Per-.so TLS error storage + aiter_get_last_error / aiter_clear_last_error
// exports. Required so that AITER_CHECK failures in our dispatcher surface as
// RuntimeError in Python instead of aborting the worker process.
AITER_CTYPES_ERROR_DEF

#ifndef EN_MLA_V4_KERNARG_PRELOAD
#define EN_MLA_V4_KERNARG_PRELOAD 1
#endif

struct __attribute__((packed)) MlaV4KernelArgsLegacy
{
    void* ptr_R;
    p2 _p_r; // 0:  splitData (logits) [FP32]
    void* ptr_LSE;
    p2 _p_lse; // 1:  splitLse (attn_lse) [FP32]
    void* ptr_Q;
    p2 _p_q; // 2:  Q packed FP8 + e8m0 scale
    void* ptr_KV;
    p2 _p_kv; // 3:  KV packed FP8
    void* ptr_LTP;
    p2 _p_ltp; // 4:  kv_indptr
    void* ptr_LTD;
    p2 _p_ltd; // 5:  kv_page_indices
    void* ptr_LTL;
    p2 _p_ltl; // 6:  kv_last_page_lens
    float scalar_f;
    p3 _p_sc; // 7:  1.0f/sqrtf(kV4DimNope+kV4DimRope)
    unsigned int s_gqa_ratio;
    p3 _p_gr; // 8:  raw gqa_ratio
    unsigned int s_kv_split;
    p3 _p_ps; // 9:  num_kv_splits == passes
    unsigned int s_total_kv;
    p3 _p_tk; // 10: kv_seq_lens * num_seqs
    unsigned int s_stride_page;
    p3 _p_sp; // 11: page_size * dim_qk_packed (bytes)
    unsigned int s_log2_page;
    p3 _p_lp; // 12: log2(page_size)
    void* ptr_QTP;
    p2 _p_qtp; // 13: qo_indptr
    void* ptr_STP;
    p2 _p_stp; // 14: split_indptr
    unsigned int out_16_nosplit;
    p3 _p_o16; // 15: 0 = fp32 split, 1 = bf16 nosplit
    void* ptr_QROPE;
    p2 _p_qrope; // 16
    void* ptr_KVROPE;
    p2 _p_kvrope; // 17
    void* ptr_sink;
    p2 _p_sink; // 18: [num_heads] FP32 attention sink logit
    void* ptr_valid_split;
    p2 _p_vs; // 19 (0x130)
    unsigned int s_use_valid_split;
    p3 _p_uvs; // 20 (0x140)
};

#if EN_MLA_V4_KERNARG_PRELOAD
struct __attribute__((packed)) MlaV4KernelArgsPreload
{
    void* ptr_R;                    // 0x00  preload  splitData (logits) [FP32] (rw)
    void* ptr_Q;                    // 0x08  preload  Q packed FP8 + e8m0 scale
    void* ptr_KV;                   // 0x10  preload  KV packed FP8
    void* ptr_LTP;                  // 0x18  preload  kv_indptr
    void* ptr_LTL;                  // 0x20  preload  kv_last_page_lens
    void* ptr_QTP;                  // 0x28  preload  qo_indptr
    void* ptr_QROPE;                // 0x30  preload  Q rope BF16
    void* ptr_KVROPE;               // 0x38  preload  KV rope BF16
    float scalar_f;                 // 0x40  preload  1.0f/sqrtf(kV4DimNope+kV4DimRope)
    unsigned int s_gqa_ratio;       // 0x44  preload  q_seq_lens (MQA = gqa*max_seqlen_q)
    unsigned int s_kv_split;        // 0x48  preload  num_kv_splits == passes
    unsigned int s_total_kv;        // 0x4C  preload  kv_seq_lens * num_seqs
    unsigned int out_16_nosplit;    // 0x50  preload  0 = fp32 split, 1 = bf16 nosplit
    void* ptr_LSE;                  // 0x54  tail     splitLse (attn_lse) [FP32] (rw)
    void* ptr_LTD;                  // 0x5C  tail     kv_page_indices
    void* ptr_valid_split;          // 0x64  tail     [num_seqs] i32 scratch (rw)
    unsigned int s_use_valid_split; // 0x6C  tail     gates the valid_split write
    void* ptr_sink;                 // 0x70  tail     [num_heads] FP32 attention sink logit
};
#endif

// ----------------------------------------------------------------------------
// kV4DimNope + kV4DimRope = 448 + 64 = 512. The kernel hardcodes
// 1/sqrt(512) as its softmax pre-scale. 
// ----------------------------------------------------------------------------
static constexpr int kV4DimNope = 448;
static constexpr int kV4DimRope = 64;

// ----------------------------------------------------------------------------
// Kernel selection — mirrors csrc/py_itfs_cu/asm_mla.cu::get_heuristic_kernel_mla
// 1:1 in key set so v3 and v4 stay structurally identical.
//
// Lookup keys: (qType, kvType, Gqa, ps, qSeqLen, prefill, causal, lse).
// `sub_Q` and `page_size` are NOT keys — sub_Q is derived in the dispatcher
// (see the V3-style decision tree below) and page_size comes from KV->size(1).
//
// `num_kv_splits` ("passes") is also NOT a key — the .co supports any value
// at runtime via slot 9 of the kernarg packet.
// ----------------------------------------------------------------------------
static std::string get_heuristic_kernel_mla_v4(const std::string& q_type,
                                               const std::string& kv_type,
                                               int gqa,
                                               int ps,
                                               int prefill,
                                               int causal,
                                               int qseqlen,
                                               int lse,
                                               const std::string& arch_id,
                                               CFG* cfgs)
{
    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.qType != q_type || cfg.kvType != kv_type)
            continue;
        if(cfg.Gqa != gqa || cfg.ps != ps || cfg.prefill != prefill)
            continue;
        if(cfg.causal != causal || cfg.qSeqLen != qseqlen)
            continue;
        if(cfg.lse != lse)
            continue;
        return el.first;
    }
    AITER_CHECK(false,
                __func__,
                ": no shipped variant for "
                " q_type:",
                q_type,
                " kv_type:",
                kv_type,
                " gqa:",
                gqa,
                " ps:",
                ps,
                " qSeqLen:",
                qseqlen,
                " prefill:",
                prefill,
                " causal:",
                causal,
                " lse:",
                lse,
                " arch:",
                arch_id);
    return "";
}

// ----------------------------------------------------------------------------
// AITER_C_ITFS entry — exposed to Python via
//   aiter/ops/attention.py::mla_decode_v4_asm  (@compile_ops ffi_type=ctypes)
//
// Mirrors mla_decode_stage1_asm_fwd in asm_mla.cu shape-wise but writes
// the v4 nm 18-slot kernarg layout. Q/KV/output are aiter_tensor_t* (NOT
// torch::Tensor) — see csrc/include/aiter_tensor.h for the C-friendly POD.
//
// Wrapped in AITER_CTYPES_DEFINE_ENTRYPOINT_VOID so that AITER_CHECK / HIP_CALL
// failures (e.g. unsupported variant lookup, dtype mismatch) surface as a
// clean Python RuntimeError via the aiter_get_last_error TLS bridge instead
// of std::abort()-ing the worker process.
// ----------------------------------------------------------------------------
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mla_decode_v4_asm,
    (aiter_tensor_t * Q,              // [total_query_len, num_heads, head_size]   FP8 packed Q+e8m0
     aiter_tensor_t* qrope,           // [total_query_len, num_heads, kv_rotary]   BF16
     aiter_tensor_t* KV,              // [num_page, page_size, num_kv_heads, head_size] FP8
     aiter_tensor_t* kvrope,          // [num_page, page_size, num_kv_heads, kv_rotary] BF16
     aiter_tensor_t* qo_indptr,       // [num_seqs+1]
     aiter_tensor_t* kv_indptr,       // [num_seqs+1]
     aiter_tensor_t* kv_page_indices, // [num_page_used]
     aiter_tensor_t* split_indptr,      // [num_seqs+1]
     aiter_tensor_t* sink,              // [num_heads] FP32 — see "ptr_sink" note above
     int max_seqlen_q,
     float softmax_scale, // ignored; v4 hardcodes 1/sqrt(512). Kept for API parity.
     int out_16_nosplit,
     int num_kv_splits, //
     // outputs
     aiter_tensor_t*
         splitData, // [num_seqs, num_kv_splits, num_kv_heads, gqa*max_seqlen_q, v_head_dim] FP32
     aiter_tensor_t* splitLse, // [num_seqs, num_kv_splits, num_kv_heads, gqa*max_seqlen_q, 1] FP32
     aiter_tensor_t*
         output, // [total_query_len, num_heads, v_head_dim] BF16 (used when out_16_nosplit==1)
     aiter_tensor_t* valid_split_count, // [num_seqs] int32 scratch (slot 19), nullable
     int use_valid_split_count_reduce,  // slot 20 flag; gates the kernel's valid_split write
     // Moved to the tail: UNUSED on the nm path (page_size=1 -> kv_seq_len comes
     // from the token-level kv_indptr). Nullable; the host guards the deref below
     // and the kernel never loads through the pointer.
     aiter_tensor_t* kv_last_page_lens, // [num_seqs] int32, nullable
     hipStream_t stream),
    (Q,
     qrope,
     KV,
     kvrope,
     qo_indptr,
     kv_indptr,
     kv_page_indices,
     split_indptr,
     sink,
     max_seqlen_q,
     softmax_scale,
     out_16_nosplit,
     num_kv_splits,
     splitData,
     splitLse,
     output,
     valid_split_count,
     use_valid_split_count_reduce,
     kv_last_page_lens,
     stream))
{
    (void)softmax_scale;
    (void)out_16_nosplit;
    const unsigned int out_16_nosplit_derived =
        (num_kv_splits == 1) ? 1u : 0u;
    AITER_CHECK(sink != nullptr, __func__, ": `sink` must not be NULL");
    AITER_CHECK(sink->data_ptr() != nullptr,
                __func__,
                ": `sink` data_ptr is NULL — caller must allocate "
                "even when no sink is desired (use torch.full(-inf))");
    AITER_CHECK(Q->is_contiguous(), __func__, ": only support Q.is_contiguous() for now");
    AITER_CHECK(KV->is_contiguous(), __func__, ": only support KV.is_contiguous() for now");
    AITER_CHECK(qrope->is_contiguous(), __func__, ": only support qrope.is_contiguous()");
    AITER_CHECK(kvrope->is_contiguous(), __func__, ": only support kvrope.is_contiguous()");

    const int num_seqs      = qo_indptr->size(0) - 1;
    const int num_heads     = Q->size(1);
    const int num_kv_heads  = KV->size(2);
    const int gqa_ratio     = num_heads / num_kv_heads;
    const int page_size     = KV->size(1);
    const int dim_qk_packed = KV->size(3); // per-token kernel stride in BYTES (FP8 = 1 byte/elem)

    AITER_CHECK(num_kv_heads == 1, __func__, ": only support num_kv_heads==1 for now");
    AITER_CHECK(Q->size(2) == dim_qk_packed,
                __func__,
                ": Q head_size must equal KV head_size (= dim_qk_packed)");

    const HipDeviceGuard device_guard(Q->device_id);

    // Kernel-hardcoded constants (q_dtype-independent on v4 nm).
    constexpr int qk_elem_dim    = kV4DimNope + kV4DimRope; // 448 + 64 = 512 elems
    const float scalar_f         = 1.0f / std::sqrt(static_cast<float>(qk_elem_dim));
    const unsigned int log2_page = static_cast<unsigned int>(__builtin_ctz(page_size));

    auto fill_common_kargs = [&](auto& a) {
        a.ptr_R       = splitData->data_ptr();
        a.ptr_LSE     = splitLse->data_ptr();
        a.ptr_Q       = Q->data_ptr();
        a.ptr_KV      = KV->data_ptr();
        a.ptr_LTP     = kv_indptr->data_ptr();
        a.ptr_LTD     = kv_page_indices->data_ptr();
        a.ptr_LTL     = kv_last_page_lens ? kv_last_page_lens->data_ptr() : nullptr;
        a.scalar_f    = scalar_f;
        a.s_gqa_ratio = static_cast<unsigned int>(gqa_ratio);
        a.s_kv_split  = static_cast<unsigned int>(num_kv_splits);
        // a.s_total_kv     left as 0 here — set per-arch in fill_gfx1250_kargs below.
        a.ptr_QTP        = qo_indptr->data_ptr();
        a.out_16_nosplit = out_16_nosplit_derived;
        a.ptr_QROPE      = qrope->data_ptr();
        a.ptr_KVROPE     = kvrope->data_ptr();
        a.ptr_sink       = sink->data_ptr();
    };

    // gfx1250-specific overrides (shared by both legacy-on-gfx1250 and the
    // compact preload ABI): s_gqa_ratio carries the flattened MQA, s_total_kv
    // is real, and the valid_split scratch (slot 19/20) is validated+forwarded.
    auto fill_gfx1250_kargs = [&](auto& a) {
        a.s_gqa_ratio = static_cast<unsigned int>(gqa_ratio * max_seqlen_q);
        a.s_total_kv  = static_cast<unsigned int>(KV->size(0) * page_size);
        if(use_valid_split_count_reduce != 0)
        {
            AITER_CHECK(valid_split_count != nullptr && valid_split_count->data_ptr() != nullptr,
                        __func__,
                        ": gfx1250 requires valid_split_count scratch tensor when "
                        "use_valid_split_count_reduce!=0");
        }
        if(valid_split_count != nullptr && valid_split_count->data_ptr() != nullptr)
        {
            AITER_CHECK(valid_split_count->dtype() == AITER_DTYPE_i32,
                        __func__,
                        ": valid_split_count must be int32");
            AITER_CHECK(valid_split_count->size(0) >= num_seqs,
                        __func__,
                        ": valid_split_count must have at least num_seqs entries");
            a.ptr_valid_split = valid_split_count->data_ptr();
        }
        else
        {
            a.ptr_valid_split = nullptr;
        }
        a.s_use_valid_split = (use_valid_split_count_reduce != 0) ? 1u : 0u;
    };

    // dtype dispatch
    auto q_dtype  = Q->dtype();
    auto kv_dtype = KV->dtype();
    std::string q_type, kv_type;
    if(q_dtype == AITER_DTYPE_fp8)
        q_type = "fp8";
    else if(q_dtype == AITER_DTYPE_bf16)
        q_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport Q dtype:", AiterDtype_to_str(q_dtype));

    if(kv_dtype == AITER_DTYPE_fp8)
        kv_type = "fp8";
    else if(kv_dtype == AITER_DTYPE_bf16)
        kv_type = "bf16";
    else
        AITER_CHECK(false, __func__, ": unsupport KV dtype:", AiterDtype_to_str(kv_dtype));

    // ------------------------------------------------------------------
    // V3-style per-shape heuristic. Mirrors the decision tree in
    // csrc/py_itfs_cu/asm_mla.cu (~lines 272-318) for gqa_ratio=16 fp8;
    // produces a `sub_Q` (per-WG Q tile, used in grid math) and a
    // `config_max_seqlen_q` (padded qseq used as CSV lookup key against
    // `qSeqLen`). v4 nm ships exactly one variant today; the heuristic
    // mirrors V3's structure so adding future variants is mechanical.
    // ------------------------------------------------------------------
    int sub_Q               = 64; // default (matches V3 default)
    int config_max_seqlen_q = max_seqlen_q;
    int ps                  = 0; // v4 nm always non-persistent today
    int prefill             = 0; // decode stage
    int causal              = 0;
    int lse_flag            = 0;

    // Supported (gqa, max_seqlen_q) entry points for fp8/fp8. The v4 nm .co
    // ships a single 64 q-row tile, so a pair is serviceable iff gqa*msq <= 64
    // AND it is on the whitelist below. Currently:
    //   gqa=16  -> msq in {1, 2, 4}
    //   gqa=32  -> msq == 1   (msq=2 deliberately narrowed out)
    //   gqa=64  -> msq == 1
    //   gqa=128 -> msq == 1
    // A single `supported` predicate drives BOTH the (sub_Q, config) setup and
    // the CSV lookup-key normalization below, so the two can never disagree.
    const bool fp8 = (q_type == "fp8" && kv_type == "fp8");
    bool supported = false;
    if(fp8)
    {
        switch(gqa_ratio)
        {
        case 16: supported = (max_seqlen_q == 1 || max_seqlen_q == 2 ||
                              max_seqlen_q == 4); break;
        case 32:
        case 64:
        case 128: supported = (max_seqlen_q == 1); break;
        default: break;
        }
    }

    // For a supported pair: sub_Q = min(64, gqa*msq) (capped by the 64 q-row
    // tile) and config_max_seqlen_q = msq. Unsupported pairs are left at the
    // defaults (sub_Q=64, config=max_seqlen_q); they will NOT match the CSV
    // normalization below, so the kernel lookup fails loudly (no silent
    // downgrade to a different msq).
    if(supported)
    {
        sub_Q               = std::min(64, gqa_ratio * max_seqlen_q);
        config_max_seqlen_q = max_seqlen_q;
    }

    // ---- CSV lookup-key normalization ---------------------------------------
    // v4 nm ships ONE kernel binary (the 32n-tile .co, symbol
    // mla_a8w8_qh64_qseqlen1_gqaratio64_nm). Its 64 q-row tile satisfies the
    // invariant `gqa * q_seq_logical = 64`, so it serves all three shipped
    // entry points: (gqa=16, qSeqLen=4), (gqa=64, qSeqLen=1) and (gqa=128,
    // qSeqLen=1). The CSV carries a single (Gqa=16, qSeqLen=4) row; normalize
    // every supported (gqa, qSeqLen) caller to that lookup key here. `sub_Q`
    // and the per-launch grid geometry stay set to the gqa-correct values from
    // the heuristic above — this remap only picks which CSV row (== which .co)
    // to load.
    const std::string arch_id = get_gpu_arch();
    const bool is_gfx1250     = (arch_id == "gfx1250");
    int csv_gqa               = gqa_ratio;
    int csv_qseqlen           = config_max_seqlen_q;
    if(!is_gfx1250)
    {
        // `supported` (fp8-gated, set in the sub_Q block above) drives BOTH the
        // (sub_Q, config) setup AND this CSV lookup-key normalization, so the two
        // can never disagree -- every whitelisted (gqa, msq) pair maps to the
        // single shipped (Gqa=64, qSeqLen=1) row. NOTE: an intermediate change
        // had narrowed this to only (gqa16,msq4)/(gqa64|128,msq1), which dropped
        // the gqa16+msq{1,2} and gqa32+msq1 entry points -- they hit "cannot find
        // suitable kernel" even though the qh64 .co serves them and the sub_Q
        // block still whitelists them (see test_v4_nm_gqa16_qseqlen1_*).
        if(supported)
        {
            csv_gqa     = 64;
            csv_qseqlen = 1;
        }
    }
    else
    {
        // shared kernel mla_a8w8_qh64_1tg_16mx4_64nx1_sparse
        if(q_type == "fp8" && kv_type == "fp8" &&
           ((gqa_ratio == 64 || gqa_ratio == 128) && config_max_seqlen_q == 1))
        {
            csv_gqa     = 64;
            csv_qseqlen = 1;
        }
    }

    const int block_dim = 4 * static_cast<int>(get_warp_size_func());

#if EN_MLA_V4_KERNARG_PRELOAD
    const bool use_preload = is_gfx1250;
#else
    const bool use_preload = false;
#endif

    MlaV4KernelArgsLegacy args_legacy = {};
#if EN_MLA_V4_KERNARG_PRELOAD
    MlaV4KernelArgsPreload args_preload = {};
#endif
    void* arg_buf   = nullptr;
    size_t arg_size = 0;

    if(use_preload)
    {
#if EN_MLA_V4_KERNARG_PRELOAD
        fill_common_kargs(args_preload);
        fill_gfx1250_kargs(args_preload);
        arg_buf  = &args_preload;
        arg_size = sizeof(args_preload);
#endif
    }
    else
    {
        fill_common_kargs(args_legacy);
        args_legacy.s_log2_page = log2_page;
        args_legacy.ptr_STP     = split_indptr->data_ptr();
        if(is_gfx1250)
        {
            fill_gfx1250_kargs(args_legacy);
        }
        else
        {
            if(valid_split_count != nullptr && valid_split_count->data_ptr() != nullptr)
            {
                AITER_CHECK(valid_split_count->dtype() == AITER_DTYPE_i32,
                            __func__,
                            ": valid_split_count must be int32");
                AITER_CHECK(valid_split_count->size(0) >= num_seqs,
                            __func__,
                            ": valid_split_count must have at least num_seqs entries");
                args_legacy.ptr_valid_split = valid_split_count->data_ptr();
            }
            else
            {
                args_legacy.ptr_valid_split = nullptr;
            }
            args_legacy.s_use_valid_split = (use_valid_split_count_reduce != 0) ? 1u : 0u;
        }
        arg_buf  = &args_legacy;
        arg_size = sizeof(args_legacy);
    }

    CFG* config_map = &cfg_mla_v4_asm;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    std::string kernelName = get_heuristic_kernel_mla_v4(
        q_type, kv_type, csv_gqa, ps, prefill, causal, csv_qseqlen, lse_flag, arch_id, config_map);
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");

    AiterAsmKernel* impl_ptr = nullptr;
    auto it                  = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
    {
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);
    }
    AITER_CHECK(impl_ptr != nullptr,
                __func__,
                ": unsupport current data type or shape. please refer to asm_mla_v4.cu");

    // Launch geometry: gdx = ceil(q_seq_lens_internal / sub_Q),
    // gdy = num_seqs, gdz = num_kv_splits. Block dim = 256 (wave64 * 4).
    const int q_seq_lens_internal = gqa_ratio * max_seqlen_q;
    const int gdx                 = (q_seq_lens_internal + sub_Q - 1) / sub_Q;
    const int gdy                 = num_seqs;
    const int gdz                 = num_kv_splits;

    // ----- DEBUG: env-gated 304B kernarg dump for cross-check. -----
    if(const char* dbg = std::getenv("AITER_V4_NM_DUMP_KERNARG"))
    {
        if(dbg[0] == '1')
        {
            fprintf(stderr, "[aiter kernarg %zuB]\n", arg_size);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(arg_buf);
            for(size_t i = 0; i < arg_size; ++i)
            {
                fprintf(stderr, "%02x%s", bytes[i], ((i + 1) % 16 == 0) ? "\n" : " ");
            }
            fprintf(stderr, "[aiter grid (%d,%d,%d) block (%d,1,1)]\n", gdx, gdy, gdz, block_dim);
            fflush(stderr);
        }
    }

    impl_ptr->launch_kernel({arg_buf, &arg_size, gdx, gdy, gdz, block_dim, 1, 1, stream});
}

// ----------------------------------------------------------------------------
// Persistent-scheduling v4 nm decode (CSV row ps=1)
//
// One launch per call: grid (2 * P, 1, 1), block 256. The kernel plans the
// KV split over P partitions in-kernel from kv_indptr, runs the attention
// and merges the split partials itself (the last arriving partition of a row
// combines it), so there is no host split plan and no stage-2 merge.
// Same math and packed Q / KV / rope layouts as mla_decode_v4_asm with
// gqa=128, max_seqlen_q=1 and page_size=1; rows with K=0 are left unwritten.
//
// The kernarg block keeps the 21 x 16 B layout of MlaV4KernelArgsLegacy;
// slots marked "repurposed" carry a different value than the non-persistent
// kernel, and several upper halves of the pointer slots carry an extra field.
// ----------------------------------------------------------------------------
struct __attribute__((packed)) MlaV4PsKernelArgs
{
    void* ptr_O_acc; // 0x00: workspace o_acc  [2P, 128, 512] FP32 split partials
                     // (rw)
    void* ptr_out;   // 0x08: out [N, 128, 512] BF16 (upper half of slot 0)
    void* ptr_L_acc; // 0x10: workspace lse_acc [2P, 128] FP32 split partials (rw)
    void* ptr_lse;   // 0x18: final LSE [N, 128] FP32, nullptr = not written
    void* ptr_Q;
    p2 _p_q; // 0x20: Q packed FP8 + e8m0 scale
    void* ptr_KV;
    p2 _p_kv; // 0x30: KV packed FP8
    void* ptr_LTP;
    p2 _p_ltp; // 0x40: kv_indptr
    void* ptr_LTD;
    p2 _p_ltd;      // 0x50: kv_page_indices
    void* ptr_desc; // 0x60: repurposed: workspace desc [P, 8] int32 (rw)
    void* ptr_dbg;  // 0x68: unused, nullptr
    float scalar_f;
    p3 _p_sc; // 0x70: 1.0f/sqrtf(kV4DimNope+kV4DimRope)
    unsigned int s_gqa_ratio;
    p3 _p_gr;                // 0x80: 128 (heads)
    unsigned int s_kv_split; // 0x90: 1
    unsigned int _p_ks;      // 0x94: pad
    void* ptr_cnt;           // 0x98: workspace cnt int32 counters (rw, zero at rest)
    unsigned int s_num_p;    // 0xa0: repurposed: number of partitions P
    unsigned int s_num_q;    // 0xa4: number of query rows N
    p2 _p_nq;                // 0xa8: pad
    unsigned int s_plan;     // 0xb0: repurposed: planner config F | (MT << 8)
    unsigned int _p_pl;      // 0xb4: pad
    void* ptr_unused;        // 0xb8: not read by this kernel, nullptr
    unsigned int s_zero;
    p3 _p_z; // 0xc0: 0
    void* ptr_QTP;
    p2 _p_qtp; // 0xd0: repurposed: workspace arange int32 (read-only)
    void* ptr_STP;
    p2 _p_stp; // 0xe0: repurposed: the same arange
    unsigned int out_16_nosplit;
    p3 _p_o16; // 0xf0: 1
    void* ptr_QROPE;
    p2 _p_qrope; // 0x100: Q rope BF16
    void* ptr_KVROPE;
    p2 _p_kvrope; // 0x110: KV rope BF16
    void* ptr_sink;
    p2 _p_sink; // 0x120: [128] FP32 attention sink logit
    void* ptr_zero;
    p2 _p_pz; // 0x130: nullptr
    unsigned int s_zero2;
    p3 _p_z2; // 0x140: 0
};
static_assert(sizeof(MlaV4PsKernelArgs) == 21 * 16, "persistent v4 nm kernarg is 0x150 bytes");
static_assert(offsetof(MlaV4PsKernelArgs, ptr_cnt) == 0x98 &&
                  offsetof(MlaV4PsKernelArgs, s_num_q) == 0xa4 &&
                  offsetof(MlaV4PsKernelArgs, ptr_unused) == 0xb8 &&
                  offsetof(MlaV4PsKernelArgs, ptr_sink) == 0x120,
              "persistent v4 nm kernarg offsets");

// Workspace sizes the kernel indexes into (aiter/mla.py
// get_mla_v4_nm_ps_workspace).
// cnt holds [0, 2*65536) row counters, [2*65536, +16*512) reserved (unused)
// and [.., +4*1024) group counters.
static constexpr int kV4PsHeads            = 128;
static constexpr int kV4PsDim              = kV4DimNope + kV4DimRope;
static constexpr int kV4PsMaxParts         = 1024;
static constexpr int kV4PsMaxRows          = 32768; // byte offsets of out / q wrap above this
static constexpr int64_t kV4PsArange       = 65537;
static constexpr int64_t kV4PsCntInts      = 2 * 65536 + 16 * 512 + 4 * 1024;
static constexpr unsigned int kV4PsPlanCfg = 6u | (1u << 8);

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mla_decode_v4_ps_asm,
    (aiter_tensor_t * Q,              // [N, 128, 512] FP8 packed Q + e8m0
     aiter_tensor_t* qrope,           // [N, 128, 64] BF16
     aiter_tensor_t* KV,              // [rows, ..., 512] FP8 packed KV, row-dense
     aiter_tensor_t* kvrope,          // [rows, ..., 64] BF16, row-dense
     aiter_tensor_t* kv_indptr,       // [>= N+1] int32
     aiter_tensor_t* kv_page_indices, // [*] int32
     aiter_tensor_t* sink,            // [128] FP32
     aiter_tensor_t* o_acc,           // workspace [2P, 128, 512] FP32
     aiter_tensor_t* lse_acc,         // workspace [2P, 128] FP32
     aiter_tensor_t* desc,            // workspace [P, 8] int32
     aiter_tensor_t* cnt,             // workspace [kV4PsCntInts] int32, zero at rest
     aiter_tensor_t* arange,          // workspace arange(kV4PsArange) int32
     aiter_tensor_t* output,          // [N, 128, 512] BF16
     aiter_tensor_t* lse,             // [N, 128] FP32, nullable
     hipStream_t stream),
    (Q,
     qrope,
     KV,
     kvrope,
     kv_indptr,
     kv_page_indices,
     sink,
     o_acc,
     lse_acc,
     desc,
     cnt,
     arange,
     output,
     lse,
     stream))
{
    auto check_buf = [&](aiter_tensor_t* t, AiterDtype dt, const char* name) {
        AITER_CHECK(t != nullptr && t->data_ptr() != nullptr,
                    "mla_decode_v4_ps_asm",
                    ": `",
                    name,
                    "` is NULL");
        AITER_CHECK(t->dtype() == dt,
                    "mla_decode_v4_ps_asm",
                    ": `",
                    name,
                    "` has dtype ",
                    AiterDtype_to_str(t->dtype()),
                    ", expected ",
                    AiterDtype_to_str(dt));
        AITER_CHECK(
            t->is_contiguous(), "mla_decode_v4_ps_asm", ": `", name, "` must be contiguous");
        AITER_CHECK(t->device_id == Q->device_id,
                    "mla_decode_v4_ps_asm",
                    ": `",
                    name,
                    "` is on another device");
    };
    AITER_CHECK(Q != nullptr, __func__, ": `Q` is NULL");
    check_buf(Q, AITER_DTYPE_fp8, "Q");
    check_buf(qrope, AITER_DTYPE_bf16, "qrope");
    check_buf(KV, AITER_DTYPE_fp8, "KV");
    check_buf(kvrope, AITER_DTYPE_bf16, "kvrope");
    check_buf(kv_indptr, AITER_DTYPE_i32, "kv_indptr");
    check_buf(kv_page_indices, AITER_DTYPE_i32, "kv_page_indices");
    check_buf(sink, AITER_DTYPE_fp32, "sink");
    check_buf(o_acc, AITER_DTYPE_fp32, "o_acc");
    check_buf(lse_acc, AITER_DTYPE_fp32, "lse_acc");
    check_buf(desc, AITER_DTYPE_i32, "desc");
    check_buf(cnt, AITER_DTYPE_i32, "cnt");
    check_buf(arange, AITER_DTYPE_i32, "arange");
    check_buf(output, AITER_DTYPE_bf16, "output");
    if(lse != nullptr && lse->data_ptr() != nullptr)
        check_buf(lse, AITER_DTYPE_fp32, "lse");

    AITER_CHECK(Q->dim() == 3 && Q->size(1) == kV4PsHeads && Q->size(2) == kV4PsDim,
                __func__,
                ": Q must be [N, 128, 512]");
    const int64_t num_q = Q->size(0);
    AITER_CHECK(num_q <= kV4PsMaxRows, __func__, ": N=", num_q, " exceeds ", kV4PsMaxRows);
    AITER_CHECK(qrope->numel() == static_cast<size_t>(num_q * kV4PsHeads * kV4DimRope),
                __func__,
                ": qrope must be [N, 128, 64]");
    AITER_CHECK(output->dim() == 3 && output->size(0) == num_q && output->size(1) == kV4PsHeads &&
                    output->size(2) == kV4PsDim,
                __func__,
                ": output must be [N, 128, 512]");
    if(lse != nullptr && lse->data_ptr() != nullptr)
        AITER_CHECK(lse->numel() == static_cast<size_t>(num_q * kV4PsHeads),
                    __func__,
                    ": lse must be [N, 128]");
    AITER_CHECK(KV->size(-1) == kV4PsDim && kvrope->size(-1) == kV4DimRope &&
                    KV->numel() / kV4PsDim == kvrope->numel() / kV4DimRope,
                __func__,
                ": KV / kvrope must be row-dense [rows, 512] / [rows, 64] with "
                "equal rows");
    AITER_CHECK(static_cast<int64_t>(kv_indptr->numel()) >= num_q + 1,
                __func__,
                ": kv_indptr needs at least N+1 entries");
    AITER_CHECK(sink->numel() == kV4PsHeads, __func__, ": sink must have 128 entries");

    AITER_CHECK(desc->dim() == 2 && desc->size(1) == 8, __func__, ": desc must be [P, 8]");
    const int64_t num_p = desc->size(0);
    AITER_CHECK(num_p >= 1 && num_p <= kV4PsMaxParts,
                __func__,
                ": num_partitions=",
                num_p,
                " must be in [1, ",
                kV4PsMaxParts,
                "]");
    AITER_CHECK(o_acc->numel() == static_cast<size_t>(2 * num_p * kV4PsHeads * kV4PsDim),
                __func__,
                ": o_acc must be [2P, 128, 512]");
    AITER_CHECK(lse_acc->numel() == static_cast<size_t>(2 * num_p * kV4PsHeads),
                __func__,
                ": lse_acc must be [2P, 128]");
    AITER_CHECK(cnt->numel() >= static_cast<size_t>(kV4PsCntInts),
                __func__,
                ": cnt needs ",
                kV4PsCntInts,
                " entries");
    AITER_CHECK(arange->numel() >= static_cast<size_t>(kV4PsArange),
                __func__,
                ": arange needs ",
                kV4PsArange,
                " entries");
    if(num_q == 0)
        return;

    const HipDeviceGuard device_guard(Q->device_id);

    MlaV4PsKernelArgs args = {};
    args.ptr_O_acc         = o_acc->data_ptr();
    args.ptr_out           = output->data_ptr();
    args.ptr_L_acc         = lse_acc->data_ptr();
    args.ptr_lse           = (lse != nullptr) ? lse->data_ptr() : nullptr;
    args.ptr_Q             = Q->data_ptr();
    args.ptr_KV            = KV->data_ptr();
    args.ptr_LTP           = kv_indptr->data_ptr();
    args.ptr_LTD           = kv_page_indices->data_ptr();
    args.ptr_desc          = desc->data_ptr();
    args.ptr_dbg           = nullptr;
    args.scalar_f          = 1.0f / std::sqrt(static_cast<float>(kV4PsDim));
    args.s_gqa_ratio       = kV4PsHeads;
    args.s_kv_split        = 1;
    args.ptr_cnt           = cnt->data_ptr();
    args.s_num_p           = static_cast<unsigned int>(num_p);
    args.s_num_q           = static_cast<unsigned int>(num_q);
    args.s_plan            = kV4PsPlanCfg;
    args.ptr_unused        = nullptr;
    args.ptr_QTP           = arange->data_ptr();
    args.ptr_STP           = arange->data_ptr();
    args.out_16_nosplit    = 1;
    args.ptr_QROPE         = qrope->data_ptr();
    args.ptr_KVROPE        = kvrope->data_ptr();
    args.ptr_sink          = sink->data_ptr();

    CFG* config_map = &cfg_mla_v4_asm;
    // Separate cache from mla_decode_v4_asm's; keys are the distinct symbol
    // names.
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const std::string arch_id    = get_gpu_arch();
    const std::string kernelName = get_heuristic_kernel_mla_v4(
        "fp8", "fp8", /*gqa=*/64, /*ps=*/1, 0, 0, /*qseqlen=*/1, /*lse=*/1, arch_id, config_map);
    auto it = config_map->find(kernelName);
    AITER_CHECK(it != config_map->end(), __func__, " not find kernel ", kernelName);
    const char* name    = it->second.knl_name.c_str();
    const char* co_name = it->second.co_name.c_str();
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });

    size_t arg_size = sizeof(args);
    impl_ptr->launch_kernel(
        {&args, &arg_size, static_cast<int>(2 * num_p), 1, 1, 256, 1, 1, stream});
}

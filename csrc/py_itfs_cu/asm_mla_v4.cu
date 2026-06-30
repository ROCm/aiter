// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// v4 MLA decode dispatcher (mi350 nm-recompile family).
//
// This is a peer of csrc/py_itfs_cu/asm_mla.cu but targets the v4 18-slot
// kernarg ABI which is *binary incompatible* with v3's 14-slot layout:
//
//   slot 8  = raw gqa_ratio          (not s_MQA = gqa_ratio*max_seqlen_q)
//   slot 9  = num_kv_splits          (== poc_kl `passes`)
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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// Per-.so TLS error storage + aiter_get_last_error / aiter_clear_last_error
// exports. Required so that AITER_CHECK failures in our dispatcher surface as
// RuntimeError in Python instead of aborting the worker process.
AITER_CTYPES_ERROR_DEF

// ----------------------------------------------------------------------------
// 19-slot kernarg buffer (304 bytes).verified by
// op_tests/test_mla_v4_nm.py::test_v4_nm_kernarg_scalar_slots.
//
// `ptr_sink` (slot 18, byte offset 0x120) is the attention-sink logit
// pointer. Pass `torch.full((num_heads,), -inf)` for "no sink"
// math (exp(-inf - max) = 0 → no contribution); the wrapper does NOT
// substitute a -1e9 sentinel for you — pure -inf works because the kernel
// writes its running max in fp32 with no rescaling at the sink merge site.
// ----------------------------------------------------------------------------
struct __attribute__((packed)) MlaV4KernelArgs
{
    void *ptr_R;          p2 _p_r;     // 0:  splitData (logits) [FP32]
    void *ptr_LSE;        p2 _p_lse;   // 1:  splitLse (attn_lse) [FP32]
    void *ptr_Q;          p2 _p_q;     // 2:  Q packed FP8 + e8m0 scale
    void *ptr_KV;         p2 _p_kv;    // 3:  KV packed FP8
    void *ptr_LTP;        p2 _p_ltp;   // 4:  kv_indptr
    void *ptr_LTD;        p2 _p_ltd;   // 5:  kv_page_indices
    void *ptr_LTL;        p2 _p_ltl;   // 6:  kv_last_page_lens
    float        scalar_f;        p3 _p_sc;     // 7:  1.0f/sqrtf(kV4DimNope+kV4DimRope)
    unsigned int s_gqa_ratio;     p3 _p_gr;     // 8:  raw gqa_ratio
    unsigned int s_kv_split;      p3 _p_ps;     // 9:  num_kv_splits == passes
    unsigned int s_total_kv;      p3 _p_tk;     // 10: kv_seq_lens * num_seqs
    unsigned int s_stride_page;   p3 _p_sp;     // 11: page_size * dim_qk_packed (bytes)
    unsigned int s_log2_page;     p3 _p_lp;     // 12: log2(page_size)
    void *ptr_QTP;        p2 _p_qtp;   // 13: qo_indptr
    void *ptr_STP;        p2 _p_stp;   // 14: split_indptr
    unsigned int out_16_nosplit;  p3 _p_o16;    // 15: 0 = fp32 split, 1 = bf16 nosplit
    void *ptr_QROPE;      p2 _p_qrope; // 16
    void *ptr_KVROPE;     p2 _p_kvrope;// 17
    void *ptr_sink;       p2 _p_sink;  // 18: [num_heads] FP32 attention sink logit
                                       //     loaded by 3_13.s @ kernarg+0x120
};
static_assert(sizeof(MlaV4KernelArgs) == 304,
              "MLA v4 kernarg pack must be 19 * 16 bytes (sink slot at 0x120)");
static_assert(offsetof(MlaV4KernelArgs, ptr_sink) == 0x120,
              "ptr_sink must land at kernarg byte offset 0x120 "
              "(matches 3_13.s `s_load_dwordx2 s[..], s[0:1], 0x120`)");

// ----------------------------------------------------------------------------
// kV4DimNope + kV4DimRope = 448 + 64 = 512. The kernel hardcodes
// 1/sqrt(512) as its softmax pre-scale. Keep the constant here so the
// dispatcher and the regression test agree without #include'ing poc_kl.
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
// at runtime via slot 9 of the kernarg packet (mirrors poc_kl `params.passes`).
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
                " q_type:", q_type,
                " kv_type:", kv_type,
                " gqa:", gqa,
                " ps:", ps,
                " qSeqLen:", qseqlen,
                " prefill:", prefill,
                " causal:", causal,
                " lse:", lse,
                " arch:", arch_id);
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
    (aiter_tensor_t* Q,                  // [total_query_len, num_heads, head_size]   FP8 packed Q+e8m0
     aiter_tensor_t* qrope,              // [total_query_len, num_heads, kv_rotary]   BF16
     aiter_tensor_t* KV,                 // [num_page, page_size, num_kv_heads, head_size] FP8
     aiter_tensor_t* kvrope,             // [num_page, page_size, num_kv_heads, kv_rotary] BF16
     aiter_tensor_t* qo_indptr,          // [num_seqs+1]
     aiter_tensor_t* kv_indptr,          // [num_seqs+1]
     aiter_tensor_t* kv_page_indices,    // [num_page_used]
     aiter_tensor_t* kv_last_page_lens,  // [num_seqs]
     aiter_tensor_t* split_indptr,       // [num_seqs+1]
     aiter_tensor_t* sink,               // [num_heads] FP32 — see "ptr_sink" note above
     int max_seqlen_q,
     float softmax_scale,                // ignored; v4 hardcodes 1/sqrt(512). Kept for API parity.
     int out_16_nosplit,
     int num_kv_splits,                  //
     // outputs
     aiter_tensor_t* splitData,          // [num_seqs, num_kv_splits, num_kv_heads, gqa*max_seqlen_q, v_head_dim] FP32
     aiter_tensor_t* splitLse,           // [num_seqs, num_kv_splits, num_kv_heads, gqa*max_seqlen_q, 1]          FP32
     aiter_tensor_t* output,             // [total_query_len, num_heads, v_head_dim] BF16 (used when out_16_nosplit==1)
     aiter_tensor_t* valid_split_count,  // [num_seqs] i32 scratch (nullable). Mirrors V3
                                         // mla_decode_stage1_asm_fwd's valid-split-count slot;
     int use_valid_split_count_reduce,   // ABI parity with V3 stage1's trailing scalar.
     hipStream_t stream),
    (Q, qrope, KV, kvrope, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
     split_indptr, sink, max_seqlen_q, softmax_scale, out_16_nosplit, num_kv_splits,
     splitData, splitLse, output, valid_split_count, use_valid_split_count_reduce, stream))
{
    (void)softmax_scale;
    // valid_split_count / use_valid_split_count_reduce: ABI parity with V3
    // stage1 (nullable / passive). The shipped v4 nm .co does not consume them;
    // accept and ignore so callers can plumb a fixed buffer through for
    // CUDA-graph capture without a separate codepath.
    (void)valid_split_count;
    (void)use_valid_split_count_reduce;
    AITER_CHECK(sink != nullptr, __func__, ": `sink` must not be NULL");
    AITER_CHECK(sink->data_ptr() != nullptr,
                __func__, ": `sink` data_ptr is NULL — caller must allocate "
                          "even when no sink is desired (use torch.full(-inf))");
    AITER_CHECK(Q->is_contiguous(),    __func__, ": only support Q.is_contiguous() for now");
    AITER_CHECK(KV->is_contiguous(),   __func__, ": only support KV.is_contiguous() for now");
    AITER_CHECK(qrope->is_contiguous(),  __func__, ": only support qrope.is_contiguous()");
    AITER_CHECK(kvrope->is_contiguous(), __func__, ": only support kvrope.is_contiguous()");

    const int num_seqs      = qo_indptr->size(0) - 1;
    const int num_heads     = Q->size(1);
    const int num_kv_heads  = KV->size(2);
    const int gqa_ratio     = num_heads / num_kv_heads;
    const int page_size     = KV->size(1);
    const int dim_qk_packed = KV->size(3);  // per-token kernel stride in BYTES (FP8 = 1 byte/elem)

    AITER_CHECK(num_kv_heads == 1, __func__, ": only support num_kv_heads==1 for now");
    AITER_CHECK(Q->size(2) == dim_qk_packed,
                __func__, ": Q head_size must equal KV head_size (= dim_qk_packed)");

    const HipDeviceGuard device_guard(Q->device_id);

    // Kernel-hardcoded constants (q_dtype-independent on v4 nm).
    constexpr int qk_elem_dim = kV4DimNope + kV4DimRope;  // 448 + 64 = 512 elems
    const float   scalar_f    = 1.0f / std::sqrt(static_cast<float>(qk_elem_dim));
    const unsigned int log2_page   = static_cast<unsigned int>(__builtin_ctz(page_size));

    // ---- Dead kernarg slots ---------------------------------------------------
    // Disassembling slot 10 (s_total_kv) at offset 0xA0 and slot 11 (s_stride_page) at
    // offset 0xB0 are NEVER read. They were carried over from earlier kernel
    // variants for ABI parity; computing s_total_kv used to require a per-call
    // 4-byte D2H readback of `kv_indptr[-1]` plus `hipStreamSynchronize`,
    // which created a host-side stall that cost ~5-7us on every launch.
    //
    MlaV4KernelArgs args = {};
    size_t arg_size = sizeof(args);
    args.ptr_R          = splitData->data_ptr();
    args.ptr_LSE        = splitLse->data_ptr();
    args.ptr_Q          = Q->data_ptr();
    args.ptr_KV         = KV->data_ptr();
    args.ptr_LTP        = kv_indptr->data_ptr();
    args.ptr_LTD        = kv_page_indices->data_ptr();
    args.ptr_LTL       = kv_last_page_lens->data_ptr();
    args.scalar_f       = scalar_f;
    args.s_gqa_ratio    = static_cast<unsigned int>(gqa_ratio);
    args.s_kv_split     = static_cast<unsigned int>(num_kv_splits);
    // args.s_total_kv     left as 0 — kernel never reads slot 10 (offset 0xA0)
    // args.s_stride_page  left as 0 — kernel never reads slot 11 (offset 0xB0)
    args.s_log2_page    = log2_page;
    args.ptr_QTP        = qo_indptr->data_ptr();
    args.ptr_STP        = split_indptr->data_ptr();
    args.out_16_nosplit = static_cast<unsigned int>(out_16_nosplit);
    args.ptr_QROPE      = qrope->data_ptr();
    args.ptr_KVROPE     = kvrope->data_ptr();
    args.ptr_sink       = sink->data_ptr();

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
    int sub_Q               = 64;            // default (matches V3 default)
    int config_max_seqlen_q = max_seqlen_q;
    int ps                  = 0;              // v4 nm always non-persistent today
    int prefill             = 0;              // decode stage
    int causal              = 0;
    int lse_flag            = 0;

    if(gqa_ratio == 16 && q_type == "fp8" && kv_type == "fp8")
    {
        if(max_seqlen_q == 4)
        {
            sub_Q               = 64;
            config_max_seqlen_q = 4;
        }
        else if(max_seqlen_q == 1)
        {
            sub_Q               = 16;
            config_max_seqlen_q = 1;
        }
        else if(max_seqlen_q == 2)
        {
            sub_Q               = 32;
            config_max_seqlen_q = 2;
        }
        else
        {
            config_max_seqlen_q = 4;
        }
    }
    else if (gqa_ratio == 64 && q_type == "fp8" && kv_type == "fp8")
    {
        if(max_seqlen_q == 1)
        {
            sub_Q               = 64;
            config_max_seqlen_q = 1;
        }
        else if(max_seqlen_q == 2)
        {
            sub_Q               = 128;
            config_max_seqlen_q = 2;
        }
        else
        {
            config_max_seqlen_q = 1;
        }
    }
    else if (gqa_ratio == 128 && q_type == "fp8" && kv_type == "fp8")
    {
        if(max_seqlen_q == 1)
        {
            sub_Q               = 64;
            config_max_seqlen_q = 1;
        }
    }

    // ---- CSV lookup-key normalization ---------------------------------------
    // v4 nm ships ONE kernel binary (the 32n-tile .co, symbol
    // mla_a8w8_qh64_qseqlen1_gqaratio64_nm). Its 64 q-row tile satisfies the
    // invariant `gqa * q_seq_logical = 64`, so it serves all three shipped
    // entry points: (gqa=16, qSeqLen=4), (gqa=64, qSeqLen=1) and (gqa=128,
    // qSeqLen=1). The CSV carries a single (Gqa=64, qSeqLen=1) row; normalize
    // every supported (gqa, qSeqLen) caller to that lookup key here. `sub_Q`
    // and the per-launch grid geometry stay set to the gqa-correct values from
    // the heuristic above — this remap only picks which CSV row (== which .co)
    // to load.
    int csv_gqa     = gqa_ratio;
    int csv_qseqlen = config_max_seqlen_q;
    if(q_type == "fp8" && kv_type == "fp8" &&
       ((gqa_ratio == 16 && ( config_max_seqlen_q == 4 || config_max_seqlen_q == 1)) ||
        ((gqa_ratio == 64 || gqa_ratio == 128) && config_max_seqlen_q == 1)))
    {
        csv_gqa     = 64;
        csv_qseqlen = 1;
    }

    // Kernel lookup (cached across calls).
    std::string arch_id  = get_gpu_arch();
    CFG* config_map      = &cfg_mla_v4_asm;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    std::string kernelName = get_heuristic_kernel_mla_v4(
        q_type, kv_type, csv_gqa, ps, prefill, causal, csv_qseqlen,
        lse_flag, arch_id, config_map);
    AITER_CHECK(!kernelName.empty(), __func__, ": cannot find suitable kernel");

    AiterAsmKernel* impl_ptr = nullptr;
    auto it = config_map->find(kernelName);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        impl_ptr = &impl_ptr_map.get_or_create(
            name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
    {
        AITER_CHECK(false, __func__, " not find kernel ", kernelName);
    }
    AITER_CHECK(impl_ptr != nullptr, __func__,
                ": unsupport current data type or shape. please refer to asm_mla_v4.cu");

    // Launch geometry: gdx = ceil(q_seq_lens_internal / sub_Q),
    // gdy = num_seqs, gdz = num_kv_splits. Block dim = 256 (wave64 * 4).
    const int q_seq_lens_internal = gqa_ratio * max_seqlen_q;
    const int gdx = (q_seq_lens_internal + sub_Q - 1) / sub_Q;
    const int gdy = num_seqs;
    const int gdz = num_kv_splits;

    // ----- DEBUG: env-gated 304B kernarg dump for cross-check vs poc_kl. -----
    // Used by op_tests/test_mla_v4_nm.py::test_v4_nm_kernarg_scalar_slots to
    // lock in the 19-slot layout. Has no runtime cost when env unset.
    if(const char* dbg = std::getenv("AITER_V4_NM_DUMP_KERNARG"))
    {
        if(dbg[0] == '1')
        {
            fprintf(stderr, "[aiter kernarg 304B]\n");
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&args);
            for(size_t i = 0; i < sizeof(args); ++i)
            {
                fprintf(stderr, "%02x%s", bytes[i], ((i + 1) % 16 == 0) ? "\n" : " ");
            }
            fprintf(stderr, "[aiter grid (%d,%d,%d) block (256,1,1)]\n", gdx, gdy, gdz);
            fflush(stderr);
        }
    }

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,
                             gdy,
                             gdz,
                             256,
                             1,
                             1,
                             stream});
}

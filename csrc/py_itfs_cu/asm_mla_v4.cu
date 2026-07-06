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
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>

// Per-.so TLS error storage + aiter_get_last_error / aiter_clear_last_error
// exports. Required so that AITER_CHECK failures in our dispatcher surface as
// RuntimeError in Python instead of aborting the worker process.
AITER_CTYPES_ERROR_DEF

// ----------------------------------------------------------------------------
// Debug buffer dump
// ----------------------------------------------------------------------------
static void mla_v4_dump_debug_buffer(const std::string& dump_dir,
                                     const char* name,
                                     const aiter_tensor_t* t)
{
    if(dump_dir.empty() || t == nullptr || t->data_ptr() == nullptr)
    {
        return;
    }
    mkdir(dump_dir.c_str(), 0777);

    const size_t nbytes = t->numel() * t->element_size();
    std::vector<char> host(nbytes);
    if(nbytes > 0)
    {
        hipError_t err =
            hipMemcpy(host.data(), t->data_ptr(), nbytes, hipMemcpyDeviceToHost);
        if(err != hipSuccess)
        {
            std::printf("[aiter][v4 nm][debug] dump %s: hipMemcpy D2H failed: %s\n",
                        name, hipGetErrorString(err));
            return;
        }
    }

    const std::string bin_path = dump_dir + "/" + name + ".bin";
    if(FILE* bin = std::fopen(bin_path.c_str(), "wb"))
    {
        std::fwrite(host.data(), 1, nbytes, bin);
        std::fclose(bin);
    }
    else
    {
        std::printf("[aiter][v4 nm][debug] failed to open dump file %s\n",
                    bin_path.c_str());
    }

    std::string shape  = "(";
    std::string stride = "(";
    for(int i = 0; i < t->ndim; ++i)
    {
        shape += std::to_string(t->size(i));
        stride += std::to_string(t->stride(i));
        if(i + 1 < t->ndim)
        {
            shape += ",";
            stride += ",";
        }
    }
    shape += ")";
    stride += ")";

    const std::string meta_path = dump_dir + "/" + name + ".meta.txt";
    if(FILE* meta = std::fopen(meta_path.c_str(), "w"))
    {
        std::fprintf(meta, "name=%s\n", name);
        std::fprintf(meta, "dtype=%s\n", AiterDtype_to_str(t->dtype()).c_str());
        std::fprintf(meta, "shape=%s\n", shape.c_str());
        std::fprintf(meta, "stride=%s\n", stride.c_str());
        std::fprintf(meta, "element_size=%zu\n", t->element_size());
        std::fprintf(meta, "numel=%zu\n", t->numel());
        std::fprintf(meta, "nbytes=%zu\n", nbytes);
        std::fprintf(meta, "layout=contiguous raw tensor bytes (device->host copy)\n");
        std::fclose(meta);
    }
}

// ----------------------------------------------------------------------------
// 19-slot kernarg buffer (304 bytes).
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
    // v4 nm 64 q-row tile serves every supported entry point, so normalize each to the single (Gqa=64,
    // qSeqLen=1) CSV row. 
    int csv_gqa     = gqa_ratio;
    int csv_qseqlen = config_max_seqlen_q;
    if(supported)
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

    // ----- DEBUG: env-gated 304B kernarg dump for cross-check. -----
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

    // ----- DEBUG: env-gated verbose launch trace (mirrors asm_mla.cu's
    // ASM_DEBUG block, but runtime-gated by AITER_MLA_DEBUG_VERBOSE so the
    // shipped .so can dump on demand). Prints kernel selection, the resolved
    // dispatch inputs, tensor shapes/strides, the kernarg scalars + buffer
    // ptrs, and the launch grid. Unset == zero cost.
    const char* verbose_env = std::getenv("AITER_MLA_DEBUG_VERBOSE");
    const bool verbose =
        verbose_env != nullptr && verbose_env[0] != '\0' &&
        !(verbose_env[0] == '0' && verbose_env[1] == '\0');
    if(verbose)
    {
        std::printf("[aiter][v4 nm][debug] kernelName=%s\n", kernelName.c_str());
        if(it != config_map->end())
        {
            const auto& cfg = it->second;
            std::printf("[aiter][v4 nm][debug] knl_name=%s co_name=%s\n",
                        cfg.knl_name.c_str(), cfg.co_name.c_str());
        }
        std::printf("[aiter][v4 nm][debug] inputs: arch=%s num_seqs=%d gqa_ratio=%d "
                    "max_seqlen_q=%d config_max_seqlen_q=%d sub_Q=%d num_kv_splits=%d "
                    "csv_gqa=%d csv_qseqlen=%d q_type=%s kv_type=%s scalar=%g\n",
                    arch_id.c_str(), num_seqs, gqa_ratio, max_seqlen_q,
                    config_max_seqlen_q, sub_Q, num_kv_splits, csv_gqa, csv_qseqlen,
                    q_type.c_str(), kv_type.c_str(), scalar_f);
        std::printf("[aiter][v4 nm][debug] tensor shapes: Q=(%ld,%ld,%ld) "
                    "KV=(%ld,%ld,%ld,%ld) splitData=(%ld,%ld,%ld,%ld) "
                    "splitLse=(%ld,%ld,%ld,%ld) output=(%ld,%ld,%ld)\n",
                    Q->size(0), Q->size(1), Q->size(2),
                    KV->size(0), KV->size(1), KV->size(2), KV->size(3),
                    splitData->size(0), splitData->size(1), splitData->size(2),
                    splitData->size(3),
                    splitLse->size(0), splitLse->size(1), splitLse->size(2),
                    splitLse->size(3),
                    output->size(0), output->size(1), output->size(2));
        std::printf("[aiter][v4 nm][debug] tensor strides: Q=(%ld,%ld,%ld) "
                    "KV=(%ld,%ld,%ld,%ld) splitData=(%ld,%ld,%ld,%ld) "
                    "splitLse=(%ld,%ld,%ld,%ld) output=(%ld,%ld,%ld)\n",
                    Q->stride(0), Q->stride(1), Q->stride(2),
                    KV->stride(0), KV->stride(1), KV->stride(2), KV->stride(3),
                    splitData->stride(0), splitData->stride(1), splitData->stride(2),
                    splitData->stride(3),
                    splitLse->stride(0), splitLse->stride(1), splitLse->stride(2),
                    splitLse->stride(3),
                    output->stride(0), output->stride(1), output->stride(2));
        std::printf("[aiter][v4 nm][debug] ABI=19slot-304B arg_size=%zu\n", arg_size);
        std::printf("[aiter][v4 nm][debug] ptrs: R=%p LSE=%p Q=%p KV=%p LTP=%p LTD=%p "
                    "LTL=%p QTP=%p STP=%p QROPE=%p KVROPE=%p sink=%p output=%p "
                    "valid_split_count=%p stream=%p\n",
                    args.ptr_R, args.ptr_LSE, args.ptr_Q, args.ptr_KV, args.ptr_LTP,
                    args.ptr_LTD, args.ptr_LTL, args.ptr_QTP, args.ptr_STP,
                    args.ptr_QROPE, args.ptr_KVROPE, args.ptr_sink,
                    output->data_ptr(),
                    valid_split_count == nullptr ? nullptr
                                                 : valid_split_count->data_ptr(),
                    stream);
        std::printf("[aiter][v4 nm][debug] kernargs: scalar=%g gqa_ratio=%u "
                    "num_kv_splits=%u log2_page=%u out_16_nosplit=%u\n",
                    args.scalar_f, args.s_gqa_ratio, args.s_kv_split,
                    static_cast<unsigned int>(args.s_log2_page), args.out_16_nosplit);
        std::printf("[aiter][v4 nm][debug] launch: grid=(%d,%d,%d) block=(256,1,1)\n",
                    gdx, gdy, gdz);
        std::fflush(stdout);
    }

    // SKIP_KERNEL: env-gated guard to dump/inspect inputs without launching
    // (mirrors asm_mla.cu's AITER_MLA_DEBUG_SKIP_KERNEL). When set, splitData/
    // splitLse keep their pre-launch contents.
    const char* skip_kernel_env = std::getenv("AITER_MLA_DEBUG_SKIP_KERNEL");
    const bool skip_kernel =
        skip_kernel_env != nullptr && skip_kernel_env[0] != '\0' &&
        !(skip_kernel_env[0] == '0' && skip_kernel_env[1] == '\0');
    if(verbose)
    {
        std::printf(skip_kernel
                        ? "[aiter][v4 nm][debug] skipping kernel launch (SKIP_KERNEL=%s)\n"
                        : "[aiter][v4 nm][debug] launching kernel.%s\n",
                    skip_kernel ? skip_kernel_env : "");
        std::fflush(stdout);
    }

    if(!skip_kernel)
    {
        impl_ptr->launch_kernel({&args,
                                 &arg_size,
                                 gdx,
                                 gdy,
                                 gdz,
                                 256,
                                 1,
                                 1,
                                 stream});
        if(verbose)
        {
            hipError_t launch_status = hipGetLastError();
            std::printf("[aiter][v4 nm][debug] after launch enqueue: "
                        "hipGetLastError=%s (%d)\n",
                        hipGetErrorString(launch_status),
                        static_cast<int>(launch_status));
            hipError_t sync_status = hipStreamSynchronize(stream);
            std::printf("[aiter][v4 nm][debug] after hipStreamSynchronize: %s (%d)\n",
                        hipGetErrorString(sync_status),
                        static_cast<int>(sync_status));
            std::fflush(stdout);
        }
    }

    if(const char* dump_env = std::getenv("AITER_MLA_DEBUG_DUMP_DIR"))
    {
        if(dump_env[0] != '\0')
        {
            const std::string dump_dir(dump_env);
            (void)hipStreamSynchronize(stream);
            mla_v4_dump_debug_buffer(dump_dir, "q", Q);
            mla_v4_dump_debug_buffer(dump_dir, "qrope", qrope);
            mla_v4_dump_debug_buffer(dump_dir, "kv_buffer", KV);
            mla_v4_dump_debug_buffer(dump_dir, "kvrope", kvrope);
            mla_v4_dump_debug_buffer(dump_dir, "qo_indptr", qo_indptr);
            mla_v4_dump_debug_buffer(dump_dir, "kv_indptr", kv_indptr);
            mla_v4_dump_debug_buffer(dump_dir, "kv_page_indices", kv_page_indices);
            mla_v4_dump_debug_buffer(dump_dir, "kv_last_page_lens", kv_last_page_lens);
            mla_v4_dump_debug_buffer(dump_dir, "split_indptr", split_indptr);
            mla_v4_dump_debug_buffer(dump_dir, "sink", sink);
            if(!skip_kernel)
            {
                // Only meaningful after a real launch; with SKIP_KERNEL these
                // hold pre-launch garbage.
                mla_v4_dump_debug_buffer(dump_dir, "splitData", splitData);
                mla_v4_dump_debug_buffer(dump_dir, "splitLse", splitLse);
                mla_v4_dump_debug_buffer(dump_dir, "output", output);
            }
            std::printf("[aiter][v4 nm][debug] dumped raw buffers to %s\n",
                        dump_dir.c_str());
            std::fflush(stdout);
        }
    }
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM dispatch for the qkptph/vph (PER_TOKEN_HEAD) FP8 causal paged kernel used
// by mha_batch_prefill. Only the varlen variant (cfg_fmha_fwd_fp8_ps, mode=1,
// packed-Q via cu_seqlens_q) is shipped and launched via AiterAsmKernel; it is
// correct for all batch sizes (b=1 -> a single [0, S] segment). The 768-byte
// kernarg ABI mirrors scripts/f8_fmha_prefill (poc_kl): see the offset-annotated
// struct below.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include "aiter_hip_common.h"
#include "asm_fmha_v3_fwd_configs.hpp"
#include "torch/mha_batch_prefill_asm.h"

#include <string>

namespace aiter {
namespace torch_itfs {

namespace {

// Kernarg layout (packed) shared with the standalone harness fwd_fp8.cpp. Scalar
// fields are 4 bytes at 16-byte-aligned offsets; pointers are 8 bytes.
struct p3 { uint32_t _p0, _p1, _p2; };
struct p2 { uint32_t _p0, _p1; };

struct __attribute__((packed)) qkptph_kernargs
{
    void*        ptr_o;          // 0x00
    p2           _0;
    const void*  ptr_q;          // 0x10
    p2           _1;
    const void*  ptr_k;          // 0x20
    p2           _2;
    const void*  ptr_v;          // 0x30
    p2           _3;
    void*        ptr_lse;        // 0x40
    p2           _4;
    float        scalar;         // 0x50
    p3           _5;
    uint32_t     s_seq_len;      // 0x60
    p3           _6;
    uint32_t     s_Seqs;         // 0x70
    p3           _7;
    uint32_t     s_Ts;           // 0x80
    p3           _8;
    uint32_t     s_Hs;           // 0x90
    p3           _9;
    uint32_t     s_Bs;           // 0xA0
    p3           _10;
    uint32_t     s_gqa;          // 0xB0
    p3           _11;
    uint32_t     s_k_Seqs;       // 0xC0
    p3           _12;
    uint32_t     s_k_Hs;         // 0xD0
    p3           _13;
    uint32_t     s_k_Bs;         // 0xE0
    p3           _14;
    uint32_t     s_opt;          // 0xF0
    p3           _15;
    uint32_t     s_lse;          // 0x100
    p3           _16;
    uint32_t     s_kv_seq_len;   // 0x110
    p3           _17;
    uint32_t     s_qk_head_dim;  // 0x120
    p3           _18;
    uint32_t     s_v_head_dim;   // 0x130
    p3           _19;
    uint32_t     s_q_head_num;   // 0x140
    p3           _20;
    uint32_t     s_v_Seqs;       // 0x150
    p3           _21;
    uint32_t     s_v_Hs;         // 0x160
    p3           _22;
    uint32_t     s_v_Bs;         // 0x170
    p3           _23;
    uint32_t     s_o_Seqs;       // 0x180
    p3           _24;
    uint32_t     s_o_Hs;         // 0x190
    p3           _25;
    uint32_t     s_o_Bs;         // 0x1A0
    p3           _26;
    const void*  ptr_qseq;       // 0x1B0  -> LTD (kv_page_indices)
    p2           _27;
    const void*  ptr_kseq;       // 0x1C0  -> LTP (kv_indptr)
    p2           _28;
    uint32_t     s_lse_Hs;       // 0x1D0  -> num_total_pages (paged SRD bound)
    p3           _29;
    const void*  ptr_qseq_pad;   // 0x1E0  -> K_page_stride (bytes)
    p2           _30;
    const void*  ptr_kseq_pad;   // 0x1F0  -> V_page_stride (bytes)
    p2           _31;
    const void*  ptr_q_descale;  // 0x200
    p2           _32;
    const void*  ptr_k_descale;  // 0x210
    p2           _33;
    const void*  ptr_v_descale;  // 0x220
    p2           _34;
    uint32_t     s_desc_q_Bs;    // 0x230  q descale per-token byte stride
    p3           _35;
    uint32_t     s_desc_q_Hs;    // 0x240  q descale per-head byte stride
    p3           _36;
    uint32_t     s_desc_k_Bs;    // 0x250  k descale per-page byte stride
    p3           _37;
    uint32_t     s_desc_k_Hs;    // 0x260  k descale per-token byte stride
    p3           _38;
    uint32_t     s_desc_v_Bs;    // 0x270
    p3           _39;
    uint32_t     s_desc_v_Hs;    // 0x280  v descale per-head byte stride
    p3           _40;
    const void*  ptr_p_scale;    // 0x290
    p2           _41;
    uint32_t     s_p_scale_Bs;   // 0x2A0
    p3           _42;
    uint32_t     s_p_scale_Hs;   // 0x2B0  p_scale per-head byte stride
    p3           _43;
    uint32_t     s_sched_groups; // 0x2C0  runtime snake bin count G = #CU/nheads
    p3           _44;
    uint32_t     s_desc_k_head_stride; // 0x2D0  k descale per-kv-head byte stride
    p3           _45;
    // Unified paged+varlen tail (only consumed by the mode=1 / varlen kernel).
    const void*  ptr_cu_seqlens_q;     // 0x2E0  QTP (packed-Q base)
    p2           _46;
    const void*  ptr_seqlens_kvcache;  // 0x2F0  per-batch KV token length
    p2           _47;
}; // sizeof == 0x300 == 768

static_assert(sizeof(qkptph_kernargs) == 768, "qkptph kernarg ABI must be 768 bytes");

// Find the registered varlen config (mask=2 causal, mode=1), gfx942 fp8 hd128.
// Only the varlen (packed-Q / cu_seqlens) kernel is shipped: it is correct for
// all batch sizes (b=1 degenerates to a single segment) and within ~1% of a
// dedicated batch kernel at b=1, so the batch (mode=0) variant was dropped.
const fmha_v3_fwdConfig* find_qkptph_cfg()
{
    const std::string arch_id = get_gpu_arch();
    for(const auto& el : cfg_fmha_fwd_fp8_ps)
    {
        const auto& c = el.second;
        if(c.arch == arch_id && c.dtype == "fp8" && c.hdim_q == 128 && c.hdim_v == 128 &&
           c.mask == 2 && c.mode == 1)
        {
            return &c;
        }
    }
    return nullptr;
}

} // namespace

// Launch the asm qkptph/vph FP8 causal paged kernel. Writes into `out` and returns
// it. Raises if the inputs are not eligible for this asm path (there is no CK
// fallback in this standalone module).
at::Tensor
mha_batch_prefill_asm(const at::Tensor& q,                   // [total_q, hq, d] fp8
                      const at::Tensor& k,                   // [num_pages, hk, d/x, page, x] fp8
                      const at::Tensor& v,                   // [num_pages, hk, d, page] fp8 (col-major)
                      const at::Tensor& cu_seqlens_q,        // [b+1] int32 (QTP)
                      const at::Tensor& kv_indptr,           // [b+1] int32 (LTP)
                      const at::Tensor& kv_page_indices,     // [num_pages] int32 (LTD)
                      const at::Tensor& seqlens_kvcache,     // [b] int32 per-batch KV token len
                      at::Tensor& out,                       // [total_q, hq, dv] bf16
                      const at::Tensor& q_descale_per_token, // [total_q, hq] f32
                      const at::Tensor& k_descale_per_token, // [num_pages, page, hk] f32
                      const at::Tensor& v_descale_per_head,  // [hk] f32
                      int batch,
                      int num_heads,
                      int num_heads_k,
                      int head_size_q,
                      int head_size_v,
                      int page_block_size,
                      int num_total_pages,
                      int max_seqlen_q,
                      float softmax_scale,
                      std::optional<const at::Tensor> p_scale) // [hq] f32
{
    // Eligibility: gfx942, fp8 in / bf16 out, hd128, causal, page_size 16.
    const std::string arch_id = get_gpu_arch();
    TORCH_CHECK(arch_id == "gfx942",
                "mha_batch_prefill_asm: only gfx942 is supported, got ", arch_id);
    TORCH_CHECK(head_size_q == 128 && head_size_v == 128 && page_block_size == 16,
                "mha_batch_prefill_asm: requires head_size_q==head_size_v==128 and "
                "page_block_size==16");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                    q.scalar_type() == at::ScalarType::Float8_e4m3fnuz,
                "mha_batch_prefill_asm: q must be fp8 (e4m3)");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
                "mha_batch_prefill_asm: out must be bf16");

    const fmha_v3_fwdConfig* cfg = find_qkptph_cfg();
    TORCH_CHECK(cfg != nullptr,
                "mha_batch_prefill_asm: no registered varlen (mode=1) asm config");

    hipStream_t stream = at::hip::getCurrentHIPStream().stream();

    const int q_es = static_cast<int>(q.element_size());   // 1 (fp8)
    const int o_es = static_cast<int>(out.element_size());  // 2 (bf16)
    const int sc_es = 4;                                    // f32 descales

    qkptph_kernargs a;
    memset(&a, 0, sizeof(a));

    a.ptr_o = out.data_ptr();
    a.ptr_q = q.data_ptr();
    a.ptr_k = k.data_ptr();
    a.ptr_v = v.data_ptr();
    a.scalar = softmax_scale;

    // Packed (THD) Q/O: per-token row stride, per-head stride, batch stride 0
    // (the per-batch base comes from cu_seqlens_q in the kernel).
    a.s_seq_len    = max_seqlen_q;
    a.s_Seqs       = q.stride(0) * q_es;
    a.s_Ts         = cfg->ts_qo * a.s_Seqs;
    a.s_Hs         = q.stride(1) * q_es;
    a.s_Bs         = 0;
    a.s_gqa        = num_heads / num_heads_k;
    a.s_kv_seq_len = max_seqlen_q; // per-batch KV len comes from seqlens_kvcache
    a.s_qk_head_dim = head_size_q;
    a.s_v_head_dim  = head_size_v;
    a.s_q_head_num  = num_heads;
    a.s_o_Seqs      = out.stride(0) * o_es;
    a.s_o_Hs        = out.stride(1) * o_es;
    a.s_o_Bs        = 0;

    // Page table (SGLang 1D): ptr_qseq=LTD (physical page ids), ptr_kseq=LTP
    // (per-batch page-range prefix). 0x1D0 carries the pool page count.
    a.ptr_qseq  = kv_page_indices.data_ptr();
    a.ptr_kseq  = kv_indptr.data_ptr();
    a.s_lse_Hs  = num_total_pages;

    // Paged K (5D vectorized [num_pages, hk, d/x, page, x]): head-within-block
    // stride = stride(1); per-token-within-page = stride(3) (== vector width x);
    // K_page_stride (bytes/physical block) = stride(0).
    a.s_k_Hs       = k.stride(1) * q_es;
    a.s_k_Seqs     = k.stride(3) * q_es;
    a.s_k_Bs       = 0;
    a.ptr_qseq_pad = reinterpret_cast<const void*>(static_cast<uintptr_t>(k.stride(0) * q_es));

    // Paged V (4D col-major [num_pages, hk, d, page]): head-within-block stride
    // = stride(1); token-minor (stride(3)==1); V_page_stride = stride(0).
    a.s_v_Hs       = v.stride(1) * q_es;
    a.s_v_Seqs     = v.stride(3) * q_es; // == 1 (token-minor)
    a.s_v_Bs       = 0;
    a.ptr_kseq_pad = reinterpret_cast<const void*>(static_cast<uintptr_t>(v.stride(0) * q_es));

    // PER_TOKEN_HEAD descales.
    a.ptr_q_descale = q_descale_per_token.data_ptr();
    a.s_desc_q_Bs   = q_descale_per_token.stride(0) * sc_es; // per-token
    a.s_desc_q_Hs   = q_descale_per_token.stride(1) * sc_es; // per-head

    a.ptr_k_descale       = k_descale_per_token.data_ptr();
    a.s_desc_k_Bs         = k_descale_per_token.stride(0) * sc_es; // per-page
    a.s_desc_k_Hs         = k_descale_per_token.stride(1) * sc_es; // per-token in page
    a.s_desc_k_head_stride = k_descale_per_token.stride(2) * sc_es; // per-kv-head

    a.ptr_v_descale = v_descale_per_head.data_ptr();
    a.s_desc_v_Bs   = 0;
    a.s_desc_v_Hs   = v_descale_per_head.stride(0) * sc_es;

    if(p_scale.has_value())
    {
        a.ptr_p_scale  = p_scale.value().data_ptr();
        a.s_p_scale_Bs = 0;
        a.s_p_scale_Hs = p_scale.value().stride(0) * sc_es;
    }

    // Persistent snake scheduler: G cost-balanced bins/head so the bin grid fills
    // one CU wave (G*nheads ~= #CU). Grid = (G, nheads, batch).
    int num_cu = 80;
    {
        hipDeviceProp_t props;
        int dev = 0;
        if(hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&props, dev) == hipSuccess)
            num_cu = props.multiProcessorCount;
    }
    const int sub_q  = cfg->ts_qo;
    const int ntiles = (max_seqlen_q + sub_q - 1) / sub_q;
    int g            = num_cu / num_heads;
    if(g < 1) g = 1;
    if(g > ntiles) g = ntiles;
    a.s_sched_groups = static_cast<uint32_t>(g);

    // Packed-Q base + per-batch KV length tables. Always set: the varlen kernel
    // derives every per-batch Q base from cu_seqlens_q (b=1 -> [0, S]).
    a.ptr_cu_seqlens_q    = cu_seqlens_q.data_ptr();
    a.ptr_seqlens_kvcache = seqlens_kvcache.data_ptr();

    size_t arg_size = sizeof(a);

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    const char* name = cfg->knl_name.c_str();
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, cfg->co_name.c_str()); });

    const int bdx = 512;
    void* args_ptr      = &a;
    size_t* arg_size_ptr = &arg_size;
    impl_ptr->launch_kernel({args_ptr, arg_size_ptr, g, num_heads, batch, bdx, 1, 1, stream});
    return out;
}

} // namespace torch_itfs
} // namespace aiter

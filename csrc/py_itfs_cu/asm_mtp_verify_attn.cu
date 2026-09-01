// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MTP-verify attention for gfx950: the speculative-decoding verify step
// (uniform q_len = 4 query tokens per sequence) over an NHD page-16 fp8 KV
// cache with head_dim 256, GQA ratio 16 or 8. Split-KV: the main kernel
// writes per-segment partial outputs / max / expsum, the reduce kernel merges
// them into the final bf16 output.
#include "aiter_tensor.h"
#include <cmath>
#include <cstdint>

namespace {

// Must match the kernarg layout of hsa/gfx950/mtp_verify_attn/src/vattn3_core.s
struct __attribute__((packed)) VattnKernelArgs
{
    void* ptr_K;          // [num_pages, 16, num_kv_heads, 256] fp8
    void* ptr_V;
    void* ptr_Q;          // [num_tokens, num_q_heads, 256] bf16
    void* ptr_BT;         // block tables [num_seqs, max_pages] int32
    void* ptr_SL;         // kv seq lens [num_seqs] int64
    void* ptr_CU;         // cu_seqlens_q [num_seqs + 1] int32
    void* ptr_SO;         // segm out    [num_tokens, num_q_heads, segs, 256] fp32
    void* ptr_SM;         // segm max    [num_tokens, num_q_heads, segs] fp32
    void* ptr_SE;         // segm expsum [num_tokens, num_q_heads, segs] fp32
    int32_t num_segments;
    int32_t bt_stride0;   // block table row stride (elements)
    int32_t q_stride0;    // Q token stride (elements)
    int32_t num_kv_heads;
    int32_t num_q_heads;
    float scale_log2;     // softmax_scale * log2(e); kernel multiplies by *k_descale
    float v_scale;        // 1.0; kernel multiplies by *v_descale
    uint32_t magic_m;     // magic division by (num_segments * 16)
    int32_t magic_sh;
    int32_t _pad;
    void* ptr_KD;         // k_descale [1] fp32 (device)
    void* ptr_VD;         // v_descale [1] fp32 (device)
};
static_assert(sizeof(VattnKernelArgs) == 128, "kernarg layout must match vattn3_core.s");

// Must match hsa/gfx950/mtp_verify_attn/src/vred.s
struct __attribute__((packed)) VredKernelArgs
{
    void* ptr_O;          // [num_tokens, num_q_heads, 256] bf16
    void* ptr_SO;
    void* ptr_SM;
    void* ptr_SE;
    uint32_t num_q_heads;
    uint32_t num_segments;
    uint32_t o_stride0;   // elements
    uint32_t o_stride1;   // elements
    uint32_t magic_m;     // magic division by num_q_heads
    uint32_t magic_sh;
};
static_assert(sizeof(VredKernelArgs) == 56, "kernarg layout must match vred.s");

// (M, sh) such that ((n * M) >> 32) >> sh == n / d for all n < 2^31.
void magic_u32(uint32_t d, uint32_t& M, uint32_t& sh)
{
    if((d & (d - 1)) == 0)
    {
        M  = 1u << 31;
        sh = 0;
        for(uint32_t x = d; x > 1; x >>= 1)
            sh++;
        sh -= 1;
        return;
    }
    uint32_t s = 0;
    for(uint32_t x = d; x > 1; x >>= 1)
        s++;
    uint64_t m = ((1ull << (32 + s)) + d - 1) / d;
    AITER_CHECK(m < (1ull << 32), "magic_u32: divisor ", d, " out of range");
    M  = static_cast<uint32_t>(m);
    sh = s;
}

const float k_log2e = 1.4426950408889634f;

} // namespace

AITER_C_ITFS void mtp_verify_attn_fwd(
    aiter_tensor_t* Q,            // [num_tokens, num_q_heads, 256] bf16, num_tokens = 4 * num_seqs
    aiter_tensor_t* K,            // [num_pages, 16, num_kv_heads, 256] fp8 e4m3 (NHD page 16)
    aiter_tensor_t* V,            // same layout as K
    aiter_tensor_t* block_tables, // [num_seqs, max_pages] int32
    aiter_tensor_t* seq_lens,     // [num_seqs] int64, kv length including the 4 query tokens
    aiter_tensor_t* cu_seqlens_q, // [num_seqs + 1] int32
    aiter_tensor_t* k_descale,    // [1] fp32, device
    aiter_tensor_t* v_descale,    // [1] fp32, device
    aiter_tensor_t* segm_out,     // [num_tokens, num_q_heads, num_segments, 256] fp32 workspace
    aiter_tensor_t* segm_max,     // [num_tokens, num_q_heads, num_segments] fp32 workspace
    aiter_tensor_t* segm_expsum,  // [num_tokens, num_q_heads, num_segments] fp32 workspace
    aiter_tensor_t* out,          // [num_tokens, num_q_heads, 256] bf16
    float softmax_scale,
    hipStream_t stream)
{
    const int num_tokens   = static_cast<int>(Q->size(0));
    const int num_q_heads  = static_cast<int>(Q->size(1));
    const int head_size    = static_cast<int>(Q->size(2));
    const int page_size    = static_cast<int>(K->size(1));
    const int num_kv_heads = static_cast<int>(K->size(2));
    const int num_seqs     = static_cast<int>(seq_lens->size(0));
    const int num_segments = static_cast<int>(segm_max->size(2));

    AITER_CHECK(Q->dtype() == AITER_DTYPE_bf16, __func__, ": Q must be bf16");
    AITER_CHECK(K->dtype() == AITER_DTYPE_fp8 && V->dtype() == AITER_DTYPE_fp8,
                __func__, ": K/V must be fp8 e4m3");
    AITER_CHECK(head_size == 256 && K->size(3) == 256, __func__, ": head_dim must be 256");
    AITER_CHECK(page_size == 16, __func__, ": page size must be 16");
    AITER_CHECK(num_tokens == 4 * num_seqs, __func__,
                ": uniform q_len of 4 required (num_tokens = 4 * num_seqs)");
    AITER_CHECK(num_q_heads % num_kv_heads == 0, __func__, ": bad GQA layout");
    const int gqa = num_q_heads / num_kv_heads;
    AITER_CHECK(gqa == 16 || gqa == 8, __func__, ": GQA ratio must be 16 or 8, got ", gqa);
    AITER_CHECK(Q->stride(2) == 1 && Q->stride(1) == head_size, __func__, ": Q heads must be contiguous");
    AITER_CHECK(K->stride(3) == 1 && K->stride(2) == head_size &&
                K->stride(1) == num_kv_heads * head_size &&
                K->stride(0) == page_size * num_kv_heads * head_size,
                __func__, ": K must be a contiguous NHD page-16 cache");
    AITER_CHECK(V->stride(3) == 1 && V->stride(2) == head_size &&
                V->stride(1) == num_kv_heads * head_size &&
                V->stride(0) == page_size * num_kv_heads * head_size,
                __func__, ": V must be a contiguous NHD page-16 cache");
    AITER_CHECK(seq_lens->element_size() == 8, __func__, ": seq_lens must be int64");
    AITER_CHECK(cu_seqlens_q->element_size() == 4 && block_tables->element_size() == 4,
                __func__, ": cu_seqlens_q / block_tables must be int32");
    AITER_CHECK(k_descale->element_size() == 4 && v_descale->element_size() == 4,
                __func__, ": k/v descale must be fp32");
    AITER_CHECK(segm_out->size(0) == num_tokens && segm_out->size(1) == num_q_heads &&
                segm_out->size(2) == num_segments && segm_out->size(3) == head_size &&
                segm_out->is_contiguous(),
                __func__, ": segm_out must be [num_tokens, num_q_heads, num_segments, 256] contiguous");
    AITER_CHECK(segm_max->is_contiguous() && segm_expsum->is_contiguous(), __func__,
                ": segm_max / segm_expsum must be contiguous");
    AITER_CHECK(num_segments >= 1 && num_segments <= 64, __func__, ": num_segments must be in [1, 64]");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16 && out->stride(2) == 1, __func__,
                ": out must be bf16 with a contiguous head dim");

    const HipDeviceGuard device_guard(Q->device_id);

    VattnKernelArgs args = {};
    size_t arg_size      = sizeof(args);
    args.ptr_K           = K->data_ptr();
    args.ptr_V           = V->data_ptr();
    args.ptr_Q           = Q->data_ptr();
    args.ptr_BT          = block_tables->data_ptr();
    args.ptr_SL          = seq_lens->data_ptr();
    args.ptr_CU          = cu_seqlens_q->data_ptr();
    args.ptr_SO          = segm_out->data_ptr();
    args.ptr_SM          = segm_max->data_ptr();
    args.ptr_SE          = segm_expsum->data_ptr();
    args.num_segments    = num_segments;
    args.bt_stride0      = static_cast<int32_t>(block_tables->stride(0));
    args.q_stride0       = static_cast<int32_t>(Q->stride(0));
    args.num_kv_heads    = num_kv_heads;
    args.num_q_heads     = num_q_heads;
    args.scale_log2      = softmax_scale * k_log2e;
    args.v_scale         = 1.0f;
    uint32_t m, sh;
    magic_u32(static_cast<uint32_t>(num_segments) * 16u, m, sh);
    args.magic_m  = m;
    args.magic_sh = static_cast<int32_t>(sh);
    args._pad     = 0;
    args.ptr_KD   = k_descale->data_ptr();
    args.ptr_VD   = v_descale->data_ptr();

    static AiterAsmKernel impl_gqa16("vattn_asm", "/mtp_verify_attn/vattn_hd256_fp8_gqa16.co");
    static AiterAsmKernel impl_gqa8("vattn_asm", "/mtp_verify_attn/vattn_hd256_fp8_gqa8.co");
    static AiterAsmKernel impl_reduce("vred_asm", "/mtp_verify_attn/vattn_reduce.co");

    AiterAsmKernel* impl = (gqa == 16) ? &impl_gqa16 : &impl_gqa8;
    impl->launch_kernel({&args,
                         &arg_size,
                         num_segments, // gdx: KV segments
                         num_seqs,     // gdy
                         num_kv_heads, // gdz
                         512,          // bdx: 8 waves
                         1,
                         1,
                         stream});

    VredKernelArgs rargs = {};
    size_t rarg_size     = sizeof(rargs);
    rargs.ptr_O          = out->data_ptr();
    rargs.ptr_SO         = segm_out->data_ptr();
    rargs.ptr_SM         = segm_max->data_ptr();
    rargs.ptr_SE         = segm_expsum->data_ptr();
    rargs.num_q_heads    = static_cast<uint32_t>(num_q_heads);
    rargs.num_segments   = static_cast<uint32_t>(num_segments);
    rargs.o_stride0      = static_cast<uint32_t>(out->stride(0));
    rargs.o_stride1      = static_cast<uint32_t>(out->stride(1));
    magic_u32(static_cast<uint32_t>(num_q_heads), m, sh);
    rargs.magic_m  = m;
    rargs.magic_sh = sh;
    impl_reduce.launch_kernel({&rargs,
                               &rarg_size,
                               num_tokens * num_q_heads, // one wave per (token, head, 64-dim quarter)
                               1,
                               1,
                               256,
                               1,
                               1,
                               stream});
}

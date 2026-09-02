// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 (MI400) DSA sparse-prefill MLA asm dispatcher.

#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mla_v4_configs.hpp"
#include <cstddef>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

AITER_CTYPES_ERROR_DEF

namespace {

struct __attribute__((packed)) MlaSparsePrefillKargs
{
    const void* q_nope;            // 0x00  [T, H, 512] fp8   (448 data + 14 e8m0 scales + pad)
    const void* q_rope;            // 0x08  [T, H, 64]  bf16
    const void* unified_kv_nope;   // 0x10  [total_pages, 512] fp8   <- prefix source
    const void* unified_kv_rope;   // 0x18  [total_pages, 64]  bf16
    const void* kv_nope;           // 0x20  [total_tokens, 512] fp8  <- extend source
    const void* kv_rope;           // 0x28  [total_tokens, 64]  bf16
    const void* attn_sink;         // 0x30  [H] fp32
    void* out;                     // 0x38  [T, H, 512] bf16 (read_write)
    const void* kv_indptr_prefix;  // 0x40  [T+1] int32
    const void* kv_indices_prefix; // 0x48  [nnz_prefix] int32
    const void* kv_indptr_extend;  // 0x50  [T+1] int32
    const void* kv_indices_extend; // 0x58  [nnz_extend] int32
    float softmax_scale;           // 0x60  f32 by_value
    unsigned int _tail_pad;        // 0x64
};

static_assert(sizeof(MlaSparsePrefillKargs) == 104,
              "kernarg packet must stay 104 bytes (.kernarg_segment_size in the .co)");
static_assert(offsetof(MlaSparsePrefillKargs, q_nope) == 0x00, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, q_rope) == 0x08, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, unified_kv_nope) == 0x10, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, unified_kv_rope) == 0x18, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, kv_nope) == 0x20, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, kv_rope) == 0x28, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, attn_sink) == 0x30, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, out) == 0x38, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, kv_indptr_prefix) == 0x40, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, kv_indices_prefix) == 0x48, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, kv_indptr_extend) == 0x50, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, kv_indices_extend) == 0x58, "kernarg offset drift");
static_assert(offsetof(MlaSparsePrefillKargs, softmax_scale) == 0x60, "kernarg offset drift");

constexpr int kHeads      = 128; // == the CSV's Gqa for the one prefill row
constexpr int kHeadDim    = 512; // NoPE packed row: 448 fp8 + 14 e8m0 scales + 50 pad
constexpr int kRopeDim    = 64;
constexpr int kBlockDim   = 128; // 4 x wave32

std::string dtype_name(const aiter_tensor_t* t)
{
    return AiterDtype_to_str(t->dtype());
}

// `allow_empty` covers the CSR index lists: a region with no entries at all is
// a legal input (the kernel branches straight past an empty prefix stream and
// parks the prefix->extend switch index out of reach when extend is empty), and
// torch hands us a NULL data pointer for a zero-element tensor. Everything else
// must be a real allocation.
void check_tensor(const aiter_tensor_t* t,
                  const char* name,
                  AiterDtype want,
                  bool allow_empty = false)
{
    AITER_CHECK(t != nullptr, "mla_sparse_prefill_fp8_asm: `", name, "` must not be NULL");
    AITER_CHECK(t->data_ptr() != nullptr || (allow_empty && t->numel() == 0),
                "mla_sparse_prefill_fp8_asm: `", name, "` has a NULL data pointer");
    AITER_CHECK(t->dtype() == want,
                "mla_sparse_prefill_fp8_asm: `", name, "` must be ",
                AiterDtype_to_str(want), ", got ", dtype_name(t));
    AITER_CHECK(t->is_contiguous(),
                "mla_sparse_prefill_fp8_asm: `", name, "` must be contiguous");
}

// CSR row pointers are read as [T+1] int32. An empty index list is legal.
void check_csr(const aiter_tensor_t* indptr,
               const aiter_tensor_t* indices,
               const char* indptr_name,
               const char* indices_name,
               int64_t t)
{
    check_tensor(indptr, indptr_name, AITER_DTYPE_i32);
    check_tensor(indices, indices_name, AITER_DTYPE_i32, /*allow_empty=*/true);
    AITER_CHECK(indptr->numel() >= static_cast<size_t>(t + 1),
                "mla_sparse_prefill_fp8_asm: `", indptr_name, "` must have at least T+1 = ",
                t + 1, " elements, got ", indptr->numel());
}

} // namespace

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mla_sparse_prefill_fp8_asm_fwd,
    (aiter_tensor_t * q_nope,             // [T, H, 512] fp8
     aiter_tensor_t* q_rope,              // [T, H, 64]  bf16
     aiter_tensor_t* unified_kv_nope,     // [total_pages, 512] fp8
     aiter_tensor_t* unified_kv_rope,     // [total_pages, 64]  bf16
     aiter_tensor_t* kv_indices_prefix,   // [nnz_prefix] int32
     aiter_tensor_t* kv_indptr_prefix,    // [T+1] int32
     aiter_tensor_t* kv_nope,             // [total_tokens, 512] fp8
     aiter_tensor_t* kv_rope,             // [total_tokens, 64]  bf16
     aiter_tensor_t* kv_indices_extend,   // [nnz_extend] int32
     aiter_tensor_t* kv_indptr_extend,    // [T+1] int32
     aiter_tensor_t* attn_sink,           // [H] fp32
     aiter_tensor_t* out,                 // [T, H, 512] bf16 (written)
     float softmax_scale,
     hipStream_t stream),
    (q_nope,
     q_rope,
     unified_kv_nope,
     unified_kv_rope,
     kv_indices_prefix,
     kv_indptr_prefix,
     kv_nope,
     kv_rope,
     kv_indices_extend,
     kv_indptr_extend,
     attn_sink,
     out,
     softmax_scale,
     stream))
{
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK(arch_id == "gfx1250",
                "mla_sparse_prefill_fp8_asm: only gfx1250 is supported, got ", arch_id);

    check_tensor(q_nope, "q_nope", AITER_DTYPE_fp8);
    check_tensor(q_rope, "q_rope", AITER_DTYPE_bf16);
    check_tensor(unified_kv_nope, "unified_kv_nope", AITER_DTYPE_fp8);
    check_tensor(unified_kv_rope, "unified_kv_rope", AITER_DTYPE_bf16);
    check_tensor(kv_nope, "kv_nope", AITER_DTYPE_fp8);
    check_tensor(kv_rope, "kv_rope", AITER_DTYPE_bf16);
    check_tensor(attn_sink, "attn_sink", AITER_DTYPE_fp32);
    check_tensor(out, "out", AITER_DTYPE_bf16);

    AITER_CHECK(q_nope->dim() == 3,
                "mla_sparse_prefill_fp8_asm: `q_nope` must be 3-D [T, H, 512], got ndim=",
                q_nope->dim());
    const int64_t t = q_nope->size(0);
    const int64_t h = q_nope->size(1);

    AITER_CHECK(h == kHeads,
                "mla_sparse_prefill_fp8_asm: this kernel is built for exactly H=", kHeads,
                " heads (one workgroup serves one query token x ", kHeads,
                " heads, and the Q address math requires gridDim.y == 1); got H=", h);
    AITER_CHECK(q_nope->size(2) == kHeadDim,
                "mla_sparse_prefill_fp8_asm: `q_nope` last dim must be ", kHeadDim, ", got ",
                q_nope->size(2));
    AITER_CHECK(q_rope->size(2) == kRopeDim,
                "mla_sparse_prefill_fp8_asm: `q_rope` last dim must be ", kRopeDim, ", got ",
                q_rope->size(2));
    AITER_CHECK(unified_kv_nope->size(-1) == kHeadDim && kv_nope->size(-1) == kHeadDim,
                "mla_sparse_prefill_fp8_asm: KV NoPE rows must be ", kHeadDim, " wide, got ",
                unified_kv_nope->size(-1), " / ", kv_nope->size(-1));
    AITER_CHECK(unified_kv_rope->size(-1) == kRopeDim && kv_rope->size(-1) == kRopeDim,
                "mla_sparse_prefill_fp8_asm: KV RoPE rows must be ", kRopeDim, " wide, got ",
                unified_kv_rope->size(-1), " / ", kv_rope->size(-1));
    AITER_CHECK(attn_sink->numel() == static_cast<size_t>(h),
                "mla_sparse_prefill_fp8_asm: `attn_sink` must have H=", h, " elements, got ",
                attn_sink->numel());
    AITER_CHECK(out->dim() == 3 && out->size(0) == t && out->size(1) == h &&
                    out->size(2) == kHeadDim,
                "mla_sparse_prefill_fp8_asm: `out` must be [", t, ", ", h, ", ", kHeadDim, "]");

    check_csr(kv_indptr_prefix, kv_indices_prefix, "kv_indptr_prefix", "kv_indices_prefix", t);
    check_csr(kv_indptr_extend, kv_indices_extend, "kv_indptr_extend", "kv_indices_extend", t);

    MlaSparsePrefillKargs args;
    size_t arg_size          = sizeof(args);
    args.q_nope              = q_nope->data_ptr();
    args.q_rope              = q_rope->data_ptr();
    args.unified_kv_nope     = unified_kv_nope->data_ptr();
    args.unified_kv_rope     = unified_kv_rope->data_ptr();
    args.kv_nope             = kv_nope->data_ptr();
    args.kv_rope             = kv_rope->data_ptr();
    args.attn_sink           = attn_sink->data_ptr();
    args.out                 = out->data_ptr();
    args.kv_indptr_prefix    = kv_indptr_prefix->data_ptr();
    args.kv_indices_prefix   = kv_indices_prefix->data_ptr();
    args.kv_indptr_extend    = kv_indptr_extend->data_ptr();
    args.kv_indices_extend   = kv_indices_extend->data_ptr();
    args.softmax_scale       = softmax_scale;
    args._tail_pad           = 0;

    CFG* config_map = &cfg_mla_v4_asm;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = nullptr;

    // The manifest is shared with the v4 sparse DECODE kernels, so prefill must be
    // part of the key -- otherwise a decode row with a matching Gqa would be picked
    // and launched with the wrong kernarg packet. Gqa is the head count (one
    // workgroup serves one query token x Gqa heads) and qSeqLen is 1 by
    // construction on this path.
    std::string kernelName;
    for(const auto& el : *config_map)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.prefill != 1 || cfg.causal != 0 || cfg.lse != 0 || cfg.ps != 0)
            continue;
        if(cfg.qType != "fp8" || cfg.kvType != "fp8")
            continue;
        if(cfg.Gqa != static_cast<int>(h) || cfg.qSeqLen != 1)
            continue;
        kernelName = el.first;
        break;
    }
    AITER_CHECK(!kernelName.empty(),
                "mla_sparse_prefill_fp8_asm: no prefill kernel for arch=", arch_id,
                " qType=fp8 kvType=fp8 Gqa=", h);

    auto it = config_map->find(kernelName);
    AITER_CHECK(it != config_map->end(),
                "mla_sparse_prefill_fp8_asm: kernel not found: ", kernelName);
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }

    // gdx = one workgroup per query token. gdy MUST stay 1 (see kHeads note).
    AITER_CHECK(t >= 0 && (t >> 31) == 0, "mla_sparse_prefill_fp8_asm: T too large: ", t);
    const int gdx = static_cast<int>(t);
    if(gdx == 0)
        return; // nothing to do; `out` stays as the caller left it

    if(const char* dbg = std::getenv("AITER_MLA_SPARSE_PREFILL_DUMP_KERNARG"))
    {
        if(dbg[0] == '1')
        {
            fprintf(stderr, "[aiter pa_sparse_prefill kernarg %zuB]\n", arg_size);
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&args);
            for(size_t i = 0; i < arg_size; ++i)
                fprintf(stderr, "%02x%s", bytes[i], ((i + 1) % 16 == 0) ? "\n" : " ");
            fprintf(stderr, "\n[aiter grid (%d,1,1) block (%d,1,1)]\n", gdx, kBlockDim);
            fflush(stderr);
        }
    }

    const HipDeviceGuard device_guard(q_nope->device_id);
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdx,       // gdx: one query token per workgroup
                             1,         // gdy: MUST be 1
                             1,         // gdz
                             kBlockDim, // bdx: 4 x wave32
                             1,         // bdy
                             1,         // bdz
                             stream});
}

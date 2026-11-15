// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM FMHA forward, VARLEN / packed, FP8 (e4m3fn) with per-tensor descales, gfx1250.
// qk head dim 192, v head dim 128; output bf16.
//
//   q   : (total_q, nheads,   192) fp8 e4m3fn
//   k   : (total_k, nheads_k, 192) fp8 e4m3fn
//   v   : (total_k, nheads_k, 128) fp8 e4m3fn
//   out : (total_q, nheads,   128) bf16
//   lse : packed [total_q, nheads] fp32, natural log (caller may shape (total_q, nheads, 1))
//   q/k/v_descale : fp32, one element each (per-tensor), read by the kernel from device memory
//   cu_seqlens_q/k : int32 [batch+1] cumulative, cu[batch] == total
//
// q/k/v/out are addressed through their token / head strides (contiguous last dim), so views
// of a larger buffer work in place.  Causal masking is bottom-right aligned.
//
// The softmax probabilities P are rounded to fp8 before the PV product, and the row sum (the
// denominator of O and the LSE) is the sum of those rounded P: O is consistent with the P actually
// used, and the LSE is the log of that same sum, an approximation of the exact log-sum-exp.
//
// Memory-allocation policy: all tensors are allocated by the Python caller.  This entry point
// only fills the kernel arguments and launches -- no GPU allocation, no torch dependency.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"   // HipDeviceGuard, AiterAsmKernel, ...
#include "asm_fmha_fwd_fp8_varlen_configs.hpp"
#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Kernel argument block (136 B).  The first 120 B mirror the bf16 D192x128 strided varlen ABI
// (FmhaFwdVarlenStrideKernelArgs) with the sink pointer slot at 0x50 holding the q descale
// pointer; k/v descale pointers follow at 0x78 / 0x80.  The kernel preloads the first 120 B
// (s2..s31) and loads the last 16 B itself.
#pragma pack(push, 1)
struct FmhaFwdFp8VarlenKernelArgs
{
    void*       d_addr;          // off 0x00  O output     (total_q, nheads, 128) bf16
    const void* q_addr;          // off 0x08  Q            (total_q, nheads, 192)
    const void* k_addr;          // off 0x10  K            (total_k, nheads_k, 192)
    const void* v_addr;          // off 0x18  V            (total_k, nheads_k, 128)
    void*       lse_addr;        // off 0x20  LSE packed   [total_q, nheads]
    const void* qseq_addr;       // off 0x28  cu_seqlens_q int32[batch+1]
    const void* kseq_addr;       // off 0x30  cu_seqlens_k int32[batch+1]
    float       scalar;          // off 0x38  softmax_scale (descales are applied by the kernel)
    int         gqa;             // off 0x3C  nheads / nheads_k
    int         q_head_num;      // off 0x40  nheads
    int         opt;             // off 0x44  bit1 double_q | bit2 remap_xy (bit0 ignored)
    int         lse;             // off 0x48  1 = write LSE
    int         max_q_len;       // off 0x4C  max over batches of q seqlen (dispatch basis)
    const void* q_descale_addr;  // off 0x50  fp32 [1]
    int         q_hs;            // off 0x58  Q   head  stride (bytes)
    int         k_hs;            // off 0x5C  K   head  stride (bytes)
    int         v_hs;            // off 0x60  V   head  stride (bytes)
    int         d_hs;            // off 0x64  out head  stride (bytes)
    int         q_seqs;          // off 0x68  Q   token stride (bytes)
    int         k_seqs;          // off 0x6C  K   token stride (bytes)
    int         v_seqs;          // off 0x70  V   token stride (bytes)
    int         d_seqs;          // off 0x74  out token stride (bytes)
    const void* k_descale_addr;  // off 0x78  fp32 [1]
    const void* v_descale_addr;  // off 0x80  fp32 [1]
};
#pragma pack(pop)
static_assert(sizeof(FmhaFwdFp8VarlenKernelArgs) == 0x88,
              "fmha_fwd_fp8_varlen_asm: FmhaFwdFp8VarlenKernelArgs must be 136B packed");

// ---- helpers ---------------------------------------------------------------

// Kernel selection: (dtype, hdim_q, hdim_v, mask); mask = is_causal (the causal kernel is a
// separate build).
static std::string get_heuristic_kernel_fmha_fwd_fp8_varlen(const std::string& dtype,
                                                            int hdim_q,
                                                            int hdim_v,
                                                            int mask_flag,
                                                            const std::string& arch_id,
                                                            CFG* cfgs)
{
    for (const auto& el : *cfgs)
    {
        if (el.first.find(arch_id) != 0) continue;
        const auto& cfg = el.second;
        if (cfg.dtype   != dtype)       continue;
        if (cfg.hdim_q  != hdim_q)      continue;
        if (cfg.hdim_v  != hdim_v)      continue;
        if (cfg.mask    != mask_flag)   continue;
        return el.first;
    }
    AITER_CHECK(false,
                "fmha_fwd_fp8_varlen_asm: no kernel for dtype=", dtype,
                " hdim_q=", hdim_q, " hdim_v=", hdim_v,
                " mask=", mask_flag,
                " arch=", arch_id);
    return "";
}

// The kernel steps the KV loop by 64 * token stride as a 32-bit value and keeps the strides in
// 32-bit registers; token stride < 2^24 B keeps the same margin as the bf16 D192x128 kernel.
static void check_strided_thd(const aiter_tensor_t* t, const char* name)
{
    const int64_t esz = (int64_t)t->element_size();
    const int64_t ts  = t->stride(0) * esz;
    const int64_t hs  = t->stride(1) * esz;
    AITER_CHECK(t->stride(-1) == 1,
                "fmha_fwd_fp8_varlen_asm: ", name, " must have contiguous last dim");
    AITER_CHECK(ts > 0 && ts < (int64_t(1) << 24) && hs > 0 && hs <= INT32_MAX,
                "fmha_fwd_fp8_varlen_asm: ", name, " strides out of range (token ", ts,
                " B must be in (0, 2^24), head ", hs, " B must be in (0, INT32_MAX])");
}

static void check_descale(const aiter_tensor_t* t, const char* name, int device_id)
{
    AITER_CHECK(t->dtype() == AITER_DTYPE_fp32 && t->numel() == 1,
                "fmha_fwd_fp8_varlen_asm: ", name, " must be a single fp32 value (per-tensor)");
    AITER_CHECK(t->device_id == device_id,
                "fmha_fwd_fp8_varlen_asm: ", name, " must be on the same GPU as q");
}

static int cu_count(int device)
{
    static std::atomic<int> cache[64];
    if (device < 0 || device >= 64)
        device = 0;
    int n = cache[device].load(std::memory_order_relaxed);
    if (n <= 0)
    {
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess || n <= 0)
            n = 256;
        cache[device].store(n, std::memory_order_relaxed);
    }
    return n;
}

// Fraction of the last round of workgroups that is busy when n workgroups run s at a time.
static double wave_fill(long n, int s)
{
    if (n <= 0)
        return 1.0;
    const long rounds = (n + s - 1) / s;
    return (double)n / ((double)rounds * s);
}

// Launch options (bit1 double_q, bit2 remap_xy).  One 4-wave workgroup runs per CU.
//   double_q: a workgroup runs Q tiles g and tiles-1-g one after the other, so the grid has half as
//     many workgroups.  Causal: every workgroup gets the same work (a heavy tile plus a light one).
//     Non-causal: the two tiles share K/V in L2 -- a gain unless halving the grid leaves the last
//     round of workgroups emptier, so the fill of the last round is compared with and without it.
//     Causal without double_q relies on the heaviest-first launch order instead (remap_xy, below).
//   remap_xy: x indexes heads, so workgroups launched together are different heads of one Q tile.
//     It helps while the heads' K/V together are small (q heads x kv length under the threshold
//     below); with many heads over long sequences they stream K/V through L2 at once and it hurts.
//     Causal without double_q always uses it: it launches the heavy tiles of every head first.
// The thresholds come from timing the four options over head counts and sequence lengths on gfx1250,
// weighted towards many heads with long sequences.
// AITER_FMHA_FP8_VARLEN_ASM_OPT=<int> overrides the choice (bits 1 and 2 only).
static int select_opt(int batch, int max_seqlen_q, int max_seqlen_k, int q_head_num, int is_causal, int device)
{
    const char* env = std::getenv("AITER_FMHA_FP8_VARLEN_ASM_OPT");
    if (env && *env)
        return (int)std::strtol(env, nullptr, 0) & 0x6;
    const int  s     = cu_count(device);
    const long tiles = (max_seqlen_q + 255) / 256;
    const long wg    = tiles * q_head_num * batch;                   // workgroups without double_q
    const long wg_dq = (tiles + 1) / 2 * q_head_num * batch;         // with double_q
    const bool remap = (long)q_head_num * max_seqlen_k <= (1L << 18);
    const bool dq    = is_causal ? wave_fill(wg_dq, s) >= 0.93
                                 : wave_fill(wg_dq, s) * 1.03 >= wave_fill(wg, s);
    if (dq)
        return 0x2 | (remap ? 0x4 : 0x0);
    return is_causal ? 0x4 : 0x0;
}

// ---- main entry ------------------------------------------------------------

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmha_fwd_fp8_varlen_asm,
    (aiter_tensor_t* q,
     aiter_tensor_t* k,
     aiter_tensor_t* v,
     aiter_tensor_t* q_descale,
     aiter_tensor_t* k_descale,
     aiter_tensor_t* v_descale,
     aiter_tensor_t* out,
     aiter_tensor_t* lse,
     aiter_tensor_t* cu_seqlens_q,
     aiter_tensor_t* cu_seqlens_k,
     int             max_seqlen_q,
     int             max_seqlen_k,
     float           softmax_scale,
     int             is_causal,
     int             return_lse,
     hipStream_t     stream),
    (q, k, v, q_descale, k_descale, v_descale, out, lse, cu_seqlens_q, cu_seqlens_k,
     max_seqlen_q, max_seqlen_k, softmax_scale, is_causal, return_lse, stream))
{
    // ---- null safety ---------------------------------------------------------
    AITER_CHECK(q && k && v && out && lse && cu_seqlens_q && cu_seqlens_k,
                "fmha_fwd_fp8_varlen_asm: q/k/v/out/lse/cu_seqlens_q/cu_seqlens_k must all be non-null");
    AITER_CHECK(q_descale && k_descale && v_descale,
                "fmha_fwd_fp8_varlen_asm: q/k/v_descale must all be non-null");

    HipDeviceGuard device_guard{q->device_id};

    // ---- arch + dtype validation ----------------------------------------------
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK((arch_id == "gfx1250" || arch_id == "gfx1250-strict"),
                "fmha_fwd_fp8_varlen_asm: only supported on gfx1250, got ", arch_id);

    AITER_CHECK(q->dtype() == AITER_DTYPE_fp8 &&
                k->dtype() == AITER_DTYPE_fp8 &&
                v->dtype() == AITER_DTYPE_fp8,
                "fmha_fwd_fp8_varlen_asm: q/k/v must be fp8 (e4m3fn)");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_fp8_varlen_asm: out must be bf16");
    AITER_CHECK(lse->dtype() == AITER_DTYPE_fp32,
                "fmha_fwd_fp8_varlen_asm: lse must be fp32");
    AITER_CHECK(cu_seqlens_q->dtype() == AITER_DTYPE_i32 &&
                cu_seqlens_k->dtype() == AITER_DTYPE_i32,
                "fmha_fwd_fp8_varlen_asm: cu_seqlens_q/k must be int32");
    check_descale(q_descale, "q_descale", q->device_id);
    check_descale(k_descale, "k_descale", q->device_id);
    check_descale(v_descale, "v_descale", q->device_id);

    // ---- shape extraction (packed thd) ----------------------------------------
    AITER_CHECK(q->dim() == 3 && k->dim() == 3 && v->dim() == 3,
                "fmha_fwd_fp8_varlen_asm: q/k/v must be 3-D packed tensors (total, head, dim)");

    const int total_q     = (int)q->size(0);
    const int q_head_num  = (int)q->size(1);
    const int qk_head_dim = (int)q->size(2);

    const int total_k     = (int)k->size(0);
    const int kv_head_num = (int)k->size(1);
    const int v_head_dim  = (int)v->size(2);

    AITER_CHECK(qk_head_dim == 192 && v_head_dim == 128 && (int)k->size(2) == qk_head_dim,
                "fmha_fwd_fp8_varlen_asm: only (qk_head_dim, v_head_dim) = (192, 128) is supported; got (",
                qk_head_dim, ",", v_head_dim, ")");
    AITER_CHECK((int)v->size(0) == total_k,     "fmha_fwd_fp8_varlen_asm: v total_k mismatch with k");
    AITER_CHECK((int)v->size(1) == kv_head_num, "fmha_fwd_fp8_varlen_asm: v head_num mismatch with k");
    AITER_CHECK(q_head_num > 0 && kv_head_num > 0,
                "fmha_fwd_fp8_varlen_asm: q/kv head counts must be positive");
    AITER_CHECK(q_head_num % kv_head_num == 0,
                "fmha_fwd_fp8_varlen_asm: q_head_num must be a multiple of kv_head_num");

    AITER_CHECK(out->dim() == 3 &&
                (int)out->size(0) == total_q && (int)out->size(1) == q_head_num &&
                (int)out->size(2) == v_head_dim,
                "fmha_fwd_fp8_varlen_asm: out shape must be [total_q, q_head_num, 128]");
    AITER_CHECK(lse->dim() >= 2 &&
                (int)lse->size(0) == total_q && (int)lse->size(1) == q_head_num,
                "fmha_fwd_fp8_varlen_asm: lse leading dims must be [total_q, q_head_num]");
    AITER_CHECK(lse->is_contiguous(),
                "fmha_fwd_fp8_varlen_asm: lse must be contiguous");

    check_strided_thd(q, "q");
    check_strided_thd(k, "k");
    check_strided_thd(v, "v");
    check_strided_thd(out, "out");

    AITER_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1,
                "fmha_fwd_fp8_varlen_asm: cu_seqlens_q/k must be 1-D");
    const int batch = (int)cu_seqlens_q->size(0) - 1;
    AITER_CHECK(batch >= 1,
                "fmha_fwd_fp8_varlen_asm: cu_seqlens_q must have length batch+1 (>=2)");
    AITER_CHECK((int)cu_seqlens_k->size(0) == batch + 1,
                "fmha_fwd_fp8_varlen_asm: cu_seqlens_k length must match cu_seqlens_q");
    AITER_CHECK(max_seqlen_q > 0,
                "fmha_fwd_fp8_varlen_asm: max_seqlen_q must be > 0");

    const int mask_flag = is_causal ? 1 : 0;
    const int opt       = select_opt(batch, max_seqlen_q, max_seqlen_k, q_head_num, is_causal, q->device_id);

    // ---- kernel args -------------------------------------------------------------
    FmhaFwdFp8VarlenKernelArgs args;
    memset(&args, 0, sizeof(args));
    args.d_addr         = out->data_ptr();
    args.q_addr         = q->data_ptr();
    args.k_addr         = k->data_ptr();
    args.v_addr         = v->data_ptr();
    args.lse_addr       = lse->data_ptr();
    args.qseq_addr      = cu_seqlens_q->data_ptr();
    args.kseq_addr      = cu_seqlens_k->data_ptr();
    args.scalar         = softmax_scale;
    args.gqa            = q_head_num / kv_head_num;
    args.q_head_num     = q_head_num;
    args.opt            = opt;
    args.lse            = return_lse ? 1 : 0;
    args.max_q_len      = max_seqlen_q;
    args.q_descale_addr = q_descale->data_ptr();
    args.q_hs           = (int)(q->stride(1) * q->element_size());
    args.k_hs           = (int)(k->stride(1) * k->element_size());
    args.v_hs           = (int)(v->stride(1) * v->element_size());
    args.d_hs           = (int)(out->stride(1) * out->element_size());
    args.q_seqs         = (int)(q->stride(0) * q->element_size());
    args.k_seqs         = (int)(k->stride(0) * k->element_size());
    args.v_seqs         = (int)(v->stride(0) * v->element_size());
    args.d_seqs         = (int)(out->stride(0) * out->element_size());
    args.k_descale_addr = k_descale->data_ptr();
    args.v_descale_addr = v_descale->data_ptr();
    size_t arg_size     = sizeof(args);

    // ---- kernel selection ----------------------------------------------------------
    const std::string dtype = "fp8";
    CFG* cfg_map            = &cfg_fmha_fwd_fp8_varlen;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const std::string kernel_key = get_heuristic_kernel_fmha_fwd_fp8_varlen(
        dtype, qk_head_dim, v_head_dim, mask_flag, arch_id, cfg_map);
    auto it = cfg_map->find(kernel_key);
    AITER_CHECK(it != cfg_map->end(),
                "fmha_fwd_fp8_varlen_asm: kernel not found in CFG: ", kernel_key);

    const char* name    = it->second.knl_name.c_str();
    const char* co_name = it->second.co_name.c_str();
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        name, [&]() { return AiterAsmKernel(name, co_name); });

    // ---- launch ------------------------------------------------------------------------
    // x = Q tiles of 256 rows over max_q_len (halved by double_q), y = q heads, z = batch;
    // remap_xy swaps x and y.  4 waves of 32 lanes per workgroup.  Tiles beyond a batch's own
    // length exit early.
    const int sub_Q        = 256;
    const int bdx          = 128;
    const int q_tile_count = (max_seqlen_q + sub_Q - 1) / sub_Q;
    const int gdx          = (opt & 0x2) ? (q_tile_count + 1) / 2 : q_tile_count;
    const int gdy          = q_head_num;
    const int gdz          = batch;
    const bool remap       = (opt & 0x4) != 0;

    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             remap ? gdy : gdx,
                             remap ? gdx : gdy,
                             gdz,
                             bdx,
                             1,
                             1,
                             stream});
}

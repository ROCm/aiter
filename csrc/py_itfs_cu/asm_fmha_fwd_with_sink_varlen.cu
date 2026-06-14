// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ASM FMHA forward, VARLEN / packed (BF16, gfx1250).
//
// Layout: q/k/v/out are **packed [token, head, dim]** (batch folded into the
// token axis).  Per-batch sequence boundaries are described by cumulative
// length arrays cu_seqlens_q / cu_seqlens_k (int32, length batch+1, no
// padding).  Unlike the fixed-batch path, the kernel computes all addresses
// internally from (q_head_num, gqa, head_dim, cu_seqlens) -- so the kernarg
// block carries NO strides and the tensors MUST be densely packed / contiguous.
//
//   q   : (total_q, nheads,   hdim_q)
//   k   : (total_k, nheads_k, hdim_q)
//   v   : (total_k, nheads_k, hdim_v)
//   out : (total_q, nheads,   hdim_v)
//   lse : packed [total_q, nheads]  (token-major; caller may shape (total_q, nheads, 1))
//   cu_seqlens_q/k : int32 [batch+1] cumulative, cu[batch] == total
//
// Memory-allocation policy: all tensors are allocated by the Python caller.
// This C++ entry point performs only pointer bookkeeping + kernel launch --
// no GPU allocation, no temporaries, no torch dependency.
//
// sink: passed through verbatim (the value the kernel consumes directly, no
// host-side scaling).  Optional -- may be null; whether the kernel reads it is
// decided inside the .co (ENABLE_SINK).
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "aiter_hip_common.h"   // HipDeviceGuard, AiterAsmKernel, ...
#include "asm_fmha_fwd_bf16_varlen_configs.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <memory>

// Kernel argument block -- packed varlen ABI (0x58 = 88 B), matches the
// FmhaFwdVarlenKernelArgs layout in the poc host code (s_load order of the
// BF16_FMHA_FWD_VARLEN_*.s kernels).  No strides: packed [token, head, dim].
#pragma pack(push, 1)
struct FmhaFwdVarlenKernelArgs
{
    void*        d_addr;       // off 0x00  O output     (total_q, nheads, dv)
    const void*  q_addr;       // off 0x08  Q            (total_q, nheads, dq)
    const void*  k_addr;       // off 0x10  K            (total_k, nheads_k, dq)
    const void*  v_addr;       // off 0x18  V            (total_k, nheads_k, dv)
    void*        lse_addr;     // off 0x20  LSE packed   [total_q, nheads]
    const void*  qseq_addr;    // off 0x28  cu_seqlens_q int32[batch+1]
    const void*  kseq_addr;    // off 0x30  cu_seqlens_k int32[batch+1]
    float        scalar;       // off 0x38  softmax_scale
    int          gqa;          // off 0x3C  nheads / nheads_k
    int          q_head_num;   // off 0x40  nheads
    int          opt;          // off 0x44  bit0 reverse_kv | bit1 double_q | bit2 remap_xy
    int          lse;          // off 0x48  1 = write LSE
    int          max_q_len;    // off 0x4C  max over batches of q seqlen (dispatch basis)
    void*        sink_addr;    // off 0x50  per-Q-head f32 sink (verbatim; may be null)
};
#pragma pack(pop)
static_assert(sizeof(FmhaFwdVarlenKernelArgs) == 0x58,
              "fmha_fwd_with_sink_varlen_asm: FmhaFwdVarlenKernelArgs must be 88B packed");

// ---- helpers ---------------------------------------------------------------

// Kernel selection: only (dtype, hdim_q, hdim_v, mask).  Only the _brd (border)
// causal kernels are shipped, so mask is always 1.
static std::string get_heuristic_kernel_fmha_fwd_bf16_varlen(const std::string& dtype,
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
                "fmha_fwd_with_sink_varlen_asm: no kernel for dtype=", dtype,
                " hdim_q=", hdim_q, " hdim_v=", hdim_v,
                " mask=", mask_flag,
                " arch=", arch_id);
    return "";
}

// ---- main entry ------------------------------------------------------------

AITER_CTYPES_ERROR_DEF

// C ABI: every tensor is caller-allocated.  No GPU memory is allocated here;
// no torch dependency.
//
// q/k/v/out are packed [token, head, dim] (densely contiguous).  cu_seqlens_q/k
// are int32 [batch+1] cumulative arrays.  max_seqlen_q is the maximum per-batch
// Q sequence length (host-supplied; used for the launch tile count).  sink is
// optional and forwarded verbatim.
AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmha_fwd_with_sink_varlen_asm,
    (aiter_tensor_t* q,
     aiter_tensor_t* k,
     aiter_tensor_t* v,
     aiter_tensor_t* out,
     aiter_tensor_t* lse,
     aiter_tensor_t* sink,
     aiter_tensor_t* cu_seqlens_q,
     aiter_tensor_t* cu_seqlens_k,
     int             max_seqlen_q,
     float           softmax_scale,
     int             is_causal,
     int             return_lse,
     hipStream_t     stream),
    (q, k, v, out, lse, sink, cu_seqlens_q, cu_seqlens_k,
     max_seqlen_q, softmax_scale, is_causal, return_lse, stream))
{
    // ---- null safety (sink is optional) -----------------------------------
    AITER_CHECK(q && k && v && out && lse && cu_seqlens_q && cu_seqlens_k,
                "fmha_fwd_with_sink_varlen_asm: q/k/v/out/lse/cu_seqlens_q/cu_seqlens_k must all be non-null");

    // Pin current HIP device to q.device() (torch-free) so kernel symbol
    // resolution + launch target the tensors' device.
    HipDeviceGuard device_guard{q->device_id};

    // ---- arch + dtype validation ------------------------------------------
    const std::string arch_id = get_gpu_arch();
    AITER_CHECK(arch_id == "gfx1250",
                "fmha_fwd_with_sink_varlen_asm: only supported on gfx1250, got ", arch_id);

    AITER_CHECK(q->dtype() == AITER_DTYPE_bf16 &&
                k->dtype() == AITER_DTYPE_bf16 &&
                v->dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_with_sink_varlen_asm: q/k/v must be bf16");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                "fmha_fwd_with_sink_varlen_asm: out must be bf16");
    AITER_CHECK(lse->dtype() == AITER_DTYPE_fp32,
                "fmha_fwd_with_sink_varlen_asm: lse must be fp32");
    AITER_CHECK(cu_seqlens_q->dtype() == AITER_DTYPE_i32 &&
                cu_seqlens_k->dtype() == AITER_DTYPE_i32,
                "fmha_fwd_with_sink_varlen_asm: cu_seqlens_q/k must be int32");
    if (sink)
    {
        AITER_CHECK(sink->dtype() == AITER_DTYPE_fp32,
                    "fmha_fwd_with_sink_varlen_asm: sink must be fp32");
    }

    // ---- shape extraction (packed thd) ------------------------------------
    AITER_CHECK(q->dim() == 3 && k->dim() == 3 && v->dim() == 3,
                "fmha_fwd_with_sink_varlen_asm: q/k/v must be 3-D packed tensors (total, head, dim)");
    AITER_CHECK(q->stride(-1) == 1 && k->stride(-1) == 1 && v->stride(-1) == 1,
                "fmha_fwd_with_sink_varlen_asm: q/k/v must have contiguous last dim");

    const int total_q     = (int)q->size(0);
    const int q_head_num  = (int)q->size(1);
    const int qk_head_dim = (int)q->size(2);

    const int total_k     = (int)k->size(0);
    const int kv_head_num = (int)k->size(1);
    const int v_head_dim  = (int)v->size(2);

    AITER_CHECK((int)k->size(2) == qk_head_dim, "fmha_fwd_with_sink_varlen_asm: k head_dim mismatch");
    AITER_CHECK((int)v->size(0) == total_k,     "fmha_fwd_with_sink_varlen_asm: v total_k mismatch with k");
    AITER_CHECK((int)v->size(1) == kv_head_num, "fmha_fwd_with_sink_varlen_asm: v head_num mismatch with k");
    AITER_CHECK(q_head_num % kv_head_num == 0,  "fmha_fwd_with_sink_varlen_asm: q_head_num must be a multiple of kv_head_num");
    AITER_CHECK(qk_head_dim == 64 || qk_head_dim == 128,
                "fmha_fwd_with_sink_varlen_asm: only head_dim 64 or 128 supported, got ", qk_head_dim);
    AITER_CHECK(v_head_dim == qk_head_dim,
                "fmha_fwd_with_sink_varlen_asm: v_head_dim must equal qk_head_dim");

    AITER_CHECK(out->dim() == 3 &&
                (int)out->size(0) == total_q && (int)out->size(1) == q_head_num &&
                (int)out->size(2) == v_head_dim,
                "fmha_fwd_with_sink_varlen_asm: out shape must be [total_q, q_head_num, v_head_dim]");
    AITER_CHECK(out->stride(-1) == 1,
                "fmha_fwd_with_sink_varlen_asm: out must have contiguous last dim");

    // lse packed [total_q, nheads]; caller may pass shape (total_q, nheads, 1).
    AITER_CHECK(lse->dim() >= 2 &&
                (int)lse->size(0) == total_q && (int)lse->size(1) == q_head_num,
                "fmha_fwd_with_sink_varlen_asm: lse leading dims must be [total_q, q_head_num]");

    AITER_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1,
                "fmha_fwd_with_sink_varlen_asm: cu_seqlens_q/k must be 1-D");
    const int batch = (int)cu_seqlens_q->size(0) - 1;
    AITER_CHECK(batch >= 1,
                "fmha_fwd_with_sink_varlen_asm: cu_seqlens_q must have length batch+1 (>=2)");
    AITER_CHECK((int)cu_seqlens_k->size(0) == batch + 1,
                "fmha_fwd_with_sink_varlen_asm: cu_seqlens_k length must match cu_seqlens_q");
    AITER_CHECK(max_seqlen_q > 0,
                "fmha_fwd_with_sink_varlen_asm: max_seqlen_q must be > 0");

    if (sink)
    {
        AITER_CHECK(sink->dim() == 1 && (int)sink->size(0) == q_head_num,
                    "fmha_fwd_with_sink_varlen_asm: sink must be 1-D with size q_head_num (", q_head_num, ")");
    }

    const int gqa       = q_head_num / kv_head_num;
    const int mask_flag = is_causal ? 1 : 0;

    // ---- kernel args (88 B packed; no strides) ----------------------------
    FmhaFwdVarlenKernelArgs args;
    memset(&args, 0, sizeof(args));
    args.d_addr     = out->data_ptr();
    args.q_addr     = q->data_ptr();
    args.k_addr     = k->data_ptr();
    args.v_addr     = v->data_ptr();
    args.lse_addr   = lse->data_ptr();
    args.qseq_addr  = cu_seqlens_q->data_ptr();
    args.kseq_addr  = cu_seqlens_k->data_ptr();
    args.scalar     = softmax_scale;
    args.gqa        = gqa;
    args.q_head_num = q_head_num;
    // s_opt: bit0 reverse_kv | bit1 double_q | bit2 remap_xy.
    // 6 = 0b110 -> reverse_kv=0, double_q=1, remap_xy=1.  Must match how the
    // shipped VARLEN .co was built.
    args.opt        = 4;
    args.lse        = return_lse ? 1 : 0;
    args.max_q_len  = max_seqlen_q;
    args.sink_addr  = sink ? sink->data_ptr() : nullptr;

    size_t arg_size = sizeof(args);

    // ---- kernel selection --------------------------------------------------
    const std::string dtype = "bf16";
    CFG* cfg_map            = &cfg_fmha_fwd_bf16_varlen;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const std::string kernel_key = get_heuristic_kernel_fmha_fwd_bf16_varlen(
        dtype, qk_head_dim, v_head_dim, mask_flag, arch_id, cfg_map);
    auto it = cfg_map->find(kernel_key);
    AITER_CHECK(it != cfg_map->end(),
                "fmha_fwd_with_sink_varlen_asm: kernel not found in CFG: ", kernel_key);

    const char* name    = it->second.knl_name.c_str();
    const char* co_name = it->second.co_name.c_str();
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        name, [&]() { return AiterAsmKernel(name, co_name); });

    // ---- launch ------------------------------------------------------------
    // Dispatch along max_q_len (DOUBLE_Q halves via tg_div); z = batch index.
    // The kernel early-exits q-tiles that fall beyond a batch's actual seqlen
    // using cu_seqlens_q.  Shipped VARLEN kernels: ts_qo=128, double_q=1
    // (tg_div=2), wv_tg=4 (block=128), remap_xy=1.
    const int sub_Q        = 128;   // ts_qo
    const int wv_tg        = 4;
    const int bdx          = (wv_tg == 4) ? 128 : 256;
    const int tg_div       = 1;     // double_q = 0
    const int q_tile_count = (max_seqlen_q + sub_Q - 1) / sub_Q;
    const int gdx          = (q_tile_count + tg_div - 1) / tg_div;
    const int gdy          = q_head_num;
    const int gdz          = batch;

    // remap_xy=1: swap gdx<->gdy at launch so bid.x indexes heads, bid.y Q-tiles.
    impl_ptr->launch_kernel({&args,
                             &arg_size,
                             gdy,   // launch_gdx = head count  (swapped)
                             gdx,   // launch_gdy = Q-tile count (swapped)
                             gdz,
                             bdx,
                             1,
                             1,
                             stream});
}

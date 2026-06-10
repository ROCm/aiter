// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Block-sparse FMHA forward (Sage i8fp8, hd=128, gfx950) C++ dispatcher.
//
// Mirrors aiter::fmha_fwd_v3 in mha_fwd.cu but routes to the hand-written
// sparse ASM kernel (fwd_hd128_i8fp8_sparse.co). The 704-byte kernarg blob is
// the dense 656-byte fmha_fwd_v3_args layout plus 3 trailing pointers
// (kv_block_indices, lut_start, lut_count) -- see
// /workspace/mi350_fmha_hd128_i8fp8_sparse.py docstring.

#include "mha_fwd_sparse.h"
#include "aiter_hip_common.h"
#include <memory>
#include <string>

namespace aiter {

// Hardcoded for the single shape this kernel currently supports.
// (BLOCK_M, BLOCK_N) = (256, 128); hd_q = hd_v = 128; non-causal.
// The mxfp4 sibling shares the SAME 704-byte kernarg layout (sparse args
// are only the trailing 48-byte LUT pointer block, identical for both).
static constexpr int      kSparseTileQ = 256;
static constexpr int      kSparseTileN = 128;
static constexpr int      kSparseBdx   = 512;
static constexpr const char* kSparseKernelName =
    "_ZN5aiter35fmha_fwd_hd128_i8fp8_sparse_gfx950E";
static constexpr const char* kSparseCoName =
    "fmha_v3_fwd/fwd_hd128_i8fp8_sparse.co";
static constexpr const char* kSparseMxfp4KernelName =
    "_ZN5aiter35fmha_fwd_hd128_mxfp4_sparse_gfx950E";
static constexpr const char* kSparseMxfp4CoName =
    "fmha_v3_fwd/fwd_hd128_mxfp4_sparse.co";
// fp8-quantized sibling (E4M3 Q/K/V). Same 704-byte kernarg layout and
// same in_bpe=1 byte stride as the i8fp8 path, so init_sparse_v3_args is
// reused unchanged; only the kernel symbol + .co name differ.
static constexpr const char* kSparseFp8KernelName =
    "_ZN5aiter32fmha_fwd_hd128_fp8_sparse_gfx950E";
static constexpr const char* kSparseFp8CoName =
    "fmha_v3_fwd/fwd_hd128_fp8_sparse.co";

// Pack the 704-byte blob. The first 656 bytes mirror init_fmha_fwd_v3_args
// (see mha_fwd.cu); the trailing 48 bytes hold the 3 LUT pointers (each 16
// bytes with p2 padding, matching the host struct in mha_fwd.h).
//
// We pack manually instead of calling init_fmha_fwd_v3_args to keep the
// sparse path self-contained and avoid mha_fwd.cu's static configs (cfg_fmha_fwd)
// which key on (dtype, hdim, mask, mode, ts_*). The values written here are
// the same ones init_fmha_fwd_v3_args would produce for {data_type="i8fp8bf16",
// is_group_mode=false, mask_type=0, ts_qo=256, in_bpe=1, out_bpe=2}.
static void init_sparse_v3_args(fmha_fwd_v3_sparse_args& args,
                                const mha_fwd_sparse_args& a)
{
    // ---- dense portion (matches mha_fwd.cu::init_fmha_fwd_v3_args path
    // for i8fp8bf16, batch mode, mask=0). ----
    constexpr int in_bpe  = 1;
    constexpr int out_bpe = 2;
    constexpr int ts_qo   = kSparseTileQ;

    args.ptr_o            = a.o_ptr;
    args.ptr_q            = a.q_ptr;
    args.ptr_k            = a.k_ptr;
    args.ptr_v            = a.v_ptr;
    args.ptr_lse          = nullptr;
    args.ptr_qseq         = nullptr;
    args.ptr_kseq         = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    args.ptr_q_descale    = a.q_descale_ptr;
    args.ptr_k_descale    = a.k_descale_ptr;
    args.ptr_v_descale    = a.v_descale_ptr;
    args.s_descale_q_Bs   = a.batch_stride_q_descale * 4;
    args.s_descale_q_Hs   = a.nhead_stride_q_descale * 4;
    args.s_descale_k_Bs   = a.batch_stride_k_descale * 4;
    args.s_descale_k_Hs   = a.nhead_stride_k_descale * 4;
    args.s_descale_v_Bs   = a.batch_stride_v_descale * 4;
    args.s_descale_v_Hs   = a.nhead_stride_v_descale * 4;

    args.scalar        = a.scale_s;
    args.s_seq_len     = a.seqlen_q;
    args.s_Seqs        = a.stride_q * in_bpe;
    args.s_Ts          = ts_qo * a.stride_q * in_bpe;
    args.s_Hs          = a.nhead_stride_q * in_bpe;
    args.s_Bs          = a.batch_stride_q * in_bpe;
    args.s_gqa         = a.nhead_q / a.nhead_k;
    args.s_k_Seqs      = a.stride_k * in_bpe;
    args.s_k_Hs        = a.nhead_stride_k * in_bpe;
    args.s_k_Bs        = a.batch_stride_k * in_bpe;
    args.s_opt         = 0; // tune_opt unused by the sparse kernel (no mask path)
    args.s_lse         = 0;
    args.s_kv_seq_len  = a.seqlen_k;
    args.s_qk_head_dim = a.hdim_q;
    args.s_v_head_dim  = a.hdim_v;
    args.s_q_head_num  = a.nhead_q;
    args.s_v_Seqs      = a.stride_v * in_bpe;
    args.s_v_Hs        = a.nhead_stride_v * in_bpe;
    args.s_v_Bs        = a.batch_stride_v * in_bpe;
    args.s_o_Seqs      = a.stride_o * out_bpe;
    args.s_o_Hs        = a.nhead_stride_o * out_bpe;
    args.s_o_Bs        = a.batch_stride_o * out_bpe;
    args.s_lse_Hs      = 0;

    // ---- sparse-specific tail ----
    args.ptr_kv_block_indices = a.kv_block_indices_ptr;
    args.ptr_lut_start        = a.lut_start_ptr;
    args.ptr_lut_count        = a.lut_count_ptr;
}

float fmha_fwd_v3_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_sparse: only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.data_type != "i8fp8bf16")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_sparse: only data_type=i8fp8bf16 is "
                          "supported (got " << a.data_type << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_v3_sparse: unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kSparseKernelName,
        [&]() { return AiterAsmKernel(kSparseKernelName, kSparseCoName); });

    fmha_fwd_v3_sparse_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a);

    // Grid: (num_q_blocks, nhead_q, batch). Same ordering the .py kernel
    // uses for tgid_x/y/z (see process_current_work_sparse()).
    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// Block-sparse mxfp4 fmha sibling. Same 704-byte kernarg blob as the
// i8fp8 sparse path; the only on-device difference is the kernel symbol
// + .co name. The mxfp4 kernel re-computes its own Q/K E8M0 per-block
// scale offsets from _s_KV_cur / _s_seq_len / _s_q_head_num so the
// init_sparse_v3_args path can be reused unchanged -- the kernel does
// NOT consume args.s_descale_*_Bs / _Hs for mxfp4 (only the base
// pointers q_descale_ptr / k_descale_ptr / v_descale_ptr matter, which
// init_sparse_v3_args sets from a.{q,k,v}_descale_ptr).
float fmha_fwd_v3_mxfp4_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_mxfp4_sparse: only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    // Accept any caller-tag; we keep the dtype tag opaque to the host so
    // the same dispatcher works whether the wrapper says "mxfp4fp8bf16",
    // "mxfp4bf16", etc. The validation that Q/K really are fp4-packed
    // happens in the torch entry (asm_mha_fwd_sparse.cu::fmha_v3_fwd_mxfp4_sparse).
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_v3_mxfp4_sparse: unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kSparseMxfp4KernelName,
        [&]() { return AiterAsmKernel(kSparseMxfp4KernelName, kSparseMxfp4CoName); });

    fmha_fwd_v3_sparse_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a);

    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

// Block-sparse fp8 fmha sibling (E4M3 Q/K and fp8 V, per-tensor fp32
// descales -- same descale contract as the i8fp8 path). Shares the
// identical 704-byte kernarg blob and in_bpe=1 byte stride as the i8fp8
// path, so init_sparse_v3_args is reused unchanged. The only on-device
// difference is the kernel symbol + .co name.
float fmha_fwd_v3_fp8_sparse(mha_fwd_sparse_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    const std::string arch_id = get_gpu_arch();
    if(arch_id != "gfx950")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_fp8_sparse: only gfx950 is supported "
                          "(detected arch: " << arch_id << ")");
        return -1;
    }
    if(a.data_type != "fp8bf16")
    {
        AITER_LOG_WARNING("fmha_fwd_v3_fp8_sparse: only data_type=fp8bf16 is "
                          "supported (got " << a.data_type << ")");
        return -1;
    }
    if(a.is_group_mode || a.mask_type != 0 || a.has_lse || a.p_drop > 0.f ||
       a.bias_type != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_v3_fp8_sparse: unsupported feature combination "
                          "(group/mask/lse/dropout/bias must all be off)");
        return -1;
    }

    if(a.v3_api_check)
    {
        return 1;
    }

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        kSparseFp8KernelName,
        [&]() { return AiterAsmKernel(kSparseFp8KernelName, kSparseFp8CoName); });

    fmha_fwd_v3_sparse_args args{};
    size_t arg_size = sizeof(args);
    init_sparse_v3_args(args, a);

    const int num_q_blocks = (a.seqlen_q + kSparseTileQ - 1) / kSparseTileQ;
    const int gdx = num_q_blocks;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = kSparseBdx;

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz,
                                 bdx, 1, 1, s_.stream_id_});
    });
}

} // namespace aiter

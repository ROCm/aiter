// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "unified_attention.hpp"

void unified_attention_fwd(
    torch::Tensor& output,          // [num_tokens, num_heads_q, head_size]
    torch::Tensor& query,           // [num_tokens, num_heads_q, head_size]
    torch::Tensor& key_cache,       // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& value_cache,     // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,        // [num_seqs]
    torch::Tensor& query_start_len, // [num_seqs + 1]
    int mask_type,
    float scale_s,
    float scale,
    float scale_k,
    float scale_v,
    float scale_out,
    bool cache_ptr_int32_overflow_possible,
    // KV-segment parallelism (FlashDecoding-style split-KV). The kernel
    // launches a 3D grid with z-dim = num_splits and writes each CTA's
    // partial (o_acc, lse) into the FP32 workspaces below; the host then
    // reduces across the split axis to produce the final output. When
    // num_splits == 1 the workspaces are ignored (pass None/empty).
    //   o_acc_workspace   : float32 [num_q_heads, num_splits, num_tokens, hdim]
    //   lse_acc_workspace : float32 [num_q_heads, num_splits, num_tokens]
    int num_splits,
    std::optional<torch::Tensor> o_acc_workspace,
    std::optional<torch::Tensor> lse_acc_workspace,
    float q_descale,
    float k_descale,
    float v_descale,
    // Optional caller-side override for max_seqlen_q. When 0 (default) we fall
    // back to the conservative heuristic at line below. Pass the real per-seq
    // max here to let the C++ dispatcher pick a tighter (smaller) BlockM tier
    // (e.g. decode_d128_m128 with 4 warps vs prefill_d128 with 8 warps).
    int64_t max_seqlen_q_override)
{
    auto dtype = query.dtype();
    // FP8 path: Q is FP8E4M3 (e4m3fn on gfx950 / e4m3fnuz on gfx942). The
    // output dtype is allowed to be bf16 or fp16 — we mirror the Triton
    // reference which writes bf16 output for FP8 inputs (and the CK fp8
    // problem traits also use bf16 for `o_dtype`). When q/k/v are not FP8
    // the descale floats default to 1.0f at the Python layer so this code
    // path is identical to the pre-FP8 behaviour.
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
                    dtype == torch::kFloat8_e4m3fn ||
                    dtype == torch::kFloat8_e4m3fnuz,
                "unified_attention supports fp16, bf16, fp8_e4m3fn, fp8_e4m3fnuz inputs");
    TORCH_CHECK(key_cache.dtype() == dtype && value_cache.dtype() == dtype,
                "unified_attention requires q/k/v to share dtype");
    TORCH_CHECK(block_tables.dtype() == torch::kInt32, "block_tables must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(query_start_len.dtype() == torch::kInt32, "query_start_len must be int32");

    ck_tile::unified_attention_args::data_type_enum dt;
    if (dtype == torch::kFloat16)
        dt = ck_tile::unified_attention_args::data_type_enum::fp16;
    else if (dtype == torch::kBFloat16)
        dt = ck_tile::unified_attention_args::data_type_enum::bf16;
    else
        dt = ck_tile::unified_attention_args::data_type_enum::fp8;

    ck_tile::index_t num_tokens   = query.size(0);
    ck_tile::index_t num_head_q   = query.size(1);
    ck_tile::index_t hdim         = query.size(2);
    ck_tile::index_t num_kv_heads = key_cache.size(2);
    ck_tile::index_t num_blks     = key_cache.size(0);
    ck_tile::index_t page_blk_size = key_cache.size(1);
    ck_tile::index_t num_seqs     = seq_lens.size(0);
    ck_tile::index_t num_queries_per_kv = num_head_q / num_kv_heads;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(query));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    ck_tile::unified_attention_args args;
    args.data_type          = dt;
    args.mask_type          = mask_type;
    args.num_tokens         = num_tokens;
    args.num_blks           = num_blks;
    args.num_head_q         = num_head_q;
    args.num_queries_per_kv = num_queries_per_kv;
    args.page_blk_size      = page_blk_size;
    args.hdim               = hdim;
    args.scale_s            = scale_s;
    args.scale              = scale;
    args.scale_k            = scale_k;
    args.scale_v            = scale_v;
    args.scale_out          = scale_out;
    args.q_descale          = q_descale;
    args.k_descale          = k_descale;
    args.v_descale          = v_descale;

    args.q_ptr          = query.data_ptr();
    args.query_stride_0 = query.stride(0);
    args.query_stride_1 = query.stride(1);

    args.k_ptr            = key_cache.data_ptr();
    args.stride_k_cache_0 = key_cache.stride(0);
    args.stride_k_cache_1 = key_cache.stride(1);
    args.stride_k_cache_2 = key_cache.stride(2);
    args.stride_k_cache_3 = key_cache.stride(3);

    args.v_ptr            = value_cache.data_ptr();
    args.stride_v_cache_0 = value_cache.stride(0);
    args.stride_v_cache_1 = value_cache.stride(1);
    args.stride_v_cache_2 = value_cache.stride(2);
    args.stride_v_cache_3 = value_cache.stride(3);

    args.o_ptr           = output.data_ptr();
    args.output_stride_0 = output.stride(0);
    args.output_stride_1 = output.stride(1);

    args.block_tables_ptr  = block_tables.data_ptr<int32_t>();
    args.block_table_stride = block_tables.stride(0);
    args.seq_lens_ptr       = seq_lens.data_ptr<int32_t>();
    args.query_start_len_ptr = query_start_len.data_ptr<int32_t>();
    args.num_seqs            = num_seqs;

    // Graph-capture-safe max_seqlen_q estimation (no GPU→CPU copy).
    //   - Caller-provided override (>0) wins — lets the wrapper pick a tighter
    //     BlockM tier (e.g. decode_d128_m128 4-warp instead of prefill_d128
    //     8-warp). Callers that can't supply this stay on the heuristic.
    //   - Pure decode: num_tokens == num_seqs → every seq has exactly 1 token.
    //   - Otherwise: use num_tokens as conservative upper bound (forces medium
    //     tier which handles any seqlen_q correctly via 1D grid w/ Q iter).
    if (max_seqlen_q_override > 0)
        args.max_seqlen_q = static_cast<ck_tile::index_t>(max_seqlen_q_override);
    else
        args.max_seqlen_q = (num_tokens == num_seqs) ? 1 : num_tokens;

    // Routes the K/V async-load path inside the CK pipeline. False (default)
    // → fast `buffer_load_dword_lds` with shared SRD (valid as long as the
    // cache fits in 4 GB). True → `global_load_lds` with per-lane 64-bit base
    // pointer (slower but lifts the 4 GB limit).
    args.cache_ptr_int32_overflow_possible = cache_ptr_int32_overflow_possible;

    // Wire up the split-KV workspace pointers/strides. For num_splits == 1 the
    // workspace tensors are ignored by the kernel (and the *_acc fields stay
    // at their no-split defaults). The split index is now derived from
    // blockIdx.z inside the kernel — the host does not pass `i_split`.
    args.num_splits = num_splits;
    if (num_splits > 1) {
        TORCH_CHECK(o_acc_workspace.has_value() && lse_acc_workspace.has_value(),
                    "unified_attention: num_splits>1 requires o_acc_workspace and "
                    "lse_acc_workspace tensors");
        auto& oacc = *o_acc_workspace;
        auto& lacc = *lse_acc_workspace;
        TORCH_CHECK(oacc.dtype() == torch::kFloat32 && lacc.dtype() == torch::kFloat32,
                    "unified_attention: split workspaces must be float32");
        TORCH_CHECK(oacc.dim() == 4 && lacc.dim() == 3,
                    "unified_attention: o_acc must be 4-D [nhead, splits, tokens, hdim], "
                    "lse_acc must be 3-D [nhead, splits, tokens]");
        args.o_acc_ptr             = oacc.data_ptr();
        args.lse_acc_ptr           = lacc.data_ptr();
        args.nhead_stride_o_acc    = oacc.stride(0);
        args.split_stride_o_acc    = oacc.stride(1);
        args.nhead_stride_lse_acc  = lacc.stride(0);
        args.split_stride_lse_acc  = lacc.stride(1);
    }

    auto [launched, elapsed] = ck_tile::unified_attention(args, {stream});
    TORCH_CHECK(launched,
                "unified_attention: no matching kernel for hdim=", hdim,
                " num_queries_per_kv=", num_queries_per_kv,
                " mask_type=", mask_type);
}

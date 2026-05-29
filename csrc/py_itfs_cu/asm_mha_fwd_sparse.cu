// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Block-sparse Sage i8fp8 FMHA forward (hd=128, gfx950).
// Top-level torch entry point. Mirrors csrc/py_itfs_cu/asm_mha_fwd.cu::fmha_v3_fwd
// but takes 3 extra LUT tensors and only dispatches to the sparse i8fp8 .co.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>

#include "py_itfs_common.h"
#include "mha_common.h"
#include "mha_fwd_sparse.h"

namespace aiter {
namespace torch_itfs {

static constexpr int ASM_SPARSE_BLOCK_M = 256; // kTileQ
static constexpr int ASM_SPARSE_BLOCK_N = 128; // kTileKV
static constexpr int ASM_SPARSE_HEAD_DIM = 128;

std::vector<at::Tensor>
fmha_v3_fwd_sparse(at::Tensor& q,
                   const at::Tensor& k,
                   const at::Tensor& v,
                   const at::Tensor& q_descale,
                   const at::Tensor& k_descale,
                   const at::Tensor& v_descale,
                   const at::Tensor& kv_block_indices,
                   const at::Tensor& lut_start,
                   const at::Tensor& lut_count,
                   float softmax_scale,
                   std::optional<at::Tensor> out_)
{
    // ---- dtype + device checks --------------------------------------------
    TORCH_CHECK(q.dtype() == at::ScalarType::Char && k.dtype() == at::ScalarType::Char,
                "fmha_v3_fwd_sparse: Q and K must be int8 (Sage i8fp8 path).");
    TORCH_CHECK(v.dtype() == at::ScalarType::Float8_e4m3fnuz ||
                    v.dtype() == at::ScalarType::Float8_e4m3fn,
                "fmha_v3_fwd_sparse: V must be fp8 (Sage i8fp8 path).");
    TORCH_CHECK(q_descale.dtype() == torch::kFloat32 &&
                    k_descale.dtype() == torch::kFloat32 &&
                    v_descale.dtype() == torch::kFloat32,
                "fmha_v3_fwd_sparse: descale tensors must be fp32.");
    TORCH_CHECK(kv_block_indices.dtype() == torch::kInt32 &&
                    lut_start.dtype() == torch::kInt32 &&
                    lut_count.dtype() == torch::kInt32,
                "fmha_v3_fwd_sparse: LUT tensors must be int32.");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(q_descale); CHECK_DEVICE(k_descale); CHECK_DEVICE(v_descale);
    CHECK_DEVICE(kv_block_indices); CHECK_DEVICE(lut_start); CHECK_DEVICE(lut_count);

    // ---- shape checks (bshd) ----------------------------------------------
    TORCH_CHECK(q.stride(-1) == 1, "Q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "K must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "V must have contiguous last dimension");

    const auto sz = q.sizes();
    const int batch_size = sz[0];
    const int seqlen_q = sz[1];
    const int num_heads = sz[2];
    const int head_size_q = sz[3];
    const int head_size_v = v.sizes()[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    TORCH_CHECK(head_size_q == ASM_SPARSE_HEAD_DIM && head_size_v == ASM_SPARSE_HEAD_DIM,
                "fmha_v3_fwd_sparse: only hd=", ASM_SPARSE_HEAD_DIM, " is supported "
                "(got Qd=", head_size_q, " Vd=", head_size_v, ").");
    // Arbitrary seqlen_q and seqlen_k are supported via:
    //   * Q/O buffer descriptor clamping (init_buffer_addresses adjusts
    //     num_records for the partial last Q tile).
    //   * K/V buffer descriptor clamping (num_records = kv_seq_len * stride)
    //     causes OOB loads to return 0; apply_mask in the kernel then sets
    //     the corresponding S entries to -inf so they contribute 0 mass to
    //     the softmax.
    // The LUT producer must size num_q_blocks = ceil(sq/256) and
    // num_kv_blocks = ceil(sk/128); the kernel iterates over the LUT.
    TORCH_CHECK(num_heads % num_heads_k == 0,
                "fmha_v3_fwd_sparse: HQ must be divisible by HK (got HQ=",
                num_heads, " HK=", num_heads_k, ").");
    const int gqa_ratio = num_heads / num_heads_k;
    TORCH_CHECK((gqa_ratio & (gqa_ratio - 1)) == 0,
                "fmha_v3_fwd_sparse: GQA ratio (HQ/HK) must be a power of 2 "
                "(got ", gqa_ratio, "). The ASM kernel uses a fixed log2 shift table.");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_q);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_q);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_v);

    // ---- LUT shape checks --------------------------------------------------
    // num_q_blocks = ceil(seqlen_q / BLOCK_M) -- must match what the bench's
    // build_block_mask / block_attn_mask_to_ragged_lut produced. The partial
    // last Q tile is handled by Q/O buffer-descriptor clamping in the kernel.
    const int num_q_blocks =
        (seqlen_q + ASM_SPARSE_BLOCK_M - 1) / ASM_SPARSE_BLOCK_M;
    const int64_t expected_lut_meta =
        static_cast<int64_t>(batch_size) * num_heads * num_q_blocks;
    TORCH_CHECK(lut_start.numel() == expected_lut_meta,
                "fmha_v3_fwd_sparse: lut_start.numel() = ", lut_start.numel(),
                ", expected ", expected_lut_meta,
                " (= batch*HQ*num_q_blocks).");
    TORCH_CHECK(lut_count.numel() == expected_lut_meta,
                "fmha_v3_fwd_sparse: lut_count.numel() = ", lut_count.numel(),
                ", expected ", expected_lut_meta, ".");

    // ---- output tensor -----------------------------------------------------
    auto opts = q.options();
    at::Tensor out;
    if (out_.has_value())
    {
        out = out_.value();
        TORCH_CHECK(out.dtype() == torch::kBFloat16,
                    "fmha_v3_fwd_sparse: out must be bf16 (i8fp8bf16 path).");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
    }
    else
    {
        out = torch::empty({batch_size, seqlen_q, num_heads, head_size_v},
                           opts.dtype(torch::kBFloat16));
    }

    // Otherwise the kernel will be launched from cuda:0 device.
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // ---- build mha_fwd_sparse_args (mostly mirrors get_asm_fmha_fwd_args) --
    mha_fwd_sparse_args a{};
    a.use_asm_v3       = true;
    a.v3_api_check     = false;
    a.how_v3_bf16_cvt  = 0;
    a.data_type        = "i8fp8bf16";
    a.is_group_mode    = false;
    a.bias_type        = 0;
    a.has_lse          = false;
    a.qscale_type      = 0;
    a.has_sink         = false;

    a.q_ptr            = q.data_ptr();
    a.k_ptr            = k.data_ptr();
    a.v_ptr            = v.data_ptr();
    a.bias_ptr         = nullptr;
    a.q_descale_ptr    = q_descale.data_ptr();
    a.k_descale_ptr    = k_descale.data_ptr();
    a.v_descale_ptr    = v_descale.data_ptr();
    a.rand_val_ptr     = nullptr;
    a.lse_ptr          = nullptr;
    a.o_ptr            = out.data_ptr();

    a.seqstart_q_ptr = nullptr;
    a.seqstart_k_ptr = nullptr;
    a.seqlen_q_ptr   = nullptr;
    a.seqlen_k_ptr   = nullptr;
    a.cu_seqlen_q_ptr = nullptr;
    a.cu_seqlen_k_ptr = nullptr;
    a.block_scale_seqstart_q_ptr = nullptr;
    a.block_scale_seqstart_k_ptr = nullptr;
    a.sink_ptr = nullptr;

    a.seqlen_q       = seqlen_q;
    a.seqlen_k       = seqlen_k;
    a.batch          = batch_size;
    a.max_seqlen_q   = seqlen_q;
    a.hdim_q         = head_size_q;
    a.hdim_v         = head_size_v;
    a.nhead_q        = num_heads;
    a.nhead_k        = num_heads_k;
    a.scale_s        = softmax_scale;
    a.logits_soft_cap = 0.0f;

    a.stride_q       = q.stride(1);
    a.stride_k       = k.stride(1);
    a.stride_v       = v.stride(1);
    a.stride_bias    = 0;
    a.stride_randval = 0;
    a.stride_o       = out.stride(1);
    a.nhead_stride_q = q.stride(2);
    a.nhead_stride_k = k.stride(2);
    a.nhead_stride_v = v.stride(2);
    a.nhead_stride_bias = 0;
    a.nhead_stride_randval = 0;
    a.nhead_stride_lse = 0;
    a.nhead_stride_o = out.stride(2);
    a.nhead_stride_q_descale = (q_descale.dim() == 2) ? q_descale.stride(1) : 0;
    a.nhead_stride_k_descale = (k_descale.dim() == 2) ? k_descale.stride(1) : 0;
    a.nhead_stride_v_descale = (v_descale.dim() == 2) ? v_descale.stride(1) : 0;
    a.batch_stride_q = q.stride(0);
    a.batch_stride_k = k.stride(0);
    a.batch_stride_v = v.stride(0);
    a.batch_stride_bias = 0;
    a.batch_stride_randval = 0;
    a.batch_stride_lse = 0;
    a.batch_stride_o = out.stride(0);
    a.batch_stride_q_descale = (q_descale.dim() == 2) ? q_descale.stride(0) : 0;
    a.batch_stride_k_descale = (k_descale.dim() == 2) ? k_descale.stride(0) : 0;
    a.batch_stride_v_descale = (v_descale.dim() == 2) ? v_descale.stride(0) : 0;

    a.window_size_left = -1;
    a.window_size_right = -1;
    a.sink_size = 0;
    a.mask_type = 0;
    a.min_seqlen_q = 0;
    a.p_drop = 0.0f;
    a.s_randval = false;
    a.drop_seed_offset =
        std::pair<const void*, const void*>{nullptr, nullptr};
    a.block_scale_size_q = 128;
    a.block_scale_size_kv = 128;

    // Sparse-specific fields:
    a.kv_block_indices_ptr = kv_block_indices.data_ptr();
    a.lut_start_ptr        = lut_start.data_ptr();
    a.lut_count_ptr        = lut_count.data_ptr();

    ck_tile::stream_config stream_config{stream};
    float t = aiter::fmha_fwd_v3_sparse(a, stream_config);
    TORCH_CHECK(t >= 0, "fmha_v3_fwd_sparse: dispatcher returned an error code "
                        "(unsupported config or .co not found).");
    return {out};
}

} // namespace torch_itfs
} // namespace aiter

// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>
#include <torch/all.h>

#include "aiter_stream.h"
#include "fmha_fwd_trek.hpp"

namespace {

constexpr int64_t kBlockSize = 128;

void check_bhsd_tensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be on a GPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous BHSD");
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, H, S, D]");
    TORCH_CHECK(
        tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16,
        name,
        " must have dtype float16 or bfloat16");
}

} // namespace

at::Tensor vsa_sparse_attention(const at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const at::Tensor& block_lut,
                                const at::Tensor& block_counts)
{
    check_bhsd_tensor(q, "q");
    check_bhsd_tensor(k, "k");
    check_bhsd_tensor(v, "v");

    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same GPU");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q, k, and v must have the same dtype");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0,
                "batch size and head counts must be positive");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, and v must have the same batch size");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    TORCH_CHECK(q.size(1) % k.size(1) == 0,
                "the number of query heads must be divisible by the number of KV heads");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128,
                "VSA sparse attention currently supports head dimension 128 only");
    TORCH_CHECK(q.size(2) > 0 && k.size(2) > kBlockSize,
                "query length must be positive and key length must exceed 128");

    TORCH_CHECK(block_lut.is_cuda() && block_counts.is_cuda(),
                "block_lut and block_counts must be on a GPU");
    TORCH_CHECK(block_lut.device() == q.device() && block_counts.device() == q.device(),
                "block_lut and block_counts must be on the same GPU as q");
    TORCH_CHECK(block_lut.scalar_type() == at::kInt &&
                    block_counts.scalar_type() == at::kInt,
                "block_lut and block_counts must have dtype int32");
    TORCH_CHECK(block_lut.is_contiguous() && block_counts.is_contiguous(),
                "block_lut and block_counts must be contiguous");

    const int64_t batch      = q.size(0);
    const int64_t nhead_q    = q.size(1);
    const int64_t nhead_k    = k.size(1);
    const int64_t seqlen_q   = q.size(2);
    const int64_t seqlen_k   = k.size(2);
    const int64_t q_blocks   = (seqlen_q + kBlockSize - 1) / kBlockSize;
    const int64_t kv_blocks  = (seqlen_k + kBlockSize - 1) / kBlockSize;

    TORCH_CHECK(
        block_lut.sizes() == at::IntArrayRef({batch, nhead_q, q_blocks, kv_blocks}),
        "block_lut must have shape [B, Hq, ceil(Sq/128), ceil(Sk/128)]");
    TORCH_CHECK(
        block_counts.sizes() == at::IntArrayRef({batch, nhead_q, q_blocks}),
        "block_counts must have shape [B, Hq, ceil(Sq/128)]");

    // The CK pipeline currently prefetches lut[i + 1] before checking whether
    // the current iteration is the last one. Reserve the final LUT slot as a
    // sentinel until that lookahead is guarded in CK.
    const int32_t min_count = block_counts.min().item<int32_t>();
    const int32_t max_count = block_counts.max().item<int32_t>();
    TORCH_CHECK(min_count >= 1, "every query block must select at least one KV block");
    TORCH_CHECK(max_count < kv_blocks,
                "block_counts must be smaller than the LUT row capacity; "
                "the final slot is reserved for CK's lookahead");

    auto out = torch::empty_like(q);

    const auto mask = mask_info::decode("0", seqlen_q, seqlen_k);
    fmha_vsa_fwd_traits traits{
        128,
        128,
        q.scalar_type() == at::kBFloat16 ? "bf16" : "fp16",
        true,
        mask.type};

    fmha_vsa_fwd_args args{
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        block_lut.data_ptr(),
        block_counts.data_ptr(),
        out.data_ptr(),
        static_cast<ck_tile::index_t>(seqlen_q),
        static_cast<ck_tile::index_t>(seqlen_k),
        static_cast<ck_tile::index_t>(batch),
        static_cast<ck_tile::index_t>(seqlen_q),
        128,
        128,
        static_cast<ck_tile::index_t>(nhead_q),
        static_cast<ck_tile::index_t>(nhead_k),
        1.0f / std::sqrt(128.0f),
        static_cast<ck_tile::index_t>(q.stride(2)),
        static_cast<ck_tile::index_t>(k.stride(2)),
        static_cast<ck_tile::index_t>(v.stride(2)),
        static_cast<ck_tile::index_t>(out.stride(2)),
        static_cast<ck_tile::index_t>(q.stride(1)),
        static_cast<ck_tile::index_t>(k.stride(1)),
        static_cast<ck_tile::index_t>(v.stride(1)),
        static_cast<ck_tile::index_t>(out.stride(1)),
        static_cast<ck_tile::index_t>(q.stride(0)),
        static_cast<ck_tile::index_t>(k.stride(0)),
        static_cast<ck_tile::index_t>(v.stride(0)),
        static_cast<ck_tile::index_t>(out.stride(0)),
        mask.left,
        mask.right,
        static_cast<ck_tile::index_t>(mask.type)};

    const ck_tile::stream_config stream_config{aiter::getCurrentHIPStream()};
    const float result = fmha_vsa_fwd(traits, args, stream_config);
    TORCH_CHECK(result >= 0, "no CK VSA kernel instance matched the input");
    return out;
}

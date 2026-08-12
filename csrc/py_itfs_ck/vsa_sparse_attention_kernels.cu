// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <cmath>

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "vsa_sparse_attention.h"
#include "fmha_fwd_trek.hpp"

namespace {

constexpr int64_t kBlockSize = 128;

void check_bhsd_tensor(const aiter_tensor_t& tensor, const char* name)
{
    AITER_CHECK(tensor.is_gpu(), name, " must be on a GPU");
    AITER_CHECK(tensor.is_contiguous(), name, " must be contiguous BHSD");
    AITER_CHECK(tensor.dim() == 4, name, " must have shape [B, H, S, D]");
    AITER_CHECK(
        tensor.dtype() == AITER_DTYPE_fp16 || tensor.dtype() == AITER_DTYPE_bf16,
        name,
        " must have dtype float16 or bfloat16");
}

bool same_shape(const aiter_tensor_t& lhs, const aiter_tensor_t& rhs)
{
    if(lhs.dim() != rhs.dim())
        return false;
    for(int index = 0; index < lhs.dim(); ++index)
        if(lhs.size(index) != rhs.size(index))
            return false;
    return true;
}

} // namespace

void vsa_sparse_attention_fwd(aiter_tensor_t& q,
                              aiter_tensor_t& k,
                              aiter_tensor_t& v,
                              aiter_tensor_t& block_lut,
                              aiter_tensor_t& block_counts,
                              aiter_tensor_t& out)
{
    check_bhsd_tensor(q, "q");
    check_bhsd_tensor(k, "k");
    check_bhsd_tensor(v, "v");
    check_bhsd_tensor(out, "out");

    AITER_CHECK(q.device_id == k.device_id && q.device_id == v.device_id &&
                    q.device_id == out.device_id,
                "q, k, and v must be on the same GPU");
    AITER_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype() &&
                    q.dtype() == out.dtype(),
                "q, k, v, and out must have the same dtype");
    AITER_CHECK(q.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0,
                "batch size and head counts must be positive");
    AITER_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "q, k, and v must have the same batch size");
    AITER_CHECK(same_shape(k, v), "k and v must have the same shape");
    AITER_CHECK(same_shape(q, out), "out must have the same shape as q");
    AITER_CHECK(q.size(1) % k.size(1) == 0,
                "the number of query heads must be divisible by the number of KV heads");
    AITER_CHECK(q.size(3) == 128 && k.size(3) == 128,
                "VSA sparse attention currently supports head dimension 128 only");
    AITER_CHECK(q.size(2) > 0 && k.size(2) > kBlockSize,
                "query length must be positive and key length must exceed 128");

    AITER_CHECK(block_lut.is_gpu() && block_counts.is_gpu(),
                "block_lut and block_counts must be on a GPU");
    AITER_CHECK(block_lut.device_id == q.device_id &&
                    block_counts.device_id == q.device_id,
                "block_lut and block_counts must be on the same GPU as q");
    AITER_CHECK(block_lut.dtype() == AITER_DTYPE_i32 &&
                    block_counts.dtype() == AITER_DTYPE_i32,
                "block_lut and block_counts must have dtype int32");
    AITER_CHECK(block_lut.is_contiguous() && block_counts.is_contiguous(),
                "block_lut and block_counts must be contiguous");

    const int64_t batch     = q.size(0);
    const int64_t nhead_q   = q.size(1);
    const int64_t nhead_k   = k.size(1);
    const int64_t seqlen_q  = q.size(2);
    const int64_t seqlen_k  = k.size(2);
    const int64_t q_blocks  = (seqlen_q + kBlockSize - 1) / kBlockSize;
    const int64_t kv_blocks = (seqlen_k + kBlockSize - 1) / kBlockSize;

    AITER_CHECK(
        block_lut.dim() == 4 && block_lut.size(0) == batch &&
            block_lut.size(1) == nhead_q && block_lut.size(2) == q_blocks &&
            block_lut.size(3) == kv_blocks,
        "block_lut must have shape [B, Hq, ceil(Sq/128), ceil(Sk/128)]");
    AITER_CHECK(
        block_counts.dim() == 3 && block_counts.size(0) == batch &&
            block_counts.size(1) == nhead_q && block_counts.size(2) == q_blocks,
        "block_counts must have shape [B, Hq, ceil(Sq/128)]");

    const auto mask = mask_info::decode("0", seqlen_q, seqlen_k);
    fmha_vsa_fwd_traits traits{
        128,
        128,
        q.dtype() == AITER_DTYPE_bf16 ? "bf16" : "fp16",
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

    HipDeviceGuard device_guard(q.device_id);
    const ck_tile::stream_config stream_config{aiter::getCurrentHIPStream()};
    const float result = fmha_vsa_fwd(traits, args, stream_config);
    AITER_CHECK(result >= 0, "no CK VSA kernel instance matched the input");
}

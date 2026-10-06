// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_enum.h"
#include "mla_reduce_cases.h"
#include <cstdint>
#include <string_view>

enum class MlaPlanner : int32_t
{
    V1_0 = 0,
    V1_1 = 1,
    V1_2 = 2,
};

enum class MlaHeadPlan : int32_t
{
    Native            = 0,
    Folded            = 1,
    AcceptedNonNative = 2,
    Unsupported       = 3,
};

struct MlaPlannerFlags
{
    bool enable_experimental;
    bool flydsl_ps1;
};

struct MlaDecodeHeadPlan
{
    MlaPlanner planner;
    MlaHeadPlan plan;
    bool natively_supported;
    int32_t qk_batch_ratio;
    int32_t kernel_num_heads;
    int32_t seqlen_fold;
    int32_t packed_qo_len_per_wg;
};

inline constexpr int32_t kMlaMetadataV10V11PackedQoLenPerWg = 128;

namespace aiter_mla_decode_shape_detail {

inline MlaHeadPlan
classify(const bool natively_supported, const int32_t qk_batch_ratio, const bool accepted)
{
    if(!accepted)
    {
        return MlaHeadPlan::Unsupported;
    }
    if(qk_batch_ratio != 1)
    {
        return MlaHeadPlan::Folded;
    }
    return natively_supported ? MlaHeadPlan::Native : MlaHeadPlan::AcceptedNonNative;
}

} // namespace aiter_mla_decode_shape_detail

inline MlaDecodeHeadPlan mla_decode_head_plan_v1_0(int32_t num_heads, const int32_t uni_seqlen_qo)
{
    const bool natively_supported = (num_heads == 16) || (num_heads == 128);
    int32_t qk_batch_ratio        = 1;
    int32_t seqlen_fold           = 1;

    // In the following cases, we use #head=16 to simulate cases which is not natively supported by
    // mla main kernel.
    if((num_heads != 16) &&
       (num_heads != 128) && // main kernel natively supports #head=16 or #head=128
       (num_heads % 16 == 0) && (num_heads < 128))
    {
        qk_batch_ratio = num_heads / 16;
        num_heads      = 16;
    }

    if(num_heads == 128)
    {
        seqlen_fold = uni_seqlen_qo;
    }

    const bool accepted = (num_heads == 16) || (num_heads == 128);

    return {MlaPlanner::V1_0,
            aiter_mla_decode_shape_detail::classify(natively_supported, qk_batch_ratio, accepted),
            natively_supported,
            qk_batch_ratio,
            num_heads,
            seqlen_fold,
            kMlaMetadataV10V11PackedQoLenPerWg};
}

inline MlaDecodeHeadPlan mla_decode_head_plan_v1_1(int32_t num_heads)
{
    const bool natively_supported = (num_heads == 16) || (num_heads == 128);
    int32_t qk_batch_ratio        = 1;

    // In the following cases, we use #head=16 to simulate cases which is not natively supported by
    // mla main kernel.
    if((num_heads != 16) &&
       (num_heads != 128) && // main kernel natively supports #head=16 or #head=128
       (num_heads % 16 == 0) && (num_heads < 128))
    {
        qk_batch_ratio = num_heads / 16;
        num_heads      = 16;
    }

    const bool accepted = (num_heads == 16) || (num_heads == 128);

    return {MlaPlanner::V1_1,
            aiter_mla_decode_shape_detail::classify(natively_supported, qk_batch_ratio, accepted),
            natively_supported,
            qk_batch_ratio,
            num_heads,
            1,
            kMlaMetadataV10V11PackedQoLenPerWg};
}

inline MlaDecodeHeadPlan mla_decode_head_plan_v1_2(const std::string_view arch_id,
                                                   int32_t num_heads,
                                                   const int32_t max_seqlen_qo,
                                                   const AiterDtype q_dtype,
                                                   const AiterDtype kv_dtype,
                                                   const MlaPlannerFlags flags)
{
    int32_t qk_batch_ratio = 1;

    // In the following cases, we use #head=16 to simulate cases which is not natively supported by
    // mla main kernel.
    const bool q_is_fp8  = (q_dtype == AITER_DTYPE_fp8);
    const bool kv_is_fp8 = (kv_dtype == AITER_DTYPE_fp8);

    const bool enable_experimental = flags.enable_experimental;

    // Gate on arch_id consistent with hk_mla_v32_decode_fwd dispatch (gfx942/gfx950).
    // Otherwise this would mark shapes as natively supported on archs where the
    // HK kernels are unavailable, producing metadata that downstream kernels
    // cannot consume.
    const bool hk_mtp_experimental =
        (arch_id == "gfx942" || arch_id == "gfx950") && (q_is_fp8 && kv_is_fp8) &&
        (num_heads * max_seqlen_qo == 128) &&
        ((num_heads == 16) || (num_heads == 32) || (num_heads == 64) || (num_heads == 128)) &&
        enable_experimental;

    // FlyDSL PS1 on gfx1250 consumes the full 32/64/128 Q heads in one work
    // item. Without this gate the planner folds those shapes to 16-head
    // pseudo-batches (qk_batch_ratio), which the FlyDSL kernel does not read.
    // Keep it behind AITER_MLA_DECODE_PS1_FLYDSL so gfx1250 persistent ASM
    // (16-head fold + host Q fold) is unchanged when FlyDSL is off.
    const bool flydsl_ps1 = flags.flydsl_ps1;
    const bool gfx1250_flydsl_ps1_heads =
        flydsl_ps1 && (arch_id == "gfx1250") && q_is_fp8 && kv_is_fp8 &&
        ((num_heads == 96) ||
         (((num_heads == 32) || (num_heads == 64) || (num_heads == 128)) && (max_seqlen_qo == 1)));

    const bool natively_supported =
        (num_heads == 16) || gfx1250_flydsl_ps1_heads ||
        ((arch_id == "gfx942" || arch_id == "gfx950") && (num_heads == 64) && q_is_fp8 &&
         kv_is_fp8 && (max_seqlen_qo == 1)) ||
        ((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8) ||
        ((arch_id == "gfx942") && (num_heads == 128) && q_is_fp8 && kv_is_fp8) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 &&
         ((num_heads == 32) || (num_heads == 64) || (num_heads == 128))) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 && (num_heads == 96) &&
         (max_seqlen_qo <= 6)) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 && (num_heads == 12) &&
         ((num_heads * max_seqlen_qo) <= 128)) ||
        hk_mtp_experimental;

    if(!natively_supported && (num_heads % 16 == 0))
    {
        qk_batch_ratio = num_heads / 16;
        num_heads      = 16;
    }

    const bool accepted = natively_supported || (num_heads == 16) || (num_heads == 128) ||
                          ((num_heads == 32) && q_is_fp8 && kv_is_fp8) ||
                          ((num_heads == 64) && q_is_fp8 && kv_is_fp8 && (max_seqlen_qo == 1)) ||
                          ((arch_id == "gfx950") && (num_heads == 8) && (max_seqlen_qo == 4) &&
                           q_is_fp8 && kv_is_fp8) ||
                          ((arch_id == "gfx942") && (num_heads == 8) && (max_seqlen_qo == 2) &&
                           !q_is_fp8 && !kv_is_fp8) ||
                          ((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8) ||
                          ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 &&
                           (((num_heads == 32) && (max_seqlen_qo == 4)) || (num_heads == 64) ||
                            (num_heads == 128))) ||
                          hk_mtp_experimental;

    int32_t kPackedQoLenPerWg = 128;
    if((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8 && (num_heads * max_seqlen_qo >= 64) &&
       (num_heads <= 64) && (((num_heads * max_seqlen_qo) < 128) || (num_heads == 48)))
    {
        kPackedQoLenPerWg = 64;
    }
    else if((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 && (num_heads == 32) &&
            (max_seqlen_qo == 3))
    {
        kPackedQoLenPerWg = 64;
    }

    return {MlaPlanner::V1_2,
            aiter_mla_decode_shape_detail::classify(natively_supported, qk_batch_ratio, accepted),
            natively_supported,
            qk_batch_ratio,
            num_heads,
            1,
            kPackedQoLenPerWg};
}

inline MlaDecodeHeadPlan mla_decode_head_plan(const std::string_view arch_id,
                                              const int32_t num_heads_k,
                                              const int32_t num_heads_per_head_k,
                                              const int32_t max_seqlen_qo,
                                              const int32_t uni_seqlen_qo,
                                              const AiterDtype q_nope_dtype,
                                              const AiterDtype kv_nope_dtype,
                                              const bool fast_mode,
                                              const bool intra_batch_mode,
                                              const MlaPlannerFlags flags)
{
    const int32_t num_heads = num_heads_k * num_heads_per_head_k;
    if(fast_mode)
    {
        return mla_decode_head_plan_v1_2(
            arch_id, num_heads, max_seqlen_qo, q_nope_dtype, kv_nope_dtype, flags);
    }
    else if(intra_batch_mode)
    {
        return mla_decode_head_plan_v1_0(num_heads, uni_seqlen_qo);
    }
    return mla_decode_head_plan_v1_1(num_heads);
}

enum class MlaBackendId : int32_t
{
    Asm       = 0,
    FlydslPs1 = 1,
    Opus      = 2,
    Hk        = 3,
    Ps1Fp8Asm = 4,
};

inline MlaBackendId mla_decode_backend(const std::string_view arch_id,
                                       const int32_t num_heads,
                                       const int32_t kernel_num_heads,
                                       const int32_t max_seqlen_qo,
                                       const AiterDtype q_dtype,
                                       const AiterDtype kv_dtype,
                                       const int32_t page_size,
                                       const int32_t cp_world_size,
                                       const bool cp_round_robin,
                                       const bool intra_batch_mode,
                                       const bool has_scales,
                                       const bool use_opus,
                                       const bool use_ps1_asm,
                                       const MlaPlannerFlags flags)
{
    const bool q_is_fp8   = (q_dtype == AITER_DTYPE_fp8);
    const bool kv_is_fp8  = (kv_dtype == AITER_DTYPE_fp8);
    const bool q_is_bf16  = (q_dtype == AITER_DTYPE_bf16);
    const bool kv_is_bf16 = (kv_dtype == AITER_DTYPE_bf16);

    const bool use_flydsl_ps1 =
        flags.flydsl_ps1 && (arch_id == "gfx1250") && (page_size == 1) && q_is_fp8 && kv_is_fp8 &&
        ((num_heads == 16) || (num_heads == 32) || (num_heads == 64) || (num_heads == 96) ||
         (num_heads == 128)) &&
        ((num_heads == 16) || (num_heads == 96) || (max_seqlen_qo == 1)) &&
        ((cp_world_size == 1) || cp_round_robin) && !intra_batch_mode && has_scales;
    if(use_flydsl_ps1)
    {
        // Head counts with code objects exported from the FlyDSL PS1 kernel
        // (hsa/gfx1250/mla_dsl/mla_dsl.csv) take them by default;
        // AITER_MLA_DECODE_PS1_ASM=0 keeps them on FlyDSL JIT.
        if(use_ps1_asm && ((num_heads == 96) || (num_heads == 128)))
        {
            return MlaBackendId::Ps1Fp8Asm;
        }
        return MlaBackendId::FlydslPs1;
    }

    const bool opus_is_fp8  = q_is_fp8 && kv_is_fp8 && has_scales;
    const bool opus_is_bf16 = q_is_bf16 && kv_is_bf16;
    if(use_opus && (arch_id == "gfx950") && (page_size == 1) && (opus_is_fp8 || opus_is_bf16))
    {
        return MlaBackendId::Opus;
    }

    const bool hk_page_size = (page_size == 1) || (page_size == 64);
    const bool use_hk       = ((arch_id == "gfx942" || arch_id == "gfx950") &&
                         (kernel_num_heads * max_seqlen_qo == 128) && q_is_fp8 && kv_is_fp8 &&
                         hk_page_size && flags.enable_experimental) ||
                        ((arch_id == "gfx950") && (kernel_num_heads * max_seqlen_qo == 64) &&
                         q_is_fp8 && kv_is_fp8 && hk_page_size && flags.enable_experimental);
    if(use_hk)
    {
        return MlaBackendId::Hk;
    }

    return MlaBackendId::Asm;
}

// HK MLA m16x4 kernel runs at occupancy=2 (gfx950 + 64 q-tokens per tile, gated on
// AITER_ENABLE_EXPERIMENTAL same as the dispatch in aiter/mla.py:use_hk). When it
// applies, the m16x4 launch site spawns 2*num_cu workgroups; the work distribution
// here must produce work_indptr sized to match so the second occupancy slot actually
// receives work. Detection mirrors hk_decode_fwd dispatch (num_heads * max_seqlen_qo
// == 64) and uses ORIGINAL num_heads/max_seqlen_qo (pre-fold). V32 uses fp8 across
// nope+rope; V40 uses fp8 nope + bf16 rope.
inline int32_t mla_metadata_cluster_multiplier(const std::string_view arch_id,
                                               const bool enable_experimental,
                                               const int32_t num_heads,
                                               const int32_t max_seqlen_qo,
                                               const MlaVersion mla_version,
                                               const AiterDtype q_nope_dtype,
                                               const AiterDtype q_rope_dtype,
                                               const AiterDtype kv_nope_dtype,
                                               const AiterDtype kv_rope_dtype)
{
    auto is_fp8  = [](const AiterDtype dtype) { return dtype == AITER_DTYPE_fp8; };
    auto is_bf16 = [](const AiterDtype dtype) { return dtype == AITER_DTYPE_bf16; };

    const bool dtype_ok =
        ((mla_version == MlaVersion::V32) && is_fp8(q_nope_dtype) && is_fp8(q_rope_dtype) &&
         is_fp8(kv_nope_dtype) && is_fp8(kv_rope_dtype)) ||
        ((mla_version == MlaVersion::V40) && is_fp8(q_nope_dtype) && is_bf16(q_rope_dtype) &&
         is_fp8(kv_nope_dtype) && is_bf16(kv_rope_dtype));

    const bool is_hk_m16x4 = enable_experimental && (arch_id == "gfx950") &&
                             (num_heads * max_seqlen_qo == 64) && dtype_ok;

    return is_hk_m16x4 ? 2 : 1;
}

#define AITER_MLA_REDUCE_SUPPORTS_X(NUM_HEAD_C, HEAD_DIM_C, NUM_HEAD, HEAD_DIM) \
    || (((NUM_HEAD) == (NUM_HEAD_C)) && ((HEAD_DIM) == (HEAD_DIM_C)))
inline bool mla_reduce_v1_supports(const int32_t num_heads, const int32_t head_dim)
{
    return false AITER_MLA_REDUCE_CASES(AITER_MLA_REDUCE_SUPPORTS_X, num_heads, head_dim);
}
#undef AITER_MLA_REDUCE_SUPPORTS_X

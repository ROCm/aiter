// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/fmha.hpp"

#include "mask.hpp"
#include "bias.hpp"

#include <type_traits>
#include <utility>
#include <variant>

struct FmhaFwdFp16
{
};

struct FmhaFwdBf16
{
};

struct FmhaFwdFp8
{
};

struct FmhaFwdBf8
{
};

struct FmhaFwdFp8Fp16
{
};

struct FmhaFwdFp8Bf16
{
};

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp16>
{
    using QDataType             = ck_tile::half_t;
    using KDataType             = ck_tile::half_t;
    using VDataType             = ck_tile::half_t;
    using BiasDataType          = ck_tile::half_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::half_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::half_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdBf16>
{
    using QDataType             = ck_tile::bf16_t;
    using KDataType             = ck_tile::bf16_t;
    using VDataType             = ck_tile::bf16_t;
    using BiasDataType          = ck_tile::bf16_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf16_t; // data type for A matrix of second gemm
    using OaccDataType          = float;           // data type for second gemm accumulation
    using ODataType             = ck_tile::bf16_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdFp8>
{
    using QDataType             = ck_tile::fp8_t;
    using KDataType             = ck_tile::fp8_t;
    using VDataType             = ck_tile::fp8_t;
    using BiasDataType          = float;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::fp8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::fp8_t;
};

template <>
struct FmhaFwdTypeConfig<FmhaFwdBf8>
{
    using QDataType             = ck_tile::bf8_t;
    using KDataType             = ck_tile::bf8_t;
    using VDataType             = ck_tile::bf8_t;
    using BiasDataType          = ck_tile::bf8_t;
    using RandValOutputDataType = uint8_t;
    using LSEDataType           = float; // data type for lse(logsumexp L_j = max_j + log(l_j))
    using SaccDataType          = float; // data type for first gemm accumulation
    using SMPLComputeDataType   = float; // data type for reduction, softmax
    using PDataType             = ck_tile::bf8_t; // data type for A matrix of second gemm
    using OaccDataType          = float;          // data type for second gemm accumulation
    using ODataType             = ck_tile::bf8_t;
};

struct FmhaMasks
{
    using NoMask      = ck_tile::GenericAttentionMask<false>;
    using GenericMask = ck_tile::GenericAttentionMask<true, true>;
    using CausalMask  = ck_tile::GenericAttentionMask<true, false>;
};

struct fmha_sparge_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* lut_ptr;
    const void* valid_block_num_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void*
        seqlen_k_ptr; // only used if both 'seqstart_q_ptr' & 'seqstart_k_ptr' are not nullptr

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float pv_threshold;
    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

template <typename FmhaKernel>
auto fmha_fwd_create_kargs_and_grids(fmha_sparge_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.lut_ptr,
                                             args.valid_block_num_ptr,
                                             args.bias_ptr,
                                             args.rand_val_ptr,
                                             args.lse_ptr,
                                             args.o_ptr,
                                             args.seqstart_q_ptr,
                                             args.seqstart_k_ptr,
                                             args.seqlen_k_ptr,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.pv_threshold,
                                             args.scale_s,
                                             args.scale_p,
                                             args.scale_o,
                                             args.logits_soft_cap,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_bias,
                                             args.stride_randval,
                                             args.stride_o,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             args.nhead_stride_bias,
                                             args.nhead_stride_randval,
                                             args.nhead_stride_lse,
                                             args.nhead_stride_o,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.mask_type,
                                             args.min_seqlen_q,
                                             args.p_drop,
                                             args.s_randval,
                                             args.drop_seed_offset);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                             args.k_ptr,
                                             args.v_ptr,
                                             args.lut_ptr,
                                             args.valid_block_num_ptr,
                                             args.bias_ptr,
                                             args.rand_val_ptr,
                                             args.lse_ptr,
                                             args.o_ptr,
                                             args.seqlen_q,
                                             args.seqlen_k,
                                             args.hdim_q,
                                             args.hdim_v,
                                             args.nhead_q,
                                             args.nhead_q / args.nhead_k,
                                             args.pv_threshold,
                                             args.scale_s,
                                             args.scale_p,
                                             args.scale_o,
                                             args.logits_soft_cap,
                                             args.stride_q,
                                             args.stride_k,
                                             args.stride_v,
                                             args.stride_bias,
                                             args.stride_randval,
                                             args.stride_o,
                                             args.nhead_stride_q,
                                             args.nhead_stride_k,
                                             args.nhead_stride_v,
                                             args.nhead_stride_bias,
                                             args.nhead_stride_randval,
                                             args.nhead_stride_lse,
                                             args.nhead_stride_o,
                                             args.batch_stride_q,
                                             args.batch_stride_k,
                                             args.batch_stride_v,
                                             args.batch_stride_bias,
                                             args.batch_stride_randval,
                                             args.batch_stride_lse,
                                             args.batch_stride_o,
                                             args.window_size_left,
                                             args.window_size_right,
                                             args.mask_type,
                                             args.p_drop,
                                             args.s_randval,
                                             args.drop_seed_offset);
        }
    }();

    if constexpr(FmhaKernel::kIsGroupMode)
    {
        dim3 grids = FmhaKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
        return ck_tile::make_tuple(kargs, grids);
    }
    else
    {
        dim3 grids =
            FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, false);
        return ck_tile::make_tuple(kargs, grids);
    }
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          bool kDoFp8StaticQuant_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kSkipMinSeqlenQ_ = false>
struct fmha_sparge_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kHasDropout                = kHasDropout_;
    static constexpr bool kDoFp8StaticQuant          = kDoFp8StaticQuant_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kUseTrLoad                 = kUseTrLoad_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
};

struct fmha_sparge_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool has_dropout;
    bool do_fp8_static_quant;
    bool skip_min_seqlen_q = false;
    // TODO: padding check is inside this api
};

float fmha_sparge_fwd(fmha_sparge_fwd_traits, fmha_sparge_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_sparge_fwd_(const ck_tile::stream_config&, fmha_sparge_fwd_args);

float fmha_sparge_fwd(fmha_sparge_fwd_args, const ck_tile::stream_config&);

// jenga
struct fmha_jenga_fwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* block_relation_onehot_ptr;
    const void* lut_ptr;
    const void* valid_block_num_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void*
        seqlen_k_ptr; // only used if both 'seqstart_q_ptr' & 'seqstart_k_ptr' are not nullptr

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;
    float scale_p;
    float scale_o;

    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

template <typename FmhaKernel, bool VSA=false>
auto fmha_fwd_create_kargs_and_grids(fmha_jenga_fwd_args args)
{
    assert(args.nhead_q % args.nhead_k == 0);
    auto kargs = [&] {
        if constexpr(VSA) {
            // create group mode kernel arguments
            if constexpr(FmhaKernel::kIsGroupMode)
            {
                return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                                args.k_ptr,
                                                args.v_ptr,
                                                args.lut_ptr,
                                                args.valid_block_num_ptr,
                                                args.bias_ptr,
                                                args.rand_val_ptr,
                                                args.lse_ptr,
                                                args.o_ptr,
                                                args.seqstart_q_ptr,
                                                args.seqstart_k_ptr,
                                                args.seqlen_k_ptr,
                                                args.hdim_q,
                                                args.hdim_v,
                                                args.nhead_q,
                                                args.nhead_q / args.nhead_k,
                                                args.scale_s,
                                                args.scale_p,
                                                args.scale_o,
                                                args.logits_soft_cap,
                                                args.stride_q,
                                                args.stride_k,
                                                args.stride_v,
                                                args.stride_bias,
                                                args.stride_randval,
                                                args.stride_o,
                                                args.nhead_stride_q,
                                                args.nhead_stride_k,
                                                args.nhead_stride_v,
                                                args.nhead_stride_bias,
                                                args.nhead_stride_randval,
                                                args.nhead_stride_lse,
                                                args.nhead_stride_o,
                                                args.window_size_left,
                                                args.window_size_right,
                                                args.mask_type,
                                                args.min_seqlen_q,
                                                args.p_drop,
                                                args.s_randval,
                                                args.drop_seed_offset);
            }
            else
            { // create batch mode kernel arguments
                return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                                args.k_ptr,
                                                args.v_ptr,
                                                args.lut_ptr,
                                                args.valid_block_num_ptr,
                                                args.bias_ptr,
                                                args.rand_val_ptr,
                                                args.lse_ptr,
                                                args.o_ptr,
                                                args.seqlen_q,
                                                args.seqlen_k,
                                                args.hdim_q,
                                                args.hdim_v,
                                                args.nhead_q,
                                                args.nhead_q / args.nhead_k,
                                                args.scale_s,
                                                args.scale_p,
                                                args.scale_o,
                                                args.logits_soft_cap,
                                                args.stride_q,
                                                args.stride_k,
                                                args.stride_v,
                                                args.stride_bias,
                                                args.stride_randval,
                                                args.stride_o,
                                                args.nhead_stride_q,
                                                args.nhead_stride_k,
                                                args.nhead_stride_v,
                                                args.nhead_stride_bias,
                                                args.nhead_stride_randval,
                                                args.nhead_stride_lse,
                                                args.nhead_stride_o,
                                                args.batch_stride_q,
                                                args.batch_stride_k,
                                                args.batch_stride_v,
                                                args.batch_stride_bias,
                                                args.batch_stride_randval,
                                                args.batch_stride_lse,
                                                args.batch_stride_o,
                                                args.window_size_left,
                                                args.window_size_right,
                                                args.mask_type,
                                                args.p_drop,
                                                args.s_randval,
                                                args.drop_seed_offset);
            }
        } else {
            // create group mode kernel arguments
            if constexpr(FmhaKernel::kIsGroupMode)
            {
                return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                                args.k_ptr,
                                                args.v_ptr,
                                                args.block_relation_onehot_ptr,
                                                args.bias_ptr,
                                                args.rand_val_ptr,
                                                args.lse_ptr,
                                                args.o_ptr,
                                                args.seqstart_q_ptr,
                                                args.seqstart_k_ptr,
                                                args.seqlen_k_ptr,
                                                args.hdim_q,
                                                args.hdim_v,
                                                args.nhead_q,
                                                args.nhead_q / args.nhead_k,
                                                args.scale_s,
                                                args.scale_p,
                                                args.scale_o,
                                                args.logits_soft_cap,
                                                args.stride_q,
                                                args.stride_k,
                                                args.stride_v,
                                                args.stride_bias,
                                                args.stride_randval,
                                                args.stride_o,
                                                args.nhead_stride_q,
                                                args.nhead_stride_k,
                                                args.nhead_stride_v,
                                                args.nhead_stride_bias,
                                                args.nhead_stride_randval,
                                                args.nhead_stride_lse,
                                                args.nhead_stride_o,
                                                args.window_size_left,
                                                args.window_size_right,
                                                args.mask_type,
                                                args.min_seqlen_q,
                                                args.p_drop,
                                                args.s_randval,
                                                args.drop_seed_offset);
            }
            else
            { // create batch mode kernel arguments
                return FmhaKernel::MakeKargsImpl(args.q_ptr,
                                                args.k_ptr,
                                                args.v_ptr,
                                                args.block_relation_onehot_ptr,
                                                args.bias_ptr,
                                                args.rand_val_ptr,
                                                args.lse_ptr,
                                                args.o_ptr,
                                                args.seqlen_q,
                                                args.seqlen_k,
                                                args.hdim_q,
                                                args.hdim_v,
                                                args.nhead_q,
                                                args.nhead_q / args.nhead_k,
                                                args.scale_s,
                                                args.scale_p,
                                                args.scale_o,
                                                args.logits_soft_cap,
                                                args.stride_q,
                                                args.stride_k,
                                                args.stride_v,
                                                args.stride_bias,
                                                args.stride_randval,
                                                args.stride_o,
                                                args.nhead_stride_q,
                                                args.nhead_stride_k,
                                                args.nhead_stride_v,
                                                args.nhead_stride_bias,
                                                args.nhead_stride_randval,
                                                args.nhead_stride_lse,
                                                args.nhead_stride_o,
                                                args.batch_stride_q,
                                                args.batch_stride_k,
                                                args.batch_stride_v,
                                                args.batch_stride_bias,
                                                args.batch_stride_randval,
                                                args.batch_stride_lse,
                                                args.batch_stride_o,
                                                args.window_size_left,
                                                args.window_size_right,
                                                args.mask_type,
                                                args.p_drop,
                                                args.s_randval,
                                                args.drop_seed_offset);
            }
        }
    }();

    if constexpr(FmhaKernel::kIsGroupMode)
    {
        dim3 grids = FmhaKernel::GridSize(
            args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, args.seqlen_k_ptr != nullptr);
        return ck_tile::make_tuple(kargs, grids);
    }
    else
    {
        dim3 grids =
            FmhaKernel::GridSize(args.batch, args.nhead_q, args.max_seqlen_q, args.hdim_v, false);
        return ck_tile::make_tuple(kargs, grids);
    }
}

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsGroupMode_,
          ck_tile::index_t kM0_,
          ck_tile::index_t kN0_,
          ck_tile::index_t kK0_,
          ck_tile::index_t kN1_,
          ck_tile::index_t kK1_,
          ck_tile::index_t kK0BlockLength_,
          bool kIsVLayoutRowMajor_,
          ck_tile::BlockFmhaPipelineEnum FmhaPipelineEnum_,
          bool kHasLogitsSoftCap_,
          typename FmhaMask_,
          ck_tile::BlockAttentionBiasEnum BiasEnum_,
          bool kStoreLse_,
          bool kHasDropout_,
          bool kDoFp8StaticQuant_,
          bool kPadS_,
          bool kPadSK_,
          bool kPadD_,
          bool kPadDv_,
          bool kUseTrLoad_,
          bool kSkipMinSeqlenQ_ = false>
struct fmha_jenga_fwd_traits_
{
    static constexpr ck_tile::index_t HDim           = HDim_;
    using DataType                                   = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsGroupMode               = kIsGroupMode_;
    static constexpr ck_tile::index_t kM0            = kM0_;
    static constexpr ck_tile::index_t kN0            = kN0_;
    static constexpr ck_tile::index_t kK0            = kK0_;
    static constexpr ck_tile::index_t kN1            = kN1_;
    static constexpr ck_tile::index_t kK1            = kK1_;
    static constexpr ck_tile::index_t kK0BlockLength = kK0BlockLength_;
    static constexpr bool kIsVLayoutRowMajor         = kIsVLayoutRowMajor_;
    static constexpr auto FmhaPipelineEnum           = FmhaPipelineEnum_;
    static constexpr bool kHasLogitsSoftCap          = kHasLogitsSoftCap_;
    using FmhaMask                                   = ck_tile::remove_cvref_t<FmhaMask_>;
    static constexpr auto BiasEnum                   = BiasEnum_;
    static constexpr bool kStoreLse                  = kStoreLse_;
    static constexpr bool kHasDropout                = kHasDropout_;
    static constexpr bool kDoFp8StaticQuant          = kDoFp8StaticQuant_;
    static constexpr bool kPadS                      = kPadS_;
    static constexpr bool kPadSK                     = kPadSK_;
    static constexpr bool kPadD                      = kPadD_;
    static constexpr bool kPadDv                     = kPadDv_;
    static constexpr bool kUseTrLoad                 = kUseTrLoad_;
    static constexpr bool kSkipMinSeqlenQ            = kSkipMinSeqlenQ_;
};

struct fmha_jenga_fwd_traits
{
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    bool is_v_rowmajor;
    bool has_logits_soft_cap;
    mask_enum mask_type;
    bias_enum bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    bool has_dropout;
    bool do_fp8_static_quant;
    bool skip_min_seqlen_q = false;
    // TODO: padding check is inside this api
};

float fmha_jenga_fwd(fmha_jenga_fwd_traits, fmha_jenga_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_jenga_fwd_(const ck_tile::stream_config&, fmha_jenga_fwd_args);

float fmha_jenga_fwd(fmha_jenga_fwd_args, const ck_tile::stream_config&);

float fmha_vsa_fwd(fmha_jenga_fwd_traits, fmha_jenga_fwd_args, const ck_tile::stream_config&);

template <typename Traits_>
float fmha_vsa_fwd_(const ck_tile::stream_config&, fmha_jenga_fwd_args);

float fmha_vsa_fwd(fmha_jenga_fwd_args, const ck_tile::stream_config&);
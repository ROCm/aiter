// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "wp_bias_pipeline_agmem_bgmem_creg_v2.hpp"
#include "gemm_bias_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm_bias_kernel.hpp"

#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3
#define CK_TILE_PIPELINE_COMPUTE_V5 4
#define CK_TILE_PIPELINE_PRESHUFFLE_V1 5
#define CK_TILE_PIPELINE_PRESHUFFLE_V2 6

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if defined(CK_GFX950_SUPPORT)
    constexpr bool is_8bit_float =
        std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
    if constexpr(M_Warp_Tile == 32)
        return is_8bit_float ? 64 : 16;
    else
        return is_8bit_float ? 128 : 32;
#else
    if constexpr(M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
}
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile_flatmm()
{
#if defined(CK_GFX950_SUPPORT)
    if constexpr(M_Warp_Tile == 32)
        return sizeof(PrecType) == 2 ? 16 : 64;
    else
        return sizeof(PrecType) == 2 ? 32 : 128;
#else
    if constexpr(M_Warp_Tile == 32)
        return sizeof(PrecType) == 2 ? 16 : 32;
    else
        return sizeof(PrecType) == 2 ? 32 : 64;
#endif
}

struct GemmConfigBase
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t Pipeline      = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
};

template <typename PrecType>
struct GemmConfigMemoryInterwave : public GemmConfigBase
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(PrecType) == 2 ? 8 : 16;

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_MEMORY;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct GemmConfigMemoryIntrawave : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(PrecType) == 2 ? 8 : 16;

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_MEMORY;
};

template <typename PrecType>
struct GemmConfigComputeV3 : public GemmConfigBase
{
    // Compute V3 only support Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigComputeV3_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigComputeV3_2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct GemmConfigComputeV4 : public GemmConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
};

template <typename PrecType>
struct GemmConfigComputeV4_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
};

template <typename PrecType>
struct GemmConfigComputeV5 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer               = false;
    static constexpr ck_tile::index_t Pipeline           = CK_TILE_PIPELINE_COMPUTE_V5;
    static constexpr ck_tile::index_t NumWaNumWaveGroups = 2;
};

template <typename PrecType>
struct GemmConfigPreshuffle_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile_flatmm<PrecType, M_Warp_Tile>();

    static constexpr int kBlockPerCu           = 2;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V1;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = false;
};

template <typename PrecType>
struct GemmConfigPreshuffle_2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr int kBlockPerCu           = 2;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
};

template <typename ADataType, typename BDataType = ADataType, typename CDataType = ADataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
};

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<int32_t>
{
    static constexpr const char* name = "int32";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::pk_int4_t>
{
    static constexpr const char* name = "pk_int4_t";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <ck_tile::index_t PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3_bias<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3_bias<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV5<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_PRESHUFFLE_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV1<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline =
        ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV1<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_PRESHUFFLE_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2_bias<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline =
        ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2_bias<PipelineProblem>;
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Column by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("persistent", "0", "0:non-persistent, 1:persistent")
        .insert("flush_cache", "true", "flush cache before running the kernel, defaults to true")
        .insert("rotating_count", "1", "rotating count, defaults to 1");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Type aliases for memory operation integral constants
using MemoryOpSet =
    std::integral_constant<ck_tile::memory_operation_enum, ck_tile::memory_operation_enum::set>;
using MemoryOpAtomicAdd = std::integral_constant<ck_tile::memory_operation_enum,
                                                 ck_tile::memory_operation_enum::atomic_add>;

// host API
template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          bool Persistent = false,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs_bias& args, const ck_tile::stream_config& s);

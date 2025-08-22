
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "gemm_pipeline_ag_bg_cr_comp_v3_block_quant.hpp"
#include "block_quant_gemm_kernel.hpp"

#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3
#define CK_TILE_PIPELINE_COMPUTE_V5 4

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif

#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrMem
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrMem
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Interwave
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrComp_block_quant
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrComp_block_quant
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V4)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV4
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV4
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V5)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV5
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV5
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave
#else
#error "unsupported CK_TILE_PIPELINE_DEFAULT value"
#endif


struct GemmConfig_fp8
{
 static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool DoubleSmemBuffer = false;
};
struct GemmConfig
{
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
#endif
#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
    // Compute friendly for Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V4)
    // Compute friendly for Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V5)
    // Compute friendly for Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer = false;

    // Available wavegroups will be split into `NumWaveGroups` and each of these wavegroups
    // will be responsible for specific jobs. For instance, perform Global Memory read operations,
    // perform block-gemm operation etc...
    static constexpr ck_tile::index_t NumWaveGroups = 2;
#endif

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
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
    using ScaleDataType = float;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
    using ScaleDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::bf16_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
    using ScaleDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    using ScaleDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    using ScaleDataType = float;
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
        .insert("stride_a_scale", "0", "Tensor a_scale stride")
        .insert("stride_b_scale", "0", "Tensor b_scale stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("persistent", "0", "0:non-persistent, 1:persistent");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// host API
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          bool Persistent = false>
float gemm_calc(const ck_tile::Block_quant_GemmHostArgs& args, const ck_tile::stream_config& s);

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "gemm_moe_ck2stages.h"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp"
#include <iostream>

template <
    typename A0DataType,
    typename B0DataType,
    typename AccDataType,
    typename EDataType,
    typename CDEElementOp,
    PipelineVersion PipelineVer,
    int BLOCKSIZE,
    int MPerBlock,
    int NPerBlock,
    int KPerBlock,
    int MWaves,
    int NWaves,
    bool Nswizzle,
    bool PerTensorQuant,
    bool MulRoutedWeight,
    int ActOP>
void ck_moe_stage1_gemm(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&hidden_states,     // [m, k], input token
                        void *&w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,  // [max_num_tokens_padded]
                        void *&sorted_expert_ids, // [max_num_m_blocks]
                        void *&sorted_weights,
                        void *&num_valid_ids,           // [1]
                        void *&out,                     // [max_num_tokens_padded, inter_dim]
                        std::optional<void *> w1_scale, // [e, 1, n], gate(up) scale
                        std::optional<void *> a1_scale  // [m, 1], token scale
)
{
    // ~~~~~~~~~~~~~~~~~~~~~~~~following start with ck things
    using A1DataType = F32;
    using B1DataType = F32;
    using CShuffleDataType = F32;
    using D2DataType = F32;
    using DsDataType = ck::Tuple<D2DataType>;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;
    ck::index_t KBatch = 1;

    using A0Layout = Row;
    using B0Layout = Col;
    using ELayout = Row;
    using D0Layout = Row;
    using D1Layout = Col;
    using D2Layout = ELayout;
    using DsLayout = ck::Tuple<D2Layout>;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs = std::array<ck::index_t, NumDTensor>{0};

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr ck::index_t MNPerXDL = 16;
    static constexpr ck::index_t WAVES = BLOCKSIZE / 64;
    static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * MWaves);
    static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * NWaves);
    // static constexpr ck::index_t NPerBlock = PipelineVer == ck::BlockGemmPipelineVersion::v1 ? 64 : 128;
    static constexpr ck::index_t CShuffleMXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : MXDLPerWave;
    static constexpr ck::index_t CShuffleNXDLPerWave = ck::is_same_v<B0DataType, I4> ? 1 : NXDLPerWave;
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = ck::is_same_v<B0DataType, I4> ? 32 : 16 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 16 / sizeof(EDataType);
    static constexpr ck::index_t K0_A = KPerBlock / AK1;
    static constexpr ck::index_t K0_B = KPerBlock / BK1;
    static constexpr ck::index_t K0_M_A = BLOCKSIZE / K0_A;
    static constexpr ck::index_t K0_N_B = BLOCKSIZE / K0_B;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = PerTensorQuant ? 1 : EVec;
    static constexpr ck::index_t D2Vec = 1;
    static constexpr ck::index_t Scale_Block_M = 1;
    static constexpr ck::index_t Scale_Block_N = 128;
    static constexpr ck::index_t Scale_Block_K = 128;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemmBlockScale
        // clang-format off
          <     Row,  Col,  DsLayout, ELayout, 
                A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
                AElementOp,  BElementOp, CDEElementOp,       GemmSpec,
                BLOCKSIZE,  Scale_Block_M, Scale_Block_N, Scale_Block_K,
                MPerBlock,   NPerBlock,    KPerBlock,
                AK1,   BK1,
                MNPerXDL,   MNPerXDL,
                4,    2,
                S<K0_A, K0_M_A, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, AK1, AK1, 0,
                S<K0_B, K0_N_B, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, BK1, BK1, 0,
                4,    2,   S<1, 32, 1, 8>, S<2, 1, 1, 1>,
                ck::BlockGemmPipelineScheduler::Intrawave, PipelineVer, ActOP, Nswizzle, true, MulRoutedWeight, int32_t, A0DataType>;

    // clang-format on

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    // do GEMM
    auto device_op = DeviceOpInstance{};
    const void *a1_scale_ptr = *a1_scale;
    const void *w1_scale_ptr = *w1_scale;

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               hidden_states,
                               w1,
                               std::array<const void *, NumDTensor>{MulRoutedWeight ? sorted_weights : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               StrideDs,
                               StrideE,
                               a1_scale_ptr,
                               w1_scale_ptr,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if (!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    invoker.Run(argument, StreamConfig{stream});
}

#define CK_MOE_STAGE1_GEMM_DEFINE(BLOCKSIZE, MPerfBlock, NPerBlock, KPerBlock, MWaves, NWaves, PipelineVer)                                                                                                                     \
    template void ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, BLOCKSIZE, MPerfBlock, NPerBlock, KPerBlock, MWaves, NWaves, Nswizzle, PerTensorQuant, MulRoutedWeight, ActOP>( \
        const hipStream_t &stream,                                                                                                                                                                                              \
        int tokens, int sorted_size, int N, int K,                                                                                                                                                                              \
        int topk,                                                                                                                                                                                                               \
        void *&hidden_states,                                                                                                                                                                                                   \
        void *&w1,                                                                                                                                                                                                              \
        void *&w2,                                                                                                                                                                                                              \
        void *&sorted_token_ids,                                                                                                                                                                                                \
        void *&sorted_expert_ids,                                                                                                                                                                                               \
        void *&sorted_weights,                                                                                                                                                                                                  \
        void *&num_valid_ids,                                                                                                                                                                                                   \
        void *&out,                                                                                                                                                                                                             \
        std::optional<void *> w1_scale,                                                                                                                                                                                         \
        std::optional<void *> a1_scale);

template <
    typename A0DataType,
    typename B0DataType,
    typename AccDataType,
    typename EDataType,
    typename CDEElementOp,
    PipelineVersion PipelineVer,
    int BLOCKSIZE,
    int MPerBlock,
    int NPerBlock,
    int KPerBlock,
    int MWaves,
    int NWaves,
    bool Nswizzle,
    bool PerTensorQuant,
    bool MulRoutedWeight,
    int ActOP = 0>
void ck_moe_stage2_gemm(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&inter_states,            // [max_num_tokens_padded, k], input token
                        void *&w1,                      // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                      // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,        // [max_num_tokens_padded]
                        void *&sorted_expert_ids,       // [max_num_m_blocks]
                        void *&sorted_weights,          // [max_num_tokens_padded]
                        void *&num_valid_ids,           //[1]
                        void *&out,                     // [m, out_dim]
                        std::optional<void *> w2_scale, // [e, 1, n], gate(up) scale
                        std::optional<void *> a2_scale  // [max_num_tokens_padded, 1], token scale
)
{
    // ~~~~~~~~~~~~~~~~~~~~~~~~following start with ck things
    using A1DataType = F32; // input scale
    using B1DataType = F32; // input scale
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;
    ck::index_t KBatch = 1;

    using CShuffleDataType = F32;
    using D2DataType = F32;
    using DsDataType = ck::Tuple<D2DataType>;

    using A0Layout = Row;
    using B0Layout = Col;
    using ELayout = Row;
    using D0Layout = Row;
    using D1Layout = Col;
    using D2Layout = ELayout;
    using DsLayout = ck::Tuple<D2Layout>;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto StrideDs = std::array<ck::index_t, NumDTensor>{0};

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

    // static constexpr ck::index_t BLOCKSIZE = 256;
    static constexpr ck::index_t WAVES = BLOCKSIZE / 64;
    static constexpr ck::index_t MNPerXDL = 16;
    static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * MWaves);
    static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * NWaves);
    static constexpr ck::index_t CShuffleMXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : MXDLPerWave;
    static constexpr ck::index_t CShuffleNXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : NXDLPerWave;
    static constexpr ck::index_t CShuffleNLane = ck::is_same_v<B0DataType, I4> ? 32 : NPerBlock / 2 / NXDLPerWave;
    static constexpr ck::index_t CShuffleMLane = BLOCKSIZE / CShuffleNLane;
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = ck::is_same_v<B0DataType, I4> ? 32 / sizeof(B0DataType) : 16 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 2;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = PerTensorQuant ? 1 : EVec;
    static constexpr ck::index_t D2Vec = 1;
    static constexpr ck::index_t K0_A = KPerBlock / AK1;
    static constexpr ck::index_t K0_B = KPerBlock / BK1;
    static constexpr ck::index_t K0_M = BLOCKSIZE / K0_A;
    static constexpr ck::index_t K0_N = BLOCKSIZE / K0_B;
    static constexpr ck::index_t Scale_Block_M = 1;
    static constexpr ck::index_t Scale_Block_N = 128;
    static constexpr ck::index_t Scale_Block_K = 128;
    using DeviceOpInstance = ck::tensor_operation::device::DeviceMoeGemmBlockScale
        // clang-format off
            < Row, Col, DsLayout, ELayout,
              A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
              AElementOp,  BElementOp, CDEElementOp,   GemmSpec,   
              256,  Scale_Block_M, Scale_Block_N, Scale_Block_K,
              MPerBlock,   128,    128,
              AK1,   BK1,
              MNPerXDL,   MNPerXDL,
              4, 2,
              S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
              S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
              2,    2,   S<1, 32, 1, 8>, S<2, 1, 1, 1>,
              ck::BlockGemmPipelineScheduler::Intrawave, PipelineVer, 0, false, false, MulRoutedWeight, int32_t, A0DataType>;



    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    // do GEMM
    auto device_op = DeviceOpInstance{};
    const void* a2_scale_ptr = *a2_scale; 
    const void* w2_scale_ptr = *w2_scale;  

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               inter_states,
                               w2,
                               std::array<const void *, NumDTensor>{MulRoutedWeight ? sorted_weights : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               StrideB,
                               StrideDs,
                               StrideE,
                               a2_scale_ptr,
                               w2_scale_ptr,
                               KBatch,
                               a_element_op,
                               b_element_op,
                               cde_element_op);

    if (!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }
    invoker.Run(argument, StreamConfig{stream});
}

#define CK_MOE_STAGE2_GEMM_DEFINE(BLOCKSIZE, MPerfBlock, NPerfBlock, KPerBlock, MWaves, NWaves, PipelineVer)                                                                                    \
    template void ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, BLOCKSIZE, MPerfBlock, NPerfBlock, KPerBlock, MWaves, NWaves, Nswizzle, PerTensorQuant, MulRoutedWeight, ActOP>( \
        const hipStream_t &stream,                                                                                                                                   \
        int tokens, int sorted_size, int N, int K,                                                                                                                   \
        int topk,                                                                                                                                                    \
        void *&inter_states,                                                                                                                                         \
        void *&w1,                                                                                                                                                   \
        void *&w2,                                                                                                                                                   \
        void *&sorted_token_ids,                                                                                                                                     \
        void *&sorted_expert_ids,                                                                                                                                    \
        void *&sorted_weights,                                                                                                                                       \
        void *&num_valid_ids,                                                                                                                                        \
        void *&out,                                                                                                                                                  \
        std::optional<void *> w2_scale,                                                                                                                              \
        std::optional<void *> a2_scale);
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "moe_ck_gemm.hpp"
#include <iostream>

// void preShuffleBuffer(const F4* src, F4* dst, int N, int K, int NXdl)
// {
//     int KPack = 32;
//     int NLane = NXdl;
//     int KLane = 64 / NLane;

//     int K0 = K / (KLane * KPack);
//     // K -> K0 KLane KPack
//     // N -> N0 NLane
//     // N, K -> N0 K0 KLane NLane KPack
//     int tempk;
//     for(int n = 0; n < N; ++n)
//     {
//         for(int k = 0; k < K; ++k)
//         {
//             int n0 = n / NLane;
//             int n1 = n % NLane;

//             int k0 = k / (KLane * KPack);
//             tempk  = k % (KLane * KPack);
//             int k1 = tempk / KPack;
//             int k2 = tempk % KPack;

//             int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
//                               k1 * KPack * NLane + n1 * KPack + k2;

//             dst[outputIndex / 2] = src[(n * K + k) / 2];
//         }
//     }
// }
template <
    typename A0DataType, 
    typename A1DataType, 
    typename B0DataType, 
    typename B1DataType, 
    typename AccDataType, 
    typename EDataType, 
    typename CDEElementOp, 
    PipelineVersion PipelineVer,
    int MPerBlock, 
    int KPerBlock, 
    int MWaves, 
    int NWaves, 
    bool Nswizzle, 
    bool PerTensorQuant, 
    bool MulRoutedWeight, 
    int ActOP
    >
void ck_moe_stage1_gemm_mxfp4(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&hidden_states,           // [m, k], input token
                        void *&w1,                      // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                      // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,        // [max_num_tokens_padded]
                        void *&sorted_expert_ids,       // [max_num_m_blocks]
                        void *&sorted_weights,
                        void *&num_valid_ids,           //[1]
                        void *&out,                     // [max_num_tokens_padded, inter_dim]
                        std::optional<void *> w1_scale, // [e, 1, n], gate(up) scale
                        std::optional<void *> a1_scale  // [m, 1], token scale
)
{
    // ~~~~~~~~~~~~~~~~~~~~~~~~following start with ck things
    static constexpr ck::index_t ScaleBlockSize  = 32; // scaling block size
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;
    ck::index_t Scale_Stride_AM      = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    ck::index_t Scale_Stride_BN      = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    ck::index_t KBatch = 1;
    // using AccDataType = F32;
    using CShuffleDataType = F32;
    using DsDataType = ck::Tuple<F32, F32, F32>;

    using A0Layout = Row;
    using B0Layout = Col;
    using D0Layout = Row;
    using D1Layout = Col;
    using ELayout  = Row;
    using D2Layout = ELayout;
    using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    // using CDEElementOp = MultiplyMultiply;

    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
    // static constexpr ck::index_t MPerBlock = 128;
    static constexpr ck::index_t MNPerXDL = 16;
    static constexpr ck::index_t BLOCKSIZE = 256;
    static constexpr ck::index_t NPerBlock = PipelineVer == ck::BlockGemmPipelineVersion::v1 ? 128 : 256;
    static constexpr ck::index_t WAVES = BLOCKSIZE / 64;
    // static constexpr ck::index_t MWaves = 1;
    // static constexpr ck::index_t NWaves = WAVES / MWaves;
    static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * MWaves);
    static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * NWaves);
    static constexpr ck::index_t CShuffleMXDLPerWave = MXDLPerWave;
    static constexpr ck::index_t CShuffleNXDLPerWave = NXDLPerWave;
    // static constexpr ck::index_t KPerBlock = ck::is_same_v<B0DataType, I4> ? 128 : 256 / sizeof(A0DataType);
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = 32 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 16 / sizeof(EDataType);
    static constexpr ck::index_t K0_A = KPerBlock / AK1;
    static constexpr ck::index_t K0_B = KPerBlock / BK1;
    static constexpr ck::index_t K0_M_A = BLOCKSIZE / K0_A;
    static constexpr ck::index_t K0_N_B = BLOCKSIZE / K0_B;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = PerTensorQuant ? 1 : EVec;
    static constexpr ck::index_t D2Vec = 1;

    // preShuffleBuffer((cosntF4*)w1_scale, w1, 4096, 6144, 16);

    // std::cout << "ck_preshuffle" << std::endl;
    // for (int i = 0; i < 128; i++) {
    //     std::cout << (int)((uint8_t*)w1)[i] << ", ";
    // }
    
    using DeviceOpInstance                     = ck::tensor_operation::device::DeviceMoeGemmMXBNS<      
        A0Layout,    B0Layout,    DsLayout,    ELayout, 
        A0DataType,  A1DataType,  B0DataType,  B1DataType,  DsDataType, EDataType, AccDataType, CShuffleDataType,
        AElementOp,  BElementOp, CDEElementOp, GemmSpec,   
        ScaleBlockSize, 256,   
        64,      128,    128,
        16,   16,
        16,   16,
        4,     2,
        S<8, 32, 1>, S<1, 0, 2>,     S<1, 0, 2>,    2, 16, 16, 0,
        S<8, 32, 1>, S<1, 0, 2>,     S<1, 0, 2>,    2, 16, 16, 0,
        2,    2,     S<1, 32, 1, 8>, S<8, 1, 1, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 
        ActOP, Nswizzle, true, MulRoutedWeight, ck::index_t, A0DataType>;// clang-format on

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    constexpr auto I0 = ck::Number<0>{};
    constexpr auto I1 = ck::Number<1>{};
    static constexpr auto DStride = PerTensorQuant ? I0 : I1;

    // Tensor<A0DataType> a0_t_k(HostTensorDescriptor({1024, K}, {K, 1}));
    // Tensor<A1DataType> a1_t_k(HostTensorDescriptor(
    //     {1024, (K + ScaleBlockSize - 1) / ScaleBlockSize}, {Scale_Stride_AM, 1}));
    // Tensor<B0DataType> b0_e_n_k(HostTensorDescriptor({8, K, N * 2}, {N * 2 * K, 1, K}));
    // Tensor<B1DataType> b1_e_n_k(
    //     HostTensorDescriptor({8, (K + ScaleBlockSize - 1) / ScaleBlockSize, N * 2},
    //                          {(N * 2 * Scale_Stride_BN), 1, Scale_Stride_BN}));
    
    // a0_t_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{6, 6});
    // b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{6, 6});
    // a1_t_k.GenerateTensorValue(GeneratorTensor_3<A1DataType>{125, 125});
    // b1_e_n_k.GenerateTensorValue(GeneratorTensor_3<B1DataType>{125, 125});

    // DeviceMem a0_device_buf(sizeof(A0DataType) * a0_t_k.mDesc.GetElementSpaceSize() / 2);
    // DeviceMem a1_device_buf(sizeof(A1DataType) * a1_t_k.mDesc.GetElementSpaceSize());
    // DeviceMem b0_device_buf(sizeof(B0DataType) * b0_e_n_k.mDesc.GetElementSpaceSize() / 2);
    // DeviceMem b1_device_buf(sizeof(B1DataType) * b1_e_n_k.mDesc.GetElementSpaceSize());

    // a0_device_buf.ToDevice(a0_t_k.mData.data());
    // a1_device_buf.ToDevice(a1_t_k.mData.data());
    // b0_device_buf.ToDevice(b0_e_n_k.mData.data);
    // b1_device_buf.ToDevice(b1_e_n_k.mData.data());
    // // do GEMM
    auto device_op = DeviceOpInstance{};

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               hidden_states,
                               a1_scale.value(),
                               w1,
                               w1_scale.value(),
                               std::array<const void *, NumDTensor>{nullptr,
                                                                    nullptr,
                                                                    MulRoutedWeight ? sorted_weights : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               Scale_Stride_AM,
                               StrideB,
                               Scale_Stride_BN,
                               std::array<ck::index_t, NumDTensor>{I0, I0, I0},
                               StrideE,
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

#define CK_MOE_STAGE1_GEMM_MXFP4_DEFINE(MPerfBlock, KPerBlock, MWaves, NWaves, PipelineVer, MulRoutedWeight, ActOP)                                                                                                                            \
    template void ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, MPerfBlock, KPerBlock, MWaves, NWaves, Nswizzle, PerTensorQuant, MulRoutedWeight, ActOP>( \
        const hipStream_t &stream,                                                                                                                                          \
        int tokens, int sorted_size, int N, int K,                                                                                                                          \
        int topk,                                                                                                                                                           \
        void *&hidden_states,                                                                                                                                               \
        void *&w1,                                                                                                                                                          \
        void *&w2,                                                                                                                                                          \
        void *&sorted_token_ids,                                                                                                                                            \
        void *&sorted_expert_ids,                                                                                                                                           \
        void *&sorted_weights,                                                                                                                                              \
        void *&num_valid_ids,                                                                                                                                               \
        void *&out,                                                                                                                                                         \
        std::optional<void *> w1_scale,                                                                                                                                     \
        std::optional<void *> a1_scale);

template <
    typename A0DataType, 
    typename A1DataType, 
    typename B0DataType, 
    typename B1DataType, 
    typename AccDataType, 
    typename EDataType, 
    typename CDEElementOp, 
    PipelineVersion PipelineVer,
    int MPerBlock, 
    int KPerBlock, 
    int MWaves, 
    int NWaves, 
    bool Nswizzle, 
    bool PerTensorQuant, 
    bool MulRoutedWeight
>
void ck_moe_stage2_gemm_mxfp4(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
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
    static constexpr ck::index_t ScaleBlockSize  = 32; // scaling block size
    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideD = 0;
    ck::index_t StrideE = N;
    ck::index_t Scale_Stride_AM      = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    ck::index_t Scale_Stride_BN      = (K + ScaleBlockSize - 1) / ScaleBlockSize;
    ck::index_t KBatch = 1;

    printf("%dx%dx%d", tokens, N, K);
    // using AccDataType = F32;
    using CShuffleDataType = F32;
    using DsDataType = ck::Tuple<F32, F32, F32>;

    using A0Layout = Row;
    using B0Layout = Col;
    using ELayout = Row;
    using D0Layout = Row;
    using D1Layout = Col;
    using DsLayout = ck::Tuple<D0Layout, D1Layout, ELayout>;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    using AElementOp = PassThrough;
    using BElementOp = PassThrough;
    // using CDEElementOp = MultiplyMultiply;
    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
    static constexpr ck::index_t BLOCKSIZE = 256;
    static constexpr ck::index_t WAVES = BLOCKSIZE / 64;
    static constexpr ck::index_t NPerBlock = 128;
    static constexpr ck::index_t MNPerXDL = 16;
    // static constexpr ck::index_t MWaves = 1;
    // static constexpr ck::index_t NWaves = WAVES / MWaves;
    static constexpr ck::index_t MXDLPerWave = MPerBlock / (MNPerXDL * MWaves);
    static constexpr ck::index_t NXDLPerWave = NPerBlock / (MNPerXDL * NWaves);
    // static constexpr ck::index_t KPerBlock = ck::is_same_v<B0DataType, I4> ? 128 : 256 / sizeof(A0DataType);
    static constexpr ck::index_t CShuffleMXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : MXDLPerWave;
    static constexpr ck::index_t CShuffleNXDLPerWave = ck::is_same_v<B0DataType, I4> ? 2 : NXDLPerWave;
    static constexpr ck::index_t CShuffleNLane = ck::is_same_v<B0DataType, I4> ? 32 : NPerBlock / 2 / NXDLPerWave; // 64
    static constexpr ck::index_t CShuffleMLane = BLOCKSIZE / CShuffleNLane;
    static constexpr ck::index_t AK1 = 16 / sizeof(A0DataType);
    static constexpr ck::index_t BK1 = 32 / sizeof(B0DataType);
    static constexpr ck::index_t EVec = 2;
    static constexpr ck::index_t D0Vec = 1;
    static constexpr ck::index_t D1Vec = PerTensorQuant ? 1 : EVec;
    static constexpr ck::index_t D2Vec = 1;
    static constexpr ck::index_t K0_A = KPerBlock / AK1;
    static constexpr ck::index_t K0_B = KPerBlock / BK1;
    static constexpr ck::index_t K0_M = BLOCKSIZE / K0_A;
    static constexpr ck::index_t K0_N = BLOCKSIZE / K0_B;
// clang-format off
using DeviceOpInstance                     = ck::tensor_operation::device::DeviceMoeGemmMXBNS<      
    A0Layout,    B0Layout,    DsLayout,    ELayout, 
    A0DataType,  A1DataType,  B0DataType,  B1DataType,  DsDataType, EDataType, AccDataType, CShuffleDataType,
    AElementOp,  BElementOp, CDEElementOp, GemmSpec,   
    ScaleBlockSize,      256,   
    128,   128,    128,
    16,   16,
    16,   16,
    8,    2,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
    2,    2,   S<1, 32, 1, 8>, S<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, 0, false, false, MulRoutedWeight, ck::index_t, A0DataType>;
// clang-format on

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    constexpr auto I0 = ck::Number<0>{};
    constexpr auto I1 = ck::Number<1>{};
    static constexpr auto DStride = PerTensorQuant ? I0 : I1;

    // do GEMM
    auto device_op = DeviceOpInstance{};

    auto invoker = device_op.MakeInvoker();
    auto argument =
        device_op.MakeArgument(sorted_token_ids,
                               sorted_expert_ids,
                               num_valid_ids,
                               inter_states,
                               a2_scale.value(),
                               w2,
                               w2_scale.value(),
                               std::array<const void *, NumDTensor>{nullptr,
                                                                    nullptr,
                                                                    MulRoutedWeight ? sorted_weights : nullptr},
                               out,
                               tokens,
                               topk,
                               sorted_size,
                               N,
                               K,
                               StrideA,
                               Scale_Stride_AM,
                               StrideB,
                               Scale_Stride_BN,
                               std::array<ck::index_t, NumDTensor>{DStride, DStride, I0},
                               StrideE,
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

#define CK_MOE_STAGE2_GEMM_MXFP4_DEFINE(MPerfBlock, KPerBlock, MWaves, NWaves, PipelineVer, MulRoutedWeight)                                                                                    \
    template void ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, MPerfBlock, KPerBlock, MWaves, NWaves, Nswizzle, PerTensorQuant, MulRoutedWeight>( \
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
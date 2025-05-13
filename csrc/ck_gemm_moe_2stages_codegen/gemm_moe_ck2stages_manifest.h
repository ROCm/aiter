// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/check_err.hpp"
#include <torch/torch.h>
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include <hip/hip_runtime.h>

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
                        void *&hidden_states,                          // [m, k], input token
                        void *&w1,                                     // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                                     // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,                       // [max_num_tokens_padded]
                        void *&sorted_expert_ids,                      // [max_num_m_blocks]
                        void *&sorted_weights,                         // null for stage1
                        void *&num_valid_ids,                          // [1]
                        void *&out,                                    // [max_num_tokens_padded, inter_dim]
                        std::optional<void *> w1_scale = std::nullopt, // [e, 1, n], gate(up) scale
                        std::optional<void *> a1_scale = std::nullopt  // [m, 1], token scale
);

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
    int MNPerXDL,
    bool Nswizzle,
    bool PerTensorQuant,
    bool MulRoutedWeight>
void ck_moe_stage2_gemm(const hipStream_t &stream, int tokens, int sorted_size, int N, int K,
                        int topk,
                        void *&inter_states,                           // [max_num_tokens_padded, k], input token
                        void *&w1,                                     // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                        void *&w2,                                     // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
                        void *&sorted_token_ids,                       // [max_num_tokens_padded]
                        void *&sorted_expert_ids,                      // [max_num_m_blocks]
                        void *&sorted_weights,                         // [max_num_tokens_padded]
                        void *&num_valid_ids,                          //[1]
                        void *&out,                                    // [m, out_dim]
                        std::optional<void *> w2_scale = std::nullopt, // [e, 1, n], gate(up) scale
                        std::optional<void *> a2_scale = std::nullopt  // [max_num_tokens_padded, 1], token scale
);        

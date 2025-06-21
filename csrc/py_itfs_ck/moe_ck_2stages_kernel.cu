// // SPDX-License-Identifier: MIT
// // Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// #include <torch/all.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include "py_itfs_common.h"
// #include "moe_ck_gemm.hpp"

// #define CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, Nswizzle, isPerTensorQuant, MPerBlock, MulRoutedWeight, ActOP) \
//     if (ActOP == 0)                                                                                               \
//     {                                                                                                           \
//         if (isPerTensorQuant)                                                                                               \
//         {                                                                                                                   \
//             if (MPerBlock == 32)                                                                                            \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                    \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 64)                                                                                       \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                    \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 128)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }                                                                                                                   \
//         else                                                                                                                \
//         {                                                                                                                   \
//             if (MPerBlock == 32)                                                                                            \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                    \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 64)                                                                                       \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                   \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 128)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }\
//     }\
//     else if (ActOP == 1)                                                                                       \
//     {\
//         if (isPerTensorQuant)                                                                                               \
//         {                                                                                                                   \
//             if (MPerBlock == 32)                                                                                            \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                    \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 64)                                                                                       \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                  \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 128)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }                                                                                                                   \
//         else                                                                                                                \
//         {                                                                                                                   \
//             if (MPerBlock == 32)                                                                                            \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                    \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 64)                                                                                       \
//             {                                                                                                               \
//                 if (K % (256 / sizeof(A0DataType)) == 0)                                                                  \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//                 else                                                                                                        \
//                 {                                                                                                           \
//                     ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//                 }                                                                                                           \
//             }                                                                                                               \
//             else if (MPerBlock == 128)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }\
//     }

// #define CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, Nswizzle, isPerTensorQuant, MPerBlock, MulRoutedWeight, ActOP)                                                                            \
//     if (ActOP == 0)     \
//     {   \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//         }   \
//     }\
//     else if (ActOP == 1)                                                                                       \
//     {   \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//         }   \
//     }                                                                                                                                                                                                                                                                                                                                                                  \


// #define CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, Nswizzle, isPerTensorQuant, MPerBlock, MulRoutedWeight, ActOP)                                                                            \
//     if (ActOP == 0)     \
//     {   \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 0>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//         }   \
//     }\
//     else if (ActOP == 1)                                                                                       \
//     {   \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);   \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr,  num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage1_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, MulRoutedWeight, 1>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, hidden_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w1_scale_ptr, a1_scale_ptr); \
//         }   \
//     }                                                                                                                                                                                                                                                                                                                                                                  \

// void ck_moe_stage1(torch::Tensor &hidden_states,     // [m, k], input token
//                    torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
//                    torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
//                    torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
//                    torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
//                    torch::Tensor &num_valid_ids,     // [1]
//                    torch::Tensor &out,               // [m * topk, inter_dim]
//                    int topk,
//                    std::optional<torch::Tensor> w1_scale        = std::nullopt, // [e, 1, n], gate(up) scale
//                    std::optional<torch::Tensor> a1_scale        = std::nullopt, // [m, 1], token scale
//                    std::optional<int> block_m                   = 32,
//                    std::optional<torch::Tensor> sorted_weights  = std::nullopt,
//                    std::optional<int> act_op                    = 0,
//                    std::optional<int> pipe_ver                  = 1)
// {
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
//     // TORCH_CHECK(hidden_states.dtype() == w1.dtype(),
//     //             "Weights and activations should both be same dtype!");

//     TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
//                 "Out dtype only support BFloat16/Float16!")

//     int tokens = hidden_states.size(0);
//     int sorted_size = sorted_token_ids.size(0);
//     int E = w1.size(0);
//     int N = w1.size(1) / 2;
//     int K = hidden_states.size(-1);
//     // printf("%dx%dx%d\n", E, N, K);
//     // int max_num_tokens_padded = sorted_token_ids.size(0);
//     // int agvtokens_per_expert = max_num_tokens_padded / E;
//     int MPerBlock = block_m.value();
//     bool isPerTensorQuant = (!w1_scale.has_value()) || (w1_scale.value().numel() == E);
// #if defined(__gfx942__)
//     PipelineVersion PipelineVer = (pipe_ver == 1 || MPerBlock < 64) ? PipelineVersion::v1 : PipelineVersion::v3;
// #else
//     PipelineVersion PipelineVer = (pipe_ver == 1 || MPerBlock < 128) ? PipelineVersion::v1 : PipelineVersion::v3;
// #endif
//     // int M = agvtokens_per_expert < 32 ? 32 : (agvtokens_per_expert < 64 ? 64 : 128);

//     void *hidden_states_ptr = hidden_states.data_ptr();
//     void *w1_ptr = w1.transpose(1, 2).data_ptr();
//     void *w2_ptr = w2.data_ptr();
//     void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
//     void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
//     void *sorted_weights_ptr = sorted_weights.has_value() ? sorted_weights.value().data_ptr() : nullptr;
//     void *num_valid_ids_ptr = num_valid_ids.data_ptr();
//     void *out_ptr = out.data_ptr();
//     void *w1_scale_ptr = w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr;
//     void *a1_scale_ptr = a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr;
//     // BF16
//     if (hidden_states.dtype() == at::ScalarType::BFloat16)
//     {
//         using A0DataType = B16;
//         using B0DataType = B16;
//         using AccDataType = F32;
//         using EDataType = B16;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1) 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//             }
//         }
//         else 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//             }
//         }
//     }
//     // FP16
//     else if (hidden_states.dtype() == at::ScalarType::Half)
//     {
//         using A0DataType = F16;
//         using B0DataType = F16;
//         using AccDataType = F32;
//         using EDataType = F16;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1) 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//             }
//         }
//         else 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//             }
//         }
//     }
//     // FP8 Wint4
//     else if (hidden_states.dtype() == torch_fp8 && w1.dtype() == at::ScalarType::UInt32)
//     {
//         using A0DataType = F8;
//         using B0DataType = I4;
//         const bool Nswizzle = false;
//         TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
//                     "MoE Quant must input scale!");
//         TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
//                     "Scales must be Float dtype!");
//         using AccDataType = F32;
//         using CDEElementOp = MulABScaleWint4;
//         if (sorted_weights.has_value()) 
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//             }
//         }
//         else
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE1_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//             }
//         }
//     }
//     // FP8
//     else if (hidden_states.dtype() == torch_fp8)
//     {
//         using A0DataType = F8;
//         using B0DataType = F8;
//         TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
//                     "MoE Quant must input scale!");
//         TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
//                     "Scales must be Float dtype!");
//         using AccDataType = F32;
//         using CDEElementOp = MulABScale;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1) 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//             }
//             else
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//             }
//         }
//         else 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//             }
//             else
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//             }
//         }
//     }
//     // I8
//     else if (hidden_states.dtype() == at::ScalarType::Char)
//     {
//         using A0DataType = I8;
//         using B0DataType = I8;
//         TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
//                     "MoE Quant must input scale!");
//         TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
//                     "Scales must be Float dtype!");
//         using AccDataType = I32;
//         using CDEElementOp = MulABScale;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1) 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//             }
//             else
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//             }
//         }
//         else 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//             }
//             else
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//             }
//         }
//     }
//     // mxfp4
//     else if ((hidden_states.dtype() == at::ScalarType::Byte && w2.dtype() == at::ScalarType::Byte))
//     {
//         K *= 2; // packed fp4
//         using A0DataType = F4;
//         using B0DataType = F4;
//         using A1DataType = XDataType;
//         using B1DataType = XDataType;
//         TORCH_CHECK(a1_scale.has_value() && w1_scale.has_value(),
//                     "MoE Quant must input scale!");
//         // TORCH_CHECK(a1_scale.value().dtype() == at::ScalarType::Float,
//         //             "Scales must be Float dtype!");
//         using AccDataType = F32;
//         using CDEElementOp = MulABScaleExpertWeight;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1) 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//             }
//             else
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//             }
//         }
//         else 
//         {
//             if (sorted_weights.has_value()) 
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, true, act_op);
//                 }
//             }
//             else
//             {
//                 if (out.dtype() == at::ScalarType::Half)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//                 else if (out.dtype() == at::ScalarType::BFloat16)
//                 {
//                     CK_MOE_STAGE1_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock, false, act_op);
//                 }
//             }
//         }
//     }
// }

// #define CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                                       \
//     if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
//     {                                                                                                                                                                                                                                                                                                                                                                       \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//     }                                                                                                                                                                                                                                                                                                                                                                       \
//     else                                                                                                                                                                                                                                                                                                                                                                    \
//     {                                                                                                                                                                                                                                                                                                                                                                       \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 256 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//     }

// #define CK_MOE_STAGE2_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                                  \
//     if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
//     {                                                                                                                                                                                                                                                                                                                                                                       \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//     }                                                                                                                                                                                                                                                                                                                                                                       \
//     else                                                                                                                                                                                                                                                                                                                                                    \
//     {                                                                                                                                                                                                                                                                                                                                                                       \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 2, 2, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm_mxfp4<A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//     } 

// #define CK_MOE_STAGE2_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, Nswizzle, isPerTensorQuant, MPerBlock)                                                                                                                                                                                                                                                  \
//     if (MulRoutedWeight)                                                                                                                                                                                                                                                                                                                                                    \
//     {                                                                                                                                                                                                                                                                                                                                                                       \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, true>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//     }                                                                                                                                                                                                                                                                                                                                                                       \
//     else                                                                                                                                                                                                                                                                                                                                                    \
//     {                                                                                                                                                                                                                                                                                                                                                                       \
//         if (isPerTensorQuant)                                                                                                                                                                                                                                                                                                                                               \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);   \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, true, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//         else                                                                                                                                                                                                                                                                                                                                                                \
//         {                                                                                                                                                                                                                                                                                                                                                                   \
//             if (MPerBlock == 32)                                                                                                                                                                                                                                                                                                                                            \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 32, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 64)                                                                                                                                                                                                                                                                                                                                       \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 64, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr);  \
//             else if (MPerBlock == 128)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 128, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//             else if (MPerBlock == 256)                                                                                                                                                                                                                                                                                                                                      \
//                 ck_moe_stage2_gemm<A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVer, 256, 128 / sizeof(A0DataType), 1, 4, Nswizzle, false, false>(at::cuda::getCurrentCUDAStream().stream(), tokens, sorted_size, N, K, topk, inter_states_ptr, w1_ptr, w2_ptr, sorted_token_ids_ptr, sorted_expert_ids_ptr, sorted_weights_ptr, num_valid_ids_ptr, out_ptr, w2_scale_ptr, a2_scale_ptr); \
//         }                                                                                                                                                                                                                                                                                                                                                                   \
//     }  

// void ck_moe_stage2(torch::Tensor &inter_states,      // [m, k], input token
//                    torch::Tensor &w1,                // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
//                    torch::Tensor &w2,                // [expert, dim, inter_dim], pre-shuffle([e, nr, kr, w])
//                    torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
//                    torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
//                    torch::Tensor &num_valid_ids,     // [1]
//                    torch::Tensor &out,               // [max_num_tokens_padded, inter_dim]
//                    int topk,
//                    std::optional<torch::Tensor> w2_scale = std::nullopt, // [e, 1, n], gate(up) scale
//                    std::optional<torch::Tensor> a2_scale = std::nullopt, // [m, 1], token scale
//                    std::optional<int> block_m = 32,
//                    std::optional<torch::Tensor> sorted_weights = std::nullopt, // [max_num_tokens_padded])
//                    std::optional<int> pipe_ver                 = 1)    
// {
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
//     // TORCH_CHECK(inter_states.dtype() == w2.dtype(),
//     //             "Weights and activations should both be same dtype!");
//     //
//     TORCH_CHECK(out.dtype() == at::ScalarType::BFloat16 || out.dtype() == at::ScalarType::Half,
//                 "Out dtype only support BFloat16/Float16!")

//     int tokens = inter_states.size(0);
//     int sorted_size = sorted_token_ids.size(0);
//     int E = w1.size(0);
//     int N = w2.size(1);
//     int K = inter_states.size(-1);
//     // printf("%dx%dx%d\n", E, N, K);
//     // int max_num_tokens_padded = sorted_token_ids.size(0);
//     // int agvtokens_per_expert = max_num_tokens_padded / E;
//     int MPerBlock = block_m.value();
//     // int M = agvtokens_per_expert < 32 ? 32 : (agvtokens_per_expert < 64 ? 64 : 128);
//     bool isPerTensorQuant = (!w2_scale.has_value()) || (w2_scale.value().numel() == E);
//     bool MulRoutedWeight = sorted_weights.has_value();
// #if defined(__gfx942__)
//     PipelineVersion PipelineVer = (pipe_ver == 1 || MPerBlock < 64) ? PipelineVersion::v1 : PipelineVersion::v3;
// #else
//     PipelineVersion PipelineVer = (pipe_ver == 1 || MPerBlock < 128) ? PipelineVersion::v1 : PipelineVersion::v3;
// #endif    

//     void *inter_states_ptr = inter_states.data_ptr();
//     void *w1_ptr = w1.data_ptr();
//     void *w2_ptr = w2.data_ptr();
//     void *sorted_token_ids_ptr = sorted_token_ids.data_ptr();
//     void *sorted_expert_ids_ptr = sorted_expert_ids.data_ptr();
//     void *sorted_weights_ptr = MulRoutedWeight ? sorted_weights.value().data_ptr() : nullptr;
//     void *num_valid_ids_ptr = num_valid_ids.data_ptr();
//     void *out_ptr = out.data_ptr();
//     void *w2_scale_ptr = w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr;
//     void *a2_scale_ptr = a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr;

//     // BF16
//     if (inter_states.dtype() == at::ScalarType::BFloat16)
//     {
//         using A0DataType = B16;
//         using B0DataType = B16;
//         using AccDataType = F32;
//         using EDataType = B16;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1)
//         {
//             if (MulRoutedWeight) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//         else 
//         {
//             if (MulRoutedWeight) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//     }
//     // FP16
//     else if (inter_states.dtype() == at::ScalarType::Half)
//     {
//         using A0DataType = F16;
//         using B0DataType = F16;
//         using AccDataType = F32;
//         using EDataType = F16;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1)
//         {
//             if (MulRoutedWeight) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//         else 
//         {
//             if (MulRoutedWeight) 
//             {
//                 using CDEElementOp = TypeCastExpertWeight;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else 
//             {
//                 using CDEElementOp = TypeCast;
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, EDataType, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//     }
//     // FP8 wint4
//     else if (inter_states.dtype() == torch_fp8 && w1.dtype() == at::ScalarType::UInt32)
//     {
//         using A0DataType = F8;
//         using B0DataType = I4;
//         const bool Nswizzle = false;
//         TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
//                     "MoE Quant must input scale!");
//         TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
//                     "Scales must be Float dtype!");
//         using AccDataType = F32;
//         using CDEElementOp = MulABScaleExpertWeightWin4;

//         // TODO: need to add the v3 support of int4
//         if (out.dtype() == at::ScalarType::Half)
//         {
//             CK_MOE_STAGE2_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//         }
//         else if (out.dtype() == at::ScalarType::BFloat16)
//         {
//             CK_MOE_STAGE2_GEMM_IMPL_INT4(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//         }
//     }
//     // FP8
//     else if (inter_states.dtype() == torch_fp8)
//     {
//         using A0DataType = F8;
//         using B0DataType = F8;
//         TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
//                     "MoE Quant must input scale!");
//         TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
//                     "Scales must be Float dtype!");
//         using AccDataType = F32;
//         using CDEElementOp = MulABScaleExpertWeight;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1)
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//         else
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//     }
//     // I8
//     else if (inter_states.dtype() == at::ScalarType::Char)
//     {
//         using A0DataType = I8;
//         using B0DataType = I8;
//         TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
//                     "MoE Quant must input scale!");
//         TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
//                     "Scales must be Float dtype!");
//         using AccDataType = I32;
//         using CDEElementOp = MulABScaleExpertWeight;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1)
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//         else
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL(A0DataType, B0DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//     }
//     else if (inter_states.dtype() == at::ScalarType::Byte && w2.dtype() == at::ScalarType::Byte)
//     {
//         K *= 2;
//         using A0DataType = F4;
//         using B0DataType = F4;
//         using A1DataType = ck::e8m0_bexp_t;
//         using B1DataType = ck::e8m0_bexp_t;
//         TORCH_CHECK(a2_scale.has_value() && w2_scale.has_value(),
//                     "MoE Quant must input scale!");
//         // TORCH_CHECK(a2_scale.value().dtype() == at::ScalarType::Float,
//         //             "Scales must be Float dtype!");
//         using AccDataType = F32;
//         using CDEElementOp = MulABScaleExpertWeight;
//         const bool Nswizzle = false;
//         if (PipelineVer == PipelineVersion::v1)
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v1, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//         else
//         {
//             if (out.dtype() == at::ScalarType::Half)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, F16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//             else if (out.dtype() == at::ScalarType::BFloat16)
//             {
//                 CK_MOE_STAGE2_GEMM_IMPL_MXFP4(A0DataType, A1DataType, B0DataType, B1DataType, AccDataType, B16, CDEElementOp, PipelineVersion::v3, Nswizzle, isPerTensorQuant, MPerBlock);
//             }
//         }
//     }
// }
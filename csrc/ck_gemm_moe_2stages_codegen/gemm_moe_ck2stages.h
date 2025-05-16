// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/check_err.hpp"

#include <torch/torch.h>
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include <hip/hip_runtime.h>

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using I4 = ck::pk_i4_t;
using I8 = int8_t;
using I32 = int;
using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using F8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PipelineVersion = ck::BlockGemmPipelineVersion;

const auto V1 = ck::BlockGemmPipelineVersion::v1;
const auto V3 = ck::BlockGemmPipelineVersion::v3;

struct TypeCast
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;

    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, float>(F16 &e, const float &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, float>(B16 &e, const float &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<F16, F16, float, float>(F16 &e, const F16 &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, B16, float, float>(B16 &e, const B16 &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }
};

// for gate, a_scale, b_scale
struct MulABScale
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;

    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, float>(F16 &e,
                                                                            const float &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, float>(B16 &e,
                                                                            const float &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }
};

struct MulABScaleWint4
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;

    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, float>(F16 &e,
                                                                            const float &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, float>(B16 &e,
                                                                            const float &c,
                                                                            const float &d0,
                                                                            const float &d1,
                                                                            const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<F16, int, float, float, float>(F16 &e,
                                                                          const int &c,
                                                                          const float &d0,
                                                                          const float &d1,
                                                                          const float &d2) const
    {
        e = ck::type_convert<F16>(ck::type_convert<F32>(c));
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, int, float, float, float>(B16 &e,
                                                                          const int &c,
                                                                          const float &d0,
                                                                          const float &d1,
                                                                          const float &d2) const
    {
        e = ck::type_convert<B16>(ck::type_convert<F32>(c));
    }
};

struct TypeCastExpertWeight
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;
    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, float>(F16 &e,
                                                                                   const float &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, float>(B16 &e,
                                                                                   const float &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<F16, F16, float, float, float>(F16 &e,
                                                                                   const F16 &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, B16, float, float, float>(B16 &e,
                                                                                   const B16 &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<F16, int, float, float, float>(F16 &e,
                                                                                   const int &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(ck::type_convert<F32>(c));
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, int, float, float, float>(B16 &e,
                                                                                   const int &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(ck::type_convert<F32>(c));
    }
};

// d0: ascale, d1: bscale, d2:expert weight
//warning: hack hack hack here!!!! ignore d0 right now as kernel mul d0 * d2 outside. tofix:felix 
struct MulABScaleExpertWeight
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;
    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, float>(F16 &e,
                                                                                   const float &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<F16, F16, float, float, float>(F16 &e,
                                                                                   const F16 &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, B16, float, float, float>(B16 &e,
                                                                                   const B16 &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, float>(B16 &e,
                                                                                   const float &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(c);
    }

    template <>
    __host__ __device__ constexpr void operator()<F16, int, float, float, float>(F16 &e,
                                                                                    const int &c,
                                                                                    const float &d0,
                                                                                    const float &d1,
                                                                                    const float &d2) const
    {
        e = ck::type_convert<F16>(ck::type_convert<F32>(c));
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, int, float, float, float>(B16 &e,
                                                                                    const int &c,
                                                                                    const float &d0,
                                                                                    const float &d1,
                                                                                    const float &d2) const
    {
        e = ck::type_convert<B16>(ck::type_convert<F32>(c));
    }
    

    template <typename E, typename C, typename D2>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D2& d2) const;
    // for real kernel use
    template <>
    __host__ __device__ constexpr void
    operator()<F16, float, float>(F16& e, const float& c, const float& d2) const
    {
        // for real kernel use
        e = ck::type_convert<F16>(c * d2);
    }

    // for reference cpu
    template <>
    __host__ __device__ constexpr void
    operator()<float, float, float>(float& e, const float& c, const float& d2) const
    {
        // for reference cpu
        e = ck::type_convert<F16>(c * d2);
    }



};

// d0: ascale, d1: bscale, d2:expert weight
// warning: hack hack hack here!!!! ignore d0 right now as kernel mul d0 * d2 outside. tofix:felix
struct MulABScaleExpertWeightWin4
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;
    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, float>(F16 &e,
                                                                                   const float &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(c * 16.f);
    }
    template <>
    __host__ __device__ constexpr void operator()<F16, F16, float, float, float>(F16 &e,
                                                                                   const F16 &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<F16>(c * 16.f);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, float>(B16 &e,
                                                                                   const float &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(c * 16.f);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, B16, float, float, float>(B16 &e,
                                                                                   const B16 &c,
                                                                                   const float &d0,
                                                                                   const float &d1,
                                                                                   const float &d2) const
    {
        e = ck::type_convert<B16>(c * 16.f);
    }


    template <>
    __host__ __device__ constexpr void operator()<F16, int, float, float, float>(F16 &e,
                                                                                 const int &c,
                                                                                 const float &d0,
                                                                                 const float &d1,
                                                                                 const float &d2) const
    {
        e = ck::type_convert<F16>(ck::type_convert<F32>(c) * 16.f);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, int, float, float, float>(B16 &e,
                                                                                 const int &c,
                                                                                 const float &d0,
                                                                                 const float &d1,
                                                                                 const float &d2) const
    {
        e = ck::type_convert<B16>(ck::type_convert<F32>(c) * 16.f);
    }
};

using MoeKernel = std::function<
    void(const hipStream_t &stream, int, int, int, int,
         int,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         void *&,
         std::optional<void *>,
         std::optional<void *>)>;


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
bool Nswizzle,
bool PerTensorQuant,
bool MulRoutedWeight,
int ActOP>
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

// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/// @brief The GEMM problem definition.
///
/// @par Overview
///      This structure defines the GEMM problem configuration by stating all required information
///      like M,N,K sizes and respective strides.
struct Block_quant_GemmProblem
{
    CK_TILE_HOST Block_quant_GemmProblem() = default;
    CK_TILE_HOST Block_quant_GemmProblem(
       index_t M_, index_t N_, index_t K_, index_t stride_A_, index_t stride_B_, index_t stride_C_)
        : M(M_), N(N_), K(K_), stride_A(stride_A_), stride_B(stride_B_), stride_C(stride_C_)
    {
    }

    index_t M;
    index_t N;
    index_t K;
    index_t stride_A;
    index_t stride_B;
    index_t stride_C;
 
};

/// @brief The GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref GemmKernel "GemmKernel" when creating kernel arguments
///      object. It contain all necessary information required to build proper kernel argument
///      and launch kernel on GPU.
struct Block_quant_GemmHostArgs : public Block_quant_GemmProblem
{
    CK_TILE_HOST Block_quant_GemmHostArgs() = default;
    CK_TILE_HOST Block_quant_GemmHostArgs(const void* a_ptr_,
                              const void* b_ptr_,
                              void* c_ptr_,
                              const void* a_scale_ptr_,
                              const void* b_scale_ptr_,
                              index_t k_batch_,
                              index_t M_,
                              index_t N_,
                              index_t K_,
                              index_t stride_A_,
                              index_t stride_B_,
                              index_t stride_C_)
        : Block_quant_GemmProblem(M_, N_, K_, stride_A_, stride_B_, stride_C_),
          a_ptr(a_ptr_),
          b_ptr(b_ptr_),
          a_scale_ptr(a_scale_ptr_),
          b_scale_ptr(b_scale_ptr_),
          c_ptr(c_ptr_),
          k_batch(k_batch_)
    {
    }

    const void* a_ptr;
    const void* b_ptr;
    const void* a_scale_ptr;
    const void* b_scale_ptr;
    void* c_ptr;
    index_t k_batch;
};

/// @brief The GEMM kernel device arguments.
struct Block_quant_GemmKernelArgs
{
    /// @brief The A input tensor's pointer to device memory.
    const void* a_ptr;
    /// @brief The B input tensor's pointer to device memory.
    const void* b_ptr;
   
    /// @brief The C output tensor's pointer to device memory.
    void* c_ptr;
    /// @brief GEMM's M dimension size.
     const void* a_scale_ptr;
    /// @brief The B input tensor's pointer to device memory.
    const void* b_scale_ptr;
    index_t M;
    /// @brief GEMM's N dimension size.
    index_t N;
    /// @brief GEMM's K dimension size.
    index_t K;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of A tensor.
    index_t stride_A;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of B tensor.
    index_t stride_B;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of C tensor.
    index_t stride_C;
    index_t k_batch;
};

/// @brief The GEMM kernel template.
///
/// @paragraph Overview Overview
///            This class provides the generic matrix multiplication kernel template. By semantic
///            division of GEMM algorithm into following parts we achieve flexible, versatile
///            and robust kernel implementation.
///
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator" which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part @a "heart" of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The @a "final" part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            Additionally both @ref GemmPipeline_ "GemmPipeline" and @ref EpiloguePipeline_
///            "EpiloguePipeline" are parameterized with so called @a Policy which determines all
///            internal details of those functional parts. You can think of it like both gemm and
///            epilogue pipelines provides the control-flow logic controlled by policies. Moreover
///            the policy is responsible for definition of all necessary data layouts and thread's
///            work distribution.
///
/// @tparam TilePartitioner_    The type of class providing mapping of workgroup index into the
///                             output data tile to be calculated. It determines the workgroup to
///                             data relationship (or in other words - which data would be
///                             processed and calculated by which workgroup).
/// @tparam GemmPipeline_       The type of class which provides the core part of matrix
///                             multiplication. This class should provide implementation of data
///                             loading from global memory and performing block-wise matrix
///                             multiplication. You can think of it as a work done by single
///                             workgroup point of view.
/// @tparam EpiloguePipeline_   The type of class providing the final part of matrix
///                             multiplication implementation. It is responsible for storing
///                             results calculated by @ref GemmPipeline_ "GemmPipeline" to
///                             the output C tensor in global memory.
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct Block_quant_GemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    static constexpr index_t ScaleBlockM = 1;
    static constexpr index_t ScaleBlockN = 128;
    static constexpr index_t ScaleBlockK = 128;
    // Get the persistent kernel if the pipeline has it available
    struct has_persistent_kernel
    {
        template <typename T>
        using has_persistent_type = decltype(T::UsePersistentKernel);

        static constexpr bool value = []() {
            if constexpr(is_detected<has_persistent_type, GemmPipeline>{})
                return GemmPipeline::UsePersistentKernel;
            else
                return false;
        }();
    };
    static constexpr bool PersistentKernel = has_persistent_kernel::value;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm", gemm_prec_str<ADataType, BDataType>, GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using Kernel      = Block_quant_GemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto kernel = kentry<KernelBlockSize, 1, Kernel, Block_quant_GemmKernelArgs>;
        int occupancy;
        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, KernelBlockSize, 0));
        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr Block_quant_GemmKernelArgs MakeKernelArgs(const Block_quant_GemmHostArgs& hostArgs)
    {   
        return Block_quant_GemmKernelArgs{hostArgs.a_ptr,
                              hostArgs.b_ptr,
                              hostArgs.c_ptr,
                              hostArgs.a_scale_ptr,
                              hostArgs.b_scale_ptr,
                              hostArgs.M,
                              hostArgs.N,
                              hostArgs.K,
                              hostArgs.stride_A,
                              hostArgs.stride_B,
                              hostArgs.stride_C,
                              hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const Block_quant_GemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = __builtin_amdgcn_readfirstlane(kargs.k_batch * K1);
            const index_t KRead = __builtin_amdgcn_readfirstlane((kargs.K + K_t - 1) / K_t * K1);

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_A);
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = __builtin_amdgcn_readfirstlane(KRead);
            }
            else
            {
                splitted_k = __builtin_amdgcn_readfirstlane(kargs.K - KRead * (kargs.k_batch - 1));
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const Block_quant_GemmKernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<CDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % GemmPipeline::GetVectorSizeA() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
               GemmPipeline::kPadK == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Can't support K that is not a multiple of k_batch * KPerBlock "
                                  "without padding!");
                }
                return false;
            }
            if(kargs.K % GemmPipeline::GetVectorSizeB() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("K is not a multiple of vector load size for B tensor!");
                }
                return false;
            }
        }

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeGemmTensorViews(const ADataType* a_ptr,
                                                   const BDataType* b_ptr,
                                                   CDataType* c_ptr,
                                                   const float* a_scale_ptr,
                                                   const Block_quant_GemmKernelArgs& kargs,
                                                   const SplitKBatchOffset& splitk_batch_offset)
    {   
        (void)a_scale_ptr;
        static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();
       
        const auto& b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = splitk_batch_offset.splitted_k / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(splitk_batch_offset.splitted_k, kargs.N),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
            else
            {
                if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                {
                    constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                    const index_t K0              = splitk_batch_offset.splitted_k / K1;
                    constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                    const auto b_k0_n_k1_desc =
                        make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                     make_tuple(kargs.N * K1, K1, I1),
                                                     number<VectorSizeB>{},
                                                     number<1>{});
                    const auto b_n_k_desc = transform_tensor_descriptor(
                        b_k0_n_k1_desc,
                        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                   make_pass_through_transform(kargs.N)),
                        make_tuple(sequence<0, 2>{}, sequence<1>{}),
                        make_tuple(sequence<1>{}, sequence<0>{}));
                    return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        b_ptr,
                        make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                        make_tuple(kargs.stride_B, 1),
                        number<GemmPipeline::GetVectorSizeB()>{},
                        number<1>{});
                }
            }
        }();

        // TODO: enable vector write for C in ColMajor
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();
         const auto& a_scale_tensor_view = [&]() {
           if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_scale_ptr,
                    make_tuple(kargs.M, (splitk_batch_offset.splitted_k + ScaleBlockK-1)/ScaleBlockK ),
                    make_tuple((kargs.stride_A + ScaleBlockK - 1)/ScaleBlockK, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_scale_ptr,
                    make_tuple((splitk_batch_offset.splitted_k+ScaleBlockK-1)/ScaleBlockK , kargs.M),
                    make_tuple((kargs.stride_A+ScaleBlockK-1)/ScaleBlockK, 1),
                    number<1>{},
                    number<1>{});
            }
    //    return make_naive_tensor_view<address_space_enum::global>(
    //                 a_scale_ptr,
    //                 make_tuple(kargs.M),
    //                 make_tuple(1),
    //                 number<1>{},
    //                 number<1>{});

           
        }();
        return make_tuple(a_tensor_view, b_tensor_view, c_tensor_view,a_scale_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadM>{});
            }
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I1);
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
        }();

        // TODO vector write in for C in ColMajor
        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I2);
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        const auto& a_scale_pad_view = [&]() {
            const auto& a_scale_tensor_view = views.at(I3);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_scale_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, false>{});
            }
            else
            {
                return pad_tensor_view(a_scale_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, false>{});
            }
        }();
        //  const auto& a_scale_pad_view = [&]() {
        //  const auto& a_scale_tensor_view = views.at(I3);
        //         return pad_tensor_view(a_scale_tensor_view,
        //                                make_tuple(number<TilePartitioner::MPerBlock>{}),
        //                               sequence< GemmPipeline::kPadM>{});
          
        // }();
        return make_tuple(a_pad_view, b_pad_view, c_pad_view,a_scale_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view = views.at(I0);
        const auto& b_pad_view = views.at(I1);
        const auto& c_pad_view = views.at(I2);
        const auto& a_scale_pad_view = views.at(I3);
        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, i_m});
            }
        }();

        const auto& b_block_window = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::NPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_n, 0});
            }
            else
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {0, i_n});
            }
        }();

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});
        const auto& a_scale_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_scale_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<1>{}),
                                        {i_m, 0});
            }
            else
            {
                return make_tile_window(a_scale_pad_view,
                                        make_tuple(number<1>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, i_m});
            }
        }();
    //      const auto& a_scale_block_window = [&]() {
            
            
    //             return make_tile_window(
    //             a_scale_pad_view, make_tuple( number<TilePartitioner::MPerBlock>{}), {i_m});
            
    //     }();
        return make_tuple(a_block_window, b_block_window, c_block_window,a_scale_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const ADataType* a_ptr,
                                       const BDataType* b_ptr,
                                       CDataType* c_ptr,
                                       const float* a_scale_ptr,
                                       const float* b_scale_ptr,
                                       void* smem_ptr_0,
                                       const Block_quant_GemmKernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {   
        
        const float* b_per_blcok_scale=b_scale_ptr+(block_idx_n)/ScaleBlockN*((kargs.K+ScaleBlockK-1)/ScaleBlockK);
      
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, c_ptr, a_scale_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);
        const auto& a_scale_block_window = gemm_tile_windows.at(I3);
        (void)a_scale_block_window;
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window,a_scale_block_window,b_per_blcok_scale, num_loop,  smem_ptr_0);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I2);

        EpiloguePipeline{}.template operator()<decltype(c_block_window), decltype(c_block_tile)>(
            c_block_window, c_block_tile,b_block_window, smem_ptr_0);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note RunGEMM2LDS in with two shared memory buffers using the ping pong buffer mechanism.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The starting pointer of 1st shared memory block.
     * @param smem_ptr_1 The starting pointer of 2nd shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm2LDS(const ADataType* a_ptr,
                                           const BDataType* b_ptr,
                                           CDataType* c_ptr,
                                           void* __restrict__ smem_ptr_0,
                                           void* __restrict__ smem_ptr_1,
                                           const Block_quant_GemmKernelArgs& kargs,
                                           const SplitKBatchOffset& splitk_batch_offset,
                                           const index_t block_idx_m,
                                           const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, c_ptr, kargs, splitk_batch_offset);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0, smem_ptr_1);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I2);

        EpiloguePipeline{}.template operator()<decltype(c_block_window), decltype(c_block_tile)>(
            c_block_window, c_block_tile, smem_ptr_0);
    }

    // Non-persistent kernel entry point
    template <bool U = !PersistentKernel, typename = std::enable_if_t<U>>
    CK_TILE_DEVICE void operator()(Block_quant_GemmKernelArgs kargs) const
    {
        const auto blockId  = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const SplitKBatchOffset splitk_batch_offset(kargs);
        // options
        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_ptr =
            static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
        CDataType* c_ptr = static_cast<CDataType*>(kargs.c_ptr);
        
        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        if constexpr(GemmPipeline::DoubleSmemBuffer == true)
        {
            __shared__ char smem_ptr_1[GetSmemSize()];
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<CDataType, fp16_t, bf16_t>::value))
            {
                RunGemm2LDS(a_ptr,
                            b_ptr,
                            c_ptr,
                            smem_ptr_0,
                            smem_ptr_1,
                            kargs,
                            splitk_batch_offset,
                            i_m,
                            i_n);
            }
        }
        else
        {
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<CDataType, fp16_t, bf16_t>::value))
            {
                RunGemm(a_ptr, b_ptr, c_ptr, static_cast<const float*>(kargs.a_scale_ptr), static_cast<const float*>(kargs.b_scale_ptr), smem_ptr_0, kargs, splitk_batch_offset, i_m, i_n);
            }
        }
    }

    // Persistent kernel entry point
    template <bool U = PersistentKernel, typename = std::enable_if_t<U>, typename = void>
    CK_TILE_DEVICE void operator()(Block_quant_GemmKernelArgs kargs) const
    {
        const auto grid_size = __builtin_amdgcn_readfirstlane(get_grid_size());
        const auto num_tiles =
            __builtin_amdgcn_readfirstlane(TilePartitioner::GridSize(kargs.M, kargs.N));
        const auto num_work = __builtin_amdgcn_readfirstlane(num_tiles * kargs.k_batch);
        auto block_id       = __builtin_amdgcn_readfirstlane(get_block_id());

        while(block_id < num_work)
        {
            // Get the tile index for this block
            const auto tile_idx = __builtin_amdgcn_readfirstlane(block_id % num_tiles);
            const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(tile_idx);
            const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
            const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

            // Get the SplitK offset for this block
            const auto k_batch = __builtin_amdgcn_readfirstlane(block_id / num_tiles);
            const SplitKBatchOffset splitk_batch_offset(kargs, k_batch);
            const ADataType* a_ptr =
                static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
            const BDataType* b_ptr =
                static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
            CDataType* c_ptr = static_cast<CDataType*>(kargs.c_ptr);
            
            // allocate LDS
            __shared__ char smem_ptr_0[GetSmemSize()];
            // Run the GEMM
            if constexpr(GemmPipeline::DoubleSmemBuffer == true)
            {
                __shared__ char smem_ptr_1[GetSmemSize()];
                if constexpr(!(EpiloguePipeline::MemoryOperation ==
                                   memory_operation_enum::atomic_add &&
                               EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                               is_any_of<CDataType, fp16_t, bf16_t>::value))
                {
                    RunGemm2LDS(a_ptr,
                                b_ptr,
                                c_ptr,
                                smem_ptr_0,
                                smem_ptr_1,
                                kargs,
                                splitk_batch_offset,
                                i_m,
                                i_n);
                }
            }
            else
            {
                if constexpr(!(EpiloguePipeline::MemoryOperation ==
                                   memory_operation_enum::atomic_add &&
                               EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                               is_any_of<CDataType, fp16_t, bf16_t>::value))
                {
                    RunGemm(a_ptr, b_ptr, c_ptr, static_cast<const float*>(kargs.a_scale_ptr), static_cast<const float*>(kargs.b_scale_ptr), smem_ptr_0, kargs, splitk_batch_offset, i_m, i_n);
                }
            }
            // Advance to the next work item
            block_id += grid_size;
            if(block_id >= num_work)
            {
                break;
            }
        }
    }
};

} // namespace ck_tile

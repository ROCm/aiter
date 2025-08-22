// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "gemm_universal_pipeline_ag_bg_cr_policy_block_quant.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"
// #include "ck/utility/get_id.hpp"
namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrComp_block_quant
{
    static constexpr index_t PrefetchStages   = 2;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(BlockHasHotloop(num_loop))
        {
            return TailNumber::Full;
        }
        else
        {
            if(num_loop == 1)
            {
                return TailNumber::Odd;
            }
            else
            {
                return TailNumber::Even;
            }
        }
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
    {
        // Handle all the valid cases.
        if(has_hot_loop)
        {
            if(tail_number == TailNumber::Full)
            {
                return run_func(bool_constant<true>{},
                                integral_constant<TailNumber, TailNumber::Full>{});
            }
        }
        else
        {
            if(tail_number == TailNumber::Odd)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Odd>{});
            }
            else if(tail_number == TailNumber::Even)
            {
                return run_func(bool_constant<false>{},
                                integral_constant<TailNumber, TailNumber::Even>{});
            }
        }
#if defined(__HIP_DEVICE_COMPILE__)
        // This path should be unreachable in device code if tail_number is valid.
        __builtin_unreachable();
#else
        // If execution reaches here, it's an invalid combination of arguments.
        if(has_hot_loop)
        {
            throw std::logic_error("Invalid TailNumber: If has_hot_loop is true, tail_number must "
                                   "be TailNumber::Full.");
        }
        else
        {
            throw std::logic_error("Invalid TailNumber: If has_hot_loop is false, tail_number must "
                                   "be TailNumber::Odd or TailNumber::Even.");
        }
#endif
    }
};

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1
template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrPolicy_block_quant>
struct GemmPipelineAgBgCrComp_block_quant : public BaseGemmPipelineAgBgCrComp_block_quant<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrComp_block_quant<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;
    using I3        = number<3>;
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    using Base::PrefetchStages;
    using Base::UsePersistentKernel;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompV3", BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(),  GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string Print()
    {
        constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
        constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
        constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = 64;
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

        // Below should be equal to AK1|BK1
        constexpr index_t A_LDS_Read_Width = GetSmemPackA();
        constexpr index_t B_LDS_Read_Width = GetSmemPackB();

        constexpr index_t A_LDS_Write_Width = GetSmemPackA();
        constexpr index_t B_LDS_Write_Width = GetSmemPackB();

        constexpr index_t A_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
        constexpr index_t B_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

        constexpr index_t A_LDS_Write_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
        constexpr index_t B_LDS_Write_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

        constexpr index_t A_LDS_Read_Inst_Num =
            WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

        constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                            (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        auto str = std::stringstream{};

        str << "A/B vector size: " << GetVectorSizeA() << ", " << GetVectorSizeB() << "\n"
            << "A/B LDS read/write width: " << A_LDS_Read_Width << ", " << B_LDS_Read_Width << "\n"
            << "A/B buffer load inst: " << A_Buffer_Load_Inst_Num << ", " << B_Buffer_Load_Inst_Num
            << "\n"
            << "A/B LDS write inst: " << A_LDS_Write_Inst_Num << ", " << B_LDS_Write_Inst_Num
            << "\n"
            << "A/B LDS read inst: " << A_LDS_Read_Inst_Num << ", " << B_LDS_Read_Inst_Num << "\n"
            << "C MFMA inst: " << C_MFMA_Inst_Num << "\n"
            << "KPack: " << BlockGemm::Traits::KPack << "\n"
            << "PrefetchStages: " << PrefetchStages << "\n";
        return str.str();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
            constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
            constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

            // Below should be equal to AK1|BK1
            constexpr index_t A_LDS_Read_Width = GetSmemPackA();
            constexpr index_t B_LDS_Read_Width = GetSmemPackB();

            constexpr index_t A_LDS_Write_Width = GetSmemPackA();
            constexpr index_t B_LDS_Write_Width = GetSmemPackB();

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
            constexpr index_t B_LDS_Write_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            // A/B split schedule
            // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num
                                                                         : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? B_LDS_Read_Inst_Num
                                                                         : B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
            constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

            constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
            constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

            constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_a_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
            constexpr auto ds_read_b_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

            constexpr auto num_dsread_a_mfma =
                (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_b_mfma =
                (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

            // stage 1
            // Separate this part?
            // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
            //                                               sizeof(ComputeDataType) /
            //                                               sizeof(BDataType)
            //                                           ? sizeof(ComputeDataType) /
            //                                           sizeof(ADataType) : sizeof(ComputeDataType)
            //                                           / sizeof(BDataType);
            constexpr auto num_mfma_stage1 =
                num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
            constexpr auto num_mfma_per_issue =
                num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
            constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
            constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

            static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0); // MFMA
            });
            static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0); // MFMA
            });

            // stage 2
            static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename A_scaleDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const A_scaleDramBlockWindowTmp& a_scale_dram_block_window_tmp,
                                       const float* b_per_block_scale,
                                       index_t num_loop,                                      
                                       void* p_smem) const
        {  
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles
            int b_blcok_stride=0;
            // A/B tiles in LDS
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);


           
            // A DRAM tile window for load
            auto a_scale_copy_dram_window =
                make_tile_window(a_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                              a_scale_dram_block_window_tmp.get_window_lengths(),
                                a_scale_dram_block_window_tmp.get_window_origin(),
                                Policy::template MakeA_scaleTileDistribution<Problem>());
           
            auto a_scale_block_tile  = decltype(load_tile(a_scale_copy_dram_window)){};
            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();
            auto acc_block_tile=block_gemm.MakeCBlockTile();
            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());
           
            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));
            
            ABlockTile a_block_tile;
            BBlockTile b_block_tile;
           
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
          
            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            
            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            // auto a_scale_block_tile  = decltype(load_tile(a_scale_copy_dram_window)){};
            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);
            
            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
          
            
            // LDS write 0
            if constexpr(is_a_col_major)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, a_block_tile);
                Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
            }
            else
            {
                Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
            }
           
            if constexpr(is_b_row_major)
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, b_block_tile);
                Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
            }
            else
            {
                Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
            }

            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);
            
            block_sync_lds();
            block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);

            __builtin_amdgcn_sched_barrier(0);

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    block_sync_lds();

                    if constexpr(is_a_col_major)
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, a_block_tile);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                    }
                    if constexpr(is_b_row_major)
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, b_block_tile);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                    }

                    Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
                    Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);
                  
                    a_scale_block_tile = load_tile(a_scale_copy_dram_window);
                    move_tile_window(a_scale_copy_dram_window, {0, 1});
                    tile_elementwise_inout([](auto& c) { c = 0; }, acc_block_tile);
                    block_gemm(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                    constexpr auto p_idx0 = tile_distributed_index<0>{};
                    constexpr auto c_block = decltype(acc_block_tile)::get_distributed_spans();
                    sweep_tile_span(c_block[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(c_block[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        constexpr auto i_idx   = make_tuple(idx0, p_idx0);
                        
                        acc_block_tile(i_j_idx) = type_convert<CDataType>((type_convert<float>(acc_block_tile(i_j_idx))) * a_scale_block_tile(i_idx)*(*(b_per_block_scale+b_blcok_stride)));
                        c_block_tile(i_j_idx) +=acc_block_tile(i_j_idx);
                    });
                });
              
                    block_sync_lds();

                    block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                    b_blcok_stride +=1;
                } while(i < (num_loop - 1));
            }
            tile_elementwise_inout([](auto& c) { c = 0; }, acc_block_tile);
            // tail
            if constexpr((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd))
            {
                // Leak last MFMA block to epilogue region, cover the potential lds-shuffle
                // latency
                // printf("1\n");
                block_gemm(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);
               
                     a_scale_block_tile = load_tile(a_scale_copy_dram_window);
                    
                    
                    constexpr auto p_idx0 = tile_distributed_index<0>{};
                   
                    
                    (void)p_idx0;
                    constexpr auto c_block = decltype(acc_block_tile)::get_distributed_spans();
                    sweep_tile_span(c_block[number<0>{}], [&](auto idx0) {
                        constexpr auto i_idx   = make_tuple(idx0,p_idx0);
                        
                    sweep_tile_span(c_block[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                      
                        acc_block_tile(i_j_idx) = type_convert<CDataType>((type_convert<float>(acc_block_tile(i_j_idx))) * a_scale_block_tile(i_idx)*(*(b_per_block_scale+b_blcok_stride)));
                        c_block_tile(i_j_idx) +=acc_block_tile(i_j_idx);
                    });
                });
               
            }
            else
            {   
                
                block_gemm(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                
                 a_scale_block_tile = load_tile(a_scale_copy_dram_window);
                move_tile_window(a_scale_copy_dram_window, {0, 1});
                    constexpr auto p_idx0 = tile_distributed_index<0>{};
                    constexpr auto c_block = decltype(acc_block_tile)::get_distributed_spans();
                    sweep_tile_span(c_block[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(c_block[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        constexpr auto i_idx   = make_tuple(idx0,p_idx0);
                       
                        acc_block_tile(i_j_idx) = type_convert<CDataType>((type_convert<float>(acc_block_tile(i_j_idx))) * a_scale_block_tile[i_idx]*(*(b_per_block_scale+b_blcok_stride)));
                        c_block_tile(i_j_idx) +=acc_block_tile(i_j_idx);
                    });
                });
               
                b_blcok_stride +=1;
                block_sync_lds();

                if constexpr(is_a_col_major)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, a_block_tile);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                }
                if constexpr(is_b_row_major)
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, b_block_tile);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                }
                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);

                tile_elementwise_inout([](auto& c) { c = 0; }, acc_block_tile);

                block_gemm(acc_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                a_scale_block_tile = load_tile(a_scale_copy_dram_window);
                
                    sweep_tile_span(c_block[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(c_block[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        constexpr auto i_idx   = make_tuple(idx0,p_idx0);
                       
                        acc_block_tile(i_j_idx) = type_convert<CDataType>((type_convert<float>(acc_block_tile(i_j_idx))) * a_scale_block_tile[i_idx]*(*(b_per_block_scale+b_blcok_stride)));
                        c_block_tile(i_j_idx) +=acc_block_tile(i_j_idx);
                    });
                });
                block_sync_lds();
              
            }
            // __builtin_amdgcn_sched_barrier(0);
            

             
        
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem);
    }

    /**
     * @brief This function runs the pipeline by wrapping it with the tail handler.
     *
     * @note This is used by the persistent gemm kernel variants that don't determine
     *       hot loop and tail number on the host side, e.g. grouped gemm kernel.
     */
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        const auto RunPipeline = [&](auto hot_loop_, auto tail_num_) {
            constexpr bool hot_loop    = hot_loop_.value;
            constexpr auto tail_num    = tail_num_.value;
            constexpr auto PassThrough = [](const auto& x) { return x; };
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop, tail_num>(
                a_dram_block_window_tmp,
                PassThrough,
                b_dram_block_window_tmp,
                PassThrough,
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    /**
     * @brief This function runs the pipeline using compile-time known hot loop and tail number.
     * @param num_loop The number of loop iterations. This is determined at runtime due to e.g.
     * SplitK.
     * @note This is used by the kernel variants that are able to determine
     *       hot loop and tail number on the host side, e.g. non-persistent gemm kernel.
     */
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp,typename A_scaleDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const A_scaleDramBlockWindowTmp& a_scale_dram_block_window_tmp,
                                   const float* b_per_block_scale,
                                   index_t num_loop,
                                   void* p_smem) const
    {   (void)a_scale_dram_block_window_tmp;
        (void)b_per_block_scale;
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            a_scale_dram_block_window_tmp,
            b_per_block_scale,
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile

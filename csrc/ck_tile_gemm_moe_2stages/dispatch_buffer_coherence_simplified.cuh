// Simplified dispatch for buffer coherence enum (针对162行的dispatch代码)
// 直接替换原来的 Run lambda 定义

// 方案1: 编译期dispatch（推荐，如果buffer_coherence是编译期常量）
template <ck_tile::amd_buffer_coherence_enum BufferCoherence>
const auto RunWithBufferCoherence = [&](const auto has_hot_loop_,
                                        const auto tail_number_,
                                        const auto memory_operation_) {
    constexpr bool has_hot_loop_v   = has_hot_loop_.value;
    constexpr auto tail_number_v    = tail_number_.value;
    constexpr auto scheduler        = FlatmmConfig::Scheduler;
    constexpr auto memory_operation = memory_operation_.value;

    using CodegenPipelineProblem =
        std::conditional_t<MXFP4_Pipeline,
                           ck_tile::F16xMXF4FlatmmPipelineProblem<ADataType,
                                                                  BDataType,
                                                                  AccDataType,
                                                                  CodegenFlatmmShape,
                                                                  CodegenGemmTraits,
                                                                  scheduler,
                                                                  has_hot_loop_v,
                                                                  tail_number_v,
                                                                  BufferCoherence>,  // <-- 这里使用模板参数
                           ck_tile::FlatmmPipelineProblem<ADataType,
                                                          BDataType,
                                                          AccDataType,
                                                          CodegenFlatmmShape,
                                                          CodegenGemmTraits,
                                                          scheduler,
                                                          has_hot_loop_v,
                                                          tail_number_v,
                                                          BufferCoherence>>;  // <-- 这里使用模板参数

    constexpr int BlockedXDLN_PerWarp =
        (MXFP4_Pipeline || (moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up))
            ? 2
            : 1;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                         ComputeDataType,
                                         DsDatatype,
                                         AccDataType,
                                         CDataType,
                                         DsLayout,
                                         ELayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         FlatmmConfig::M_Warp,
                                         FlatmmConfig::N_Warp,
                                         FlatmmConfig::M_Warp_Tile,
                                         FlatmmConfig::N_Warp_Tile,
                                         FlatmmConfig::K_Warp_Tile,
                                         CodegenPipelineProblem::TransposeC,
                                         BlockedXDLN_PerWarp>>;

    using CodegenFlatmmPipeline =
        std::conditional_t<MXFP4_Pipeline,
                           ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>,
                           ck_tile::FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>>;

    using Kernel = ck_tile::MoeFlatmmKernel<TilePartitioner,
                                            CodegenFlatmmPipeline,
                                            GemmEpilogue,
                                            moe_kind,
                                            FusedActivation>;

    const dim3 grids  = Kernel::GridSize(args);
    const dim3 blocks = Kernel::BlockSize();

    auto kargs = Kernel::MakeKargs(args.p_sorted_token_ids,
                                    args.p_sorted_expert_ids,
                                    args.p_max_token_id,
                                    args.p_sorted_expert_weights,
                                    args.a_ptr,
                                    args.b_ptr,
                                    args.expert_bias.ptr,
                                    nullptr,
                                    args.e_ptr,
                                    args.NumExperts,
                                    args.NumTokens,
                                    args.TopK,
                                    args.N,
                                    args.K,
                                    args.stride_A,
                                    args.stride_B,
                                    args.stride_E,
                                    K_split,
                                    args.k_batch,
                                    args.n_padded_zeros,
                                    args.k_padded_zeros,
                                    args.scale_m,
                                    args.scale_n,
                                    args.expert_bias);

    ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
};

// 使用示例（替换原来的 RunSplitk 调用）:
// 在 RunSplitk 中调用:
const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
    // 根据配置选择buffer coherence类型
    constexpr auto buffer_coherence = FlatmmConfig::BufferCoherence; // 需要在Config中添加这个字段
    
    if(args.k_batch == 1)
    {
        RunWithBufferCoherence<buffer_coherence>(
            has_hot_loop_,
            tail_number_,
            ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        RunWithBufferCoherence<buffer_coherence>(
            has_hot_loop_,
            tail_number_,
            ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::atomic_add>{});
    }
};

// ================================================================================
// 方案2: 运行时dispatch（如果buffer_coherence需要运行时决定）
// ================================================================================

const auto DispatchBufferCoherenceRuntime = [&](const auto has_hot_loop_,
                                                const auto tail_number_,
                                                const auto memory_operation_,
                                                ck_tile::amd_buffer_coherence_enum buffer_coherence) 
{
    switch(buffer_coherence)
    {
    case ck_tile::amd_buffer_coherence_enum::WAVE_NT1:
        RunWithBufferCoherence<ck_tile::amd_buffer_coherence_enum::WAVE_NT1>(
            has_hot_loop_, tail_number_, memory_operation_);
        break;
    case ck_tile::amd_buffer_coherence_enum::SYSTEM_NT1:
        RunWithBufferCoherence<ck_tile::amd_buffer_coherence_enum::SYSTEM_NT1>(
            has_hot_loop_, tail_number_, memory_operation_);
        break;
    case ck_tile::amd_buffer_coherence_enum::DEFAULT:
        RunWithBufferCoherence<ck_tile::amd_buffer_coherence_enum::DEFAULT>(
            has_hot_loop_, tail_number_, memory_operation_);
        break;
    default:
        throw std::runtime_error("Unsupported buffer coherence type!");
    }
};

// 运行时dispatch的使用示例:
const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
    // 从args或其他地方获取运行时的buffer_coherence值
    auto buffer_coherence = args.buffer_coherence; // 假设args中有这个字段
    
    if(args.k_batch == 1)
    {
        DispatchBufferCoherenceRuntime(
            has_hot_loop_,
            tail_number_,
            ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{},
            buffer_coherence);
    }
    else
    {
        DispatchBufferCoherenceRuntime(
            has_hot_loop_,
            tail_number_,
            ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::atomic_add>{},
            buffer_coherence);
    }
};

// ================================================================================
// 需要在FlatmmConfig中添加的字段（如果使用方案1）：
// ================================================================================
/*
template <...>
struct FlatmmConfig {
    // ... 其他字段
    
    // 添加 buffer coherence 配置
    static constexpr ck_tile::amd_buffer_coherence_enum BufferCoherence = 
        ck_tile::amd_buffer_coherence_enum::WAVE_NT1;  // 或其他值
};
*/

// ================================================================================
// 需要在 MoeFlatmmHostArgs 中添加的字段（如果使用方案2）：
// ================================================================================
/*
template <...>
struct MoeFlatmmHostArgs {
    // ... 其他字段
    
    // 添加运行时 buffer coherence 参数
    ck_tile::amd_buffer_coherence_enum buffer_coherence = 
        ck_tile::amd_buffer_coherence_enum::WAVE_NT1;
};
*/


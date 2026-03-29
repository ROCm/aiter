// TurboQuant INT4 FlatMM: bf16 A × pk_int4_t B → bf16 C
// Uses CK flatmm pipeline with constexpr Lloyd-Max codebook dequant.
//
// KEY DESIGN: bypasses FlatmmKernel because it assumes BDataType=bf16 for stride
// computation, but our B data is pk_int4_t (1 byte, not 2). Instead, we launch
// a custom kernel that sets up windows with correct types and calls the pipeline.
#ifdef USE_ROCM
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <hip/hip_runtime.h>
#include "ck_tile/core_hip.hpp"
#include "ck_tile/host/kernel_launch_hip.hpp"
#include "ck_tile/ops/gemm_hip.hpp"
#include "ck_tile/ops/epilogue_hip.hpp"
#include "ck_tile/ops/flatmm_hip.hpp"
#include "ck_tile/ops/flatmm/pipeline/int4_flatmm_pipeline_agmem_bgmem_creg_v1_hip.hpp"

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

namespace {
using namespace ck_tile;

using AType = bf16_t;
using BType = pk_int4_t;  // the ACTUAL storage type (1 byte = 2 int4)
using AccType = float;
using CType = bf16_t;
using ALay = tensor_layout::gemm::RowMajor;
using BLay = tensor_layout::gemm::ColumnMajor;
using CLay = tensor_layout::gemm::RowMajor;

struct Int4Config
{
    // Smaller N_Tile for better CU occupancy with small M
    static constexpr index_t M_Tile = 16, N_Tile = 64, K_Tile = 128;
    static constexpr index_t M_Warp = 1, N_Warp = 4, K_Warp = 1;
    static constexpr index_t M_Warp_Tile = 16, N_Warp_Tile = 16, K_Warp_Tile = 32;
    static constexpr bool kPadM = true, kPadN = true, kPadK = true;
    static constexpr bool TransposeC = false, UseStructuredSparsity = false, DoubleSmemBuffer = false;
    static constexpr int kBlockPerCu = 1, NumWaveGroups = 1;
    static constexpr auto Scheduler = GemmPipelineScheduler::Default;
};

using Shape = TileGemmShape<
    sequence<Int4Config::M_Tile, Int4Config::N_Tile, Int4Config::K_Tile>,
    sequence<Int4Config::M_Warp, Int4Config::N_Warp, Int4Config::K_Warp>,
    sequence<Int4Config::M_Warp_Tile, Int4Config::N_Warp_Tile, Int4Config::K_Warp_Tile>>;

using Traits = TileGemmUniversalTraits<true, true, true, false,
    ALay, BLay, CLay, false, false, false, 1, true>;

// Pipeline problem: base BDataType=bf16 for MFMA, QuantType=pk_int4_t for loads
template <bool HasHotLoop, TailNumber TailNum>
using PP = F16xInt4FlatmmPipelineProblem<AType, BType, AccType,
    Shape, Traits, Int4Config::Scheduler, HasHotLoop, TailNum>;

template <bool HasHotLoop, TailNumber TailNum>
using Pipeline = F16xInt4FlatmmPipelineAGmemBGmemCRegV1<PP<HasHotLoop, TailNum>>;

using Part = GemmSpatiallyLocalTilePartitioner<Shape, 8, 4>;
using Policy = F16xInt4FlatmmPipelineAgBgCrPolicy;

using Epi = CShuffleEpilogue<CShuffleEpilogueProblem<
    AType, AType, tuple<>, AccType, CType, tuple<>, CLay,
    element_wise::PassThrough,
    Part::MPerBlock, Part::NPerBlock,
    Int4Config::M_Warp, Int4Config::N_Warp,
    Int4Config::M_Warp_Tile, Int4Config::N_Warp_Tile,
    Int4Config::K_Warp_Tile, false, 1, false, 1, false>>;

// Custom kernel: creates B window with pk_int4_t (uint8) strides, not bf16 strides
template <bool HasHotLoop, TailNumber TailNum>
__global__ void int4_flatmm_kernel(
    const AType* __restrict__ a_ptr,
    const uint8_t* __restrict__ b_ptr,
    CType* __restrict__ c_ptr,
    const float* __restrict__ norms_ptr,  // (N, n_groups) or nullptr
    index_t M, index_t N, index_t K,
    index_t stride_A, index_t stride_C,
    index_t norms_stride_n,  // stride along N for norms (= n_groups)
    float inv_sqrt_gs)
{
    using Pipe = Pipeline<HasHotLoop, TailNum>;
    constexpr index_t kMPerBlock = Shape::kM;
    constexpr index_t kNPerBlock = Shape::kN;
    constexpr index_t kKPerBlock = Shape::kK;
    constexpr index_t flatNPerWarp = Pipe::flatNPerWarp;
    constexpr index_t flatKPerWarp = Pipe::flatKPerWarp;

    const auto [iM, iN] = Part{M, N}.GetOutputTileIndex(blockIdx.x);
    const index_t i_m = iM * Part::MPerBlock;
    const index_t i_n = iN * Part::NPerBlock;

    // A window: standard row-major (M, K) bf16
    auto a_view = make_naive_tensor_view<address_space_enum::global>(
        a_ptr,
        make_tuple(M, K),
        make_tuple(stride_A, 1),
        number<Pipe::GetVectorSizeA()>{},
        number<1>{});
    auto a_pad_view = pad_tensor_view(
        a_view,
        make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
        sequence<true, true>{});
    auto a_window = make_tile_window(
        a_pad_view,
        make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
        {i_m, 0});

    // B window: uint8_t buffer in preshuffle flat layout
    // Same view as mixed_prec: (kFlatN, kFlatK) with stride (kFlatK, 1)
    const index_t K_half = K / 2;
    constexpr index_t NWT = Shape::WarpTile::at(number<1>{});  // 16
    const index_t kFlatK = K_half * NWT;  // Total K bytes per N-block
    const index_t kFlatN = N * K_half / kFlatK;  // = N / NWT

    // Use pk_int4_t (same as uint8_t) for the view — CK handles buffer_load for this type.
    const auto* b_typed = reinterpret_cast<const pk_int4_t*>(b_ptr);
    auto b_view = make_naive_tensor_view<address_space_enum::global>(
        b_typed,
        make_tuple(kFlatN, kFlatK),
        make_tuple(kFlatK, 1),
        number<Pipe::GetVectorSizeB()>{},
        number<1>{});
    auto b_window = make_tile_window(
        b_view,
        make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
        {static_cast<int>(i_n / NWT), 0});

    // SMEM
    __shared__ char smem_ping[Pipe::GetSmemSize()];
    __shared__ char smem_pong[Pipe::GetSmemSize()];

    const index_t num_loop = Part::GetLoopNum(K);

    // Run pipeline (with per-group norms if provided)
    auto c_block_tile = Pipe{}(a_window, b_window,
                               norms_ptr, norms_stride_n, i_n, inv_sqrt_gs,
                               num_loop, smem_ping, smem_pong);

    // Epilogue: write C
    auto c_view = make_naive_tensor_view<address_space_enum::global>(
        c_ptr,
        make_tuple(M, N),
        make_tuple(stride_C, 1),
        number<Pipe::GetVectorSizeC()>{},
        number<1>{});
    auto c_pad_view = pad_tensor_view(
        c_view,
        make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
        sequence<true, true>{});
    auto c_window = make_tile_window(
        c_pad_view,
        make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{}),
        {i_m, i_n});

    Epi{}(c_window, c_block_tile, c_window, smem_ping);
}

} // namespace

torch::Tensor turboquant_int4_flatmm_gemm(
    torch::Tensor x_rot, torch::Tensor packed_idx,
    torch::Tensor norms, int group_size)
{
    auto x_bf16 = (x_rot.scalar_type() == torch::kBFloat16 && x_rot.is_contiguous())
                 ? x_rot : x_rot.to(torch::kBFloat16).contiguous();
    const int M = x_bf16.size(0), K = x_bf16.size(1);
    const int N = packed_idx.size(0);
    auto output = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(x_rot.device()));
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    // Norms: (N, n_groups) float32
    auto norms_f32 = norms.to(torch::kFloat32).contiguous();
    const float* norms_ptr = norms_f32.data_ptr<float>();
    const int n_groups = K / group_size;
    const int norms_stride = (norms.dim() == 2) ? n_groups : 0;
    const float inv_sqrt_gs = 1.0f / sqrtf(static_cast<float>(group_size));

    constexpr int KT = Int4Config::K_Tile;
    const int num_loop = (K + KT - 1) / KT;
    const dim3 grid(Part::GridSize(M, N), 1, 1);
    constexpr int block_size = 256;

#define LAUNCH(HL, TN) \
    int4_flatmm_kernel<HL, TN><<<grid, block_size, 0, stream.stream()>>>( \
        (const AType*)x_bf16.data_ptr(), packed_idx.data_ptr<uint8_t>(), \
        (CType*)output.data_ptr(), norms_ptr, M, N, K, K, N, norms_stride, inv_sqrt_gs)

    if(num_loop > 2) {
        if(num_loop % 2 == 0) { LAUNCH(true, TailNumber::Even); }
        else                  { LAUNCH(true, TailNumber::Odd); }
    } else {
        if(num_loop % 2 == 0) { LAUNCH(false, TailNumber::Even); }
        else                  { LAUNCH(false, TailNumber::Odd); }
    }
#undef LAUNCH

    return output;
}

#endif

#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("turboquant_int4_flatmm_gemm", &turboquant_int4_flatmm_gemm,
          "TurboQuant INT4 FlatMM with per-group norms");
}

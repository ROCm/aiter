// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef __gfx908__
// // Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below
// just for gfx908 and not for others
// // below lines enable hip float to half conversion which are disabled by
// default in hip_fp16.h #undef __HIP_NO_HALF_OPERATORS__ #undef
// __HIP_NO_HALF_CONVERSIONS__ #endif

#include "hipbsolgemm.cuh"

// #include <rocblas/rocblas.h>

// #ifdef USE_ROCM
// #define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 +
// ROCBLAS_VERSION_MINOR) #define USE_GEMM_FLAGS_FP16_ALT_IMPL
// (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242) #endif

// #ifdef __HIP_PLATFORM_HCC__
// 	#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 +
// ROCBLAS_VERSION_MINOR) 	#define USE_GEMM_FLAGS_FP16_ALT_IMPL
// (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242) 	#if USE_GEMM_FLAGS_FP16_ALT_IMPL
// 	  #ifdef ROCM_BACKWARD_PASS_GUARD
// 		flag = at::BackwardPassGuard::is_backward_pass() ?
// rocblas_gemm_flags_fp16_alt_impl : 0; 	  #endif 	#endif #endif

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                    \
    if(error != HIPBLAS_STATUS_SUCCESS)               \
    {                                                 \
        fprintf(stderr,                               \
                "hipBLAS error: '%s'(%d) at %s:%d\n", \
                hipblasStatusToString(error),         \
                error,                                \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

namespace {
/*thread_local*/ hipStream_t weight_stream;
// BUG: DLM has event and stream on different devices error
// In multi-GPU scenerio, do names defined in this namespace exist on all
// devices? C++ keyword: thread_local <- maybe this can help?
/*thread_local*/ hipEvent_t event;

// hipBLASLt
hipblasLtHandle_t hipblaslt_handle;
hipblasLtMatmulPreference_t preference;
size_t workspace_size = 2 * 128 * 1024 * 1024;
// uint64_t workspace_size = 0;
void* d_workspace;
int request_solutions = 1;
int returnedAlgoCount = 0;

struct MatMulConfig
{
    hipblasOperation_t op_A;
    hipblasOperation_t op_B;
    int M;
    int N;
    int K;
    hipDataType dtype;

    friend auto operator<(const MatMulConfig& left, const MatMulConfig& right) -> bool
    {
        return std::tie(left.op_A, left.op_B, left.M, left.N, left.K, left.dtype) <
               std::tie(right.op_A, right.op_B, right.M, right.N, right.K, right.dtype);
    }
};

// std::map<std::tuple<int, int, int, int, int, int>,
// std::vector<hipblasLtMatmulHeuristicResult_t>> heuristic_map;
std::map<MatMulConfig, hipblasLtMatmulHeuristicResult_t> heuristic_map;

hipEvent_t start, stop;
int bench_iters{1};
int warmup_iters{1};

bool cout_print = false;

torch::Tensor dTensor;

std::map<at::ScalarType, hipDataType> dtype_map{{at::kHalf, HIP_R_16F},
                                                {at::kBFloat16, HIP_R_16BF},
                                                {at::kFloat, HIP_R_32F},
                                                {at::kChar, HIP_R_8I}
#ifdef ENABLE_TORCH_FP8
                                                ,
                                                {at::kFloat8_e4m3fnuz, HIP_R_8F_E4M3_FNUZ},
                                                {at::kFloat8_e4m3fn, HIP_R_8F_E4M3}
#endif
};

// std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
} // namespace

// find all hipblaslt solutions for given gemm problem
std::vector<int> hipblasLtMatmul_findallsols_wrapper(hipblasLtHandle_t handle,
                                                     hipblasOperation_t op_A,
                                                     hipblasOperation_t op_B,
                                                     int m,
                                                     int n,
                                                     int k,
                                                     const void* alpha,
                                                     const void* a,
                                                     int lda,
                                                     const void* b,
                                                     int ldb,
                                                     const void* beta,
                                                     void* c,
                                                     int ldc,
                                                     const void* bias,
                                                     hipDataType intype,
                                                     hipDataType outtype,
                                                     const void* scaleA,
                                                     const void* scaleB,
                                                     const void* scaleC,
                                                     hipStream_t& stream,
                                                     bool use_rowwise = false,
                                                     bool bpreshuffle = false)
{
    int flag{0};
    hipblasLtMatrixLayout_t matA, matB, matC;
    hipblasLtMatmulDesc_t matmul;
    if(op_A == HIPBLAS_OP_N)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, m, k, lda));
    }
    else
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, k, m, lda));
    }
#if (HIPBLASLT_VERSION_MAJOR >= 1) || \
    (HIPBLASLT_VERSION_MAJOR == 0 && HIPBLASLT_VERSION_MINOR >= 15)
    if(bpreshuffle)
    {
        hipblasLtOrder_t orderA;
        if(scaleA != nullptr)
            orderA = HIPBLASLT_ORDER_COL16_4R16;
        else
            orderA = HIPBLASLT_ORDER_COL16_4R8;

        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
    }
#else
    if(bpreshuffle)
    {
        std::cerr << "Warning: hipblasLt version lower than 0.15 does not support "
                     "bpreshuffle. Please upgrade hipblasLt."
                  << std::endl;
    }
#endif
    if(op_B == HIPBLAS_OP_N)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, k, n, ldb));
    }
    else
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, n, k, ldb));
    }
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matC, outtype, m, n, ldc));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));

    if(bias)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
        auto epilogue = HIPBLASLT_EPILOGUE_BIAS;
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    }

    if(scaleA != nullptr)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA, sizeof(scaleA)));
#if (HIPBLASLT_VERSION_MAJOR >= 1)
        if(use_rowwise)
        {
            auto scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(scale_mode_a)));
        }
#endif
    }
    if(scaleB != nullptr)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB)));
#if (HIPBLASLT_VERSION_MAJOR >= 1)
        if(use_rowwise)
        {
            auto scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(scale_mode_b)));
        }
#endif
    }
#if (HIPBLASLT_VERSION_MAJOR < 1)
    TORCH_CHECK(!(use_rowwise), "Rowwise scaling requires hipBLASLt >= 1.0");
#endif
    if(scaleC != nullptr)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleC, sizeof(scaleC)));
    }

    // std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(10);
    // CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
    //     handle, matmul, matA, matB, matC, matC,
    //     preference, 10, heuristicResult.data(), &returnedAlgoCount));
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLAS_ERROR(hipblaslt_ext::getAllAlgos(handle,
                                                   hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                   op_A,
                                                   op_B,
                                                   intype,
                                                   intype,
                                                   outtype,
                                                   outtype,
                                                   HIPBLAS_COMPUTE_32F,
                                                   heuristicResult));

    std::vector<int> algoIndex;
    int returned_algo_count = heuristicResult.size();
    // for (int i = 0; i < returnedAlgoCount; i++) {
    for(int i = 0; i < returned_algo_count; i++)
    {
        auto algo                 = heuristicResult[i].algo;
        size_t ret_workspace_size = 0;
        auto status               = hipblaslt_ext::matmulIsAlgoSupported(
            handle, matmul, alpha, matA, matB, beta, matC, matC, algo, ret_workspace_size);
        if(status == HIPBLAS_STATUS_SUCCESS)
        {
            if(ret_workspace_size < workspace_size)
            {
                algoIndex.push_back(hipblaslt_ext::getIndexFromAlgo(algo));
            }
        }
    }

    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    return algoIndex;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * hipBLASLt GEMM call
 */
hipblasStatus_t hipblasLtMatmul_sol_wrapper(hipblasLtHandle_t handle,
                                            hipblasOperation_t op_A,
                                            hipblasOperation_t op_B,
                                            int m,
                                            int n,
                                            int k,
                                            const void* alpha,
                                            const void* a,
                                            int lda,
                                            const void* scaleA,
                                            const void* b,
                                            int ldb,
                                            const void* scaleB,
                                            const void* beta,
                                            void* c,
                                            int ldc,
                                            const void* scaleC,
                                            const void* bias,
                                            hipDataType intype,
                                            hipDataType outtype,
                                            hipStream_t& stream,
                                            int solution_index = -1,
                                            bool bpreshuffle   = false,
                                            bool use_rowwise   = false)
{
    // TODO: flag is not supported for hipblasLt yet
    int flag{0};
    // if (dtype == HIPBLAS_R_16F) {
    //  use fp16 alt impl for MI200
    //  https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    // flag = rocblas_gemm_flags_fp16_alt_impl;
    //}

    // nvtxRangePushA("hipBLASLt variables creation");
    hipblasLtMatrixLayout_t matA, matB, matC;
    hipblasLtMatmulDesc_t matmul;
    if(op_A == HIPBLAS_OP_N)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, m, k, lda));
    }
    else
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, intype, k, m, lda));
    }
#if (HIPBLASLT_VERSION_MAJOR >= 1) || \
    (HIPBLASLT_VERSION_MAJOR == 0 && HIPBLASLT_VERSION_MINOR >= 15)
    if(bpreshuffle)
    {
        hipblasLtOrder_t orderA;
        if(scaleA != nullptr)
            orderA = HIPBLASLT_ORDER_COL16_4R16;
        else
            orderA = HIPBLASLT_ORDER_COL16_4R8;

        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
            matA, HIPBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));
    }
#else
    if(bpreshuffle)
    {
        std::cerr << "Warning: hipblasLt version lower than 0.15 does not support "
                     "bpreshuffle. Please upgrade hipblasLt."
                  << std::endl;
    }
#endif
    if(op_B == HIPBLAS_OP_N)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, k, n, ldb));
    }
    else
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, intype, n, k, ldb));
    }
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matC, outtype, m, n, ldc));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));

    // Set scale attributes with proper rowwise support
    // hipBLASLt >= 1.0 supports rowwise scaling via OUTER_VEC mode
    if(scaleA != nullptr)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA, sizeof(scaleA)));
#if (HIPBLASLT_VERSION_MAJOR >= 1)
        if(use_rowwise)
        {
            // Set the scale mode to OUTER_VEC for rowwise scaling on A
            auto scale_mode_a = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a, sizeof(scale_mode_a)));
        }
#endif
    }
    if(scaleB != nullptr)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB)));
#if (HIPBLASLT_VERSION_MAJOR >= 1)
        if(use_rowwise)
        {
            // Set the scale mode to OUTER_VEC for rowwise scaling on B
            auto scale_mode_b = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
                matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b, sizeof(scale_mode_b)));
        }
#endif
    }
#if (HIPBLASLT_VERSION_MAJOR < 1)
    TORCH_CHECK(!(use_rowwise), "Rowwise scaling requires hipBLASLt >= 1.0");
#endif

    if(scaleC != nullptr)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &scaleC, sizeof(scaleC)));
    }
    if(bias)
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(void*)));
        auto epilogue = HIPBLASLT_EPILOGUE_BIAS;
        static_assert(sizeof(epilogue) == sizeof(int32_t));
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    }
    // nvtxRangePop();
    //  if heuristic does not exist in the map, do search and push into the map
    // auto gemm_key { MatMulConfig { op_A, op_B, m, n, k, dtype } };
    // if (heuristic_map.count(gemm_key) <= 0) {
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(1);
    if(solution_index < 0)
    {
        // nvtxRangePushA("hipblasLtMatmulAlgoGetHeuristic");
        // std::cout
        //     << "Warning! HipbSolId Gemm Fallback Path used for solution index <0"
        //     << std::endl;
        if(cout_print)
        {
            std::cout << (op_A == HIPBLAS_OP_N ? "N" : "T") << (op_B == HIPBLAS_OP_N ? "N" : "T")
                      << " (" << m << ", " << n << ", " << k << "), dtype: " << intype
                      << ", (lda, ldb, ldc): (" << lda << ", " << ldb << ", " << ldc << "), "
                      << std::endl;
        }
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                            matmul,
                                                            matA,
                                                            matB,
                                                            matC,
                                                            matC,
                                                            preference,
                                                            request_solutions,
                                                            heuristicResult.data(),
                                                            &returnedAlgoCount));
        if((returnedAlgoCount != request_solutions) && cout_print)
        {
            std::cout << "less solution found! request: " << request_solutions
                      << ", found: " << returnedAlgoCount << std::endl;
        }
    }
    else
    {
        std::vector<int> algoIndex(1);
        algoIndex[0] = solution_index;
        CHECK_HIPBLAS_ERROR(hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResult));
    }

    hipblasStatus_t status = hipblasLtMatmul(handle,
                                             matmul,
                                             alpha,
                                             a,
                                             matA,
                                             b,
                                             matB,
                                             beta,
                                             c,
                                             matC,
                                             c,
                                             matC,
                                             &heuristicResult[0].algo,
                                             d_workspace,
                                             workspace_size,
                                             stream);

    CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));

    return status;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////

enum class ScalingType
{
    TensorWise, // Both A and B use per-tensor scaling (scalar)
    RowWise,    // Both A and B use rowwise scaling
    Error
};

// Helper function to determine scaling type
static ScalingType get_scaling_type(const torch::Tensor& scale_a,
                                    const torch::Tensor& scale_b,
                                    int64_t dim_m,
                                    int64_t dim_n)
{
    // Both Per-Tensor and Row-wise scaling expect fp32 tensors
    TORCH_CHECK(scale_a.scalar_type() == at::kFloat && scale_b.scalar_type() == at::kFloat,
                "Both scale_a and scale_b must be float (fp32) tensors.");

    // Check if scales are scalars (per-tensor)
    bool scale_a_is_scalar = (scale_a.numel() == 1);
    bool scale_b_is_scalar = (scale_b.numel() == 1);

    // Case 1: Both per-tensor (scalar)
    if(scale_a_is_scalar && scale_b_is_scalar)
    {
        return ScalingType::TensorWise;
    }

    // For non-scalar scaling, enforce 2D input tensors
    TORCH_CHECK(scale_a.dim() == 2 && scale_b.dim() == 2,
                "For non-TensorWise scaling, scale tensors must be 2-dimensional, "
                "but got scale_a.dim()=",
                scale_a.dim(),
                " and scale_b.dim()=",
                scale_b.dim());

    // Check if scales match rowwise pattern
    bool scale_a_is_rowwise = (scale_a.size(0) == dim_m && scale_a.size(1) == 1);
    bool scale_b_is_rowwise = (scale_b.size(0) == 1 && scale_b.size(1) == dim_n);

#if (HIPBLASLT_VERSION_MAJOR >= 1) || \
    (HIPBLASLT_VERSION_MAJOR == 0 && HIPBLASLT_VERSION_MINOR >= 15)
    // Case 2: Both rowwise
    if(scale_a_is_rowwise && scale_b_is_rowwise)
    {
        TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous(),
                    "Both scale_a and scale_b must be contiguous for RowWise scaling.");
        return ScalingType::RowWise;
    }
#else
    // Older hipBLASLt versions don't support rowwise
    if(scale_a_is_rowwise || scale_b_is_rowwise)
    {
        TORCH_CHECK(false, "Per-row scaling is not supported for this hipBLASLt version!");
        return ScalingType::Error;
    }
#endif

    // If we reach here, the input doesn't match any valid scaling type
    TORCH_CHECK(false,
                "Invalid scaling configuration. Supported modes:\n"
                "  - TensorWise: both scales are scalars (1 element) of shape (1,1)\n"
                "  - RowWise: scale_a=(",
                dim_m,
                ", 1), scale_b=(1, ",
                dim_n,
                ")\n"
                "Got scale_a.size()=(",
                scale_a.size(0),
                ", ",
                scale_a.size(1),
                ") and "
                "scale_b.size()=(",
                scale_b.size(0),
                ", ",
                scale_b.size(1),
                ")");

    return ScalingType::Error;
}

torch::Tensor hipb_mm(const torch::Tensor& mat1,
                      const torch::Tensor& mat2,
                      const int solution_index,
                      std::optional<torch::Tensor> bias,
                      std::optional<c10::ScalarType> out_dtype,
                      std::optional<torch::Tensor> scaleA,
                      std::optional<torch::Tensor> scaleB,
                      std::optional<torch::Tensor> scaleOut,
                      std::optional<bool> bpreshuffle)
{
    bool bpreshuffle_flag = bpreshuffle.value_or(false);

    int version;
    hipblasLtGetVersion(hipblaslt_handle, &version);
    TORCH_CHECK(!bpreshuffle_flag || version >= 1500,
                " to use bpreshuffle feature, hipblaslt version should be at least 1500.");

    auto mat1_strides{mat1.strides()};
    auto mat2_strides{mat2.strides()};
    auto mat1_sizes{mat1.sizes()};
    auto mat2_sizes{mat2.sizes()};

    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(mat1.dtype() == mat2.dtype(),
                "expected mat1 and mat2 to have the same dtype, but got: ",
                mat1.dtype(),
                " != ",
                mat2.dtype());
    TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");

    auto inDtype{mat1.options().dtype().toScalarType()};
    auto outDtype{out_dtype.has_value() ? out_dtype.value() : inDtype};
    auto options{at::TensorOptions().dtype(outDtype).device(at::kCUDA)};
    auto result{torch::empty({mat1_sizes[0], mat2_sizes[1]}, options)};

    bool transpose_result = true;
    bool transpose_mat1;
    bool transpose_mat2;
    if((mat2_strides[0] == 1) && (mat2_strides[1] >= std::max<int64_t>(1, mat2_sizes[0])))
    {
        transpose_mat2 = false;
    }
    else if((mat2_strides[1] == 1) && (mat2_strides[0] >= std::max<int64_t>(1, mat2_sizes[1])))
    {
        transpose_mat2 = true;
    }
    else
    {
        assert(false && "unusual strides detected, may need to clone a contiguous tensor");
    }
    if((mat1_strides[0] == 1) && (mat1_strides[1] >= std::max<int64_t>(1, mat1_sizes[0])))
    {
        transpose_mat1 = false;
    }
    else if((mat1_strides[1] == 1) && (mat1_strides[0] >= std::max<int64_t>(1, mat1_sizes[1])))
    {
        transpose_mat1 = true;
    }
    else
    {
        assert(false && "unusual strides detected, may need to clone a contiguous tensor");
    }

    if(transpose_result)
    {
        bool tmp       = transpose_mat1;
        transpose_mat1 = !transpose_mat2;
        transpose_mat2 = !tmp;
        mat1_strides   = mat2.strides();
        mat2_strides   = mat1.strides();
        mat1_sizes     = mat2.sizes();
        mat2_sizes     = mat1.sizes();
    }

    float one{1.0f};
    float zero{0.0f};
    int64_t m         = mat1_sizes[transpose_result ? 1 : 0];
    int64_t k         = mat1_sizes[transpose_result ? 0 : 1];
    int64_t n         = mat2_sizes[transpose_result ? 0 : 1];
    int64_t mat1_ld   = mat1_strides[(transpose_mat1 == transpose_result) ? 1 : 0];
    int64_t mat2_ld   = mat2_strides[(transpose_mat2 == transpose_result) ? 1 : 0];
    int64_t result_ld = result.stride(transpose_result ? 0 : 1);

    void *d_scaleA = nullptr, *d_scaleB = nullptr, *d_scaleOut = nullptr;
    bool use_rowwise = false;

    // Determine scaling type if scales are provided
    // The API expects scaleA for mat1 and scaleB for mat2 in the original orientation
    if(scaleA.has_value() && scaleB.has_value())
    {
        // Determine scaling type based on original input dimensions (before transpose_result)
        // The scales should match mat1 (m_orig x k) and mat2 (k x n_orig)
        int64_t m_orig = mat1.sizes()[0];
        int64_t n_orig = mat2.sizes()[1];
        int64_t k_orig = mat1.sizes()[1];

        // Determine scaling type - will throw error for unsupported configurations
        ScalingType scaling_type = get_scaling_type(scaleA.value(), scaleB.value(), m_orig, n_orig);

        // Enable rowwise scaling based on detected scaling type
        // Note: hipBLASLt only supports uniform scaling modes (both per-tensor OR both rowwise)
        if(scaling_type == ScalingType::RowWise)
        {
            // Both matrices use rowwise scaling
            // Rowwise scaling is only supported for FP8 input with BFloat16 output
            // For bpreshuffle (swizzled layout), proper alignment is required
            // Note: m can be any value >= 1, but n should be >= 16 and aligned
            TORCH_CHECK(outDtype == at::kBFloat16,
                        "hipblaslt rowwise scaled_mm only supports BFloat16 output but got ",
                        outDtype);
            TORCH_CHECK(inDtype == at::kFloat8_e4m3fn || inDtype == at::kFloat8_e4m3fnuz,
                        "hipblaslt rowwise scaled_mm only supports FP8 input but got ",
                        inDtype);
            TORCH_CHECK(n_orig >= 16, "hipblaslt rowwise scaled_mm requires n >= 16");
            TORCH_CHECK(n_orig % 16 == 0,
                        "hipblaslt rowwise scaled_mm requires n to be divisible by 16");
            // TORCH_CHECK(m_orig % 16 == 0 || m_orig < 16, "hipblaslt rowwise scaled_mm requires m
            // to be divisible by 16 or less than 16, but got ", m_orig);
            TORCH_CHECK(k_orig % 16 == 0,
                        "hipblaslt rowwise scaled_mm requires k to be divisible by 16");

            use_rowwise = true;
        }

        d_scaleA = static_cast<void*>(scaleA.value().data_ptr());
        d_scaleB = static_cast<void*>(scaleB.value().data_ptr());
    }
    else
    {
        if(scaleA.has_value())
        {
            d_scaleA = static_cast<void*>(scaleA.value().data_ptr());
        }
        if(scaleB.has_value())
        {
            d_scaleB = static_cast<void*>(scaleB.value().data_ptr());
        }
    }

    if(scaleOut.has_value())
    {
        d_scaleOut = static_cast<void*>(scaleOut.value().data_ptr());
    }

    auto hipblasInType  = dtype_map.at(inDtype);
    auto hipblasOutType = dtype_map.at(outDtype);

    void* ptrA{static_cast<void*>((transpose_result ? mat2 : mat1).data_ptr())};
    void* ptrB{static_cast<void*>((transpose_result ? mat1 : mat2).data_ptr())};
    void* ptrC{static_cast<void*>(result.data_ptr())};
    if(transpose_result)
        std::swap(d_scaleA, d_scaleB);
    auto current_stream{torch::hip::getCurrentHIPStream().stream()};
    void* bias_ptr = bias.has_value() ? static_cast<void*>(bias.value().data_ptr()) : nullptr;

    CHECK_HIPBLAS_ERROR(hipblasLtMatmul_sol_wrapper(hipblaslt_handle,
                                                    transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                                    transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                                    m,
                                                    n,
                                                    k,
                                                    &one,
                                                    ptrA,
                                                    mat1_ld,
                                                    d_scaleA,
                                                    ptrB,
                                                    mat2_ld,
                                                    d_scaleB,
                                                    &zero,
                                                    ptrC,
                                                    result_ld,
                                                    d_scaleOut,
                                                    bias_ptr,
                                                    hipblasInType,
                                                    hipblasOutType,
                                                    current_stream,
                                                    solution_index,
                                                    bpreshuffle_flag,
                                                    use_rowwise));

    return result;
}

// find all hipblas solutions and return them to python land
std::vector<int> hipb_findallsols(const torch::Tensor& mat1,
                                  const torch::Tensor& mat2,
                                  std::optional<torch::Tensor> bias,
                                  std::optional<c10::ScalarType> out_dtype,
                                  std::optional<torch::Tensor> scaleA,
                                  std::optional<torch::Tensor> scaleB,
                                  std::optional<torch::Tensor> scaleC,
                                  bool bpreshuffle)
{
    auto mat1_strides{mat1.strides()};
    auto mat2_strides{mat2.strides()};
    auto mat1_sizes{mat1.sizes()};
    auto mat2_sizes{mat2.sizes()};
    TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(mat1.dtype() == mat2.dtype(),
                "expected mat1 and mat2 to have the same dtype, but got: ",
                mat1.dtype(),
                " != ",
                mat2.dtype());
    TORCH_CHECK(mat1_sizes[1] == mat2_sizes[0], "mat1 dim 1 must match mat2 dim 0");

    auto inType{mat1.options().dtype().toScalarType()};
    auto outType{out_dtype.has_value() ? out_dtype.value() : inType};

    auto options{at::TensorOptions().dtype(outType).device(at::kCUDA)};
    auto result{torch::empty({mat1_sizes[0], mat2_sizes[1]}, options)};
    bool transpose_result = true;
    bool transpose_mat1;
    bool transpose_mat2;
    if((mat2_strides[0] == 1) && (mat2_strides[1] >= std::max<int64_t>(1, mat2_sizes[0])))
    {
        transpose_mat2 = false;
    }
    else if((mat2_strides[1] == 1) && (mat2_strides[0] >= std::max<int64_t>(1, mat2_sizes[1])))
    {
        transpose_mat2 = true;
    }
    else
    {
        assert(false && "unusual strides detected, may need to clone a contiguous tensor");
    }
    if((mat1_strides[0] == 1) && (mat1_strides[1] >= std::max<int64_t>(1, mat1_sizes[0])))
    {
        transpose_mat1 = false;
    }
    else if((mat1_strides[1] == 1) && (mat1_strides[0] >= std::max<int64_t>(1, mat1_sizes[1])))
    {
        transpose_mat1 = true;
    }
    else
    {
        assert(false && "unusual strides detected, may need to clone a contiguous tensor");
    }
    if(transpose_result)
    {
        bool tmp       = transpose_mat1;
        transpose_mat1 = !transpose_mat2;
        transpose_mat2 = !tmp;
        mat1_strides   = mat2.strides();
        mat2_strides   = mat1.strides();
        mat1_sizes     = mat2.sizes();
        mat2_sizes     = mat1.sizes();
    }
    float one{1.0f};
    float zero{0.0f};
    int64_t m                  = mat1_sizes[transpose_result ? 1 : 0];
    int64_t k                  = mat1_sizes[transpose_result ? 0 : 1];
    int64_t n                  = mat2_sizes[transpose_result ? 0 : 1];
    int64_t mat1_ld            = mat1_strides[(transpose_mat1 == transpose_result) ? 1 : 0];
    int64_t mat2_ld            = mat2_strides[(transpose_mat2 == transpose_result) ? 1 : 0];
    int64_t result_ld          = result.stride(transpose_result ? 0 : 1);
    hipDataType hipblasInType  = dtype_map.at(inType);
    hipDataType hipblasOutType = dtype_map.at(outType);

    void* ptrA{static_cast<void*>((transpose_result ? mat2 : mat1).data_ptr())};
    void* ptrB{static_cast<void*>((transpose_result ? mat1 : mat2).data_ptr())};
    void* ptrC{static_cast<void*>(result.data_ptr())};
    auto current_stream{torch::hip::getCurrentHIPStream().stream()};

    auto bias_ptr = bias.has_value() ? static_cast<void*>(bias.value().data_ptr()) : nullptr;

    auto scaleA_ptr = scaleA.has_value() ? static_cast<void*>(scaleA.value().data_ptr()) : nullptr;

    auto scaleB_ptr = scaleB.has_value() ? static_cast<void*>(scaleB.value().data_ptr()) : nullptr;

    auto scaleC_ptr = scaleC.has_value() ? static_cast<void*>(scaleC.value().data_ptr()) : nullptr;

    bool use_rowwise = false;
    if(scaleA.has_value() && scaleB.has_value())
    {
        int64_t m_orig = mat1.sizes()[0];
        int64_t n_orig = mat2.sizes()[1];

        ScalingType scaling_type = get_scaling_type(scaleA.value(), scaleB.value(), m_orig, n_orig);

        if(scaling_type == ScalingType::RowWise)
        {
            use_rowwise = true;
        }
    }

    return hipblasLtMatmul_findallsols_wrapper(hipblaslt_handle,
                                               transpose_mat1 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               transpose_mat2 ? HIPBLAS_OP_T : HIPBLAS_OP_N,
                                               m,
                                               n,
                                               k,
                                               &one,
                                               ptrA,
                                               mat1_ld,
                                               ptrB,
                                               mat2_ld,
                                               &zero,
                                               ptrC,
                                               result_ld,
                                               bias_ptr,
                                               hipblasInType,
                                               hipblasOutType,
                                               scaleA_ptr,
                                               scaleB_ptr,
                                               scaleC_ptr,
                                               current_stream,
                                               use_rowwise,
                                               bpreshuffle);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////

void hipb_create_extension()
{
    // CHECK_HIP_ERROR(hipStreamCreate(&weight_stream));
    // CHECK_HIP_ERROR(hipEventCreateWithFlags(&event, cudaEventDisableTiming));

    // hipBLASLt
    CHECK_HIPBLAS_ERROR(hipblasLtCreate(&hipblaslt_handle));
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceCreate(&preference));
    CHECK_HIPBLAS_ERROR(
        hipblasLtMatmulPreferenceSetAttribute(preference,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_size,
                                              sizeof(workspace_size)));

    // CHECK_HIP_ERROR(hipEventCreate(&start));
    // CHECK_HIP_ERROR(hipEventCreate(&stop));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void hipb_destroy_extension()
{
    // CHECK_HIP_ERROR(hipStreamDestroy(weight_stream));
    // CHECK_HIP_ERROR(hipEventDestroy(event));

    // hipBLASLt
    CHECK_HIPBLAS_ERROR(hipblasLtDestroy(hipblaslt_handle));
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceDestroy(preference));
    CHECK_HIP_ERROR(hipFree(d_workspace));

    // CHECK_HIP_ERROR(hipEventDestroy(start));
    // CHECK_HIP_ERROR(hipEventDestroy(stop));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string getHipblasltKernelName(int solution_index)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult(1);
    std::vector<int> algoIndex(1);
    algoIndex[0] = solution_index;
    CHECK_HIPBLAS_ERROR(
        hipblaslt_ext::getAlgosFromIndex(hipblaslt_handle, algoIndex, heuristicResult));
    return hipblaslt_ext::getKernelNameFromAlgo(hipblaslt_handle, heuristicResult[0].algo);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hipb_create_extension", &hipb_create_extension, "create_extension");
    m.def("hipb_destroy_extension", &hipb_destroy_extension, "destroy_extension");
    m.def("hipb_mm",
          &hipb_mm,
          "hipb_mm",
          py::arg("mat1"),
          py::arg("mat2"),
          py::arg("solution_index"),
          py::arg("bias")        = std::nullopt,
          py::arg("out_dtype")   = std::nullopt,
          py::arg("scaleA")      = std::nullopt,
          py::arg("scaleB")      = std::nullopt,
          py::arg("scaleOut")    = std::nullopt,
          py::arg("bpreshuffle") = std::nullopt);
    m.def("hipb_findallsols",
          &hipb_findallsols,
          "hipb_findallsols",
          py::arg("mat1"),
          py::arg("mat2"),
          py::arg("bias")        = std::nullopt,
          py::arg("out_dtype")   = std::nullopt,
          py::arg("scaleA")      = std::nullopt,
          py::arg("scaleB")      = std::nullopt,
          py::arg("scaleC")      = std::nullopt,
          py::arg("bpreshuffle") = false);
    m.def("getHipblasltKernelName", &getHipblasltKernelName);
}

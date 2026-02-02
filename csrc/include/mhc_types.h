#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>

namespace cooperative_groups {
template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T, typename Op>
__device__ __forceinline__ T reduce(::cooperative_groups::thread_block_tile<32> tile, T val, Op op)
{
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = op(val, other);
    }
    return val;
}
}  // namespace cooperative_groups

namespace mhc {

using floatX = __hip_bfloat16;
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
using floatN = float;

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

#define CHECK_HIP(call) mhc::check_hip((call), __FILE__, __LINE__)
#define CHECK_CUBLAS(call) mhc::check_cublas((call), __FILE__, __LINE__)
#define CHECK_CUDA(call) CHECK_HIP(call)

#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t
#define cublasLtMatmulHeuristicResult_t hipblasLtMatmulHeuristicResult_t
#define cublasLtPointerMode_t hipblasLtPointerMode_t
#define cublasLtOrder_t hipblasLtOrder_t
#define cublasOperation_t hipblasOperation_t
#define cublasComputeType_t hipblasComputeType_t
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define cublasLtCreate hipblasLtCreate
#define cublasLtDestroy hipblasLtDestroy
#define cublasLtMatmulDescCreate hipblasLtMatmulDescCreate
#define cublasLtMatmulDescSetAttribute hipblasLtMatmulDescSetAttribute
#define cublasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define cublasLtMatrixLayoutCreate hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutSetAttribute hipblasLtMatrixLayoutSetAttribute
#define cublasLtMatrixLayoutDestroy hipblasLtMatrixLayoutDestroy
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceSetAttribute hipblasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulAlgoGetHeuristic hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatmul hipblasLtMatmul
#define CUBLASLT_ORDER_ROW HIPBLASLT_ORDER_ROW
#define CUBLASLT_MATRIX_LAYOUT_ORDER HIPBLASLT_MATRIX_LAYOUT_ORDER
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_POINTER_MODE HIPBLASLT_MATMUL_DESC_POINTER_MODE
#define CUBLASLT_POINTER_MODE_HOST HIPBLASLT_POINTER_MODE_HOST
#define CUBLASLT_POINTER_MODE_DEVICE HIPBLASLT_POINTER_MODE_DEVICE
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUDA_R_16BF HIP_R_16BF
#define CUDA_R_32F HIP_R_32F

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetErrorString hipGetErrorString
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess

inline void check_hip(hipError_t err, const char* file, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s:%d: %s\n", file, line, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void check_cublas(hipblasStatus_t status, const char* file, int line) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "hipBLASLt error at %s:%d: %d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}
} // namespace mhc

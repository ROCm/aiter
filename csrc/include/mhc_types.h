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

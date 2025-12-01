#include <iostream>
#include <functional>
#include <limits>
#include <vector>
#include <array>
#include <tuple>

#include "hip_float8.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_cooperative_groups.h>

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

using namespace std;
using namespace at;

#define NBLOCKS_PER_GPU 256

namespace trtllm {

namespace cg = cooperative_groups;
using __bfloat16 = __hip_bfloat16;

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

namespace details {

static constexpr int kBytesPerAccess = 16;

template <bool RELAXED = true>
__device__ __forceinline__ void st_flag(int *addr, int flag) {
    __scoped_atomic_store_n(addr, flag,
                            RELAXED ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                            __MEMORY_SCOPE_SYSTEM);
}

template <bool RELAXED = true>
__device__ __forceinline__ int ld_flag(int *addr) {
    int flag;
    flag = __scoped_atomic_load_n(addr,
                                  RELAXED ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM);
    return flag;
}

} // namespace details

#define gpuSuccess hipSuccess
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuStream_t hipStream_t
#define gpuLaunchCooperativeKernel hipLaunchCooperativeKernel
#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define gpuDeviceptr_t hipDeviceptr_t
#define gpuPointerGetAttribute hipPointerGetAttribute

namespace kernel_utils {

struct alignas(1) fp8e4m3fn {
    enum {
        max_value = 240,
    };
    struct from_bits_t {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() {
        return from_bits_t();
    }
    uint8_t data;

    fp8e4m3fn() = default;
    __host__ __device__ constexpr fp8e4m3fn(const fp8e4m3fn &) = default;
    __host__ __device__ constexpr fp8e4m3fn(uint8_t v) = delete;
    explicit __host__ __device__ constexpr fp8e4m3fn(uint8_t v, from_bits_t) :
        data(v) {
    }

    explicit __host__ __device__ fp8e4m3fn(float v) {
        data = hip_fp8_impl::to_float8<4, 3, float, false /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fn(double v) :
        fp8e4m3fn(static_cast<float>(v)) {
    }

    explicit inline __host__ __device__ operator float() const {
        return hip_fp8_impl::from_float8<4, 3, float, false /*negative_zero_nan*/>(data);
    }
};

struct alignas(1) fp8e4m3fnuz {
    enum {
        max_value = 120,
    };
    struct from_bits_t {
    };
    __host__ __device__ static constexpr from_bits_t from_bits() {
        return from_bits_t();
    }
    uint8_t data;

    fp8e4m3fnuz() = default;
    __host__ __device__ constexpr fp8e4m3fnuz(const fp8e4m3fnuz &) = default;
    __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v) = delete;
    explicit __host__ __device__ constexpr fp8e4m3fnuz(uint8_t v, from_bits_t) :
        data(v) {
    }

    explicit __host__ __device__ fp8e4m3fnuz(float v) {
        data = hip_fp8_impl::to_float8<4, 3, float, true /*negative_zero_nan*/, true /*clip*/>(v);
    }

    explicit __host__ __device__ fp8e4m3fnuz(double v) :
        fp8e4m3fnuz(static_cast<float>(v)) {
    }

    explicit inline __host__ __device__ operator float() const {
        return hip_fp8_impl::from_float8<4, 3, float, true /*negative_zero_nan*/>(data);
    }
};

template <typename T, int WARP_SIZE, typename func_t>
__device__ __forceinline__ T warp_reduce(T val, func_t fn) {
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        val = fn(val, __shfl_xor(val, offset, WARP_SIZE));
    }
    return val;
}

template <typename T, int WARP_SIZE, typename func_t>
__device__ __forceinline__ T block_reduce(T val, func_t fn) {
    static __shared__ T shared[1024 / WARP_SIZE];
    const int tid = threadIdx.x;
    const int w_tid = tid % WARP_SIZE;
    const int wid = tid / WARP_SIZE;
    val = warp_reduce<T, WARP_SIZE, func_t>(val, fn);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    bool is_mask = threadIdx.x < (blockDim.x / (float)WARP_SIZE);
    val = is_mask ? shared[w_tid] : (T)(0.0f);
    __syncthreads();
    val = warp_reduce<T, WARP_SIZE, func_t>(val, fn);
    return val;
}

template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) vec_t {
    T data[VEC_SIZE];
    __device__ __forceinline__ T &operator[](int i) {
        return data[i];
    }
    __device__ __forceinline__ T const &operator[](int i) const {
        return data[i];
    }
    __device__ __forceinline__ void load(const T *ptr) {
        *this = *reinterpret_cast<vec_t<T, VEC_SIZE> *>(const_cast<T *>(ptr));
    }
    __device__ __forceinline__ void store(T *ptr) {
        *reinterpret_cast<vec_t<T, VEC_SIZE> *>(ptr) = *this;
    }
    __device__ __forceinline__ void nontemporal_load(const T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] =
                __builtin_nontemporal_load((uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void nontemporal_store(T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __builtin_nontemporal_store(reinterpret_cast<uint32_t *>(&data)[i],
                                        (uint32_t *)ptr + i);
        }
    }
    __device__ __forceinline__ void volatile_load(const T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            reinterpret_cast<uint32_t *>(&data)[i] = __scoped_atomic_load_n(
                (uint32_t *)ptr + i, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __device__ __forceinline__ void volatile_store(T *ptr) {
        constexpr int ITERS = VEC_SIZE * sizeof(T) / sizeof(uint32_t);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            __scoped_atomic_store_n((uint32_t *)ptr + i,
                                    reinterpret_cast<uint32_t *>(&data)[i],
                                    __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        }
    }
    __device__ __forceinline__ void fill(T val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            data[i] = val;
        }
    }
    template <typename VT>
    __device__ __forceinline__ void cast_fill(VT val) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            *reinterpret_cast<VT *>(&data[i]) = val;
        }
    }
};

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void vec_add_(vec_t<T, VEC_SIZE> &self,
                                         const vec_t<T, VEC_SIZE> &other) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[i] = (float)self[i] + (float)other[i];
    }
}

template <typename T, int VEC_SIZE, int NRanks>
__device__ __forceinline__ void vec_add_r_(vec_t<T, VEC_SIZE> (&self)[NRanks]) {
    vec_t<float, VEC_SIZE> acc;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        acc[i] = (float)self[0][i];
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            acc[i] += (float)self[r][i];
        }
    }
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        self[0][i] = (T)acc[i];
    }
}

} // namespace kernel_utils

using namespace kernel_utils;

#define WARP_SIZE 64

template <int NRanks>
struct CommPtrs {
    std::array<void *, NRanks> barrier_flag_ptrs;
    std::array<void *, NRanks> data_ptrs;
    void *sync_clock;
    int rank;
};

struct HostCommPtrs {
    std::vector<void *> barrier_flag_ptrs;
    std::vector<void *> data_ptrs;
    void *sync_clock;
    int rank;
    int nranks;
};

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(CommPtrs<NRanks> &cptrs) {
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            barrier_flags[r] = cptrs.barrier_flag_ptrs[r];
        }
        flag_ptr = ((int *)cptrs.sync_clock) + blockIdx.x;
        int rank = cptrs.rank;
        __syncthreads();
        if (threadIdx.x < NRanks) {
            int target_rank = threadIdx.x;
            target_flag = reinterpret_cast<int *>(barrier_flags[target_rank]) + blockIdx.x * NRanks + rank;
            current_flag = reinterpret_cast<int *>(barrier_flags[rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    template <bool RELAXED = true>
    __device__ __forceinline__ void sync() {
        auto flag = (*flag_ptr) + 1;
        if (threadIdx.x < NRanks) {
            details::st_flag<RELAXED>(target_flag, flag);
            while (details::ld_flag<RELAXED>(current_flag) < flag) {
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *flag_ptr = flag;
        }
    }

    int *flag_ptr;
    void *barrier_flags[NRanks];
    int *target_flag;
    int *current_flag;
};

enum QuantType {
    NONE = 0,
    FP8E4M3FN = 1,
    FP8E4M3FNUZ = 2,
};

template <typename T>
struct AllReduceFusionParams {
    int nranks;
    int rank;
    int size;
    int hidden_dim;
    void *allreduce_in;
    void *residual_in;
    void *residual_out;
    void *norm_out;
    void *rms_gamma;
    float rms_eps;
    // per token quant
    QuantType quant_type;
    void *scale_out;
};

template <typename T, int VEC_SIZE, typename QuantT>
__device__ __forceinline__ vec_t<QuantT, VEC_SIZE> convert_to_fp8(vec_t<T, VEC_SIZE> &in_vec, float scale) {
    vec_t<QuantT, VEC_SIZE> out_vec;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        volatile float out = static_cast<float>(in_vec[i]) / scale;
        out_vec[i] = static_cast<QuantT>(out);
    }
    return out_vec;
}

template <typename T, int VEC_SIZE, typename OutT>
__device__ __forceinline__ vec_t<OutT, VEC_SIZE> rms_norm(AllReduceFusionParams<T> const &m_params,
                                                          vec_t<T, VEC_SIZE> const &residual, vec_t<T, VEC_SIZE> const &gamma) {
    __shared__ float s_val;
    vec_t<OutT, VEC_SIZE> norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]);
        acc += v * v;
    }
    acc = block_reduce<float, WARP_SIZE>(acc, std::plus<float>());
    if (threadIdx.x == 0) {
        s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float out = static_cast<float>(reinterpret_cast<T const *>(&residual)[i]) * s_val * static_cast<float>(reinterpret_cast<T const *>(&gamma)[i]);
        norm_out[i] = static_cast<OutT>(out);
    }
    return norm_out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ float reduce_abs_max(vec_t<T, VEC_SIZE> const &data) {
    __shared__ float s_val;
    auto fn = [](float a, float b) { return a > b ? a : b; };
    float acc = -1.f;
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float v = static_cast<float>(reinterpret_cast<T const *>(&data)[i]);
        acc = fn(acc, std::abs(v));
    }
    acc = block_reduce<float, WARP_SIZE>(acc, fn);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename T, int VEC_SIZE, bool STORE = true>
__device__ __forceinline__ void epilogue(
    AllReduceFusionParams<T> const &params,
    vec_t<T, VEC_SIZE> &rms_in,
    vec_t<T, VEC_SIZE> &rms_weight,
    int idx, int tidx) {
    if constexpr (STORE)
        rms_in.store(reinterpret_cast<T *>(params.residual_out) + idx);
    if (params.quant_type == QuantType::NONE) {
        auto val = rms_norm<T, VEC_SIZE, T>(params, rms_in, rms_weight);
        val.store(reinterpret_cast<T *>(params.norm_out) + idx);
    } else {
        auto val = rms_norm<T, VEC_SIZE, float>(params, rms_in, rms_weight);
        float scale = reduce_abs_max<float, VEC_SIZE>(val);
        if (params.quant_type == QuantType::FP8E4M3FN) {
            scale = scale == 0.f ? 1.f : scale / (float)fp8e4m3fn::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fn>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fn *>(params.norm_out) + idx);
        } else {
            scale = scale == 0.f ? 1.f : scale / (float)fp8e4m3fnuz::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fnuz>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fnuz *>(params.norm_out) + idx);
        }
        if (threadIdx.x == 0)
            reinterpret_cast<float *>(params.scale_out)[tidx] = scale;
    }
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot(AllReduceFusionParams<T> params, CommPtrs<NRanks> cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id_begin =
        (blockIdx.x * NRanks + 0) * params.hidden_dim + access_id_in_token;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    SyncComm<NRanks> comm(cptrs);
    comm.sync();

    // allreduce
    for (int idx = (blockIdx.x * NRanks + params.rank) * params.hidden_dim + access_id_in_token;
         idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
        vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[r].load(reinterpret_cast<T *>(cptrs.data_ptrs[r]) + idx);
        }
        vec_add_r_<T, VEC_SIZE, NRanks>(vals);
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[0].store(reinterpret_cast<T *>(cptrs.data_ptrs[r]) + params.size + idx);
        }
    }

    comm.template sync<false>();

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        int token_id = blockIdx.x * NRanks + r;
        for (int idx = token_id * params.hidden_dim + access_id_in_token, tidx = token_id;
             idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim, tidx += gridDim.x * NRanks) {
            vec_t<T, VEC_SIZE> data[2];
            data[0].load(reinterpret_cast<T *>(params.residual_in) + idx);
            data[1].load(reinterpret_cast<T *>(cptrs.data_ptrs[params.rank]) + params.size + idx);
            vec_add_<T, VEC_SIZE>(data[0], data[1]);
            epilogue<T, VEC_SIZE>(params, data[0], gamma, idx, tidx);
        }
    }
}

template <typename T, int NRanks, int BLOCK_SIZE, bool USE_EPILOGUE>
__global__ void __launch_bounds__(BLOCK_SIZE, 1) allreduce_fusion_kernel_w(AllReduceFusionParams<T> params, CommPtrs<NRanks> cptrs) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int WARP_SIZE_ = BLOCK_SIZE / NRanks;
    SyncComm<NRanks> comm(cptrs);

    __shared__ T shared[NRanks * WARP_SIZE_ * VEC_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE_;
    int lane_id = threadIdx.x % WARP_SIZE_;

    comm.sync();

    for (
        int idx = ((blockIdx.x * NRanks + params.rank) * WARP_SIZE_ + lane_id) * VEC_SIZE;
        // int idx = ((params.rank * gridDim.x + blockIdx.x) * WARP_SIZE_ + lane_id) * VEC_SIZE;
        idx < params.size;
        idx += gridDim.x * NRanks * WARP_SIZE_ * VEC_SIZE) {
        vec_t<T, VEC_SIZE> val;
        val.load(reinterpret_cast<T *>(cptrs.data_ptrs[warp_id]) + idx);
        val.store(&shared[0] + threadIdx.x * VEC_SIZE);
        __syncthreads();
        if (warp_id == 0) {
            vec_t<T, VEC_SIZE> acc;
            acc.load(&shared[0] + lane_id * VEC_SIZE);
#pragma unroll
            for (int r = 1; r < NRanks; ++r) {
                vec_t<T, VEC_SIZE> vec;
                vec.load(&shared[0] + (r * WARP_SIZE_ + lane_id) * VEC_SIZE);
                vec_add_<T, VEC_SIZE>(acc, vec);
            }
            acc.store(&shared[0] + lane_id * VEC_SIZE);
        }
        __syncthreads();
        val.load(&shared[0] + lane_id * VEC_SIZE);
        vec_t<T, VEC_SIZE> res;
        res.load(reinterpret_cast<T *>(params.residual_in) + idx);
        vec_add_<T, VEC_SIZE>(val, res);
        val.store(reinterpret_cast<T *>(cptrs.data_ptrs[warp_id]) + idx);
    }

    if constexpr (USE_EPILOGUE) {
        comm.template sync<false>();
        int access_id_in_token = threadIdx.x * VEC_SIZE;
        vec_t<T, VEC_SIZE> gamma;
        gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);
        for (
            int idx = blockIdx.x * params.hidden_dim + access_id_in_token, tidx = blockIdx.x;
            idx < params.size;
            idx += gridDim.x * params.hidden_dim, tidx += gridDim.x) {
            vec_t<T, VEC_SIZE> val;
            val.load(reinterpret_cast<T *>(cptrs.data_ptrs[params.rank]) + idx);
            epilogue<T, VEC_SIZE, true>(params, val, gamma, idx, tidx);
        }
    } else {
        comm.sync();
    }
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_twoshot_launcher(
    AllReduceFusionParams<T> const &params,
    CommPtrs<NRanks> const &cptrs,
    gpuStream_t stream) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    int token_num = params.size / params.hidden_dim;
    int threads_per_token = params.hidden_dim / VEC_SIZE;
    dim3 threadsPerBlock(threads_per_token);
    int nblocks = std::min((token_num + NRanks - 1) / NRanks, NBLOCKS_PER_GPU);
    if (params.size * sizeof(T) >= 2048 * 1024 * 128) {
        nblocks /= 2;
    }
    // void *args[] = {(void *)&params};
    dim3 numBlocks(nblocks);
    allreduce_fusion_kernel_twoshot<T, NRanks><<<numBlocks, threadsPerBlock, 0, stream>>>(params, cptrs);
}

template <typename T, int NRanks, int HIDDEN_DIM>
void allreduce_fusion_kernel_w_launcher(
    AllReduceFusionParams<T> const &params,
    CommPtrs<NRanks> const &cptrs,
    gpuStream_t stream) {
    constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    constexpr int BLOCK_SIZE = HIDDEN_DIM / VEC_SIZE;
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    assert(params.hidden_dim == HIDDEN_DIM);
    int token_num = params.size / params.hidden_dim;
    dim3 threadsPerBlock(BLOCK_SIZE);
    token_num = token_num <= 2 ? 1 : token_num;
    dim3 numBlocks(token_num);
    allreduce_fusion_kernel_w<T, NRanks, BLOCK_SIZE, true><<<numBlocks, threadsPerBlock, 0, stream>>>(params, cptrs);
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params,
                                      CommPtrs<NRanks> const &cptrs,
                                      gpuStream_t stream) {
    int token_num = params.size / params.hidden_dim;
    if (token_num <= 1024) {
        switch (params.hidden_dim) {
        case 4096:
            allreduce_fusion_kernel_w_launcher<T, NRanks, 4096>(params, cptrs, stream);
            return;
        case 2048:
            allreduce_fusion_kernel_w_launcher<T, NRanks, 2048>(params, cptrs, stream);
            return;
        case 1024:
            allreduce_fusion_kernel_w_launcher<T, NRanks, 1024>(params, cptrs, stream);
            return;
        default:
            break;
        }
    }
    allreduce_fusion_kernel_twoshot_launcher<T, NRanks>(params, cptrs, stream);
}

template <typename T>
void allreduce_rms_fusion_impl(HostCommPtrs host_cptrs, int size,
                               int hidden_dim, void *allreduce_in,
                               void *residual_in, void *residual_out,
                               void *norm_out, void *rms_gamma, float eps,
                               int quant_type = 0, void *scale_out = nullptr,
                               gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = host_cptrs.nranks;
    params.rank = host_cptrs.rank;
    params.size = size;
    params.hidden_dim = hidden_dim;
    params.allreduce_in = allreduce_in;
    params.residual_in = residual_in;
    params.residual_out = residual_out;
    params.norm_out = norm_out;
    params.rms_gamma = rms_gamma;
    params.rms_eps = eps;
    params.scale_out = scale_out;
    params.quant_type = (QuantType)quant_type;

#define DISPATCH_NRANKS(NRANKS)                                             \
    {                                                                       \
        CommPtrs<NRANKS> cptrs;                                             \
        for (int i = 0; i < NRANKS; ++i) {                                  \
            cptrs.barrier_flag_ptrs[i] = host_cptrs.barrier_flag_ptrs[i];   \
            cptrs.data_ptrs[i] = host_cptrs.data_ptrs[i];                   \
        }                                                                   \
        cptrs.sync_clock = host_cptrs.sync_clock;                           \
        cptrs.rank = host_cptrs.rank;                                       \
        allreduce_fusion_kernel_launcher<T, NRANKS>(params, cptrs, stream); \
    }

    int nranks = host_cptrs.nranks;
    if (nranks == 8) {
        DISPATCH_NRANKS(8)
    } else if (nranks == 4) {
        DISPATCH_NRANKS(4)
    } else if (nranks == 2) {
        DISPATCH_NRANKS(2)
    } else {
        assert(false);
    }

#undef DISPATCH_NRANKS
}

namespace ipc_details {

Tensor get_handle(void *ptr) {
    gpuIpcMemHandle_t handle;
    TORCH_CHECK(gpuIpcGetMemHandle(&handle, ptr) == gpuSuccess);
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto data_handle = torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
    std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
    return data_handle;
}

void open_handles(int rank, std::vector<Tensor> &handles, void *ptr, std::vector<void *> &ipc_ptrs) {
    std::vector<gpuIpcMemHandle_t> ipc_handles;
    int world_size = handles.size();
    ipc_handles.reserve(world_size);
    ipc_ptrs.resize(world_size);
    for (auto &handle : handles) {
        // Ensure the tensor is on the same device as the current device.
        gpuIpcMemHandle_t ipc_handle;
        std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(gpuIpcMemHandle_t));
        ipc_handles.push_back(ipc_handle);
    }
    for (int i = 0; i < world_size; ++i) {
        if (i != rank) {
            TORCH_CHECK(
                gpuIpcOpenMemHandle((void **)&ipc_ptrs[i], ipc_handles[i], gpuIpcMemLazyEnablePeerAccess) == gpuSuccess);
        } else {
            ipc_ptrs[i] = ptr;
        }
    }
}

} // namespace ipc_details

class CommWorkspace {
public:
    CommWorkspace(int64_t rank, int64_t world_size, int64_t size_in_bytes, int64_t max_thread_blocks = NBLOCKS_PER_GPU) {
        TORCH_CHECK(rank < world_size);
        gpuSetDevice(rank);
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        max_thread_blocks_ = max_thread_blocks;
        gpuMalloc(&sync_clock_, max_thread_blocks_ * sizeof(int));
        gpuMalloc(&barrier_flags_, max_thread_blocks_ * world_size_ * sizeof(int));
        gpuMalloc(&data_, size_in_bytes_ * 2);
        gpuMemset(sync_clock_, 0, max_thread_blocks_ * sizeof(int));
        gpuMemset(barrier_flags_, 0, max_thread_blocks_ * world_size_ * sizeof(int));
    }

    ~CommWorkspace() {
        gpuFree(sync_clock_);
        gpuFree(barrier_flags_);
        gpuFree(data_);
    }

    Tensor get_barrier_handle() {
        return ipc_details::get_handle(barrier_flags_);
    }

    Tensor get_data_handle() {
        return ipc_details::get_handle(data_);
    }

    void open_barrier_handles(std::vector<Tensor> handles) {
        ipc_details::open_handles(rank_, handles, barrier_flags_, ipc_barrier_flags_);
    }

    void open_data_handles(std::vector<Tensor> handles) {
        ipc_details::open_handles(rank_, handles, data_, ipc_data_);
    }

    HostCommPtrs get_cptrs(const Tensor &input, gpuStream_t stream) {
        HostCommPtrs cptrs;
        cptrs.data_ptrs.resize(world_size_);
        cptrs.barrier_flag_ptrs.resize(world_size_);
        void *ptr = (void *)input.data_ptr();
        auto it = cached_ipc_data_.find(ptr);
        if (it != cached_ipc_data_.end()) {
            for (int r = 0; r < world_size_; ++r) {
                cptrs.data_ptrs[r] = (it->second)[r];
            }
        } else {
            gpuMemcpyAsync(data_, ptr, input.numel() * input.element_size(), gpuMemcpyDeviceToDevice, stream);
            for (int r = 0; r < world_size_; ++r) {
                cptrs.data_ptrs[r] = ipc_data_[r];
            }
        }
        for (int r = 0; r < world_size_; ++r) {
            cptrs.barrier_flag_ptrs[r] = ipc_barrier_flags_[r];
        }
        cptrs.sync_clock = sync_clock_;
        cptrs.rank = rank_;
        cptrs.nranks = world_size_;
        return cptrs;
    }

    void capture(const Tensor &input) {
        if (input.numel() * input.element_size() > 1024 * 4096 * 16) {
            return;
        }
        void *ptr = (void *)input.data_ptr();
        void *base_ptr;
        if (gpuPointerGetAttribute(&base_ptr, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR, (gpuDeviceptr_t)ptr) != gpuSuccess) {
            throw std::runtime_error("failed to get pointer attr");
        }
        cached_ptrs_.push_back(ptr);
        cached_base_ptrs_.push_back(base_ptr);
        cached_offsets_.push_back(((char *)ptr) - ((char *)base_ptr));
    }

    void capture_clear() {
        cached_ptrs_.clear();
        cached_base_ptrs_.clear();
        cached_offsets_.clear();
    }

    std::tuple<std::vector<Tensor>, std::vector<int64_t>> get_captured_handles() {
        int num_datas = cached_ptrs_.size();
        std::vector<Tensor> ipc_handles;
        std::vector<int64_t> offsets;
        ipc_handles.reserve(num_datas);
        offsets.reserve(num_datas);
        for (int i = 0; i < num_datas; ++i) {
            ipc_handles.push_back(ipc_details::get_handle(cached_base_ptrs_[i]));
            offsets.push_back(cached_offsets_[i]);
        }
        return {ipc_handles, offsets};
    }

    void open_captured_handles(std::vector<Tensor> &handles, std::vector<int64_t> &offsets, int64_t ptr_idx) {
        auto ptr = cached_ptrs_[ptr_idx];
        auto base_ptr = cached_base_ptrs_[ptr_idx];
        std::vector<void *> ipc_data;
        ipc_details::open_handles(rank_, handles, base_ptr, ipc_data);
        for (int i = 0; i < offsets.size(); ++i) {
            ipc_data[i] = (void *)((char *)ipc_data[i] + offsets[i]);
        }
        cached_ipc_data_[ptr] = ipc_data;
    }

private:
    int rank_;
    int world_size_;
    int size_in_bytes_;
    int max_thread_blocks_;
    void *sync_clock_;
    void *barrier_flags_;
    void *data_;
    std::vector<void *> ipc_barrier_flags_;
    std::vector<void *> ipc_data_;
    // capture
    std::vector<void *> cached_ptrs_;
    std::vector<void *> cached_base_ptrs_;
    std::vector<int64_t> cached_offsets_;
    std::unordered_map<void *, std::vector<void *>> cached_ipc_data_;
};

fptr_t init_ar_fusion(int64_t rank, int64_t world_size, int64_t max_size_in_bytes) {
    switch (world_size) {
    case 8:
    case 4:
    case 2:
        break;
    default:
        throw std::invalid_argument("world size is not supported");
    }
    if (rank < 0 || rank >= world_size)
        throw std::invalid_argument("invalid rank passed in");
    return (fptr_t) new CommWorkspace(rank, world_size, max_size_in_bytes);
}

void destroy_ar_fusion(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    delete ptr;
}

Tensor get_ar_fusion_barrier_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_barrier_handle();
}

Tensor get_ar_fusion_data_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_data_handle();
}

void open_ar_fusion_barrier_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_barrier_handles(handles);
}

void open_ar_fusion_data_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_data_handles(handles);
}

void ar_fusion_capture(fptr_t fptr, const Tensor &input) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->capture(input);
}

void ar_fusion_capture_clear(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->capture_clear();
}

std::tuple<std::vector<Tensor>, std::vector<int64_t>> get_ar_fusion_captured_handles(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_captured_handles();
}

void open_ar_fusion_captured_handles(fptr_t fptr, std::vector<Tensor> handles, std::vector<int64_t> offsets, int64_t ptr_idx) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_captured_handles(handles, offsets, ptr_idx);
}

template <typename T>
struct KernelElementType { using type = T; };

template <>
struct KernelElementType<c10::Half> { using type = __half; };

template <>
struct KernelElementType<c10::BFloat16> {
    using type = __bfloat16;
};

void allreduce_rms(fptr_t fptr, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, Tensor &scale_out,
                   double eps, int64_t quant_type) {
    TORCH_CHECK(allreduce_in.is_contiguous() && residual_in.is_contiguous() && rms_gamma.is_contiguous());
    TORCH_CHECK(residual_out.is_contiguous() && norm_out.is_contiguous() && scale_out.is_contiguous());
    auto dev = allreduce_in.device();
    c10::DeviceGuard dev_guard(dev);
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    auto cptrs = ptr->get_cptrs(allreduce_in, stream);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        allreduce_in.scalar_type(),
        "allreduce_rms", [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_rms_fusion_impl<k_scalar_t>(
                cptrs,
                size,
                hidden_dim,
                (void *)allreduce_in.data_ptr<scalar_t>(),
                (void *)residual_in.data_ptr<scalar_t>(),
                (void *)residual_out.data_ptr<scalar_t>(),
                (void *)norm_out.data_ptr(),
                (void *)rms_gamma.data_ptr<scalar_t>(),
                eps,
                quant_type,
                (void *)scale_out.data_ptr<float>(),
                stream);
        });
}

}

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
static constexpr int kOneShotMaxToken = 128;
static constexpr int kOneShotMaxSize =
    kOneShotMaxToken * 1024 * kBytesPerAccess;

} // namespace details

#define gpuSuccess hipSuccess
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
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

template <typename T, typename func_t>
__device__ __forceinline__ T warp_reduce(T val, func_t fn) {
#pragma unroll
    for (int offset = (32 >> 1); offset > 0; offset >>= 1) {
        val = fn(val, __shfl_xor(val, offset, 32));
    }
    return val;
}

template <typename T, typename func_t>
__inline__ __device__ T block_reduce(T val, func_t fn) {
    static __shared__ T shared[32];
    const int tid = threadIdx.x;
    const int w_tid = tid % 32;
    const int wid = tid / 32;
    val = warp_reduce<T, func_t>(val, fn);
    if (w_tid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
    val = is_mask ? shared[w_tid] : (T)(0.0f);
    __syncthreads();
    val = warp_reduce<T, func_t>(val, fn);
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

namespace comm {

template <int NRanks>
struct SyncComm {
    __device__ __forceinline__ SyncComm(void **workspace) {
        counter_ptr = (int *)workspace[NRanks * 3 + 0];
        flag_ptr = (int *)workspace[NRanks * 3 + 1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r) {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (atomicAdd(counter_ptr, 0) != gridDim.x) {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    void *comm_bufs[NRanks];
    void *barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
class Barrier {
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const &comm) {
        if (threadIdx.x < NRanks) {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag = reinterpret_cast<int *>(comm.barrier_flags[target_rank]) + current_rank;
            m_current_flag =
                reinterpret_cast<int *>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync() {
        constexpr int kBarrierFlagCount = NBLOCKS_PER_GPU;
        __syncthreads();
        if (threadIdx.x < NRanks) {
            m_flag_value = next_flag(m_flag_value);
            // To avoid the ABA problem, we need to synchronize the correct flag value
            // to all barrier_flags, even if the corresponding CTA has not been
            // launched.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount;
                 flag_idx += gridDim.x) {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value)) {
            }
        }
        __syncthreads();
    }

protected:
    __device__ void st_flag(int *addr, int flag) {
#ifdef __CUDACC__
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
#else
        __scoped_atomic_store_n(addr, flag, __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
#endif
    }

    __device__ int ld_flag(int *addr) {
        int flag;
#ifdef __CUDACC__
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];"
                     : "=r"(flag)
                     : "l"(addr));
#else
        flag =
            __scoped_atomic_load_n(addr, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
#endif
        return flag;
    }

    __device__ __forceinline__ int next_flag(int flag) {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag) {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    volatile int m_flag_value;

private:
    int *m_target_flag;
    int *m_current_flag;
};

template <int NRanks>
struct LamportComm {
    __device__ __forceinline__ LamportComm(void **workspace, int rank) {
        counter_ptr = (int *)workspace[NRanks * 3 + 0];
        flag_ptr = (int *)workspace[NRanks * 3 + 2];
        int comm_size = *reinterpret_cast<int *>(workspace[NRanks * 3 + 3]);
        clear_ptr = (int *)workspace[NRanks * 3 + 4];
        flag_value = *flag_ptr;
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r) {
            data_bufs[r] = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + r]) + static_cast<int64_t>(data_offset) * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t *>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_clear_size) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            while (atomicAdd(counter_ptr, 0) != gridDim.x) {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int *counter_ptr;
    int *flag_ptr;
    int *clear_ptr;
    uint8_t *data_bufs[NRanks];
    uint8_t *clear_buf;
    int clear_size;
    int flag_value;
};

} // namespace comm

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
    void **workspace;
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
    acc = block_reduce<float>(acc, std::plus<float>());
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
    acc = block_reduce<float>(acc, fn);
    if (threadIdx.x == 0) {
        s_val = acc;
    }
    __syncthreads();
    acc = s_val;
    return acc;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void epilogue(
    AllReduceFusionParams<T> const &params,
    vec_t<T, VEC_SIZE> &rms_in,
    vec_t<T, VEC_SIZE> &rms_weight,
    int idx, int tidx) {
    rms_in.store(reinterpret_cast<T *>(params.residual_out) + idx);
    if (params.quant_type == QuantType::NONE) {
        auto val = rms_norm<T, VEC_SIZE, T>(params, rms_in, rms_weight);
        val.store(reinterpret_cast<T *>(params.norm_out) + idx);
    } else {
        auto val = rms_norm<T, VEC_SIZE, float>(params, rms_in, rms_weight);
        float scale = reduce_abs_max<float, VEC_SIZE>(val);
        if (params.quant_type == QuantType::FP8E4M3FN) {
            scale = scale == 0.f ? 1.f : scale / fp8e4m3fn::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fn>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fn *>(params.norm_out) + idx);
        } else {
            scale = scale == 0.f ? 1.f : scale / fp8e4m3fnuz::max_value;
            auto val_fp8 = convert_to_fp8<float, VEC_SIZE, fp8e4m3fnuz>(val, scale);
            val_fp8.store(reinterpret_cast<fp8e4m3fnuz *>(params.norm_out) + idx);
        }
        if (threadIdx.x == 0)
            reinterpret_cast<float *>(params.scale_out)[tidx] = scale;
    }
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot_direct(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

    int access_id_in_token = threadIdx.x * VEC_SIZE;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    comm::SyncComm<NRanks> comm(params.workspace);

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        for (int idx =
                 (blockIdx.x * NRanks + r) * params.hidden_dim + access_id_in_token;
             idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
            reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
                reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];
        }
    }

    comm::Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();

    // allreduce
    for (int idx = (blockIdx.x * NRanks + params.rank) * params.hidden_dim + access_id_in_token;
         idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim) {
        vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r]) + idx);
        }
        vec_add_r_<T, VEC_SIZE, NRanks>(vals);
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            vals[0].store(reinterpret_cast<T *>(comm.comm_bufs[r]) + params.size + idx);
        }
    }

    barrier.sync();

#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        int token_id = blockIdx.x * NRanks + r;
        for (int idx = token_id * params.hidden_dim + access_id_in_token, tidx = token_id;
             idx < params.size; idx += gridDim.x * NRanks * params.hidden_dim, tidx += gridDim.x * NRanks) {
            vec_t<T, VEC_SIZE> data[2];
            data[0].load(reinterpret_cast<T *>(params.residual_in) + idx);
            data[1].load(reinterpret_cast<T *>(comm.comm_bufs[params.rank]) + params.size + idx);
            vec_add_<T, VEC_SIZE>(data[0], data[1]);
            data[0].store(reinterpret_cast<T *>(params.residual_out) + idx);
            epilogue<T, VEC_SIZE>(params, data[0], gamma, idx, tidx);
        }
    }

    comm.update(barrier.m_flag_value);
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_twoshot_single_load(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);

    int access_id_in_token = threadIdx.x * VEC_SIZE;

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    comm::SyncComm<NRanks> comm(params.workspace);

    int idx = blockIdx.x * params.hidden_dim + access_id_in_token;
    reinterpret_cast<float4 *>(comm.comm_bufs[params.rank])[idx / VEC_SIZE] =
        reinterpret_cast<float4 *>(params.allreduce_in)[idx / VEC_SIZE];

    comm::Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();

    // cross-device load
    vec_t<T, VEC_SIZE> vals[NRanks];
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
        vals[r].load(reinterpret_cast<T *>(comm.comm_bufs[r]) + idx);
    }
    vec_add_r_<T, VEC_SIZE, NRanks>(vals);

    int tidx = blockIdx.x;
    vec_t<T, VEC_SIZE> residual;
    residual.load(reinterpret_cast<T *>(params.residual_in) + idx);
    vec_add_<T, VEC_SIZE>(residual, vals[0]);
    epilogue<T, VEC_SIZE>(params, residual, gamma, idx, tidx);

    comm.update(barrier.m_flag_value);
}

template <typename T>
struct neg_zero { static constexpr T value = -T(0); };

template <>
struct neg_zero<__half> {
    static constexpr unsigned short neg_zero_bits = 0x8000U;
    static constexpr __half value = __half_raw{neg_zero_bits};
    using bits_type = unsigned short;
};

template <>
struct neg_zero<__bfloat16> {
    static constexpr unsigned short neg_zero_bits = 0x8000U;
    static constexpr __bfloat16 value = __hip_bfloat16_raw{neg_zero_bits};
    using bits_type = unsigned short;
};

template <>
struct neg_zero<float> {
    static constexpr unsigned int neg_zero_bits = 0x80000000U;
    static constexpr float value = -0.0f;
    using bits_type = unsigned int;
};

template <>
struct neg_zero<double> {
    static constexpr uint64_t neg_zero_bits = 0x8000000000000000ULL;
    static constexpr double value = -0.0f;
    using bits_type = uint64_t;
};

template <typename T>
__device__ static constexpr T neg_zero_v = neg_zero<T>::value;

template <typename T>
__device__ bool is_negative_zero(T) {
    return false;
}

// float specialization
template <>
__device__ bool is_negative_zero<float>(float x) {
    return (__float_as_int(x) == 0x80000000);
}

// double specialization
template <>
__device__ bool is_negative_zero<double>(double x) {
    return (__double_as_longlong(x) == 0x8000000000000000ULL);
}

// __half specialization
template <>
__device__ bool is_negative_zero<__half>(__half x) {
    return (__half_as_ushort(x) == 0x8000);
}

// __bfloat16 specialization
template <>
__device__ bool is_negative_zero<__bfloat16>(__bfloat16 x) {
    return (__bfloat16_as_ushort(x) == 0x8000);
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ bool has_neg_zero(const vec_t<T, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        if (is_negative_zero<T>(vec[i])) {
            return true;
        }
    }
    return false;
}

template <typename T, uint32_t VEC_SIZE>
__device__ __forceinline__ void remove_neg_zero(vec_t<T, VEC_SIZE> &vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        vec[i] = (is_negative_zero<T>(vec[i])) ? static_cast<T>(0.f) : vec[i];
    }
}

template <typename T, int NRanks>
__global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams<T> params) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    int token_id = blockIdx.x;
    int access_id_in_token = threadIdx.x * VEC_SIZE;
    int access_id = token_id * params.hidden_dim + access_id_in_token;
    int access_stride = gridDim.x * params.hidden_dim;

    vec_t<T, VEC_SIZE> clear_vec;
    clear_vec.cast_fill(neg_zero<T>::neg_zero_bits);

    vec_t<T, VEC_SIZE> gamma;
    gamma.load(reinterpret_cast<T *>(params.rms_gamma) + access_id_in_token);

    comm::LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size;

    for (int idx = access_id; idx < params.size; idx += access_stride) {
        vec_t<T, VEC_SIZE> val;
        val.load(reinterpret_cast<T *>(params.allreduce_in) + idx);
        remove_neg_zero<T, VEC_SIZE>(val);
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
            // Push data to other ranks
            val.store(reinterpret_cast<T *>(comm.data_bufs[r]) + params.rank * params.size + idx);
        }
    }

    for (int idx = access_id; idx < clear_access; idx += access_stride) {
        // Clear comm buffer that previous kernel used
        clear_vec.store(reinterpret_cast<T *>(comm.clear_buf) + idx);
    }

    for (int idx = access_id, tidx = token_id; idx < params.size;
         idx += access_stride, tidx += gridDim.x) {
        vec_t<T, VEC_SIZE> residual;
        residual.load(reinterpret_cast<T *>(params.residual_in) + idx);

        vec_t<T, VEC_SIZE> vals[NRanks];
        volatile bool done = false;
        while (!done) {
            done = true;
            __threadfence();
#pragma unroll
            for (int r = 0; r < NRanks; ++r) {
                // LDG.128 from local rank
                vals[r].load(reinterpret_cast<T *>(comm.data_bufs[params.rank]) + r * params.size + idx);
                done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
            }
        }
        vec_add_r_<T, VEC_SIZE, NRanks>(vals);
        vec_add_<T, VEC_SIZE>(vals[0], residual);
        epilogue<T, VEC_SIZE>(params, vals[0], gamma, idx, tidx);
    }

    comm.update(params.size * NRanks);
}

template <typename T, int NRanks>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams<T> const &params,
                                      gpuStream_t stream) {
    static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
    assert(params.size % params.hidden_dim == 0);
    assert(params.hidden_dim % VEC_SIZE == 0);
    int token_num = params.size / params.hidden_dim;
    int threads_per_token = params.hidden_dim / VEC_SIZE;
    dim3 threadsPerBlock(threads_per_token);
    // void *args[] = {(void *)&params};
    if (token_num <= NBLOCKS_PER_GPU) {
        dim3 numBlocks(token_num);
        allreduce_fusion_kernel_twoshot_single_load<T, NRanks><<<numBlocks,
                                                                 threadsPerBlock, 0, stream>>>(params);
    } else {
        int nblocks = std::min(token_num, NBLOCKS_PER_GPU);
        if (params.size * sizeof(T) >= 1024 * 1024 * 128) {
            nblocks /= 2;
        }
        dim3 numBlocks(nblocks);
        allreduce_fusion_kernel_twoshot_direct<T, NRanks><<<numBlocks,
                                                            threadsPerBlock, 0, stream>>>(params);
    }
}

template <typename T>
void allreduce_rms_fusion_impl(void **workspace, int rank, int nranks, int size,
                               int hidden_dim, void *allreduce_in,
                               void *residual_in, void *residual_out,
                               void *norm_out, void *rms_gamma, float eps,
                               int quant_type = 0, void *scale_out = nullptr,
                               gpuStream_t stream = 0) {
    AllReduceFusionParams<T> params;
    params.nranks = nranks;
    params.rank = rank;
    params.size = size;
    params.hidden_dim = hidden_dim;
    params.workspace = workspace;
    params.allreduce_in = allreduce_in;
    params.residual_in = residual_in;
    params.residual_out = residual_out;
    params.norm_out = norm_out;
    params.rms_gamma = rms_gamma;
    params.rms_eps = eps;
    params.scale_out = scale_out;
    params.quant_type = (QuantType)quant_type;
    if (nranks == 8) {
        allreduce_fusion_kernel_launcher<T, 8>(params, stream);
    } else if (nranks == 4) {
        allreduce_fusion_kernel_launcher<T, 4>(params, stream);
    } else if (nranks == 2) {
        allreduce_fusion_kernel_launcher<T, 2>(params, stream);
    } else {
        assert(false);
    }
}

class CommWorkspace {
    static constexpr int MAX_RANKS = 16;

    template <typename T>
    void flush_data(void *data, int one_shot_comm_size) {
        using element_t = typename neg_zero<T>::bits_type;
        std::vector<element_t> arr;
        arr.resize(one_shot_comm_size / sizeof(T));
        for (int i = 0; i < one_shot_comm_size / sizeof(element_t); ++i) {
            volatile element_t v = neg_zero<T>::neg_zero_bits;
            arr[i] = v;
        }
        gpuMemcpy(data, arr.data(), one_shot_comm_size, gpuMemcpyHostToDevice);
    }

public:
    CommWorkspace(int64_t rank, int64_t world_size, int64_t size_in_bytes) {
        TORCH_CHECK(world_size < MAX_RANKS && rank < world_size);
        gpuSetDevice(rank);
        rank_ = rank;
        world_size_ = world_size;
        size_in_bytes_ = size_in_bytes;
        int data_size = size_in_bytes * 2 + NBLOCKS_PER_GPU * world_size * sizeof(int);
        int one_shot_comm_size = details::kOneShotMaxSize * world_size_ * 3;
        data_size += one_shot_comm_size;
        gpuMalloc(&data_, data_size);
        gpuMalloc(&counter_, sizeof(int));
        gpuMemset(counter_, 0, sizeof(int));
        gpuMalloc(&twoshot_sync_clock_, sizeof(int));
        gpuMemset(twoshot_sync_clock_, 0, sizeof(int));
        // oneshot
        gpuMalloc(&oneshot_sync_clock_, sizeof(int));
        gpuMemset(oneshot_sync_clock_, 0, sizeof(int));
        int size = details::kOneShotMaxSize * world_size;
        gpuMalloc(&oneshot_comm_size_, sizeof(int));
        gpuMemcpy(oneshot_comm_size_, &size, sizeof(int), gpuMemcpyHostToDevice);
        gpuMalloc(&oneshot_clear_, sizeof(int));
        gpuMemset(oneshot_clear_, 0, sizeof(int));
        flush_data<float>((void *)((char *)data_ + size_in_bytes * 2 + NBLOCKS_PER_GPU * world_size * sizeof(int)), one_shot_comm_size);
        dtype_ = ScalarType::Float;
        gpuDeviceSynchronize();
    }

    ~CommWorkspace() {
        gpuFree(counter_);
        gpuFree(twoshot_sync_clock_);
        gpuFree(data_);
        gpuFree(oneshot_sync_clock_);
        gpuFree(oneshot_clear_);
        gpuFree(oneshot_comm_size_);
    }

    Tensor get_handle() {
        gpuIpcMemHandle_t handle;
        TORCH_CHECK(gpuIpcGetMemHandle(&handle, data_) == gpuSuccess);
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto data_handle = torch::empty({static_cast<int64_t>(sizeof(gpuIpcMemHandle_t))}, options);
        std::memcpy(data_handle.data_ptr(), &handle, sizeof(gpuIpcMemHandle_t));
        return data_handle;
    }

    void open_handles(std::vector<Tensor> handles) {
        std::vector<gpuIpcMemHandle_t> ipc_handles;
        ipc_handles.reserve(world_size_);
        for (auto &handle : handles) {
            // Ensure the tensor is on the same device as the current device.
            gpuIpcMemHandle_t ipc_handle;
            std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(gpuIpcMemHandle_t));
            ipc_handles.push_back(ipc_handle);
        }
        for (int i = 0; i < world_size_; ++i) {
            if (i != rank_) {
                TORCH_CHECK(
                    gpuIpcOpenMemHandle((void **)&ipc_data_[i], ipc_handles[i], gpuIpcMemLazyEnablePeerAccess) == gpuSuccess);
            } else {
                ipc_data_[i] = data_;
            }
        }
        for (int i = 0; i < world_size_; ++i) {
            twoshot_comm_bufs_[i] = ipc_data_[i];
            twoshot_barrier_flags_[i] = (int *)((char *)ipc_data_[i] + 2 * size_in_bytes_);
            // oneshot
            oneshot_comm_bufs_[i] = (void *)((char *)ipc_data_[i] + 2 * size_in_bytes_ + NBLOCKS_PER_GPU * world_size_ * sizeof(int));
        }
    }

    Tensor get_workspace(const Tensor &ref) {
        std::vector<void *> workspace(world_size_ * 3 + 5);
        auto dtype = ref.scalar_type();
        int one_shot_comm_size = details::kOneShotMaxSize * world_size_ * 3;
        if (dtype != dtype_) {
            if (dtype == ScalarType::Float) {
                flush_data<float>(oneshot_comm_bufs_[rank_], one_shot_comm_size);
            } else if (dtype == ScalarType::Half) {
                flush_data<__half>(oneshot_comm_bufs_[rank_], one_shot_comm_size);
            } else if (dtype == ScalarType::BFloat16) {
                flush_data<__bfloat16>(oneshot_comm_bufs_[rank_], one_shot_comm_size);
            } else {
                TORCH_CHECK("datatype not support!");
            }
            dtype_ = dtype;
        }
        for (int peer = 0; peer < world_size_; ++peer) {
            workspace[peer] = (void *)twoshot_comm_bufs_[peer];
            workspace[world_size_ + peer] = (void *)twoshot_barrier_flags_[peer];
            workspace[2 * world_size_ + peer] = (void *)oneshot_comm_bufs_[peer];
        }
        workspace[world_size_ * 3 + 0] = (void *)counter_;
        workspace[world_size_ * 3 + 1] = (void *)twoshot_sync_clock_;
        // oneshot
        workspace[world_size_ * 3 + 2] = (void *)oneshot_sync_clock_;
        workspace[world_size_ * 3 + 3] = (void *)oneshot_comm_size_;
        workspace[world_size_ * 3 + 4] = (void *)oneshot_clear_;
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        auto workspace_tensor = torch::empty({static_cast<int64_t>(workspace.size() * sizeof(void *))}, options);
        std::memcpy(workspace_tensor.data_ptr(), workspace.data(), workspace.size() * sizeof(void *));
        return workspace_tensor.to(ref.device());
    }

private:
    // meta
    int rank_;
    int world_size_;
    int size_in_bytes_;

    // data
    void *data_;
    void *ipc_data_[MAX_RANKS];

    int *counter_;
    // twoshot
    void *twoshot_comm_bufs_[MAX_RANKS];    // 2 * size * sizeof(T)
    int *twoshot_barrier_flags_[MAX_RANKS]; // nblocks * world_size
    int *twoshot_sync_clock_;
    // oneshot
    void *oneshot_comm_bufs_[MAX_RANKS];
    int *oneshot_sync_clock_;
    int *oneshot_comm_size_;
    int *oneshot_clear_;
    ScalarType dtype_;
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

Tensor get_ar_fusion_handle(fptr_t fptr) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_handle();
}

void open_ar_fusion_handles(fptr_t fptr, std::vector<Tensor> handles) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    ptr->open_handles(handles);
}

Tensor get_ar_fusion_workspace(fptr_t fptr, const Tensor &ref) {
    auto ptr = reinterpret_cast<CommWorkspace *>(fptr);
    return ptr->get_workspace(ref);
}

template <typename T>
struct KernelElementType { using type = T; };

template <>
struct KernelElementType<c10::Half> { using type = __half; };

template <>
struct KernelElementType<c10::BFloat16> {
    using type = __bfloat16;
};

void allreduce_rms(int64_t rank, int64_t nranks, Tensor &allreduce_in, Tensor &residual_in,
                   Tensor &rms_gamma, Tensor &residual_out, Tensor &norm_out, Tensor &scale_out,
                   double eps, int64_t quant_type, Tensor &workspace) {
    auto dev = allreduce_in.device();
    c10::DeviceGuard dev_guard(dev);
    auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
    int size = allreduce_in.numel();
    int hidden_dim = allreduce_in.size(-1);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        allreduce_in.scalar_type(),
        "allreduce_rms", [&] {
            using k_scalar_t = KernelElementType<scalar_t>::type;
            allreduce_rms_fusion_impl<k_scalar_t>(
                (void **)workspace.data_ptr(),
                rank,
                nranks,
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

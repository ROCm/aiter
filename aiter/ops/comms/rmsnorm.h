#ifndef __RMSNORM_H__
#define __RMSNORM_H__

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <ranges>
#include <type_traits>
#include <vector>

#define HIP_CALL(cmd)                                             \
    do                                                            \
    {                                                             \
        hipError_t error = (cmd);                                 \
        if (error != hipSuccess)                                  \
        {                                                         \
            std::cerr << "Encountered HIP error ("                \
                      << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__      \
                      << "\n";                                    \
            exit(-1);                                             \
        }                                                         \
    } while (0)

auto hipMallocFunc = [](std::size_t size) -> void* {
    void* ptr;
    HIP_CALL(hipMalloc(&ptr, size));
    return ptr;
};

struct hipDeleteHelper
{
    int _device_id;
    hipDeleteHelper(int i) : _device_id(i) {}
    void operator()(void* ptr)
    {
        if (ptr)
        {
            HIP_CALL(hipSetDevice(_device_id));
            HIP_CALL(hipFree(ptr));
        }
    }
};

template <int LEN>
struct VecType;

template <>
struct VecType<4>
{
    using vectype = __attribute__((
        __vector_size__(2 * sizeof(uint32_t)))) uint32_t;
};

template <>
struct VecType<8>
{
    using vectype = __attribute__((
        __vector_size__(4 * sizeof(uint32_t)))) uint32_t;
};

template <typename T, int LEN>
union PackData;

template <>
union PackData<__half, 4>
{
    using vec_type = typename VecType<4>::vectype;
    vec_type vec_;
    uint32_t u32_[2];
    __half f_[4];

    __host__ __device__ __half& operator[](int i) { return f_[i]; }
    __host__ __device__ const __half& operator[](int i) const
    {
        return f_[i];
    }
};

template <>
union PackData<__half, 8>
{
    using vec_type = typename VecType<8>::vectype;
    vec_type vec_;
    uint32_t u32_[4];
    __half f_[8];

    __host__ __device__ half& operator[](int i) { return f_[i]; }
    __host__ __device__ const __half& operator[](int i) const
    {
        return reinterpret_cast<const __half*>(f_)[i];
    }
};

template <>
union PackData<__hip_bfloat16, 4>
{
    using vec_type = typename VecType<4>::vectype;
    vec_type vec_;
    uint32_t u32_[2];
    __hip_bfloat16_raw raw_data_[4];

    __host__ __device__ __hip_bfloat16& operator[](int i)
    {
        return reinterpret_cast<__hip_bfloat16*>(raw_data_)[i];
    }
    __host__ __device__ const __hip_bfloat16& operator[](int i) const
    {
        return reinterpret_cast<const __hip_bfloat16*>(raw_data_)[i];
    }
};

template <>
union PackData<__hip_bfloat16, 8>
{
    using vec_type = typename VecType<8>::vectype;
    vec_type vec_;
    uint32_t u32_[4];
    __hip_bfloat16_raw raw_data_[8];

    __host__ __device__ __hip_bfloat16& operator[](int i)
    {
        return reinterpret_cast<__hip_bfloat16*>(raw_data_)[i];
    }
    __host__ __device__ const __hip_bfloat16& operator[](int i) const
    {
        return reinterpret_cast<const __hip_bfloat16*>(raw_data_)[i];
    }
};

template <>
union PackData<__hip_fp8_storage_t, 8>
{
    using vec_type = typename VecType<4>::vectype;
    vec_type vec_;
    __hip_fp8_storage_t data_[8];
    __host__ __device__ __hip_fp8_storage_t& operator[](int i)
    {
        return data_[i];
    }
    __host__ __device__ const __hip_fp8_storage_t& operator[](
        int i) const
    {
        return data_[i];
    }
};

template <>
union PackData<__hip_fp8_storage_t, 16>
{
    using vec_type = typename VecType<8>::vectype;
    vec_type vec_;
    __hip_fp8_storage_t data_[16];

    __host__ __device__ __hip_fp8_storage_t& operator[](int i)
    {
        return data_[i];
    }
    __host__ __device__ const __hip_fp8_storage_t& operator[](
        int i) const
    {
        return data_[i];
    }
};

template <typename T>
__device__ __forceinline__ T loadnt(T* addr)
{
    return __builtin_nontemporal_load(addr);
    // return *((T*)addr);
}

template <typename T>
struct dtype;

template <>
struct dtype<__half>
{
    using packed_type = __half2;
    static inline __device__ __host__ float to_float(__half v)
    {
        return __half2float(v);
    }
    static inline __device__ float2 packed_to_float2(__half2 v)
    {
        return __half22float2(v);
    }
    static inline __device__ __host__ __half from_float(float v)
    {
        return __float2half(v);
    }

    template <int LEN>
    static inline __device__ __host__ __half* from_pack_data(
        PackData<__half, LEN>& data)
    {
        return &(data._f[0]);
    }
    template <int LEN>
    static inline __device__ __host__ int vec_length(
        PackData<__half, LEN>& data)
    {
        return LEN;
    }
};

template <>
struct dtype<__hip_bfloat16>
{
    using packed_type = __hip_bfloat162;

    static inline __device__ __host__ float to_float(__hip_bfloat16 v)
    {
        return __bfloat162float(v);
    }
    static inline __device__ float2
    packed_to_float2(__hip_bfloat162 v)
    {
        return __bfloat1622float2(v);
    }
    static inline __device__ __host__ __hip_bfloat16
    from_float(float v)
    {
        return __float2bfloat16(v);
    }

    template <int LEN>
    static inline __device__ __host__ __hip_bfloat16* from_pack_data(
        PackData<__hip_bfloat16, LEN>& data)
    {
        return &(data[0]);
    }

    template <int LEN>
    static inline __device__ __host__ int vec_length(
        PackData<__hip_bfloat16, LEN>& data)
    {
        return LEN;
    }
};

template <>
struct dtype<__hip_fp8_storage_t>
{
    using packed_type = __hip_fp8_storage_t;

    static inline __device__ __host__ float to_float(
        packed_type v,
        __hip_fp8_interpretation_t interpret = __HIP_E4M3)
    {
        __half hf = __hip_cvt_fp8_to_halfraw(v, interpret);
        return __half2float(hf);
    }

    static inline __device__ packed_type
    from_float(float v,
               __hip_saturation_t sat = __HIP_SATFINITE,
               __hip_fp8_interpretation_t interpret = __HIP_E4M3)
    {
        return __hip_cvt_float_to_fp8(v, sat, interpret);
    }

    template <int LEN>
    static inline __device__ packed_type* from_pack_data(
        PackData<__hip_fp8_storage_t, LEN>& data)
    {
        return &(data[0]);
    }

    template <int LEN>
    static inline __device__ int vec_length(
        PackData<__hip_fp8_storage_t, LEN>& data)
    {
        return LEN;
    }
};

template <typename T>
void generate_random(std::vector<T>& v);

template <typename InT, typename OutT = InT>
struct RMSNormHelper
{
    int nGPU_;
    int rank_;
    int n_rows_;
    int n_cols_;
    int input_row_stride_;
    int output_row_stride_;

    std::vector<InT**> input_ptrs_;
    std::vector<OutT**> output_ptrs_;
    std::vector<InT**> g_ptrs_;
    std::vector<OutT**> rsigma_ptrs_;

    InT** mapped_input_ptrs_;
    OutT** mapped_output_ptrs_;
    InT** mapped_g_ptrs_;
    OutT** mapped_rsigma_ptrs_;

    float epsilon_;

    int warp_size_;
    int num_cus_;

    std::vector<std::shared_ptr<float>> sh_ptr_per_wg_vec_;
    std::vector<float*> ptr_per_wg_vec_;

    std::vector<int> n_local_rows_;
    std::vector<int> dev_vec_;
    std::vector<int> start_vec_;

    std::vector<hipGraph_t> graph_vec_;
    std::vector<hipGraphExec_t> graph_instance_vec_;
    std::vector<hipGraphNode_t> graph_nodeA_vec_;
    std::vector<hipGraphNode_t> graph_nodeB_vec_;
    std::vector<hipGraphNode_t> graph_nodeC_vec_;

    std::shared_ptr<float> dist_sh_ptr_per_wg_;
    float* dist_ptr_per_wg_;

    int dist_local_rows_;
    int dist_start_;

    hipGraph_t dist_graph_;
    hipGraphExec_t dist_graph_instance_;
    hipGraphNode_t dist_graph_nodeA_;
    hipGraphNode_t dist_graph_nodeB_;
    hipGraphNode_t dist_graph_nodeC_;

    RMSNormHelper(int nGPU,
                  int n_rows,
                  int n_cols,
                  int input_row_stride,
                  int output_row_stride,
                  std::vector<InT**> input_ptrs,
                  std::vector<OutT**> output_ptrs,
                  std::vector<InT**> g_ptrs,
                  std::vector<OutT**> rsigma_ptrs,
                  float epsilon);

    RMSNormHelper(int nGPU,
                  int rank,
                  int n_rows,
                  int n_cols,
                  int input_row_stride,
                  int output_row_stride,
                  InT** mapped_input_ptrs,
                  OutT** mapped_output_ptrs,
                  InT** mapped_g_ptrs,
                  OutT** mapped_rsigma_ptrs,
                  float epsilon);

    ~RMSNormHelper() {}

    void launch_graph(int dev)
    {
        HIP_CALL(hipSetDevice(dev));
        HIP_CALL(hipGraphLaunch(graph_instance_vec_[dev], 0));
    }

    void launch_graph()
    {
        HIP_CALL(hipSetDevice(rank_));
        HIP_CALL(hipGraphLaunch(dist_graph_instance_, 0));
    }
};

template <typename InT, typename OutT = InT>
void rms_norm(RMSNormHelper<InT, OutT>& rms_helper);

template <typename T>
int check(int dev,
          std::shared_ptr<T*>* input_ptr,
          std::shared_ptr<T*>* output_ptr,
          std::shared_ptr<T*>* g_ptr,
          float epsilon,
          int input_row_stride,
          int output_row_stride,
          int n_rows,
          int n_cols);

template <typename T>
int check_input(T* input_ptr,
                T** input_d_ptr,
                int n_rows,
                int n_cols);

#if 1
template <typename T>
void generate_random(std::vector<T>& v);
#endif

void setup_multigpu();

#endif

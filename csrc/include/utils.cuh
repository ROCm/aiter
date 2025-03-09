// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>
#include <type_traits>


class Error : public std::exception {
 private:
  std::string message_;

 public:
  Error(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  virtual const char* what() const noexcept override { return message_.c_str(); }
};

#define FLASHINFER_ERROR(message) throw Error(__FUNCTION__, __FILE__, __LINE__, message)

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                    \
  {                                                                                        \
    hipError_t e = (func);                                                                 \
    if (e != hipSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << hipGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;     \
      return e;                                                                            \
    }                                                                                      \
  }
#else
#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    hipError_t e = (func);              \
    if (e != hipSuccess) {              \
      return e;                         \
    }                                   \
  }
#endif

#define DISPATCH_ALLOW_FP16_QK_REDUCTION(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, ...) \
  if (allow_fp16_qk_reduction) {                                                                \
    FLASHINFER_ERROR("FP16_QK_REDUCTION disabled at compile time");                             \
  } else {                                                                                      \
    constexpr bool ALLOW_FP16_QK_REDUCTION = false;                                             \
    __VA_ARGS__                                                                                 \
  }

#define DISPATCH_NUM_MMA_Q(num_mma_q, NUM_MMA_Q, ...)  \
  if (num_mma_q == 1) {                                \
    constexpr size_t NUM_MMA_Q = 1;                    \
    __VA_ARGS__                                        \
  } else if (num_mma_q == 2) {                         \
    constexpr size_t NUM_MMA_Q = 2;                    \
    __VA_ARGS__                                        \
  } else {                                             \
    std::ostringstream err_msg;                        \
    err_msg << "Unsupported num_mma_q: " << num_mma_q; \
    FLASHINFER_ERROR(err_msg.str());                   \
  }

#define DISPATCH_NUM_MMA_KV(max_mma_kv, NUM_MMA_KV, ...) \
  if (max_mma_kv >= 8) {                                 \
    constexpr size_t NUM_MMA_KV = 8;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 4) {                          \
    constexpr size_t NUM_MMA_KV = 4;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 2) {                          \
    constexpr size_t NUM_MMA_KV = 2;                     \
    __VA_ARGS__                                          \
  } else if (max_mma_kv >= 1) {                          \
    constexpr size_t NUM_MMA_KV = 1;                     \
    __VA_ARGS__                                          \
  } else {                                               \
    std::ostringstream err_msg;                          \
    err_msg << "Unsupported max_mma_kv: " << max_mma_kv; \
    FLASHINFER_ERROR(err_msg.str());                     \
  }

#define DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)   \
  switch (cta_tile_q) {                                    \
    case 128: {                                            \
      constexpr uint32_t CTA_TILE_Q = 128;                 \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    case 64: {                                             \
      constexpr uint32_t CTA_TILE_Q = 64;                  \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    case 16: {                                             \
      constexpr uint32_t CTA_TILE_Q = 16;                  \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    default: {                                             \
      std::ostringstream err_msg;                          \
      err_msg << "Unsupported cta_tile_q: " << cta_tile_q; \
      FLASHINFER_ERROR(err_msg.str());                     \
    }                                                      \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 3) {                              \
    constexpr size_t GROUP_SIZE = 3;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else {                                                   \
    std::ostringstream err_msg;                              \
    err_msg << "Unsupported group_size: " << group_size;     \
    FLASHINFER_ERROR(err_msg.str());                         \
  }

#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)         \
  switch (mask_mode) {                                        \
    case MaskMode::kNone: {                                   \
      constexpr MaskMode MASK_MODE = MaskMode::kNone;         \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    case MaskMode::kCausal: {                                 \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal;       \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    case MaskMode::kCustom: {                                 \
      constexpr MaskMode MASK_MODE = MaskMode::kCustom;       \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    default: {                                                \
      std::ostringstream err_msg;                             \
      err_msg << "Unsupported mask_mode: " << int(mask_mode); \
      FLASHINFER_ERROR(err_msg.str());                        \
    }                                                         \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)     \
  switch (head_dim) {                                  \
    case 64: {                                         \
      constexpr size_t HEAD_DIM = 64;                  \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 128: {                                        \
      constexpr size_t HEAD_DIM = 128;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 256: {                                        \
      constexpr size_t HEAD_DIM = 256;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 512: {                                        \
      constexpr size_t HEAD_DIM = 512;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    default: {                                         \
      std::ostringstream err_msg;                      \
      err_msg << "Unsupported head_dim: " << head_dim; \
      FLASHINFER_ERROR(err_msg.str());                 \
    }                                                  \
  }

#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)    \
  switch (pos_encoding_mode) {                                                   \
    case PosEncodingMode::kNone: {                                               \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kRoPELlama: {                                          \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama; \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kALiBi: {                                              \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi;     \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::ostringstream err_msg;                                                \
      err_msg << "Unsupported pos_encoding_mode: " << int(pos_encoding_mode);    \
      FLASHINFER_ERROR(err_msg.str());                                           \
    }                                                                            \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      FLASHINFER_ERROR(err_msg.str());                                     \
    }                                                                      \
  }

#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, ...) \
  if (compute_capacity.first >= 8) {                                                        \
    constexpr uint32_t NUM_STAGES_SMEM = 2;                                                 \
    __VA_ARGS__                                                                             \
  } else {                                                                                  \
    constexpr uint32_t NUM_STAGES_SMEM = 1;                                                 \
    __VA_ARGS__                                                                             \
  }

namespace flashinfer {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

inline std::pair<int, int> GetHIPComputeCapability() {
  int device_id = 0;
  hipError_t result = hipGetDevice(&device_id);
  int major = 0, minor = 0;
  hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device_id);
  hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

template <typename T>
inline void DebugPrintCUDAArray(T* device_ptr, size_t size, std::string prefix = "") {
  std::vector<T> host_array(size);
  std::cout << prefix;
  hipMemcpy(host_array.data(), device_ptr, size * sizeof(T), hipMemcpyDeviceToHost);
  for (size_t i = 0; i < size; ++i) {
    std::cout << host_array[i] << " ";
  }
  std::cout << std::endl;
}

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

__device__ __forceinline__ void swap(uint32_t& a, uint32_t& b) {
  uint32_t tmp = a;
  a = b;
  b = tmp;
}

__device__ __forceinline__ uint32_t dim2_offset(const uint32_t& dim_a, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return idx_b * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim3_offset(const uint32_t& dim_b, const uint32_t& dim_a,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return (idx_c * dim_b + idx_b) * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim4_offset(const uint32_t& dim_c, const uint32_t& dim_b,
                                                const uint32_t& dim_a, const uint32_t& idx_d,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return ((idx_d * dim_c + idx_c) * dim_b + idx_b) * dim_a + idx_a;
}


template <typename T>
__device__ __host__ T convert_float_to_16bits(float val) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half(0.f);
  } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
    return __float2bfloat16(0.f);
  } else {
    return val;
  }
}

}  // namespace flashinfer

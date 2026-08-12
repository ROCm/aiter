// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include <cstring>
#include <limits>
#include <optional>

#ifdef USE_ROCM

  #include "quick_all_reduce.cuh"

namespace aiter {
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                                   std::optional<int64_t> qr_max_size) {
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size == 6)
    throw std::invalid_argument("world size == 6 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");
  DeviceComms* fptr = new DeviceComms();
  fptr->init(world_size, rank, qr_max_size);
  return (fptr_t)fptr;
}

void qr_destroy(fptr_t _fa) {
  if (_fa) {
    auto fa = reinterpret_cast<DeviceComms*>(_fa);
    fa->destroy();
    delete fa;
  }
}

void qr_get_handle(fptr_t _fa, int64_t out_ptr) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  hipIpcMemHandle_t handle = fa->get_handle();
  std::memcpy((void*)out_ptr, &handle, sizeof(hipIpcMemHandle_t));
}

void qr_open_handles(fptr_t _fa, const std::vector<int64_t>& handle_ptrs) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  std::vector<hipIpcMemHandle_t> ipc_handles;
  ipc_handles.reserve(handle_ptrs.size());
  for (auto ptr : handle_ptrs) {
    hipIpcMemHandle_t ipc_handle;
    std::memcpy(&ipc_handle, (void*)ptr, sizeof(hipIpcMemHandle_t));
    ipc_handles.push_back(ipc_handle);
  }
  fa->open_ipc_handles(ipc_handles);
}

void qr_all_reduce(fptr_t _fa, const aiter_tensor_t& inp,
                   const aiter_tensor_t& out, int64_t quant_level, bool cast_bf2half) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  HipDeviceGuard device_guard(inp.device_id);
  hipStream_t stream = aiter::getCurrentHIPStream();

  if (inp.dtype() != out.dtype())
    throw std::invalid_argument("qr_all_reduce: inp/out dtype mismatch");
  if (inp.numel() != out.numel())
    throw std::invalid_argument("qr_all_reduce: inp/out numel mismatch");
  if ((int64_t)out.numel() > fa->kMaxProblemSize)
    throw std::invalid_argument("qr_all_reduce: numel exceeds kMaxProblemSize");

  uint32_t N = static_cast<uint32_t>(out.numel());
  if (out.dtype() == AITER_DTYPE_fp16) {
    fa->allreduce<half, false>(reinterpret_cast<half*>(inp.data_ptr()),
                               reinterpret_cast<half*>(out.data_ptr()),
                               N, quant_level, stream);
  } else if (out.dtype() == AITER_DTYPE_bf16) {
    if (cast_bf2half) {
      fa->allreduce<half, true>(reinterpret_cast<half*>(inp.data_ptr()),
                                reinterpret_cast<half*>(out.data_ptr()),
                                N, quant_level, stream);
    } else {
      fa->allreduce<__hip_bfloat16, false>(
          reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
          N, quant_level, stream);
    }
  } else {
    throw std::runtime_error(
        "quick allreduce only supports float16 and bfloat16");
  }
}

void qr_all_reduce_rmsnorm(fptr_t _fa, const aiter_tensor_t& inp,
                           const aiter_tensor_t& residual_inp,
                           const aiter_tensor_t& residual_out, const aiter_tensor_t& out,
                           const aiter_tensor_t& weight, double eps,
                           int64_t hidden_dim, int64_t quant_level,
                           bool cast_bf2half) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  HipDeviceGuard device_guard(inp.device_id);
  hipStream_t stream = aiter::getCurrentHIPStream();

  if (inp.dtype() != out.dtype() || inp.dtype() != residual_inp.dtype() ||
      inp.dtype() != residual_out.dtype() || inp.dtype() != weight.dtype())
    throw std::invalid_argument("qr_all_reduce_rmsnorm: dtype mismatch");
  if (inp.numel() != out.numel() || inp.numel() != residual_inp.numel() ||
      inp.numel() != residual_out.numel())
    throw std::invalid_argument("qr_all_reduce_rmsnorm: numel mismatch");
  if (weight.numel() != (size_t)hidden_dim)
    throw std::invalid_argument("qr_all_reduce_rmsnorm: weight numel != hidden_dim");
  if (hidden_dim <= 0)
    throw std::invalid_argument("qr_all_reduce_rmsnorm: hidden_dim must be > 0");
  if (inp.numel() % hidden_dim != 0)
    throw std::invalid_argument("qr_all_reduce_rmsnorm: numel not divisible by hidden_dim");
  if ((int64_t)out.numel() > fa->kMaxProblemSize)
    throw std::invalid_argument("qr_all_reduce_rmsnorm: numel exceeds kMaxProblemSize");

  uint32_t N = static_cast<uint32_t>(out.numel());
  if (out.dtype() == AITER_DTYPE_fp16) {
    fa->allreduce_rmsnorm<half, half, false>(
        reinterpret_cast<half*>(inp.data_ptr()),
        reinterpret_cast<half*>(residual_inp.data_ptr()),
        reinterpret_cast<half*>(residual_out.data_ptr()),
        reinterpret_cast<half*>(out.data_ptr()),
        reinterpret_cast<half*>(weight.data_ptr()), static_cast<float>(eps),
        N, hidden_dim, quant_level, stream);
  } else if (out.dtype() == AITER_DTYPE_bf16) {
    if (cast_bf2half) {
      fa->allreduce_rmsnorm<__hip_bfloat16, half, true>(
          reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(residual_inp.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(residual_out.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(weight.data_ptr()),
          static_cast<float>(eps), N, hidden_dim, quant_level, stream);
    } else {
      fa->allreduce_rmsnorm<__hip_bfloat16, __hip_bfloat16, false>(
          reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(residual_inp.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(residual_out.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(weight.data_ptr()),
          static_cast<float>(eps), N, hidden_dim, quant_level, stream);
    }
  } else {
    throw std::runtime_error(
        "quick allreduce rmsnorm only supports float16 and bfloat16");
  }
}

int64_t qr_max_size() {
  // The default is 2GB (2,147,483,648 bytes)
  return static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
}

  #define INSTANTIATE_FOR_WORLDSIZE(T, Codec, cast_bf2half)                         \
    template struct AllReduceTwoshot<T, Codec<T, 2>, cast_bf2half>;          \
    template struct AllReduceTwoshot<T, Codec<T, 4>, cast_bf2half>;          \
    template struct AllReduceTwoshot<T, Codec<T, 8>, cast_bf2half>;          \

  // INT3 (CodecQ3) is restricted to TP2 only, so we only instantiate the
  // world_size == 2 kernel for it.
  #define INSTANTIATE_FOR_WORLDSIZE_TP2_ONLY(T, Codec, cast_bf2half)                \
    template struct AllReduceTwoshot<T, Codec<T, 2>, cast_bf2half>;

  #define INSTANTIATE_RMSNORM_FOR_WORLDSIZE(T, CommT, Codec, cast_bf2half)          \
    template struct AllReduceTwoshotRMSNorm<T, CommT, Codec<CommT, 2>, cast_bf2half>; \
    template struct AllReduceTwoshotRMSNorm<T, CommT, Codec<CommT, 4>, cast_bf2half>; \
    template struct AllReduceTwoshotRMSNorm<T, CommT, Codec<CommT, 8>, cast_bf2half>; \

INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP8, false)
INSTANTIATE_FOR_WORLDSIZE_TP2_ONLY(__hip_bfloat16, CodecQ3, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP, true)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ4, true)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ6, true)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP8, true)
INSTANTIATE_FOR_WORLDSIZE_TP2_ONLY(__hip_bfloat16, CodecQ3, true)

INSTANTIATE_FOR_WORLDSIZE(half, CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecFP8, false)
INSTANTIATE_FOR_WORLDSIZE_TP2_ONLY(half, CodecQ3, false)

INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, __hip_bfloat16, CodecFP, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, __hip_bfloat16, CodecQ4, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, __hip_bfloat16, CodecQ6, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, __hip_bfloat16, CodecFP8, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, half, CodecFP, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, half, CodecQ4, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, half, CodecQ6, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, half, CodecFP8, true)

INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, half, CodecFP, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, half, CodecQ4, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, half, CodecQ6, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, half, CodecFP8, false)

#endif  // USE_ROCM
} // namespace aiter
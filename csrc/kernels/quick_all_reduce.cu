// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

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

torch::Tensor qr_get_handle(fptr_t _fa) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  hipIpcMemHandle_t handle = fa->get_handle();
  auto options =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  auto data_handle =
      torch::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))}, options);
  std::memcpy(data_handle.data_ptr(), &handle, sizeof(hipIpcMemHandle_t));
  return data_handle;
}

void qr_open_handles(fptr_t _fa,
                     const std::vector<torch::Tensor>& handles) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  std::vector<hipIpcMemHandle_t> ipc_handles;
  ipc_handles.reserve(handles.size());
  for (auto& handle : handles) {
    // Ensure the tensor is on the same device as the current device.
    hipIpcMemHandle_t ipc_handle;
    std::memcpy(&ipc_handle, handle.data_ptr(), sizeof(hipIpcMemHandle_t));
    ipc_handles.push_back(ipc_handle);
  }
  fa->open_ipc_handles(ipc_handles);
}

void qr_all_reduce(fptr_t _fa, torch::Tensor& inp,
                   torch::Tensor& out, int64_t quant_level, bool cast_bf2half) {
  auto fa = reinterpret_cast<DeviceComms*>(_fa);
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
  auto stream = at::hip::getCurrentHIPStream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_LE(out.numel(), fa->kMaxProblemSize);
  if (out.scalar_type() == at::ScalarType::Half) {
    fa->allreduce<half, false>(reinterpret_cast<half*>(inp.data_ptr()),
                               reinterpret_cast<half*>(out.data_ptr()),
                               out.numel(), quant_level, stream);
  } else if (out.scalar_type() == at::ScalarType::BFloat16) {
    if (cast_bf2half) {
      fa->allreduce<half, true>(reinterpret_cast<half*>(inp.data_ptr()),
                                reinterpret_cast<half*>(out.data_ptr()),
                                out.numel(), quant_level, stream);
    } else {
      fa->allreduce<__hip_bfloat16, false>(
          reinterpret_cast<__hip_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),
          out.numel(), quant_level, stream);
    }
  } else {
    throw std::runtime_error(
        "quick allreduce only supports float16 and bfloat16");
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

INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP8, false)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP, true)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ4, true)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecQ6, true)
INSTANTIATE_FOR_WORLDSIZE(__hip_bfloat16, CodecFP8, true)

INSTANTIATE_FOR_WORLDSIZE(half, CodecFP, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ4, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecQ6, false)
INSTANTIATE_FOR_WORLDSIZE(half, CodecFP8, false)

#endif  // USE_ROCM
} // namespace aiter
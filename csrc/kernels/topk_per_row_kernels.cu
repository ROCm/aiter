// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "dispatch_utils.h"
#include <hipcub/hipcub.hpp>
#include <hipcub/util_type.hpp>

namespace aiter {
template <int TPB>
__launch_bounds__(TPB) __global__ void topKPerRow(const float* logits,
                                                  const int* rowStarts,
                                                  const int* rowEnds,
                                                  int* outIndices,
                                                  float* outLogits,
                                                  int stride0,
                                                  int stride1){

};


} // namespace aiter

void topk_per_row(const torch::Tensor& logits,
                   const torch::Tensor& rowStarts,
                   const torch::Tensor& rowEnds,
                   torch::Tensor& indices,
                   torch::Tensor& values,
                   int64_t numRows,
                   int64_t stride0,
                   int64_t stride1)
{
     // Compute the results on the device.
  constexpr int kNumThreadsPerBlock = 512;
  const hipStream_t stream = at::hip::getCurrentHIPStream();

  aiter::topKPerRow<kNumThreadsPerBlock>
      <<<numRows, kNumThreadsPerBlock, 0, stream>>>(
          logits.data_ptr<float>(), rowStarts.data_ptr<int>(),
          rowEnds.data_ptr<int>(), indices.data_ptr<int>(),
          values.data_ptr<float>(), static_cast<int>(stride0),
          static_cast<int>(stride1));
}
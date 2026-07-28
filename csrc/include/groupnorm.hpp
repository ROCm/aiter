#pragma once

#include "aiter_tensor.h"

namespace aiter {

// Group Normalization. All memory (output and scratch workspace) is allocated by
// the caller (Python) and passed in; the C side only computes.
//
//   y:         output, same shape/dtype as x
//   workspace: float32 scratch, capacity (in elements) must be
//              >= 2 * ceil(4096 / (num_tokens * num_groups)) * (num_tokens * num_groups)
//   x:         input [num_tokens, num_channels, ...], contiguous
//   weight:    affine weight [num_channels], contiguous
//   bias:      affine bias   [num_channels], contiguous
//   epsilon:   numerical stability term
//
// Supported dtypes for x/y/weight/bias: fp32, fp16, bf16.
void groupnorm_run(aiter_tensor_t& y,
                   aiter_tensor_t& workspace,
                   const aiter_tensor_t& x,
                   int num_groups,
                   const aiter_tensor_t& weight,
                   const aiter_tensor_t& bias,
                   double epsilon);

} // namespace aiter

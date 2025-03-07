// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <speculative_sampling.cuh>

#include "speculative_sampling.h"
#include <ATen/cuda/CUDAContext.h>
using namespace flashinfer;


#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, num_spec_step]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// uniform_samples: [bs, num_draft_tokens]
// target_probs: [bs, num_draft_tokens, vocab_size]

void tree_speculative_sampling_target_only(at::Tensor predicts, at::Tensor accept_index,
                                           at::Tensor accept_token_num,  // mutable
                                           at::Tensor candidates, at::Tensor retrive_index,
                                           at::Tensor retrive_next_token, at::Tensor retrive_next_sibling,
                                           at::Tensor uniform_samples, at::Tensor target_probs, at::Tensor draft_probs,
                                           bool deterministic){
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(target_probs);
  auto device = target_probs.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_EQ(target_probs.device(), device);
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, uniform_samples);
  CHECK_DIM(3, target_probs);
  CHECK_DIM(3, draft_probs);
  unsigned int batch_size = uniform_samples.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_probs.size(2);
  CHECK_EQ(batch_size, candidates.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(num_draft_tokens, uniform_samples.size(1));
  CHECK_EQ(num_draft_tokens, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));
  if (predicts.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'predicts' to be of type int (torch.int32).");
  }
  if (accept_index.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_index' to be of type int (torch.int32).");
  }
  if (accept_token_num.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'accept_token_num' to be of type int (torch.int32).");
  }
  if (candidates.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'candidates' to be of type int (torch.int32).");
  }
  if (retrive_index.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'retrive_index' to be of type int (torch.int32).");
  }
  if (retrive_next_token.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'retrive_next_token' to be of type int (torch.int32).");
  }
  if (retrive_next_sibling.scalar_type() != at::kInt) {
    throw std::runtime_error("Expected 'retrive_next_sibling' to be of type int (torch.int32).");
  }
  if (uniform_samples.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'uniform_samples' to be of type float (torch.float32).");
  }
  if (target_probs.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'target_probs' to be of type float (torch.float32).");
  }
  if (draft_probs.scalar_type() != at::kFloat) {
    throw std::runtime_error("Expected 'target_probs' to be of type float (torch.float32).");
  }
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cudaError_t status = sampling::TreeSpeculativeSamplingTargetOnly<float, int>(
      static_cast<int*>(predicts.data_ptr()), static_cast<int*>(accept_index.data_ptr()),
      static_cast<int*>(accept_token_num.data_ptr()), static_cast<int*>(candidates.data_ptr()),
      static_cast<int*>(retrive_index.data_ptr()), static_cast<int*>(retrive_next_token.data_ptr()),
      static_cast<int*>(retrive_next_sibling.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<float*>(target_probs.data_ptr()), static_cast<float*>(draft_probs.data_ptr()), batch_size,
      num_spec_step, num_draft_tokens, vocab_size, deterministic, stream);

  TORCH_CHECK(status == cudaSuccess,
              "TreeSpeculativeSamplingTargetOnly failed with error code " + std::string(cudaGetErrorString(status)));
}

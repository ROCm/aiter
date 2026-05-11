// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Torch-free entry point for the HipKittens MLA decode kernel. Implementation
// in csrc/kernels/mla/hk_decode_fwd.cu dispatches to per-arch .cuh kernels.

#include "aiter_tensor.h"

void hk_mla_decode_fwd(aiter_tensor_t& query,
                       aiter_tensor_t& kv_buffer,
                       const aiter_tensor_t& qo_indptr,
                       const aiter_tensor_t& kv_indptr,
                       const aiter_tensor_t& kv_page_indices,
                       const aiter_tensor_t& kv_last_page_lens,
                       const aiter_tensor_t& work_indptr,
                       const aiter_tensor_t& work_info_set,
                       const int max_seqlen_q,
                       const float softmax_scale,
                       aiter_tensor_t& split_output,
                       aiter_tensor_t& split_lse,
                       aiter_tensor_t& final_output);

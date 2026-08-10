// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

void vsa_sparse_attention_fwd(aiter_tensor_t& q,
                              aiter_tensor_t& k,
                              aiter_tensor_t& v,
                              aiter_tensor_t& block_lut,
                              aiter_tensor_t& block_counts,
                              aiter_tensor_t& out);

#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>
#include <vector>

using fptr_t = int64_t;

namespace aiter {

fptr_t
init_custom_qr(int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
int64_t qr_handle_nbytes();
void qr_get_handle(fptr_t _fa, int64_t out_ptr, int64_t out_nbytes);
void qr_open_handles(fptr_t _fa, const std::vector<int64_t>& handle_ptrs, int64_t handle_nbytes);
void qr_all_reduce(fptr_t _fa,
                   const aiter_tensor_t& inp,
                   const aiter_tensor_t& out,
                   int64_t quant_level,
                   bool cast_bf2half = false);
int64_t qr_max_size();

} // namespace aiter

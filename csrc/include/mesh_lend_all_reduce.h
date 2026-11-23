// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <vector>

using fptr_t = int64_t;

namespace aiter {

/// One rank's end of a mesh-lending all-reduce group of 2 or 4 GPUs, with at
/// most `max_lenders` GPUs outside the group lending their links.
fptr_t mesh_lend_create(
    int64_t device_id, int64_t rank, int64_t world_size, int64_t max_bytes, int64_t max_lenders);

/// What the other ranks need: this rank's GPU, the GPUs its ROCr sees and the
/// IPC handle of its inboxes. Written to host memory at out_ptr,
/// mesh_lend_export_size() bytes.
int64_t mesh_lend_export_size();
void mesh_lend_export(fptr_t fa, int64_t out_ptr);

/// Takes the exports of all ranks in rank order. Picks the lenders, which
/// every rank derives identically, and opens the SDMA queues.
void mesh_lend_connect(fptr_t fa, const std::vector<int64_t>& export_ptrs);

/// out = sum over ranks of inp, bfloat16, on the caller's current stream; out
/// may be inp. bits = 0: exact. bits = 4 or 8 (two ranks only): each rank's
/// input is quantized once; the result is the same on both ranks.
void mesh_lend_all_reduce(fptr_t fa,
                          const aiter_tensor_t& inp,
                          const aiter_tensor_t& out,
                          int64_t bits = 0);

/// GPUs outside the group whose links this instance uses.
int64_t mesh_lend_num_lenders(fptr_t fa);

void mesh_lend_destroy(fptr_t fa);

} // namespace aiter

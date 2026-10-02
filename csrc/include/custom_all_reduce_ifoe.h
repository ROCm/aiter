// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Host API for the IFOE cross-node custom all-reduce (gfx1250).  Peer buffers
// are shared with HIP fabric handles (node independent), so the same kernel
// runs cross-node over the IFOE fabric.  Bootstrap / handle exchange is done in
// Python via torch.distributed; this layer only allocates fabric buffers,
// imports peer handles, and launches the kernel.
#pragma once

#include <cstdint>
#include <vector>

// Raw-pointer API on purpose: this module must not pull in aiter_tensor.h /
// ck_tile, which does not device-compile on gfx1250.
using fptr_t = int64_t;

namespace aiter {

// Allocate a fabric-exportable VMM buffer of `bytes`; write its 64-byte
// hipMemFabricHandle_t to the host buffer at `handle_out_ptr`.  Returns the
// device pointer (as int64).
int64_t ifoe_alloc_fabric(int64_t bytes, int64_t handle_out_ptr);

// Import a peer's fabric buffer from the 64-byte handle at `handle_ptr`.
// Returns the mapped device pointer (as int64).
int64_t ifoe_import_fabric(int64_t handle_ptr, int64_t bytes);

// Build an all-reduce context.  The peer_* vectors are length `world`, indexed
// by rank, and include this rank's own pointers at index `rank`.
fptr_t ifoe_init(int64_t rank,
                 int64_t world,
                 int64_t self_input_ptr,
                 int64_t self_signal_ptr,
                 int64_t self_bf_ptr,
                 const std::vector<int64_t>& peer_input_ptrs,
                 const std::vector<int64_t>& peer_signal_ptrs,
                 const std::vector<int64_t>& peer_bf_ptrs);

// Run one all-reduce.  mode: 0 = fp32 (opt), 1 = bf16 on-wire.  unroll/blocks
// <= 0 select tuned defaults.  The tensor at `inp_ptr` (numel * elt_size bytes,
// fp32) is copied into the registered fabric input buffer; the reduced result
// is written directly to `out_ptr`.
void ifoe_all_reduce(fptr_t ctx,
                     int64_t inp_ptr,
                     int64_t out_ptr,
                     int64_t numel,
                     int64_t elt_size,
                     int64_t mode,
                     int64_t unroll,
                     int64_t blocks);

int64_t ifoe_meta_size(); // sizeof(Signal)
void ifoe_dispose(fptr_t ctx);

} // namespace aiter

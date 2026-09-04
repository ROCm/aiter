// SPDX-License-Identifier: MIT
// Copyright (c) 2026 MoonshotAI
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

#include <cstdint>
#include <hip/hip_runtime.h>

namespace flashkda_hip {

// This layout is part of the hand-off ABI between FlashKDA's preparation and
// recurrence kernels. Keep it in sync with the original FlashKDA HIP backend.
struct WorkspaceSizes {
    static constexpr int CHUNK = 16;
    static constexpr int D = 128;
    static constexpr int kKDecayed = CHUNK * D * 2;
    static constexpr int kQDecayed = CHUNK * D * 2;
    static constexpr int kKRestored = CHUNK * D * 2;
    static constexpr int kGTotal = D * 4;
    static constexpr int kINV = CHUNK * CHUNK * 2;
    static constexpr int kMqk = CHUNK * CHUNK * 2;
    static constexpr int kCsplitU = CHUNK * D * 2;
    static constexpr int kCsplitSin = D * D * 2;
    static constexpr int kCsplitCross = CHUNK * CHUNK * 2;
    static constexpr int kCsplitCross64 = 4 * CHUNK * CHUNK * 2;
    static constexpr int kCsplitBeta = 64 * 4;
    static constexpr int kCsplitSegmentA = 10 * CHUNK * CHUNK * 2;
    static constexpr int64_t kPerTile =
        kKDecayed + kQDecayed + kKRestored + kGTotal + kINV + kMqk;

    // Packed prefix ABI: three (N + 1) integer prefix arrays, one N-entry
    // sequence worklist, one sequence count, and one atomic task counter.
    // Dense calls reserve the same aligned span so all later arenas retain a
    // single architecture-neutral layout.
    static constexpr int64_t prefix_bytes(int64_t N) {
        return ((4 * N + 5) * int64_t(sizeof(int32_t)) + 127) / 128 * 128;
    }
};

int64_t get_workspace_size_hip(int64_t T_total, int64_t H, int64_t N);

// Native FlashKDA launcher. Unlike the upstream extension ABI, aiter keeps
// packed sequence offsets as int32 on device so no per-call conversion kernel
// is needed by K3/ATOM callers.
void launch_fwd_hip(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const void* g_ptr,
    const void* beta_ptr,
    float scale,
    void* out_ptr,
    void* workspace_ptr,
    const float* A_log_ptr,
    const float* dt_bias_ptr,
    float gate_scale,
    int total_tiles,
    int T_total,
    int H_q,
    int H,
    int N,
    const void* init_state,
    void* final_state,
    bool has_state_in,
    bool has_state_out,
    bool state_fp32,
    const int32_t* cu_seqlens,
    hipStream_t stream,
    int max_seqlen_upper_bound);

} // namespace flashkda_hip

namespace aiter {

// aiter JIT/pybind ABI. Optional tensors are represented by an empty tensor
// plus the corresponding boolean, avoiding torch-specific optional types in
// the native implementation. The output and final state are written in place.
void flash_kda_fwd_hip(
    aiter_tensor_t q,
    aiter_tensor_t k,
    aiter_tensor_t v,
    aiter_tensor_t g,
    aiter_tensor_t beta,
    aiter_tensor_t out,
    aiter_tensor_t workspace,
    aiter_tensor_t A_log,
    aiter_tensor_t dt_bias,
    aiter_tensor_t initial_state,
    aiter_tensor_t final_state,
    aiter_tensor_t cu_seqlens,
    double scale,
    double lower_bound,
    bool has_initial_state,
    bool output_final_state,
    bool is_varlen);

// Fast internal ABI for an already-validated Python call.  Tensor dtype/shape
// admission remains the responsibility of the public Python adapter; this
// entry independently validates all integer geometry, pointer presence,
// workspace capacity, state flags, active device, and stream ownership before
// launching.  Explicit device/stream arguments avoid constructing twelve
// transient pybind aiter_tensor_t descriptors on every invocation.
void flash_kda_fwd_hip_raw(
    std::uintptr_t q_ptr,
    std::uintptr_t k_ptr,
    std::uintptr_t v_ptr,
    std::uintptr_t g_ptr,
    std::uintptr_t beta_ptr,
    std::uintptr_t out_ptr,
    std::uintptr_t workspace_ptr,
    std::uintptr_t A_log_ptr,
    std::uintptr_t dt_bias_ptr,
    std::uintptr_t initial_state_ptr,
    std::uintptr_t final_state_ptr,
    std::uintptr_t cu_seqlens_ptr,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t N,
    int64_t workspace_bytes,
    double scale,
    double lower_bound,
    bool has_initial_state,
    bool output_final_state,
    bool is_varlen,
    bool state_fp32,
    int64_t device_id,
    std::uintptr_t stream_ptr);

// Additive raw ABI carrying a graph-safe host routing hint.  The original
// 25-argument symbol above is retained byte-for-byte at the call boundary;
// zero keeps its legacy policy semantics.
void flash_kda_fwd_hip_raw_v2(
    std::uintptr_t q_ptr,
    std::uintptr_t k_ptr,
    std::uintptr_t v_ptr,
    std::uintptr_t g_ptr,
    std::uintptr_t beta_ptr,
    std::uintptr_t out_ptr,
    std::uintptr_t workspace_ptr,
    std::uintptr_t A_log_ptr,
    std::uintptr_t dt_bias_ptr,
    std::uintptr_t initial_state_ptr,
    std::uintptr_t final_state_ptr,
    std::uintptr_t cu_seqlens_ptr,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t N,
    int64_t workspace_bytes,
    double scale,
    double lower_bound,
    bool has_initial_state,
    bool output_final_state,
    bool is_varlen,
    bool state_fp32,
    int64_t device_id,
    std::uintptr_t stream_ptr,
    int64_t max_seqlen_upper_bound);

// Additive grouped-value-attention ABI. ``H`` remains the value/gate/state
// head count carried by raw-v1/v2; ``H_q`` is appended so the established
// argument prefix and all equal-head callers remain source-compatible.
void flash_kda_fwd_hip_raw_v3(
    std::uintptr_t q_ptr,
    std::uintptr_t k_ptr,
    std::uintptr_t v_ptr,
    std::uintptr_t g_ptr,
    std::uintptr_t beta_ptr,
    std::uintptr_t out_ptr,
    std::uintptr_t workspace_ptr,
    std::uintptr_t A_log_ptr,
    std::uintptr_t dt_bias_ptr,
    std::uintptr_t initial_state_ptr,
    std::uintptr_t final_state_ptr,
    std::uintptr_t cu_seqlens_ptr,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t N,
    int64_t workspace_bytes,
    double scale,
    double lower_bound,
    bool has_initial_state,
    bool output_final_state,
    bool is_varlen,
    bool state_fp32,
    int64_t device_id,
    std::uintptr_t stream_ptr,
    int64_t max_seqlen_upper_bound,
    int64_t H_q);

} // namespace aiter

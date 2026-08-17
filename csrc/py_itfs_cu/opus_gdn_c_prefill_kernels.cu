// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Standalone dense C-input GDN prefill launcher for gfx942.

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include "opus_gdn_c_prefill.h"

extern "C" hipError_t opus_gdn_k1_c_fwd(
    const void* ptr_k,
    const void* ptr_g,
    const void* ptr_beta,
    void* ptr_c,
    void* ptr_g_cumsum,
    int B,
    int T,
    int H,
    int Hg,
    const void* ptr_cu_seqlens,
    const void* ptr_chunk_indices,
    int total_chunks,
    hipStream_t stream);

void opus_gdn_k2_c_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor c,
    torch::Tensor beta,
    torch::Tensor g,
    torch::Tensor o,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor chunk_offsets,
    bool has_initial_state,
    bool output_final_state,
    float scale,
    int c_mode,
    bool use_env_overrides = true);

namespace {

constexpr int kCModeAuto = 0;
constexpr int kCModeFused = 1;
constexpr int kCModeSplit = 2;

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    const c10::Device& device) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(
        tensor.device() == device, name, " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(
        tensor.scalar_type() == dtype, name, " has an unexpected dtype");
}

void check_bthd(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    const c10::Device& device,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t D) {
    check_tensor(tensor, name, dtype, device);
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, T, H, D]");
    TORCH_CHECK(
        tensor.size(0) == B && tensor.size(1) == T &&
            tensor.size(2) == H && tensor.size(3) == D,
        name,
        " has an unexpected shape");
}

void check_bth(
    const torch::Tensor& tensor,
    const char* name,
    const c10::Device& device,
    int64_t B,
    int64_t T,
    int64_t H) {
    check_tensor(tensor, name, at::kFloat, device);
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape [B, T, H]");
    TORCH_CHECK(
        tensor.size(0) == B && tensor.size(1) == T && tensor.size(2) == H,
        name,
        " has an unexpected shape");
}

void check_state(
    const torch::Tensor& tensor,
    const char* name,
    const c10::Device& device,
    int64_t B,
    int64_t H) {
    check_tensor(tensor, name, at::kFloat, device);
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, H, V, K]");
    TORCH_CHECK(
        tensor.size(0) == B && tensor.size(1) == H &&
            tensor.size(2) == 128 && tensor.size(3) == 128,
        name,
        " has an unexpected shape");
}

void check_gfx942() {
    int device = 0;
    hipDeviceProp_t properties{};
    const hipError_t device_status = hipGetDevice(&device);
    TORCH_CHECK(
        device_status == hipSuccess,
        "failed to query the active HIP device: ",
        hipGetErrorString(device_status));
    const hipError_t properties_status =
        hipGetDeviceProperties(&properties, device);
    TORCH_CHECK(
        properties_status == hipSuccess,
        "failed to query HIP device properties: ",
        hipGetErrorString(properties_status));
    TORCH_CHECK(
        std::strstr(properties.gcnArchName, "gfx942") != nullptr,
        "opus_gdn_c_prefill currently requires gfx942, got ",
        properties.gcnArchName);
}

int resolve_c_mode(int requested_mode, int64_t T, int64_t batch_heads) {
    TORCH_CHECK(
        requested_mode == kCModeAuto || requested_mode == kCModeFused ||
            requested_mode == kCModeSplit,
        "unsupported c_mode=",
        requested_mode,
        "; expected 0 (auto), 1 (CF), or 2 (CS)");
    if (requested_mode != kCModeAuto) {
        return requested_mode;
    }

    // Conservative subset of the measured 80-CU gfx942 dense envelope.
    // CS is consistently ahead for long, grid-starved chains in these two
    // regions.  Unmeasured/intermediate shapes intentionally fall back to CF;
    // callers that own a workload-specific policy should request 1 or 2.
    const bool use_split =
        (T >= 256 && batch_heads <= 20) ||
        (T >= 128 && batch_heads <= 8);
    return use_split ? kCModeSplit : kCModeFused;
}

}  // namespace

void opus_gdn_c_prefill_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor o,
    float scale,
    torch::Tensor initial_state,
    torch::Tensor final_state,
    torch::Tensor cu_seqlens,
    torch::Tensor chunk_indices,
    torch::Tensor chunk_offsets,
    bool has_initial_state,
    bool output_final_state,
    int c_mode,
    bool use_env_overrides) {
    TORCH_CHECK(q.defined(), "q must be defined");
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must have dtype bfloat16");
    TORCH_CHECK(
        q.dim() == 4 && q.size(3) == 128,
        "q must have shape [B, T, Hg, 128]");
    TORCH_CHECK(
        v.defined() && v.dim() == 4 && v.size(3) == 128,
        "v must have shape [B, T, H, 128]");

    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    // H counts value heads; Hg counts the q/k key heads that H / Hg value
    // heads share.  Hg == H is plain MHA.
    const int64_t H = v.size(2);
    const int64_t Hg = q.size(2);
    const c10::Device device = q.device();
    TORCH_CHECK(B > 0 && T > 0 && H > 0 && Hg > 0,
                "B, T, H, and Hg must be positive");
    TORCH_CHECK(
        H % Hg == 0,
        "v head count ", H,
        " must be a multiple of the q/k head count ", Hg);
    TORCH_CHECK(T % 64 == 0, "T must be divisible by the dense BT=64");
    TORCH_CHECK(std::isfinite(scale), "scale must be finite");
    TORCH_CHECK(
        B <= std::numeric_limits<int>::max() &&
            T <= std::numeric_limits<int>::max() &&
            H <= std::numeric_limits<int>::max(),
        "B, T, and H must fit in int");
    TORCH_CHECK(
        B <= std::numeric_limits<int>::max() / H,
        "B * H must fit in a signed kernel grid index");

    check_bthd(k, "k", at::kBFloat16, device, B, T, Hg, 128);
    check_bthd(v, "v", at::kBFloat16, device, B, T, H, 128);
    check_bthd(o, "o", at::kBFloat16, device, B, T, H, 128);
    TORCH_CHECK(!o.is_alias_of(v), "out must not alias v storage");
    check_bth(g, "g", device, B, T, H);
    check_bth(beta, "beta", device, B, T, H);

    // A packed batch folds every sequence into the token axis, so the state and
    // the auto policy below are sized by the sequence count rather than by B.
    // The C-input kernels emit no token-tail predicate, hence the BT-multiple
    // requirement that total_chunks * 64 == T expresses.
    const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() != 0;
    int64_t num_sequences = B;
    if (is_varlen) {
        TORCH_CHECK(B == 1, "packed varlen expects B=1, got ", B);
        for (const auto& item : {
                 std::pair<const char*, torch::Tensor>("cu_seqlens", cu_seqlens),
                 std::pair<const char*, torch::Tensor>(
                     "chunk_indices", chunk_indices),
                 std::pair<const char*, torch::Tensor>(
                     "chunk_offsets", chunk_offsets)}) {
            check_tensor(item.second, item.first, at::kInt, device);
        }
        TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                    "cu_seqlens must have shape [N + 1]");
        num_sequences = cu_seqlens.numel() - 1;
        TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                        chunk_indices.size(0) > 0,
                    "chunk_indices must have shape [total_chunks, 2]");
        TORCH_CHECK(chunk_indices.size(0) * 64 == T,
                    "the C-input packed path requires every sequence length to "
                    "be a multiple of BT=64");
    }
    if (has_initial_state) {
        check_state(initial_state, "initial_state", device, num_sequences, H);
    }
    if (output_final_state) {
        check_state(final_state, "final_state", device, num_sequences, H);
    }

    const int64_t total_chunks = is_varlen ? chunk_indices.size(0) : T / 64;
    // The CS auto region is a dense grid-starvation policy keyed on a per-batch
    // T, which a packed token total does not describe, so auto takes the
    // family's general-purpose CF path and only an explicit c_mode picks CS.
    const int resolved_mode = (is_varlen && c_mode == kCModeAuto)
        ? kCModeFused
        : resolve_c_mode(c_mode, T, num_sequences * H);
    auto bf16_options =
        torch::TensorOptions().dtype(torch::kBFloat16).device(device);
    auto fp32_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto c = torch::empty({B, T, H, 64}, bf16_options);
    auto g_cumsum = torch::empty({B, T, H}, fp32_options);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
        at::device_of(q));
    check_gfx942();
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    const hipError_t k1_status = opus_gdn_k1_c_fwd(
        k.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        c.data_ptr(),
        g_cumsum.data_ptr(),
        // K1 resolves the sequence from chunk_indices when packed, so it takes
        // the sequence count here for the same reason K2 does.
        static_cast<int>(num_sequences),
        static_cast<int>(T),
        static_cast<int>(H),
        static_cast<int>(Hg),
        is_varlen ? cu_seqlens.data_ptr() : nullptr,
        is_varlen ? chunk_indices.data_ptr() : nullptr,
        is_varlen ? static_cast<int>(total_chunks) : 0,
        stream);
    TORCH_CHECK(
        k1_status == hipSuccess,
        "gdn_k1_neumann_c_kernel launch failed: ",
        hipGetErrorString(k1_status));

    opus_gdn_k2_c_fwd(
        q,
        k,
        v,
        c,
        beta,
        g_cumsum,
        o,
        initial_state,
        final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        has_initial_state,
        output_final_state,
        scale,
        resolved_mode,
        use_env_overrides);
}

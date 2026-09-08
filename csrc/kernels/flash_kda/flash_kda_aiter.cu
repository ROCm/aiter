// SPDX-License-Identifier: MIT
// Copyright (c) 2026 MoonshotAI
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "flash_kda.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void check_gpu_contiguous(const aiter_tensor_t& tensor, const char* name)
{
    AITER_CHECK(tensor.is_gpu(), name, " must be a GPU tensor");
    AITER_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_same_device(const aiter_tensor_t& tensor,
                       const aiter_tensor_t& q,
                       const char* name)
{
    AITER_CHECK(tensor.device_id == q.device_id,
                name,
                " must be on the same GPU as q (expected device ",
                q.device_id,
                ", got ",
                tensor.device_id,
                ")");
}

void check_same_shape(const aiter_tensor_t& tensor,
                      const aiter_tensor_t& q,
                      const char* name)
{
    AITER_CHECK(tensor.dim() == q.dim(), name, " must have the same rank as q");
    for(int i = 0; i < q.dim(); ++i)
    {
        AITER_CHECK(tensor.size(i) == q.size(i),
                    name,
                    " shape must match q at dimension ",
                    i,
                    " (expected ",
                    q.size(i),
                    ", got ",
                    tensor.size(i),
                    ")");
    }
}

void check_state_shape(const aiter_tensor_t& state,
                       int64_t n,
                       int64_t h,
                       const char* name)
{
    constexpr int64_t d = flashkda_hip::WorkspaceSizes::D;
    AITER_CHECK(state.dim() == 4 && state.size(0) == n && state.size(1) == h &&
                    state.size(2) == d && state.size(3) == d,
                name,
                " must have V-first shape [N,H,128,128]; expected [",
                n,
                ",",
                h,
                ",128,128]");
}

[[noreturn]] void raw_fail(const std::string& message)
{
    throw std::invalid_argument("flash_kda_fwd_hip_raw: " + message);
}

void raw_check(bool condition, const char* message)
{
    if(!condition)
        raw_fail(message);
}

void raw_check_pointer(std::uintptr_t pointer, const char* name, std::uintptr_t alignment)
{
    if(pointer == 0)
        raw_fail(std::string(name) + " must not be null");
    if(pointer % alignment != 0)
        raw_fail(std::string(name) + " does not satisfy its required alignment");
}

void raw_check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string("flash_kda_fwd_hip_raw: ") + operation +
                                 " failed: " + hipGetErrorString(status));
    }
}

int64_t checked_raw_workspace_size(int64_t total_tokens, int64_t H, int64_t N)
{
    using W = flashkda_hip::WorkspaceSizes;
    using wide = __int128_t;
    const wide tokens = total_tokens;
    const wide heads = H;
    const wide sequences = N;
    const wide total_tiles = (tokens + W::CHUNK - 1) / W::CHUNK + sequences;
    const wide total_pairs = (tokens + 31) / 32 + sequences;
    const wide total_segments = (tokens + 63) / 64 + sequences;
    const wide prefix_bytes =
        ((4 * sequences + 5) * static_cast<wide>(sizeof(int32_t)) + 127) / 128 * 128;
    const wide required =
        heads * total_tiles * W::kPerTile + prefix_bytes +
        heads * total_tiles * W::kCsplitU +
        heads * total_segments * W::kCsplitSin +
        heads * total_pairs * W::kCsplitCross +
        heads * total_segments * W::kCsplitCross64 +
        heads * total_segments * W::kCsplitBeta +
        heads * total_segments * W::kCsplitSegmentA;
    raw_check(required >= 0 && required <= std::numeric_limits<int64_t>::max(),
              "workspace size overflows int64_t");
    const int64_t checked = static_cast<int64_t>(required);
    raw_check(checked == flashkda_hip::get_workspace_size_hip(total_tokens, H, N),
              "internal workspace-size calculation mismatch");
    return checked;
}

} // namespace

namespace aiter {

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
    bool is_varlen)
{
    check_gpu_contiguous(q, "q");
    check_gpu_contiguous(k, "k");
    check_gpu_contiguous(v, "v");
    check_gpu_contiguous(g, "g");
    check_gpu_contiguous(beta, "beta");
    check_gpu_contiguous(out, "out");
    check_gpu_contiguous(workspace, "workspace");
    check_gpu_contiguous(A_log, "A_log");
    check_gpu_contiguous(dt_bias, "dt_bias");

    check_same_device(k, q, "k");
    check_same_device(v, q, "v");
    check_same_device(g, q, "g");
    check_same_device(beta, q, "beta");
    check_same_device(out, q, "out");
    check_same_device(workspace, q, "workspace");
    check_same_device(A_log, q, "A_log");
    check_same_device(dt_bias, q, "dt_bias");

    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16, "q must be bfloat16");
    AITER_CHECK(k.dtype() == AITER_DTYPE_bf16, "k must be bfloat16");
    AITER_CHECK(v.dtype() == AITER_DTYPE_bf16, "v must be bfloat16");
    AITER_CHECK(g.dtype() == AITER_DTYPE_bf16, "g must be bfloat16");
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16, "out must be bfloat16");
    AITER_CHECK(beta.dtype() == AITER_DTYPE_fp32,
                "beta must be float32 (the Python adapter widens bf16 logits)");
    AITER_CHECK(workspace.dtype() == AITER_DTYPE_u8,
                "workspace must be a byte (uint8) tensor");
    AITER_CHECK(A_log.dtype() == AITER_DTYPE_fp32, "A_log must be float32");
    AITER_CHECK(dt_bias.dtype() == AITER_DTYPE_fp32, "dt_bias must be float32");

    AITER_CHECK(q.dim() == 4, "q must have shape [B,T,H,128]");
    check_same_shape(k, q, "k");
    AITER_CHECK(v.dim() == 4, "v must have shape [B,T,HV,128]");
    check_same_shape(g, v, "g");
    check_same_shape(out, v, "out");

    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t H_q = q.size(2);
    const int64_t H = v.size(2);
    const int64_t K = q.size(3);
    const int64_t V = v.size(3);
    AITER_CHECK(B > 0 && T > 0 && H_q > 0 && H > 0,
                "B, T, H_q and H_v must all be positive");
    AITER_CHECK(v.size(0) == B && v.size(1) == T,
                "v must match q's batch and token dimensions");
    AITER_CHECK(H >= H_q && H % H_q == 0,
                "native FlashKDA requires H_v >= H_q and H_v divisible by H_q");
    AITER_CHECK(K == flashkda_hip::WorkspaceSizes::D &&
                    V == flashkda_hip::WorkspaceSizes::D,
                "native FlashKDA requires K=V=128, got ",
                K,
                " and ",
                V);
    AITER_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == T &&
                    beta.size(2) == H,
                "beta must have shape [B,T,H]");
    AITER_CHECK(A_log.dim() == 1 && A_log.size(0) == H,
                "A_log must have shape [H]");
    AITER_CHECK((dt_bias.dim() == 1 || dt_bias.dim() == 2) &&
                    static_cast<int64_t>(dt_bias.numel()) == H * K,
                "dt_bias must be contiguous float32 [H*128] or [H,128]");
    if(dt_bias.dim() == 2)
    {
        AITER_CHECK(dt_bias.size(0) == H && dt_bias.size(1) == K,
                    "2D dt_bias must have shape [H,128]");
    }

    AITER_CHECK(std::isfinite(scale) && scale > 0.0,
                "scale must be finite and positive, got ",
                scale);
    AITER_CHECK(std::isfinite(lower_bound) && lower_bound >= -5.0 &&
                    lower_bound < 0.0,
                "lower_bound must be in [-5,0), got ",
                lower_bound);

    AITER_CHECK(B <= std::numeric_limits<int>::max() &&
                    T <= std::numeric_limits<int>::max() &&
                    H_q <= std::numeric_limits<int>::max() &&
                    H <= std::numeric_limits<int>::max() &&
                    B <= std::numeric_limits<int64_t>::max() / T,
                "FlashKDA dimensions exceed the native launch ABI");
    const int64_t total_tokens = B * T;
    AITER_CHECK(total_tokens <= std::numeric_limits<int>::max(),
                "B*T exceeds the native launch ABI");

    int64_t N = B;
    const int32_t* cu_seqlens_ptr = nullptr;
    if(is_varlen)
    {
        check_gpu_contiguous(cu_seqlens, "cu_seqlens");
        check_same_device(cu_seqlens, q, "cu_seqlens");
        AITER_CHECK(cu_seqlens.dtype() == AITER_DTYPE_i32,
                    "cu_seqlens must be int32 for the native aiter ABI");
        AITER_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                    "cu_seqlens must have shape [N+1] with N >= 1");
        AITER_CHECK(B == 1, "packed varlen mode requires B == 1");
        N = static_cast<int64_t>(cu_seqlens.numel()) - 1;
        cu_seqlens_ptr = static_cast<const int32_t*>(cu_seqlens.data_ptr());
    }
    AITER_CHECK(N > 0 && N <= std::numeric_limits<int>::max(),
                "number of sequences exceeds the native launch ABI");
    AITER_CHECK(H <= std::numeric_limits<int>::max() / N,
                "N*H_v exceeds the native launch-grid ABI");
    constexpr int64_t max_grid_y = 65535;
    AITER_CHECK(is_varlen ? H <= max_grid_y : H <= max_grid_y / N,
                "FlashKDA dense N*H_v (or packed H_v) exceeds grid.y limit");

    bool state_fp32 = false;
    if(has_initial_state)
    {
        check_gpu_contiguous(initial_state, "initial_state");
        check_same_device(initial_state, q, "initial_state");
        AITER_CHECK(initial_state.dtype() == AITER_DTYPE_bf16 ||
                        initial_state.dtype() == AITER_DTYPE_fp32,
                    "initial_state must be bfloat16 or float32");
        check_state_shape(initial_state, N, H, "initial_state");
        state_fp32 = initial_state.dtype() == AITER_DTYPE_fp32;
    }
    if(output_final_state)
    {
        check_gpu_contiguous(final_state, "final_state");
        check_same_device(final_state, q, "final_state");
        AITER_CHECK(final_state.dtype() == AITER_DTYPE_bf16 ||
                        final_state.dtype() == AITER_DTYPE_fp32,
                    "final_state must be bfloat16 or float32");
        check_state_shape(final_state, N, H, "final_state");
        if(has_initial_state)
        {
            AITER_CHECK(final_state.dtype() == initial_state.dtype(),
                        "initial_state and final_state must have the same dtype");
        }
        state_fp32 = final_state.dtype() == AITER_DTYPE_fp32;
    }

    constexpr int64_t chunk = flashkda_hip::WorkspaceSizes::CHUNK;
    // A one-sequence packed tensor is byte-for-byte identical to the dense
    // B=1 layout: cu_seqlens is necessarily [0, B*T], so no device metadata
    // is needed to resolve a tile.  Dispatch it through the dense kernels to
    // remove the prefix launch, binary searches, and the conservative gap
    // tile.  Multi-sequence packed calls retain the original prefix ABI.
    const bool single_sequence_packed = is_varlen && N == 1;
    const int64_t total_tiles64 = single_sequence_packed
        ? (total_tokens + chunk - 1) / chunk
        : (is_varlen
            ? (total_tokens + chunk - 1) / chunk + N
            : N * ((T + chunk - 1) / chunk));
    AITER_CHECK(total_tiles64 > 0 &&
                    total_tiles64 <= std::numeric_limits<int>::max(),
                "number of FlashKDA tiles exceeds the native launch ABI");

    const int64_t workspace_bytes =
        flashkda_hip::get_workspace_size_hip(total_tokens, H, N);
    const size_t workspace_element_size = workspace.element_size();
    AITER_CHECK(workspace_element_size != 0 &&
                    workspace.numel() <=
                        std::numeric_limits<size_t>::max() / workspace_element_size,
                "workspace byte size overflows size_t");
    const size_t supplied_workspace_bytes =
        workspace.numel() * workspace_element_size;
    AITER_CHECK(workspace_bytes >= 0 &&
                    supplied_workspace_bytes >= static_cast<size_t>(workspace_bytes),
                "workspace is too small: need ",
                workspace_bytes,
                " bytes, got ",
                supplied_workspace_bytes);

    HipDeviceGuard device_guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const float gate_scale =
        static_cast<float>(lower_bound * 1.4426950408889634074); // LOG2E

    flashkda_hip::launch_fwd_hip(
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        static_cast<float>(scale),
        out.data_ptr(),
        workspace.data_ptr(),
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const float*>(dt_bias.data_ptr()),
        gate_scale,
        static_cast<int>(total_tiles64),
        static_cast<int>(total_tokens),
        static_cast<int>(H_q),
        static_cast<int>(H),
        static_cast<int>(N),
        has_initial_state ? initial_state.data_ptr() : nullptr,
        output_final_state ? final_state.data_ptr() : nullptr,
        has_initial_state,
        output_final_state,
        state_fp32,
        single_sequence_packed ? nullptr : cu_seqlens_ptr,
        stream,
        0);
    HIP_CALL_LAUNCH(hipGetLastError());
}

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
    int64_t H_q)
{
    raw_check_pointer(q_ptr, "q_ptr", alignof(uint16_t));
    raw_check_pointer(k_ptr, "k_ptr", alignof(uint16_t));
    raw_check_pointer(v_ptr, "v_ptr", alignof(uint16_t));
    raw_check_pointer(g_ptr, "g_ptr", alignof(uint16_t));
    raw_check_pointer(beta_ptr, "beta_ptr", alignof(float));
    raw_check_pointer(out_ptr, "out_ptr", alignof(uint16_t));
    raw_check_pointer(workspace_ptr, "workspace_ptr", alignof(uint16_t));
    raw_check_pointer(A_log_ptr, "A_log_ptr", alignof(float));
    raw_check_pointer(dt_bias_ptr, "dt_bias_ptr", alignof(float));
    if(has_initial_state)
    {
        raw_check_pointer(initial_state_ptr,
                          "initial_state_ptr",
                          state_fp32 ? alignof(float) : alignof(uint16_t));
    }
    if(output_final_state)
    {
        raw_check_pointer(final_state_ptr,
                          "final_state_ptr",
                          state_fp32 ? alignof(float) : alignof(uint16_t));
    }

    raw_check(B > 0 && T > 0 && H_q > 0 && H > 0 && N > 0,
              "B, T, H_q, H_v, and N must all be positive");
    raw_check(H >= H_q && H % H_q == 0,
              "H_v must be an integer multiple of H_q");
    raw_check(B <= std::numeric_limits<int>::max() &&
                  T <= std::numeric_limits<int>::max() &&
                  H_q <= std::numeric_limits<int>::max() &&
                  H <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max(),
              "B, T, H_q, H_v, or N exceeds the native int launch ABI");
    raw_check(H <= std::numeric_limits<int>::max() / N,
              "N*H_v exceeds the native launch-grid ABI");
    constexpr int64_t max_grid_y = 65535;
    raw_check(is_varlen ? H <= max_grid_y : H <= max_grid_y / N,
              "FlashKDA dense N*H_v (or packed H_v) exceeds grid.y limit");
    raw_check(B <= std::numeric_limits<int64_t>::max() / T,
              "B*T overflows int64_t");
    const int64_t total_tokens = B * T;
    raw_check(total_tokens <= std::numeric_limits<int>::max(),
              "B*T exceeds the native int launch ABI");
    raw_check(workspace_bytes >= 0, "workspace_bytes must be nonnegative");
    raw_check(std::isfinite(scale) && scale > 0.0 &&
                  scale <= std::numeric_limits<float>::max(),
              "scale must be finite, positive, and representable as float");
    raw_check(std::isfinite(lower_bound) && lower_bound >= -5.0 &&
                  lower_bound < 0.0,
              "lower_bound must be finite and in [-5,0)");
    raw_check(device_id >= 0 && device_id <= std::numeric_limits<int>::max(),
              "device_id is outside the HIP device-id range");

    if(is_varlen)
    {
        raw_check(B == 1, "packed varlen mode requires B == 1");
    }
    else
    {
        raw_check(N == B, "dense mode requires N == B");
    }
    raw_check(max_seqlen_upper_bound >= 0,
              "max_seqlen_upper_bound must be nonnegative");
    raw_check(max_seqlen_upper_bound <= std::numeric_limits<int>::max(),
              "max_seqlen_upper_bound exceeds the native int policy ABI");
    if(max_seqlen_upper_bound > 0)
    {
        if(is_varlen)
        {
            const int64_t minimum_upper = total_tokens / N +
                                          (total_tokens % N != 0 ? 1 : 0);
            raw_check(max_seqlen_upper_bound >= minimum_upper &&
                          max_seqlen_upper_bound <= total_tokens,
                      "packed max_seqlen_upper_bound must be in "
                      "[ceil(B*T/N), B*T]");
        }
        else
        {
            raw_check(max_seqlen_upper_bound == T,
                      "dense max_seqlen_upper_bound must be zero or equal T");
        }
    }

    constexpr int64_t chunk = flashkda_hip::WorkspaceSizes::CHUNK;
    const bool single_sequence_packed = is_varlen && N == 1;
    if(is_varlen && !single_sequence_packed)
        raw_check_pointer(cu_seqlens_ptr, "cu_seqlens_ptr", alignof(int32_t));

    const int64_t launch_tiles = single_sequence_packed
        ? (total_tokens + chunk - 1) / chunk
        : (is_varlen ? (total_tokens + chunk - 1) / chunk + N
                     : N * ((T + chunk - 1) / chunk));
    raw_check(launch_tiles > 0 && launch_tiles <= std::numeric_limits<int>::max(),
              "number of launch tiles exceeds the native int ABI");

    const int64_t required_workspace = checked_raw_workspace_size(total_tokens, H, N);
    if(workspace_bytes < required_workspace)
    {
        raw_fail("workspace is too small: need " + std::to_string(required_workspace) +
                 " bytes, got " + std::to_string(workspace_bytes));
    }

    int active_device = -1;
    raw_check_hip(hipGetDevice(&active_device), "hipGetDevice");
    if(active_device != device_id)
    {
        raw_fail("active HIP device " + std::to_string(active_device) +
                 " does not match device_id " + std::to_string(device_id));
    }

    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    hipDevice_t stream_device = -1;
    raw_check_hip(hipStreamGetDevice(stream, &stream_device), "hipStreamGetDevice");
    if(stream_device != device_id)
    {
        raw_fail("stream belongs to device " + std::to_string(stream_device) +
                 ", expected " + std::to_string(device_id));
    }

    const float gate_scale =
        static_cast<float>(lower_bound * 1.4426950408889634074); // LOG2E
    flashkda_hip::launch_fwd_hip(
        reinterpret_cast<const void*>(q_ptr),
        reinterpret_cast<const void*>(k_ptr),
        reinterpret_cast<const void*>(v_ptr),
        reinterpret_cast<const void*>(g_ptr),
        reinterpret_cast<const void*>(beta_ptr),
        static_cast<float>(scale),
        reinterpret_cast<void*>(out_ptr),
        reinterpret_cast<void*>(workspace_ptr),
        reinterpret_cast<const float*>(A_log_ptr),
        reinterpret_cast<const float*>(dt_bias_ptr),
        gate_scale,
        static_cast<int>(launch_tiles),
        static_cast<int>(total_tokens),
        static_cast<int>(H_q),
        static_cast<int>(H),
        static_cast<int>(N),
        has_initial_state ? reinterpret_cast<const void*>(initial_state_ptr) : nullptr,
        output_final_state ? reinterpret_cast<void*>(final_state_ptr) : nullptr,
        has_initial_state,
        output_final_state,
        state_fp32,
        single_sequence_packed ? nullptr
                               : reinterpret_cast<const int32_t*>(cu_seqlens_ptr),
        stream,
        static_cast<int>(max_seqlen_upper_bound));
    raw_check_hip(hipGetLastError(), "hipGetLastError");
}

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
    int64_t max_seqlen_upper_bound)
{
    flash_kda_fwd_hip_raw_v3(q_ptr,
                             k_ptr,
                             v_ptr,
                             g_ptr,
                             beta_ptr,
                             out_ptr,
                             workspace_ptr,
                             A_log_ptr,
                             dt_bias_ptr,
                             initial_state_ptr,
                             final_state_ptr,
                             cu_seqlens_ptr,
                             B,
                             T,
                             H,
                             N,
                             workspace_bytes,
                             scale,
                             lower_bound,
                             has_initial_state,
                             output_final_state,
                             is_varlen,
                             state_fp32,
                             device_id,
                             stream_ptr,
                             max_seqlen_upper_bound,
                             H);
}

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
    std::uintptr_t stream_ptr)
{
    flash_kda_fwd_hip_raw_v2(q_ptr,
                             k_ptr,
                             v_ptr,
                             g_ptr,
                             beta_ptr,
                             out_ptr,
                             workspace_ptr,
                             A_log_ptr,
                             dt_bias_ptr,
                             initial_state_ptr,
                             final_state_ptr,
                             cu_seqlens_ptr,
                             B,
                             T,
                             H,
                             N,
                             workspace_bytes,
                             scale,
                             lower_bound,
                             has_initial_state,
                             output_final_state,
                             is_varlen,
                             state_fp32,
                             device_id,
                             stream_ptr,
                             0);
}

} // namespace aiter

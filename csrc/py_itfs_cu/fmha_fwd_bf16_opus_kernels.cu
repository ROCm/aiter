// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared host launcher for the OPUS gfx950 bf16 flash-attention forward kernels.
// A single entry point (`fmha_fwd_bf16_opus_fwd`) dispatches by head dim:
//   * (D_QK,D_V) = (128,128) -> gqa_d128_kernel        (batch mode only; logic unchanged)
//   * (D_QK,D_V) = (192,128) -> gqa_d192_v128_kernel   (batch + group / varlen)
//
// Both device kernel templates are pulled in (IMPL-guarded, single-header) so they can
// be launched from this one translation unit.

#define FMHA_FWD_HD128_BF16_OPUS_IMPL
#include "fmha_fwd_hd128_bf16_opus.h"
#define FMHA_FWD_HD192_V128_BF16_OPUS_IMPL
#include "fmha_fwd_hd192_v128_bf16_opus.h"

#include "fmha_fwd_bf16_opus.h"
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

#include <cmath>

namespace {

// ─── D_QK=128 / D_V=128 (symmetric) launch — logic unchanged from the original
//     fmha_fwd_hd128_bf16_opus_fwd, only moved under the shared entry point. ───
void launch_d128(aiter_tensor_t& q,
                 aiter_tensor_t& k,
                 aiter_tensor_t& v,
                 aiter_tensor_t& out,
                 bool causal,
                 float softmax_scale)
{
    AITER_CHECK(q.dim() == 4, "q must be 4-D [B, N, H, D], got ndim=", q.dim());
    AITER_CHECK(k.dim() == 4, "k must be 4-D [B, N, H_KV, D], got ndim=", k.dim());
    AITER_CHECK(v.dim() == 4, "v must be 4-D [B, N, H_KV, D], got ndim=", v.dim());
    AITER_CHECK(out.dim() == 4, "out must be 4-D [B, N, H, D], got ndim=", out.dim());

    const int B    = static_cast<int>(q.size(0));
    const int N    = static_cast<int>(q.size(1));      // seqlen_q
    const int H    = static_cast<int>(q.size(2));
    const int D    = static_cast<int>(q.size(3));
    const int H_KV = static_cast<int>(k.size(2));
    const int N_KV = static_cast<int>(k.size(1));      // seqlen_kv (cross-attn: may != N)

    AITER_CHECK(D == 128, "launch_d128 only compiles D=128, got D=", D);
    AITER_CHECK(k.size(0) == B && v.size(0) == B, "k/v batch must equal q batch B");
    AITER_CHECK(v.size(1) == N_KV, "k/v seqlen must match (v seqlen != k seqlen)");
    AITER_CHECK(v.size(2) == H_KV, "k/v must share H_KV");
    AITER_CHECK(k.size(3) == D && v.size(3) == D, "k/v head dim must equal D=128");
    AITER_CHECK(H_KV > 0 && (H % H_KV) == 0, "H must be divisible by H_KV (GQA group)");
    AITER_CHECK(out.size(0) == B && out.size(1) == N && out.size(2) == H && out.size(3) == D,
                "out shape must match q [B, N, H, D]");

    AITER_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1 && out.stride(3) == 1,
                "q/k/v/out must be contiguous along the head dim D");

    // 32-bit KV buffer-offset guard: extent >= 2^32 wraps the async-load soffset (silent
    // wrong output), reject instead.
    const long long kv_slice_bytes = (long long)N_KV * (long long)k.stride(1) * 2LL;  // bf16
    AITER_CHECK(kv_slice_bytes < (1LL << 32),
                "OPUS D=128: KV byte extent ", kv_slice_bytes,
                " reaches the 32-bit buffer-offset limit (2^32); reduce seqlen_kv or use another backend");

    if (B == 0 || N == 0 || H == 0) return;

    opus_gqa_kargs kargs{};
    kargs.ptr_q = q.data_ptr();
    kargs.ptr_k = k.data_ptr();
    kargs.ptr_v = v.data_ptr();
    kargs.ptr_o = out.data_ptr();
    kargs.B     = B;
    kargs.N     = N;
    kargs.N_KV  = N_KV;
    kargs.H     = H;
    kargs.H_KV  = H_KV;
    kargs.D     = D;
    kargs.stride_q_b  = static_cast<int>(q.stride(0));
    kargs.stride_q_n  = static_cast<int>(q.stride(1));
    kargs.stride_q_h  = static_cast<int>(q.stride(2));
    kargs.stride_kv_b = static_cast<int>(k.stride(0));
    kargs.stride_kv_n = static_cast<int>(k.stride(1));
    kargs.stride_kv_h = static_cast<int>(k.stride(2));

    if (softmax_scale <= 0.0f) {
        softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
    }
    kargs.softmax_scale = softmax_scale;  // kernel applies scale * log2(e) to Q

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    using TraitsCausal    = opus_gqa_traits<32, 64, 128, 8, true>;
    using TraitsNonCausal = opus_gqa_traits<32, 64, 128, 8, false>;

    auto launch = [&](auto traits_tag) {
        using Traits          = decltype(traits_tag);
        const int num_q_tiles = ceil_div(N, Traits::Q_TILE_SIZE);
        const int num_q_blk   = ceil_div(num_q_tiles, Traits::NUM_WARPS);
        dim3 grid(H, num_q_blk, B);
        dim3 block(Traits::BLOCK_SIZE);
        gqa_d128_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    if (causal) {
        launch(TraitsCausal{});
    } else {
        launch(TraitsNonCausal{});
    }
}

// ─── D_QK=192 / D_V=128 (asymmetric) launch — batch + group (varlen). ───
void launch_d192_v128(aiter_tensor_t& q,
                      aiter_tensor_t& k,
                      aiter_tensor_t& v,
                      aiter_tensor_t& out,
                      bool causal,
                      float softmax_scale,
                      std::optional<aiter_tensor_t>& seqstart_q,
                      std::optional<aiter_tensor_t>& seqstart_k,
                      std::optional<aiter_tensor_t>& seqstart_q_pad,
                      std::optional<aiter_tensor_t>& seqstart_k_pad,
                      int max_seqlen_q,
                      int max_seqlen_k)
{
    constexpr int D_QK = 192;
    constexpr int D_V  = 128;
    constexpr int Q_TILE_SIZE = 32, KV_TILE_SIZE = 64, NUM_WARPS = 8;
    constexpr int Q_BLOCK = Q_TILE_SIZE * NUM_WARPS;  // 256

    const bool is_group = seqstart_q.has_value() && seqstart_q->numel() > 0;

    opus_gqa_d192_kargs kargs{};
    kargs.ptr_q = q.data_ptr();
    kargs.ptr_k = k.data_ptr();
    kargs.ptr_v = v.data_ptr();
    kargs.ptr_o = out.data_ptr();
    kargs.D_QK  = D_QK;
    kargs.D_V   = D_V;

    if (softmax_scale <= 0.0f) {
        softmax_scale = 1.0f / std::sqrt(static_cast<float>(D_QK));
    }
    // Plumbed into the kernel via kargs; the kernel folds in log2(e) for its exp2 softmax.
    kargs.softmax_scale = softmax_scale;

    HipDeviceGuard guard(q.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    int B, N, N_KV, H, H_KV;
    int num_q_blocks;   // for grid / merge decision
    int grid_x, grid_y, grid_z;

    if (is_group) {
        // Packed / varlen: q [total_q, H, D_QK], k [total_k, H_KV, D_QK],
        // v [total_k, H_KV, D_V], out [total_q, H, D_V]. group = num sequences.
        AITER_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && out.dim() == 3,
                    "group mode expects packed 3-D q/k/v/out [total, H, D]");
        AITER_CHECK(static_cast<int>(q.size(2)) == D_QK && static_cast<int>(k.size(2)) == D_QK,
                    "group mode q/k head dim must be 192");
        AITER_CHECK(static_cast<int>(v.size(2)) == D_V && static_cast<int>(out.size(2)) == D_V,
                    "group mode v/out head dim must be 128");
        AITER_CHECK(seqstart_q.has_value() && seqstart_k.has_value(),
                    "group mode requires seqstart_q and seqstart_k");
        H    = static_cast<int>(q.size(1));
        H_KV = static_cast<int>(k.size(1));
        AITER_CHECK(static_cast<int>(v.size(1)) == H_KV, "group mode k/v must share H_KV");
        AITER_CHECK(static_cast<int>(out.size(0)) == static_cast<int>(q.size(0)) &&
                    static_cast<int>(out.size(1)) == H,
                    "group mode out must be [total_q, H, D_V]");
        B    = static_cast<int>(seqstart_q->numel()) - 1;   // num groups
        AITER_CHECK(B > 0, "group mode requires seqstart_q length >= 2");
        AITER_CHECK(max_seqlen_q > 0 && max_seqlen_k > 0,
                    "group mode requires max_seqlen_q / max_seqlen_k > 0");
        N    = max_seqlen_q;
        N_KV = max_seqlen_k;

        // Validate the cumulative-length arrays before reinterpreting their storage as
        // int32: a wrong dtype (e.g. int64), non-contiguous layout, or wrong length would
        // otherwise silently corrupt the per-group offsets or fault.
        auto check_seqstart = [&](const aiter_tensor_t& s, const char* name) {
            AITER_CHECK(s.dtype() == AITER_DTYPE_i32, name, " must be int32");
            AITER_CHECK(s.dim() == 1, name, " must be 1-D");
            AITER_CHECK(s.is_contiguous(), name, " must be contiguous");
            AITER_CHECK(static_cast<int>(s.numel()) == B + 1, name, " length must be num_groups+1");
        };
        check_seqstart(*seqstart_q, "seqstart_q");
        check_seqstart(*seqstart_k, "seqstart_k");
        if (seqstart_q_pad.has_value()) check_seqstart(*seqstart_q_pad, "seqstart_q_pad");
        if (seqstart_k_pad.has_value()) check_seqstart(*seqstart_k_pad, "seqstart_k_pad");

        // Packed single-sequence strides (no batch stride).
        kargs.stride_q_b = 0; kargs.stride_q_n = static_cast<int>(q.stride(0));   kargs.stride_q_h = static_cast<int>(q.stride(1));
        kargs.stride_o_b = 0; kargs.stride_o_n = static_cast<int>(out.stride(0)); kargs.stride_o_h = static_cast<int>(out.stride(1));
        kargs.stride_k_b = 0; kargs.stride_k_n = static_cast<int>(k.stride(0));   kargs.stride_k_h = static_cast<int>(k.stride(1));
        kargs.stride_v_b = 0; kargs.stride_v_n = static_cast<int>(v.stride(0));   kargs.stride_v_h = static_cast<int>(v.stride(1));

        kargs.ptr_seqstart_q     = reinterpret_cast<const int*>(seqstart_q->data_ptr());
        kargs.ptr_seqstart_k     = reinterpret_cast<const int*>(seqstart_k->data_ptr());
        kargs.ptr_seqstart_q_pad = reinterpret_cast<const int*>(
            (seqstart_q_pad.has_value() ? *seqstart_q_pad : *seqstart_q).data_ptr());
        kargs.ptr_seqstart_k_pad = reinterpret_cast<const int*>(
            (seqstart_k_pad.has_value() ? *seqstart_k_pad : *seqstart_k).data_ptr());

        // Rotated axis order (matches production asm GROUP_MODE): head=x, group=y, Q-block=z.
        num_q_blocks = ceil_div(N, Q_BLOCK);          // nqb_cap from max_seqlen_q
    } else {
        // Dense batch: q/k/v/out 4-D [B, N, H, D]. Cross-attention allowed (N != N_KV).
        AITER_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
                    "batch mode expects 4-D q/k/v/out [B, N, H, D]");
        B    = static_cast<int>(q.size(0));
        N    = static_cast<int>(q.size(1));
        H    = static_cast<int>(q.size(2));
        H_KV = static_cast<int>(k.size(2));
        N_KV = static_cast<int>(k.size(1));
        AITER_CHECK(v.size(0) == B && v.size(1) == N_KV && v.size(2) == H_KV,
                    "k/v must share [B, N_KV, H_KV]");
        AITER_CHECK(static_cast<int>(q.size(3)) == D_QK && static_cast<int>(k.size(3)) == D_QK,
                    "q/k head dim must be 192");
        AITER_CHECK(static_cast<int>(v.size(3)) == D_V && static_cast<int>(out.size(3)) == D_V,
                    "v/out head dim must be 128");
        AITER_CHECK(out.size(0) == B && out.size(1) == N && out.size(2) == H,
                    "out shape must match q [B, N, H, D_V]");

        kargs.stride_q_b = static_cast<int>(q.stride(0));   kargs.stride_q_n = static_cast<int>(q.stride(1));   kargs.stride_q_h = static_cast<int>(q.stride(2));
        kargs.stride_o_b = static_cast<int>(out.stride(0)); kargs.stride_o_n = static_cast<int>(out.stride(1)); kargs.stride_o_h = static_cast<int>(out.stride(2));
        kargs.stride_k_b = static_cast<int>(k.stride(0));   kargs.stride_k_n = static_cast<int>(k.stride(1));   kargs.stride_k_h = static_cast<int>(k.stride(2));
        kargs.stride_v_b = static_cast<int>(v.stride(0));   kargs.stride_v_n = static_cast<int>(v.stride(1));   kargs.stride_v_h = static_cast<int>(v.stride(2));

        num_q_blocks = ceil_div(N, Q_BLOCK);
    }

    AITER_CHECK(H_KV > 0 && (H % H_KV) == 0, "H must be divisible by H_KV (GQA group)");
    AITER_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1,
                "q/k/v/out must be contiguous along the head dim");
    if (B == 0 || H == 0) return;

    kargs.B = B; kargs.N = N; kargs.N_KV = N_KV; kargs.H = H; kargs.H_KV = H_KV;

    // 32-bit KV buffer-offset guard (same as D=128); N_KV bounds the per-group extent in
    // group mode (max_seqlen_k).
    const long long k_slice_bytes = (long long)N_KV * (long long)kargs.stride_k_n * 2LL;  // bf16
    const long long v_slice_bytes = (long long)N_KV * (long long)kargs.stride_v_n * 2LL;
    AITER_CHECK(k_slice_bytes < (1LL << 32) && v_slice_bytes < (1LL << 32),
                "OPUS D_QK=192/D_V=128: K/V byte extent (k=", k_slice_bytes, " v=", v_slice_bytes,
                ") reaches the 32-bit buffer-offset limit (2^32); reduce seqlen_kv");

    // Head/tail merge (causal load balance): host is the single source of truth; the
    // kernel reads the OPT_MERGE_HEADTAIL bit and never recomputes it.
    const bool small_shape = (long long)num_q_blocks * H * B < (long long)HEADTAIL_MIN_WG;
    const bool merge_ht    = causal && !small_shape;
    kargs.opt = merge_ht ? OPT_MERGE_HEADTAIL : 0;

    if (is_group) {
        grid_x = H;
        grid_y = B;
        grid_z = merge_ht ? ceil_div(num_q_blocks, 2) : num_q_blocks;
    } else {
        grid_x = merge_ht ? ceil_div(num_q_blocks, 2) : num_q_blocks;   // config A: q-block=x
        grid_y = H;
        grid_z = B;
    }
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(NUM_WARPS * 64);

    auto launch = [&](auto traits_tag) {
        using Traits = decltype(traits_tag);
        gqa_d192_v128_kernel<Traits><<<grid, block, 0, stream>>>(kargs);
        HIP_CALL_LAUNCH(hipGetLastError());
    };

    if (is_group) {
        if (causal) launch(opus_gqa_d192_traits<32, 64, 8, true,  true>{});
        else        launch(opus_gqa_d192_traits<32, 64, 8, false, true>{});
    } else {
        if (causal) launch(opus_gqa_d192_traits<32, 64, 8, true,  false>{});
        else        launch(opus_gqa_d192_traits<32, 64, 8, false, false>{});
    }
}

} // namespace

void fmha_fwd_bf16_opus_fwd(aiter_tensor_t& q,
                            aiter_tensor_t& k,
                            aiter_tensor_t& v,
                            aiter_tensor_t& out,
                            bool causal,
                            float softmax_scale,
                            std::optional<aiter_tensor_t> seqstart_q,
                            std::optional<aiter_tensor_t> seqstart_k,
                            std::optional<aiter_tensor_t> seqstart_q_pad,
                            std::optional<aiter_tensor_t> seqstart_k_pad,
                            int max_seqlen_q,
                            int max_seqlen_k)
{
    AITER_CHECK(q.dtype() == k.dtype() && q.dtype() == v.dtype() && q.dtype() == out.dtype(),
                "q/k/v/out must share dtype");
    AITER_CHECK(q.dtype() == AITER_DTYPE_bf16, "fmha_fwd_bf16_opus_fwd only supports bf16");

    const int D_QK = static_cast<int>(q.size(-1));
    const int D_V  = static_cast<int>(v.size(-1));
    const bool is_group = seqstart_q.has_value() && seqstart_q->numel() > 0;

    if (D_QK == 128 && D_V == 128) {
        AITER_CHECK(!is_group, "OPUS D=128 kernel supports batch mode only (no varlen)");
        launch_d128(q, k, v, out, causal, softmax_scale);
    } else if (D_QK == 192 && D_V == 128) {
        launch_d192_v128(q, k, v, out, causal, softmax_scale,
                         seqstart_q, seqstart_k, seqstart_q_pad, seqstart_k_pad,
                         max_seqlen_q, max_seqlen_k);
    } else {
        AITER_CHECK(false, "OPUS fwd supports (D_QK,D_V) in {(128,128),(192,128)}, got (",
                    D_QK, ",", D_V, ")");
    }
}

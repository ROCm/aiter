#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cstdlib>
#include <limits>

#include "opus_gdn/gdn_k2_c_defs.h"

// Keep the device implementation out of the host compiler pass.  This is the
// same split used by the existing K2 translation unit: the host pass only
// needs a launchable kernel declaration, while hipcc's device pass sees the
// full MFMA implementation.
#ifndef __HIP_DEVICE_COMPILE__
template <typename Traits>
__global__ void gdn_k2_c_kernel(gdn_k2_c_kargs) {}
#else
#include "opus_gdn/gdn_k2_c_kernel_template.hpp"
#endif

// Reuse the tuned chunk-parallel K6 from the existing W/U split path.  Its
// inputs are only q/k/g, the pre-update H snapshots, and corrected values.
template <typename Traits>
__global__ void gdn_k2_out_kernel(gdn_k2_kargs);

using gdn_k2_c_bt64_traits = gdn_k2_c_traits<64, 128, 128, 64, 4>;
using gdn_k2_c_bt64_persist_k_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, true>;
using gdn_k2_c_bt64_persist_k_gate_cache_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, true, true>;
using gdn_k2_c_bt64_low_lds_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, false, false, 2>;
using gdn_k2_c_bt64_low_lds_prefetch_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true>;
using gdn_k2_c_bt64_low_lds_gate_cache_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, false>;
using gdn_k2_c_bt64_low_lds_q_prefetch_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, false, false, 2, true>;
using gdn_k2_c_bt64_low_lds_relaxed_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true>;
using gdn_k2_c_bt64_low_lds_relaxed_vec_c_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    true>;
using gdn_k2_c_bt64_low_lds_relaxed_retain_k_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true>;
using gdn_k2_c_bt64_low_lds_direct_av_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true>;
using gdn_k2_c_bt64_low_lds_wave_owned_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true, true>;
using gdn_k2_c_bt64_low_lds_wave_owned_fused_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true, true, true>;
using gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true, true, true, true>;
using gdn_k2_c_bt64_low_lds_wave_owned_fused_unroll_d_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true, true, true, false, true>;
using gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d1_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true, true, true, false, true, 1>;
using gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d2_traits =
    gdn_k2_c_traits<64, 128, 128, 64, 4, false, true, false, 2, true, true,
                    false, true, true, true, true, false, true, 2>;
template <int BV>
using gdn_k2_c_bt64_split_scan_traits =
    gdn_k2_c_traits<64, 128, 128, BV, 4,
                    false, true, false, 2, false, true, true,
                    (BV == 64), true, true, true, false, true, 2, true>;

// Packed varlen C-input kernels.  This family emits no token-tail predicate at
// all, so a packed batch only needs the metadata addressing and the host
// guarantees BT divides every sequence; that is why there is no separate
// aligned/ragged pair the way the W/U fused kernel has.  Only the measured
// default fused variant is packed-aware, matching the split scan's BV sweep.
using gdn_k2_c_bt64_varlen_fused_traits = gdn_varlen_traits<
    gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d2_traits>;
template <int BV>
using gdn_k2_c_bt64_varlen_split_scan_traits =
    gdn_varlen_traits<gdn_k2_c_bt64_split_scan_traits<BV>>;

// Packed varlen K6 for the split C path.  Both forms are already instantiated
// by opus_gdn_k2_out.cu, which this module also compiles.  Aligned is the
// default because a packed C batch is BT-aligned by contract; the predicated
// generic form stays available as the runtime rollback.
using gdn_k2_c_varlen_out_aligned_traits = gdn_varlen_aligned_traits<
    gdn_k2_out_traits<gdn_k2_traits<64, 128, 128, 128, 8>, true, false>>;
using gdn_k2_c_varlen_out_generic_traits = gdn_varlen_traits<
    gdn_k2_out_traits<gdn_k2_traits<64, 128, 128, 128, 8>>>;

template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_persist_k_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_persist_k_gate_cache_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_prefetch_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_gate_cache_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_q_prefetch_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_relaxed_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_relaxed_vec_c_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_relaxed_retain_k_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_direct_av_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d_traits>(
    gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_unroll_d_traits>(
    gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d1_traits>(
    gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d2_traits>(
    gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_split_scan_traits<16>>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_split_scan_traits<32>>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_split_scan_traits<64>>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_varlen_fused_traits>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_varlen_split_scan_traits<16>>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_varlen_split_scan_traits<32>>(gdn_k2_c_kargs);
template __global__ void
gdn_k2_c_kernel<gdn_k2_c_bt64_varlen_split_scan_traits<64>>(gdn_k2_c_kargs);

namespace {

void check_cuda_contiguous(const torch::Tensor& tensor,
                           const char* name,
                           at::ScalarType dtype,
                           const c10::Device& device) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same device as q");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unexpected dtype");
}

void check_bth(const torch::Tensor& tensor,
               const char* name,
               at::ScalarType dtype,
               const c10::Device& device,
               int64_t B,
               int64_t T,
               int64_t H,
               int64_t D) {
    check_cuda_contiguous(tensor, name, dtype, device);
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, T, H, D]");
    TORCH_CHECK(tensor.size(0) == B && tensor.size(1) == T &&
                    tensor.size(2) == H && tensor.size(3) == D,
                name, " has an unexpected shape");
}

void check_bth_scalar(const torch::Tensor& tensor,
                      const char* name,
                      const c10::Device& device,
                      int64_t B,
                      int64_t T,
                      int64_t H) {
    check_cuda_contiguous(tensor, name, at::kFloat, device);
    TORCH_CHECK(tensor.dim() == 3, name, " must have shape [B, T, H]");
    TORCH_CHECK(tensor.size(0) == B && tensor.size(1) == T && tensor.size(2) == H,
                name, " has an unexpected shape");
}

dim3 flat_y_grid(unsigned int x, int64_t logical_y) {
    // gfx942 exposes only a 16-bit blockIdx.y/z.  A packed batch can hold far
    // more (sequence, head) pairs than that, so keep the dense grid exactly and
    // slice any larger logical extent across z.
    constexpr int64_t SAFE_GRID_Y = 65535;
    TORCH_CHECK(logical_y > 0 && logical_y <= std::numeric_limits<int>::max(),
                "logical grid y must be positive and fit in int, got ",
                logical_y);
    const auto y = static_cast<unsigned int>(
        logical_y < SAFE_GRID_Y ? logical_y : SAFE_GRID_Y);
    const auto z = static_cast<unsigned int>(
        logical_y / y + (logical_y % y != 0));
    TORCH_CHECK(z <= SAFE_GRID_Y,
                "logical grid y requires too many z slices: ", z);
    return dim3(x, y, z);
}

void check_varlen_metadata(const torch::Tensor& tensor,
                           const char* name,
                           const c10::Device& device) {
    check_cuda_contiguous(tensor, name, at::kInt, device);
}

void check_state(const torch::Tensor& tensor,
                 const char* name,
                 const c10::Device& device,
                 int64_t B,
                 int64_t H) {
    check_cuda_contiguous(tensor, name, at::kFloat, device);
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, H, V, K]");
    TORCH_CHECK(tensor.size(0) == B && tensor.size(1) == H &&
                    tensor.size(2) == 128 && tensor.size(3) == 128,
                name, " has an unexpected shape");
}

} // namespace

// Internal ABI for the C-input backend.  The public launcher resolves auto into
// an explicit mode before entering this translation unit:
//   1 = CF (fused recurrence/output), 2 = CS (split scan + shared K6).
// An empty cu_seqlens selects the dense layout; otherwise q/k/v are packed as
// [1, total_tokens, ...] and B below counts sequences.
void opus_gdn_k2_c_fwd(torch::Tensor q,
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
                       bool use_env_overrides) {
    TORCH_CHECK(q.defined(), "q must be defined");
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must have dtype bfloat16");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == 128,
                "q must have shape [B, T, Hg, 128]");
    TORCH_CHECK(v.defined() && v.dim() == 4 && v.size(3) == 128,
                "v must have shape [B, T, H, 128]");

    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    // H counts value heads (v/C/o/g/beta/state); Hg counts the q/k key heads
    // that H / Hg value heads share.  Every grid and buffer below stays
    // value-head indexed, so only q/k addressing changes for GQA.
    const int64_t H = v.size(2);
    const int64_t Hg = q.size(2);
    const c10::Device device = q.device();

    TORCH_CHECK(B > 0 && T > 0 && H > 0 && Hg > 0,
                "B, T, H, and Hg must all be positive");
    TORCH_CHECK(H % Hg == 0,
                "v head count ", H,
                " must be a multiple of the q/k head count ", Hg);
    TORCH_CHECK(T % 64 == 0,
                "the K2-C prototype only supports sequence lengths divisible by 64");
    TORCH_CHECK(B <= std::numeric_limits<int>::max() &&
                    T <= std::numeric_limits<int>::max() &&
                    H <= std::numeric_limits<int>::max(),
                "B, T, and H must fit in int");
    TORCH_CHECK(B <= std::numeric_limits<int>::max() / H,
                "B * H must fit in a signed kernel grid index");
    TORCH_CHECK(H <= std::numeric_limits<int>::max() / (64 * 128),
                "one BT64 HBM tile stride must fit in int");

    // Packed batches replace the batch axis with a sequence axis.  The C-input
    // kernels carry no token-tail predicate, so a packed batch is only
    // supported when BT divides every sequence; total_chunks * BT == T is
    // exactly that condition.
    const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() != 0;
    int64_t num_sequences = B;
    int64_t total_chunks = T / 64;
    if (is_varlen) {
        TORCH_CHECK(B == 1,
                    "packed varlen expects q/k/v batch dimension B=1, got ", B);
        check_varlen_metadata(cu_seqlens, "cu_seqlens", device);
        check_varlen_metadata(chunk_indices, "chunk_indices", device);
        check_varlen_metadata(chunk_offsets, "chunk_offsets", device);
        TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                    "cu_seqlens must have shape [N + 1]");
        num_sequences = cu_seqlens.numel() - 1;
        TORCH_CHECK(chunk_offsets.dim() == 1 &&
                        chunk_offsets.numel() == num_sequences + 1,
                    "chunk_offsets must have shape [N + 1]");
        TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                        chunk_indices.size(0) > 0,
                    "chunk_indices must have shape [total_chunks, 2]");
        total_chunks = chunk_indices.size(0);
        TORCH_CHECK(total_chunks * 64 == T,
                    "the C-input packed path requires every sequence length to "
                    "be a multiple of BT=64; got total_tokens=", T,
                    " for ", total_chunks, " chunks");
        TORCH_CHECK(num_sequences <= std::numeric_limits<int>::max() / H,
                    "N * H must fit in a signed kernel grid index");
    } else {
        TORCH_CHECK(!chunk_indices.defined() || chunk_indices.numel() == 0,
                    "chunk_indices must be empty for dense input");
        TORCH_CHECK(!chunk_offsets.defined() || chunk_offsets.numel() == 0,
                    "chunk_offsets must be empty for dense input");
    }
    // Every grid and buffer below is indexed by the logical sequence axis,
    // which is the batch axis for dense input and N for a packed batch.
    const int64_t nh64 = num_sequences * H;
    const unsigned int grid_bh = static_cast<unsigned int>(nh64);

    check_bth(k, "k", at::kBFloat16, device, B, T, Hg, 128);
    check_bth(v, "v", at::kBFloat16, device, B, T, H, 128);
    check_bth(c, "c", at::kBFloat16, device, B, T, H, 64);
    check_bth_scalar(beta, "beta", device, B, T, H);
    check_bth_scalar(g, "g", device, B, T, H);
    check_bth(o, "o", at::kBFloat16, device, B, T, H, 128);

    // State is per sequence, so a packed batch carries N entries rather than B.
    if (has_initial_state) {
        check_state(initial_state, "initial_state", device, num_sequences, H);
    }
    if (output_final_state) {
        check_state(final_state, "final_state", device, num_sequences, H);
    }

    const int NT = static_cast<int>(T / 64);
    gdn_k2_c_kargs args{
        reinterpret_cast<const __hip_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(v.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(c.data_ptr()),
        reinterpret_cast<const float*>(beta.data_ptr()),
        reinterpret_cast<const float*>(g.data_ptr()),
        has_initial_state
            ? reinterpret_cast<const float*>(initial_state.data_ptr())
            : nullptr,
        reinterpret_cast<__hip_bfloat16*>(o.data_ptr()),
        output_final_state
            ? reinterpret_cast<float*>(final_state.data_ptr())
            : nullptr,
        // The kernel derives (sequence, head) from this extent, so a packed
        // batch reports its sequence count here instead of the B=1 token axis.
        static_cast<int>(num_sequences),
        static_cast<int>(T),
        static_cast<int>(H),
        static_cast<int>(Hg),
        128,
        128,
        NT,
        scale,
        nullptr,
        nullptr,
        is_varlen ? cu_seqlens.data_ptr<int32_t>() : nullptr,
        is_varlen ? chunk_offsets.data_ptr<int32_t>() : nullptr,
    };

    const dim3 block(256);
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    TORCH_CHECK(c_mode == 1 || c_mode == 2,
                "internal C-prefill mode must be 1 (CF) or 2 (CS), got ",
                c_mode);
    const bool split_scan = c_mode == 2;
    if (split_scan) {
        auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        // Dense snapshots are indexed per batch chunk; packed snapshots are a
        // flat run over every sequence's chunks, which chunk_offsets indexes.
        auto h_snap = is_varlen
            ? torch::empty({1, total_chunks, H, 128, 128}, opts_bf16)
            : torch::empty({B, NT, H, 128, 128}, opts_bf16);
        args.ptr_h_snap = h_snap.data_ptr();
        // The split scan writes corrected values into the final output.  K6
        // reads its complete CTA tile before replacing it in place.
        args.ptr_v_new = o.data_ptr();

        // On the local 80-CU gfx942, BV16 wins through roughly 20
        // chunk-head chains; above that point BV64 avoids redundant C/K/V
        // traffic and wins the measured BV16/32/64 sweep.
        int scan_bv = nh64 <= 20 ? 16 : 64;
        if (use_env_overrides) {
            if (const char* env = std::getenv("OPUS_GDN_K2C_SCAN_BV")) {
                scan_bv = std::atoi(env);
            }
        }
        TORCH_CHECK(scan_bv == 16 || scan_bv == 32 || scan_bv == 64,
                    "OPUS_GDN_K2C_SCAN_BV must be 16, 32, or 64");
        // The packed K6 spends blockIdx.x on the flattened chunk axis, so it has
        // no V-tile axis left and needs one full-width tile.
        int out_bv = is_varlen
            ? 128
            : (static_cast<int64_t>(NT) * nh64 >= 128 ? 128 : 64);
        if (use_env_overrides) {
            if (const char* env = std::getenv("OPUS_GDN_K2C_OUT_BV")) {
                out_bv = std::atoi(env);
            }
        }
        TORCH_CHECK(out_bv == 64 || out_bv == 128,
                    "OPUS_GDN_K2C_OUT_BV must be 64 or 128");
        TORCH_CHECK(!is_varlen || out_bv == 128,
                    "the packed K6 requires OPUS_GDN_K2C_OUT_BV=128");
        int out_variant = 1;
        if (use_env_overrides) {
            if (const char* env = std::getenv("OPUS_GDN_OUT_VARIANT")) {
                out_variant = std::atoi(env);
            }
        }
        TORCH_CHECK(out_variant >= 0 && out_variant <= 2,
                    "OPUS_GDN_OUT_VARIANT must be 0 (generic), "
                    "1 (dense forward), or 2 (dense reverse)");
        // Reverse chunk scheduling is a dense-only K6 specialization.
        TORCH_CHECK(!is_varlen || out_variant != 2,
                    "the packed K6 accepts OPUS_GDN_OUT_VARIANT 0 (generic) "
                    "or 1 (aligned), not 2 (dense reverse)");
        if (out_variant != 0) {
            // T%64 is already a contract of this C-input launcher; keep the
            // value-tile condition next to the specialization dispatch.
            TORCH_CHECK(T % 64 == 0 && 128 % out_bv == 0,
                        "dense K6 requires complete BT64 and BV tiles");
        }

        #define LAUNCH_C_SCAN(TY, BVP) do { \
            const dim3 scan_grid = is_varlen \
                ? flat_y_grid(128 / BVP, nh64) \
                : dim3(128 / BVP, grid_bh); \
            gdn_k2_c_kernel<TY><<<scan_grid, block, TY::smem_size_bytes(), \
                stream>>>(args); \
        } while (0)
        #define DISPATCH_C_SCAN(BVP) do { \
            if (is_varlen) \
                LAUNCH_C_SCAN(gdn_k2_c_bt64_varlen_split_scan_traits<BVP>, BVP); \
            else \
                LAUNCH_C_SCAN(gdn_k2_c_bt64_split_scan_traits<BVP>, BVP); \
        } while (0)
        if (scan_bv == 16) DISPATCH_C_SCAN(16);
        else if (scan_bv == 32) DISPATCH_C_SCAN(32);
        else DISPATCH_C_SCAN(64);
        #undef DISPATCH_C_SCAN
        #undef LAUNCH_C_SCAN

        gdn_k2_kargs out_args{};
        out_args.ptr_q = q.data_ptr();
        out_args.ptr_k = k.data_ptr();
        out_args.ptr_g_cumsum = g.data_ptr();
        out_args.ptr_o = o.data_ptr();
        out_args.ptr_h_snap = h_snap.data_ptr();
        out_args.ptr_v_new = o.data_ptr();
        out_args.B = static_cast<int>(num_sequences);
        out_args.T = static_cast<int>(T);
        out_args.H = static_cast<int>(H);
        out_args.Hg = static_cast<int>(Hg);
        out_args.K = 128;
        out_args.V = 128;
        out_args.NT = NT;
        out_args.scale = scale;
        if (is_varlen) {
            out_args.ptr_cu_seqlens = cu_seqlens.data_ptr<int32_t>();
            out_args.ptr_chunk_indices = chunk_indices.data_ptr<int32_t>();
            out_args.ptr_chunk_offsets = chunk_offsets.data_ptr<int32_t>();
        }

        #define LAUNCH_C_OUT(BVP, DENSEP, REVERSEP) do { \
            using OT = gdn_k2_out_traits< \
                gdn_k2_traits<64, 128, 128, BVP, 8>, \
                DENSEP, REVERSEP>; \
            const dim3 out_grid(128 / BVP, NT, grid_bh); \
            gdn_k2_out_kernel<OT><<<out_grid, dim3(OT::BLOCK_SIZE), \
                OT::smem_out_bytes(), stream>>>(out_args); \
        } while (0)
        // The packed K6 owns one (global chunk, head) pair, so chunk_indices
        // replaces the dense (chunk, batch*head) grid axes entirely.
        #define LAUNCH_C_OUT_VARLEN(TY) do { \
            using OT = TY; \
            const dim3 out_grid = flat_y_grid( \
                static_cast<unsigned int>(total_chunks), H); \
            gdn_k2_out_kernel<OT><<<out_grid, dim3(OT::BLOCK_SIZE), \
                OT::smem_out_bytes(), stream>>>(out_args); \
        } while (0)
        if (is_varlen) {
            if (out_variant == 1) {
                LAUNCH_C_OUT_VARLEN(gdn_k2_c_varlen_out_aligned_traits);
            } else {
                LAUNCH_C_OUT_VARLEN(gdn_k2_c_varlen_out_generic_traits);
            }
        } else if (out_bv == 64) {
            if (out_variant == 1) LAUNCH_C_OUT(64, true, false);
            else if (out_variant == 2) LAUNCH_C_OUT(64, true, true);
            else LAUNCH_C_OUT(64, false, false);
        } else {
            if (out_variant == 1) LAUNCH_C_OUT(128, true, false);
            else if (out_variant == 2) LAUNCH_C_OUT(128, true, true);
            else LAUNCH_C_OUT(128, false, false);
        }
        #undef LAUNCH_C_OUT_VARLEN
        #undef LAUNCH_C_OUT

        const hipError_t split_status = hipGetLastError();
        TORCH_CHECK(split_status == hipSuccess,
                    "split gdn_k2_c launch failed: ",
                    hipGetErrorString(split_status));
        return;
    }

    const dim3 grid = is_varlen ? flat_y_grid(128 / 64, nh64)
                                : dim3(128 / 64, grid_bh);
    // The two-pack Phase-D candidate is the validated default.  Keep the
    // environment override so variants 0-15 remain available for rollback
    // and controlled ceiling studies.
    int variant = 16;
    if (use_env_overrides) {
        if (const char* env = std::getenv("OPUS_GDN_K2C_VARIANT")) {
            variant = std::atoi(env);
        }
    }
    if (is_varlen) {
        // Only the measured default carries the packed addressing; the rollback
        // variants stay dense so a packed request cannot silently pick one.
        TORCH_CHECK(variant == 16,
                    "packed varlen fused C-input requires "
                    "OPUS_GDN_K2C_VARIANT=16, got ", variant);
        using VT = gdn_k2_c_bt64_varlen_fused_traits;
        gdn_k2_c_kernel<VT>
            <<<grid, block, VT::smem_size_bytes(), stream>>>(args);
        const hipError_t varlen_status = hipGetLastError();
        TORCH_CHECK(varlen_status == hipSuccess,
                    "packed gdn_k2_c_kernel launch failed: ",
                    hipGetErrorString(varlen_status));
        return;
    }
    if (variant == 16) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d2_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d2_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 15) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d1_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d1_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 14) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_unroll_d_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_wave_owned_fused_unroll_d_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 13) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_wave_owned_fused_prefetch_d_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 12) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_fused_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_wave_owned_fused_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 11) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_wave_owned_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_wave_owned_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 10) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_direct_av_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_direct_av_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 9) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_relaxed_retain_k_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_relaxed_retain_k_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 8) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_relaxed_vec_c_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_relaxed_vec_c_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 7) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_relaxed_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_relaxed_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 6) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_q_prefetch_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_q_prefetch_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 5) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_gate_cache_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_gate_cache_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 4) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_prefetch_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_prefetch_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 3) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_low_lds_traits>
            <<<grid, block,
               gdn_k2_c_bt64_low_lds_traits::smem_size_bytes(), stream>>>(args);
    } else if (variant == 2) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_persist_k_gate_cache_traits>
            <<<grid, block,
               gdn_k2_c_bt64_persist_k_gate_cache_traits::smem_size_bytes(),
               stream>>>(args);
    } else if (variant == 1) {
        gdn_k2_c_kernel<gdn_k2_c_bt64_persist_k_traits>
            <<<grid, block,
               gdn_k2_c_bt64_persist_k_traits::smem_size_bytes(), stream>>>(args);
    } else {
        TORCH_CHECK(variant == 0,
                    "unsupported OPUS_GDN_K2C_VARIANT=", variant,
                    "; expected 0 (baseline), 1 (persistent K), or "
                    "2 (persistent K + gate cache), 3 (low LDS), or "
                    "4 (low LDS + gate cache + Q prefetch), "
                    "5 (low LDS + gate cache), 6 (low LDS + Q prefetch), or "
                    "7 (variant 4 + relaxed barriers), or "
                    "8 (variant 7 + vectorized C loads), or "
                    "9 (variant 7 + retained final K slab), or "
                    "10 (variant 9 + direct A-to-Vd handoff), or "
                    "11 (variant 10 + wave-owned LDS staging), or "
                    "12 (variant 11 + merged Vd/K0 publication), or "
                    "13 (variant 12 + deferred Phase-D K0 prefetch), or "
                    "14 (variant 12 + unrolled Phase-D pack loop), "
                    "15 (variant 14 + deferred K0 pack 0), or "
                    "16 (variant 14 + deferred K0 packs 0-1)");
        gdn_k2_c_kernel<gdn_k2_c_bt64_traits>
            <<<grid, block,
               gdn_k2_c_bt64_traits::smem_size_bytes(), stream>>>(args);
    }

    const hipError_t launch_status = hipGetLastError();
    TORCH_CHECK(launch_status == hipSuccess,
                "gdn_k2_c_kernel launch failed: ",
                hipGetErrorString(launch_status));
}

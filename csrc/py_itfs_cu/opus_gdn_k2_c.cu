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

// Internal dense ABI for the C-input backend.  The public launcher resolves
// auto into an explicit mode before entering this translation unit:
//   1 = CF (fused recurrence/output), 2 = CS (split scan + shared K6).
void opus_gdn_k2_c_fwd(torch::Tensor q,
                       torch::Tensor k,
                       torch::Tensor v,
                       torch::Tensor c,
                       torch::Tensor beta,
                       torch::Tensor g,
                       torch::Tensor o,
                       torch::Tensor initial_state,
                       torch::Tensor final_state,
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
                "q must have shape [B, T, H, 128]");

    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t H = q.size(2);
    const c10::Device device = q.device();

    TORCH_CHECK(B > 0 && T > 0 && H > 0, "B, T, and H must all be positive");
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
    const int64_t bh64 = B * H;
    const unsigned int grid_bh = static_cast<unsigned int>(bh64);

    check_bth(k, "k", at::kBFloat16, device, B, T, H, 128);
    check_bth(v, "v", at::kBFloat16, device, B, T, H, 128);
    check_bth(c, "c", at::kBFloat16, device, B, T, H, 64);
    check_bth_scalar(beta, "beta", device, B, T, H);
    check_bth_scalar(g, "g", device, B, T, H);
    check_bth(o, "o", at::kBFloat16, device, B, T, H, 128);

    if (has_initial_state) {
        check_state(initial_state, "initial_state", device, B, H);
    }
    if (output_final_state) {
        check_state(final_state, "final_state", device, B, H);
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
        static_cast<int>(B),
        static_cast<int>(T),
        static_cast<int>(H),
        128,
        128,
        NT,
        scale,
        nullptr,
        nullptr,
    };

    const dim3 block(256);
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    TORCH_CHECK(c_mode == 1 || c_mode == 2,
                "internal C-prefill mode must be 1 (CF) or 2 (CS), got ",
                c_mode);
    const bool split_scan = c_mode == 2;
    if (split_scan) {
        auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
        auto h_snap = torch::empty({B, NT, H, 128, 128}, opts_bf16);
        args.ptr_h_snap = h_snap.data_ptr();
        // The split scan writes corrected values into the final output.  K6
        // reads its complete CTA tile before replacing it in place.
        args.ptr_v_new = o.data_ptr();

        // On the local 80-CU gfx942, BV16 wins through roughly 20
        // chunk-head chains; above that point BV64 avoids redundant C/K/V
        // traffic and wins the measured BV16/32/64 sweep.
        int scan_bv = bh64 <= 20 ? 16 : 64;
        if (use_env_overrides) {
            if (const char* env = std::getenv("OPUS_GDN_K2C_SCAN_BV")) {
                scan_bv = std::atoi(env);
            }
        }
        TORCH_CHECK(scan_bv == 16 || scan_bv == 32 || scan_bv == 64,
                    "OPUS_GDN_K2C_SCAN_BV must be 16, 32, or 64");
        int out_bv = static_cast<int64_t>(NT) * bh64 >= 128 ? 128 : 64;
        if (use_env_overrides) {
            if (const char* env = std::getenv("OPUS_GDN_K2C_OUT_BV")) {
                out_bv = std::atoi(env);
            }
        }
        TORCH_CHECK(out_bv == 64 || out_bv == 128,
                    "OPUS_GDN_K2C_OUT_BV must be 64 or 128");
        int out_variant = 1;
        if (use_env_overrides) {
            if (const char* env = std::getenv("OPUS_GDN_OUT_VARIANT")) {
                out_variant = std::atoi(env);
            }
        }
        TORCH_CHECK(out_variant >= 0 && out_variant <= 2,
                    "OPUS_GDN_OUT_VARIANT must be 0 (generic), "
                    "1 (dense forward), or 2 (dense reverse)");
        if (out_variant != 0) {
            // T%64 is already a contract of this C-input launcher; keep the
            // value-tile condition next to the specialization dispatch.
            TORCH_CHECK(T % 64 == 0 && 128 % out_bv == 0,
                        "dense K6 requires complete BT64 and BV tiles");
        }

        #define LAUNCH_C_SCAN(BVP) do { \
            using ST = gdn_k2_c_bt64_split_scan_traits<BVP>; \
            const dim3 scan_grid(128 / BVP, grid_bh); \
            gdn_k2_c_kernel<ST><<<scan_grid, block, ST::smem_size_bytes(), stream>>>(args); \
        } while (0)
        if (scan_bv == 16) LAUNCH_C_SCAN(16);
        else if (scan_bv == 32) LAUNCH_C_SCAN(32);
        else LAUNCH_C_SCAN(64);
        #undef LAUNCH_C_SCAN

        gdn_k2_kargs out_args{};
        out_args.ptr_q = q.data_ptr();
        out_args.ptr_k = k.data_ptr();
        out_args.ptr_g_cumsum = g.data_ptr();
        out_args.ptr_o = o.data_ptr();
        out_args.ptr_h_snap = h_snap.data_ptr();
        out_args.ptr_v_new = o.data_ptr();
        out_args.B = static_cast<int>(B);
        out_args.T = static_cast<int>(T);
        out_args.H = static_cast<int>(H);
        out_args.K = 128;
        out_args.V = 128;
        out_args.NT = NT;
        out_args.scale = scale;

        #define LAUNCH_C_OUT(BVP, DENSEP, REVERSEP) do { \
            using OT = gdn_k2_out_traits< \
                gdn_k2_traits<64, 128, 128, BVP, 8>, \
                DENSEP, REVERSEP>; \
            const dim3 out_grid(128 / BVP, NT, grid_bh); \
            gdn_k2_out_kernel<OT><<<out_grid, dim3(OT::BLOCK_SIZE), \
                OT::smem_out_bytes(), stream>>>(out_args); \
        } while (0)
        if (out_bv == 64) {
            if (out_variant == 1) LAUNCH_C_OUT(64, true, false);
            else if (out_variant == 2) LAUNCH_C_OUT(64, true, true);
            else LAUNCH_C_OUT(64, false, false);
        } else {
            if (out_variant == 1) LAUNCH_C_OUT(128, true, false);
            else if (out_variant == 2) LAUNCH_C_OUT(128, true, true);
            else LAUNCH_C_OUT(128, false, false);
        }
        #undef LAUNCH_C_OUT

        const hipError_t split_status = hipGetLastError();
        TORCH_CHECK(split_status == hipSuccess,
                    "split gdn_k2_c launch failed: ",
                    hipGetErrorString(split_status));
        return;
    }

    const dim3 grid(128 / 64, grid_bh);
    // The two-pack Phase-D candidate is the validated default.  Keep the
    // environment override so variants 0-15 remain available for rollback
    // and controlled ceiling studies.
    int variant = 16;
    if (use_env_overrides) {
        if (const char* env = std::getenv("OPUS_GDN_K2C_VARIANT")) {
            variant = std::atoi(env);
        }
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

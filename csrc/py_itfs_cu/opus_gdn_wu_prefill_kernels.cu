// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based GDN prefill kernel — host launcher for aiter JIT.
// Target: gfx942 (MI300X) / gfx950 (MI350)
// Fuses all 4 steps of Gated DeltaNet chunkwise recurrence into 2 HIP kernels:
//   K1 = Steps 1+2 (cumsum, KKT, trisol, WY factors)  — token-parallel
//   K2 = Steps 3+4 (h update, output)                  — head-parallel, chunk-serial

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <torch/all.h>
#include <hip/hip_runtime.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/ref_fwd_h.hpp"   // reference FLA fwd_h (BV=16/32/64, beats triton)

// gfx950 (MI350) uses the neumann LDS C_inv path (OCC=2); gfx942 (MI300) uses
// the register-cached path (OCC=3). Host can't see __gfx950__, so detect at
// runtime (the JIT build is per-machine, so device arch == runtime arch).
static inline bool gdn_is_gfx950(int device) {
    // 0 = unknown, 1 = other architecture, 2 = gfx950.  JIT modules can be
    // built for multiple architectures and used from multiple host threads,
    // so cache per logical HIP device without a racy process-wide singleton.
    constexpr int MAX_CACHED_DEVICES = 128;
    static std::atomic<unsigned char> cached[MAX_CACHED_DEVICES]{};
    if (device >= 0 && device < MAX_CACHED_DEVICES) {
        unsigned char value = cached[device].load(std::memory_order_acquire);
        if (value == 0) {
            hipDeviceProp_t p;
            const hipError_t status = hipGetDeviceProperties(&p, device);
            TORCH_CHECK(status == hipSuccess,
                        "failed to query HIP device properties: ",
                        hipGetErrorString(status));
            const bool is_gfx950 =
                std::strstr(p.gcnArchName, "gfx950") != nullptr;
            value = is_gfx950 ? 2 : 1;
            cached[device].store(value, std::memory_order_release);
        }
        return value == 2;
    }
    hipDeviceProp_t p;
    const hipError_t status = hipGetDeviceProperties(&p, device);
    TORCH_CHECK(status == hipSuccess,
                "failed to query HIP device properties: ",
                hipGetErrorString(status));
    return std::strstr(p.gcnArchName, "gfx950") != nullptr;
}

static inline bool gdn_is_gfx942() {
    int device = 0;
    hipDeviceProp_t p;
    return hipGetDevice(&device) == hipSuccess
        && hipGetDeviceProperties(&p, device) == hipSuccess
        && std::strstr(p.gcnArchName, "gfx942") != nullptr;
}

static inline int gdn_checked_int(int64_t value, const char* name) {
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
                name, " must be positive and fit in int, got ", value);
    return static_cast<int>(value);
}

static inline unsigned int gdn_checked_grid_dim(int64_t value,
                                                const char* name) {
    // The kernels materialize blockIdx.{y,z} in signed int locals, so the
    // practical limit is INT_MAX even though dim3 itself stores uint32_t.
    TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
                name, " must fit in a signed kernel grid index, got ", value);
    return static_cast<unsigned int>(value);
}

static inline dim3 gdn_grid_with_flat_y(unsigned int x,
                                        int64_t logical_y) {
    // gfx942/gfx950 expose only a 16-bit blockIdx.y/z.  Keep the common path's
    // exact grid, and slice larger logical y dimensions across z.  65535 is a
    // conservative extent accepted by both supported targets.
    constexpr int64_t SAFE_GRID_Y = 65535;
    TORCH_CHECK(logical_y > 0 &&
                    logical_y <= std::numeric_limits<int>::max(),
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

static inline void gdn_check_grid_extent(int device,
                                         hipDeviceAttribute_t attribute,
                                         int64_t extent,
                                         const char* name) {
    // Avoid a runtime query on all production-sized launches.  Values above
    // this conservative fast-path bound are checked against the actual HIP
    // device extent (gfx942 accepts 65536, while 65537 is invalid).
    if (extent <= 65535) return;
    int limit = 0;
    const hipError_t status =
        hipDeviceGetAttribute(&limit, attribute, device);
    TORCH_CHECK(status == hipSuccess,
                "failed to query HIP grid limit for ", name, ": ",
                hipGetErrorString(status));
    TORCH_CHECK(extent <= limit,
                name, " grid extent ", extent,
                " exceeds device limit ", limit);
}

// Forward declarations — definitions live in separate TUs to avoid ODR conflicts
template<typename Traits>
__global__ void gdn_k1_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_neumann_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_bt32_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k1_bt128_kernel(gdn_k1_kargs kargs);
template<typename Traits>
__global__ void gdn_k2_kernel(gdn_k2_kargs kargs);
// Split-path K2 (grid-starved / long single-sequence): two dedicated kernels,
// independent of the fused gdn_k2_kernel — serial scan + chunk-parallel output.
template<typename Traits>
__global__ void gdn_k2_scan_kernel(gdn_k2_kargs kargs);
template<typename Traits>
__global__ void gdn_k2_out_kernel(gdn_k2_kargs kargs);
// Pure-HIP single-warp scan (raw MFMA, register-resident H, no opus template).
template<typename Traits>
__global__ void gdn_k2_scan_hip_kernel(gdn_k2_kargs kargs);
// Faithful triton fwd_h port: 32x32x16 MFMA, nw=2.
template<typename Traits>
__global__ void gdn_k2_scan32_kernel(gdn_k2_kargs kargs);
enum class K1Algo { BASIC, NEUMANN, BT32, BT128 };

template<K1Algo algo, typename K1Traits, typename K2Traits,
         bool IS_VARLEN = false>
void launch_gdn_prefill_impl(
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
    int varlen_max_chunks,
    bool has_initial_state,
    bool output_final_state,
    int k2_mode,
    bool use_env_overrides)
{
    static_assert(!IS_VARLEN || algo == K1Algo::NEUMANN,
                  "packed varlen currently requires the BT64 Neumann K1");
    using LaunchK1Traits = std::conditional_t<
        IS_VARLEN, gdn_varlen_traits<K1Traits>, K1Traits>;

    const int input_B = gdn_checked_int(q.size(0), "B");
    const int T = gdn_checked_int(q.size(1), "T");
    const int H = gdn_checked_int(q.size(2), "H");
    const int K = gdn_checked_int(q.size(3), "K");
    const int V = gdn_checked_int(v.size(3), "V");
    constexpr int BT = K1Traits::BT;
    constexpr int BV = K2Traits::BV;
    const int NT = gdn_checked_int(
        (static_cast<int64_t>(T) + BT - 1) / BT, "NT");
    const int N = IS_VARLEN
        ? gdn_checked_int(cu_seqlens.numel() - 1, "N")
        : input_B;
    const int total_chunks = IS_VARLEN
        ? gdn_checked_int(chunk_indices.size(0), "total_chunks")
        : gdn_checked_int(static_cast<int64_t>(input_B) * NT,
                          "B * NT");
    const int64_t nh64 = static_cast<int64_t>(N) * H;
    const unsigned int grid_nh = gdn_checked_grid_dim(nh64, "N * H");
    if constexpr (!IS_VARLEN) {
        const int device = q.get_device();
        gdn_check_grid_extent(
            device, hipDeviceAttributeMaxGridDimY, nh64, "dense N * H (y)");
    }
    // Tile-local HBM offsets intentionally stay 32-bit for the hot loops.
    // Validate their maximum span once on the host; only the CTA-wide batch
    // base needs 64-bit device arithmetic.
    gdn_checked_int(static_cast<int64_t>(BT) * H * K, "BT * H * K");
    gdn_checked_int(static_cast<int64_t>(BT) * H * V, "BT * H * V");
    const bool needs_64bit_k1_address =
        q.numel() > std::numeric_limits<int>::max() ||
        v.numel() > std::numeric_limits<int>::max();
    const bool needs_64bit_legacy_scan_address =
        needs_64bit_k1_address ||
        static_cast<int64_t>(N) * H * V * K >
            std::numeric_limits<int>::max();
    if constexpr (algo != K1Algo::NEUMANN) {
        TORCH_CHECK(!needs_64bit_k1_address,
                    "64-bit flattened addressing is currently supported only "
                    "by the BT64 Neumann GDN K1 path");
    }

    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

    auto w_bar    = torch::empty({input_B, T, H, K}, opts_bf16);
    auto u_bar    = torch::empty({input_B, T, H, V}, opts_bf16);
    auto g_cumsum = torch::empty({input_B, T, H},    opts_fp32);
    gdn_k1_kargs k1args{};
    k1args.ptr_k       = k.data_ptr();
    k1args.ptr_v       = v.data_ptr();
    k1args.ptr_beta    = beta.data_ptr();
    k1args.ptr_g       = g.data_ptr();
    k1args.ptr_w_bar   = w_bar.data_ptr();
    k1args.ptr_u_bar   = u_bar.data_ptr();
    k1args.ptr_g_cumsum = g_cumsum.data_ptr();
    k1args.ptr_cu_seqlens = IS_VARLEN
        ? static_cast<const int32_t*>(cu_seqlens.data_ptr()) : nullptr;
    k1args.ptr_chunk_indices = IS_VARLEN
        ? static_cast<const int32_t*>(chunk_indices.data_ptr()) : nullptr;
    k1args.g_is_bf16 = g.scalar_type() == at::ScalarType::BFloat16;
    k1args.beta_is_bf16 = beta.scalar_type() == at::ScalarType::BFloat16;
    k1args.B = N; k1args.T = T; k1args.H = H; k1args.K = K; k1args.V = V;

    gdn_k2_kargs k2args{};
    k2args.ptr_q       = q.data_ptr();
    k2args.ptr_k       = k.data_ptr();
    k2args.ptr_w_bar   = w_bar.data_ptr();
    k2args.ptr_u_bar   = u_bar.data_ptr();
    k2args.ptr_g_cumsum = g_cumsum.data_ptr();
    k2args.ptr_h0      = has_initial_state ? initial_state.data_ptr() : nullptr;
    k2args.ptr_o       = o.data_ptr();
    k2args.ptr_ht      = output_final_state ? final_state.data_ptr() : nullptr;
    k2args.ptr_h_snap  = nullptr;
    k2args.ptr_v_new   = nullptr;
    k2args.ptr_cu_seqlens = IS_VARLEN
        ? static_cast<const int32_t*>(cu_seqlens.data_ptr()) : nullptr;
    k2args.ptr_chunk_indices = IS_VARLEN
        ? static_cast<const int32_t*>(chunk_indices.data_ptr()) : nullptr;
    k2args.ptr_chunk_offsets = IS_VARLEN
        ? static_cast<const int32_t*>(chunk_offsets.data_ptr()) : nullptr;
    k2args.B = N; k2args.T = T; k2args.H = H; k2args.K = K; k2args.V = V;
    k2args.NT = NT;
    k2args.scale = scale;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(q));
    const auto current_stream = at::hip::getCurrentHIPStream();
    const hipStream_t stream = current_stream;
    if constexpr (IS_VARLEN) {
        // These tensors are private wrapper metadata and may have been
        // allocated on a different stream before a cache hit.  Associate all
        // three allocations with this consumer stream so replacing the
        // process-wide cache cannot recycle their storage while the kernels
        // below are still reading it.
        c10::cuda::CUDACachingAllocator::recordStream(
            cu_seqlens.storage().data_ptr(), current_stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            chunk_indices.storage().data_ptr(), current_stream);
        c10::cuda::CUDACachingAllocator::recordStream(
            chunk_offsets.storage().data_ptr(), current_stream);
    }

    dim3 k1_grid = IS_VARLEN
        ? gdn_grid_with_flat_y(static_cast<unsigned int>(total_chunks), H)
        : dim3(static_cast<unsigned int>(NT), grid_nh);
    dim3 k1_block(LaunchK1Traits::BLOCK_SIZE);
    size_t k1_smem = LaunchK1Traits::smem_size_bytes();

    dim3 k2_grid(ceil_div(V, BV), grid_nh);
    dim3 k2_block(K2Traits::BLOCK_SIZE);
    size_t k2_smem = K2Traits::smem_size_bytes();

    if constexpr (algo == K1Algo::BASIC)
        gdn_k1_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);
    else if constexpr (algo == K1Algo::NEUMANN) {
        const bool use_gfx950_lds = gdn_is_gfx950(q.get_device());
        if (k1args.g_is_bf16 || k1args.beta_is_bf16) {
            using ScalarTraits =
                gdn_k1_dynamic_scalar_traits<LaunchK1Traits>;
            gdn_k1_neumann_kernel<ScalarTraits><<<
                k1_grid, k1_block,
                use_gfx950_lds
                    ? ScalarTraits::smem_size_bytes_cinv_lds()
                    : ScalarTraits::smem_size_bytes(),
                stream>>>(k1args);
        } else {
            gdn_k1_neumann_kernel<LaunchK1Traits><<<
                k1_grid, k1_block,
                use_gfx950_lds
                    ? LaunchK1Traits::smem_size_bytes_cinv_lds()
                    : k1_smem,
                stream>>>(k1args);
        }
    }
    else if constexpr (algo == K1Algo::BT128) {
        // BT128 keeps the fp32 inverse and its bf16 copy live together during
        // the WY GEMMs, so it needs the full phase-2c LDS allocation.
        gdn_k1_bt128_kernel<K1Traits><<<
            k1_grid, k1_block,
            LaunchK1Traits::smem_size_bytes_cinv_lds(), stream>>>(k1args);
    }
    else
        gdn_k1_bt32_kernel<K1Traits><<<k1_grid, k1_block, k1_smem, stream>>>(k1args);

    // K2: fused (register-resident scan+output, wins high-parallelism) vs
    // split (serial scan + chunk-parallel output, wins grid-starved long seq).
    // k2_mode: 0=auto, 1=force-fused, 2=force-split. Split is BT=64 only.
    // Mode 3 is intercepted by opus_gdn_prefill_fwd before this generic W/U
    // implementation is instantiated.
    //
    // The split path ALWAYS uses its own tuned config — BV=32 (4 v-tiles) +
    // num_warps=8 — independent of the fused config: more v-tiles give more
    // workgroups (the whole point when grid-starved) and less work per warp,
    // matching what triton's autotuner picks for fwd_h. Auto-split engages when
    // the FUSED grid under-fills the device (then the split's extra parallelism
    // and chunk-parallel output win despite materializing h_snap).
    // Keep the scalar threshold as an explicit rollback for non-production
    // traits and OPUS_GDN_SPLIT_THRESHOLD overrides.  The measured gfx942
    // BT64/BV64/NW8 path below uses a T/BH/state envelope instead.
    const auto get_env_override = [use_env_overrides](const char* name) {
        return use_env_overrides ? std::getenv(name) : nullptr;
    };
    int split_thr = 129;  // fused_grid < this => split
    const char* split_env = get_env_override("OPUS_GDN_SPLIT_THRESHOLD");
    if (split_env) split_thr = atoi(split_env);
    const int64_t fused_grid = static_cast<int64_t>(ceil_div(V, BV)) * nh64;
    bool use_split;
    if constexpr (IS_VARLEN) {
        use_split = true;
    } else if (k2_mode == 2) {
        use_split = true;
    } else if (k2_mode == 1) {
        use_split = false;
    } else if constexpr (K2Traits::BT == 64 && K2Traits::K == 128 &&
                         K2Traits::V == 128 && K2Traits::BV == 64 &&
                         K2Traits::NUM_WARPS == 8) {
        if (gdn_is_gfx942() && split_env == nullptr) {
            const bool with_state = has_initial_state || output_final_state;
            bool prefer_fused;
            if (T <= 64) {
                prefer_fused = true;
            } else if (T <= 256) {
                prefer_fused = nh64 >= (with_state ? 24 : 21);
            } else if (T <= 512) {
                prefer_fused = nh64 >= 40;
            } else if (T <= 8192) {
                prefer_fused = nh64 >= 48;
            } else {
                prefer_fused = nh64 >= 64;
            }
            use_split = !prefer_fused;
        } else {
            use_split = fused_grid < split_thr;
        }
    } else {
        use_split = fused_grid < split_thr;
    }

    if constexpr (K2Traits::BT == 64) {
        if (use_split) {
            if constexpr (!IS_VARLEN) {
                const int device = q.get_device();
                gdn_check_grid_extent(
                    device, hipDeviceAttributeMaxGridDimY, NT,
                    "dense split NT (y)");
                gdn_check_grid_extent(
                    device, hipDeviceAttributeMaxGridDimZ, nh64,
                    "dense split N * H (z)");
            }
            // Dense aligned forward is the measured default.  Keep generic
            // for padded/tail input and as the explicit variant-0 rollback.
            int out_variant = (T % 64 == 0) ? 1 : 0;
            if (const char* e = get_env_override("OPUS_GDN_OUT_VARIANT")) {
                out_variant = atoi(e);
            }
            TORCH_CHECK(out_variant >= 0 && out_variant <= 2,
                        "OPUS_GDN_OUT_VARIANT must be 0 (generic), "
                        "1 (dense forward), or 2 (dense reverse)");
            auto h_snap = IS_VARLEN
                ? torch::empty({1, total_chunks, H, V, K}, opts_bf16)
                : torch::empty({input_B, NT, H, V, K}, opts_bf16);
            k2args.ptr_h_snap = h_snap.data_ptr();
            // K5 materializes corrected values directly into the final output;
            // K6 consumes each disjoint tile before overwriting it in place.
            k2args.ptr_v_new  = o.data_ptr();
            using OT = gdn_k2_traits<64, 128, 128, 32, 8>;  // out kernel: BV=32, nw=8
            dim3 out_grid(ceil_div(V, OT::BV), NT, grid_nh);
            dim3 scan_grid(ceil_div(V, OT::BV), grid_nh);
            // Scan num_warps: opus-template scan now supports nw=2 (H_E_M>1 fixed).
            // env OPUS_GDN_SPLIT_NW selects 2/4/8 (default 8). OPUS_GDN_HIP_SCAN=1
            // uses the pure-HIP single-warp scan instead (bit-exact, slow ref).
            int snw = 8;
            if (const char* e = get_env_override("OPUS_GDN_SPLIT_NW")) snw = atoi(e);
            bool hip_scan = false;
            if (const char* e = get_env_override("OPUS_GDN_HIP_SCAN")) hip_scan = atoi(e) != 0;
            bool scan32 = false;   // faithful triton port: 32x32x16 MFMA, nw=2
            if (const char* e = get_env_override("OPUS_GDN_SCAN32")) scan32 = atoi(e) != 0;
            // Reference FLA fwd_h as the scan (beats triton: ~353us; reads opus
            // token-major w/u/k directly, writes token-major v_new -> no transposes).
            // Default scan for the split path.  On the 80-CU gfx942, choose
            // 8/4/2 V tiles so the B*H grid approaches 80 resident CTAs.
            // OPUS_GDN_REF remains the explicit 0/16/32/64 override.
            int ref_bv = (nh64 <= 10) ? 16 : (nh64 <= 20 ? 32 : 64);
            if constexpr (IS_VARLEN) {
                // N*H alone overestimates useful parallelism when one long
                // sequence controls the scan critical path.  Estimate the
                // number of full-length-equivalent heads, then refine only
                // scans long enough to amortize narrower V tiles.  The
                // thresholds are the no-regression envelope from the gfx942
                // balanced/ragged/skew sweep; short scans keep the dense rule.
                const int64_t effective_nh =
                    (static_cast<int64_t>(total_chunks) * H +
                     varlen_max_chunks - 1) /
                    varlen_max_chunks;
                if (ref_bv == 32) {
                    if (varlen_max_chunks >= 64 && effective_nh <= 10)
                        ref_bv = 16;
                } else if (ref_bv == 64) {
                    if (varlen_max_chunks >= 32 && effective_nh <= 10) {
                        ref_bv = 16;
                    } else if ((varlen_max_chunks >= 32 && effective_nh <= 20) ||
                               (varlen_max_chunks >= 16 && effective_nh <= 16)) {
                        ref_bv = 32;
                    }
                }
            }
            if (const char* e = get_env_override("OPUS_GDN_REF")) ref_bv = atoi(e);  // 0=off, 16/32/64
            if constexpr (IS_VARLEN) {
                TORCH_CHECK(
                    ref_bv == 16 || ref_bv == 32 || ref_bv == 64,
                    "packed varlen requires the metadata-aware reference "
                    "state scan; OPUS_GDN_REF must be 16, 32, or 64");
            }
            if (ref_bv == 16 || ref_bv == 32 || ref_bv == 64) {
                // reference now reads opus token-major w/u/k directly (strided loads)
                // and writes v_new token-major [B,T,H,V] -> zero transposes.
                // Its dense sequence bounds and strided buffer-resource byte
                // ranges are still represented as signed int in ref_fwd_h.
                gdn_checked_int(static_cast<int64_t>(input_B) * T, "B * T");
                gdn_checked_int(
                    static_cast<int64_t>(BT) * H * K * sizeof(hip_bfloat16),
                    "BT * H * K * sizeof(bfloat16)");
                const int k_stride_t = gdn_checked_int(
                    static_cast<int64_t>(H) * K, "H * K");
                const int batch_chunks = total_chunks;
                const int64_t g_sb = IS_VARLEN ? 0 : (int64_t)T * H;
                const int64_t g_sh = 1, g_st = H;
                dim3 rg = IS_VARLEN
                    ? gdn_grid_with_flat_y(V / ref_bv, nh64)
                    : dim3(V / ref_bv, grid_nh);
                dim3 rb(256);
                const hip_bfloat16* kp = reinterpret_cast<const hip_bfloat16*>(k.data_ptr());
                const hip_bfloat16* wp = reinterpret_cast<const hip_bfloat16*>(w_bar.data_ptr());
                const hip_bfloat16* up = reinterpret_cast<const hip_bfloat16*>(u_bar.data_ptr());
                const float* gp = reinterpret_cast<const float*>(g_cumsum.data_ptr());
                const void* h0p = has_initial_state ? initial_state.data_ptr() : nullptr;
                hip_bfloat16* hsp = reinterpret_cast<hip_bfloat16*>(h_snap.data_ptr());
                hip_bfloat16* vnp = reinterpret_cast<hip_bfloat16*>(o.data_ptr());
                void* htp = output_final_state ? final_state.data_ptr() : nullptr;
                #define REF_LAUNCH(BVP, UINIT, SFIN) \
                    hipLaunchKernelGGL((chunk_gated_delta_rule_fwd_h_hip_kernel<BVP, UINIT, SFIN, true, IS_VARLEN, false, false, false, false>), \
                        rg, rb, 0, stream, kp, wp, up, gp, (const float*)nullptr, h0p, hsp, vnp, htp, \
                        IS_VARLEN ? static_cast<const int32_t*>(cu_seqlens.data_ptr()) : nullptr, \
                        IS_VARLEN ? static_cast<const int32_t*>(chunk_offsets.data_ptr()) : nullptr, \
                        N, batch_chunks, T, H, H, k_stride_t, g_sb, g_sh, g_st)
                #define REF_BV(UINIT, SFIN) do { \
                    if (ref_bv==16) REF_LAUNCH(16, UINIT, SFIN); \
                    else if (ref_bv==32) REF_LAUNCH(32, UINIT, SFIN); \
                    else REF_LAUNCH(64, UINIT, SFIN); } while(0)
                if (has_initial_state && output_final_state) REF_BV(true, true);
                else if (has_initial_state)                  REF_BV(true, false);
                else if (output_final_state)                 REF_BV(false, true);
                else                                         REF_BV(false, false);
                #undef REF_BV
                #undef REF_LAUNCH
                // v_new already written token-major into k2args.ptr_v_new.
                // out kernel is HBM-BW bound; larger V-tile (BV) cuts redundant
                // q/k re-reads + intra q@k^T recompute. Sweep via OPUS_GDN_OUT_BV.
                // BV=128 (single v-tile) is fastest: no q/k re-read or intra
                // q@k^T recompute. Fall back to 64 only when the out grid
                // (NT*B*H, since grid.x=V/BV=1 at BV=128) would starve the device.
                if constexpr (IS_VARLEN) {
                    const bool all_chunks_aligned =
                        static_cast<int64_t>(total_chunks) * BT == T;
                    bool use_aligned = all_chunks_aligned;
                    if (const char* e = get_env_override(
                            "OPUS_GDN_VARLEN_ALIGNED")) {
                        const int requested = atoi(e);
                        TORCH_CHECK(requested == 0 || requested == 1,
                                    "OPUS_GDN_VARLEN_ALIGNED must be 0 or 1");
                        use_aligned = all_chunks_aligned && requested != 0;
                    }
                    if (use_aligned) {
                        using VOUT = gdn_varlen_aligned_traits<
                            gdn_k2_out_traits<
                                gdn_k2_traits<64, 128, 128, 128, 8>,
                                true, false>>;
                        dim3 vog = gdn_grid_with_flat_y(
                            static_cast<unsigned int>(total_chunks), H);
                        gdn_k2_out_kernel<VOUT><<<
                            vog, dim3(VOUT::BLOCK_SIZE),
                            VOUT::smem_out_bytes(), stream>>>(k2args);
                        return;
                    }
                    using VOUT = gdn_varlen_traits<gdn_k2_out_traits<
                        gdn_k2_traits<64, 128, 128, 128, 8>>>;
                    dim3 vog = gdn_grid_with_flat_y(
                        static_cast<unsigned int>(total_chunks), H);
                    gdn_k2_out_kernel<VOUT><<<
                        vog, dim3(VOUT::BLOCK_SIZE), VOUT::smem_out_bytes(),
                        stream>>>(k2args);
                    return;
                }

                int out_bv = (static_cast<int64_t>(NT) * nh64 >= 128)
                    ? 128 : 64;
                int out_nw = 8;
                if (const char* e = get_env_override("OPUS_GDN_OUT_BV")) out_bv = atoi(e);
                if (const char* e = get_env_override("OPUS_GDN_OUT_NW")) out_nw = atoi(e);
                TORCH_CHECK(out_bv == 32 || out_bv == 64 || out_bv == 128,
                            "OPUS_GDN_OUT_BV must be 32, 64, or 128");
                TORCH_CHECK(out_nw == 4 || out_nw == 8,
                            "OPUS_GDN_OUT_NW must be 4 or 8");
                if (out_variant != 0 && out_nw == 8 && out_bv >= 64) {
                    TORCH_CHECK(T % 64 == 0 && V % out_bv == 0,
                                "dense K6 requires complete BT64 and BV tiles");
                }
                #define LAUNCH_OUT(OBV, ONW, DENSEP, REVERSEP) do { \
                    using OUTT = gdn_k2_out_traits< \
                        gdn_k2_traits<64,128,128,OBV,ONW>, \
                        DENSEP, REVERSEP>; \
                    dim3 og(ceil_div(V, OUTT::BV), NT, grid_nh); \
                    gdn_k2_out_kernel<OUTT><<<og, dim3(OUTT::BLOCK_SIZE), \
                        OUTT::smem_out_bytes(), stream>>>(k2args); } while(0)
                // NW4 and BV32 intentionally stay on the generic rollback.
                if (out_nw == 4) {
                    if (out_bv == 128) LAUNCH_OUT(128,4,false,false);
                    else if (out_bv == 64) LAUNCH_OUT(64,4,false,false);
                    else LAUNCH_OUT(32,4,false,false);
                } else if (out_bv == 32) {
                    LAUNCH_OUT(32,8,false,false);
                } else if (out_bv == 64) {
                    if (out_variant == 1) LAUNCH_OUT(64,8,true,false);
                    else if (out_variant == 2) LAUNCH_OUT(64,8,true,true);
                    else LAUNCH_OUT(64,8,false,false);
                } else {
                    if (out_variant == 1) LAUNCH_OUT(128,8,true,false);
                    else if (out_variant == 2) LAUNCH_OUT(128,8,true,true);
                    else LAUNCH_OUT(128,8,false,false);
                }
                #undef LAUNCH_OUT
                return;
            }
            TORCH_CHECK(!needs_64bit_legacy_scan_address,
                        "large flattened tensors require the default reference "
                        "W/U split scan; OPUS_GDN_REF=0 alternatives are not "
                        "yet 64-bit-address safe");
            #define DO_SCAN(NW) gdn_k2_scan_kernel<gdn_k2_traits<64,128,128,32,NW>> \
                <<<scan_grid, dim3((NW)*64), gdn_k2_traits<64,128,128,32,NW>::smem_scan_bytes(), stream>>>(k2args)
            if (scan32) {
                TORCH_CHECK(gdn_is_gfx950(q.get_device()),
                            "OPUS_GDN_SCAN32 requires gfx950");
                gdn_k2_scan32_kernel<OT><<<scan_grid, dim3(128), 0, stream>>>(k2args);
            } else if (hip_scan) {
                gdn_k2_scan_hip_kernel<OT><<<scan_grid, dim3(64), 0, stream>>>(k2args);
            } else if (snw==2) DO_SCAN(2);
            else if (snw==4) DO_SCAN(4);
            else             DO_SCAN(8);
            #undef DO_SCAN
            // The legacy scan pairs with BV32 output, which has no dense
            // specialization; OPUS_GDN_OUT_VARIANT therefore rolls back here.
            using OUTT = gdn_k2_out_traits<OT>;
            gdn_k2_out_kernel<OUTT><<<out_grid, dim3(OUTT::BLOCK_SIZE),
                                      OUTT::smem_out_bytes(), stream>>>(k2args);
            return;
        }
    }
    // Runtime A/B switch for the gfx942 dense fused W/U kernel.  The public
    // wrapper above guarantees complete BT and BV tiles; split K2 has already
    // returned, so none of its scan/output specializations are affected.
    if constexpr (K2Traits::BT == 64 && K2Traits::K == 128 &&
                  K2Traits::V == 128 && K2Traits::BV == 64 &&
                  K2Traits::NUM_WARPS == 8) {
        if (gdn_is_gfx942()) {
            // BV128/NW16 fills all four wave slots per SIMD while staying at
            // scratch=0.  Keep the next-chunk W/Q prefetch only once there is
            // enough serial chunk work to amortize its longer live range.
            const int64_t wf_serial_work = static_cast<int64_t>(NT) * nh64;
            int wf_variant = (wf_serial_work >= 4096) ? 5 : 6;
            if (const char* e = get_env_override("OPUS_GDN_WF_VARIANT")) {
                wf_variant = atoi(e);
            }
            TORCH_CHECK(wf_variant >= 0 && wf_variant <= 6,
                        "OPUS_GDN_WF_VARIANT must be 0 (baseline), "
                        "1 (dense/no-aux), 2 (+K reuse), "
                        "3 (+gate cache), 4 (+early prefetch), or "
                        "5 (BV128/NW16 early), or 6 (BV128/NW16 normal)");

            using WF_V1 = gdn_k2_fused_traits<
                K2Traits, true, true, false, false, false>;
            using WF_V2 = gdn_k2_fused_traits<
                K2Traits, true, true, false, true, false>;
            using WF_V3 = gdn_k2_fused_traits<
                K2Traits, true, true, true, true, false>;
            using WF_V4 = gdn_k2_fused_traits<
                K2Traits, true, true, true, true, true>;
            using WF_V5 = gdn_k2_fused_traits<
                gdn_k2_traits<64,128,128,128,16>,
                true, true, true, true, true>;
            using WF_V6 = gdn_k2_fused_traits<
                gdn_k2_traits<64,128,128,128,16>,
                true, true, true, true, false>;

            if (wf_variant == 1) {
                gdn_k2_kernel<WF_V1><<<k2_grid, dim3(WF_V1::BLOCK_SIZE),
                    WF_V1::smem_size_bytes(), stream>>>(k2args);
                return;
            }
            if (wf_variant == 2) {
                gdn_k2_kernel<WF_V2><<<k2_grid, dim3(WF_V2::BLOCK_SIZE),
                    WF_V2::smem_size_bytes(), stream>>>(k2args);
                return;
            }
            if (wf_variant == 3) {
                gdn_k2_kernel<WF_V3><<<k2_grid, dim3(WF_V3::BLOCK_SIZE),
                    WF_V3::smem_size_bytes(), stream>>>(k2args);
                return;
            }
            if (wf_variant == 4) {
                gdn_k2_kernel<WF_V4><<<k2_grid, dim3(WF_V4::BLOCK_SIZE),
                    WF_V4::smem_size_bytes(), stream>>>(k2args);
                return;
            }
            if (wf_variant == 5) {
                dim3 wf_grid(ceil_div(V, WF_V5::BV), grid_nh);
                gdn_k2_kernel<WF_V5><<<wf_grid, dim3(WF_V5::BLOCK_SIZE),
                    WF_V5::smem_size_bytes(), stream>>>(k2args);
                return;
            }
            if (wf_variant == 6) {
                dim3 wf_grid(ceil_div(V, WF_V6::BV), grid_nh);
                gdn_k2_kernel<WF_V6><<<wf_grid, dim3(WF_V6::BLOCK_SIZE),
                    WF_V6::smem_size_bytes(), stream>>>(k2args);
                return;
            }
        }
    }
    gdn_k2_kernel<K2Traits><<<k2_grid, k2_block, k2_smem, stream>>>(k2args);
}

void opus_gdn_wu_prefill_fwd(
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
    int varlen_max_chunks,
    bool has_initial_state,
    bool output_final_state,
    int BT,
    int BV,
    int num_warps,
    int k1_algo,
    int k2_mode,
    bool use_env_overrides)
{
    const bool is_varlen = cu_seqlens.defined() && cu_seqlens.numel() != 0;
    TORCH_CHECK(BT == 16 || BT == 32 || BT == 64 || BT == 128,
                "Unsupported BT=", BT, ". Supported: 16, 32, 64, 128");
    TORCH_CHECK(BV == 64 || BV == 32,
                "Unsupported BV=", BV, ". Supported: 64, 32");
    TORCH_CHECK(num_warps == 2 || num_warps == 4 || num_warps == 8,
                "Unsupported num_warps=", num_warps,
                ". Supported: 2, 4, 8");
    TORCH_CHECK(k1_algo == 0 || k1_algo == 1,
                "Unsupported k1_algo=", k1_algo, ". Supported: 0, 1");
    TORCH_CHECK(k2_mode >= 0 && k2_mode <= 2,
                "Unsupported k2_mode=", k2_mode, ". Supported: 0, 1, 2");
    TORCH_CHECK(std::isfinite(scale), "scale must be finite");
    TORCH_CHECK(varlen_max_chunks >= 0,
                "varlen_max_chunks must be non-negative");

    auto check_hip_tensor = [&q](const torch::Tensor& tensor,
                                 const char* name) {
        TORCH_CHECK(tensor.defined(), name, " must be defined");
        TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
        TORCH_CHECK(tensor.device() == q.device(), name,
                    " must be on the same device as q");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    };

    TORCH_CHECK(q.defined(), "q must be defined");
    TORCH_CHECK(q.is_cuda(), "q must be a HIP tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, T, H, K]");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0,
                "B, T, and H must be positive");
    TORCH_CHECK(q.size(3) == 128, "K must be 128");
    for (const auto& item : {
             std::pair<const char*, torch::Tensor>("k", k),
             std::pair<const char*, torch::Tensor>("v", v),
             std::pair<const char*, torch::Tensor>("o", o)}) {
        check_hip_tensor(item.second, item.first);
        TORCH_CHECK(item.second.dim() == 4 &&
                        item.second.size(0) == q.size(0) &&
                        item.second.size(1) == q.size(1) &&
                        item.second.size(2) == q.size(2) &&
                        item.second.size(3) == 128,
                    item.first, " must match q shape [B, T, H, 128]");
        TORCH_CHECK(item.second.scalar_type() == at::ScalarType::BFloat16,
                    item.first, " must be bf16");
    }
    const bool native_scalar_loads = BT == 64 && k1_algo == 1;
    for (const auto& item : {
             std::pair<const char*, torch::Tensor>("g", g),
             std::pair<const char*, torch::Tensor>("beta", beta)}) {
        check_hip_tensor(item.second, item.first);
        TORCH_CHECK(item.second.dim() == 3 &&
                        item.second.size(0) == q.size(0) &&
                        item.second.size(1) == q.size(1) &&
                        item.second.size(2) == q.size(2),
                    item.first, " must have shape [B, T, H]");
        if (native_scalar_loads) {
            TORCH_CHECK(
                item.second.scalar_type() == at::ScalarType::Float ||
                    item.second.scalar_type() == at::ScalarType::BFloat16,
                item.first, " must be fp32 or bf16 for the BT64 Neumann path");
        } else {
            TORCH_CHECK(item.second.scalar_type() == at::ScalarType::Float,
                        item.first, " must be fp32");
        }
    }
    TORCH_CHECK(v.size(3) == 128, "V must be 128");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(is_varlen || q.size(1) % BT == 0,
                "dense T must be a multiple of BT");
    TORCH_CHECK(v.size(3) % BV == 0, "V must be a multiple of BV");
    TORCH_CHECK(k2_mode != 2 || BT == 64,
                "k2_mode=2 (WS) requires BT=64");

    if (is_varlen) {
        TORCH_CHECK(q.size(0) == 1,
                    "packed varlen expects B=1 and T=total_tokens");
        TORCH_CHECK(BT == 64 && k1_algo == 1,
                    "packed varlen requires BT=64 and k1_algo=1 (Neumann)");
        TORCH_CHECK(k2_mode != 1,
                    "packed varlen currently supports the split WS path only");
        for (const auto& item : {
                 std::pair<const char*, torch::Tensor>("cu_seqlens", cu_seqlens),
                 std::pair<const char*, torch::Tensor>("chunk_indices", chunk_indices),
                 std::pair<const char*, torch::Tensor>("chunk_offsets", chunk_offsets)}) {
            TORCH_CHECK(item.second.is_cuda(), item.first,
                        " must be a HIP tensor");
            TORCH_CHECK(item.second.device() == q.device(), item.first,
                        " must be on the same device as q");
            TORCH_CHECK(item.second.is_contiguous(), item.first,
                        " must be contiguous");
            TORCH_CHECK(item.second.scalar_type() == at::ScalarType::Int,
                        item.first, " must have dtype int32");
        }
        TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
                    "cu_seqlens must have shape [N + 1]");
        TORCH_CHECK(chunk_offsets.dim() == 1 &&
                        chunk_offsets.numel() == cu_seqlens.numel(),
                    "chunk_offsets must have shape [N + 1]");
        TORCH_CHECK(chunk_indices.dim() == 2 && chunk_indices.size(1) == 2 &&
                        chunk_indices.size(0) > 0,
                    "chunk_indices must have shape [total_chunks, 2]");
        TORCH_CHECK(varlen_max_chunks > 0 &&
                        varlen_max_chunks <= chunk_indices.size(0),
                    "varlen_max_chunks must be in [1, total_chunks]");
    } else {
        TORCH_CHECK(varlen_max_chunks == 0,
                    "dense input requires varlen_max_chunks=0");
    }

    const int64_t state_batch = is_varlen ? cu_seqlens.numel() - 1 : q.size(0);
    auto check_state = [&](const torch::Tensor& state, const char* name) {
        check_hip_tensor(state, name);
        TORCH_CHECK(state.scalar_type() == at::ScalarType::Float,
                    name, " must be fp32");
        TORCH_CHECK(state.dim() == 4 && state.size(0) == state_batch &&
                        state.size(1) == q.size(2) && state.size(2) == 128 &&
                        state.size(3) == 128,
                    name, " must have shape [N, H, 128, 128]");
    };
    if (has_initial_state) check_state(initial_state, "initial_state");
    if (output_final_state) check_state(final_state, "final_state");

    auto storage_ranges_overlap = [](const torch::Tensor& a,
                                     const torch::Tensor& b) {
        const auto a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
        const auto b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
        const auto a_end = a_begin +
            static_cast<uintptr_t>(a.numel() * a.element_size());
        const auto b_end = b_begin +
            static_cast<uintptr_t>(b.numel() * b.element_size());
        return a_begin < b_end && b_begin < a_end;
    };
    auto check_disjoint = [&](const torch::Tensor& writable,
                              const char* writable_name,
                              const torch::Tensor& read_only,
                              const char* read_only_name) {
        TORCH_CHECK(!storage_ranges_overlap(writable, read_only),
                    writable_name, " must not overlap ", read_only_name,
                    " storage");
    };
    for (const auto& item : {
             std::pair<const char*, torch::Tensor>("q", q),
             std::pair<const char*, torch::Tensor>("k", k),
             std::pair<const char*, torch::Tensor>("g", g),
             std::pair<const char*, torch::Tensor>("beta", beta)}) {
        check_disjoint(o, "o", item.second, item.first);
    }
    if (has_initial_state) {
        check_disjoint(o, "o", initial_state, "initial_state");
    }
    if (output_final_state) {
        check_disjoint(o, "o", final_state, "final_state");
    }
    if (is_varlen) {
        for (const auto& item : {
                 std::pair<const char*, torch::Tensor>("cu_seqlens", cu_seqlens),
                 std::pair<const char*, torch::Tensor>("chunk_indices", chunk_indices),
                 std::pair<const char*, torch::Tensor>("chunk_offsets", chunk_offsets)}) {
            check_disjoint(o, "o", item.second, item.first);
        }
    }
    if (storage_ranges_overlap(o, v)) {
        TORCH_CHECK(o.data_ptr() == v.data_ptr() && o.numel() == v.numel(),
                    "o may overlap v only as the exact same contiguous view");
    }
    if (output_final_state) {
        for (const auto& item : {
                 std::pair<const char*, torch::Tensor>("q", q),
                 std::pair<const char*, torch::Tensor>("k", k),
                 std::pair<const char*, torch::Tensor>("v", v),
                 std::pair<const char*, torch::Tensor>("g", g),
                 std::pair<const char*, torch::Tensor>("beta", beta),
                 std::pair<const char*, torch::Tensor>("o", o)}) {
            check_disjoint(final_state, "final_state", item.second, item.first);
        }
        if (has_initial_state) {
            check_disjoint(final_state, "final_state", initial_state,
                           "initial_state");
        }
        if (is_varlen) {
            for (const auto& item : {
                     std::pair<const char*, torch::Tensor>("cu_seqlens", cu_seqlens),
                     std::pair<const char*, torch::Tensor>("chunk_indices", chunk_indices),
                     std::pair<const char*, torch::Tensor>("chunk_offsets", chunk_offsets)}) {
                check_disjoint(final_state, "final_state", item.second,
                               item.first);
            }
        }
    }

#define DISPATCH(algo, bt, bv, nw) \
    launch_gdn_prefill_impl<K1Algo::algo, \
        gdn_k1_traits<bt, 128, 128, 4>, \
        gdn_k2_traits<bt, 128, 128, bv, nw>>( \
        q, k, v, g, beta, o, scale, \
        initial_state, final_state, cu_seqlens, chunk_indices, chunk_offsets, \
        varlen_max_chunks, \
        has_initial_state, output_final_state, \
        k2_mode, use_env_overrides)

    if (is_varlen) {
        launch_gdn_prefill_impl<
            K1Algo::NEUMANN,
            gdn_k1_traits<64, 128, 128, 4>,
            gdn_k2_traits<64, 128, 128, 64, 8>,
            true>(
                q, k, v, g, beta, o, scale, initial_state, final_state,
                cu_seqlens, chunk_indices, chunk_offsets,
                varlen_max_chunks,
                has_initial_state, output_final_state, k2_mode,
                use_env_overrides);
        return;
    }

    if (BT == 128) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(BT128, 128, 64, 4);
        } else if (BV == 32 && num_warps == 4) {
            DISPATCH(BT128, 128, 32, 4);
        } else if (BV == 64 && num_warps == 8) {
            DISPATCH(BT128, 128, 64, 8);
        } else if (BV == 32 && num_warps == 8) {
            DISPATCH(BT128, 128, 32, 8);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=128, BV=", BV,
                        " num_warps=", num_warps);
        }
    } else if (BT == 64 && k1_algo == 1) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(NEUMANN, 64, 64, 4);
        } else if (BV == 32 && num_warps == 4) {
            DISPATCH(NEUMANN, 64, 32, 4);
        } else if (BV == 64 && num_warps == 8) {
            DISPATCH(NEUMANN, 64, 64, 8);
        } else if (BV == 32 && num_warps == 8) {
            DISPATCH(NEUMANN, 64, 32, 8);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=", BT, " BV=", BV,
                        " num_warps=", num_warps);
        }
    } else if (BT == 64) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(BASIC, 64, 64, 4);
        } else if (BV == 32 && num_warps == 4) {
            DISPATCH(BASIC, 64, 32, 4);
        } else if (BV == 64 && num_warps == 8) {
            DISPATCH(BASIC, 64, 64, 8);
        } else if (BV == 32 && num_warps == 8) {
            DISPATCH(BASIC, 64, 32, 8);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=", BT, " BV=", BV,
                        " num_warps=", num_warps);
        }
    } else if (BT == 32) {
        if (BV == 64 && num_warps == 4) {
            DISPATCH(BT32, 32, 64, 4);
        } else {
            TORCH_CHECK(false, "Unsupported combination BT=32, BV=", BV,
                        " num_warps=", num_warps, ". Use BV=64 num_warps=4");
        }
    } else if (BT == 16 && num_warps == 4) {
        DISPATCH(BASIC, 16, 64, 4);
    } else {
        TORCH_CHECK(false, "Unsupported combination BT=", BT, " BV=", BV,
                    " num_warps=", num_warps);
    }
#undef DISPATCH
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ============================================================================
// gfx1250 F4GEMM ASM Support Matrix (256x256 persistent kernels)
// ----------------------------------------------------------------------------
//  OUTTYPE | A_PRESHUFFLE | B_PRESHUFFLE | INTYPE |   M    |   N    |   K
// ---------+--------------+--------------+--------+--------+--------+--------
//  BF16    |      0       |      1       | MXFP4  | %1==0  | %16==0 | %32==0
//  BF16    |      0       |      1       | NVFP4  | %1==0  | %16==0 | %32==0
//  BF16    |      1       |      1       | MXFP4  | %16==0 | %16==0 | %32==0
//  BF16    |      1       |      1       | NVFP4  | %16==0 | %16==0 | %32==0
//  FP8     |      0       |      1       | MXFP4  | %1==0  | %16==0 | %32==0
//  FP8     |      0       |      1       | NVFP4  | %1==0  | %16==0 | %32==0
//  FP8     |      1       |      1       | MXFP4  | %16==0 | %16==0 | %32==0
//  FP8     |      1       |      1       | NVFP4  | %16==0 | %16==0 | %32==0
// ----------------------------------------------------------------------------
// Notes:
//  - B_PRESHUFFLE is always 1 (B is always pre-shuffled).
//  - A_PRESHUFFLE=1 tightens the M constraint from %1==0 to %16==0.
//  - K is always a multiple of 32.
//  - The current 192x256 non-persistent (NPS) path accepts MXFP4 -> BF16,
//    A_PRESHUFFLE=0, M%768==0, N%1024==0 and K%256==0, with 4x4 clusters.
//    Automatic selection is limited to 48 clusters (768 TG);
//    larger shapes and other formats use the persistent kernels.
//  - Python GEMM entrypoints return empty outputs for M==0 without a launch.
//  - See hsa/gfx1250/f4gemm/README.md for runtime geometry and tuning controls.
// ============================================================================
//
// gfx1250 F4GEMM ASM dispatch (preload SGPR mode).
// Two entrypoints:
//   - mxfp4_gemm_asm: D = A[M,K/2] mxfp4 * B[N,K/2] mxfp4 (e8m0 scales)
//   - nvfp4_gemm_asm: D = A[M,K/2] nvfp4 * B[N,K/2] nvfp4 (e4m3 scales + GlobalScale)
// Output D is bf16 [M,N] or fp8: FP8 e4m3 [M,N] data + a per-128-block E8M0
// scale [M,N/128] (out_scale).
//
// KernelArgs uses the ROCm kernarg-preload layout (sgpr_mode==1): pointers
// first (dw 0..9, MEM-first), then 4B-tight scalars. Bytes shipped to HW,
// keyed by intype (dw18..21 layout) and whether the output is fp8:
//   MXFP4      : 80B (struct minus the 2 trailing persistent log2 dwords)
//   MXFP4 + fp8: 88B (ScaleD overlaid on the unused dw20/21 log2 slots)
//   NVFP4      : 88B (full struct incl. GlobalScaleA/B + trailing log2)
//   NVFP4 + fp8: 96B (full struct + dedicated dw22/23 ScaleD pointer)
//
// The fp8 output-scale pointer reuses the per-intype overlay trick that log2
// does: NVFP4 gets a dedicated slot (dw22/23); MXFP4 (no GlobalScale, log2 in
// dw18/19) overlays ScaleD onto its still-unused dw20/21 slots.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_f4gemm_configs.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <hip/hip_runtime.h>

constexpr int MXFP4_SCALE_BLOCK = 32;
constexpr int NVFP4_SCALE_BLOCK = 16;

constexpr int F4GEMM_N_ALIGN      = 16;
constexpr int F4GEMM_K_ALIGN      = 32;
constexpr int F4GEMM_M_ALIGN_APRE = 16;
// Bound automatic NPS selection to output sizes with a clear measured gain.
// Larger sampled outputs are near parity or regress versus PS. Explicit
// kernelName remains available; see hsa/gfx1250/f4gemm/README.md for measurements.
constexpr int F4GEMM_NPS_MAX_CLUSTERS = 48;

namespace aiter::f4gemm
{
constexpr int TILE_M        = 256;
constexpr int TILE_N        = 256;
constexpr int PERSISTENT_TG = 256;

struct LaunchGeometry
{
    int cluster_x; // N direction; HIP cluster attribute / ttmp6
    int cluster_y; // M direction
    int grid_x;    // Cluster grid, not the HIP workgroup grid
    int grid_y;
};

constexpr bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

static unsigned int log2_power_of_two(int value)
{
    unsigned int result = 0;
    for(; value > 1; value >>= 1)
        ++result;
    return result;
}

static LaunchGeometry make_launch_geometry(int cluster_x, int cluster_y, int grid_y)
{
    // 1x1 bypasses cluster launch, but these shaders need ttmp6 and barriers.
    // A single axis of 16 hung the current stack; automatic axes stay <= 4.
    if(!is_power_of_two(cluster_x) || !is_power_of_two(cluster_y) ||
       cluster_x > 8 || cluster_y > 8 || cluster_x * cluster_y < 2 ||
       cluster_x * cluster_y > 16)
        throw std::invalid_argument("f4gemm: cluster axes must be powers of two <= 8, "
                                    "with 2..16 TG per cluster");

    const int clusters = PERSISTENT_TG / (cluster_x * cluster_y);
    if(!is_power_of_two(grid_y) || grid_y > clusters)
        throw std::invalid_argument("f4gemm: grid_y must be a power-of-two divisor "
                                    "of the persistent cluster count");
    return {cluster_x, cluster_y, clusters / grid_y, grid_y};
}

static LaunchGeometry select_launch_geometry(int M, int N)
{
    if(M < 0 || N < 0)
        throw std::invalid_argument("f4gemm: M and N must be nonnegative");
    const int tiles_m = M / TILE_M + (M % TILE_M != 0);
    const int tiles_n = N / TILE_N + (N % TILE_N != 0);

    // Shrink axes that would add whole phantom tiles to the default 4x4 cluster.
    int cluster_x = 4;
    int cluster_y = 4;
    while(cluster_x > 1 && tiles_n % cluster_x != 0)
        cluster_x >>= 1;
    while(cluster_y > 1 && tiles_m % cluster_y != 0)
        cluster_y >>= 1;
    if(cluster_x * cluster_y == 1)
    {
        if(tiles_n >= tiles_m)
            cluster_x = 2;
        else
            cluster_y = 2;
    }

    const int cluster_rows = (tiles_m + cluster_y - 1) / cluster_y;
    int grid_y = 1;
    while(grid_y < 4 && grid_y < cluster_rows)
        grid_y <<= 1;
    return make_launch_geometry(cluster_x, cluster_y, grid_y);
}
} // namespace aiter::f4gemm

// Preload-mode KernelArgs (4B-tight, MEM-first). Offsets in comments are the
// kernarg byte offsets the preload-aware shader s_load's from.
struct __attribute__((packed)) KernelArgs
{
    void*        ptr_D;            // dw 0..1   (off 0x00)
    void*        ptr_A;            // dw 2..3   (off 0x08)
    void*        ptr_B;            // dw 4..5   (off 0x10)
    void*        ptr_ScaleA;       // dw 6..7   (off 0x18)
    void*        ptr_ScaleB;       // dw 8..9   (off 0x20)
    unsigned int strideD0;         // dw 10     (off 0x28)
    unsigned int strideA0;         // dw 11     (off 0x2C)
    unsigned int strideB0;         // dw 12     (off 0x30)
    unsigned int ScaleA_stride0;   // dw 13     (off 0x34)
    unsigned int ScaleB_stride0;   // dw 14     (off 0x38)
    unsigned int M;                // dw 15     (off 0x3C)
    unsigned int N;                // dw 16     (off 0x40)
    unsigned int K;                // dw 17     (off 0x44)
    float        GlobalScaleA;     // dw 18     (off 0x48) NVFP4 only
    float        GlobalScaleB;     // dw 19     (off 0x4C) NVFP4 only
    unsigned int log2_grid_x;      // dw 20     NVFP4 persistent; MXFP4+fp8: ScaleD low
    unsigned int log2_grid_y;      // dw 21     NVFP4 persistent; MXFP4+fp8: ScaleD high
    void*        ptr_ScaleD;       // dw 22..23 (off 0x58) NVFP4+fp8 output E8M0 scale
};
// 5 ptrs (40B) + 8 scalars (32B) + GlobalScaleA/B (8B) + 2 log2 (8B) + ScaleD (8B) = 96B.
// Only fp8 outputs ship ScaleD; other outputs ship 80B (MXFP4) / 88B (NVFP4).
static_assert(sizeof(KernelArgs) == 96, "f4gemm preload KernelArgs must be 96B");

// Keep NPS on the validated full 4x4 clusters and K256 steps. Both shaders
// read cluster dimensions from ttmp6; 4x4 is the current NPS dispatch policy.
template <typename Config>
static bool supports_shape(const Config& cfg, int M, int N, int K)
{
    if(M <= 0 || N <= 0 || K <= 0)
        return false;
    const int m_align = cfg.a_preshuffle ? F4GEMM_M_ALIGN_APRE : 1;
    if(M % m_align || N % F4GEMM_N_ALIGN || K % F4GEMM_K_ALIGN)
        return false;
    if(cfg.ps_mode == 1)
        return true;
    return cfg.ps_mode == 0 && cfg.tile_m == 192 && cfg.tile_n == 256 &&
           cfg.cluster_x == 4 && cfg.cluster_y == 4 && cfg.intype == "mxfp4" &&
           cfg.outtype == "bf16" && cfg.a_preshuffle == 0 && cfg.scale == 1 &&
           M % (192 * 4) == 0 && N % (256 * 4) == 0 && K % 256 == 0;
}

static std::tuple<std::string, int> get_heuristic_kernel(
    int M, int N, int K, std::string arch_id, const std::string& intype,
    const std::string& outtype, int a_preshuffle, CFG* cfgs)
{
    // Prefer NPS within the measured size bound. Select deterministically
    // regardless of unordered-map iteration order, with persistent fallback.
    std::string selectedKernelName = "";

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.intype != intype || cfg.a_preshuffle != a_preshuffle)
            continue;
        if(cfg.outtype != outtype)
            continue;

        if(!supports_shape(cfg, M, N, K))
            continue;
        if(cfg.ps_mode == 0 &&
           static_cast<int64_t>(M) * N >
               static_cast<int64_t>(F4GEMM_NPS_MAX_CLUSTERS) * cfg.tile_m * cfg.tile_n *
                   cfg.cluster_x * cfg.cluster_y)
            continue;

        if(selectedKernelName.empty() ||
           cfg.ps_mode < cfgs->at(selectedKernelName).ps_mode ||
           (cfg.ps_mode == cfgs->at(selectedKernelName).ps_mode && el.first < selectedKernelName))
            selectedKernelName = el.first;
    }

    AITER_CHECK(selectedKernelName != "",
                __func__,
                ": cannot get heuristic kernel for intype=",
                intype,
                ", outtype=",
                outtype,
                ", a_preshuffle=",
                a_preshuffle,
                ", M=",
                M,
                ", N=",
                N,
                ", K=",
                K);
    return std::make_tuple(selectedKernelName, 1);
}

static const f4gemmConfig& get_kernel_config(
    int M, int N, int K, const std::string& arch_id, const std::string& intype,
    const std::string& outtype, int a_preshuffle, const char* kernelName)
{
    static CFG* config_map = &cfg_f4gemm;
    AITER_CHECK(!config_map->empty(),
                __func__,
                " no kernel registered for f4gemm; check AITER_GPU_ARCHS=gfx1250");

    std::string selectedName =
        (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";

    // Keep identical shapes on different GPU architectures in separate entries.
    const int intype_id = (intype == "nvfp4") ? 1 : 0;   // else mxfp4
    const int out_id    = (outtype == "fp8") ? 1 : 0;  // bf16=0
    using DictKey = std::tuple<std::string, int, int, int, int, int, int>;
    struct DictHash
    {
        size_t operator()(const DictKey& k) const
        {
            const auto& [arch, m, n, kk, it, ap, ot] = k;
            size_t h = std::hash<std::string>{}(arch);
            for(int v : {m, n, kk, it, ap, ot})
                h = (h ^ static_cast<size_t>(static_cast<unsigned>(v))) * 1099511628211ull;
            return h;
        }
    };
    static SynchronizedCache<DictKey, std::string, DictHash> heuristic_kernel_dict;

    if(selectedName.empty())
    {
        selectedName = heuristic_kernel_dict.get_or_create(
            DictKey(arch_id, M, N, K, intype_id, a_preshuffle, out_id), [&]() {
                auto [name, _] = get_heuristic_kernel(
                    M, N, K, arch_id, intype, std::string(outtype),
                    a_preshuffle, config_map);
                return name;
            });
    }

    auto it = config_map->find(selectedName);
    AITER_CHECK(it != config_map->end(),
                __func__,
                " kernel not in cfg_f4gemm: ",
                selectedName);

    const auto& cfg     = it->second;
    // Guard the explicit-kernelName path. outtype MUST match: a mismatched .co
    // keeps the same kernarg size (HIP won't catch it) but sizes stride_d / the
    // output buffer for a different element width -> device-side OOB write.
    AITER_CHECK(cfg.intype == intype && cfg.a_preshuffle == a_preshuffle &&
                    cfg.outtype == outtype,
                __func__,
                " selected kernel ",
                selectedName,
                " mismatches requested intype/a_preshuffle/outtype (got intype=",
                cfg.intype,
                ", outtype=",
                cfg.outtype,
                "; requested intype=",
                intype,
                ", outtype=",
                outtype,
                ")");
    AITER_CHECK(supports_shape(cfg, M, N, K),
                __func__, " selected kernel does not support M/N/K=",
                M, "/", N, "/", K,
                cfg.ps_mode == 0
                    ? "; 192x256 NPS requires MXFP4/BF16/AP0, M%768=0, N%1024=0, K%256=0"
                    : "; persistent kernels require N%16=0, K%32=0 and M%16=0 for AP1");

    return cfg;
}

static aiter::f4gemm::LaunchGeometry get_launch_geometry(
    const f4gemmConfig& cfg, int M, int N)
{
    if(cfg.ps_mode == 0)
        return {cfg.cluster_x, cfg.cluster_y,
                N / (cfg.tile_n * cfg.cluster_x), M / (cfg.tile_m * cfg.cluster_y)};

    AITER_CHECK(cfg.tile_m == aiter::f4gemm::TILE_M && cfg.tile_n == aiter::f4gemm::TILE_N,
                "f4gemm persistent geometry requires the registered 256x256 tile");
    // Snapshot the override on the first persistent selection.
    // Later environment changes do not alter dispatch. NPS keeps its fixed 4x4
    // clusters. See hsa/gfx1250/f4gemm/README.md for the format and constraints.
    static const auto override_geometry = []() -> std::optional<aiter::f4gemm::LaunchGeometry> {
        const char* tune = std::getenv("AITER_F4GEMM_TUNE");
        if(tune == nullptr)
            return std::nullopt;
        int cluster_x, cluster_y, grid_y;
        char trailing;
        AITER_CHECK(std::sscanf(tune, "%d,%d,%d%c", &cluster_x, &cluster_y, &grid_y,
                               &trailing) == 3,
                    "AITER_F4GEMM_TUNE must be cluster_x,cluster_y,grid_y");
        return aiter::f4gemm::make_launch_geometry(cluster_x, cluster_y, grid_y);
    }();
    return override_geometry ? *override_geometry : aiter::f4gemm::select_launch_geometry(M, N);
}

// Shared dispatch body. arg_size (shipped bytes) is set per (intype, out_is_fp8)
// below; see the KernelArgs ABI table at the top of this file.
static void f4gemm_launch(aiter_tensor_t* A,
                                aiter_tensor_t* B,
                                aiter_tensor_t* ScaleA,
                                aiter_tensor_t* ScaleB,
                                aiter_tensor_t* out,
                                aiter_tensor_t* out_scale,   // fp8 output E8M0 scale (null otherwise)
                                const char*     kernelName,
                                const std::string& intype,
                                int             a_preshuffle,
                                float           GlobalScaleA,
                                float           GlobalScaleB,
                                hipStream_t     stream)
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16 || out->dtype() == AITER_DTYPE_fp8,
                __func__,
                " only supports BFloat16 or FP8 (e4m3 + E8M0 scale) output");
    const bool out_is_fp8 = (out->dtype() == AITER_DTYPE_fp8);
    // CSV outtype name (fp8|bf16); const char* to avoid a per-call alloc.
    const char* out_type = out_is_fp8 ? "fp8" : "bf16";
    AITER_CHECK(!out_is_fp8 || out_scale != nullptr,
                __func__,
                " fp8 output requires an out_scale (E8M0) tensor");
    AITER_CHECK(intype == "mxfp4" || intype == "nvfp4",
                __func__,
                " unsupported intype ",
                intype);
    AITER_CHECK(a_preshuffle == 0 || a_preshuffle == 1,
                __func__,
                " a_preshuffle must be 0 or 1");

    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1) * 2; // packed fp4: stored dim = K/2 bytes

    int scale_block = (intype == "nvfp4") ? NVFP4_SCALE_BLOCK : MXFP4_SCALE_BLOCK;
    AITER_CHECK(Kdim % scale_block == 0,
                __func__,
                " K must be divisible by scale block size (",
                scale_block,
                ")");

    // Strides in bytes.
    unsigned int stride_a = static_cast<unsigned int>(Kdim / 2);     // fp4 packed
    unsigned int stride_b = static_cast<unsigned int>(Kdim / 2);     // fp4 packed
    // Output row stride in bytes: bf16 = Ndim*2; fp8 (fp8 e4m3, 1 byte/val) =
    // Ndim. Output format is compile-time per .co; the host only needs the
    // matching stride + buffer dtype.
    unsigned int stride_d = out_is_fp8 ? static_cast<unsigned int>(Ndim)
                                       : static_cast<unsigned int>(Ndim) * 2;
    // Scale row stride: the shader loads scales in 128-K super-columns, so the
    // per-row scale count is padded up to a multiple of 128/scale_block. Equals
    // Kdim/scale_block when K%128==0.
    unsigned int stride_sa =
        static_cast<unsigned int>(((Kdim + 127) / 128) * (128 / scale_block));
    unsigned int stride_sb = stride_sa;

    KernelArgs args{};                       // zero-init; log2_grid_x/y set below if persistent
    args.ptr_D           = out->ptr;
    args.ptr_A           = A->ptr;
    args.ptr_B           = B->ptr;
    args.ptr_ScaleA      = ScaleA->ptr;
    args.ptr_ScaleB      = ScaleB->ptr;
    args.strideD0        = stride_d;
    args.strideA0        = stride_a;
    args.strideB0        = stride_b;
    args.ScaleA_stride0  = stride_sa;
    args.ScaleB_stride0  = stride_sb;
    args.M               = Mdim;
    args.N               = Ndim;
    args.K               = Kdim;
    if(intype == "nvfp4")
    {
        args.GlobalScaleA = GlobalScaleA;
        args.GlobalScaleB = GlobalScaleB;
    }

    // Bytes shipped to HW: each .co declares a matching kernarg segment size, so
    // this must be exact (HIP validates against it). Sizes: MXFP4 80B, NVFP4 /
    // MXFP4+fp8 88B, NVFP4+fp8 96B (see the ABI table at the top of this file).
    size_t arg_size;
    if(out_is_fp8)
        arg_size = (intype == "nvfp4") ? sizeof(KernelArgs)
                                               : offsetof(KernelArgs, ptr_ScaleD);
    else
        arg_size = (intype == "nvfp4")
                       ? offsetof(KernelArgs, ptr_ScaleD)
                       : (offsetof(KernelArgs, ptr_ScaleD) - 2 * sizeof(unsigned int));

    const HipDeviceGuard device_guard(A->device_id);

    const auto& cfg = get_kernel_config(
        Mdim, Ndim, Kdim, get_gpu_arch(), intype, out_type, a_preshuffle, kernelName);

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        cfg.knl_name, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    const auto geometry = get_launch_geometry(cfg, Mdim, Ndim);
    const int gdx = geometry.grid_x * geometry.cluster_x;
    const int gdy = geometry.grid_y * geometry.cluster_y;
    const int gdz = 1;
    if(cfg.ps_mode == 1)
    {
        const auto log2_grid_x = aiter::f4gemm::log2_power_of_two(geometry.grid_x);
        const auto log2_grid_y = aiter::f4gemm::log2_power_of_two(geometry.grid_y);
        // MXFP4 overlays the persistent grid on the unused GlobalScale slots.
        if(intype == "nvfp4")
        {
            args.log2_grid_x = log2_grid_x;
            args.log2_grid_y = log2_grid_y;
        }
        else
        {
            std::memcpy(&args.GlobalScaleA, &log2_grid_x, sizeof(unsigned int));
            std::memcpy(&args.GlobalScaleB, &log2_grid_y, sizeof(unsigned int));
        }
    }

    // fp8 output E8M0 scale pointer (set after the persistent block so the log2
    // writes can't clobber it): NVFP4 uses dw22/23; MXFP4 overlays it on dw20/21.
    if(out_is_fp8)
    {
        if(intype == "nvfp4")
            args.ptr_ScaleD = out_scale->ptr;
        else
            std::memcpy(&args.log2_grid_x, &out_scale->ptr, sizeof(void*));
    }

    const int bdx = 128; // 4 wave * 32 thread on gfx1250

    impl_ptr->launch_kernel(
        {&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, stream,
         geometry.cluster_x, geometry.cluster_y, 1});
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp4_gemm_asm,
    (aiter_tensor_t* A,         // A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,         // B:[N, K/2] fp4x2 (always preshuffled)
     aiter_tensor_t* ScaleA,    // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB,    // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,       // Out: bf16 [M,N] / fp8 [M,N]
     aiter_tensor_t* out_scale, // fp8 only: E8M0 [M, N/128] (null otherwise)
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, out, out_scale, kernelName, a_preshuffle, stream))
{
    f4gemm_launch(A, B, ScaleA, ScaleB, out, out_scale,
                        kernelName, "mxfp4", a_preshuffle,
                        0.0f, 0.0f, stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    nvfp4_gemm_asm,
    (aiter_tensor_t* A,         // A:[M, K/2] fp4x2 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,         // B:[N, K/2] fp4x2 (always preshuffled)
     aiter_tensor_t* ScaleA,    // ScaleA:[M, K/32] e4m3 (shuffled)
     aiter_tensor_t* ScaleB,    // ScaleB:[N, K/32] e4m3 (shuffled)
     float           GlobalScaleA,
     float           GlobalScaleB,
     aiter_tensor_t* out,       // Out: bf16 [M,N] / fp8 [M,N]
     aiter_tensor_t* out_scale, // fp8 only: E8M0 [M, N/128] (null otherwise)
     const char*     kernelName,
     int             a_preshuffle,
     hipStream_t     stream),
    (A, B, ScaleA, ScaleB, GlobalScaleA, GlobalScaleB,
     out, out_scale, kernelName, a_preshuffle, stream))
{
    f4gemm_launch(A, B, ScaleA, ScaleB, out, out_scale,
                        kernelName, "nvfp4", a_preshuffle,
                        GlobalScaleA, GlobalScaleB, stream);
}

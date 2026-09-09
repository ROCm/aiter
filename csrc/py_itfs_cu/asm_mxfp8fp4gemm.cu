// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// ============================================================================
// gfx1250 F8GEMM ASM Support Matrix
// ----------------------------------------------------------------------------
//  OUTTYPE | A_PRESHUFFLE | B_INTYPE |   M    |   N    |   K
// ---------+--------------+----------+--------+--------+--------
//  BF16    |      0       |  MXFP8   | %1==0  | %16==0 | %128==0
//  BF16    |      0       |  MXFP4   | %1==0  | %16==0 | %128==0
//  BF16    |      1       |  MXFP8   | %2==0  | %16==0 | %128==0
//  BF16    |      1       |  MXFP4   | %2==0  | %16==0 | %128==0
// ----------------------------------------------------------------------------
// Notes:
//  - Currently only support BF16 output.
//  - B_PRESHUFFLE is always 1 (B is always pre-shuffled).
//  - A_PRESHUFFLE=1 tightens the M constraint from %1==0 to %2==0.
//  - K is always a multiple of 128.
// ============================================================================
//
// gfx1250 MXFP8 x {MXFP8, MXFP4} GEMM ASM dispatch (preload SGPR mode).
// A (activation) is always MXFP8 (e4m3, 1 byte/elem); B (weight) is either
// MXFP8 (a8w8) or MXFP4 (a8w4, e2m1, 2 elems/byte). Both operands carry OCP
// micro-scaling block scales (e8m0, one per 32 K-elements).
//
// Two entrypoints:
//   - mxfp8_mxfp8_gemm_asm: D[M,N] bf16 = A[M,K] mxfp8 * B[N,K] mxfp8   (a8w8)
//   - mxfp8_mxfp4_gemm_asm: D[M,N] bf16 = A[M,K] mxfp8 * B[N,K/2] mxfp4 (a8w4)
//
// KernelArgs is the packed preload layout the POC silicon host ships (80B):
// 5 pointers (MEM-first), then 10 tight 4B scalars. The persistent + cluster
// shaders do their own tile scheduling, so unlike f4gemm there are no
// log2_grid kernargs -- the host only supplies M/N/K/batch/splitk and launches
// on a fixed cluster grid.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_mxfp8fp4gemm_configs.hpp"
#include <cmath>
#include <cstring>
#include <memory>
#include <hip/hip_runtime.h>

constexpr int MX_SCALE_BLOCK = 32;

constexpr int F8GEMM_N_ALIGN      = 16;
constexpr int F8GEMM_K_ALIGN      = 128;
constexpr int F8GEMM_M_ALIGN_APRE = 2;
// Every f8gemm .co is a persistent shader launching exactly this many threadgroups.
constexpr int F8GEMM_WG_MAX = 256;
// Split-K: PF_TILES(4) * TILE_K(128). A shallower split leaves the persistent
// prologue's prefetch past the round's K range and the next round reads stale LDS.
constexpr int F8GEMM_SPLITK_MIN_K = 512;

// Preload-mode KernelArgs (4B-tight, MEM-first). Offsets in comments are the
// kernarg byte offsets the preload-aware shader s_load's from.
struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;             // s[2:3]   off 0x00
    void* ptr_A;             // s[4:5]   off 0x08
    void* ptr_B;             // s[6:7]   off 0x10
    void* ptr_ScaleA;        // s[8:9]   off 0x18
    void* ptr_ScaleB;        // s[10:11] off 0x20
    unsigned int stride_C;   // s12      off 0x28  (bytes)
    unsigned int stride_A;   // s13      off 0x2c  (bytes)
    unsigned int stride_B;   // s14      off 0x30  (bytes)
    unsigned int ScaleA_K;   // s15      off 0x34  (= K/32)
    unsigned int ScaleB_K;   // s16      off 0x38  (= K/32)
    unsigned int M;          // s17      off 0x3c
    unsigned int N;          // s18      off 0x40
    unsigned int K;          // s19      off 0x44
    unsigned int batch_size; // s20      off 0x48
    unsigned int splitk;     // s21      off 0x4c
};
static_assert(sizeof(KernelArgs) == 80, "mxfp8fp4 preload KernelArgs must be 80B");

// Pick the best registered kernel variant for (M,N,K) given the B dtype and
// a_preshuffle.
static std::tuple<std::string, int> get_heuristic_kernel(int M,
                                                         int N,
                                                         int K,
                                                         std::string arch_id,
                                                         const std::string& b_intype,
                                                         const std::string& outtype,
                                                         int a_preshuffle,
                                                         CFG* cfgs)
{
    // ---- Pass 1: tile band by M (availability-aware) ----
    // A tiny M wastes a taller tile's rows: M<=16 prefers the 16x512 decode tile
    // (FP4 + a_preshuffle=0 only), M<=64 the 64x512 variant, larger M 256x256. Only
    // the tiles actually registered in the csv are eligible, so when a preferred tile
    // isn't shipped for this combo the next rank resolves it (e.g. a 256x256-only
    // deployment lands every M on 256x256).
    const int(*tile_prefs)[2];
    int n_tile_prefs;
    static const int tp_m16[][2] = {{16, 512}, {64, 512}, {256, 256}};
    static const int tp_m64[][2] = {{64, 512}, {256, 256}};
    static const int tp_big[][2] = {{256, 256}, {64, 512}};
    if(M <= 16)
    {
        tile_prefs   = tp_m16;
        n_tile_prefs = 3;
    }
    else if(M <= 64)
    {
        tile_prefs   = tp_m64;
        n_tile_prefs = 2;
    }
    else
    {
        tile_prefs   = tp_big;
        n_tile_prefs = 2;
    }

    const int  m_align  = a_preshuffle ? F8GEMM_M_ALIGN_APRE : 1;
    const bool align_ok = (M % m_align) == 0 && (N % F8GEMM_N_ALIGN) == 0 &&
                          (K % F8GEMM_K_ALIGN) == 0;

    int         bestTileRank = n_tile_prefs; // == "no preferred tile registered"
    int         selTileM = 0, selTileN = 0;
    std::string fallbackKernelName = ""; // any valid variant if no preferred tile is present
    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.b_intype != b_intype || cfg.a_preshuffle != a_preshuffle)
            continue;
        if(cfg.outtype != outtype)
            continue;
        if(!align_ok)
            continue;

        // Remember the first valid variant so an odd combo that ships one tile resolves.
        if(fallbackKernelName.empty())
            fallbackKernelName = el.first;

        for(int r = 0; r < bestTileRank; ++r)
        {
            if(cfg.tile_m == tile_prefs[r][0] && cfg.tile_n == tile_prefs[r][1])
            {
                bestTileRank = r;
                selTileM     = tile_prefs[r][0];
                selTileN     = tile_prefs[r][1];
                break;
            }
        }
    }

    // ---- Pass 2: cluster that best fits the selected tile's grid ----
    // cluster_x groups TILE_N columns (N), cluster_y groups TILE_M rows (M). The
    // largest cluster that DIVIDES the tile grid evenly maximizes data reuse; ties
    // break toward the aspect closest to the tile grid, then larger cx. 1x1 always
    // fits (n % 1 == 0), so a valid pick always exists.
    //
    // Divisibility, not just cx<=ntiles: a ragged last block leaves its trailing
    // lanes with an out-of-range tile id. The kernel clamps them onto the last valid
    // tile (calc_wg_coord) so they stay lock-step for the multicast TDM loads, but
    // that work is redundant -- up to (cx*cy-1)/(cx*cy) of the block is wasted.
    // Measured on N=1280 (5 N-tiles, 5 % 4 != 0), a8w8 K=8192: cluster 4x4 runs
    // 31.9us at M=4096 and 44.7us at M=8192 against 25.3us / 35.7us for 1x1, i.e.
    // the "fitting" cluster was ~20% SLOWER than no cluster at all.
    std::string selectedKernelName = "";
    if(bestTileRank < n_tile_prefs)
    {
        const int mtiles = (M + selTileM - 1) / selTileM;
        const int ntiles = (N + selTileN - 1) / selTileN;
        int       bestScore     = -1;   // cx*cy for a fitting cluster, else 0
        double    bestAspectErr = 1e30; // |cx*mtiles - cy*ntiles|, smaller = better
        int       bestCx        = -1;
        for(const auto& el : *cfgs)
        {
            if(el.first.find(arch_id) != 0)
                continue;
            const auto& cfg = el.second;
            if(cfg.b_intype != b_intype || cfg.a_preshuffle != a_preshuffle)
                continue;
            if(cfg.outtype != outtype)
                continue;
            if(cfg.tile_m != selTileM || cfg.tile_n != selTileN)
                continue;

            const int    cx        = cfg.cluster_x > 0 ? cfg.cluster_x : 1;
            const int    cy        = cfg.cluster_y > 0 ? cfg.cluster_y : 1;
            const bool   fits      = (ntiles % cx == 0) && (mtiles % cy == 0);
            const int    score     = fits ? cx * cy : 0;
            const double aspectErr = std::fabs((double)cx * mtiles - (double)cy * ntiles);

            bool better = score > bestScore;
            if(!better && score == bestScore)
                better = (aspectErr < bestAspectErr) ||
                         (aspectErr == bestAspectErr && cx > bestCx);
            if(better)
            {
                bestScore          = score;
                bestAspectErr      = aspectErr;
                bestCx             = cx;
                selectedKernelName = el.first;
            }
        }
    }

    if(selectedKernelName.empty())
        selectedKernelName = fallbackKernelName;

    AITER_CHECK(selectedKernelName != "",
                __func__,
                ": cannot get heuristic kernel for b_intype=",
                b_intype,
                ", a_preshuffle=",
                a_preshuffle,
                ", M=",
                M,
                ", N=",
                N,
                ", K=",
                K,
                " (require N%16==0, K%128==0, and M%2==0 when a_preshuffle=1)");
    return std::make_tuple(selectedKernelName, 1);
}

// Resolve the variant to run: an explicit mangled kernelName, else the cached
// heuristic. Shared by the launch and the splitk query so both see one decision.
static const mxfp8fp4gemmConfig& resolve_kernel(int M,
                                                int N,
                                                int K,
                                                const std::string& b_intype,
                                                const std::string& out_type,
                                                int a_preshuffle,
                                                const char* kernelName)
{
    static CFG* config_map = &cfg_mxfp8fp4gemm;
    AITER_CHECK(!config_map->empty(),
                __func__,
                " no kernel registered for mxfp8fp4gemm; check AITER_GPU_ARCHS=gfx1250");

    std::string arch_id      = get_gpu_arch();
    std::string selectedName = (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";

    const int intype_id = (b_intype == "mxfp4") ? 1 : 0;       // else mxfp8
    using DictKey       = std::tuple<int, int, int, int, int>; // M,N,K,intype_id,apre
    struct DictHash
    {
        size_t operator()(const DictKey& k) const
        {
            const auto& [m, n, kk, it, ap] = k;
            size_t h                       = 1469598103934665603ull;
            for(int v : {m, n, kk, it, ap})
                h = (h ^ static_cast<size_t>(static_cast<unsigned>(v))) * 1099511628211ull;
            return h;
        }
    };
    static SynchronizedCache<DictKey, std::string, DictHash> heuristic_kernel_dict;

    if(selectedName.empty())
    {
        selectedName =
            heuristic_kernel_dict.get_or_create(DictKey(M, N, K, intype_id, a_preshuffle), [&]() {
                auto [name, _] = get_heuristic_kernel(
                    M, N, K, arch_id, b_intype, out_type, a_preshuffle, config_map);
                return name;
            });
    }

    auto it = config_map->find(selectedName);
    AITER_CHECK(
        it != config_map->end(), __func__, " kernel not in cfg_mxfp8fp4gemm: ", selectedName);

    const auto& cfg = it->second;
    // Guard the explicit-kernelName path. outtype MUST match: a mismatched .co
    // keeps the same kernarg size (HIP won't catch it) but sizes stride_d / the
    // output buffer for a different element width -> device-side OOB write.
    AITER_CHECK(cfg.b_intype == b_intype && cfg.a_preshuffle == a_preshuffle &&
                    cfg.outtype == out_type,
                __func__,
                " selected kernel ",
                selectedName,
                " mismatches requested b_intype/a_preshuffle/outtype (got outtype=",
                cfg.outtype,
                ", requested ",
                out_type,
                ")");
    return cfg;
}

// Split-K count for this shape, 1 = unsplit. The kernel gives split s the K range
// [s*K/splitk, (s+1)*K/splitk) and writes D as (splitk, M, N) WITHOUT reducing it --
// summing the planes is the caller's job (aiter/ops/mxfp8fp4gemm_common.py).
//
// It checks none of the constraints below and a violation does not fault, it reads
// garbage, so every one of them is enforced here. See
// poc_kl/mi400/mxfp8fp4gemm/README.md "Split-K"; keep the two in sync.
static bool splitk_is_valid(int M, int N, int K, const mxfp8fp4gemmConfig& cfg, int s)
{
    if(s == 1)
        return true;
    // cfg.splitk marks the variants whose .co actually declares _s_splitk (the
    // 256x256 master). The others stop their kernarg preload short, so s21 would
    // hold whatever was left in the SGPR.
    if(cfg.splitk == 0)
        return false;
    if(s <= 0 || (s & (s - 1)) != 0) // pow2: the SP3 takes log2 with s_ctz_i32_b32
        return false;
    if(K % s != 0 || (K / s) % F8GEMM_K_ALIGN != 0)
        return false; // each split must be a whole number of TILE_K steps
    if(K / s < F8GEMM_SPLITK_MIN_K)
        return false; // shallower than the persistent prologue's prefetch -> stale LDS
    // One TG per tile per split; past WG_MAX the work needs a second persistent round.
    const int tiles = ((M + cfg.tile_m - 1) / cfg.tile_m) * ((N + cfg.tile_n - 1) / cfg.tile_n);
    return (long long)s * tiles <= F8GEMM_WG_MAX;
}

// Split count to use when the caller did not name one: the deepest split that is
// still valid and still pays for its reduce.
static int choose_splitk(int M, int N, int K, const mxfp8fp4gemmConfig& cfg)
{
    int best = 1;
    for(int s = 2; splitk_is_valid(M, N, K, cfg, s); s *= 2)
    {
        // The reduce moves (s+1)*M*N*2 B in a separate pass. Filling the idle TGs only
        // buys back a fraction of the GEMM, so keep that pass under a quarter of the
        // GEMM's own operand traffic -- past that the split loses end to end (measured
        // on 128x1280x8192, 2048x1280x8192 and 128x8192x1024).
        if((long long)(s + 1) * M * N * 8 > (long long)(M + N) * K)
            break;
        best = s;
    }
    return best;
}

// Shared dispatch body for both a8w8 (B=mxfp8) and a8w4 (B=mxfp4).
static void mxfp8fp4_launch(aiter_tensor_t* A,
                            aiter_tensor_t* B,
                            aiter_tensor_t* ScaleA,
                            aiter_tensor_t* ScaleB,
                            aiter_tensor_t* out,
                            const char* kernelName,
                            const std::string& b_intype,
                            int a_preshuffle,
                            int splitk,
                            hipStream_t stream)
{
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, __func__, " only supports BFloat16 output");
    const char* out_type = "bf16";
    AITER_CHECK(
        b_intype == "mxfp8" || b_intype == "mxfp4", __func__, " unsupported b_intype ", b_intype);
    AITER_CHECK(a_preshuffle == 0 || a_preshuffle == 1, __func__, " a_preshuffle must be 0 or 1");

    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1); // A is mxfp8: 1 byte/elem, so col count == K

    AITER_CHECK(Kdim % F8GEMM_K_ALIGN == 0,
                __func__,
                " K must be divisible by ",
                F8GEMM_K_ALIGN,
                " (got K=",
                Kdim,
                ")");

    // Strides in bytes. A is fp8 (1 byte); B fp8 (1 byte) or fp4 (0.5 byte);
    // D is bf16 (2 bytes). Scales are e8m0, one per 32-K block.
    unsigned int stride_a = static_cast<unsigned int>(Kdim);
    unsigned int stride_b = (b_intype == "mxfp4") ? static_cast<unsigned int>(Kdim / 2)
                                                  : static_cast<unsigned int>(Kdim);
    unsigned int stride_d = static_cast<unsigned int>(Ndim) * 2;
    unsigned int scale_k  = static_cast<unsigned int>(Kdim / MX_SCALE_BLOCK);

    KernelArgs args{};
    args.ptr_D      = out->ptr;
    args.ptr_A      = A->ptr;
    args.ptr_B      = B->ptr;
    args.ptr_ScaleA = ScaleA->ptr;
    args.ptr_ScaleB = ScaleB->ptr;
    args.stride_C   = stride_d;
    args.stride_A   = stride_a;
    args.stride_B   = stride_b;
    args.ScaleA_K   = scale_k;
    args.ScaleB_K   = scale_k;
    args.M          = Mdim;
    args.N          = Ndim;
    args.K          = Kdim;
    args.batch_size = 1;
    args.splitk     = splitk;
    size_t arg_size = sizeof(KernelArgs);

    const HipDeviceGuard device_guard(A->device_id);

    const auto& cfg = resolve_kernel(Mdim, Ndim, Kdim, b_intype, out_type, a_preshuffle, kernelName);

    // splitk<=0 means "let the heuristic decide"; an explicit count is only checked
    // against the kernel's hard constraints (it validates none of them itself), so a
    // caller may deliberately go deeper than choose_splitk would.
    if(splitk <= 0)
        splitk = choose_splitk(Mdim, Ndim, Kdim, cfg);
    AITER_CHECK(splitk_is_valid(Mdim, Ndim, Kdim, cfg, splitk),
                __func__,
                " splitk=",
                splitk,
                " is not valid for ",
                cfg.knl_name,
                " at M=",
                Mdim,
                ", N=",
                Ndim,
                ", K=",
                Kdim);
    // D is (splitk, M, N) and the kernel does not reduce it -- the caller must have
    // sized the buffer for every plane.
    AITER_CHECK(out->numel() == (long long)splitk * Mdim * Ndim,
                __func__,
                " out must hold splitk*M*N elements (splitk=",
                splitk,
                ", got numel=",
                out->numel(),
                ")");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr = &impl_ptr_map.get_or_create(
        cfg.knl_name, [&]() { return AiterAsmKernel(cfg.knl_name.c_str(), cfg.co_name.c_str()); });

    // ----- Launch geometry: cluster + persistent -----
    // Every f8gemm .co is a persistent shader launching exactly WG_MAX threadgroups
    // regardless of M/N/K. The tile-walk swizzle (GRID_X/GRID_Y) is baked into the .co
    // and re-derived from a flat workgroup id, so the launch is NOT free to reshape the
    // grid: it must reproduce the geometry the shader was assembled and validated for.
    //
    // That geometry is the reference host's persistent branch,
    // scripts/mi400/mxfp8fp4gemm/mxfp8fp4gemm.cpp:487-503:
    //   clusters = WG_MAX / (cluster_x*cluster_y)   (gridX; gridY=1)
    //   blocks_x = cluster_x * clusters             (== gridX*CLUSTER_X)
    //   blocks_y = cluster_y * 1                     (== gridY*CLUSTER_Y)
    //   blockDim = 32 * WAVES(=4) = 128 threads, 1 TG
    // clusterDim=(cluster_x,cluster_y) then evenly divides (blocks_x,blocks_y) and the
    // total is blocks_x*blocks_y == WG_MAX. cluster_x/cluster_y are compile-time per .co.
    const int cluster_x = cfg.cluster_x > 0 ? cfg.cluster_x : 1;
    const int cluster_y = cfg.cluster_y > 0 ? cfg.cluster_y : 1;

    const int cluster_size = cluster_x * cluster_y;
    AITER_CHECK((F8GEMM_WG_MAX % cluster_size) == 0,
                __func__,
                " persistent WG_MAX=",
                F8GEMM_WG_MAX,
                " not divisible by cluster_x*cluster_y=",
                cluster_size);

    const int clusters = F8GEMM_WG_MAX / cluster_size; // reference gridX (gridY is 1)
    const int gdx      = clusters * cluster_x;  // blocks along X
    const int gdy      = cluster_y;             // blocks along Y (gridY==1)
    const int gdz      = 1;

    const int bdx = 128; // 4 waves * 32 threads on gfx1250

    impl_ptr->launch_kernel(
        {&args, &arg_size, gdx, gdy, gdz, bdx, 1, 1, stream, cluster_x, cluster_y, 1});
}

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_mxfp8_gemm_asm,
    (aiter_tensor_t * A,     // A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,      // B:[N, K]   mxfp8 e4m3 (always preshuffled)
     aiter_tensor_t* ScaleA, // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB, // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,    // Out:[M, N] bf16
     const char* kernelName,
     int a_preshuffle,
     int splitk, // <=0: pick with choose_splitk; out must then hold splitk*M*N
     hipStream_t stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, splitk, stream))
{
    mxfp8fp4_launch(
        A, B, ScaleA, ScaleB, out, kernelName, "mxfp8", a_preshuffle, splitk, stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    mxfp8_mxfp4_gemm_asm,
    (aiter_tensor_t * A,     // A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
     aiter_tensor_t* B,      // B:[N, K/2] mxfp4 e2m1 (always preshuffled)
     aiter_tensor_t* ScaleA, // ScaleA:[M, K/32] e8m0 (shuffled)
     aiter_tensor_t* ScaleB, // ScaleB:[N, K/32] e8m0 (shuffled)
     aiter_tensor_t* out,    // Out:[M, N] bf16
     const char* kernelName,
     int a_preshuffle,
     int splitk, // <=0: pick with choose_splitk; out must then hold splitk*M*N
     hipStream_t stream),
    (A, B, ScaleA, ScaleB, out, kernelName, a_preshuffle, splitk, stream))
{
    mxfp8fp4_launch(
        A, B, ScaleA, ScaleB, out, kernelName, "mxfp4", a_preshuffle, splitk, stream);
}

// Split-K count the dispatch will use for this shape. The caller needs it up front:
// D becomes (splitk, M, N) and only the caller can size that buffer and reduce it.
AITER_CTYPES_DEFINE_ENTRYPOINT(
    mxfp8fp4_gemm_splitk,
    (int M, int N, int K, int b_is_fp4, int a_preshuffle, const char* kernelName,
     hipStream_t stream),
    (M, N, K, b_is_fp4, a_preshuffle, kernelName, stream))
{
    return choose_splitk(
        M,
        N,
        K,
        resolve_kernel(M, N, K, b_is_fp4 ? "mxfp4" : "mxfp8", "bf16", a_preshuffle, kernelName));
}

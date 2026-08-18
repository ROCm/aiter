// Minimal CK warp-decode benchmark for SILOTIGER-667.
//
// Covers the kernel variants relevant to the FlyDSL comparison:
//   gate_up_bf16     BF16 act  x FP8 weight, default (pkf32)
//   gate_bf16_d2     BF16 act  x FP8 weight, dot2  <- primary comparison
//   gate_up_fp8      FP8  act  x FP8 weight, default
//   gate_fp8_d2      FP8  act  x FP8 weight, dot2
//   gate_up_fp4      BF16 act  x FP4 weight, packed (A4; needs the gate_up kernel
//                    packed-stride fix)
//   down_h2_d2       FP8 down weight, H2 layout, dot2  (current fp8 best)
//   down_fp4_h2      FP4 down weight, H2 layout        (best overall)
//
// Does NOT include persistent/jtile variants (those headers are not in this
// commit).
//
// Build (standalone, no CMake needed -- CK-Tile is header-only):
//   /opt/rocm/bin/amdclang++ -x hip -std=c++20 -O3 \
//     --offload-arch=gfx950 -DCK_TILE_USE_OCP_FP8 \
//     -I <ck_src>/include \
//     -o bench_ck_warp_decode ck_bench_warp_decode.cpp
//
// Run:
//   CK_WD_SHAPES=deepseek-v3,qwen3next CK_WD_BATCHES=1,2,4,8 ./bench_ck_warp_decode

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/warp_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ck_tile;

// Output format: pretty human table (default) or machine-readable CSV
// (CK_WD_FORMAT=csv) with schema "shape,H,I,K,E,B,kernel,us" for compare.py.
static bool g_csv = false;

// ---------------------------------------------------------------------------
// Kernel type aliases (mirrors bench_warp_decode.cpp, non-persistent only)
// ---------------------------------------------------------------------------

constexpr index_t kVec  = 8;   // BF16 / FP8 lane chunk
constexpr index_t kVec4 = 8;   // FP4 lane chunk
constexpr index_t kBN   = 128; // Block2D scale Block_N
constexpr index_t kBK   = 128; // Block2D scale Block_K
constexpr index_t kBXK  = 128; // X scale Block_K

using XScaleBF16 = WarpDecodeScaleLayout::PerTensor;
using XScaleFP8  = WarpDecodeScaleLayout::Block2D<1, kBXK>;
using WScaleAll  = WarpDecodeScaleLayout::Block2D<kBN, kBK>;
using WScalePT   = WarpDecodeScaleLayout::PerTensor; // for FP4 down

// gate_up BF16-act x FP8-weight
using GUProbBF16 = WarpDecodeGateUpProblem<bf16_t,
                                           fp8_t,
                                           float,
                                           bf16_t,
                                           float,
                                           float,
                                           XScaleBF16,
                                           WScaleAll,
                                           element_wise::Silu,
                                           kVec>;
using GUKernBF16 = WarpDecodeGateUpKernel<GUProbBF16, WarpDecodePolicy>;

using GUProbBF16D2 = WarpDecodeGateUpProblem<bf16_t,
                                             fp8_t,
                                             float,
                                             bf16_t,
                                             float,
                                             float,
                                             XScaleBF16,
                                             WScaleAll,
                                             element_wise::Silu,
                                             kVec,
                                             true>;
using GUKernBF16D2 = WarpDecodeGateUpKernel<GUProbBF16D2, WarpDecodePolicy>;

// gate_up FP8-act x FP8-weight
using GUProbFP8 = WarpDecodeGateUpProblem<fp8_t,
                                          fp8_t,
                                          float,
                                          bf16_t,
                                          float,
                                          float,
                                          XScaleFP8,
                                          WScaleAll,
                                          element_wise::Silu,
                                          kVec>;
using GUKernFP8 = WarpDecodeGateUpKernel<GUProbFP8, WarpDecodePolicy>;

using GUProbFP8D2 = WarpDecodeGateUpProblem<fp8_t,
                                            fp8_t,
                                            float,
                                            bf16_t,
                                            float,
                                            float,
                                            XScaleFP8,
                                            WScaleAll,
                                            element_wise::Silu,
                                            kVec,
                                            true>;
using GUKernFP8D2 = WarpDecodeGateUpKernel<GUProbFP8D2, WarpDecodePolicy>;

// gate_up BF16-act x FP4-weight (packed pk_fp4_t; PerTensor dummy scale, mirrors
// down_fp4_h2). NPerWarp=1 + non-dot2 scalar path (dot2/NPerWarp=2 reject packed).
using GUProbFP4 = WarpDecodeGateUpProblem<bf16_t,
                                          pk_fp4_t,
                                          float,
                                          bf16_t,
                                          float,
                                          float,
                                          XScaleBF16,
                                          WScalePT,
                                          element_wise::Silu,
                                          kVec4>;
using GUKernFP4 = WarpDecodeGateUpKernel<GUProbFP4, WarpDecodePolicy>;

// down FP8-weight, H2 (2 outputs/wave), dot2
using DnProbFP8H2D2 = WarpDecodeDownReduceProblem<bf16_t,
                                                  fp8_t,
                                                  float,
                                                  bf16_t,
                                                  float,
                                                  WScaleAll,
                                                  kVec,
                                                  true,
                                                  false,
                                                  1,
                                                  2>;
using DnKernFP8H2D2 = WarpDecodeDownReduceKernel<DnProbFP8H2D2, WarpDecodePolicy>;

// down FP4-weight, H2 (2 outputs/wave), dot2
using DnProbFP4H2 = WarpDecodeDownReduceProblem<bf16_t,
                                                pk_fp4_t,
                                                float,
                                                bf16_t,
                                                float,
                                                WScalePT,
                                                kVec4,
                                                true,
                                                false,
                                                1,
                                                2>;
using DnKernFP4H2 = WarpDecodeDownReduceKernel<DnProbFP4H2, WarpDecodePolicy>;

// -- D7 validation-only variants: FP4 with a real MXFP4 Block2D<1,32> weight scale
// (float scale values; the generic load_block2d_scale path handles any Block_N/
// Block_K, so no kernel change is needed).  Only instantiated for CK_WD_VALIDATE.
using WScaleMX = WarpDecodeScaleLayout::Block2D<1, 32>;

using GUProbFP4MX = WarpDecodeGateUpProblem<bf16_t,
                                            pk_fp4_t,
                                            float,
                                            bf16_t,
                                            float,
                                            float,
                                            XScaleBF16,
                                            WScaleMX,
                                            element_wise::Silu,
                                            kVec4>;
using GUKernFP4MX = WarpDecodeGateUpKernel<GUProbFP4MX, WarpDecodePolicy>;

using DnProbFP4H2MX = WarpDecodeDownReduceProblem<bf16_t,
                                                  pk_fp4_t,
                                                  float,
                                                  bf16_t,
                                                  float,
                                                  WScaleMX,
                                                  kVec4,
                                                  true,
                                                  false,
                                                  1,
                                                  2>;
using DnKernFP4H2MX = WarpDecodeDownReduceKernel<DnProbFP4H2MX, WarpDecodePolicy>;

// ---------------------------------------------------------------------------
// Shapes and timing helpers
// ---------------------------------------------------------------------------

struct Shape
{
    std::string name;
    index_t H, I, K, E;
};

static const std::vector<Shape> ALL_SHAPES = {
    {"deepseek-v3", 7168, 2048, 8, 256},
    {"minimax", 3072, 1536, 8, 256},
    {"qwen3next", 2048, 512, 10, 512},
};

// ---------------------------------------------------------------------------
// Cold-HBM timing loop with disjoint-expert router rotation.
//
// WHY NOT stream_config flush/rotate: launch_warp_decode_* -> launch_kernel()
// only ever calls timing_loop_impl(); it never reads s.flush_cache_ /
// s.rotating_count_ (those are consumed by launch_kernel_time_mask_flush_cache
// and the gemm-universal profiler, not this path). And ck_tile's "flush_cache"
// is s_icache_inv -- it evicts the *instruction* cache, not the L2/MALL data
// cache that holds the weights. So neither flag would make weight reads cold.
//
// Instead we mirror the FlyDSL cold harness: keep the full E-expert weight pool
// resident, precompute `rotate` disjoint router-id groups that tile the pool
// (group g -> experts [g*BK, g*BK+BK) mod E), and march a *continuous* launch
// counter across warmup+timed so every launch reads a different expert group.
// Any repeat is `rotate` launches apart == a full-pool sweep of HBM traffic
// between reuses, so each timed launch reads its weights cold from HBM.
template <typename Kern>
static float bench_cold(
    typename Kern::Kargs a, const int32_t* rids_base, int rotate, int bk, int cold, int iters)
{
    stream_config s{}; // default stream (stream_id_ == 0)
    int idx        = 0;
    auto do_launch = [&]() {
        a.p_router_ids = rids_base + static_cast<std::size_t>(idx % rotate) * bk;
        ++idx;
        make_kernel(Kern{}, Kern::GridSize(a), Kern::BlockSize(), 0, a)(s);
    };

    for(int i = 0; i < cold; ++i)
        do_launch();
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK_ERROR(hipEventCreate(&start));
    HIP_CHECK_ERROR(hipEventCreate(&stop));
    HIP_CHECK_ERROR(hipEventRecord(start, s.stream_id_));
    for(int i = 0; i < iters; ++i)
        do_launch();
    HIP_CHECK_ERROR(hipEventRecord(stop, s.stream_id_));
    HIP_CHECK_ERROR(hipEventSynchronize(stop));

    float ms = 0.f;
    HIP_CHECK_ERROR(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK_ERROR(hipEventDestroy(start));
    HIP_CHECK_ERROR(hipEventDestroy(stop));
    return (iters > 0) ? ms / iters : 0.f;
}

static void print_row(
    const Shape& sh, index_t B, const std::string& kernel, float ms, double flops, double bytes)
{
    if(g_csv)
    {
        // Dimension-based join key + raw time in microseconds. Derived metrics
        // (TB/s, TFLOPs) are recomputed downstream by the shared compute_metrics
        // helper (B3) applied identically to both harnesses, so only us is emitted.
        std::cout << sh.name << ',' << sh.H << ',' << sh.I << ',' << sh.K << ',' << sh.E << ',' << B
                  << ',' << kernel << ',' << std::fixed << std::setprecision(4) << (ms * 1e3)
                  << "\n";
        return;
    }
    double tflops = (ms > 0) ? flops / (ms * 1e9) : 0.0;
    double gbs    = (ms > 0) ? bytes / (ms * 1e6) : 0.0;
    std::cout << std::left << std::setw(16) << sh.name << std::right << std::setw(4) << B
              << std::setw(18) << kernel << std::setw(10) << std::fixed << std::setprecision(4)
              << ms << std::setw(10) << std::fixed << std::setprecision(2) << tflops
              << std::setw(10) << std::fixed << std::setprecision(1) << gbs << "\n";
}

template <typename T>
static constexpr double ebytes()
{
    return static_cast<double>(sizeof(T)) / static_cast<double>(numeric_traits<T>::PackedSize);
}

static double gu_flops(index_t B, index_t H, index_t I, index_t K)
{
    return static_cast<double>(B) * K * I * (4.0 * H + 5.0);
}
static double gu_bytes(index_t B, index_t H, index_t I, index_t K, double xe, double we)
{
    return static_cast<double>(B * K * I) * (H * xe + 2 * H * we + 1 * 2.0 /*inter bf16*/);
}
static double dn_flops(index_t B, index_t H, index_t I, index_t K)
{
    return static_cast<double>(B) * H * K * I * 3.0;
}
static double dn_bytes(index_t B, index_t H, index_t I, index_t K, double we)
{
    return static_cast<double>(B * H) *
           (K * I * 2.0 /*inter bf16*/ + K * I * we + 2 * K * 4.0 /*router*/ + 2.0 /*y bf16*/);
}

// ---------------------------------------------------------------------------
// Bench one shape x batch
// ---------------------------------------------------------------------------

static void bench(const Shape& sh, index_t B, int cold, int iters, int rotate_env)
{
    const index_t H = sh.H, I = sh.I, K = sh.K, E = sh.E;

    // Cold rotation: `rotate` disjoint router-id groups that tile the E-expert
    // pool (BK = B*K experts per group). Default (rotate_env <= 0) covers the
    // whole pool -> ceil(E / BK); env CK_WD_ROTATE overrides (use 1 for the
    // warm baseline). Any value >= 1 is valid; groups wrap over E when
    // rotate*BK > E, and a wrap is still a full-pool traffic apart == cold.
    const int bk     = static_cast<int>(B) * static_cast<int>(K);
    const int rotate = (rotate_env > 0) ? rotate_env : std::max(1, (E + bk - 1) / bk);

    DeviceMem x_bf16_dev(B * H * sizeof(bf16_t));
    DeviceMem x_fp8_dev(B * H * sizeof(fp8_t));
    DeviceMem x_scale_fp8_dev(B * (H / 128) * sizeof(float));
    DeviceMem router_ids_dev(static_cast<std::size_t>(rotate) * bk * sizeof(int32_t));
    DeviceMem router_wts_dev(B * K * sizeof(float));
    DeviceMem inter_dev(B * K * I * sizeof(bf16_t));
    DeviceMem y_dev(B * H * sizeof(bf16_t));

    // Gate/up weights: allocated together, freed before down weights.
    // Peak memory = 2 * E*I*H*sizeof(fp8) (gate+up simultaneously).
    // For deepseek-v3: 2 * 3.75 GB = 7.5 GB -- fine on MI350X/MI355X (192+ GB HBM).
    DeviceMem w_gate_dev(static_cast<std::size_t>(E) * I * H * sizeof(fp8_t));
    DeviceMem w_gate_scale_dev(static_cast<std::size_t>(E) * I * (H / 128) * sizeof(float));
    DeviceMem w_up_dev(static_cast<std::size_t>(E) * I * H * sizeof(fp8_t));
    DeviceMem w_up_scale_dev(static_cast<std::size_t>(E) * I * (H / 128) * sizeof(float));

    // Fill small tensors. router_ids: `rotate` disjoint groups tiling the pool
    // (rids[i] = i % E) so consecutive launches march sequentially through the
    // experts and each group reads a fresh slice of the weight pool from HBM.
    {
        std::vector<int32_t> rids(static_cast<std::size_t>(rotate) * bk);
        for(std::size_t i = 0; i < rids.size(); ++i)
            rids[i] = static_cast<int32_t>(i % static_cast<std::size_t>(E));
        router_ids_dev.ToDevice(rids.data(), rids.size() * sizeof(int32_t));
    }
    {
        std::vector<float> rwts(B * K, 1.0f / K);
        router_wts_dev.ToDevice(rwts.data(), rwts.size() * sizeof(float));
    }

    auto rids_ptr  = static_cast<const int32_t*>(router_ids_dev.GetDeviceBuffer());
    auto rwts_ptr  = static_cast<const float*>(router_wts_dev.GetDeviceBuffer());
    auto inter_ptr = static_cast<void*>(inter_dev.GetDeviceBuffer());
    auto y_ptr     = static_cast<void*>(y_dev.GetDeviceBuffer());
    auto x_sc_ptr  = static_cast<const float*>(x_scale_fp8_dev.GetDeviceBuffer());
    auto wg_sc_ptr = static_cast<const float*>(w_gate_scale_dev.GetDeviceBuffer());
    auto wu_sc_ptr = static_cast<const float*>(w_up_scale_dev.GetDeviceBuffer());

    // -- gate_up kernels (gate/up weights live here, freed before down) -------
    {
        typename GUKernBF16::Kargs a{};
        a.p_x                 = x_bf16_dev.GetDeviceBuffer();
        a.p_x_scale           = nullptr;
        a.p_w_gate            = w_gate_dev.GetDeviceBuffer();
        a.p_w_gate_scale      = wg_sc_ptr;
        a.p_w_up              = w_up_dev.GetDeviceBuffer();
        a.p_w_up_scale        = wu_sc_ptr;
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = inter_ptr;
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H;
        a.stride_w_up         = H;
        a.stride_intermediate = I;
        if(GUKernBF16::IsSupportedArgument(a))
        {
            float ms = bench_cold<GUKernBF16>(a, rids_ptr, rotate, bk, cold, iters);
            print_row(sh,
                      B,
                      "gate_up_bf16",
                      ms,
                      gu_flops(B, H, I, K),
                      gu_bytes(B, H, I, K, ebytes<bf16_t>(), ebytes<fp8_t>()));
        }
    }
    {
        typename GUKernBF16D2::Kargs a{};
        a.p_x                 = x_bf16_dev.GetDeviceBuffer();
        a.p_x_scale           = nullptr;
        a.p_w_gate            = w_gate_dev.GetDeviceBuffer();
        a.p_w_gate_scale      = wg_sc_ptr;
        a.p_w_up              = w_up_dev.GetDeviceBuffer();
        a.p_w_up_scale        = wu_sc_ptr;
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = inter_ptr;
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H;
        a.stride_w_up         = H;
        a.stride_intermediate = I;
        if(GUKernBF16D2::IsSupportedArgument(a))
        {
            float ms = bench_cold<GUKernBF16D2>(a, rids_ptr, rotate, bk, cold, iters);
            print_row(sh,
                      B,
                      "gate_bf16_d2",
                      ms,
                      gu_flops(B, H, I, K),
                      gu_bytes(B, H, I, K, ebytes<bf16_t>(), ebytes<fp8_t>()));
        }
    }
    {
        typename GUKernFP8D2::Kargs a{};
        a.p_x                 = x_fp8_dev.GetDeviceBuffer();
        a.p_x_scale           = x_sc_ptr;
        a.p_w_gate            = w_gate_dev.GetDeviceBuffer();
        a.p_w_gate_scale      = wg_sc_ptr;
        a.p_w_up              = w_up_dev.GetDeviceBuffer();
        a.p_w_up_scale        = wu_sc_ptr;
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = inter_ptr;
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H;
        a.stride_w_up         = H;
        a.stride_intermediate = I;
        if(GUKernFP8D2::IsSupportedArgument(a))
        {
            float ms = bench_cold<GUKernFP8D2>(a, rids_ptr, rotate, bk, cold, iters);
            print_row(sh,
                      B,
                      "gate_fp8_d2",
                      ms,
                      gu_flops(B, H, I, K),
                      gu_bytes(B, H, I, K, ebytes<fp8_t>(), ebytes<fp8_t>()));
        }
    }
    // gate_up FP4 (packed): separate packed pools (E*I*H/2 bytes each) + PerTensor
    // dummy scale, mirroring down_fp4_h2. hidden/2 stride is now accepted by the
    // patched gate_up kernel. Only run when hidden is a multiple of the FP4 tile.
    if(H % (64 * kVec4) == 0)
    {
        DeviceMem w_gate_fp4(static_cast<std::size_t>(E) * I * (H / 2) * sizeof(uint8_t));
        DeviceMem w_up_fp4(static_cast<std::size_t>(E) * I * (H / 2) * sizeof(uint8_t));
        DeviceMem w_gu_fp4_sc(sizeof(float));
        float one = 1.0f;
        w_gu_fp4_sc.ToDevice(&one, sizeof(float));
        auto gu4_sc = static_cast<const float*>(w_gu_fp4_sc.GetDeviceBuffer());

        typename GUKernFP4::Kargs a{};
        a.p_x                 = x_bf16_dev.GetDeviceBuffer();
        a.p_x_scale           = nullptr;
        a.p_w_gate            = w_gate_fp4.GetDeviceBuffer();
        a.p_w_gate_scale      = gu4_sc;
        a.p_w_up              = w_up_fp4.GetDeviceBuffer();
        a.p_w_up_scale        = gu4_sc;
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = inter_ptr;
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H / 2;
        a.stride_w_up         = H / 2;
        a.stride_intermediate = I;
        if(GUKernFP4::IsSupportedArgument(a))
        {
            float ms = bench_cold<GUKernFP4>(a, rids_ptr, rotate, bk, cold, iters);
            print_row(sh,
                      B,
                      "gate_up_fp4",
                      ms,
                      gu_flops(B, H, I, K),
                      gu_bytes(B, H, I, K, ebytes<bf16_t>(), ebytes<pk_fp4_t>()));
        }
    }
    // gate/up weights go out of scope here -- freed before down weights arrive.

    // -- down kernels (allocate on demand, freed at end of scope) -------------
    if(I % 128 == 0)
    {
        DeviceMem w_dn_fp8(static_cast<std::size_t>(E) * H * I * sizeof(fp8_t));
        DeviceMem w_dn_fp8_sc(E * (H / 128) * (I / 128) * sizeof(float));
        auto wd_sc = static_cast<const float*>(w_dn_fp8_sc.GetDeviceBuffer());

        typename DnKernFP8H2D2::Kargs a{};
        a.p_intermediate      = inter_ptr;
        a.p_w_down            = w_dn_fp8.GetDeviceBuffer();
        a.p_w_down_scale      = wd_sc;
        a.p_router_ids        = rids_ptr;
        a.p_router_wts        = rwts_ptr;
        a.p_y                 = y_ptr;
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_intermediate = I;
        a.stride_w_down       = I;
        a.stride_y            = H;
        if(DnKernFP8H2D2::IsSupportedArgument(a))
        {
            float ms = bench_cold<DnKernFP8H2D2>(a, rids_ptr, rotate, bk, cold, iters);
            print_row(sh,
                      B,
                      "down_h2_d2",
                      ms,
                      dn_flops(B, H, I, K),
                      dn_bytes(B, H, I, K, ebytes<fp8_t>()));
        }
    }
    if(I % (64 * kVec4) == 0)
    {
        DeviceMem w_dn_fp4(static_cast<std::size_t>(E) * H * (I / 2) * sizeof(uint8_t));
        DeviceMem w_dn_fp4_sc(sizeof(float));
        float one = 1.0f;
        w_dn_fp4_sc.ToDevice(&one, sizeof(float));
        auto wd4_sc = static_cast<const float*>(w_dn_fp4_sc.GetDeviceBuffer());

        typename DnKernFP4H2::Kargs a{};
        a.p_intermediate      = inter_ptr;
        a.p_w_down            = w_dn_fp4.GetDeviceBuffer();
        a.p_w_down_scale      = wd4_sc;
        a.p_router_ids        = rids_ptr;
        a.p_router_wts        = rwts_ptr;
        a.p_y                 = y_ptr;
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_intermediate = I;
        a.stride_w_down       = I / 2;
        a.stride_y            = H;
        if(DnKernFP4H2::IsSupportedArgument(a))
        {
            float ms = bench_cold<DnKernFP4H2>(a, rids_ptr, rotate, bk, cold, iters);
            print_row(sh,
                      B,
                      "down_fp4_h2",
                      ms,
                      dn_flops(B, H, I, K),
                      dn_bytes(B, H, I, K, ebytes<pk_fp4_t>()));
        }
    }
}

// ---------------------------------------------------------------------------
// Full-D7 numerical validation (CK_WD_VALIDATE=1): run each FP8/BF16/FP4 kernel
// ONCE on real, host-quantized inputs with *non-uniform per-block* scales, and
// dump inputs+scales+output so a Python validator can rebuild the torch reference
// on the identical bytes and compare (cos / allclose).  Perf mode is untouched --
// this path returns before the timing loop.
//
// Scales are real Block2D arrays (weight 128x128 for FP8, x 1x128 for FP8-act,
// weight 1x32 for FP4/MXFP4) filled with random per-block values -- so this also
// validates CK's Block2D scale-index layout, not just the matmul.  FP4 uses the
// generic load_block2d_scale (1,32) path with power-of-two float scales (exactly
// representable, matching FlyDSL's e8m0 range) -- no CK kernel change required.
// ---------------------------------------------------------------------------

static void write_raw(const std::string& path, const void* data, std::size_t bytes)
{
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

template <typename Kern>
static bool run_once(typename Kern::Kargs a)
{
    if(!Kern::IsSupportedArgument(a))
        return false;
    stream_config s{};
    make_kernel(Kern{}, Kern::GridSize(a), Kern::BlockSize(), 0, a)(s);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    return true;
}

// Deterministic host fills -----------------------------------------------------
static std::vector<bf16_t> host_bf16(std::size_t n, std::mt19937& rng, float lo, float hi)
{
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<bf16_t> v(n);
    for(auto& e : v)
        e = type_convert<bf16_t>(d(rng));
    return v;
}
static std::vector<fp8_t> host_fp8(std::size_t n, std::mt19937& rng, float lo, float hi)
{
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<fp8_t> v(n);
    for(auto& e : v)
        e = type_convert<fp8_t>(d(rng));
    return v;
}
// Non-uniform float scales in [lo,hi] (one per Block2D block).
static std::vector<float> host_scale(std::size_t n, std::mt19937& rng, float lo, float hi)
{
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for(auto& e : v)
        e = d(rng);
    return v;
}
// Power-of-two float scales 2^exp, exp in [elo,ehi] -- exactly representable, so
// the FP4 leg's only error source is bf16 rounding (mirrors FlyDSL's e8m0 range).
static std::vector<float> host_scale_pow2(std::size_t n, std::mt19937& rng, int elo, int ehi)
{
    std::uniform_int_distribution<int> d(elo, ehi);
    std::vector<float> v(n);
    for(auto& e : v)
        e = std::ldexp(1.0f, d(rng));
    return v;
}
// Pack 2 FP4 codes/byte (low nibble = even element), matching FlyDSL's convention.
static std::vector<uint8_t>
host_fp4_packed(std::size_t rows, std::size_t cols, std::mt19937& rng, std::vector<uint8_t>& codes)
{
    std::uniform_int_distribution<int> d(0, 15);
    std::vector<uint8_t> packed(rows * cols / 2);
    codes.resize(rows * cols);
    for(std::size_t r = 0; r < rows; ++r)
        for(std::size_t c = 0; c < cols; c += 2)
        {
            uint8_t lo                 = static_cast<uint8_t>(d(rng));
            uint8_t hi                 = static_cast<uint8_t>(d(rng));
            codes[r * cols + c]        = lo;
            codes[r * cols + c + 1]    = hi;
            packed[(r * cols + c) / 2] = static_cast<uint8_t>((hi << 4) | lo);
        }
    return packed;
}

// Rounds up a/b.
static std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

static void validate(const std::string& dir)
{
    // Small shapes but honoring the kernels' divisibility gates:
    //   gate_up needs HIDDEN % (warp_size*kVector==512) == 0,
    //   down needs INTER % 512 == 0; Block2D<128,128> needs dims % 128 == 0;
    //   FP4 needs HIDDEN/INTER % (64*kVec4==512)==0; bk = B*K <= E.
    const index_t B = 4, H = 512, I = 512, K = 2, E = 8;
    const int bk = static_cast<int>(B) * static_cast<int>(K);
    std::mt19937 rng(20260818u);

    // Block2D block dims (must match the CK type aliases above).
    constexpr int WBN = 128, WBK = 128; // FP8 weight scale
    constexpr int XBN = 1, XBK = 128;   // FP8 activation scale
    constexpr int MBN = 1, MBK = 32;    // FP4/MXFP4 weight scale

    // Router ids: distinct experts per (token,slot); router wts uniform 1/K.
    std::vector<int32_t> rids(bk);
    for(int i = 0; i < bk; ++i)
        rids[i] = i % static_cast<int>(E);
    std::vector<float> rwts(bk, 1.0f / static_cast<float>(K));

    DeviceMem rids_dev(bk * sizeof(int32_t));
    rids_dev.ToDevice(rids.data());
    DeviceMem rwts_dev(bk * sizeof(float));
    rwts_dev.ToDevice(rwts.data());
    auto rids_ptr = static_cast<const int32_t*>(rids_dev.GetDeviceBuffer());
    auto rwts_ptr = static_cast<const float*>(rwts_dev.GetDeviceBuffer());

    std::vector<std::string> done;

    auto upload_scale = [](DeviceMem& m, const std::vector<float>& s) {
        m.ToDevice(s.data(), s.size() * sizeof(float));
    };

    // Number of Block2D scale floats the kernel indexes: ceil(rows/BN) * (cols/BK).
    auto n_block2d = [](std::size_t rows, std::size_t cols, int bn, int bk) {
        return ceil_div(rows, bn) * (cols / bk);
    };

    // ------------------------------------------------------------------ FP8 gate
    // Shared FP8 gate/up weights + non-uniform 128x128 scales (rows = E*I, cols=H).
    auto wg_h = host_fp8(static_cast<std::size_t>(E) * I * H, rng, -0.25f, 0.25f);
    auto wu_h = host_fp8(static_cast<std::size_t>(E) * I * H, rng, -0.25f, 0.25f);
    DeviceMem wg_dev(wg_h.size() * sizeof(fp8_t)), wu_dev(wu_h.size() * sizeof(fp8_t));
    wg_dev.ToDevice(wg_h.data());
    wu_dev.ToDevice(wu_h.data());
    const std::size_t gu_nsc = n_block2d(static_cast<std::size_t>(E) * I, H, WBN, WBK);
    auto wgs_h               = host_scale(gu_nsc, rng, 0.5f, 1.5f);
    auto wus_h               = host_scale(gu_nsc, rng, 0.5f, 1.5f);
    DeviceMem wgs_dev(gu_nsc * sizeof(float)), wus_dev(gu_nsc * sizeof(float));
    upload_scale(wgs_dev, wgs_h);
    upload_scale(wus_dev, wus_h);
    auto wgs_ptr = static_cast<const float*>(wgs_dev.GetDeviceBuffer());
    auto wus_ptr = static_cast<const float*>(wus_dev.GetDeviceBuffer());

    DeviceMem gu_out_dev(static_cast<std::size_t>(B) * K * I * sizeof(bf16_t));

    auto dump_gate = [&](const std::string& name, bool fp8_act) {
        std::vector<bf16_t> out(static_cast<std::size_t>(B) * K * I);
        gu_out_dev.FromDevice(out.data());
        write_raw(dir + "/" + name + ".wg.bin", wg_h.data(), wg_h.size() * sizeof(fp8_t));
        write_raw(dir + "/" + name + ".wu.bin", wu_h.data(), wu_h.size() * sizeof(fp8_t));
        write_raw(dir + "/" + name + ".wgs.bin", wgs_h.data(), wgs_h.size() * sizeof(float));
        write_raw(dir + "/" + name + ".wus.bin", wus_h.data(), wus_h.size() * sizeof(float));
        write_raw(dir + "/" + name + ".rids.bin", rids.data(), rids.size() * sizeof(int32_t));
        write_raw(dir + "/" + name + ".out.bin", out.data(), out.size() * sizeof(bf16_t));
        static_cast<void>(fp8_act);
        done.push_back(name);
    };

    // -- gate_bf16_d2 : bf16 act x fp8 weight -----------------------------------
    {
        auto x_h = host_bf16(static_cast<std::size_t>(B) * H, rng, -1.0f, 1.0f);
        DeviceMem x_dev(x_h.size() * sizeof(bf16_t));
        x_dev.ToDevice(x_h.data());
        typename GUKernBF16D2::Kargs a{};
        a.p_x                 = x_dev.GetDeviceBuffer();
        a.p_x_scale           = nullptr;
        a.p_w_gate            = wg_dev.GetDeviceBuffer();
        a.p_w_gate_scale      = wgs_ptr;
        a.p_w_up              = wu_dev.GetDeviceBuffer();
        a.p_w_up_scale        = wus_ptr;
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = gu_out_dev.GetDeviceBuffer();
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H;
        a.stride_w_up         = H;
        a.stride_intermediate = I;
        if(run_once<GUKernBF16D2>(a))
        {
            write_raw(dir + "/gate_bf16_d2.x.bin", x_h.data(), x_h.size() * sizeof(bf16_t));
            dump_gate("gate_bf16_d2", /*fp8_act=*/false);
        }
        else
            std::cerr << "# validate: gate_bf16_d2 unsupported, skipped\n";
    }

    // -- gate_fp8_d2 : fp8 act x fp8 weight (non-uniform 1x128 x-scale) ----------
    {
        auto x_h = host_fp8(static_cast<std::size_t>(B) * H, rng, -1.0f, 1.0f);
        DeviceMem x_dev(x_h.size() * sizeof(fp8_t));
        x_dev.ToDevice(x_h.data());
        const std::size_t x_nsc = n_block2d(B, H, XBN, XBK);
        auto xs_h               = host_scale(x_nsc, rng, 0.5f, 1.5f);
        DeviceMem x_sc(x_nsc * sizeof(float));
        upload_scale(x_sc, xs_h);
        typename GUKernFP8D2::Kargs a{};
        a.p_x                 = x_dev.GetDeviceBuffer();
        a.p_x_scale           = static_cast<const float*>(x_sc.GetDeviceBuffer());
        a.p_w_gate            = wg_dev.GetDeviceBuffer();
        a.p_w_gate_scale      = wgs_ptr;
        a.p_w_up              = wu_dev.GetDeviceBuffer();
        a.p_w_up_scale        = wus_ptr;
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = gu_out_dev.GetDeviceBuffer();
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H;
        a.stride_w_up         = H;
        a.stride_intermediate = I;
        if(run_once<GUKernFP8D2>(a))
        {
            write_raw(dir + "/gate_fp8_d2.x.bin", x_h.data(), x_h.size() * sizeof(fp8_t));
            write_raw(dir + "/gate_fp8_d2.xs.bin", xs_h.data(), xs_h.size() * sizeof(float));
            dump_gate("gate_fp8_d2", /*fp8_act=*/true);
        }
        else
            std::cerr << "# validate: gate_fp8_d2 unsupported, skipped\n";
    }

    // -- gate_up_fp4 : bf16 act x fp4 weight, 1x32 pow2 scale -------------------
    {
        std::vector<uint8_t> gcodes, ucodes;
        auto wg4 = host_fp4_packed(static_cast<std::size_t>(E) * I, H, rng, gcodes);
        auto wu4 = host_fp4_packed(static_cast<std::size_t>(E) * I, H, rng, ucodes);
        DeviceMem wg4_dev(wg4.size()), wu4_dev(wu4.size());
        wg4_dev.ToDevice(wg4.data());
        wu4_dev.ToDevice(wu4.data());
        const std::size_t nsc = n_block2d(static_cast<std::size_t>(E) * I, H, MBN, MBK);
        auto wgs4             = host_scale_pow2(nsc, rng, -4, 0);
        auto wus4             = host_scale_pow2(nsc, rng, -4, 0);
        DeviceMem wgs4_dev(nsc * sizeof(float)), wus4_dev(nsc * sizeof(float));
        upload_scale(wgs4_dev, wgs4);
        upload_scale(wus4_dev, wus4);
        auto x_h = host_bf16(static_cast<std::size_t>(B) * H, rng, -1.0f, 1.0f);
        DeviceMem x_dev(x_h.size() * sizeof(bf16_t));
        x_dev.ToDevice(x_h.data());
        typename GUKernFP4MX::Kargs a{};
        a.p_x                 = x_dev.GetDeviceBuffer();
        a.p_x_scale           = nullptr;
        a.p_w_gate            = wg4_dev.GetDeviceBuffer();
        a.p_w_gate_scale      = static_cast<const float*>(wgs4_dev.GetDeviceBuffer());
        a.p_w_up              = wu4_dev.GetDeviceBuffer();
        a.p_w_up_scale        = static_cast<const float*>(wus4_dev.GetDeviceBuffer());
        a.p_router_ids        = rids_ptr;
        a.p_intermediate      = gu_out_dev.GetDeviceBuffer();
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_x            = H;
        a.stride_w_gate       = H / 2;
        a.stride_w_up         = H / 2;
        a.stride_intermediate = I;
        if(run_once<GUKernFP4MX>(a))
        {
            std::vector<bf16_t> out(static_cast<std::size_t>(B) * K * I);
            gu_out_dev.FromDevice(out.data());
            write_raw(dir + "/gate_up_fp4.x.bin", x_h.data(), x_h.size() * sizeof(bf16_t));
            write_raw(dir + "/gate_up_fp4.wg.bin", wg4.data(), wg4.size());
            write_raw(dir + "/gate_up_fp4.wu.bin", wu4.data(), wu4.size());
            write_raw(dir + "/gate_up_fp4.wgs.bin", wgs4.data(), wgs4.size() * sizeof(float));
            write_raw(dir + "/gate_up_fp4.wus.bin", wus4.data(), wus4.size() * sizeof(float));
            write_raw(dir + "/gate_up_fp4.rids.bin", rids.data(), rids.size() * sizeof(int32_t));
            write_raw(dir + "/gate_up_fp4.out.bin", out.data(), out.size() * sizeof(bf16_t));
            done.push_back("gate_up_fp4");
        }
        else
            std::cerr << "# validate: gate_up_fp4 (1x32) unsupported, skipped\n";
    }

    // -- down_h2_d2 : bf16 inter x fp8 weight, 128x128 scale --------------------
    {
        auto inter_h = host_bf16(static_cast<std::size_t>(B) * K * I, rng, -1.0f, 1.0f);
        DeviceMem inter_dev(inter_h.size() * sizeof(bf16_t));
        inter_dev.ToDevice(inter_h.data());
        auto wd_h = host_fp8(static_cast<std::size_t>(E) * H * I, rng, -0.25f, 0.25f);
        DeviceMem wd_dev(wd_h.size() * sizeof(fp8_t));
        wd_dev.ToDevice(wd_h.data());
        const std::size_t nsc = n_block2d(static_cast<std::size_t>(E) * H, I, WBN, WBK);
        auto wds_h            = host_scale(nsc, rng, 0.5f, 1.5f);
        DeviceMem wd_sc(nsc * sizeof(float));
        upload_scale(wd_sc, wds_h);
        DeviceMem y_dev(static_cast<std::size_t>(B) * H * sizeof(bf16_t));
        typename DnKernFP8H2D2::Kargs a{};
        a.p_intermediate      = inter_dev.GetDeviceBuffer();
        a.p_w_down            = wd_dev.GetDeviceBuffer();
        a.p_w_down_scale      = static_cast<const float*>(wd_sc.GetDeviceBuffer());
        a.p_router_ids        = rids_ptr;
        a.p_router_wts        = rwts_ptr;
        a.p_y                 = y_dev.GetDeviceBuffer();
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_intermediate = I;
        a.stride_w_down       = I;
        a.stride_y            = H;
        if(run_once<DnKernFP8H2D2>(a))
        {
            std::vector<bf16_t> out(static_cast<std::size_t>(B) * H);
            y_dev.FromDevice(out.data());
            write_raw(
                dir + "/down_h2_d2.inter.bin", inter_h.data(), inter_h.size() * sizeof(bf16_t));
            write_raw(dir + "/down_h2_d2.wd.bin", wd_h.data(), wd_h.size() * sizeof(fp8_t));
            write_raw(dir + "/down_h2_d2.wds.bin", wds_h.data(), wds_h.size() * sizeof(float));
            write_raw(dir + "/down_h2_d2.rids.bin", rids.data(), rids.size() * sizeof(int32_t));
            write_raw(dir + "/down_h2_d2.rwts.bin", rwts.data(), rwts.size() * sizeof(float));
            write_raw(dir + "/down_h2_d2.out.bin", out.data(), out.size() * sizeof(bf16_t));
            done.push_back("down_h2_d2");
        }
        else
            std::cerr << "# validate: down_h2_d2 unsupported, skipped\n";
    }

    // -- down_fp4_h2 : bf16 inter x fp4 weight, 1x32 pow2 scale -----------------
    {
        auto inter_h = host_bf16(static_cast<std::size_t>(B) * K * I, rng, -1.0f, 1.0f);
        DeviceMem inter_dev(inter_h.size() * sizeof(bf16_t));
        inter_dev.ToDevice(inter_h.data());
        std::vector<uint8_t> dcodes;
        auto wd4 = host_fp4_packed(static_cast<std::size_t>(E) * H, I, rng, dcodes);
        DeviceMem wd4_dev(wd4.size());
        wd4_dev.ToDevice(wd4.data());
        const std::size_t nsc = n_block2d(static_cast<std::size_t>(E) * H, I, MBN, MBK);
        auto wds4             = host_scale_pow2(nsc, rng, -4, 0);
        DeviceMem wd4_sc(nsc * sizeof(float));
        upload_scale(wd4_sc, wds4);
        DeviceMem y_dev(static_cast<std::size_t>(B) * H * sizeof(bf16_t));
        typename DnKernFP4H2MX::Kargs a{};
        a.p_intermediate      = inter_dev.GetDeviceBuffer();
        a.p_w_down            = wd4_dev.GetDeviceBuffer();
        a.p_w_down_scale      = static_cast<const float*>(wd4_sc.GetDeviceBuffer());
        a.p_router_ids        = rids_ptr;
        a.p_router_wts        = rwts_ptr;
        a.p_y                 = y_dev.GetDeviceBuffer();
        a.b                   = B;
        a.hidden              = H;
        a.inter               = I;
        a.top_k               = K;
        a.e                   = E;
        a.stride_intermediate = I;
        a.stride_w_down       = I / 2;
        a.stride_y            = H;
        if(run_once<DnKernFP4H2MX>(a))
        {
            std::vector<bf16_t> out(static_cast<std::size_t>(B) * H);
            y_dev.FromDevice(out.data());
            write_raw(
                dir + "/down_fp4_h2.inter.bin", inter_h.data(), inter_h.size() * sizeof(bf16_t));
            write_raw(dir + "/down_fp4_h2.wd.bin", wd4.data(), wd4.size());
            write_raw(dir + "/down_fp4_h2.wds.bin", wds4.data(), wds4.size() * sizeof(float));
            write_raw(dir + "/down_fp4_h2.rids.bin", rids.data(), rids.size() * sizeof(int32_t));
            write_raw(dir + "/down_fp4_h2.rwts.bin", rwts.data(), rwts.size() * sizeof(float));
            write_raw(dir + "/down_fp4_h2.out.bin", out.data(), out.size() * sizeof(bf16_t));
            done.push_back("down_fp4_h2");
        }
        else
            std::cerr << "# validate: down_fp4_h2 (1x32) unsupported, skipped\n";
    }

    // Manifest: dims + Block2D block dims + which kernels dumped (fixed file names).
    std::ostringstream js;
    js << "{\n  \"B\": " << B << ", \"H\": " << H << ", \"I\": " << I << ", \"K\": " << K
       << ", \"E\": " << E << ",\n"
       << "  \"w_block\": [" << WBN << ", " << WBK << "], \"x_block\": [" << XBN << ", " << XBK
       << "], \"mx_block\": [" << MBN << ", " << MBK << "],\n  \"kernels\": [";
    for(std::size_t i = 0; i < done.size(); ++i)
        js << (i ? ", " : "") << '"' << done[i] << '"';
    js << "]\n}\n";
    write_raw(dir + "/manifest.json", js.str().data(), js.str().size());
    std::cerr << "# validate: wrote " << done.size() << " kernel dump(s) to " << dir << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static std::vector<std::string> split_csv(const char* s)
{
    std::vector<std::string> out;
    if(!s || !*s)
        return out;
    std::istringstream ss(s);
    std::string tok;
    while(std::getline(ss, tok, ','))
    {
        if(!tok.empty())
            out.push_back(tok);
    }
    return out;
}

int main()
{
    // D7 numerical validation: dump FP8/BF16/FP4 kernel I/O for a Python torch
    // cross-check, then exit (perf mode stays the default when the flag is unset).
    if(const char* v = std::getenv("CK_WD_VALIDATE"); v && std::string(v) != "0")
    {
        const char* d = std::getenv("CK_WD_VALIDATE_DIR");
        validate(d && *d ? std::string(d) : std::string("./ck_validate_dump"));
        return 0;
    }

    // Shape filter from env
    auto shape_env = split_csv(std::getenv("CK_WD_SHAPES"));
    auto batch_env = split_csv(std::getenv("CK_WD_BATCHES"));
    int cold       = std::getenv("CK_WD_COLD") ? std::stoi(std::getenv("CK_WD_COLD")) : 5;
    int iters      = std::getenv("CK_WD_ITERS") ? std::stoi(std::getenv("CK_WD_ITERS")) : 30;
    // Cold-HBM router rotation: <=0 => auto (tile the whole E pool per shape),
    // 1 => warm baseline (single fixed expert group), >1 => that many groups.
    int rotate_env = std::getenv("CK_WD_ROTATE") ? std::stoi(std::getenv("CK_WD_ROTATE")) : 0;
    // Output format: CK_WD_FORMAT=csv -> machine-readable; anything else -> pretty table.
    if(const char* fmt = std::getenv("CK_WD_FORMAT"))
        g_csv = (std::string(fmt) == "csv");

    std::set<std::string> shape_filter(shape_env.begin(), shape_env.end());
    std::vector<index_t> batches;
    for(const auto& b : batch_env)
        batches.push_back(std::stoi(b));
    if(batches.empty())
        batches = {1, 2, 4, 8};

    // Config/provenance to stderr (keeps stdout table clean for parsing).
    // base_commit = pinned CK checkout; patch = local A4 fix on top (the gate_up
    // kernel packed-FP4 stride acceptance), so the worktree is not pristine.
    std::cerr << "# ck_bench_warp_decode  base_commit=62e30c9098 patch=A4-gateup-fp4-packed-stride"
              << "  cold=" << cold << "  iters=" << iters << "  rotate="
              << (rotate_env > 0 ? std::to_string(rotate_env) : std::string("auto(ceil(E/BK))"))
              << "  format=" << (g_csv ? "csv" : "table")
              << "  mechanism=manual-hipEvent+disjoint-router-rotation\n";

    if(g_csv)
    {
        std::cout << "shape,H,I,K,E,B,kernel,us\n";
    }
    else
    {
        std::cout << std::left << std::setw(16) << "shape" << std::right << std::setw(4) << "B"
                  << std::setw(18) << "kernel" << std::setw(10) << "ms" << std::setw(10)
                  << "TFLOP/s" << std::setw(10) << "GB/s" << "\n"
                  << std::string(68, '-') << "\n";
    }

    for(const auto& sh : ALL_SHAPES)
    {
        if(!shape_filter.empty() && shape_filter.find(sh.name) == shape_filter.end())
            continue;
        for(index_t B : batches)
            bench(sh, B, cold, iters, rotate_env);
    }
}

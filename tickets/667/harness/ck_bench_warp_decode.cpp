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
#include <cstdlib>
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

// NVFP4 MoE kernels for gfx950: one GEMM template for both stages and every tile, configured by
// the kernel name (Cfg in moe_nvfp4_common.h). The fused sort + cast is in
// moe_nvfp4_sort.hip.
//
// A block (BM = 16 m_tiles rows of one expert, sorted) x N group (waves x n_tiles weight
// tiles of 16 rows) is one WG. Each wave walks its K slice (K / ksplit K-tiles) in chunks:
// weights + E4M3 scale groups + X -> fp4 decode + f16 MFMA (mma_ktile), then the K partial sums
// of the ksplit waves are reduced in LDS and the stage epilogue runs (stage 1: SwiGLU-OAI, f16
// x_inter; stage 2: route weight, bf16 / f32 atomics). The X placement (Cfg::x) picks the weight
// pipeline: a register ring (moe_nvfp4_wreg.h) for X in registers / an LDS stream / LDS
// resident, an LDS-DMA ring for X by DMA.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <utility>

#include "moe_nvfp4_common.h"
#include "moe_nvfp4_wdma.h"
#include "moe_nvfp4_wreg.h"

// ---------------------------------------------------------------- the GEMM kernel
// LDS v4i of a kernel: K partial sums + C tiles (X_REG), X ring + C tiles (X_STREAM), the X
// block, reused by the C tiles (X_RESIDENT), the W rings + X ring (X_DMA).
template <Cfg C>
constexpr int lds_v4()
{
    constexpr int PARTIAL = (C.ksplit - 1) * C.waves * C.n_tiles * C.m_tiles * 64;
    if constexpr(C.x == X_DMA)
        return dma_lds_v4(C);
    else if constexpr(C.x == X_RESIDENT)
        return x_lds_v4(C) > c_tile_v4(C) ? x_lds_v4(C) : c_tile_v4(C);
    else
        return PARTIAL + x_lds_v4(C) + c_tile_v4(C);
}

template <Cfg C, int MA>
DEV void gemm_body(const Params& p, int mblk, int e, v4i* lds)
{
    const WaveIds w         = wave_ids<C>();
    const Sources<C.n_tiles> src = make_sources<C>(p, e, w, C.k_tiles ? C.k_tiles : p.K / 128);
    Routing<C.m_tiles> rt;
    v4f acc[C.n_tiles][C.m_tiles];
    float rw[C.m_tiles]; // stage 2 route weights (C.staged: read by the K loop)
    if constexpr(C.x == X_DMA)
        dma_k_loop<C, MA>(p, mblk, e, w, src, rt, acc, lds);
    else
        reg_k_loop<C, MA>(p, mblk, e, w, src, rt, acc, lds, rw);
    if(!k_reduce<C, MA>(acc, w, (v4f*)lds))
        return;
    if constexpr(C.s1)
        s1_epilogue<C, MA>(p, acc, e, rt, w);
    else
    {
        if constexpr(!C.staged)
            read_route_weights<C.m_tiles, MA>(rw, p, e, mblk, w.lane_col);
        constexpr int C_V4 = C.x == X_RESIDENT ? 0 : lds_v4<C>() - c_tile_v4(C);
        if constexpr(C.x == X_RESIDENT)
            __syncthreads(); // the C tiles reuse the X LDS
        float* c_lds = (float*)(lds + C_V4) + w.wave_n * 16 * C.m_tiles * c_row_floats(C.n_tiles);
        s2_epilogue<C, MA>(p, acc, rw, rt, c_lds, w);
    }
}

// Grid: x = N group, y = sorted block (XCD remap: any WG order). Empty for other archs of a
// multi-arch build (moe_nvfp4 rejects them before any launch).
template <Cfg C>
__global__ __launch_bounds__(64 * C.ksplit * C.waves)
    __attribute__((amdgpu_waves_per_eu(waves_hint(C)))) void nvfp4_moe_gemm(Params p)
{
#if defined(__gfx950__)
    constexpr int BM = 16 * C.m_tiles;
    const int nvalid = p.num_valid[0];
    int mblk         = blockIdx.y;
    if constexpr(C.xcdremap)
    {
        mblk = xcd_remap<1>(BM, nvalid);
        if(mblk < 0)
            return;
    }
    else if constexpr(C.staged)
        lds_n_block = blockIdx.x;
    // the expert load is issued before the exit branch (grid y never exceeds the sorted buffers)
    const int e = p.sorted_eids[mblk];
    require_sgpr(e);
    if(mblk * BM >= nvalid)
        return;
    __shared__ v4i lds[lds_v4<C>() > 0 ? lds_v4<C>() : 1];
    dispatch_active_tiles<C.m_tiles>(
        C.m_tiles > 1 ? active_tiles<C.m_tiles>(p, mblk) : C.m_tiles,
        [&](auto ma) { gemm_body<C, decltype(ma)::value>(p, mblk, e, lds); });
#endif
}

// ---------------------------------------------------------------- host side
// Whether a kernel takes reduction length K: compile-time K equal; runtime K: whole units per
// wave slice and at least the peeled cycles.
template <Cfg C>
static bool k_fits(int K)
{
    if constexpr(C.x == X_DMA)
        return dma_k_fits<C>(K);
    const int k_tiles = K / 128;
    if(K % 128 || k_tiles % C.ksplit)
        return false;
    if constexpr(C.k_tiles > 0)
        return k_tiles == C.k_tiles;
    const int chunks = k_tiles / C.ksplit / C.chunk_k_tiles;
    return k_tiles / C.ksplit % C.chunk_k_tiles == 0 && chunks % reg_k_unit(C) == 0 &&
           chunks >= reg_head(C) * reg_cycle(C) + lookahead(C) && chunks > 0;
}

using LaunchFn = int (*)(const Params&, int, int, hipStream_t);
template <Cfg C>
static int launch(const Params& p, int grid_x, int grid_y, hipStream_t stream)
{
    if(!k_fits<C>(p.K))
        return -3;
    hipLaunchKernelGGL(
        (nvfp4_moe_gemm<C>), dim3(grid_x, grid_y), dim3(64 * C.ksplit * C.waves), 0, stream, p);
    return (int)hipGetLastError();
}

struct Instance
{
    const char* name;
    Cfg cfg;
    LaunchFn launch;
    bool (*fits)(int K);
};
template <Cfg C>
constexpr Instance instance(const char* name)
{
    static_assert(cfg_valid(C), "bad NVFP4 MoE kernel config");
    return {name, C, &launch<C>, &k_fits<C>};
}

// The kernel table of a module: generated instance files (gen_instances.py) define the
// Instance arrays and the registry functions. The name is the configuration (kernel_name); the
// host selects kernels by name (aiter/ops/moe_nvfp4.py, aiter/configs/moe_nvfp4_dispatch.csv).
extern "C" int moe_nvfp4_kernels_num_cfgs();
extern "C" const char* moe_nvfp4_kernels_cfg_name(int i);
// out: CfgField order (moe_nvfp4_entry.h)
extern "C" void moe_nvfp4_kernels_cfg_info(int i, int* out);
extern "C" int moe_nvfp4_kernels_cfg_fits(int i, int K);
extern "C" int
moe_nvfp4_kernels_launch(int kernel, const Params* hp, int grid_x, int grid_y, hipStream_t stream);

// NVFP4 MoE kernels for gfx950: one GEMM template for both stages and every tile, configured by
// the kernel name (Cfg, parse_cfg in moe_nvfp4_common.h), and the fused sort + cast.
//
// A block (BM = 16 mt rows of one expert, sorted) x N group (wn waves x nt weight tiles of 16
// rows) is one WG. Each wave walks its K slice (K / wk K-tiles) in chunks: weights + E4M3 scale
// groups + X -> fp4 decode + f16 MFMA (mma_ktile), then the K partial sums of the wk waves are
// reduced in LDS and the stage epilogue runs (stage 1: SwiGLU-OAI, f16 x_inter; stage 2: route
// weight, bf16 atomics). The X placement (Cfg::x) picks the weight pipeline: a register ring
// (moe_nvfp4_wreg.h) for X in registers / an LDS stream / LDS resident, an LDS-DMA ring for
// X by DMA.
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
    constexpr int PARTIAL = (C.wk - 1) * C.wn * C.nt * C.mt * 64;
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
    const Sources<C.nt> src = make_sources<C>(p, e, w, C.kt ? C.kt : p.K / 128);
    Routing<C.mt> rt;
    v4f acc[C.nt][C.mt];
    float rw[C.mt]; // stage 2 route weights (C.staged: read by the K loop)
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
            read_route_weights<C.mt, MA>(rw, p, e, mblk, w.lane_col);
        constexpr int C_V4 = C.x == X_RESIDENT ? 0 : lds_v4<C>() - c_tile_v4(C);
        if constexpr(C.x == X_RESIDENT)
            __syncthreads(); // the C tiles reuse the X LDS
        float* c_lds = (float*)(lds + C_V4) + w.wave_n * 16 * C.mt * c_row_floats(C.nt);
        s2_epilogue<C, MA>(p, acc, rw, rt, c_lds, w);
    }
}

// Grid: x = N group, y = sorted block (XCD remap: any WG order).
template <Cfg C>
__global__ __launch_bounds__(64 * C.wk * C.wn)
__attribute__((amdgpu_waves_per_eu(waves_hint(C)))) void nvfp4_moe_gemm(Params p)
{
    constexpr int BM = 16 * C.mt;
    const int nvalid = p.num_valid[0];
    int mblk         = blockIdx.y;
    if constexpr(C.xg)
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
    dispatch_active_tiles<C.mt>(C.mt > 1 ? active_tiles<C.mt>(p, mblk) : C.mt, [&](auto ma) {
        gemm_body<C, decltype(ma)::value>(p, mblk, e, lds);
    });
}

// ---------------------------------------------------------------- fused prep
// One launch replaces moe_sorting + activation cast: block 0 sorts the routing (one
// workgroup), blocks 1.. zero the bf16 output for the stage2 atomics and cast X (bf16) to f16
// (x_f16 null: no cast, for _xbf16 stage-1 kernels). Sorted layout matches the gemm kernels:
// per expert, ceil(count / BM) blocks of BM entries (token | slot << 24, padding token = T).
// Cast blocks: all SORT_THREADS lanes, 8 elements per lane (one 16 B load / zero / store); the
// f16 X store is non-temporal (gives stage1 2-4% of layer time back at M384-768).
constexpr int SORT_THREADS     = 1024;
constexpr int SORT_PER_THREAD  = 4; // T * topk <= 4096
constexpr int SORT_MAX_EXPERTS = 512;
constexpr int SORT_MAX_BLOCKS  = SORT_THREADS * SORT_PER_THREAD / 16 + SORT_MAX_EXPERTS;
static_assert(SORT_MAX_BLOCKS < (1 << 15), "packed block counts of the sort scan");

struct PrepParams
{
    const uint16_t* x;
    uint8_t* x_f16;
    uint16_t* out;
    const int* ids;
    const float* w;
    int* sorted_ids;
    float* sorted_w;
    int* sorted_eids;
    int* num_valid;
    int T, H, topk, E, BM, nblk;
};

__global__ __launch_bounds__(SORT_THREADS) void nvfp4_moe_sort_cast(PrepParams p)
{
    const int tid = threadIdx.x;
    if(blockIdx.x > 0)
    {
        const int vec = (blockIdx.x - 1) * SORT_THREADS + tid;
        if(vec >= p.T * p.H / 8)
            return;
        ((v4i*)p.out)[vec] = v4i{0, 0, 0, 0};
        if(p.x_f16)
            __builtin_nontemporal_store(cast_bf16x8_f16(((const v4i*)p.x)[vec]),
                                        (v4i*)p.x_f16 + vec);
        return;
    }

    __shared__ int count[SORT_MAX_EXPERTS];
    __shared__ int first_block[SORT_MAX_EXPERTS + 1]; // first block of each expert
    __shared__ int block_expert[SORT_MAX_BLOCKS];     // expert of each block
    const int n_rows = p.T * p.topk;
    const int BM     = p.BM;
    // all global loads first: one memory latency instead of SORT_PER_THREAD dependent ones
    int expert[SORT_PER_THREAD], rank[SORT_PER_THREAD];
    float weight[SORT_PER_THREAD];
#pragma unroll
    for(int i = 0; i < SORT_PER_THREAD; i++)
    {
        const int idx = tid + SORT_THREADS * i;
        expert[i]     = idx < n_rows ? p.ids[idx] : 0;
        weight[i]     = idx < n_rows ? p.w[idx] : 0.0f;
    }
    for(int e = tid; e < p.E; e += SORT_THREADS)
        count[e] = 0;
    __syncthreads();
#pragma unroll
    for(int i = 0; i < SORT_PER_THREAD; i++)
        rank[i] = tid + SORT_THREADS * i < n_rows ? atomicAdd(&count[expert[i]], 1) : 0;
    __syncthreads();
    if(tid < 64)
    {
        // Experts with more than one block go first (stable), so the long multi-block experts
        // (e.g. a fused shared expert) do not form the grid tail. One scan of both block
        // counts: multi-block experts' blocks in bits 16.., single-block experts' in bits
        // 0..15 (each total at most SORT_MAX_BLOCKS < 2^16).
        const int experts_per_lane = (p.E + 63) / 64;
        const int expert_begin     = tid * experts_per_lane;
        auto n_blocks              = [&](int e) { return (count[e] + BM - 1) / BM; };
        int blocks                 = 0;
        for(int k = 0; k < experts_per_lane; k++)
            if(const int e = expert_begin + k; e < p.E)
                blocks += n_blocks(e) > 1 ? n_blocks(e) << 16 : n_blocks(e);
        int scan = blocks;
#pragma unroll
        for(int d = 1; d < 64; d <<= 1)
        {
            const int lane_scan = __shfl_up(scan, d);
            scan += tid >= d ? lane_scan : 0;
        }
        const int total       = __shfl(scan, 63);
        const int total_multi = total >> 16;
        int next_multi        = (scan - blocks) >> 16;
        int next_single       = total_multi + ((scan - blocks) & 0xffff);
        for(int k = 0; k < experts_per_lane; k++)
            if(const int e = expert_begin + k; e < p.E)
            {
                int& next      = n_blocks(e) > 1 ? next_multi : next_single;
                first_block[e] = next;
                next += n_blocks(e);
            }
        if(tid == 63)
        {
            first_block[p.E] = total_multi + (total & 0xffff);
            p.num_valid[0]   = first_block[p.E] * BM;
        }
    }
    __syncthreads();
    for(int e = tid; e < p.E; e += SORT_THREADS)
        for(int b = first_block[e]; b < first_block[e] + (count[e] + BM - 1) / BM; b++)
            block_expert[b] = e;
    __syncthreads();
    const int n_valid_blocks = first_block[p.E];
    for(int b = tid; b < p.nblk; b += SORT_THREADS)
        p.sorted_eids[b] = b < n_valid_blocks ? block_expert[b] : 0;
    for(int pos = tid; pos < n_valid_blocks * BM; pos += SORT_THREADS)
    {
        const int e = block_expert[pos / BM];
        if(pos - first_block[e] * BM >= count[e])
        {
            p.sorted_ids[pos] = p.T;
            p.sorted_w[pos]   = 0.0f;
        }
    }
#pragma unroll
    for(int i = 0; i < SORT_PER_THREAD; i++)
    {
        const int idx = tid + SORT_THREADS * i;
        if(idx < n_rows)
        {
            const int pos     = first_block[expert[i]] * BM + rank[i];
            p.sorted_ids[pos] = ((idx % p.topk) << 24) | (idx / p.topk);
            p.sorted_w[pos]   = weight[i];
        }
    }
}

extern "C" int moe_nvfp4_kernels_sort(const PrepParams* hp, hipStream_t stream)
{
    const PrepParams p = *hp;
    if(p.T * p.topk > SORT_THREADS * SORT_PER_THREAD || p.E > SORT_MAX_EXPERTS)
        return -2;
    // one 8-element vector per cast lane
    const int n_cast_blocks = (p.T * p.H / 8 + SORT_THREADS - 1) / SORT_THREADS;
    hipLaunchKernelGGL(
        nvfp4_moe_sort_cast, dim3(1 + n_cast_blocks), dim3(SORT_THREADS), 0, stream, p);
    return (int)hipGetLastError();
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
    if(K % 128 || k_tiles % C.wk)
        return false;
    if constexpr(C.kt > 0)
        return k_tiles == C.kt;
    const int chunks = k_tiles / C.wk / C.kc;
    return k_tiles / C.wk % C.kc == 0 && chunks % reg_k_unit(C) == 0 &&
           chunks >= reg_head(C) * reg_cycle(C) + lookahead(C) && chunks > 0;
}

using LaunchFn = int (*)(const Params&, int, int, hipStream_t);
template <Cfg C>
static int launch(const Params& p, int grid_x, int grid_y, hipStream_t stream)
{
    if(!k_fits<C>(p.K))
        return -3;
    hipLaunchKernelGGL((nvfp4_moe_gemm<C>), dim3(grid_x, grid_y), dim3(64 * C.wk * C.wn), 0, stream,
                       p);
    return (int)hipGetLastError();
}

struct Instance
{
    const char* name;
    Cfg cfg;
    LaunchFn launch;
};
#define KERNEL(name) Instance{"nvfp4_moe_" name, parse_cfg(name), &launch<parse_cfg(name)>}

// The kernel instances, by name (the host selects kernels by name: aiter/ops/moe_nvfp4.py,
// aiter/configs/moe_nvfp4_dispatch.csv). The name is the configuration (parse_cfg).
// Tune-only builds: -DNVFP4_MOE_NAMES_INC='"<file>"' replaces the production list;
// -DNVFP4_MOE_TUNE_NAMES='"<file>"' appends the names of <file>; -DNVFP4_MOE_TUNE_ONLY drops
// the production list.
#ifndef NVFP4_MOE_NAMES_INC
#define NVFP4_MOE_NAMES_INC "moe_nvfp4_names.inc"
#endif
static const Instance g_instances[] = {
#ifndef NVFP4_MOE_TUNE_ONLY
#include NVFP4_MOE_NAMES_INC
#endif
#ifdef NVFP4_MOE_TUNE_NAMES
#include NVFP4_MOE_TUNE_NAMES
#endif
};

extern "C" int moe_nvfp4_kernels_num_cfgs()
{
    return sizeof(g_instances) / sizeof(g_instances[0]);
}
extern "C" const char* moe_nvfp4_kernels_cfg_name(int i) { return g_instances[i].name; }
// s1, mt, nt, wn, gs, xbf16
extern "C" void moe_nvfp4_kernels_cfg_info(int i, int* out)
{
    const Cfg& c = g_instances[i].cfg;
    int v[6]     = {c.s1, c.mt, c.nt, c.wn, c.gs, c.xbf16};
    for(int k = 0; k < 6; k++)
        out[k] = v[k];
}

extern "C" int
moe_nvfp4_kernels_launch(int cfg, const Params* hp, int grid_x, int grid_y, hipStream_t stream)
{
    if(cfg < 0 || cfg >= moe_nvfp4_kernels_num_cfgs())
        return -1;
    return g_instances[cfg].launch(*hp, grid_x, grid_y, stream);
}

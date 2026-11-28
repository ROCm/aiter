// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Muse Spark attention block, TP4 / gfx950: ONE persistent kernel per call per rank (construction checkpoint 1).
//
// grid = 256 CTAs x 256 threads (1 CTA per CU, full residency asserted by the host). Regions in program order,
// separated by grid barriers (epoch-scaled monotone arrival counter, bounded spins):
//   a  qkv_gemm      unit (144-col group, K eighth ks; 256 equal units): FP8 x FP8 MFMA (16x16x32_fp8_fp8), fixed-order
//                    cross-wave sum, fp32 partial [ks][m][n] written through.                         -- barrier 0
//   b  qk_epilogue   unit (row m, kv head h): sum of the 8 K partials * x_scale * w_scale -> bf16, per-head
//                    RMSNorm, q * scale_query_by, GPT-J RoPE (sliding), FP8 K/V at out_cache_loc (-> swa_mapping),
//                    q / gate bf16 scratch.                                                         -- barrier 1
//   c  attention     static equal-token split of all (request, kv head) segments (32-token steps, in-kernel
//                    prefix sum of seqused_k): S^T = K Q^T and O^T = V^T P^T on bf16 MFMAs from the FP8 cache, log2
//                    online softmax per wave, in-CTA merge of the 4 waves, one (o, max, denom) partial per item;
//                    denom = 0 for empty items. Lazy rescale (cross-lane column max only when a lane's local max
//                    grows by > 8 in log2 units); K/V prefetched NPF steps ahead with page ids two rings ahead.
//                                                                                                     -- barrier 2
//   d  merge_norm_gate unit (request, kv head, verify token): exp2 merge, bf16 round, per-head RMSNorm,
//                    * sigmoid(gate) -> o bf16 [M, 2048].                                            -- barrier 3
//   e  o_proj        tile t = bid: 32 output columns x K 2048 (software-pipelined bf16 MFMA), bf16 partial
//                    pushed write-through to owner(t) (reduce-scatter landing zone), flag = epoch.
//   f  RS/AG         owner reduce in source order 0..3 -> bf16, push to every rank's all-gather zone, flags.
//   g  boundary      per row: post RMSNorm, bf16 round, alpha*stream + beta*normed, pre RMSNorm (eff weight),
//                    ffn_in bf16, new_stream fp32, FP8 copy with amax/448 scale.
// Epoch g = ctrl[GEN] + 1, published by the last CTA through the ticket. Cross-rank data: system-scope
// write-through (sc0 sc1) stores + s_waitcnt vmcnt(0) before the flag; cross-CTA data inside the launch:
// device-scope write-through (sc1) stores + vmcnt(0) before the barrier arrival, plain loads after it. No fences.
// Every spin is bounded: on timeout (or a broken internal invariant) it records {code, observed, target} in
// ctrl[ERR..] and traps (FAIL_TRAP=0: record only), so the launch fails instead of returning a wrong result.
#include <hip/hip_runtime.h>
#include <cstdint>

#define NR 4
#define HID 8192
#define KLOC 2048
#define MAXTOK 64
#define MAXBS 8
#define NTILE 256
#define TPR 64
#define NTHR 256
#define NCHUNK_MAX 16
#define NQKV 4608
#define KQKV 8192
#define NQH 32
#define NKH 4
#define HD 64
#define NSEG_MAX (MAXBS * NKH)
#define NPIECE 272          // (CTA, request) pieces of the token-balanced split (<= GRID + MAXBS)
#define NPART (NPIECE * NKH) // one partial per (piece, kv head)
// READY=1: readiness edges replace grid barriers 0 (qkv -> epilogue), 1 (epilogue -> attention) and 3 (merge ->
// o_proj); READY=0: compile-time fallback with all four grid barriers (ablation / liveness escape hatch)
#ifndef READY
#define READY 1
#endif
#if READY
#define NBAR 1
#else
#define NBAR 4
#endif
#ifndef QLDS
#define QLDS 0
#endif
#ifndef PIN_REGS
#define PIN_REGS 1
#endif
#ifndef STATIC_MAX
#define STATIC_MAX 1
#endif
#define QLDS_OFF 20480   // floats: Q fragments at byte 80 KB of the region-c LDS (merge scratch uses < 68 KB)
#ifndef NPF
#define NPF 2
#endif
#ifndef PSPLIT
#define PSPLIT 0
#endif
#ifndef QSCL
#define QSCL 1
#endif
#ifndef LSUM_MFMA
#define LSUM_MFMA 1
#endif
#if (QSCL || LSUM_MFMA) && !STATIC_MAX
#error "QSCL / LSUM_MFMA assume the static softmax reference"
#endif
#ifndef MTQ_S
#define MTQ_S 1          // verify tokens per merge unit, sliding layer (bs > 2)
#endif
#ifndef MTQ_F
#define MTQ_F 1          // verify tokens per merge unit, full layer (bs > 2)
#endif
#ifndef MREV
#define MREV 1           // merge units on the highest CTA ids (item-free under the per-request split)
#endif
#ifndef WO_POL
#define WO_POL 1         // Wo slice deferred only by CTAs holding an attention item and a merge unit
#endif
#ifndef WO_WARM
#define WO_WARM 0         // item-free CTAs warm the item CTAs' Wo slices into L2 / MALL during the attention
#endif
#ifndef PROJ_DIAG
#define PROJ_DIAG 0
#endif
#ifndef PBF16
#define PBF16 1          // attention partial o rows stored in bf16
#endif
// pieces per merge load batch (bs > 2). Sliding (PSPLIT): 17 = the per-request piece count at QSMIN 4, so the merge
// is one load batch (S56 merge 4.8 -> 3.5 us; 24 / 32 measured slower). Full layer: 16.
#ifndef MBAT
#if PSPLIT
#define MBAT 17
#else
#define MBAT 16
#endif
#endif
#ifndef MERGE_DIAG
#define MERGE_DIAG 0
#endif
#ifndef ATT_DIAG
#define ATT_DIAG 0
#endif
#ifndef WO_LATE
#define WO_LATE 64
#endif
// ATT_RDY=1: grid barrier 2 replaced by per-request attention readiness (merge of request b waits only for the
// CTAs holding items of b); the barrier-2 counter keeps its arrivals (Wo lateness rank) but nobody waits on it
#ifndef ATT_RDY
#define ATT_RDY 1
#endif
#if ATT_RDY && !(READY && WO_LATE)
#undef ATT_RDY
#define ATT_RDY 0
#endif
#define PID_MAX 1024   // page ids per attention item (2 per 32-token step)
#ifndef ATT_AGG
#define ATT_AGG 1      // att_ready arrivals only from item CTAs (item 0 adds the item-free CTAs' share)
#endif
#ifndef ENTRY_LD
#define ENTRY_LD 1     // region-a W / X loads issued at kernel entry; the segment table is built under them
#endif
#ifndef MFMA128
#define MFMA128 1      // region a on v_mfma_f32_16x16x128_f8f6f4 (0: 16x16x32_fp8_fp8)
#endif
typedef int i32x8 __attribute__((ext_vector_type(8)));
#ifndef EPI_PF
#define EPI_PF 1       // region-b input operands loaded before the readiness wait (0: after it)
#endif
#ifndef SEG_PAR
#define SEG_PAR 1      // segment table built lane-parallel in wave 0 (0: serial build by thread 0 behind a CTA barrier)
#endif
#ifndef A_DIAG
#define A_DIAG 0       // diagnostic: stamp 12 = region-a load issue, 13 = loads landed (adds a vmcnt(0) wait)
#endif
#ifndef QKV_ABL
#define QKV_ABL 0      // diagnostic: 1 = region-a loads only (no MFMA, zero partials)
#endif
#ifndef PID_PF
#define PID_PF PSPLIT  // first item's page ids loaded into registers before region b (sliding build only: the full
                       // layer read slower with it in paired legs, F56 0.1562 -> 0.1591 ms together with ATT_AGG)
#endif
#ifndef SLOT_FENCE
#define SLOT_FENCE 1
#endif
#ifndef FAIL_TRAP
#define FAIL_TRAP 1   // 0: record the failure in ctrl and continue (diagnostics only)
#endif
#ifndef SPIN_TICKS
#define SPIN_TICKS (100ull * 1000ull * 1000ull * 600ull)   // 600 s at 100 MHz, NCCL's default timeout
#endif

// OSPLITK=2: o_proj split-K x2. Unit t = (64-col N tile t >> 1, K half t & 1): each CTA reads half of o (M x 1024)
// in 4 K stages, its Wo slice (64 rows x K 1024 = 128 KB) stays the same size, and the bf16 partial of each K half
// lands in the owner rank's reduce-scatter zone as its own source (src = rank * 2 + K half, 8 sources per 32-col
// tile); the owner reduce sums all 8 in the same RS pass (no extra cross-rank round). OSPLITK=1: 32-col x K 2048.
#ifndef OSPLITK
#define OSPLITK 2
#endif
// OSK_XCH=1 (OSPLITK=2): the two K-half CTAs of a 64-col tile exchange fp32 halves on-rank (each sums its own
// 32-col half), so the cross-rank push keeps the OSPLITK=1 bytes and 4 sources; OSK_XCH=0: 8-source owner reduce
#ifndef OSK_XCH
#define OSK_XCH 1
#endif
#if OSPLITK == 2 && !OSK_XCH
#define NSRC (NR * 2)
#else
#define NSRC NR
#endif
#define RS_OFF 0u
#define AG_OFF (NSRC * TPR * MAXTOK * 64u)
#define RSF_OFF (AG_OFF + MAXTOK * HID * 2u)
#define AGF_OFF (RSF_OFF + 4096u)
#define IPC_BYTES (AGF_OFF + NCHUNK_MAX * NTILE * 4u)

#define CTRL_GEN 0
#define CTRL_TICKET 32
#define CTRL_BAR 48
#define CTRL_ERR 64
// readiness counters (monotone, epoch scaled; one 64-B slot each)
#define CTRL_CG 256      // + cg * 16: qkv column group cg complete (8 K-eighth units per generation)
#define CTRL_EPI 1024    // + b * 16: request b epilogue complete (8 rows x 4 kv heads per generation; padded for b >= bs)
#define CTRL_ATT 1536    // + b * 16: attention partials of request b complete (one arrival per CTA per generation)
#define CTRL_XCH 2560    // + 2 nt + half: partner K-half fp32 partial of (64-col tile nt, 32-col half) landed
#define CTRL_SINK 2048    // + bid: scratch sink of the Wo warm-up loads (never read)
#define CTRL_SLAB 1280   // + h * 16: o rows of kv head h (K slab h of o_proj) complete (one arrival per CTA per generation)

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;
#define CPOL_SYS 17
// intra-GPU data published across CTAs inside one launch: device-scope write-through stores (sc1); consumers read
// after a grid barrier with plain loads (a reader's L2 never held these lines earlier in the launch, and the
// kernel-start acquire drops lines from the previous launch)
#ifndef ST_DEV
#define ST_DEV 16
#endif
#ifndef LD_DEV
#define LD_DEV 0
#endif

struct Args {
    uint8_t* peer[NR];
    uint32_t* ctrl;
    // front
    const uint8_t* xq;        // [M, 8192] fp8 e4m3
    const float* xs;          // [M]
    const uint8_t* wq;        // [4608, 8192] fp8 e4m3
    const float* ws;          // [4608]
    float* qkv_part;          // [QKV_KS][MAXTOK][4608]
    uint16_t* qbuf;           // [MAXTOK][2048] bf16 (normed / scaled / rotated q)
    uint16_t* gbuf;           // [MAXTOK][2048] bf16 gate
    uint8_t* kc;              // [slots][4][64] fp8
    uint8_t* vc;
    const int64_t* out_loc;   // [M]
    const int64_t* swa_map;   // nullable
    const int64_t* positions; // [M]
    const float* cos_sin;     // [maxpos][cs_stride]
    const int32_t* block_table;
    const int32_t* seqused;   // [bs]
    float* att_o;             // [NPART][64 q][64 d]
    float* att_ml;            // [NPART][2][64]
    // tail
    const uint16_t* x;        // o: [MAXTOK][2048] bf16 (written by region d)
    const uint16_t* w;        // [8192, 2048] bf16
    const float* stream;
    const float* alpha;
    const float* beta;
    const float* pre_w;
    uint16_t* out;
    float* new_stream;
    uint8_t* out_fp8;
    float* out_scale;
    unsigned long long* stamps;
    float post_eps;
    float pre_eps;
    float norm_eps;
    float q_scale;
    float sm_scale_log2;
    int32_t M;
    int32_t rank;
    int32_t nchunk;
    int32_t opt;
    int32_t bs;
    int32_t window;           // 0 = full attention
    int32_t rope;
    int32_t bt_stride;
    int32_t cs_stride;
    int32_t qs_min;           // minimum 32-token steps per attention piece (partial bytes vs K/V bytes per piece)
};

__device__ __forceinline__ rsrc_t make_rsrc(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, 0, 0x7fffffff, 0x00020000);
}
// K / V cache: the per-layer pool can exceed 2 GiB (serving c24: mem fraction 0.96); every K/V byte offset is a
// uint32 (slot * 256, page * 4096), so the full 4 GiB num_records range is addressable.
__device__ __forceinline__ rsrc_t make_rsrc_kv(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, 0, (int)0xffffffffu, 0x00020000);
}
__device__ __forceinline__ void st_wt(rsrc_t r, uint32_t off, uint4 v) {
    u32x4 t = {v.x, v.y, v.z, v.w};
    __builtin_amdgcn_raw_buffer_store_b128(t, r, off, 0, CPOL_SYS);
}
__device__ __forceinline__ uint4 ld_sys(rsrc_t r, uint32_t off) {
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, CPOL_SYS);
    return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ uint4 ld_pub(rsrc_t r, uint32_t off) { return ld_sys(r, off); }
// all buffer ops take a wave-uniform descriptor (a per-lane descriptor would serialise into a waterfall loop)
__device__ __forceinline__ void st_dev32(rsrc_t r, uint32_t off, uint32_t v) {
    __builtin_amdgcn_raw_buffer_store_b32(v, r, off, 0, ST_DEV);
}
__device__ __forceinline__ void st_dev16(rsrc_t r, uint32_t off, uint16_t v) {
    __builtin_amdgcn_raw_buffer_store_b16(v, r, off, 0, ST_DEV);
}
__device__ __forceinline__ void st_dev8(rsrc_t r, uint32_t off, uint8_t v) {
    __builtin_amdgcn_raw_buffer_store_b8(v, r, off, 0, ST_DEV);
}
__device__ __forceinline__ void st_dev(rsrc_t r, uint32_t off, uint4 v) {
    u32x4 t = {v.x, v.y, v.z, v.w};
    __builtin_amdgcn_raw_buffer_store_b128(t, r, off, 0, ST_DEV);
}
__device__ __forceinline__ uint4 ld_dev(rsrc_t r, uint32_t off) {
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, LD_DEV);
    return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ float bf2f(uint32_t h) { return __uint_as_float(h << 16); }
typedef __bf16 bf16x2_t __attribute__((ext_vector_type(2)));
typedef float f32x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ uint32_t pkbf(float lo, float hi) {
    const f32x2_t v = {lo, hi};
    return __builtin_bit_cast(uint32_t, __builtin_convertvector(v, bf16x2_t));
}
__device__ __forceinline__ float rbf(float f) { return bf2f(pkbf(f, 0.f) & 0xffffu); }
__device__ __forceinline__ void vm_drain() { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); }
__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ uint32_t ld_sys32(uint32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void st_sys32(uint32_t* p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint8_t f2fp8(float v) {
    return (uint8_t)(__builtin_amdgcn_cvt_pk_fp8_f32(v, v, 0, false) & 0xffu);
}

// Record the first failure and abort the launch.
__device__ __noinline__ void fail(uint32_t* ctrl, uint32_t code, uint32_t observed, uint32_t target) {
    if (atomicCAS(ctrl + CTRL_ERR, 0u, code) == 0u) {
        st_sys32(ctrl + CTRL_ERR + 1, observed);
        st_sys32(ctrl + CTRL_ERR + 2, target);
    }
#if FAIL_TRAP
    __builtin_trap();
#endif
}
__device__ __forceinline__ void spin_ge(uint32_t* p, uint32_t target, uint32_t* ctrl, uint32_t code) {
    uint64_t t0 = 0;
    uint32_t n = 0;
    while (true) {
        const uint32_t v = ld_sys32(p);
        if ((int32_t)(v - target) >= 0) break;
        if (n == 0) t0 = __builtin_amdgcn_s_memrealtime();
        if ((++n & 255u) == 0u && __builtin_amdgcn_s_memrealtime() - t0 > SPIN_TICKS) {
            fail(ctrl, code, v, target);
            break;   // only reached with FAIL_TRAP=0: record and continue
        }
        __builtin_amdgcn_s_sleep(1);
    }
    asm volatile("" ::: "memory");   // no load of the guarded data may move above the wait
}
// one arrival on a readiness counter after this CTA's write-through stores have completed
__device__ __forceinline__ void arrive(uint32_t* p, uint32_t n) {
    vm_drain();
    __syncthreads();
    if (threadIdx.x == 0) __hip_atomic_fetch_add(p, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// grid barrier k of epoch g: monotone counter, target = ((g - 1) * NBAR + k + 1) * G (mod 2^32)
__device__ __forceinline__ void grid_barrier(uint32_t* ctrl, uint32_t g, int k, int G) {
    vm_drain();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_fetch_add(ctrl + CTRL_BAR, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const uint32_t target = ((g - 1u) * NBAR + (uint32_t)k + 1u) * (uint32_t)G;
        spin_ge(ctrl + CTRL_BAR, target, ctrl, 0x400u + (uint32_t)k);
    }
    __syncthreads();
}

#define STAMP(slot) do { if (a.stamps && tid == 0) a.stamps[bid * 16 + (slot)] = __builtin_amdgcn_s_memrealtime(); } while (0)

__device__ __forceinline__ float block_sum(float v, float* sh, int tid) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    if ((tid & 63) == 0) sh[tid >> 6] = v;
    __syncthreads();
    const float s = ((sh[0] + sh[1]) + sh[2]) + sh[3];
    __syncthreads();
    return s;
}
__device__ __forceinline__ float block_max(float v, float* sh, int tid) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
    if ((tid & 63) == 0) sh[tid >> 6] = v;
    __syncthreads();
    const float s = fmaxf(fmaxf(sh[0], sh[1]), fmaxf(sh[2], sh[3]));
    __syncthreads();
    return s;
}
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}

// ---------------------------------------------------------------------------------------------- region a
// unit u (one per CTA, 256 equal units): K eighth ks = u & 7 (1024 K), column group cg = u >> 3 (QKV_CG = 144
// columns = 9 MFMA column blocks); wave w covers K [ks*1024 + w*256, +256). Fixed-order cross-wave sum in LDS,
// fp32 partial [ks][m][n] written through.
#define QKV_KS 8
#define QKV_CG 144
#define QKV_NCB 9
template <int NRB, typename Pre>
__device__ __forceinline__ void qkv_unit(const Args& a, int cg, int ks, float* red, Pre&& pre) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6, lr = lane & 15, lg = lane >> 4;
    const int bid = blockIdx.x;
    (void)bid;
    const int M = a.M;
    const int kb = ks * 1024 + w * 256 + lg * 16;
    const rsrc_t rx = make_rsrc(a.xq), rw = make_rsrc(a.wq);
    uint32_t xo[NRB], wo[QKV_NCB];
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb) xo[rb] = (uint32_t)min(rb * 16 + lr, M - 1) * KQKV + kb;
#pragma unroll
    for (int cb = 0; cb < QKV_NCB; ++cb) wo[cb] = (uint32_t)(cg * QKV_CG + cb * 16 + lr) * KQKV + kb;
    f32x4 acc[NRB][QKV_NCB];
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
        for (int cb = 0; cb < QKV_NCB; ++cb) acc[rb][cb] = (f32x4){0.f, 0.f, 0.f, 0.f};
    // all 4 K steps of this wave in flight before the first MFMA (W streamed from HBM once per launch)
    uint4 X[4][NRB], W[4][QKV_NCB];
    if (A_DIAG) STAMP(12);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int cb = 0; cb < QKV_NCB; ++cb) W[p][cb] = ld_dev(rw, wo[cb] + p * 64);
#pragma unroll
        for (int rb = 0; rb < NRB; ++rb) X[p][rb] = ld_dev(rx, xo[rb] + p * 64);
    }
    pre();   // work that needs no region-a data runs under the loads in flight
    if (A_DIAG) { vm_drain(); STAMP(13); }
#if QKV_ABL
    {
        uint32_t s = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
#pragma unroll
            for (int cb = 0; cb < QKV_NCB; ++cb) s ^= W[p][cb].x ^ W[p][cb].w;
#pragma unroll
            for (int rb = 0; rb < NRB; ++rb) s ^= X[p][rb].y;
        }
        acc[0][0][0] = s == 0x12345678u ? 1.f : 0.f;
    }
#elif MFMA128
    // gfx950 16x16x128 FP8 MFMA: lane bytes of K steps 2q and 2q+1 (the same lane -> K permutation for x and W, so the
    // dot product is unchanged), a quarter of the 16x16x32 instruction count
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
            for (int cb = 0; cb < QKV_NCB; ++cb) {
                const uint4 x0 = X[2 * q][rb], x1 = X[2 * q + 1][rb], w0 = W[2 * q][cb], w1 = W[2 * q + 1][cb];
                const i32x8 xa = {(int)x0.x, (int)x0.y, (int)x0.z, (int)x0.w, (int)x1.x, (int)x1.y, (int)x1.z, (int)x1.w};
                const i32x8 wb = {(int)w0.x, (int)w0.y, (int)w0.z, (int)w0.w, (int)w1.x, (int)w1.y, (int)w1.z, (int)w1.w};
                acc[rb][cb] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(xa, wb, acc[rb][cb], 0, 0, 0, 0, 0, 0);
            }
#else
#pragma unroll
    for (int p = 0; p < 4; ++p)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf)
#pragma unroll
            for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
                for (int cb = 0; cb < QKV_NCB; ++cb) {
                    const long xa = hf ? (long)(((uint64_t)X[p][rb].w << 32) | X[p][rb].z) : (long)(((uint64_t)X[p][rb].y << 32) | X[p][rb].x);
                    const long wb = hf ? (long)(((uint64_t)W[p][cb].w << 32) | W[p][cb].z) : (long)(((uint64_t)W[p][cb].y << 32) | W[p][cb].x);
                    acc[rb][cb] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(xa, wb, acc[rb][cb], 0, 0, 0);
                }
#endif
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
        for (int cb = 0; cb < QKV_NCB; ++cb)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[(w * 64 + rb * 16 + lg * 4 + i) * QKV_CG + cb * 16 + lr] = acc[rb][cb][i];
    lds_barrier();
    const rsrc_t rp = make_rsrc(a.qkv_part);
    for (int idx = tid; idx < M * (QKV_CG / 4); idx += NTHR) {
        const int row = idx / (QKV_CG / 4), c4 = (idx - row * (QKV_CG / 4)) * 4;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
            v[e] = ((red[(0 * 64 + row) * QKV_CG + c4 + e] + red[(1 * 64 + row) * QKV_CG + c4 + e]) +
                    red[(2 * 64 + row) * QKV_CG + c4 + e]) + red[(3 * 64 + row) * QKV_CG + c4 + e];
        st_dev(rp, (((uint32_t)ks * MAXTOK + row) * NQKV + cg * QKV_CG + c4) * 4,
               make_uint4(__float_as_uint(v[0]), __float_as_uint(v[1]), __float_as_uint(v[2]), __float_as_uint(v[3])));
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------- region b
// all loads of the unit (row scalars, RoPE coefficients, the 8 K partials of every vector, w_scale) are issued
// before any math; fp8 / bf16 results are packed across lanes so every store is >= 4 bytes
// the unit's input-only operands (row scale, cache slot, RoPE coefficients, w_scale): no dependence on region a, so
// EPI_PF=1 loads them before the column-group readiness wait (the slot -> swa_mapping and position -> cos / sin chains
// leave the post-wait path)
constexpr int EPI_NV = 5;   // vectors per wave: vi = w + 4 i < 18
struct EpiIn {
    float xsm, c, sn, wsv[EPI_NV];
    int64_t loc;
};
__device__ __forceinline__ int epi_col(int vi, int h) {
    return vi < 8 ? (h * 8 + vi) * 64 : vi == 8 ? 2048 + h * 64 : vi == 9 ? 2304 + h * 64 : 2560 + (h * 8 + vi - 10) * 64;
}
__device__ __forceinline__ EpiIn epi_inputs(const Args& a, int m, int h) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    EpiIn e;
    e.xsm = a.xs[m];
    e.loc = a.out_loc[m];
    e.c = 1.f;
    e.sn = 0.f;
    if (a.rope) {
        const int64_t pos = a.positions[m];
        e.c = rbf(a.cos_sin[pos * a.cs_stride + (lane >> 1)]);
        e.sn = rbf(a.cos_sin[pos * a.cs_stride + 32 + (lane >> 1)]);
    }
#pragma unroll
    for (int i = 0; i < EPI_NV; ++i) {
        const int vi = w + 4 * i;
        e.wsv[i] = vi < 18 ? a.ws[epi_col(vi, h) + lane] : 0.f;
    }
    if (e.loc >= 0 && a.swa_map) e.loc = a.swa_map[e.loc];
    return e;
}
__device__ __forceinline__ void qk_epilogue(const Args& a, int m, int h, const EpiIn& e) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const float xsm = e.xsm, c = e.c, sn = e.sn;
    const int64_t loc = e.loc;
    constexpr int NV = EPI_NV;
    float pv[NV][QKV_KS], wsv[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int vi = w + 4 * i;
        wsv[i] = e.wsv[i];
        if (vi < 18) {
            const int n = epi_col(vi, h) + lane;
#pragma unroll
            for (int k = 0; k < QKV_KS; ++k)
                pv[i][k] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(
                    make_rsrc(a.qkv_part), (((uint32_t)k * MAXTOK + m) * NQKV + n) * 4, 0, READY ? CPOL_SYS : 0));
        }
    }
    const rsrc_t rq = make_rsrc(a.qbuf), rg = make_rsrc(a.gbuf), rk = make_rsrc_kv(a.kc), rv = make_rsrc_kv(a.vc);
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        const int vi = w + 4 * i;
        if (vi >= 18) break;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < QKV_KS; ++k) s += pv[i][k];
        float val = rbf(s * xsm * wsv[i]);
        if (vi < 9) {
            const float ss = wave_sum(val * val);
            const float inv = rsqrtf(ss / 64.f + a.norm_eps);
            val = rbf(val * inv);
            if (vi < 8) val = rbf(val * a.q_scale);
            if (a.rope) {
                const float other = __shfl_xor(val, 1, 64);
                const float rot = (lane & 1) ? other : -other;
                val = rbf(rbf(val * c) + rbf(rot * sn));
            }
        }
        if (vi < 8 || vi >= 10) {   // bf16 pair per even lane
            const float hi = __shfl_xor(val, 1, 64);
            if ((lane & 1) == 0) {
                const uint32_t off = vi < 8 ? ((uint32_t)m * 2048 + (h * 8 + vi) * 64 + lane) * 2
                                            : ((uint32_t)m * 2048 + (h * 8 + vi - 10) * 64 + lane) * 2;
                st_dev32(vi < 8 ? rq : rg, off, pkbf(val, hi));
            }
        } else if (loc >= 0) {      // K (vi 8) / V (vi 9): fp8 quad per lane % 4 == 0
            const uint32_t b = (uint32_t)f2fp8(val);
            const uint32_t q = b | ((uint32_t)__shfl_down(b, 1, 64) << 8) | ((uint32_t)__shfl_down(b, 2, 64) << 16) |
                               ((uint32_t)__shfl_down(b, 3, 64) << 24);
            if ((lane & 3) == 0) st_dev32(vi == 8 ? rk : rv, (uint32_t)loc * 256 + h * 64 + lane, q);
        }
    }
}

// ---------------------------------------------------------------------------------------------- region c
__device__ __forceinline__ uint32_t pk_fp8lo_bf16(uint32_t wd) {   // bytes 0,1 -> 2 bf16
    const f32x2_t f = __builtin_amdgcn_cvt_pk_f32_fp8((int)wd, false);
    return pkbf(f[0], f[1]);
}
__device__ __forceinline__ uint32_t pk_fp8hi_bf16(uint32_t wd) {   // bytes 2,3 -> 2 bf16
    const f32x2_t f = __builtin_amdgcn_cvt_pk_f32_fp8((int)wd, true);
    return pkbf(f[0], f[1]);
}
template <bool HI>
__device__ __forceinline__ uint32_t cvt2(uint32_t wd) {   // fp8 bytes {0,1} (HI: {2,3}) -> 2 bf16 (exact)
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8((int)wd, 1.0f, HI));
}
__device__ __forceinline__ bf16x8 fp8x8_bf16(uint32_t w0, uint32_t w1) {
    uint4 r = make_uint4(cvt2<false>(w0), cvt2<true>(w0), cvt2<false>(w1), cvt2<true>(w1));
    return __builtin_bit_cast(bf16x8, r);
}
__device__ __forceinline__ size_t kv_off(const int32_t* bt, int pos, int h) {
    const int page = bt[pos >> 4];
    return ((size_t)page * 16 + (pos & 15)) * 256 + h * 64;
}

// one item: request b, kv head h, steps [st0, st1) of 32 tokens from token lo; partial -> slot
// sp: LDS copy of the item's page ids, sp[r] = block_table[b][min((lo >> 4) + 2 * st0 + r, last page)] for
// r < 2 * (st1 - st0), staged by the caller before the epilogue-readiness wait (producer independent), so no page-id
// load sits on the K/V critical path or shares the in-order vmcnt with the K/V stream
// rdy / rdy_target / rdy_code: epilogue-readiness counter of request b (nullptr: no wait)
__device__ void attn_item(const Args& a, int b, int h, int lo, int L, int st0, int st1, int slot, float* lds,
                          const int* sp, uint32_t* rdy, uint32_t rdy_target, uint32_t rdy_code) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6, lr = lane & 15, lg = lane >> 4;
    const int s0 = L - 8;                       // position of the first verify token
    const int win = a.window;
    const int32_t* bt = a.block_table + (size_t)b * a.bt_stride;
    f32x4 O[4][4];
#pragma unroll
    for (int db = 0; db < 4; ++db)
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) O[db][qb] = (f32x4){0.f, 0.f, 0.f, 0.f};
    float mrow[4], lrow[4];
#pragma unroll
    for (int qb = 0; qb < 4; ++qb) { mrow[qb] = STATIC_MAX ? 0.f : -INFINITY; lrow[qb] = 0.f; }
#if LSUM_MFMA
    f32x4 Lacc[4];
#pragma unroll
    for (int qb = 0; qb < 4; ++qb) Lacc[qb] = (f32x4){0.f, 0.f, 0.f, 0.f};
    const bf16x8 ones = __builtin_bit_cast(bf16x8, make_uint4(0x3F803F80u, 0x3F803F80u, 0x3F803F80u, 0x3F803F80u));
#endif
    const float scl = a.sm_scale_log2;
    // Steps are page aligned (lo % 16 == 0): step st covers the two 16-token pages (lo + 32 st) / 16 + {0, 1}, so
    // its page ids are two wave-uniform values (clamped to the request's last page; tokens >= L are masked and
    // their V words zeroed; ids are loaded as uniform vector values). Software pipeline per wave: page ids two rings ahead, K/V one ring ahead.
    // lane tokens: K rows (lr>>2)*8 + tb*4 + (lr&3) (tb = 0, 1; page lr >> 3); V tokens lg*8 + j (page lg >> 1)
    const int lastp = (L - 1) >> 4;
    const int vlo = win ? max(0, L - 8 - win + 1) : 0;   // first token of the window (lo is page aligned below it)
    const bool ksel = (lr >> 3) != 0, vsel = (lg >> 1) != 0;
    uint32_t kofs[2];
#pragma unroll
    for (int tb = 0; tb < 2; ++tb) kofs[tb] = (uint32_t)((((lr >> 2) * 8 + (lr & 3) + tb * 4) & 15) * 256 + h * 64 + lg * 16);
    const uint32_t vofs = (uint32_t)(((lg & 1) * 8) * 256 + h * 64 + lr * 4);
    auto pages = [&](int st, int (&pg)[2]) {
        const int r = __builtin_amdgcn_readfirstlane(2 * (st - st0));
        pg[0] = sp[r];
        pg[1] = sp[r + 1];
    };
    const rsrc_t rk = make_rsrc_kv(a.kc), rv = make_rsrc_kv(a.vc);
    auto kvload = [&](const int (&pg)[2], uint4 (&Kr)[2], uint32_t (&Vw)[8]) {
        const uint32_t kp = (uint32_t)(ksel ? pg[1] : pg[0]) * 4096u, vp = (uint32_t)(vsel ? pg[1] : pg[0]) * 4096u;
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) Kr[tb] = ld_dev(rk, kp + kofs[tb]);
#pragma unroll
        for (int j = 0; j < 8; ++j) Vw[j] = __builtin_amdgcn_raw_buffer_load_b32(rv, vp + vofs + j * 256, 0, LD_DEV);
    };
    // NPF steps of this wave in flight: slot i holds K/V of step st + 4 i and the page ids two rings ahead
    int pg[NPF][2];
    uint4 Kr[NPF][2];
    uint32_t Vw[NPF][8];
    // Every slot load is issued unconditionally (step index clamped to the piece's last step, whose lines are then
    // L2 hits): a conditional reload makes the loop-carried slot registers a phi of {old value, in-flight load}, and
    // the copy that resolves it at the back edge forces s_waitcnt vmcnt(0), i.e. a full drain of the K/V stream once
    // per NPF steps.
    const int stl = st1 - 1;
    auto prologue = [&]() {
#pragma unroll
    for (int i = 0; i < NPF; ++i) {
        const int sti = st0 + i;
        pages(min(sti, stl), pg[i]);
        kvload(pg[i], Kr[i], Vw[i]);
#if SLOT_FENCE
        // keep the prologue's issue order equal to the loop's (slot 0 K, V, ids, slot 1 ...): the vmcnt scoreboard is
        // merged at the loop header, so a reordered prologue would force vmcnt(0) on every loop iteration
        asm volatile("" ::: "memory");
#endif
    }
    };
    // History K/V (every prologue step ends before the 8 verify tokens, whose K/V the epilogue writes in this
    // launch) is producer independent: its prologue is issued before the epilogue-readiness wait. Otherwise the
    // prologue follows the wait. Q (written by the epilogue) is always loaded after the wait, behind the prologue,
    // so the first K/V latency is not serialised behind the Q load + scale.
    const bool early = rdy != nullptr && lo + min(st0 + NPF, st1) * 32 <= L - 8;
    if (early) prologue();
    if (rdy != nullptr) {
        if (tid == 0) spin_ge(rdy, rdy_target, a.ctrl, rdy_code);
        __syncthreads();
    }
    if (a.stamps && tid == 0) a.stamps[blockIdx.x * 16 + 15] = __builtin_amdgcn_s_memrealtime();
    if (!early) prologue();
    const rsrc_t rq = make_rsrc(a.qbuf);
    uint4 Q[4][2];
#pragma unroll
    for (int qb = 0; qb < 4; ++qb) {
        const int qc = qb * 16 + lr, tq = qc >> 3, hh = qc & 7;
        const uint32_t off = (((uint32_t)(b * 8 + tq) * 2048) + (h * 8 + hh) * 64 + lg * 16) * 2;
        Q[qb][0] = ld_dev(rq, off);
        Q[qb][1] = ld_dev(rq, off + 16);
    }
#if QSCL
    // log2 softmax scale folded into the Q operand once per item (bf16 re-round, relative 2^-9), so the step loop
    // feeds S straight to exp2 (one v_mul per score less: the step body is VALU bound at 1 wave per SIMD)
#pragma unroll
    for (int qb = 0; qb < 4; ++qb)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            uint4& v = Q[qb][hf];
            const float sc = a.sm_scale_log2;
            v = make_uint4(pkbf(bf2f(v.x & 0xffffu) * sc, bf2f(v.x >> 16) * sc), pkbf(bf2f(v.y & 0xffffu) * sc, bf2f(v.y >> 16) * sc),
                           pkbf(bf2f(v.z & 0xffffu) * sc, bf2f(v.z >> 16) * sc), pkbf(bf2f(v.w & 0xffffu) * sc, bf2f(v.w >> 16) * sc));
        }
#define SCL(x) (x)
#else
#define SCL(x) ((x) * scl)
#endif
#if QLDS
    // Q B-fragments parked in wave-private LDS (lane-contiguous 16-B slots) to keep the loop's VGPR set small
    uint4* qL = (uint4*)(lds + QLDS_OFF) + w * 512;
#pragma unroll
    for (int f = 0; f < 8; ++f) qL[f * 64 + lane] = Q[f >> 1][f & 1];
#define QF(qb, hf) qL[((qb) * 2 + (hf)) * 64 + lane]
#else
#define QF(qb, hf) Q[qb][hf]
#endif
#if ATT_DIAG
    if (a.stamps) { vm_drain(); if (tid == 0) a.stamps[blockIdx.x * 16 + 12] = __builtin_amdgcn_s_memrealtime(); }
#endif
    for (int stb = st0; stb < st1; stb += NPF) {
#pragma unroll
    for (int i = 0; i < NPF; ++i) {
        const int st = stb + i;
        const int tok0 = lo + st * 32;
        uint32_t Vc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) Vc[j] = Vw[i][j];
        if (tok0 + 32 > L || tok0 < vlo) {   // V rows past L or before the window may hold stale (NaN) FP8 bytes
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int t = tok0 + lg * 8 + j;
                Vc[j] = (t < L && t >= vlo) ? Vc[j] : 0u;
            }
        }
        // slot i is converted to bf16 MFMA operands BEFORE its reload is issued, and the fence below keeps the
        // reload after the conversion: the raw slot registers die first, so the reload lands in the same registers
        // and the loop-carried slot needs no back-edge copy (a copy of an in-flight load waits on vmcnt)
        u32x4 kb[2][2];
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
            kb[tb][0] = __builtin_bit_cast(u32x4, fp8x8_bf16(Kr[i][tb].x, Kr[i][tb].y));
            kb[tb][1] = __builtin_bit_cast(u32x4, fp8x8_bf16(Kr[i][tb].z, Kr[i][tb].w));
        }
        // V^T operand of d-block db: byte db of the 8 token words -> 8 bf16 (k slot j = token lg*8 + j)
        uint32_t vp[4][4];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            // words j = 2jj, 2jj+1 -> bytes {db of w0, db of w1} for db = 0..3 via byte permutes
            const uint32_t w0 = Vc[2 * jj], w1 = Vc[2 * jj + 1];
            const uint32_t x02 = __builtin_amdgcn_perm(w1, w0, 0x06020400u);   // {w0.b0, w1.b0, w0.b2, w1.b2}
            const uint32_t x13 = __builtin_amdgcn_perm(w1, w0, 0x07030501u);   // {w0.b1, w1.b1, w0.b3, w1.b3}
            vp[0][jj] = cvt2<false>(x02);
            vp[2][jj] = cvt2<true>(x02);
            vp[1][jj] = cvt2<false>(x13);
            vp[3][jj] = cvt2<true>(x13);
        }
#if SLOT_FENCE
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) asm volatile("" : "+v"(kb[tb][0]), "+v"(kb[tb][1]) :: "memory");
#pragma unroll
        for (int db = 0; db < 4; ++db)
            asm volatile("" : "+v"(vp[db][0]), "+v"(vp[db][1]), "+v"(vp[db][2]), "+v"(vp[db][3]) :: "memory");
#endif
        pages(min(st + NPF, stl), pg[i]);
        kvload(pg[i], Kr[i], Vw[i]);
#if SLOT_FENCE
        asm volatile("" ::: "memory");
#endif
        if (st < st1) {
        f32x4 S[2][4];
#pragma unroll
        for (int tb = 0; tb < 2; ++tb) {
            const bf16x8 k0 = __builtin_bit_cast(bf16x8, kb[tb][0]), k1 = __builtin_bit_cast(bf16x8, kb[tb][1]);
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) {
                f32x4 c = (f32x4){0.f, 0.f, 0.f, 0.f};
                c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(k0, __builtin_bit_cast(bf16x8, QF(qb, 0)), c, 0, 0, 0);
                c = __builtin_amdgcn_mfma_f32_16x16x32_bf16(k1, __builtin_bit_cast(bf16x8, QF(qb, 1)), c, 0, 0, 0);
#if PIN_REGS
                asm volatile("" : "+v"(c));   // S straight into VGPRs (consumed by VALU)
#endif
                S[tb][qb] = c;
            }
        }
        uint4 P[4];
        const bool nomask = (tok0 + 32 <= s0) && (win == 0 || tok0 >= s0 + 8 - win);   // wave-uniform
#if STATIC_MAX
        // static softmax reference: q and k are RMS-normalised per head, so |s * scale| <= |q|_2 |k|_2 scale is
        // bounded on the host (MAX_LOG2_SCORE); p = exp2(s * scale) then stays inside fp32 / bf16 range and no
        // running max, cross-lane reduction or O rescale is needed (bf16 P has the fp32 exponent range)
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            float pv[8];
            float ps = 0.f;
            if (nomask) {
#pragma unroll
                for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                    for (int i = 0; i < 4; ++i) pv[tb * 4 + i] = __builtin_amdgcn_exp2f(SCL(S[tb][qb][i]));
            } else {
                const int qpos = s0 + ((qb * 16 + lr) >> 3);
#pragma unroll
                for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int pos = tok0 + lg * 8 + tb * 4 + i;
                        const bool ok = pos < L && pos <= qpos && (win == 0 || pos > qpos - win);
                        pv[tb * 4 + i] = ok ? __builtin_amdgcn_exp2f(SCL(S[tb][qb][i])) : 0.f;
                    }
            }
#if !LSUM_MFMA
#pragma unroll
            for (int j = 0; j < 8; ++j) ps += pv[j];
            lrow[qb] += ps;
#else
            (void)ps;
#endif
            P[qb] = make_uint4(pkbf(pv[0], pv[1]), pkbf(pv[2], pv[3]), pkbf(pv[4], pv[5]), pkbf(pv[6], pv[7]));
#if LSUM_MFMA
            // denominator on the matrix pipe: ones^T P^T gives the column sums of the bf16 P actually multiplied
            // with V, replicated over the 16 output rows (no per-score v_add, no cross-lane reduction)
            Lacc[qb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ones, __builtin_bit_cast(bf16x8, P[qb]), Lacc[qb], 0, 0, 0);
#endif
        }
#else
        float sv[4][8], mx[4];
        if (nomask) {
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) {
#pragma unroll
                for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                    for (int i = 0; i < 4; ++i) sv[qb][tb * 4 + i] = S[tb][qb][i];
                mx[qb] = fmaxf(fmaxf(fmaxf(sv[qb][0], sv[qb][1]), fmaxf(sv[qb][2], sv[qb][3])),
                               fmaxf(fmaxf(sv[qb][4], sv[qb][5]), fmaxf(sv[qb][6], sv[qb][7]))) * scl;
            }
        } else {
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) {
                const int qpos = s0 + ((qb * 16 + lr) >> 3);
                mx[qb] = -INFINITY;
#pragma unroll
                for (int tb = 0; tb < 2; ++tb)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int pos = tok0 + lg * 8 + tb * 4 + i;
                        const bool ok = pos < L && pos <= qpos && (win == 0 || pos > qpos - win);
                        sv[qb][tb * 4 + i] = ok ? S[tb][qb][i] : -INFINITY;
                        mx[qb] = fmaxf(mx[qb], ok ? S[tb][qb][i] * scl : -INFINITY);
                    }
            }
        }
        // lazy rescale, one vote for all four column blocks: the column max (cross-lane) is formed only when some
        // lane's local max exceeds the running max by more than 8 (log2 units), so p <= 2^8 between rescales
        const bool grow = (mx[0] > mrow[0] + 8.0f) || (mx[1] > mrow[1] + 8.0f) || (mx[2] > mrow[2] + 8.0f) ||
                          (mx[3] > mrow[3] + 8.0f);
        if (__builtin_expect(__any(grow), 0)) {
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) {
                float m2 = fmaxf(mx[qb], __shfl_xor(mx[qb], 16, 64));
                m2 = fmaxf(m2, __shfl_xor(m2, 32, 64));
                const float mn = fmaxf(mrow[qb], m2);
                const float al = __builtin_amdgcn_exp2f(mrow[qb] - (mn == -INFINITY ? 0.f : mn));
                lrow[qb] *= al;
#pragma unroll
                for (int db = 0; db < 4; ++db) O[db][qb] *= al;
                mrow[qb] = mn;
            }
        }
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            const float mu = mrow[qb] == -INFINITY ? 0.f : mrow[qb];
            float ps = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) { sv[qb][j] = __builtin_amdgcn_exp2f(__builtin_fmaf(sv[qb][j], scl, -mu)); ps += sv[qb][j]; }
            lrow[qb] += ps;
            P[qb] = make_uint4(pkbf(sv[qb][0], sv[qb][1]), pkbf(sv[qb][2], sv[qb][3]), pkbf(sv[qb][4], sv[qb][5]), pkbf(sv[qb][6], sv[qb][7]));
        }
#endif
#pragma unroll
        for (int db = 0; db < 4; ++db) {
            const uint4 vr = make_uint4(vp[db][0], vp[db][1], vp[db][2], vp[db][3]);
            const bf16x8 va = __builtin_bit_cast(bf16x8, vr);
#pragma unroll
            for (int qb = 0; qb < 4; ++qb)
                O[db][qb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(va, __builtin_bit_cast(bf16x8, P[qb]), O[db][qb], 0, 0, 0);
        }
#if PIN_REGS
#pragma unroll
        for (int db = 0; db < 4; ++db)
#pragma unroll
            for (int qb = 0; qb < 4; ++qb) asm volatile("" : "+a"(O[db][qb]));   // O stays in AGPRs across steps
#endif
        }   // st < st1
    }
    }
    vm_drain();   // trailing (clamped) slot reloads land before the partial stores reuse the counters
#if ATT_DIAG
    if (a.stamps && tid == 0) a.stamps[blockIdx.x * 16 + 13] = __builtin_amdgcn_s_memrealtime();
#endif
#if LSUM_MFMA
#pragma unroll
    for (int qb = 0; qb < 4; ++qb) lrow[qb] = Lacc[qb][0];
#else
#pragma unroll
    for (int qb = 0; qb < 4; ++qb) {
        lrow[qb] += __shfl_xor(lrow[qb], 16, 64);
        lrow[qb] += __shfl_xor(lrow[qb], 32, 64);
    }
#endif
    // wave w owns kv head h of this piece: its (o, max, denom) partial goes to slot (piece * 4 + h) through a
    // wave-private LDS transpose (row-contiguous 16-B stores)
    float* oL = lds + w * 4096;      // [64 q][64 d]
#pragma unroll
    for (int db = 0; db < 4; ++db)
#pragma unroll
        for (int qb = 0; qb < 4; ++qb)
#pragma unroll
            for (int i = 0; i < 4; ++i) oL[(qb * 16 + lr) * 64 + (lg * 4 + i) * 4 + db] = O[db][qb][i];
    const rsrc_t ro = make_rsrc(a.att_o), rml = make_rsrc(a.att_ml);
    if (lg == 0) {
#pragma unroll
        for (int qb = 0; qb < 4; ++qb) {
            st_dev32(rml, ((uint32_t)slot * 128 + qb * 16 + lr) * 4, __float_as_uint(mrow[qb]));
            st_dev32(rml, ((uint32_t)slot * 128 + 64 + qb * 16 + lr) * 4, __float_as_uint(lrow[qb]));
        }
    }
    __syncthreads();
#if PBF16
    // partial o rows in bf16 (max / denom stay fp32): half the partial bytes written here and read by the merge
#pragma unroll 4
    for (int it = 0; it < 8; ++it) {
        const int c = it * 64 + lane, q = c >> 3, d8 = (c & 7) * 8;
        const float4 v0 = *(const float4*)(oL + q * 64 + d8), v1 = *(const float4*)(oL + q * 64 + d8 + 4);
        st_dev(ro, (((uint32_t)slot * 64 + q) * 64 + d8) * 2,
               make_uint4(pkbf(v0.x, v0.y), pkbf(v0.z, v0.w), pkbf(v1.x, v1.y), pkbf(v1.z, v1.w)));
    }
#else
#pragma unroll 4
    for (int it = 0; it < 16; ++it) {
        const int c = it * 64 + lane, q = c >> 4, d4 = (c & 15) * 4;
        const float4 v = *(const float4*)(oL + q * 64 + d4);
        st_dev(ro, (((uint32_t)slot * 64 + q) * 64 + d4) * 4,
               make_uint4(__float_as_uint(v.x), __float_as_uint(v.y), __float_as_uint(v.z), __float_as_uint(v.w)));
    }
#endif
    __syncthreads();
}

// ---------------------------------------------------------------------------------------------- region d
#if PBF16
#define PLD(i) bf2f(((const uint16_t*)a.att_o)[i])
#else
#define PLD(i) a.att_o[i]
#endif
// unit (request b, kv head h, verify tokens tq0 .. tq0 + TQU - 1): wave w merges q rows hh = w and w + 4 of every
// token together (2 * TQU rows, lanes over the 64 head dims), 16 pieces per round with all loads in flight
template <int TQU>
__device__ __forceinline__ void merge_norm_gate(const Args& a, int b, int h, int tq0, int poff, int nch) {
    constexpr int R = 2 * TQU;
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    int q[R];
    float gv[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int tq = tq0 + (r >> 1), hh = w + 4 * (r & 1);
        q[r] = tq * 8 + hh;
        gv[r] = bf2f(a.gbuf[(size_t)(b * 8 + tq) * 2048 + (h * 8 + hh) * 64 + lane]);
    }
    // global max over pieces: identically 0 under the static softmax reference (every partial has max 0)
    float gm[R];
#pragma unroll
    for (int r = 0; r < R; ++r) gm[r] = 0.f;
#if !STATIC_MAX
#pragma unroll
    for (int r = 0; r < R; ++r) gm[r] = -INFINITY;
    for (int c = lane; c < nch; c += 64) {
        const size_t p = (size_t)((poff + c) * NKH + h) * 128;
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (a.att_ml[p + 64 + q[r]] > 0.f) gm[r] = fmaxf(gm[r], a.att_ml[p + q[r]]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int o = 32; o > 0; o >>= 1) gm[r] = fmaxf(gm[r], __shfl_xor(gm[r], o, 64));
        if (gm[r] == -INFINITY) gm[r] = 0.f;
    }
#endif
    // rounds of 64 pieces: (max, denom) one per lane, partial rows 16 per batch in flight
    float tot[R], acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) tot[r] = acc[r] = 0.f;
    for (int c0 = 0; c0 < nch; c0 += 64) {
        float wl[R], mm[R], ll[R];
#pragma unroll
        for (int r = 0; r < R; ++r) { wl[r] = 0.f; mm[r] = -INFINITY; ll[r] = 0.f; }
        if (c0 + lane < nch) {
            const size_t p = (size_t)((poff + c0 + lane) * NKH + h) * 128;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                ll[r] = a.att_ml[p + 64 + q[r]];
                mm[r] = a.att_ml[p + q[r]];
            }
        }
        const int nc = min(64, nch - c0);
        float v[R][MBAT];
        auto ldv = [&](int c) {
#pragma unroll
            for (int e = 0; e < MBAT; ++e) {
                const int cc = min(c + e, nc - 1);
                const size_t base = (size_t)((poff + c0 + cc) * NKH + h) * 64;
#pragma unroll
                for (int r = 0; r < R; ++r) v[r][e] = PLD((base + q[r]) * 64 + lane);
            }
        };
        ldv(0);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wl[r] = ll[r] > 0.f ? exp2f(mm[r] - gm[r]) : 0.f;
            tot[r] += ll[r] * wl[r];
        }
        for (int c = 0; c < nc; c += MBAT) {
            float vc[R][MBAT];
#pragma unroll
            for (int e = 0; e < MBAT; ++e)
#pragma unroll
                for (int r = 0; r < R; ++r) vc[r][e] = v[r][e];
            if (c + MBAT < nc) ldv(c + MBAT);
#pragma unroll
            for (int e = 0; e < MBAT; ++e)
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float wv = __shfl(wl[r], c + e, 64);
                    acc[r] += (c + e < nc) ? vc[r][e] * wv : 0.f;
                }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float t = wave_sum(tot[r]);
        const float inv = t > 0.f ? 1.f / t : 0.f;
        float x = rbf(acc[r] * inv);
        const float rstd = 1.f / sqrtf(wave_sum(x * x) / 64.f + a.norm_eps);
        x = x * rstd;
        const int m = b * 8 + tq0 + (r >> 1);
        const int col = (h * 8 + w + 4 * (r & 1)) * 64 + lane;
        const float y = x * (1.f / (1.f + expf(-gv[r])));
        const float y1 = __shfl_xor(y, 1, 64);   // bf16 pair per even lane: 4-B write-through stores
        if ((lane & 1) == 0) st_dev32(make_rsrc(a.x), ((uint32_t)m * 2048 + col) * 2, pkbf(y, y1));
    }
}

// small batch (bs <= 2: up to ~GRID / bs pieces per request, only bs * 32 row-group units): unit = one q row of
// (request b, kv head h); the 4 waves split the pieces, fixed-order cross-wave sum in LDS, wave 0 finishes the row
__device__ __forceinline__ void merge_row(const Args& a, int b, int h, int qi, int poff, int nch, float* sh) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6;
    const int tq = qi >> 3, hh = qi & 7, m = b * 8 + tq;
    const int CH = (nch + 3) >> 2, cb = w * CH, ce = min(nch, cb + CH);
    float gm = 0.f;
#if !STATIC_MAX
    gm = -INFINITY;
    for (int c = lane; c < nch; c += 64) {
        const size_t p = (size_t)((poff + c) * NKH + h) * 128;
        if (a.att_ml[p + 64 + qi] > 0.f) gm = fmaxf(gm, a.att_ml[p + qi]);
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) gm = fmaxf(gm, __shfl_xor(gm, o, 64));
    if (gm == -INFINITY) gm = 0.f;
#endif
    float tot = 0.f, acc = 0.f;
    for (int c0 = cb; c0 < ce; c0 += 64) {
        float wl = 0.f;
        const int nc = min(64, ce - c0);
        float v[16];
        auto ldv = [&](int c) {
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const int cc = min(c + e, nc - 1);
                v[e] = PLD(((size_t)((poff + c0 + cc) * NKH + h) * 64 + qi) * 64 + lane);
            }
        };
        ldv(0);
        if (lane < nc) {
            const size_t p = (size_t)((poff + c0 + lane) * NKH + h) * 128;
            const float l = a.att_ml[p + 64 + qi];
            wl = l > 0.f ? exp2f(a.att_ml[p + qi] - gm) : 0.f;
            tot += l * wl;
        }
        for (int c = 0; c < nc; c += 16) {
            float vc[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) vc[e] = v[e];
            if (c + 16 < nc) ldv(c + 16);
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const float wv = __shfl(wl, c + e, 64);
                acc += (c + e < nc) ? vc[e] * wv : 0.f;
            }
        }
    }
    tot = wave_sum(tot);
    sh[w * 64 + lane] = acc;
    if (lane == 0) sh[256 + w] = tot;
    __syncthreads();
    if (w == 0) {
        const float t = ((sh[256] + sh[257]) + sh[258]) + sh[259];
        const float s4 = ((sh[lane] + sh[64 + lane]) + sh[128 + lane]) + sh[192 + lane];
        const float inv = t > 0.f ? 1.f / t : 0.f;
        float x = rbf(s4 * inv);
        const float rstd = 1.f / sqrtf(wave_sum(x * x) / 64.f + a.norm_eps);
        x = x * rstd;
        const int col = (h * 8 + hh) * 64 + lane;
        const float gv = bf2f(a.gbuf[(size_t)m * 2048 + col]);
        const float y = x * (1.f / (1.f + expf(-gv)));
        const float y1 = __shfl_xor(y, 1, 64);
        if ((lane & 1) == 0) st_dev32(make_rsrc(a.x), ((uint32_t)m * 2048 + col) * 2, pkbf(y, y1));
    }
    __syncthreads();
}

#define PIPE 1
// Software-pipelined o_proj tile (PIPE): 8 K stages of 256 (production's BK), loads issued PD stages ahead in
// K order (x and W of a stage together, as the Triton row-major GEMM does), so the x (L2) and W (HBM) streams
// overlap. x stage: full-line loads (32 lanes per 512-B row) -> registers -> padded LDS double buffer ->
// MFMA fragments; W stage: wave w's 64-K slice, fragments straight into registers. All loads are plain
// global loads, so the compiler's vmcnt tracking stays exact.
#ifndef PD
#define PD 2                 // stages in flight (x + W of a stage are issued together)
#endif
#define PROW 528             // LDS row of one x stage: 256 bf16 + 16 B pad
#ifndef WLDS
#define WLDS 1               // 1: W stage loaded full-line (32 lanes per 512-B row) and re-fragmented through LDS
#endif
// The whole Wo slice of tile t (32 rows x K 2048 bf16 = 128 KB) is producer independent: it is issued into
// registers (8 stages x 4 full-line 16-B chunks per thread) right after this CTA's attention work, so the HBM
// stream overlaps barrier 2, the merge and the merge-readiness wait instead of sitting on the o_proj critical path.
#if !WLDS
#error "Wo prefetch needs the full-line W layout (WLDS=1)"
#endif
__device__ __forceinline__ void wo_prefetch(const Args& a, int t, uint4 (&wp)[8][4]) {
    // the W slice is producer independent; t = this CTA's o_proj tile
    const int tid = threadIdx.x;
    const rsrc_t rw = make_rsrc(a.w + (size_t)t * 32 * KLOC);
    const uint32_t xo = (tid >> 5) * (KLOC * 2) + (tid & 31) * 16;
#pragma unroll
    for (int st = 0; st < 8; ++st)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rw, xo, i * 8 * KLOC * 2 + st * 512, 0);
            wp[st][i] = make_uint4(v[0], v[1], v[2], v[3]);
        }
}
template <int NRB>
__device__ __forceinline__ void gemm_tile_pipe(const Args& a, int t, uint8_t* xs, float (*redk)[MAXTOK][32], uint4* tileb,
                                               const uint4 (&wp)[8][4], uint32_t* slab, uint32_t slab_tgt) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6, lr = lane & 15, lg = lane >> 4;
    const int M = a.M;
    constexpr int XPT = NRB * 2;          // 16-B x chunks per thread per stage: NRB*16 rows x 32 chunks / 256
    uint4 xr[PD][XPT], wf[2][2];
    // buffer loads: one per-lane voffset + compile-time soffset per chunk / stage (no per-chunk 64-bit addresses
    // to keep live); x rows >= M fall outside num_records and read as 0
    const rsrc_t rx = __builtin_amdgcn_make_buffer_rsrc((void*)a.x, 0, M * KLOC * 2, 0x00020000);
    const uint32_t xo = (tid >> 5) * (KLOC * 2) + (tid & 31) * 16;
#define PIPE_ISSUE(ST, SLOT)                                                                                   \
    do {                                                                                                       \
        _Pragma("unroll") for (int i = 0; i < XPT; ++i) {                                                      \
            const u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rx, xo, i * 8 * KLOC * 2 + (ST) * 512, 0);   \
            xr[SLOT][i] = make_uint4(v[0], v[1], v[2], v[3]);                                                  \
        }                                                                                                      \
    } while (0)
    // K stages 2h, 2h + 1 read o columns of kv head h: their x loads wait on slab_ready[h] (W is already in registers)
#define SLAB_WAIT(ST)                                                                                          \
    do {                                                                                                       \
        if (slab && ((ST) & 1) == 0) {                                                                         \
            if (tid == 0) spin_ge(slab + ((ST) >> 1) * 16, slab_tgt, a.ctrl, 0x700u + ((ST) >> 1));            \
            __syncthreads();                                                                                   \
        }                                                                                                      \
    } while (0)
    SLAB_WAIT(0);
    PIPE_ISSUE(0, 0);
    if (PD > 1) { SLAB_WAIT(1); PIPE_ISSUE(1, 1 % PD); }
    if (PD > 2) { SLAB_WAIT(2); PIPE_ISSUE(2, 2 % PD); }
    if (PD > 3) { SLAB_WAIT(3); PIPE_ISSUE(3, 3 % PD); }
    f32x4 acc[NRB][2];
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb) acc[rb][0] = acc[rb][1] = (f32x4){0.f, 0.f, 0.f, 0.f};
#define PIPE_STAGE(ST)                                                                                         \
    do {                                                                                                       \
        constexpr int slot = (ST) % PD;                                                                        \
        uint8_t* buf = xs + ((ST) & 1) * (MAXTOK * PROW);                                                      \
        _Pragma("unroll") for (int i = 0; i < XPT; ++i) {                                                      \
            const int c = tid + i * 256;                                                                       \
            *(uint4*)(buf + (c >> 5) * PROW + (c & 31) * 16) = xr[slot][i];                                    \
        }                                                                                                      \
        uint8_t* wbuf = xs + 2 * (MAXTOK * PROW) + ((ST) & 1) * (32 * PROW);                                   \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                                                          \
            *(uint4*)(wbuf + ((tid >> 5) + i * 8) * PROW + (tid & 31) * 16) = wp[ST][i];                       \
        __syncthreads();                                                                                       \
        _Pragma("unroll") for (int kk = 0; kk < 2; ++kk)                                                       \
        _Pragma("unroll") for (int cb = 0; cb < 2; ++cb)                                                       \
            wf[kk][cb] = *(const uint4*)(wbuf + (cb * 16 + lr) * PROW + (w * 64 + kk * 32 + lg * 8) * 2);       \
        _Pragma("unroll") for (int kk = 0; kk < 2; ++kk) {                                                     \
            uint4 X[NRB];                                                                                      \
            _Pragma("unroll") for (int rb = 0; rb < NRB; ++rb)                                                 \
                X[rb] = *(const uint4*)(buf + (rb * 16 + lr) * PROW + (w * 64 + kk * 32 + lg * 8) * 2);         \
            _Pragma("unroll") for (int rb = 0; rb < NRB; ++rb)                                                 \
            _Pragma("unroll") for (int cb = 0; cb < 2; ++cb)                                                   \
                acc[rb][cb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(bf16x8, X[rb]),      \
                    __builtin_bit_cast(bf16x8, wf[kk][cb]), acc[rb][cb], 0, 0, 0);                             \
        }                                                                                                      \
        if ((ST) + PD < 8) { SLAB_WAIT((ST) + PD); PIPE_ISSUE((ST) + PD, slot); }                             \
    } while (0)
    PIPE_STAGE(0); PIPE_STAGE(1); PIPE_STAGE(2); PIPE_STAGE(3);
    PIPE_STAGE(4);
#if PROJ_DIAG
    if (a.stamps && tid == 0) a.stamps[blockIdx.x * 16 + 12] = __builtin_amdgcn_s_memrealtime();   // last slab wait done
#endif
    PIPE_STAGE(5); PIPE_STAGE(6); PIPE_STAGE(7);
    __syncthreads();
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
        for (int cb = 0; cb < 2; ++cb)
#pragma unroll
            for (int i = 0; i < 4; ++i) redk[w][rb * 16 + lg * 4 + i][cb * 16 + lr] = acc[rb][cb][i];
    lds_barrier();
    if (tid < NRB * 64) {
        const int row = tid >> 2, c8 = (tid & 3) * 8;
        uint32_t pk[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = c8 + 2 * q;
            const float v0 = ((redk[0][row][c] + redk[1][row][c]) + redk[2][row][c]) + redk[3][row][c];
            const float v1 = ((redk[0][row][c + 1] + redk[1][row][c + 1]) + redk[2][row][c + 1]) + redk[3][row][c + 1];
            pk[q] = pkbf(v0, v1);
        }
        tileb[tid] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
}

#if OSPLITK == 2
// Wo slice of split-K unit (64-col tile nt, K half kh): stage st (K 256) = 64 rows x 512 B, chunk i of 8 per thread
// (row (tid >> 5) + 8 i); kept in the same 32 x 16-B registers as the 32-col slice (wp[2 st + (i >> 2)][i & 3])
__device__ __forceinline__ void wo_prefetch_sk(const Args& a, int nt, int kh, uint4 (&wp)[8][4]) {
    const int tid = threadIdx.x;
    const rsrc_t rw = make_rsrc(a.w + (size_t)nt * 64 * KLOC + kh * 1024);
    const uint32_t xo = (tid >> 5) * (KLOC * 2) + (tid & 31) * 16;
#pragma unroll
    for (int st = 0; st < 4; ++st)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rw, xo, i * 8 * KLOC * 2 + st * 512, 0);
            wp[2 * st + (i >> 2)][i & 3] = make_uint4(v[0], v[1], v[2], v[3]);
        }
}
// 64 output columns x K half kh (4 stages of 256, PD in flight); stage ST reads o columns of kv head 2 kh + ST / 2
// and waits on that slab; result: bf16 [M][64] in tileb (8 x 16 B per row)
template <int NRB>
__device__ __forceinline__ void gemm_tile_sk(const Args& a, int kh, uint8_t* xs, uint4* tileb, const uint4 (&wp)[8][4],
                                             uint32_t* slab, uint32_t slab_tgt, uint32_t xch_tgt) {
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6, lr = lane & 15, lg = lane >> 4;
    const int M = a.M;
    constexpr int XPT = NRB * 2;
    uint4 xr[PD][XPT], wf[2][4];
    const rsrc_t rx = __builtin_amdgcn_make_buffer_rsrc((void*)a.x, 0, M * KLOC * 2, 0x00020000);
    const uint32_t xo = (tid >> 5) * (KLOC * 2) + (tid & 31) * 16 + kh * 2048;
#define SK_ISSUE(ST, SLOT)                                                                                     \
    do {                                                                                                       \
        _Pragma("unroll") for (int i = 0; i < XPT; ++i) {                                                      \
            const u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rx, xo, i * 8 * KLOC * 2 + (ST) * 512, 0);   \
            xr[SLOT][i] = make_uint4(v[0], v[1], v[2], v[3]);                                                  \
        }                                                                                                      \
    } while (0)
#define SK_WAIT(ST)                                                                                            \
    do {                                                                                                       \
        if (slab && ((ST) & 1) == 0) {                                                                         \
            if (tid == 0) spin_ge(slab + (kh * 2 + ((ST) >> 1)) * 16, slab_tgt, a.ctrl, 0x700u + kh * 2 + ((ST) >> 1)); \
            __syncthreads();                                                                                   \
        }                                                                                                      \
    } while (0)
    SK_WAIT(0);
    SK_ISSUE(0, 0);
    if (PD > 1) { SK_WAIT(1); SK_ISSUE(1, 1 % PD); }
    if (PD > 2) { SK_WAIT(2); SK_ISSUE(2, 2 % PD); }
    if (PD > 3) { SK_WAIT(3); SK_ISSUE(3, 3 % PD); }
    f32x4 acc[NRB][4];
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
        for (int cb = 0; cb < 4; ++cb) acc[rb][cb] = (f32x4){0.f, 0.f, 0.f, 0.f};
#define SK_STAGE(ST)                                                                                           \
    do {                                                                                                       \
        constexpr int slot = (ST) % PD;                                                                        \
        uint8_t* buf = xs + ((ST) & 1) * (MAXTOK * PROW);                                                      \
        _Pragma("unroll") for (int i = 0; i < XPT; ++i) {                                                      \
            const int c = tid + i * 256;                                                                       \
            *(uint4*)(buf + (c >> 5) * PROW + (c & 31) * 16) = xr[slot][i];                                    \
        }                                                                                                      \
        uint8_t* wbuf = xs + 2 * (MAXTOK * PROW) + ((ST) & 1) * (64 * PROW);                                   \
        _Pragma("unroll") for (int i = 0; i < 8; ++i)                                                          \
            *(uint4*)(wbuf + ((tid >> 5) + i * 8) * PROW + (tid & 31) * 16) = wp[2 * (ST) + (i >> 2)][i & 3];  \
        __syncthreads();                                                                                       \
        _Pragma("unroll") for (int kk = 0; kk < 2; ++kk) {                                                     \
            _Pragma("unroll") for (int cb = 0; cb < 4; ++cb)                                                   \
                wf[kk & 1][cb] = *(const uint4*)(wbuf + (cb * 16 + lr) * PROW + (w * 64 + kk * 32 + lg * 8) * 2); \
            uint4 X[NRB];                                                                                      \
            _Pragma("unroll") for (int rb = 0; rb < NRB; ++rb)                                                 \
                X[rb] = *(const uint4*)(buf + (rb * 16 + lr) * PROW + (w * 64 + kk * 32 + lg * 8) * 2);         \
            _Pragma("unroll") for (int rb = 0; rb < NRB; ++rb)                                                 \
            _Pragma("unroll") for (int cb = 0; cb < 4; ++cb)                                                   \
                acc[rb][cb] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(bf16x8, X[rb]),      \
                    __builtin_bit_cast(bf16x8, wf[kk & 1][cb]), acc[rb][cb], 0, 0, 0);                         \
        }                                                                                                      \
        if ((ST) + PD < 4) { SK_WAIT((ST) + PD); SK_ISSUE((ST) + PD, slot); }                                 \
    } while (0)
    SK_STAGE(0); SK_STAGE(1);
#if PROJ_DIAG
    if (a.stamps && tid == 0) a.stamps[blockIdx.x * 16 + 12] = __builtin_amdgcn_s_memrealtime();
#endif
    SK_STAGE(2); SK_STAGE(3);
    __syncthreads();
    float (*red)[MAXTOK][64] = (float (*)[MAXTOK][64])xs;   // 4 waves x 64 rows x 64 cols fp32 = 64 KB
#pragma unroll
    for (int rb = 0; rb < NRB; ++rb)
#pragma unroll
        for (int cb = 0; cb < 4; ++cb)
#pragma unroll
            for (int i = 0; i < 4; ++i) red[w][rb * 16 + lg * 4 + i][cb * 16 + lr] = acc[rb][cb][i];
    lds_barrier();
#if OSK_XCH
    // own 32-col half (kh) summed into tileb as bf16 after adding the partner's fp32 half; the other half goes to
    // the partner through qkv_part scratch (free once every epilogue unit is done, which slab_ready implies)
    {
        const int nt = blockIdx.x >> 1;
        const rsrc_t rx = make_rsrc(a.qkv_part);
        const uint32_t xb_out = (uint32_t)(nt * 2 + (1 - kh)) * (MAXTOK * 32 * 4), xb_in = (uint32_t)(nt * 2 + kh) * (MAXTOK * 32 * 4);
        float* own = (float*)(xs + 65536);   // [64][32] fp32
        for (int e = tid; e < NRB * 16 * 8; e += NTHR) {
            const int row = e >> 3, c4 = (e & 7) * 4;
            float o4[4], p4[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int co = (1 - kh) * 32 + c4 + q, cm = kh * 32 + c4 + q;
                p4[q] = ((red[0][row][co] + red[1][row][co]) + red[2][row][co]) + red[3][row][co];
                o4[q] = ((red[0][row][cm] + red[1][row][cm]) + red[2][row][cm]) + red[3][row][cm];
            }
            st_wt(rx, xb_out + (uint32_t)(row * 32 + c4) * 4, make_uint4(__float_as_uint(p4[0]), __float_as_uint(p4[1]),
                                                                      __float_as_uint(p4[2]), __float_as_uint(p4[3])));
            *(float4*)(own + row * 32 + c4) = make_float4(o4[0], o4[1], o4[2], o4[3]);
        }
        arrive(a.ctrl + CTRL_XCH + nt * 2 + (1 - kh), 1u);
        if (tid == 0) spin_ge(a.ctrl + CTRL_XCH + nt * 2 + kh, xch_tgt, a.ctrl, 0xB00u + (uint32_t)nt);
        __syncthreads();
        for (int e = tid; e < M * 4; e += NTHR) {   // tileb[row * 4 + cq]: 8 bf16 of the own half
            const int row = e >> 2, c8 = (e & 3) * 8;
            const uint4 pa = ld_sys(rx, xb_in + (uint32_t)(row * 32 + c8) * 4);
            const uint4 pb = ld_sys(rx, xb_in + (uint32_t)(row * 32 + c8 + 4) * 4);
            const float4 oa = *(const float4*)(own + row * 32 + c8), ob = *(const float4*)(own + row * 32 + c8 + 4);
            tileb[e] = make_uint4(pkbf(oa.x + __uint_as_float(pa.x), oa.y + __uint_as_float(pa.y)),
                                  pkbf(oa.z + __uint_as_float(pa.z), oa.w + __uint_as_float(pa.w)),
                                  pkbf(ob.x + __uint_as_float(pb.x), ob.y + __uint_as_float(pb.y)),
                                  pkbf(ob.z + __uint_as_float(pb.z), ob.w + __uint_as_float(pb.w)));
        }
    }
#else
    for (int e = tid; e < NRB * 16 * 8; e += NTHR) {
        const int row = e >> 3, c8 = (e & 7) * 8;
        uint32_t pk[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = c8 + 2 * q;
            const float v0 = ((red[0][row][c] + red[1][row][c]) + red[2][row][c]) + red[3][row][c];
            const float v1 = ((red[0][row][c + 1] + red[1][row][c + 1]) + red[2][row][c + 1]) + red[3][row][c + 1];
            pk[q] = pkbf(v0, v1);
        }
        tileb[e] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
#endif
#undef SK_ISSUE
#undef SK_WAIT
#undef SK_STAGE
}
#endif

#define XS_BYTES 147456
extern "C" __global__ void __launch_bounds__(NTHR, 1) __attribute__((amdgpu_waves_per_eu(1, 1))) muse_attnblk_mega(Args a) {
    __shared__ __attribute__((aligned(16))) uint8_t xs[XS_BYTES];
    float (*redk)[MAXTOK][32] = (float (*)[MAXTOK][32])xs;
    __shared__ __attribute__((aligned(16))) uint4 tileb[MAXTOK * 4 * OSPLITK];
    __shared__ float shs[16];
    __shared__ int sg_lo[NSEG_MAX], sg_L[NSEG_MAX], sg_F[NSEG_MAX], sg_n[NSEG_MAX], sg_first[NSEG_MAX], sg_poff[NSEG_MAX];
    __shared__ int sg_tot[2], sg_q[NSEG_MAX], sg_np[NSEG_MAX];
    __shared__ int s_pid[PID_MAX];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x, G = gridDim.x;
    const int M = a.M, rank = a.rank;
    uint32_t* ctrl = a.ctrl;
    const uint32_t g = (uint32_t)__builtin_amdgcn_readfirstlane(
        (int)__hip_atomic_load(ctrl + CTRL_GEN, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) + 1u;
    const int nrb = (M + 15) >> 4;
    STAMP(0);
#if READY
    // request slots b >= bs publish nothing this generation: pad their epilogue counters so every target stays
    // (g * arrivals) whatever bs the previous launches used
    if (tid == 0 && bid < MAXBS && bid >= a.bs)
        __hip_atomic_fetch_add(ctrl + CTRL_EPI + bid * 16, 32u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (tid == 0 && bid == 0 && G != NTILE) fail(ctrl, 0x900u, (uint32_t)G, NTILE);
#endif
    // segment table (every CTA, identical): seg = b * 4 + h, 32-token steps, equal-step split over the grid.
    // Inputs only (seqused_k): built under the region-a weight loads (ENTRY_LD=1; seqused_k is loaded before them)
    // or at kernel start (ENTRY_LD=0), off the region-b critical path either way.
    const int sl_in = tid < MAXBS && tid < a.bs ? a.seqused[tid] : 0;
    auto seg_table = [&]() {
#if SEG_PAR
    // lane-parallel build in wave 0 (lane = request slot; prefix sums by shuffles, 32-bit integer math with the same
    // results as the serial build): no CTA barrier, no serial loop in front of / under the region-a loads
    if (tid < 64) {
        const int nseg = a.bs, sg = tid;
        const bool on = sg < nseg;
        const int L = sl_in;
        const int lo = on && a.window ? (max(0, L - 8 - a.window + 1) & ~15) : 0;
        const int n = on ? (L - lo + 31) >> 5 : 0;
        int inc = n;
#pragma unroll
        for (int o = 1; o < MAXBS; o <<= 1) { const int t = __shfl_up(inc, o, 64); if (sg >= o) inc += t; }
        const int F = __shfl(inc, MAXBS - 1, 64), Fx = inc - n;
        if (sg < MAXBS) sg_L[sg] = L;
        if (on) { sg_lo[sg] = lo; sg_F[sg] = Fx; sg_n[sg] = n; }
#if PSPLIT
        int P = max(1, (int)(((uint32_t)n * (uint32_t)(G - nseg)) / (uint32_t)max(F, 1)));
        const int q = max((n + P - 1) / P, max(1, a.qs_min));
        P = on ? (n + q - 1) / q : 0;
        int pinc = P;
#pragma unroll
        for (int o = 1; o < MAXBS; o <<= 1) { const int t = __shfl_up(pinc, o, 64); if (sg >= o) pinc += t; }
        const int po = __shfl(pinc, MAXBS - 1, 64), px = pinc - P;
        if (on) { sg_q[sg] = q; sg_np[sg] = P; sg_first[sg] = px; sg_poff[sg] = px; }
        if (sg == 0) {
            sg_tot[0] = F;
            sg_tot[1] = po;   // pieces = CTAs 0 .. po - 1 hold one item each
            if (po > G && bid == 0) fail(ctrl, 0x800u, (uint32_t)po, (uint32_t)G);
        }
#else
        const int qs = max((F + G - 1) / G, max(1, a.qs_min));
        const int first = Fx / qs, last = (Fx + n - 1) / qs;
        const int np = on ? last - first + 1 : 0;
        int pinc = np;
#pragma unroll
        for (int o = 1; o < MAXBS; o <<= 1) { const int t = __shfl_up(pinc, o, 64); if (sg >= o) pinc += t; }
        const int po = __shfl(pinc, MAXBS - 1, 64);
        if (on) { sg_first[sg] = first; sg_poff[sg] = pinc - np; sg_q[sg] = qs; sg_np[sg] = np; }
        if (sg == 0) {
            sg_tot[0] = F;
            sg_tot[1] = qs;
            if (po > NPIECE && bid == 0) fail(ctrl, 0x801u, (uint32_t)po, NPIECE);
        }
#endif
    }
    if (A_DIAG) STAMP(2);
#else
    if (tid < MAXBS) sg_L[tid] = sl_in;
    __syncthreads();
    if (tid == 0) {
        const int nseg = a.bs;
        int F = 0;
        for (int sg = 0; sg < nseg; ++sg) {
            const int L = sg_L[sg];
            const int lo = a.window ? (max(0, L - 8 - a.window + 1) & ~15) : 0;   // page aligned; the window mask
                                                                                    // drops tokens before the window
            sg_lo[sg] = lo;
            sg_F[sg] = F;
            sg_n[sg] = (L - lo + 31) >> 5;
            F += sg_n[sg];
        }
#if PSPLIT
        // per-request split: request sg gets ~G * n / F pieces of q steps, one piece per CTA (CTA = piece index), so
        // no CTA pays two serial piece prologues (Q / page-id / first K-V latency chain + partial write) at a request
        // boundary
        int po = 0;
        for (int sg = 0; sg < nseg; ++sg) {
            const int n = sg_n[sg];
            int P = max(1, (int)(((long long)n * (G - nseg)) / max(F, 1)));
            const int q = max((n + P - 1) / P, max(1, a.qs_min));
            P = (n + q - 1) / q;
            sg_q[sg] = q;
            sg_np[sg] = P;
            sg_first[sg] = po;
            sg_poff[sg] = po;
            po += P;
        }
        sg_tot[0] = F;
        sg_tot[1] = po;   // pieces = CTAs 0 .. po - 1 hold one item each
        if (po > G && bid == 0) fail(ctrl, 0x800u, (uint32_t)po, (uint32_t)G);
#else
        // a piece writes a 64 KB (o, max, denom) partial for its 4 kv heads and reads 512 B of K/V per token, so short
        // contexts (sliding window) use fewer, longer pieces instead of spreading ~2 steps over every CTA
        const int qs = max((F + G - 1) / G, max(1, a.qs_min));
        int po = 0;
        for (int sg = 0; sg < nseg; ++sg) {
            const int first = sg_F[sg] / qs, last = (sg_F[sg] + sg_n[sg] - 1) / qs;
            sg_first[sg] = first;
            sg_poff[sg] = po;
            sg_q[sg] = qs;
            sg_np[sg] = last - first + 1;
            po += last - first + 1;
        }
        sg_tot[0] = F;
        sg_tot[1] = qs;
        if (po > NPIECE && bid == 0) fail(ctrl, 0x801u, (uint32_t)po, NPIECE);
#endif
    }
    if (A_DIAG) STAMP(2);
#endif
    };
    bool seg_pending = true;
    if (!ENTRY_LD || bid >= (NQKV / QKV_CG) * QKV_KS) { seg_table(); __syncthreads(); seg_pending = false; }
    auto pre = [&]() { if (seg_pending) { seg_table(); seg_pending = false; } };

    // ------------------------------------------------------------------ a: qkv_gemm
    for (int u = bid; u < (NQKV / QKV_CG) * QKV_KS; u += G) {
        const int nt = u >> 3, ks = u & 7;
        switch (nrb) {
            case 1: qkv_unit<1>(a, nt, ks, (float*)xs, pre); break;
            case 2: qkv_unit<2>(a, nt, ks, (float*)xs, pre); break;
            case 3: qkv_unit<3>(a, nt, ks, (float*)xs, pre); break;
            default: qkv_unit<4>(a, nt, ks, (float*)xs, pre); break;
        }
#if READY
        arrive(ctrl + CTRL_CG + nt * 16, 1u);   // head_ready edge: column group nt gains one of its 8 K eighths
#endif
    }
    STAMP(1);
#if !READY
    grid_barrier(ctrl, g, 0, G);
#endif
    if (!A_DIAG) STAMP(2);
#if PID_PF
    // page ids of this CTA's first attention item (block table: an input, producer independent) are loaded into
    // registers here, before the region-b readiness spins, and parked in LDS at the item start: the cold block-table
    // load no longer sits between the end of region b and the item's K/V prologue (the segment table built by
    // thread 0 is visible: region a synchronised the CTA)
    int pf_sg = -1, pf_nid = 0;
    int pidr[PID_MAX / NTHR];
    {
        int pf_st0 = 0, pf_st1 = 0;
#if PSPLIT
        for (int sg = 0; sg < a.bs; ++sg) {
            const int j = bid - sg_first[sg];
            if (j < 0 || j >= sg_np[sg]) continue;
            pf_sg = sg; pf_st0 = j * sg_q[sg]; pf_st1 = min(sg_n[sg], pf_st0 + sg_q[sg]);
            break;
        }
#else
        const int qs = sg_tot[1], c0 = bid * qs, c1 = min(sg_tot[0], c0 + qs);
        for (int sg = 0; sg < a.bs; ++sg) {
            const int l0 = max(c0, sg_F[sg]), l1 = min(c1, sg_F[sg] + sg_n[sg]);
            if (l0 >= l1) continue;
            pf_sg = sg; pf_st0 = l0 - sg_F[sg]; pf_st1 = l1 - sg_F[sg];
            break;
        }
#endif
        if (pf_sg >= 0) {
            const int p0 = (sg_lo[pf_sg] >> 4) + 2 * pf_st0, lastp = (sg_L[pf_sg] - 1) >> 4;
            pf_nid = min(2 * (pf_st1 - pf_st0), PID_MAX);
            const int32_t* bt = a.block_table + (size_t)pf_sg * a.bt_stride;
#pragma unroll
            for (int k = 0; k < PID_MAX / NTHR; ++k) {
                const int r = tid + k * NTHR;
                pidr[k] = r < pf_nid ? bt[min(p0 + r, lastp)] : 0;
            }
        }
    }
#endif
    // ------------------------------------------------------------------ b: qk_epilogue
    for (int u = bid; u < M * NKH; u += G) {
        const int m = u >> 2, h = u & 3;
#if EPI_PF
        const EpiIn ein = epi_inputs(a, m, h);
#endif
#if READY
        // wait only on the column groups holding this unit's q / k / v / gate columns of kv head h
        if (tid < NQKV / QKV_CG) {
            const int c0 = tid * QKV_CG, c1 = c0 + QKV_CG;
            auto ov = [&](int lo, int n) { return lo < c1 && c0 < lo + n; };
            if (ov(h * 512, 512) || ov(2048 + h * 64, 64) || ov(2304 + h * 64, 64) || ov(2560 + h * 512, 512))
                spin_ge(ctrl + CTRL_CG + tid * 16, 8u * g, ctrl, 0x500u + (uint32_t)tid);
        }
        __syncthreads();
#endif
#if !EPI_PF
        const EpiIn ein = epi_inputs(a, m, h);
#endif
        qk_epilogue(a, m, h, ein);
#if READY
        arrive(ctrl + CTRL_EPI + (m >> 3) * 16, 1u);   // request m / 8 gains one of its 32 (row, kv head) units
#endif
    }
    STAMP(3);
#if !READY
    grid_barrier(ctrl, g, 1, G);
#endif
    STAMP(4);
    // ------------------------------------------------------------------ c: attention partials
    bool had_item = false;
    {
        const int nseg = a.bs;
#if ATT_RDY
        // att_ready[b] (b < MAXBS) gains exactly one arrival per CTA per generation: at once for a request this CTA
        // holds no attention item of, otherwise after that item's partial stores completed (target g * G)
#if ATT_AGG
        // aggregated: only the CTAs holding an item of request b arrive on att_ready[b]; the first item's CTA
        // (j == 0) also adds the G - np_b arrivals of the item-free CTAs, and CTA b adds G for a padded slot b >= bs.
        // The target stays g * G, and region c no longer issues ~bs * G contended atomics whose completion every
        // CTA's next workgroup barrier (the page-id parking before its item) waits on
        if (tid == 0 && bid < MAXBS && bid >= nseg)
            __hip_atomic_fetch_add(ctrl + CTRL_ATT + bid * 16, (uint32_t)G, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
        if (tid < MAXBS) {
            bool has = false;
            if (tid < nseg) {
#if PSPLIT
                const int j = bid - sg_first[tid];
                has = j >= 0 && j < sg_np[tid];
#else
                const int qs_ = sg_tot[1], c0_ = bid * qs_, c1_ = min(sg_tot[0], c0_ + qs_);
                has = max(c0_, sg_F[tid]) < min(c1_, sg_F[tid] + sg_n[tid]);
#endif
            }
            if (!has) __hip_atomic_fetch_add(ctrl + CTRL_ATT + tid * 16, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
#endif
#endif
#if PSPLIT
        for (int sg = 0; sg < nseg; ++sg) {
            const int j = bid - sg_first[sg];
            if (j < 0 || j >= sg_np[sg]) continue;
            const int F = 0, l0 = j * sg_q[sg], l1 = min(sg_n[sg], l0 + sg_q[sg]);
            const int slot = bid;
#else
        const int qs = sg_tot[1];
        const int c0 = bid * qs, c1 = min(sg_tot[0], c0 + qs);
        for (int sg = 0; sg < nseg; ++sg) {
            const int F = sg_F[sg], n = sg_n[sg];
            const int l0 = max(c0, F), l1 = min(c1, F + n);
            if (l0 >= l1) continue;
            const int slot = sg_poff[sg] + (bid - sg_first[sg]);
#endif
            const int w = tid >> 6;
            had_item = true;
            {
                // page ids of the whole item into LDS (inputs only: issued before the readiness wait)
                const int st0 = l0 - F, st1 = l1 - F, lo = sg_lo[sg], lastp = (sg_L[sg] - 1) >> 4;
                const int p0 = (lo >> 4) + 2 * st0, nid = 2 * (st1 - st0);
                const int32_t* bt = a.block_table + (size_t)sg * a.bt_stride;
                if (nid > PID_MAX && tid == 0) fail(ctrl, 0xA00u, (uint32_t)nid, PID_MAX);
#if PID_PF
                if (sg == pf_sg) {
#pragma unroll
                    for (int k = 0; k < PID_MAX / NTHR; ++k)
                        if (tid + k * NTHR < pf_nid) s_pid[tid + k * NTHR] = pidr[k];
                } else
#endif
                for (int r = tid; r < min(nid, PID_MAX); r += NTHR) s_pid[r] = bt[min(p0 + r, lastp)];
            }
            __syncthreads();   // page ids visible
            STAMP(14);
            // READY: q rows, gate rows and the new K/V rows of request sg must be complete (all 4 kv heads: K/V token
            // rows share cache lines across heads); the wait is inside the item, after its history prologue
            if (slot < NPIECE)
                attn_item(a, sg, w, sg_lo[sg], sg_L[sg], l0 - F, l1 - F, slot * NKH + w, (float*)xs, s_pid,
                          READY ? ctrl + CTRL_EPI + sg * 16 : nullptr, 32u * g, 0x600u + (uint32_t)sg);
#if READY
            else {
                if (tid == 0) spin_ge(ctrl + CTRL_EPI + sg * 16, 32u * g, ctrl, 0x600u + (uint32_t)sg);
                __syncthreads();
            }
#endif
#if ATT_RDY
#if ATT_AGG
            // this item's (o, max, denom) partials are written through (+ the item-free CTAs' share on item 0)
            arrive(ctrl + CTRL_ATT + sg * 16, bid == sg_first[sg] ? 1u + (uint32_t)(G - sg_np[sg]) : 1u);
#else
            arrive(ctrl + CTRL_ATT + sg * 16, 1u);   // this item's (o, max, denom) partials are written through
#endif
#endif
        }
    }
    // Wo slice of this CTA's o_proj tile (t = bid), issued once this CTA's attention work is done: CTAs without an
    // attention piece stream it during attention; it overlaps barrier 2 and the merge (placement inside / after the
    // merge measured no better: the merge's slab_ready publication drains vmcnt, W loads included)
    uint4 wp[8][4];
#if OSPLITK == 2
#define WO_PREFETCH() wo_prefetch_sk(a, bid >> 1, bid & 1, wp)
#else
#define WO_PREFETCH() wo_prefetch(a, bid, wp)
#endif
    // merge unit of this CTA (at most one: bs * 32 / tqu <= G). bs > 2: unit = (request, kv head, tqu verify tokens),
    // mapped from the top CTA id down (MREV) so the item-free CTAs of the per-request piece split take them
    const int tqu = a.window ? MTQ_S : MTQ_F;
    const int nu = a.bs <= 2 ? a.bs * 256 : a.bs * 32 / tqu;
    const int mu = (a.bs > 2 && MREV) ? G - 1 - bid : bid;
    const bool mu_has = mu < nu;
    const int upr = 32 / tqu, uph = 8 / tqu;   // units per request / per (request, kv head)
    STAMP(5);
#if WO_LATE
    // barrier 2 split into arrive / wait so the Wo slice (128 KB per CTA, in-order vmcnt) never delays this CTA's
    // arrival, and the last WO_LATE arrivers (the attention critical path) defer their Wo slice until their merge
    // has published slab_ready: their merge loads and slab publication then do not wait behind the Wo stream
    __shared__ uint32_t s_bar;
    vm_drain();
    __syncthreads();
    const uint32_t btarget = ((g - 1u) * NBAR + (READY ? 0u : 2u) + 1u) * (uint32_t)G;
    if (tid == 0) s_bar = __hip_atomic_fetch_add(ctrl + CTRL_BAR, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
#if WO_POL
    // a CTA defers its Wo slice only when it holds both an attention item (its vmcnt was busy until now) and a merge
    // unit (whose L2-bypass loads would retire behind the 128 KB Wo stream); every other CTA streams Wo at once:
    // item-free CTAs during the attention, merge-free item CTAs during the merge
    // (sliding layer, where item-free CTAs exist and warm the item CTAs' slices; WO_POL 2: no CTA defers, the warmed
    // slice lands from L2 in ~1 us and the o_proj x loads never queue behind a deferred 128 KB stream). The full
    // layer keeps the arrival-order rule (last WO_LATE arrivers defer)
    const bool wo_late = (a.window && a.bs > 2) ? (WO_POL == 1 && had_item && mu_has)
                                                : (int32_t)(btarget - (s_bar + 1u)) < WO_LATE;
#else
    const bool wo_late = (int32_t)(btarget - (s_bar + 1u)) < WO_LATE;
#endif
#if WO_WARM && PSPLIT
    // L2 / MALL warm-up of the item CTAs' Wo slices (one 4-B load per 128-B line) by the item-free CTAs of the same
    // XCD (bid % 8) during the attention: the item CTAs' own Wo loads then hit cache instead of streaming HBM while
    // the latency-bound merge loads are in flight. Item-free CTA m of residue class r warms item CTAs k = m (mod F_r).
    if (!had_item) {
        const int np = sg_tot[1], r = bid & 7;
        const int Ir = np > r ? (np - r + 7) >> 3 : 0, Fr = ((G - 1 - r) >> 3) + 1 - Ir;
        uint32_t sink = 0;
        for (int k = (bid >> 3) - Ir; k < Ir; k += Fr) {
            const rsrc_t rw = make_rsrc(a.w + (size_t)(r + 8 * k) * 32 * KLOC);
#pragma unroll
            for (int j = 0; j < 4; ++j) sink ^= __builtin_amdgcn_raw_buffer_load_b32(rw, (uint32_t)(tid + 256 * j) * 128u, 0, 0);
        }
        if (sink == 0x5A5A5A5Au) ctrl[CTRL_SINK + bid] = (uint32_t)tid;   // keeps every lane's loads; scratch, never read
    }
#endif
    if (!wo_late) WO_PREFETCH();
#if !ATT_RDY
    if (tid == 0) spin_ge(ctrl + CTRL_BAR, btarget, ctrl, 0x400u + (READY ? 0u : 2u));
    __syncthreads();
#endif
#else
    WO_PREFETCH();
    grid_barrier(ctrl, g, READY ? 0 : 2, G);
#endif
    STAMP(6);
    // ------------------------------------------------------------------ d: merge + o-norm + gate
#if READY
    // every merge unit of one CTA belongs to one kv head (myh); arrive at once on the other three slabs
    const int myh = !mu_has ? -1 : (a.bs <= 2 ? (mu >> 6) & 3 : (mu / uph) & 3);
    if (tid == 0) {
#pragma unroll
        for (int h = 0; h < NKH; ++h)
            if (h != myh) __hip_atomic_fetch_add(ctrl + CTRL_SLAB + h * 16, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
#endif
    if (a.bs <= 2) {
        for (int u = bid; u < a.bs * 256; u += G) {
            const int b = u >> 8, h = (u >> 6) & 3, qi = u & 63;
#if ATT_RDY
            if (tid == 0) spin_ge(ctrl + CTRL_ATT + b * 16, g * (uint32_t)G, ctrl, 0x700u + (uint32_t)b);
            __syncthreads();
#endif
            merge_row(a, b, h, qi, sg_poff[b], sg_np[b], (float*)xs);
        }
    } else if (mu_has) {
        const int b = mu / upr, h = (mu / uph) & 3, tq0 = (mu % uph) * tqu;
#if ATT_RDY
        if (tid == 0) spin_ge(ctrl + CTRL_ATT + b * 16, g * (uint32_t)G, ctrl, 0x700u + (uint32_t)b);
        __syncthreads();
#endif
        if (MERGE_DIAG) STAMP(12);
        switch (tqu) {
            case 1: merge_norm_gate<1>(a, b, h, tq0, sg_poff[b], sg_np[b]); break;
            case 2: merge_norm_gate<2>(a, b, h, tq0, sg_poff[b], sg_np[b]); break;
            default: merge_norm_gate<4>(a, b, h, tq0, sg_poff[b], sg_np[b]); break;
        }
        if (MERGE_DIAG) STAMP(13);
    }
#if READY
    if (myh >= 0) arrive(ctrl + CTRL_SLAB + myh * 16, 1u);
    uint32_t* const slab = ctrl + CTRL_SLAB;
#else
    uint32_t* const slab = nullptr;
#endif
#if WO_LATE
    if (wo_late) WO_PREFETCH();
#endif
    STAMP(7);
#if !READY
    grid_barrier(ctrl, g, 3, G);
#endif
    STAMP(8);

    const int nchunk = a.nchunk;
    const int rp = (M + nchunk - 1) / nchunk;
    uint8_t* mine = a.peer[rank];
    const rsrc_t rmine = make_rsrc(mine);
    // ------------------------------------------------------------------ e: o_proj tile + RS push
#if OSPLITK == 2
    for (int t = bid; t < NTILE; t += G) {
        const int nt = t >> 1, kh = t & 1, owner = (2 * nt) / TPR, j0 = (2 * nt) % TPR, src = rank * 2 + kh;
        switch (nrb) {
            case 1: gemm_tile_sk<1>(a, kh, xs, tileb, wp, slab, (uint32_t)G * g, g); break;
            case 2: gemm_tile_sk<2>(a, kh, xs, tileb, wp, slab, (uint32_t)G * g, g); break;
            case 3: gemm_tile_sk<3>(a, kh, xs, tileb, wp, slab, (uint32_t)G * g, g); break;
            default: gemm_tile_sk<4>(a, kh, xs, tileb, wp, slab, (uint32_t)G * g, g); break;
        }
        __syncthreads();
        if (PROJ_DIAG) STAMP(13);
        const rsrc_t dst = make_rsrc(a.peer[owner]);
#if OSK_XCH
        {
            const int j = j0 + kh;
            const uint32_t base = RS_OFF + ((uint32_t)(rank * TPR + j) * MAXTOK) * 64u;
            for (int idx = tid; idx < M * 4; idx += NTHR) st_wt(dst, base + idx * 16, tileb[idx]);
            vm_drain();
            __syncthreads();
            if (tid == 0) st_sys32((uint32_t*)(a.peer[owner] + RSF_OFF) + j * NSRC + rank, g);
        }
        continue;
#endif
        for (int idx = tid; idx < M * 8; idx += NTHR) {
            const int row = idx >> 3, jj = (idx >> 2) & 1, cq = idx & 3;
            st_wt(dst, RS_OFF + ((uint32_t)((src * TPR + j0 + jj) * MAXTOK + row) * 4 + cq) * 16, tileb[idx]);
        }
        vm_drain();
        __syncthreads();
        if (tid < 2) st_sys32((uint32_t*)(a.peer[owner] + RSF_OFF) + (j0 + tid) * NSRC + src, g);
    }
#else
    for (int t = bid; t < NTILE; t += G) {
        const int owner = t / TPR, j = t % TPR;
        switch (nrb) {
            case 1: gemm_tile_pipe<1>(a, t, xs, redk, tileb, wp, slab, (uint32_t)G * g); break;
            case 2: gemm_tile_pipe<2>(a, t, xs, redk, tileb, wp, slab, (uint32_t)G * g); break;
            case 3: gemm_tile_pipe<3>(a, t, xs, redk, tileb, wp, slab, (uint32_t)G * g); break;
            default: gemm_tile_pipe<4>(a, t, xs, redk, tileb, wp, slab, (uint32_t)G * g); break;
        }
        __syncthreads();
        if (PROJ_DIAG) STAMP(13);
        const rsrc_t dst = make_rsrc(a.peer[owner]);
        const uint32_t base = RS_OFF + ((uint32_t)(rank * TPR + j) * MAXTOK) * 64u;
        for (int idx = tid; idx < M * 4; idx += NTHR) st_wt(dst, base + idx * 16, tileb[idx]);
        vm_drain();
        __syncthreads();
        if (tid == 0) st_sys32((uint32_t*)(a.peer[owner] + RSF_OFF) + j * NSRC + rank, g);
    }
#endif
    STAMP(9);
    // ------------------------------------------------------------------ f: owner reduce + AG push
    for (int u = bid; u < TPR * nchunk; u += G) {
        const int j = u / nchunk, c = u % nchunk;
        const int T = rank * TPR + j;
        const int r0 = c * rp;
        if (r0 >= M) continue;
        const int r1 = min(M, r0 + rp);
        if (tid < NSRC) spin_ge((uint32_t*)(mine + RSF_OFF) + j * NSRC + tid, g, ctrl, 0x100u + (uint32_t)u);
        __syncthreads();
        if (!ATT_DIAG && !MERGE_DIAG && !PROJ_DIAG && !A_DIAG) STAMP(12);
        const int nv = (r1 - r0) * 4;
        for (int idx = tid; idx < nv; idx += NTHR) {
            const int row = r0 + (idx >> 2), cq = idx & 3;
            uint4 pk[NSRC];
#pragma unroll
            for (int s = 0; s < NSRC; ++s)
                pk[s] = ld_pub(rmine, RS_OFF + ((uint32_t)((s * TPR + j) * MAXTOK + row) * 4 + cq) * 16);
            float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int s = 0; s < NSRC; ++s) {
                const uint32_t q[4] = {pk[s].x, pk[s].y, pk[s].z, pk[s].w};
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[2 * e] += bf2f(q[e] & 0xffffu);
                    v[2 * e + 1] += bf2f(q[e] >> 16);
                }
            }
            tileb[idx] = make_uint4(pkbf(v[0], v[1]), pkbf(v[2], v[3]), pkbf(v[4], v[5]), pkbf(v[6], v[7]));
        }
        __syncthreads();
        for (int e = tid; e < nv * NR; e += NTHR) {
            const int k = e / nv, idx = e - k * nv;
            const int pr = (k + ((a.opt & 1) ? u : 0) + ((a.opt & 2) ? (idx >> 2) : 0)) & (NR - 1);
            const int row = r0 + (idx >> 2), cq = idx & 3;
            uint8_t* pb = a.peer[0];
#pragma unroll
            for (int q = 1; q < NR; ++q) pb = pr == q ? a.peer[q] : pb;
            st_wt(make_rsrc(pb), AG_OFF + ((uint32_t)row * HID + T * 32 + cq * 8) * 2, tileb[idx]);
        }
        vm_drain();
        __syncthreads();
        if (tid < NR) st_sys32((uint32_t*)(a.peer[tid] + AGF_OFF) + c * NTILE + T, g);
    }
    STAMP(10);
    // ------------------------------------------------------------------ g: boundary epilogue per row
    for (int m = bid; m < M; m += G) {
        const int c = m / rp;
        float s[32], al[32], be[32], pw[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = i * 2048 + tid * 8;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const float4 t0 = ((const float4*)(a.stream + (size_t)m * HID + col))[hh];
                const float4 t1 = ((const float4*)(a.alpha + col))[hh];
                const float4 t2 = ((const float4*)(a.beta + col))[hh];
                const float4 t3 = ((const float4*)(a.pre_w + col))[hh];
                const int e = i * 8 + hh * 4;
                s[e] = t0.x; s[e + 1] = t0.y; s[e + 2] = t0.z; s[e + 3] = t0.w;
                al[e] = t1.x; al[e + 1] = t1.y; al[e + 2] = t1.z; al[e + 3] = t1.w;
                be[e] = t2.x; be[e + 1] = t2.y; be[e + 2] = t2.z; be[e + 3] = t2.w;
                pw[e] = t3.x; pw[e + 1] = t3.y; pw[e + 2] = t3.z; pw[e + 3] = t3.w;
            }
        }
        spin_ge((uint32_t*)(mine + AGF_OFF) + c * NTILE + tid, g, ctrl, 0x200u + (uint32_t)m);
        __syncthreads();
        if (!ATT_DIAG && !MERGE_DIAG && !PROJ_DIAG && !A_DIAG) STAMP(13);
        uint4 pk[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) pk[i] = ld_pub(rmine, AG_OFF + ((uint32_t)m * HID + i * 2048 + tid * 8) * 2);
        float o[32];
        float ss1 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t q[4] = {pk[i].x, pk[i].y, pk[i].z, pk[i].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float v0 = bf2f(q[e] & 0xffffu), v1 = bf2f(q[e] >> 16);
                ss1 += v0 * v0;
                ss1 += v1 * v1;
                o[i * 8 + 2 * e] = v0;
                o[i * 8 + 2 * e + 1] = v1;
            }
        }
        const float post_rstd = rsqrtf(block_sum(ss1, shs, tid) / HID + a.post_eps);
        float ss2 = 0.f;
#pragma unroll
        for (int e = 0; e < 32; e += 2) {
            const uint32_t nb = pkbf(o[e] * post_rstd, o[e + 1] * post_rstd);
            const float ns0 = al[e] * s[e] + be[e] * bf2f(nb & 0xffffu);
            const float ns1 = al[e + 1] * s[e + 1] + be[e + 1] * bf2f(nb >> 16);
            s[e] = ns0;
            s[e + 1] = ns1;
            ss2 += ns0 * ns0;
            ss2 += ns1 * ns1;
        }
        const float tot = block_sum(ss2, shs, tid);
        const float pre_rstd = rsqrtf(tot / HID + a.pre_eps);
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 32; ++e) {
            o[e] = s[e] * pre_rstd * pw[e];
            amax = fmaxf(amax, fabsf(o[e]));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = i * 2048 + tid * 8;
            uint32_t ob[4];
#pragma unroll
            for (int hh = 0; hh < 4; ++hh) ob[hh] = pkbf(o[i * 8 + 2 * hh], o[i * 8 + 2 * hh + 1]);
            *(uint4*)(a.out + (size_t)m * HID + col) = make_uint4(ob[0], ob[1], ob[2], ob[3]);
            float4* np = (float4*)(a.new_stream + (size_t)m * HID + col);
            np[0] = make_float4(s[i * 8 + 0], s[i * 8 + 1], s[i * 8 + 2], s[i * 8 + 3]);
            np[1] = make_float4(s[i * 8 + 4], s[i * 8 + 5], s[i * 8 + 6], s[i * 8 + 7]);
        }
        amax = block_max(amax, shs, tid);
        const float scale = fmaxf(amax, 1e-20f) / 448.0f;
        const float inv = 1.0f / scale;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int col = i * 2048 + tid * 8;
            uint32_t lo = 0, hi = 0;
            lo = __builtin_amdgcn_cvt_pk_fp8_f32(o[i * 8 + 0] * inv, o[i * 8 + 1] * inv, lo, false);
            lo = __builtin_amdgcn_cvt_pk_fp8_f32(o[i * 8 + 2] * inv, o[i * 8 + 3] * inv, lo, true);
            hi = __builtin_amdgcn_cvt_pk_fp8_f32(o[i * 8 + 4] * inv, o[i * 8 + 5] * inv, hi, false);
            hi = __builtin_amdgcn_cvt_pk_fp8_f32(o[i * 8 + 6] * inv, o[i * 8 + 7] * inv, hi, true);
            *(uint2*)(a.out_fp8 + (size_t)m * HID + col) = make_uint2(lo, hi);
        }
        if (tid == 0) a.out_scale[m] = scale;
    }
    STAMP(11);
    // ------------------------------------------------------------------ epoch close
    __syncthreads();
    if (tid == 0) {
        const uint32_t old = __hip_atomic_fetch_add(ctrl + CTRL_TICKET, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (old == (uint32_t)G - 1u) {
            __hip_atomic_store(ctrl + CTRL_TICKET, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(ctrl + CTRL_GEN, g, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

static_assert(sizeof(Args) == 328, "mirrored by ctypes");
struct FillArgs { uint4* p; uint64_t n16; };
extern "C" __global__ void muse_attnblk_mega_fill(FillArgs a) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < a.n16; i += (uint64_t)gridDim.x * blockDim.x)
        a.p[i] = make_uint4(0, 0, 0, 0);
}

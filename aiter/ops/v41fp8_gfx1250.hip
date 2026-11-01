// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// DSV4.1 MXFP8 sparse MLA attention (gfx1250 backend of mla_sparse_prefill_v41).
// Built at first use by aiter/ops/v41fp8_gfx1250.py (hipcc, cached), enabled with AITER_V41_FP8_GFX1250=1.
// Structure: persistent (grid = #CU, one WG per CU, XCD-aware order), warp-specialized, TDM gather, rows never split.
// WG = 14 wave32 (448 threads):
//   waves 0-3   "S-waves": QK^T (native MX-FP8 v_wmma_scale_f32_16x16x128_f8f6f4, Q fp8 in VGPRs) + online softmax
//               (additive key bias from the loader: one FMA does scale + mask). wave a owns heads [16a,16a+16).
//               Q rows are TDM-loaded into LDS with a 16 B pad per 512 B (no bank conflict on the per-lane read).
//   waves 4-11  "O-waves": V scale table (v_perm), V read raw with ds_load_tr8_b64, exact 2^e scaling
//               (v_cvt_scale_pk8 + v_pk_mul_bf16, factors via LDS broadcast b128), PV as O^T = V^T P^T with
//               BF16 WMMA (lane = head), epilogue writes [head][dim] bf16 rows straight to LDS staging + TDM store.
//               wave b owns dims [64b,64b+64), all 64 heads.
//   waves 12-13 loaders: row ids -> VGPR ring, 4 TDM gathers (8 rows x 528 B) per wave per tile into an NSTAGE ring
//               (OOB / -1 ids are dropped and zero-filled by the engine); index lines prefetched at start-up.
// Step s: S-waves QK(tile s) -> P(s) bf16 in LDS; O-waves PV(tile s-1). One barrier per step.
// Keys of a row: prefix rows padded to a multiple of 8 slots, then extend rows; 64 slots per tile.
// Work: all rows' tiles concatenated; WG g owns the rows whose first tile lies in [g*N/G', (g+1)*N/G').
// Requirements (checked on host): H=64, D=512, row stride 528 B, scales inline at +512.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef __attribute__((ext_vector_type(16))) int v16i;
typedef __attribute__((ext_vector_type(8))) float v8f;
typedef __attribute__((ext_vector_type(2))) float v2f;
typedef __attribute__((ext_vector_type(2))) int v2i;
typedef __attribute__((ext_vector_type(4))) int v4i;
typedef __attribute__((ext_vector_type(8))) int v8i;
typedef __attribute__((ext_vector_type(16))) __bf16 v16bf;
typedef __attribute__((ext_vector_type(8))) __bf16 v8bf;
#define LDSP __attribute__((address_space(3)))

#ifndef FRESH
#define NOFRESH 1
#endif
// v55 defaults (codegen clean-ups found from ATT instruction stats); -DV55_OLD reverts to the v58 code paths
#ifndef V55_OLD
#define SPK 1
#define DPPU 1
#define NOPM 1
#define FBIAS 1
#endif
#ifndef FTB
#define FTB 1
#endif
#ifndef LSPK
#define LSPK 0  // measured neutral (ATT/gn10); kept as an option
#endif
#ifndef FTW2
#define FTW2 1
#endif
#ifndef QKMODE
#define QKMODE 1
#endif
#ifndef NSTAGE
#define NSTAGE 4
#endif
#ifndef PFCPOL
#define PFCPOL 0
#endif
#ifndef XCD_REMAP
#define XCD_REMAP 1
#endif

namespace v41fp8_gfx1250 {

constexpr int H = 64, D = 512, ROWB = 528, TILE = 64;
constexpr int NT = 448, NS_THREADS = 128;  // 4 S + 8 O + 2 loader waves
constexpr int SLOT_BYTES = TILE * ROWB;  // 33792
constexpr int PROW = 144;                // P buffer row: 64 heads bf16 + 16 B pad
#ifndef NOQPAD
// Q rows padded 512 -> 528 B in LDS (TDM load pad: 16 B after every 512 B): the per-lane Q read (16 lanes at a 512 B
// stride) was a 16-way bank conflict and blocked the CU's LDS for ~2k cycles at every row start
constexpr int QROW = D + 16;
#else
constexpr int QROW = D;
#endif
constexpr int QW_BYTES = 16 * QROW + 16 * 16;  // per S-wave Q: 16 heads x QROW + 16 x 16 scales
constexpr int OFF_RING = 0;
constexpr int OFF_FLAGS = OFF_RING + NSTAGE * SLOT_BYTES;
constexpr int OFF_P = OFF_FLAGS + NSTAGE * TILE * 4;
constexpr int OFF_A = OFF_P + 2 * TILE * PROW;
constexpr int OFF_F = OFF_A + 2 * H * 4 + 64;
constexpr int OFF_Q = OFF_F + 2 * 16 * TILE * 2;
constexpr int OFF_ML = OFF_Q + 2 * 4 * QW_BYTES;
constexpr int OFF_SINK = OFF_ML + 2 * 2 * H * 4;
constexpr int OFF_MISC = OFF_SINK + H * 4;
#ifndef DI_LEAD
#define DI_LEAD 3
#endif
constexpr int DI = DI_LEAD;           // index prefetch lead (steps)
constexpr int IDXR = DI_LEAD + 1;               // index ring entries per O-wave
constexpr int IDXE = 160;             // entry: 32 row ids + 32 source flags
constexpr int OFF_IDX = OFF_MISC + 64 * 4;
constexpr int OFF_ROWS = OFF_IDX + 2 * IDXR * IDXE;
constexpr int MAXR = 512;             // rows per WG kept in LDS (2 x int4 each)
#ifndef DRAIN_PER_STEP
#define DRAIN_PER_STEP 4
#endif
constexpr int OSTG_ROW = 144;         // output staging: [64 dims][64 heads] bf16 (+pad) per O-wave
#ifdef NOPAD
constexpr int STG_RS = 128;
#else
constexpr int STG_RS = OSTG_ROW;
#endif
constexpr int OFF_OSTG = OFF_ROWS + MAXR * 32;
constexpr int LDS_BYTES = OFF_OSTG + 8 * 64 * OSTG_ROW;
static_assert(LDS_BYTES <= 327680, "LDS budget");
constexpr float NEG_BIG = -1.0e30f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float TAU = 8.0f;
constexpr int MIN_TILES_PER_WG = 4;

struct Params {
  const uint8_t* q; long long q_st, q_sh;
  const uint8_t* qs; long long qs_st, qs_sh;
  const uint8_t* p_rows_base; long long p_rows;   // inline rows [n, 528]
  const uint8_t* e_rows_base; long long e_rows;
  const int* p_idx; const int* p_ptr; const int* e_idx; const int* e_ptr;
  const float* sink;
  __bf16* out; long long o_st, o_sh;
  float scale_log2;
  int T, G;
  const int* minus1;  // >= 32 ints of -1 (row id source for padding slots)
  long long* prof;
  long long n_pidx, n_eidx;  // element counts of p_idx / e_idx (for the start-up prefetch guess)
};

// ---------------------------------------------------------------- helpers
__device__ __forceinline__ float dpp_x(float v, int k) {
  int x = __float_as_int(v), r;
  switch (k) {
    case 1: r = __builtin_amdgcn_update_dpp(0, x, 0x161, 0xf, 0xf, false); break;
    case 2: r = __builtin_amdgcn_update_dpp(0, x, 0x162, 0xf, 0xf, false); break;
    case 4: r = __builtin_amdgcn_update_dpp(0, x, 0x164, 0xf, 0xf, false); break;
    default: r = __builtin_amdgcn_update_dpp(0, x, 0x168, 0xf, 0xf, false); break;
  }
  return __int_as_float(r);
}
__device__ __forceinline__ float rowmax16(float v) {
  v = fmaxf(v, dpp_x(v, 1)); v = fmaxf(v, dpp_x(v, 2)); v = fmaxf(v, dpp_x(v, 4)); return fmaxf(v, dpp_x(v, 8));
}
__device__ __forceinline__ float rowsum16(float v) {
  v += dpp_x(v, 1); v += dpp_x(v, 2); v += dpp_x(v, 4); return v + dpp_x(v, 8);
}
template <int SEL>
__device__ __forceinline__ v8bf cvt8(v2i x, int sc) { return __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(x, sc, SEL); }
__device__ __forceinline__ v16bf cvt16(v4i x, int scdw, int byte) {  // scale_sel: byte0:0 byte1:4 byte2:2 byte3:6
  v2i lo = {x[0], x[1]}, hi = {x[2], x[3]};
  v8bf a, b;
  switch (byte) {
    case 0: a = cvt8<0>(lo, scdw); b = cvt8<0>(hi, scdw); break;
    case 1: a = cvt8<4>(lo, scdw); b = cvt8<4>(hi, scdw); break;
    case 2: a = cvt8<2>(lo, scdw); b = cvt8<2>(hi, scdw); break;
    default: a = cvt8<6>(lo, scdw); b = cvt8<6>(hi, scdw); break;
  }
  return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ __forceinline__ unsigned lds_addr(LDSP const void* p) { return (unsigned)(uintptr_t)p; }

// TDM gather: 8 rows of `rowb` bytes (row stride rowb) from gbase[idx*rowb] into LDS (consecutive).
// Rows with (unsigned)idx >= nrows are dropped and zero-filled.
__device__ __forceinline__ void tdm_gather8(const void* gbase, unsigned lds, unsigned nrows, unsigned rowb,
                                            const int* ix) {
  const unsigned long long ga = (unsigned long long)gbase;
  v4i g0 = {(int)(1u | (1u << 30) | (1u << 31)), (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | (1u << 31))};
  v8i g1 = {0, (int)((rowb & 0xFFFF) << 16), (int)(((rowb >> 16) & 0xFFFF) | ((nrows & 0xFFFF) << 16)),
            (int)(((nrows >> 16) & 0xFFFF) | ((rowb & 0xFFFF) << 16)), 8, (int)rowb, 0, 0};
  v4i g2 = {ix[0], ix[1], ix[2], ix[3]};
  v4i g3 = {ix[4], ix[5], ix[6], ix[7]};
  v8i g4 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, g4, 0);
}
// TDM 2-D load: `rows` rows of `rowb` bytes, global row stride `stride` bytes -> LDS dense
// pad512: LDS destination padded by 16 B after every 512 B (pad_enable bit 20, interval log2(512/4)-1 = 6 at bits 22-24,
// amount 16/4-1 = 3 at bits 25-31); pad only applies to loads (TDM store ignores it, probe p50)
__device__ __forceinline__ void tdm_load2d(const void* gbase, unsigned lds, unsigned rows, unsigned rowb, unsigned long long stride,
                                           bool pad512 = false) {
  const unsigned long long ga = (unsigned long long)gbase;
  v4i g0 = {1, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | (1u << 31))};
  v8i g1 = {pad512 ? (int)((1u << 20) | (6u << 22) | (3u << 25)) : 0, (int)((rowb & 0xFFFF) << 16), (int)(((rowb >> 16) & 0xFFFF) | ((rows & 0xFFFF) << 16)),
            (int)(((rows >> 16) & 0xFFFF) | ((rowb & 0xFFFF) << 16)), (int)(rows & 0xFFFF), (int)(unsigned)stride,
            (int)(((stride >> 32) & 0xFFFF) | ((rows & 0xFFFF) << 16)), (int)(rows >> 16)};
  v4i z4 = {0, 0, 0, 0};
  v8i z8 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, z4, z4, z8, 0);
}
// broadcast lane k of each 16-lane row (DPP row_share)
template <int K>
#ifdef DPPU
// mov_dpp has no 'old' operand (undef): all lanes are written (full masks), so no zero-init mov is needed
__device__ __forceinline__ int row_share(int v) { return __builtin_amdgcn_mov_dpp(v, 0x150 + K, 0xf, 0xf, false); }
#else
__device__ __forceinline__ int row_share(int v) { return __builtin_amdgcn_update_dpp(0, v, 0x150 + K, 0xf, 0xf, false); }
#endif
__device__ __forceinline__ int row_share_dyn(int v, int k) {
  switch (k) {
    case 0: return row_share<0>(v); case 1: return row_share<1>(v); case 2: return row_share<2>(v); case 3: return row_share<3>(v);
    case 4: return row_share<4>(v); case 5: return row_share<5>(v); case 6: return row_share<6>(v); case 7: return row_share<7>(v);
    case 8: return row_share<8>(v); case 9: return row_share<9>(v); case 10: return row_share<10>(v); case 11: return row_share<11>(v);
    case 12: return row_share<12>(v); case 13: return row_share<13>(v); case 14: return row_share<14>(v); default: return row_share<15>(v);
  }
}
// TDM 2-D store: LDS dense rows -> global rows of `rowb` bytes at stride `stride`
__device__ __forceinline__ void tdm_store2d(void* gbase, unsigned lds, unsigned rows, unsigned rowb, unsigned long long stride) {
  const unsigned long long ga = (unsigned long long)gbase;
  v4i g0 = {1, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | (1u << 31))};
  v8i g1 = {0, (int)((rowb & 0xFFFF) << 16), (int)(((rowb >> 16) & 0xFFFF) | ((rows & 0xFFFF) << 16)),
            (int)(((rows >> 16) & 0xFFFF) | ((rowb & 0xFFFF) << 16)), (int)(rows & 0xFFFF), (int)(unsigned)stride,
            (int)(((stride >> 32) & 0xFFFF) | ((rows & 0xFFFF) << 16)), (int)(rows >> 16)};
  v4i z4 = {0, 0, 0, 0};
  v8i z8 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_store_from_lds(g0, g1, z4, z4, z8, 0);
}
// same, LDS source rows of 144 B (128 B data + 16 B bank pad): TDM store ignores the pad fields, so the tile is
// 144 B wide and the tensor only 128 B -> the pad columns are out of bounds and dropped (probe p50)
__device__ __forceinline__ void tdm_store2d_pad(void* gbase, unsigned lds, unsigned rows, unsigned rowb, unsigned long long stride) {
  const unsigned long long ga = (unsigned long long)gbase;
  const unsigned tileb = rowb + 16;
  v4i g0 = {1, (int)lds, (int)(unsigned)ga, (int)((unsigned)(ga >> 32) | (1u << 31))};
  v8i g1 = {0, (int)((rowb & 0xFFFF) << 16), (int)(((rowb >> 16) & 0xFFFF) | ((rows & 0xFFFF) << 16)),
            (int)(((rows >> 16) & 0xFFFF) | ((tileb & 0xFFFF) << 16)), (int)(rows & 0xFFFF), (int)(unsigned)stride,
            (int)(((stride >> 32) & 0xFFFF) | ((rows & 0xFFFF) << 16)), (int)(rows >> 16)};
  v4i z4 = {0, 0, 0, 0};
  v8i z8 = {0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_store_from_lds(g0, g1, z4, z4, z8, 0);
}
// workgroup barrier without the generic fence (which would also wait for outstanding global loads)
__device__ __forceinline__ void wg_barrier() {
  asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::: "memory");
}

struct RowGeo { int pp0, np, p8, ep0, ne, nslots, ntiles; };
__device__ __forceinline__ RowGeo row_geo(const Params& p, int r) {
  RowGeo g;
  g.pp0 = p.p_ptr[r]; g.np = max(p.p_ptr[r + 1] - g.pp0, 0);
  g.ep0 = p.e_ptr[r]; g.ne = max(p.e_ptr[r + 1] - g.ep0, 0);
  g.p8 = (g.np + 7) & ~7;
  g.nslots = g.p8 + g.ne;
  g.ntiles = g.nslots > 0 ? (g.nslots + TILE - 1) / TILE : 1;
  return g;
}
// per-row info kept in LDS for the WG's rows: {pp0, np, ep0, ne}, {tstart, ntiles, p8, 0}
struct RowL { int pp0, np, ep0, ne, tstart, ntiles, p8; };
__device__ __forceinline__ RowL rowl(LDSP const int* rows, int k) {
  const v4i a = *(LDSP const v4i*)(rows + 8 * k), b = *(LDSP const v4i*)(rows + 8 * k + 4);
  return RowL{a[0], a[1], a[2], a[3], b[0], b[1], b[2]};
}
struct Cursor {  // (row k relative to the WG's first row, tile ti of ntr)
  int k, ti, ntr, nk;
  __device__ void init(LDSP const int* rows, int ti0, int nrows) { k = 0; ti = ti0; nk = nrows; ntr = rows[5]; }
  __device__ void next(LDSP const int* rows) {
    if (++ti == ntr) { ++k; ti = 0; ntr = k < nk ? rows[8 * k + 5] : 1; }
  }
};
__device__ __forceinline__ int eff_groups(long long N, int G) {
  long long ge = N / MIN_TILES_PER_WG;
  return (int)(ge < 1 ? 1 : (ge < G ? ge : G));
}
__device__ __forceinline__ int logical_wg(int b, int G) {
#if XCD_REMAP
  if ((G & 7) == 0) return (b & 7) * (G >> 3) + (b >> 3);  // consecutive logical WGs share an XCD
#endif
  return b;
}

#ifdef PROF
#define PROF_INIT() long long _pt = __builtin_readcyclecounter(), _acc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#define PROF_MARK(k) { long long _n = __builtin_readcyclecounter(); _acc[k] += _n - _pt; _pt = _n; }
#define PROF_FLUSH() if ((threadIdx.x & 31) == 0 && p.prof) { for (int _k = 0; _k < 10; ++_k) p.prof[((long long)blockIdx.x * 16 + (threadIdx.x >> 5)) * 10 + _k] = _acc[_k]; }
#else
#define PROF_INIT()
#define PROF_MARK(k)
#define PROF_FLUSH()
#endif

// ---------------------------------------------------------------- main kernel
__global__ void __launch_bounds__(NT, 1) attn_kernel(Params p) {
  extern __shared__ __attribute__((aligned(16))) uint8_t smem_generic[];
  LDSP uint8_t* smem = (LDSP uint8_t*)smem_generic;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int hl = lane >> 4, l16 = lane & 15;
#ifdef STOUCH
  // touch this wave's scratch (spill) slot first thing: the first spill store otherwise waits ~2-4.5k cycles on
  // s_wait_xcnt (cold scratch translation) inside the prologue; here it overlaps the kernarg wait
  asm volatile("scratch_store_b32 off, %0, off" :: "v"(0) : "memory");
#endif
  const int g = logical_wg(blockIdx.x, p.G);
  // sink loaded first thing (its global round trip overlaps the setup instead of the prologue)
  const float sinkv = (tid < H && p.sink) ? p.sink[tid] * LOG2E : -INFINITY;
#ifndef NOWARM
  // start-up warm-up (v56): the prologue is a chain of cold accesses (row table -> row ids -> KV rows), each paying a
  // TLB miss (~2.5k cycles cold vs ~0.7-1k warm-TLB DRAM, probe p51). The loader waves prefetch the index lines this WG
  // will most likely need first (uniform-row guess) while the row scan runs.
  if (wave >= 12) {
    const int lw = tid - 12 * 32;  // 0..63
    if (p.n_pidx > 0) {
      long long c = (long long)g * p.n_pidx / p.G - 256 + (long long)lw * 32;
      c = c < 0 ? 0 : (c >= p.n_pidx ? p.n_pidx - 1 : c);
      __builtin_amdgcn_global_prefetch((const __attribute__((address_space(1))) void*)(p.p_idx + c), PFCPOL);
    }
    if (p.n_eidx > 0) {
      long long c = (long long)g * p.n_eidx / p.G - 64 + (long long)lw * 32;
      c = c < 0 ? 0 : (c >= p.n_eidx ? p.n_eidx - 1 : c);
      __builtin_amdgcn_global_prefetch((const __attribute__((address_space(1))) void*)(p.e_idx + c), PFCPOL);
    }
  }
#endif
  // ---- speculative start (v58) ----
  // The loader waves guess the WG's first row assuming all rows have the geometry of row 0 (exact for uniform decode
  // batches), load its row ids and issue the TDM gathers of tiles 0 and 1 while the row scan / prefix / row table run.
  // After setup the guess is verified (r0 == r0g); on a miss the normal prologue runs (after draining the spec gathers).
  // During setup the scan scratch (pre[], geo[]) lives in ring slots 2-3, slots 0-1 receive the spec gathers.
  int ridx[IDXR], rext[IDXR];
  int sp_r0g = -1;
  RowL sp_rl = {0, 0, 0, 0, 0, 0, 0};
  const int* const a_pidx = p.p_idx;
  const int* const a_eidx = p.e_idx;
  const int* const a_m1 = p.minus1;
  const uint8_t* const a_pb = p.p_rows_base;
  const uint8_t* const a_eb = p.e_rows_base;
  const unsigned a_pr = (unsigned)p.p_rows, a_er = (unsigned)p.e_rows;
  const int lb = wave - 12;  // loader index (valid for wave >= 12)
  auto issue_tile = [&](const int sl, const int my, const int myext) {  // 4 TDM gathers of 8 rows into ring slot sl
    LDSP uint8_t* dst = smem + OFF_RING + sl * SLOT_BYTES + lb * 32 * ROWB;
#pragma unroll
    for (int gq = 0; gq < 4; ++gq) {
      int ix[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) ix[k] = __builtin_amdgcn_readlane(my, gq * 8 + k);
      const int ext = __builtin_amdgcn_readlane(myext, gq * 8);
#ifndef ABL_NOTDM
      tdm_gather8(ext ? a_eb : a_pb, lds_addr(dst + gq * 8 * ROWB), ext ? a_er : a_pr, ROWB, ix);
#endif
    }
    const unsigned nr = myext ? a_er : a_pr;
#ifdef FBIAS
    // per-slot additive score bias (0 valid / -3e38 invalid): the S-wave masks with one FMA per score
    ((LDSP float*)(smem + OFF_FLAGS))[sl * TILE + lb * 32 + lane] = (unsigned)my < nr ? 0.f : -3.0e38f;
#else
    (smem + OFF_FLAGS)[sl * TILE + lb * 32 + lane] = (unsigned)my < nr ? 1 : 0;
#endif
  };
  auto tile_src = [&](const RowL& rl, const int ti, int& ext) -> const int* {  // row-id address of this lane's slot
    const int k = ti * TILE + lb * 32 + lane;
    const int* src = a_m1;
    ext = 0;
    if (k < rl.p8) {
      if (k < rl.np) src = a_pidx + rl.pp0 + k;
    } else {
      ext = 1;
      if (k - rl.p8 < rl.ne) src = a_eidx + rl.ep0 + (k - rl.p8);
    }
    return src;
  };
#ifdef SPEC
  const bool sp_fit = ((p.T + 1) * 4 + 15) / 16 * 16 <= 2 * SLOT_BYTES;
#else
  const bool sp_fit = false;
#endif
  int sp_v0 = 0, sp_w0 = 0;
  if (wave >= 12 && sp_fit) {  // stage 0: geometry of row 0 (lane l: entry l & 1)
    sp_v0 = p.p_ptr[min(lane & 1, p.T)];
    sp_w0 = p.e_ptr[min(lane & 1, p.T)];
  }
  // ---- tiles prefix over rows (scratch: ring), WG range, start row ----
  int N, gb, ge, r0, ti0, nrows, Ge;
  {
    const int scr_off = sp_fit ? 2 * SLOT_BYTES : 0;
    LDSP int* pre = (LDSP int*)(smem + OFF_RING + scr_off);
    LDSP int* misc = (LDSP int*)(smem + OFF_MISC);
    const int per = (p.T + NT - 1) / NT;
    const int rb = min(p.T, tid * per), re = min(p.T, rb + per);
    int sum = 0;
    // row geometry cached in LDS behind pre[] (ring area, free during setup) when it fits
    LDSP v4i* geo = (LDSP v4i*)(smem + OFF_RING + scr_off + ((p.T + 1) * 4 + 15) / 16 * 16);
    const bool cgeo = scr_off + ((p.T + 1) * 4 + 15) / 16 * 16 + p.T * 16 <= NSTAGE * SLOT_BYTES;
#ifndef NOSCAN2
    if (per <= 4) {
      // all pointer loads of this thread's rows issued at once (the per-row loop serialized two cold round trips)
      int pp[5], ep[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) { const int r = min(rb + i, p.T); pp[i] = p.p_ptr[r]; ep[i] = p.e_ptr[r]; }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = rb + i;
        if (r < re) {
          const int np = max(pp[i + 1] - pp[i], 0), ne = max(ep[i + 1] - ep[i], 0);
          const int ns = ((np + 7) & ~7) + ne;
          pre[r] = sum;
          sum += ns > 0 ? (ns + TILE - 1) / TILE : 1;
          if (cgeo) geo[r] = (v4i){pp[i], np, ep[i], ne};
        }
      }
    } else
#endif
    {
      for (int r = rb; r < re; ++r) {
        const RowGeo rg = row_geo(p, r);
        pre[r] = sum; sum += rg.ntiles;
        if (cgeo) geo[r] = (v4i){rg.pp0, rg.np, rg.ep0, rg.ne};
      }
    }
    int sp_v1 = 0, sp_w1 = 0;
    if (wave >= 12 && sp_fit) {  // stage 1: guess r0 from row 0's geometry, load the guessed row's pointers
      const int a0 = __builtin_amdgcn_readlane(sp_v0, 0), a1 = __builtin_amdgcn_readlane(sp_v0, 1);
      const int c0 = __builtin_amdgcn_readlane(sp_w0, 0), c1 = __builtin_amdgcn_readlane(sp_w0, 1);
      const int np = max(a1 - a0, 0), ne = max(c1 - c0, 0), ns = ((np + 7) & ~7) + ne;
      const int nt0 = ns > 0 ? (ns + TILE - 1) / TILE : 1;
      const int Ng = p.T * nt0, Geg = eff_groups(Ng, p.G);
      const int gbg = g < Geg ? (int)((long long)g * Ng / Geg) : Ng;
      const int r0g = (gbg + nt0 - 1) / nt0;
      if (p.T >= 1 && r0g < p.T && nt0 >= 3) {
        sp_r0g = r0g;
        sp_v1 = p.p_ptr[r0g + (lane & 1)];
        sp_w1 = p.e_ptr[r0g + (lane & 1)];
      }
    }
    int x = sum;
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up(x, o, 32); if (lane >= o) x += y; }
    if (lane == 31) misc[wave] = x;
#ifndef SETUP5
    if (tid == 0) misc[16] = 0;
#endif
    __syncthreads();
    if (wave >= 12 && sp_r0g >= 0) {  // stage 2: guessed row geometry -> row ids of its tiles 0..2
      const int a0 = __builtin_amdgcn_readlane(sp_v1, 0), a1 = __builtin_amdgcn_readlane(sp_v1, 1);
      const int c0 = __builtin_amdgcn_readlane(sp_w1, 0), c1 = __builtin_amdgcn_readlane(sp_w1, 1);
      sp_rl.pp0 = a0; sp_rl.np = max(a1 - a0, 0); sp_rl.ep0 = c0; sp_rl.ne = max(c1 - c0, 0);
      sp_rl.p8 = (sp_rl.np + 7) & ~7;
      const int ns = sp_rl.p8 + sp_rl.ne;
      sp_rl.ntiles = ns > 0 ? (ns + TILE - 1) / TILE : 1;
      sp_rl.tstart = 0;
      if (sp_rl.ntiles >= 3) {
#pragma unroll
        for (int t = 0; t < 3; ++t) { int e; const int* src = tile_src(sp_rl, t, e); ridx[t] = __builtin_nontemporal_load(src); rext[t] = e; }
      } else {
        sp_r0g = -1;
      }
    }
#ifndef SETUP5
    // 3-barrier setup: every thread derives N from the per-wave totals, r0 / r1 are counts of rows whose first tile
    // lies before gb / ge (pre[] is strictly increasing), reduced per wave and added to one LDS word
    int off = x - sum;
    N = 0;
    for (int w2 = 0; w2 < NT / 32; ++w2) { const int t = misc[w2]; off += w2 < wave ? t : 0; N += t; }
    Ge = eff_groups(N, p.G);
    gb = g < Ge ? (int)((long long)g * N / Ge) : N;
    ge = g < Ge ? (int)((long long)(g + 1) * N / Ge) : N;
    int cnt = 0;
    for (int r = rb; r < re; ++r) {
      const int pr = pre[r] + off;
      pre[r] = pr;
      cnt += (pr < gb ? 1 : 0) + (pr < ge ? 0x10000 : 0);
    }
    if (tid == NT - 1) pre[p.T] = off + sum;
    for (int o = 16; o >= 1; o >>= 1) cnt += __shfl_xor(cnt, o, 32);
    if (lane == 0 && cnt) __hip_atomic_fetch_add((LDSP int*)(misc + 16), cnt, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
    __syncthreads();
    r0 = misc[16] & 0xffff; ti0 = 0;
    const int r1 = misc[16] >> 16;
    nrows = r1 - r0;
    gb = pre[r0]; ge = pre[r1];
#else
    int off = x - sum;
    for (int w2 = 0; w2 < wave; ++w2) off += misc[w2];
    for (int r = rb; r < re; ++r) pre[r] += off;
    if (tid == NT - 1) pre[p.T] = off + sum;
    __syncthreads();
    if (wave >= 12 && sp_r0g >= 0) {  // stage 3: speculative gathers of tiles 0 and 1 into ring slots 0 and 1
      issue_tile(0, ridx[0], rext[0]);
      issue_tile(1, ridx[1], rext[1]);
    }
    N = pre[p.T];
    Ge = eff_groups(N, p.G);
    gb = g < Ge ? (int)((long long)g * N / Ge) : N;
    ge = g < Ge ? (int)((long long)(g + 1) * N / Ge) : N;
    // rows are never split: WG g owns the rows whose first tile lies in [g*N/Ge, (g+1)*N/Ge)
    if (tid == 0) { misc[16] = p.T; misc[18] = p.T; }
    __syncthreads();
    for (int r = rb; r < re; ++r) {
      if (pre[r] >= gb && (r == 0 || pre[r - 1] < gb)) misc[16] = r;
      if (pre[r] >= ge && (r == 0 || pre[r - 1] < ge)) misc[18] = r;
    }
    __syncthreads();
    r0 = misc[16]; ti0 = 0;
    const int r1 = misc[18];
    nrows = r1 - r0;
    gb = pre[r0]; ge = pre[r1];
#endif
    // per-row info of this WG's rows (pre[] is still intact here)
    LDSP int* rows = (LDSP int*)(smem + OFF_ROWS);
    for (int k = tid; k < min(nrows, MAXR); k += NT) {
      RowGeo rg;
      if (cgeo) {
        const v4i x = geo[r0 + k];
        rg.pp0 = x[0]; rg.np = x[1]; rg.ep0 = x[2]; rg.ne = x[3];
        rg.p8 = (rg.np + 7) & ~7; rg.nslots = rg.p8 + rg.ne; rg.ntiles = rg.nslots > 0 ? (rg.nslots + TILE - 1) / TILE : 1;
      } else {
        rg = row_geo(p, r0 + k);
      }
      *(LDSP v4i*)(rows + 8 * k) = (v4i){rg.pp0, rg.np, rg.ep0, rg.ne};
      *(LDSP v4i*)(rows + 8 * k + 4) = (v4i){pre[r0 + k], rg.ntiles, rg.p8, 0};
    }
    __syncthreads();
  }
  const int n = ge - gb;
  if (n <= 0) return;
  if (nrows > MAXR) __builtin_trap();
  LDSP const int* rows = (LDSP const int*)(smem + OFF_ROWS);
  LDSP float* sinkl = (LDSP float*)(smem + OFF_SINK);
  if (tid < H) sinkl[tid] = sinkv;
  if (tid < 8) ((LDSP int*)(smem + OFF_A + 2 * H * 4))[tid] = 0;
  (void)N;
  LDSP float* abuf = (LDSP float*)(smem + OFF_A);
  LDSP float* mlbuf = (LDSP float*)(smem + OFF_ML);
  LDSP uint8_t* pbuf = smem + OFF_P;
  LDSP uint16_t* ftab = (LDSP uint16_t*)(smem + OFF_F);
  LDSP uint8_t* flagsb = smem + OFF_FLAGS;
  PROF_INIT();

  if (tid < NS_THREADS) {
    // ================================ S-waves ================================
    const int a = __builtin_amdgcn_readfirstlane(wave);
    Cursor cur; cur.init(rows, ti0, nrows);
    int seg = 0;
    auto q_load = [&](int row, int buf) {  // own 16 heads -> LDS (TDM, wave-private)
      LDSP uint8_t* qb = smem + OFF_Q + (buf * 4 + a) * QW_BYTES;
      tdm_load2d(p.q + (long long)row * p.q_st + (long long)(16 * a) * p.q_sh, lds_addr(qb), 16, D, (unsigned long long)p.q_sh, QROW != D);
      tdm_load2d(p.qs + (long long)row * p.qs_st + (long long)(16 * a) * p.qs_sh, lds_addr(qb + 16 * QROW), 16, 16, (unsigned long long)p.qs_sh);
    };
    static_assert(QKMODE == 1, "v53+ only implement the MX-FP8 QK path");
    q_load(r0, 0);
    v16i qa[4];
    int qsc[4];
    auto q_regs = [&](int buf) {  // Q of the current segment LDS -> VGPRs; lane offsets from an opaque lane id (no spill)
#ifdef QLANEV
      int lv;
      asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0" : "=v"(lv));
      const int hlx = lv >> 4, l16x = lv & 15;
#else
      const int hlx = hl, l16x = l16;
#endif
      __builtin_amdgcn_s_wait_tensorcnt(0);  // this segment's Q
      LDSP uint8_t* qb = smem + OFF_Q + (buf * 4 + a) * QW_BYTES;
      const v4i sc4 = *(LDSP v4i*)(qb + 16 * QROW + l16x * 16);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        qsc[c] = sc4[c];
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          v4i x = *(LDSP v4i*)(qb + l16x * QROW + c * 128 + jj * 32 + hlx * 16);
          qa[c][4 * jj] = x[0]; qa[c][4 * jj + 1] = x[1]; qa[c][4 * jj + 2] = x[2]; qa[c][4 * jj + 3] = x[3];
        }
      }
    };
#ifndef NOQH
    q_regs(0);  // first segment's Q while the loader's first gathers are still in flight
#endif
    wg_barrier();  // prologue (pairs with the O-waves')

    float m_r = NEG_BIG, l_r = 0.f;  // per lane: head 16a+l16 (partner lane l^16 holds the other 8 tokens of each tile)
    for (int s = 0; s <= n; ++s) {
      if (s < n) {
        const bool first = (s == 0) || cur.ti == 0;
        const bool last = (s == n - 1) || (cur.ti == cur.ntr - 1);
        LDSP uint8_t* slot = smem + OFF_RING + (s % NSTAGE) * SLOT_BYTES;
        if (first) {
#ifndef NOQH
          if (s > 0) q_regs(seg & 1);
#else
          q_regs(seg & 1);
#endif
          m_r = NEG_BIG; l_r = 0.f;
          if (cur.k + 1 < nrows && (n - s) > (cur.ntr - cur.ti)) q_load(r0 + cur.k + 1, (seg + 1) & 1);
        }
        // key validity flags of this tile (loaded early: LDS latency is ~300 cycles under load)
        LDSP const uint8_t* fl = flagsb + (s % NSTAGE) * TILE;
        uint64_t fv[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) fv[nt] = *(LDSP const uint64_t*)(fl + nt * 16 + 8 * hl);
        // ---- S^T = K Q^T (A = 16 keys x 128 dims from LDS, B = Q^T 128 dims x 16 heads in VGPRs) ----
        // D layout: lane = head (16a + l16), vgpr i = key nt*16 + 8*hl + i
        v8f st[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) st[nt] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
        // software pipeline: K operand of step it+1 loaded while step it multiplies (sched_barrier bounds the lookahead)
#ifndef ABL_NOQK
        struct KOp { v4i x[4]; int sc; };
        auto ldk = [&](int it) {
          const int c = it >> 2, nt = it & 3;
          LDSP uint8_t* krow = slot + (nt * 16 + l16) * ROWB;
          KOp k;
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) k.x[jj] = *(LDSP v4i*)(krow + c * 128 + jj * 32 + hl * 16);
          k.sc = *(LDSP int*)(krow + 512 + 4 * c);
          return k;
        };
        // grouped + double-buffered: the K operands of step c+1 are loaded while the 4 WMMAs of step c run
        // (VALU <-> WMMA alternation is expensive on gfx1250; keep the matrix ops contiguous)
        KOp kbuf[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) kbuf[0][nt] = ldk(nt);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (c + 1 < 4) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) kbuf[(c + 1) & 1][nt] = ldk((c + 1) * 4 + nt);
          }
          KOp (&kk)[4] = kbuf[c & 1];
#if QKMODE == 1
          v16i ka[4];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
            ka[nt] = (v16i){kk[nt].x[0][0], kk[nt].x[0][1], kk[nt].x[0][2], kk[nt].x[0][3], kk[nt].x[1][0], kk[nt].x[1][1], kk[nt].x[1][2], kk[nt].x[1][3],
                            kk[nt].x[2][0], kk[nt].x[2][1], kk[nt].x[2][2], kk[nt].x[2][3], kk[nt].x[3][0], kk[nt].x[3][1], kk[nt].x[3][2], kk[nt].x[3][3]};
          __builtin_amdgcn_sched_barrier(0);
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
#ifndef ABL_NOWMMA_QK
            st[nt] = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(0, ka[nt], 0, qa[c], (short)0, st[nt], 0, 0, kk[nt].sc, 0, 0,
                                                                      qsc[c], false, false);
#else
            st[nt][0] += __int_as_float((ka[nt][0] ^ ka[nt][5] ^ qa[c][3] ^ kk[nt].sc) & 0x3f00ffff);
#endif
          }
          __builtin_amdgcn_sched_barrier(0);
#else
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            v16bf kbf[4];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) kbf[jj] = cvt16(kk[nt].x[jj], kk[nt].sc, jj);
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
              st[nt] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, kbf[jj], false, qbf[4 * c + jj], (short)0, st[nt], false, false);
            __builtin_amdgcn_sched_barrier(0);
          }
#endif
        }
#endif
        PROF_MARK(0);
#ifndef ABL_NOSM
        // ---- online softmax (log2 domain, lazy rescale), all in-lane + one cross-half exchange ----
#ifdef FBIAS
        v8f fb[4];  // loaded after QK: keeps 32 VGPRs out of the K double-buffer phase
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) fb[nt] = *(LDSP const v8f*)((LDSP const float*)(smem + OFF_FLAGS) + (s % NSTAGE) * TILE + nt * 16 + 8 * hl);
#endif
        float mx = NEG_BIG;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
          for (int i = 0; i < 8; ++i) {
#if defined(FBIAS)
            const float v = __builtin_fmaf(st[nt][i], p.scale_log2, fb[nt][i]);
#elif defined(NOPM)
            // invalid keys get -3e38 (< NEG_BIG = initial m_r): exp2(-3e38 - m) == 0 for every reachable m, so the
            // probability needs no second select
            const float v = ((fv[nt] >> (8 * i)) & 1) ? st[nt][i] * p.scale_log2 : -3.0e38f;
#else
            const float v = ((fv[nt] >> (8 * i)) & 1) ? st[nt][i] * p.scale_log2 : NEG_BIG;
#endif
            st[nt][i] = v;
            mx = fmaxf(mx, v);
          }
        mx = fmaxf(mx, __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(mx), __float_as_int(mx), 0x76543210, 0xfedcba98, false, false)));
        const bool up = mx > m_r + TAU;
        const float al = up ? __builtin_amdgcn_exp2f(m_r - mx) : 1.f;
        m_r = up ? mx : m_r;
        LDSP uint8_t* pb = pbuf + (s & 1) * H * PROW + (16 * a + l16) * PROW;
        float ls = 0.f;
#if LSPK
        v2f ls2[2] = {{0.f, 0.f}, {0.f, 0.f}};
#endif
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#ifdef SPK
          v8f pv;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
#ifdef NOPM
            pv[i] = __builtin_amdgcn_exp2f(st[nt][i] - m_r);
#else
            pv[i] = ((fv[nt] >> (8 * i)) & 1) ? __builtin_amdgcn_exp2f(st[nt][i] - m_r) : 0.f;
#endif
#if !LSPK
            ls += pv[i];
#endif
          }
#if LSPK
#pragma unroll
          for (int i = 0; i < 8; i += 4) {
            const v2f x0 = {pv[i], pv[i + 1]}, x1 = {pv[i + 2], pv[i + 3]};
            asm("v_pk_add_f32 %0, %1, %2" : "=v"(ls2[0]) : "v"(ls2[0]), "v"(x0));
            asm("v_pk_add_f32 %0, %1, %2" : "=v"(ls2[1]) : "v"(ls2[1]), "v"(x1));
          }
#endif
          *(LDSP v8bf*)(pb + (nt * 16 + 8 * hl) * 2) = __builtin_convertvector(pv, v8bf);
#else
          uint32_t pk[4];
#pragma unroll
          for (int i = 0; i < 8; i += 2) {
            const float p0 = ((fv[nt] >> (8 * i)) & 1) ? __builtin_amdgcn_exp2f(st[nt][i] - m_r) : 0.f;
            const float p1 = ((fv[nt] >> (8 * i + 8)) & 1) ? __builtin_amdgcn_exp2f(st[nt][i + 1] - m_r) : 0.f;
            ls += p0 + p1;
            __bf16 b0 = (__bf16)p0, b1 = (__bf16)p1;
            pk[i >> 1] = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
          }
          *(LDSP v4i*)(pb + (nt * 16 + 8 * hl) * 2) = (v4i){(int)pk[0], (int)pk[1], (int)pk[2], (int)pk[3]};
#endif
        }
#if LSPK
        ls = (ls2[0][0] + ls2[1][0]) + (ls2[0][1] + ls2[1][1]);
#endif
        l_r = l_r * al + ls;
        if (hl == 0) abuf[(s & 1) * H + 16 * a + l16] = al;
        if (__builtin_amdgcn_ballot_w32(up) && lane == 0) ((LDSP int*)(abuf + 2 * H))[(s & 1) * 4 + a] = s + 1;
        if (last) {
          const float lt = l_r + __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(l_r), __float_as_int(l_r), 0x76543210, 0xfedcba98, false, false));
          if (hl == 0) {
            LDSP float* ml = mlbuf + (seg & 1) * 2 * H;
            ml[16 * a + l16] = m_r;
            ml[H + 16 * a + l16] = lt;
          }
          ++seg;
        }
#endif
        cur.next(rows);
      }
      PROF_MARK(1);
      wg_barrier();
      PROF_MARK(2);
    }
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_wait_asynccnt(0);
  } else if (wave >= 12) {
    // ================================ loader waves (2) ================================
    // wave b gathers slots [32b, 32b+32) of every tile: 4 TDM gathers of 8 rows
    const int b = __builtin_amdgcn_readfirstlane(wave - 12);
#ifndef NORING
    // v51: row ids are loaded with plain global loads into a 4-entry VGPR ring (LOADcnt, in-order return) instead of
    // global_load_async_to_lds + LDS read-back. The current row's RowL is cached in registers.
    int lk = 0, lti = ti0;
    RowL crl = rowl(rows, 0);
    int fetched = 0, issued = 0;
    auto fetch = [&](int& dv, int& de) {  // row ids of tile #fetched -> VGPR (one global b32 per lane)
      const int* src = a_m1;
      int ext = 0;
      if (fetched < n) {
        src = tile_src(crl, lti, ext);
        if (++lti == crl.ntiles) {
          lti = 0; ++lk;
          if (lk < nrows) crl = rowl(rows, lk);
        }
      }
      dv = __builtin_nontemporal_load(src);
      de = ext;
      ++fetched;
    };
    auto issue = [&](const int my, const int myext) {
      issue_tile(issued % NSTAGE, my, myext);
      ++issued;
    };
    static_assert(IDXR == 4 && NSTAGE == 4 && DI == 3, "loader ring assumes NSTAGE 4, DI 3");
    if (sp_r0g >= 0 && sp_r0g == r0) {
      // speculation hit: row ids of tiles 0..2 are in ridx[0..2], tiles 0 and 1 already issued
      fetched = 3; issued = 2;
      lti = 3;
      if (lti == crl.ntiles) { lti = 0; lk = 1; if (lk < nrows) crl = rowl(rows, lk); }
      fetch(ridx[3], rext[3]);
      fetch(ridx[0], rext[0]);
    } else {
      if (sp_r0g >= 0) __builtin_amdgcn_s_wait_tensorcnt(0);  // speculative gathers must land before slots 0/1 are re-filled
      // prologue: fetch 0,1,2 | issue 0, fetch 3 | issue 1, fetch 4
      fetch(ridx[0], rext[0]); fetch(ridx[1], rext[1]); fetch(ridx[2], rext[2]);
      issue(ridx[0], rext[0]); fetch(ridx[3], rext[3]);
      issue(ridx[1], rext[1]); fetch(ridx[0], rext[0]);
    }
    __builtin_amdgcn_s_wait_tensorcnt(4 * (NSTAGE - 3));  // tile 0 landed
    wg_barrier();
    // step s: issue tile s+2 (ring slot (s+2)&3), fetch tile s+5 (slot (s+1)&3)
    for (int s0 = 0; s0 <= n; s0 += 4) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (s0 + j > n) break;
        issue(ridx[(j + 2) & 3], rext[(j + 2) & 3]);
        PROF_MARK(3);
        fetch(ridx[(j + 1) & 3], rext[(j + 1) & 3]);
        PROF_MARK(0);
        __builtin_amdgcn_s_wait_tensorcnt(4 * (NSTAGE - 3));  // tile s+1 landed
        PROF_MARK(1);
        wg_barrier();
        PROF_MARK(2);
      }
    }
#endif
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __builtin_amdgcn_s_wait_asynccnt(0);
  } else {
    // ================================ O-waves ================================
    const int otid = tid - NS_THREADS, ow = __builtin_amdgcn_readfirstlane(wave - 4);
#ifdef OPRIO
    __builtin_amdgcn_s_setprio(OPRIO);  // O-waves are the critical role: win issue arbitration over S/loader waves
#endif
    wg_barrier();  // prologue

    v8f o[4][4];
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
      for (int y = 0; y < 4; ++y) o[x][y] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    Cursor pc; pc.init(rows, ti0, nrows);
    int seg = 0;
    LDSP uint8_t* stg = smem + OFF_OSTG + ow * 64 * OSTG_ROW;
    for (int s = 0; s <= n; ++s) {
      PROF_MARK(3);
      // early LDS loads (their latency overlaps the work below)
#if FTW2
      // lane: tokens 2tp, 2tp+1, groups gp, gp+1 (E8M0 bytes, u16 per token row)
      uint32_t e4 = 0, e4b = 0;
      const int tp = otid >> 3, gp = (otid & 7) * 2;
      if (s < n) {
        LDSP const uint8_t* er = smem + OFF_RING + (s % NSTAGE) * SLOT_BYTES + (2 * tp) * ROWB + 512 + gp;
        e4 = *(LDSP const uint16_t*)er;
        e4b = *(LDSP const uint16_t*)(er + ROWB);
      }
#else
      uint32_t e4 = 0;
      const int tok = otid >> 2, g4 = (otid & 3) * 4;
      if (s < n) e4 = *(LDSP uint32_t*)(smem + OFF_RING + (s % NSTAGE) * SLOT_BYTES + tok * ROWB + 512 + g4);
#endif
      v4i rf = {0, 0, 0, 0};
      if (s >= 1) rf = *(LDSP v4i*)(abuf + 2 * H + ((s - 1) & 1) * 4);
      PROF_MARK(4);
      if (s >= 1) {
        const int j = s - 1;
        LDSP uint8_t* slot = smem + OFF_RING + (j % NSTAGE) * SLOT_BYTES;
        LDSP uint8_t* pb = pbuf + (j & 1) * H * PROW;
        LDSP uint16_t* ft = ftab + (j & 1) * 16 * TILE;
#ifndef NOFRESH
        const bool fresh = pc.ti == 0;
#else
        const bool fresh = false;
#endif  // first tile of a row: accumulator starts from C = 0 (no zeroing, no rescale)
        PROF_MARK(6);
#ifndef ABL_NOPV
        // V scale factors: this half-wave needs 2 groups x 4 key blocks x 8 keys = 32 dwords (bf16 pairs);
        // lane l16 holds dwords 2*l16, 2*l16+1 (one b64 load), the rest is broadcast by DPP row_share
#if !FTB
        int fdw[2];
        {
          const int gsel = l16 >> 3, bb = (l16 & 7) >> 1;
          const uint64_t f2 = *(LDSP const uint64_t*)(ft + (2 * ow + gsel) * TILE + 16 * bb + 8 * hl + 4 * (l16 & 1));
          fdw[0] = (int)(uint32_t)f2; fdw[1] = (int)(uint32_t)(f2 >> 32);
        }
#endif
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          v16bf pa[4];
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) {
            // P stored [head][key] bf16: A operand rows = heads, elements = keys (no transpose needed)
            LDSP uint8_t* prow = pb + (mt * 16 + l16) * PROW + (ks * 32 + 8 * hl) * 2;
            const v8bf lo = *(LDSP v8bf*)prow;
            const v8bf hi = *(LDSP v8bf*)(prow + 32);
            pa[mt] = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
          }
          if (ks == 0 && !fresh) {  // rescale (only if some S-wave moved its reference max in tile j (flag = j+1)
          const int jt = j + 1;
          if (rf[0] == jt || rf[1] == jt || rf[2] == jt || rf[3] == jt) {
            // O^T layout: lane = head (mt*16 + l16) -> one alpha per lane per mt
            float alv[4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) alv[mt] = abuf[(j & 1) * H + mt * 16 + l16];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
              for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[nt][mt][i] *= alv[mt];
          }
          }
#if FTB
          v8bf fq[2][2];  // identical for the 16 lanes of a half-wave: LDS broadcast
#pragma unroll
          for (int gs = 0; gs < 2; ++gs)
#pragma unroll
            for (int half = 0; half < 2; ++half)
              fq[gs][half] = *(LDSP const v8bf*)(ft + (2 * ow + gs) * TILE + 16 * (2 * ks + half) + 8 * hl);
#endif
          v16bf vbs[4];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int n0 = ow * 64 + nt * 16;
            const int grp = n0 >> 5;
            v8bf vh[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
              const int T0 = ks * 32 + half * 16 + hl * 8;
              const int tok = T0 + (l16 & 3) + 4 * (l16 >> 3);
              const int doff = n0 + 8 * ((l16 >> 2) & 1);
              const v2i raw = __builtin_amdgcn_ds_load_tr8_b64_v2i32((LDSP v2i*)(slot + tok * ROWB + doff));
              const v8bf vv = __builtin_amdgcn_cvt_scale_pk8_bf16_fp8(raw, 127, 0);
              // factor dwords d=0..3 of vector (gsel = nt>>1, block = 2*ks + half): list index k = gsel*16 + block*4 + d
#if FTB
              const v8bf f = fq[nt >> 1][half];
#else
              v8bf f;
              {
                constexpr int dummy = 0; (void)dummy;
                int fd[4];
#define FDW(d) { const int k = (nt >> 1) * 16 + (2 * ks + half) * 4 + (d);                  fd[d] = row_share_dyn(fdw[k & 1], k >> 1); }
                FDW(0) FDW(1) FDW(2) FDW(3)
#undef FDW
                const v4i fi = {fd[0], fd[1], fd[2], fd[3]};
                f = __builtin_bit_cast(v8bf, fi);
              }
#endif
              vh[half] = vv * f;
            }
            vbs[nt] = __builtin_shufflevector(vh[0], vh[1], 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
          }
          __builtin_amdgcn_sched_barrier(0);
          // O^T[dim][head] = V^T P^T: A = V^T (rows = dims, lane l16 = dim), B = P^T (cols = heads)
          // D: lane = head mt*16 + l16, vgpr i = dim nt*16 + 8*hl + i
#ifndef ABL_NOWMMA_PV
#ifndef NOFRESH
          if (ks == 0 && fresh) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
              for (int mt = 0; mt < 4; ++mt)
                o[nt][mt] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, vbs[nt], false, pa[mt], (short)0, (v8f){0, 0, 0, 0, 0, 0, 0, 0}, false, false);
          } else
#endif
          {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
              for (int mt = 0; mt < 4; ++mt)
                o[nt][mt] = __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, vbs[nt], false, pa[mt], (short)0, o[nt][mt], false, false);
          }
#else
#pragma unroll
          for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
              o[nt][mt][0] += (float)pa[mt][0] * (float)vbs[nt][1] + (float)pa[mt][9] * (float)vbs[nt][14];
#endif
          __builtin_amdgcn_sched_barrier(0);
        }
#endif
        PROF_MARK(8);
        const bool last = pc.ti == pc.ntr - 1;
        if (last) {
          LDSP float* ml = mlbuf + (seg & 1) * 2 * H;
          const int t = r0 + pc.k;
          // O^T layout: lane = head, 8 consecutive dims per (nt, mt). Normalize (one 1/den per lane per mt),
          // store [head][dim] bf16 rows (128 B + 16 B pad) and TDM-store with LDS padding (no transpose pass).
          float inv[4];
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) {
            const int head = mt * 16 + l16;
            const float m = ml[head], l = ml[H + head];
            const float den = l + __builtin_amdgcn_exp2f(sinkl[head] - m);
            inv[mt] = den > 0.f ? __builtin_amdgcn_rcpf(den) : 0.f;
          }
          __builtin_amdgcn_s_wait_tensorcnt(0);  // previous row's TDM store has read the staging area
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#ifndef NOEPK
              // packed: 4 v_pk_mul_f32 + 4 v_cvt_pk_bf16_f32, no per-element shuffles
              const v8f x = o[nt][mt] * inv[mt];
              *(LDSP v8bf*)(stg + (16 * mt + l16) * STG_RS + (16 * nt + 8 * hl) * 2) = __builtin_convertvector(x, v8bf);
#else
              uint32_t w[4];
#pragma unroll
              for (int i = 0; i < 8; i += 2) {
                __bf16 b0 = (__bf16)(o[nt][mt][i] * inv[mt]), b1 = (__bf16)(o[nt][mt][i + 1] * inv[mt]);
                w[i >> 1] = (uint32_t)(*(uint16_t*)&b0) | ((uint32_t)(*(uint16_t*)&b1) << 16);
              }
              *(LDSP v4i*)(stg + (16 * mt + l16) * STG_RS + (16 * nt + 8 * hl) * 2) = (v4i){(int)w[0], (int)w[1], (int)w[2], (int)w[3]};
#endif
            }
          }
          asm volatile("s_wait_dscnt 0x0" ::: "memory");
#ifndef ABL_NOOUT
#ifdef NOPAD
          tdm_store2d((uint8_t*)p.out + (long long)t * p.o_st + (long long)ow * 128, lds_addr(stg), 64, 128, (unsigned long long)p.o_sh);
#else
          tdm_store2d_pad((uint8_t*)p.out + (long long)t * p.o_st + (long long)ow * 128, lds_addr(stg), 64, 128, (unsigned long long)p.o_sh);
#endif
#endif
#ifdef NOFRESH
#pragma unroll
          for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int y = 0; y < 4; ++y) o[x][y] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#endif
          ++seg;
        }
        pc.next(rows);
      }
#ifndef ABL_NOFTAB
      if (s < n) {  // V scale table for tile s: f[g][tok] = 2^(e-127) bf16 bits (consumed next step)
        LDSP uint16_t* ftw = ftab + (s & 1) * 16 * TILE;
#if FTW2
        // bf16 bits of 2^(e-127) = e << 7 (e = 0 -> +0 instead of the subnormal 2^-127: only zero-filled/invalid rows
        // or blocks whose values are all ~1e-36 are affected). Bytes -> 16-bit halves with one v_perm, then one shift.
        const uint32_t w0 = __builtin_amdgcn_perm(e4b, e4, 0x0c040c00u) << 7;  // group gp:   {tok 2tp, tok 2tp+1}
        const uint32_t w1 = __builtin_amdgcn_perm(e4b, e4, 0x0c050c01u) << 7;  // group gp+1
        *(LDSP uint32_t*)(ftw + gp * TILE + 2 * tp) = w0;
        *(LDSP uint32_t*)(ftw + (gp + 1) * TILE + 2 * tp) = w1;
#else
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const uint32_t e = (e4 >> (8 * b)) & 255u;
          ftw[(g4 + b) * TILE + tok] = e ? (uint16_t)(e << 7) : (uint16_t)0x0040;
        }
#endif
      }
#else
      (void)e4;
#endif
      PROF_MARK(5);
      wg_barrier();
      PROF_MARK(7);
    }
    __builtin_amdgcn_s_wait_tensorcnt(0);
  }
  PROF_FLUSH();
}

}  // namespace v41fp8_gfx1250

extern "C" int v41fp8_gfx1250_launch(const v41fp8_gfx1250::Params* pp, hipStream_t stream) {
  using namespace v41fp8_gfx1250;
  static bool init = false;
  if (!init) {
    hipFuncSetAttribute((const void*)attn_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, LDS_BYTES);
    init = true;
  }
  const Params& p = *pp;
  if (p.T == 0) return 0;
  hipLaunchKernelGGL(attn_kernel, dim3(p.G), dim3(NT), LDS_BYTES, stream, p);
  return (int)hipGetLastError();
}
extern "C" int v41fp8_gfx1250_lds_bytes() { return v41fp8_gfx1250::LDS_BYTES; }
extern "C" int v41fp8_gfx1250_max_T() { return NSTAGE * v41fp8_gfx1250::SLOT_BYTES / 4 - 1; }

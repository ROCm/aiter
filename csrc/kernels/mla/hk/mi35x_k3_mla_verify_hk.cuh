// K3 MLA target-verify (QLEN = 8, 12 heads, latent 576 = 512 nope + 64 rope,
// FP8 e4m3 KV, page size 1) split-KV attention for gfx950, hand-scheduled HIP.
//
// One CTA = 4 waves (1 wave / SIMD, 512 regs each) owns all 96 query rows
// (8 q_pos x 12 heads) of one request and one contiguous KV chunk:
//   waves 0..2  compute: wave w owns query rows 32w..32w+31.
//               Q (FP8, per-row scale) stays in VGPRs as the B operand of
//               S^T = K Q^T (MFMA 32x32x64 f8f6f4), so every lane holds ONE
//               query row and the softmax row max is a lane-local max plus a
//               single permlane32 swap. P (FP8) stays in registers and is the
//               B operand of O^T += V^T P^T; O^T (32 rows x 512 fp32) lives in
//               AGPRs. Online softmax with lazy rescaling (tau = 2).
//   wave 3      loader: streams 64-token KV tiles global -> LDS with
//               buffer_load_dwordx4 ... lds (one kv index per lane, 36
//               instructions per tile), 4-stage ring, one s_barrier per tile.
// LDS tile layout: chunk-major [36 chunks of 16 B][64 tokens][16 B]; odd
// chunks store token t at position t ^ 4 (bank spread for the V transposed
// reads). K is read with ds_read_b128, V with ds_read_b64_tr_b8.
//
// Output: per-split normalized partial O (bf16) + base-2 LSE, merged by
// k3_mla_verify_hk_reduce1 (or the Triton merge in aiter/ops/triton/attention/
// mla_verify_reduce.py) with the same split formula, or written directly when NSPLIT = 1.

#pragma once

#include <hip/hip_runtime.h>

#include <stdint.h>

namespace k3hk {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float v2f __attribute__((ext_vector_type(2)));
#define K3_LDS __attribute__((address_space(3)))

constexpr int kH = 12;
constexpr int kQLen = 8;
constexpr int kRows = kH * kQLen;  // 96
constexpr int kD = 576;
constexpr int kDV = 512;
constexpr int kBN = 64;
constexpr int kNCh = kD / 16;  // 36 chunks of 16 B
constexpr int kTileBytes = kNCh * 1024;
constexpr int kStages = 4;
constexpr int kMboxBytes = 4 * 768;  // DMA voffset mailbox: [tile % 4][tg 0..2][lane] u32
constexpr int kThreads = 256;
constexpr float kTau = 2.0f;
constexpr float kPScale = 64.0f;  // p <= 2^tau = 4 -> 256 < 448

struct Params {
  const uint16_t* q;         // bf16 [bs*8, 12, 576] (strided rows)
  const uint8_t* kv;         // fp8 e4m3 pool [N, kv_stride]
  const int32_t* kv_indptr;  // [bs+1]
  const int32_t* kv_indices;
  const float* kv_scale;  // [1]
  uint16_t* o_part;       // bf16 [bs, nsplit, 96, 512]
  float* lse_part;        // [bs, nsplit, 96]
  uint16_t* o_final;      // bf16 [bs*8, 12, 512] (strided), used when nsplit == 1
  int64_t stride_q_tok, stride_q_h, stride_o_tok, stride_o_h;
  int32_t kv_stride;
  uint32_t kv_bytes;  // buffer num_records
  int32_t nsplit, min_chunk;
  float sm_scale_log2;
  int32_t q_lds;  // 1: q rows are contiguous ([tok][12][576] dense) -> coalesced LDS staging of Q
  int32_t bs;
  // fp8 partials: when a request has >= part8_min active splits (0 = never), its partial O is
  // stored as fp8 e4m3 of O / kv_scale (|.| <= max |V| <= 448: no per-row scale needed),
  // in the first 512 B of each 1 KB o_part row; the merge multiplies by kv_scale.
  int32_t part8_min;
};

__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_rsrc(const void* p, uint32_t bytes) {
  return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short)0, (int)bytes, 0x00020000);
}

__device__ __forceinline__ float bf16_to_f(uint32_t bits16) {
  return __uint_as_float(bits16 << 16);
}

__device__ __forceinline__ uint32_t pack_fp8x4(float a, float b, float c, float d) {
  int r = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  r = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, r, true);
  return (uint32_t)r;
}

// max / sum of x over lanes l and l ^ 32 (result in every lane)
__device__ __forceinline__ float xhalf_max(float x) {
  auto r = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return fmaxf(__uint_as_float(r[0]), __uint_as_float(r[1]));
}
__device__ __forceinline__ float xhalf_sum(float x) {
  auto r = __builtin_amdgcn_permlane32_swap(__float_as_uint(x), __float_as_uint(x), false, false);
  return __uint_as_float(r[0]) + __uint_as_float(r[1]);
}

__device__ __forceinline__ v16f mfma(v8i a, v8i b, v16f c) {
  return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}

typedef uint32_t v4u_ __attribute__((ext_vector_type(4)));
#include "k3_mla_verify_hk_asm.h"

__device__ __forceinline__ int split_chunk(int L, int nsplit, int min_chunk) {
  int per = (L + nsplit - 1) / nsplit;
  per = (per + kBN - 1) / kBN * kBN;
  return per > min_chunk ? per : min_chunk;
}

// ---------------------------------------------------------------- loader wave
__device__ __forceinline__ void barrier_raw() {
  asm volatile("s_barrier" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wait_vm() {
  asm volatile("s_waitcnt vmcnt(%0)" ::"n"(N) : "memory");
}

typedef uint32_t v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v4u make_rsrc_s(const void* ptr, uint32_t bytes) {
  const uint64_t a = (uint64_t)ptr;
  v4u r;
  r[0] = __builtin_amdgcn_readfirstlane((uint32_t)a);
  r[1] = __builtin_amdgcn_readfirstlane((uint32_t)(a >> 32) & 0xffffu);
  r[2] = __builtin_amdgcn_readfirstlane(bytes);
  r[3] = 0x00020000u;
  return r;
}

// LDS tile layout ("16 tokens x 4 chunks" blocks): instruction k (0..35) covers token
// group tg = k / 9 (16 tokens) and chunk group cg = k % 9 (4 x 16 B); lane i -> token
// 16 tg + i / 4, LDS slot i % 4 holding chunk 4 cg + ((i % 4) ^ (i >> 4)). So
//   LDS(t, c) = 9216 (t / 16) + 1024 (c / 4) + 64 (t % 16) + 16 ((c % 4) ^ ((t % 16) >> 2)).
// Every instruction touches 16 KV rows x 64 B; the XOR spreads banks for both the
// K row reads and the V transposed reads. Loads are inline asm so the compiler's
// waitcnt pass never inserts a vmcnt(0) (the loader manages vmcnt by hand).
template <int R>
__device__ __forceinline__ void issue_tile(v4u kv_rs, uint32_t slot, uint32_t stride, uint32_t ba, uint32_t co) {
  issue_tile_asm<R>(kv_rs, slot, stride, ba, co);
}

// kv index of token `lane` of the tile -> hardcoded v[56 + R] (async asm load: the
// destination must not be a compiler-managed register).
template <int R>
__device__ __forceinline__ void load_idx(v4u idx_rs, int pos) {
  load_idx_asm<R>(idx_rs, pos * 4);
}

#define K3_RING_SWITCH(r, F, ...) \
  switch ((r) & 7) {              \
    case 0:                       \
      F<0>(__VA_ARGS__);          \
      break;                      \
    case 1:                       \
      F<1>(__VA_ARGS__);          \
      break;                      \
    case 2:                       \
      F<2>(__VA_ARGS__);          \
      break;                      \
    case 3:                       \
      F<3>(__VA_ARGS__);          \
      break;                      \
    case 4:                       \
      F<4>(__VA_ARGS__);          \
      break;                      \
    case 5:                       \
      F<5>(__VA_ARGS__);          \
      break;                      \
    case 6:                       \
      F<6>(__VA_ARGS__);          \
      break;                      \
    default:                      \
      F<7>(__VA_ARGS__);          \
      break;                      \
  }

// Loader wave. Per iteration i (after barrier B_i):
//   load idx(i+4), LDS-DMA tile i+3 into the slot tile i-1 used.
// idx(t) lives in the hardcoded ring v[56 + t % 8]. vmcnt bookkeeping by hand.
__device__ __forceinline__ void
loader_wave(const Params& p, uint8_t K3_LDS* smem, int kv_start, int start, int end, int ntiles, int lane) {
  const v4u kv_rs = make_rsrc_s(p.kv, p.kv_bytes);
  const v4u idx_rs = make_rsrc_s(p.kv_indices + kv_start, (uint32_t)end * 4u);
  const uint32_t stride = (uint32_t)p.kv_stride;
  const uint32_t ba = (uint32_t)(lane >> 2) << 2;                  // bpermute source lane (t % 16) * 4
  const uint32_t co = 16u * (uint32_t)((lane & 3) ^ (lane >> 4));  // chunk-in-group byte offset
  const uint32_t base = (uint32_t)(uintptr_t)smem;
  bool skip = false;
  auto pos_of = [&](int t) {
    int pos = start + t * kBN + lane;
    return pos < end ? pos : end - 1;  // tail tokens re-read the last valid one (masked)
  };
  auto idx = [&](int t) { K3_RING_SWITCH(t, load_idx_asm, idx_rs, pos_of(t) * 4); };
  rl_precompute((uint32_t)lane);
  auto dma = [&](int t) {
    if (!skip) K3_RING_SWITCH(t, issue_tile_rl_asm, kv_rs, base + (t % kStages) * kTileBytes, stride);
  };
  // prologue: idx(0 .. 3), DMA tiles 0..2
  for (int t = 0; t < 4 && t < ntiles; ++t)
    idx(t);
  wait_vm<0>();
  dma(0);
  // Q loaded straight into VGPRs (q_lds == 0): slots 1, 2 are free, so DMA tiles 1, 2 now
  const bool early = p.q_lds == 0;
  if (early)
    for (int t = 1; t < 3 && t < ntiles; ++t)
      dma(t);
  barrier_raw();  // B_q: the compute waves have read their Q out of slots 1..3
  if (!early)
    for (int t = 1; t < 3 && t < ntiles; ++t)
      dma(t);
  // B_pre needs tiles 0 and 1 (B_0): younger than DMA(1) = DMA(2)
  if (ntiles > 2) {
    wait_vm<36>();
  } else {
    wait_vm<0>();
  }
  barrier_raw();
  for (int i = 0; i < ntiles; ++i) {
    if (i > 0) {
      // need DMA(i+1) (issued last in iteration i-2). Younger: iteration i-1's idx, DMA(i+2)
      if (i + 3 < ntiles) {
        wait_vm<1 + 36>();
      } else if (i + 2 < ntiles) {
        wait_vm<36>();
      } else {
        wait_vm<0>();
      }
    }
    barrier_raw();
    // issue order per iteration: idx(i+4), DMA(i+3)
    if (i + 4 < ntiles) idx(i + 4);
    if (i + 3 < ntiles) {
      // idx(i+3) was loaded first in iteration i-1 (or in the prologue for i = 0);
      // younger than it: idx(i+4), DMA(i+2) of iteration i-1 + idx just issued
      if (i > 0) {
        if (i + 4 < ntiles) {
          wait_vm<1 + 36>();
        } else {
          wait_vm<36>();
        }
      }
      dma(i + 3);
    }
  }
}

// Shared-DMA loader: tiles 0..2 entirely by the loader (prologue); from tile 3 on,
// compute wave w issues token group w (9 loads) itself and the loader only token
// group 3, so per-wave vmcnt (max 63) no longer caps the bytes in flight.
// Iteration i (after B_i): load idx(i+5); mailbox(i+4) <- voffsets of token groups 0..2;
// DMA tg3 of tile i+3. Compute iteration i: read mailbox(i+3), DMA its group of tile i+3.
__device__ __forceinline__ void
loader_wave_sd(const Params& p, uint8_t K3_LDS* smem, int kv_start, int start, int end, int ntiles, int lane) {
  const v4u kv_rs = make_rsrc_s(p.kv, p.kv_bytes);
  const v4u idx_rs = make_rsrc_s(p.kv_indices + kv_start, (uint32_t)end * 4u);
  const uint32_t stride = (uint32_t)p.kv_stride;
  const uint32_t ba = (uint32_t)(lane >> 2) << 2;
  const uint32_t co = 16u * (uint32_t)((lane & 3) ^ (lane >> 4));
  const uint32_t base = (uint32_t)(uintptr_t)smem;
  const uint32_t mb = base + kStages * kTileBytes + 4u * lane;
  auto pos_of = [&](int t) {
    int pos = start + t * kBN + lane;
    return pos < end ? pos : end - 1;
  };
  auto idx = [&](int t) { K3_RING_SWITCH(t, load_idx_asm, idx_rs, pos_of(t) * 4); };
  auto dma = [&](int t) {
    K3_RING_SWITCH(t, issue_tile_asm, kv_rs, base + (t % kStages) * kTileBytes, stride, ba, co);
  };
  auto tg3 = [&](int t) { K3_RING_SWITCH(t, issue_tg3_asm, kv_rs, base + (t % kStages) * kTileBytes, stride, ba, co); };
  auto mbox = [&](int t) { K3_RING_SWITCH(t, mbox_write_asm, mb + 768u * (t & 3), ba, stride, co); };
  for (int t = 0; t < 5 && t < ntiles; ++t)
    idx(t);
  wait_vm<0>();
  for (int t = 0; t < 3 && t < ntiles; ++t)
    dma(t);
  if (ntiles > 3) mbox(3);
  if (ntiles > 2) {
    wait_vm<36>();
  } else {
    wait_vm<0>();
  }
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_pre
  for (int i = 0; i < ntiles; ++i) {
    if (i > 0) {
      // own part of tile i+1 (DMA(2) in the prologue for i = 1, else tg3 of iteration i-2);
      // younger: iteration i-1's idx(i+4) and tg3(i+2)
      if (i + 2 < ntiles) {
        wait_vm<9>();
      } else {
        wait_vm<0>();
      }
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_i
    if (i + 5 < ntiles) idx(i + 5);
    if (i + 4 < ntiles) {
      if (i > 0) wait_vm<9>();  // idx(i+4) (iteration i-1); younger: tg3(i+2) (+ idx(i+5))
      mbox(i + 4);
    }
    if (i + 3 < ntiles) tg3(i + 3);
  }
}

// ---------------------------------------------------------------- compute waves
// LDS byte offset (within a tile) this lane supplies to ds_read_b64_tr_b8 for
// V column block 0, token group T_m = 0. Tokens needed by output lane
// (col c, half h): T_m + 4h + {0,1,2,3,8,9,10,11}.
// gfx950 ds_read_b64_tr_b8 (probed): within each 16-lane group, output lane i
// byte j = input lane 2j + i/8, byte i%8. So input lane q supplies row j = q/2,
// 8-column half q%2; output lane i holds column i of the group's 16 columns.
// Returns the per-lane LDS offset for V column blocks with cb even (odd: +1 -> ^),
// T_m = 0; the asm adds 64 * (cb / 2) + 576 * T_m as immediates.
// token-major 576 B rows; 16-B chunk c of token t at position c ^ ((t >> 2) & 3)
__device__ __forceinline__ uint32_t lds_off(uint32_t t, uint32_t c) {
  return 576u * t + 16u * (c ^ ((t >> 2) & 3u));
}
// V: per-lane offset for column blocks with cb even (cb_odd = 0) or odd; the asm adds
// 1024 * (cb / 2) + 9216 * (T_m / 16) as immediates (T_m multiple of 16).
__device__ __forceinline__ uint32_t tr_lane_offset(int lane, int cb_odd) {
  const int g = lane >> 4, q = lane & 15;
  const int h = g >> 1;   // lane / 32
  const int par = g & 1;  // 16-col chunk within the 32-col block
  const int j = q >> 1;
  const int tok = 4 * h + (j < 4 ? j : j + 4);
  return lds_off((uint32_t)tok, (uint32_t)(2 * cb_odd + par)) + 8u * (q & 1);
}

// Epilogue staging: O^T block CB (lane = query row l32, cols 32 CB + 8 a + 4 h + 0..3)
// -> bf16 row-major [32][kOPitch] in LDS, so the global stores are full 1 KB rows.
constexpr int kOPitch = 520;  // bf16 elements (1040 B: spreads banks across rows)
template <int CB>
__device__ __forceinline__ void stage_o(uint16_t K3_LDS* buf, int l32, int h, float o_mul) {
  const v16f o = o_read<CB>();
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const bf16x2 lo = __builtin_convertvector((v2f{o[4 * a] * o_mul, o[4 * a + 1] * o_mul}), bf16x2);
    const bf16x2 hi = __builtin_convertvector((v2f{o[4 * a + 2] * o_mul, o[4 * a + 3] * o_mul}), bf16x2);
    v2i pk = {__builtin_bit_cast(int, lo), __builtin_bit_cast(int, hi)};
    *(v2i K3_LDS*)(buf + l32 * kOPitch + 32 * CB + 8 * a + 4 * h) = pk;
  }
  if constexpr (CB + 1 < 16) stage_o<CB + 1>(buf, l32, h, o_mul);
}

constexpr int kOPitch8 = 544;  // bytes
template <int CB>
__device__ __forceinline__ void stage_o8(uint8_t K3_LDS* buf, int l32, int h, float o_mul) {
  const v16f o = o_read<CB>();
#pragma unroll
  for (int a = 0; a < 4; ++a)
    *(uint32_t K3_LDS*)(buf + l32 * kOPitch8 + 32 * CB + 8 * a + 4 * h) =
        pack_fp8x4(o[4 * a] * o_mul, o[4 * a + 1] * o_mul, o[4 * a + 2] * o_mul, o[4 * a + 3] * o_mul);
  if constexpr (CB + 1 < 16) stage_o8<CB + 1>(buf, l32, h, o_mul);
}

template <int KS>
__device__ __forceinline__ void load_q(const uint16_t* qrow, float q_inv) {
  v4i r[4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
    r[j] = *(const v4i*)(qrow + 64 * KS + 8 * j);
  v8i qf;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const uint32_t u0 = (uint32_t)r[k >> 1][(k & 1) * 2];
    const uint32_t u1 = (uint32_t)r[k >> 1][(k & 1) * 2 + 1];
    qf[k] = (int)pack_fp8x4(
        bf16_to_f(u0 & 0xffffu) * q_inv,
        bf16_to_f(u0 >> 16) * q_inv,
        bf16_to_f(u1 & 0xffffu) * q_inv,
        bf16_to_f(u1 >> 16) * q_inv);
  }
  q_set<KS>(qf);
  if constexpr (KS + 1 < 9) load_q<KS + 1>(qrow, q_inv);
}

struct WaveState {
  float m_i;  // running max (log2 units, scaled) used for p
  float nb;   // 6 - m_i (log2 kPScale folded in)
  float al;   // rescale factor of the pending decision (applied to l in the next exp)
  float l_i;  // running sum (x kPScale)
};

// One pipelined iteration on tile i (scores in S buffer D, decision for it already made):
//   QK(i+1) -> S[1-D]  ||  p = exp(S[D]) -> P, l update
//   [mask S[1-D] if tile i+1 needs it]
//   PV(i)  ||  max / rescale decision of tile i+1;  O *= alpha if any row wants it.
struct DmaCtx {
  v8i ones;  // fp8 1.0 x 32 (A operand of the P row-sum MFMA)
  v4u rs;
  uint32_t base, mb;
  int w, i, ntiles;
};

template <int D>
__device__ __forceinline__ void iteration(
    WaveState& st,
    uint32_t slot,
    uint32_t nslot,
    uint32_t koff_e,
    uint32_t koff_o,
    uint32_t voff,
    uint32_t voff_o,
    float s_scale,
    bool has_next,
    int next_thr,
    bool next_mask,
    const DmaCtx& dc) {
  // own DMA part of tile i+1 (issued in iteration i-2) must have landed; younger: part of tile i+2
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_i
  qk_exp<D>(nslot + koff_e, nslot + koff_o, s_scale, st.nb, st.al, st.l_i);
  if (has_next) {
    if (next_mask) mask_s<1 - D>(next_thr);
    uint64_t rs;
    pv_decide<1 - D>(slot + voff, slot + voff_o, s_scale, st.m_i, st.nb, st.al, rs, dc.ones, st.l_i);
    if (rs != 0) {
      o_scale(st.al);
    }
  } else {
    pv_tile_asm<D>(slot + voff, slot + voff_o, dc.ones, st.al, st.l_i);
  }
}

__device__ __forceinline__ void compute_wave(
    const Params& p, uint8_t K3_LDS* smem, int b, int split, int L, int start, int end, int ntiles, int w, int lane) {
  const int h = lane >> 5, l32 = lane & 31;
  const int row = 32 * w + l32;
  const int qpos = row / kH, head = row - qpos * kH;

  // ---- Q: this lane owns d = 64 ks + 32 h + [0, 32) of its row, ks = 0..8.
  // Pass 1: per-row amax (over all 576 dims); pass 2 (L2 hits): quantize into v[48:119].
  const uint16_t* qrow = p.q + (int64_t)(b * kQLen + qpos) * p.stride_q_tok + (int64_t)head * p.stride_q_h + 32 * h;
  // q8 = fp8(q * 448 / amax_row); qscale = amax_row / 448
  float qscale;
  const float kv_scale = *p.kv_scale;
  if (p.q_lds) {
    // this wave's 32 rows are one contiguous 36 KB block: DMA it into LDS slot 1 + w (tiles 1, 2
    // are issued by the loader only after barrier B_q), read back in the MFMA layout; the slot
    // is released (B_q) before the quantization VALU so the loader's DMA(1, 2) overlap it
    const uint16_t* qblk = p.q + ((int64_t)b * kRows + 32 * w) * kD;
    const uint32_t sl = (uint32_t)(uintptr_t)smem + (uint32_t)((1 + w) * kTileBytes);
    q_load_lds_asm(
        make_rsrc_s(qblk, 32 * kD * 2),
        sl,
        16u * (uint32_t)lane,
        sl + 1152u * (uint32_t)l32 + 64u * (uint32_t)h,
        1024u * (uint32_t)__builtin_amdgcn_readfirstlane((blockIdx.x >> 3) % 36));
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_q: Q staging slots free
    qscale = q_quant_asm();
  } else {
    qscale = q_prologue_asm(qrow);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");  // B_q
  }
  const float s_scale = qscale * (kv_scale * p.sm_scale_log2);

  const uint32_t base = (uint32_t)(uintptr_t)smem;
  // K fragment (ks, tb) of lane (l32, h): token t = 32 tb + l32, chunks 4 ks + 2h (+1)
  // (immediates in the asm: 1024 ks + 18432 tb)
  const uint32_t koff_e = lds_off((uint32_t)l32, 2u * h);
  const uint32_t koff_o = lds_off((uint32_t)l32, 2u * h + 1u);
  const uint32_t voff = tr_lane_offset(lane, 0), voff_o = tr_lane_offset(lane, 1);
  const int lim = min(L - kQLen + qpos, end - 1);  // last visible token of this row (in this split)
  const int mask_from = min(L - kQLen, end - 1);   // tiles ending after this need a mask

  if (!p.q_lds) o_zero();  // q_lds: zeroed during the Q loads (q_load_lds_asm)
  WaveState st{-1e30f, 0.f, 0.f, 0.f};
  auto slot_of = [&](int t) { return base + (uint32_t)((t % kStages) * kTileBytes); };
  auto thr_of = [&](int t) { return lim - (start + t * kBN) - 4 * h; };
  auto mask_of = [&](int t) { return start + t * kBN + kBN - 1 > mask_from; };

  DmaCtx dc{
      v8i{0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838, 0x38383838},
      make_rsrc_s(p.kv, p.kv_bytes),
      base,
      base + kStages * kTileBytes + 256u * (uint32_t)w + 4u * (uint32_t)lane,
      w,
      0,
      ntiles};
  asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier" ::: "memory");  // B_pre: tiles 0, 1 landed
  qk_first(base + koff_e, base + koff_o);
  if (mask_of(0)) mask_s<0>(thr_of(0));
  {
    uint64_t rs;
    decide<0>(s_scale, st.m_i, st.nb, st.al, rs);
  }
  int i = 0;
  for (; i + 2 <= ntiles; i += 2) {
    dc.i = i;
    iteration<0>(
        st, slot_of(i), slot_of(i + 1), koff_e, koff_o, voff, voff_o, s_scale, true, thr_of(i + 1), mask_of(i + 1), dc);
    const bool nx = i + 2 < ntiles;
    dc.i = i + 1;
    iteration<1>(
        st,
        slot_of(i + 1),
        slot_of(nx ? i + 2 : i + 1),
        koff_e,
        koff_o,
        voff,
        voff_o,
        s_scale,
        nx,
        thr_of(i + 2),
        nx && mask_of(i + 2),
        dc);
  }
  if (i < ntiles) {
    dc.i = i;
    iteration<0>(st, slot_of(i), slot_of(i), koff_e, koff_o, voff, voff_o, s_scale, false, 0, false, dc);
  }

  // ---- epilogue
  // l: sum of the fp8-rounded P (x kPScale) from the row-sum MFMA -> already the full row
  // in both lane halves (gen_asm LSUM8); otherwise per-half partial sums of the unrounded p.
  const float l_tot = st.l_i;
  const bool has = l_tot > 0.f;
  const float o_mul = has ? kv_scale / l_tot : 0.f;
  bool part8 = false;
  if (p.part8_min > 0 && p.nsplit != 1) {
    const int chunk = split_chunk(L, p.nsplit, p.min_chunk);
    part8 = min((L + chunk - 1) / chunk, p.nsplit) >= p.part8_min;
  }
  if (p.nsplit != 1 && h == 0) {
    const int64_t prow = ((int64_t)b * p.nsplit + split) * kRows + row;
    const float lse = has ? st.m_i + __builtin_amdgcn_logf(l_tot) - 6.0f : -INFINITY;
    p.lse_part[prow] = lse;
  }
  // After the last barrier only the last tile's slot is still read (by other waves):
  // wave w stages its 32 x 512 bf16 O block in slot (ntiles + w) % 4.
  uint16_t K3_LDS* buf = (uint16_t K3_LDS*)(smem + ((ntiles + w) % kStages) * kTileBytes);
  if (part8) {
    uint8_t K3_LDS* b8 = (uint8_t K3_LDS*)buf;
    stage_o8<0>(b8, l32, h, has ? 1.f / l_tot : 0.f);
    __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0)
    // 1 KB row stride (= the bf16 layout, first half used): no overlap with bf16 requests
    uint8_t* o8 = (uint8_t*)p.o_part + (((int64_t)b * p.nsplit + split) * kRows + 32 * w) * (2 * kDV);
#pragma unroll 4
    for (int r = 0; r < 32; r += 2) {
      const int rr = r + (lane >> 5);
      const v4i v = *(const v4i K3_LDS*)(b8 + rr * kOPitch8 + 16 * (lane & 31));
      *(v4i*)(o8 + rr * (2 * kDV) + 16 * (lane & 31)) = v;
    }
    if (w == 0) {
      __builtin_amdgcn_s_waitcnt(0);
    }
    return;
  }
  stage_o<0>(buf, l32, h, o_mul);
  __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0): own LDS writes done before the reads below
#pragma unroll 4
  for (int r = 0; r < 32; ++r) {
    const int rr = 32 * w + r;
    uint16_t* dst;
    if (p.nsplit == 1) {
      const int qp = rr / kH;
      dst = p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(rr - qp * kH) * p.stride_o_h;
    } else {
      dst = p.o_part + (((int64_t)b * p.nsplit + split) * kRows + rr) * kDV;
    }
    const v4i v = *(const v4i K3_LDS*)(buf + r * kOPitch + 8 * lane);
    *(v4i*)(dst + 8 * lane) = v;
  }
  if (w == 0) {
    __builtin_amdgcn_s_waitcnt(0);
  }
}

__global__ __launch_bounds__(kThreads, 1)
    __attribute__((amdgpu_num_vgpr(48))) void k3_mla_verify_hk_kernel(const Params p) {
  __shared__ __attribute__((aligned(1024))) uint8_t smem_raw[kStages * kTileBytes + kMboxBytes];
  uint8_t K3_LDS* smem = (uint8_t K3_LDS*)smem_raw;
  const int cid = blockIdx.x;
  const int b = cid / p.nsplit, split = cid - b * p.nsplit;
  const int kv_start = p.kv_indptr[b];
  const int L = p.kv_indptr[b + 1] - kv_start;
  const int chunk = split_chunk(L, p.nsplit, p.min_chunk);
  const int start = split * chunk;
  if (start >= L) return;
  const int end = min(start + chunk, L);
  const int ntiles = (end - start + kBN - 1) / kBN;
  const int w = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  if (w == 3) {
    loader_wave(p, smem, kv_start, start, end, ntiles, lane);
  } else {
    compute_wave(p, smem, b, split, L, start, end, ntiles, w, lane);
  }
}

// ---------------------------------------------------------------- split merge
// Workgroup (128-column chunk cc, row, request b); thread t = (g = t / 16, c = t % 16): splits
// s = g + 16 j (j < J), columns 128 cc + 8 c .. + 7. All partial / lse loads are issued at once
// (addresses do not depend on kv_indptr: o_part / lse_part are allocated for all nsplit splits;
// inactive splits are masked by selects, so their uninitialised contents never matter), so the
// merge costs one memory round trip instead of indptr -> lse -> max -> partials.

// fp8-partial variant of reduce1_body (o_part rows = 512 fp8 bytes of O / kv_scale); thread loads 8 B
template <int J>
__device__ __forceinline__ void reduce1_body8(const Params& p, float* s_m, float* s_acc, int ip0, int ip1) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  const int c = t & 15, g = t >> 4;
  const int row = blockIdx.x >> 2, cc = blockIdx.x & 3, b = blockIdx.y;
  const int ns = p.nsplit;
  const int64_t rbase = (int64_t)b * ns * kRows + row;
  const int col = 128 * cc + 8 * c;
  const uint8_t* o8 = (const uint8_t*)p.o_part;
  const int L = ip1 - ip0;
  const int chunk = split_chunk(L, ns, p.min_chunk);
  const int nact = min((L + chunk - 1) / chunk, ns);
  v2i x[J];
  float lv[J];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int s = min(g + 16 * j, nact - 1);
    lv[j] = p.lse_part[rbase + (int64_t)s * kRows];
    x[j] = *(const v2i*)(o8 + (rbase + (int64_t)s * kRows) * (2 * kDV) + col);
  }
  const float kvs = *p.kv_scale;
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < J; ++j)
    if (g + 16 * j < nact) m = fmaxf(m, lv[j]);
  m = fmaxf(m, __shfl_xor(m, 16, 64));
  m = fmaxf(m, __shfl_xor(m, 32, 64));
  if (lane == 0) s_m[w] = m;
  __syncthreads();
  m = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
  if (m == -INFINITY) m = 0.f;
  float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const bool ok = g + 16 * j < nact;
    const float wj = ok ? __builtin_amdgcn_exp2f(lv[j] - m) : 0.f;
    acc[8] += wj;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int u = ok ? x[j][e] : 0;
      const v2f lo = __builtin_amdgcn_cvt_pk_f32_fp8(u, false);
      const v2f hi = __builtin_amdgcn_cvt_pk_f32_fp8(u, true);
      acc[4 * e] += wj * lo[0];
      acc[4 * e + 1] += wj * lo[1];
      acc[4 * e + 2] += wj * hi[0];
      acc[4 * e + 3] += wj * hi[1];
    }
  }
#pragma unroll
  for (int e = 0; e < 9; ++e) {
    acc[e] += __shfl_xor(acc[e], 16, 64);
    acc[e] += __shfl_xor(acc[e], 32, 64);
  }
  if (lane < 16) {
#pragma unroll
    for (int e = 0; e < 9; ++e)
      s_acc[(w * 16 + c) * 9 + e] = acc[e];
  }
  __syncthreads();
  if (t >= 16) return;
  float o[9];
#pragma unroll
  for (int e = 0; e < 9; ++e)
    o[e] = s_acc[c * 9 + e] + s_acc[(16 + c) * 9 + e] + s_acc[(32 + c) * 9 + e] + s_acc[(48 + c) * 9 + e];
  const float inv = o[8] > 0.f ? kvs / o[8] : 0.f;
  v4i r;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const bf16x2 pk = __builtin_convertvector((v2f{o[2 * e] * inv, o[2 * e + 1] * inv}), bf16x2);
    r[e] = __builtin_bit_cast(int, pk);
  }
  const int qp = row / kH;
  *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h + col) = r;
}

template <int J>
__device__ __forceinline__ void reduce1_body(const Params& p, float* s_m, float* s_acc) {
  const int t = threadIdx.x, lane = t & 63, w = t >> 6;
  const int c = t & 15, g = t >> 4;
  const int row = blockIdx.x >> 2, cc = blockIdx.x & 3, b = blockIdx.y;
  const int ns = p.nsplit;
  const int ip0 = p.kv_indptr[b], ip1 = p.kv_indptr[b + 1];
  const int64_t rbase = (int64_t)b * ns * kRows + row;
  const int col = 128 * cc + 8 * c;
  if (p.part8_min > 0 && ip1 > ip0) {
    const int ch = split_chunk(ip1 - ip0, ns, p.min_chunk);
    if (min((ip1 - ip0 + ch - 1) / ch, ns) >= p.part8_min) {
      reduce1_body8<J>(p, s_m, s_acc, ip0, ip1);
      return;
    }
  }
  v4i x[J];
  float lv[J];
  // splits 0..31 are loaded unconditionally (no wait on kv_indptr), the rest only if active
  constexpr int J0 = J < 2 ? J : 2;
#pragma unroll
  for (int j = 0; j < J0; ++j) {
    const int s = min(g + 16 * j, ns - 1);
    lv[j] = p.lse_part[rbase + (int64_t)s * kRows];
    x[j] = *(const v4i*)(p.o_part + (rbase + (int64_t)s * kRows) * kDV + col);
  }
  const int L = ip1 - ip0;
  const int chunk = split_chunk(L, ns, p.min_chunk);
  const int nact = L > 0 ? min((L + chunk - 1) / chunk, ns) : 0;
#pragma unroll
  for (int j = J0; j < J; ++j) {
    const int s = g + 16 * j;
    lv[j] = -INFINITY;
    x[j] = v4i{0, 0, 0, 0};
    if (s < nact) {
      lv[j] = p.lse_part[rbase + (int64_t)s * kRows];
      x[j] = *(const v4i*)(p.o_part + (rbase + (int64_t)s * kRows) * kDV + col);
    }
  }
  float m = -INFINITY;
#pragma unroll
  for (int j = 0; j < J; ++j)
    if (g + 16 * j < nact) m = fmaxf(m, lv[j]);
  m = fmaxf(m, __shfl_xor(m, 16, 64));
  m = fmaxf(m, __shfl_xor(m, 32, 64));
  if (lane == 0) s_m[w] = m;
  __syncthreads();
  m = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3]));
  if (m == -INFINITY) m = 0.f;
  float acc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const bool ok = g + 16 * j < nact;
    const float wj = ok ? __builtin_amdgcn_exp2f(lv[j] - m) : 0.f;
    acc[8] += wj;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t u = ok ? (uint32_t)x[j][e] : 0u;
      acc[2 * e] += wj * bf16_to_f(u & 0xffffu);
      acc[2 * e + 1] += wj * bf16_to_f(u >> 16);
    }
  }
#pragma unroll
  for (int e = 0; e < 9; ++e) {
    acc[e] += __shfl_xor(acc[e], 16, 64);
    acc[e] += __shfl_xor(acc[e], 32, 64);
  }
  if (lane < 16) {
#pragma unroll
    for (int e = 0; e < 9; ++e)
      s_acc[(w * 16 + c) * 9 + e] = acc[e];
  }
  __syncthreads();
  if (t >= 16) return;
  float o[9];
#pragma unroll
  for (int e = 0; e < 9; ++e)
    o[e] = s_acc[c * 9 + e] + s_acc[(16 + c) * 9 + e] + s_acc[(32 + c) * 9 + e] + s_acc[(48 + c) * 9 + e];
  const float inv = o[8] > 0.f ? 1.f / o[8] : 0.f;
  v4i r;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const bf16x2 pk = __builtin_convertvector((v2f{o[2 * e] * inv, o[2 * e + 1] * inv}), bf16x2);
    r[e] = __builtin_bit_cast(int, pk);
  }
  const int qp = row / kH;
  *(v4i*)(p.o_final + (int64_t)(b * kQLen + qp) * p.stride_o_tok + (int64_t)(row - qp * kH) * p.stride_o_h + col) = r;
}

template <int J>
__global__ __launch_bounds__(256) void k3_mla_verify_hk_reduce1(const Params p) {
  __shared__ float s_m[4];
  __shared__ float s_acc[64 * 9];
  reduce1_body<J>(p, s_m, s_acc);
}

// launch: grid (96 * 4, bs), 256 threads; J = ceil(nsplit / 16) rounded up to a power of two
inline void launch_reduce1(const Params& p, int bs, hipStream_t st) {
  const dim3 grid(kRows * 4, bs), blk(256);
  const int J = (p.nsplit + 15) / 16;
  if (J <= 1)
    hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<1>, grid, blk, 0, st, p);
  else if (J <= 2)
    hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<2>, grid, blk, 0, st, p);
  else if (J <= 4)
    hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<4>, grid, blk, 0, st, p);
  else if (J <= 8)
    hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<8>, grid, blk, 0, st, p);
  else
    hipLaunchKernelGGL(k3_mla_verify_hk_reduce1<16>, grid, blk, 0, st, p);
}

}  // namespace k3hk

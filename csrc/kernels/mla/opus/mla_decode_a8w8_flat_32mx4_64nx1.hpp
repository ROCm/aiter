// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// MLA decode forward on gfx950, fp8 Q x fp8 KV, 32 rows per wave x 4 waves, full-rate MFMA only.
//
// Work items of the flat-row plan (opus_mla_decode_flat.cu): q/o rows are token * H + head, an item
// covers up to 128 of them (any head count, a request's tokens and heads packed together) and a KV
// range [kv_start, kv_end) of kv_indices. An item with partial_slot >= 0 writes its normalised bf16
// O and fp32 LSE to the 128-row partial slot partial_slot; partial_slot -1 writes out / lse
// directly. Built with -ffast-math: every row of an item must see at least one key (the planner
// guarantees it).
//
// Core:
//   * one wave per SIMD (512 registers: O lives in 256 AGPRs), 32 query rows per wave;
//   * GEMM0 S^T = K Q^T and GEMM1 O^T = V^T P^T both on v_mfma_f32_32x32x64_f8f6f4 (64 cycles,
//     full rate) -- no half-rate 16x16x32 fp8 MFMA anywhere; rope is just the 9th 64-wide K step;
//   * KV tile = 64 tokens. Every K/V byte read from LDS feeds 32 rows instead of 16, so LDS
//     traffic per FLOP halves.
//
// LDS tile layout: token-major rows of 37 16-byte units (36 data units = 576 fp8, 1 pad unit).
// The pad makes the unit stride odd mod 16, so
//   * QK A-operand reads (ds_read_b128, 16 lanes = 16 tokens at one d-unit) and
//   * V^T reads (ds_read_b64_tr_b8, 8 tokens of one parity x 2 adjacent d-units per 32 lanes)
// are both bank-conflict free with purely affine (immediate-offset) addressing.
//
// Token order. S comes out of the 32x32 MFMA as lane (n = row = l%32, g = l/32), value i of
// M-block mb <-> M-row m = (i/4)*8 + g*4 + i%4. The A-operand rows are permuted with
//   sigma(m) = m4<<4 | m3<<3 | m1<<2 | m0<<1 | m2
// so that value i of M-block mb holds token mb*32 + 2*i + g. P (= fp8(S), packed byte j = mb*16 +
// i) is then directly the PV B-operand, and the matching V^T A-operand for byte block r (tokens 16r
// + 2b + g, b = 0..7) is one ds_read_b64_tr_b8 with lane L = 16q + 2b + h addressing V[16r + 2b +
// q/2][dblk*32 + (q%2)*16 + 8h].
//
// Pipeline (per wave, one s_barrier per KV tile), iteration i:
//   wait own vmcnt (tile i+1 DMA + idx(i+3) landed) ; s_barrier
//   idx(i+4) -> private LDS ring ; QK(i+1) -> S[(i+1)&1] ; DMA(i+3) -> slot (i+3)&3
//   softmax(i) on S[i&1] -> P ; PV(i) from slot i&3
// 4 KV slots (37,888 B each) + 4 x 4 x 256 B private page-index rings = 155,648 B of LDS.

#include "mla_decode_kargs.h"

#include <hip/hip_runtime.h>

struct opus_mla_decode_a8w8_flat_kargs
{
    const void* __restrict__ q_buffer_ptr;  // [total_q * H, 576] fp8
    const void* __restrict__ q_scale_ptr;   // float[1] descale
    const void* __restrict__ kv_buffer_ptr; // [pages, 576] fp8, page_size 1
    const void* __restrict__ kv_scale_ptr;  // float[1] descale
    void* __restrict__ o_accum;             // [slots * 128, 512] bf16 partials
    void* __restrict__ lse_accum;           // [slots * 128] fp32 partial LSE
    void* __restrict__ out_ptr;             // [total_q * H, 512] bf16
    void* __restrict__ lse_ptr;             // [total_q * H] fp32
    const int* __restrict__ q_indptr;
    const int* __restrict__ kv_indptr;
    const int* __restrict__ kv_indices;

    const int* __restrict__ work_indptr;
    const opus_mla_decode_work_info* __restrict__ work_info_set;

    int H;
    int total_tokens;
    int stride_q_b;
    int stride_q_h;
    int stride_o_b;
    int stride_o_h;
    int stride_kv_page;
    float softmax_scale;
    // Round-robin CP: KV shard j of this rank sits at global position j * cp_world_size + cp_rank.
    // g_kv_indptr is the global kv indptr; cp_world_size == 1 is the plain causal mask.
    const int* __restrict__ g_kv_indptr;
    int cp_world_size;
    int cp_rank;
    // Optional: each workgroup stores {core cycles, 100 MHz ticks} of its work loop at [2 * wg].
    int* wg_clock;
};

#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx950__)

template <bool CAUSAL>
__global__ void opus_mla_decode_a8w8_flat_32mx4_kernel(opus_mla_decode_a8w8_flat_kargs)
{
}

#else

#include <bit>
#include <cstdint>
#include <utility>

namespace opus_mla_flat {

using i32x2  = int __attribute__((ext_vector_type(2)));
using f32x2  = float __attribute__((ext_vector_type(2)));
using i32x4  = int __attribute__((ext_vector_type(4)));
using u32x2  = unsigned __attribute__((ext_vector_type(2)));
using u32x4  = unsigned __attribute__((ext_vector_type(4)));
using i32x8  = int __attribute__((ext_vector_type(8)));
using f32x16 = float __attribute__((ext_vector_type(16)));
using u16x4  = unsigned short __attribute__((ext_vector_type(4)));

#define OPUS_MLA_FLAT_LDS __attribute__((address_space(3)))

constexpr int KT    = 64; // tokens per KV tile
constexpr int NSLOT = 4;  // KV ring slots
// Slot layout: 1 KB block (ug, tg) = units 4ug..4ug+3 (bytes 64ug..64ug+63) of tokens
// 16tg..16tg+15, at (4ug + tg) KB; inside it unit 4ug + k of token 16tg + t sits at 256k + 16((t +
// k) % 16) (the rotation keeps the QK and V^T reads conflict-free). DMA ug of wave tg fills block
// (ug, tg): lane l holds unit k = l / 16 of token (l - k) % 16, i.e. a fixed token per lane and 64
// contiguous bytes per token.
constexpr int NUNIT  = 36; // 16-byte units per token (576 B)
constexpr int BLK    = 1024;
constexpr int SLOT_B = NUNIT * KT * 16; // 36864
constexpr int NWAVE  = 4;
constexpr int IDX_B  = KT * 4; // one tile of page indices (per wave)
constexpr int SLOT_S =
    SLOT_B + NWAVE * IDX_B; // slot stride: KV tile + the waves' page indices of the same tile
constexpr int LDS_BYTES = NSLOT * SLOT_S;
// Epilogue O staging (per wave, 32 rows x one 32-column block of bf16): the MFMA layout gives each
// lane one row, so direct stores put 4 rows into every lane quad (4 lines per quad, ~2x the store
// issue time); through LDS every quad stores 64 contiguous bytes. Pitch 80 B keeps the b64 writes
// and b128 reads conflict-free.
constexpr int OST_PITCH    = 80;
constexpr int OST_B        = 32 * OST_PITCH;
constexpr int LDS_ALLOC    = LDS_BYTES + NWAVE * OST_B;
constexpr int DMA_PER_WAVE = NUNIT / 4; // unit groups
constexpr int SLOT_ROWS    = 128;       // rows per partial slot (matches the reduce kernel)
constexpr int D_HEAD       = 576;
constexpr int D_V          = 512;

constexpr float RESCALE_THRESHOLD = 8.0f;
constexpr float NEG_INF           = -__builtin_inff();

__device__ inline float readfirstlane_f32(float v)
{ return std::bit_cast<float>(__builtin_amdgcn_readfirstlane(std::bit_cast<int>(v))); }

__device__ inline f32x16 mfma32(const i32x8& a, const i32x8& b, const f32x16& c)
{ return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c, 0, 0, 0, 0, 0, 0); }

// PV MFMA as inline asm so that O is allocated in AGPRs and never copied (the builtin is selected
// in VGPR form). The hazard recognizer cannot see this MFMA: P is written >= 2 slots before its
// first PV use (VALU write -> MFMA read), o_settle() covers MFMA write -> VALU / accvgpr read of O.
__device__ inline void mfma32_acc(const i32x8& a, const i32x8& b, f32x16& c)
{ asm("v_mfma_f32_32x32x64_f8f6f4 %0, %1, %2, %0" : "+a"(c) : "v"(a), "v"(b)); }

__device__ inline void o_settle(f32x16 (&o)[16])
{
    asm volatile("s_nop 7\n\ts_nop 7\n\ts_nop 7"
                 : "+a"(o[0]),
                   "+a"(o[1]),
                   "+a"(o[2]),
                   "+a"(o[3]),
                   "+a"(o[4]),
                   "+a"(o[5]),
                   "+a"(o[6]),
                   "+a"(o[7]),
                   "+a"(o[8]),
                   "+a"(o[9]),
                   "+a"(o[10]),
                   "+a"(o[11]),
                   "+a"(o[12]),
                   "+a"(o[13]),
                   "+a"(o[14]),
                   "+a"(o[15]));
}

// LDS DMA: M0 = base + MOFF, lane l lands at M0 + IOFF + 16*l (4*l for dword); IOFF
// is the instruction offset of the global address. GFX9 needs one wait state between the M0 write
// and the LDS DMA that reads it (invisible to the hazard recognizer here). s_add_u32 writes SCC,
// hence the clobber.
template <int MOFF, int IOFF>
__device__ inline void dma_b128(uint32_t base, int vo, __amdgpu_buffer_rsrc_t rsrc)
{
    asm volatile("s_add_u32 m0, %0, %3\n\t"
                 "s_nop 0\n\t"
                 "buffer_load_dwordx4 %1, %2, 0 offen offset:%4 lds"
                 :
                 : "s"(base), "v"(vo), "s"(rsrc), "i"(MOFF - IOFF), "i"(IOFF)
                 : "memory", "scc");
}

// voffset = 4 * pos; returns the voffset register. A VALU write to it while the DMA still waits in
// the TA queue (behind the KV DMAs) stalls the wave, so the caller keeps it alive for a few slots.
__device__ inline int dma_b32(uint32_t m0, int pos, __amdgpu_buffer_rsrc_t rsrc)
{
    int vo;
    asm volatile("s_mov_b32 m0, %1\n\t"
                 "v_lshlrev_b32 %0, 2, %2\n\t"
                 "buffer_load_dword %0, %3, 0 offen lds"
                 : "=&v"(vo)
                 : "s"(m0), "v"(pos), "s"(rsrc)
                 : "memory");
    return vo;
}

template <int N>
__device__ inline void wait_vmcnt_barrier()
{ asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier" ::"n"(N) : "memory"); }

template <class T>
__device__ inline T lds_load(uint32_t addr)
{ return *reinterpret_cast<OPUS_MLA_FLAT_LDS const T*>(static_cast<uintptr_t>(addr)); }

__device__ inline i32x2 lds_tr8(uint32_t addr)
{
    return __builtin_amdgcn_ds_read_tr8_b64_v2i32(
        reinterpret_cast<OPUS_MLA_FLAT_LDS i32x2*>(static_cast<uintptr_t>(addr)));
}

// a * b - c as one scalar VALU op (the SLP vectorizer would pair these into v_pk_fma_f32)
__device__ inline float fma_scalar(float a, float b, float c)
{
    float r;
    asm("v_fma_f32 %0, %1, %2, -%3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}

__device__ inline float permlane32_max(float v)
{
    auto r = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<unsigned>(v), std::bit_cast<unsigned>(v), false, true);
    return fmaxf(std::bit_cast<float>(r[0]), std::bit_cast<float>(r[1]));
}

__device__ inline float permlane32_sum(float v)
{
    auto r = __builtin_amdgcn_permlane32_swap(
        std::bit_cast<unsigned>(v), std::bit_cast<unsigned>(v), false, true);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

__device__ inline unsigned pk_bf16(float a, float b)
{
    using bf16x2 = __bf16 __attribute__((ext_vector_type(2)));
    return std::bit_cast<unsigned>(__builtin_convertvector(f32x2{a, b}, bf16x2));
}

struct Ctx
{
    int lane, m, g, w;
    uint32_t k_ofs[2]; // QK operand read bases inside a slot (low / high 16 bytes)
    uint32_t v_ofs[2]; // V^T tr-read bases inside a slot (even / odd d-block)
    uint32_t lds0;     // LDS byte address of the KV ring
    uint32_t idxw;     // slot-relative offset of this wave's page indices
    // DMA instruction j (k = w + 4j), lane l fills slot unit U = 64k + l = token t * 37 + unit du
    int pofs; // slot-relative offset of the page index of this lane's DMA token
    int vo_k; // 16k: byte offset of this lane's DMA unit inside the 64-byte segment
};

// Buffer resources shared by all items.
struct Bufs
{ __amdgpu_buffer_rsrc_t kv_rsrc, idx_rsrc; };

// State of one work item, shared by all iterations. The items of a WG form one tile stream: item
// tile t is stream tile base + t (ring slot (base + t) & 3). Tiles t >= nt are the next item's
// tiles t - nt when it is prefetched, else this item's last tokens again; either way they are only
// read by fully masked QK, so every iteration fetches unconditionally.
struct Item
{
    int kv_s, len, nt;
    int base;        // stream index of tile 0
    int last;        // kv_indices index of the item's last token
    int kvn0, lastn; // tiles t >= nt: first token index kvn0 + 64t, clamped to lastn
    int bound_g;     // last visible item-relative KV position of this lane's row, minus g
    float scale;
};

// Per-item row bookkeeping.
struct Rows
{ int b, qs, q_rows, pslot, n_clean; };

// Page indices of item tile `tile` (64 tokens, clamped) -> this wave's private idx ring. Tiles past
// the item end (NEXT only) are the next item's first tiles.
struct IdxArgs
{
    uint32_t m0;
    int pos;
};

__device__ inline IdxArgs idx_args(const Ctx& c, const Item& it, int tile, uint32_t slot)
{
    const bool nx = tile >= it.nt;
    const int b0  = __builtin_amdgcn_readfirstlane((nx ? it.kvn0 : it.kv_s) + tile * KT);
    const int lst = __builtin_amdgcn_readfirstlane(nx ? it.lastn : it.last);
    int pos       = b0 + c.lane;
    pos           = pos < lst ? pos : lst;
    return {static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(slot + c.idxw)), pos};
}

__device__ inline void
issue_idx(const Ctx& c, const Bufs& bf, const Item& it, int tile, uint32_t slot)
{
    const IdxArgs a = idx_args(c, it, tile, slot);
    dma_b32(a.m0, a.pos, bf.idx_rsrc);
}

// Page index of this lane's DMA token of the tile in `slot`.
__device__ inline int read_page(const Ctx& c, uint32_t slot)
{ return lds_load<int>(c.pofs + slot); }

// voffset of this lane's DMAs: page * 576 + 16k (unit group ug adds 64ug via the instruction
// offset)
__device__ inline int page_vo(const Ctx& c, int page)
{ return static_cast<int>(__umul24(page, D_HEAD)) + c.vo_k; }

// LDS address of the slot of stream tile g
__device__ inline uint32_t slot_addr(const Ctx& c, int g)
{ return static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(c.lds0 + (g & 3) * SLOT_S)); }

// a pair of per-lane LDS read bases
struct KBase
{ uint32_t lo, hi; };
__device__ inline KBase kbase(const Ctx& c, uint32_t slot)
{ return {slot + c.k_ofs[0], slot + c.k_ofs[1]}; }
__device__ inline KBase vbase(const Ctx& c, uint32_t slot)
{ return {slot + c.v_ofs[0], slot + c.v_ofs[1]}; }

// slots of stream tiles gi, gi+1, gi+2, gi+3 (rotated by one per iteration); kb = kbase(a1)
struct Ring
{
    uint32_t a0, a1, a2, a3;
    KBase kb;
};
__device__ inline void rotate(Ring& r, KBase kb_next) { r = {r.a1, r.a2, r.a3, r.a0, kb_next}; }

// LDS base of this wave's blocks (tg = w) in the target slot
__device__ inline uint32_t dma_dst(const Ctx& c, uint32_t slot)
{ return static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(slot + c.w * BLK)); }

// DMA j of this wave: block (ug = j, tg = w)
template <int J = 0>
__device__ inline void dma_kv_rt(uint32_t dst, int j, int vo, __amdgpu_buffer_rsrc_t rs)
{
    if constexpr(J < DMA_PER_WAVE)
    {
        if(j == J)
            dma_b128<J * NWAVE * BLK, J * 64>(dst, vo, rs);
        else
            dma_kv_rt<J + 1>(dst, j, vo, rs);
    }
}

__device__ inline void issue_kv(const Ctx& c, const Bufs& bf, const Item& it, int tile)
{
    const uint32_t slot = slot_addr(c, it.base + tile);
    const int vo        = page_vo(c, read_page(c, slot));
    const uint32_t dst  = dma_dst(c, slot);
#pragma unroll
    for(int j = 0; j < DMA_PER_WAVE; ++j)
        dma_kv_rt(dst, j, vo, bf.kv_rsrc);
}

__device__ inline void zero(f32x16 (&s)[2])
{
#pragma unroll
    for(int mb = 0; mb < 2; ++mb)
#pragma unroll
        for(int i = 0; i < 16; ++i)
            s[mb][i] = 0.0f;
}

// Mask (if needed) and reduce the raw row max of one tile's scores; returns max * scale.
template <bool MASK>
__device__ inline float tile_max(const Ctx& c, const Item& it, f32x16 (&s)[2], int tile)
{
    if constexpr(MASK)
    {
        // value (mb, i) of this lane is item-relative token tile*64 + mb*32 + 2i + g
        const int rel = it.bound_g - tile * KT;
#pragma unroll
        for(int mb = 0; mb < 2; ++mb)
#pragma unroll
            for(int i = 0; i < 16; ++i)
                s[mb][i] = (mb * 32 + 2 * i <= rel) ? s[mb][i] : NEG_INF;
    }
    float mx = fmaxf(s[0][0], s[1][0]);
#pragma unroll
    for(int i = 1; i < 16; ++i)
        mx = fmaxf(mx, fmaxf(s[0][i], s[1][i]));
    return permlane32_max(mx) * it.scale;
}

// Lazy online-softmax rescale: only when some row's max grew by more than the threshold.
__device__ inline void
rescale_apply(float nm, f32x16 (&o)[16], float& m_row, float& l_row, bool o_live)
{
    const float alpha = (nm == NEG_INF) ? 1.0f : __builtin_amdgcn_exp2f(m_row - nm);
    if(o_live)
    {
        o_settle(o);
        // one 16-register chunk through VGPRs at a time: the volatile fences keep the
        // compiler from reading all of O at once (which would push Q out to scratch)
#pragma unroll
        for(int db = 0; db < 16; ++db)
        {
            asm volatile("" : "+a"(o[db]));
#pragma unroll
            for(int i = 0; i < 16; ++i)
                o[db][i] *= alpha;
            asm volatile("" : "+a"(o[db]));
        }
    }
    l_row *= alpha;
    m_row = nm;
}

__device__ inline bool rescale_needed(float nm, float m_row)
{ return __builtin_amdgcn_ballot_w64(nm > m_row + RESCALE_THRESHOLD) != 0; }

__device__ inline void
rescale(float new_m, f32x16 (&o)[16], float& m_row, float& l_row, bool o_live)
{
    const float nm = fmaxf(m_row, new_m);
    if(rescale_needed(nm, m_row))
        rescale_apply(nm, o, m_row, l_row, o_live);
}

// ---- one pipeline iteration (see the header), with an explicit schedule ----
// 34 MFMA slots (QK(i+1) steps 0..17, then PV(i) d-blocks 0..15) separated by sched_barrier(0), so
// that source order is issue order. The operands of slot x are read PF slots ahead; softmax(i)
// runs in the shadow of the QK slots, the row max of tile i+1 and the KV DMA in the shadow of PV.
// Every iteration issues the same VMEM (DMA past the item end is pointed out of bounds) and runs
// QK(i+1) even past the end (masked to -inf), so the steady state and the tail share one body.
constexpr int PF = 3;
// DMA(i+3) j in slot DMA_SLOT + DMA_STRIDE * j, the last one in the barrier slot; its page index is
// read in PAGE_SLOT, its LDS base and voffset computed 2 and 3 slots later
constexpr int DMA_STRIDE = 2;
constexpr int PAGE_SLOT  = 1;
constexpr int DMA_LAST   = 29;
constexpr int LSUM_SLOT  = 32; // l_row += the row sums of tile i
constexpr int DMA_SLOT   = DMA_LAST - DMA_STRIDE * (DMA_PER_WAVE - 1);
static_assert(DMA_LAST <= 30, "tile i+3 DMAs before the barrier's vmcnt");
static_assert(PAGE_SLOT + 3 <= DMA_SLOT, "DMA before its voffset");
constexpr int IDX_SLOT  = 22; // idx(i+5), its address computed in IDXA_SLOT
constexpr int IDX_HOLD  = 8;  // slots the idx DMA's voffset register stays reserved
constexpr int IDXA_SLOT = 18;
static_assert(IDXA_SLOT < IDX_SLOT, "idx DMA before its address");

// QK step st: token (st & 1) * 32 + sig(m), bytes (st >> 1) * 64 + 32g .. +32 = units 2g, 2g+1 of
// unit group st >> 1, token group sig / 16 + 2(st & 1)
__device__ inline i32x8 load_k(const KBase& kb, int st)
{
    const uint32_t o = (st >> 1) * 4 * BLK + (st & 1) * 2 * BLK;
    const i32x4 lo   = lds_load<i32x4>(kb.lo + o);
    const i32x4 hi   = lds_load<i32x4>(kb.hi + o);
    return i32x8{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
}

// bytes 32db.. = unit group db / 2; token rows +16r = token group r
__device__ inline i32x8 load_v(const KBase& vb, int db)
{
    const uint32_t a = ((db & 1) ? vb.hi : vb.lo) + (db >> 1) * 4 * BLK;
    const i32x2 r0   = lds_tr8(a);
    const i32x2 r1   = lds_tr8(a + BLK);
    const i32x2 r2   = lds_tr8(a + 2 * BLK);
    const i32x2 r3   = lds_tr8(a + 3 * BLK);
    return i32x8{r0[0], r0[1], r1[0], r1[1], r2[0], r2[1], r3[0], r3[1]};
}

// S^T(tile) = K Q^T for a whole tile with the same PF-deep operand pipeline (prologue only).
__device__ __attribute__((always_inline)) void
qk_tile(KBase kb, const i32x8 (&q)[9], f32x16 (&s)[2])
{
    i32x8 op[18];
#pragma unroll
    for(int x = 0; x < PF; ++x)
        op[x] = load_k(kb, x);
    __builtin_amdgcn_sched_barrier(0);
    zero(s);
#pragma unroll
    for(int x = 0; x < 18; ++x)
    {
        if(x + PF < 18)
            op[x + PF] = load_k(kb, x + PF);
        s[x & 1] = mfma32(op[x], q[x >> 1], s[x & 1]);
        __builtin_amdgcn_sched_barrier(0);
    }
}

// kpre: the first PF K operands of QK(i+1) on entry, of QK(i+2) on exit (read after the slot-30
// barrier, which is also what makes tile i+2 visible and frees slot (i-1)&3 for DMA(i+4)).
// IDLE: a wave without valid rows only does its share of the DMAs and the barrier.
template <bool MASK, bool IDLE = false>
__device__ __attribute__((always_inline)) void iteration(const Ctx& c,
                                                         const Bufs& bf,
                                                         const Item& it,
                                                         const i32x8 (&q)[9],
                                                         f32x16 (&s_cur)[2],
                                                         f32x16 (&s_nxt)[2],
                                                         f32x16 (&o)[16],
                                                         float& m_row,
                                                         float& l_row,
                                                         i32x8 (&kpre)[PF],
                                                         Ring& rg,
                                                         int i)
{
    // the slot-dependent addresses are computed where the schedule has spare issue
    const KBase kb = rg.kb;
    KBase kb2{}, vb{};
    // DMA(i+3) and its page indices: slot a3; idx(i+5): slot a1
    uint32_t dst = 0;
    int pg = 0, vo = 0, idx_vo = 0;
    IdxArgs ia{};
    i32x8 op[34];
#pragma unroll
    for(int x = 0; x < PF; ++x)
        op[x] = kpre[x];
    const float m_use = (m_row == NEG_INF) ? 0.0f : m_row;
    float sum0 = 0.0f, sum1 = 0.0f;
    float a0[16], a1[16];
    i32x8 p;
    float mx = NEG_INF, nm = NEG_INF;
    bool need = false;
    if constexpr(!IDLE)
        zero(s_nxt);
    __builtin_amdgcn_sched_barrier(0);

    // one MFMA slot; instantiated per x (a 34-trip unroll of the generic body can exceed the
    // pragma-unroll size threshold, which leaves op/a0/a1/page dynamically indexed on the stack)
    auto slot = [&]<int x>() __attribute__((always_inline)) {
        if constexpr(x == IDXA_SLOT)
            ia = idx_args(c, it, i + 5, rg.a1);
        if constexpr(x == 1 && !IDLE)
            vb = vbase(c, rg.a0);
        if constexpr(x == 29 && !IDLE)
            kb2 = kbase(c, rg.a2);
        if constexpr(x == PAGE_SLOT)
            pg = read_page(c, rg.a3);
        if constexpr(x == PAGE_SLOT + 2)
            dst = dma_dst(c, rg.a3);
        if constexpr(x == PAGE_SLOT + 3)
            vo = page_vo(c, pg);
        if constexpr(!IDLE)
        {
            if constexpr(x + PF < 18)
                op[x + PF] = load_k(kb, x + PF);
            else if constexpr(x + PF < 34)
                op[x + PF] = load_v(vb, x + PF - 18);
            if constexpr(x >= 34 - PF)
                kpre[x - (34 - PF)] = load_k(kb2, x - (34 - PF));

            if constexpr(x < 18)
                s_nxt[x & 1] = mfma32(op[x], q[x >> 1], s_nxt[x & 1]);
            else
                mfma32_acc(op[x], p, o[x - 18]);
        }

        // softmax(i), two scores per pair, pair n = slot pair (2d, 2d+1) of M-block mb -> P dword
        // mb*4+d; staged fma (slot n) -> exp (n+1) -> pack + row sum (n+2) so no slot waits on
        // its own transcendental results. Scalar ops only: packed FP32 VALU stalls behind MFMA.
        if constexpr(x < 16 && !IDLE)
        {
            const int mb = x >> 3, d = (x >> 1) & 3, h = x & 1;
            a0[x] = fma_scalar(s_cur[mb][4 * d + 2 * h], it.scale, m_use);
            a1[x] = fma_scalar(s_cur[mb][4 * d + 2 * h + 1], it.scale, m_use);
        }
        if constexpr(x >= 1 && x <= 16 && !IDLE)
        {
            a0[x - 1] = __builtin_amdgcn_exp2f(a0[x - 1]);
            a1[x - 1] = __builtin_amdgcn_exp2f(a1[x - 1]);
        }
        if constexpr(x >= 2 && x <= 17 && !IDLE)
        {
            const int n = x - 2, mb = n >> 3, d = (n >> 1) & 3, h = n & 1;
            if constexpr(h ==
                         0) // the high half is written by h = 1: any old value (p is undef here)
                p[mb * 4 + d] = __builtin_amdgcn_cvt_pk_fp8_f32(a0[n], a1[n], p[mb * 4 + d], false);
            else
                p[mb * 4 + d] = __builtin_amdgcn_cvt_pk_fp8_f32(a0[n], a1[n], p[mb * 4 + d], true);
            sum0 += a0[n];
            sum1 += a1[n];
            asm volatile("" : "+v"(sum0), "+v"(sum1)); // keep the row sum in this slot
        }
        if constexpr(x == IDX_SLOT)
            idx_vo = dma_b32(ia.m0, ia.pos, bf.idx_rsrc);
        if constexpr(x == IDX_SLOT + IDX_HOLD)
            asm volatile("" ::"v"(idx_vo));

        // row max of tile i+1 once its QK is done
        if constexpr(x >= 20 && x < 28 && !IDLE)
        {
            const int k = x - 20;
            if constexpr(MASK)
            {
                // value (mb, i) is item-relative token (i+1)*64 + mb*32 + 2i + g
                const int rel = it.bound_g - (i + 1) * KT;
#pragma unroll
                for(int mb = 0; mb < 2; ++mb)
#pragma unroll
                    for(int h = 0; h < 2; ++h)
                        s_nxt[mb][2 * k + h] =
                            (mb * 32 + 4 * k + 2 * h <= rel) ? s_nxt[mb][2 * k + h] : NEG_INF;
            }
            mx = fmaxf(mx,
                       fmaxf(fmaxf(s_nxt[0][2 * k], s_nxt[0][2 * k + 1]),
                             fmaxf(s_nxt[1][2 * k], s_nxt[1][2 * k + 1])));
        }
        if constexpr(x == 28 && !IDLE)
            mx = permlane32_max(mx) * it.scale;
        if constexpr(x == LSUM_SLOT && !IDLE)
        {
            l_row += sum0 + sum1;
            asm volatile("" : "+v"(l_row));
        }
        if constexpr(x == 31 && !IDLE)
        {
            nm   = fmaxf(m_row, mx);
            need = rescale_needed(nm, m_row);
        }

        if constexpr(x >= DMA_SLOT && (x - DMA_SLOT) % DMA_STRIDE == 0)
            dma_kv_rt(dst, (x - DMA_SLOT) / DMA_STRIDE, vo, bf.kv_rsrc);

        // own DMA(i+2) landed (only idx(i+5) and DMA(i+3) may be in flight), then everyone's
        if constexpr(x == 34 - PF - 1)
            wait_vmcnt_barrier<DMA_PER_WAVE + 1>();
        __builtin_amdgcn_sched_barrier(0);
    };
    [&]<int... X>(std::integer_sequence<int, X...>) __attribute__((always_inline)) {
        (slot.template operator()<X>(), ...);
    }(std::make_integer_sequence<int, 34>{});
    if(__builtin_expect(need, 0))
        rescale_apply(nm, o, m_row, l_row, true);
    rotate(rg, kb2);
}

// Scalar description of item wi (everything the previous item needs to prefetch it).
__device__ __attribute__((always_inline)) void
item_scalars(const opus_mla_decode_a8w8_flat_kargs& kargs, int wi, float scale, Item& it, Rows& r)
{
    const opus_mla_decode_work_info item = kargs.work_info_set[wi];
    // SGPRs on every path: a VGPR input to the item phi of the WG loop moves the whole item (and
    // the index-DMA address math of every iteration) to VGPRs
    auto u   = [](int v) { return __builtin_amdgcn_readfirstlane(v); };
    r.b      = u(item.batch_idx);
    r.pslot  = u(item.partial_slot);
    r.qs     = u(item.qo_start);
    r.q_rows = u(item.qo_end) - r.qs;
    it.kv_s  = u(item.kv_start);
    it.len   = u(item.kv_end) - it.kv_s;
    it.nt    = (it.len + KT - 1) / KT;
    it.last  = it.kv_s + it.len - 1;
    it.kvn0  = it.kv_s;
    it.lastn = it.last;
    it.scale = scale;
}

__device__ inline bool row_valid(const Ctx& c, const Rows& r) { return c.w * 32 + c.m < r.q_rows; }

// Per-lane causal bound and the wave's unmasked tile count (computed when the item starts).
template <bool CAUSAL>
__device__ __attribute__((always_inline)) void
item_lanes(const opus_mla_decode_a8w8_flat_kargs& kargs, const Ctx& c, Item& it, Rows& r)
{
    int row_local = c.w * 32 + c.m;
    asm volatile("" : "+v"(row_local)); // per-item recompute: keep lane invariants out of the loop
    int bound = it.len - 1;
    if constexpr(CAUSAL)
    {
        const int b     = r.b;
        const int kv_b0 = __builtin_amdgcn_readfirstlane(kargs.kv_indptr[b]);
        const int q_b0  = __builtin_amdgcn_readfirstlane(kargs.q_indptr[b]);
        const int q_b1  = __builtin_amdgcn_readfirstlane(kargs.q_indptr[b + 1]);
        const int g_len = __builtin_amdgcn_readfirstlane(kargs.g_kv_indptr[b + 1]) -
                          __builtin_amdgcn_readfirstlane(kargs.g_kv_indptr[b]);
        const int diag  = kv_b0 - it.kv_s;
        const int dcp0  = g_len - (q_b1 - q_b0) - kargs.cp_rank;
        const int row0  = r.qs - q_b0 * kargs.H;
        int H = kargs.H, cpw = kargs.cp_world_size;
        asm volatile("" : "+s"(H), "+s"(cpw));
        const int num = dcp0 + (row0 + (row_local < r.q_rows ? row_local : 0)) / H;
        const int tq  = num < 0 ? -1 : num / cpw;
        const int cb  = tq + diag;
        bound         = cb < bound ? cb : bound;
    }
    int g = c.g;
    asm volatile("" : "+v"(g));
    it.bound_g = bound - g;
    // tiles [0, n_clean) need no mask on any lane of this wave
    int bmin = bound;
#pragma unroll
    for(int sh = 32; sh >= 1; sh >>= 1)
        bmin = min(bmin, __shfl_xor(bmin, sh));
    bmin      = __builtin_amdgcn_readfirstlane(bmin);
    r.n_clean = __builtin_amdgcn_readfirstlane(bmin < 0 ? 0 : (bmin + 1) / KT);
}

// Q B-operand: lane (row = l%32, g) holds Q[row][kc*64 + g*32 .. +32) for kc = 0..8. Rows >= q_rows
// load row 0 instead: rows stay independent through softmax and PV, and those rows are never
// stored, so they need no zeroing (which would wait for the loads, and with them every DMA in
// flight, inside the epilogue).
__device__ __attribute__((always_inline)) void
load_q(const opus_mla_decode_a8w8_flat_kargs& kargs, const Ctx& c, const Rows& r, i32x8 (&q)[9])
{
    int row_local = c.w * 32 + c.m;
    asm volatile("" : "+v"(row_local));
    int lofs        = (r.qs + (row_local < r.q_rows ? row_local : 0)) * D_HEAD + c.g * 32;
    const i32x4* qp = reinterpret_cast<const i32x4*>(
        reinterpret_cast<const uint8_t*>(kargs.q_buffer_ptr) + static_cast<unsigned>(lofs));
#pragma unroll
    for(int kc = 0; kc < 9; ++kc)
    {
        const i32x4 lo = qp[kc * 4];
        const i32x4 hi = qp[kc * 4 + 1];
        q[kc]          = i32x8{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
    }
}

// One item. warm: the previous item already fetched page indices 0..4 and KV tiles 0..2 (tiles 0
// and 1 landed everywhere) and loaded q. pf: this item's tail prefetches the next one (rn), and q
// is reloaded with the next item's Q before the epilogue.
template <bool CAUSAL>
__device__ __attribute__((always_inline)) void
run_item(const opus_mla_decode_a8w8_flat_kargs& kargs,
         const Ctx& c,
         const Bufs& bf,
         Item& it,
         Rows& r,
         i32x8 (&q)[9],
         bool warm,
         bool pf,
         int wi_next,
         Item& nx,
         Rows& rn,
         float descale_k)
{
    const int row_local = c.w * 32 + c.m;
    {
        // the WG loop may carry these in VGPRs; the iterations need them in SGPRs
        auto u = [](int& v) { v = __builtin_amdgcn_readfirstlane(v); };
        u(it.kv_s), u(it.len), u(it.nt), u(it.base), u(it.last), u(it.kvn0), u(it.lastn);
        u(r.b), u(r.qs), u(r.q_rows), u(r.pslot), u(r.n_clean);
    }

    f32x16 o[16];
#pragma unroll
    for(int db = 0; db < 16; ++db)
#pragma unroll
        for(int i = 0; i < 16; ++i)
            o[db][i] = 0.0f;
    float m_row = NEG_INF;
    float l_row = 0.0f;
    f32x16 s[2][2];

    if(!warm)
    {
        // cold: everyone is done with the previous item's slots; idx 0..4, KV 0..2
        asm volatile("s_waitcnt vmcnt(0)\n\ts_barrier" ::: "memory");
#pragma unroll
        for(int t = 0; t < 4; ++t)
            issue_idx(c, bf, it, t, slot_addr(c, it.base + t));
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        issue_kv(c, bf, it, 0);
        issue_kv(c, bf, it, 1);
        issue_kv(c, bf, it, 2);
        issue_idx(c, bf, it, 4, slot_addr(c, it.base + 4));
        wait_vmcnt_barrier<DMA_PER_WAVE + 1>(); // KV 0 and 1 landed everywhere
    }
    Ring rg = {slot_addr(c, it.base),
               slot_addr(c, it.base + 1),
               slot_addr(c, it.base + 2),
               slot_addr(c, it.base + 3),
               {}};
    i32x8 kpre[PF];
    // the padded tail of a request (576 rows = 4.5 items): waves without rows skip the math
    if(__builtin_amdgcn_readfirstlane(c.w * 32) >= r.q_rows)
    {
        for(int i = 0; i < it.nt; ++i)
            iteration<false, true>(c, bf, it, q, s[0], s[1], o, m_row, l_row, kpre, rg, i);
    }
    else
    {
        rg.kb = kbase(c, rg.a1);
        qk_tile(kbase(c, rg.a0), q, s[0]);
        {
            const float mx = tile_max<true>(c, it, s[0], 0);
            rescale(mx, o, m_row, l_row, false);
        }
#pragma unroll
        for(int x = 0; x < PF; ++x)
            kpre[x] = load_k(rg.kb, x);

        // iteration i is clean if tile i+1 needs no mask (i + 1 < n_clean)
        int i = 0;
        for(; i + 3 <= r.n_clean; i += 2)
        {
            iteration<false>(c, bf, it, q, s[0], s[1], o, m_row, l_row, kpre, rg, i);
            iteration<false>(c, bf, it, q, s[1], s[0], o, m_row, l_row, kpre, rg, i + 1);
        }
        for(; i + 1 < it.nt; i += 2)
        {
            iteration<true>(c, bf, it, q, s[0], s[1], o, m_row, l_row, kpre, rg, i);
            iteration<true>(c, bf, it, q, s[1], s[0], o, m_row, l_row, kpre, rg, i + 1);
        }
        if(i < it.nt)
            iteration<true>(c, bf, it, q, s[0], s[1], o, m_row, l_row, kpre, rg, i);
    }

    // next item: descriptor, causal bounds and Q, in the shadow of the epilogue (always, so the
    // old q is dead here on every path)
    if(wi_next >= 0)
    {
        item_scalars(kargs, wi_next, it.scale, nx, rn);
        nx.base = it.base + it.nt;
        item_lanes<CAUSAL>(kargs, c, nx, rn);
        load_q(kargs, c, rn, q);
    }

    // --- epilogue: normalise, write partial (or final) O and LSE ---
    const float l_tot   = permlane32_sum(l_row);
    const float o_scale = l_tot > 0.0f ? descale_k / l_tot : 0.0f;
    constexpr float LN2 = 0.69314718055994531f;
    const float lse     = l_tot > 0.0f ? (m_row + __builtin_log2f(l_tot)) * LN2 : -3.0e38f;

    unsigned short* obase;
    float* lbase;
    if(r.pslot >= 0)
    {
        obase = reinterpret_cast<unsigned short*>(kargs.o_accum) +
                static_cast<size_t>(r.pslot) * SLOT_ROWS * D_V;
        lbase =
            reinterpret_cast<float*>(kargs.lse_accum) + static_cast<size_t>(r.pslot) * SLOT_ROWS;
    }
    else
    {
        obase = reinterpret_cast<unsigned short*>(kargs.out_ptr) + static_cast<size_t>(r.qs) * D_V;
        lbase = kargs.lse_ptr ? reinterpret_cast<float*>(kargs.lse_ptr) + r.qs : nullptr;
    }
    o_settle(o);
    // lane (m, g) holds O[m][db*32 + i4*8 + g*4 + k]: each 8-byte bf16 piece goes to the wave's
    // staging tile, then lane L reads 16 bytes of row L/4 (+16) and the 4 lanes of a quad store 64
    // contiguous bytes. The whole wave takes part (lanes store other lanes' rows); rows >= q_rows
    // fall outside the buffer range.
    {
        const int lane      = c.lane;
        const uint32_t st   = c.lds0 + LDS_BYTES + c.w * OST_B;
        const uint32_t wofs = st + (lane & 31) * OST_PITCH + (lane >> 5) * 8;
        const uint32_t rofs = st + (lane >> 2) * OST_PITCH + (lane & 3) * 16;
        const auto orsrc =
            __builtin_amdgcn_make_buffer_rsrc(obase, 0, r.q_rows * D_V * 2, 0x00020000);
        const int vofs = ((c.w * 32 + (lane >> 2)) * D_V + (lane & 3) * 8) * 2;
        unsigned dep   = 0;
        u32x4 v0, v1;
        // software-pipelined by one block: block db's pieces are written (DS ops of a wave execute
        // in order, so they land after block db-1's reads), then block db-1 is stored
#pragma unroll
        for(int db = 0; db < 16; ++db)
        {
            // read O one 16-register chunk at a time (the next item's Q is already in flight)
            asm volatile("" : "+a"(o[db]) : "v"(dep));
            const f32x16 od = o[db];
#pragma unroll
            for(int i4 = 0; i4 < 4; ++i4)
            {
                const unsigned a = pk_bf16(od[i4 * 4 + 0] * o_scale, od[i4 * 4 + 1] * o_scale);
                const unsigned b = pk_bf16(od[i4 * 4 + 2] * o_scale, od[i4 * 4 + 3] * o_scale);
                *reinterpret_cast<OPUS_MLA_FLAT_LDS u32x2*>(
                    static_cast<uintptr_t>(wofs + i4 * 16)) = u32x2{a, b};
                dep                                         = b;
            }
            if(db > 0)
            {
                __builtin_amdgcn_raw_buffer_store_b128(v0, orsrc, vofs + (db - 1) * 64, 0, 0);
                __builtin_amdgcn_raw_buffer_store_b128(
                    v1, orsrc, vofs + 16 * D_V * 2 + (db - 1) * 64, 0, 0);
            }
            v0 = lds_load<u32x4>(rofs);
            v1 = lds_load<u32x4>(rofs + 16 * OST_PITCH);
        }
        __builtin_amdgcn_raw_buffer_store_b128(v0, orsrc, vofs + 15 * 64, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b128(v1, orsrc, vofs + 16 * D_V * 2 + 15 * 64, 0, 0);
        if(row_valid(c, r) && c.g == 0 && lbase != nullptr)
            lbase[row_local] = lse;
    }
}

} // namespace opus_mla_flat

template <bool CAUSAL>
__global__ __launch_bounds__(256, 1) void opus_mla_decode_a8w8_flat_32mx4_kernel(
    opus_mla_decode_a8w8_flat_kargs kargs)
{
    using namespace opus_mla_flat;
    __shared__ __attribute__((aligned(1024))) uint8_t smem[LDS_ALLOC];

    Ctx c;
    c.lane = threadIdx.x & 63;
    c.w    = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    c.m    = c.lane & 31;
    c.g    = c.lane >> 5;
    {
        const int m   = c.m;
        const int sig = (((m >> 4) & 1) << 4) | (((m >> 3) & 1) << 3) | (((m >> 1) & 1) << 2) |
                        ((m & 1) << 1) | ((m >> 2) & 1);
        // unit k of token t of a token group: 64 p(t) + 16((k + r(t)) % 4) with r = (t & 1) + 2(t
        // >> 3), p = ((t >> 1) & 3) + 4r: each DMA lane quad fills one token's 64 contiguous bytes
        // (one L1 access instead of four) and the QK b128 / V^T tr8 reads stay conflict-free
        auto at = [](int t, int k) {
            const int tt = t & 15, r = (tt & 1) + 2 * (tt >> 3), p = ((tt >> 1) & 3) + 4 * r;
            return (t >> 4) * BLK + p * 64 + ((k + r) & 3) * 16;
        };
        for(int h = 0; h < 2; ++h)
            c.k_ofs[h] = at(sig, 2 * c.g + h);
        // V^T tr read: token 2bb + (qq >> 1) (+16r), unit 2db + (qq & 1) (bytes +8hh)
        const int L = c.lane, qq = L >> 4, bb = (L & 15) >> 1, hh = L & 1;
        for(int h = 0; h < 2; ++h)
            c.v_ofs[h] = at(2 * bb + (qq >> 1), 2 * h + (qq & 1)) + hh * 8;
    }
    c.lds0 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>((OPUS_MLA_FLAT_LDS uint8_t*)(smem)));
    c.idxw = SLOT_B + c.w * IDX_B;
    {
        // lane l = 4p + j lands at 64p + 16j: token t(p), unit (j - r) % 4
        const int r = c.lane >> 4;
        const int t = 2 * ((c.lane >> 2) & 3) + (r & 1) + 8 * (r >> 1);
        const int k = ((c.lane & 3) - r) & 3;
        c.pofs      = static_cast<int>(c.idxw) + (c.w * 16 + t) * 4;
        c.vo_k      = k * 16;
    }

    Bufs bf;
    bf.kv_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(kargs.kv_buffer_ptr),
        0,
        static_cast<int>(static_cast<unsigned>(kargs.total_tokens) * D_HEAD),
        0x00020000);
    bf.idx_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<int*>(kargs.kv_indices), 0, 0x7FFFFFFF, 0x00020000);

    const int wg           = blockIdx.x;
    const int beg          = kargs.work_indptr[wg];
    const int end          = kargs.work_indptr[wg + 1];
    constexpr float LOG2_E = 1.44269504089f;
    const float descale_q = readfirstlane_f32(reinterpret_cast<const float*>(kargs.q_scale_ptr)[0]);
    const float descale_k =
        readfirstlane_f32(reinterpret_cast<const float*>(kargs.kv_scale_ptr)[0]);
    const float scale = readfirstlane_f32(kargs.softmax_scale * LOG2_E * descale_q * descale_k);
    // per-WG clock (kargs.wg_clock, when set)
    const unsigned long long fb_c0 = __builtin_amdgcn_s_memtime();
    const unsigned long long fb_r0 = __builtin_amdgcn_s_memrealtime();
    i32x8 q[9];
    Item cur;
    Rows rc;
    if(beg < end)
    {
        item_scalars(kargs, beg, scale, cur, rc);
        item_lanes<CAUSAL>(kargs, c, cur, rc);
        load_q(kargs, c, rc, q);
    }
    cur.base  = 0;
    bool warm = false;
    for(int wi = beg; wi < end; ++wi)
    {
        const int wn = wi + 1 < end ? wi + 1 : -1;
        // prefetching needs >= 5 tiles here (the next item's idx 0 is fetched by iteration nt-5)
        bool pf = false;
        if(wn >= 0 && cur.nt >= 5)
        {
            const opus_mla_decode_work_info nw = kargs.work_info_set[wn];
            const int ks                       = __builtin_amdgcn_readfirstlane(nw.kv_start);
            const int ke                       = __builtin_amdgcn_readfirstlane(nw.kv_end);
            pf                                 = ke > ks;
            if(pf)
            {
                cur.kvn0  = ks - cur.nt * KT;
                cur.lastn = ke - 1;
            }
        }
        Item nx;
        Rows rn;
        if(cur.nt > 0)
        {
            run_item<CAUSAL>(kargs, c, bf, cur, rc, q, warm, pf, wn, nx, rn, descale_k);
        }
        else if(wn >= 0)
        {
            item_scalars(kargs, wn, scale, nx, rn);
            nx.base = cur.base;
            item_lanes<CAUSAL>(kargs, c, nx, rn);
            load_q(kargs, c, rn, q);
        }
        warm = pf && cur.nt > 0;
        cur  = nx;
        rc   = rn;
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if(kargs.wg_clock != nullptr && threadIdx.x == 0 && beg < end)
    {
        const unsigned cyc  = static_cast<unsigned>(__builtin_amdgcn_s_memtime() - fb_c0);
        const unsigned tick = static_cast<unsigned>(__builtin_amdgcn_s_memrealtime() - fb_r0);
        unsigned* fb        = reinterpret_cast<unsigned*>(kargs.wg_clock) + 2 * wg;
        __hip_atomic_store(fb, cyc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        __hip_atomic_store(fb + 1, tick, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

#undef OPUS_MLA_FLAT_LDS

#endif

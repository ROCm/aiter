// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Splice source: fused last-arriver combine, spliced by asm/patch.py at the end of every SPLIT piece
// (after the ASM epilogue stored the piece's partial o_acc[slot] / lse_acc[slot], before the walk's barrier).
// Only compiled to assembly (hipcc -S). ABI of the splice: kernarg pointer s[0:1], thread id v0, exec = all.
// Inputs (uniform, AGPRs written by the wrapper): a230 row r, a231 slot, a232 partition p, a233 hg, a234 tb,
// a235 te (unclamped), a236 kv_indptr[r], a237 kv_indptr[r+1]; from pre.hip: a248 per, a249 Pa, a250 org.
// Kernarg: 0x00 o_acc, 0x08 out, 0x10 lse_acc, 0x18 lse_out (0 = none), 0x98 cnt (uint32, zero at rest:
// [0, 2*65536) row counters, [2*65536, +16*512) reserved (unused), [CNTG, +4*1024) group counters), 0xb0 planner cfg, 0x120 sink.
//
// Protocol (per row r and head group hg). The pieces of r live in partitions pf..pl (pf = owner(r, 0), pl = owner(r, T-1)
// on the planner's cost line); a partition q in between holds a piece iff its tile range [ftg(q*per), ftg((q+1)*per)) is
// non-empty (partitions may be empty when per < 32); slot(q) = 2q, except the first piece: 2pf + (r is not the begin row
// of pf), whose parity travels through the counters (bit 30, added by the piece with tb == 0).
//  * nq = pl - pf + 1 <= FIN_TREE_MIN ("narrow" row): every piece, after its partial is complete (s_waitcnt vmcnt(0) in
//    every wave + barrier), does ONE agent-scope atomic add on cnt[2r+hg] of ntiles (+ parity). The piece that brings the
//    tile sum to T(r) is the last arriver: it zeroes the counter (next launch starts from 0; ordered by the kernel
//    boundary) and merges all pieces (+ sink) into out (bf16) and lse_out.
//  * wider rows (two levels, fan-in FIN_G): partitions are grouped by (q - pf) / FIN_G; group counter
//    cntg[4*q0 + 2*(q0 == pf) + hg] (q0 = first partition of the group; unique per (row, group)); the group's last arriver
//    merges its pieces (no sink) into an fp32 partial + log2 lse stored IN PLACE in the slot of the group's first piece,
//    then adds the group's tiles (+ parity for group 0) to the row counter; the row's last arriver merges the group
//    partials (+ sink). The two levels spread the merge of wide rows over several CUs.
// Coherence: sc1 instruction scope instead of cache maintenance. The ASM's partial stores are built with sc1
// (patch.py sc1: agent-scope write-through = LLVM's "store atomic monotonic agent" encoding on gfx942/950), group
// partials are stored with sc1 buffer stores, and every merge read is an sc1 load ("load atomic monotonic agent"): the
// release is s_waitcnt vmcnt(0) + barrier before the counter atomic, the acquire needs no buffer_inv; no buffer_wbl2.
// Merge math (FlashMLA combine, log2 domain): Mx = max(sk, lse_z); S = exp2(sk-Mx) + sum exp2(lse_z-Mx);
// out = sum exp2(lse_z-Mx) o_z / S; lse_out = (Mx + log2 S) ln2. Wave w merges heads hg*64 + 16w .. +15.
#include "hca_common.hip"

typedef unsigned v4u __attribute__((ext_vector_type(4)));
struct KA { char b[336]; };
// kernarg reads through the constant-address-space kernarg pointer -> s_load (the by-value struct compiled to vector loads)
#define KARG(T, off) (*(T const __attribute__((address_space(4)))*)((const __attribute__((address_space(4))) char*)__builtin_amdgcn_kernarg_segment_ptr() + (off)))
template <int A> __device__ __forceinline__ int rda() {
  int v;
  asm volatile("v_accvgpr_read_b32 %0, a%c1" : "=v"(v) : "i"(A));
  return __builtin_amdgcn_readfirstlane(v);
}
template <class T> __device__ __forceinline__ T* uni(T* p) {   // make a pointer provably uniform (SGPR pair)
  const unsigned long long u = (unsigned long long)p;
  return (T*)(((unsigned long long)__builtin_amdgcn_readfirstlane((unsigned)(u >> 32)) << 32) |
              (unsigned)__builtin_amdgcn_readfirstlane((unsigned)u));
}
#define FIN_MAXJ 256
#define FIN_NPF 100000 // rows spanning more than FIN_NPF partitions would be left to an external combine: none
#define FIN_TREE_MIN 8 // rows spanning more partitions than this use the two-level merge
#define FIN_G 4        // fan-in of the first level
#define FIN_LR 4       // lse values per lane held in registers (pieces 0..15); more pieces are re-read from memory
#define CNTG (2 * 65536 + 16 * 512)
#define CPOL_SC1 16
#define OCP " sc1"

extern "C" __global__ void __launch_bounds__(256) ps_merge(KA ka) {
  __shared__ unsigned s_dec[2];
  __shared__ int s_slot[4][FIN_MAXJ];
  __shared__ float s_wt[4][FIN_MAXJ + 1][16];
  const int r = rda<230>(), slot = rda<231>(), p = rda<232>(), hg = rda<233>(), tb = rda<234>(), te = rda<235>();
  const int i0 = rda<236>(), i1 = rda<237>();
  const int per = rda<248>(), Pa = rda<249>(), org = rda<250>();
  float* o_acc = uni(KARG(float*, 0x00));
  __bf16* out = uni(KARG(__bf16*, 0x08));
  float* lse_acc = uni(KARG(float*, 0x10));
  float* lse_out = uni(KARG(float*, 0x18));
  unsigned* cnt = uni(KARG(unsigned*, 0x98));
  const int F = __builtin_amdgcn_readfirstlane(KARG(int, 0xb0));
  const float* sink = uni(KARG(const float*, 0x120));
  const int T = (i1 - i0 + TILE - 1) >> 5;
  const int nt = min(te, T) - tb;
  const int C0 = cfg_C0(F);
  const int b = i0 - org + r * C0;
  const int pf = min(Pa - 1, (b + C0) / per), pl = min(Pa - 1, (b + C0 + TILE * (T - 1)) / per);
  auto ftg = [&](int pos) { const int d = pos - b - C0; const int i = d <= 0 ? 0 : (d + TILE - 1) >> 5; return min(i, T); };
  auto t0of = [&](int q) { return q == pf ? 0 : ftg(q * per); };
  auto t1of = [&](int q) { return q == pl ? T : ftg((q + 1) * per); };
  const int nq = pl - pf + 1;
  if (nq > FIN_NPF) return;   // never taken (FIN_NPF exceeds any partition count); pre.hip mirrors it for desc.comb
  const bool tree = nq > FIN_TREE_MIN;
  const int gi = tree ? (p - pf) / FIN_G : 0;
  const int q0 = tree ? pf + gi * FIN_G : pf, q1 = tree ? min(pl, q0 + FIN_G - 1) : pl;
  unsigned* c1 = tree ? cnt + CNTG + 4 * q0 + 2 * (q0 == pf) + hg : cnt + 2 * r + hg;
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63;
  const int target1 = tree ? t1of(q1) - t0of(q0) : T;
  const unsigned par = tb == 0 ? (unsigned)(slot & 1) << 30 : 0u;

  // one counter step: all waves' stores complete -> barrier -> thread 0 atomic (+reset if last) -> barrier -> decision
  auto arrive = [&](unsigned* c, unsigned add, int target) -> unsigned {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned nw = __hip_atomic_fetch_add(c, add, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + add;
      const bool last = (int)(nw & 0x3fffffffu) == target;
      if (last) __hip_atomic_store(c, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      s_dec[0] = last ? 1u | ((nw >> 30) & 1u) << 1 : 0u;
    }
    __syncthreads();
    return s_dec[0];
  };
  unsigned dec = arrive(c1, (unsigned)nt | par, target1);
  if (!(dec & 1u)) return;

  const int hw0 = hg * 64 + 16 * w;       // first head of this wave
  const int hl = l & 15, jl = l >> 4;
  const __amdgpu_buffer_rsrc_t rl = __builtin_amdgcn_make_buffer_rsrc(lse_acc, 0, 0x7fffffff, 0x00020000);
  const __amdgpu_buffer_rsrc_t ro = __builtin_amdgcn_make_buffer_rsrc(o_acc, 0, 0x7fffffff, 0x00020000);
  auto ldl = [&](int j) {
    return __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rl, (s_slot[w][j] * NH + hw0 + hl) * 4, 0, CPOL_SC1));
  };
  // slots of the pieces in partitions [qa, qb] (lanes over partitions) -> s_slot[w][0..np), one private copy per wave
  auto slots_pieces = [&](int qa, int qb, unsigned pr) -> int {
    int np = 0;
    for (int j0 = qa; j0 <= qb; j0 += 64) {
      const int q = j0 + l;
      const bool has = q <= qb && t0of(q) < t1of(q);
      const unsigned long long m = __ballot(has);
      if (has) {
        const int k = np + __popcll(m & ((1ull << l) - 1));
        if (k < FIN_MAXJ) s_slot[w][k] = 2 * q + (q == pf ? (int)pr : 0);
      }
      np += __popcll(m);
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);   // lgkmcnt(0)
    __builtin_amdgcn_wave_barrier();
    return __builtin_amdgcn_readfirstlane(min(np, FIN_MAXJ));
  };

  // ---- merge s_slot[w][0..np) for heads hw0..hw0+15. final: + sink -> out bf16 / lse_out (natural log);
  //      !final: no sink -> fp32 partial + log2 lse written in place into slot s_slot[w][0] (sc1 stores) ----
  auto merge = [&](int np, bool final) {
    // weights: lane l = (piece j = jl + 4i, head hl); lse of pieces 0..15 in registers, one round trip
    const float sk = final ? sink[hw0 + hl] * LOG2E : -__builtin_inff();
    float lz[FIN_LR];
#pragma unroll
    for (int i = 0; i < FIN_LR; ++i) lz[i] = ldl(min(jl + 4 * i, np - 1));
    // O pipeline: 2 head groups (8 heads) x np pieces, batch b = g*npp + j (npp = np rounded up to even), 16 x 1 KB
    // sc1 loads per wave per batch into one of 2 register buffers; inline asm with explicit s_waitcnt vmcnt(16) before
    // each consume (vmcnt completes in order; stores in between only make the wait conservative). The compiler's
    // own waitcnt pass would drain vmcnt(0) at the loop head, serialising every batch.
    constexpr int HB = 8;
    const int npp = (np + 1) & ~1, nbt = 2 * npp;
    v4f buf0[HB][2], buf1[HB][2];
    auto issue = [&](int bb, v4f (&bf)[HB][2]) {
      bb = min(bb, nbt - 1);
      const int g = bb >= npp ? 1 : 0, jj = min(bb - g * npp, np - 1);
      const int sl = __builtin_amdgcn_readfirstlane(s_slot[w][jj]);
      const float* base = uni(o_acc + ((size_t)sl * NH + hw0 + g * HB) * DV);
      const int vo = 16 * l;
      // the hazard recognizer does not look into inline asm: VALU (v_readfirstlane) SGPR write -> VMEM saddr read
      // needs 5 wait states on gfx9
      asm volatile("s_nop 4" :: "s"(base));
#pragma unroll
      for (int k = 0; k < HB; ++k)
        asm volatile("global_load_dwordx4 %0, %2, %3" OCP "\n\tglobal_load_dwordx4 %1, %2, %3 offset:1024" OCP
                     : "=&v"(bf[k][0]), "=&v"(bf[k][1]) : "v"(vo + k * DV * 4), "s"(base) : "memory");
    };
#define FIN_TIE16(bf, CNT)                                                                                             \
    asm volatile("s_waitcnt vmcnt(" #CNT ")" : "+v"(bf[0][0]), "+v"(bf[0][1]), "+v"(bf[1][0]), "+v"(bf[1][1]),          \
                 "+v"(bf[2][0]), "+v"(bf[2][1]), "+v"(bf[3][0]), "+v"(bf[3][1]), "+v"(bf[4][0]), "+v"(bf[4][1]),            \
                 "+v"(bf[5][0]), "+v"(bf[5][1]), "+v"(bf[6][0]), "+v"(bf[6][1]), "+v"(bf[7][0]), "+v"(bf[7][1]) :: "memory")
    issue(0, buf0);
    float Mx = sk;
#pragma unroll
    for (int i = 0; i < FIN_LR; ++i) if (jl + 4 * i < np) Mx = fmaxf(Mx, lz[i]);
    for (int j = jl + 4 * FIN_LR; j < np; j += 4) Mx = fmaxf(Mx, ldl(j));
    Mx = fmaxf(Mx, __shfl_xor(Mx, 16)); Mx = fmaxf(Mx, __shfl_xor(Mx, 32));
    const bool dead = Mx == -__builtin_inff();
    float S = 0.f;
#pragma unroll
    for (int i = 0; i < FIN_LR; ++i) {
      if (jl + 4 * i < np) {
        const float wt = dead ? 0.f : __builtin_amdgcn_exp2f(lz[i] - Mx);
        s_wt[w][jl + 4 * i][hl] = wt;
        S += wt;
      }
    }
    for (int j = jl + 4 * FIN_LR; j < np; j += 4) {
      const float wt = dead ? 0.f : __builtin_amdgcn_exp2f(ldl(j) - Mx);
      s_wt[w][j][hl] = wt;
      S += wt;
    }
    S += __shfl_xor(S, 16); S += __shfl_xor(S, 32);
    if (final) S += dead ? 0.f : __builtin_amdgcn_exp2f(sk - Mx);
    const float invS = S > 0.f ? 1.f / S : 0.f;
    if (jl == 0) {
      if (final) {
        if (lse_out) lse_out[(size_t)r * NH + hw0 + hl] = (__builtin_amdgcn_logf(S) + Mx) * LN2;
      } else {   // group partial: log2-domain lse without sink (dead -> -inf); read later only by the row merge
        const float lg = dead ? -__builtin_inff() : Mx + __builtin_amdgcn_logf(S);
        __builtin_amdgcn_raw_buffer_store_b32(__builtin_bit_cast(unsigned, lg), rl, (s_slot[w][0] * NH + hw0 + hl) * 4, 0,
                                              CPOL_SC1);
      }
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    const int dslot = s_slot[w][0];
    // ---- accumulate: lane l owns dims 4l..4l+3 and 256+4l..+3 ----
    auto consume = [&](v4f (&acc)[HB][2], v4f (&bf)[HB][2], int j, int g) {
#pragma unroll
      for (int k = 0; k < HB; ++k) {
        const float wz = s_wt[w][j][g * HB + k];
        acc[k][0] += wz * bf[k][0]; acc[k][1] += wz * bf[k][1];
      }
    };
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      v4f acc[HB][2];
#pragma unroll
      for (int k = 0; k < HB; ++k) acc[k][0] = acc[k][1] = (v4f){0, 0, 0, 0};
#pragma unroll 1
      for (int j = 0; j + 1 < np; j += 2) {
        issue(g * npp + j + 1, buf1); FIN_TIE16(buf0, 16); consume(acc, buf0, j, g);
        issue(g * npp + j + 2, buf0); FIN_TIE16(buf1, 16); consume(acc, buf1, j + 1, g);
      }
      if (np & 1) {   // odd np: batch np-1 is in buf0; the next group's first batch goes to buf1, moved to buf0 after it landed
        issue(g * npp + npp, buf1); FIN_TIE16(buf0, 16); consume(acc, buf0, np - 1, g);
        FIN_TIE16(buf1, 0);
#pragma unroll
        for (int k = 0; k < HB; ++k) { buf0[k][0] = buf1[k][0]; buf0[k][1] = buf1[k][1]; }
      }
#pragma unroll
      for (int k = 0; k < HB; ++k) {
        const float inv = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, invS), g * HB + k));
        if (final) {
          __bf16* dst = out + ((size_t)r * NH + hw0 + g * HB + k) * DV + 4 * l;
          *(v4bf*)dst = __builtin_convertvector(acc[k][0] * inv, v4bf);
          *(v4bf*)(dst + 256) = __builtin_convertvector(acc[k][1] * inv, v4bf);
        } else {   // in place into the group's first slot (its own reads of these addresses are complete: batch order)
          const int off = ((dslot * NH + hw0 + g * HB + k) * DV + 4 * l) * 4;
          __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(v4u, acc[k][0] * inv), ro, off, 0, CPOL_SC1);
          __builtin_amdgcn_raw_buffer_store_b128(__builtin_bit_cast(v4u, acc[k][1] * inv), ro, off + 1024, 0, CPOL_SC1);
        }
      }
    }
    // the last (clamped, unused) prefetch: all 16 destination registers stay allocated until it has landed
    FIN_TIE16(buf0, 0);
  };

  if (!tree) {
    merge(slots_pieces(pf, pl, (dec >> 1) & 1u), true);
  } else {
    // level 1: this group's pieces -> fp32 partial in the slot of the group's first piece (a 1-piece group is already there)
    const int n1 = slots_pieces(q0, q1, (dec >> 1) & 1u);
    if (n1 > 1) merge(n1, false);
    // level 2: the group's tiles (+ the first-piece parity for group 0) go to the row counter
    const unsigned par2 = gi == 0 ? (dec & 2u) << 29 : 0u;
    const unsigned dec2 = arrive(cnt + 2 * r + hg, (unsigned)target1 | par2, T);
    if (!(dec2 & 1u)) return;
    // group partial slots: first piece of each group (lanes over groups)
    const int ng = (nq + FIN_G - 1) / FIN_G;
    const unsigned pr = (dec2 >> 1) & 1u;
    int n2 = 0;
    for (int g0 = 0; g0 < ng; g0 += 64) {
      const int g = g0 + l;
      int sl = -1;
      if (g < ng) {
        const int qa = pf + g * FIN_G, qb = min(pl, qa + FIN_G - 1);
        for (int q = qa; q <= qb; ++q)
          if (t0of(q) < t1of(q)) { sl = 2 * q + (q == pf ? (int)pr : 0); break; }
      }
      const unsigned long long m = __ballot(sl >= 0);
      if (sl >= 0) {
        const int k = n2 + __popcll(m & ((1ull << l) - 1));
        if (k < FIN_MAXJ) s_slot[w][k] = sl;
      }
      n2 += __popcll(m);
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    merge(__builtin_amdgcn_readfirstlane(min(n2, FIN_MAXJ)), true);
  }
}

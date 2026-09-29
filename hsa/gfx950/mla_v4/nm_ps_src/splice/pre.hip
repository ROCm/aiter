// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Splice source: the in-kernel split planner, compiled for gfx950 and spliced into the prologue of the patched ASM by
// asm/patch.py. This file is only compiled to assembly (hipcc -S); the kernel is never launched.
// ABI assumed by the splice (checked by splice.py): kernarg pointer in s[0:1], workgroup id x in s2, thread id in v0.
// Outputs (uniform) in AGPRs: a243 begin_row, a244 end_row, a245 begin_tile, a246 end_tile, a247 flags,
// a248 per, a249 Pa (active partitions), a250 org = kv_indptr[0] -- the planner's cost-line constants, used by the
// fused combine (fin.hip).
// The descriptor is also published to desc[p] (hg == 0 WG, thread 0).
#include "hca_common.hip"
#define FIN_NPF 100000   // split rows spanning <= FIN_NPF partitions are merged by the fused combine (fin.hip): all rows

struct KA { char b[336]; };
extern "C" __device__ int __ockl_wfred_max_i32(int);
// LDS-only barrier: __syncthreads() also waits for every outstanding global load (fence), which would make the first
// barrier wait for a kv_indptr round trip
#define NS_BAR() do { __builtin_amdgcn_s_waitcnt(0xc07f); __builtin_amdgcn_s_barrier(); } while (0)
// Split / no-split choice: no-split iff  Minf - (per + C0) < NS_WANY  (token units; 384 = 12 tiles; an empirically
// chosen threshold). Minf = max partition load of the no-split plan, exact, per row in closed form: row r (K > 0)
// goes to partition q(r) = min(Pa-1, m_r / per), m_r = b + C0 + 32*(ceil(T/2)-1), load += C0 + 32T.
// WG-cooperative (all 256 threads; every WG computes the same result): thread t walks rows [t*c, t*c+c), c = ceil(N/256),
// adding each row's load into an LDS per-partition array (atomics); the max by an LDS atomic.
// Code size matters (straight-line instructions executed at kernel start cost instruction fetch): no unrolling, one
// division per row (float reciprocal estimate + exact correction).
#define NS_WANY 384
#define NS_MAXP 1024
// Split descriptor (closed-form planner, cut at p*per / (p+1)*per) + the no-split descriptor (every cut snapped to the
// nearer end of the row it falls in: tile i -> 0 if 2i < T, else T) from the same wsearch2 (one search)
__device__ __forceinline__ void plan_both(const int* __restrict__ ind, int N, int P, int Fcfg, int p, Desc& ds, Desc& dn,
                                          int& total, int& per) {
  const int C0 = cfg_C0(Fcfg);
  const int tN = ind[N], tNm1 = ind[max(N - 1, 0)], org = ind[0];
  int posA, posB;
  WRes A, B;
  wsearch2(ind, N, Fcfg, tN, org, P, p, total, per, posA, posB, A, B);
  const int Pa = active_parts(total, P, Fcfg);
  auto Tof = [](int a, int b) { return (b - a + TILE - 1) >> 5; };
  auto ftg = [&](int r, int ir, int T_, int pos) {
    int d = pos - (ir - org + r * C0) - C0;
    int i = d <= 0 ? 0 : (d + TILE - 1) >> 5;
    return min(i, T_);
  };
  // ---- split ----
  int br, bt, Tb;
  const int T0 = Tof(A.i0, A.i1);
  const int iA = posA >= total ? 0 : ftg(A.r, A.i0, T0, posA);
  if (posA >= total) { br = N; bt = 0; Tb = 0; }
  else if (iA < T0) { br = A.r; bt = iA; Tb = T0; }
  else { br = A.r + 1; bt = 0; Tb = (br < N) ? Tof(A.i1, A.i2) : 0; }
  int er, et, Tm1, T0e;
  Tm1 = Tof(B.im1, B.i0); T0e = Tof(B.i0, B.i1);
  const int TN1 = Tof(tNm1, tN);
  int Te_minus = 0;
  const bool endN = p >= Pa - 1 || posB >= total;
  const int iB = endN ? 0 : ftg(B.r, B.i0, T0e, posB);
  if (endN) { er = N; et = 0; Te_minus = TN1; }
  else if (iB < T0e) { er = B.r; et = iB; Te_minus = Tm1; }
  else { er = B.r + 1; et = 0; Te_minus = T0e; }
  ds.comb = ds.pad1 = 0;
  if (!((br < er) || (br == er && bt < et))) {
    ds.begin_row = N; ds.end_row = -1; ds.begin_tile = ds.end_tile = ds.flags = ds.kv_begin_tok = 0;
  } else {
    int Te;
    if (et == 0) { er -= 1; et = Te_minus; Te = Te_minus; }
    else Te = (er == B.r) ? T0e : 0;
    const int f0 = (bt > 0) || (er == br && et < Tb);
    const int f1 = (et < Te) || (er == br && bt > 0);
    ds.begin_row = br; ds.end_row = er; ds.begin_tile = bt; ds.end_tile = et;
    ds.flags = f0 | (f1 << 1); ds.kv_begin_tok = (br == A.r) ? A.i0 : A.i1;
    ds.comb = bt > 0 && (er > br || et == Tb);
  }
  // ---- no-split ----
  const int nb = posA >= total ? N : ((2 * iA < T0 || T0 == 0) ? A.r : A.r + 1);
  const int ne = endN ? N : ((2 * iB < T0e || T0e == 0) ? B.r : B.r + 1);
  dn.comb = dn.pad1 = 0; dn.begin_tile = 0; dn.flags = 0;
  if (nb >= ne) {
    dn.begin_row = N; dn.end_row = -1; dn.end_tile = dn.kv_begin_tok = 0;
  } else {
    dn.begin_row = nb; dn.end_row = ne - 1;
    dn.end_tile = ne == N ? TN1 : (ne == B.r ? Tm1 : T0e);
    dn.kv_begin_tok = nb == A.r ? A.i0 : A.i1;
  }
}
// kernarg reads through the constant-address-space kernarg pointer -> s_load (the by-value struct compiled to vector loads)
#define KARG(T, off) (*(T const __attribute__((address_space(4)))*)((const __attribute__((address_space(4))) char*)__builtin_amdgcn_kernarg_segment_ptr() + (off)))

extern "C" __global__ void __launch_bounds__(256) t8pre(KA ka) {
  const int* ind = KARG(const int*, 0x40);
  Desc* desc = KARG(Desc*, 0x60);
  const int P = KARG(int, 0xa0), N = KARG(int, 0xa4), F = KARG(int, 0xb0);
  int p, hg;
  wg_map(blockIdx.x, P, p, hg);
  int total, per;
  // Wave 0 runs the planner search (plan_both: split and no-split descriptors from one search) while waves
  // 1..3 compute the no-split load profile (ns rows, LDS atomics); the descriptors travel through LDS, so the decision
  // overlaps the search. Every wave executes the same 4 barriers.
  Desc d0;
  if (N > 0) {
    __shared__ int s_load[NS_MAXP];
    __shared__ int s_max;
    __shared__ Desc s_dd[2];
    __shared__ int s_tp[3];   // total, per, Pa (wave 0's search)
    const int C0 = cfg_C0(F), tid = threadIdx.x, wv = tid >> 6;
    // LDS init before any memory dependence (the first barrier must not wait for a kv_indptr round trip)
#pragma unroll
    for (int i = 0; i < NS_MAXP / 256; ++i) s_load[tid + 256 * i] = 0;
    if (tid == 0) s_max = 0;
    int r0 = 0, r1 = 0;
    if (wv) {
      const int cs = (N + 191) / 192;
      r0 = min(N, (tid - 64) * cs); r1 = min(N, r0 + cs);
    }
    NS_BAR();
    if (wv == 0) {
      Desc ds, dn;
      int tt, pp;
      plan_both(ind, N, P, F, p, ds, dn, tt, pp);
      if (tid == 0) { s_dd[0] = ds; s_dd[1] = dn; s_tp[0] = tt; s_tp[1] = pp; s_tp[2] = active_parts(tt, P, F); }
    } else {
      int i0 = ind[r0], i1 = ind[min(r0 + 1, N)];
      const int org = ind[0], tot0 = ind[N] - org + N * C0;   // == wsearch2's cost-line constants
      const int Pa0 = active_parts(tot0, P, F), per0 = max(1, (tot0 + Pa0 - 1) / Pa0);
      const float rf = __builtin_amdgcn_rcpf((float)per0);
#pragma unroll 1
      for (int r = r0; r < r1; ++r) {
        if (r != r0) i1 = ind[r + 1];
        const int T = (i1 - i0 + TILE - 1) >> 5;
        if (T > 0) {
          const int m = i0 - org + r * C0 + C0 + TILE * (((T + 1) >> 1) - 1);
          // q <= total/per <= Pa <= 1024: the float estimate has |error| < 1 (relative error ~2^-22), one correction
          int q = (int)((float)m * rf);
          const int rm = m - q * per0;
          q += rm < 0 ? -1 : (rm >= per0 ? 1 : 0);
          atomicAdd(&s_load[min(Pa0 - 1, q)], C0 + TILE * T);
        }
        i0 = i1;
      }
    }
    NS_BAR();
    const int tot0 = s_tp[0], per0 = s_tp[1], Pa0 = s_tp[2];
    int mx = 0;
    for (int i = tid; i < Pa0; i += 256) mx = max(mx, s_load[i]);
    // wave max first: 256 lanes atomicMax-ing one LDS word would serialise
    mx = __ockl_wfred_max_i32(mx);   // DPP wave reduction
    if ((tid & 63) == 0 && mx) atomicMax(&s_max, mx);
    NS_BAR();
    d0 = s_dd[(s_max - per0 - C0 < NS_WANY) ? 1 : 0];
    NS_BAR();   // every wave has read LDS before any wave enters the ASM (which reuses LDS)
    total = tot0; per = per0;
  } else {   // N = 0: all partitions empty (without instantiating a second search)
    d0.begin_row = N; d0.end_row = -1; d0.begin_tile = d0.end_tile = d0.flags = d0.kv_begin_tok = d0.comb = d0.pad1 = 0;
    total = 0; per = 1;
  }
  const int Pa = active_parts(total, P, F);
  Desc d = d0;
  // fused combine (fin.hip): a split row spanning <= FIN_NPF partitions (every row) is merged in the attention kernel
  // by its last-arriving piece; desc.comb is cleared for it. Same closed form as fin.hip.
  if (d.comb) {
    const int br = d.begin_row, org = ind[0], C0 = cfg_C0(F);
    const int b = ind[br] - org + br * C0, T = (ind[br + 1] - ind[br] + TILE - 1) >> 5;
    const int pf = min(Pa - 1, (b + C0) / per), pl = min(Pa - 1, (b + C0 + TILE * (T - 1)) / per);
    if (pl - pf + 1 <= FIN_NPF) d.comb = 0;
  }
  if (hg == 0 && threadIdx.x == 0) desc[p] = d;
  asm volatile("v_accvgpr_write_b32 a243, %0\n\tv_accvgpr_write_b32 a244, %1\n\tv_accvgpr_write_b32 a245, %2\n\t"
               "v_accvgpr_write_b32 a246, %3\n\tv_accvgpr_write_b32 a247, %4\n\tv_accvgpr_write_b32 a248, %5\n\t"
               "v_accvgpr_write_b32 a249, %6\n\tv_accvgpr_write_b32 a250, %7"
               :: "v"(d.begin_row), "v"(d.end_row), "v"(d.begin_tile), "v"(d.end_tile), "v"(d.flags), "v"(per), "v"(Pa),
                  "v"(N > 0 ? ind[0] : 0)
               : "memory");
}

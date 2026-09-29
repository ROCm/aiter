// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Common definitions of the splice sources (pre.hip, fin.hip): the closed-form partition planner's cost line,
// the per-partition descriptor, the wave-cooperative 64-way boundary search and the WG -> (partition, head group)
// map. Only compiled as part of pre.hip / fin.hip (hipcc -S); nothing here is a kernel.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int      v4i  __attribute__((ext_vector_type(4)));
typedef float    v4f  __attribute__((ext_vector_type(4)));
typedef __bf16   v4bf __attribute__((ext_vector_type(4)));

#define TILE 32
#define NH 128           // heads
#define DV 512
#define LOG2E 1.4426950408889634f
#define LN2 0.6931471805599453f

// ---------------- closed-form partition planner: cost line ----------------
// Planner config word Fcfg = F | (MT << 8):
//   C0 = 32*F + 16 (fixed cost per row piece, token units, charged once per row on the cost line);
//   MT = minimum partition size in tiles beyond one C0 (0 = off): the number of ACTIVE partitions is
//   Pa = clamp(total / (C0 + 32*MT), 1, P), per = ceil(total / Pa); partitions p >= Pa are empty.
//   A cut inside a row adds at most one extra C0 per partition (the fixed cost is the same for every
//   partition, so balance is unaffected); Pa bounds how many pieces tiny shapes are cut into.
__device__ __forceinline__ int cfg_C0(int Fcfg) { return 32 * (Fcfg & 0xff) + 16; }
__device__ __forceinline__ int active_parts(int total, int P, int Fcfg) {
  const int MT = Fcfg >> 8;
  if (MT <= 0) return P;
  return max(1, min(P, total / (cfg_C0(Fcfg) + 32 * MT)));
}

struct __attribute__((aligned(32))) Desc {
  // comb = 1 iff begin_row is split and its LAST piece is in this partition (begin_tile > 0 and the row ends here):
  // this partition's combine WG merges that row (only used by an external combine; the fused combine clears it).
  int begin_row, end_row, begin_tile, end_tile, flags, kv_begin_tok, comb, pad1;
};

// ---------------- wave-cooperative planner search ----------------
// The two partition-boundary binary searches of partition p (positions p*per and (p+1)*per on the cost
// line), 64-way: each round every lane samples one row of the current interval, a ballot counts the samples
// with base() <= pos, and the interval shrinks 64x. N <= 64: 1 round, N <= 4096: 2 rounds (+ the
// ind[N], ind[N-1] loads, issued in parallel with round 0). Each lane loads ind[j-1..j+2] so the final
// step-1 round also yields every neighbour value the descriptor needs (no extra dependent round).
// Must be called by all 64 lanes of a wave with uniform arguments.
struct WRes { int r, im1, i0, i1, i2; };  // row, ind[r-1], ind[r], ind[r+1], ind[r+2] (clamped)

__device__ __forceinline__ int rdl(int v, int lane) { return __builtin_amdgcn_readlane(v, lane); }

__device__ __forceinline__ void wsearch2(const int* __restrict__ ind, int N, int Fcfg, int tN, int org, int P, int p,
                                         int& total, int& per, int& posA, int& posB, WRes& A, WRes& B) {
  const int C0 = cfg_C0(Fcfg);  // positions are relative to org = ind[0] (tN is ind[N], absolute)
  const int lane = threadIdx.x & 63;
  int loA = 0, lenA = N, stA = (N + 63) >> 6;
  int loB = 0, lenB = N, stB = stA;
  bool doneA = false, doneB = false, first = true;
  total = 0; per = 1; posA = posB = 0;
  while (!(doneA && doneB)) {
    const bool vA = !doneA && lane * stA < lenA, vB = !doneB && lane * stB < lenB;
    const int jA = vA ? loA + lane * stA : loA, jB = vB ? loB + lane * stB : loB;
    const int a_m1 = ind[max(jA - 1, 0)], a0 = ind[jA], a1 = ind[jA + 1], a2 = ind[min(jA + 2, N)];
    const int b_m1 = ind[max(jB - 1, 0)], b0 = ind[jB], b1 = ind[jB + 1], b2 = ind[min(jB + 2, N)];
    if (first) {
      total = tN - org + N * C0;
      const int Pa = active_parts(total, P, Fcfg);
      per = max(1, (total + Pa - 1) / Pa);
      posA = min(p, Pa) * per;                      // p >= Pa: posA = Pa*per >= total -> empty (no int overflow)
      posB = (p >= Pa - 1) ? posA : (p + 1) * per;
      first = false;
    }
    const int cA = __popcll(__ballot(vA && a0 - org + jA * C0 <= posA));
    const int cB = __popcll(__ballot(vB && b0 - org + jB * C0 <= posB));
    if (!doneA) {
      if (stA == 1) {
        A.r = loA + cA - 1; A.im1 = rdl(a_m1, cA - 1); A.i0 = rdl(a0, cA - 1); A.i1 = rdl(a1, cA - 1); A.i2 = rdl(a2, cA - 1);
        doneA = true;
      } else {
        loA += (cA - 1) * stA; lenA = min(stA, lenA - (cA - 1) * stA); stA = (lenA + 63) >> 6;
      }
    }
    if (!doneB) {
      if (stB == 1) {
        B.r = loB + cB - 1; B.im1 = rdl(b_m1, cB - 1); B.i0 = rdl(b0, cB - 1); B.i1 = rdl(b1, cB - 1); B.i2 = rdl(b2, cB - 1);
        doneB = true;
      } else {
        loB += (cB - 1) * stB; lenB = min(stB, lenB - (cB - 1) * stB); stB = (lenB + 63) >> 6;
      }
    }
  }
}

__device__ __forceinline__ void wg_map(int wg, int P, int& p, int& hg) {
  if ((P & 7) == 0) {  // keep the hg pair and neighbouring partitions on one XCD (round-robin dispatch, inferred)
    int xcd = wg & 7, loc = wg >> 3;
    p = xcd * (P >> 3) + (loc >> 1); hg = loc & 1;
    return;
  }
  p = wg >> 1; hg = wg & 1;
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: the topk-prefill-avo repo. Regenerate with
//   python3 scripts/export_aiter_op.py --aiter <this aiter checkout>
// and verify an existing tree with the same command plus --check.
//
// This is benchmark_topk.hip.cpp up to its AITER_EXPORT_END marker (the kernels
// and their dispatch) followed by csrc/topk_aiter_entry.inc.hip (the aiter op
// entry). The harness half of that file -- CPU/GPU verification oracles, timing,
// CLI -- is deliberately not here.
#pragma once

// GPU kernels for generalized top-k paths. Include AFTER block_select_lds / block_gather_topk
// are visible in the translation unit.

__global__ void phase_small_n_topk(const float* __restrict__ input, int N, int K,
                                   int* __restrict__ out_idx, int npasses) {
  const int row = blockIdx.x;
  const float* ri = input + (size_t)row * N;
  // No index array: the whole row is resident, so LDS slot i IS column i. A
  // [N_LDS_MAX] index array would cost 32 KB of LDS to store the identity.
  extern __shared__ uint32_t s_keys[];
  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
  __shared__ unsigned s_wgt, s_weq;

  const int n4 = N / FP32_EPT;
  for (int u = threadIdx.x; u < n4; u += blockDim.x) {
    vfloat4 v = *(reinterpret_cast<const vfloat4*>(ri) + u);
    const int base = u * FP32_EPT;
    s_keys[base + 0] = fp32_to_sortable(v[0]);
    s_keys[base + 1] = fp32_to_sortable(v[1]);
    s_keys[base + 2] = fp32_to_sortable(v[2]);
    s_keys[base + 3] = fp32_to_sortable(v[3]);
  }
  __syncthreads();

  uint32_t pivot;
  int eq_needed;
  block_select_lds(s_keys, N, K, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses, true);

  int* out = out_idx + (size_t)row * K;
  if (threadIdx.x == 0) {
    s_wgt = 0;
    s_weq = 0;
  }
  __syncthreads();
  block_gather_topk(N, pivot, K - eq_needed, eq_needed, out, &s_wgt, &s_weq,
                    [&](int i) { return s_keys[i]; }, [](int i) { return i; });
}

__global__ void phase_b_filter_coop(const float* __restrict__ input, int N, int n4_per_row,
                                    const float* __restrict__ threshold_f, uint64_t* __restrict__ cand_pack,
                                    unsigned int* __restrict__ cand_reserved,
                                    unsigned int* __restrict__ cand_bad, int cap) {
  const int row = blockIdx.y;
  const int bid = blockIdx.x;
  const vfloat4* ri = reinterpret_cast<const vfloat4*>(input + (size_t)row * N);
  const float th = threshold_f[row];
  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;

  __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
  uint64_t* buf = wbuf + (size_t)wid * WSTAGE_CAP;
  uint64_t* row_base = cand_pack + (size_t)row * cap;
  (void)nwaves;

  const int G = gridDim.x;
  const int chunk_n4 = (n4_per_row + G - 1) / G;
  const int i0 = bid * chunk_n4;
  const int i1 = min(i0 + chunk_n4, n4_per_row);
  const int stride = blockDim.x;
  // All lanes must run the same number of iterations: with
  // `for (i = i0 + tid; i < i1; i += stride)` the tail iteration leaves some
  // lanes inactive, and then __ballot sees only the active lanes, so bcnt stops
  // being wave-uniform and every flush offset below is wrong. Mask with `live`
  // instead, exactly as phase_b_filter_wavestage does.
  const int iters = (i1 > i0) ? ((i1 - i0) + stride - 1) / stride : 0;

  int bcnt = 0;

  // Overflow valve, NOT the normal path. The original version accumulated the
  // block's whole chunk into `buf` with no bound check against WSTAGE_CAP=320,
  // so any wave producing more than 320 passers wrote into the next wave's
  // slice and the last wave wrote past wbuf onto the count variables (measured:
  // M=128 N=65536 adversarial -> rows_fail=4, 124 rows with garbage counts that
  // CHANGED between identical runs).
  //
  // Draining per-wave on every >=64 candidates fixes that but is 1.5-2x slower
  // (M=8 N=524288: 34.8 -> 54.0 us), because at normal density a wave holds
  // only ~6 candidates, so every wave ends up doing one 48-byte scattered write
  // instead of the block doing one contiguous ~400-byte write. Small scattered
  // stores are this kernel's known pathology.
  //
  // So: keep the block-level aggregation below as the fast path, and only drain
  // a wave mid-loop when its buffer is actually about to overflow. An iteration
  // can add at most 4*64 = 256, so draining once bcnt exceeds
  // WSTAGE_CAP - 256 = 64 keeps the write in bounds. At normal density bcnt
  // never reaches that and the valve never fires.
#define COOP_DRAIN_WAVE()                                                          \
  do {                                                                             \
    unsigned _off = 0;                                                             \
    if (lane == 0) _off = atomicAdd(&cand_reserved[row], (unsigned)bcnt);           \
    _off = (unsigned)__shfl((int)_off, 0);                                          \
    if (_off + (unsigned)bcnt > (unsigned)cap) {                                    \
      if (lane == 0) atomicExch(&cand_bad[row], 1u);                                \
      bcnt = -1;                                                                    \
      break;                                                                        \
    }                                                                               \
    for (int _j = lane; _j < bcnt; _j += WAVE_SIZE) row_base[_off + _j] = buf[_j];  \
    bcnt = 0;                                                                       \
  } while (0)

  for (int it = 0; it < iters; it++) {
    const int i = i0 + it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    const bool live = (i < i1);
    if (live) v = load_f4(ri + i);
    const uint64_t b0 = __ballot(live && !(v[0] < th));
    const uint64_t b1 = __ballot(live && !(v[1] < th));
    const uint64_t b2 = __ballot(live && !(v[2] < th));
    const uint64_t b3 = __ballot(live && !(v[3] < th));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);
    if (wtotal > 0) {
      const int base_idx = i * FP32_EPT;
      if (b0 & (1ull << lane))
        buf[bcnt + __popcll(b0 & lt)] =
            ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
      if (b1 & (1ull << lane))
        buf[bcnt + t0 + __popcll(b1 & lt)] =
            ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
      if (b2 & (1ull << lane))
        buf[bcnt + t1 + __popcll(b2 & lt)] =
            ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
      if (b3 & (1ull << lane))
        buf[bcnt + t2 + __popcll(b3 & lt)] =
            ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
      bcnt += wtotal;
    }
    if (bcnt > WSTAGE_CAP - 4 * WAVE_SIZE) {
      __builtin_amdgcn_wave_barrier();
      COOP_DRAIN_WAVE();
    }
  }
#undef COOP_DRAIN_WAVE

  // Fast path: one atomic and one contiguous region per BLOCK for everything
  // still staged. bcnt < 0 marks a wave that already tripped the cap.
  __shared__ int s_local[MAX_WAVES_PER_BLOCK];
  __shared__ int s_off[MAX_WAVES_PER_BLOCK];
  __shared__ unsigned s_base;
  if (lane == 0) s_local[wid] = bcnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    bool bad = false;
    for (int w = 0; w < nwaves; w++) {
      if (s_local[w] < 0) bad = true;
      else {
        s_off[w] = total;
        total += s_local[w];
      }
    }
    if (bad || total > cap) {
      if (bad || total > cap) atomicExch(&cand_bad[row], 1u);
      s_base = 0xFFFFFFFFu;
    } else if (total == 0) {
      s_base = 0xFFFFFFFEu;                       // nothing to write
    } else {
      s_base = atomicAdd(&cand_reserved[row], (unsigned)total);
      if (s_base + (unsigned)total > (unsigned)cap) {
        atomicExch(&cand_bad[row], 1u);
        s_base = 0xFFFFFFFFu;
      }
    }
  }
  __syncthreads();
  if (s_base == 0xFFFFFFFFu || s_base == 0xFFFFFFFEu) return;

  // Whole block writes one contiguous run; wave w owns [s_off[w], +s_local[w]).
  for (int w = 0; w < nwaves; w++) {
    const int cnt = s_local[w];
    if (cnt <= 0) continue;
    uint64_t* dst = row_base + s_base + s_off[w];
    const uint64_t* src = wbuf + (size_t)w * WSTAGE_CAP;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) dst[j] = src[j];
  }
}

// Folds what used to be a separate finalize_coop_counts kernel: the reserved
// total and the overflow flag are reduced to the usable candidate count right
// here, saving a dispatch. cand_count is still written so --dump-stats keeps
// working, which costs one store rather than a kernel.
__global__ void phase_c_select_contig(const float* __restrict__ input, int N,
                                      const uint64_t* __restrict__ cand_pack,
                                      const unsigned int* __restrict__ cand_reserved,
                                      const unsigned int* __restrict__ cand_bad,
                                      unsigned int* __restrict__ cand_count, int cap, int K,
                                      int* __restrict__ out_idx, int* __restrict__ fb_rows,
                                      int* __restrict__ fb_count, int npasses, bool keys_only) {
  const int row = blockIdx.x;
  const unsigned int c_raw = cand_bad[row] ? 0xFFFFFFFFu : cand_reserved[row];
  if (threadIdx.x == 0) cand_count[row] = c_raw;

  // Dynamic LDS: cap keys, plus cap indices unless the keys-only variant
  // re-reads them from global. Sizing either statically at PHASE_C_CAP_MAX costs
  // 32 KB unconditionally and halves occupancy.
  extern __shared__ uint32_t s_dyn[];
  uint32_t* s_keys_ext = s_dyn;
  int* s_idx = reinterpret_cast<int*>(s_dyn + cap);
  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
  __shared__ unsigned s_wgt, s_weq;

  int* out = out_idx + (size_t)row * K;
  if (c_raw < (unsigned)K || c_raw > (unsigned)cap) {
    // Unusable candidate set: exact full-row select right here instead of in a
    // separate fallback kernel. Saves the 4th dispatch on every call, and is
    // more parallel in the worst case (M blocks instead of FB_GRID=64).
    if (threadIdx.x == 0) fb_rows[atomicAdd(fb_count, 1)] = row;   // diagnostics only
    exact_row_select(input, N, K, row, out, s_hist, s_red, s_scan, &s_wgt, &s_weq);
    return;
  }
  const int c = (int)c_raw;
  const uint64_t* base = cand_pack + (size_t)row * cap;

  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    uint64_t p = base[i];
    s_keys_ext[i] = fp32_to_sortable_bits((uint32_t)(p >> 32));
    if (!keys_only) s_idx[i] = (int)(uint32_t)p;
  }
  __syncthreads();

  uint32_t pivot;
  int eq_needed;
  block_select_lds(s_keys_ext, c, K, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses, true);

  if (threadIdx.x == 0) {
    s_wgt = 0;
    s_weq = 0;
  }
  __syncthreads();

  if (keys_only) {
    block_gather_topk(c, pivot, K - eq_needed, eq_needed, out, &s_wgt, &s_weq,
                      [&](int i) { return s_keys_ext[i]; },
                      [&](int i) { return (int)(uint32_t)(base[i] & 0xFFFFFFFFull); });
  } else {
    block_gather_topk(c, pivot, K - eq_needed, eq_needed, out, &s_wgt, &s_weq,
                      [&](int i) { return s_keys_ext[i]; }, [&](int i) { return s_idx[i]; });
  }
}

__global__ __launch_bounds__(1024) void phase_ab_fused(
    const float* __restrict__ input, int N, int rank, int S, int npasses, int chunk_stride,
    int seg_stride, uint64_t* __restrict__ cand_pack, int* __restrict__ cand_seg,
    unsigned int* __restrict__ cand_count, int* __restrict__ fb_count) {
  const int row = blockIdx.x;
  const float* ri = input + (size_t)row * N;

  if (threadIdx.x == 0 && row == 0) *fb_count = 0;

  extern __shared__ uint32_t s_keys[];
  __shared__ uint32_t s_hist[HIST_SLOTS];
  __shared__ uint32_t s_red[256];
  __shared__ uint32_t s_scan[2];
  __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];

  // From the host, for the reason spelled out in phase_a_threshold.
  const int v4_per_chunk = SAMPLE_CHUNK_ELEMS / FP32_EPT;
  const int total_v4 = S / FP32_EPT;
  for (int u = threadIdx.x; u < total_v4; u += blockDim.x) {
    const int chunk = u / v4_per_chunk;
    const int off4 = u % v4_per_chunk;
    vfloat4 v = *(reinterpret_cast<const vfloat4*>(ri + (size_t)chunk * chunk_stride) + off4);
    const int base = u * FP32_EPT;
    s_keys[base + 0] = fp32_to_sortable(v[0]);
    s_keys[base + 1] = fp32_to_sortable(v[1]);
    s_keys[base + 2] = fp32_to_sortable(v[2]);
    s_keys[base + 3] = fp32_to_sortable(v[3]);
  }
  __syncthreads();

  uint32_t pivot;
  int eq_needed;
  block_select_lds(s_keys, S, rank, s_hist, s_red, s_scan, s_mm, pivot, eq_needed, npasses);
  const float th = sortable_to_fp32(pivot);

  const vfloat4* ri4 = reinterpret_cast<const vfloat4*>(ri);
  const int lane = threadIdx.x & (WAVE_SIZE - 1);
  const int wid = threadIdx.x / WAVE_SIZE;
  const int nwaves = blockDim.x / WAVE_SIZE;
  const uint64_t lt = (1ull << lane) - 1ull;
  __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
  uint64_t* buf = wbuf + (size_t)wid * WSTAGE_CAP;
  uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;
  const int n4 = N / FP32_EPT;
  const int stride = blockDim.x;
  const int iters = (n4 + stride - 1) / stride;
  int wcnt = 0, bcnt = 0;
  bool overflow = false;
  for (int it = 0; it < iters; it++) {
    const int i = it * stride + threadIdx.x;
    vfloat4 v = {0.f, 0.f, 0.f, 0.f};
    const bool live = (i < n4);
    if (live) v = load_f4(ri4 + i);
    const uint64_t b0 = __ballot(live && !(v[0] < th));
    const uint64_t b1 = __ballot(live && !(v[1] < th));
    const uint64_t b2 = __ballot(live && !(v[2] < th));
    const uint64_t b3 = __ballot(live && !(v[3] < th));
    const int t0 = __popcll(b0);
    const int t1 = t0 + __popcll(b1);
    const int t2 = t1 + __popcll(b2);
    const int wtotal = t2 + __popcll(b3);
    if (wtotal > 0) {
      const int base_idx = i * FP32_EPT;
      if (b0 & (1ull << lane))
        buf[bcnt + __popcll(b0 & lt)] =
            ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
      if (b1 & (1ull << lane))
        buf[bcnt + t0 + __popcll(b1 & lt)] =
            ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
      if (b2 & (1ull << lane))
        buf[bcnt + t1 + __popcll(b2 & lt)] =
            ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
      if (b3 & (1ull << lane))
        buf[bcnt + t2 + __popcll(b3 & lt)] =
            ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
      bcnt += wtotal;
    }
    if (bcnt >= WAVE_SIZE) {
      __builtin_amdgcn_wave_barrier();
      if (wcnt + bcnt <= seg_stride) {
        for (int j = lane; j < bcnt; j += WAVE_SIZE) seg[wcnt + j] = buf[j];
      } else {
        overflow = true;
      }
      wcnt += bcnt;
      bcnt = 0;
    }
  }
  if (bcnt > 0) {
    __builtin_amdgcn_wave_barrier();
    if (wcnt + bcnt <= seg_stride) {
      for (int j = lane; j < bcnt; j += WAVE_SIZE) seg[wcnt + j] = buf[j];
    } else {
      overflow = true;
    }
    wcnt += bcnt;
  }
  __shared__ int s_seg[MAX_WAVES_PER_BLOCK];
  if (lane == 0) s_seg[wid] = overflow ? -1 : wcnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned total = 0;
    bool bad = false;
    for (int w = 0; w < nwaves; w++) {
      cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + w] = s_seg[w];
      if (s_seg[w] < 0) bad = true;
      else total += (unsigned)s_seg[w];
    }
    cand_count[row] = bad ? 0xFFFFFFFFu : total;
  }
}

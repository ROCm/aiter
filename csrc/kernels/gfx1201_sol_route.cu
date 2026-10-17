// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// gfx1201 Sol sparse attention, routing: pooled K/V statistics, INT8/FP8 pooled operands for the Phase-A proxy,
// Sol-Attn block selection per 64-row query block (NVlabs Sol-Attn "diag" threshold, Sol-H3 sink/band policy),
// union over each 512-row workgroup -> tile list for the ASM core + Phase-A mask.
// Workgroup order matches the ASM core: wg = head * nqt + qtile.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

static constexpr int D = 128, KBR = 64, QBR = 64, WGR = 512, QB_PER_WG = WGR / QBR;

__device__ __forceinline__ float fp8_to_f32(uint32_t w, int sel) {
  switch (sel) {
    case 0: return __builtin_amdgcn_cvt_f32_fp8((int)w, 0);
    case 1: return __builtin_amdgcn_cvt_f32_fp8((int)w, 1);
    case 2: return __builtin_amdgcn_cvt_f32_fp8((int)w, 2);
    default: return __builtin_amdgcn_cvt_f32_fp8((int)w, 3);
  }
}

// grid (nkbp, H), block 128: kbar [H,nkbp,128] (mean of dequant K = Sol kc), vsum [H,128,nkbp] (sum of dequant V = Sol vc)
extern "C" __global__ void gfx1201_sol_pool_kv(const int8_t* K, const float* KS, const uint8_t* V, const float* VS,
                                   float* kbar, float* vsum, int Sp, int valid, int H, int nkbp) {
  const int b = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int r0 = b * KBR;
  const int n = max(0, min(KBR, valid - r0));
  float ks = 0.f;
  for (int g = 0; g < n; g += 32) {
    int acc = 0;
    const int m = min(32, n - g);
    for (int r = 0; r < m; ++r) acc += K[((long long)(r0 + g + r) * H + h) * D + d];
    ks += (float)acc * KS[(long long)h * (Sp / 32) + (r0 + g) / 32];
  }
  kbar[((long long)h * nkbp + b) * D + d] = ks / (float)max(n, 1);
  const uint8_t* vp = V + ((long long)h * D + d) * Sp + r0;
  float vsm = 0.f;
  for (int r = 0; r < n; r += 4) {
    const uint32_t w = *reinterpret_cast<const uint32_t*>(vp + r);
    const int m = min(4, n - r);
    for (int i = 0; i < m; ++i) vsm += fp8_to_f32(w, i);
  }
  vsum[((long long)h * D + d) * nkbp + b] = vsm * VS[h * D + d];
}

// grid (nkbp/32, H), block 256: per 32-block tile INT8 quantisation of kbar -> kb [nkbp,H,128], kb_scale [H,nkbp/32]
extern "C" __global__ void gfx1201_sol_quant_k(const float* kbar, int8_t* kb, float* kb_scale, int H, int nkbp) {
  __shared__ float red[256];
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const float* src = kbar + ((long long)h * nkbp + t * 32) * D;
  float am = 0.f;
  for (int i = tid; i < 32 * D; i += 256) am = fmaxf(am, fabsf(src[i]));
  red[tid] = am;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float sc = fmaxf(red[0] / 127.f, 1e-12f);
  if (tid == 0) kb_scale[h * (nkbp / 32) + t] = sc;
  for (int i = tid; i < 32 * D; i += 256) {
    const float x = fminf(fmaxf(rintf(src[i] / sc), -127.f), 127.f);
    kb[((long long)(t * 32 + i / D) * H + h) * D + i % D] = (int8_t)x;
  }
}

// grid H, block 128: Sol kc_mean / kc_var_diag (over all key blocks); FP8 pooled V with per-channel scale
extern "C" __global__ void gfx1201_sol_stats(const float* kbar, const float* vsum, uint8_t* vb, float* vb_scale,
                                 float* mu, float* var, int H, int nkb, int nkbp) {
  const int h = blockIdx.x, d = threadIdx.x;
  float s = 0.f, s2 = 0.f;
  for (int b = 0; b < nkb; ++b) {
    const float x = kbar[((long long)h * nkbp + b) * D + d];
    s += x;
    s2 += x * x;
  }
  const float m = s / nkb;
  mu[h * D + d] = m;
  var[h * D + d] = s2 / nkb - m * m;
  const float* vs = vsum + ((long long)h * D + d) * nkbp;
  float am = 0.f;
  for (int b = 0; b < nkbp; ++b) am = fmaxf(am, fabsf(vs[b]));
  const float sc = fmaxf(am / 448.f, 1e-20f);
  vb_scale[h * D + d] = sc;
  uint8_t* dst = vb + ((long long)h * D + d) * nkbp;
  for (int b = 0; b < nkbp; b += 4) {
    int w = __builtin_amdgcn_cvt_pk_fp8_f32(vs[b] / sc, vs[b + 1] / sc, 0, false);
    w = __builtin_amdgcn_cvt_pk_fp8_f32(vs[b + 2] / sc, vs[b + 3] / sc, w, true);
    *reinterpret_cast<int*>(dst + b) = w;
  }
}

__device__ void gfx1201_sol_emit_list(uint8_t* uni, int* scan, int32_t* list, int32_t* count, uint32_t* mask, int wg,
                          int valid, int nkb, int nkbp, int stride) {
  const int tid = threadIdx.x;
  const int per = (nkb + 255) / 256;
  const int b0 = tid * per, b1 = min(nkb, b0 + per);
  const int full_end = valid / 32 * 32;
  for (int b = nkb + tid; b < nkbp; b += 256) uni[b] = 1;
  int ntile = 0;
  for (int b = b0; b < b1; ++b)
    if (uni[b]) ntile += (b * KBR < full_end) + (b * KBR + 32 < full_end);
  scan[tid] = ntile;
  __syncthreads();
  for (int off = 1; off < 256; off <<= 1) {
    const int v = tid >= off ? scan[tid - off] : 0;
    __syncthreads();
    scan[tid] += v;
    __syncthreads();
  }
  int pos = scan[tid] - ntile;
  int32_t* lst = list + (long long)wg * stride;
  for (int b = b0; b < b1; ++b) {
    if (!uni[b]) continue;
    if (b * KBR < full_end) lst[pos++] = b * KBR;
    if (b * KBR + 32 < full_end) lst[pos++] = b * KBR + 32;
  }
  const int total = scan[255];
  if (tid == 0) {
    count[wg] = total;
    lst[total] = full_end;
    lst[total + 1] = 0;
  }
  for (int t = tid; t < nkbp / 32; t += 256) {
    uint32_t w = 0;
    for (int j = 0; j < 32; ++j) w |= (uint32_t)uni[t * 32 + j] << j;
    mask[(long long)wg * (nkbp / 32) + t] = w;
  }
}


// gfx1201_sol_select: grid W (wg = head * nqt + qtile), 256 threads; wave w routes query block qtile * 8 + w (64 rows).
// Sol-Attn "diag": qc = block-mean query (Q scales carry softmax_scale * log2 e), threshold = qc.mu + tau * sqrt(sum
// qc^2 var + 1e-6); key block j is exact if qc.kc_j > threshold, |qb - j| <= 1, j or qb inside the exact prefix
// (Sol-H3 sink, block-rounded outward), or j is the last block (its partial 32-row tail is always run exactly).
static constexpr int MAX_NKB = 4096, KC_CHUNK = 32, KC_STRIDE = D + 1;
extern "C" __global__ __launch_bounds__(256) void gfx1201_sol_select(const int8_t* Q, const float* QS,
                                                        const float* kbar, const float* mu, const float* var,
                                                        int32_t* list, int32_t* count, uint32_t* mask, int Sp,
                                                        int valid, int H, int nkb, int nkbp, int nqt, int stride,
                                                        int prefix, float tau) {
  __shared__ uint8_t uni[MAX_NKB + 32];
  __shared__ int scan[256];
  __shared__ float sq[QB_PER_WG][D];
  __shared__ float skc[KC_CHUNK * KC_STRIDE];
  const int wg = blockIdx.x, tid = threadIdx.x;
  const int h = wg / nqt, qt = wg % nqt;
  const int wave = tid / 32, lane = tid % 32;
  for (int i = tid; i < nkbp; i += 256) uni[i] = 0;
  const int qb = qt * QB_PER_WG + wave;
  const int q0 = qb * QBR, n = max(0, min(QBR, valid - q0));
  const int sink = (prefix + KBR - 1) / KBR;
  // qc: lane owns dims 4*lane .. 4*lane+3
  float qc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int g = 0; g < n; g += 32) {
    int acc[4] = {0, 0, 0, 0};
    const int m = min(32, n - g);
    for (int r = 0; r < m; ++r) {
      const int w = *reinterpret_cast<const int*>(Q + ((long long)(q0 + g + r) * H + h) * D + 4 * lane);
      #pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] += (int)(int8_t)(w >> (8 * i));
    }
    const float sc = QS[(long long)h * (Sp / 32) + (q0 + g) / 32];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qc[i] += (float)acc[i] * sc;
  }
  float mean = 0.f, vv = 0.f;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    qc[i] /= (float)max(n, 1);
    sq[wave][4 * lane + i] = qc[i];
    mean += qc[i] * mu[h * D + 4 * lane + i];
    vv += qc[i] * qc[i] * var[h * D + 4 * lane + i];
  }
  for (int o = 16; o > 0; o >>= 1) {
    mean += __shfl_xor(mean, o, 32);
    vv += __shfl_xor(vv, o, 32);
  }
  const float thr = mean + tau * sqrtf(fmaxf(vv, 0.f) + 1e-6f);
  const bool active = n > 0, qsink = qb < sink;
  for (int j0 = 0; j0 < nkb; j0 += KC_CHUNK) {
    __syncthreads();
    for (int i = tid; i < KC_CHUNK * D; i += 256) {
      const int jb = j0 + i / D;
      skc[(i / D) * KC_STRIDE + i % D] = jb < nkb ? kbar[((long long)h * nkbp + jb) * D + i % D] : 0.f;
    }
    __syncthreads();
    const int j = j0 + lane;
    if (active && j < nkb) {
      float s = 0.f;
      #pragma unroll 8
      for (int d = 0; d < D; ++d) s += sq[wave][d] * skc[lane * KC_STRIDE + d];
      if (s > thr || abs(qb - j) <= 1 || j < sink || qsink || j == nkb - 1) uni[j] = 1;
    }
  }
  __syncthreads();
  gfx1201_sol_emit_list(uni, scan, list, count, mask, wg, valid, nkb, nkbp, stride);
}

void launch_gfx1201_sol_route(int64_t Q, int64_t QS, int64_t K, int64_t KS, int64_t V, int64_t VS, int64_t kbar,
                              int64_t vsum, int64_t kb, int64_t kb_scale, int64_t vb, int64_t vb_scale, int64_t mu,
                              int64_t var, int64_t list, int64_t count, int64_t mask, int64_t Sp, int64_t valid,
                              int64_t H, int64_t prefix, double tau, int64_t stream) {
  hipStream_t st = (hipStream_t)stream;
  const int nkb = (int)((valid + KBR - 1) / KBR), nkbp = (nkb + 31) / 32 * 32;
  const int nqt = (int)((valid + WGR - 1) / WGR);
  const int h = (int)H;
  hipLaunchKernelGGL(gfx1201_sol_pool_kv, dim3(nkbp, h), dim3(D), 0, st, (const int8_t*)K, (const float*)KS,
                     (const uint8_t*)V, (const float*)VS, (float*)kbar, (float*)vsum, (int)Sp, (int)valid, h, nkbp);
  hipLaunchKernelGGL(gfx1201_sol_quant_k, dim3(nkbp / 32, h), dim3(256), 0, st, (const float*)kbar, (int8_t*)kb,
                     (float*)kb_scale, h, nkbp);
  hipLaunchKernelGGL(gfx1201_sol_stats, dim3(h), dim3(D), 0, st, (const float*)kbar, (const float*)vsum,
                     (uint8_t*)vb, (float*)vb_scale, (float*)mu, (float*)var, h, nkb, nkbp);
  hipLaunchKernelGGL(gfx1201_sol_select, dim3(nqt * h), dim3(256), 0, st, (const int8_t*)Q, (const float*)QS,
                     (const float*)kbar, (const float*)mu, (const float*)var, (int32_t*)list, (int32_t*)count,
                     (uint32_t*)mask, (int)Sp, (int)valid, h, nkb, nkbp, nqt, 2 * nkb + 2, (int)prefix, (float)tau);
}

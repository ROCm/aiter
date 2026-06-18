// SPDX-License-Identifier: MIT
// Torch-free standalone driver for the CK unified-attention kernel, built so
// that rocprofv3 ATT only has to disassemble THIS executable's (tiny) code
// object -- not PyTorch's ~26MB HIP library, which is what floors the
// python-harness trace at ~2.5 min (see ua-test-scripts/PREFILL_PERF_PLAN.md
// section 5, "Collection-cost learnings").
//
// It fills `unified_attention_args` for the contiguous (is_paged=false),
// non-causal/causal, fp8 prefill instance exactly as the production glue does
// (csrc/py_itfs_ck/unified_attention_ck_kernels.cu) but with raw HIP buffers.
//
// Inputs are RANDOM (normal). An optional float reference (CHECK=1) recomputes
// O = softmax(scale * Q.K^T [+ causal mask]) . V on the host and compares, so
// we can be sure the bench setup (layouts / strides / fp8 encoding / GQA map)
// is faithful and isn't silently corrupting the profiled kernel's results.
//
// IMPORTANT (fp8): build.sh forces -DCK_TILE_USE_OCP_FP8=1 so the HOST agrees
// with the DEVICE on the e4m3 encoding (OCP e4m3fn on gfx950). Without it the
// host pass defaults to e4m3fnuz and the random fill bytes would be decoded
// differently by the kernel -> spurious mismatch.
//
// Usage: ua_trace_main [sq] [hq] [hk] [d] [mask] [iters]
//   sq=8192 hq=16 hk=2 d=128 mask=0(non-causal)/2(causal) iters=3   (defaults)
// Env:
//   CHECK=1      run the host reference accuracy check (use a SMALL sq -- the
//                reference is O(hq * sq^2 * d) on the CPU)
//   PAGED=1      exercise the PAGED address path (default 0 = contiguous nopage).
//                Uses an IDENTITY block table so the paged pool is byte-identical
//                to contiguous -> same DRAM pattern, isolates page-index overhead.
//                Build the matching instance, e.g.
//                  TARGET_INSTANCE=unified_attention_d128_fp8_nmask_ps128 ./build.sh
//   PAGE_BLK=128 paging granularity (tokens/page). 128 == the kv tile (one page
//                per tile, kRebaseKSrd fast path); 16/32/64 span multiple pages.
//                Must match the built ps{N} instance.
//   SK           KV context length (default = sq). Set sq=1 SK=<context> to
//                profile the decode tier (selected via max_seqlen_q==sq). The
//                decode instances are paged-only, so pair with PAGED=1.
//   NUM_SEQS     number of independent sequences (default 1). Decode grid is
//                dim3(num_kv_heads, num_seqs, num_splits) -> this is the knob
//                for resident CTAs/CU (production decode runs batch>1). sq/sk
//                are PER-SEQ; each seq gets its own KV region.
//   NUM_SPLITS   split-KV count (gridDim.z). 0/unset = production heuristic
//                (aiter/ops/unified_attention.py _pick_num_splits); >=1 forces
//                a value. >1 launches the 3D split grid, writes fp32
//                (o_acc,lse_acc) partials, and the CHECK path merges them.
//   SEED=42      RNG seed
//   QKV_STD=1.0  stddev of the normal fill for Q/K/V
//   RTOL=2e-2 ATOL=1e-2   accuracy tolerances (|got-ref| <= ATOL + RTOL*|ref|)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "unified_attention.hpp"  // pulls in ck_tile fp8_t / bf16_t (don't include
                                  // float8.hpp directly -- it has no re-include guard)

using fp8_t  = ck_tile::fp8_t;   // OCP e4m3fn on gfx950 (forced via -DCK_TILE_USE_OCP_FP8=1)
using bf16_t = ck_tile::bf16_t;

#define HIP_CHECK(expr)                                                                 \
    do {                                                                                \
        hipError_t _e = (expr);                                                         \
        if(_e != hipSuccess) {                                                          \
            std::cerr << "HIP error " << hipGetErrorString(_e) << " at " << __FILE__    \
                      << ":" << __LINE__ << std::endl;                                  \
            std::exit(1);                                                               \
        }                                                                               \
    } while(0)

template <typename T>
static T* device_from(const std::vector<T>& host)
{
    T* p = nullptr;
    HIP_CHECK(hipMalloc(&p, host.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

static int32_t* device_i32(const std::vector<int32_t>& host)
{
    int32_t* p = nullptr;
    HIP_CHECK(hipMalloc(&p, host.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(p, host.data(), host.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    return p;
}

static double env_d(const char* k, double dflt)
{
    const char* s = std::getenv(k);
    return s ? std::atof(s) : dflt;
}

// Host reference: O[i,h,:] = sum_j softmax_j( scale * <Q[i,h], K[j,kv]> ) * V[j,kv],
// reading the SAME fp8 bytes the kernel reads (cast to float). GQA: kv = h / nr.
// Causal (mask==2, bottom-right anchor): key j valid iff j <= i + (sk - sq).
template <typename T>
static std::vector<float> reference(const std::vector<T>& q,
                                    const std::vector<T>& k,
                                    const std::vector<T>& v,
                                    int sq, int sk, int hq, int hk, int d, int mask,
                                    float scale_s, float q_descale, float k_descale,
                                    float v_descale)
{
    const int   nr      = hq / hk;
    const float qk_scale = scale_s * q_descale * k_descale;
    const long  off      = (long)sk - (long)sq;  // bottom-right causal anchor

    std::vector<float> o((size_t)sq * hq * d, 0.f);
    std::vector<float> sc(sk);

    for(int h = 0; h < hq; ++h)
    {
        const int kv = h / nr;
        for(int i = 0; i < sq; ++i)
        {
            const T* qrow = &q[((size_t)i * hq + h) * d];
            float m = -std::numeric_limits<float>::infinity();
            for(int j = 0; j < sk; ++j)
            {
                if(mask == 2 && (long)j > (long)i + off) { sc[j] = -std::numeric_limits<float>::infinity(); continue; }
                const T* krow = &k[((size_t)j * hk + kv) * d];
                float dot = 0.f;
                for(int e = 0; e < d; ++e)
                    dot += ck_tile::type_convert<float>(qrow[e]) * ck_tile::type_convert<float>(krow[e]);
                sc[j] = dot * qk_scale;
                if(sc[j] > m) m = sc[j];
            }
            float sum = 0.f;
            for(int j = 0; j < sk; ++j)
            {
                if(sc[j] == -std::numeric_limits<float>::infinity()) { sc[j] = 0.f; continue; }
                sc[j] = std::exp(sc[j] - m);
                sum += sc[j];
            }
            const float inv = sum > 0.f ? 1.f / sum : 0.f;
            float* orow = &o[((size_t)i * hq + h) * d];
            for(int j = 0; j < sk; ++j)
            {
                if(sc[j] == 0.f) continue;
                const float p = sc[j] * inv;
                const T* vrow = &v[((size_t)j * hk + kv) * d];
                for(int e = 0; e < d; ++e)
                    orow[e] += p * ck_tile::type_convert<float>(vrow[e]);
            }
            for(int e = 0; e < d; ++e)
                orow[e] *= v_descale;
        }
    }
    return o;
}

// ---------------------------------------------------------------------------
// GPU reference: a deliberately NAIVE attention kernel, one thread per output
// row (h,i), flash-style online softmax so no O(sk) scratch is needed. It is
// INDEPENDENT of the CK unified_attention pipeline (own code, own math) so it
// is a real cross-check, not a kernel that could share the same bug. It reads
// the SAME fp8 bytes (device-side ck_tile::type_convert) and reproduces the
// host reference()'s math (max-subtracted softmax, GQA kv=h/nr, bottom-right
// causal anchor, V descale) exactly -- just ~1000x faster because it runs on
// the GPU. Result matches reference() to fp32 rounding, so the same loose fp8
// tolerance applies. Use REF=cpu to fall back to the host loop.
// d is bounded by kRefMaxD (these instances are d<=128); guarded in main().
static constexpr int kRefMaxD = 256;

// Multi-seq aware: each of `total_q` global query tokens belongs to seq =
// token/sq (equal sq per seq) and attends only to its own KV region, whose
// physical base token is seq*kv_seq_stride (kv_seq_stride = pages_per_seq*
// page_blk for paged, sk for contiguous). sq/sk are PER-SEQ lengths.
template <typename T>
__global__ void ref_attn_kernel(const T* __restrict__ q,
                                const T* __restrict__ k,
                                const T* __restrict__ v,
                                float* __restrict__ o,
                                int total_q, int sq, int sk, int hq, int hk, int d, int mask,
                                long kv_seq_stride, float qk_scale, float v_descale)
{
    const long row = (long)blockIdx.x * blockDim.x + threadIdx.x; // 0..hq*total_q
    if(row >= (long)hq * total_q)
        return;
    const int  h       = (int)(row % hq); // O/Q layout is [token, head, d]
    const int  i       = (int)(row / hq); // global query token
    const int  seq     = i / sq;
    const int  i_local = i % sq;          // query index within its seq
    const int  nr      = hq / hk;
    const int  kv      = h / nr;
    const long off     = (long)sk - (long)sq; // bottom-right causal anchor (per-seq)
    const long kv_base = (long)seq * kv_seq_stride;

    const T*     qrow = &q[((long)i * hq + h) * d];
    float        acc[kRefMaxD];
    for(int e = 0; e < d; ++e)
        acc[e] = 0.f;
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.f;

    for(int j = 0; j < sk; ++j)
    {
        if(mask == 2 && (long)j > (long)i_local + off)
            break; // causal: no valid keys past the diagonal
        const T*     krow = &k[((kv_base + j) * hk + kv) * d];
        float        dot  = 0.f;
        for(int e = 0; e < d; ++e)
            dot += ck_tile::type_convert<float>(qrow[e]) * ck_tile::type_convert<float>(krow[e]);
        const float s     = dot * qk_scale;
        const float m_new = fmaxf(m, s);
        const float corr  = expf(m - m_new); // 0 on the first valid key (m=-inf)
        const float p     = expf(s - m_new);
        l                 = l * corr + p;
        const T*     vrow = &v[((kv_base + j) * hk + kv) * d];
        for(int e = 0; e < d; ++e)
            acc[e] = acc[e] * corr + p * ck_tile::type_convert<float>(vrow[e]);
        m = m_new;
    }

    const float inv  = l > 0.f ? 1.f / l : 0.f; // masked-empty row -> zeros
    float*      orow = &o[((long)i * hq + h) * d];
    for(int e = 0; e < d; ++e)
        orow[e] = acc[e] * inv * v_descale;
}

template <typename T>
static std::vector<float> reference_gpu(const T* q, const T* k, const T* v,
                                        int total_q, int sq, int sk, int hq, int hk, int d,
                                        int mask, long kv_seq_stride,
                                        float scale_s, float q_descale, float k_descale,
                                        float v_descale)
{
    const float qk_scale = scale_s * q_descale * k_descale;
    const size_t n       = (size_t)total_q * hq * d;
    float*       o_dev   = nullptr;
    HIP_CHECK(hipMalloc(&o_dev, n * sizeof(float)));

    const long  rows  = (long)hq * total_q;
    const int   block = 128;
    const int   grid  = (int)((rows + block - 1) / block);
    ref_attn_kernel<T><<<grid, block>>>(q, k, v, o_dev, total_q, sq, sk, hq, hk, d, mask,
                                        kv_seq_stride, qk_scale, v_descale);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> o_host(n);
    HIP_CHECK(hipMemcpy(o_host.data(), o_dev, n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(o_dev));
    return o_host;
}

// ---------------------------------------------------------------------------
// Split-KV count heuristic, mirroring aiter/ops/unified_attention.py
// (_pick_num_splits) so the standalone launches the SAME 3D grid (gridDim.z ==
// num_splits) the production wrapper would for this shape. NUM_SPLITS env
// overrides: >=1 forces that value, 0 (default) runs the heuristic.
//   base_ctas   = num_kv_heads * q_tiles
//   occ_splits  = ceil(num_cus * 2 / base_ctas)          # fill ~2 waves/CU
//   work_splits = sk_ub // MIN_SPLIT_KV_TOKENS           # combine-amortize cap
//   num_splits  = clamp(next_pow2(min(occ, work)), 1, num_cus*2)
// with the prefill saturation guard (avg_q>8 && base_ctas>=num_cus -> 1).
// sk_ub is the per-seq KV upper bound (pages_per_seq * page_blk in production).
static int pick_num_splits(long total_q, int hq, int hk, int num_seqs, int num_cus,
                           long sk_ub)
{
    if(const char* e = std::getenv("NUM_SPLITS"))
    {
        const int v = std::atoi(e);
        if(v >= 1) return v;           // explicit force (0 -> fall through to heuristic)
    }
    if(num_seqs <= 0) return 1;
    const int  num_qpkv = hq / hk;
    const long avg_q    = total_q / num_seqs;
    int kBlockQ;
    if(num_qpkv == 1)
        kBlockQ = avg_q <= 16 ? 16 : avg_q <= 32 ? 32 : avg_q <= 128 ? 128 : 256;
    else
    {
        const int tiny = 16 / num_qpkv, small = 64 / num_qpkv;
        kBlockQ = avg_q <= tiny ? tiny : avg_q <= small ? small : 128 / num_qpkv;
    }
    kBlockQ            = std::max(1, kBlockQ);
    const long q_tiles = std::max(1L, (total_q + kBlockQ - 1) / kBlockQ);
    const long base    = (long)hk * q_tiles;
    if(avg_q > 8 && base >= num_cus) return 1;   // prefill already saturates
    const long hard_cap   = (long)num_cus * 2;
    const long occ_splits = (hard_cap + base - 1) / std::max(1L, base);
    long min_split_kv = 128;
    if(const char* e = std::getenv("AITER_UA_MIN_SPLIT_KV_TOKENS"))
        min_split_kv = std::max(1, std::atoi(e));
    const long work_splits = std::max(1L, sk_ub / std::max(1L, min_split_kv));
    const long raw         = std::min(occ_splits, work_splits);
    int pow2 = 1;
    while(pow2 < raw) pow2 <<= 1;
    return std::max(1, (int)std::min(hard_cap, (long)pow2));
}

// FlashDecoding LSE merge: combine the per-split fp32 partials into the bf16
// output. One thread per (token, head). Layouts:
//   o_acc   [hq, splits, total_q, d]   lse_acc [hq, splits, total_q]
//   out     [total_q, hq, d]           (output_stride_0 = hq*d, _1 = d)
__global__ void combine_splits_kernel(const float* __restrict__ o_acc,
                                      const float* __restrict__ lse_acc,
                                      bf16_t* __restrict__ out,
                                      int total_q, int hq, int d, int num_splits)
{
    const long row = (long)blockIdx.x * blockDim.x + threadIdx.x; // 0..total_q*hq
    if(row >= (long)total_q * hq) return;
    const int t = (int)(row / hq);
    const int h = (int)(row % hq);

    const long lse_h = (long)h * num_splits * total_q;
    const long o_h   = (long)h * num_splits * total_q * d;

    float m = -std::numeric_limits<float>::infinity();
    for(int s = 0; s < num_splits; ++s)
        m = fmaxf(m, lse_acc[lse_h + (long)s * total_q + t]);

    float wsum = 0.f;
    for(int s = 0; s < num_splits; ++s)
    {
        const float l = lse_acc[lse_h + (long)s * total_q + t];
        if(l != -std::numeric_limits<float>::infinity()) wsum += expf(l - m);
    }
    const float inv = wsum > 0.f ? 1.f / wsum : 0.f;

    bf16_t* orow = &out[(long)t * hq * d + (long)h * d];
    for(int e = 0; e < d; ++e)
    {
        float acc = 0.f;
        for(int s = 0; s < num_splits; ++s)
        {
            const float l = lse_acc[lse_h + (long)s * total_q + t];
            if(l == -std::numeric_limits<float>::infinity()) continue;
            const float w = expf(l - m) * inv;
            acc += w * o_acc[o_h + (long)s * total_q * d + (long)t * d + e];
        }
        orow[e] = ck_tile::type_convert<bf16_t>(acc);
    }
}

// ---------------------------------------------------------------------------
// COMBINE VARIANTS (gated by env COMBINE=1, selected by COMBINE_KIND).
// All consume the same fp32 partials the main split-KV kernel wrote:
//   o_acc [hq, splits, total_q, d]   lse_acc [hq, splits, total_q]
// and produce the bf16 output [total_q, hq, d]. We time them in isolation to
// quantify the combine overhead the e2e A/B showed exploding with num_splits
// (b=1: ~20us @512 -> ~92us @1024). The serial kernel is the production-shaped
// reduce (1 thread per (token,head)); the others are the fix candidates.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float blk_max(float v, float* sh)
{
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for(int o = blockDim.x >> 1; o > 0; o >>= 1)
    { if(t < o) sh[t] = fmaxf(sh[t], sh[t + o]); __syncthreads(); }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float blk_sum(float v, float* sh)
{
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for(int o = blockDim.x >> 1; o > 0; o >>= 1)
    { if(t < o) sh[t] += sh[t + o]; __syncthreads(); }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// PARALLEL REDUCE: one block per (token,head); threads cooperate across splits
// (max + wsum block-reduce, weights cached in shared) then across the d dims.
// Dynamic shared = (blockDim.x + num_splits) floats.
__global__ void combine_par_kernel(const float* __restrict__ o_acc,
                                   const float* __restrict__ lse_acc,
                                   bf16_t* __restrict__ out,
                                   int total_q, int hq, int d, int num_splits)
{
    extern __shared__ float smem[];
    float* sred = smem;                 // blockDim.x scratch for reductions
    float* sw   = smem + blockDim.x;     // num_splits cached weights/lse
    const long row = blockIdx.x;
    if(row >= (long)total_q * hq) return;
    const int  t     = (int)(row / hq);
    const int  h     = (int)(row % hq);
    const long lse_h = (long)h * num_splits * total_q;
    const long o_h   = (long)h * num_splits * total_q * d;
    const int  tid   = threadIdx.x;

    float lm = -std::numeric_limits<float>::infinity();
    for(int s = tid; s < num_splits; s += blockDim.x)
    { const float l = lse_acc[lse_h + (long)s * total_q + t]; sw[s] = l; lm = fmaxf(lm, l); }
    const float m = blk_max(lm, sred);

    float ls = 0.f;
    for(int s = tid; s < num_splits; s += blockDim.x)
    { const float l = sw[s]; const float w = (l == -std::numeric_limits<float>::infinity()) ? 0.f : __expf(l - m);
      sw[s] = w; ls += w; }
    const float wsum = blk_sum(ls, sred);
    const float inv  = wsum > 0.f ? 1.f / wsum : 0.f;

    bf16_t* orow = &out[(long)t * hq * d + (long)h * d];
    for(int e = tid; e < d; e += blockDim.x)
    {
        float acc = 0.f;
        for(int s = 0; s < num_splits; ++s)
        { const float w = sw[s]; if(w != 0.f) acc += w * o_acc[o_h + (long)s * total_q * d + (long)t * d + e]; }
        orow[e] = ck_tile::type_convert<bf16_t>(acc * inv);
    }
}

// ATOMIC ADD (the "no buffer, atomic into output" idea). Correctness still needs
// the global per-(token,head) softmax max, so phase 1 computes (m, inv) exactly
// like the reduce; phase 2 has every (row, split) block atomicAdd its rescaled
// contribution into an fp32 accumulator (contended: num_splits adds per element).
__global__ void combine_atomic_prep(const float* __restrict__ lse_acc,
                                     float* __restrict__ m_out, float* __restrict__ inv_out,
                                     int total_q, int hq, int num_splits)
{
    extern __shared__ float smem[];
    const long row = blockIdx.x;
    if(row >= (long)total_q * hq) return;
    const int  t     = (int)(row / hq);
    const int  h     = (int)(row % hq);
    const long lse_h = (long)h * num_splits * total_q;
    const int  tid   = threadIdx.x;
    float lm = -std::numeric_limits<float>::infinity();
    for(int s = tid; s < num_splits; s += blockDim.x)
        lm = fmaxf(lm, lse_acc[lse_h + (long)s * total_q + t]);
    const float m = blk_max(lm, smem);
    float ls = 0.f;
    for(int s = tid; s < num_splits; s += blockDim.x)
    { const float l = lse_acc[lse_h + (long)s * total_q + t];
      if(l != -std::numeric_limits<float>::infinity()) ls += __expf(l - m); }
    const float wsum = blk_sum(ls, smem);
    if(tid == 0) { m_out[row] = m; inv_out[row] = wsum > 0.f ? 1.f / wsum : 0.f; }
}
// grid = (total_q*hq, num_splits); block = d. Each thread atomicAdds into oat.
__global__ void combine_atomic_accum(const float* __restrict__ o_acc,
                                     const float* __restrict__ lse_acc,
                                     const float* __restrict__ m_in, const float* __restrict__ inv_in,
                                     float* __restrict__ oat,
                                     int total_q, int hq, int d, int num_splits)
{
    const long row = blockIdx.x;
    const int  s   = blockIdx.y;
    const int  e   = threadIdx.x;
    if(row >= (long)total_q * hq || e >= d) return;
    const int  t     = (int)(row / hq);
    const int  h     = (int)(row % hq);
    const long lse_h = (long)h * num_splits * total_q;
    const long o_h   = (long)h * num_splits * total_q * d;
    const float l = lse_acc[lse_h + (long)s * total_q + t];
    if(l == -std::numeric_limits<float>::infinity()) return;
    const float w = __expf(l - m_in[row]) * inv_in[row];
    atomicAdd(&oat[row * d + e], w * o_acc[o_h + (long)s * total_q * d + (long)t * d + e]);
}
// CHUNKED atomic: grid = (rows, ceil(splits/chunk)); each block serially folds
// `chunk` splits in registers, then ONE atomicAdd per element. This cuts both
// the serial split-walk (par's bottleneck, now `chunk` long) and the atomic
// contention depth (plain-atomic's, now splits/chunk). Standard segmented reduce.
__global__ void combine_atomic_accum_chunked(const float* __restrict__ o_acc,
                                             const float* __restrict__ lse_acc,
                                             const float* __restrict__ m_in, const float* __restrict__ inv_in,
                                             float* __restrict__ oat,
                                             int total_q, int hq, int d, int num_splits, int chunk)
{
    const long row = blockIdx.x;
    const int  s0  = blockIdx.y * chunk;
    const int  e   = threadIdx.x;
    if(row >= (long)total_q * hq || e >= d) return;
    const int  t     = (int)(row / hq);
    const int  h     = (int)(row % hq);
    const long lse_h = (long)h * num_splits * total_q;
    const long o_h   = (long)h * num_splits * total_q * d;
    const float m = m_in[row], inv = inv_in[row];
    float acc = 0.f;
    const int s1 = ck_tile::min(s0 + chunk, num_splits);
    for(int s = s0; s < s1; ++s)
    {
        const float l = lse_acc[lse_h + (long)s * total_q + t];
        if(l == -std::numeric_limits<float>::infinity()) continue;
        const float w = __expf(l - m) * inv;
        acc += w * o_acc[o_h + (long)s * total_q * d + (long)t * d + e];
    }
    if(acc != 0.f) atomicAdd(&oat[row * d + e], acc);
}
__global__ void combine_atomic_finalize(const float* __restrict__ oat, bf16_t* __restrict__ out,
                                        int total_q, int hq, int d)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= (long)total_q * hq * d) return;
    const int  t = (int)(i / ((long)hq * d));
    const int  r = (int)(i % ((long)hq * d));
    const int  h = r / d, e = r % d;
    out[(long)t * hq * d + (long)h * d + e] = ck_tile::type_convert<bf16_t>(oat[i]);
}

// LAST-CTA (Lean fused proxy): one block per (kv_head, token) output tile merges
// ALL nqpkv query heads of that group over splits. This is what the last-arriving
// split CTA would have to do in a fused kernel -- so it measures whether dumping
// a whole tile's reduction on a single CTA is viable at high split counts (the
// pitfall: at low batch there's only one tile -> no cross-CTA parallelism).
// grid = dim3(hk, total_q); block = 256, threads parallel over (local_head, e).
__global__ void combine_lastcta_kernel(const float* __restrict__ o_acc,
                                       const float* __restrict__ lse_acc,
                                       bf16_t* __restrict__ out,
                                       int total_q, int hq, int hk, int d, int num_splits)
{
    const int nqpkv = hq / hk;
    const int kvh   = blockIdx.x;
    const int t     = blockIdx.y;
    const int work  = nqpkv * d;
    for(int idx = threadIdx.x; idx < work; idx += blockDim.x)
    {
        const int  lh = idx / d, e = idx % d;
        const int  h  = kvh * nqpkv + lh;
        const long lse_h = (long)h * num_splits * total_q;
        const long o_h   = (long)h * num_splits * total_q * d;
        float m = -std::numeric_limits<float>::infinity();
        for(int s = 0; s < num_splits; ++s) m = fmaxf(m, lse_acc[lse_h + (long)s * total_q + t]);
        float wsum = 0.f, acc = 0.f;
        for(int s = 0; s < num_splits; ++s)
        {
            const float l = lse_acc[lse_h + (long)s * total_q + t];
            if(l == -std::numeric_limits<float>::infinity()) continue;
            const float w = __expf(l - m);
            wsum += w;
            acc  += w * o_acc[o_h + (long)s * total_q * d + (long)t * d + e];
        }
        out[(long)t * hq * d + (long)h * d + e] =
            ck_tile::type_convert<bf16_t>(wsum > 0.f ? acc / wsum : 0.f);
    }
}

int main(int argc, char** argv)
{
    using DT = ck_tile::unified_attention_args::data_type_enum;

    // The driver historically hard-coded fp8 inputs. The input element type AND
    // args.data_type must match the instance the slim build compiled as the real
    // (non-stub) kernel -- build.sh derives both from DTYPE. A mismatch lands the
    // runtime dispatch on a stubbed instance ("no matching kernel"). UA_TRACE_BF16
    // (injected by build.sh when DTYPE=bf16) switches inputs + data_type to bf16.
#if defined(UA_TRACE_BF16)
    using qkv_t                  = bf16_t;
    constexpr DT kTraceDtype     = DT::bf16;
#else
    using qkv_t                  = fp8_t;
    constexpr DT kTraceDtype     = DT::fp8;
#endif

    const ck_tile::index_t sq   = argc > 1 ? std::atoi(argv[1]) : 8192;
    const ck_tile::index_t hq   = argc > 2 ? std::atoi(argv[2]) : 16;
    const ck_tile::index_t hk   = argc > 3 ? std::atoi(argv[3]) : 2;
    const ck_tile::index_t d    = argc > 4 ? std::atoi(argv[4]) : 128;
    const int              mask = argc > 5 ? std::atoi(argv[5]) : 0;
    const int              iters = argc > 6 ? std::atoi(argv[6]) : 3;

    // SK decouples the KV context length from the query length. Default sk==sq
    // (self-attention prefill); set SK to profile decode (sq=1, sk=context),
    // which selects the decode tier via max_seqlen_q==sq below. The decode
    // tiers are paged-only instances, so a decode run also needs PAGED=1.
    //
    // NUM_SEQS replicates the shape across `num_seqs` independent sequences
    // (each sq queries / sk context, its own KV region) -- the decode grid is
    // dim3(num_kv_heads, num_seqs, num_splits), so num_seqs is the direct knob
    // for resident CTAs/CU (production decode runs batch>1). sq/sk stay PER-SEQ.
    const char*            sk_env    = std::getenv("SK");
    const ck_tile::index_t sk        = sk_env ? (ck_tile::index_t)std::atol(sk_env) : sq;
    const ck_tile::index_t num_seqs  = std::getenv("NUM_SEQS")
                                           ? (ck_tile::index_t)std::max(1, std::atoi(std::getenv("NUM_SEQS")))
                                           : 1;
    const ck_tile::index_t total_q   = sq * num_seqs;        // all query tokens
    const ck_tile::index_t total_kv  = sk * num_seqs;        // contiguous KV pool tokens
    const ck_tile::index_t nqpkv     = hq / hk;

    const bool         do_check = std::getenv("CHECK") && std::atoi(std::getenv("CHECK")) != 0;
    const bool         do_perf  = std::getenv("PERF") && std::atoi(std::getenv("PERF")) != 0;
    const int          warmup   = (int)env_d("WARMUP", 5);
    // Rotate among ROTATE independent Q/K/V copies so each timed iter reads
    // fresh DRAM (defeats L2 reuse) -- mirrors @perftest's num_rotate_args, so
    // bandwidth/TFLOPs stay comparable to the Python harness. Bump for small
    // shapes that fit in L2; 1 = reuse (fine for large compute-bound shapes).
    const int          rotate   = std::max(1, (int)env_d("ROTATE", 1));
    const unsigned     seed     = std::getenv("SEED") ? (unsigned)std::atoi(std::getenv("SEED")) : 42u;
    const float        qkv_std  = (float)env_d("QKV_STD", 1.0f);
    const float        q_descale = 1.0f, k_descale = 1.0f, v_descale = 1.0f;
    const float        scale_s   = 1.0f / std::sqrt((float)d);

    // ---- paging config (PAGED=1) -------------------------------------------
    // Exercise the PAGED address path on the SAME canonical shape. With an
    // IDENTITY block table (logical page p -> physical block p) the paged K/V
    // pool is byte-identical to the contiguous one, so (a) the host/GPU
    // reference is unchanged and (b) the DRAM access pattern matches contiguous
    // EXACTLY. That isolates the pure page-index-fetch / address-computation
    // overhead -- the thing we want at ~0 -- with no page-scatter confound.
    //   PAGED=1        enable the paged instance (default 0 = contiguous nopage)
    //   PAGE_BLK=128   paging granularity (tokens/page). 128 == the kv tile, so
    //                  one page per tile (kRebaseKSrd fast path). 16/32/64 span
    //                  multiple pages (dedup path). Must match the built instance.
    const bool paged          = std::getenv("PAGED") && std::atoi(std::getenv("PAGED")) != 0;
    const int  page_blk       = std::max(1, (int)env_d("PAGE_BLK", 128));
    const int  pages_per_seq  = paged ? (int)(((long)sk + page_blk - 1) / page_blk) : 0;
    const int  num_pages      = pages_per_seq * (int)num_seqs;          // total physical pages
    // Tokens between consecutive per-seq KV bases in the physical pool: a whole
    // (padded) page run for paged, exactly sk for contiguous.
    const long kv_seq_stride  = paged ? (long)pages_per_seq * page_blk : (long)sk;
    const long padded_kv      = kv_seq_stride * (long)num_seqs;        // total pool tokens

    // ---- random host inputs (fp8 in, bf16 out) ----
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.f, qkv_std);
    // NOTE: fp8_t/bf16_t are _BitInt/native types here (CK_TILE_USE_CUSTOM_DATA_TYPE=0),
    // so float<->fp8 MUST go through ck_tile::type_convert (a static_cast would
    // truncate to an integer). type_convert honours CK_TILE_USE_OCP_FP8=1 (forced
    // in build.sh) so the host encodes the SAME e4m3fn bytes the gfx950 kernel reads.
    auto fill_qkv = [&](size_t n) {
        std::vector<qkv_t> h(n);
        for(size_t i = 0; i < n; ++i) h[i] = ck_tile::type_convert<qkv_t>(nd(gen));
        return h;
    };
    // KV pool spans padded_kv = kv_seq_stride*num_seqs tokens (per-seq region +
    // page padding). Filling it all random is fine: the kernel reads only the
    // first sk tokens of each seq region (seq_len=sk), the page-padding tail is
    // never touched, and the reference reads the same [kv_base, kv_base+sk).
    std::vector<qkv_t>  q_h = fill_qkv((size_t)total_q * hq * d);
    std::vector<qkv_t>  k_h = fill_qkv((size_t)padded_kv * hk * d);
    std::vector<qkv_t>  v_h = fill_qkv((size_t)padded_kv * hk * d);
    std::vector<bf16_t> o_h((size_t)total_q * hq * d, ck_tile::type_convert<bf16_t>(0.0f));

    qkv_t*  q = device_from(q_h);
    qkv_t*  k = device_from(k_h);
    qkv_t*  v = device_from(v_h);
    bf16_t* o = device_from(o_h);

    // Rotation copies (copy 0 == q/k/v); distinct DRAM addresses defeat L2 reuse.
    std::vector<qkv_t*> q_rot{q}, k_rot{k}, v_rot{v};
    for(int r = 1; r < rotate; ++r)
    {
        q_rot.push_back(device_from(q_h));
        k_rot.push_back(device_from(k_h));
        v_rot.push_back(device_from(v_h));
    }

    // Per-seq block table for paged mode: [num_seqs, pages_per_seq], seq s's
    // logical page i -> physical page s*pages_per_seq + i. Each seq owns a
    // DISTINCT physical page run, so its KV lives at distinct DRAM addresses
    // (no cross-seq L2 reuse confound). block_table_stride = pages_per_seq.
    std::vector<int32_t> bt_host;
    if(paged) { bt_host.resize(num_pages); for(int i = 0; i < num_pages; ++i) bt_host[i] = i; }
    else      { bt_host = {0}; }                              // ignored when !is_paged
    // seq_lens [num_seqs] = sk each; query_start_len/kv_start_len are the
    // cumulative [num_seqs+1] offsets into the packed query / contiguous-KV pools.
    std::vector<int32_t> sl_host(num_seqs, (int32_t)sk);
    std::vector<int32_t> qsl_host(num_seqs + 1), ksl_host(num_seqs + 1);
    for(ck_tile::index_t s = 0; s <= num_seqs; ++s)
    { qsl_host[s] = (int32_t)(s * sq); ksl_host[s] = (int32_t)(s * sk); }
    int32_t* block_tables    = device_i32(bt_host);
    int32_t* seq_lens        = device_i32(sl_host);
    int32_t* query_start_len = device_i32(qsl_host);
    int32_t* kv_start_len    = device_i32(ksl_host);

    // ---- split-KV (FlashDecoding) ------------------------------------------
    // The production wrapper (aiter/ops/unified_attention.py) launches a 3D grid
    // with gridDim.z == num_splits to fan the KV range across CTAs (essential to
    // fill the machine in decode, where the base grid num_kv_heads*num_seqs is
    // tiny). Mirror that here so the profiled grid matches deployment. When
    // num_splits>1 the kernel writes fp32 (o_acc,lse_acc) partials and a combine
    // merges them into o; num_splits==1 writes o_ptr directly (no workspace).
    int num_cus = 0;
    { hipDeviceProp_t p{}; HIP_CHECK(hipGetDeviceProperties(&p, 0)); num_cus = p.multiProcessorCount; }
    // sk upper bound = max_num_blocks_per_seq * page_blk (paged) -- the same
    // capture-safe shape-derived bound production reads from block_tables.
    const long sk_ub = paged ? (long)pages_per_seq * page_blk : (long)sk;
    const int num_splits = pick_num_splits(total_q, hq, hk, num_seqs, num_cus, sk_ub);

    float* o_acc   = nullptr;   // [hq, num_splits, total_q, d]   fp32
    float* lse_acc = nullptr;   // [hq, num_splits, total_q]      fp32
    if(num_splits > 1)
    {
        const size_t n_o   = (size_t)hq * num_splits * total_q * d;
        const size_t n_lse = (size_t)hq * num_splits * total_q;
        HIP_CHECK(hipMalloc(&o_acc, n_o * sizeof(float)));
        HIP_CHECK(hipMalloc(&lse_acc, n_lse * sizeof(float)));
        // lse seeded to -inf so any split the kernel leaves unwritten (fewer KV
        // pages than splits) is inert in the combine; o_acc zeroed for safety.
        HIP_CHECK(hipMemset(o_acc, 0, n_o * sizeof(float)));
        std::vector<float> lse_init(n_lse, -std::numeric_limits<float>::infinity());
        HIP_CHECK(hipMemcpy(lse_acc, lse_init.data(), n_lse * sizeof(float), hipMemcpyHostToDevice));
    }

    ck_tile::unified_attention_args args{};
    args.data_type          = kTraceDtype;
    args.mask_type          = mask;
    if(mask == 0) { args.window_size_left = -1; args.window_size_right = -1; args.is_top_left = false; }
    else          { args.window_size_left = -1; args.window_size_right = 0;  args.is_top_left = false; }

    args.num_tokens         = total_q;
    // contiguous: page dim folded to 1, num_blks == token count.
    // paged: num_blks == physical page count, page_blk_size == tokens/page.
    args.num_blks           = paged ? num_pages : total_kv;
    args.num_head_q         = hq;
    args.num_queries_per_kv = nqpkv;
    args.page_blk_size      = paged ? page_blk : 1;
    args.hdim               = d;
    args.scale_s            = scale_s;
    args.q_descale = q_descale; args.k_descale = k_descale; args.v_descale = v_descale;

    args.q_ptr          = q;
    args.query_stride_0 = hq * d;
    args.query_stride_1 = d;

    // K/V as 4-D [num_blks, page_blk_size, hk, d].
    //   contiguous: page_blk_size==1 -> stride_0 == stride_1 == hk*d
    //   paged:      stride_0 == page_blk*hk*d (page), stride_1 == hk*d (token-in-page)
    const ck_tile::index_t kv_page_stride =
        paged ? (ck_tile::index_t)((long)page_blk * hk * d) : (ck_tile::index_t)(hk * d);
    args.k_ptr            = k;
    args.stride_k_cache_0 = kv_page_stride;
    args.stride_k_cache_1 = hk * d;
    args.stride_k_cache_2 = d;
    args.stride_k_cache_3 = 1;
    args.v_ptr            = v;
    args.stride_v_cache_0 = kv_page_stride;
    args.stride_v_cache_1 = hk * d;
    args.stride_v_cache_2 = d;
    args.stride_v_cache_3 = 1;

    args.o_ptr           = o;
    args.output_stride_0 = hq * d;
    args.output_stride_1 = d;

    args.block_tables_ptr   = block_tables;
    args.block_table_stride = paged ? pages_per_seq : 1;  // max_num_blocks_per_seq
    args.seq_lens_ptr       = seq_lens;
    args.query_start_len_ptr = query_start_len;
    args.num_seqs           = num_seqs;
    args.max_seqlen_q       = sq;        // tier selector: sq>1 -> prefill, sq==1 -> decode
    args.cache_ptr_int32_overflow_possible = false;
    args.num_splits         = num_splits;
    if(num_splits > 1)
    {
        // Workspaces are [hq, num_splits, total_q, *] contiguous, so the q-token
        // axis stride is implicit (d for o_acc, 1 for lse_acc) and only the
        // nhead/split strides are passed -- matching the .cu glue + kernel.
        args.o_acc_ptr            = o_acc;
        args.lse_acc_ptr          = lse_acc;
        args.nhead_stride_o_acc   = (ck_tile::index_t)((long)num_splits * total_q * d);
        args.split_stride_o_acc   = (ck_tile::index_t)((long)total_q * d);
        args.nhead_stride_lse_acc = (ck_tile::index_t)((long)num_splits * total_q);
        args.split_stride_lse_acc = (ck_tile::index_t)total_q;
    }
    args.is_paged           = paged;     // PAGED=1 -> ps{page_blk} instance
    args.kv_start_len_ptr   = kv_start_len;  // ignored when is_paged

    auto launch = [&](int rot_idx) {
        args.q_ptr = q_rot[rot_idx % rotate];
        args.k_ptr = k_rot[rot_idx % rotate];
        args.v_ptr = v_rot[rot_idx % rotate];
        return ck_tile::unified_attention(args, ck_tile::stream_config{}).first;
    };

    // Warmup. ATT (--att-consecutive-kernels 1) captures the first dispatch
    // under rocprofv3; for PERF we discard these and time a steady-state loop.
    bool launched = false;
    for(int i = 0; i < std::max(1, do_perf ? warmup : iters); ++i)
    {
        launched = launch(i);
        HIP_CHECK(hipDeviceSynchronize());
    }
    if(!launched)
    {
        std::cerr << "unified_attention: no matching kernel (slim build missing this instance?)"
                  << std::endl;
        return 2;
    }
    std::cout << "[ua_trace] launched ok  sq=" << sq << " sk=" << sk << " seqs=" << num_seqs
              << " hq=" << hq << " hk=" << hk
              << " d=" << d << " mask=" << mask << " std=" << qkv_std
              << "  splits=" << num_splits << " (cus=" << num_cus
              << ", grid_ctas=" << (num_seqs * (hq / nqpkv) * num_splits) << ")"
              << (paged ? ("  [paged blk=" + std::to_string(page_blk)
                           + " pages/seq=" + std::to_string(pages_per_seq) + "]")
                        : "  [contiguous]")
              << (do_perf ? "  [perf]" : "") << std::endl;

    if(do_perf)
    {
        // Steady-state timing: one hipEvent pair around `iters` back-to-back
        // launches (rotating inputs), latency = elapsed / iters. Matches the CK
        // example + Python harness cost model so TFLOPs/bandwidth are comparable.
        hipEvent_t ev0, ev1;
        HIP_CHECK(hipEventCreate(&ev0));
        HIP_CHECK(hipEventCreate(&ev1));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(ev0, nullptr));
        for(int i = 0; i < iters; ++i) (void)launch(i);
        HIP_CHECK(hipEventRecord(ev1, nullptr));
        HIP_CHECK(hipEventSynchronize(ev1));
        float ms_total = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms_total, ev0, ev1));
        const double lat_s  = (double)ms_total / 1e3 / iters;
        const double lat_us = lat_s * 1e6;

        // valid (unmasked) (q,k) pairs -- same accounting as the Python harness,
        // summed over all num_seqs sequences. non-causal: sq*sk per seq; causal
        // (bottom-right anchor): key j<=i+(sk-sq).
        const long off = (long)sk - (long)sq;
        long long valid_per_seq = 0;
        if(mask == 0)
            valid_per_seq = (long long)sq * sk;
        else
            for(long i = 0; i < sq; ++i)
            { long c = i + off + 1; if(c < 0) c = 0; if(c > sk) c = sk; valid_per_seq += c; }
        const long long valid = valid_per_seq * (long long)num_seqs;

        const double flops = 4.0 * (double)valid * d * hq;             // QK + PV
        const long long mem = (long long)total_q * hq * d * 1          // Q (fp8)
                            + (long long)num_seqs * sk * hk * d * 1     // K (fp8)
                            + (long long)num_seqs * sk * hk * d * 1     // V (fp8)
                            + (long long)total_q * hq * d * 2;         // O (bf16)
        const double tflops = flops / lat_s / 1e12;
        const double tbs    = (double)mem / lat_s / 1e12;
        std::printf("[perf] lat=%.2f us  %.1f TFLOP/s  %.2f TB/s  "
                    "(iters=%d warmup=%d rotate=%d, valid=%.3g, mem=%.2f MB)\n",
                    lat_us, tflops, tbs, iters, warmup, rotate, (double)valid,
                    mem / 1048576.0);
        HIP_CHECK(hipEventDestroy(ev0));
        HIP_CHECK(hipEventDestroy(ev1));

        // ---- combine A/B (COMBINE=1) ---------------------------------------
        // Time the split-KV combine in isolation on the partials the main loop
        // just wrote. COMBINE_KIND=serial|par|atomic|all (default all). This
        // quantifies the overhead that the e2e A/B showed exploding with splits
        // and compares the production-shaped serial reduce against the fixes.
        const bool combine = std::getenv("COMBINE") && std::atoi(std::getenv("COMBINE")) != 0;
        if(combine && num_splits > 1)
        {
            const std::string kind = std::getenv("COMBINE_KIND") ? std::getenv("COMBINE_KIND") : "all";
            const long rows = (long)total_q * hq;
            // atomic scratch
            float* m_buf   = nullptr; float* inv_buf = nullptr; float* oat = nullptr;
            HIP_CHECK(hipMalloc(&m_buf, rows * sizeof(float)));
            HIP_CHECK(hipMalloc(&inv_buf, rows * sizeof(float)));
            HIP_CHECK(hipMalloc(&oat, rows * d * sizeof(float)));
            const int cblk = 256;

            auto time_kernel = [&](const char* name, auto&& fn) {
                for(int i = 0; i < warmup; ++i) fn();
                HIP_CHECK(hipDeviceSynchronize());
                hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
                HIP_CHECK(hipEventRecord(a, nullptr));
                for(int i = 0; i < iters; ++i) fn();
                HIP_CHECK(hipEventRecord(b, nullptr));
                HIP_CHECK(hipEventSynchronize(b));
                float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
                std::printf("[combine] %-8s lat=%.2f us  (rows=%ld, splits=%d)\n",
                            name, (double)ms / iters * 1e3, rows, num_splits);
                HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipEventDestroy(b));
            };

            if(kind == "serial" || kind == "all")
                time_kernel("serial", [&]{
                    const int grid = (int)((rows + 127) / 128);
                    combine_splits_kernel<<<grid, 128>>>(o_acc, lse_acc, o, total_q, hq, d, num_splits);
                });
            if(kind == "par" || kind == "all")
                time_kernel("par", [&]{
                    const size_t sh = (size_t)(cblk + num_splits) * sizeof(float);
                    combine_par_kernel<<<(int)rows, cblk, sh>>>(o_acc, lse_acc, o, total_q, hq, d, num_splits);
                });
            if(kind == "atomic" || kind == "all")
                time_kernel("atomic", [&]{
                    combine_atomic_prep<<<(int)rows, cblk, cblk * sizeof(float)>>>(
                        lse_acc, m_buf, inv_buf, total_q, hq, num_splits);
                    HIP_CHECK(hipMemsetAsync(oat, 0, rows * d * sizeof(float), nullptr));
                    dim3 ag((unsigned)rows, (unsigned)num_splits, 1);
                    combine_atomic_accum<<<ag, d>>>(o_acc, lse_acc, m_buf, inv_buf, oat,
                                                    total_q, hq, d, num_splits);
                    const long n = rows * d;
                    combine_atomic_finalize<<<(int)((n + 255) / 256), 256>>>(oat, o, total_q, hq, d);
                });
            if(kind == "lastcta" || kind == "all")
                time_kernel("lastcta", [&]{
                    dim3 g((unsigned)hk, (unsigned)total_q, 1);
                    combine_lastcta_kernel<<<g, 256>>>(o_acc, lse_acc, o, total_q, hq, hk, d, num_splits);
                });
            if(kind == "chunk" || kind == "all")
            {
                const int chunk = (int)env_d("COMBINE_CHUNK", 16);
                char nm[16]; std::snprintf(nm, sizeof(nm), "chunk%d", chunk);
                time_kernel(nm, [&]{
                    combine_atomic_prep<<<(int)rows, cblk, cblk * sizeof(float)>>>(
                        lse_acc, m_buf, inv_buf, total_q, hq, num_splits);
                    HIP_CHECK(hipMemsetAsync(oat, 0, rows * d * sizeof(float), nullptr));
                    dim3 ag((unsigned)rows, (unsigned)((num_splits + chunk - 1) / chunk), 1);
                    combine_atomic_accum_chunked<<<ag, d>>>(o_acc, lse_acc, m_buf, inv_buf, oat,
                                                            total_q, hq, d, num_splits, chunk);
                    const long n = rows * d;
                    combine_atomic_finalize<<<(int)((n + 255) / 256), 256>>>(oat, o, total_q, hq, d);
                });
            }
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipFree(m_buf)); HIP_CHECK(hipFree(inv_buf)); HIP_CHECK(hipFree(oat));
        }
    }

    int rc = 0;
    if(do_check)
    {
        // Tolerances mirror op_tests/test_unified_attention_ck.py for fp8: loose
        // (1.5e-1) per-element, and checkAllclose-style PASS if the FRACTION of
        // elements outside tolerance stays below tol_err_ratio (5%). This
        // matches torch.isclose: |ref-got| <= atol + rtol*|got|. The loose
        // bound is the expected fp8 (e4m3) + internal-P-quant approximation,
        // NOT a setup error -- a layout/encoding/GQA bug shows up as gross
        // (catastrophic) error on most elements, which this still catches.
        const double rtol          = env_d("RTOL", 1.5e-1);
        const double atol          = env_d("ATOL", 1.5e-1);
        const double tol_err_ratio = env_d("TOL_ERR_RATIO", 0.05);

        // With split-KV the kernel wrote fp32 partials (o_acc,lse_acc), not o.
        // Merge them into o exactly as the production combine kernel would, so
        // the accuracy check still validates the bench setup end-to-end.
        if(num_splits > 1)
        {
            const long rows = (long)total_q * hq;
            const std::string ckind = std::getenv("COMBINE_KIND") ? std::getenv("COMBINE_KIND") : "serial";
            const int cblk = 256;
            if(ckind == "par")
            {
                const size_t sh = (size_t)(cblk + num_splits) * sizeof(float);
                combine_par_kernel<<<(int)rows, cblk, sh>>>(o_acc, lse_acc, o, total_q, hq, d, num_splits);
            }
            else if(ckind == "lastcta")
            {
                dim3 g((unsigned)hk, (unsigned)total_q, 1);
                combine_lastcta_kernel<<<g, 256>>>(o_acc, lse_acc, o, total_q, hq, hk, d, num_splits);
            }
            else if(ckind == "atomic" || ckind == "chunk")
            {
                const int chunk = (int)env_d("COMBINE_CHUNK", 16);
                float *m_buf, *inv_buf, *oat;
                HIP_CHECK(hipMalloc(&m_buf, rows * sizeof(float)));
                HIP_CHECK(hipMalloc(&inv_buf, rows * sizeof(float)));
                HIP_CHECK(hipMalloc(&oat, rows * d * sizeof(float)));
                combine_atomic_prep<<<(int)rows, cblk, cblk * sizeof(float)>>>(
                    lse_acc, m_buf, inv_buf, total_q, hq, num_splits);
                HIP_CHECK(hipMemset(oat, 0, rows * d * sizeof(float)));
                if(ckind == "chunk")
                {
                    dim3 ag((unsigned)rows, (unsigned)((num_splits + chunk - 1) / chunk), 1);
                    combine_atomic_accum_chunked<<<ag, d>>>(o_acc, lse_acc, m_buf, inv_buf, oat,
                                                            total_q, hq, d, num_splits, chunk);
                }
                else
                {
                    dim3 ag((unsigned)rows, (unsigned)num_splits, 1);
                    combine_atomic_accum<<<ag, d>>>(o_acc, lse_acc, m_buf, inv_buf, oat,
                                                    total_q, hq, d, num_splits);
                }
                const long n = rows * d;
                combine_atomic_finalize<<<(int)((n + 255) / 256), 256>>>(oat, o, total_q, hq, d);
                HIP_CHECK(hipDeviceSynchronize());
                HIP_CHECK(hipFree(m_buf)); HIP_CHECK(hipFree(inv_buf)); HIP_CHECK(hipFree(oat));
            }
            else
            {
                const int grid = (int)((rows + 127) / 128);
                combine_splits_kernel<<<grid, 128>>>(o_acc, lse_acc, o, total_q, hq, d, num_splits);
            }
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
        }

        HIP_CHECK(hipMemcpy(o_h.data(), o, o_h.size() * sizeof(bf16_t), hipMemcpyDeviceToHost));

        // Default to the GPU reference (independent naive kernel, ~ms even at
        // sq=8192). REF=cpu forces the slow host loop (O(hq*sq^2*d)) -- useful
        // as a paranoia cross-check that the GPU reference itself is faithful.
        const char* ref_mode = std::getenv("REF");
        const bool  use_cpu  = ref_mode && std::string(ref_mode) == "cpu";
        std::vector<float> o_ref;
        if(use_cpu)
        {
            if(num_seqs != 1)
            {
                std::cerr << "[check] REF=cpu only supports num_seqs==1; use the GPU reference"
                          << std::endl;
                return 2;
            }
            std::cout << "[check] computing HOST reference (O(hq*sq^2*d), slow) ..." << std::endl;
            o_ref = reference(q_h, k_h, v_h, sq, sk, hq, hk, d, mask,
                              scale_s, q_descale, k_descale, v_descale);
        }
        else
        {
            if(d > kRefMaxD)
            {
                std::cerr << "[check] d=" << d << " exceeds kRefMaxD=" << kRefMaxD
                          << "; use REF=cpu" << std::endl;
                return 2;
            }
            std::cout << "[check] computing GPU reference (naive independent kernel) ..."
                      << std::endl;
            o_ref = reference_gpu(q, k, v, total_q, sq, sk, hq, hk, d, mask, kv_seq_stride,
                                  scale_s, q_descale, k_descale, v_descale);
        }

        double  max_abs = 0.0, ref_at_max = 0.0, got_at_max = 0.0;
        size_t  n_fail = 0, idx_max = 0;
        for(size_t i = 0; i < o_ref.size(); ++i)
        {
            const float ref = o_ref[i];
            const float got = ck_tile::type_convert<float>(o_h[i]);
            const double a  = std::fabs((double)ref - (double)got);
            if(a > atol + rtol * std::fabs((double)got)) ++n_fail;
            if(a > max_abs) { max_abs = a; ref_at_max = ref; got_at_max = got; idx_max = i; }
        }
        const size_t total   = o_ref.size();
        const double fail_pct = total ? (double)n_fail / (double)total : 0.0;
        const bool   pass     = fail_pct < tol_err_ratio;
        std::printf("[check] %s  | mismatch=%.4f%% (%zu/%zu, allowed<%.2f%%)  "
                    "max_abs=%.4g (ref=%.4g got=%.4g @%zu)  (rtol=%.2g atol=%.2g)\n",
                    pass ? "PASS" : "FAIL", 100.0 * fail_pct, n_fail, total,
                    100.0 * tol_err_ratio, max_abs, ref_at_max, got_at_max, idx_max, rtol, atol);
        rc = pass ? 0 : 3;
    }

    for(int r = 0; r < rotate; ++r)
    { HIP_CHECK(hipFree(q_rot[r])); HIP_CHECK(hipFree(k_rot[r])); HIP_CHECK(hipFree(v_rot[r])); }
    HIP_CHECK(hipFree(o));
    if(o_acc)   HIP_CHECK(hipFree(o_acc));
    if(lse_acc) HIP_CHECK(hipFree(lse_acc));
    HIP_CHECK(hipFree(block_tables));    HIP_CHECK(hipFree(seq_lens));
    HIP_CHECK(hipFree(query_start_len)); HIP_CHECK(hipFree(kv_start_len));
    return rc;
}

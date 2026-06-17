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

template <typename T>
__global__ void ref_attn_kernel(const T* __restrict__ q,
                                const T* __restrict__ k,
                                const T* __restrict__ v,
                                float* __restrict__ o,
                                int sq, int sk, int hq, int hk, int d, int mask,
                                float qk_scale, float v_descale)
{
    const long row = (long)blockIdx.x * blockDim.x + threadIdx.x; // 0..hq*sq
    if(row >= (long)hq * sq)
        return;
    const int  h   = (int)(row % hq); // O/Q layout is [token, head, d]
    const int  i   = (int)(row / hq);
    const int  nr  = hq / hk;
    const int  kv  = h / nr;
    const long off = (long)sk - (long)sq; // bottom-right causal anchor

    const T*     qrow = &q[((long)i * hq + h) * d];
    float        acc[kRefMaxD];
    for(int e = 0; e < d; ++e)
        acc[e] = 0.f;
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.f;

    for(int j = 0; j < sk; ++j)
    {
        if(mask == 2 && (long)j > (long)i + off)
            break; // causal: no valid keys past the diagonal
        const T*     krow = &k[((long)j * hk + kv) * d];
        float        dot  = 0.f;
        for(int e = 0; e < d; ++e)
            dot += ck_tile::type_convert<float>(qrow[e]) * ck_tile::type_convert<float>(krow[e]);
        const float s     = dot * qk_scale;
        const float m_new = fmaxf(m, s);
        const float corr  = expf(m - m_new); // 0 on the first valid key (m=-inf)
        const float p     = expf(s - m_new);
        l                 = l * corr + p;
        const T*     vrow = &v[((long)j * hk + kv) * d];
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
                                        int sq, int sk, int hq, int hk, int d, int mask,
                                        float scale_s, float q_descale, float k_descale,
                                        float v_descale)
{
    const float qk_scale = scale_s * q_descale * k_descale;
    const size_t n       = (size_t)sq * hq * d;
    float*       o_dev   = nullptr;
    HIP_CHECK(hipMalloc(&o_dev, n * sizeof(float)));

    const long  rows  = (long)hq * sq;
    const int   block = 128;
    const int   grid  = (int)((rows + block - 1) / block);
    ref_attn_kernel<T><<<grid, block>>>(q, k, v, o_dev, sq, sk, hq, hk, d, mask, qk_scale, v_descale);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> o_host(n);
    HIP_CHECK(hipMemcpy(o_host.data(), o_dev, n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(o_dev));
    return o_host;
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

    const ck_tile::index_t sk        = sq;          // self-attention prefill
    const ck_tile::index_t num_seqs  = 1;
    const ck_tile::index_t total_q   = sq;
    const ck_tile::index_t total_kv  = sk;
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
    const bool paged     = std::getenv("PAGED") && std::atoi(std::getenv("PAGED")) != 0;
    const int  page_blk  = std::max(1, (int)env_d("PAGE_BLK", 128));
    const int  num_pages = paged ? (int)(((long)sk + page_blk - 1) / page_blk) : 0;
    const long padded_kv = paged ? (long)num_pages * page_blk : (long)total_kv;

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
    std::vector<qkv_t>  q_h = fill_qkv((size_t)total_q * hq * d);
    std::vector<qkv_t>  k_h = fill_qkv((size_t)total_kv * hk * d);
    std::vector<qkv_t>  v_h = fill_qkv((size_t)total_kv * hk * d);
    std::vector<bf16_t> o_h((size_t)total_q * hq * d, ck_tile::type_convert<bf16_t>(0.0f));
    if(paged)
    {
        // Pad the pool tail (tokens [sk, num_pages*page_blk)) with zeros. The
        // kernel never reads past seq_len=sk (identity table), so the tail is
        // inert -- it only exists so the buffer descriptor's (num_blks*page_blk)
        // rows stay in-bounds of the allocation.
        k_h.resize((size_t)padded_kv * hk * d, ck_tile::type_convert<qkv_t>(0.f));
        v_h.resize((size_t)padded_kv * hk * d, ck_tile::type_convert<qkv_t>(0.f));
    }

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

    // Identity block table for paged mode (logical page p -> physical block p),
    // so the paged pool is byte-identical to contiguous. [1, num_pages] int32.
    std::vector<int32_t> bt_host;
    if(paged) { bt_host.resize(num_pages); for(int i = 0; i < num_pages; ++i) bt_host[i] = i; }
    else      { bt_host = {0}; }                              // ignored when !is_paged
    int32_t* block_tables    = device_i32(bt_host);
    int32_t* seq_lens        = device_i32({(int32_t)sk});     // [num_seqs]
    int32_t* query_start_len = device_i32({0, (int32_t)sq});  // [num_seqs+1]
    int32_t* kv_start_len    = device_i32({0, (int32_t)sk});  // [num_seqs+1]

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
    args.block_table_stride = paged ? num_pages : 1;  // max_num_blocks_per_seq
    args.seq_lens_ptr       = seq_lens;
    args.query_start_len_ptr = query_start_len;
    args.num_seqs           = num_seqs;
    args.max_seqlen_q       = sq;        // force the prefill_d128 tier
    args.cache_ptr_int32_overflow_possible = false;
    args.num_splits         = 1;
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
    std::cout << "[ua_trace] launched ok  sq=" << sq << " hq=" << hq << " hk=" << hk
              << " d=" << d << " mask=" << mask << " std=" << qkv_std
              << (paged ? ("  [paged blk=" + std::to_string(page_blk)
                           + " pages=" + std::to_string(num_pages) + "]")
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

        // valid (unmasked) (q,k) pairs -- same accounting as the Python harness.
        // non-causal: sq*sk ; causal (bottom-right anchor): key j<=i+(sk-sq).
        const long off = (long)sk - (long)sq;
        long long valid = 0;
        if(mask == 0)
            valid = (long long)sq * sk;
        else
            for(long i = 0; i < sq; ++i)
            { long c = i + off + 1; if(c < 0) c = 0; if(c > sk) c = sk; valid += c; }

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

        HIP_CHECK(hipMemcpy(o_h.data(), o, o_h.size() * sizeof(bf16_t), hipMemcpyDeviceToHost));

        // Default to the GPU reference (independent naive kernel, ~ms even at
        // sq=8192). REF=cpu forces the slow host loop (O(hq*sq^2*d)) -- useful
        // as a paranoia cross-check that the GPU reference itself is faithful.
        const char* ref_mode = std::getenv("REF");
        const bool  use_cpu  = ref_mode && std::string(ref_mode) == "cpu";
        std::vector<float> o_ref;
        if(use_cpu)
        {
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
            o_ref = reference_gpu(q, k, v, sq, sk, hq, hk, d, mask,
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
    HIP_CHECK(hipFree(block_tables));    HIP_CHECK(hipFree(seq_lens));
    HIP_CHECK(hipFree(query_start_len)); HIP_CHECK(hipFree(kv_start_len));
    return rc;
}

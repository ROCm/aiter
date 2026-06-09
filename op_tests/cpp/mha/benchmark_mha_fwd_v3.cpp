// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CK-excluded benchmark host for the fmha_fwd_with_sink_asm ASM path.
// Pair with libmha_fwd_asm.so built via `compile.py --api=fwd_v3` (ck_exclude=True),
// which defines ENABLE_CK=0 and links both csrc/cpp_itfs/mha_fwd.cu and
// csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu into libmha_fwd_asm.so.  The latter
// provides the `fmha_fwd_with_sink_asm` C-ABI entry point used here.
// A distinct library name (libmha_fwd_asm vs libmha_fwd) prevents the JIT blob
// directory from being contaminated by CK-generated blobs from a prior full-CK
// build, which would cause compile errors on gfx1250 where CK fails.
//
// Supported features (matches kernel capabilities in hsa/gfx1250/fmha_fwd_bf16/):
//   - arch: gfx1250 only
//   - prec: bf16 only (kernel does not accept fp16)
//   - head_dim: 64 or 128
//   - mask: causal only (mask=1; only causal kernels are shipped in the CSV)
//   - layout: bshd ([b, s, h, d]) shape; kernel reads strides directly so the
//             underlying memory layout can differ (this host uses contiguous bshd)
//   - sink (D64 only): per-head fp32 logit in AITER post-scale domain required
//             for D64 (ENABLE_SINK=1 kernels).  D128 kernels (ENABLE_SINK=0)
//             ignore sink contents; a zero-filled buffer is still passed.
//
// Not supported: fp16, group/varlen mode, bias, dropout, sliding-window (non-causal).

#include "aiter_tensor.h" // AiterTensor, aiter_tensor_t, AiterDtype, HipDeviceGuard

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// C-ABI declaration for the ASM FWD with sink entry point.
// Defined in csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu, linked via libmha_fwd.so.
// The AITER_CTYPES_DEFINE_ENTRYPOINT_VOID macro generates an extern "C" int
// wrapper; returns 0 on success, -1 on error (error string in TLS).
// ---------------------------------------------------------------------------
extern "C" int fmha_fwd_with_sink_asm(aiter_tensor_t* q,
                                       aiter_tensor_t* k,
                                       aiter_tensor_t* v,
                                       aiter_tensor_t* out,
                                       aiter_tensor_t* lse,
                                       aiter_tensor_t* sink,
                                       float           softmax_scale,
                                       int             is_causal,
                                       int             return_lse,
                                       hipStream_t     stream);

// TLS error accessors from the same .so (AITER_CTYPES_ERROR_DEF).
extern "C" const char* aiter_get_last_error();
extern "C" void        aiter_clear_last_error();

// ---------------------------------------------------------------------------
// HIP error check helper
// ---------------------------------------------------------------------------

#ifndef HIP_CHECK
#define HIP_CHECK(expr)                                                      \
    do                                                                       \
    {                                                                        \
        hipError_t _e = (expr);                                              \
        if(_e != hipSuccess)                                                 \
        {                                                                    \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at "   \
                      << __FILE__ << ":" << __LINE__ << " (" #expr ")"      \
                      << std::endl;                                          \
            std::abort();                                                    \
        }                                                                    \
    } while(0)
#endif

// ---------------------------------------------------------------------------
// Dtype <-> float conversions
// ---------------------------------------------------------------------------

template <typename T>
inline float to_float(T x);
template <>
inline float to_float<float>(float x)
{
    return x;
}
template <>
inline float to_float<__half>(__half x)
{
    return __half2float(x);
}
template <>
inline float to_float<__hip_bfloat16>(__hip_bfloat16 x)
{
    return static_cast<float>(x);
}

template <typename T>
inline T from_float(float x);
template <>
inline float from_float<float>(float x)
{
    return x;
}
template <>
inline __half from_float<__half>(float x)
{
    return __float2half_rn(x);
}
template <>
inline __hip_bfloat16 from_float<__hip_bfloat16>(float x)
{
    return static_cast<__hip_bfloat16>(x);
}

// ---------------------------------------------------------------------------
// HostTensor<T>: flat row-major storage indexed by variadic indices
// ---------------------------------------------------------------------------

using index_t = int32_t;

template <typename T>
class HostTensor
{
  public:
    HostTensor() = default;

    explicit HostTensor(std::vector<index_t> shape) : shape_(std::move(shape))
    {
        strides_.assign(shape_.size(), 1);
        for(int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        data_.assign(numel(), T{});
    }

    template <std::size_t N>
    explicit HostTensor(std::array<index_t, N> shape)
        : HostTensor(std::vector<index_t>(shape.begin(), shape.end()))
    {
    }

    HostTensor(std::initializer_list<index_t> shape)
        : HostTensor(std::vector<index_t>(shape.begin(), shape.end()))
    {
    }

    const std::vector<index_t>& shape() const { return shape_; }

    std::size_t numel() const
    {
        std::size_t n = 1;
        for(auto s : shape_)
            n *= static_cast<std::size_t>(s);
        return n;
    }

    std::size_t bytes() const { return numel() * sizeof(T); }

    T*       data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    template <typename... Idxs>
    std::size_t offset(Idxs... idxs) const
    {
        const index_t ii[] = {static_cast<index_t>(idxs)...};
        std::size_t   off  = 0;
        for(std::size_t i = 0; i < sizeof...(idxs); ++i)
            off += static_cast<std::size_t>(ii[i]) * static_cast<std::size_t>(strides_[i]);
        return off;
    }

    template <typename... Idxs>
    T& operator()(Idxs... idxs)
    {
        return data_[offset(idxs...)];
    }
    template <typename... Idxs>
    const T& operator()(Idxs... idxs) const
    {
        return data_[offset(idxs...)];
    }

  private:
    std::vector<index_t> shape_;
    std::vector<index_t> strides_;
    std::vector<T>       data_;
};

// ---------------------------------------------------------------------------
// Fill helpers
// ---------------------------------------------------------------------------

template <typename T, typename Engine>
void fill_uniform_int(HostTensor<T>& t, float lo, float hi, Engine& eng)
{
    std::uniform_int_distribution<int> dist(static_cast<int>(lo), static_cast<int>(hi));
    for(std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = from_float<T>(static_cast<float>(dist(eng)));
}

template <typename T, typename Engine>
void fill_uniform(HostTensor<T>& t, float lo, float hi, Engine& eng)
{
    std::uniform_real_distribution<float> dist(lo, hi);
    for(std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = from_float<T>(dist(eng));
}

template <typename T>
void fill_trig(HostTensor<T>& t)
{
    for(std::size_t i = 0; i < t.numel(); ++i)
    {
        float x    = static_cast<float>(i);
        t.data()[i] = from_float<T>(0.5f * std::sin(x) + 0.5f * std::cos(0.5f * x));
    }
}

template <typename T>
void fill_constant(HostTensor<T>& t, float v)
{
    const T tv = from_float<T>(v);
    for(std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = tv;
}

// ---------------------------------------------------------------------------
// check_err: element-wise comparison with a bad-fraction budget.
// Pass criterion: (#fails / N) <= bad_fraction  (default 0.5%).
// ---------------------------------------------------------------------------

template <typename T, typename Ref>
bool check_err(const HostTensor<T>&   out,
               const HostTensor<Ref>& ref,
               const std::string&     msg,
               double                 rtol,
               double                 atol,
               double                 bad_fraction = 5e-3)
{
    if(out.numel() != ref.numel())
    {
        std::cerr << msg << " size mismatch " << out.numel() << " vs " << ref.numel()
                  << std::endl;
        return false;
    }
    std::size_t bad_cnt = 0;
    double      max_abs = 0.0;
    double      sq_err  = 0.0;
    double      sq_ref  = 0.0;
    for(std::size_t i = 0; i < out.numel(); ++i)
    {
        double a = static_cast<double>(to_float<T>(out.data()[i]));
        double b = static_cast<double>(to_float<Ref>(ref.data()[i]));
        double d = std::fabs(a - b);
        if(d > max_abs)
            max_abs = d;
        sq_err += d * d;
        sq_ref += b * b;
        if(d > atol + rtol * std::fabs(b))
            ++bad_cnt;
    }
    const double frac = static_cast<double>(bad_cnt) / static_cast<double>(out.numel());
    const double nrms = std::sqrt(sq_err / (sq_ref + 1e-30));
    const bool   pass = frac <= bad_fraction;
    if(!pass)
    {
        std::cerr << msg << " bad=" << bad_cnt << "/" << out.numel() << " ("
                  << (frac * 100.0) << "%) max_abs=" << max_abs << " nrms=" << nrms
                  << std::endl;
    }
    return pass;
}

// ---------------------------------------------------------------------------
// Tiny ArgParser
// ---------------------------------------------------------------------------

class ArgParser
{
  public:
    void insert(const std::string& key,
                const std::string& def,
                const std::string& help)
    {
        if(map_.count(key) == 0)
            map_[key] = def;
        order_.push_back({key, def, help});
    }

    bool parse(int argc, char** argv)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            if(s == "--help" || s == "-h" || s == "-?")
            {
                print_help(argv[0]);
                return false;
            }
            if(s.rfind("-", 0) != 0)
            {
                std::cerr << "unknown arg: " << s << std::endl;
                return false;
            }
            auto eq = s.find('=');
            if(eq == std::string::npos)
            {
                std::cerr << "expected -k=v, got: " << s << std::endl;
                return false;
            }
            std::string key = s.substr(1, eq - 1);
            std::string val = s.substr(eq + 1);
            if(map_.count(key) == 0)
            {
                std::cerr << "unknown option: " << key << std::endl;
                return false;
            }
            map_[key] = val;
        }
        return true;
    }

    std::string get_str(const std::string& k) const { return get(k); }
    int         get_int(const std::string& k) const { return std::stoi(get(k)); }
    float       get_float(const std::string& k) const { return std::stof(get(k)); }
    bool        get_bool(const std::string& k) const { return get_int(k) != 0; }

  private:
    std::string get(const std::string& k) const
    {
        auto it = map_.find(k);
        if(it == map_.end())
        {
            std::cerr << "missing key " << k << std::endl;
            std::abort();
        }
        return it->second;
    }

    void print_help(const char* prog) const
    {
        std::cout << "Usage: " << prog << " [-key=val ...]" << std::endl;
        for(const auto& e : order_)
        {
            std::cout << "  -" << std::get<0>(e) << " (=" << std::get<1>(e) << ") "
                      << std::get<2>(e) << std::endl;
        }
    }

    std::map<std::string, std::string>                              map_;
    std::vector<std::tuple<std::string, std::string, std::string>>  order_;
};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

std::tuple<bool, ArgParser> create_args(int argc, char** argv)
{
    ArgParser p;
    p.insert("v",      "1",     "1: run CPU validation, 0: perf only");
    p.insert("b",      "2",     "batch size");
    p.insert("h",      "64",    "num q heads");
    p.insert("h_k",    "-1",    "num k/v heads (-1 = same as h)");
    p.insert("s",      "4096",  "seqlen_q");
    p.insert("s_k",    "-1",    "seqlen_k (-1 = same as s)");
    p.insert("d",      "128",   "head dim; 64 or 128 only (kernel constraint)");
    p.insert("scale",  "0",     "softmax scale (0 = 1/sqrt(d))");
    p.insert("causal", "1",     "causal mask; only causal kernels are shipped");
    p.insert("sink",   "1.0",
             "per-head sink logit in AITER post-scale domain. "
             "D64 kernels require a valid sink tensor (ENABLE_SINK=1); "
             "D128 kernels always receive a zero buffer (kernel ignores contents). "
             "Set to 0.0 for an all-zero D64 sink (valid but no sink effect).");
    p.insert("init",   "1",     "init: 0=randint, 1=rand, 2=trig, 3=const(0.25)");
    p.insert("seed",   "11939", "random seed");
    p.insert("warmup", "10",    "warmup iterations");
    p.insert("repeat", "10",    "timed iterations");
    p.insert("timer",  "gpu",   "gpu / cpu (HIP events always used for timing)");
    p.insert("kname",  "0",     "reserved (no effect in ASM path)");
    bool ok = p.parse(argc, argv);
    return {ok, std::move(p)};
}

// ---------------------------------------------------------------------------
// CPU forward-attention reference
//
// Matches _ref_attn() in op_tests/test_fmha_fwd_with_sink_asm.py:
//
//   S[b,h,m,n] = sum_d Q[b,m,h,d] * K[b,n,hk,d]         (raw, no scale)
//   causal mask: S[b,h,m,n] = -inf  if n > m + (sk - sq) (bottom-right)
//   max_total[b,h,m] = max(S[b,h,m,:])
//   if has_sink: max_total = max(max_total, sink_raw[h])
//                where sink_raw = sink_user * sqrt(d)  (post-scale -> raw)
//   denom = sum_n exp((S - max_total) * scale)
//         + (has_sink ? exp((sink_raw - max_total) * scale) : 0)
//   P[b,h,m,n] = exp((S - max_total) * scale) / denom
//   O[b,m,h,v] = sum_n P[b,h,m,n] * V[b,n,hk,v]
//   LSE[b,h,m]  = max_total * scale + log(denom)
//
// All tensors stored bshd: HostTensor shape [b,s,h,d] or [b,h,s].
// Accumulation in fp32.
// GQA: Q head h uses K/V head (h / (nhead / nhead_k)).
// ---------------------------------------------------------------------------

template <typename BF16, typename F32>
void cpu_fwd_ref(
    const HostTensor<BF16>& q_host,    // [batch, sq, nhead,   hdim]
    const HostTensor<BF16>& k_host,    // [batch, sk, nhead_k, hdim]
    const HostTensor<BF16>& v_host,    // [batch, sk, nhead_k, hdim]
    const HostTensor<F32>&  sink_host, // [nhead]  post-scale domain
    HostTensor<BF16>&       o_ref,     // [batch, sq, nhead,   hdim]  output
    HostTensor<F32>&        lse_ref,   // [batch, nhead, sq]           output
    int   batch,
    int   nhead,
    int   nhead_k,
    int   seqlen_q,
    int   seqlen_k,
    int   hdim,
    float scale,
    bool  is_causal,
    bool  has_sink)
{
    const int ratio = nhead / nhead_k;
    // Bottom-right causal: mask n if n > m + (sk - sq).
    // sq == sk => n > m (standard causal).
    const int causal_offset = seqlen_k - seqlen_q;

    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            const int hk = h / ratio;

            // Convert sink from AITER post-scale domain to raw logit.
            const float sink_raw =
                has_sink
                    ? (to_float<F32>(sink_host(h)) *
                       std::sqrt(static_cast<float>(hdim)))
                    : 0.f;

            for(int m = 0; m < seqlen_q; ++m)
            {
                // --- Step 1: S = Q[b,m,h,:] @ K[b,:,hk,:]^T (no scale) ---
                std::vector<float> s(seqlen_k);
                for(int n = 0; n < seqlen_k; ++n)
                {
                    if(is_causal && n > m + causal_offset)
                    {
                        s[n] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float acc = 0.f;
                    for(int d = 0; d < hdim; ++d)
                        acc += to_float<BF16>(q_host(b, m, h,  d)) *
                               to_float<BF16>(k_host(b, n, hk, d));
                    s[n] = acc;
                }

                // --- Step 2: max_total (including optional sink) ---
                float max_total = s[0];
                for(int n = 1; n < seqlen_k; ++n)
                    if(s[n] > max_total)
                        max_total = s[n];
                if(has_sink && sink_raw > max_total)
                    max_total = sink_raw;

                // --- Step 3: unnormalized softmax + denominator ---
                // Reuse s[] as unnormalized P values.
                float denom = 0.f;
                for(int n = 0; n < seqlen_k; ++n)
                {
                    // Masked / -inf positions → 0 (matches kernel behavior).
                    float v = std::isfinite(s[n])
                                  ? std::exp((s[n] - max_total) * scale)
                                  : 0.f;
                    s[n] = v;
                    denom += v;
                }
                if(has_sink)
                    denom += std::exp((sink_raw - max_total) * scale);

                // Guard fully-masked rows (denom == 0).
                if(denom == 0.f)
                    denom = 1.f;

                // --- Step 4: O = P @ V ---
                for(int dv = 0; dv < hdim; ++dv)
                {
                    float acc = 0.f;
                    for(int n = 0; n < seqlen_k; ++n)
                        acc += (s[n] / denom) *
                               to_float<BF16>(v_host(b, n, hk, dv));
                    o_ref(b, m, h, dv) = from_float<BF16>(acc);
                }

                // --- Step 5: LSE = max_total * scale + log(denom) ---
                lse_ref(b, h, m) =
                    from_float<F32>(max_total * scale + std::log(denom));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// run_bench: orchestrate one benchmark run
// ---------------------------------------------------------------------------

bool run_bench(const ArgParser& arg)
{
    using BF16 = __hip_bfloat16;
    using F32  = float;

    const bool do_validation = arg.get_bool("v");
    const int  batch         = arg.get_int("b");
    int        nhead         = arg.get_int("h");
    int        nhead_k       = arg.get_int("h_k");
    if(nhead_k < 0)
        nhead_k = nhead;
    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead=" << nhead << " must be a multiple of nhead_k=" << nhead_k
                  << std::endl;
        return false;
    }
    int seqlen_q = arg.get_int("s");
    int seqlen_k = arg.get_int("s_k");
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    const int hdim = arg.get_int("d");
    if(hdim != 64 && hdim != 128)
    {
        std::cerr << "head_dim must be 64 or 128, got " << hdim << std::endl;
        return false;
    }

    float scale = arg.get_float("scale");
    if(scale == 0.f)
        scale = 1.f / std::sqrt(static_cast<float>(hdim));

    const bool  is_causal = arg.get_bool("causal");
    const float sink_val  = arg.get_float("sink");

    // D64 kernels compile ENABLE_SINK=1: must receive a non-null sink tensor.
    // D128 kernels compile ENABLE_SINK=0: sink contents ignored; zero buffer is fine.
    const bool has_sink = (hdim == 64);

    const int init_method = arg.get_int("init");
    const int seed_val    = arg.get_int("seed");
    const int warmup      = arg.get_int("warmup");
    const int repeat      = arg.get_int("repeat");

    // ---- Host tensors (contiguous bshd: [b, s, h, d]) ----
    HostTensor<BF16> q_host({batch, seqlen_q, nhead,   hdim});
    HostTensor<BF16> k_host({batch, seqlen_k, nhead_k, hdim});
    HostTensor<BF16> v_host({batch, seqlen_k, nhead_k, hdim});
    // sink: [nhead] fp32 in AITER post-scale domain
    HostTensor<F32>  sink_host({nhead});

    std::mt19937 eng(static_cast<unsigned>(seed_val));
    if(init_method == 0)
    {
        fill_uniform_int(q_host, -2.f, 2.f, eng);
        fill_uniform_int(k_host, -2.f, 2.f, eng);
        fill_uniform_int(v_host, -2.f, 2.f, eng);
    }
    else if(init_method == 1)
    {
        fill_uniform(q_host, -1.f, 1.f, eng);
        fill_uniform(k_host, -1.f, 1.f, eng);
        fill_uniform(v_host, -1.f, 1.f, eng);
    }
    else if(init_method == 2)
    {
        fill_trig(q_host);
        fill_trig(k_host);
        fill_trig(v_host);
    }
    else
    {
        fill_constant(q_host, 0.25f);
        fill_constant(k_host, 0.25f);
        fill_constant(v_host, 0.25f);
    }
    // D64: fill sink with sink_val (same value across all heads for simplicity).
    // D128: leave zeros; kernel ignores sink contents.
    fill_constant(sink_host, has_sink ? sink_val : 0.f);

    // ---- GPU tensors via AiterTensor ----
    // AiterTensor::empty() allocates contiguous row-major GPU memory and fills
    // the aiter_tensor_t descriptor (shape, strides, dtype, device_id) that
    // fmha_fwd_with_sink_asm reads directly.
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));

    AiterTensor q_dev   = AiterTensor::empty({batch, seqlen_q, nhead,   hdim},
                                             AITER_DTYPE_bf16, device_id);
    AiterTensor k_dev   = AiterTensor::empty({batch, seqlen_k, nhead_k, hdim},
                                             AITER_DTYPE_bf16, device_id);
    AiterTensor v_dev   = AiterTensor::empty({batch, seqlen_k, nhead_k, hdim},
                                             AITER_DTYPE_bf16, device_id);
    AiterTensor out_dev = AiterTensor::empty({batch, seqlen_q, nhead,   hdim},
                                             AITER_DTYPE_bf16, device_id);
    // lse layout required by kernel ABI: [batch, q_head_num, q_seq_len] fp32
    AiterTensor lse_dev  = AiterTensor::empty({batch, nhead, seqlen_q},
                                              AITER_DTYPE_fp32, device_id);
    // sink: [nhead] fp32.  D64 filled with sink_val; D128 zeros (ignored).
    AiterTensor sink_dev = AiterTensor::zeros({nhead}, AITER_DTYPE_fp32, device_id);

    // Upload inputs to device
    HIP_CHECK(hipMemcpy(q_dev.data_ptr(),    q_host.data(),
                        q_host.bytes(),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_dev.data_ptr(),    k_host.data(),
                        k_host.bytes(),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_dev.data_ptr(),    v_host.data(),
                        v_host.bytes(),    hipMemcpyHostToDevice));
    if(has_sink)
        HIP_CHECK(hipMemcpy(sink_dev.data_ptr(), sink_host.data(),
                            sink_host.bytes(), hipMemcpyHostToDevice));

    hipStream_t stream = nullptr; // default (null) stream

    // Lambda that calls the ASM kernel and aborts on error
    auto run_kernel = [&]() {
        aiter_clear_last_error();
        int ret = fmha_fwd_with_sink_asm(
            &q_dev, &k_dev, &v_dev,
            &out_dev, &lse_dev, &sink_dev,
            scale,
            is_causal ? 1 : 0,
            1 /*return_lse*/,
            stream);
        if(ret != 0)
        {
            const char* err = aiter_get_last_error();
            std::cerr << "fmha_fwd_with_sink_asm failed: "
                      << (err ? err : "(no error message)") << std::endl;
            std::abort();
        }
    };

    // ---- FLOPs / bandwidth estimate ----
    // FWD has 2 matrix multiplications per head:
    //   GEMM0: Q @ K^T  => 2 * sq * sk * hdim MACs
    //   GEMM1: P @ V    => 2 * sq * sk * hdim MACs
    std::size_t flop =
        4ULL * static_cast<std::size_t>(batch)   *
               static_cast<std::size_t>(nhead)   *
               static_cast<std::size_t>(seqlen_q) *
               static_cast<std::size_t>(seqlen_k) *
               static_cast<std::size_t>(hdim);
    if(is_causal)
        flop /= 2;

    // Bytes: Q + O + LSE per q-head; K + V per kv-head.
    std::size_t num_byte =
        static_cast<std::size_t>(batch) *
        (static_cast<std::size_t>(nhead) *
             (static_cast<std::size_t>(seqlen_q) * hdim * sizeof(BF16)   // Q
              + static_cast<std::size_t>(seqlen_q) * hdim * sizeof(BF16) // O
              + static_cast<std::size_t>(seqlen_q) * sizeof(F32))        // LSE
         + static_cast<std::size_t>(nhead_k) *
             (static_cast<std::size_t>(seqlen_k) * hdim * sizeof(BF16)   // K
              + static_cast<std::size_t>(seqlen_k) * hdim * sizeof(BF16))); // V

    // ---- Print benchmark header ----
    std::string sink_str =
        has_sink ? (", sink=" + std::to_string(sink_val)) : ", sink=none(D128)";
    std::cout << "[bf16|bshd|causal=" << is_causal << "]"
              << " b:" << batch
              << ", h:" << nhead << "/" << nhead_k
              << ", s:" << seqlen_q << "/" << seqlen_k
              << ", d:" << hdim
              << ", scale:" << scale
              << sink_str
              << std::flush;

    // ---- Warmup ----
    for(int i = 0; i < warmup; ++i)
        run_kernel();
    HIP_CHECK(hipStreamSynchronize(stream));

    // ---- Timed run (HIP events) ----
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));
    HIP_CHECK(hipEventRecord(ev_start, stream));
    for(int i = 0; i < repeat; ++i)
        run_kernel();
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    float elapsed_ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));

    const float ave_time   = elapsed_ms / static_cast<float>(repeat);
    const float tflops     = static_cast<float>(flop) / 1.E9f / ave_time;
    const float gb_per_sec = static_cast<float>(num_byte) / 1.E6f / ave_time;

    std::cout << std::fixed
              << ", " << std::setprecision(3) << ave_time   << " ms"
              << ", " << std::setprecision(2) << tflops     << " TFlops"
              << ", " << std::setprecision(2) << gb_per_sec << " GB/s";

    if(!do_validation)
    {
        std::cout << std::endl;
        return true;
    }

    // ---- CPU reference ----
    HostTensor<BF16> o_ref  ({batch, seqlen_q, nhead, hdim});
    HostTensor<F32>  lse_ref({batch, nhead,    seqlen_q});

    cpu_fwd_ref<BF16, F32>(
        q_host, k_host, v_host, sink_host,
        o_ref, lse_ref,
        batch, nhead, nhead_k, seqlen_q, seqlen_k, hdim,
        scale, is_causal, has_sink);

    // ---- Copy kernel outputs to host ----
    HostTensor<BF16> out_got({batch, seqlen_q, nhead, hdim});
    HostTensor<F32>  lse_got({batch, nhead,    seqlen_q});
    HIP_CHECK(hipMemcpy(out_got.data(), out_dev.data_ptr(),
                        out_got.bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(lse_got.data(), lse_dev.data_ptr(),
                        lse_got.bytes(), hipMemcpyDeviceToHost));

    // ---- Validate (rtol=atol=1e-2 matching Python test thresholds) ----
    const double rtol = 1e-2;
    const double atol = 1e-2;
    bool ok_out = check_err<BF16, BF16>(out_got, o_ref,   "[out]", rtol, atol);
    bool ok_lse = check_err<F32,  F32> (lse_got, lse_ref, "[lse]", rtol, atol);
    const bool pass = ok_out && ok_lse;

    std::cout << ", valid:" << (pass ? "y" : "n") << std::endl;
    return pass;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    auto [ok, arg] = create_args(argc, argv);
    if(!ok)
        return -1;
    return run_bench(arg) ? 0 : -2;
}

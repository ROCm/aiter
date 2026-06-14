// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Unified benchmark/correctness host for the gfx1250 ASM FWD sink path.
//
// Two calling modes selected at run time via -via= (default: mha_fwd):
//
//   direct   -- calls fmha_fwd_with_sink_asm() directly (bypasses mha_fwd wrapper).
//               The host pre-initialises O=0 and LSE=-inf before each call because
//               the kernel is a streaming accumulator that does not self-initialise.
//   mha_fwd  -- calls aiter::mha_fwd() (same dispatch path as TE's ck_fused_attn_fwd).
//               No host-side initialisation required: fmha_fwd_gfx1250_batched
//               handles O=0 / LSE=-inf initialisation internally.
//
// Both modes share the same CPU reference, validation thresholds, and shape args.
//
// Prerequisites:
//   python3 compile.py --api=fwd_v3 && bash build_mha.sh fwd_v3

#include "aiter_tensor.h"  // aiter_tensor_t, AITER_DTYPE_*
#include "mha_fwd.h"       // aiter::mha_fwd, mha_fwd_args
#include "ck_tile_shim.h"  // ck_tile::stream_config

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// Direct-mode kernel (csrc/py_itfs_cu/asm_fmha_fwd_with_sink.cu → libmha_fwd_asm.so)
// ---------------------------------------------------------------------------
extern "C" int         fmha_fwd_with_sink_asm(aiter_tensor_t* q,
                                               aiter_tensor_t* k,
                                               aiter_tensor_t* v,
                                               aiter_tensor_t* out,
                                               aiter_tensor_t* lse,
                                               aiter_tensor_t* sink,
                                               float           softmax_scale,
                                               int             is_causal,
                                               int             return_lse,
                                               hipStream_t     stream);
extern "C" const char* aiter_get_last_error();
extern "C" void        aiter_clear_last_error();

// ---------------------------------------------------------------------------
// HIP error helper
// ---------------------------------------------------------------------------
#ifndef HIP_CHECK
#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if(_e != hipSuccess) {                                                 \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at "     \
                      << __FILE__ << ":" << __LINE__ << " (" #expr ")\n";     \
            std::abort();                                                      \
        }                                                                      \
    } while(0)
#endif

// ---------------------------------------------------------------------------
// DeviceMem: RAII hipMalloc wrapper
// ---------------------------------------------------------------------------
class DeviceMem
{
public:
    DeviceMem() = default;
    explicit DeviceMem(std::size_t bytes) { realloc(bytes); }
    ~DeviceMem() { if(buf_) (void)hipFree(buf_); }
    DeviceMem(const DeviceMem&)            = delete;
    DeviceMem& operator=(const DeviceMem&) = delete;

    void realloc(std::size_t bytes)
    {
        if(buf_) { HIP_CHECK(hipFree(buf_)); buf_ = nullptr; }
        size_ = bytes;
        if(bytes > 0) HIP_CHECK(hipMalloc(&buf_, bytes));
    }
    void to_device(const void* src)
    {
        if(src && buf_ && size_)
            HIP_CHECK(hipMemcpy(buf_, src, size_, hipMemcpyHostToDevice));
    }
    void from_device(void* dst) const
    {
        if(dst && buf_ && size_)
            HIP_CHECK(hipMemcpy(dst, buf_, size_, hipMemcpyDeviceToHost));
    }
    void set_zero() { if(buf_ && size_) HIP_CHECK(hipMemset(buf_, 0, size_)); }

    void*       ptr()  const { return buf_; }
    std::size_t size() const { return size_; }

private:
    void*       buf_  = nullptr;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// make_tensor_desc: contiguous row-major aiter_tensor_t; strides in elements
// ---------------------------------------------------------------------------
template <typename... Dims>
inline aiter_tensor_t make_tensor_desc(void* ptr, AiterDtype dtype, int dev, Dims... dims)
{
    static_assert(sizeof...(dims) <= 8, "aiter_tensor_t supports at most 8 dimensions");
    aiter_tensor_t t{};
    t.ptr = ptr; t.dtype_ = dtype; t.device_id = dev;
    t.ndim = (int)sizeof...(dims);
    const int64_t da[] = {(int64_t)dims...};
    t.numel_ = 1;
    for(int i = 0; i < t.ndim; ++i) { t.shape[i] = da[i]; t.numel_ *= (std::size_t)da[i]; }
    if(t.ndim > 0) {
        t.strides[t.ndim - 1] = 1;
        for(int i = t.ndim - 2; i >= 0; --i)
            t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
    }
    return t;
}

// ---------------------------------------------------------------------------
// HostTensor<T>: flat row-major CPU storage
// ---------------------------------------------------------------------------
using index_t = int32_t;

template <typename T>
class HostTensor
{
public:
    explicit HostTensor(std::vector<index_t> shape) : shape_(std::move(shape))
    {
        strides_.assign(shape_.size(), 1);
        for(int i = (int)shape_.size() - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        data_.assign(numel(), T{});
    }
    HostTensor(std::initializer_list<index_t> s)
        : HostTensor(std::vector<index_t>(s.begin(), s.end())) {}

    std::size_t numel() const { std::size_t n=1; for(auto s:shape_) n*=(std::size_t)s; return n; }
    std::size_t bytes() const { return numel() * sizeof(T); }
    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }

    template <typename... Idx>
    T& operator()(Idx... idx)
    {
        const index_t ii[] = {(index_t)idx...};
        std::size_t off = 0;
        for(std::size_t i = 0; i < sizeof...(idx); ++i)
            off += (std::size_t)ii[i] * (std::size_t)strides_[i];
        return data_[off];
    }
    template <typename... Idx>
    const T& operator()(Idx... idx) const
    {
        const index_t ii[] = {(index_t)idx...};
        std::size_t off = 0;
        for(std::size_t i = 0; i < sizeof...(idx); ++i)
            off += (std::size_t)ii[i] * (std::size_t)strides_[i];
        return data_[off];
    }

private:
    std::vector<index_t> shape_, strides_;
    std::vector<T>       data_;
};

// ---------------------------------------------------------------------------
// Type helpers (BF16 only — the only precision the gfx1250 sink kernel accepts)
// ---------------------------------------------------------------------------
template <typename T> float to_float(T x);
template <> float to_float<float>(float x)                    { return x; }
template <> float to_float<__hip_bfloat16>(__hip_bfloat16 x) { return (float)x; }

template <typename T> T from_float(float x);
template <> float          from_float<float>(float x)          { return x; }
template <> __hip_bfloat16 from_float<__hip_bfloat16>(float x) { return (__hip_bfloat16)x; }

// ---------------------------------------------------------------------------
// Fill helpers
// ---------------------------------------------------------------------------
template <typename T>
void fill_uniform(HostTensor<T>& t, float lo, float hi, std::mt19937& eng)
{
    std::uniform_real_distribution<float> d(lo, hi);
    for(std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = from_float<T>(d(eng));
}

template <typename T>
void fill_constant(HostTensor<T>& t, float v)
{
    for(std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = from_float<T>(v);
}

// ---------------------------------------------------------------------------
// check_err: element-wise comparison; pass if bad fraction <= bad_fraction
// ---------------------------------------------------------------------------
template <typename T, typename R>
bool check_err(const HostTensor<T>& out, const HostTensor<R>& ref,
               const std::string& msg, double rtol, double atol,
               double bad_fraction = 5e-3)
{
    std::size_t bad = 0;
    double max_abs = 0, sq_err = 0, sq_ref = 0;
    for(std::size_t i = 0; i < out.numel(); ++i)
    {
        double a = (double)to_float<T>(out.data()[i]);
        double b = (double)to_float<R>(ref.data()[i]);
        double d = std::fabs(a - b);
        if(d > max_abs) max_abs = d;
        sq_err += d * d; sq_ref += b * b;
        if(d > atol + rtol * std::fabs(b)) ++bad;
    }
    bool pass = (double)bad / (double)out.numel() <= bad_fraction;
    if(!pass)
        std::cerr << msg << " bad=" << bad << "/" << out.numel()
                  << " (" << (100.0 * bad / out.numel()) << "%)"
                  << " max_abs=" << max_abs
                  << " nrms=" << std::sqrt(sq_err / (sq_ref + 1e-30)) << "\n";
    return pass;
}

// ---------------------------------------------------------------------------
// ArgParser
// ---------------------------------------------------------------------------
class ArgParser
{
public:
    void insert(const std::string& key, const std::string& def, const std::string& help)
    {
        if(!map_.count(key)) map_[key] = def;
        order_.push_back({key, def, help});
    }
    bool parse(int argc, char** argv)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            if(s == "--help" || s == "-h" || s == "-?") { print_help(argv[0]); return false; }
            auto eq = s.find('=');
            if(eq == std::string::npos || s.rfind("-", 0) != 0)
                { std::cerr << "expected -k=v, got: " << s << "\n"; return false; }
            std::string k = s.substr(1, eq - 1), v = s.substr(eq + 1);
            if(!map_.count(k)) { std::cerr << "unknown option: " << k << "\n"; return false; }
            map_[k] = v;
        }
        return true;
    }
    std::string get_str  (const std::string& k) const { return map_.at(k); }
    int         get_int  (const std::string& k) const { return std::stoi(map_.at(k)); }
    float       get_float(const std::string& k) const { return std::stof(map_.at(k)); }
    bool        get_bool (const std::string& k) const { return get_int(k) != 0; }

private:
    void print_help(const char* prog) const
    {
        std::cout << "Usage: " << prog << " [-key=val ...]\n";
        for(const auto& e : order_)
            std::cout << "  -" << std::get<0>(e) << " (=" << std::get<1>(e) << ") "
                      << std::get<2>(e) << "\n";
    }
    std::map<std::string, std::string>                             map_;
    std::vector<std::tuple<std::string, std::string, std::string>> order_;
};

// ---------------------------------------------------------------------------
// CPU reference (shared by both modes)
//
// Bottom-right causal attention with optional sink token:
//   scores[b,h,m,n] = scale * (Q[b,m,h,:] @ K[b,n,hk,:]^T)
//   mask: n > m + (sk - sq) → -inf   (bottom-right; reduces to top-left when sq==sk)
//   max_total = max(max(scores), sink_val)     [if has_sink]
//   denom = sum exp(scores - max_total) + exp(sink_val - max_total)  [if has_sink]
//   O[b,m,h,:] = sum_n softmax_n * V[b,n,hk,:]
//   LSE[b,h,m]  = max_total + log(denom)
//
// sink_val is in the AITER post-scale domain (scale already applied).
// TE uses sink_val = -1e30 → exp(-1e30) ≈ 0 → no sink contribution.
// ---------------------------------------------------------------------------
template <typename BF16, typename F32>
void cpu_fwd_ref(
    const HostTensor<BF16>& q,
    const HostTensor<BF16>& k,
    const HostTensor<BF16>& v,
    HostTensor<BF16>&       o,
    HostTensor<F32>&        lse,
    int batch, int nhead, int nhead_k, int sq, int sk, int hdim,
    float scale, bool is_causal, bool has_sink, float sink_val)
{
    const int ratio         = nhead / nhead_k;
    const int causal_offset = sk - sq; // bottom-right causal

    for(int b = 0; b < batch; ++b)
    for(int h = 0; h < nhead; ++h)
    {
        const int   hk          = h / ratio;
        const float sink_scaled = has_sink ? sink_val : 0.f;

        for(int m = 0; m < sq; ++m)
        {
            std::vector<float> scores(sk);
            for(int n = 0; n < sk; ++n)
            {
                if(is_causal && n > m + causal_offset)
                    { scores[n] = -std::numeric_limits<float>::infinity(); continue; }
                float acc = 0.f;
                for(int d = 0; d < hdim; ++d)
                    acc += to_float<BF16>(q(b,m,h,d)) * to_float<BF16>(k(b,n,hk,d));
                scores[n] = acc * scale;
            }

            float max_total = scores[0];
            for(int n = 1; n < sk; ++n)
                if(scores[n] > max_total) max_total = scores[n];
            if(has_sink && sink_scaled > max_total) max_total = sink_scaled;

            float denom = 0.f;
            for(int n = 0; n < sk; ++n)
            {
                float w = std::isfinite(scores[n]) ? std::exp(scores[n] - max_total) : 0.f;
                scores[n] = w; denom += w;
            }
            if(has_sink) denom += std::exp(sink_scaled - max_total);
            if(denom == 0.f) denom = 1.f;

            for(int dv = 0; dv < hdim; ++dv)
            {
                float acc = 0.f;
                for(int n = 0; n < sk; ++n)
                    acc += (scores[n] / denom) * to_float<BF16>(v(b,n,hk,dv));
                o(b,m,h,dv) = from_float<BF16>(acc);
            }
            lse(b,h,m) = from_float<F32>(max_total + std::log(denom));
        }
    }
}

// ---------------------------------------------------------------------------
// run_bench
// ---------------------------------------------------------------------------
bool run_bench(const ArgParser& arg)
{
    using BF16 = __hip_bfloat16;
    using F32  = float;

    const bool via_direct = (arg.get_str("via") == "direct");
    const bool do_val     = arg.get_bool("v");
    const int  batch      = arg.get_int("b");
    int        nhead      = arg.get_int("h");
    int        nhead_k    = arg.get_int("h_k");
    if(nhead_k < 0) nhead_k = nhead;
    int sq = arg.get_int("s");
    int sk = arg.get_int("s_k");
    if(sk < 0) sk = sq;

    const int hdim = arg.get_int("d");
    if(hdim != 64 && hdim != 128)
        { std::cerr << "head_dim must be 64 or 128\n"; return false; }
    if(nhead % nhead_k != 0)
        { std::cerr << "nhead must be a multiple of nhead_k\n"; return false; }

    float scale = arg.get_float("scale");
    if(scale == 0.f) scale = 1.f / std::sqrt((float)hdim);

    const bool  is_causal = arg.get_bool("causal");
    const float sink_val  = arg.get_float("sink");
    // D64 kernels have ENABLE_SINK=1 and require a valid sink tensor.
    // D128 kernels have ENABLE_SINK=0 and ignore the sink buffer.
    const bool  has_sink  = (hdim == 64);

    // ---- Host tensors ----
    HostTensor<BF16> q_h({batch, sq, nhead,   hdim});
    HostTensor<BF16> k_h({batch, sk, nhead_k, hdim});
    HostTensor<BF16> v_h({batch, sk, nhead_k, hdim});
    HostTensor<F32>  sink_h({nhead});

    std::mt19937 eng((unsigned)arg.get_int("seed"));
    fill_uniform(q_h, -1.f, 1.f, eng);
    fill_uniform(k_h, -1.f, 1.f, eng);
    fill_uniform(v_h, -1.f, 1.f, eng);
    fill_constant(sink_h, has_sink ? sink_val : 0.f);

    // ---- GPU buffers ----
    const std::size_t lse_elems  = (std::size_t)batch * nhead * sq;
    const std::size_t lse_bytes  = lse_elems * sizeof(F32);
    const std::size_t sink_bytes = (std::size_t)nhead * sizeof(F32);

    DeviceMem q_buf(q_h.bytes()), k_buf(k_h.bytes()), v_buf(v_h.bytes());
    DeviceMem o_buf(q_h.bytes()), lse_buf(lse_bytes),  sink_buf(sink_bytes);

    q_buf.to_device(q_h.data());
    k_buf.to_device(k_h.data());
    v_buf.to_device(v_h.data());
    if(has_sink) sink_buf.to_device(sink_h.data());

    hipStream_t stream = nullptr;

    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));

    // ---- Accumulator reset ----
    // direct:  kernel reads running O/LSE as initial state; caller must initialise.
    //          LSE=-inf is the log-sum-exp identity: exp(-inf)=0 contributes nothing.
    //          0xFF800000 = IEEE 754 -inf for float32.
    // mha_fwd: fmha_fwd_gfx1250_batched initialises internally; nothing to do here.
    auto reset_accumulators = [&]() {
        if(via_direct)
        {
            o_buf.set_zero();
            (void)hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(lse_buf.ptr()),
                               0xFF800000u, lse_elems);
        }
    };

    // ---- Strides for mha_fwd mode (in elements, bshd contiguous) ----
    const int stride_q    = nhead   * hdim;
    const int stride_k    = nhead_k * hdim;
    const int stride_v    = nhead_k * hdim;
    const int bstride_q   = sq * stride_q;
    const int bstride_k   = sk * stride_k;
    const int bstride_v   = sk * stride_v;
    const int bstride_lse = nhead * sq;
    const int mask_type   = is_causal ? 1 : 0;

    auto run_kernel = [&]()
    {
        if(via_direct)
        {
            auto q_d    = make_tensor_desc(q_buf.ptr(),    AITER_DTYPE_bf16, device_id,
                                           batch, sq, nhead,   hdim);
            auto k_d    = make_tensor_desc(k_buf.ptr(),    AITER_DTYPE_bf16, device_id,
                                           batch, sk, nhead_k, hdim);
            auto v_d    = make_tensor_desc(v_buf.ptr(),    AITER_DTYPE_bf16, device_id,
                                           batch, sk, nhead_k, hdim);
            auto o_d    = make_tensor_desc(o_buf.ptr(),    AITER_DTYPE_bf16, device_id,
                                           batch, sq, nhead,   hdim);
            auto lse_d  = make_tensor_desc(lse_buf.ptr(),  AITER_DTYPE_fp32, device_id,
                                           batch, nhead, sq);
            auto sink_d = make_tensor_desc(sink_buf.ptr(), AITER_DTYPE_fp32, device_id,
                                           nhead);
            aiter_clear_last_error();
            int ret = fmha_fwd_with_sink_asm(&q_d, &k_d, &v_d, &o_d, &lse_d, &sink_d,
                                              scale, is_causal ? 1 : 0, 1, stream);
            if(ret != 0)
            {
                const char* err = aiter_get_last_error();
                std::cerr << "fmha_fwd_with_sink_asm failed: "
                          << (err ? err : "(no error message)") << "\n";
                std::abort();
            }
        }
        else
        {
            ck_tile::stream_config sc{stream};
            aiter::mha_fwd_args a{};
            a.use_asm_v3 = true; a.v3_api_check = false; a.how_v3_bf16_cvt = 0;
            a.data_type = "bf16"; a.is_group_mode = false;
            a.bias_type = 0; a.has_lse = true; a.qscale_type = 0; a.has_sink = false;

            a.q_ptr = q_buf.ptr(); a.k_ptr = k_buf.ptr(); a.v_ptr = v_buf.ptr();
            a.bias_ptr = nullptr; a.q_descale_ptr = nullptr; a.k_descale_ptr = nullptr;
            a.v_descale_ptr = nullptr; a.rand_val_ptr = nullptr;
            a.lse_ptr = lse_buf.ptr(); a.o_ptr = o_buf.ptr();
            a.sink_ptr = has_sink ? sink_buf.ptr() : nullptr;

            a.seqstart_q_ptr = nullptr; a.seqstart_k_ptr = nullptr;
            a.seqlen_q_ptr = nullptr; a.seqlen_k_ptr = nullptr;
            a.cu_seqlen_q_ptr = nullptr; a.cu_seqlen_k_ptr = nullptr;
            a.block_scale_seqstart_q_ptr = nullptr; a.block_scale_seqstart_k_ptr = nullptr;

            a.seqlen_q = sq; a.seqlen_k = sk; a.batch = batch; a.max_seqlen_q = sq;
            a.hdim_q = hdim; a.hdim_v = hdim; a.nhead_q = nhead; a.nhead_k = nhead_k;
            a.scale_s = scale; a.logits_soft_cap = 0.f;

            a.stride_q = stride_q; a.stride_k = stride_k; a.stride_v = stride_v;
            a.stride_bias = 0; a.stride_randval = 0; a.stride_o = stride_q;
            a.nhead_stride_q = hdim; a.nhead_stride_k = hdim; a.nhead_stride_v = hdim;
            a.nhead_stride_bias = 0; a.nhead_stride_randval = 0;
            a.nhead_stride_lse = sq; a.nhead_stride_o = hdim;
            a.nhead_stride_q_descale = 0; a.nhead_stride_k_descale = 0;
            a.nhead_stride_v_descale = 0;
            a.batch_stride_q = bstride_q; a.batch_stride_k = bstride_k;
            a.batch_stride_v = bstride_v; a.batch_stride_bias = 0;
            a.batch_stride_randval = 0; a.batch_stride_lse = bstride_lse;
            a.batch_stride_o = bstride_q;
            a.batch_stride_q_descale = 0; a.batch_stride_k_descale = 0;
            a.batch_stride_v_descale = 0;

            a.window_size_left = -1; a.window_size_right = 0; a.sink_size = 0;
            a.mask_type = mask_type; a.min_seqlen_q = 0;
            a.p_drop = 0.f; a.s_randval = false;
            a.drop_seed_offset = std::pair<uint64_t, uint64_t>{0, 0};
            a.block_scale_size_q = 0; a.block_scale_size_kv = 0;

            aiter::mha_fwd(a, sc);
        }
    };

    // ---- FLOPs estimate (2 GEMMs, each sq*sk*hdim MACs per head) ----
    const std::size_t flop =
        4ULL * (std::size_t)batch * nhead * sq * sk * hdim / (is_causal ? 2 : 1);

    std::cout << "[bf16|bshd|causal=" << is_causal << "|via=" << arg.get_str("via") << "]"
              << " b=" << batch << " h=" << nhead << "/" << nhead_k
              << " s=" << sq << "/" << sk << " d=" << hdim << " scale=" << scale
              << (has_sink ? (" sink=" + std::to_string(sink_val)) : " sink=none(D128)")
              << std::flush;

    // ---- Warmup ----
    for(int i = 0; i < arg.get_int("warmup"); ++i)
        { reset_accumulators(); run_kernel(); }
    HIP_CHECK(hipStreamSynchronize(stream));

    // ---- Timed runs ----
    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0, stream));
    for(int i = 0; i < arg.get_int("repeat"); ++i)
        { reset_accumulators(); run_kernel(); }
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    const float ave_ms = ms / (float)arg.get_int("repeat");
    std::cout << std::fixed
              << ", " << std::setprecision(3) << ave_ms << " ms"
              << ", " << std::setprecision(2) << (float)flop / 1e9f / ave_ms << " TFlops";

    if(!do_val) { std::cout << "\n"; return true; }

    // ---- Validation: single call from a clean initial state ----
    reset_accumulators();
    run_kernel();
    HIP_CHECK(hipStreamSynchronize(stream));

    // ---- CPU reference ----
    HostTensor<BF16> o_ref  ({batch, sq, nhead, hdim});
    HostTensor<F32>  lse_ref({batch, nhead, sq});
    cpu_fwd_ref<BF16, F32>(q_h, k_h, v_h, o_ref, lse_ref,
                            batch, nhead, nhead_k, sq, sk, hdim,
                            scale, is_causal, has_sink, sink_val);

    HostTensor<BF16> o_got  ({batch, sq, nhead, hdim});
    HostTensor<F32>  lse_got({batch, nhead, sq});
    o_buf.from_device(o_got.data());
    lse_buf.from_device(lse_got.data());

    bool pass = check_err<BF16,BF16>(o_got,   o_ref,   "[out]", 1e-2, 1e-2)
             && check_err<F32, F32> (lse_got, lse_ref, "[lse]", 1e-2, 1e-2);
    std::cout << ", valid:" << (pass ? "y" : "n") << "\n";
    return pass;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Force HIP context creation before any hipMalloc (no PyTorch pre-init here).
    HIP_CHECK(hipInit(0));

    ArgParser p;
    p.insert("via",    "mha_fwd",
             "calling mode: direct (fmha_fwd_with_sink_asm) or mha_fwd (aiter::mha_fwd)");
    p.insert("v",      "1",     "1: CPU validation, 0: perf only");
    p.insert("b",      "2",     "batch size");
    p.insert("h",      "8",     "num q heads");
    p.insert("h_k",    "-1",    "num k/v heads (-1 = same as h)");
    p.insert("s",      "1024",  "seqlen_q");
    p.insert("s_k",    "-1",    "seqlen_k (-1 = same as s)");
    p.insert("d",      "64",    "head dim (64 or 128 only)");
    p.insert("scale",  "0",     "softmax scale (0 = 1/sqrt(d))");
    p.insert("causal", "1",     "1: causal mask, 0: no mask");
    p.insert("sink",   "-1e30", "per-head sink logit (AITER post-scale domain); "
                                "D64: -1e30 → exp(-1e30)≈0 (no effect); "
                                "D128: ignored (kernel has ENABLE_SINK=0)");
    p.insert("seed",   "11939", "random seed");
    p.insert("warmup", "2",     "warmup iterations");
    p.insert("repeat", "3",     "timed iterations");

    if(!p.parse(argc, argv)) return -1;

    const std::string via = p.get_str("via");
    if(via != "direct" && via != "mha_fwd")
    {
        std::cerr << "unknown -via=" << via << "; expected 'direct' or 'mha_fwd'\n";
        return -1;
    }

    return run_bench(p) ? 0 : -2;
}

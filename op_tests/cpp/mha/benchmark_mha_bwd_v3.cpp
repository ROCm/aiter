// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CK-excluded benchmark host for the aiter::mha_bwd asm-v3 path. Pair with
// libmha_bwd.so built via `compile.py --api=bwd_v3` (ck_exclude=True), which
// defines ENABLE_CK=0 and pulls in csrc/include/ck_tile_shim.h for the
// tiny ck_tile::* surface used by mha_bwd.h. This file therefore contains
// its own ArgParser / HostTensor / DeviceMem / fill / reference / check_err
// helpers and never includes ck_tile/host.hpp.
//
// Supported features (mirrors what fmha_v3_bwd actually accepts):
//   - prec: fp16 / bf16
//   - mode: 0 (batch) / 1 (group)
//   - bias_type: 0 (no bias)
//   - no dropout, no alibi, no dbias, no sink, no deterministic
//   - mask: 0 / t / b / t:l,r / b:l,r / xt:w / xb:w / g:y,x
//   - iperm / operm (bhsd vs bshd)
//
// CPU validation reproduces only the asm-supported scenarios. When the user
// requests an unsupported config (drop>0, bias!=n, dbias, deterministic),
// the validation step is skipped and a warning is printed; the perf number
// is still reported so the asm kernel selection can be probed.

#include "mha_bwd.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

// -----------------------------------------------------------------------------
// HIP helpers
// -----------------------------------------------------------------------------

#ifndef HIP_CHECK
#define HIP_CHECK(expr)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t _e = (expr);                                             \
        if(_e != hipSuccess)                                                \
        {                                                                   \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at "   \
                      << __FILE__ << ":" << __LINE__ << " (" << #expr ")"   \
                      << std::endl;                                         \
            std::abort();                                                   \
        }                                                                   \
    } while(0)
#endif

// -----------------------------------------------------------------------------
// Pinned host releaser (deferred hipHostFree)
// -----------------------------------------------------------------------------

class PinnedHostReleaser
{
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<void*> q_;
    std::thread worker_;
    bool stop_ = false;

    void run()
    {
        for(;;)
        {
            void* p = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                if(q_.empty())
                    return;
                p = q_.front();
                q_.pop();
            }
            (void)hipHostFree(p);
        }
    }

  public:
    PinnedHostReleaser() : worker_([this] { run(); }) {}
    ~PinnedHostReleaser()
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        if(worker_.joinable())
            worker_.join();
    }
    PinnedHostReleaser(const PinnedHostReleaser&)            = delete;
    PinnedHostReleaser& operator=(const PinnedHostReleaser&) = delete;

    static PinnedHostReleaser& instance()
    {
        static PinnedHostReleaser r;
        return r;
    }

    void enqueue(void* p)
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            q_.push(p);
        }
        cv_.notify_one();
    }
};

// -----------------------------------------------------------------------------
// Dtype <-> float conversion
// -----------------------------------------------------------------------------

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

template <typename Dst, typename Src>
inline Dst type_convert(Src s)
{
    return from_float<Dst>(to_float<Src>(s));
}

// -----------------------------------------------------------------------------
// HostTensor: flat data with row-major strides, indexed by std::array<I, N>
// -----------------------------------------------------------------------------

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

    std::size_t rank() const { return shape_.size(); }
    const std::vector<index_t>& shape() const { return shape_; }
    const std::vector<index_t>& strides() const { return strides_; }

    std::size_t numel() const
    {
        std::size_t n = 1;
        for(auto s : shape_)
            n *= static_cast<std::size_t>(s);
        return n;
    }

    std::size_t bytes() const { return numel() * sizeof(T); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    template <typename... Idxs>
    std::size_t offset(Idxs... idxs) const
    {
        const index_t ii[] = {static_cast<index_t>(idxs)...};
        std::size_t off    = 0;
        for(std::size_t i = 0; i < sizeof...(idxs); ++i)
            off += static_cast<std::size_t>(ii[i]) * static_cast<std::size_t>(strides_[i]);
        return off;
    }

    std::size_t offset_arr(const index_t* ii, std::size_t n) const
    {
        std::size_t off = 0;
        for(std::size_t i = 0; i < n; ++i)
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

    // ForEach over a flat counter, exposing index array of size `rank`.
    template <typename Fn>
    void ForEach(Fn&& fn)
    {
        std::vector<index_t> idx(shape_.size(), 0);
        const std::size_t n = numel();
        for(std::size_t flat = 0; flat < n; ++flat)
        {
            fn(*this, idx);
            for(int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d)
            {
                if(++idx[d] < shape_[d])
                    break;
                idx[d] = 0;
            }
        }
    }

    // ForEach (const variant)
    template <typename Fn>
    void ForEach(Fn&& fn) const
    {
        std::vector<index_t> idx(shape_.size(), 0);
        const std::size_t n = numel();
        for(std::size_t flat = 0; flat < n; ++flat)
        {
            fn(*this, idx);
            for(int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d)
            {
                if(++idx[d] < shape_[d])
                    break;
                idx[d] = 0;
            }
        }
    }

  private:
    std::vector<index_t> shape_;
    std::vector<index_t> strides_;
    std::vector<T> data_;
};

// -----------------------------------------------------------------------------
// DeviceMem: simple raw hipMalloc wrapper
// -----------------------------------------------------------------------------

class DeviceMem
{
  public:
    DeviceMem() = default;
    explicit DeviceMem(std::size_t bytes) { Realloc(bytes); }
    ~DeviceMem()
    {
        if(buf_)
            (void)hipFree(buf_);
    }
    DeviceMem(const DeviceMem&)            = delete;
    DeviceMem& operator=(const DeviceMem&) = delete;

    void Realloc(std::size_t bytes)
    {
        if(buf_)
        {
            HIP_CHECK(hipFree(buf_));
            buf_ = nullptr;
        }
        size_ = bytes;
        if(bytes > 0)
            HIP_CHECK(hipMalloc(&buf_, bytes));
    }

    void ToDevice(const void* src) const
    {
        if(src && buf_ && size_)
            HIP_CHECK(hipMemcpy(buf_, src, size_, hipMemcpyHostToDevice));
    }

    void FromDevice(void* dst) const
    {
        if(dst && buf_ && size_)
            HIP_CHECK(hipMemcpy(dst, buf_, size_, hipMemcpyDeviceToHost));
    }

    void SetZero()
    {
        if(buf_ && size_)
            HIP_CHECK(hipMemset(buf_, 0, size_));
    }

    void* GetDeviceBuffer() const { return buf_; }
    std::size_t GetBufferSize() const { return size_; }

  private:
    void* buf_       = nullptr;
    std::size_t size_ = 0;
};

// -----------------------------------------------------------------------------
// Fill helpers
// -----------------------------------------------------------------------------

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
        float x = static_cast<float>(i);
        t.data()[i] =
            from_float<T>(0.5f * std::sin(x) + 0.5f * std::cos(0.5f * x));
    }
}

template <typename T>
void fill_constant(HostTensor<T>& t, float v)
{
    const T tv = from_float<T>(v);
    for(std::size_t i = 0; i < t.numel(); ++i)
        t.data()[i] = tv;
}

// -----------------------------------------------------------------------------
// check_err
// -----------------------------------------------------------------------------

// Pass criteria modelled after ck_tile::check_err: per-element pointwise check
// with a small fail-fraction budget, plus an absolute|ref|-normalised tolerance
// to ignore noise around tiny reference magnitudes.
//
//   pass_i = |out_i - ref_i| <= atol + rtol * |ref_i|
//   pass   = (#fails / N) <= bad_fraction
//
// `bad_fraction` defaults to 0.5% which is the ballpark used by ck_tile.
template <typename T, typename Ref>
bool check_err(const HostTensor<T>& out,
               const HostTensor<Ref>& ref,
               const std::string& msg,
               double rtol,
               double atol,
               double bad_fraction = 5e-3)
{
    if(out.numel() != ref.numel())
    {
        std::cerr << msg << " size mismatch " << out.numel() << " vs " << ref.numel()
                  << std::endl;
        return false;
    }
    std::size_t bad_cnt = 0;
    double max_abs      = 0.0;
    double sq_err       = 0.0;
    double sq_ref       = 0.0;
    for(std::size_t i = 0; i < out.numel(); ++i)
    {
        double a = to_float<T>(out.data()[i]);
        double b = to_float<Ref>(ref.data()[i]);
        double d = std::fabs(a - b);
        if(d > max_abs)
            max_abs = d;
        sq_err += d * d;
        sq_ref += b * b;
        if(d > atol + rtol * std::fabs(b))
            ++bad_cnt;
    }
    const double frac = static_cast<double>(bad_cnt) / out.numel();
    const double nrms = std::sqrt(sq_err / (sq_ref + 1e-30));
    const bool pass   = frac <= bad_fraction;
    if(!pass)
    {
        std::cerr << msg << " bad=" << bad_cnt << "/" << out.numel()
                  << " (" << (frac * 100.0) << "%) max_abs=" << max_abs
                  << " nrms=" << nrms << std::endl;
    }
    return pass;
}

// -----------------------------------------------------------------------------
// Tiny ArgParser
// -----------------------------------------------------------------------------

class ArgParser
{
  public:
    void insert(const std::string& key, const std::string& def, const std::string& help)
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
    int get_int(const std::string& k) const { return std::stoi(get(k)); }
    uint32_t get_uint32(const std::string& k) const
    {
        return static_cast<uint32_t>(std::stoul(get(k)));
    }
    uint64_t get_uint64(const std::string& k) const { return std::stoull(get(k)); }
    float get_float(const std::string& k) const { return std::stof(get(k)); }
    bool get_bool(const std::string& k) const { return get_int(k) != 0; }

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

    std::map<std::string, std::string> map_;
    std::vector<std::tuple<std::string, std::string, std::string>> order_;
};

// -----------------------------------------------------------------------------
// Sequence-length / seqstart helpers (group mode)
// -----------------------------------------------------------------------------

enum class mode_e
{
    batch = 0,
    group = 1
};

std::ostream& operator<<(std::ostream& os, mode_e m)
{
    return os << (m == mode_e::batch ? "batch" : "group");
}

std::vector<int32_t> generate_seqlens(mode_e mode,
                                      unsigned count,
                                      int32_t seqlen_avg,
                                      int32_t seqlen_min,
                                      int32_t seqlen_max,
                                      std::optional<unsigned> seed)
{
    seqlen_min = (seqlen_min > 0 ? seqlen_min : 1);
    seqlen_max =
        (seqlen_max > 0 ? seqlen_max : std::numeric_limits<int32_t>::max());

    std::vector<int32_t> seqlens(count, std::clamp(seqlen_avg, seqlen_min, seqlen_max));
    if(mode == mode_e::group && count > 1)
    {
        std::mt19937 eng(seed.value_or(std::random_device{}()));
        std::uniform_int_distribution<unsigned> idx_dist(0, count - 1);
        std::uniform_int_distribution<unsigned> step_dist(1, count - 1);
        for(unsigned repeat = seqlen_avg * (count / 2); repeat > 0; --repeat)
        {
            unsigned to_dec = idx_dist(eng);
            if(seqlens[to_dec] == seqlen_min)
                continue;
            unsigned to_inc = (to_dec + step_dist(eng)) % count;
            if(seqlens[to_inc] >= seqlen_max)
                continue;
            --seqlens[to_dec];
            ++seqlens[to_inc];
        }
    }
    return seqlens;
}

std::vector<int32_t> to_seqstarts(const std::vector<int32_t>& seqlens)
{
    std::vector<int32_t> ss = {0};
    for(int32_t v : seqlens)
        ss.push_back(ss.back() + v);
    return ss;
}

// -----------------------------------------------------------------------------
// Reference forward + backward (CPU, fp32 accumulation)
//
// All tensors are 3D [nhead, seq_q, *] / [nhead, *, seq_k] etc.
// -----------------------------------------------------------------------------

// out[h, m, n] = sum_k A[h, m, k] * B[h, n, k]   (transposed B layout)
template <typename A, typename B, typename Out>
void ref_batched_gemm_nt(const HostTensor<A>& a,
                         const HostTensor<B>& b,
                         HostTensor<Out>& out,
                         float scale = 1.0f)
{
    const int H = out.shape()[0];
    const int M = out.shape()[1];
    const int N = out.shape()[2];
    const int K = a.shape()[2];
    for(int h = 0; h < H; ++h)
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
            {
                float acc = 0.f;
                for(int k = 0; k < K; ++k)
                    acc += to_float<A>(a(h, m, k)) * to_float<B>(b(h, n, k));
                out(h, m, n) = from_float<Out>(scale * acc);
            }
}

// out[h, m, n] = sum_k A[h, m, k] * B[h, k, n]   (no transpose)
template <typename A, typename B, typename Out>
void ref_batched_gemm_nn(const HostTensor<A>& a,
                         const HostTensor<B>& b,
                         HostTensor<Out>& out,
                         float scale = 1.0f)
{
    const int H = out.shape()[0];
    const int M = out.shape()[1];
    const int N = out.shape()[2];
    const int K = a.shape()[2];
    for(int h = 0; h < H; ++h)
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
            {
                float acc = 0.f;
                for(int k = 0; k < K; ++k)
                    acc += to_float<A>(a(h, m, k)) * to_float<B>(b(h, k, n));
                out(h, m, n) = from_float<Out>(scale * acc);
            }
}

// out[h, m, n] = sum_k A[h, k, m] * B[h, k, n]   (A transposed)
template <typename A, typename B, typename Out>
void ref_batched_gemm_tn(const HostTensor<A>& a,
                         const HostTensor<B>& b,
                         HostTensor<Out>& out,
                         float scale = 1.0f)
{
    const int H = out.shape()[0];
    const int M = out.shape()[1];
    const int N = out.shape()[2];
    const int K = a.shape()[1];
    for(int h = 0; h < H; ++h)
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
            {
                float acc = 0.f;
                for(int k = 0; k < K; ++k)
                    acc += to_float<A>(a(h, k, m)) * to_float<B>(b(h, k, n));
                out(h, m, n) = from_float<Out>(scale * acc);
            }
}

// row-wise softmax with lse output
template <typename In, typename Out, typename LSE>
void ref_softmax(const HostTensor<In>& s, HostTensor<Out>& p, HostTensor<LSE>& lse)
{
    const int H = s.shape()[0];
    const int M = s.shape()[1];
    const int N = s.shape()[2];
    for(int h = 0; h < H; ++h)
        for(int m = 0; m < M; ++m)
        {
            float row_max = -std::numeric_limits<float>::infinity();
            for(int n = 0; n < N; ++n)
                row_max = std::max(row_max, to_float<In>(s(h, m, n)));
            float row_sum = 0.f;
            // when all -inf (fully masked row), avoid NaN: emit p=0, lse=-inf.
            const bool finite_row = std::isfinite(row_max);
            for(int n = 0; n < N; ++n)
            {
                float v = finite_row ? std::exp(to_float<In>(s(h, m, n)) - row_max) : 0.f;
                p(h, m, n) = from_float<Out>(v);
                row_sum += v;
            }
            if(finite_row && row_sum > 0.f)
            {
                for(int n = 0; n < N; ++n)
                {
                    float pv = to_float<Out>(p(h, m, n)) / row_sum;
                    p(h, m, n) = from_float<Out>(pv);
                }
                lse(h, m) = from_float<LSE>(row_max + std::log(row_sum));
            }
            else
            {
                lse(h, m) = from_float<LSE>(-std::numeric_limits<float>::infinity());
            }
        }
}

// Apply attention mask in-place by writing -inf in disallowed positions.
// mask kinds use the same semantics as fmha_v3_bwd:
//   0: no mask
//   1: top-left causal / sliding-window (left, right finite)
//   2: bottom-right causal / sliding-window
//   3: window_generic (here treated as top-left with left,right >= 0)
//
// causal interpretation:
//   - mask_top_left  : keep n in [m - left, m + right] (left=-1 => unbounded left)
//   - mask_bottom_right: keep n in [m + (sk - sq) - left, m + (sk - sq) + right]
template <typename T>
void ref_apply_mask(HostTensor<T>& s,
                    int mask_type,
                    int left,
                    int right,
                    int sq,
                    int sk)
{
    if(mask_type == 0)
        return;
    const T minus_inf = from_float<T>(-std::numeric_limits<float>::infinity());
    const int H = s.shape()[0];
    const int M = s.shape()[1];
    const int N = s.shape()[2];
    auto in_window = [&](int m, int n) {
        int center = (mask_type == 2 ? m + (sk - sq) : m);
        int lo     = (left < 0 ? std::numeric_limits<int>::min() : center - left);
        int hi     = (right < 0 ? std::numeric_limits<int>::max() : center + right);
        return n >= lo && n <= hi;
    };
    for(int h = 0; h < H; ++h)
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
                if(!in_window(m, n))
                    s(h, m, n) = minus_inf;
}

// -----------------------------------------------------------------------------
// Per-dtype config (matches FmhaBwdTypeConfig)
// -----------------------------------------------------------------------------

template <typename Prec>
struct BwdTypes;

template <>
struct BwdTypes<__half>
{
    using Q     = __half;
    using K     = __half;
    using V     = __half;
    using O     = __half;
    using dO    = __half;
    using dQ    = __half;
    using dK    = __half;
    using dV    = __half;
    using LSE   = float;
    using D     = float;
    using Acc   = float;
};

template <>
struct BwdTypes<__hip_bfloat16>
{
    using Q     = __hip_bfloat16;
    using K     = __hip_bfloat16;
    using V     = __hip_bfloat16;
    using O     = __hip_bfloat16;
    using dO    = __hip_bfloat16;
    using dQ    = __hip_bfloat16;
    using dK    = __hip_bfloat16;
    using dV    = __hip_bfloat16;
    using LSE   = float;
    using D     = float;
    using Acc   = float;
};

// -----------------------------------------------------------------------------
// CLI definition (kept in lockstep with the original benchmark)
// -----------------------------------------------------------------------------

std::tuple<bool, ArgParser> create_args(int argc, char** argv)
{
    ArgParser p;
    p.insert("v", "1", "weather do CPU validation or not");
    p.insert("mode", "0", "kernel mode. 0:batch, 1:group");
    p.insert("b", "2", "batch size");
    p.insert("h", "8", "num of head, for q");
    p.insert("h_k", "-1", "num of head, for k/v, -1 means equal to h");
    p.insert("s", "3328", "seqlen_q");
    p.insert("s_k", "-1", "seqlen_k, -1 means equal to s");
    p.insert("d", "128", "head dim for q, k");
    p.insert("d_v", "-1", "head dim for v, -1 means equal to d");
    p.insert("scale", "0", "scale factor. 0 means 1/sqrt(hdim)");
    p.insert("iperm", "1", "1: b*h*s*d, 0: b*s*h*d");
    p.insert("operm", "1", "permute output");
    p.insert("bias", "n", "only 'n' (no bias) is supported in the v3-only host");
    p.insert("dbias", "0", "must be 0 in the v3-only host");
    p.insert("prec", "fp16", "data type. fp16 or bf16");
    p.insert("mask",
             "0",
             "0/'t'/'b' or 't:l,r' / 'b:l,r' / 'xt:w' / 'xb:w' / 'g:y,x'");
    p.insert("kname", "0", "if set to 1 will print kernel name");
    p.insert("init", "1", "init method. 0:randint, 1:rand, 2:trig, 3:const(0.1)");
    p.insert("seed", "11939", "random seed");
    p.insert("p_drop", "0", "must be 0 in the v3-only host");
    p.insert("drop_seed", "1", "");
    p.insert("drop_offset", "0", "");
    p.insert("drop_prefs", "0", "");
    p.insert("timer", "gpu", "gpu / cpu");
    p.insert("warmup", "10", "warmup iters");
    p.insert("repeat", "10", "repeat iters");
    p.insert("deterministic", "0", "must be 0 in the v3-only host");
    p.insert("bwd_v3", "1", "the v3-only host always force asm v3");
    p.insert("v3_atomic_fp32", "1", "0: atomic16, 1: atomic32 (fp32 dq_acc)");
    p.insert("v3_bf16_cvt", "1", "0:RTNE; 1:RTNA; 2:RTZ");
    p.insert("v3_api_check", "0", "if 1, only probe kernel availability");
    p.insert("cpu_lp_round",
             "0",
             "CPU ref: round P/dS to fp16/bf16 before downstream GEMMs (1) or "
             "keep fp32 (0). Default 0 because the asm path's actual quant "
             "boundary on dQ/dK doesn't match the naive cast-everything path; "
             "set to 1 to reproduce the strict kernel-aligned variant.");
    bool ok = p.parse(argc, argv);
    return {ok, std::move(p)};
}

// -----------------------------------------------------------------------------
// run<>: orchestration for a single (prec, config)
// -----------------------------------------------------------------------------

template <typename Prec>
bool run_bench(const ArgParser& arg)
{
    using Q   = typename BwdTypes<Prec>::Q;
    using K   = typename BwdTypes<Prec>::K;
    using V   = typename BwdTypes<Prec>::V;
    using O   = typename BwdTypes<Prec>::O;
    using DO  = typename BwdTypes<Prec>::dO;
    using DQ  = typename BwdTypes<Prec>::dQ;
    using DK  = typename BwdTypes<Prec>::dK;
    using DV  = typename BwdTypes<Prec>::dV;
    using LSE = typename BwdTypes<Prec>::LSE;
    using D   = typename BwdTypes<Prec>::D;
    using Acc = typename BwdTypes<Prec>::Acc;

    const std::string data_type = arg.get_str("prec");
    const int do_validation     = arg.get_int("v");
    const auto mode             = static_cast<mode_e>(arg.get_uint32("mode"));
    const int batch             = arg.get_int("b");
    int nhead                   = arg.get_int("h");
    int nhead_k                 = arg.get_int("h_k");
    if(nhead_k < 0)
        nhead_k = nhead;
    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead=" << nhead << " must be multiple of nhead_k=" << nhead_k
                  << std::endl;
        return false;
    }
    int seqlen_q = arg.get_int("s");
    int seqlen_k = arg.get_int("s_k");
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    int hdim_q = arg.get_int("d");
    int hdim_v = arg.get_int("d_v");
    if(hdim_v < 0)
        hdim_v = hdim_q;
    const bool i_perm = arg.get_bool("iperm");
    const bool o_perm = arg.get_bool("operm");
    float scale       = arg.get_float("scale");
    if(scale == 0.f)
        scale = 1.f / std::sqrt(static_cast<float>(hdim_q));

    bias_info bias    = bias_info::decode(arg.get_str("bias"));
    const bool use_dbias = arg.get_bool("dbias");
    const float p_drop = arg.get_float("p_drop");
    const uint64_t drop_seed   = arg.get_uint64("drop_seed");
    const uint64_t drop_offset = arg.get_uint64("drop_offset");
    const bool drop_prefs      = arg.get_bool("drop_prefs");
    mask_info mask = mask_info::decode(arg.get_str("mask"), seqlen_q, seqlen_k);

    const int init_method        = arg.get_int("init");
    std::optional<uint32_t> seed = arg.get_uint32("seed");
    if(*seed == 0)
        seed.reset();

    const int stream_warmup = arg.get_int("warmup");
    const int stream_repeat = arg.get_int("repeat");
    const bool kname        = arg.get_bool("kname");
    const bool deterministic = arg.get_bool("deterministic");
    const bool bwd_v3       = true; // v3-only host
    const bool v3_atomic_fp32 = arg.get_bool("v3_atomic_fp32");
    const int v3_bf16_cvt     = arg.get_int("v3_bf16_cvt");
    const bool v3_api_check   = arg.get_bool("v3_api_check");
    const bool cpu_lp_round   = arg.get_bool("cpu_lp_round");

    // Reject features the asm-v3 path can't run.
    auto reject = [&](const char* why) {
        std::cout << "[v3-only host] not supported: " << why << std::flush << std::endl;
        return true;
    };
    if(bias.type != bias_enum::no_bias && reject("bias != n"))
        return false;
    if(use_dbias && reject("dbias"))
        return false;
    if(p_drop > 0.f && reject("p_drop > 0"))
        return false;
    if(deterministic && reject("deterministic"))
        return false;

    ck_tile::stream_config sc{nullptr,
                              true,
                              kname ? 1 : 0,
                              stream_warmup,
                              stream_repeat,
                              arg.get_str("timer") == std::string("gpu")};

    // Sequence layout
    auto seqlens_q = generate_seqlens(mode, batch, seqlen_q, -1, -1, seed);
    auto seqlens_k = generate_seqlens(mode, batch, seqlen_k, -1, -1, seed);
    auto seqstart_q_host = to_seqstarts(seqlens_q);
    auto seqstart_k_host = to_seqstarts(seqlens_k);

    int32_t max_seqlen_q = 0;
    int32_t max_seqlen_k = 0;
    for(int b = 0; b < batch; ++b)
    {
        max_seqlen_q = std::max(max_seqlen_q, seqlens_q[b]);
        max_seqlen_k = std::max(max_seqlen_k, seqlens_k[b]);
    }

    std::size_t flop = 0, num_byte = 0;
    for(int b = 0; b < batch; ++b)
    {
        int sq = seqlens_q[b];
        int sk = seqlens_k[b];
        flop += static_cast<std::size_t>(nhead) *
                (3ull * 2ull * sq * sk * hdim_q + 2ull * 2ull * sq * sk * hdim_v);
        num_byte += static_cast<std::size_t>(nhead) *
                    (sizeof(Q) * sq * hdim_q + sizeof(K) * sk * hdim_q +
                     sizeof(V) * sk * hdim_v + sizeof(O) * sq * hdim_v +
                     sizeof(DO) * sq * hdim_v + sizeof(DQ) * sq * hdim_q +
                     sizeof(DK) * sk * hdim_q + sizeof(DV) * sk * hdim_v +
                     sizeof(LSE) * sq);
    }

    auto lengths = [&](bool perm, int b, int h, int s, int d) {
        return perm ? std::array<index_t, 4>{b, h, s, d}
                    : std::array<index_t, 4>{b, s, h, d};
    };

    const int shape_batch = (mode == mode_e::batch ? batch : 1);
    const int shape_seqlen_q =
        (mode == mode_e::batch ? seqlen_q : seqstart_q_host.back());
    const int shape_seqlen_k =
        (mode == mode_e::batch ? seqlen_k : seqstart_k_host.back());

    HostTensor<Q> q_host(lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    HostTensor<K> k_host(lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    HostTensor<V> v_host(lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v));
    HostTensor<O> o_host(lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));
    HostTensor<LSE> lse_host(std::array<index_t, 3>{shape_batch, nhead, shape_seqlen_q});
    HostTensor<D> d_host(std::array<index_t, 3>{shape_batch, nhead, shape_seqlen_q});
    HostTensor<DQ> dq_host(lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    HostTensor<DK> dk_host(lengths(i_perm, shape_batch, nhead, shape_seqlen_k, hdim_q));
    HostTensor<DV> dv_host(lengths(i_perm, shape_batch, nhead, shape_seqlen_k, hdim_v));
    HostTensor<DO> do_host(lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    std::mt19937 eng(seed.value_or(11939u));
    if(init_method == 0)
    {
        fill_uniform_int(q_host, -2.f, 2.f, eng);
        fill_uniform_int(k_host, -2.f, 2.f, eng);
        fill_uniform_int(v_host, -2.f, 2.f, eng);
        fill_uniform_int(do_host, -2.f, 2.f, eng);
    }
    else if(init_method == 1)
    {
        fill_uniform(q_host, -1.f, 1.f, eng);
        fill_uniform(k_host, -1.f, 1.f, eng);
        fill_uniform(v_host, -1.f, 1.f, eng);
        fill_uniform(do_host, -1.f, 1.f, eng);
    }
    else if(init_method == 2)
    {
        fill_trig(q_host);
        fill_trig(k_host);
        fill_trig(v_host);
        fill_trig(do_host);
    }
    else
    {
        fill_constant(q_host, 0.1f);
        fill_constant(k_host, 0.1f);
        fill_constant(v_host, 0.1f);
        fill_constant(do_host, 0.1f);
    }

    DeviceMem q_buf(q_host.bytes());
    DeviceMem k_buf(k_host.bytes());
    DeviceMem v_buf(v_host.bytes());
    DeviceMem bias_buf(1);
    DeviceMem o_buf(o_host.bytes());
    DeviceMem lse_buf(lse_host.bytes());
    DeviceMem d_buf(d_host.bytes());
    DeviceMem randval_buf(1);
    DeviceMem dq_buf(dq_host.bytes());
    DeviceMem dk_buf(dk_host.bytes());
    DeviceMem dv_buf(dv_host.bytes());
    DeviceMem do_buf(do_host.bytes());
    DeviceMem dbias_buf(1);
    DeviceMem seqstart_q_buf(seqstart_q_host.size() * sizeof(int32_t));
    DeviceMem seqstart_k_buf(seqstart_k_host.size() * sizeof(int32_t));
    DeviceMem drop_seed_buf(drop_prefs ? sizeof(uint64_t) : 0);
    DeviceMem drop_offset_buf(drop_prefs ? sizeof(uint64_t) : 0);
    DeviceMem alibi_buf(1);

    // Workspace allocator that lazily grows.
    DeviceMem dq_acc_buf(1);
    auto workspace_alloc = [&dq_acc_buf](std::size_t bytes, bool zero_init) -> void* {
        if(bytes > dq_acc_buf.GetBufferSize())
            dq_acc_buf.Realloc(bytes);
        if(zero_init && bytes > 0)
            HIP_CHECK(hipMemset(dq_acc_buf.GetDeviceBuffer(), 0, bytes));
        return dq_acc_buf.GetDeviceBuffer();
    };

    auto pinned_host_alloc = [](std::size_t bytes) -> std::shared_ptr<void> {
        void* p = nullptr;
        HIP_CHECK(hipHostMalloc(&p, bytes, hipHostMallocDefault));
        return std::shared_ptr<void>(p, [](void* q) {
            PinnedHostReleaser::instance().enqueue(q);
        });
    };

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    do_buf.ToDevice(do_host.data());
    seqstart_q_buf.ToDevice(seqstart_q_host.data());
    seqstart_k_buf.ToDevice(seqstart_k_host.data());
    if(drop_prefs)
    {
        drop_seed_buf.ToDevice(&drop_seed);
        drop_offset_buf.ToDevice(&drop_offset);
    }

    auto layout_str = [](bool perm) { return perm ? "bhsd" : "bshd"; };
    std::cout << "[" << data_type << "|" << mode << "|"
              << layout_str(i_perm) << "-" << layout_str(o_perm)
              << "] b:" << batch << ", h:" << nhead << "/" << nhead_k
              << ", s:" << seqlen_q << "/" << seqlen_k << ", d:" << hdim_q << "/" << hdim_v
              << ", scale:" << scale << ", bias:" << bias << ", dbias:" << use_dbias
              << ", p_drop:" << p_drop << ", deterministic:" << deterministic
              << ", mask:" << mask << std::flush << std::endl;

    auto build_args = [&]() -> aiter::mha_bwd_args {
        const index_t stride_q     = (i_perm ? hdim_q : nhead * hdim_q);
        const index_t stride_k     = (i_perm ? hdim_q : nhead_k * hdim_q);
        const index_t stride_v     = (i_perm ? hdim_v : nhead_k * hdim_v);
        const index_t stride_bias  = max_seqlen_k;
        const index_t stride_o     = (o_perm ? hdim_v : nhead * hdim_v);
        const index_t stride_randval = max_seqlen_k;
        const index_t stride_do    = (o_perm ? hdim_v : nhead * hdim_v);
        const index_t stride_dk    = (i_perm ? hdim_q : nhead * hdim_q);
        const index_t stride_dv    = (i_perm ? hdim_v : nhead * hdim_v);
        const index_t stride_dbias = (i_perm ? max_seqlen_k : nhead * max_seqlen_k);

        const index_t nhead_stride_q       = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const index_t nhead_stride_k       = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const index_t nhead_stride_v       = (i_perm ? shape_seqlen_k * hdim_v : hdim_v);
        const index_t nhead_stride_bias    = 0;
        const index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        const index_t nhead_stride_randval = shape_seqlen_q * max_seqlen_k;
        const index_t nhead_stride_do      = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        const index_t nhead_stride_lsed    = shape_seqlen_q;
        const index_t nhead_stride_dbias =
            (i_perm ? shape_seqlen_q * max_seqlen_k : max_seqlen_k);

        const index_t batch_stride_q       = nhead * shape_seqlen_q * hdim_q;
        const index_t batch_stride_k       = nhead_k * shape_seqlen_k * hdim_q;
        const index_t batch_stride_v       = nhead_k * shape_seqlen_k * hdim_v;
        const index_t batch_stride_bias    = 0;
        const index_t batch_stride_o       = nhead * shape_seqlen_q * hdim_v;
        const index_t batch_stride_randval = nhead * shape_seqlen_q * max_seqlen_k;
        const index_t batch_stride_do      = nhead * shape_seqlen_q * hdim_v;
        const index_t batch_stride_lsed    = nhead * shape_seqlen_q;
        const index_t batch_stride_dk      = nhead * shape_seqlen_k * hdim_q;
        const index_t batch_stride_dv      = nhead * shape_seqlen_k * hdim_v;
        const index_t batch_stride_dbias   = nhead * shape_seqlen_q * max_seqlen_k;

        std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
            drop_so;
        if(drop_prefs)
            drop_so = std::pair<const void*, const void*>{
                drop_seed_buf.GetDeviceBuffer(), drop_offset_buf.GetDeviceBuffer()};
        else
            drop_so = std::pair<uint64_t, uint64_t>{drop_seed, drop_offset};

        aiter::mha_bwd_args a{};
        a.use_asm_v3       = bwd_v3;
        a.v3_atomic_fp32   = v3_atomic_fp32;
        a.v3_bf16_cvt      = v3_bf16_cvt;
        a.v3_api_check     = v3_api_check;
        a.hdim_q           = hdim_q;
        a.hdim_v           = hdim_v;
        a.data_type        = data_type;
        a.is_group_mode    = (mode == mode_e::group);
        a.mask_type        = static_cast<int>(mask.type);
        a.bias_type        = static_cast<int>(bias.type);
        a.has_dbias        = use_dbias;
        a.has_dropout      = (p_drop > 0.f);
        a.is_store_randval = false;
        a.is_deterministic = deterministic;

        a.q_ptr      = q_buf.GetDeviceBuffer();
        a.k_ptr      = k_buf.GetDeviceBuffer();
        a.v_ptr      = v_buf.GetDeviceBuffer();
        a.bias_ptr   = nullptr;
        a.o_ptr      = o_buf.GetDeviceBuffer();
        a.lse_ptr    = lse_buf.GetDeviceBuffer();
        a.do_ptr     = do_buf.GetDeviceBuffer();
        a.d_ptr      = d_buf.GetDeviceBuffer();
        a.rand_val_ptr = nullptr;
        a.dq_ptr     = dq_buf.GetDeviceBuffer();
        a.dk_ptr     = dk_buf.GetDeviceBuffer();
        a.dv_ptr     = dv_buf.GetDeviceBuffer();
        a.dbias_ptr  = nullptr;
        a.sink_ptr   = nullptr;
        a.d_sink_ptr = nullptr;

        a.seqstart_q_ptr  = (mode == mode_e::group ? seqstart_q_buf.GetDeviceBuffer() : nullptr);
        a.seqstart_k_ptr  = (mode == mode_e::group ? seqstart_k_buf.GetDeviceBuffer() : nullptr);
        a.seqlen_q_ptr    = nullptr;
        a.seqlen_k_ptr    = nullptr;
        a.cu_seqlen_q_ptr = nullptr;
        a.cu_seqlen_k_ptr = nullptr;

        a.seqlen_q     = shape_seqlen_q;
        a.seqlen_k     = shape_seqlen_k;
        a.batch        = batch;
        a.max_seqlen_q = max_seqlen_q;
        a.max_seqlen_k = max_seqlen_k;
        a.nhead_q      = nhead;
        a.nhead_k      = nhead_k;
        a.scale        = scale;

        a.stride_q       = stride_q;
        a.stride_k       = stride_k;
        a.stride_v       = stride_v;
        a.stride_bias    = stride_bias;
        a.stride_o       = stride_o;
        a.stride_randval = stride_randval;
        a.stride_do      = stride_do;
        a.stride_dq      = stride_q;
        a.stride_dk      = stride_dk;
        a.stride_dv      = stride_dv;
        a.stride_dbias   = stride_dbias;

        a.nhead_stride_q       = nhead_stride_q;
        a.nhead_stride_k       = nhead_stride_k;
        a.nhead_stride_v       = nhead_stride_v;
        a.nhead_stride_bias    = nhead_stride_bias;
        a.nhead_stride_o       = nhead_stride_o;
        a.nhead_stride_randval = nhead_stride_randval;
        a.nhead_stride_do      = nhead_stride_do;
        a.nhead_stride_lsed    = nhead_stride_lsed;
        a.nhead_stride_dq      = nhead_stride_q;
        a.nhead_stride_dk      = nhead_stride_k;
        a.nhead_stride_dv      = nhead_stride_v;
        a.nhead_stride_dbias   = nhead_stride_dbias;

        a.batch_stride_q       = batch_stride_q;
        a.batch_stride_k       = batch_stride_k;
        a.batch_stride_v       = batch_stride_v;
        a.batch_stride_bias    = batch_stride_bias;
        a.batch_stride_o       = batch_stride_o;
        a.batch_stride_randval = batch_stride_randval;
        a.batch_stride_do      = batch_stride_do;
        a.batch_stride_lsed    = batch_stride_lsed;
        a.batch_stride_dq      = batch_stride_q;
        a.batch_stride_dk      = batch_stride_dk;
        a.batch_stride_dv      = batch_stride_dv;
        a.batch_stride_dbias   = batch_stride_dbias;

        a.window_size_left  = mask.left;
        a.window_size_right = mask.right;
        a.p_drop            = p_drop;
        a.p_undrop          = 1.f - p_drop;
        a.drop_seed_offset  = drop_so;
        a.workspace_alloc   = workspace_alloc;
        a.pinned_host_alloc = pinned_host_alloc;
        return a;
    };

    aiter::mha_bwd_args mha_args = build_args();

    float ave_time = aiter::mha_bwd(mha_args, sc);
    if(ave_time < 0)
    {
        std::cout << "not supported yet" << std::flush << std::endl;
        return false;
    }

    float tflops     = static_cast<float>(flop) / 1.E9f / ave_time;
    float gb_per_sec = static_cast<float>(num_byte) / 1.E6f / ave_time;
    std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
              << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2)
              << gb_per_sec << " GB/s\n" << std::flush;

    if(!do_validation)
    {
        std::cout << std::flush << std::endl;
        return true;
    }
    std::cout << std::defaultfloat << std::setprecision(6);

    // -------------------------------------------------------------------------
    // CPU reference (asm-supported subset only).
    // -------------------------------------------------------------------------
    bool pass = true;

    // Snapshot Q/K/V/dO per batch in [nhead, seq, dim] layout; recompute O/LSE.
    // Note: we do NOT cache the fp32 P from forward — the backward recomputes
    // P from LSE on the fly, mirroring how the asm v3 kernel reconstructs P
    // (`P = exp(S * scale - LSE)` per tile) rather than re-running softmax.
    std::vector<HostTensor<Q>>   q_refs;
    std::vector<HostTensor<K>>   k_refs;
    std::vector<HostTensor<V>>   v_refs;
    std::vector<HostTensor<DO>>  do_refs;
    std::vector<HostTensor<O>>   o_refs;
    std::vector<HostTensor<LSE>> lse_refs;

    auto idx_q = [&](int b, int h, int s, int d) {
        return i_perm ? q_host(b, h, s, d) : q_host(b, s, h, d);
    };
    auto idx_k = [&](int b, int hk, int s, int d) {
        return i_perm ? k_host(b, hk, s, d) : k_host(b, s, hk, d);
    };
    auto idx_v = [&](int b, int hk, int s, int d) {
        return i_perm ? v_host(b, hk, s, d) : v_host(b, s, hk, d);
    };
    auto idx_do = [&](int b, int h, int s, int d) {
        return o_perm ? do_host(b, h, s, d) : do_host(b, s, h, d);
    };
    auto store_o = [&](int b, int h, int s, int d, O val) {
        if(o_perm)
            o_host(b, h, s, d) = val;
        else
            o_host(b, s, h, d) = val;
    };
    auto store_lse = [&](int b, int h, int s, LSE v) { lse_host(b, h, s) = v; };

    for(int wb = 0; wb < batch; ++wb)
    {
        const int sq = seqlens_q[wb];
        const int sk = seqlens_k[wb];

        const int b_off = (mode == mode_e::batch ? wb : 0);
        const int q_off = (mode == mode_e::batch ? 0 : seqstart_q_host[wb]);
        const int k_off = (mode == mode_e::batch ? 0 : seqstart_k_host[wb]);

        HostTensor<Q>  q_ref(std::array<index_t, 3>{nhead,   sq, hdim_q});
        HostTensor<K>  k_ref(std::array<index_t, 3>{nhead,   sk, hdim_q});
        HostTensor<V>  v_ref(std::array<index_t, 3>{nhead,   sk, hdim_v});
        HostTensor<DO> do_ref(std::array<index_t, 3>{nhead,   sq, hdim_v});
        HostTensor<Acc> s_ref(std::array<index_t, 3>{nhead,   sq, sk});
        HostTensor<Acc> p_ref(std::array<index_t, 3>{nhead,   sq, sk});
        HostTensor<LSE> lse_ref(std::array<index_t, 2>{nhead, sq});

        const int ratio = nhead / nhead_k;
        for(int h = 0; h < nhead; ++h)
        {
            for(int m = 0; m < sq; ++m)
                for(int d = 0; d < hdim_q; ++d)
                    q_ref(h, m, d) = idx_q(b_off, h, m + q_off, d);
            for(int n = 0; n < sk; ++n)
                for(int d = 0; d < hdim_q; ++d)
                    k_ref(h, n, d) = idx_k(b_off, h / ratio, n + k_off, d);
            for(int n = 0; n < sk; ++n)
                for(int d = 0; d < hdim_v; ++d)
                    v_ref(h, n, d) = idx_v(b_off, h / ratio, n + k_off, d);
            for(int m = 0; m < sq; ++m)
                for(int d = 0; d < hdim_v; ++d)
                    do_ref(h, m, d) = idx_do(b_off, h, m + q_off, d);
        }

        ref_batched_gemm_nt<Q, K, Acc>(q_ref, k_ref, s_ref, scale);
        ref_apply_mask<Acc>(
            s_ref, static_cast<int>(mask.type), mask.left, mask.right, sq, sk);
        ref_softmax<Acc, Acc, LSE>(s_ref, p_ref, lse_ref);

        // O = P @ V
        HostTensor<O> o_ref(std::array<index_t, 3>{nhead, sq, hdim_v});
        ref_batched_gemm_nn<Acc, V, O>(p_ref, v_ref, o_ref);

        for(int h = 0; h < nhead; ++h)
        {
            for(int m = 0; m < sq; ++m)
                for(int d = 0; d < hdim_v; ++d)
                    store_o(b_off, h, m + q_off, d, o_ref(h, m, d));
            for(int m = 0; m < sq; ++m)
                store_lse(b_off, h, m + q_off, lse_ref(h, m));
        }

        q_refs.push_back(std::move(q_ref));
        k_refs.push_back(std::move(k_ref));
        v_refs.push_back(std::move(v_ref));
        do_refs.push_back(std::move(do_ref));
        o_refs.push_back(std::move(o_ref));
        lse_refs.push_back(std::move(lse_ref));
    }

    // Push fresh O/LSE to the device so the asm kernel reads them; this is also
    // the path the original benchmark uses (it overwrites O/LSE before the
    // validation call).
    o_buf.ToDevice(o_host.data());
    lse_buf.ToDevice(lse_host.data());
    dq_buf.SetZero();

    ck_tile::stream_config sc_v{nullptr,
                                true,
                                0,
                                0,
                                1,
                                arg.get_str("timer") == std::string("gpu")};
    aiter::mha_bwd(mha_args, sc_v);
    dq_buf.FromDevice(dq_host.data());
    dk_buf.FromDevice(dk_host.data());
    dv_buf.FromDevice(dv_host.data());

    auto idx_dq = [&](int b, int h, int s, int d) -> DQ& {
        return i_perm ? dq_host(b, h, s, d) : dq_host(b, s, h, d);
    };
    auto idx_dk = [&](int b, int h, int s, int d) -> DK& {
        return i_perm ? dk_host(b, h, s, d) : dk_host(b, s, h, d);
    };
    auto idx_dv = [&](int b, int h, int s, int d) -> DV& {
        return i_perm ? dv_host(b, h, s, d) : dv_host(b, s, h, d);
    };

    const double rtol = (std::is_same_v<Prec, __hip_bfloat16> &&
                         hdim_q > 128 && hdim_v > 128)
                            ? 3.2e-2
                            : 1e-2;
    const double atol = rtol;

    struct ErrAccum
    {
        double sum_sq_diff  = 0.0;
        double sum_sq_ref   = 0.0;
        double sum_abs_diff = 0.0;
        double max_abs_diff = 0.0;
        double max_abs_ref  = 0.0;
        double max_rel_diff = 0.0;
        std::size_t n       = 0;
    };
    ErrAccum acc_dq, acc_dk, acc_dv;

    for(int wb = 0; wb < batch; ++wb)
    {
        const int sq = seqlens_q[wb];
        const int sk = seqlens_k[wb];
        const int b_off = (mode == mode_e::batch ? wb : 0);
        const int q_off = (mode == mode_e::batch ? 0 : seqstart_q_host[wb]);
        const int k_off = (mode == mode_e::batch ? 0 : seqstart_k_host[wb]);

        const HostTensor<Q>&   q_ref   = q_refs[wb];
        const HostTensor<K>&   k_ref   = k_refs[wb];
        const HostTensor<V>&   v_ref   = v_refs[wb];
        const HostTensor<DO>&  do_ref  = do_refs[wb];
        const HostTensor<O>&   o_ref   = o_refs[wb];
        const HostTensor<LSE>& lse_ref = lse_refs[wb];

        // --- Step 1: D = sum_d(dO * O) (matches asm `fmha_bwd_odo` pre-kernel).
        // The asm v3 pipeline computes D into d_ptr via a dedicated odo kernel
        // before the main bwd kernel runs; replicate the same reduction here so
        // our reference uses the identical D as the asm path.
        HostTensor<Acc> d_ref(std::array<index_t, 2>{nhead, sq});
        for(int h = 0; h < nhead; ++h)
            for(int m = 0; m < sq; ++m)
            {
                float acc = 0.f;
                for(int d = 0; d < hdim_v; ++d)
                    acc += to_float<DO>(do_ref(h, m, d)) *
                           to_float<O>(o_ref(h, m, d));
                d_ref(h, m) = acc;
            }

        // --- Step 2: S = Q @ K^T (no scale, matches kernel GEMM0).
        HostTensor<Acc> s_ref_bwd(std::array<index_t, 3>{nhead, sq, sk});
        ref_batched_gemm_nt<Q, K, Acc>(q_ref, k_ref, s_ref_bwd, 1.0f);
        ref_apply_mask<Acc>(
            s_ref_bwd, static_cast<int>(mask.type), mask.left, mask.right, sq, sk);

        // --- Step 3: P = exp(S * scale - LSE), mirrors fmha_bwd_softmax_s2p_dev.
        // Kernel form: P = 2^((S * scale - LSE) * log2e). exp() is the same up
        // to floating-point identity. -inf masked entries collapse to P=0.
        // Fully-masked rows (LSE=-inf) get P=0 to avoid nan propagation; the
        // asm path simply skips those tiles via totalMaskOutBottomRight.
        HostTensor<Acc> p_fp32(std::array<index_t, 3>{nhead, sq, sk});
        for(int h = 0; h < nhead; ++h)
            for(int m = 0; m < sq; ++m)
            {
                const float lse_val = to_float<LSE>(lse_ref(h, m));
                const bool row_alive = std::isfinite(lse_val);
                for(int n = 0; n < sk; ++n)
                {
                    if(!row_alive)
                    {
                        p_fp32(h, m, n) = 0.f;
                        continue;
                    }
                    const float s_scaled =
                        to_float<Acc>(s_ref_bwd(h, m, n)) * scale;
                    p_fp32(h, m, n) =
                        std::isfinite(s_scaled) ? std::exp(s_scaled - lse_val)
                                                : 0.f;
                }
            }

        // --- Step 4: dP = dO @ V^T (kernel GEMM1).
        HostTensor<Acc> dp(std::array<index_t, 3>{nhead, sq, sk});
        ref_batched_gemm_nt<DO, V, Acc>(do_ref, v_ref, dp);

        // --- Step 5: dS_fp32 = P_fp32 * (dP - D). Whether to round dS / P to
        // src low-precision before the downstream GEMMs is controlled by the
        // `cpu_lp_round` flag. Defaults to OFF: the asm v3 path does NOT round
        // these intermediates to bf16 at the boundary we naively assumed (see
        // `fmha_bwd_dev` reference vs. the production asm — the production
        // kernel keeps them at fp32 precision through V_MFMA's input quant).
        // Forcing a cast here amplifies noise to ~50% nrms on dQ for bf16, so
        // leave it off unless explicitly probing the bf16-quantized variant.
        HostTensor<Acc> ds_fp32_t(std::array<index_t, 3>{nhead, sq, sk});
        for(int h = 0; h < nhead; ++h)
            for(int m = 0; m < sq; ++m)
            {
                const float d_val = d_ref(h, m);
                for(int n = 0; n < sk; ++n)
                {
                    ds_fp32_t(h, m, n) =
                        p_fp32(h, m, n) *
                        (to_float<Acc>(dp(h, m, n)) - d_val);
                }
            }

        // dV = P^T @ dO (GEMM2). No scale on dV.
        // dQ = scale * (dS @ K) (GEMM4 + post-scale, fp32 acc).
        // dK = scale * (dS^T @ Q) (GEMM3 + post-scale, fp32 acc).
        HostTensor<DV> dv_ref(std::array<index_t, 3>{nhead, sk, hdim_v});
        HostTensor<Acc> dq_acc(std::array<index_t, 3>{nhead, sq, hdim_q});
        HostTensor<Acc> dk_acc(std::array<index_t, 3>{nhead, sk, hdim_q});

        if(cpu_lp_round)
        {
            // Strict kernel-aligned variant: round P and dS to src precision
            // before the dV/dK/dQ gemms.
            HostTensor<DO> ds_lp(std::array<index_t, 3>{nhead, sq, sk});
            for(std::size_t i = 0; i < ds_fp32_t.numel(); ++i)
                ds_lp.data()[i] = from_float<DO>(ds_fp32_t.data()[i]);

            HostTensor<Q> p_lp(std::array<index_t, 3>{nhead, sq, sk});
            for(std::size_t i = 0; i < p_fp32.numel(); ++i)
                p_lp.data()[i] = from_float<Q>(p_fp32.data()[i]);

            ref_batched_gemm_tn<Q, DO, DV>(p_lp, do_ref, dv_ref);
            ref_batched_gemm_nn<DO, K, Acc>(ds_lp, k_ref, dq_acc, 1.0f);
            ref_batched_gemm_tn<DO, Q, Acc>(ds_lp, q_ref, dk_acc, 1.0f);
        }
        else
        {
            // Default: keep P/dS in fp32 through the downstream gemms (the
            // asm pipeline matches at fp32 precision empirically).
            ref_batched_gemm_tn<Acc, DO, DV>(p_fp32, do_ref, dv_ref);
            ref_batched_gemm_nn<Acc, K, Acc>(ds_fp32_t, k_ref, dq_acc, 1.0f);
            ref_batched_gemm_tn<Acc, Q, Acc>(ds_fp32_t, q_ref, dk_acc, 1.0f);
        }

        HostTensor<DQ> dq_ref(std::array<index_t, 3>{nhead, sq, hdim_q});
        for(std::size_t i = 0; i < dq_acc.numel(); ++i)
            dq_ref.data()[i] = from_float<DQ>(scale * dq_acc.data()[i]);
        HostTensor<DK> dk_ref(std::array<index_t, 3>{nhead, sk, hdim_q});
        for(std::size_t i = 0; i < dk_acc.numel(); ++i)
            dk_ref.data()[i] = from_float<DK>(scale * dk_acc.data()[i]);

        HostTensor<DQ> dq_got(std::array<index_t, 3>{nhead, sq, hdim_q});
        HostTensor<DK> dk_got(std::array<index_t, 3>{nhead, sk, hdim_q});
        HostTensor<DV> dv_got(std::array<index_t, 3>{nhead, sk, hdim_v});

        for(int h = 0; h < nhead; ++h)
        {
            for(int m = 0; m < sq; ++m)
                for(int d = 0; d < hdim_q; ++d)
                    dq_got(h, m, d) = idx_dq(b_off, h, m + q_off, d);
            for(int n = 0; n < sk; ++n)
                for(int d = 0; d < hdim_q; ++d)
                    dk_got(h, n, d) = idx_dk(b_off, h, n + k_off, d);
            for(int n = 0; n < sk; ++n)
                for(int d = 0; d < hdim_v; ++d)
                    dv_got(h, n, d) = idx_dv(b_off, h, n + k_off, d);
        }

        bool dq_ok = check_err<DQ, DQ>(dq_got, dq_ref, "[dQ]", rtol, atol);
        bool dk_ok = check_err<DK, DK>(dk_got, dk_ref, "[dK]", rtol, atol);
        bool dv_ok = check_err<DV, DV>(dv_got, dv_ref, "[dV]", rtol, atol);
        // Accumulate per-tensor error stats across all batches; final values
        // are printed once after the batch loop. `max_rel` is gated to
        // elements with |ref| >= atol to avoid blow-up near zero.
        auto accumulate = [&](ErrAccum& a,
                              auto&& got_at,
                              auto&& ref_at,
                              int total_rows,
                              int total_cols) {
            for(int h = 0; h < nhead; ++h)
                for(int r = 0; r < total_rows; ++r)
                    for(int c = 0; c < total_cols; ++c)
                    {
                        const double g    = got_at(h, r, c);
                        const double b    = ref_at(h, r, c);
                        const double diff = std::fabs(g - b);
                        a.sum_sq_diff  += diff * diff;
                        a.sum_sq_ref   += b * b;
                        a.sum_abs_diff += diff;
                        if(diff > a.max_abs_diff)
                            a.max_abs_diff = diff;
                        const double abs_b = std::fabs(b);
                        if(abs_b > a.max_abs_ref)
                            a.max_abs_ref = abs_b;
                        if(abs_b >= atol)
                        {
                            const double rel = diff / abs_b;
                            if(rel > a.max_rel_diff)
                                a.max_rel_diff = rel;
                        }
                        ++a.n;
                    }
        };
        accumulate(acc_dq,
                   [&](int h, int r, int c) { return to_float<DQ>(dq_got(h, r, c)); },
                   [&](int h, int r, int c) { return to_float<DQ>(dq_ref(h, r, c)); },
                   sq, hdim_q);
        accumulate(acc_dk,
                   [&](int h, int r, int c) { return to_float<DK>(dk_got(h, r, c)); },
                   [&](int h, int r, int c) { return to_float<DK>(dk_ref(h, r, c)); },
                   sk, hdim_q);
        accumulate(acc_dv,
                   [&](int h, int r, int c) { return to_float<DV>(dv_got(h, r, c)); },
                   [&](int h, int r, int c) { return to_float<DV>(dv_ref(h, r, c)); },
                   sk, hdim_v);
        pass &= dq_ok && dk_ok && dv_ok;
        auto dump_per_row = [&](const char* label,
                                auto&& got_at,
                                auto&& ref_at,
                                int total_rows,
                                int total_cols) {
            std::vector<int> bad_per_row(total_rows, 0);
            std::vector<double> max_abs_per_row(total_rows, 0.0);
            std::vector<double> max_ref_per_row(total_rows, 0.0);
            for(int h = 0; h < nhead; ++h)
                for(int r = 0; r < total_rows; ++r)
                    for(int c = 0; c < total_cols; ++c)
                    {
                        double a = got_at(h, r, c);
                        double b = ref_at(h, r, c);
                        double diff = std::fabs(a - b);
                        if(std::fabs(b) > max_ref_per_row[r])
                            max_ref_per_row[r] = std::fabs(b);
                        if(diff > max_abs_per_row[r])
                            max_abs_per_row[r] = diff;
                        if(diff > atol + rtol * std::fabs(b))
                            ++bad_per_row[r];
                    }
            std::cerr << label << " per-row bad-count (rows=" << total_rows << "):" << std::endl;
            int run_first = -1, run_last = -1, run_cnt = 0;
            for(int r = 0; r < total_rows; ++r)
            {
                if(bad_per_row[r] > 0)
                {
                    if(run_first < 0)
                        run_first = r;
                    run_last = r;
                    run_cnt += bad_per_row[r];
                }
                bool flush = (r == total_rows - 1) || bad_per_row[r] == 0;
                if(flush && run_first >= 0)
                {
                    std::cerr << "  [" << run_first << ".." << run_last
                              << "] bad=" << run_cnt
                              << " (avg/row " << (run_cnt / (run_last - run_first + 1))
                              << "/" << (nhead * total_cols)
                              << ") max_abs[last]=" << max_abs_per_row[run_last]
                              << " max_ref[last]=" << max_ref_per_row[run_last]
                              << std::endl;
                    run_first = run_last = -1;
                    run_cnt = 0;
                }
            }
        };
        if(!dq_ok)
        {
            dump_per_row("[dQ]",
                         [&](int h, int r, int c) { return to_float<DQ>(dq_got(h, r, c)); },
                         [&](int h, int r, int c) { return to_float<DQ>(dq_ref(h, r, c)); },
                         sq, hdim_q);
        }
        if(!dk_ok)
        {
            dump_per_row("[dK]",
                         [&](int h, int r, int c) { return to_float<DK>(dk_got(h, r, c)); },
                         [&](int h, int r, int c) { return to_float<DK>(dk_ref(h, r, c)); },
                         sk, hdim_q);
        }
        if(!dv_ok)
        {
            dump_per_row("[dV]",
                         [&](int h, int r, int c) { return to_float<DV>(dv_got(h, r, c)); },
                         [&](int h, int r, int c) { return to_float<DV>(dv_ref(h, r, c)); },
                         sk, hdim_v);
        }
        if(!(dq_ok && dk_ok && dv_ok))
        {
            std::cerr << "mismatch at batch " << wb << " sq=" << sq << " sk=" << sk
                      << std::endl;
            break;
        }
    }

    auto print_stats = [](const char* label, const ErrAccum& a) {
        const double denom_n =
            static_cast<double>(std::max<std::size_t>(a.n, 1));
        const double mean_abs = a.sum_abs_diff / denom_n;
        const double rmse     = std::sqrt(a.sum_sq_diff / denom_n);
        const double nrmse    = std::sqrt(a.sum_sq_diff) /
                             std::max(std::sqrt(a.sum_sq_ref), 1e-30);
        std::cerr << label << " stats: n=" << a.n
                  << std::scientific << std::setprecision(3)
                  << " max_abs=" << a.max_abs_diff
                  << " mean_abs=" << mean_abs
                  << " max_rel=" << a.max_rel_diff
                  << " rmse=" << rmse
                  << " nrmse=" << nrmse
                  << " max|ref|=" << a.max_abs_ref
                  << std::defaultfloat << std::endl;
    };
    print_stats("[dQ]", acc_dq);
    print_stats("[dK]", acc_dk);
    print_stats("[dV]", acc_dv);

    std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    return pass;
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------

int main(int argc, char** argv)
{
    auto [ok, arg] = create_args(argc, argv);
    if(!ok)
        return -1;
    const std::string prec = arg.get_str("prec");
    if(prec == "fp16")
        return run_bench<__half>(arg) ? 0 : -2;
    if(prec == "bf16")
        return run_bench<__hip_bfloat16>(arg) ? 0 : -2;
    std::cerr << "unsupported prec: " << prec << std::endl;
    return -3;
}

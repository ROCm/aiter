#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Minimal shim providing ck_tile:: types and FMHA enums/structs when
// compiling without the full Composable Kernel dependency (ENABLE_CK==0).

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ck_tile {

using index_t      = int32_t;
using long_index_t = int64_t;

struct stream_config
{
    hipStream_t stream_id_ = nullptr;
    bool time_kernel_      = false;
    int log_level_         = 0;
    int cold_niters_       = 3;
    int nrepeat_           = 10;
    bool is_gpu_timer_     = true;
    bool flush_cache_      = false;
    int rotating_count_    = 1;
};

template <typename T>
constexpr T log2e_v = static_cast<T>(1.4426950408889634);

__device__ constexpr index_t get_warp_size()
{
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
    return 64;
#else
    return 32;
#endif
#else
    return 64;
#endif
}

inline __host__ index_t get_warp_size()
{
#if !(defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC))
    int device  = 0;
    int result  = 0;
    auto status = hipGetDevice(&device);
    if(status == hipSuccess)
    {
        status = hipDeviceGetAttribute(&result, hipDeviceAttributeWarpSize, device);
        if(status == hipSuccess)
        {
            return result;
        }
    }
#endif
    return 64;
}

static __global__ void flush_cache()
{
    asm __volatile__("s_icache_inv \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t"
                     "s_nop 0 \n\t" ::
                         :);
}

inline void hip_check_error(hipError_t err)
{
    if(err != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(err) << ". " << __FILE__ << ": "
           << __LINE__ << " in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}

struct gpu_timer
{
    gpu_timer()
    {
        hip_check_error(hipEventCreate(&start_evt));
        hip_check_error(hipEventCreate(&stop_evt));
    }

    ~gpu_timer() noexcept(false)
    {
        hip_check_error(hipEventDestroy(start_evt));
        hip_check_error(hipEventDestroy(stop_evt));
    }

    void start(const hipStream_t& s)
    {
        hip_check_error(hipStreamSynchronize(s));
        hip_check_error(hipEventRecord(start_evt, s));
    }

    void stop(const hipStream_t& s)
    {
        hip_check_error(hipEventRecord(stop_evt, s));
        hip_check_error(hipEventSynchronize(stop_evt));
    }

    float duration() const
    {
        float ms = 0;
        hip_check_error(hipEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }

    hipEvent_t start_evt = nullptr;
    hipEvent_t stop_evt  = nullptr;
};

struct cpu_timer
{
    using clock_t = std::chrono::steady_clock;

    void start(const hipStream_t& s)
    {
        hip_check_error(hipStreamSynchronize(s));
        start_tick = clock_t::now();
    }

    void stop(const hipStream_t& s)
    {
        hip_check_error(hipStreamSynchronize(s));
        stop_tick = clock_t::now();
    }

    float duration() const
    {
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(stop_tick - start_tick);
        return static_cast<float>(us.count()) / 1.0e3f;
    }

    clock_t::time_point start_tick{};
    clock_t::time_point stop_tick{};
};

template <typename... Callables>
void launch_and_check(const stream_config& sc, Callables&&... callables)
{
    if(!((static_cast<void>(callables(sc)), hipPeekAtLastError() == hipSuccess) && ...))
    {
        hip_check_error(hipGetLastError());
    }
}

inline void flush_icache()
{
    hipDeviceProp_t device_props;
    hip_check_error(hipGetDeviceProperties(&device_props, 0));

    constexpr int32_t blocks_per_cu = 60;
    int32_t gpu_block3              = device_props.multiProcessorCount * blocks_per_cu;

    flush_cache<<<dim3(gpu_block3), dim3(64), 0, nullptr>>>();
    hip_check_error(hipGetLastError());
}

template <typename TimerType, typename PreprocessFunc>
double preprocess_profiling_impl(TimerType timer, const stream_config& s, PreprocessFunc preprocess)
{
    timer.start(s.stream_id_);
    for(int i = 0; i < s.nrepeat_; ++i)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
    }
    timer.stop(s.stream_id_);

    return timer.duration() / s.nrepeat_;
}

template <typename TimerType, typename CallablesFunc, typename PreprocessFunc = std::nullptr_t>
double timing_loop_impl(TimerType timer,
                        const stream_config& s,
                        CallablesFunc&& callables_func,
                        PreprocessFunc preprocess = nullptr)
{
    for(int i = 0; i < s.cold_niters_; ++i)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
    }

    int i = 0;
    timer.start(s.stream_id_);
    while(i < s.nrepeat_)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
        ++i;
    }
    timer.stop(s.stream_id_);

    if(i == 0)
    {
        return 0.0;
    }
    return timer.duration() / s.nrepeat_;
}

template <typename TimerType, typename CallablesFunc, typename PreprocessFunc = std::nullptr_t>
double timing_loop_flush_cache_impl(TimerType timer,
                                    const stream_config& s,
                                    CallablesFunc&& callables_func,
                                    PreprocessFunc preprocess = nullptr)
{
    auto run_flush_cache = [&]() { flush_icache(); };

    for(int i = 0; i < s.cold_niters_; ++i)
    {
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
    }

    int i = 0;
    timer.start(s.stream_id_);
    while(i < s.nrepeat_)
    {
        run_flush_cache();
        if constexpr(!std::is_same_v<PreprocessFunc, std::nullptr_t>)
        {
            preprocess();
        }
        callables_func();
        ++i;
    }
    timer.stop(s.stream_id_);

    if(i == 0)
    {
        return 0.0;
    }

    const double flush_cache_time = preprocess_profiling_impl(gpu_timer{}, s, run_flush_cache);
    return (timer.duration() / s.nrepeat_) - flush_cache_time;
}

template <typename... Callables>
float launch_kernel(const stream_config& s, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_impl(gpu_timer{}, s, callables_func);
    }
    return timing_loop_impl(cpu_timer{}, s, callables_func);
}

template <typename PreprocessFunc, typename... Callables>
float launch_kernel_time_mask(const stream_config& s, PreprocessFunc preprocess, Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        preprocess();
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_impl(gpu_timer{}, s, callables_func, preprocess);
    }
    return timing_loop_impl(cpu_timer{}, s, callables_func, preprocess);
}

template <typename PreprocessFunc, typename... Callables>
float launch_kernel_time_mask_flush_cache(const stream_config& s,
                                          PreprocessFunc preprocess,
                                          Callables&&... callables)
{
    static_assert(sizeof...(callables) > 0, "At least one callable is required!");

    if(!s.time_kernel_)
    {
        preprocess();
        launch_and_check(s, std::forward<Callables>(callables)...);
        return 0;
    }

    auto callables_func = [&]() { launch_and_check(s, std::forward<Callables>(callables)...); };

    if(s.is_gpu_timer_)
    {
        return timing_loop_flush_cache_impl(gpu_timer{}, s, callables_func, preprocess);
    }
    return timing_loop_flush_cache_impl(cpu_timer{}, s, callables_func, preprocess);
}

} // namespace ck_tile

// keep this in sync with ck_tile::GenericAttentionMaskEnum
enum class mask_enum
{
    no_mask = 0,
    mask_top_left,
    mask_bottom_right,
    window_generic,
};

// keep sync with BlockAttentionBiasEnum
enum class bias_enum
{
    no_bias          = 0,
    elementwise_bias = 1,
    alibi            = 2,
};

inline std::pair<ck_tile::index_t, ck_tile::index_t>
compute_mask_coordinates(ck_tile::index_t left_size,
                         ck_tile::index_t right_size,
                         ck_tile::index_t sink_size,
                         ck_tile::index_t y_total,
                         ck_tile::index_t x_total,
                         bool is_top_left)
{
    (void)sink_size;
    ck_tile::index_t left_tmp  = is_top_left ? y_total - 1 : x_total - 1;
    ck_tile::index_t right_tmp = is_top_left ? x_total - 1 : y_total - 1;
    left_size     = left_size < 0 ? left_tmp : left_size;
    right_size    = right_size < 0 ? right_tmp : right_size;
    ck_tile::index_t x_off     = is_top_left ? 0 : x_total - y_total;
    ck_tile::index_t y_off     = is_top_left ? 0 : y_total - x_total;
    return {1 + left_size + y_off, 1 + right_size + x_off};
}

struct mask_info
{
    mask_enum type;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t y, x;
    ck_tile::index_t left, right;
    ck_tile::index_t sink;

    void serialize(std::ostream& os) const
    {
        if(type == mask_enum::no_mask)
            os << "n";
        else if(type == mask_enum::mask_top_left)
            os << "t(" << left << ":" << right << ")";
        else if(type == mask_enum::mask_bottom_right)
            os << "b(" << left << ":" << right << ")";
        else
            os << "g(" << y << ":" << x << ")";
    }

    static mask_info decode(std::string str, ck_tile::index_t sq, ck_tile::index_t sk)
    {
        mask_info tmp{};
        tmp.seqlen_q = sq;
        tmp.seqlen_k = sk;
        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            if(t == "xt" || t == "xb")
            {
                ck_tile::index_t window_size = std::stoi(v);
                ck_tile::index_t left_size   = -1;
                ck_tile::index_t right_size  = 0;
                if(window_size > 0)
                {
                    left_size  = window_size / 2;
                    right_size = window_size - 1 - left_size;
                }
                auto [my, mx] = compute_mask_coordinates(
                    left_size, right_size, 0, sq, sk, t == "xt");
                tmp.type  = t == "xt" ? mask_enum::mask_top_left : mask_enum::mask_bottom_right;
                tmp.y     = my;
                tmp.x     = mx;
                tmp.left  = left_size;
                tmp.right = right_size;
                tmp.sink  = 0;
            }
            else if(t == "t" || t == "b" || t == "g")
            {
                auto found_1 = v.find(",");
                if(found_1 == std::string::npos)
                    throw std::invalid_argument("invalid mask value: " + str);
                ck_tile::index_t v0   = atoi(v.substr(0, found_1).c_str());
                auto found_2          = v.find(',', found_1 + 1);
                ck_tile::index_t v1   = 0;
                ck_tile::index_t sink = 0;
                if(t == "t")
                {
                    if(found_2 != std::string::npos)
                    {
                        v1   = atoi(v.substr(found_1 + 1, found_2 - found_1 - 1).c_str());
                        sink = atoi(v.substr(found_2 + 1).c_str());
                    }
                    else
                    {
                        v1   = atoi(v.substr(found_1 + 1).c_str());
                        sink = 0;
                    }
                    tmp.type = mask_enum::mask_top_left;
                    auto [my, mx] =
                        compute_mask_coordinates(v0, v1, sink, sq, sk, true);
                    tmp.y    = my;
                    tmp.x    = mx;
                    tmp.left = v0;
                    tmp.right = v1;
                    tmp.sink = sink;
                }
                else if(t == "b")
                {
                    if(found_2 != std::string::npos)
                    {
                        v1   = atoi(v.substr(found_1 + 1, found_2 - found_1 - 1).c_str());
                        sink = atoi(v.substr(found_2 + 1).c_str());
                    }
                    else
                    {
                        v1   = atoi(v.substr(found_1 + 1).c_str());
                        sink = 0;
                    }
                    tmp.type = mask_enum::mask_bottom_right;
                    auto [my, mx] =
                        compute_mask_coordinates(v0, v1, sink, sq, sk, false);
                    tmp.y     = my;
                    tmp.x     = mx;
                    tmp.left  = v0;
                    tmp.right = v1;
                    tmp.sink  = sink;
                }
                else if(t == "g")
                {
                    tmp.type  = mask_enum::window_generic;
                    tmp.y     = v0;
                    tmp.x     = atoi(v.substr(found_1 + 1).c_str());
                    tmp.left  = v0;
                    tmp.right = tmp.x;
                    tmp.sink  = 0;
                }
            }
            else
            {
                throw std::invalid_argument("invalid mask value: " + str);
            }
        }
        else if(str == "0")
        {
            tmp.type  = mask_enum::no_mask;
            tmp.left  = -1;
            tmp.right = -1;
            tmp.sink  = 0;
        }
        else if(str == "1" || str == "t")
        {
            tmp.type  = mask_enum::mask_top_left;
            tmp.y     = sq;
            tmp.x     = 1;
            tmp.left  = -1;
            tmp.right = 0;
            tmp.sink  = 0;
        }
        else if(str == "2" || str == "b")
        {
            tmp.type  = mask_enum::mask_bottom_right;
            tmp.y     = sq;
            tmp.x     = sk - sq + 1;
            tmp.left  = -1;
            tmp.right = 0;
            tmp.sink  = 0;
        }
        else
        {
            throw std::invalid_argument("invalid mask value: " + str);
        }
        return tmp;
    }

    std::size_t get_unmaskarea() const
    {
        if(type == mask_enum::no_mask)
            return static_cast<std::size_t>(seqlen_q) * seqlen_k;
        std::size_t area = 0;
        for(ck_tile::index_t i_y = 0; i_y < seqlen_q; ++i_y)
        {
            ck_tile::index_t x_start = std::max(-y + i_y + 1, static_cast<ck_tile::index_t>(0));
            ck_tile::index_t x_end   = std::min(i_y + x, seqlen_k);
            if(x_end > x_start)
            {
                area += (x_end - x_start);
            }
        }
        return area;
    }

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi)
    {
        mi.serialize(os);
        return os;
    }
};

struct bias_info
{
    bias_enum type;
    int rank_info;

    void serialize(std::ostream& os) const
    {
        if(type == bias_enum::no_bias)
            os << "n";
        else if(type == bias_enum::elementwise_bias)
        {
            os << "e";
            if(rank_info != 0)
            {
                os << "[" << rank_info << "]";
            }
        }
        else if(type == bias_enum::alibi)
        {
            os << "alibi";
            if(rank_info != 0)
            {
                os << "[" << rank_info << "]";
            }
        }
    }

    static bias_info decode(std::string str)
    {
        bias_info info{bias_enum::no_bias, 0};
        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            if(t == "e" || t == "elementwise")
            {
                info.type      = bias_enum::elementwise_bias;
                info.rank_info = std::stoi(v);
                if(info.rank_info < 0 || info.rank_info > 2)
                    throw std::invalid_argument("invalid bias rank: " + str);
            }
            else if(t == "a" || t == "alibi")
            {
                info.type      = bias_enum::alibi;
                info.rank_info = std::stoi(v);
                if(info.rank_info < 0 || info.rank_info > 1)
                    throw std::invalid_argument("invalid bias rank: " + str);
            }
            else
            {
                throw std::invalid_argument("invalid bias value: " + str);
            }
        }
        else if(str == "0" || str == "n")
        {
            info.type = bias_enum::no_bias;
        }
        else if(str == "1" || str == "e" || str == "elementwise")
        {
            info.type = bias_enum::elementwise_bias;
        }
        else if(str == "2" || str == "a" || str == "alibi")
        {
            info.type = bias_enum::alibi;
        }
        else
        {
            throw std::invalid_argument("invalid bias value: " + str);
        }
        return info;
    }

    friend std::ostream& operator<<(std::ostream& os, const bias_info& bi)
    {
        bi.serialize(os);
        return os;
    }
};

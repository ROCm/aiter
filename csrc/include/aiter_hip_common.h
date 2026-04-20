// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#define AITER_C_ITFS extern "C" __attribute__((visibility("default")))

#include "aiter_enum.h"
#include "aiter_logger.h"
#if !ENABLE_CK
#include "ck_tile_shim.h"
#else
#include "ck_tile/core.hpp"
#endif
#include <cstdint>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <fstream>
#include <mutex>
#include <memory>
#ifdef AITER_EMBEDDED_HSA_HEADER
#include AITER_EMBEDDED_HSA_HEADER
#endif

namespace aiter_detail {

inline thread_local bool g_aiter_can_throw = false;

template <typename... Args>
[[noreturn, gnu::noinline]] inline void aiter_check_fatal(const char* file, size_t line, Args&&... args)
{
    std::cerr << "[AITER] " << file << ":" << line << " ";
    (std::cerr << ... << std::forward<Args>(args)) << std::endl;
    std::abort();
}

template <typename... Args>
[[noreturn]] inline void check_fail(const char* file, int line, Args&&... args)
{
    std::ostringstream oss;
    oss << "[AITER] " << file << ":" << line << " ";
    (oss << ... << std::forward<Args>(args));
    std::string msg = oss.str();
    std::cerr << msg << std::endl;
    if(g_aiter_can_throw)
    {
        throw std::runtime_error(std::move(msg));
    }
    std::abort();
}
} // namespace aiter_detail

#define AITER_CHECK(x, ...)                                            \
    do                                                                 \
    {                                                                  \
        if(!(x)) [[unlikely]]                                          \
        {                                                              \
            aiter_detail::check_fail(__FILE__, __LINE__, __VA_ARGS__); \
        }                                                              \
    } while(0)

// Fatal on any HIP error -- use for init/teardown/resource management where
// failure means unrecoverable state.
#define HIP_CALL(call)                                                            \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if(err != hipSuccess) [[unlikely]]                                        \
        {                                                                         \
            aiter_detail::aiter_check_fatal(__FILE__,                                 \
                                        __LINE__,                                 \
                                        "fail to call " #call " ---> [HIP error](", \
                                        hipGetErrorString(err),                   \
                                        ')');                                       \
        }                                                                         \
    } while(0)

// Launch-specific HIP error handling.
// - hipErrorInvalidValue is treated as recoverable because it commonly means
//   a software configuration problem (for example invalid grid/block dims)
//   that tuning code can catch and skip without leaving the GPU in a bad state.
// - All other launch failures remain fatal because they may indicate runtime
//   or hardware problems after which continuing is unsafe.
#define HIP_CALL_LAUNCH(call)                                                     \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if(err != hipSuccess) [[unlikely]]                                        \
        {                                                                         \
            if(err == hipErrorInvalidValue)                                        \
            {                                                                     \
                aiter_detail::check_fail(__FILE__, __LINE__,                       \
                    "fail to call " #call " ---> [HIP error](",                    \
                    hipGetErrorString(err), ')');                                   \
            }                                                                     \
            else                                                                  \
            {                                                                     \
                aiter_detail::aiter_check_fatal(__FILE__, __LINE__,               \
                    "fail to call " #call " ---> [HIP error](",                    \
                    hipGetErrorString(err), ')');                                   \
            }                                                                     \
        }                                                                         \
    } while(0)

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct p1
{
    unsigned int _p0;
};

struct AiterAsmKernelArgs
{
    void* args_ptr;
    size_t* arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

static const std::string get_gpu_arch();

namespace aiter_detail {
// Taken from
// https://github.com/llvm/llvm-project/blob/b0230f59969b9e8e7e0aff44cd34718987098462/llvm/lib/Frontend/Offloading/OffloadWrapper.cpp#L226
struct FatBinaryWrapper
{
    uint32_t magic        = 0x48495046; // "HIPF";
    uint32_t version      = 1;
    const void* binary = nullptr;
    intptr_t __pad        = 0;
};

extern "C" void* __hipRegisterFatBinary(const FatBinaryWrapper* data) noexcept;
extern "C" void __hipUnregisterFatBinary(void* module) noexcept;
extern "C" void __hipRegisterFunction(void* module,
                                      const void* hostFunction,
                                      const char* deviceFunction,
                                      const char* deviceName,
                                      int threadLimit,
                                      void* tid,
                                      void* bid,
                                      void* blockDim,
                                      void* gridDim,
                                      void* wSize) noexcept;
} // namespace aiter_detail

namespace {

class AiterAsmKernelFast
{
    private:
    void* module = nullptr;

    protected:
    AiterAsmKernelFast() = default;
    void init(const char* kernel_name, const void* hsaco)
    {
        aiter_detail::FatBinaryWrapper fat_bin{};
        fat_bin.binary = hsaco;
        module         = aiter_detail::__hipRegisterFatBinary(&fat_bin);
        AITER_CHECK(module != nullptr, "failed to load module for ", kernel_name);
        aiter_detail::__hipRegisterFunction(module,
                                            static_cast<void*>(this),
                                            kernel_name,
                                            kernel_name,
                                            -1,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr);
    }

    public:
    AiterAsmKernelFast(const char* kernel_name, const void* hsaco)
    {
        init(kernel_name, hsaco);
    };

    ~AiterAsmKernelFast() { aiter_detail::__hipUnregisterFatBinary(module); }

    AiterAsmKernelFast(AiterAsmKernelFast&)             = delete;
    AiterAsmKernelFast(AiterAsmKernelFast&&)            = delete;
    AiterAsmKernelFast& operator=(AiterAsmKernelFast&)  = delete;
    AiterAsmKernelFast& operator=(AiterAsmKernelFast&&) = delete;

    void launch_kernel(const AiterAsmKernelArgs& kargs)
    {
        void* config[]            = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                     kargs.args_ptr,
                                     HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                     kargs.arg_size_ptr,
                                     HIP_LAUNCH_PARAM_END};
        hipFunction_t kernel_func = nullptr;
        // TODO Ask runtime folks to provide an API for hipLaunchKernel with extra arg
        // Don't error check here.
        // Failure to load the func would cause hipModuleLaunchKernel to fail anyways.
        (void)hipGetFuncBySymbol(&kernel_func, reinterpret_cast<void*>(this));

        HIP_CALL_LAUNCH(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx,
                                       kargs.gdy,
                                       kargs.gdz,
                                       kargs.bdx,
                                       kargs.bdy,
                                       kargs.bdz,
                                       0,
                                       kargs.stream,
                                       nullptr,
                                       (void**)&config));
    };
};


class AiterAsmKernel: private AiterAsmKernelFast
{
    private:
    std::unique_ptr<char[]> hsaco_data;

    const void* load_hsaco_file(const char* kernel_name, const char* hsaco_path)
    {
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::string arch_name     = get_gpu_arch();
        if(AITER_ASM_DIR != nullptr)
        {
            std::string full_path = std::string(AITER_ASM_DIR) + "/" + arch_name + "/" + hsaco_path;
            AITER_LOG_INFO("RegisterKernel: " << kernel_name << " hsaco: " << full_path);

            std::ifstream file(full_path, std::ios::binary | std::ios::ate);

            AITER_CHECK(file.is_open(), "failed to open ", full_path.c_str());

            size_t file_size = file.tellg();
            hsaco_data.reset(new char[file_size]);

            file.seekg(0, std::ios::beg);
            AITER_CHECK(
                file.read(hsaco_data.get(), file_size), "failed to read ", full_path.c_str());
            return hsaco_data.get();
        }
        else
        {
#if defined(AITER_EMBEDDED_HSA_HEADER) && defined(AITER_EMBEDDED_HSA_MAP)
            std::string fname = "hsa/" + arch_name + "/" + hsaco_path;
            auto hasco_obj    = AITER_EMBEDDED_HSA_MAP.find(fname);
            AITER_CHECK(hasco_obj != AITER_EMBEDDED_HSA_MAP.end(), "hasco_obj not found");
            AITER_CHECK(hasco_obj->second.data() != nullptr, "hasco_obj is nullptr");
            AITER_LOG_INFO("RegisterKernel: " << kernel_name << " hsaco: [embedded] " << fname);
            return hasco_obj->second.data();
#else
            AITER_CHECK(AITER_ASM_DIR != nullptr, "AITER_ASM_DIR not set");
            return nullptr;
#endif
        }
    }

    public:
    AiterAsmKernel(const char* kernel_name, const char* hsaco_path)
    {
        init(kernel_name, load_hsaco_file(kernel_name, hsaco_path));
    };

    using AiterAsmKernelFast::launch_kernel;
};


} // namespace

static const std::string get_gpu_arch()
{
    int device_count;
    HIP_CALL(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        return "No GPU Found";
    }

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    std::string arch_full = dev_prop.gcnArchName;
    size_t colon_pos      = arch_full.find(':');
    if(colon_pos != std::string::npos)
    {
        return arch_full.substr(0, colon_pos);
    }
    else
    {
        return arch_full;
    }
}

static uint32_t get_num_cu_func()
{
    auto get_num_cu_local = []() {
        hipDevice_t dev;
        hipDeviceProp_t dev_prop;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        return dev_prop.multiProcessorCount;
    };
    static const uint32_t num_cu = get_num_cu_local();
    return num_cu;
}

static uint32_t get_warp_size_func()
{
    static const uint32_t warp_size = []() {
        hipDevice_t dev;
        hipDeviceProp_t dev_prop;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
        return static_cast<uint32_t>(dev_prop.warpSize);
    }();
    return warp_size;
}

struct WarpSizeValue
{
    __host__ __device__ constexpr operator int() const
    {
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
        return 64;
#else
        return 32;
#endif
#else
        if(__builtin_is_constant_evaluated())
        {
            return 64; // host pass fallback
        }
        return static_cast<int>(get_warp_size_func());
#endif
    }
};

// WARNING: Do not use WARP_SIZE as const/constexpr in host code;
// it will take the host-pass fallback path and cause a compile error.
inline constexpr WarpSizeValue WARP_SIZE{};

static int get_pci_chip_id()
{
    static const int chip_id = []() {
        hipDevice_t dev;
        int id = 0;
        HIP_CALL(hipGetDevice(&dev));
        HIP_CALL(hipDeviceGetAttribute(&id, hipDeviceAttributePciChipId, dev));
        AITER_LOG_INFO("pciChipId: 0x" << std::hex << id << std::dec
                                       << ", CU count: " << get_num_cu_func());
        return id;
    }();
    return chip_id;
}

static bool is_mi308_device()
{
    int chip_id = get_pci_chip_id();
    return chip_id == 0x74a2 || chip_id == 0x74a8 || chip_id == 0x74b6 || chip_id == 0x74bc;
}

class HipDeviceGuard
{
    public:
    explicit HipDeviceGuard(int device_id)
    {
        HIP_CALL(hipGetDevice(&prev_device_));
        HIP_CALL(hipSetDevice(device_id));
    }
    ~HipDeviceGuard() noexcept { HIP_CALL(hipSetDevice(prev_device_)); }
    HipDeviceGuard(const HipDeviceGuard&)            = delete;
    HipDeviceGuard& operator=(const HipDeviceGuard&) = delete;

    private:
    int prev_device_{};
};

template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
struct SynchronizedCache
{
    template <typename K, typename F>
    inline T& get_or_create(K&& k, F&& factory)
    {
        std::lock_guard<std::mutex> map_mu_guard(map_mu);

        struct Wrapper
        {
            F& f;
            // Makes sure we only invoke lambda on insert
            operator T() && { return f(); }
        };

        auto [it, _] = map.try_emplace(std::forward<K>(k), Wrapper{factory});
        return it->second;
    }

    private:
    std::mutex map_mu;
    std::unordered_map<Key, T, Hash, KeyEqual> map;
};

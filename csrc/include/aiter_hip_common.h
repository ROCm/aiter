// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_logger.h"
#include "ck_tile/core.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <memory>
#ifdef AITER_EMBEDDED_HSA_HEADER
#include AITER_EMBEDDED_HSA_HEADER
#endif

enum class GPUArch
{
    gfx942,
    gfx950
};

#define CHECK_COND(x)                                                                             \
    do                                                                                            \
    {                                                                                             \
        if(!(x))                                                                                  \
        {                                                                                         \
            std::cerr << "check failed, file=" << __FILE__ << ", line=" << __LINE__ << std::endl; \
            std::terminate();                                                                     \
        }                                                                                         \
    } while(0)

#define HIP_CALL(call)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t err = call;                                               \
        if(err != hipSuccess)                                                \
        {                                                                    \
            printf("\n[AITER] %s:%d fail to call %s ---> [HIP error](%s)\n", \
                   __FILE__,                                                 \
                   __LINE__,                                                 \
                   #call,                                                    \
                   hipGetErrorString(err));                                  \
            exit(0);                                                         \
        }                                                                    \
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

namespace detail {
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
} // namespace detail


namespace {

class AiterAsmKernelFast
{
    private:
    void* module = nullptr;

    protected:
    AiterAsmKernelFast() = default;
    void init(const char* kernel_name, const void* hsaco)
    {
        detail::FatBinaryWrapper fat_bin{};
        fat_bin.binary = hsaco;
        module         = detail::__hipRegisterFatBinary(&fat_bin);
        CHECK_COND(module != nullptr);
        detail::__hipRegisterFunction(module,
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

    ~AiterAsmKernelFast() { detail::__hipUnregisterFatBinary(module); }

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

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
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

    const void* load_hsaco_file(const char* hsaco_path)
    {
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::string arch_name     = get_gpu_arch();
        if(AITER_ASM_DIR != nullptr)
        {
            std::string full_path = std::string(AITER_ASM_DIR) + "/" + arch_name + "/" + hsaco_path;

            std::ifstream file(full_path, std::ios::binary | std::ios::ate);

            CHECK_COND(file.is_open());

            size_t file_size = file.tellg();
            hsaco_data.reset(new char[file_size]);

            file.seekg(0, std::ios::beg);
            CHECK_COND(file.read(hsaco_data.get(), file_size));
            return hsaco_data.get();
        }
        else
        {
#if defined(AITER_EMBEDDED_HSA_HEADER) && defined(AITER_EMBEDDED_HSA_MAP)
            std::string fname = "hsa/" + arch_name + "/" + hsaco;
            auto hasco_obj    = AITER_EMBEDDED_HSA_MAP.find(fname);
            CHECK_COND(hasco_obj != AITER_EMBEDDED_HSA_MAP.end());
            CHECK_COND(hasco_obj->second.data() != nullptr);
            return hasco_obj->second.data();
#else
            CHECK_COND(AITER_ASM_DIR != nullptr);
            return nullptr;
#endif
        }
    }

    public:
    AiterAsmKernel(const char* kernel_name, const char* hsaco_path)
    {
        init(kernel_name, load_hsaco_file(hsaco_path));
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

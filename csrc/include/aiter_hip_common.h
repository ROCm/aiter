// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_logger.h"
#include "ck_tile/core.hpp"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <iostream>
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

// ASM kernel loading requires embedded HSA blobs (provided by QoLA at
// build time via -DAITER_EMBEDDED_HSA_HEADER).  Consumers that only need
// type definitions (e.g. aiter::mha_fwd_args) can include this header
// without the embedded blobs.
#ifdef AITER_EMBEDDED_HSA_HEADER

inline void load_asm_kernel(const char* name,
                            const char* hsaco,
                            hipModule_t& module,
                            hipFunction_t& kernel_func)
{
#if !defined(AITER_EMBEDDED_HSA_MAP)
#error "AITER_EMBEDDED_HSA_HEADER is defined but AITER_EMBEDDED_HSA_MAP is missing."
#endif
    std::string arch_name = get_gpu_arch();
    std::string fname = "hsa/" + arch_name + "/" + hsaco;
    auto it = AITER_EMBEDDED_HSA_MAP.find(fname);
    CHECK_COND(it != AITER_EMBEDDED_HSA_MAP.end());
    CHECK_COND(it->second.data() != nullptr);
    AITER_LOG_INFO("hipModuleLoad (embedded): " << fname << " GetFunction: " << name);
    HIP_CALL(hipModuleLoadData(&module, it->second.data()));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    AITER_LOG_INFO("hipModuleGetFunction: " << name << " Success");
}

class AiterAsmKernel
{
    private:
    hipModule_t module;
    hipFunction_t kernel_func;

    public:
    AiterAsmKernel(const char* name, const char* hsaco)
    {
        load_asm_kernel(name, hsaco, module, kernel_func);
    };

    ~AiterAsmKernel() { HIP_CALL(hipModuleUnload(module)); }

    void launch_kernel(const AiterAsmKernelArgs& kargs)
    {
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

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

class AiterAsmKernelFast
{
    private:
    hipModule_t module;
    hipFunction_t kernel_func;

    public:
    AiterAsmKernelFast(const char* name, void* hsaco)
    {
        HIP_CALL(hipModuleLoadData(&module, hsaco));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        AITER_LOG_INFO("hipModuleGetFunction: " << name << " Success");
    };

    ~AiterAsmKernelFast() { HIP_CALL(hipModuleUnload(module)); }

    void launch_kernel(const AiterAsmKernelArgs& kargs)
    {
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

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

#endif // AITER_EMBEDDED_HSA_HEADER

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

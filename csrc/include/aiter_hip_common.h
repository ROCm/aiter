// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
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
    void* arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

static const std::string get_gpu_arch();

inline void load_asm_kernel(const char* name,
                            const char* hsaco,
                            hipModule_t& module,
                            hipFunction_t& kernel_func)
{
    const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
    std::string arch_name     = get_gpu_arch();
    if(AITER_ASM_DIR != nullptr)
    {
        std::string hsa_path = std::string(AITER_ASM_DIR) + "/" + arch_name + "/" + hsaco;
        std::cout << "[aiter] hipModuleLoad: " << hsa_path << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, hsa_path.c_str()));
    }
    else
    {
#if defined(AITER_EMBEDDED_HSA_HEADER) && defined(AITER_EMBEDDED_HSA_MAP)
        std::string fname = "hsa/" + arch_name + "/" + hsaco;
        auto hasco_obj    = AITER_EMBEDDED_HSA_MAP.find(fname);
        CHECK_COND(hasco_obj != AITER_EMBEDDED_HSA_MAP.end());
        CHECK_COND(hasco_obj->second.data() != nullptr);
        std::cout << "hipModuleLoad: " << fname << " GetFunction: " << name << std::endl;
        HIP_CALL(hipModuleLoadData(&module, hasco_obj->second.data()));
#endif
    }
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
    std::cout << " Success" << std::endl;
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
        std::cout << " Success" << std::endl;
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

// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include <dlfcn.h>   // For dladdr
#include <filesystem>

enum class GPUArch
{
    gfx942,
    gfx950
};


#define CHECK_COND(x) \
    do { \
        if (!(x)) { \
            std::cerr << "check failed, file=" \
                << __FILE__ << ", line=" \
                << __LINE__ << std::endl; \
            std::terminate(); \
        } \
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
// Helper macro to perform the actual stringization
#define STRINGIZE_HELPER(x) #x

// Macro that expands its argument before passing it to the helper
#define STRINGIZE(x) STRINGIZE_HELPER(x)

class AiterAsmKernel
{
    private:
    hipModule_t module;
    hipFunction_t kernel_func;

    public:
    AiterAsmKernel(const char* name, const char* hsaco)
    {
#if USE_AITER_ASM_DIR == 0
        //extract gpu arch
        std::string arch_id = get_gpu_arch();
        //extract current .so location
        void* func_ptr = (void*)&AiterAsmKernel::staticMethod;
        Dl_info info;
        std::filesystem::path aiter_asm_dir;
        if (dladdr(func_ptr, &info)){
          aiter_asm_dir = std::filesystem::path(info.dli_fname).parent_path() / STRINGIZE(REL_PATH_LIB_TO_HSA)/ "hsa" / arch_id.c_str();
        }else{
          std::cerr<<"Failed the dladdr when trying to find aiter lib*.so"<<std::endl;
        }
        std::cout << "[aiter] hipModuleLoad: " << (aiter_asm_dir/hsaco).c_str()
                  << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (aiter_asm_dir/hsaco).c_str()));
#else
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::cout << "[aiter] hipModuleLoad: " << (std::string(AITER_ASM_DIR) + hsaco).c_str()
                  << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + hsaco).c_str()));
#endif
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        std::cout << " Success" << std::endl;
    };

#if USE_AITER_ASM_DIR == 0
    static void staticMethod(){
      std::cout<<"Help to find out abspath of aiter lib*.so with AiterAsmKernel instantiated"<<std::endl;
    }
#endif

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

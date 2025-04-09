// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CALL(call)                                                                                                           \
    do                                                                                                                           \
    {                                                                                                                            \
        hipError_t err = call;                                                                                                   \
        if (err != hipSuccess)                                                                                                   \
        {                                                                                                                        \
            printf("\n[AITER] %s:%d fail to call %s ---> [HIP error](%s)\n", __FILE__, __LINE__, #call, hipGetErrorString(err)); \
            exit(0);                                                                                                             \
        }                                                                                                                        \
    } while (0)

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
    void *args_ptr;
    void *arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

class AiterAsmKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;

public:
    AiterAsmKernel(const char *name, const char *hsaco)
    {
        const char *AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::cout << "hipModuleLoad: " << (std::string(AITER_ASM_DIR) + hsaco).c_str() << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        std::cout << " Success" << std::endl;
    };

    ~AiterAsmKernel()
    {
        HIP_CALL(hipModuleUnload(module));
    }

    void launch_kernel(const AiterAsmKernelArgs &kargs)
    {
        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE, kargs.arg_size_ptr,
                          HIP_LAUNCH_PARAM_END};

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       kargs.gdx, kargs.gdy, kargs.gdz,
                                       kargs.bdx, kargs.bdy, kargs.bdz,
                                       0, kargs.stream, nullptr, (void **)&config));
    };
};


template <typename T, typename F>
__device__ constexpr T wave_reduce(T local, F reduce_f)
{
  constexpr int reduce_stage = 6; // 1<<6=64
  T v_local = local;
#pragma unroll
  for (int i_stage = 0; i_stage < reduce_stage; i_stage++)
  {
    int src_lane = __lane_id() ^ (1 << i_stage);
    int32_t v_remote_tmp =
        __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, v_local));
    T v_remote = __builtin_bit_cast(T, v_remote_tmp);
    v_local = reduce_f(v_local, v_remote);
  }
  return v_local;
}


template <typename T, typename F>
__device__ constexpr T cross_wave_reduce(T local, F reduce_f, T* smem)
{
    int blockSize = blockDim.x;
    int waves     = blockDim.x / WARP_SIZE;
    int wave_size = WARP_SIZE;
    int lane_id   = threadIdx.x % wave_size;
    
    __syncthreads();
    smem[threadIdx.x] = local;
    __syncthreads();

    // the data within single wave is the same
    // but for simplicity, we still use data from each lane.
    T v_local = smem[lane_id];
#pragma unroll
    for(int i_stage = 1; i_stage < waves; i_stage++)
    {
        T v_remote = smem[i_stage * wave_size + lane_id];
        v_local    = reduce_f(v_local, v_remote);
    } 
    return v_local;
}
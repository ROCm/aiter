#pragma once

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <type_traits>

#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        hipError_t e = cmd;                                                                    \
        if(e != hipSuccess)                                                                    \
        {                                                                                      \
            printf("Failed: HIP error %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
            throw std::runtime_error("HIP error");                                             \
        }                                                                                      \
    } while(0)


constexpr int BLOCK_SIZE = 1024;
constexpr int UNROLL_FACTOR = 8; 
constexpr int OCCUPANCY_PER_CU = 16; 
constexpr int WARMUP = 25;
constexpr int LOOP = 100;


enum class HFMemOp { 
    GlobalLoad,      
    GlobalLoadNT,   
    GlobalStore,    
    GlobalStoreNT,  
    BufferStore,
    BufferLoadLDS,
    BufferLoad,
    DsRead,
    DsWrite
};


struct BenchmarkResult {
    std::string operation;   
    std::string vector_width; 
    double size_mb;           
    double bandwidth_gb_s;  
};


typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef u32x4 dwordx4_t;
#define BUFFER_CONFIG 0x00020000
struct buffer_resource { const void * ptr; uint32_t range; uint32_t config; };


template <typename T> __device__ __forceinline__ float consume(const T&);

template <typename VectorType, HFMemOp Op>
BenchmarkResult run_benchmark_return_result(const std::string& operation_name, int num_cu, int64_t dwords);

void write_results_to_file(const std::vector<BenchmarkResult>& results, const std::string& filename,
                         const std::string& title, const hipDeviceProp_t& props, int num_cu);

__device__ dwordx4_t make_buffer_resource(const void * ptr, uint32_t size) {
    buffer_resource res {ptr, size, BUFFER_CONFIG};
    return __builtin_bit_cast(dwordx4_t, res);
}

__device__ __forceinline__ void do_global_load(float2& reg, const float2* addr) { 
    asm volatile("global_load_dwordx2 %0, %1, off" : "=v"(reg) : "v"(addr) : "memory"); 
}

__device__ __forceinline__ void do_global_load(float4& reg, const float4* addr) { 
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(reg) : "v"(addr) : "memory"); 
}

__device__ __forceinline__ void do_global_load_nt(float2& reg, const float2* addr) { 
    asm volatile("global_load_dwordx2 %0, %1, off nt" : "=v"(reg) : "v"(addr) : "memory"); 
}

__device__ __forceinline__ void do_global_load_nt(float4& reg, const float4* addr) { 
    asm volatile("global_load_dwordx4 %0, %1, off nt" : "=v"(reg) : "v"(addr) : "memory"); 
}

__device__ __forceinline__ void do_global_store(float2* addr, float2 reg) { 
    asm volatile("global_store_dwordx2 %0, %1, off" : : "v"(addr), "v"(reg) : "memory"); 
}

// TODO:x4 store instruction not supported yet
__device__ __forceinline__ void do_global_store(float4* addr, float4 reg) {
    asm volatile("global_store_dwordx2 %0, %1, off" : : "v"((float2*)addr), "v"(*(float2*)&reg) : "memory");
    asm volatile("global_store_dwordx2 %0, %1, off" : : "v"(((float2*)addr) + 1), "v"(*((float2*)&reg + 1)) : "memory");
    // asm volatile("global_store_dwordx4 %0, %1, off" : : "v"(addr), "v"(reg) : "memory");
}

__device__ __forceinline__ void do_global_store_nt(float2* addr, float2 reg) { 
    asm volatile("global_store_dwordx2 %0, %1, off nt" : : "v"(addr), "v"(reg) : "memory"); 
}

// TODO:x4 store instruction not supported yet
__device__ __forceinline__ void do_global_store_nt(float4* addr, float4 reg) {
    asm volatile("global_store_dwordx2 %0, %1, off nt" : : "v"((float2*)addr), "v"(*(float2*)&reg) : "memory");
    asm volatile("global_store_dwordx2 %0, %1, off nt" : : "v"(((float2*)addr) + 1), "v"(*((float2*)&reg + 1)) : "memory");
}


__device__ void buffer_load(float2& reg, dwordx4_t res, uint32_t v_offset, uint32_t s_offset, uint32_t i_offset = 0) {
    asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4" 
        : "=v"(reg) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

__device__ void buffer_load(float4& reg, dwordx4_t res, uint32_t v_offset, uint32_t s_offset, uint32_t i_offset = 0) {
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4" 
        : "=v"(reg) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}


__device__ void buffer_store(const float2& vdata, dwordx4_t res, uint32_t v_offset, uint32_t s_offset, uint32_t i_offset = 0) {
    asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4" 
                : : "v"(vdata), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

// TODO:x4 store instruction not supported yet
__device__ void buffer_store(const float4& vdata, dwordx4_t res, uint32_t v_offset, uint32_t s_offset, uint32_t i_offset = 0) {
    asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4" 
                : : "v"(*(reinterpret_cast<const float2*>(&vdata))), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
    asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4" 
                : : "v"(*(reinterpret_cast<const float2*>(&vdata) + 1)), "v"(v_offset + 8), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

// only supports dword
__device__ void buffer_load_lds(void * smem, dwordx4_t res , uint32_t v_offset, uint32_t s_offset, uint32_t i_offset , uint32_t = 0){
    asm volatile("buffer_load_dword %1, %2, %3 offen offset:%4 lds"
        : "=r"(smem) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}


__device__ __forceinline__ void do_lds_read(float& reg, unsigned int offset)  { asm volatile("ds_read_b32 %0, %1" : "=v"(reg) : "v"(offset) : "memory"); }
__device__ __forceinline__ void do_lds_read(float2& reg, unsigned int offset) { asm volatile("ds_read_b64 %0, %1" : "=v"(reg) : "v"(offset) : "memory"); }
__device__ __forceinline__ void do_lds_read(float4& reg, unsigned int offset){ asm volatile("ds_read_b128 %0, %1" : "=v"(reg) : "v"(offset) : "memory"); }

__device__ __forceinline__ void do_lds_write(unsigned int offset, float reg)  { asm volatile("ds_write_b32 %0, %1" : : "v"(offset), "v"(reg) : "memory"); }
__device__ __forceinline__ void do_lds_write(unsigned int offset, float2 reg) { asm volatile("ds_write_b64 %0, %1" : : "v"(offset), "v"(reg) : "memory"); }



// TODO:x4 store instruction not supported yet
__device__ __forceinline__ void do_lds_write(unsigned int offset, float4 reg){
    asm volatile("ds_write_b64 %0, %1" : : "v"(offset), "v"(*(float2*)&reg) : "memory");
    asm volatile("ds_write_b64 %0, %1" : : "v"(offset + 8), "v"(*((float2*)&reg + 1)) : "memory");
}

template <> __device__ __forceinline__ float consume<float2>(const float2& v) { return v.x + v.y; }
template <> __device__ __forceinline__ float consume<float4>(const float4& v) { return v.x + v.y + v.z + v.w; }


// global_load_dword
template <typename T>
__global__ void global_load_kernel(const T* in_data, int num_elements_per_block, int iters, float* g_sum)
{
    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;
    
    T temp_reg{};
    float local_sum = 0.f;

    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + threadIdx.x;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            size_t aligned_offset = (block_base_offset + offs) * sizeof(T) / sizeof(T);
            do_global_load(temp_reg, &in_data[aligned_offset]);
            local_sum += consume(temp_reg);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }

    if (local_sum > 99999999.f) {
        g_sum[0] = local_sum;
    }
}

// global_load_dword  nt
template <typename T>
__global__ void global_load_nt_kernel(const T* in_data, int num_elements_per_block, int iters, float* g_sum)
{
    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;
    
    T temp_reg{};
    float local_sum = 0.f;

    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + threadIdx.x;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            size_t aligned_offset = (block_base_offset + offs) * sizeof(T) / sizeof(T);
            do_global_load_nt(temp_reg, &in_data[aligned_offset]);
            local_sum += consume(temp_reg);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }

    if (local_sum > 99999999.f) {
        g_sum[0] = local_sum;
    }
}

// global_store_dword
template <typename T>
__global__ void global_store_kernel(T* out_data, int num_elements_per_block, int iters)
{
    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;
    T reg{};
    reg.x = 1.23f;
    if constexpr (sizeof(T) > 4)  reg.y = 2.34f;
    if constexpr (sizeof(T) > 8)  reg.z = 3.45f;
    if constexpr (sizeof(T) > 12) reg.w = 4.56f;

    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + threadIdx.x;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            size_t aligned_offset = (block_base_offset + offs) * sizeof(T) / sizeof(T);
            do_global_store(&out_data[aligned_offset], reg);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }
}

// global_store_dword nt
template <typename T>
__global__ void global_store_nt_kernel(T* out_data, int num_elements_per_block, int iters)
{
    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;

    T reg{};
    reg.x = 1.23f;
    if constexpr (sizeof(T) > 4)  reg.y = 2.34f;
    if constexpr (sizeof(T) > 8)  reg.z = 3.45f;
    if constexpr (sizeof(T) > 12) reg.w = 4.56f;

    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + threadIdx.x;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
   
            size_t aligned_offset = (block_base_offset + offs) * sizeof(T) / sizeof(T);
            do_global_store_nt(&out_data[aligned_offset], reg);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }
}


// buffer load 
template <typename T>
__global__ void buffer_load_reg_kernel(const T* in_data, int num_elements_per_block, int iters, float* g_sum)
{
    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;
    
    T temp_reg{};
    float local_sum = 0.f;
    
    dwordx4_t src_res = make_buffer_resource(in_data, 0xffffffff);

    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + threadIdx.x;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {    
            buffer_load(temp_reg, src_res, offs, block_base_offset, 0);                   
            local_sum += consume(temp_reg);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }

    if (local_sum > 99999999.f) {
        g_sum[0] = local_sum;
    }
}

// reg -> buffer, buffer_store_dword
template <typename T>
__global__ void buffer_store_kernel(T* out_data, int num_elements_per_block, int iters)
{
    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;

    T reg{};
    reg.x = 1.23f;
    if constexpr (sizeof(T) > 4)  reg.y = 2.34f;
    if constexpr (sizeof(T) > 8)  reg.z = 3.45f;
    if constexpr (sizeof(T) > 12) reg.w = 4.56f;
    
    dwordx4_t dst_res = make_buffer_resource(out_data, 0xffffffff);

    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + threadIdx.x;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            const size_t idx = block_base_offset + offs;
            uint32_t v_offset = (uint32_t)(idx * sizeof(T));
            buffer_store(reg, dst_res, v_offset, 0, 0);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }
}


template <typename T>
__global__ void buffer_load_lds_kernel(const T* in_data, int num_elements_per_block, int iters)
{

    extern __shared__ char lds_raw_data[];

    const size_t block_base_offset = (size_t)blockIdx.x * num_elements_per_block;
    uint32_t wave_id_in_block = threadIdx.x / warpSize;
    uint32_t wave_per_block = blockDim.x / warpSize;
    uint32_t num_elememts_per_wave = num_elements_per_block / wave_per_block;
    uint32_t wave_lds_base = __builtin_amdgcn_readfirstlane(num_elememts_per_wave * sizeof(T));

    dwordx4_t buffer_res = make_buffer_resource(in_data, 0xffffffff);

    asm volatile("s_mov_b32 m0, %0" : : "s"(wave_lds_base));
    uint32_t wave_base_offset = wave_id_in_block * num_elememts_per_wave;
    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * warpSize * i + threadIdx.x % warpSize;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            uint32_t v_offset =  wave_base_offset + offs;
            buffer_load_lds(lds_raw_data, buffer_res, v_offset, block_base_offset, 0);
            offs += warpSize;
        }
        asm volatile("s_waitcnt vmcnt(0)");
    }
}


template <typename T>
__global__ void lds_load_kernel(const T* in_data, int iters, float* g_sum)
{
    extern __shared__ char lds_raw_data[];
    T* lds_data = reinterpret_cast<T*>(lds_raw_data);
    
    const size_t tid = threadIdx.x;
 
    T temp_reg{};
    float local_sum = 0.f;
    
    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + tid;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            uint32_t addr_lo = offs & 0xFFFF;
            do_lds_read(temp_reg, addr_lo);    
            local_sum += consume(temp_reg);
            offs += blockDim.x;
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
    }
    
    if (local_sum > 99999999.f) {
        g_sum[0] = local_sum;
    }
}

template <typename T>
__global__ void lds_write_kernel(T* out_data,  int iters)
{
    extern __shared__ char lds_raw_data[];
    T* lds_data = reinterpret_cast<T*>(lds_raw_data);

    T reg{};
    reg.x = 1.23f;
    if constexpr (sizeof(T) > 4)  reg.y = 2.34f;
    if constexpr (sizeof(T) > 8)  reg.z = 3.45f;
    if constexpr (sizeof(T) > 12) reg.w = 4.56f;
    
    const size_t tid = threadIdx.x;
    
    for (int i = 0; i < iters; ++i)
    {
        size_t offs = UNROLL_FACTOR * blockDim.x * i + tid;
        
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            uint32_t addr_lo = offs & 0xFFFF;
            do_lds_write(addr_lo, reg);
            offs += blockDim.x;
        }

        asm volatile("s_waitcnt lgkmcnt(0)");
    } 
    const size_t global_idx = blockIdx.x * blockDim.x + tid;
    // prevent it from being optimalized by compiler
    out_data[global_idx] = lds_data[tid];
}


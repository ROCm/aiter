// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <initializer_list>
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "iq2r.h"
#include "mx_quant_utils.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace aiter {
namespace {

constexpr int kThreads     = 256;
constexpr int kTaskColumns = 3;
constexpr int kQuantGroup  = 32;

// GLM-5.3 IQ2R packed MoE routing.
// GLM-5.3 routing: 256 routed experts plus the fused shared expert 256, top-9.
constexpr int kGlm53Experts = 257;
constexpr int kGlm53TopK    = 9;
constexpr int kGlm53Hidden  = 6144;
constexpr int kGlm53Groups  = kGlm53Hidden / kQuantGroup;

// Prefill route sort in three passes. Each 256-route chunk owns its histogram
// and ranks; the prefix passes assign disjoint expert ranges before the scatter.
__global__ __launch_bounds__(256) void glm53_sort_histogram(
    const int32_t* ids, int32_t* local_ranks, int32_t* histogram, int routes)
{
    __shared__ int counts[512];
    const int t=threadIdx.x;
    counts[t]=counts[t+256]=0;
    __syncthreads();
    const int route=blockIdx.x*256+t;
    if(route<routes)
    {
        const int id=ids[route];
        const int bucket=(id>=0 && id<257)?id:257;
        local_ranks[route]=atomicAdd(counts+bucket,1);
    }
    __syncthreads();
    histogram[blockIdx.x*512+t]=counts[t];
    histogram[blockIdx.x*512+t+256]=counts[t+256];
}

// Inclusive block scan; waves holds one total per wave of the block.
__device__ __forceinline__ int glm53_block_scan(int x,int* waves)
{
    const int lane=threadIdx.x%64,wave=threadIdx.x/64;
#pragma unroll
    for(int delta=1;delta<64;delta*=2)
    {
        const int y=__shfl_up(x,delta);
        if(lane>=delta)x+=y;
    }
    if(lane==63)waves[wave]=x;
    __syncthreads();
    int prior=0;
    for(int w=0;w<wave;++w)prior+=waves[w];
    __syncthreads();
    return x+prior;
}

__global__ __launch_bounds__(512) void glm53_sort_chunk_prefix(int32_t* histogram,int chunks)
{
    __shared__ int waves[8];
    const int expert=blockIdx.x,chunk=threadIdx.x;
    const int n=chunk<chunks?histogram[chunk*512+expert]:0;
    const int prefix=glm53_block_scan(n,waves);
    if(chunk<chunks)histogram[chunk*512+expert]=prefix-n;
    if(chunk==511)histogram[chunks*512+expert]=prefix;
}

__global__ __launch_bounds__(512) void glm53_sort_expert_prefix(
    int32_t* histogram,int32_t* tasks,int32_t* task_count,
    int chunks,int task_rows,int task_capacity)
{
    __shared__ int waves[8];
    const int expert=threadIdx.x;
    const int count=expert<258?histogram[chunks*512+expert]:0;
    const int begin=glm53_block_scan(count,waves)-count;
    histogram[chunks*512+512+expert]=begin;
    const int nt=(count+task_rows-1)/task_rows;
    const int task_end=glm53_block_scan(nt,waves);
    if(expert==511)task_count[0]=task_end<=task_capacity?task_end:-1;
    for(int i=0;i<nt;++i)
    {
        const int task=task_end-nt+i;
        if(task<task_capacity)
        {
            tasks[task*3]=begin+i*task_rows;
            tasks[task*3+1]=min(task_rows,count-i*task_rows);
            tasks[task*3+2]=expert<257?expert:-1;
        }
    }
}

__global__ __launch_bounds__(256) void glm53_sort_scatter(
    const int32_t* ids,int32_t* gather,int32_t* scatter,
    const int32_t* histogram,int routes)
{
    const int route=blockIdx.x*256+threadIdx.x;
    if(route<routes)
    {
        const int id=ids[route];
        const int bucket=(id>=0 && id<257)?id:257;
        const int rank=scatter[route]+histogram[blockIdx.x*512+bucket]+histogram[((routes+255)/256)*512+512+bucket];
        gather[rank]=route;
        scatter[route]=rank;
    }
}

using glm53_fp8_group = opus::vector_t<opus::fp8_t, kQuantGroup>;

// Quantizes 32 BF16 values to FP8 with one E8M0 block scale; returns the scale.
__device__ __forceinline__ uint8_t glm53_quant32(const opus::bf16_t* __restrict__ input,
                                                 glm53_fp8_group& quantized)
{
    using input_vector        = opus::vector_t<opus::bf16_t, kQuantGroup>;
    const input_vector values = *reinterpret_cast<const input_vector*>(input);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    union { glm53_fp8_group quantized; uint32_t words[8]; } packed;
#pragma unroll
    for(int word=0;word<8;++word) {
        const float x0=static_cast<float>(values[word*4])*inverse_scale;
        const float x1=static_cast<float>(values[word*4+1])*inverse_scale;
        const float x2=static_cast<float>(values[word*4+2])*inverse_scale;
        const float x3=static_cast<float>(values[word*4+3])*inverse_scale;
        int bits=__builtin_amdgcn_cvt_pk_fp8_f32(x0,x1,0,0);
        bits=__builtin_amdgcn_cvt_pk_fp8_f32(x2,x3,bits,1);
        packed.words[word]=static_cast<uint32_t>(bits);
    }
    quantized = packed.quantized;
    return block_scale.byte;
}

// Quantizes one 32-column group of a token row (group_id = token * 192 + group).
__device__ __forceinline__ void glm53_quant_group(
    const opus::bf16_t* __restrict__ input,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int tokens,
    int input_stride,
    int group_id)
{
    if(group_id >= tokens * kGlm53Groups)
        return;
    const int token = group_id / kGlm53Groups;
    const int group = group_id % kGlm53Groups;
    glm53_fp8_group quantized;
    const uint8_t scale =
        glm53_quant32(input + static_cast<int64_t>(token) * input_stride + group * kQuantGroup, quantized);
    *reinterpret_cast<glm53_fp8_group*>(output + static_cast<int64_t>(token) * kGlm53Hidden +
                                        group * kQuantGroup) = quantized;
    scales[static_cast<int64_t>(token) * kGlm53Groups + group] = scale;
}

__global__ __launch_bounds__(256) void glm53_quant_kernel(
    const opus::bf16_t* input,opus::fp8_t* output,uint8_t* scales,int tokens,int stride)
{
    glm53_quant_group(input,output,scales,tokens,stride,
        static_cast<int>(blockIdx.x)*256+static_cast<int>(threadIdx.x));
}

// M1 front end: the nine routes of one token become nine one-row tasks in
// route order (sorting them costs more than it saves). The token is quantized
// once and its row copied to all nine routes.
__global__ __launch_bounds__(256) void glm53_m1_route_quant_kernel(
    const opus::bf16_t* input,const int32_t* ids,int32_t* scatter,int32_t* tasks,
    int32_t* count,opus::fp8_t* output,uint8_t* scales)
{
    const int t=threadIdx.x;
    if(t<kGlm53TopK)
    {
        const int id=ids[t];
        scatter[t]=t;
        tasks[t*kTaskColumns]=t;
        tasks[t*kTaskColumns+1]=1;
        tasks[t*kTaskColumns+2]=id>=0 && id<kGlm53Experts?id:-1;
        if(t==0)count[0]=kGlm53TopK;
    }
    if(t>=kGlm53Groups)return;
    glm53_fp8_group quantized;
    const uint8_t scale=glm53_quant32(input+t*kQuantGroup,quantized);
#pragma unroll
    for(int route=0;route<kGlm53TopK;++route)
    {
        *reinterpret_cast<glm53_fp8_group*>(output+route*kGlm53Hidden+t*kQuantGroup)=quantized;
        scales[route*kGlm53Groups+t]=scale;
    }
}

// Decode route sort in one CTA: one thread per routed expert, wave scans for
// offsets, and two task lists over the same sorted routes: M32 tasks for the
// down GEMM and M16 tasks for the gate GEMM. Expert 256 (shared) and invalid
// routes are appended after the routed experts.
__device__ __forceinline__ void glm53_sort_routes(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    int routes,
    int task_capacity,
    int32_t* __restrict__ gate_tasks,
    int32_t* __restrict__ gate_count,
    int gate_capacity)
{
    constexpr int kRouted = kGlm53Experts - 1;
    constexpr int task_rows = 32;
    __shared__ int counts[kRouted + 2];
    __shared__ int offsets[kRouted + 2];
    __shared__ int cursors[kRouted + 2];
    __shared__ int wave_totals[4];
    __shared__ int valid_task_count;

    const int expert = static_cast<int>(threadIdx.x);
    counts[expert]   = 0;
    if(expert == 0)
    {
        counts[kRouted]     = 0;
        counts[kRouted + 1] = 0;
    }
    __syncthreads();

    auto local_expert = [](int id) { return id >= 0 && id < kGlm53Experts ? id : -1; };
    for(int route = expert; route < routes; route += kThreads)
    {
        const int local = local_expert(expert_ids[route]);
        atomicAdd(counts + (local >= 0 ? local : kGlm53Experts), 1);
    }
    __syncthreads();

    const int count_prefix = glm53_block_scan(counts[expert], wave_totals);
    offsets[expert]        = count_prefix - counts[expert];
    cursors[expert]        = offsets[expert];
    if(expert == kRouted - 1)
    {
        offsets[kRouted] = count_prefix;
        cursors[kRouted] = count_prefix;
    }
    __syncthreads();

    if(expert == 0)
    {
        offsets[kRouted + 1] = offsets[kRouted] + counts[kRouted];
        cursors[kRouted + 1] = offsets[kRouted + 1];
    }
    __syncthreads();

    for(int route = expert; route < routes; route += kThreads)
    {
        const int local                 = local_expert(expert_ids[route]);
        const int bucket                = local >= 0 ? local : kGlm53Experts;
        const int sorted_route       = atomicAdd(cursors + bucket, 1);
        gather_indices[sorted_route] = route;
        scatter_indices[route]       = sorted_route;
    }
    __syncthreads();

    const int local_tasks = (counts[expert] + task_rows - 1) / task_rows;
    // Each task prefix fits in 16 bits: routes <= 9216 give at most 834 tasks. Summing the packed pair
    // cannot carry from the low field to the high.
    const int local_gate_tasks=(counts[expert]+15)/16;
    const int packed_prefix=glm53_block_scan(local_tasks+(local_gate_tasks<<16),wave_totals);
    const int task_prefix=packed_prefix&65535;
    const int gate_begin=(packed_prefix>>16)-local_gate_tasks;
    const int task_begin  = task_prefix - local_tasks;
    int task              = task_begin;
    for(int local = 0; local < counts[expert]; local += task_rows, ++task)
    {
        if(task < task_capacity)
        {
            tasks[task * kTaskColumns]     = offsets[expert] + local;
            tasks[task * kTaskColumns + 1] = min(task_rows, counts[expert] - local);
            tasks[task * kTaskColumns + 2] = expert;
        }
    }
    for(int local=0,t=gate_begin;local<counts[expert];local+=16,++t)
    {
        if(t<gate_capacity) {
            gate_tasks[t*3]=offsets[expert]+local;
            gate_tasks[t*3+1]=min(16,counts[expert]-local);
            gate_tasks[t*3+2]=expert;
        }
    }
    if(expert == kRouted - 1)
        valid_task_count=packed_prefix;
    __syncthreads();

    const int routed_tasks=valid_task_count&65535;
    const int routed_gate_tasks=valid_task_count>>16;
    const int shared_count=counts[kRouted];
    const int invalid_count=counts[kGlm53Experts];
    const int shared_tasks=(shared_count+task_rows-1)/task_rows;
    const int shared_gate_tasks=(shared_count+15)/16;
    const int invalid_tasks=(invalid_count+task_rows-1)/task_rows;
    const int invalid_gate_tasks=(invalid_count+15)/16;
    for(int part=expert;part<shared_tasks;part+=kThreads) {
        const int t=routed_tasks+part,local=part*task_rows;
        if(t<task_capacity) {
            tasks[t*3]=offsets[kRouted]+local;
            tasks[t*3+1]=min(task_rows,shared_count-local);
            tasks[t*3+2]=kRouted;
        }
    }
    for(int part=expert;part<shared_gate_tasks;part+=kThreads) {
        const int t=routed_gate_tasks+part,local=part*16;
        if(t<gate_capacity) {
            gate_tasks[t*3]=offsets[kRouted]+local;
            gate_tasks[t*3+1]=min(16,shared_count-local);
            gate_tasks[t*3+2]=kRouted;
        }
    }
    for(int part=expert;part<invalid_tasks;part+=kThreads) {
        const int t=routed_tasks+shared_tasks+part,local=part*task_rows;
        if(t<task_capacity) {
            tasks[t*3]=offsets[kGlm53Experts]+local;
            tasks[t*3+1]=min(task_rows,invalid_count-local);
            tasks[t*3+2]=-1;
        }
    }
    for(int part=expert;part<invalid_gate_tasks;part+=kThreads) {
        const int t=routed_gate_tasks+shared_gate_tasks+part,local=part*16;
        if(t<gate_capacity) {
            gate_tasks[t*3]=offsets[kGlm53Experts]+local;
            gate_tasks[t*3+1]=min(16,invalid_count-local);
            gate_tasks[t*3+2]=-1;
        }
    }
    if(expert==0) {
        const int total=routed_tasks+shared_tasks+invalid_tasks;
        const int gate_total=routed_gate_tasks+shared_gate_tasks+invalid_gate_tasks;
        task_count[0]=total<=task_capacity?total:-1;
        gate_count[0]=gate_total<=gate_capacity?gate_total:-1;
    }
}

// CTA 0 sorts the routes while the other CTAs quantize each token once; the
// gate GEMM gathers token rows through the sorted route order.
__global__ __launch_bounds__(256) void glm53_sort_quant_kernel(
    const opus::bf16_t* input,const int32_t* ids,
    int32_t* gather,int32_t* scatter,int32_t* tasks,int32_t* count,
    opus::fp8_t* output,uint8_t* scales,int tokens,int stride,int capacity,int32_t* gate_tasks,int32_t* gate_count,int gate_capacity)
{
    if(blockIdx.x==0)
        glm53_sort_routes(ids,gather,scatter,tasks,count,
            tokens*kGlm53TopK,capacity,gate_tasks,gate_count,gate_capacity);
    else
        glm53_quant_group(input,output,scales,tokens,stride,
            (static_cast<int>(blockIdx.x)-1)*256+static_cast<int>(threadIdx.x));
}

// Weighted top-9 sum: one wave owns 512 columns; nine FP32 FMAs in route
// order, then one BF16 rounding.
__global__ __launch_bounds__(64) void glm53_route_reduce_kernel(
    const __hip_bfloat16* __restrict__ route_output,
    const float* __restrict__ route_weights,const int32_t* __restrict__ scatter,
    __hip_bfloat16* __restrict__ output,int tokens)
{
    constexpr int Items=8,Batch=3;
    constexpr int Columns=64*Items,Tiles=6144/Columns;
    const int token=blockIdx.x/Tiles,tile=blockIdx.x%Tiles,lane=threadIdx.x;
    if(token>=tokens)return;
    const int column=tile*Columns+lane*Items;
    const int r=lane<9?scatter[token*9+lane]:-1;
    const float w=lane<9?route_weights[token*9+lane]:0.0f;
    int rows[9];float weights[9];
#pragma unroll
    for(int i=0;i<9;++i){rows[i]=__shfl(r,i);weights[i]=__shfl(w,i);}
    float acc[Items]={};
#pragma unroll
    for(int first=0;first<9;first+=Batch)
    {
        alignas(16) uint64_t packed[Batch][Items/4]={};
#pragma unroll
        for(int i=0;i<Batch;++i)
            if(rows[first+i]>=0)
                reinterpret_cast<uint4*>(packed[i])[0]=reinterpret_cast<const uint4*>(route_output+static_cast<int64_t>(rows[first+i])*6144+column)[0];
        // Explicit register operands keep every independent load before conversion.
        asm volatile("" : "+v"(packed[0][0]),"+v"(packed[0][1]),"+v"(packed[1][0]),"+v"(packed[1][1]),"+v"(packed[2][0]),"+v"(packed[2][1]));
#pragma unroll
        for(int i=0;i<Batch;++i)
            if(rows[first+i]>=0)
            {
#pragma unroll
                for(int group=0;group<Items/4;++group)
                {
                    union {uint64_t bits;__hip_bfloat16 values[4];} loaded;
                    loaded.bits=packed[i][group];
#pragma unroll
                    for(int item=0;item<4;++item)
                        acc[group*4+item]=fmaf(__bfloat162float(loaded.values[item]),weights[first+i],acc[group*4+item]);
                }
            }
    }
    opus::vector_t<opus::bf16_t,Items> result;
#pragma unroll
    for(int i=0;i<Items;++i)result[i]=__builtin_bit_cast(opus::bf16_t,__float2bfloat16(acc[i]));
    reinterpret_cast<uint4*>(output+static_cast<int64_t>(token)*6144+column)[0]=reinterpret_cast<uint4*>(&result)[0];
}

} // namespace

namespace {

void glm53_check_routes(std::initializer_list<const aiter_tensor_t*> tensors,
                        int device,
                        int64_t routes)
{
    for(const auto* tensor : tensors)
        AITER_CHECK(tensor->is_gpu() && tensor->device_id == device &&
                        tensor->dtype() == AITER_DTYPE_i32 && tensor->is_contiguous() &&
                        tensor->numel() == routes,
                    "GLM-5.3 IQ2R route arrays must be contiguous int32 on the input GPU");
}

void glm53_check_task_table(const aiter_tensor_t& tasks,
                            const aiter_tensor_t& task_count,
                            int device,
                            int64_t routes,
                            int64_t rows)
{
    AITER_CHECK(tasks.is_gpu() && tasks.device_id == device && tasks.is_contiguous() &&
                    tasks.dtype() == AITER_DTYPE_i32 && tasks.dim() == 2 && tasks.size(1) == 3 &&
                    tasks.size(0) >= (routes + rows - 1) / rows + std::min<int64_t>(routes, 258),
                "GLM-5.3 IQ2R task table is too small");
    AITER_CHECK(task_count.is_gpu() && task_count.device_id == device &&
                    task_count.dtype() == AITER_DTYPE_i32 && task_count.numel() == 1,
                "GLM-5.3 IQ2R task count must be one int32");
}

void glm53_check_input(const aiter_tensor_t& input, int64_t min_tokens, int64_t max_tokens)
{
    AITER_CHECK(input.is_gpu() && input.dtype() == AITER_DTYPE_bf16 && input.dim() == 2 &&
                    input.size(1) == kGlm53Hidden && input.stride(1) == 1 &&
                    input.size(0) >= min_tokens && input.size(0) <= max_tokens,
                "GLM-5.3 IQ2R input must be BF16 [tokens, 6144] with contiguous rows and a "
                "supported token count");
}

void glm53_check_quant_output(const aiter_tensor_t& output,
                              const aiter_tensor_t& scales,
                              int device,
                              int64_t rows)
{
    AITER_CHECK(output.is_gpu() && output.device_id == device && output.is_contiguous() &&
                    output.dtype() == AITER_DTYPE_fp8 && output.dim() == 2 &&
                    output.size(0) == rows && output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R quantized input must be FP8 [rows, 6144]");
    AITER_CHECK(scales.is_gpu() && scales.device_id == device && scales.is_contiguous() &&
                    scales.dtype() == AITER_DTYPE_u8 && scales.dim() == 2 &&
                    scales.size(0) == rows && scales.size(1) == kGlm53Groups,
                "GLM-5.3 IQ2R input scales must be uint8 [rows, 192]");
}

} // namespace

// M1 front end: nine one-row tasks in route order; the quantized token row is
// written once per route.
void iq2r_glm53_m1_route_quant_out(const aiter_tensor_t& input,
                                   const aiter_tensor_t& topk_ids,
                                   aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& tasks,
                                   aiter_tensor_t& task_count,
                                   aiter_tensor_t& output,
                                   aiter_tensor_t& scales)
{
    const int device = input.device_id;
    glm53_check_input(input, 1, 1);
    glm53_check_routes({&topk_ids, &scatter_indices}, device, kGlm53TopK);
    AITER_CHECK(tasks.is_gpu() && tasks.device_id == device && tasks.is_contiguous() &&
                    tasks.dtype() == AITER_DTYPE_i32 && tasks.dim() == 2 &&
                    tasks.size(0) >= kGlm53TopK && tasks.size(1) == kTaskColumns &&
                    task_count.is_gpu() && task_count.device_id == device &&
                    task_count.dtype() == AITER_DTYPE_i32 && task_count.numel() == 1,
                "GLM-5.3 IQ2R M1 tasks must be int32 [>=9, 3] with one int32 count");
    glm53_check_quant_output(output, scales, device, kGlm53TopK);

    HipDeviceGuard device_guard(device);
    hipLaunchKernelGGL(glm53_m1_route_quant_kernel,
                       dim3(1),
                       dim3(256),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::bf16_t*>(input.data_ptr()),
                       static_cast<const int32_t*>(topk_ids.data_ptr()),
                       static_cast<int32_t*>(scatter_indices.data_ptr()),
                       static_cast<int32_t*>(tasks.data_ptr()),
                       static_cast<int32_t*>(task_count.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()));
    HIP_CALL_LAUNCH(hipGetLastError());
}

// Decode front end for 2..1024 tokens: block 0 sorts the 9 routes per token
// into M32 down tasks and M16 gate tasks, the remaining blocks quantize each
// token row once to MXFP8 in token order.
void iq2r_glm53_sort_quant_out(const aiter_tensor_t& input,
                               const aiter_tensor_t& topk_ids,
                               aiter_tensor_t& gather_indices,
                               aiter_tensor_t& scatter_indices,
                               aiter_tensor_t& tasks,
                               aiter_tensor_t& task_count,
                               aiter_tensor_t& output,
                               aiter_tensor_t& scales,
                               aiter_tensor_t& gate_tasks,
                               aiter_tensor_t& gate_task_count)
{
    const int device = input.device_id;
    glm53_check_input(input, 2, 1024);
    const int tokens     = static_cast<int>(input.size(0));
    const int64_t routes = static_cast<int64_t>(tokens) * kGlm53TopK;
    glm53_check_routes({&topk_ids, &gather_indices, &scatter_indices}, device, routes);
    glm53_check_task_table(tasks, task_count, device, routes, 32);
    glm53_check_task_table(gate_tasks, gate_task_count, device, routes, 16);
    glm53_check_quant_output(output, scales, device, tokens);

    HipDeviceGuard device_guard(device);
    hipLaunchKernelGGL(glm53_sort_quant_kernel,
                       dim3(1 + (tokens * kGlm53Groups + 255) / 256),
                       dim3(256),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::bf16_t*>(input.data_ptr()),
                       static_cast<const int32_t*>(topk_ids.data_ptr()),
                       static_cast<int32_t*>(gather_indices.data_ptr()),
                       static_cast<int32_t*>(scatter_indices.data_ptr()),
                       static_cast<int32_t*>(tasks.data_ptr()),
                       static_cast<int32_t*>(task_count.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()),
                       tokens,
                       static_cast<int>(input.stride(0)),
                       static_cast<int>(tasks.size(0)),
                       static_cast<int32_t*>(gate_tasks.data_ptr()),
                       static_cast<int32_t*>(gate_task_count.data_ptr()),
                       static_cast<int>(gate_tasks.size(0)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

// Prefill front end for 2..4096 tokens. Route sort: per-chunk expert
// histograms, a column-wise prefix over chunks, an expert prefix that emits
// M64 tasks, then a stable scatter. Each token row is quantized once.
void iq2r_glm53_prefill_sort_quant_out(const aiter_tensor_t& input,
                                       const aiter_tensor_t& topk_ids,
                                       aiter_tensor_t& gather_indices,
                                       aiter_tensor_t& scatter_indices,
                                       aiter_tensor_t& tasks,
                                       aiter_tensor_t& task_count,
                                       aiter_tensor_t& output,
                                       aiter_tensor_t& scales,
                                       aiter_tensor_t& scratch)
{
    constexpr int kTaskRows = 64;
    const int device        = input.device_id;
    glm53_check_input(input, 2, 4096);
    const int tokens        = static_cast<int>(input.size(0));
    const int64_t routes    = static_cast<int64_t>(tokens) * kGlm53TopK;
    glm53_check_routes({&topk_ids, &gather_indices, &scatter_indices}, device, routes);
    glm53_check_task_table(tasks, task_count, device, routes, kTaskRows);
    glm53_check_quant_output(output, scales, device, tokens);
    const int chunks = static_cast<int>((routes + 255) / 256);
    AITER_CHECK(scratch.is_gpu() && scratch.device_id == device && scratch.is_contiguous() &&
                    scratch.dtype() == AITER_DTYPE_i32 &&
                    scratch.numel() >= static_cast<int64_t>(chunks) * 512 + 1024,
                "GLM-5.3 IQ2R sort scratch is too small");

    HipDeviceGuard device_guard(device);
    const auto stream = getCurrentHIPStream();
    const auto* ids   = static_cast<const int32_t*>(topk_ids.data_ptr());
    auto* gather      = static_cast<int32_t*>(gather_indices.data_ptr());
    auto* scatter     = static_cast<int32_t*>(scatter_indices.data_ptr());
    auto* histogram   = static_cast<int32_t*>(scratch.data_ptr());
    const int count   = static_cast<int>(routes);
    hipLaunchKernelGGL(
        glm53_sort_histogram, dim3(chunks), dim3(256), 0, stream, ids, scatter, histogram, count);
    hipLaunchKernelGGL(
        glm53_sort_chunk_prefix, dim3(258), dim3(512), 0, stream, histogram, chunks);
    hipLaunchKernelGGL(glm53_sort_expert_prefix,
                       dim3(1),
                       dim3(512),
                       0,
                       stream,
                       histogram,
                       static_cast<int32_t*>(tasks.data_ptr()),
                       static_cast<int32_t*>(task_count.data_ptr()),
                       chunks,
                       kTaskRows,
                       static_cast<int>(tasks.size(0)));
    hipLaunchKernelGGL(glm53_sort_scatter,
                       dim3(chunks),
                       dim3(256),
                       0,
                       stream,
                       ids,
                       gather,
                       scatter,
                       histogram,
                       count);
    hipLaunchKernelGGL(glm53_quant_kernel,
                       dim3((tokens * kGlm53Groups + 255) / 256),
                       dim3(256),
                       0,
                       stream,
                       static_cast<const opus::bf16_t*>(input.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()),
                       tokens,
                       static_cast<int>(input.stride(0)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_glm53_route_reduce_out(const aiter_tensor_t& route_output,
                                 const aiter_tensor_t& route_weights,
                                 const aiter_tensor_t& scatter_indices,
                                 aiter_tensor_t& output)
{
    const int device = output.device_id;
    AITER_CHECK(output.is_gpu() && output.dtype() == AITER_DTYPE_bf16 && output.dim() == 2 &&
                    output.size(1) == kGlm53Hidden && output.is_contiguous(),
                "GLM-5.3 IQ2R reduce output must be contiguous BF16 [tokens, 6144]");
    const int tokens     = static_cast<int>(output.size(0));
    const int64_t routes = static_cast<int64_t>(tokens) * kGlm53TopK;
    AITER_CHECK(route_output.is_gpu() && route_output.device_id == device &&
                    route_output.dtype() == AITER_DTYPE_bf16 && route_output.is_contiguous() &&
                    route_output.dim() == 2 && route_output.size(0) == routes &&
                    route_output.size(1) == kGlm53Hidden,
                "GLM-5.3 IQ2R route output must be contiguous BF16 [tokens * 9, 6144]");
    AITER_CHECK(route_weights.is_gpu() && route_weights.device_id == device &&
                    route_weights.dtype() == AITER_DTYPE_fp32 && route_weights.is_contiguous() &&
                    route_weights.numel() == routes,
                "GLM-5.3 IQ2R route weights must be contiguous FP32 [tokens, 9]");
    glm53_check_routes({&scatter_indices}, device, routes);

    HipDeviceGuard device_guard(device);
    hipLaunchKernelGGL(glm53_route_reduce_kernel,
                       dim3(tokens * (kGlm53Hidden / 512)),
                       dim3(64),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                       static_cast<const float*>(route_weights.data_ptr()),
                       static_cast<const int32_t*>(scatter_indices.data_ptr()),
                       static_cast<__hip_bfloat16*>(output.data_ptr()),
                       tokens);
    HIP_CALL_LAUNCH(hipGetLastError());
}


} // namespace aiter

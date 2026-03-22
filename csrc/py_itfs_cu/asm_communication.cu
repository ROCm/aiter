// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "aiter_hip_common.h"
#include "custom_all_reduce.cuh"

extern "C" __attribute__((visibility("default"))) void all_reduce_asm(
    AiterTensor* input,
    int64_t _ca,
    AiterTensor* reg_sig,
    AiterTensor* reg_buffer,
    bool isGraph,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(input->device_id);

    auto input_size = input->numel() * input->element_size();

    void* inp_ptr = input->data_ptr();
    if(!isGraph)
    {
        AITER_CHECK(input_size <= reg_buffer->numel() * reg_buffer->element_size(),
                    "registered buffer is too small to contain the input",
                    input_size,
                    ">",
                    reg_buffer->numel() * reg_buffer->element_size());
        HIP_CALL(hipMemcpyAsync(
            reg_buffer->data_ptr(), inp_ptr, input_size, hipMemcpyDeviceToDevice, stream));
        inp_ptr = reg_buffer->data_ptr();
    }

    auto ca  = reinterpret_cast<aiter::CustomAllreduce*>(_ca);
    using RD = aiter::RankData;

    RD* input_rd = ca->get_buffer_RD(stream, inp_ptr);
    RD* sig_rd   = ca->get_buffer_RD(stream, reg_sig->data_ptr());

    struct __attribute__((packed)) KernelArgs
    {
        void* ptr_gpu0_data;
        p2 _p0;
        void* ptr_gpu0_sig;
        p2 _p8;
        void* ptr_gpu1_sig;
        p2 _p9;
        void* ptr_gpu2_sig;
        p2 _p10;
        void* ptr_gpu3_sig;
        p2 _p11;
        void* ptr_gpu4_sig;
        p2 _p12;
        void* ptr_gpu5_sig;
        p2 _p13;
        void* ptr_gpu6_sig;
        p2 _p14;
        void* ptr_gpu7_sig;
        p2 _p15;
        unsigned int gpuId;
        p3 _p16;
        unsigned int stride_gpu;
        p3 _p17;
        unsigned int stride_tg;
        p3 _p18;
        unsigned int stride_wave;
        p3 _p19;
        unsigned int loopcnt;
        p3 _p20;
    };

    int bdx        = 256;
    int gdx        = 64;
    int gdy        = 1;
    int gdz        = 1;
    int stride_GPU = input_size / ca->world_size_; // stride base on the pass in GPU id; gpu0 focus
                                                   // on 0~15; gpu1 focus on 16~31
    int stride_TG = stride_GPU /
                    gdx; // stride base on TG id; 64 TGs, every TG focus on 16*8192/64=2048 elements
    int stride_WV =
        stride_TG /
        (bdx / 64); // stride base on Wave id, 4 waves, every wave focus on 512 elements; 1024 bytes

    KernelArgs args;
    size_t arg_size    = sizeof(args);
    args.ptr_gpu0_data = reinterpret_cast<void*>(input_rd);
    args.ptr_gpu0_sig  = const_cast<void*>(sig_rd->ptrs[0]);
    args.ptr_gpu1_sig  = const_cast<void*>(sig_rd->ptrs[1]);
    args.ptr_gpu2_sig  = const_cast<void*>(sig_rd->ptrs[2]);
    args.ptr_gpu3_sig  = const_cast<void*>(sig_rd->ptrs[3]);
    args.ptr_gpu4_sig  = const_cast<void*>(sig_rd->ptrs[4]);
    args.ptr_gpu5_sig  = const_cast<void*>(sig_rd->ptrs[5]);
    args.ptr_gpu6_sig  = const_cast<void*>(sig_rd->ptrs[6]);
    args.ptr_gpu7_sig  = const_cast<void*>(sig_rd->ptrs[7]);
    args.gpuId         = ca->rank_;
    args.stride_gpu    = stride_GPU;
    args.stride_tg     = stride_TG;
    args.stride_wave   = stride_WV;
    args.loopcnt       = 10;

    static AiterAsmKernel impl("allreduce_kernel_func", "all_reduce.co");
    impl.launch_kernel({&args,
                        &arg_size,
                        gdx, // gdx
                        gdy, // gdy
                        gdz, // gdz
                        bdx, // bdx: 4 wv64
                        1,   // bdy
                        1,   // bdz
                        stream});
    // result is written in-place to inp_ptr (input or reg_buffer depending on isGraph)
}

extern "C" __attribute__((visibility("default"))) void all_reduce_rmsnorm(
    AiterTensor* input,        // [m ,n]
    AiterTensor* residual_in,  // [m ,n]
    AiterTensor* weight,       // [1 ,n]
    AiterTensor* bias,         // [1 ,n]
    float epsilon,
    // following are fused_allreduce args
    int64_t _ca,
    AiterTensor* reg_sig,
    AiterTensor* reg_buffer,
    bool isGraph,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(input->device_id);

    auto size_input = input->numel() * input->element_size();
    auto size_pad   = (size_input + 4095) & 0xfffff000;

    void* inp_ptr = input->data_ptr();
    // reg_buffer contains input|out|res_out
    auto size_needed = size_pad * 3;
    AITER_CHECK(size_needed <= reg_buffer->numel() * reg_buffer->element_size(),
                "registered buffer is too small to contain the input ",
                size_needed,
                ">",
                reg_buffer->numel() * reg_buffer->element_size());

    uint64_t out_offset = (uint64_t)size_pad;
    uint64_t res_offset = (uint64_t)size_pad * 2;
    if(!isGraph)
    {
        HIP_CALL(hipMemcpyAsync(
            reg_buffer->data_ptr(), inp_ptr, size_input, hipMemcpyDeviceToDevice, stream));
        inp_ptr = reg_buffer->data_ptr();
    }

    auto ca  = reinterpret_cast<aiter::CustomAllreduce*>(_ca);
    using RD = aiter::RankData;

    RD* sig_rd   = ca->get_buffer_RD(stream, reg_sig->data_ptr());
    RD* reg_rd   = ca->get_buffer_RD(stream, reg_buffer->data_ptr());
    RD* input_rd = ca->get_buffer_RD(stream, inp_ptr);

    uint64_t gpu_bufs[8 * 4];
    for(size_t i = 0; i < ca->world_size_; i++)
    {
        gpu_bufs[i]      = reinterpret_cast<uint64_t>(input_rd->ptrs[i]);
        gpu_bufs[i + 8]  = reinterpret_cast<uint64_t>(reg_rd->ptrs[i]) + out_offset;
        gpu_bufs[i + 16] = reinterpret_cast<uint64_t>(reg_rd->ptrs[i]) + res_offset;
    }

    uint64_t* gpu_addr_buf_in;
    uint addr_buf_size = 8 * 4 * sizeof(uint64_t);
    HIP_CALL(hipMalloc(&gpu_addr_buf_in, addr_buf_size));
    HIP_CALL(hipMemcpy(gpu_addr_buf_in, gpu_bufs, addr_buf_size, hipMemcpyHostToDevice));

    struct __attribute__((packed)) KernelArgs
    {
        void* ptr_gpu0_data;
        p2 _p0;
        void* ptr_gpu0_sig;
        p2 _p8;
        void* ptr_gpu1_sig;
        p2 _p9;
        void* ptr_gpu2_sig;
        p2 _p10;
        void* ptr_gpu3_sig;
        p2 _p11;
        void* ptr_gpu4_sig;
        p2 _p12;
        void* ptr_gpu5_sig;
        p2 _p13;
        void* ptr_gpu6_sig;
        p2 _p14;
        void* ptr_gpu7_sig;
        p2 _p15;
        void* ptr_resi_in;
        p2 _p1;
        void* ptr_weight_in;
        p2 _p2;
        void* ptr_bias_in;
        p2 _p3;
        void* ptr_xscale;
        p2 _p4;
        unsigned int gpuId;
        p3 _p16;
        unsigned int stride_gpu;
        p3 _p17;
        unsigned int N;
        p3 _p18;
        float epsilon;
        p3 _p19;
        unsigned int tgs;
        p3 _p20;
        unsigned int loopcnt;
        p3 _p21;
    };

    int N = input->size(-1);
    int M = input->numel() / N;

    int TGs = M / ca->world_size_;
    KernelArgs args;
    size_t arg_size    = sizeof(args);
    args.ptr_gpu0_data = reinterpret_cast<void*>(gpu_addr_buf_in);
    args.ptr_gpu0_sig  = const_cast<void*>(sig_rd->ptrs[0]);
    args.ptr_gpu1_sig  = const_cast<void*>(sig_rd->ptrs[1]);
    args.ptr_gpu2_sig  = const_cast<void*>(sig_rd->ptrs[2]);
    args.ptr_gpu3_sig  = const_cast<void*>(sig_rd->ptrs[3]);
    args.ptr_gpu4_sig  = const_cast<void*>(sig_rd->ptrs[4]);
    args.ptr_gpu5_sig  = const_cast<void*>(sig_rd->ptrs[5]);
    args.ptr_gpu6_sig  = const_cast<void*>(sig_rd->ptrs[6]);
    args.ptr_gpu7_sig  = const_cast<void*>(sig_rd->ptrs[7]);
    args.ptr_resi_in   = residual_in->data_ptr();
    args.ptr_weight_in = weight->data_ptr();
    args.ptr_bias_in   = bias->data_ptr();
    args.gpuId         = ca->rank_;
    args.stride_gpu    = size_input / ca->world_size_;
    args.N             = N;
    args.epsilon       = epsilon;
    args.tgs           = TGs;
    args.loopcnt       = 0;

    static AiterAsmKernel impl("allreduce_rmsnorm_N8192_kernel", "allreduce_rmsnorm_N8192.co");

    impl.launch_kernel({&args,
                        &arg_size,
                        TGs, // gdx
                        1,   // gdy
                        1,   // gdz
                        256, // bdx: 4 wv64
                        1,   // bdy
                        1,   // bdz
                        stream});
    // results are written in-place: out at reg_buffer+size_pad, residual at reg_buffer+size_pad*2
}

extern "C" __attribute__((visibility("default"))) void all_reduce_rmsnorm_quant(
    AiterTensor* input,        // [m ,n]
    AiterTensor* residual_in,  // [m ,n]
    AiterTensor* xscale,       // [1 ,n]
    AiterTensor* weight,       // [1 ,n]
    AiterTensor* bias,         // [1 ,n]
    float epsilon,
    // following are fused_allreduce args
    int64_t _ca,
    AiterTensor* reg_sig,
    AiterTensor* reg_buffer,
    bool isGraph,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(input->device_id);

    auto size_input = input->numel() * input->element_size();
    auto size_pad   = (size_input + 4095) & 0xfffff000;

    void* inp_ptr = input->data_ptr();
    // reg_buffer contains input|out|res_out
    auto size_needed = size_pad * 4;
    AITER_CHECK(size_needed <= reg_buffer->numel() * reg_buffer->element_size(),
                "registered buffer is too small to contain the input ",
                size_needed,
                ">",
                reg_buffer->numel() * reg_buffer->element_size());

    if(!isGraph)
    {
        HIP_CALL(hipMemcpyAsync(
            reg_buffer->data_ptr(), inp_ptr, size_input, hipMemcpyDeviceToDevice, stream));
        inp_ptr = reg_buffer->data_ptr();
    }

    auto ca  = reinterpret_cast<aiter::CustomAllreduce*>(_ca);
    using RD = aiter::RankData;

    RD* sig_rd   = ca->get_buffer_RD(stream, reg_sig->data_ptr());
    RD* reg_rd   = ca->get_buffer_RD(stream, reg_buffer->data_ptr());
    RD* input_rd = ca->get_buffer_RD(stream, inp_ptr);

    uint64_t gpu_bufs[8 * 4];
    for(size_t i = 0; i < ca->world_size_; i++)
    {
        gpu_bufs[i]      = reinterpret_cast<uint64_t>(input_rd->ptrs[i]);
        gpu_bufs[i + 8]  = reinterpret_cast<uint64_t>(reg_rd->ptrs[i]) + size_pad;
        gpu_bufs[i + 16] = reinterpret_cast<uint64_t>(reg_rd->ptrs[i]) + size_pad * 2;
        gpu_bufs[i + 24] = reinterpret_cast<uint64_t>(reg_rd->ptrs[i]) + size_pad * 3;
    }

    uint64_t* gpu_addr_buf_in;
    uint addr_buf_size = 8 * 4 * sizeof(uint64_t);
    HIP_CALL(hipMalloc(&gpu_addr_buf_in, addr_buf_size));
    HIP_CALL(hipMemcpy(gpu_addr_buf_in, gpu_bufs, addr_buf_size, hipMemcpyHostToDevice));

    struct __attribute__((packed)) KernelArgs
    {
        void* ptr_gpu0_data;
        p2 _p0;
        void* ptr_gpu0_sig;
        p2 _p8;
        void* ptr_gpu1_sig;
        p2 _p9;
        void* ptr_gpu2_sig;
        p2 _p10;
        void* ptr_gpu3_sig;
        p2 _p11;
        void* ptr_gpu4_sig;
        p2 _p12;
        void* ptr_gpu5_sig;
        p2 _p13;
        void* ptr_gpu6_sig;
        p2 _p14;
        void* ptr_gpu7_sig;
        p2 _p15;
        void* ptr_resi_in;
        p2 _p1;
        void* ptr_weight_in;
        p2 _p2;
        void* ptr_bias_in;
        p2 _p3;
        void* ptr_xscale;
        p2 _p4;
        unsigned int gpuId;
        p3 _p16;
        unsigned int stride_gpu;
        p3 _p17;
        unsigned int N;
        p3 _p18;
        float epsilon;
        p3 _p19;
        unsigned int tgs;
        p3 _p20;
        unsigned int loopcnt;
        p3 _p21;
    };

    int N = input->size(-1);
    int M = input->numel() / N;

    int TGs = M / ca->world_size_;
    KernelArgs args;
    size_t arg_size    = sizeof(args);
    args.ptr_gpu0_data = reinterpret_cast<void*>(gpu_addr_buf_in);
    args.ptr_gpu0_sig  = const_cast<void*>(sig_rd->ptrs[0]);
    args.ptr_gpu1_sig  = const_cast<void*>(sig_rd->ptrs[1]);
    args.ptr_gpu2_sig  = const_cast<void*>(sig_rd->ptrs[2]);
    args.ptr_gpu3_sig  = const_cast<void*>(sig_rd->ptrs[3]);
    args.ptr_gpu4_sig  = const_cast<void*>(sig_rd->ptrs[4]);
    args.ptr_gpu5_sig  = const_cast<void*>(sig_rd->ptrs[5]);
    args.ptr_gpu6_sig  = const_cast<void*>(sig_rd->ptrs[6]);
    args.ptr_gpu7_sig  = const_cast<void*>(sig_rd->ptrs[7]);
    args.ptr_resi_in   = residual_in->data_ptr();
    args.ptr_weight_in = weight->data_ptr();
    args.ptr_bias_in   = bias->data_ptr();
    args.ptr_xscale    = xscale->data_ptr();
    args.gpuId         = ca->rank_;
    args.stride_gpu    = size_input / ca->world_size_;
    args.N             = N;
    args.epsilon       = epsilon;
    args.tgs           = TGs;
    args.loopcnt       = 0;

    static AiterAsmKernel impl("allreduce_rmsnorm_qnt_N8192_kernel",
                               "allreduce_rmsnorm_qnt_N8192.co");

    impl.launch_kernel({&args,
                        &arg_size,
                        TGs, // gdx
                        1,   // gdy
                        1,   // gdz
                        256, // bdx: 4 wv64
                        1,   // bdy
                        1,   // bdz
                        stream});
    // results are written in-place: out at reg_buffer+size_pad, residual at reg_buffer+size_pad*2, yscale at reg_buffer+size_pad*3
}

// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "aiter_hip_common.h"

struct __attribute__((packed)) KernelArgs
{
    void *ptr_O;
    p2 _p0;
    void *ptr_X;
    p2 _p1;
    void *ptr_GU;
    p2 _p2;
    void *ptr_XC;
    p2 _p3;
    void *ptr_D;
    p2 _p4;
    void *ptr_XQ;
    p2 _p5;
    void *ptr_GUQ;
    p2 _p6;
    void *ptr_DQ;
    p2 _p7;
    void *ptr_SMQ;
    p2 _p8;
    void *ptr_STP;
    p2 _p9;
    void *ptr_SW;
    p2 _p10;
    void *ptr_SEP;
    p2 _p11;
    unsigned int dim;
    p3 _p12;
    unsigned int hidden_dim;
    p3 _p13;
    unsigned int token_cnt;
    p3 _p14;
    unsigned int eprt_cnt;
    p3 _p15;
    unsigned int Xs;
    p3 _p16;
    unsigned int GUs;
    p3 _p17;
    unsigned int Ds;
    p3 _p18;
    unsigned int Os;
    p3 _p19;
    unsigned int eGUs;
    p3 _p20;
    unsigned int eDs;
    p3 _p21;
    unsigned int eGUQs;
    p3 _p22;
    unsigned int eDQs;
    p3 _p23;
    unsigned int eSMQs;
    p3 _p24;
    unsigned int topk;
    p3 _p25;
};

class FMoeKernel
{
private:
    hipModule_t module;
    hipFunction_t kernel_func;
    uint32_t sub_GU = 512;

public:
    FMoeKernel(const char *name, const char *hsaco, uint32_t sub_GU = 512)
    {
        std::cout << "hipModuleLoad: " << (std::string(AITER_ASM_DIR) + hsaco).c_str() << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        std::cout << " Success" << std::endl;
        this->sub_GU = sub_GU;
    };

    template <typename T, typename T_O, bool switchGxy = false>
    void launch_kernel(torch::Tensor &out,                    // [token_cnt, dim]
                       torch::Tensor &input,                  // [token_cnt, dim] M,K
                       torch::Tensor &w1,                     // [expert, inter_dim, dim] N,K
                       torch::Tensor &w2,                     // [expert, dim, inter_dim]
                       torch::Tensor &sorted_token_ids,       // [max_num_tokens_padded]
                       torch::Tensor &sorted_weight_buf,      // [max_num_tokens_padded]
                       torch::Tensor &sorted_expert_ids,      // [max_num_m_blocks]
                       torch::Tensor &num_tokens_post_padded, // [1]
                       uint32_t topk,                         //
                       std::optional<torch::Tensor> input_dqn = std::nullopt,
                       std::optional<torch::Tensor> w1_dqn = std::nullopt,
                       std::optional<torch::Tensor> w2_dqn = std::nullopt,
                       std::optional<torch::Tensor> w2_smooth_qnt = std::nullopt //
    )
    {
        int token_cnt = out.size(0);
        int dim = input.size(1);
        int sub_X_cnt = sorted_expert_ids.size(0);
        int eprt = w1.size(0);
        int hidden_dim = w2.size(2);
        uint32_t sub_GU = this->sub_GU;
        uint32_t I_elemSize = sizeof(T);
        uint32_t O_elemSize = sizeof(T_O);

        int stride_X = input.stride(0) * input.element_size();
        int stride_GU = dim * I_elemSize;
        int stride_D = hidden_dim * I_elemSize;
        int stride_expert_GU = stride_GU * hidden_dim;
        int stride_expert_D = stride_D * dim;
        int stride_expert_GUDQN = hidden_dim * sizeof(float);
        int stride_expert_DDQN = dim * sizeof(float);
        int stride_expert_SMTDQN = hidden_dim * sizeof(float);
        int stride_O = dim * O_elemSize;
        if (hidden_dim * 2 == w1.size(1))
        {
            stride_expert_GU *= 2;
            stride_expert_GUDQN *= 2;
        }

        KernelArgs args;
        size_t arg_size = sizeof(args);
        args.ptr_O = out.data_ptr();
        args.ptr_X = input.data_ptr();
        args.ptr_GU = w1.data_ptr();
        args.ptr_XC = num_tokens_post_padded.data_ptr();
        args.ptr_D = w2.data_ptr();
        if constexpr (std::is_same<T, uint8_t>::value)
        {
            args.ptr_XQ = input_dqn.value().data_ptr();
            args.ptr_GUQ = w1_dqn.value().data_ptr();
            args.ptr_DQ = w2_dqn.value().data_ptr();
            args.ptr_SMQ = w2_smooth_qnt.has_value() ? w2_smooth_qnt.value().data_ptr() : nullptr;
        }
        else
        {
            args.ptr_XQ = nullptr;
            args.ptr_GUQ = nullptr;
            args.ptr_DQ = nullptr;
            args.ptr_SMQ = nullptr;
        }
        args.ptr_STP = sorted_token_ids.data_ptr();
        args.ptr_SW = sorted_weight_buf.data_ptr();
        args.ptr_SEP = sorted_expert_ids.data_ptr();
        args.dim = dim;
        args.hidden_dim = hidden_dim;
        args.token_cnt = token_cnt;
        args.eprt_cnt = eprt;
        args.Xs = stride_X;
        args.GUs = stride_GU;
        args.Ds = stride_D;
        args.Os = stride_O;
        args.eGUs = stride_expert_GU;
        args.eDs = stride_expert_D;
        args.eGUQs = stride_expert_GUDQN;
        args.eDQs = stride_expert_DDQN;
        args.eSMQs = stride_expert_SMTDQN;
        args.topk = topk;

        // std::cout << "args.dim: " << args.dim << std::endl;
        // std::cout << "args.hidden_dim: " << args.hidden_dim << std::endl;
        // std::cout << "args.token_cnt: " << args.token_cnt << std::endl;
        // std::cout << "args.eprt_cnt: " << args.eprt_cnt << std::endl;
        // std::cout << "args.stride_X: " << args.Xs << std::endl;
        // std::cout << "args.stride_GU: " << args.GUs << std::endl;
        // std::cout << "args.stride_D: " << args.Ds << std::endl;
        // std::cout << "args.stride_O: " << args.Os << std::endl;
        // std::cout << "args.stride_expert_GU: " << args.eGUs << std::endl;
        // std::cout << "args.stride_expert_D: " << args.eDs << std::endl;
        // std::cout << "args.stride_expert_GUDQN: " << args.eGUQs << std::endl;
        // std::cout << "args.stride_expert_DDQN: " << args.eDQs << std::endl;
        // std::cout << "args.stride_expert_SMTDQN: " << args.eSMQs << std::endl;
        // std::cout << "args.topk: " << args.topk << std::endl;

        void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size, HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = ((hidden_dim + sub_GU - 1) / sub_GU);
        int gdy = sub_X_cnt;
        int gdz = 1;
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if constexpr (switchGxy)
        {
            HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                           gdy, gdx, gdz,
                                           bdx, 1, 1,
                                           0, stream, nullptr, (void **)&config));
        }
        else
        {
            HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                           gdx, gdy, gdz,
                                           bdx, 1, 1,
                                           0, stream, nullptr, (void **)&config));
        }
    };
};

void fmoe(torch::Tensor &out,                    // [token_cnt, dim]
          torch::Tensor &input,                  // [token_cnt, dim] M,K
          torch::Tensor &gate,                   // [expert, inter_dim, dim] N,K
          torch::Tensor &down,                   // [expert, dim, inter_dim]
          torch::Tensor &sorted_token_ids,       // [max_num_tokens_padded]
          torch::Tensor &sorted_weight_buf,      // [max_num_tokens_padded]
          torch::Tensor &sorted_expert_ids,      // [max_num_m_blocks]
          torch::Tensor &num_tokens_post_padded, // [1]
          uint32_t topk                          //
)
{
    // bf16 g1u0
    static FMoeKernel impl("fmoe_kernel_func", "fmoe.co");
    impl.launch_kernel<uint16_t, uint16_t>(out,
                                           input,
                                           gate,
                                           down,
                                           sorted_token_ids,
                                           sorted_weight_buf,
                                           sorted_expert_ids,
                                           num_tokens_post_padded,
                                           topk);
}

void fmoe_int8_g1u0(torch::Tensor &out,                    // [token_cnt, dim]
                    torch::Tensor &input,                  // [token_cnt, dim] M,K
                    torch::Tensor &gate,                   // [expert, inter_dim, dim] N,K
                    torch::Tensor &down,                   // [expert, dim, inter_dim]
                    torch::Tensor &sorted_token_ids,       // [max_num_tokens_padded]
                    torch::Tensor &sorted_weight_buf,      // [max_num_tokens_padded]
                    torch::Tensor &sorted_expert_ids,      // [max_num_m_blocks]
                    torch::Tensor &num_tokens_post_padded, // [1]
                    uint32_t topk,                         //
                    torch::Tensor &input_scale,            // [token_cnt, 1]
                    torch::Tensor &fc1_scale,              // [expert, 1, hidden_dim]
                    torch::Tensor &fc2_scale,              // [expert, 1, dim]
                    torch::Tensor &fc2_smooth_scale        // [expert, 1, hidden_dim]
)
{
    FMoeKernel *impl_ptr = nullptr;
    int hidden_dim = down.size(2);

    if (input.dtype() == at::ScalarType::Char)
    {
        static FMoeKernel impl_int8_512("fmoe_int8_g1u0_subGU_512", "fmoe_int8_g1u0_subGU_512.co", 512);
        static FMoeKernel impl_int8_448("fmoe_int8_g1u0_subGU_448", "fmoe_int8_g1u0_subGU_448.co", 448);
        static FMoeKernel impl_int8_384("fmoe_int8_g1u0_subGU_384", "fmoe_int8_g1u0_subGU_384.co", 384);
        static FMoeKernel impl_int8_320("fmoe_int8_g1u0_subGU_320", "fmoe_int8_g1u0_subGU_320.co", 320);
        static FMoeKernel impl_int8_256("fmoe_int8_g1u0_subGU_256", "fmoe_int8_g1u0_subGU_256.co", 256);
        static FMoeKernel impl_int8_192("fmoe_int8_g1u0_subGU_192", "fmoe_int8_g1u0_subGU_192.co", 192);
        static FMoeKernel impl_int8_128("fmoe_int8_g1u0_subGU_128", "fmoe_int8_g1u0_subGU_128.co", 128);

        if ((hidden_dim % 512) == 0)
            impl_ptr = &impl_int8_512;
        else if ((hidden_dim % 448) == 0)
            impl_ptr = &impl_int8_448;
        else if ((hidden_dim % 384) == 0)
            impl_ptr = &impl_int8_384;
        else if ((hidden_dim % 320) == 0)
            impl_ptr = &impl_int8_320;
        else if ((hidden_dim % 256) == 0)
            impl_ptr = &impl_int8_256;
        else if ((hidden_dim % 192) == 0)
            impl_ptr = &impl_int8_192;
        else if ((hidden_dim % 128) == 0)
            impl_ptr = &impl_int8_128;
        else
            TORCH_CHECK(false, __func__, " Unsupported hidden_dim " + std::to_string(hidden_dim) + ", which should be divisible by 128, 192, 256, 320, 384, 448 or 512");
    }
    else
    {
        TORCH_CHECK(false, __func__, " Input only supput Int8!");
    }

    impl_ptr->launch_kernel<uint8_t, uint16_t>(out,
                                               input,
                                               gate,
                                               down,
                                               sorted_token_ids,
                                               sorted_weight_buf,
                                               sorted_expert_ids,
                                               num_tokens_post_padded,
                                               topk,
                                               // quant args
                                               input_scale,
                                               fc1_scale,
                                               fc2_scale,
                                               fc2_smooth_scale);
}

void fmoe_g1u1(torch::Tensor &out,                                          // [token_cnt, dim]
               torch::Tensor &input,                                        // [token_cnt, dim] M,K
               torch::Tensor &gate,                                         // [expert, inter_dim*2, dim] N,K
               torch::Tensor &down,                                         // [expert, dim, inter_dim]
               torch::Tensor &sorted_token_ids,                             // [max_num_tokens_padded]
               torch::Tensor &sorted_weight_buf,                            // [max_num_tokens_padded]
               torch::Tensor &sorted_expert_ids,                            // [max_num_m_blocks]
               torch::Tensor &num_tokens_post_padded,                       // [1]
               uint32_t topk,                                               //
               torch::Tensor &input_scale,                                  // [token_cnt, 1]
               torch::Tensor &fc1_scale,                                    // [expert, 1, hidden_dim]
               torch::Tensor &fc2_scale,                                    // [expert, 1, dim]
               std::optional<torch::Tensor> fc2_smooth_scale = std::nullopt // [expert, 1, hidden_dim]
)
{
    FMoeKernel *impl_ptr = nullptr;
    int hidden_dim = down.size(2);

    if (input.dtype() == at::ScalarType::Char)
    {
        static FMoeKernel impl_int8_512("fmoe_int8_g1u1_subGU_512", "fmoe_int8_g1u1_subGU_512.co", 512);
        static FMoeKernel impl_int8_448("fmoe_int8_g1u1_subGU_448", "fmoe_int8_g1u1_subGU_448.co", 448);
        static FMoeKernel impl_int8_384("fmoe_int8_g1u1_subGU_384", "fmoe_int8_g1u1_subGU_384.co", 384);
        static FMoeKernel impl_int8_320("fmoe_int8_g1u1_subGU_320", "fmoe_int8_g1u1_subGU_320.co", 320);
        static FMoeKernel impl_int8_256("fmoe_int8_g1u1_subGU_256", "fmoe_int8_g1u1_subGU_256.co", 256);
        static FMoeKernel impl_int8_192("fmoe_int8_g1u1_subGU_192", "fmoe_int8_g1u1_subGU_192.co", 192);
        static FMoeKernel impl_int8_128("fmoe_int8_g1u1_subGU_128", "fmoe_int8_g1u1_subGU_128.co", 128);

        if ((hidden_dim % 512) == 0)
            impl_ptr = &impl_int8_512;
        else if ((hidden_dim % 448) == 0)
            impl_ptr = &impl_int8_448;
        else if ((hidden_dim % 384) == 0)
            impl_ptr = &impl_int8_384;
        else if ((hidden_dim % 320) == 0)
            impl_ptr = &impl_int8_320;
        else if ((hidden_dim % 256) == 0)
            impl_ptr = &impl_int8_256;
        else if ((hidden_dim % 192) == 0)
            impl_ptr = &impl_int8_192;
        else if ((hidden_dim % 128) == 0)
            impl_ptr = &impl_int8_128;
        else
            TORCH_CHECK(false, __func__, " Unsupported hidden_dim " + std::to_string(hidden_dim) + ", which should be divisible by 128, 192, 256, 320, 384, 448 or 512");
    }
    else if (input.dtype() == at::ScalarType::Float8_e4m3fnuz)
    {
        static FMoeKernel impl_fp8_512("fmoe_fp8_g1u1_subGU_512", "fmoe_fp8_g1u1_subGU_512.co", 512);
        static FMoeKernel impl_fp8_448("fmoe_fp8_g1u1_subGU_448", "fmoe_fp8_g1u1_subGU_448.co", 448);
        static FMoeKernel impl_fp8_384("fmoe_fp8_g1u1_subGU_384", "fmoe_fp8_g1u1_subGU_384.co", 384);
        static FMoeKernel impl_fp8_320("fmoe_fp8_g1u1_subGU_320", "fmoe_fp8_g1u1_subGU_320.co", 320);
        static FMoeKernel impl_fp8_256("fmoe_fp8_g1u1_subGU_256", "fmoe_fp8_g1u1_subGU_256.co", 256);
        static FMoeKernel impl_fp8_192("fmoe_fp8_g1u1_subGU_192", "fmoe_fp8_g1u1_subGU_192.co", 192);
        static FMoeKernel impl_fp8_128("fmoe_fp8_g1u1_subGU_128", "fmoe_fp8_g1u1_subGU_128.co", 128);
        if ((hidden_dim % 512) == 0)
            impl_ptr = &impl_fp8_512;
        else if ((hidden_dim % 448) == 0)
            impl_ptr = &impl_fp8_448;
        else if ((hidden_dim % 384) == 0)
            impl_ptr = &impl_fp8_384;
        else if ((hidden_dim % 320) == 0)
            impl_ptr = &impl_fp8_320;
        else if ((hidden_dim % 256) == 0)
            impl_ptr = &impl_fp8_256;
        else if ((hidden_dim % 192) == 0)
            impl_ptr = &impl_fp8_192;
        else if ((hidden_dim % 128) == 0)
            impl_ptr = &impl_fp8_128;
        else
            TORCH_CHECK(false, __func__, " Unsupported hidden_dim " + std::to_string(hidden_dim) + ", which should be divisible by 128, 192, 256, 320, 384, 448 or 512");
    }
    else
    {
        TORCH_CHECK(false, __func__, " Input only supput Int8/Fp8!");
    }

    impl_ptr->launch_kernel<uint8_t, uint16_t>(out,
                                               input,
                                               gate,
                                               down,
                                               sorted_token_ids,
                                               sorted_weight_buf,
                                               sorted_expert_ids,
                                               num_tokens_post_padded,
                                               topk,
                                               // quant args
                                               input_scale,
                                               fc1_scale,
                                               fc2_scale,
                                               fc2_smooth_scale);
}

void fmoe_int8_g1u0_a16(torch::Tensor &out,                    // [token_cnt, dim]
                        torch::Tensor &input,                  // [token_cnt, dim] M,K
                        torch::Tensor &gate,                   // [expert, inter_dim, dim] N,K
                        torch::Tensor &down,                   // [expert, dim, inter_dim]
                        torch::Tensor &sorted_token_ids,       // [max_num_tokens_padded]
                        torch::Tensor &sorted_weight_buf,      // [max_num_tokens_padded]
                        torch::Tensor &sorted_expert_ids,      // [max_num_m_blocks]
                        torch::Tensor &num_tokens_post_padded, // [1]
                        uint32_t topk,                         //
                        torch::Tensor &fc1_scale,              // [expert, 1, hidden_dim]
                        torch::Tensor &fc2_scale,              // [expert, 1, dim]
                        torch::Tensor &fc1_smooth_scale,       // [expert, 1, hidden_dim]
                        torch::Tensor &fc2_smooth_scale        // [expert, 1, hidden_dim]
)
{
    static FMoeKernel impl("fmoe_kernel_func", "fmoe_int8_g1u0_smf.co");
    impl.launch_kernel<uint8_t, uint16_t, true>(out,
                                                input,
                                                gate,
                                                down,
                                                sorted_token_ids,
                                                sorted_weight_buf,
                                                sorted_expert_ids,
                                                num_tokens_post_padded,
                                                topk,
                                                // quant args
                                                fc1_smooth_scale,
                                                fc1_scale,
                                                fc2_scale,
                                                fc2_smooth_scale);
}
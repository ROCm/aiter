// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_fmoe_configs.hpp"
#include "moe_op.h"
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <tuple>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_O;
    p2 _p0;
    void* ptr_X;
    p2 _p1;
    void* ptr_GU;
    p2 _p2;
    void* ptr_XC;
    p2 _p3;
    void* ptr_D;
    p2 _p4;
    void* ptr_XQ;
    p2 _p5;
    void* ptr_GUQ;
    p2 _p6;
    void* ptr_DQ;
    p2 _p7;
    void* ptr_SMQ;
    p2 _p8;
    void* ptr_STP;
    p2 _p9;
    void* ptr_SW;
    p2 _p10;
    void* ptr_SEP;
    p2 _p11;
    unsigned int dim;
    p3 _p12;
    unsigned int inter_dim;
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
    unsigned int total_tgs;
    p3 _p26;
    unsigned int ps_deno;
    p3 _p27;
};

class FMoeKernel
{
    private:
    hipModule_t module;
    hipFunction_t kernel_func;
    uint32_t sub_GU             = 512;
    bool is_int4                = false;
    uint32_t num_persistent_tgs = 0;
    const char* name            = nullptr;

    public:
    FMoeKernel(const char* name,
               const char* hsaco,
               uint32_t sub_GU             = 512,
               uint32_t num_persistent_tgs = 0)
    {
        const char* AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        std::cout << "[aiter] hipModuleLoad: " << (std::string(AITER_ASM_DIR) + hsaco).c_str()
                  << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
        std::cout << " Success" << std::endl;
        this->sub_GU             = sub_GU;
        this->num_persistent_tgs = num_persistent_tgs;
        this->name               = name;
    };

    const char* get_name() const { return name; }
    int get_num_persistent_tgs() { return num_persistent_tgs; }
    int get_sub_GU() { return sub_GU; }
    void set_4bit(bool is_4bit_) { is_int4 = is_4bit_; }

    template <typename T, typename T_O, bool switchGxy = false>
    void launch_kernel(torch::Tensor& out,               // [token_cnt, dim]
                       torch::Tensor& input,             // [token_cnt, dim] M,K
                       torch::Tensor& w1,                // [expert, inter_dim, dim] N,K
                       torch::Tensor& w2,                // [expert, dim, inter_dim]
                       torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
                       torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
                       torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
                       torch::Tensor& num_valid_ids,     // [1]
                       uint32_t topk,                    //
                       std::optional<torch::Tensor> input_dqn     = std::nullopt,
                       std::optional<torch::Tensor> w1_dqn        = std::nullopt,
                       std::optional<torch::Tensor> w2_dqn        = std::nullopt,
                       std::optional<torch::Tensor> w2_smooth_qnt = std::nullopt //
    )
    {
        int token_cnt       = out.size(0);
        int dim             = w2.size(1);
        int sub_X_cnt       = sorted_expert_ids.size(0);
        int eprt            = w1.size(0);
        int inter_dim       = w2.size(2) * (w2.size(1) / w1.size(2));
        uint32_t sub_GU     = this->sub_GU;
        uint32_t I_elemSize = sizeof(T);
        uint32_t O_elemSize = sizeof(T_O);

        int stride_X  = input.stride(0) * input.element_size();
        int stride_GU = dim * I_elemSize;
        int stride_D  = inter_dim * I_elemSize;
        if(is_int4)
        {
            stride_GU /= 2;
            stride_D /= 2;
        }
        int stride_expert_GU = stride_GU * inter_dim;
        int stride_expert_D  = stride_D * dim;
        int stride_expert_GUDQN =
            w1_dqn.has_value() ? w1_dqn.value().stride(0) * w1_dqn.value().element_size() : 0;
        int stride_expert_DDQN =
            w2_dqn.has_value() ? w2_dqn.value().stride(0) * w2_dqn.value().element_size() : 0;
        int stride_expert_SMTDQN = inter_dim * sizeof(float);
        int stride_O             = dim * O_elemSize;
        if(inter_dim * 2 == w1.size(1))
        {
            stride_expert_GU *= 2;
            // stride_expert_GUDQN *= 2;
        }

        KernelArgs args;
        size_t arg_size = sizeof(args);
        args.ptr_O      = out.data_ptr();
        args.ptr_X      = input.data_ptr();
        args.ptr_GU     = w1.data_ptr();
        args.ptr_XC     = num_valid_ids.data_ptr();
        args.ptr_D      = w2.data_ptr();
        if constexpr(std::is_same<T, uint8_t>::value)
        {
            args.ptr_XQ  = input_dqn.value().data_ptr();
            args.ptr_GUQ = w1_dqn.value().data_ptr();
            args.ptr_DQ  = w2_dqn.value().data_ptr();
            args.ptr_SMQ = w2_smooth_qnt.has_value() ? w2_smooth_qnt.value().data_ptr() : nullptr;
        }
        else
        {
            args.ptr_XQ  = nullptr;
            args.ptr_GUQ = nullptr;
            args.ptr_DQ  = nullptr;
            args.ptr_SMQ = nullptr;
        }
        args.ptr_STP   = sorted_token_ids.data_ptr();
        args.ptr_SW    = sorted_weights.data_ptr();
        args.ptr_SEP   = sorted_expert_ids.data_ptr();
        args.dim       = dim;
        args.inter_dim = inter_dim;
        args.token_cnt = token_cnt;
        args.eprt_cnt  = eprt;
        args.Xs        = stride_X;
        args.GUs       = stride_GU;
        args.Ds        = stride_D;
        args.Os        = stride_O;
        args.eGUs      = stride_expert_GU;
        args.eDs       = stride_expert_D;
        args.eGUQs     = stride_expert_GUDQN;
        args.eDQs      = stride_expert_DDQN;
        args.eSMQs     = stride_expert_SMTDQN;
        args.topk      = topk;
        args.ps_deno   = ((inter_dim + sub_GU - 1) / sub_GU);
        args.total_tgs = this->num_persistent_tgs / args.ps_deno * args.ps_deno;

        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &arg_size,
                          HIP_LAUNCH_PARAM_END};
        int bdx;
        int gdx;
        int gdy;
        int gdz;
        if(this->num_persistent_tgs != 0 && args.total_tgs > 0 &&
           (args.total_tgs % args.ps_deno) == 0) // ps
        {

            bdx = 256;
            gdx = this->num_persistent_tgs;
            gdy = 1;
            gdz = 1;
        }
        else // no-ps
        {
            bdx = 256;
            gdx = ((inter_dim + sub_GU - 1) / sub_GU);
            gdy = sub_X_cnt;
            gdz = 1;
        }
        // std::cout << "args.dim: " << args.dim << std::endl;
        // std::cout << "args.inter_dim: " << args.inter_dim << std::endl;
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
        // std::cout << "args.ps_deno: " << args.ps_deno << std::endl;
        // std::cout << "args.total_tgs: " << args.total_tgs << std::endl;
        // std::cout << "gdx: " << gdx << std::endl;
        // std::cout << "gdy: " << gdy << std::endl;

        const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        if constexpr(switchGxy)
        {
            HIP_CALL(hipModuleLaunchKernel(
                kernel_func, gdy, gdx, gdz, bdx, 1, 1, 0, stream, nullptr, (void**)&config));
        }
        else
        {
            HIP_CALL(hipModuleLaunchKernel(
                kernel_func, gdx, gdy, gdz, bdx, 1, 1, 0, stream, nullptr, (void**)&config));
        }
    };
};

FMoeKernel* get_heuristic_kernel(int inter_dim, int sub_X_cnt, CFG* cfgs, int smf = 0)
{
    FMoeKernel* impl_ptr        = nullptr;
    uint32_t num_cu             = get_num_cu_func();
    uint32_t empty_cu           = num_cu;
    uint32_t tg_num             = 0;
    uint32_t num_persistent_tgs = 0;
    uint32_t round              = 0xffffffff;
    std::string selectedKl      = "";
    int vskip                   = 0;
    static std::unordered_map<std::string, std::unique_ptr<FMoeKernel>> impl_ptr_map;

    const char* vs_env_value = std::getenv("AITER_ENABLE_VSKIP");
    if(vs_env_value != nullptr && std::string(vs_env_value) == "1")
        vskip = 1;

    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.vskip == vskip && cfg.smf == smf)
        {
            if((inter_dim % cfg.subGU_n) == 0)
            {
                tg_num = inter_dim / cfg.subGU_n *
                         sub_X_cnt; // hom many thread_groups are needed to handel inter_dim
                uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
                if(local_round < round || // fewer round is better
                   (local_round == round &&
                    (empty_cu > (local_round * num_cu - tg_num) || // fewer empty_cu is better
                     (empty_cu == (local_round * num_cu - tg_num) &&
                      cfg.ps == 1)))) // prefer PS kernel
                {
                    round      = local_round;
                    empty_cu   = local_round * num_cu - tg_num;
                    selectedKl = el.first;
                    if(cfg.ps == 1)
                        num_persistent_tgs = cfg.tg_num_perCU * num_cu;
                    else
                        num_persistent_tgs = 0;
                }
            }
        }
    }
    TORCH_CHECK(selectedKl != "",
                __func__,
                ": No suitable kernel found for inter_dim: ",
                inter_dim,
                ", sub_X_cnt: ",
                sub_X_cnt,
                ", smf: ",
                smf,
                ", vskip: ",
                vskip);
    auto it = cfgs->find(selectedKl);
    if(it != cfgs->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.name.c_str();
        const char* co_name = cfg.co_name.c_str();
        auto result         = impl_ptr_map.emplace(name, nullptr);
        if(result.second)
            result.first->second =
                std::make_unique<FMoeKernel>(name, co_name, cfg.subGU_n, num_persistent_tgs);
        impl_ptr = result.first->second.get();
    }
    else
        TORCH_CHECK(false, __func__, " not find kernel " + selectedKl);
    return impl_ptr;
}

int get_heuristic_tile(int inter_dim, int sub_X_cnt, const std::vector<int>& available_tiles)
{
    // int tiles[7] = {512, 448, 384, 320, 256, 192, 128};
    uint32_t num_cu   = get_num_cu_func();
    uint32_t empty_cu = num_cu;
    uint32_t tg_num   = 0;
    uint32_t round    = 0xffffffff;
    int selectedTile  = 0;

    for(auto tile : available_tiles)
    {
        if((inter_dim % tile) == 0)
        {
            tg_num               = inter_dim / tile * sub_X_cnt;
            uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
            if(local_round < round)
            {
                round        = local_round;
                selectedTile = tile;
                empty_cu     = local_round * num_cu - tg_num;
            }
            else if(local_round == round)
            {
                if(empty_cu > (local_round * num_cu - tg_num))
                {
                    round        = local_round;
                    selectedTile = tile;
                    empty_cu     = local_round * num_cu - tg_num;
                }
            }
        }
    }
    return selectedTile;
};

void fmoe(torch::Tensor& out,               // [token_cnt, dim]
          torch::Tensor& input,             // [token_cnt, dim] M,K
          torch::Tensor& gate,              // [expert, inter_dim, dim] N,K
          torch::Tensor& down,              // [expert, dim, inter_dim]
          torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
          torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
          torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
          torch::Tensor& num_valid_ids,     // [1]
          uint32_t topk)
{
    // g1u0
    FMoeKernel* impl_ptr = nullptr;
    if(input.dtype() == at::ScalarType::Half)
    {
        static FMoeKernel impl_f16("fmoe_kernel_func", "fmoe_f16.co");
        impl_ptr = &impl_f16;
    }
    else if(input.dtype() == at::ScalarType::BFloat16)
    {
        static FMoeKernel impl_b16("fmoe_kernel_func", "fmoe_b16.co");
        impl_ptr = &impl_b16;
    }
    TORCH_CHECK(
        impl_ptr != nullptr, __func__, ": unsupport current input type:", input.scalar_type());
    impl_ptr->launch_kernel<uint16_t, uint16_t>(out,
                                                input,
                                                gate,
                                                down,
                                                sorted_token_ids,
                                                sorted_weights,
                                                sorted_expert_ids,
                                                num_valid_ids,
                                                topk);
}

void fmoe_int8_g1u0(torch::Tensor& out,               // [token_cnt, dim]
                    torch::Tensor& input,             // [token_cnt, dim] M,K
                    torch::Tensor& gate,              // [expert, inter_dim, dim] N,K
                    torch::Tensor& down,              // [expert, dim, inter_dim]
                    torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
                    torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
                    torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
                    torch::Tensor& num_valid_ids,     // [1]
                    uint32_t topk,                    //
                    torch::Tensor& input_scale,       // [token_cnt, 1]
                    torch::Tensor& fc1_scale,         // [expert, 1, inter_dim]
                    torch::Tensor& fc2_scale,         // [expert, 1, dim]
                    torch::Tensor& fc2_smooth_scale,  // [expert, 1, inter_dim],
                    ActivationType activation)
{
    FMoeKernel* impl_ptr = nullptr;
    int inter_dim        = down.size(2);
    static std::unordered_map<std::string, std::unique_ptr<FMoeKernel>> impl_ptr_map;

    struct FMoeKernelConfig
    {
        std::string name;
        std::string co_name;
        int tile_size;
    };

    if(input.dtype() == at::ScalarType::Char || input.dtype() == at::ScalarType::Byte)
    {
        static std::unordered_map<int, FMoeKernelConfig> gelu_kernel_int8_configs = {
            {512,
             {"fmoe_int8_g1u0_subGU_512_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_512_gelu.co", 512}},
            {448,
             {"fmoe_int8_g1u0_subGU_448_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_448_gelu.co", 448}},
            {384,
             {"fmoe_int8_g1u0_subGU_384_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_384_gelu.co", 384}},
            {320,
             {"fmoe_int8_g1u0_subGU_320_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_320_gelu.co", 320}},
            {256,
             {"fmoe_int8_g1u0_subGU_256_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_256_gelu.co", 256}},
            {192,
             {"fmoe_int8_g1u0_subGU_192_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_192_gelu.co", 192}},
            {128,
             {"fmoe_int8_g1u0_subGU_128_gelu", "fmoe/gelu/fmoe_int8_g1u0_subGU_128_gelu.co", 128}}};

        static std::unordered_map<int, FMoeKernelConfig> silu_kernel_int8_configs = {
            {512, {"fmoe_int8_g1u0_subGU_512", "fmoe/silu/fmoe_int8_g1u0_subGU_512.co", 512}},
            {448, {"fmoe_int8_g1u0_subGU_448", "fmoe/silu/fmoe_int8_g1u0_subGU_448.co", 448}},
            {384, {"fmoe_int8_g1u0_subGU_384", "fmoe/silu/fmoe_int8_g1u0_subGU_384.co", 384}},
            {320, {"fmoe_int8_g1u0_subGU_320", "fmoe/silu/fmoe_int8_g1u0_subGU_320.co", 320}},
            {256, {"fmoe_int8_g1u0_subGU_256", "fmoe/silu/fmoe_int8_g1u0_subGU_256.co", 256}},
            {192, {"fmoe_int8_g1u0_subGU_192", "fmoe/silu/fmoe_int8_g1u0_subGU_192.co", 192}},
            {128, {"fmoe_int8_g1u0_subGU_128", "fmoe/silu/fmoe_int8_g1u0_subGU_128.co", 128}}};

        std::unordered_map<int, FMoeKernelConfig>* config_map = nullptr;
        if(activation == ActivationType::Gelu)
        {
            config_map = &gelu_kernel_int8_configs;
        }
        else if(activation == ActivationType::Silu)
        {
            config_map = &silu_kernel_int8_configs;
        }

        if(!config_map)
        {
            TORCH_CHECK(false, __func__, " Input only supput Int8!");
        }

        const int tiles[] = {512, 448, 384, 320, 256, 192, 128};
        int selectedTile  = 0;
        for(int tile : tiles)
        {
            if(inter_dim % tile == 0)
            {
                selectedTile = tile;
                break;
            }
        }
        if(selectedTile == 0)
        {
            TORCH_CHECK(false,
                        __func__,
                        " Unsupported inter_dim " + std::to_string(inter_dim) +
                            ", which should be divisible by 128, 192, 256, 320, 384, 448 or 512");
        }

        auto it = config_map->find(selectedTile);
        if(it != config_map->end())
        {
            const auto& config  = it->second;
            const char* name    = config.name.c_str();
            const char* co_name = config.co_name.c_str();

            auto result = impl_ptr_map.emplace(name, nullptr);
            if(result.second)
            {
                result.first->second =
                    std::make_unique<FMoeKernel>(name, co_name, config.tile_size);
            }
            impl_ptr = result.first->second.get();
        }
    }
    impl_ptr->launch_kernel<uint8_t, uint16_t>(out,
                                               input,
                                               gate,
                                               down,
                                               sorted_token_ids,
                                               sorted_weights,
                                               sorted_expert_ids,
                                               num_valid_ids,
                                               topk,
                                               // quant args
                                               input_scale,
                                               fc1_scale,
                                               fc2_scale,
                                               fc2_smooth_scale);
}
void fmoe_g1u1(torch::Tensor& out,                            // [token_cnt, dim]
               torch::Tensor& input,                          // [token_cnt, dim] M,K
               torch::Tensor& gate,                           // [expert, inter_dim*2, dim] N,K
               torch::Tensor& down,                           // [expert, dim, inter_dim]
               torch::Tensor& sorted_token_ids,               // [max_num_tokens_padded]
               torch::Tensor& sorted_weights,                 // [max_num_tokens_padded]
               torch::Tensor& sorted_expert_ids,              // [max_num_m_blocks]
               torch::Tensor& num_valid_ids,                  // [1]
               uint32_t topk,                                 //
               torch::Tensor& input_scale,                    // [token_cnt, 1]
               torch::Tensor& fc1_scale,                      // [expert, 1, inter_dim]
               torch::Tensor& fc2_scale,                      // [expert, 1, dim]
               std::optional<torch::Tensor> fc2_smooth_scale, // [expert, 1, inter_dim]
               ActivationType activation)
{
    struct FMoeKernelConfig
    {
        std::string name;
        std::string co_name;
        int tile_size;
    };

    FMoeKernel* impl_ptr = nullptr;
    CFG* config_map      = nullptr;
    int smf              = 0;
    int model_dim        = down.size(1);
    int inter_dim        = down.size(2);
    inter_dim *= model_dim / gate.size(2);
    int sub_X_cnt = sorted_expert_ids.size(0);
    static std::unordered_map<std::string, std::unique_ptr<FMoeKernel>> impl_ptr_map;
    if(gate.dtype() == at::ScalarType::UInt32 || gate.dtype() == at::ScalarType::Int) // int4
    {
        int selectedTile = get_heuristic_tile(
            inter_dim, sub_X_cnt, {512, 256, 128}); // todo,add tune interface here
        if(selectedTile == 512)
        {
            static FMoeKernel impl_int4_512(
                "fmoe_int4fp8_g1u1_subGU_512_gelu", "fmoe_int4fp8_g1u1_subGU_512_gelu.co", 512);
            impl_ptr = &impl_int4_512;
        }
        else if(selectedTile == 256)
        {
            static FMoeKernel impl_int4_256(
                "fmoe_int4fp8_g1u1_subGU_256_gelu", "fmoe_int4fp8_g1u1_subGU_256_gelu.co", 256);
            impl_ptr = &impl_int4_256;
        }
        else if(selectedTile == 128)
        {
            static FMoeKernel impl_int4_128(
                "fmoe_int4fp8_g1u1_subGU_128_gelu", "fmoe_int4fp8_g1u1_subGU_128_gelu.co", 128);
            impl_ptr = &impl_int4_128;
        }
        else
        {
            TORCH_CHECK(false,
                        __func__,
                        " Unsupported inter_dim " + std::to_string(inter_dim) +
                            ", which should be divisible by 128, 256, or 512");
        }
        impl_ptr->set_4bit(true);
    }

#if defined(__Float4_e2m1fn_x2)
    else if(input.dtype() == gate.dtype() &&
            (input.dtype() == torch::kFloat4_e2m1fn_x2 || input.dtype() == torch::kUInt8)) // fp4
    {
        if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenMXfp4_g1u1_silu;
        else if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenMXfp4_g1u1_gelu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_silu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_gelu;
        else
            TORCH_CHECK(false, __func__, " Not find proper cfg in pertokenMXfp4_g1u1. ");
        impl_ptr = get_heuristic_kernel(down.size(2), sub_X_cnt, config_map, smf);
        impl_ptr->set_4bit(true);
    }
#endif

    else if(input.dtype() == at::ScalarType::Char || input.dtype() == at::ScalarType::Byte) // int8
    {
        if(fc2_smooth_scale.has_value())
            smf = 2;
        if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_gelu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_gelu;
        else
            TORCH_CHECK(false, __func__, " Not find proper cfg in pertokenInt8_g1u1. ");
        impl_ptr = get_heuristic_kernel(down.size(2), sub_X_cnt, config_map, smf);
    }
    else if(input.dtype() == torch_fp8) // fp8
    {
        if(fc2_smooth_scale.has_value())
            smf = 2;
        if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_gelu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_gelu;
        else
            TORCH_CHECK(false, __func__, " Not find proper cfg in pertokenFp8_g1u1. ");
        impl_ptr = get_heuristic_kernel(down.size(2), sub_X_cnt, config_map, smf);
    }
    else
    {
        TORCH_CHECK(false, __func__, ": unsupport current input type:", input.scalar_type());
    }

    impl_ptr->launch_kernel<uint8_t, uint16_t>(out,
                                               input,
                                               gate,
                                               down,
                                               sorted_token_ids,
                                               sorted_weights,
                                               sorted_expert_ids,
                                               num_valid_ids,
                                               topk,
                                               // quant args
                                               input_scale,
                                               fc1_scale,
                                               fc2_scale,
                                               fc2_smooth_scale);
}

void fmoe_g1u1_tkw1(torch::Tensor& out,                            // [token_cnt, dim]
                    torch::Tensor& input,                          // [token_cnt, dim] M,K
                    torch::Tensor& gate,                           // [expert, inter_dim*2, dim] N,K
                    torch::Tensor& down,                           // [expert, dim, inter_dim]
                    torch::Tensor& sorted_token_ids,               // [max_num_tokens_padded]
                    torch::Tensor& sorted_weights,                 // [max_num_tokens_padded]
                    torch::Tensor& sorted_expert_ids,              // [max_num_m_blocks]
                    torch::Tensor& num_valid_ids,                  // [1]
                    uint32_t topk,                                 //
                    torch::Tensor& input_scale,                    // [token_cnt, 1]
                    torch::Tensor& fc1_scale,                      // [expert, 1, inter_dim]
                    torch::Tensor& fc2_scale,                      // [expert, 1, dim]
                    std::optional<torch::Tensor> fc2_smooth_scale, // [expert, 1, inter_dim]
                    ActivationType activation)
{
    FMoeKernel* impl_ptr = nullptr;
    CFG* config_map      = nullptr;

    const int token_cnt = input.size(0);
    const int block_m   = 32; // fmoe sorting kernel and fmoe kernel only support 32 for now
    const int estimated_sub_X_cnt = (token_cnt * topk + block_m - 1) / block_m;

    if(fc2_smooth_scale.has_value())
    {
        TORCH_CHECK(false, __func__, " Only supput non-smooth tkw1!");
    }

    if(input.dtype() == torch_fp8)
    {
        if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_silu_tkw1;
        else if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_gelu_tkw1;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_silu_tkw1;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_gelu_tkw1;
        else
            TORCH_CHECK(false, __func__, ": unsupport current activation type");
    }
    impl_ptr = get_heuristic_kernel(down.size(2), estimated_sub_X_cnt, config_map);
    impl_ptr->launch_kernel<uint8_t, uint16_t>(out,
                                               input,
                                               gate,
                                               down,
                                               sorted_token_ids,
                                               sorted_weights,
                                               sorted_expert_ids,
                                               num_valid_ids,
                                               topk,
                                               // quant args
                                               input_scale,
                                               fc1_scale,
                                               fc2_scale,
                                               fc2_smooth_scale);
}

void fmoe_int8_g1u0_a16(torch::Tensor& out,               // [token_cnt, dim]
                        torch::Tensor& input,             // [token_cnt, dim] M,K
                        torch::Tensor& gate,              // [expert, inter_dim, dim] N,K
                        torch::Tensor& down,              // [expert, dim, inter_dim]
                        torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
                        torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
                        torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
                        torch::Tensor& num_valid_ids,     // [1]
                        uint32_t topk,                    //
                        torch::Tensor& fc1_scale,         // [expert, 1, inter_dim]
                        torch::Tensor& fc2_scale,         // [expert, 1, dim]
                        torch::Tensor& fc1_smooth_scale,  // [expert, 1, dim]
                        torch::Tensor& fc2_smooth_scale   // [expert, 1, inter_dim]
)
{
    static FMoeKernel impl("fmoe_kernel_func", "fmoe_int8_g1u0_smf.co");
    impl.launch_kernel<uint8_t, uint16_t, true>(out,
                                                input,
                                                gate,
                                                down,
                                                sorted_token_ids,
                                                sorted_weights,
                                                sorted_expert_ids,
                                                num_valid_ids,
                                                topk,
                                                // quant args
                                                fc1_smooth_scale,
                                                fc1_scale,
                                                fc2_scale,
                                                fc2_smooth_scale);
}

void fmoe_g1u1_a16(torch::Tensor& out,               // [token_cnt, dim]
                   torch::Tensor& input,             // [token_cnt, dim] M,K
                   torch::Tensor& gate,              // [expert, inter_dim*2, dim] N,K
                   torch::Tensor& down,              // [expert, dim, inter_dim]
                   torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
                   torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor& num_valid_ids,     // [1]
                   uint32_t topk,                    //
                   torch::Tensor& fc1_scale,         // [expert, 1, inter_dim]
                   torch::Tensor& fc2_scale,         // [expert, 1, dim]
                   torch::Tensor& fc1_smooth_scale,  // [expert, 1, dim]
                   torch::Tensor& fc2_smooth_scale,  // [expert, 1, inter_dim]
                   ActivationType activation)
{
    FMoeKernel* impl_ptr = nullptr;
    int inter_dim        = down.size(2);
    int sub_X_cnt        = sorted_expert_ids.size(0);

    CFG* config_map = nullptr;
    if(gate.dtype() == at::ScalarType::Char || gate.dtype() == at::ScalarType::Byte) // int8
    {
        if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_gelu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_gelu;
        else
            TORCH_CHECK(
                false, __func__, "Unsupported output dtype or activation type for fmoe_g1u1_a16");
    }
    else if(gate.dtype() == torch_fp8) // fp8
    {
        if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::Half && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_gelu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_silu;
        else if(out.dtype() == at::ScalarType::BFloat16 && activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_gelu;
        else
            TORCH_CHECK(
                false, __func__, "Unsupported output dtype or activation type for fmoe_g1u1_a16");
    }
    else
        TORCH_CHECK(false, __func__, "Unsupported gate dtype for fmoe_g1u1_a16");

    impl_ptr = get_heuristic_kernel(down.size(2), sorted_expert_ids.size(0), config_map, 1);
    impl_ptr->launch_kernel<uint8_t, uint16_t, true>(out,
                                                     input,
                                                     gate,
                                                     down,
                                                     sorted_token_ids,
                                                     sorted_weights,
                                                     sorted_expert_ids,
                                                     num_valid_ids,
                                                     topk,
                                                     // quant args
                                                     fc1_smooth_scale,
                                                     fc1_scale,
                                                     fc2_scale,
                                                     fc2_smooth_scale);
}

void fmoe_fp8_blockscale_g1u1(torch::Tensor& out,               // [token_cnt, dim]
                              torch::Tensor& input,             // [token_cnt, dim] M,K
                              torch::Tensor& gate,              // [expert, inter_dim*2, dim] N,K
                              torch::Tensor& down,              // [expert, dim, inter_dim]
                              torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
                              torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
                              torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
                              torch::Tensor& num_valid_ids,     // [1]
                              uint32_t topk,                    //
                              torch::Tensor& input_scale,       // [expert, 1, dim]
                              torch::Tensor& fc1_scale,         // [expert, 1, inter_dim]
                              torch::Tensor& fc2_scale,         // [expert, 1, dim]
                              int fc_scale_blkn,
                              int fc_scale_blkk,
                              std::optional<torch::Tensor> fc2_smooth_scale,
                              ActivationType activation)
{
    FMoeKernel* impl_ptr     = nullptr;
    CFG* config_map          = nullptr;
    uint32_t num_cu          = get_num_cu_func();
    int inter_dim            = down.size(2);
    int sub_X_cnt            = sorted_expert_ids.size(0);
    const char* enable_vskip = std::getenv("AITER_ENABLE_VSKIP");

    if(out.dtype() == at::ScalarType::BFloat16 && inter_dim % 256 == 0 && fc_scale_blkn == 128 &&
       fc_scale_blkk == 128)
    {
        if(activation == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_blockscaleFp8_g1u1_silu;
        else if(activation == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_blockscaleFp8_g1u1_gelu;
        else
            TORCH_CHECK(
                false, __func__, "Unsupported activation type for fmoe_fp8_blockscale_g1u1");

        impl_ptr = get_heuristic_kernel(down.size(2), sorted_expert_ids.size(0), config_map);

        impl_ptr->launch_kernel<uint8_t, uint16_t, false>(out,
                                                          input,
                                                          gate,
                                                          down,
                                                          sorted_token_ids,
                                                          sorted_weights,
                                                          sorted_expert_ids,
                                                          num_valid_ids,
                                                          topk,
                                                          // quant args
                                                          input_scale,
                                                          fc1_scale,
                                                          fc2_scale,
                                                          fc2_smooth_scale);
    }
    else
        TORCH_CHECK(false, __func__, "Unsupported the type for fmoe_fp8_blockscale_g1u1");
}

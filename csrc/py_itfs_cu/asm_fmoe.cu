// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_fmoe_configs.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <memory>
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
    AiterAsmKernel kernel;
    uint32_t sub_GU             = 512;
    bool is_int4                = false;
    uint32_t num_persistent_tgs = 0;
    const char* name            = nullptr;
    // flat_mode: 0 = host-sorted, 1 = flat one-token-per-TG grid, 2 = emsort persistent grid
    int flat_mode = 0;

    public:
    FMoeKernel(const char* name,
               const char* hsaco,
               uint32_t sub_GU             = 512,
               uint32_t num_persistent_tgs = 0,
               int flat_mode               = 0) : kernel(name, hsaco)
    {
        this->sub_GU             = sub_GU;
        this->num_persistent_tgs = num_persistent_tgs;
        this->name               = name;
        this->flat_mode          = flat_mode;
    };

    const char* get_name() const { return name; }
    int get_num_persistent_tgs() { return num_persistent_tgs; }
    int get_sub_GU() { return sub_GU; }
    int get_flat_mode() const { return flat_mode; }
    void set_4bit(bool is_4bit_) { is_int4 = is_4bit_; }

    template <int I_elemSize, int O_elemSize, bool switchGxy = false>
    void launch_kernel(aiter_tensor_t* out,               // [token_cnt, dim]
                       aiter_tensor_t* input,             // [token_cnt, dim] M,K
                       aiter_tensor_t* w1,                // [expert, inter_dim, dim] N,K
                       aiter_tensor_t* w2,                // [expert, dim, inter_dim]
                       aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
                       aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
                       aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
                       aiter_tensor_t* num_valid_ids,     // [1]
                       uint32_t topk,                  //
                       aiter_tensor_t* input_dqn     = nullptr,
                       aiter_tensor_t* w1_dqn        = nullptr,
                       aiter_tensor_t* w2_dqn        = nullptr,
                       aiter_tensor_t* w2_smooth_qnt = nullptr,
                       hipStream_t stream         = nullptr)
    {
        int token_cnt       = out->size(0);
        int dim             = w2->size(1);
        int sub_X_cnt       = sorted_expert_ids->size(0);
        int eprt            = w1->size(0);
        int inter_dim       = w2->size(2) * (w2->size(1) / w1->size(2));
        uint32_t sub_GU     = this->sub_GU;

        int stride_X  = input->stride(0) * input->element_size();
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
            w1_dqn ? w1_dqn->stride(0) * w1_dqn->element_size() : 0;
        int stride_expert_DDQN =
            w2_dqn ? w2_dqn->stride(0) * w2_dqn->element_size() : 0;
        int stride_expert_SMTDQN = inter_dim * sizeof(float);
        int stride_O             = dim * O_elemSize;
        if(inter_dim * 2 == w1->size(1))
        {
            stride_expert_GU *= 2;
        }

        KernelArgs args;
        size_t arg_size = sizeof(args);
        args.ptr_O      = out->ptr;
        args.ptr_X      = input->ptr;
        args.ptr_GU     = w1->ptr;
        args.ptr_XC     = num_valid_ids->ptr;
        args.ptr_D      = w2->ptr;
        if constexpr(I_elemSize == 1)
        {
            args.ptr_XQ  = input_dqn ? input_dqn->ptr : nullptr;
            args.ptr_GUQ = w1_dqn ? w1_dqn->ptr : nullptr;
            args.ptr_DQ  = w2_dqn ? w2_dqn->ptr : nullptr;
            args.ptr_SMQ = w2_smooth_qnt ? w2_smooth_qnt->ptr : nullptr;
        }
        else
        {
            args.ptr_XQ  = nullptr;
            args.ptr_GUQ = nullptr;
            args.ptr_DQ  = nullptr;
            args.ptr_SMQ = nullptr;
        }
        args.ptr_STP   = sorted_token_ids->ptr;
        args.ptr_SW    = sorted_weights->ptr;
        args.ptr_SEP   = sorted_expert_ids->ptr;
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

        int bdx;
        int gdx;
        int gdy;
        int gdz;
        // flat_mode==1: one-TG-per-(token,topk) grid; no host moe_sort.
        // flat_mode==2: emsort persistent grid; no host moe_sort, same grid as ps.
        if(this->flat_mode == 1)
        {
            bdx = 256;
            gdx = ((inter_dim + sub_GU - 1) / sub_GU);
            gdy = static_cast<int>(topk);
            gdz = static_cast<int>(token_cnt);
        }
        else if(this->num_persistent_tgs != 0 && args.total_tgs > 0 &&
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

        if constexpr(switchGxy)
        {
            kernel.launch_kernel({&args,
                                  &arg_size,
                                  gdy, // gdx
                                  gdx, // gdy
                                  gdz, // gdz
                                  bdx, // bdx
                                  1,   // bdy
                                  1,   // bdz
                                  stream});
        }
        else
        {
            kernel.launch_kernel({&args,
                                  &arg_size,
                                  gdx, // gdx
                                  gdy, // gdy
                                  gdz, // gdz
                                  bdx, // bdx
                                  1,   // bdy
                                  1,   // bdz
                                  stream});
        }
    };
};

FMoeKernel* get_heuristic_kernel(
    int inter_dim, int sub_X_cnt, CFG* cfgs, int smf = 0, std::string kernel_name = "", int block_size_M = 32)
{
    FMoeKernel* impl_ptr        = nullptr;
    uint32_t num_cu             = get_num_cu_func();
    uint32_t empty_cu           = num_cu;
    uint32_t tg_num             = 0;
    uint32_t num_persistent_tgs = 0;
    uint32_t round              = 0xffffffff;
    std::string arch_id         = get_gpu_arch();
    std::string selectedKl      = kernel_name.empty() ? "" : arch_id + kernel_name;
    int vskip                   = 1;
    static SynchronizedCache<std::string_view, FMoeKernel> impl_ptr_map;

    const char* vs_env_value = std::getenv("AITER_ENABLE_VSKIP");
    if(vs_env_value != nullptr && std::string(vs_env_value) == "0")
        vskip = 0;
    if(selectedKl.empty())
    {
        for(const auto& el : *cfgs)
        {
            if(el.first.find(arch_id) != 0)
                continue;
            const auto& cfg = el.second;
            if(cfg.vskip == vskip && cfg.smf == smf && block_size_M == cfg.subGU_m &&
               cfg.flat == 0)
            {
                if((inter_dim % cfg.subGU_n) == 0)
                {
                    tg_num = inter_dim / cfg.subGU_n *
                             sub_X_cnt; // how many thread_groups are needed to handle inter_dim
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

        AITER_CHECK(selectedKl != "",
                    __func__,
                    ": No suitable kernel found for inter_dim: ",
                    inter_dim,
                    ", sub_X_cnt: ",
                    sub_X_cnt,
                    ", smf: ",
                    smf,
                    ", vskip: ",
                    vskip);
    }
    auto it = cfgs->find(selectedKl);
    if(it != cfgs->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        if(cfg.ps == 1)
            num_persistent_tgs = cfg.tg_num_perCU * num_cu;
        else
            num_persistent_tgs = 0;

        impl_ptr = &impl_ptr_map.get_or_create(name, [&]() {
            return FMoeKernel(name, co_name, cfg.subGU_n, num_persistent_tgs, cfg.flat);
        });
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + selectedKl);
    return impl_ptr;
}

int get_heuristic_tile(int inter_dim, int sub_X_cnt, const std::vector<int>& available_tiles)
{
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

AITER_CTYPES_ERROR_DECL;

// Direct BF16 G1U1 kernel for gfx1201 small-token-count MoE cases where
// M <= 16 and M * topk < num_experts.
template <int group_size, bool gate_is_shuffled>
__global__ void fmoe_g1u1_bf16_small_m_stage1_kernel(const hip_bfloat16* __restrict__ input,
                                                         const hip_bfloat16* __restrict__ gate,
                                                         const int32_t* __restrict__ topk_ids,
                                                         hip_bfloat16* __restrict__ act_out,
                                                         int token_cnt,
                                                         int model_dim,
                                                         int inter_dim,
                                                         int expert_cnt,
                                                         int topk)
{
    int route = blockIdx.x;
    int i_base = blockIdx.y * group_size;
    if(route >= token_cnt * topk || i_base >= inter_dim)
        return;

    int token = route / topk;
    int topk_idx = route - token * topk;
    int expert = topk_ids[token * topk + topk_idx];
    // Treat invalid expert ids as no-op routes and avoid out-of-bounds weight reads.
    if(expert < 0 || expert >= expert_cnt)
    {
        for(int g = 0; g < group_size; ++g)
        {
            int i = i_base + g;
            if(i < inter_dim)
                act_out[route * inter_dim + i] = hip_bfloat16(0.0f);
        }
        return;
    }

    const hip_bfloat16* x = input + token * model_dim;
    __shared__ float gate_smem[group_size][256];
    __shared__ float up_smem[group_size][256];
    float gate_acc[group_size] = {};
    float up_acc[group_size] = {};
    int gate_nblock_stride = (model_dim / 32) * 512;

    for(int k = threadIdx.x; k < model_dim; k += blockDim.x)
    {
        float xv = static_cast<float>(x[k]);
        for(int g = 0; g < group_size; ++g)
        {
            int i = i_base + g;
            if(i < inter_dim)
            {
                if constexpr(gate_is_shuffled)
                {
                    int kblock = k / 32;
                    int k_in_32 = k - kblock * 32;
                    int klane = k_in_32 / 8;
                    int k8 = k_in_32 - klane * 8;
                    int gate_row = expert * (2 * inter_dim) + i;
                    int up_row = gate_row + inter_dim;
                    int gate_nblock = gate_row / 16;
                    int gate_nintra = gate_row - gate_nblock * 16;
                    int up_nblock = up_row / 16;
                    int up_nintra = up_row - up_nblock * 16;
                    int gate_idx = gate_nblock * gate_nblock_stride + kblock * 512 +
                                   klane * 128 + gate_nintra * 8 + k8;
                    int up_idx = up_nblock * gate_nblock_stride + kblock * 512 +
                                 klane * 128 + up_nintra * 8 + k8;
                    gate_acc[g] += xv * static_cast<float>(gate[gate_idx]);
                    up_acc[g] += xv * static_cast<float>(gate[up_idx]);
                }
                else
                {
                    const hip_bfloat16* w_gate =
                        gate + expert * (2 * inter_dim * model_dim) + i * model_dim;
                    const hip_bfloat16* w_up = gate + expert * (2 * inter_dim * model_dim) +
                                               (inter_dim + i) * model_dim;
                    gate_acc[g] += xv * static_cast<float>(w_gate[k]);
                    up_acc[g] += xv * static_cast<float>(w_up[k]);
                }
            }
        }
    }

    for(int g = 0; g < group_size; ++g)
    {
        gate_smem[g][threadIdx.x] = gate_acc[g];
        up_smem[g][threadIdx.x] = up_acc[g];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int g = 0; g < group_size; ++g)
            {
                gate_smem[g][threadIdx.x] += gate_smem[g][threadIdx.x + stride];
                up_smem[g][threadIdx.x] += up_smem[g][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x != 0)
        return;

    for(int g = 0; g < group_size; ++g)
    {
        int i = i_base + g;
        if(i < inter_dim)
        {
            float gate_val = gate_smem[g][0];
            float up_val = up_smem[g][0];
            float silu = gate_val / (1.0f + expf(-gate_val));
            act_out[route * inter_dim + i] = hip_bfloat16(silu * up_val);
        }
    }
}

template <int group_size, bool down_is_shuffled>
__global__ void fmoe_g1u1_bf16_small_m_stage2_kernel(const hip_bfloat16* __restrict__ act,
                                                         const hip_bfloat16* __restrict__ down,
                                                         const int32_t* __restrict__ topk_ids,
                                                         const float* __restrict__ topk_weights,
                                                         hip_bfloat16* __restrict__ out,
                                                         int token_cnt,
                                                         int model_dim,
                                                         int inter_dim,
                                                         int expert_cnt,
                                                         int topk)
{
    int token = blockIdx.x;
    int h_base = blockIdx.y * group_size;
    if(token >= token_cnt || h_base >= model_dim)
        return;

    __shared__ float smem[group_size][256];
    float acc[group_size] = {};
    int total = topk * inter_dim;
    int down_nblock_stride = (inter_dim / 32) * 512;
    for(int idx = threadIdx.x; idx < total; idx += blockDim.x)
    {
        int topk_idx = idx / inter_dim;
        int i = idx - topk_idx * inter_dim;
        int route = token * topk + topk_idx;
        int expert = topk_ids[route];
        if(expert < 0 || expert >= expert_cnt)
            continue;

        float route_weight = topk_weights[route];
        const hip_bfloat16* act_route = act + route * inter_dim;
        float act_val = static_cast<float>(act_route[i]) * route_weight;
        for(int g = 0; g < group_size; ++g)
        {
            int h = h_base + g;
            if(h < model_dim)
            {
                if constexpr(down_is_shuffled)
                {
                    int iblock = i / 32;
                    int i_in_32 = i - iblock * 32;
                    int ilane = i_in_32 / 8;
                    int i8 = i_in_32 - ilane * 8;
                    int down_row = expert * model_dim + h;
                    int down_nblock = down_row / 16;
                    int down_nintra = down_row - down_nblock * 16;
                    int down_idx = down_nblock * down_nblock_stride + iblock * 512 +
                                   ilane * 128 + down_nintra * 8 + i8;
                    acc[g] += act_val * static_cast<float>(down[down_idx]);
                }
                else
                {
                    const hip_bfloat16* w2 =
                        down + expert * (model_dim * inter_dim) + h * inter_dim;
                    acc[g] += act_val * static_cast<float>(w2[i]);
                }
            }
        }
    }

    for(int g = 0; g < group_size; ++g)
        smem[g][threadIdx.x] = acc[g];
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(threadIdx.x < stride)
        {
            for(int g = 0; g < group_size; ++g)
                smem[g][threadIdx.x] += smem[g][threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x != 0)
        return;

    for(int g = 0; g < group_size; ++g)
    {
        int h = h_base + g;
        if(h < model_dim)
            out[token * model_dim + h] = hip_bfloat16(smem[g][0]);
    }
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_g1u1_bf16_small_m,
    (
    aiter_tensor_t* out,
    aiter_tensor_t* input,
    aiter_tensor_t* gate,
    aiter_tensor_t* down,
    aiter_tensor_t* topk_ids,
    aiter_tensor_t* topk_weights,
    aiter_tensor_t* act_workspace,
    int topk,
    bool gate_is_shuffled,
    bool down_is_shuffled,
    hipStream_t stream),
    (out,
     input,
     gate,
     down,
     topk_ids,
     topk_weights,
     act_workspace,
     topk,
     gate_is_shuffled,
     down_is_shuffled,
     stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16, __func__, " only supports bf16 output");
    AITER_CHECK(input->dtype() == AITER_DTYPE_bf16 && gate->dtype() == AITER_DTYPE_bf16 &&
                    down->dtype() == AITER_DTYPE_bf16,
                __func__,
                " only supports bf16 input and weights");
    AITER_CHECK(topk_ids->dtype() == AITER_DTYPE_i32, __func__, " topk_ids must be int32");
    AITER_CHECK(topk_weights->dtype() == AITER_DTYPE_fp32, __func__, " topk_weights must be fp32");
    AITER_CHECK(act_workspace->dtype() == AITER_DTYPE_bf16,
                __func__,
                " act_workspace must be bf16");

    int token_cnt = input->size(0);
    int model_dim = input->size(1);
    int expert_cnt = gate->size(0);
    int inter_dim = down->size(2);
    AITER_CHECK(gate->size(1) == inter_dim * 2,
                __func__,
                " expects G1U1 gate shape [E, 2*inter_dim, model_dim]");
    AITER_CHECK(gate->size(2) == model_dim && down->size(1) == model_dim,
                __func__,
                " model_dim mismatch");
    AITER_CHECK(topk_ids->size(0) == token_cnt && topk_ids->size(1) == topk,
                __func__,
                " topk_ids shape mismatch");
    AITER_CHECK(topk_weights->size(0) == token_cnt && topk_weights->size(1) == topk,
                __func__,
                " topk_weights shape mismatch");
    AITER_CHECK(act_workspace->size(0) == token_cnt * topk && act_workspace->size(1) == inter_dim,
                __func__,
                " act_workspace shape mismatch");

    constexpr int threads = 256;
    constexpr int group_size = 8;
    dim3 grid1(token_cnt * topk, (inter_dim + group_size - 1) / group_size);
    if(gate_is_shuffled)
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage1_kernel<group_size, true>),
                           grid1,
                           dim3(threads),
                           0,
                           stream,
                           reinterpret_cast<const hip_bfloat16*>(input->ptr),
                           reinterpret_cast<const hip_bfloat16*>(gate->ptr),
                           reinterpret_cast<const int32_t*>(topk_ids->ptr),
                           reinterpret_cast<hip_bfloat16*>(act_workspace->ptr),
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage1_kernel<group_size, false>),
                           grid1,
                           dim3(threads),
                           0,
                           stream,
                           reinterpret_cast<const hip_bfloat16*>(input->ptr),
                           reinterpret_cast<const hip_bfloat16*>(gate->ptr),
                           reinterpret_cast<const int32_t*>(topk_ids->ptr),
                           reinterpret_cast<hip_bfloat16*>(act_workspace->ptr),
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    HIP_CALL_LAUNCH(hipGetLastError());

    dim3 grid2(token_cnt, (model_dim + group_size - 1) / group_size);
    if(down_is_shuffled)
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage2_kernel<group_size, true>),
                           grid2,
                           dim3(threads),
                           0,
                           stream,
                           reinterpret_cast<const hip_bfloat16*>(act_workspace->ptr),
                           reinterpret_cast<const hip_bfloat16*>(down->ptr),
                           reinterpret_cast<const int32_t*>(topk_ids->ptr),
                           reinterpret_cast<const float*>(topk_weights->ptr),
                           reinterpret_cast<hip_bfloat16*>(out->ptr),
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((fmoe_g1u1_bf16_small_m_stage2_kernel<group_size, false>),
                           grid2,
                           dim3(threads),
                           0,
                           stream,
                           reinterpret_cast<const hip_bfloat16*>(act_workspace->ptr),
                           reinterpret_cast<const hip_bfloat16*>(down->ptr),
                           reinterpret_cast<const int32_t*>(topk_ids->ptr),
                           reinterpret_cast<const float*>(topk_weights->ptr),
                           reinterpret_cast<hip_bfloat16*>(out->ptr),
                           token_cnt,
                           model_dim,
                           inter_dim,
                           expert_cnt,
                           topk);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    // g1u0
    FMoeKernel* impl_ptr = nullptr;
    if(input->dtype() == AITER_DTYPE_fp16)
    {
        static FMoeKernel impl_f16("fmoe_kernel_func", "fmoe_f16.co");
        impl_ptr = &impl_f16;
    }
    else if(input->dtype() == AITER_DTYPE_bf16)
    {
        static FMoeKernel impl_b16("fmoe_kernel_func", "fmoe_b16.co");
        impl_ptr = &impl_b16;
    }
    AITER_CHECK(
        impl_ptr != nullptr, __func__, ": unsupport current input type:", AiterDtype_to_str(input->dtype()));
    impl_ptr->launch_kernel<2, 2>(out,
                                  input,
                                  gate,
                                  down,
                                  sorted_token_ids,
                                  sorted_weights,
                                  sorted_expert_ids,
                                  num_valid_ids,
                                  topk,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_int8_g1u0,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,                       //
    aiter_tensor_t* input_scale,       // [token_cnt, 1]
    aiter_tensor_t* fc1_scale,         // [expert, 1, inter_dim]
    aiter_tensor_t* fc2_scale,         // [expert, 1, dim]
    aiter_tensor_t* fc2_smooth_scale,  // [expert, 1, inter_dim]
    int activation,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale, fc2_smooth_scale, activation, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    FMoeKernel* impl_ptr = nullptr;
    int inter_dim        = down->size(2);
    static SynchronizedCache<std::string_view, FMoeKernel> impl_ptr_map;

    struct FMoeKernelConfig
    {
        std::string name;
        std::string co_name;
        int tile_size;
    };

    if(input->dtype() == AITER_DTYPE_i8 || input->dtype() == AITER_DTYPE_u8)
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
        if(act == ActivationType::Gelu)
        {
            config_map = &gelu_kernel_int8_configs;
        }
        else if(act == ActivationType::Silu)
        {
            config_map = &silu_kernel_int8_configs;
        }

        if(!config_map)
        {
            AITER_CHECK(false, __func__, " Input only support Int8!");
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
            AITER_CHECK(false,
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

            impl_ptr = &impl_ptr_map.get_or_create(
                name, [&]() { return FMoeKernel(name, co_name, config.tile_size); });
        }
    }
    impl_ptr->launch_kernel<1, 2>(out,
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
                                  fc2_smooth_scale,
                                  stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_g1u1,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim*2, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,                       //
    aiter_tensor_t* input_scale,       // [token_cnt, 1]
    aiter_tensor_t* fc1_scale,         // [expert, 1, inter_dim]
    aiter_tensor_t* fc2_scale,         // [expert, 1, dim]
    const char* kernel_name,
    aiter_tensor_t* fc2_smooth_scale,  // [expert, 1, inter_dim]
    int activation,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale, kernel_name, fc2_smooth_scale, activation, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    struct FMoeKernelConfig
    {
        std::string name;
        std::string co_name;
        int tile_size;
    };

    FMoeKernel* impl_ptr = nullptr;
    CFG* config_map      = nullptr;
    int smf              = 0;
    int model_dim        = down->size(1);
    int inter_dim        = down->size(2);
    inter_dim *= model_dim / gate->size(2);
    int sub_X_cnt = sorted_expert_ids->size(0);
    static SynchronizedCache<std::string_view, FMoeKernel> impl_ptr_map;
    std::string kernel_name_str = kernel_name ? kernel_name : "";

    if(gate->dtype() == AITER_DTYPE_u32 || gate->dtype() == AITER_DTYPE_i32) // int4
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
            AITER_CHECK(false,
                        __func__,
                        " Unsupported inter_dim " + std::to_string(inter_dim) +
                            ", which should be divisible by 128, 256, or 512");
        }
        impl_ptr->set_4bit(true);
    }
    else if(input->dtype() == gate->dtype() && input->dtype() == AITER_DTYPE_fp4x2) // fp4
    {
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenMXfp4_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenMXfp4_g1u1_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_gelu;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenMXfp4_g1u1. ");
        impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, smf, kernel_name_str);
        impl_ptr->set_4bit(true);
    }
    else if((input->dtype() == AITER_DTYPE_bf16 || input->dtype() == AITER_DTYPE_fp16) &&
            gate->dtype() == AITER_DTYPE_fp4x2) // bf16/fp16 X + MXFP4 weights (in-kernel X quant)
    {
        // X stays bf16/fp16; the asm kernel dynamic-quantizes X to MXFP4 internally
        // (xbf16 path), so no activation scale is consumed. Weights are still fp4
        // (set_4bit), and we reuse the pertokenMXfp4 config map keyed by out dtype.
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenMXfp4_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenMXfp4_g1u1_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_gelu;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenMXfp4_g1u1 (bf16 X). ");
        impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, smf, kernel_name_str);
        impl_ptr->set_4bit(true);
    }
    else if(input->dtype() == AITER_DTYPE_i8 || input->dtype() == AITER_DTYPE_u8) // int8
    {
        if(fc2_smooth_scale)
            smf = 2;
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_gelu;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenInt8_g1u1. ");
        impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, smf, kernel_name_str);
    }
    else if(input->dtype() == AITER_DTYPE_fp8) // fp8
    {
        if(fc2_smooth_scale)
            smf = 2;
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_gelu;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenFp8_g1u1. ");
        impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, smf, kernel_name_str);
    }
    else
    {
        AITER_CHECK(false, __func__, ": unsupport current input type:", AiterDtype_to_str(input->dtype()));
    }

    impl_ptr->launch_kernel<1, 2>(out,
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
                                  fc2_smooth_scale,
                                  stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_g1u1_tkw1,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim*2, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,                       //
    aiter_tensor_t* input_scale,       // [token_cnt, 1]
    aiter_tensor_t* fc1_scale,         // [expert, 1, inter_dim]
    aiter_tensor_t* fc2_scale,         // [expert, 1, dim]
    const char* kernel_name,
    aiter_tensor_t* fc2_smooth_scale,  // [expert, 1, inter_dim]
    int activation,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale, kernel_name, fc2_smooth_scale, activation, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    FMoeKernel* impl_ptr = nullptr;
    CFG* config_map      = nullptr;
    std::string kernel_name_str = kernel_name ? kernel_name : "";

    const int token_cnt = input->size(0);
    const int block_m   = 32; // fmoe sorting kernel and fmoe kernel only support 32 for now
    const int estimated_sub_X_cnt = (token_cnt * topk + block_m - 1) / block_m;
    int model_dim                 = down->size(1);
    int inter_dim                 = down->size(2);
    inter_dim *= model_dim / gate->size(2);

    if(fc2_smooth_scale)
    {
        AITER_CHECK(false, __func__, " Only support non-smooth tkw1!");
    }

    if(input->dtype() == AITER_DTYPE_fp8)
    {
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_silu_tkw1;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_gelu_tkw1;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_silu_tkw1;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_gelu_tkw1;
        else
            AITER_CHECK(false, __func__, ": unsupport current activation type");
    }
    impl_ptr = get_heuristic_kernel(inter_dim, estimated_sub_X_cnt, config_map, 0, kernel_name_str);
    impl_ptr->launch_kernel<1, 2>(out,
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
                                  fc2_smooth_scale,
                                  stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_int8_g1u0_a16,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,                       //
    aiter_tensor_t* fc1_scale,         // [expert, 1, inter_dim]
    aiter_tensor_t* fc2_scale,         // [expert, 1, dim]
    aiter_tensor_t* fc1_smooth_scale,  // [expert, 1, dim]
    aiter_tensor_t* fc2_smooth_scale,  // [expert, 1, inter_dim]
    int activation,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, activation, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    FMoeKernel* impl_ptr = nullptr;
    CFG* config_map      = nullptr;
    int inter_dim        = down->size(2);
    int sub_X_cnt        = sorted_expert_ids->size(0);

    if(gate->dtype() == AITER_DTYPE_i8 || gate->dtype() == AITER_DTYPE_u8)
    {
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u0_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u0_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u0_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u0_gelu;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenInt8_g1u0. ");
    }
    else
        AITER_CHECK(false, __func__, "Unsupported gate dtype for fmoe_int8_g1u0_a16");

    impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, 1);
    impl_ptr->launch_kernel<1, 2, true>(out,
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
                                        fc2_smooth_scale,
                                        stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_g1u1_a16,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim*2, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,                       //
    aiter_tensor_t* fc1_scale,         // [expert, 1, inter_dim]
    aiter_tensor_t* fc2_scale,         // [expert, 1, dim]
    aiter_tensor_t* fc1_smooth_scale,  // [expert, 1, dim]
    aiter_tensor_t* fc2_smooth_scale,  // [expert, 1, inter_dim]
    int activation,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale, activation, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    FMoeKernel* impl_ptr = nullptr;
    int inter_dim        = down->size(2);
    int sub_X_cnt        = sorted_expert_ids->size(0);

    CFG* config_map = nullptr;
    if(gate->dtype() == AITER_DTYPE_i8 || gate->dtype() == AITER_DTYPE_u8) // int8
    {
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenInt8_g1u1_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenInt8_g1u1_gelu;
        else
            AITER_CHECK(
                false, __func__, "Unsupported output dtype or activation type for fmoe_g1u1_a16");
    }
    else if(gate->dtype() == AITER_DTYPE_fp8) // fp8
    {
        if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_fp16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_fp16_pertokenFp8_g1u1_gelu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Silu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_silu;
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Gelu)
            config_map = &cfg_fmoe_bf16_pertokenFp8_g1u1_gelu;
        else
            AITER_CHECK(
                false, __func__, "Unsupported output dtype or activation type for fmoe_g1u1_a16");
    }
    else
        AITER_CHECK(false, __func__, "Unsupported gate dtype for fmoe_g1u1_a16");

    impl_ptr = get_heuristic_kernel(inter_dim, sorted_expert_ids->size(0), config_map, 1);
    impl_ptr->launch_kernel<1, 2, true>(out,
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
                                        fc2_smooth_scale,
                                        stream);
}

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    fmoe_fp8_blockscale_g1u1,
    (
    aiter_tensor_t* out,               // [token_cnt, dim]
    aiter_tensor_t* input,             // [token_cnt, dim] M,K
    aiter_tensor_t* gate,              // [expert, inter_dim*2, dim] N,K
    aiter_tensor_t* down,              // [expert, dim, inter_dim]
    aiter_tensor_t* sorted_token_ids,  // [max_num_tokens_padded]
    aiter_tensor_t* sorted_weights,    // [max_num_tokens_padded]
    aiter_tensor_t* sorted_expert_ids, // [max_num_m_blocks]
    aiter_tensor_t* num_valid_ids,     // [1]
    int topk,                       //
    aiter_tensor_t* input_scale,       // [expert, 1, dim]
    aiter_tensor_t* fc1_scale,         // [expert, 1, inter_dim]
    aiter_tensor_t* fc2_scale,         // [expert, 1, dim]
    const char* kernel_name,
    int fc_scale_blkn,
    int fc_scale_blkk,
    aiter_tensor_t* fc2_smooth_scale,  // [expert, 1, inter_dim]
    int activation,
    int block_size_M,
    hipStream_t stream),
    (out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale, kernel_name, fc_scale_blkn, fc_scale_blkk, fc2_smooth_scale, activation, block_size_M, stream))
{
    const HipDeviceGuard device_guard(input->device_id);
    ActivationType act = static_cast<ActivationType>(activation);
    FMoeKernel* impl_ptr     = nullptr;
    CFG* config_map          = nullptr;
    uint32_t num_cu          = get_num_cu_func();
    int inter_dim            = down->size(2);
    int sub_X_cnt            = sorted_expert_ids->size(0);
    std::string kernel_name_str = kernel_name ? kernel_name : "";

    if(out->dtype() == AITER_DTYPE_bf16 && inter_dim % 128 == 0 && fc_scale_blkn == 128 &&
       fc_scale_blkk == 128)
    {
        bool xquant = (input->dtype() == AITER_DTYPE_bf16);
        if(act == ActivationType::Silu)
            config_map = xquant ? &cfg_fmoe_bf16_blockscaleBf16_g1u1_silu : &cfg_fmoe_bf16_blockscaleFp8_g1u1_silu;
        else if(act == ActivationType::Gelu)
            config_map = xquant ? &cfg_fmoe_bf16_blockscaleBf16_g1u1_gelu : &cfg_fmoe_bf16_blockscaleFp8_g1u1_gelu;
        else
            AITER_CHECK(
                false, __func__, "Unsupported activation type for fmoe_fp8_blockscale_g1u1");

        impl_ptr =
            get_heuristic_kernel(inter_dim, sorted_expert_ids->size(0), config_map, 0, kernel_name_str, block_size_M);
        impl_ptr->launch_kernel<1, 2, false>(out,
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
                                             fc2_smooth_scale,
                                             stream);
    }
    else
        AITER_CHECK(false, __func__, "Unsupported the type for fmoe_fp8_blockscale_g1u1");
}

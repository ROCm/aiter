// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_fmoe_configs.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <numeric>
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
    uint32_t tgs_align          = 1;
    const char* name            = nullptr;
    // flat_mode: 0 = host-sorted, 1 = flat one-token-per-TG grid, 2 = emsort persistent grid
    int flat_mode = 0;
    // Manifest limits. 0 max_tokens means no separate token cap.
    uint32_t max_tokens      = 0;
    uint32_t max_assignments = 0;
    uint32_t max_experts     = 0;

    public:
    FMoeKernel(const char* name,
               const char* hsaco,
               uint32_t sub_GU             = 512,
               uint32_t num_persistent_tgs = 0,
               int flat_mode               = 0,
               uint32_t tgs_align          = 1,
               uint32_t max_tokens         = 0,
               uint32_t max_assignments    = 0,
               uint32_t max_experts        = 0) : kernel(name, hsaco)
    {
        this->sub_GU             = sub_GU;
        this->num_persistent_tgs = num_persistent_tgs;
        this->tgs_align          = std::max(tgs_align, 1u);
        this->name               = name;
        this->flat_mode          = flat_mode;
        this->max_tokens         = max_tokens;
        this->max_assignments    = max_assignments;
        this->max_experts        = max_experts;
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
        // Kernels that remap workgroups across XCDs need total_tgs to also be a
        // multiple of tgs_align (the XCD count), or the remap skips work.
        const uint32_t tgs_unit = std::lcm(args.ps_deno, this->tgs_align);
        args.total_tgs          = this->num_persistent_tgs / tgs_unit * tgs_unit;

        int bdx;
        int gdx;
        int gdy;
        int gdz;
        // flat_mode==1: one-TG-per-(token,topk) grid; no host moe_sort.
        // flat_mode==2: EMSORT consumes raw topk and uses the configured persistent
        // TG pool as a 2D grid: hidden slices on X, expert owners on Y. The shader
        // derives expert ownership from tgy and loops over its compacted queue.
        if(this->flat_mode == 1)
        {
            bdx = 256;
            gdx = ((inter_dim + sub_GU - 1) / sub_GU);
            gdy = static_cast<int>(topk);
            gdz = static_cast<int>(token_cnt);
        }
        else if(this->flat_mode == 2)
        {
            // Each EMSORT binary declares its own queue in the manifest
            // (max_tokens, max_assignments, max_experts). A zero max_tokens
            // means that binary has no separate token cap.
            //
            // The producer is the last Y workgroup. Every other workgroup spins
            // on its ready flag, so this launch assumes the whole grid is
            // resident together. num_persistent_tgs is tg_num_perCU * num_cu,
            // rounded down to lcm(ps_deno, tgs_align), and the grid below is
            // exactly that pool. Occupancy (whether tg_num_perCU fits in a CU)
            // and room left by another stream or a CU mask are not queried.
            AITER_CHECK(this->max_assignments > 0 && this->max_experts > 0,
                        __func__,
                        ": EMSORT manifest row must set max_assignments and max_experts for ",
                        this->name);
            if(this->max_tokens > 0)
            {
                AITER_CHECK(token_cnt <= static_cast<int>(this->max_tokens),
                            __func__,
                            ": EMSORT ",
                            this->name,
                            " supports token_cnt <= ",
                            this->max_tokens,
                            "; got ",
                            token_cnt);
            }
            AITER_CHECK(static_cast<uint64_t>(token_cnt) * topk <= this->max_assignments,
                        __func__,
                        ": EMSORT ",
                        this->name,
                        " supports token_cnt*topk <= ",
                        this->max_assignments,
                        "; got ",
                        static_cast<uint64_t>(token_cnt) * topk);
            AITER_CHECK(eprt <= static_cast<int>(this->max_experts),
                        __func__,
                        ": EMSORT ",
                        this->name,
                        " supports at most ",
                        this->max_experts,
                        " experts; got ",
                        eprt);
            AITER_CHECK(inter_dim % sub_GU == 0,
                        __func__,
                        ": EMSORT requires inter_dim divisible by sub_GU; got inter_dim=",
                        inter_dim,
                        ", sub_GU=",
                        sub_GU);
            AITER_CHECK(this->num_persistent_tgs > 0 && args.total_tgs > 0,
                        __func__,
                        ": EMSORT requires the dispatcher resident grid; got total_tgs=",
                        args.total_tgs,
                        ", pool=",
                        this->num_persistent_tgs);
            gdx = inter_dim / sub_GU;
            const int emsort_tgs = static_cast<int>(args.total_tgs);
            AITER_CHECK(gdx > 0 && (emsort_tgs % gdx) == 0,
                        __func__,
                        ": EMSORT requires total_tgs divisible by inter_dim/sub_GU; got total_tgs=",
                        emsort_tgs,
                        ", slices=",
                        gdx);
            bdx = 256;
            gdy = emsort_tgs / gdx;
            gdz = 1;
            const uint32_t launched =
                static_cast<uint32_t>(gdx) * static_cast<uint32_t>(gdy);
            // Documents the dispatcher assumption above. gdy is total_tgs / gdx
            // and total_tgs is num_persistent_tgs rounded down, so launched is
            // total_tgs and this comparison holds for every input that reached it.
            AITER_CHECK(launched <= this->num_persistent_tgs,
                        __func__,
                        ": EMSORT grid must be fully resident; launched ",
                        launched,
                        " workgroups, dispatcher pool ",
                        this->num_persistent_tgs);
        }
        else if(this->num_persistent_tgs != 0 && args.total_tgs > 0 &&
                (args.total_tgs % args.ps_deno) == 0) // ps
        {

            bdx = 256;
            gdx = args.total_tgs;
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

static int get_vskip()
{
    const char* vs_env_value = std::getenv("AITER_ENABLE_VSKIP");
    return (vs_env_value != nullptr && std::string(vs_env_value) == "0") ? 0 : 1;
}

// launch_kernel rounds total_tgs down to a multiple of lcm(ps_deno, tgs_align).
static bool ps_grid_fits(const CFG::mapped_type& cfg, int inter_dim)
{
    const int ps_deno = (inter_dim + cfg.subGU_n - 1) / cfg.subGU_n;
    return cfg.ps == 0 || std::lcm(ps_deno, std::max(cfg.tgs_align, 1)) <=
                              cfg.tg_num_perCU * static_cast<int>(get_num_cu_func());
}

// model_dim must be a multiple of the kernel's dim_align and at least its dim_min
// (0 = unconstrained).
static bool kernel_fits(
    const CFG::mapped_type& cfg, int vskip, int smf, int block_size_M, int inter_dim, int model_dim)
{
    if(cfg.smf != smf || inter_dim % cfg.subGU_n != 0 ||
       model_dim % std::max(cfg.dim_align, 1) != 0 || model_dim < cfg.dim_min ||
       !ps_grid_fits(cfg, inter_dim))
        return false;
    // flat==1 rows are the 16xN tiles. Their subGU_m is 16 and vskip is 0, so the
    // checks below would hide the only rows that divide inter_dim such as 160 and 320.
    if(cfg.flat == 1)
        return true;
    return cfg.vskip == vskip && block_size_M == cfg.subGU_m && cfg.flat == 0;
}

FMoeKernel* get_heuristic_kernel(int inter_dim,
                                 int sub_X_cnt,
                                 CFG* cfgs,
                                 int smf                 = 0,
                                 std::string kernel_name = "",
                                 int block_size_M        = 32,
                                 int model_dim           = 0)
{
    FMoeKernel* impl_ptr        = nullptr;
    uint32_t num_cu             = get_num_cu_func();
    uint32_t empty_cu           = num_cu;
    uint32_t tg_num             = 0;
    uint32_t num_persistent_tgs = 0;
    uint32_t round              = 0xffffffff;
    int selected_flat           = 0;
    std::string arch_id         = get_gpu_arch();
    std::string selectedKl      = kernel_name.empty() ? "" : arch_id + kernel_name;
    int vskip                   = get_vskip();
    static SynchronizedCache<std::string_view, FMoeKernel> impl_ptr_map;

    if(selectedKl.empty())
    {
        for(const auto& el : *cfgs)
        {
            if(el.first.find(arch_id) != 0)
                continue;
            const auto& cfg = el.second;
            if(kernel_fits(cfg, vskip, smf, block_size_M, inter_dim, model_dim))
            {
                tg_num = inter_dim / cfg.subGU_n *
                         sub_X_cnt; // how many thread_groups are needed to handle inter_dim
                uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
                const uint32_t empty = local_round * num_cu - tg_num;
                // Fewer rounds, then fewer empty CUs. At the same cost a flat==0 row
                // beats a flat==1 row with the same subGU_n, and a PS row wins a tie.
                if(local_round < round ||
                   (local_round == round &&
                    (empty_cu > empty ||
                     (empty_cu == empty && cfg.flat == selected_flat && cfg.ps == 1) ||
                     (empty_cu == empty && selected_flat != 0 && cfg.flat == 0))))
                {
                    round         = local_round;
                    empty_cu      = empty;
                    selectedKl    = el.first;
                    selected_flat = cfg.flat;
                    if(cfg.ps == 1)
                        num_persistent_tgs = cfg.tg_num_perCU * num_cu;
                    else
                        num_persistent_tgs = 0;
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
                    vskip,
                    ", model_dim: ",
                    model_dim);
    }
    auto it = cfgs->find(selectedKl);
    if(it != cfgs->end())
    {
        const auto& cfg     = it->second;
        AITER_CHECK(model_dim % std::max(cfg.dim_align, 1) == 0 && model_dim >= cfg.dim_min,
                    __func__,
                    ": ",
                    selectedKl,
                    " requires model_dim % ",
                    cfg.dim_align,
                    " == 0 and model_dim >= ",
                    cfg.dim_min,
                    "; got model_dim=",
                    model_dim);
        AITER_CHECK(ps_grid_fits(cfg, inter_dim),
                    __func__,
                    ": ",
                    selectedKl,
                    " has too few persistent workgroups on ",
                    num_cu,
                    " CUs for inter_dim=",
                    inter_dim);
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        if(cfg.ps == 1)
            num_persistent_tgs = cfg.tg_num_perCU * num_cu;
        else
            num_persistent_tgs = 0;

        impl_ptr = &impl_ptr_map.get_or_create(name, [&]() {
            return FMoeKernel(
                name,
                co_name,
                cfg.subGU_n,
                num_persistent_tgs,
                cfg.flat,
                cfg.tgs_align,
                static_cast<uint32_t>(std::max(cfg.max_tokens, 0)),
                static_cast<uint32_t>(std::max(cfg.max_assignments, 0)),
                static_cast<uint32_t>(std::max(cfg.max_experts, 0)));
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
    bool is_mxfp4        = false;
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
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Situv2)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_situv2;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenMXfp4_g1u1. ");
        impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, smf, kernel_name_str);
        impl_ptr->set_4bit(true);
        is_mxfp4 = true;
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
        else if(out->dtype() == AITER_DTYPE_bf16 && act == ActivationType::Situv2)
            config_map = &cfg_fmoe_bf16_pertokenMXfp4_g1u1_situv2;
        else
            AITER_CHECK(false, __func__, " Not find proper cfg in pertokenMXfp4_g1u1 (bf16 X). ");
        impl_ptr = get_heuristic_kernel(inter_dim, sub_X_cnt, config_map, smf, kernel_name_str);
        impl_ptr->set_4bit(true);
        is_mxfp4 = true;
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

    // The asm MXFP4 O-flush walks full 1024-dim blocks, then waves 0/1 drain a
    // 512-dim tail when dim bit 9 is set. That covers every multiple of 512
    // (dim=512 is tail-only). A 256 remainder is unwritten, so reject it.
    if(is_mxfp4)
    {
        AITER_CHECK(model_dim >= 512 && (model_dim % 512) == 0,
                    __func__,
                    " asm MXFP4 kernels require model_dim to be a positive "
                    "multiple of 512; got model_dim=" +
                        std::to_string(model_dim));
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

static CFG* get_blockscale_cfgs(ActivationType act, bool xquant)
{
    if(act == ActivationType::Silu)
        return xquant ? &cfg_fmoe_bf16_blockscaleBf16_g1u1_silu
                      : &cfg_fmoe_bf16_blockscaleFp8_g1u1_silu;
    if(act == ActivationType::Gelu)
        return xquant ? &cfg_fmoe_bf16_blockscaleBf16_g1u1_gelu
                      : &cfg_fmoe_bf16_blockscaleFp8_g1u1_gelu;
    return nullptr;
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
        config_map = get_blockscale_cfgs(act, input->dtype() == AITER_DTYPE_bf16);
        AITER_CHECK(config_map != nullptr,
                    __func__,
                    "Unsupported activation type for fmoe_fp8_blockscale_g1u1");

        impl_ptr = get_heuristic_kernel(inter_dim,
                                        sorted_expert_ids->size(0),
                                        config_map,
                                        0,
                                        kernel_name_str,
                                        block_size_M,
                                        down->size(1));
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

AITER_CTYPES_DEFINE_ENTRYPOINT(fmoe_fp8_blockscale_g1u1_has_kernel,
                               (int activation,
                                int xquant,
                                int block_size_M,
                                int inter_dim,
                                int model_dim,
                                hipStream_t stream),
                               (activation, xquant, block_size_M, inter_dim, model_dim, stream))
{
    CFG* config_map = get_blockscale_cfgs(static_cast<ActivationType>(activation), xquant != 0);
    if(config_map == nullptr || inter_dim % 128 != 0)
        return 0;
    std::string arch_id = get_gpu_arch();
    int vskip           = get_vskip();
    for(const auto& el : *config_map)
        if(el.first.find(arch_id) == 0 &&
           kernel_fits(el.second, vskip, 0, block_size_M, inter_dim, model_dim))
            return 1;
    return 0;
}

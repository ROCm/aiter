// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_f4gemm_configs.hpp"
#include <cmath>
#include <memory>
#include <hip/hip_runtime.h>

struct __attribute__((packed)) KernelArgs
{
    void* ptr_D;
    p2 _p0;
    void* ptr_C;
    p2 _p1;
    void* ptr_A;
    p2 _p2;
    void* ptr_B;
    p2 _p3;
    float alpha;
    p3 _p4;
    float beta;
    p3 _p5;
    unsigned int stride_D0;
    p3 _p6;
    unsigned int stride_D1;
    p3 _p7;
    unsigned int stride_C0;
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;
    p3 _p10;
    unsigned int stride_A1;
    p3 _p11;
    unsigned int stride_B0;
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    void* ptr_ScaleA;
    p2 _p17;
    void* ptr_ScaleB;
    p2 _p18;
    unsigned int stride_ScaleA0;
    p3 _p19;
    unsigned int stride_ScaleA1;
    p3 _p20;
    unsigned int stride_ScaleB0;
    p3 _p21;
    unsigned int stride_ScaleB1;
    p3 _p22;
    int log2_k_split;
    // p3 _p23;
};

// ps=1 kernels: the same arguments plus the CU count (the grid is min(tiles x K slices, num_cu))
struct __attribute__((packed)) PsKernelArgs
{
    KernelArgs args;
    unsigned int num_cu;
};
// The prebuilt .co files read these offsets.
static_assert(offsetof(KernelArgs, log2_k_split) == 368 && sizeof(KernelArgs) == 372 &&
                  offsetof(PsKernelArgs, num_cu) == 372 && sizeof(PsKernelArgs) == 376,
              "f4gemm kernarg layout");

// What the ps=1 kernels take: B preshuffled and contiguous, K % 64 == 0, N % 16 == 0, scales padded to whole tiles,
// A and B under 4 GiB and one C tile under 2 GiB (32-bit buffer offsets), no bias and no beta.
static bool ps_supported(const f4gemmConfig& cfg,
                         aiter_tensor_t* A,
                         aiter_tensor_t* B,
                         aiter_tensor_t* A_scale,
                         aiter_tensor_t* B_scale,
                         aiter_tensor_t* out,
                         aiter_tensor_t* bias,
                         float beta,
                         int bpreshuffle)
{
    const int64_t M = A->size(0), N = B->size(0), K = A->size(1) * 2;
    const int64_t a_row = A->stride(0), b_row = B->stride(0); // bytes: fp4_x2 elements
    const int64_t lim = int64_t(1) << 32;
    return bpreshuffle && bias == nullptr && beta == 0.f && K % 64 == 0 && N % 16 == 0 && a_row % 16 == 0 &&
           b_row * 2 == K && A_scale->size(0) >= (M + cfg.tile_M - 1) / cfg.tile_M * cfg.tile_M &&
           B_scale->size(0) >= (N + cfg.tile_N - 1) / cfg.tile_N * cfg.tile_N && M * a_row < lim &&
           N * b_row < lim && int64_t(cfg.tile_M) * out->stride(0) * 2 < lim / 2;
}

static CFG* get_cfg(AiterDtype inp_dtype, AiterDtype out_dtype)
{
    if((inp_dtype == AITER_DTYPE_fp4x2 || inp_dtype == AITER_DTYPE_u8) &&
       out_dtype == AITER_DTYPE_bf16)
    {
        return &cfg_f4gemm_bf16_per1x32Fp4;
    }
    else
    {
        AITER_CHECK(false,
                    __func__,
                    " Unsupported input_type:",
                    AiterDtype_to_str(inp_dtype),
                    ", out_type:",
                    AiterDtype_to_str(out_dtype));
        return nullptr;
    }
}

std::tuple<std::string, int> get_heuristic_kernel(int M,
                                                  int N,
                                                  int K,
                                                  std::string arch_id,
                                                  int log2_k_split,
                                                  int bpreshuffle,
                                                  bool ps_ok,
                                                  CFG* cfgs)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    uint32_t num_cu        = dev_prop.multiProcessorCount;
    uint32_t empty_cu      = num_cu;
    uint32_t tg_num        = 0;
    uint32_t round         = 0xffffffff;
    float compute2mem_effi = 1.0;
    int log2_k_split_en    = (log2_k_split >= 0 && log2_k_split != 0) ? 1 : 0;
    int bpreshuffle_en     = (bpreshuffle == 0) ? 0 : 1;
    std::string selectedKernelName = "";
    int selectedsplitK             = 1;

    // A ps=1 kernel that takes this call replaces the ps=0 kernel of the same tile; one that does not is skipped.
    auto has_ps_twin = [&](const f4gemmConfig& cfg) {
        for(const auto& el : *cfgs)
            if(el.first.find(arch_id) == 0 && el.second.ps && el.second.tile_M == cfg.tile_M &&
               el.second.tile_N == cfg.tile_N && el.second.bpreshuffle == cfg.bpreshuffle)
                return true;
        return false;
    };

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;
        const auto& cfg = el.second;
        if(cfg.ps ? !ps_ok : ps_ok && has_ps_twin(cfg))
            continue;
        if(cfg.bpreshuffle == bpreshuffle_en &&
           (cfg.splitK >= log2_k_split_en))
        {
            // tile128x512 may not support N % cfg.tile_N != 0
            if(cfg.tile_M != 128 || cfg.tile_N != 512 || (N % cfg.tile_N) == 0)
            {
                std::vector<int> splitK_list =
                    (log2_k_split >= 0 && cfg.splitK)
                        ? std::vector<int>{1 << log2_k_split}
                        : (cfg.splitK ? std::vector<int>{2, 4, 8, 16} : std::vector<int>{1});

                for(auto& splitK : splitK_list)
                {
                    int tg_num_M         = (M + cfg.tile_M - 1) / cfg.tile_M;
                    int tg_num_N         = (N + cfg.tile_N - 1) / cfg.tile_N;
                    tg_num               = tg_num_M * tg_num_N * splitK;
                    uint32_t local_round = (tg_num + num_cu - 1) / num_cu;

                    float local_compute2mem_effi =
                        cfg.tile_M * cfg.tile_N / (cfg.tile_M + cfg.tile_N);

                    bool is_earlier_round        = (local_round < round);
                    bool is_same_round           = (local_round == round);
                    bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
                    bool has_better_efficiency   = (local_compute2mem_effi > compute2mem_effi);
                    if(is_earlier_round ||
                       (is_same_round && (has_sufficient_empty_cu || has_better_efficiency)))
                    {
                        round              = local_round;
                        empty_cu           = local_round * num_cu - tg_num;
                        compute2mem_effi   = local_compute2mem_effi;
                        selectedKernelName = el.first;
                        selectedsplitK     = splitK;
                    }
                }
            }
        }
    }

    AITER_CHECK(selectedKernelName != "", __func__, ": cannot get heuristic kernel!");
    int log2_result = 0;
    while(selectedsplitK >>= 1)
        ++log2_result;
    return std::make_tuple(selectedKernelName, log2_result);
}

// A4W4 asm gemm kernel
// D=A*B*alpha+beta*C
AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a4w4_asm,
    (
    aiter_tensor_t* A,       // A:[M, K/2] f4x2
    aiter_tensor_t* B,       // B:[N, K/2] f4x2
    aiter_tensor_t* A_scale, // A_scale:[M, K/32] e8m0 padded
    aiter_tensor_t* B_scale, // B_scale:[N, K/32] e8m0 padded
    aiter_tensor_t* out,     // Out:[M, N] bf16
    const char*  kernelName,
    aiter_tensor_t* bias,    // bias:[M, N] f32, can be nullptr
    float        alpha,
    float        beta,
    int          bpreshuffle,
    int          log2_k_split,
    hipStream_t  stream),
    (A, B, A_scale, B_scale, out, kernelName, bias, alpha, beta, bpreshuffle, log2_k_split, stream))
{
    AITER_CHECK(
        out->dtype() == AITER_DTYPE_bf16, __func__, " only support BFloat16 output now!");
    int Mdim = A->size(0);
    int Ndim = B->size(0);
    int Kdim = A->size(1) * 2; // always fp4_x2
    KernelArgs args{};
    args.ptr_D      = out->ptr;
    args.ptr_C      = bias ? bias->ptr : nullptr;
    args.ptr_A      = A->ptr;
    args.ptr_B      = B->ptr;

    args.alpha          = alpha;
    args.beta           = beta;
    args.stride_C0      = out->stride(0);
    args.stride_A0      = A->stride(0) * 2; // always fp4_x2
    args.stride_B0      = B->stride(0) * 2; // always fp4_x2
    args.M              = Mdim;
    args.N              = Ndim;
    args.K              = Kdim;
    args.ptr_ScaleA     = A_scale->ptr;
    args.ptr_ScaleB     = B_scale->ptr;
    args.stride_ScaleA0 = A_scale->stride(0);
    args.stride_ScaleB0 = B_scale->stride(0);
    args.log2_k_split   = 0;

    const HipDeviceGuard device_guard(A->device_id);

    CFG* config_map = get_cfg(A->dtype(), out->dtype());
    using DictKey   = std::tuple<int, int, int, int, int, int>;
    struct SimpleHash
    {
        size_t operator()(const DictKey& key) const
        {
            const auto& [m, n, k, log2, shuffle, ps_ok] = key;
            return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(k) ^
                   std::hash<int>()(log2) ^ std::hash<int>()(shuffle) ^ std::hash<int>()(ps_ok << 1);
        }
    };
    static SynchronizedCache<DictKey, std::tuple<std::string, int>, SimpleHash>
        heuristic_kernel_dict;

    AITER_CHECK(!config_map->empty(), __func__, " no kernel support a4w4 for this gpu arch");

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    std::string arch_id = get_gpu_arch();
    std::string kname   = (kernelName && kernelName[0] != 0) ? (arch_id + kernelName) : "";

    int selectedksplit = (log2_k_split >= 0) ? log2_k_split : 0;
    if(kname.empty())
    {
        bool ps_ok = true; // every ps=1 kernel of this arch takes the call
        for(const auto& el : *config_map)
            if(el.first.find(arch_id) == 0 && el.second.ps)
                ps_ok = ps_ok && ps_supported(el.second, A, B, A_scale, B_scale, out, bias, beta, bpreshuffle);
        std::tie(kname, selectedksplit) = heuristic_kernel_dict.get_or_create(
            DictKey(Mdim, Ndim, Kdim, log2_k_split, bpreshuffle, ps_ok), [&]() {
                return get_heuristic_kernel(
                    Mdim, Ndim, Kdim, arch_id, log2_k_split, bpreshuffle, ps_ok, config_map);
            });
    }

    AiterAsmKernel* impl_ptr = nullptr;
    int SUBM                 = 0;
    int SUBN                 = 0;
    int gdz                  = 1;
    uint32_t num_cu          = 0; // ps=1 only

    auto it = config_map->find(kname);
    if(it != config_map->end())
    {
        const auto& cfg     = it->second;
        const char* name    = cfg.knl_name.c_str();
        const char* co_name = cfg.co_name.c_str();
        SUBM                = cfg.tile_M;
        SUBN                = cfg.tile_N;

        if(cfg.ps)
        {
            AITER_CHECK(bias == nullptr && beta == 0.f, __func__, " ", kname, ": bias/beta not supported");
            AITER_CHECK(ps_supported(cfg, A, B, A_scale, B_scale, out, bias, beta, bpreshuffle),
                        __func__,
                        " ",
                        kname,
                        ": needs B preshuffled and contiguous, K % 64 == 0, N % 16 == 0, scales padded to whole "
                        "tiles, A and B under 4 GiB");
            num_cu = get_num_cu_func();
        }

        if(cfg.splitK == 1)
        {
            args.log2_k_split = selectedksplit;
            int k_num         = 1 << args.log2_k_split;
            AITER_CHECK(Kdim % k_num == 0, __func__, " Kdim % (1 << args.log2_k_split) != 0 !");
            auto k_slices = [&](int split) {
                int k_per_tg = Kdim / split;
                k_per_tg     = ((k_per_tg + 256 - 1) / 256) * 256;
                return (Kdim + k_per_tg - 1) / k_per_tg;
            };
            // ps=1 kernels drop K slices until every work item has its own CU; same rule here, so grid == work
            const int64_t tiles = int64_t((Mdim + SUBM - 1) / SUBM) * ((Ndim + SUBN - 1) / SUBN);
            while(cfg.ps && k_num > 1 && tiles * k_slices(k_num) > num_cu)
            {
                k_num /= 2;
                --args.log2_k_split;
            }
            if(k_num > 1)
                hipMemsetAsync(out->ptr, 0, out->numel() * out->element_size(), stream);
            gdz = k_slices(k_num);
        }

        impl_ptr =
            &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });
    }
    else
        AITER_CHECK(false, __func__, " not find kernel " + kname);

    int gdx = (Ndim + SUBN - 1) / SUBN;
    int gdy = (Mdim + SUBM - 1) / SUBM;
    if(it->second.ps) // one workgroup per CU at most, each loops over its work items (tile x K slice)
    {
        gdx = std::min<int64_t>(int64_t(gdx) * gdy * gdz, num_cu);
        gdy = 1;
        gdz = 1;
    }
    PsKernelArgs ps_args{args, num_cu};
    size_t arg_size = it->second.ps ? sizeof(ps_args) : sizeof(args);

    impl_ptr->launch_kernel({it->second.ps ? (void*)&ps_args : (void*)&args,
                             &arg_size,
                             gdx, // gdx
                             gdy, // gdy
                             gdz, // gdz
                             256, // bdx: 4 wv64
                             1,   // bdy
                             1,   // bdz
                             stream});
}

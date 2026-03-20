// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "asm_fp8gemm_blockscale_configs.hpp"
#include <cmath>
#include <memory>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define DebugPrint 0

struct __attribute__((packed)) KernelArgs {
    void *ptr_C;
    p2 _p0;
    void *ptr_A;
    p2 _p1;
    void *ptr_B;
    p2 _p2;
    void *ptr_a_scale;
    p2 _p3;
    void *ptr_b_scale;
    p2 _p4;
    void *ptr_bias;
    p2 _p5;
    unsigned int m;
    p3 _p6;
    unsigned int n;
    p3 _p7;
    unsigned int k;
    p3 _p17;
    unsigned int lda;
    p3 _p8;
    unsigned int ldb;
    p3 _p9;
    unsigned int ldc;
    p3 _p10;
    unsigned int ks;
    p3 _p11;
    unsigned int scale_m;
    p3 _p12;
    unsigned int scale_n;
    p3 _p13;
    unsigned int scale_k;
    p3 _p14;
};

static CFG* get_cfg(AiterDtype inp_dtype, AiterDtype out_dtype) {
    if (inp_dtype == AITER_DTYPE_fp8 && out_dtype == AITER_DTYPE_bf16) {
        return &cfg_fp8gemm_bf16_blockscale;
    }
    AITER_CHECK(false, __func__, " Unsupported input_type: ", AiterDtype_to_str(inp_dtype),
                ", out_type: ", AiterDtype_to_str(out_dtype), ". Expected FP8 input and BFloat16 output.");
    return nullptr;
}

// Validation functions for fp8gemm_bf16_blockscale
// rule1: Ndim % TileN == 0 and Kdim % TileK == 0
static void validate_inputs(AiterTensor* A, AiterTensor* B, AiterTensor* out,
                           AiterTensor* A_scale, AiterTensor* B_scale) {
    constexpr int TileN = 128, TileK = 128;

    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16,
                "MI308 A8W8 blockscale asm only support BFloat16 output now!");
    AITER_CHECK(A->dtype() == AITER_DTYPE_fp8 && B->dtype() == AITER_DTYPE_fp8,
                "MI308 A8W8 blockscale asm requires FP8 input tensors!");

    int Ndim = B->size(0), Kdim = A->size(1);

    AITER_CHECK(Ndim % TileN == 0 && Kdim % TileK == 0,
                "MI308 A8W8 blockscale asm only support 128nx128k tile now!");
}

// Heuristic kernel selection
std::tuple<std::string, int> get_heuristic_fp8_kernel(int M, int N, int K, std::string arch_id,
                                                      int splitK, int bpreshuffle,
                                                      CFG* cfgs) {
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    uint32_t num_cu = dev_prop.multiProcessorCount;
    uint32_t empty_cu = num_cu;
    uint32_t round = 0xffffffff;
    float compute2mem_effi = 1.0;

    int splitK_en = (splitK >= 0 && splitK != 1) ? 1 : 0;
    int bpreshuffle_en = (bpreshuffle == 0) ? 0 : 1;
    std::string selectedKernelName = "";
    int selectedsplitK = 1;

    for (const auto& el : *cfgs) {
        if (el.first.find(arch_id) != 0) continue;

        const auto& cfg = el.second;
        if (cfg.bpreshuffle == bpreshuffle_en && ((cfg.splitK == splitK_en) || splitK < 0)) {
            if ((N % cfg.tile_n) == 0) {
                std::vector<int> splitK_list = (splitK >= 0 && cfg.splitK)
                    ? std::vector<int>{splitK}
                    : (cfg.splitK ? std::vector<int>{2, 4, 8} : std::vector<int>{1});

                for (auto& split_k : splitK_list) {
                    int tg_num_M = (M + cfg.tile_m - 1) / cfg.tile_m;
                    int tg_num_N = (N + cfg.tile_n - 1) / cfg.tile_n;
                    uint32_t tg_num = tg_num_M * tg_num_N * split_k;
                    uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
                    float local_compute2mem_effi = cfg.tile_m * cfg.tile_n / (cfg.tile_m + cfg.tile_n);

                    bool is_earlier_round = (local_round < round);
                    bool is_same_round = (local_round == round);
                    bool has_sufficient_empty_cu = (empty_cu > (local_round * num_cu - tg_num));
                    bool has_better_efficiency = (local_compute2mem_effi > compute2mem_effi);

                    if (is_earlier_round || (is_same_round && (has_sufficient_empty_cu || has_better_efficiency))) {
                        round = local_round;
                        empty_cu = local_round * num_cu - tg_num;
                        selectedKernelName = el.first;
                        selectedsplitK = split_k;
                        compute2mem_effi = local_compute2mem_effi;
                    }
                }
            }
        }
    }

    AITER_CHECK(selectedKernelName != "", __func__, ": cannot get heuristic kernel!");
    return std::make_tuple(selectedKernelName, selectedsplitK);
}

struct KernelSelector {
    using DictKey = std::tuple<int, int, int, int, int>;
    struct SimpleHash {
        size_t operator()(const DictKey& key) const {
            const auto& [m, n, k, split_k, shuffle] = key;
            return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(k) ^
                   std::hash<int>()(split_k) ^ std::hash<int>()(shuffle);
        }
    };

    static std::unordered_map<DictKey, std::tuple<std::string, int>, SimpleHash> heuristic_cache;
    static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> kernel_cache;

    static std::tuple<std::string, int> select_kernel(int M, int N, int K, const std::string& arch_id,
                                                     int splitK, int bpreshuffle,
                                                     const char* kernelName, CFG* config_map) {
        if (kernelName && kernelName[0] != 0) {
            return std::make_tuple(arch_id + kernelName, (splitK >= 0) ? splitK : 1);
        }

        DictKey key(M, N, K, splitK, bpreshuffle);
        auto it = heuristic_cache.find(key);
        if (it != heuristic_cache.end()) {
            return it->second;  // find it and return
        }
        auto result = get_heuristic_fp8_kernel(M, N, K, arch_id, splitK, bpreshuffle, config_map);
        heuristic_cache[key] = result;
        return result;
    }

    static AiterAsmKernel* get_kernel(const std::string& kernel_name, const std::string& co_name) {
        auto result = kernel_cache.emplace(kernel_name, nullptr);
        if (result.second) {
            result.first->second = std::make_unique<AiterAsmKernel>(kernel_name.c_str(), co_name.c_str());
        }
        return result.first->second.get();
    }
};


std::unordered_map<KernelSelector::DictKey, std::tuple<std::string, int>, KernelSelector::SimpleHash>
    KernelSelector::heuristic_cache;
std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> KernelSelector::kernel_cache;

static KernelArgs setup_kernel_args(AiterTensor* A, AiterTensor* B, AiterTensor* out,
                                   AiterTensor* A_scale, AiterTensor* B_scale,
                                   void* bias_ptr, int selectedsplitK) {
    constexpr int block_shape_m = 1, block_shape_k = 128, block_shape_n = 128;
    KernelArgs args;
    args.ptr_A = A->ptr;
    args.ptr_B = B->ptr;
    args.ptr_C = out->ptr;
    args.ptr_a_scale = A_scale->ptr;
    args.ptr_b_scale = B_scale->ptr;
    args.ptr_bias = bias_ptr;
    args.m = A->size(0);
    args.n = B->size(0);
    args.k = A->size(1);
    args.lda = A->size(1);
    args.ldb = A->size(1);
    args.ldc = B->size(0) * 2;  // BF16 is 2 bytes
    args.ks = selectedsplitK;
    args.scale_m = (A->size(0) + block_shape_m - 1) / block_shape_m;
    args.scale_n = (B->size(0) + block_shape_n - 1) / block_shape_n;
    args.scale_k = (A->size(1) + block_shape_k - 1) / block_shape_k;

    return args;
}

static void print_debug_info(const KernelArgs& args, const std::string& selectedKernelName,
                           int selectedsplitK, int gdx, int gdy, int gdz, hipStream_t stream,
                           AiterTensor* bias) {
    if (!DebugPrint) return;

    printf("=== MI308 A8W8 GEMM Kernel Parameters ===\n");
    printf("Selected Kernel: %s\n", selectedKernelName.c_str());
    printf("Matrix dimensions: M=%u, N=%u, K=%u\n", args.m, args.n, args.k);
    printf("Grid dimensions: gdx=%d, gdy=%d, gdz=%d\n", gdx, gdy, gdz);
    printf("splitK: %d\n", selectedsplitK);

    printf("\n=== Kernel Arguments ===\n");
    printf("args.m=%u, args.n=%u, args.k=%u\n", args.m, args.n, args.k);
    printf("args.lda=%u, args.ldb=%u, args.ldc=%u\n", args.lda, args.ldb, args.ldc);
    printf("args.ks=%u\n", args.ks);
    printf("args.scale_m=%u, args.scale_n=%u, args.scale_k=%u\n", args.scale_m, args.scale_n, args.scale_k);

    if (bias) {
        printf("Bias: provided\n");
    } else {
        printf("Bias: zero bias created\n");
    }
    printf("==========================================\n");
}

extern "C" __attribute__((visibility("default"))) void gemm_a8w8_blockscale_bpreshuffle_asm(
    AiterTensor* A,
    AiterTensor* B,
    AiterTensor* out,
    AiterTensor* A_scale,
    AiterTensor* B_scale,
    AiterTensor* bias,
    int          splitK,
    const char*  kernelName,
    int          bpreshuffle,
    AiterTensor* zero_bias_buf,
    hipStream_t  stream)
{
    validate_inputs(A, B, out, A_scale, B_scale);
    std::string arch_id = get_gpu_arch();
    CFG* config_map = get_cfg(A->dtype(), out->dtype());

    AITER_CHECK(!config_map->empty(), __func__, " no kernel support a8w8 blockscale for GPU arch: ", arch_id);

    auto [selectedKernelName, selectedsplitK] = KernelSelector::select_kernel(
        A->size(0), B->size(0), A->size(1), arch_id, splitK, bpreshuffle, kernelName, config_map);

    // Use provided bias or the zero_bias_buf passed by caller
    void* bias_ptr = bias ? bias->ptr : (zero_bias_buf ? zero_bias_buf->ptr : nullptr);

    int prev_device;
    HIP_CALL(hipGetDevice(&prev_device));
    HIP_CALL(hipSetDevice(A->device_id));

    auto it = config_map->find(selectedKernelName);
    AITER_CHECK(it != config_map->end(), __func__, " not find kernel " + selectedKernelName);

    const auto& cfg = it->second;
    constexpr int TileK = 128;

    if (cfg.splitK == 1 && selectedsplitK > 1) {
        int k_per_split = (A->size(1) + selectedsplitK - 1) / selectedsplitK;
        int k_per_split_aligned = ((k_per_split + TileK - 1) / TileK) * TileK;
        int actual_ksplit = (A->size(1) + k_per_split_aligned - 1) / k_per_split_aligned;
        if (actual_ksplit != selectedsplitK) {
            selectedsplitK = actual_ksplit;
        }
        AITER_CHECK(A->size(1) % k_per_split_aligned == 0 ||
                   (A->size(1) / k_per_split_aligned) == (selectedsplitK - 1),
                   __func__, " Kdim alignment check failed for splitK!");
        hipMemsetAsync(out->ptr, 0, out->numel() * out->element_size(), stream);
    }

    AiterAsmKernel* impl_ptr = KernelSelector::get_kernel(cfg.knl_name, cfg.co_name);
    KernelArgs args = setup_kernel_args(A, B, out, A_scale, B_scale, bias_ptr, selectedsplitK);
    size_t arg_size = sizeof(args);

    int gdx = (B->size(0) + cfg.tile_n - 1) / cfg.tile_n;
    int gdy = (A->size(0) + cfg.tile_m - 1) / cfg.tile_m;
    int gdz = 1;
    gdx = gdx * selectedsplitK;
    if (DebugPrint) {
        print_debug_info(args, selectedKernelName, selectedsplitK, gdx, gdy, gdz, stream, bias);
    }
    impl_ptr->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 256, 1, 1, stream});

    HIP_CALL(hipSetDevice(prev_device));
}

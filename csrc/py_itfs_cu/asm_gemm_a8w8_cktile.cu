// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"
#include "asm_i8gemm_cktile_configs.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

#include <optional>
#include <string>
#include <tuple>

namespace {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

std::tuple<std::string, int>
get_heuristic_kernel_cktile(int M,
                            int N,
                            int K,
                            const std::string& arch_id,
                            std::optional<int> splitK,
                            CFG* cfgs,
                            const char* kernelName = nullptr)
{
    std::string selected_kernel = "";
    int selected_splitk_exp     = 0;

    for(const auto& el : *cfgs)
    {
        if(el.first.find(arch_id) != 0)
            continue;

        const auto& cfg = el.second;

        if(kernelName && kernelName[0] != '\0')
        {
            if(el.first != (arch_id + kernelName))
                continue;

            AITER_CHECK(cfg.tile_n > 0 && N % cfg.tile_n == 0,
                        __func__,
                        " specified kernel cannot support N=", N,
                        " with tile_n=", cfg.tile_n,
                        ", kernel=", el.first);

            if(splitK.has_value() && splitK.value() > 0)
            {
                AITER_CHECK(cfg.splitK == 1,
                            __func__,
                            " splitK requested but kernel does not support splitK, kernel=",
                            el.first);
            }

            selected_kernel     = el.first;
            selected_splitk_exp = splitK.value_or(0);
            break;
        }

        if(cfg.tile_n > 0 && N % cfg.tile_n == 0)
        {
            selected_kernel = el.first;
            if(splitK.has_value())
            {
                selected_splitk_exp = splitK.value();
            }
            else
            {
                selected_splitk_exp = 0;
            }
            break;
        }
    }

    AITER_CHECK(selected_kernel != "", __func__, " no compatible cktile asm kernel found");
    return std::make_tuple(selected_kernel, selected_splitk_exp);
}

} // namespace

AITER_CTYPES_ERROR_DEF

AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(
    gemm_a8w8_cktile_asm,
    (aiter_tensor_t* A,       // [M, K] int8/fp8
     aiter_tensor_t* B,       // [N, K] int8/fp8 (no preshuffle)
     aiter_tensor_t* A_scale, // [M, 1]
     aiter_tensor_t* B_scale, // [1, N]
     aiter_tensor_t* out,     // [M, N] bf16/fp16
     const char* kernelName,
     int splitK,
     hipStream_t stream),
    (A, B, A_scale, B_scale, out, kernelName, splitK, stream))
{
    AITER_CHECK(A->dtype() == B->dtype(), __func__, " A/B dtype mismatch");
    AITER_CHECK(out->dtype() == AITER_DTYPE_bf16 || out->dtype() == AITER_DTYPE_fp16,
                __func__,
                " out must be bf16/fp16");

    const HipDeviceGuard device_guard(A->device_id);

    const int Mdim = A->size(0);
    const int Ndim = out->size(1);
    const int Kdim = A->size(1);

    const int stride_A  = static_cast<int>(A->stride(0));
    const int stride_B  = static_cast<int>(B->stride(0));
    const int stride_C  = static_cast<int>(out->stride(0));
    const int stride_AQ = static_cast<int>(A_scale->stride(0));
    const int stride_BQ = static_cast<int>(B_scale->stride(0));

    std::optional<int> opt_splitK = (splitK >= 0) ? std::optional<int>(splitK) : std::nullopt;

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    CFG* config_map = &cfg_i8gemm_cktile;
    AITER_CHECK(!config_map->empty(), __func__, " no cktile asm kernels configured");

    const std::string arch_id = get_gpu_arch();

    std::string selected_name = (kernelName && kernelName[0] != '\0') ? (arch_id + kernelName) : "";
    int selected_splitk_exp   = opt_splitK.value_or(0);

    if(selected_name.empty())
    {
        std::tie(selected_name, selected_splitk_exp) = get_heuristic_kernel_cktile(
            Mdim, Ndim, Kdim, arch_id, opt_splitK, config_map, kernelName);
    }

    auto it = config_map->find(selected_name);
    AITER_CHECK(it != config_map->end(), __func__, " kernel not found: ", selected_name);

    const auto& cfg = it->second;
    AITER_CHECK(cfg.tile_m > 0 && cfg.tile_n > 0,
                __func__,
                " invalid tile_m/tile_n in config for ",
                selected_name);
    AITER_CHECK(selected_name.find("QuantGemmMultiDKernelArgsILi0") != std::string::npos,
                __func__,
                " unsupported kernel arg ABI (expected ILi0) for ",
                selected_name);

    if(selected_splitk_exp > 0)
    {
        AITER_CHECK(cfg.splitK == 1,
                    __func__,
                    " splitK requested but kernel does not support splitK: ",
                    selected_name);
    }

    const int k_batch = 1 << selected_splitk_exp;

    // Quant-by-row/column scales.
    const int AQK = 1;
    const int BQK = 1;

    // For split-k, output is accumulated atomically.
    if(k_batch > 1)
    {
        HIP_CALL(hipMemsetAsync(out->ptr, 0, out->numel() * out->element_size(), stream));
    }

    // IMPORTANT: launch raw ASM with the *kernel* arg ABI layout.
    // QuantGemmHostArgs has a different field order and is only for CK host wrappers.
    ck_tile::QuantGemmKernelArgs args(A->ptr,
                                      B->ptr,
                                      A_scale->ptr,
                                      B_scale->ptr,
                                      out->ptr,
                                      Mdim,
                                      Ndim,
                                      Kdim,
                                      AQK,
                                      BQK,
                                      stride_A,
                                      stride_B,
                                      stride_C,
                                      stride_AQ,
                                      stride_BQ,
                                      k_batch);

    size_t arg_size = sizeof(args);

    const char* name    = cfg.knl_name.c_str();
    const char* co_name = cfg.co_name.c_str();
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name); });

    // CKTile uses a 1D grid: gridX = ceil(M/tile_m) * ceil(N/tile_n), gridZ = k_batch
    const int gdx = ceil_div(Mdim, cfg.tile_m) * ceil_div(Ndim, cfg.tile_n);
    const int gdz = k_batch;
    const int bdx = 512;

    impl_ptr->launch_kernel({&args, &arg_size, gdx, 1, gdz, bdx, 1, 1, stream});
}

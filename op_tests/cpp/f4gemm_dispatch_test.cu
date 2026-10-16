// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Include the production selectors in a test-only translation unit.
#include <unordered_map>
#include "../../csrc/py_itfs_cu/asm_f4gemm.cu"

AITER_CTYPES_DEFINE_ENTRYPOINT(
    f4gemm_test_config,
    (int M, int N, int K, const char* arch, const char* intype, const char* outtype,
     int a_preshuffle, const char* kernelName, int* values),
    (M, N, K, arch, intype, outtype, a_preshuffle, kernelName, values))
{
    const auto& cfg = get_kernel_config(M, N, K, arch, intype, outtype,
                                        a_preshuffle, kernelName);
    const auto geometry = get_launch_geometry(cfg, M, N);
    const int result[] = {geometry.cluster_x, geometry.cluster_y,
                          geometry.grid_x, geometry.grid_y,
                          cfg.tile_m, cfg.tile_n, cfg.ps_mode};
    std::copy(std::begin(result), std::end(result), values);
    return 0;
}

AITER_C_ITFS void f4gemm_test_geometry(int M, int N, int* values)
{
    const auto geometry = aiter::f4gemm::select_launch_geometry(M, N);
    values[0] = geometry.cluster_x;
    values[1] = geometry.cluster_y;
    values[2] = geometry.grid_x;
    values[3] = geometry.grid_y;
}

AITER_C_ITFS bool f4gemm_test_valid_geometry(int x, int y, int grid_y)
{
    try
    {
        aiter::f4gemm::make_launch_geometry(x, y, grid_y);
        return true;
    }
    catch(const std::invalid_argument&)
    {
        return false;
    }
}

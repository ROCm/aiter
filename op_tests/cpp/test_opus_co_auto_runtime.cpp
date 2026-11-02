// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gfx1250/opus_co_launch_gfx1250.cuh"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    assert(argc >= 3);
    const std::string root = argv[1];
    const std::string symbol = argv[2];
    aiter_detail::AiterThrowGuard throwing;
    int device = -1;
    HIP_CALL(hipGetDevice(&device));
    const auto& info = opus_gfx1250_co::device_info(device);
    auto selected = opus_gfx1250_co::select_image(root, info.arch, info.asic_revision, symbol);
    std::cout << "DEVICE=" << device << " ASIC_REVISION=" << info.asic_revision
              << " PROFILE=" << (info.asic_revision == 0 ? "A0" : "B0+")
              << " PATH=" << selected.path << std::endl;
    setenv("OPUS_GEN_CO_DIR", root.c_str(), 1);
    auto* first = &opus_gfx1250_co::co_kernel(symbol, device);
    assert(&opus_gfx1250_co::co_kernel(symbol, device) == first);
    const std::string alternate = root + (info.asic_revision == 0 ? "/gfx1250-a0" : "/named_reg");
    setenv("OPUS_GEN_CO_DIR", alternate.c_str(), 1);
    auto* second = &opus_gfx1250_co::co_kernel(symbol, device);
    if(info.asic_revision == 0)
        assert(second == first); // Direct override names the same actual image.
    else
        assert(second != first); // Same kid, different file root, separate registration.
    setenv("OPUS_GEN_CO_DIR", root.c_str(), 1);
    assert(&opus_gfx1250_co::co_kernel(symbol, device) == first);
    bool wrong_device = false;
    try { opus_gfx1250_co::co_kernel(symbol, device + 1); }
    catch(const std::runtime_error&) { wrong_device = true; }
    assert(wrong_device);
    std::cout << "CO_AUTO_RUNTIME_LOAD_CACHE_TESTS_PASS\n";
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_traits_a16w16_gfx1250.cuh"

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#include "aiter_hip_common.h"
#include "opus_co_profile_gfx1250.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Python exports the installed base root. The compiled macro is only a fallback
// for callers without Python (a wheel build-tree macro may no longer exist).
// ASIC Revision 0 selects its gfx1250-a0/ directory; B0+ selects gfx1250/.
#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif

namespace opus_gfx1250_co {

inline std::string co_root() {
    if(const char* env = std::getenv("OPUS_GEN_CO_DIR"); env && *env)
        return absolute_co_root(env);
    return absolute_co_root(OPUS_GEN_CO_DIR);
}

struct DeviceInfo {
    int device;
    int asic_revision;
    std::string arch;
};

inline const DeviceInfo& device_info(int device) {
    static SynchronizedCache<int, DeviceInfo> cache;
    return cache.get_or_create(device, [device]() {
        hipDeviceProp_t prop{};
        const hipError_t status = hipGetDeviceProperties(&prop, device);
        AITER_CHECK(status == hipSuccess, "opus_gemm: cannot query device ", device,
                    ": ", hipGetErrorString(status));
        const std::string full_arch(prop.gcnArchName);
        // This is HSA_AMD_AGENT_INFO_ASIC_REVISION, exactly rocminfo's
        // "ASIC Revision" field, NOT CHIP_REV_ID / EXTERNAL_REV_ID. Use the
        // current HIP device, not the minimum across rocminfo agents: visible
        // device ordinals can be remapped by HIP_VISIBLE_DEVICES.
        return DeviceInfo{device, prop.asicRevision, full_arch.substr(0, full_arch.find(':'))};
    });
}

inline std::vector<char> read_co_image(const SelectedImage& selected);

// Bytes must outlive the registered module: AiterAsmKernelFast does not copy
// the input hsaco. Every object is associated with one HIP device and path.
struct CoModule {
    std::vector<char> image;
    AiterAsmKernelFast kernel;

    CoModule(const std::string& symbol, const SelectedImage& selected)
        : image(read_co_image(selected)),
          kernel(symbol.c_str(), image.data(), AiterAsmKernelFast::SkipGfx1250Gate{}) {}
};

inline std::vector<char> read_co_image(const SelectedImage& selected) {
    std::ifstream file(selected.path, std::ios::binary | std::ios::ate);
    AITER_CHECK(file.is_open(), "opus_gemm: cannot open pre-compiled kernel '", selected.path,
                "' for ASIC Revision ", selected.asic_revision,
                ". Build the matching gfx1250/gfx1250-strict profile.");
    const std::streamsize size = file.tellg();
    AITER_CHECK(size >= 0, "opus_gemm: cannot determine CO size: ", selected.path);
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    AITER_CHECK(file.read(bytes.data(), size), "opus_gemm: short CO read: ", selected.path);
    validate_image_profile(bytes, selected);
    return bytes;
}

// HIP runtimes before ROCm 10.1.0 (HIP 7.16.26385) do not understand
// gfx1250-strict code objects and reject them with hipErrorNoBinaryForGpu /
// invalid device function. Detect that before registering, using the plain
// module API so the decision does not depend on AITER_CHECK's throw mode.
// Throws on a missing/invalid image (packaging or profile error); returns
// false only when the runtime itself rejects the code object.
inline bool hip_module_accepts(const SelectedImage& selected, std::string* error) {
    const std::vector<char> bytes = read_co_image(selected);
    hipModule_t module = nullptr;
    const hipError_t status = hipModuleLoadData(&module, bytes.data());
    if(status != hipSuccess) {
        *error = std::string(hipGetErrorName(status)) + " (" +
                 hipGetErrorString(status) + ")";
        return false;
    }
    (void)hipModuleUnload(module);
    return true;
}

// True only when the image file exists; a missing file stays a packaging error
// and must not be silently replaced by another profile.
inline bool co_image_exists(const SelectedImage& selected) {
    std::error_code error;
    return std::filesystem::is_regular_file(selected.path, error) && !error;
}

inline AiterAsmKernelFast& co_kernel(const std::string& symbol, int expected_device = -1) {
    int device = -1;
    const hipError_t status = hipGetDevice(&device);
    AITER_CHECK(status == hipSuccess, "opus_gemm: cannot query current HIP device: ",
                hipGetErrorString(status));
    AITER_CHECK(expected_device < 0 || expected_device == device,
                "opus_gemm: current HIP device ", device,
                " does not match input device ", expected_device);
    const std::string root = co_root();

    struct FastEntry {
        int device = -1;
        std::string root;
        AiterAsmKernelFast* kernel = nullptr;
    };
    // Thread-local hot path: no directory probe, device-properties query or
    // shared registration-cache lock while the symbol/device/root is unchanged.
    static thread_local std::unordered_map<std::string, FastEntry> fast;
    auto& entry = fast[symbol];
    if(entry.kernel && entry.device == device && entry.root == root)
        return *entry.kernel;

    const auto& info = device_info(device);
    SelectedImage selected = select_image(root, info.arch, info.asic_revision, symbol);
    if(info.asic_revision == 0 && co_image_exists(selected)) {
        std::string strict_error;
        if(!hip_module_accepts(selected, &strict_error)) {
            const auto regular = select_regular_fallback(root, info.arch, symbol);
            std::string regular_error;
            if(regular && hip_module_accepts(*regular, &regular_error)) {
                static std::once_flag warned;
                std::call_once(warned, [&]() {
                    std::fprintf(stderr,
                                 "[opus] A0 gfx1250-strict CO not loadable by this HIP runtime "
                                 "(%s); falling back to regular gfx1250 CO for this process. "
                                 "Upgrade to ROCm >= 10.1.0 (HIP >= 7.16.26385) for A0-tuned "
                                 "images.\n",
                                 strict_error.c_str());
                });
                selected = *regular;
            } else {
                const std::string detail =
                    std::string("A0 gfx1250-strict CO cannot be loaded by this HIP runtime: ") +
                    strict_error +
                    (regular ? "; regular fallback also unavailable: " + regular_error
                             : "; no regular gfx1250 image exists for fallback") +
                    ". Use ROCm >= 10.1.0 (HIP >= 7.16.26385), or ship the regular profile.";
                AITER_CHECK(false, detail);
            }
        }
    }
    static SynchronizedCache<std::string, CoModule> modules;
    auto& module = modules.get_or_create(co_cache_key(device, selected), [&]() {
        return CoModule(symbol, selected);
    });
    entry = FastEntry{device, root, &module.kernel};
    return *entry.kernel;
}

} // namespace opus_gfx1250_co

// Grid is in workgroups, rounded to complete clusters by the generated launcher.
template <typename Traits>
inline void opus_co_launch_gfx1250(AiterAsmKernelFast& kernel,
                                  dim3 grid,
                                  dim3 block,
                                  opus_gemm_4wave_compute_kargs_gfx1250& kargs,
                                  hipStream_t stream) {
    AITER_CHECK(grid.x % Traits::kClusterWgM == 0 && grid.y % Traits::kClusterWgN == 0,
                "opus_gemm gfx1250 co launch: grid (", grid.x, ",", grid.y,
                ") must be a whole number of ", Traits::kClusterWgM, "x",
                Traits::kClusterWgN, " clusters");

    size_t arg_size = sizeof(kargs);
    kernel.launch_kernel({&kargs, &arg_size,
                          (int)grid.x, (int)grid.y, (int)grid.z,
                          (int)block.x, 1, 1, stream,
                          Traits::kClusterWgM, Traits::kClusterWgN, 1});
}

#endif // host pass only

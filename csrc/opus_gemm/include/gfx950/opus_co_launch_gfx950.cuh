// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime loader for the pre-built gfx950 code objects (gen_co/build_co.py --arch gfx950).
// Host pass only. Images live under $OPUS_GEN_CO_DIR/gfx950/<family>/:
//   a8w8_blockscale_bpreshuffle: co_kernel(symbol, image_name) + opus_co_launch_gfx950 (96-byte kernarg)
//   a16w16 4-wave:               co_kernel(symbol, device) + launch (88-byte kernarg); one .co holds
//                                exactly one extern "C" kernel whose name equals the file stem.
#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

// Python exports the installed root (aiter/jit/core.py sets OPUS_GEN_CO_DIR at
// import); the compiled macro is only a fallback for callers without Python.
#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif

namespace opus_gfx950_co {

inline std::string co_dir(const char* family) {
    const char* env = std::getenv("OPUS_GEN_CO_DIR");
    const std::string root = (env && *env) ? std::string(env) : std::string(OPUS_GEN_CO_DIR);
    AITER_CHECK(!root.empty(), "opus_gemm gfx950: OPUS_GEN_CO_DIR is empty");
    return (std::filesystem::absolute(root) / "gfx950" / family).lexically_normal().string();
}

// Bare ELF: e_machine EM_AMDGPU (224), e_flags & EF_AMDGPU_MACH == EF_AMDGPU_MACH_AMDGCN_GFX950 (0x4f).
inline std::vector<char> read_co_image(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    AITER_CHECK(file.is_open(), "opus_gemm: cannot open pre-compiled gfx950 kernel '", path,
                "'. Build it with csrc/opus_gemm/gen_co/build_co.py --arch gfx950.");
    const std::streamsize size = file.tellg();
    AITER_CHECK(size > 4, "opus_gemm: invalid gfx950 CO size: ", path);
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    AITER_CHECK(static_cast<bool>(file.read(bytes.data(), size)), "opus_gemm: short CO read: ", path);
    AITER_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
                "opus_gemm: gfx950 CO is not a bare ELF code object: ", path);
    return bytes;
}

struct CoModule {
    // Keep the CO bytes alive until the registered kernel is destroyed.
    std::vector<char> image;
    AiterAsmKernelFast kernel;

    CoModule(const std::string& symbol, const std::string& path)
        : image(read_co_image(path)),
          kernel(symbol.c_str(), image.data(), AiterAsmKernelFast::SkipGfx1250Gate{}) {}
};

// a8w8_blockscale_bpreshuffle: one image may hold several symbols (normal + short loop).
inline AiterAsmKernelFast& co_kernel(const std::string& symbol, const std::string& image_name) {
    static SynchronizedCache<std::string, CoModule> cache;
    const std::string cache_key = image_name + '\0' + symbol;
    return cache
        .get_or_create(cache_key,
                       [&]() {
                           return CoModule(symbol,
                                           co_dir("a8w8_blockscale_bpreshuffle") + "/" + image_name);
                       })
        .kernel;
}

// a16w16 4-wave: <symbol>.co, one module per (device, path).
inline AiterAsmKernelFast& co_kernel(const std::string& symbol, int expected_device = -1) {
    int device = -1;
    const hipError_t status = hipGetDevice(&device);
    AITER_CHECK(status == hipSuccess, "opus_gemm: cannot query current HIP device: ",
                hipGetErrorString(status));
    AITER_CHECK(expected_device < 0 || expected_device == device, "opus_gemm: current HIP device ",
                device, " does not match input device ", expected_device);
    const std::string dir = co_dir("a16w16");

    struct FastEntry {
        int device = -1;
        std::string dir;
        AiterAsmKernelFast* kernel = nullptr;
    };
    static thread_local std::unordered_map<std::string, FastEntry> fast;
    auto& entry = fast[symbol];
    if(entry.kernel && entry.device == device && entry.dir == dir)
        return *entry.kernel;

    const std::string path = (std::filesystem::path(dir) / (symbol + ".co")).string();
    static SynchronizedCache<std::string, CoModule> modules;
    auto& module = modules.get_or_create(std::to_string(device) + ":" + path,
                                         [&]() { return CoModule(symbol, path); });
    entry = FastEntry{device, dir, &module.kernel};
    return *entry.kernel;
}

// Kargs is opus_gemm_a16w16_4wave_kargs_gfx950.
template <class Kargs>
inline void launch(AiterAsmKernelFast& kernel, dim3 grid, int block_size, Kargs& kargs,
                   hipStream_t stream) {
    static_assert(sizeof(Kargs) == 88,
                  "pre-built gfx950 4wave images expect an 88-byte kernarg segment");
    size_t arg_size = sizeof(kargs);
    kernel.launch_kernel({&kargs, &arg_size, static_cast<int>(grid.x), static_cast<int>(grid.y),
                          static_cast<int>(grid.z), block_size, 1, 1, stream});
}

} // namespace opus_gfx950_co

template <class Kargs>
inline void opus_co_launch_gfx950(AiterAsmKernelFast& kernel, dim3 grid, dim3 block,
                                  Kargs& args, hipStream_t stream, unsigned dynamic_lds = 0) {
    static_assert(sizeof(args) == 96);
    size_t arg_size = sizeof(args);
    kernel.launch_kernel({&args, &arg_size,
                          int(grid.x), int(grid.y), int(grid.z),
                          int(block.x), int(block.y), int(block.z),
                          stream, 1, 1, 1, dynamic_lds});
}

#endif // host pass only

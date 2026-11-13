// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime loader for the pre-built gfx950 a16w16 4-wave code objects
// (gen_co/gfx950/<symbol>.co, built by gen_co/build_co.py --arch gfx950).
// Host pass only. One .co holds exactly one extern "C" kernel whose name equals
// the file stem; its kernarg is opus_gemm_a16w16_4wave_kargs_gfx950 (88 bytes).
#pragma once

#include "opus_gemm_traits_a16w16_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#include "aiter_hip_common.h"
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// Python exports the installed root (aiter/jit/core.py sets OPUS_GEN_CO_DIR at
// import); the compiled macro is only a fallback for callers without Python.
#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif

namespace opus_gfx950_co {

static_assert(sizeof(opus_gemm_a16w16_4wave_kargs_gfx950) == 88,
              "pre-built gfx950 4wave images expect an 88-byte kernarg segment");

inline std::string co_dir() {
    const char* env = std::getenv("OPUS_GEN_CO_DIR");
    const std::string root = (env && *env) ? std::string(env) : std::string(OPUS_GEN_CO_DIR);
    AITER_CHECK(!root.empty(), "opus_gemm gfx950: OPUS_GEN_CO_DIR is empty");
    return (std::filesystem::absolute(root) / "gfx950").lexically_normal().string();
}

// e_machine EM_AMDGPU (224), e_flags & EF_AMDGPU_MACH == EF_AMDGPU_MACH_AMDGCN_GFX950 (0x4f).
inline std::vector<char> read_co_image(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    AITER_CHECK(file.is_open(), "opus_gemm: cannot open pre-compiled gfx950 kernel '", path,
                "'. Build it with csrc/opus_gemm/gen_co/build_co.py --arch gfx950.");
    const std::streamsize size = file.tellg();
    AITER_CHECK(size >= 64, "opus_gemm: truncated code object ", path);
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    AITER_CHECK(static_cast<bool>(file.read(bytes.data(), size)), "opus_gemm: short CO read: ", path);
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(bytes[i]); };
    AITER_CHECK(byte(0) == 0x7f && byte(1) == 'E' && byte(2) == 'L' && byte(3) == 'F' && byte(4) == 2 &&
                    byte(5) == 1,
                "opus_gemm: ", path, " is not a bare little-endian ELF64 code object");
    const uint32_t machine = byte(18) | (uint32_t(byte(19)) << 8);
    AITER_CHECK(machine == 224 && byte(48) == 0x4f, "opus_gemm: ", path,
                " is not a gfx950 code object (e_machine ", machine, ", mach 0x", std::hex,
                static_cast<int>(byte(48)), ")");
    return bytes;
}

// The bytes must outlive the registered module: AiterAsmKernelFast does not copy them.
struct CoModule {
    std::vector<char> image;
    AiterAsmKernelFast kernel;

    CoModule(const std::string& symbol, const std::string& path)
        : image(read_co_image(path)),
          kernel(symbol.c_str(), image.data(), AiterAsmKernelFast::SkipGfx1250Gate{}) {}
};

inline const std::string& device_arch(int device) {
    static SynchronizedCache<int, std::string> cache;
    return cache.get_or_create(device, [device]() {
        hipDeviceProp_t prop{};
        const hipError_t status = hipGetDeviceProperties(&prop, device);
        AITER_CHECK(status == hipSuccess, "opus_gemm: cannot query device ", device, ": ",
                    hipGetErrorString(status));
        const std::string full(prop.gcnArchName);
        return full.substr(0, full.find(':'));
    });
}

// Kernel `symbol` registered for the current HIP device (loaded once per device and path).
inline AiterAsmKernelFast& co_kernel(const std::string& symbol, int expected_device = -1) {
    int device = -1;
    const hipError_t status = hipGetDevice(&device);
    AITER_CHECK(status == hipSuccess, "opus_gemm: cannot query current HIP device: ",
                hipGetErrorString(status));
    AITER_CHECK(expected_device < 0 || expected_device == device, "opus_gemm: current HIP device ",
                device, " does not match input device ", expected_device);
    const std::string dir = co_dir();

    struct FastEntry {
        int device = -1;
        std::string dir;
        AiterAsmKernelFast* kernel = nullptr;
    };
    static thread_local std::unordered_map<std::string, FastEntry> fast;
    auto& entry = fast[symbol];
    if(entry.kernel && entry.device == device && entry.dir == dir)
        return *entry.kernel;

    const std::string& arch = device_arch(device);
    AITER_CHECK(arch == "gfx950", "opus_gemm: pre-built a16w16 4wave kernels require gfx950, device ",
                device, " is ", arch);
    const std::string path = (std::filesystem::path(dir) / (symbol + ".co")).string();
    static SynchronizedCache<std::string, CoModule> modules;
    auto& module = modules.get_or_create(std::to_string(device) + ":" + path,
                                         [&]() { return CoModule(symbol, path); });
    entry = FastEntry{device, dir, &module.kernel};
    return *entry.kernel;
}

inline void launch(AiterAsmKernelFast& kernel, dim3 grid, int block_size,
                   opus_gemm_a16w16_4wave_kargs_gfx950& kargs, hipStream_t stream) {
    size_t arg_size = sizeof(kargs);
    kernel.launch_kernel({&kargs, &arg_size, static_cast<int>(grid.x), static_cast<int>(grid.y),
                          static_cast<int>(grid.z), block_size, 1, 1, stream});
}

} // namespace opus_gfx950_co

#endif // host pass only

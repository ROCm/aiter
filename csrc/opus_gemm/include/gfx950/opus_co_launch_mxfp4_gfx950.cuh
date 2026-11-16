// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime loader / launcher for the gfx950 pre-compiled opus_gemm code objects (gen_co/gfx950/a4w4_bpreshuffle/*.co, built offline by
// gen_co/build_co.py --arch gfx950 with the pin toolchain). Host pass only.
//
// The .co directory is OPUS_GEN_CO_DIR (environment variable, else the compile-time macro set by the JIT module);
// images live in <dir>/gfx950/a4w4_bpreshuffle/<symbol>.co and contain one extern "C" kernel named <symbol>.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "aiter_hip_common.h"

#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif

namespace opus_gfx950_co {
struct MxCoModule {
    std::vector<char> image;   // kept alive while the kernel is registered
    AiterAsmKernelFast kernel;

    explicit MxCoModule(const std::string& symbol) : image(read_co(symbol)), kernel(symbol.c_str(), image.data()) {}

private:
    static std::vector<char> read_co(const std::string& symbol) {
        const char* env = std::getenv("OPUS_GEN_CO_DIR");
        const std::string root = env && *env ? env : OPUS_GEN_CO_DIR;
        AITER_CHECK(!root.empty(), "opus gfx950 .co directory is not set (OPUS_GEN_CO_DIR)");
        const std::string path = root + "/gfx950/a4w4_bpreshuffle/" + symbol + ".co";
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        AITER_CHECK(file.is_open(), "opus gfx950: cannot open code object ", path);
        const std::streamsize size = file.tellg();
        AITER_CHECK(size > 4, "opus gfx950: invalid code object size ", path);
        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(size));
        AITER_CHECK(static_cast<bool>(file.read(bytes.data(), size)), "opus gfx950: short read ", path);
        AITER_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
                    "opus gfx950: not a bare ELF code object: ", path);
        return bytes;
    }
};

inline AiterAsmKernelFast& mxfp4_co_kernel(const std::string& symbol) {
    static SynchronizedCache<std::string, MxCoModule> cache;
    return cache.get_or_create(symbol, [&]() { return MxCoModule(symbol); }).kernel;
}
} // namespace opus_gfx950_co

template <class Kargs>
inline void opus_mxfp4_co_launch_gfx950(AiterAsmKernelFast& kernel, dim3 grid, dim3 block, Kargs& args, hipStream_t stream) {
    size_t arg_size = sizeof(args);
    kernel.launch_kernel({&args, &arg_size, int(grid.x), int(grid.y), int(grid.z), int(block.x), int(block.y), int(block.z), stream});
}
#endif

// SPDX-License-Identifier: Apache-2.0
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <optional>
#include "aiter_hip_common.h"
#include "aiter_tensor.h"
#include "aiter_stream.h"
#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif
namespace opus_gfx950_co {
struct CoModule {
    // Keep the CO bytes alive until the registered kernel is destroyed.
    std::vector<char> image;
    AiterAsmKernelFast kernel;

    explicit CoModule(const std::string& symbol)
        : image(read_co(symbol)),
          kernel(symbol.c_str(), image.data(), AiterAsmKernelFast::SkipGfx1250Gate{}) {}

private:
    static std::vector<char> read_co(const std::string& symbol) {
        const char* env = std::getenv("OPUS_GEN_CO_DIR");
        const std::string root = env && *env ? env : OPUS_GEN_CO_DIR;
        AITER_CHECK(!root.empty(), "gfx950 CO directory is empty");
        const auto path = root + "/gfx950/" + symbol + ".co";
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        AITER_CHECK(file.is_open(), "gfx950 cannot open CO: ", path);
        const std::streamsize size = file.tellg();
        AITER_CHECK(size > 4, "gfx950 invalid CO size: ", path);
        file.seekg(0, std::ios::beg);
        std::vector<char> bytes(static_cast<size_t>(size));
        AITER_CHECK(file.read(bytes.data(), size), "gfx950 short CO read: ", path);
        AITER_CHECK(bytes[0] == 0x7f && bytes[1] == 'E' && bytes[2] == 'L' && bytes[3] == 'F',
                    "gfx950 CO is not a bare ELF code object: ", path);
        return bytes;
    }
};
inline AiterAsmKernelFast& co_kernel(const std::string& symbol) {
    static SynchronizedCache<std::string, CoModule> cache;
    return cache.get_or_create(symbol, [&]() { return CoModule(symbol); }).kernel;
}
} // namespace opus_gfx950_co
template<class Kargs>
inline void opus_co_launch_gfx950(AiterAsmKernelFast& kernel, dim3 grid, dim3 block,
                                  Kargs& args, hipStream_t stream, unsigned dynamic_lds = 0) {
    static_assert(sizeof(args) == 96);
    size_t arg_size = sizeof(args);
    kernel.launch_kernel({&args, &arg_size,
                          int(grid.x), int(grid.y), int(grid.z),
                          int(block.x), int(block.y), int(block.z),
                          stream, 1, 1, 1, dynamic_lds});
}

namespace opus_gfx950_co {
struct LaunchContract {
    int64_t m, n, k;
    unsigned gx, gy, gz, bx, by, bz, dynamic_lds;
};
} // namespace opus_gfx950_co
#endif

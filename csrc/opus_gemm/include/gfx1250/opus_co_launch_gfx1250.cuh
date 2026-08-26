// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Loader + launcher for the pre-compiled gfx1250 .co kernels (kernel_tag
// a16w16_4wave_co).
//
// Why a .co at all: the 4wave_compute pipeline needs __builtin_amdgcn_pin_vgpr,
// amdgpu_num_vgpr(1024) and -mllvm -amdgpu-expert-scheduling-mode. A release
// ROCm toolchain has none of them, and aiter/jit/core.py's hip_flag_checker
// SILENTLY DROPS flags the compiler rejects -- so a JIT build would not fail,
// it would just quietly produce a slower, differently-scheduled kernel. The
// kernel is therefore built ahead of time by gen_co/build_co.py with a patched
// LLVM, and only the launch happens here. Everything else about the kid (host
// launcher signature, manifest, tune lookup, dispatch, CSV schema) is identical
// to a JIT kid, so switching back to JIT later is a codegen branch and nothing
// more.
//
// The .co is an ordinary HIP-compiled kernel, but built with -D__HIPCC_RTC__,
// which strips the implicit HIP runtime wrapper: its metadata declares a single
// by_value kernarg and NO hidden/implicit args, exactly like the hand-written
// asm .co files aiter already loads this way.
#pragma once

#include "opus_gemm_traits_a16w16_gfx1250.cuh"

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#include "aiter_hip_common.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// Directory holding gen_co/<arch>/<symbol>.co, resolved in this order:
//
//   1. $OPUS_GEN_CO_DIR -- what `import aiter` sets (aiter/jit/core.py, next to
//      AITER_ASM_DIR), and what a user can override to swap in a freshly built
//      .co without rebuilding the module. This is the only value that is right
//      for a PREBUILT wheel: the macro below is baked when the module is
//      compiled, which for a prebuild is the build tree's aiter_meta/ -- a
//      directory setup.py deletes when it is done, and which in any case is not
//      where the files land after install.
//   2. -DOPUS_GEN_CO_DIR from module_deepgemm_opus
//      (aiter/jit/optCompilerConfig.json), for a build that is not driven by
//      aiter's Python entry points at all.
#ifndef OPUS_GEN_CO_DIR
#define OPUS_GEN_CO_DIR ""
#endif

namespace opus_gfx1250_co
{

inline std::string co_root()
{
    if (const char* env = std::getenv("OPUS_GEN_CO_DIR"); env && *env)
        return std::string(env);
    return std::string(OPUS_GEN_CO_DIR);
}

// One loaded .co: the file bytes plus the registered module. AiterAsmKernelFast
// takes the hsaco by pointer and does NOT copy it (only the heavier
// AiterAsmKernel owns its buffer), so `image` must outlive `kernel` -- hence
// both in one object, declared in that order.
struct CoModule
{
    std::vector<char> image;
    AiterAsmKernelFast kernel;

    explicit CoModule(const std::string& symbol)
        : image(read_co(symbol)), kernel(symbol.c_str(), image.data())
    {
    }

    private:
    static std::vector<char> read_co(const std::string& symbol)
    {
        const std::string root = co_root();
        AITER_CHECK(!root.empty(),
                    "opus_gemm gfx1250: OPUS_GEN_CO_DIR is empty -- the module was "
                    "built without -DOPUS_GEN_CO_DIR and no env override is set, so "
                    "the pre-compiled kernel '", symbol, "' cannot be located");
        // Subdirectory is the RUNNING device's arch, not the arch this header
        // is named after: on a multi-arch build the launcher symbol exists on
        // every device, so a kid selected on the wrong one should say "no
        // gfx950/<symbol>.co" and not go load a gfx1250 image that then fails
        // registration for reasons that read like corruption.
        const std::string path = root + "/" + get_gpu_arch() + "/" + symbol + ".co";
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        AITER_CHECK(f.is_open(),
                    "opus_gemm gfx1250: cannot open pre-compiled kernel '", path,
                    "'. Build it with csrc/opus_gemm/gen_co/build_co.py "
                    "--llvm-bin <pin-capable LLVM>, or point OPUS_GEN_CO_DIR at "
                    "a directory that has it.");
        const std::streamsize size = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> buf(static_cast<size_t>(size));
        AITER_CHECK(f.read(buf.data(), size), "opus_gemm gfx1250: short read on ", path);
        // Bare amdhsa ELF is what __hipRegisterFatBinary wants. build_co.py
        // unbundles the clang offload bundle for us; catch a stale/wrong file
        // here rather than as an opaque registration failure.
        AITER_CHECK(buf.size() > 4 && buf[0] == 0x7f && buf[1] == 'E' && buf[2] == 'L'
                        && buf[3] == 'F',
                    "opus_gemm gfx1250: ", path,
                    " is not a bare ELF code object (a clang offload bundle "
                    "starts with __CLANG_OFFLOAD_BUNDLE__). Re-run build_co.py.");
        return buf;
    }
};

// One registration per symbol per process. AiterAsmKernelFast is neither
// copyable nor movable; SynchronizedCache constructs in place (C++17 guaranteed
// elision through its Wrapper conversion), which is why CoModule need not be.
//
// Every generated launcher resolves this ONCE into a function-local static, so
// the map lookup (a mutex plus a std::string) is a first-call cost and not a
// per-launch one -- a launcher symbol is a compile-time constant, so its static
// can never be the wrong kernel.
inline AiterAsmKernelFast& co_kernel(const std::string& symbol)
{
    static SynchronizedCache<std::string, CoModule> cache;
    return cache.get_or_create(symbol, [&]() { return CoModule(symbol); }).kernel;
}

}  // namespace opus_gfx1250_co

// Launch a 4wave_compute .co. `grid` is in WORKGROUPS and must already be
// rounded up to whole clusters -- rocclr rejects a cluster launch whose grid is
// not a multiple of the cluster dims (LaunchParams::CheckClusterDivisibility),
// and the surplus workgroups are handled inside the kernel (tile_oob).
template <typename Traits>
inline void opus_co_launch_gfx1250(AiterAsmKernelFast& kernel,
                                   dim3 grid,
                                   dim3 block,
                                   opus_gemm_4wave_compute_kargs_gfx1250& kargs,
                                   hipStream_t stream)
{
    AITER_CHECK(grid.x % Traits::kClusterWgM == 0 && grid.y % Traits::kClusterWgN == 0,
                "opus_gemm gfx1250 co launch: grid (", grid.x, ",", grid.y,
                ") must be a whole number of ", Traits::kClusterWgM, "x",
                Traits::kClusterWgN, " clusters");

    size_t arg_size = sizeof(kargs);
    kernel.launch_kernel({&kargs,
                          &arg_size,
                          (int)grid.x,
                          (int)grid.y,
                          (int)grid.z,
                          (int)block.x,
                          1,
                          1,
                          stream,
                          Traits::kClusterWgM,
                          Traits::kClusterWgN,
                          1});
}

#endif  // host pass only

// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// opus_gemm_arch.cuh — runtime architecture probe shared by all opus dispatch
// shells. Holds:
//   * OpusGfxArch enum  — extend when a new arch ships.
//   * OpusArchInfo      — cached gcnArchName + device id snapshot.
//   * opus_get_arch_info() / opus_get_gfx_arch() — one-shot probes.
//
// Per-arch dispatch implementations live in opus_gemm_arch_<arch>.cuh and are
// only included by opus_gemm.cu. Adding a new arch:
//   1. Append the new enum value below.
//   2. Add a `name.rfind("gfxXXXX", 0) == 0` branch in the probe.
//   3. Create opus_gemm_arch_gfxXXXX.cuh mirroring the gfx950 header.
//   4. Add include + switch case in opus_gemm.cu.
#pragma once

// We deliberately include <c10/util/Exception.h> instead of <torch/torch.h>
// here: TORCH_CHECK is the only thing this header needs from torch, and the
// full <torch/torch.h> drags in all of ATen + autograd which is much heavier.
// This file is a near-leaf header; keeping its include cost tiny matters when
// future opus subpackages start including it.
#include <hip/hip_runtime.h>
#include <c10/util/Exception.h>

#include <string>
#include <utility>

enum class OpusGfxArch
{
    Unknown = 0,
    Gfx950,
    // future: Gfx942, Gfx940, Gfx1100, ...
};

namespace opus_arch_detail
{
struct OpusArchInfo
{
    OpusGfxArch arch;
    std::string name;  // full gcnArchName, e.g. "gfx950:sramecc+:xnack-"
    int dev;
};
}  // namespace opus_arch_detail

// One-shot probe of the active CUDA device. opus_gemm follows the standard
// PyTorch model of one device per process; if a future user mixes archs across
// devices in the same process they'll need to extend this to a per-device map.
inline const opus_arch_detail::OpusArchInfo &opus_get_arch_info()
{
    using namespace opus_arch_detail;
    static const OpusArchInfo info = []() {
        int dev = -1;
        TORCH_CHECK(hipGetDevice(&dev) == hipSuccess, "opus_gemm: hipGetDevice failed");
        hipDeviceProp_t prop{};
        TORCH_CHECK(hipGetDeviceProperties(&prop, dev) == hipSuccess,
                    "opus_gemm: hipGetDeviceProperties failed");
        std::string name(prop.gcnArchName);
        OpusGfxArch a = OpusGfxArch::Unknown;
        if (name.rfind("gfx950", 0) == 0)
        {
            a = OpusGfxArch::Gfx950;
        }
        // future: else if (name.rfind("gfx942", 0) == 0) a = OpusGfxArch::Gfx942;
        return OpusArchInfo{a, std::move(name), dev};
    }();
    return info;
}

inline OpusGfxArch opus_get_gfx_arch()
{
    return opus_get_arch_info().arch;
}

// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Runtime architecture probe shared by all opus dispatch shells. Per-arch
// dispatch lives in opus_gemm_arch_<arch>.cuh (included only by opus_gemm.cu).
#pragma once

#include "aiter_hip_common.h"  // AITER_CHECK + hip_runtime (torch-free)

#include <string>
#include <utility>

enum class OpusGfxArch
{
    Unknown = 0,
    Gfx950,
    Gfx942,
    Gfx1250,
    // future: Gfx940, Gfx1100, ...
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

// One-shot probe of the active CUDA device (one-device-per-process model).
inline const opus_arch_detail::OpusArchInfo &opus_get_arch_info()
{
    using namespace opus_arch_detail;
    static const OpusArchInfo info = []() {
        int dev = -1;
        AITER_CHECK(hipGetDevice(&dev) == hipSuccess, "opus_gemm: hipGetDevice failed");
        hipDeviceProp_t prop{};
        AITER_CHECK(hipGetDeviceProperties(&prop, dev) == hipSuccess,
                    "opus_gemm: hipGetDeviceProperties failed");
        std::string name(prop.gcnArchName);
        OpusGfxArch a = OpusGfxArch::Unknown;
        if (name.rfind("gfx950", 0) == 0)
        {
            a = OpusGfxArch::Gfx950;
        }
        else if (name.rfind("gfx942", 0) == 0)
        {
            a = OpusGfxArch::Gfx942;
        }
        else if (name.rfind("gfx1250", 0) == 0)
        {
            a = OpusGfxArch::Gfx1250;
        }
        return OpusArchInfo{a, std::move(name), dev};
    }();
    return info;
}

inline OpusGfxArch opus_get_gfx_arch()
{
    return opus_get_arch_info().arch;
}

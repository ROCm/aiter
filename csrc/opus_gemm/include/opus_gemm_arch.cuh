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
    int cu_num;
};
}  // namespace opus_arch_detail

// Probe of the device this thread is bound to, cached per device ordinal so a
// process that calls hipSetDevice does not keep the first device's arch and CU
// count. Both select kernels, so a stale value runs another SKU's binary.
inline const opus_arch_detail::OpusArchInfo &opus_get_arch_info()
{
    using namespace opus_arch_detail;
    static SynchronizedCache<int, OpusArchInfo> cache;
    int dev = -1;
    AITER_CHECK(hipGetDevice(&dev) == hipSuccess, "opus_gemm: hipGetDevice failed");
    return cache.get_or_create(dev, [dev]() {
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
        return OpusArchInfo{a, std::move(name), dev, prop.multiProcessorCount};
    });
}

inline OpusGfxArch opus_get_gfx_arch()
{
    return opus_get_arch_info().arch;
}

// CU count of the active device, matching the `cu_num` column the tuner stamps
// on every row. Selects between the CU-count variants baked into one build.
inline int opus_get_device_cu_num()
{
    return opus_get_arch_info().cu_num;
}

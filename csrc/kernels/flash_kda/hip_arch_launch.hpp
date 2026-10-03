#pragma once

#include "hip_common.hpp"

namespace flashkda_hip::gfx942 {
void launch_fwd(const FwdParams& params, const HipDeviceInfo& device);
}

namespace flashkda_hip::gfx950 {
void launch_fwd(const FwdParams& params, const HipDeviceInfo& device);
}

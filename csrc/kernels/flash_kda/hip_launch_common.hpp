#pragma once

#include "hip_common.hpp"

namespace flashkda_hip {

void launch_fwd_common(
    const FwdParams& params,
    const HipDeviceInfo& device,
    const HipLaunchPolicy& policy);

}  // namespace flashkda_hip

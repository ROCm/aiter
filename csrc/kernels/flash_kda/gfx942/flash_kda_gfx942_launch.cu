#include "../hip_arch_launch.hpp"
#include "../hip_launch_common.hpp"
#include "policy.hpp"

namespace flashkda_hip::gfx942 {

void launch_fwd(const FwdParams& params, const HipDeviceInfo& device) {
    launch_fwd_common(params, device, LaunchPolicy::make(params, device));
}

}  // namespace flashkda_hip::gfx942

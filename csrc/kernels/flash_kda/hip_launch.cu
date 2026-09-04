// Thin HIP entry point: identify the active GPU and hand the stable parameter
// bundle to its architecture-private launcher.
#include <cstring>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>  // hipified under ROCm

#include "flash_kda.h"
#include "hip_arch_launch.hpp"

namespace flashkda_hip {
namespace {

struct DetectedDevice {
    HipDeviceInfo info;
    std::string arch_name;
};

HipArchitecture classify_architecture(const char* name) {
    if (std::strncmp(name, "gfx942", 6) == 0 &&
        (name[6] == '\0' || name[6] == ':'))
        return HipArchitecture::gfx942;
    if (std::strncmp(name, "gfx950", 6) == 0 &&
        (name[6] == '\0' || name[6] == ':'))
        return HipArchitecture::gfx950;
    return HipArchitecture::unsupported;
}

DetectedDevice current_device() {
    static thread_local int cached_device = -1;
    static thread_local DetectedDevice cached{};

    int device = 0;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string("FlashKDA: hipGetDevice failed: ") +
            hipGetErrorString(status));
    }
    if (device == cached_device)
        return cached;

    hipDeviceProp_t properties{};
    status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string("FlashKDA: hipGetDeviceProperties failed: ") +
            hipGetErrorString(status));
    }

    cached_device = device;
    cached = {{classify_architecture(properties.gcnArchName),
               properties.multiProcessorCount},
              properties.gcnArchName};
    return cached;
}

}  // namespace

void launch_fwd_hip(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const void* g_ptr,
    const void* beta_ptr,
    float scale,
    void* out_ptr,
    void* workspace_ptr,
    const float* A_log_ptr,
    const float* dt_bias_ptr,
    float gate_scale,
    int total_tiles,
    int T_total,
    int H_q,
    int H,
    int N,
    const void* init_state,
    void* final_state,
    bool has_state_in,
    bool has_state_out,
    bool state_fp32,
    const int32_t* cu_seqlens,
    hipStream_t stream,
    int max_seqlen_upper_bound) {
    const FwdParams params{
        q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, scale, out_ptr, workspace_ptr,
        A_log_ptr, dt_bias_ptr, gate_scale, total_tiles, T_total, H_q, H, N,
        init_state, final_state, has_state_in, has_state_out, state_fp32,
        cu_seqlens, stream, max_seqlen_upper_bound};
    const DetectedDevice device = current_device();

    switch (device.info.architecture) {
        case HipArchitecture::gfx942:
            gfx942::launch_fwd(params, device.info);
            return;
        case HipArchitecture::gfx950:
            gfx950::launch_fwd(params, device.info);
            return;
        case HipArchitecture::unsupported:
            throw std::runtime_error(
                "FlashKDA HIP backend supports gfx942 and gfx950; active "
                "device is '" + device.arch_name + "'");
    }
    throw std::runtime_error("FlashKDA: invalid HIP architecture dispatch");
}

}  // namespace flashkda_hip

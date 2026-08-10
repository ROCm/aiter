// Standalone gfx942 instantiation and raw-pointer launcher for GDN K1-C.
#include <hip/hip_runtime.h>

#include <limits>

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_k1_bt64_neumann_c_kernel_template.hpp"

#ifndef __HIP_DEVICE_COMPILE__

// hipcc's host pass needs a body from which it can emit the launch stub; the
// device pass gets the real MFMA implementation from the header above.
extern "C" __global__ void
gdn_k1_neumann_c_kernel(gdn_k1_neumann_c_kargs) {}

extern "C" hipError_t opus_gdn_k1_c_fwd(
    const void* ptr_k,
    const void* ptr_g,
    const void* ptr_beta,
    void* ptr_c,
    void* ptr_g_cumsum,
    int B,
    int T,
    int H,
    hipStream_t stream) {
    if (ptr_k == nullptr || ptr_g == nullptr || ptr_beta == nullptr
        || ptr_c == nullptr || ptr_g_cumsum == nullptr
        || B <= 0 || T <= 0 || H <= 0
        || H > std::numeric_limits<int>::max() / (64 * 128)) {
        return hipErrorInvalidValue;
    }

    constexpr int BT = 64;
    using K1Traits = gdn_k1_traits<BT, 128, 128, 4>;
    constexpr size_t dynamic_smem_bytes = K1Traits::smem_size_bytes();
    static_assert(
        dynamic_smem_bytes == 18176,
        "BT64 K1-C must use the existing Opus K1 dynamic-LDS contract");

    const int NT = 1 + (T - 1) / BT;
    const uint64_t bh = static_cast<uint64_t>(B) * static_cast<uint64_t>(H);
    // The kernel converts blockIdx.y to int before deriving b/h.
    if (bh > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return hipErrorInvalidValue;
    }
    // This raw ABI can be called without tensor allocation checks.  Bound the
    // largest flattened K address before forming it as signed int64 on device.
    const uint64_t max_bth =
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 128u;
    const uint64_t b = static_cast<uint64_t>(B);
    const uint64_t t = static_cast<uint64_t>(T);
    const uint64_t h = static_cast<uint64_t>(H);
    if (b > max_bth / t) {
        return hipErrorInvalidValue;
    }
    const uint64_t bt = b * t;
    if (bt > max_bth / h) {
        return hipErrorInvalidValue;
    }
    const gdn_k1_neumann_c_kargs kargs{
        ptr_k,
        ptr_g,
        ptr_beta,
        ptr_c,
        ptr_g_cumsum,
        B,
        T,
        H,
        NT};

    hipLaunchKernelGGL(
        gdn_k1_neumann_c_kernel,
        dim3(static_cast<unsigned int>(NT), static_cast<unsigned int>(bh)),
        dim3(256),
        dynamic_smem_bytes,
        stream,
        kargs);
    return hipGetLastError();
}

#endif  // !__HIP_DEVICE_COMPILE__

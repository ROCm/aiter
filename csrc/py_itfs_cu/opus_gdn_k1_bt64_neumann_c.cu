// Standalone gfx942 instantiation and raw-pointer launcher for GDN K1-C.
#include <hip/hip_runtime.h>

#include <limits>

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_k1_bt64_neumann_c_kernel_template.hpp"

#ifndef __HIP_DEVICE_COMPILE__
// hipcc's host pass needs a body from which it can emit the launch stub; the
// device pass gets the real MFMA implementation from the header above.
template <bool IS_VARLEN>
__global__ void gdn_k1_neumann_c_kernel(gdn_k1_neumann_c_kargs) {}
#endif

// Both passes must see these: the device pass emits the code objects and the
// host pass emits the matching launch stubs.
template __global__ void gdn_k1_neumann_c_kernel<false>(
    gdn_k1_neumann_c_kargs);
template __global__ void gdn_k1_neumann_c_kernel<true>(
    gdn_k1_neumann_c_kargs);

#ifndef __HIP_DEVICE_COMPILE__

extern "C" hipError_t opus_gdn_k1_c_fwd(
    const void* ptr_k,
    const void* ptr_g,
    const void* ptr_beta,
    void* ptr_c,
    void* ptr_g_cumsum,
    int B,
    int T,
    int H,
    int Hg,
    const void* ptr_cu_seqlens,
    const void* ptr_chunk_indices,
    int total_chunks,
    hipStream_t stream) {
    if (ptr_k == nullptr || ptr_g == nullptr || ptr_beta == nullptr
        || ptr_c == nullptr || ptr_g_cumsum == nullptr
        || B <= 0 || T <= 0 || H <= 0
        || H > std::numeric_limits<int>::max() / (64 * 128)) {
        return hipErrorInvalidValue;
    }
    // H value heads share Hg key heads; Hg == H is plain MHA.
    if (Hg <= 0 || Hg > H || H % Hg != 0) {
        return hipErrorInvalidValue;
    }
    // Packed batches carry both metadata tensors and a positive chunk count;
    // dense batches carry neither.
    const bool is_varlen = ptr_cu_seqlens != nullptr;
    if (is_varlen != (ptr_chunk_indices != nullptr)) {
        return hipErrorInvalidValue;
    }
    if (is_varlen ? total_chunks <= 0 : total_chunks != 0) {
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
        Hg,
        NT,
        static_cast<const int32_t*>(ptr_cu_seqlens),
        static_cast<const int32_t*>(ptr_chunk_indices)};

    if (is_varlen) {
        // One block per (global chunk, head).  The chunk axis already spans
        // every sequence, so the head axis alone stays inside the 16-bit
        // blockIdx.y extent for any supported head count.
        if (H > 65535) {
            return hipErrorInvalidValue;
        }
        hipLaunchKernelGGL(
            (gdn_k1_neumann_c_kernel<true>),
            dim3(static_cast<unsigned int>(total_chunks),
                 static_cast<unsigned int>(H)),
            dim3(256),
            dynamic_smem_bytes,
            stream,
            kargs);
        return hipGetLastError();
    }

    hipLaunchKernelGGL(
        (gdn_k1_neumann_c_kernel<false>),
        dim3(static_cast<unsigned int>(NT), static_cast<unsigned int>(bh)),
        dim3(256),
        dynamic_smem_bytes,
        stream,
        kargs);
    return hipGetLastError();
}

#endif  // !__HIP_DEVICE_COMPILE__

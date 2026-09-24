// Prebuilt code object for the fp8 32mx4 / 64nx1 MLA decode kernel, loaded by aiter's
// AiterAsmKernel out of hsa/gfx950/mla_opus/opus_mla_decode_fp8_32mx4_64nx1.co. Build it with
// build_mla_decode_fp8_32mx4_64nx1_co.sh: only the amdgpu-pin-op-dst toolchain honours the
// register pins the kernel's register map depends on.
//
// extern "C", because AiterAsmKernel looks each entry up by this exact name. One entry per
// (causal, large KV) specialization; the in-tree dispatch picks by name.

#include <hip/hip_runtime.h>

#include "opus/mla_decode_fp8_32mx4_64nx1.hpp"

template <bool CAUSAL, bool LARGE_KV, bool SPLIT_DMA = true>
using mla32mx4_traits_t = opus_mla_decode_fp8_32mx4_64nx1_traits<32,
                                                                 64,
                                                                 4,
                                                                 fp8_t,
                                                                 fp8_t,
                                                                 bf16_t,
                                                                 CAUSAL,
                                                                 LARGE_KV,
                                                                 3,
                                                                 2,
                                                                 4,
                                                                 SPLIT_DMA>;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#ifndef MLA32MX4_HAS_PIN
#error "build this code object with the amdgpu-pin-op-dst toolchain (see the build script)"
#endif
#define MLA32MX4_CO_ENTRY(name, causal, large, split)                                            \
    extern "C" __global__ __launch_bounds__(256, 1) void name(opus_mla_decode_fp8_kargs kargs) \
    {                                                                                           \
        mla_decode_fwd_32mx4_64nx1_fp8fp8::mla_decode_fwd_persistent<                            \
            mla32mx4_traits_t<causal, large, split>>(kargs);                                    \
    }
#else
#define MLA32MX4_CO_ENTRY(name, causal, large, split) \
    extern "C" __global__ void name(opus_mla_decode_fp8_kargs) {}
#endif

// Split-DMA builds for long work items, whole-tile DMA ("_ns") for short ones; LARGE_KV is
// always whole-tile (see SPLIT_DMA in the kernel).
MLA32MX4_CO_ENTRY(opus_mla_decode_fp8_32mx4_64nx1_nc, false, false, true)
MLA32MX4_CO_ENTRY(opus_mla_decode_fp8_32mx4_64nx1_causal, true, false, true)
MLA32MX4_CO_ENTRY(opus_mla_decode_fp8_32mx4_64nx1_nc_ns, false, false, false)
MLA32MX4_CO_ENTRY(opus_mla_decode_fp8_32mx4_64nx1_causal_ns, true, false, false)
MLA32MX4_CO_ENTRY(opus_mla_decode_fp8_32mx4_64nx1_nc_large, false, true, false)
MLA32MX4_CO_ENTRY(opus_mla_decode_fp8_32mx4_64nx1_causal_large, true, true, false)

// The code object is frozen against this layout; a wrong offset is a compile error here
// rather than garbage output.
static_assert(sizeof(opus_mla_decode_fp8_kargs) == 144 &&
                  offsetof(opus_mla_decode_fp8_kargs, work_indptr) == 96 &&
                  offsetof(opus_mla_decode_fp8_kargs, H) == 112 &&
                  offsetof(opus_mla_decode_fp8_kargs, softmax_scale) == 140,
              "opus_mla_decode_fp8_kargs changed: rebuild the code object and update this check");

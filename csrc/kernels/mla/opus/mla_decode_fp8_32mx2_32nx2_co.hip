// Prebuilt code object for the fp8 32mx2 / 32nx2 MLA decode kernel, loaded by aiter's
// AiterAsmKernel out of hsa/gfx950/mla_opus/opus_mla_decode_fp8_32mx2_32nx2.co. Build it with
// build/build_mla_decode_fp8_32mx2_32nx2_co.sh: only the amdgpu-pin-op-dst toolchain honours the
// register pins the kernel's register map depends on.
//
// extern "C", because AiterAsmKernel looks each entry up by this exact name. One entry per
// (causal, large KV) specialization; the in-tree dispatch picks by name.

#include <hip/hip_runtime.h>

#include "opus/mla_decode_fp8_32mx2_32nx2.hpp"

template <bool CAUSAL, bool LARGE_KV>
using mla32mx2_traits_t =
    opus_mla_decode_fp8_32mx2_32nx2_traits<32, 64, 4, fp8_t, fp8_t, bf16_t, CAUSAL, LARGE_KV, 3>;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#ifndef MLA32MX2_HAS_PIN
#error "build this code object with the amdgpu-pin-op-dst toolchain (see the build script)"
#endif
#define MLA32MX2_CO_ENTRY(name, causal, large)                                                   \
    extern "C" __global__ __launch_bounds__(256, 1) void name(opus_mla_decode_kargs kargs) \
    {                                                                                           \
        mla_decode_fwd_32mx2_32nx2_fp8fp8::mla_decode_fwd_persistent<                            \
            mla32mx2_traits_t<causal, large>>(kargs);                                           \
    }
#else
#define MLA32MX2_CO_ENTRY(name, causal, large) \
    extern "C" __global__ void name(opus_mla_decode_kargs) {}
#endif

MLA32MX2_CO_ENTRY(opus_mla_decode_fp8_32mx2_32nx2_nc, false, false)
MLA32MX2_CO_ENTRY(opus_mla_decode_fp8_32mx2_32nx2_causal, true, false)
MLA32MX2_CO_ENTRY(opus_mla_decode_fp8_32mx2_32nx2_nc_large, false, true)
MLA32MX2_CO_ENTRY(opus_mla_decode_fp8_32mx2_32nx2_causal_large, true, true)

// The code object is frozen against this layout; a wrong offset is a compile error here
// rather than garbage output.
static_assert(sizeof(opus_mla_decode_kargs) == 144 &&
                  offsetof(opus_mla_decode_kargs, work_indptr) == 72 &&
                  offsetof(opus_mla_decode_kargs, H) == 88 &&
                  offsetof(opus_mla_decode_kargs, softmax_scale) == 116 &&
                  offsetof(opus_mla_decode_kargs, q_scale_ptr) == 120 &&
                  offsetof(opus_mla_decode_kargs, kv_last_page_lens) == 136,
              "opus_mla_decode_kargs changed: rebuild the code object and update this check");

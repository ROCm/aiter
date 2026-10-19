#pragma once

#include <opus/opus.hpp>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#define OPUS_GLOBAL_ADDR __attribute__((address_space(1)))
#endif

template<opus::index_t BYTES, int G_IMM>
__device__ inline void global_load_lds_raw(const char* src, void* lds_ptr) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    __builtin_amdgcn_global_load_lds(
        (OPUS_GLOBAL_ADDR void*)(const_cast<char*>(src)),
        (OPUS_LDS_ADDR void*)(reinterpret_cast<__UINTPTR_TYPE__>(lds_ptr)),
        BYTES, G_IMM, 0);
#else
    __builtin_memcpy(reinterpret_cast<char*>(lds_ptr) + G_IMM, src + G_IMM, BYTES);
#endif
}

template<opus::index_t VEC, int M0_IMM, int G_IMM, class D, class S>
__device__ inline void global_load_lds(const D* src, const S* lds_base) {
    constexpr opus::index_t BYTES = VEC * static_cast<opus::index_t>(sizeof(D));
    static_assert(G_IMM >= 0 && G_IMM < (1 << 12) && G_IMM % 4 == 0);
    static_assert(M0_IMM >= 0 && M0_IMM % 4 == 0);
    global_load_lds_raw<BYTES, G_IMM>(
        reinterpret_cast<const char*>(src),
        reinterpret_cast<char*>(const_cast<S*>(lds_base)) + M0_IMM);
}

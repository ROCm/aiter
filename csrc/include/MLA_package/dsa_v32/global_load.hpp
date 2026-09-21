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

template<opus::index_t VEC, class D, class LayoutG, class LayoutS>
__device__ inline void global_load(const D* g_base, void* smem_base,
                                   const LayoutG& u_gmem, const LayoutS& u_smem) {
    constexpr opus::index_t BYTES = VEC * static_cast<opus::index_t>(sizeof(D));
    constexpr opus::index_t N     = opus::layout_load_traits<LayoutG, VEC>::r_elem.value;
    constexpr auto g_imm = opus::layout_imm_offsets_v<LayoutG, VEC>;
    constexpr auto s_imm = opus::layout_imm_offsets_v<LayoutS, VEC>;

    const auto g_os = opus::layout_to_offsets<VEC>(u_gmem);
    const auto s_os = opus::layout_to_offsets<VEC>(u_smem);
    auto* s_ptr = reinterpret_cast<char*>(smem_base);
    const char* src = reinterpret_cast<const char*>(g_base)
                    + static_cast<int64_t>(g_os[0]) * static_cast<int64_t>(sizeof(D));

    opus::static_for<N>([&](auto i) {
        constexpr int g_delta = (g_imm[i.value] - g_imm[0]) * static_cast<int>(sizeof(D));
        constexpr int m0_fix  = (s_imm[i.value] - s_imm[0]) * static_cast<int>(sizeof(D)) - g_delta;
        static_assert(m0_fix >= 0 && m0_fix % 4 == 0);
        global_load_lds_raw<BYTES, g_delta>(
            src, s_ptr + s_os[0] * static_cast<int>(sizeof(D)) + m0_fix);
    });
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

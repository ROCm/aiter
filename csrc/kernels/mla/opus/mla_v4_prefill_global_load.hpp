#pragma once

#include <opus/opus.hpp>
#include <bit>
#include <cstdint>

using opus::operator""_I;

// global -> LDS async copy (GLOBAL_LOAD_LDS). Same call shape as opus::async_load; completion is
// tracked by vmcnt only (ISA 10.4). u_smem must be wave-uniform: it feeds M0.
template<opus::index_t VEC, class D, class LayoutG, class LayoutS>
__device__ inline void global_load(const D* g_base, void* smem_base,
                                   const LayoutG& u_gmem, const LayoutS& u_smem) {
    constexpr opus::index_t BYTES = VEC * static_cast<opus::index_t>(sizeof(D));
    constexpr opus::index_t N     = opus::layout_load_traits<LayoutG, VEC>::r_elem.value;
    constexpr auto g_imm = opus::layout_imm_offsets_v<LayoutG, VEC>;
    constexpr auto s_imm = opus::layout_imm_offsets_v<LayoutS, VEC>;

    const auto g_os = opus::layout_to_offsets<VEC>(u_gmem);
    const auto s_os = opus::layout_to_offsets<VEC>(u_smem);
    auto* s_ptr = reinterpret_cast<OPUS_LDS_ADDR char*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_base));
    const char* src = reinterpret_cast<const char*>(g_base)
                    + static_cast<int64_t>(g_os[0]) * static_cast<int64_t>(sizeof(D));
    const unsigned int m0_base = __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
        reinterpret_cast<__UINTPTR_TYPE__>(s_ptr + s_os[0] * static_cast<int>(sizeof(D)))));

    opus::static_for<N>([&](auto i) {
        constexpr int g_delta = (g_imm[i.value] - g_imm[0]) * static_cast<int>(sizeof(D));
        constexpr int m0_fix  = (s_imm[i.value] - s_imm[0]) * static_cast<int>(sizeof(D)) - g_delta;
        static_assert(m0_fix >= 0 && m0_fix % 4 == 0,
                      "offset: shifts the LDS address too, so M0 must absorb it (ISA 9.1.9 / 10.3)");
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        const unsigned int m0_val = m0_base + m0_fix;
        const char* addr = src;   // asm operands do not odr-use, so name it inside the lambda
        #define OPUS_MLA_V4_GLOBAL_LOAD_LDS(mnemonic)                                       \
            asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\t" mnemonic " %1, off offset:%2"    \
                         :: "s"(m0_val), "v"(addr), "n"(g_delta) : "memory")
        if      constexpr (BYTES == 16) OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_dwordx4");
        else if constexpr (BYTES == 12) OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_dwordx3");
        else if constexpr (BYTES ==  4) OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_dword");
        else if constexpr (BYTES ==  2) OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_ushort");
        else                            OPUS_MLA_V4_GLOBAL_LOAD_LDS("global_load_lds_ubyte");
        #undef OPUS_MLA_V4_GLOBAL_LOAD_LDS
#else
        *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<D, VEC>*>(
            s_ptr + s_os[i.value] * static_cast<int>(sizeof(D))) =
            *reinterpret_cast<const opus::vector_t<D, VEC>*>(src + g_delta);
#endif
    });
}

// global -> VGPR, one issue per y-position of the layout (drop-in for opus::load(gmem, layout)).
template<opus::index_t VEC, class D, class Layout,
         std::enable_if_t<opus::is_layout_v<Layout>, bool> = true>
__device__ inline auto global_load(const D* g_base, const Layout& u) {
    constexpr opus::index_t N = opus::layout_load_traits<Layout, VEC>::r_elem.value;
    const auto os = opus::layout_to_offsets<VEC>(u);
    opus::vector_t<D, VEC * N> r;
    opus::static_for<N>([&](auto i) {
        opus::set_slice(r, *reinterpret_cast<const opus::vector_t<D, VEC>*>(
                               reinterpret_cast<const char*>(g_base)
                               + static_cast<int64_t>(os[i.value]) * static_cast<int64_t>(sizeof(D))),
                        opus::number<i.value * VEC>{}, opus::number<(i.value + 1) * VEC>{});
    });
    return r;
}

// global -> VGPR at a plain element offset.
template<opus::index_t VEC, class D>
__device__ inline auto global_load(const D* g_base, opus::index_t os) {
    return *reinterpret_cast<const opus::vector_t<D, VEC>*>(
        reinterpret_cast<const char*>(g_base) + static_cast<int64_t>(os) * static_cast<int64_t>(sizeof(D)));
}

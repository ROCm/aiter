#pragma once

#include <opus/opus.hpp>

// --- global->LDS async copy on a 64-bit flat address (GLOBAL_LOAD_LDS) ---
//
// Drop-in for opus::async_load(smem_base, u_gmem, u_smem), and the reason to prefer it:
// async_load goes through a buffer descriptor, whose num_records field is 32 bits, so it
// cannot address a tensor larger than 4 GiB. An MLA KV cache reaches that at ~7.5 M tokens
// (576 fp8 each), e.g. batch 256 x ctx 32768, and past it the bound wraps and every load
// beyond the wrap returns 0 -- silently wrong results, not a fault. global_load_lds takes
// the address in a VGPR pair instead, so the reach is the pointer's.
//
// Rebasing the descriptor per tile (the fmha hd192 fix) is not an option here: MLA decode
// is paged with page_size == 1, so the 32 tokens of one KV tile are scattered anywhere in
// the cache and their offsets are per-lane, while a descriptor's base is wave-uniform.
//
// Two consequences of this being inline asm, both of which the caller must respect:
//   * the compiler does not model it, so SIInsertWaitcnts will not wait for it. Completion
//     is tracked by vmcnt only (ISA 10.4) and the caller owns that budget. The "memory"
//     clobber is what keeps it ordered against the surrounding LDS traffic.
//   * u_smem must be wave-uniform: its offset feeds M0, which is scalar.
//
// Only the widths gfx950 actually has are accepted. There is no 8-byte form, so a VEC that
// works out to 8 bytes is a compile error here rather than a silent narrowing to one byte.

// Resolves the lane's own position inside the tile into the pointer. That part is
// loop-invariant, so hoisting it leaves the per-tile address as the single v_mad_u64_u32 that
// folds in the page offset. Adding it inside global_load instead costs a whole extra 64-bit
// add per load, and pins the lane offset in a VGPR pair whose high half is a dead zero.
template <opus::index_t VEC, class D, class LayoutG>
__device__ inline const D* global_load_base(const D* g_base, const LayoutG& u_gmem)
{
    // Unsigned: layout offsets are non-negative, and this keeps the add out of sign-extension.
    return g_base + static_cast<uint32_t>(opus::layout_to_offsets<VEC>(u_gmem)[0]);
}

// g_base is a global_load_base() of the same layout, with the tile's own offset added to it.
template <opus::index_t VEC, class D, class LayoutG, class LayoutS>
__device__ inline void
global_load(const D* g_base, void* smem_base, const LayoutG&, const LayoutS& u_smem)
{
    constexpr opus::index_t BYTES = VEC * static_cast<opus::index_t>(sizeof(D));
    constexpr opus::index_t N     = opus::layout_load_traits<LayoutG, VEC>::r_elem.value;
    constexpr auto g_imm          = opus::layout_imm_offsets_v<LayoutG, VEC>;
    constexpr auto s_imm          = opus::layout_imm_offsets_v<LayoutS, VEC>;
    static_assert(BYTES == 16 || BYTES == 12 || BYTES == 4 || BYTES == 2 || BYTES == 1,
                  "global_load_lds has no such width on gfx950 (note: no 8-byte form)");

    const auto s_os = opus::layout_to_offsets<VEC>(u_smem);
    auto* s_ptr =
        reinterpret_cast<OPUS_LDS_ADDR char*>(reinterpret_cast<__UINTPTR_TYPE__>(smem_base));
    const char* src            = reinterpret_cast<const char*>(g_base);
    const unsigned int m0_base = __builtin_amdgcn_readfirstlane(static_cast<unsigned int>(
        reinterpret_cast<__UINTPTR_TYPE__>(s_ptr + s_os[0] * static_cast<int>(sizeof(D)))));

    opus::static_for<N>([&](auto i) {
        // The per-issue deltas are the layout's y-dim steps, hence compile-time. The gmem
        // one rides the instruction's `offset:` immediate; M0 has to absorb the difference,
        // because that immediate shifts the LDS end of the copy too (ISA 9.1.9 / 10.3).
        constexpr int g_delta = (g_imm[i.value] - g_imm[0]) * static_cast<int>(sizeof(D));
        constexpr int m0_fix  = (s_imm[i.value] - s_imm[0]) * static_cast<int>(sizeof(D)) - g_delta;
        static_assert(m0_fix >= 0 && m0_fix % 4 == 0, "M0 cannot absorb this LDS/gmem skew");
        static_assert(g_delta >= -4096 && g_delta <= 4095, "g_delta exceeds the 13-bit offset");
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
        const unsigned int m0_val = m0_base + m0_fix;
        const char* addr          = src; // asm operands do not odr-use, so name it in here
#define MLA_GLOBAL_LOAD_LDS(mnemonic)                                                           \
    asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\t" mnemonic " %1, off offset:%2" ::"s"(m0_val), \
                 "v"(addr),                                                                     \
                 "n"(g_delta)                                                                   \
                 : "memory")
        if constexpr(BYTES == 16)
            MLA_GLOBAL_LOAD_LDS("global_load_lds_dwordx4");
        else if constexpr(BYTES == 12)
            MLA_GLOBAL_LOAD_LDS("global_load_lds_dwordx3");
        else if constexpr(BYTES == 4)
            MLA_GLOBAL_LOAD_LDS("global_load_lds_dword");
        else if constexpr(BYTES == 2)
            MLA_GLOBAL_LOAD_LDS("global_load_lds_ushort");
        else
            MLA_GLOBAL_LOAD_LDS("global_load_lds_ubyte");
#undef MLA_GLOBAL_LOAD_LDS
#else
        *reinterpret_cast<OPUS_LDS_ADDR opus::vector_t<D, VEC>*>(
            s_ptr + s_os[i.value] * static_cast<int>(sizeof(D))) =
            *reinterpret_cast<const opus::vector_t<D, VEC>*>(src + g_delta);
#endif
    });
}

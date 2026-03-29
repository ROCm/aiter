// SPDX-License-Identifier: MIT
// TurboQuant CodebookPack8: LDS-based 16-entry bf16 codebook lookup
// Drop-in replacement for CK's PassThroughPack8 for TurboQuant dequantization
#pragma once

#include "ck_tile/core.hpp"

namespace turboquant {

// Runtime codebook lookup using 16 bf16 values resident in LDS.
// Matches CK's i4_to_bhalf4 constexpr path nibble extraction order:
//   {cb[(q>>0)&0xf], cb[(q>>16)&0xf], cb[(q>>4)&0xf], cb[(q>>20)&0xf]}
// This order is required by CK's permute_vectors_i4x4_b preshuffle layout.
struct CodebookPack8
{
    const ck_tile::bf16_t* lds_codebook; // pointer to 16 bf16 values in LDS

    CK_TILE_DEVICE ck_tile::bf16x4_t lookup_bf16x4(int q) const
    {
        return ck_tile::bf16x4_t{lds_codebook[(q >> 0) & 0xf],
                                 lds_codebook[(q >> 16) & 0xf],
                                 lds_codebook[(q >> 4) & 0xf],
                                 lds_codebook[(q >> 20) & 0xf]};
    }

    // Same signature as PassThroughPack8::operator()(bf16x8_t&, const pk_int4x4_t&)
    CK_TILE_DEVICE void operator()(ck_tile::bf16x8_t& y, const ck_tile::pk_int4x4_t& x) const
    {
        int raw = ck_tile::bit_cast<int>(x);
        y.lo    = lookup_bf16x4(raw);
        y.hi    = lookup_bf16x4(raw >> 8);
    }
};

} // namespace turboquant

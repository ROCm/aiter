// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Inclusive add-scans on the GCN DPP row_shr / row_bcast sequence. Pure VALU,
// where a __shfl_up tree costs a ds_bpermute and an LDS wait per step.

#include "opus/opus.hpp"

namespace aiter {

// x + the DPP-moved x. `old` is 0, so lanes the masks or the row edge exclude
// contribute nothing.
template <int ctrl, int row_mask, int bank_mask, typename T>
__device__ __forceinline__ T dpp_add(T x)
{
    return x + opus::upd_dpp(T{0},
                             x,
                             opus::number<ctrl>{},
                             opus::number<row_mask>{},
                             opus::number<bank_mask>{},
                             opus::bool_constant<false>{});
}

// Inclusive add-scan from the start of each 16-lane DPP row, exact on the
// row's first `Lanes` lanes.
template <int Lanes = 16, typename T>
__device__ __forceinline__ T row_inclusive_sum_dpp(T x)
{
    static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8 || Lanes == 16);
    x = dpp_add<0x111, 0xf, 0xf>(x); // row_shr:1
    if constexpr(Lanes > 2)
        x = dpp_add<0x112, 0xf, 0xf>(x); // row_shr:2
    if constexpr(Lanes > 4)
        x = dpp_add<0x114, 0xf, 0xe>(x); // row_shr:4
    if constexpr(Lanes > 8)
        x = dpp_add<0x118, 0xf, 0xc>(x); // row_shr:8
    return x;
}

// Inclusive add-scan across one wave64. Take T unsigned when two 16-bit counts
// share a dword: the total may set bit 31 though neither field carries.
template <typename T>
__device__ __forceinline__ T wave_inclusive_sum_dpp(T x)
{
    x = row_inclusive_sum_dpp(x);
    x = dpp_add<0x142, 0xa, 0xf>(x); // row_bcast:15
    x = dpp_add<0x143, 0xc, 0xf>(x); // row_bcast:31
    return x;
}

} // namespace aiter

// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared key and search for the generated (M, N, K, cu_num) -> kernel tables
// emitted by gen_instances.py :: gen_lookup_dict. One build can bake several CU
// counts of one arch (AITER_GPU_TARGETS=gfx950:128;gfx950:256), so cu_num is
// part of the key: without it the row read last would evict the other SKU's
// winner and one of the two parts would run the wrong kernel for every shape
// tuned on both.
#pragma once

#include <algorithm>

struct OpusLookupKey
{
    int M;
    int N;
    int K;
    int CU;
};

// (M, N, K) only. Entries are emitted sorted on (M, N, K, cu_num), so this is a
// valid (non-strict) ordering over the same array and lower_bound with it lands
// on the first entry of a shape's block.
template <typename Entry>
constexpr bool opus_shape_less(const Entry& a, const Entry& b) noexcept
{
    if (a.key.M != b.key.M) return a.key.M < b.key.M;
    if (a.key.N != b.key.N) return a.key.N < b.key.N;
    return a.key.K < b.key.K;
}

template <typename Entry>
constexpr bool opus_shape_eq(const Entry& a, const Entry& b) noexcept
{
    return a.key.M == b.key.M && a.key.N == b.key.N && a.key.K == b.key.K;
}

// Return the winner tuned for this device's CU count. With fallback enabled,
// prefer a legacy CU=0 shape-only entry, then preserve the historical behavior
// of using the shape's first available row. Callers that run before a parallel
// exact-CU table disable fallback so they cannot shadow that table's winner.
template <typename Entry>
inline const Entry* opus_lookup_find(const Entry* first, const Entry* last,
                                     int M, int N, int K, int cu_num,
                                     bool allow_fallback = true) noexcept
{
    const Entry needle{{M, N, K, 0}, nullptr};
    const Entry* it = std::lower_bound(first, last, needle, opus_shape_less<Entry>);
    if (it == last || !opus_shape_eq(*it, needle)) return nullptr;
    const Entry* legacy = nullptr;
    for (const Entry* p = it; p != last && opus_shape_eq(*p, needle); ++p)
    {
        if (p->key.CU == cu_num) return p;
        if (p->key.CU == 0) legacy = p;
    }
    return allow_fallback ? (legacy != nullptr ? legacy : it) : nullptr;
}

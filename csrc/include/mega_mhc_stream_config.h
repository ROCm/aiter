// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#ifndef MHC_STREAM_VARIANT
#error "Build with a Mega-MHC stream variant selected by mega_mhc.py"
#endif

namespace aiter::mega_mhc_stream {

struct Options
{
    int split, chunks, groups, lds_chunks, fixed_tokens, decode_rows, prefetch;
    bool segment;
};

// IDs match _STREAM_VARIANTS in aiter/ops/mega_mhc.py. These are compile-time
// choices, so unused branches and LDS storage disappear from each JIT module.
inline constexpr Options options[] = {
    // K   chunks groups LDS fixed_T decode prefetch segment
    {32, 5, 1, 6, 4, 4, 4, false},    //  0 decode4
    {32, 5, 1, 6, 0, 4, 4, false},    //  1 decode_small
    {32, 5, 1, 6, 0, 8, 8, false},    //  2 decode8
    {32, 5, 1, 6, 16, 0, 16, false},  //  3 small16
    {32, 5, 1, 6, 128, 0, 16, false}, //  4 small128
    {32, 5, 1, 6, 0, 0, 16, false},   //  5 small
    {16, 5, 2, 6, 0, 0, 4, false},    //  6 mid256
    {10, 4, 2, 4, 0, 0, 4, true},     //  7 mid384
    {8, 2, 2, 2, 0, 0, 4, false},     //  8 mid512
    {4, 4, 2, 4, 0, 0, 4, false},     //  9 mid1024
    {2, 2, 2, 2, 0, 0, 4, false},     // 10 mid2048
    {1, 4, 1, 4, 0, 0, 4, true},      // 11 large
    {1, 4, 1, 4, 4160, 0, 4, true},   // 12 tail (large + generic K=32)
    {0, 2, 1, 2, 0, 0, 4, false},     // 13 generic (runtime K)
};

template <int ID, bool POST = false>
struct Tuning
{
    static_assert(ID >= 0 && ID < 14);
    // The workspace reserves the base split count; a post call can use a
    // smaller prefix. Explicit split_k requests select generic instead.
    static constexpr Options selected =
        POST && ID >= 3 && ID <= 5 ? Options{20, 2, 2, 2, 0, 0, 5, true} : options[ID];
    static constexpr int split        = selected.split;
    static constexpr int chunks       = selected.chunks;
    static constexpr int groups       = selected.groups;
    static constexpr int threads      = 256 * groups;
    static constexpr int lds_chunks   = selected.lds_chunks;
    static constexpr int fixed_tokens = selected.fixed_tokens;
    static constexpr int decode_rows  = selected.decode_rows;
    static constexpr int prefetch     = selected.prefetch;
    static constexpr bool segment     = selected.segment;
    static constexpr bool distributed = ID <= 10;
    static_assert(!distributed || 5120 % (split * groups * 32) == 0);
};

} // namespace aiter::mega_mhc_stream

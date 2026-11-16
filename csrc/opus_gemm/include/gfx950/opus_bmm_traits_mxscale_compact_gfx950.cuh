// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"

template <int BM, int BN, int BK, int WM_, int WN_, int NB_, bool NT_, bool EARLY_B_,
          bool ROW_B_ = false>
struct opus_bmm_mxscale_compact_traits_gfx950
{
    static constexpr int B_M = BM, B_N = BN, B_K = BK;
    static constexpr int WM = WM_, WN = WN_, NB = NB_, BLOCK_SIZE = WM * WN * 64;
    static constexpr bool NT = NT_, EARLY_B = EARLY_B_, ROW_B = ROW_B_;
};

// Only the fields consumed by compact BMM; dimensions and strides remain
// runtime values.
struct opus_bmm_compact_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int m, n, k, batch;
    // Element strides of A, x_scale and Y along M and batch; K / N is unit.
    int sam, sab, ssm, ssb, sym, syb;
};
static_assert(sizeof(opus_bmm_compact_kargs_gfx950) == 80);

// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_vgpr_gfx950.cuh"

template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int FixedK = 7168, int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false, int GroupM = 0, int PublishM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits<
        BM, BN, NB, BD, BC, XCD, FixedK, WN, WGPerCU, ACache, StoreCache,
        Output, BlockK, Contiguous, PinA, PinC, GroupM> {
    // Select the M repeat that issues the next A-stage prefetch.
    static constexpr int PUBLISH_M = PublishM;

    // Publish before the final M repeat without changing the fragment queue.
    static_assert(PublishM >= 0 && PublishM < BM / 16 - 1);
};

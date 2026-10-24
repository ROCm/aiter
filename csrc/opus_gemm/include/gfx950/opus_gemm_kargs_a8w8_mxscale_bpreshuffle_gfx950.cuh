// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

struct opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 {
    const void *__restrict__ ptr_a;
    const void *__restrict__ ptr_b;
    void *__restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    const void *__restrict__ ptr_sfa;
    const void *__restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};
static_assert(sizeof(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) == 96);

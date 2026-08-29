// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 (MI450) a8w8 mxscale BMM with a preshuffled B -- KERNEL INSTANTIATION
// ONLY. The public entry point lives in opus_bmm.cu with every other BMM
// frontend; this TU exists solely because the tile is hand-written and so has no
// codegen'd *.device.cu to be instantiated into yet. When it moves into
// gen_instances_gfx1250.py (see the note at the bottom of the launcher header)
// this file goes away entirely and nothing else has to change.
//
// The instantiation is NOT guarded by __HIP_DEVICE_COMPILE__, deliberately:
// hipcc emits the launch stub and the fatbin registration from the HOST pass, so
// an instantiation visible only to the device pass leaves an undefined
// __device_stub__ and the launch in opus_bmm.cu silently resolves to nothing.
// The arch gate lives INSIDE the pipeline instead (`#if defined(__gfx1250__)`
// around the body, `(void)kargs` otherwise), so a non-gfx1250 device pass
// compiles an empty kernel rather than opus::tdm on the wrong target.

#include "opus_gemm_utils.cuh"
#include "gfx1250/opus_bmm_pipeline_a8w8_mxscale_bpreshuffle_gfx1250.cuh"

// Both C dtypes the frontend can dispatch. The traits alias is the shared one
// from the traits header -- see the comment there for why it is not spelled out
// again here.
template __global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250<
    opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250<bf16_t>>(
    opus_bmm_a8w8_mxscale_kargs_gfx1250);
template __global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250<
    opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250<fp32_t>>(
    opus_bmm_a8w8_mxscale_kargs_gfx1250);

// Decode tiles (kid 1..4). Same two C dtypes each; see the traits header for
// what each variant adds. They exist to be A/B'd against kid 0 at the DSV4
// decode shapes, where kid 0's grid leaves 94% of the CUs idle. kid4 is the odd
// one out: it is the only tile that is not 128 threads, and its A/B partner is
// kid1 rather than kid0.
#define OPUS_BMM_BPRESHUF_INST(TILE)                                     \
    template __global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250< \
        TILE<bf16_t>>(opus_bmm_a8w8_mxscale_kargs_gfx1250);              \
    template __global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250< \
        TILE<fp32_t>>(opus_bmm_a8w8_mxscale_kargs_gfx1250)

OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_wg2_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_k512_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w4_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n128_w6_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gn128_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_w6_gn128_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n192_w6_gn128_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfab_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_sfa_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_sfa_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_sfab_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n256_sfab_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm32_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_tdm128_gfx1250);

#undef OPUS_BMM_BPRESHUF_INST

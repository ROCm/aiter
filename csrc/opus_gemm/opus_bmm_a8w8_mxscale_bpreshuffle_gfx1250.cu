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

// Prefill tile candidates -- see the traits header for sizing rationale.
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_m256_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_n256_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_m256n256_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_w6_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_bk128_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_w6_2x2_gfx1250);
OPUS_BMM_BPRESHUF_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_pf_m256_bk128_gfx1250);

#undef OPUS_BMM_BPRESHUF_INST

// ===========================================================================
// NON-SPECIALIZED instantiations (every wave loads and computes).
// ===========================================================================
// Separate kernel symbol and separate pipeline header, so the twenty-five tiles
// above keep their exact codegen. See the _nospec pipeline's header for why the
// two cannot simply share a template flag.
#include "gfx1250/opus_bmm_pipeline_a8w8_mxscale_bpreshuffle_nospec_gfx1250.cuh"

#define OPUS_BMM_BPRESHUF_NS_INST(TILE)                                        \
    template __global__ void                                                   \
    bmm_a8w8_mxscale_bpreshuffle_nospec_kernel_gfx1250<TILE<bf16_t>>(          \
        opus_bmm_a8w8_mxscale_kargs_gfx1250);                                  \
    template __global__ void                                                   \
    bmm_a8w8_mxscale_bpreshuffle_nospec_kernel_gfx1250<TILE<fp32_t>>(          \
        opus_bmm_a8w8_mxscale_kargs_gfx1250)

OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns128_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns256_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns256_gn128_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns128_gn128_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns256_gn128_sf_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_ns128_gn128_sf_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_fly256_gfx1250);
OPUS_BMM_BPRESHUF_NS_INST(opus_bmm_a8w8_mxscale_bpreshuffle_tile_fly256_nb4_gfx1250);

#undef OPUS_BMM_BPRESHUF_NS_INST

// ===========================================================================
// CLUSTER-LAUNCH, FUSED SPLIT-K instantiations.
// ===========================================================================
// Kernel:  gemm_a8w8_mxscale_bpreshuffle_clusterclaunch_kernel_gfx1250
// Frontend: opus_bmm_a8w8_mxscale_bpreshuffle_clusterclaunch() in opus_bmm.cu
//
// THIS IS THE LIST TO EDIT to give a tile the cluster path. Each tile listed
// here costs |splitK| x |mClusterWg| x 2 dtypes kernels and roughly 2.6 s of
// build each, which is why the set is deliberately small rather than mirroring
// all 15 kids of the non-cluster entry point. Adding a tile means: one line
// here, one case in the opus_bmm.cu switch, and its id in that function's
// kernelId AITER_CHECK.
//
// DataWs follows the C dtype (bf16 C -> bf16 partials, fp32 C -> fp32), which
// is the frontend's convention -- see the comment on OPUS_BMM_CC_LAUNCH_ONE.
// Instantiating a bf16-C / fp32-ws mix would be legal but nothing dispatches to
// it, so it is left out rather than built and never called.
#include "gfx1250/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_clusterclaunch_gfx1250.cuh"

// DataWs is fp32_t for BOTH C dtypes, not DC. See the measurement note on
// OPUS_BMM_CC_LAUNCH_ONE in opus_bmm.cu: bf16 partials put 0.05%-1.3% of cells
// outside the project's atol gate at splitK>1 through cancellation, while fp32
// partials are bit-exact. The two sites must agree -- the launcher validates the
// workspace in bytes against sizeof(DataWs).
#define OPUS_BMM_BPRESHUF_CC_ONE(TILE, DC, SK, MC)                            \
    template __global__ void                                                  \
    gemm_a8w8_mxscale_bpreshuffle_clusterclaunch_kernel_gfx1250<              \
        TILE<DC>, SK, fp32_t, MC, DC>(opus_gemm_cluster_claunch_kargs_gfx1250)

// splitK 1/2/4/8 at one mClusterWg, both C dtypes.
#define OPUS_BMM_BPRESHUF_CC_SK(TILE, MC)          \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 1, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 1, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 2, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 2, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 4, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 4, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 8, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 8, MC)

// PREFILL tiles get mClusterWg 1 and 2: B_M=128, so a real prefill M has more
// than one M-tile and the B multicast has peers to reach.
#define OPUS_BMM_BPRESHUF_CC_INST_PREFILL(TILE) \
    OPUS_BMM_BPRESHUF_CC_SK(TILE, 1);           \
    OPUS_BMM_BPRESHUF_CC_SK(TILE, 2)

// DECODE tiles get mClusterWg 1 only: B_M=16 and m <= 16 at every shape these
// exist for, so ceil(M/B_M) is 1 and a second peer would be an out-of-range
// workgroup with nothing to multicast to. The frontend rejects mClusterWg>1 for
// these ids, so the missing instantiations are unreachable, not a latent gap.
#define OPUS_BMM_BPRESHUF_CC_INST_DECODE(TILE) \
    OPUS_BMM_BPRESHUF_CC_SK(TILE, 1)

OPUS_BMM_BPRESHUF_CC_INST_PREFILL(opus_bmm_a8w8_mxscale_bpreshuffle_tile_gfx1250);
OPUS_BMM_BPRESHUF_CC_INST_PREFILL(opus_bmm_a8w8_mxscale_bpreshuffle_tile_sfa_gfx1250);
OPUS_BMM_BPRESHUF_CC_INST_DECODE(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n32_gfx1250);
OPUS_BMM_BPRESHUF_CC_INST_DECODE(opus_bmm_a8w8_mxscale_bpreshuffle_tile_dec_n64_w6_gfx1250);

#undef OPUS_BMM_BPRESHUF_CC_INST_DECODE
#undef OPUS_BMM_BPRESHUF_CC_INST_PREFILL
#undef OPUS_BMM_BPRESHUF_CC_SK
#undef OPUS_BMM_BPRESHUF_CC_ONE

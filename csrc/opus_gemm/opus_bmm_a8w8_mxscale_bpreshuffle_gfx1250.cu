#include "opus_gemm_utils.cuh"
#include "gfx1250/opus_bmm_pipeline_a8w8_mxscale_bpreshuffle_gfx1250.cuh"
#include "gfx1250/opus_bmm_pipeline_a8w8_mxscale_bpreshuffle_nospec_gfx1250.cuh"
#include "gfx1250/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_clusterclaunch_gfx1250.cuh"

#define OPUS_BMM_BPRESHUF_INST(TILE)                                     \
    template __global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250< \
        TILE<bf16_t>>(opus_bmm_a8w8_mxscale_kargs_gfx1250);              \
    template __global__ void bmm_a8w8_mxscale_bpreshuffle_kernel_gfx1250< \
        TILE<fp32_t>>(opus_bmm_a8w8_mxscale_kargs_gfx1250)

#define OPUS_BMM_BPRESHUF_NS_INST(TILE)                                        \
    template __global__ void                                                   \
    bmm_a8w8_mxscale_bpreshuffle_nospec_kernel_gfx1250<TILE<bf16_t>>(          \
        opus_bmm_a8w8_mxscale_kargs_gfx1250);                                  \
    template __global__ void                                                   \
    bmm_a8w8_mxscale_bpreshuffle_nospec_kernel_gfx1250<TILE<fp32_t>>(          \
        opus_bmm_a8w8_mxscale_kargs_gfx1250)

#define OPUS_BMM_BPRESHUF_CC_ONE(TILE, DC, SK, MC)                            \
    template __global__ void                                                  \
    gemm_a8w8_mxscale_bpreshuffle_clusterclaunch_kernel_gfx1250<              \
        TILE<DC>, SK, fp32_t, MC, DC>(opus_gemm_cluster_claunch_kargs_gfx1250)

#define OPUS_BMM_BPRESHUF_CC_SK(TILE, MC)          \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 1, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 1, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 2, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 2, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 4, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 4, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, bf16_t, 8, MC); \
    OPUS_BMM_BPRESHUF_CC_ONE(TILE, fp32_t, 8, MC)

#define OPUS_BMM_BPRESHUF_CC_INST_PREFILL(TILE) \
    OPUS_BMM_BPRESHUF_CC_SK(TILE, 1);           \
    OPUS_BMM_BPRESHUF_CC_SK(TILE, 2)

#define OPUS_BMM_BPRESHUF_CC_INST_DECODE(TILE) \
    OPUS_BMM_BPRESHUF_CC_SK(TILE, 1)


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

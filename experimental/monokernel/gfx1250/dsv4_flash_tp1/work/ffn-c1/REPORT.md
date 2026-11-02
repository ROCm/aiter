# Flash FFN ablations: 1K/1K concurrency 1

This supersedes the C64 table for the intended monokernel target. All inputs have one true token row. Four independent experiments ran on four idle gfx1250 GPUs, one GPU per experiment. FFN dimensions do not depend on sequence length: hidden [1,4096], residual [1,4,4096], merged shared gate/up weights [4096,4096], shared down weights [4096,2048], router weights [256,4096], routed top-6 out of 256 experts with intermediate dimension 2048.

These are fresh isolated microbenchmarks using the installed ATOM methods and checkpoint weights, not a new serving throughput run. Each benchmark ran two rounds; all reported medians below come from round 2. Flat HIP graphs clone kernel nodes with stable buffer addresses, avoiding child-graph overhead and unequal temporary-buffer footprints. 128 repetitions, 5 warmups and 40 alternating samples per round, except routed MoE (16 repetitions, 3 warmups and 20 alternating samples). Both rounds and raw samples are saved.

| ATOM operation/group | Candidate ablation | Native us | Candidate us |
|---|---|---:|---:|
| Attention mHC post + FFN mHC pre + RMSNorm + shared-input quant | Fused quant in mHC reduction | 6.51270 | 4.81689 |
| Router BF16 projection (layer 0) | Native retained | 3.00012 | — |
| HERD routing | Native retained | 6.81831 | — |
| Shared input FP8 quant | Native retained | 1.30790 | — |
| Shared gate/up projection | Existing selected C1 Gluon WMMA/TDM GEMM | 27.75530 | 4.20942 |
| Clamped SiLU times up | Native retained | 1.45139 | — |
| Shared down-input FP8 quant | Native retained | 1.29178 | — |
| Shared down projection | Existing selected C1 Gluon WMMA/TDM GEMM | 14.25195 | 2.70061 |
| Complete shared expert chain (five launches) | Same chain with candidate GEMMs | 47.43491 | 14.06763 |
| Routed input quant + expert GEMMs/SwiGLU/requant + combine | Separate quant + persistent fused MoE | 13.13963 | 14.70947 |
| Shared/routed add + extracted mHC post | Single combine_post kernel | 4.58672 | 1.63510 |

## Interpretation and boundaries

The complete shared-expert chain is 3.37x faster with the selected C1 Gluon GEMMs. Both paths contain five launches; this gain is from GEMM implementation/configuration, not additional fusion. Native C1 dispatch selects gemm_a8w8_mx128_t16x1024x128_mw1_nw4_nb2_sk1_cm1_cn1 for both projections. The candidate explicitly uses the existing selected_shared.json entries keyed 1,4096,4096 and 1,4096,2048. The routed MoE persistent prototype remains 1.12x slower than native and is not a recommended replacement on these results. No serving integration or end-to-end gain is claimed.

Rows overlap; do not sum them. Input quant appears both in mHC grouping and shared chain; shared constituent rows are parts of the full chain. Whole-chain times are measured directly, not sums of independent operator timings.

The first row includes attention post because ATOM normally folds it into FFN pre. Our previous full-attention prototype already contains attention post. Adopting this grouping requires moving that boundary, not duplicating post. Native FFN-pre/RMSNorm alone after completed attention post is 7.26304 us; no independent candidate is measured for that cut. Native post/pre/RMSNorm without quant is 4.60735 us.

The last row is an extracted endpoint: native BF16 add, residual unshuffle, HIP mHC post versus one TDM combine_post kernel. Serving normally folds FFN post into the next layer attention pre; the three-launch extracted reference is not a literal serving launch sequence. Standalone shared/routed BF16 add is 1.21353 us. Layer-3 router projection is 2.99964 us.

## Validation

All compared outputs were bitwise equal in the tested cases: mHC all six outputs across 16 cases, shared up/down/full chain across eight input seeds, add/post across 16 cases, routed MoE initial case plus 16 additional input cases. MoE uses fixed HERD routes derived from checkpoint layer-0 gate weights and synthetic input, with zero health flags. This is not validation across arbitrary changing routes, captured serving activations, or GSM8K. Early-layer nominal hash routing is not substituted for the actual current Triton/HERD path.

## Reproduction

The compatible container is vigokhal_dsv4_tp4, image rocm/fw-bringup-mlperf:gfx1250-atom-20261015. ATOM commit f19778df4054e94d64d6f6bf6dd68f5681bd5945; AITER 7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe. Compatible installed ATOM/AITER/PyTorch/Triton and /data/DeepSeek-V4-Flash are external dependencies. Archive includes scripts, smaller fixtures, results, profiles, ISA, graph helper and hybrid wrapper at their expected /work-relative paths. It does not duplicate the checkpoint.

Extract into the directory mounted as /work, then run each bench.py from its directory with an idle GPU selected. Example:

```sh
docker exec -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/app/aiter:/app/ATOM vigokhal_dsv4_tp4 bash -c 'cd /work/ffn-c1/mhc && python bench.py'
```

Other directories are shared-router, moe and combine-post. Run moe/validate_extra.py for extra MoE checks and routing/add timings. No server is needed for these microbenchmarks. All four GPU jobs completed.

## Exact kernel symbols

The following are profiler symbols, not the latency source. Names include compiler specialization parameters. The corresponding traces retain launch metadata.

### mhc/actual_ffn_pre_norm_only-trace.json

- `mhc_res_unshuffle_h4096_hc4_ks64_b2`
- `void aiter::mhc_pre_gemm_sqrsum_kernel<std::bfloat16_t, 4, 64, 16, 64, true, true>(float*, float*, std::bfloat16_t*, float*, int, int, int, int, int, int, int)`
- `void aiter::mhc_pre_big_fuse_rmsnorm_kernel<std::bfloat16_t, 9, 4, 1, 4096, 1024, 1024, false>(float*, float*, std::bfloat16_t*, float*, float*, float*, float*, std::bfloat16_t*, std::bfloat16_t*, int, int, int, int, float, float, float, float, float, int, int)`

### mhc/actual_post_pre_norm-trace.json

- `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel_C_4096_BLOCK_M_16_K_LOOP_1_SPLIT_K_64_NUM_STAGES_2_RES_SHUFFLED_1_W_PRESHUFFLED_1`
- `_mhc_pre_reduce_apply_gfx1250_kernel_C_4096_ROWS_1_CHUNK_1024_N_CHUNKS_4_S_64_FUSE_RMSNORM_1_RES_SHUFFLED_1_FUSE_QUANT_0`

### mhc/actual_post_pre_norm_quant-trace.json

- `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel_C_4096_BLOCK_M_16_K_LOOP_1_SPLIT_K_64_NUM_STAGES_2_RES_SHUFFLED_1_W_PRESHUFFLED_1`
- `_mhc_pre_reduce_apply_gfx1250_kernel_C_4096_ROWS_1_CHUNK_1024_N_CHUNKS_4_S_64_FUSE_RMSNORM_1_RES_SHUFFLED_1_FUSE_QUANT_0`
- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`

### mhc/hybrid_post_pre_norm_quant-trace.json

- `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel_C_4096_BLOCK_M_16_K_LOOP_1_SPLIT_K_64_NUM_STAGES_2_RES_SHUFFLED_1_W_PRESHUFFLED_1`
- `_mhc_pre_reduce_apply_gfx1250_kernel_C_4096_ROWS_1_CHUNK_1024_N_CHUNKS_4_S_64_FUSE_RMSNORM_1_RES_SHUFFLED_1_FUSE_QUANT_1`

### shared-router/candidate_down-trace.json

- `_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_128_A_SCALE_TRANSPOSED_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048_EVEN_K_1_num_warps_4_waves_per_eu_1_cache_modifier_NONE_NUM_BUFFERS_3_LOOP_UNROLL_FACTOR_1_B_SCALE_TDM_1_CTAS_M_1_CTAS_N_1`

### shared-router/candidate_shared_chain-trace.json

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_128_A_SCALE_TRANSPOSED_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_num_warps_4_waves_per_eu_1_cache_modifier_NONE_NUM_BUFFERS_3_LOOP_UNROLL_FACTOR_1_B_SCALE_TDM_1_CTAS_M_1_CTAS_N_1`
- `_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_if`
- `_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_128_A_SCALE_TRANSPOSED_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048_EVEN_K_1_num_warps_4_waves_per_eu_1_cache_modifier_NONE_NUM_BUFFERS_3_LOOP_UNROLL_FACTOR_1_B_SCALE_TDM_1_CTAS_M_1_CTAS_N_1`

### shared-router/candidate_up-trace.json

- `_gemm_mxfp8_preshuffle_gfx1250_bandwidth_bound_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_64_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_A_SCALE_K_GROUP_128_A_SCALE_TRANSPOSED_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_4096_EVEN_K_1_num_warps_4_waves_per_eu_1_cache_modifier_NONE_NUM_BUFFERS_3_LOOP_UNROLL_FACTOR_1_B_SCALE_TDM_1_CTAS_M_1_CTAS_N_1`

### shared-router/down_quant-trace.json

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`

### shared-router/input_quant-trace.json

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`

### shared-router/native_down-trace.json

- `gemm_a8w8_mx128_t16x1024x128_mw1_nw4_nb2_sk1_cm1_cn1`

### shared-router/native_router_layer0-trace.json

- `_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16_BLOCK_N_16_BLOCK_K_512_NUM_BUFFERS_3_LAYOUT_TN_USE_ACTIVATION_0_ADD_BIAS_0`

### shared-router/native_router_layer3-trace.json

- `_gemm_a16w16_gfx1250_bandwidth_bound_kernel_BLOCK_M_16_BLOCK_N_16_BLOCK_K_512_NUM_BUFFERS_3_LAYOUT_TN_USE_ACTIVATION_0_ADD_BIAS_0`

### shared-router/native_shared_chain-trace.json

- `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32ELi128ELb1ELi64ELb1ELb0ELi0ELb0ELb0ELb0ELb0ELb0EEEvPT0_PfPKT_PKfliilPKii`
- `gemm_a8w8_mx128_t16x1024x128_mw1_nw4_nb2_sk1_cm1_cn1`
- `_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_if`

### shared-router/native_up-trace.json

- `gemm_a8w8_mx128_t16x1024x128_mw1_nw4_nb2_sk1_cm1_cn1`

### shared-router/swiglu-trace.json

- `_ZN5aiter18act_and_mul_kernelIDF16bDF16bTnPFfRKT_EXadL_ZNS_11silu_kernelIDF16bEEfS3_EELi8ELb1ELi3EEEvPT0_PS2_if`

### moe/candidate_prototype-trace.json

- `_downcast_to_mxfp_BLOCK_SIZE_OUT_DIM_128_BLOCK_SIZE_QUANT_DIM_32_DEQUANT_SCALE_ROUNDING_MODE_0_POW2_SCALE_0`
- `fused`

### moe/native_actual-trace.json

- `_downcast_to_mxfp_BLOCK_SIZE_OUT_DIM_128_BLOCK_SIZE_QUANT_DIM_32_DEQUANT_SCALE_ROUNDING_MODE_0_POW2_SCALE_0`
- `_moe_gemm_a8w4_decode_BLOCK_M_16_BLOCK_N_64_BLOCK_K_512_SWIZZLE_MX_SCALE_GFX1250_SCALE_APPLY_SWIGLU_1_num_warps_4_NUM_BUFFERS_3_HAS_MX_OUT_1_EP_SCATTER_0`
- `_moe_gemm_a8w4_decode_BLOCK_M_16_BLOCK_N_64_BLOCK_K_512_SWIZZLE_MX_SCALE_GFX1250_SCALE_APPLY_SWIGLU_0_num_warps_4_NUM_BUFFERS_3_HAS_MX_OUT_0_EP_SCATTER_0`
- `reduce_grouped_gluon`

### moe/routing-add-trace.json

- `_topk_BLOCK_M_1_BLOCK_N_128_BLOCK_S_128_BLOCK_SP_128_N_EXPTS_PAD_256_N_EXPTS_ACT_6_N_EXPTS_ACT_PAD_8_SCORE_MODE_SQRTSOFTPLUS_APPLY_SOFTMAX_0_HAS_BIAS_0_APPLY_RENORM_1_WRITE_POP_0`
- `_combined_routing_fused_BLOCK_M_1_EVEN_M_1_N_EXPTS_ACT_6_N_EXPTS_ACT_PAD_8_N_EXPTS_TOT_256_N_BLKS_BITMATRIX_8_tile_dim_log2_4_BLOCK_A_256_EQUAL_A_1_USE_TDM_1`
- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)`

### combine-post/trace.json

- `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)`
- `mhc_res_unshuffle_h4096_hc4_ks64_b2`
- `void aiter::mhc_post_kernel<std::bfloat16_t, 4, 4, 1024, false>(std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float*, float*, int, int, int, int, int)`
- `combine_post`

Correction from full-layer integration: HERD is enabled by environment but defaults to a minimum of 16 rows. The measured C1 routing row is stock top-6, not active HERD. Timing is unchanged.

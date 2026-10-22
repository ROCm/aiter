# gfx1250-strict kernel inventory (skeleton)

Status: skeleton generated 2026-10-22, build/classification columns pending
strict-capable toolchain run. Tracked by AITERKER-209 / ROCM-31030.

## JIT/prebuilt modules (from aiter/jit/optCompilerConfig.json)

| module | family | gfx1250 signal | strict build | classification | owner | notes |
|--------|--------|----------------|--------------|----------------|-------|-------|
| module_attention_asm | ASM | - | pending | pending | TBD | |
| module_attention_ps_reduce_asm | ASM | - | pending | pending | TBD | |
| module_f4gemm_asm | ASM | name | pending | pending | TBD | |
| module_fmha_fwd_mxfp8_asm | ASM | name | pending | pending | TBD | |
| module_fmha_fwd_with_sink_asm | ASM | name | pending | pending | TBD | |
| module_fmha_fwd_with_sink_varlen_asm | ASM | name | pending | pending | TBD | |
| module_fmha_v3_bwd | ASM | name | pending | pending | TBD | |
| module_fmha_v3_fwd | ASM | - | pending | pending | TBD | |
| module_fmha_v3_varlen_bwd | ASM | name | pending | pending | TBD | |
| module_fmha_v3_varlen_fwd | ASM | - | pending | pending | TBD | |
| module_fmha_v4_fwd | ASM | - | pending | pending | TBD | |
| module_gemm_a16w16_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a16w8_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a4w4_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a4w6_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a6w4_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a6w6_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a8w8_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_asm | ASM | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_bpreshuffle_asm | ASM | - | pending | pending | TBD | |
| module_mla_asm | ASM | name | pending | pending | TBD | |
| module_mla_sparse_prefill_asm | ASM | name | pending | pending | TBD | |
| module_moe_asm | ASM | - | pending | pending | TBD | |
| module_moe_fmoe_asm | ASM | - | pending | pending | TBD | |
| module_moe_topksoftmax_asm | ASM | - | pending | pending | TBD | |
| module_mxfp8fp4gemm_asm | ASM | name | pending | pending | TBD | |
| module_pa_decode_bf16_asm | ASM | name | pending | pending | TBD | |
| module_top_k_per_row | ASM | guard | pending | pending | TBD | |
| module_batched_gemm_a8w8 | CK | - | pending | pending | TBD | |
| module_batched_gemm_a8w8_tune | CK | - | pending | pending | TBD | |
| module_batched_gemm_bf16 | CK | - | pending | pending | TBD | |
| module_batched_gemm_bf16_tune | CK | - | pending | pending | TBD | |
| module_deepgemm | CK | - | pending | pending | TBD | |
| module_gemm_a4w4_blockscale | CK | - | pending | pending | TBD | |
| module_gemm_a4w4_blockscale_tune | CK | - | pending | pending | TBD | |
| module_gemm_a8w8 | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_bpreshuffle | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_bpreshuffle_cktile | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_bpreshuffle_tune | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_cktile | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_cktile_tune | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_blockscale_tune | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_bpreshuffle | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_bpreshuffle_tune | CK | - | pending | pending | TBD | |
| module_gemm_a8w8_tune | CK | - | pending | pending | TBD | |
| module_moe_ck2stages | CK | - | pending | pending | TBD | |
| module_moe_cktile2stages | CK | - | pending | pending | TBD | |
| module_quick_all_reduce | CK | - | pending | pending | TBD | |
| module_quick_reduce_scatter | CK | - | pending | pending | TBD | |
| libmha_bwd | HIP | name | pending | pending | TBD | |
| libmha_fwd | HIP | - | pending | pending | TBD | |
| module_activation | HIP | guard | pending | pending | TBD | |
| module_aiter_core | HIP | - | pending | pending | TBD | |
| module_aiter_operator | HIP | - | pending | pending | TBD | |
| module_aiter_unary | HIP | - | pending | pending | TBD | |
| module_attention | HIP | - | pending | pending | TBD | |
| module_cache | HIP | guard | pending | pending | TBD | |
| module_causal_conv1d_fwd_split_qkv | HIP | - | pending | pending | TBD | |
| module_causal_conv1d_update | HIP | - | pending | pending | TBD | |
| module_chunk_gdr_fwd_h | HIP | - | pending | pending | TBD | |
| module_custom | HIP | - | pending | pending | TBD | |
| module_custom_all_reduce | HIP | - | pending | pending | TBD | |
| module_custom_all_reduce_gfx1250 | HIP | name | pending | pending | TBD | |
| module_dsv4_dequant_gather_k | HIP | - | pending | pending | TBD | |
| module_dsv4_rotate_quant | HIP | - | pending | pending | TBD | |
| module_fused_ar_mhc | HIP | guard | pending | pending | TBD | |
| module_fused_qk_norm_mrope_cache_quant_shuffle | HIP | - | pending | pending | TBD | |
| module_fused_qk_norm_rope_cache_quant_shuffle | HIP | guard | pending | pending | TBD | |
| module_fused_qk_rmsnorm_group_quant | HIP | - | pending | pending | TBD | |
| module_fused_qknorm_idxrqknorm | HIP | - | pending | pending | TBD | |
| module_fused_split_gdr_update | HIP | - | pending | pending | TBD | |
| module_gated_rmsnorm_quant | HIP | - | pending | pending | TBD | |
| module_gdr_decode_packed_bf16 | HIP | - | pending | pending | TBD | |
| module_gemm_a8w8_bpreshuffle_cktile | HIP | - | pending | pending | TBD | |
| module_gemm_a8w8_bpreshuffle_cktile_tune | HIP | - | pending | pending | TBD | |
| module_gemm_common | HIP | - | pending | pending | TBD | |
| module_groupnorm | HIP | - | pending | pending | TBD | |
| module_hipbsolgemm | HIP | - | pending | pending | TBD | |
| module_hk_mla_v32_fwd_mi3xx | HIP | - | pending | pending | TBD | |
| module_hk_mla_v40_fwd_mi3xx | HIP | - | pending | pending | TBD | |
| module_inverse_rope_group_quant | HIP | guard | pending | pending | TBD | |
| module_iq2r_moe | HIP | - | pending | pending | TBD | |
| module_mha_batch_prefill | HIP | - | pending | pending | TBD | |
| module_mha_bwd | HIP | name | pending | pending | TBD | |
| module_mha_fwd | HIP | - | pending | pending | TBD | |
| module_mha_fwd_native_splitkv | HIP | - | pending | pending | TBD | |
| module_mha_v4_quant | HIP | - | pending | pending | TBD | |
| module_mha_varlen_bwd | HIP | name | pending | pending | TBD | |
| module_mha_varlen_fwd | HIP | - | pending | pending | TBD | |
| module_mhc | HIP | guard | pending | pending | TBD | |
| module_mla_metadata | HIP | name | pending | pending | TBD | |
| module_mla_reduce | HIP | - | pending | pending | TBD | |
| module_moe_mxfp4_aux | HIP | - | pending | pending | TBD | |
| module_moe_sorting | HIP | - | pending | pending | TBD | |
| module_moe_topk | HIP | - | pending | pending | TBD | |
| module_moe_topk_ck | HIP | - | pending | pending | TBD | |
| module_msa_sparse_attention | HIP | - | pending | pending | TBD | |
| module_norm | HIP | - | pending | pending | TBD | |
| module_pa_metadata | HIP | - | pending | pending | TBD | |
| module_pos_encoding | HIP | - | pending | pending | TBD | |
| module_ps_metadata | HIP | - | pending | pending | TBD | |
| module_quant | HIP | guard | pending | pending | TBD | |
| module_rmsnorm_quant | HIP | name | pending | pending | TBD | |
| module_rocsolgemm | HIP | - | pending | pending | TBD | |
| module_rope_1c_2d_bwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_2d_fwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_cached_bwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_cached_fwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_cached_positions_fwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_cached_positions_offsets_fwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_thd_bwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_thd_fwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_uncached_bwd | HIP | - | pending | pending | TBD | |
| module_rope_1c_uncached_fwd | HIP | - | pending | pending | TBD | |
| module_rope_2c_cached_bwd | HIP | - | pending | pending | TBD | |
| module_rope_2c_cached_fwd | HIP | - | pending | pending | TBD | |
| module_rope_2c_cached_positions_fwd | HIP | - | pending | pending | TBD | |
| module_rope_2c_cached_positions_offsets_fwd | HIP | - | pending | pending | TBD | |
| module_rope_2c_uncached_bwd | HIP | - | pending | pending | TBD | |
| module_rope_2c_uncached_fwd | HIP | - | pending | pending | TBD | |
| module_sample | HIP | - | pending | pending | TBD | |
| module_smoothquant | HIP | - | pending | pending | TBD | |
| module_sp_head_exchange | HIP | - | pending | pending | TBD | |
| module_topk_plain | HIP | guard | pending | pending | TBD | |
| module_vsa_sparse_attention | HIP | - | pending | pending | TBD | |
| module_deepgemm_opus | Opus | name | pending | pending | TBD | |
| module_fmha_fwd_bf16_opus | Opus | - | pending | pending | TBD | |
| module_mla_decode_opus | Opus | - | pending | pending | TBD | |
| module_mla_v4_prefill_opus | Opus | name | pending | pending | TBD | |
| module_moe_opus | Opus | - | pending | pending | TBD | |
| module_moe_sorting_opus | Opus | - | pending | pending | TBD | |
| module_pa_mqa_logits_mxfp4_opus | Opus | name | pending | pending | TBD | |
| module_rmsnorm | Opus | - | pending | pending | TBD | |

## FlyDSL kernel paths (aiter/ops/flydsl)

| path | gfx1250-specific | strict build | classification | owner | notes |
|------|------------------|--------------|----------------|-------|-------|
| __init__.py | - | pending | pending | TBD | |
| batched_gemm_a8w8.py | - | pending | pending | TBD | |
| batched_gemm_a8w8_gfx1250.py | yes | pending | pending | TBD | |
| batched_gemm_a8w8_gfx950.py | - | pending | pending | TBD | |
| batched_gemm_mxfp4.py | - | pending | pending | TBD | |
| bpreshuffle_gemm_gfx1250.py | yes | pending | pending | TBD | |
| causal_conv1d_flydsl.py | - | pending | pending | TBD | |
| causal_conv1d_update_kernels.py | - | pending | pending | TBD | |
| comm_fused_moe_host.py | - | pending | pending | TBD | |
| conv3d_policy.py | - | pending | pending | TBD | |
| conv_kernels.py | - | pending | pending | TBD | |
| dcp_topk_merge.py | - | pending | pending | TBD | |
| fhmoe.py | - | pending | pending | TBD | |
| fmha_bwd_gfx942.py | - | pending | pending | TBD | |
| fmha_kernels.py | - | pending | pending | TBD | |
| fp8_mqa_logits_kernels.py | - | pending | pending | TBD | |
| fused_moe_gfx942.py | - | pending | pending | TBD | |
| gather_kv_b_proj.py | - | pending | pending | TBD | |
| gather_kv_b_proj_gfx1250.py | yes | pending | pending | TBD | |
| gemm_a16w16_gfx1250.py | yes | pending | pending | TBD | |
| gemm_a16w16_policy.py | - | pending | pending | TBD | |
| gemm_a16wfp4.py | - | pending | pending | TBD | |
| gemm_a8w8_bpreshuffle_8wave.py | - | pending | pending | TBD | |
| gemm_a8w8_mxfp8.py | - | pending | pending | TBD | |
| gemm_a8w8_mxfp8_8wave.py | - | pending | pending | TBD | |
| gemm_kernels.py | - | pending | pending | TBD | |
| grouped_gemm_mxfp4.py | - | pending | pending | TBD | |
| grouped_moe_gfx1250.py | yes | pending | pending | TBD | |
| herd_topk.py | - | pending | pending | TBD | |
| hstu_attention.py | - | pending | pending | TBD | |
| kernels/__init__.py | - | pending | pending | TBD | |
| kernels/act.py | - | pending | pending | TBD | |
| kernels/bmm_a8w8_mxscale_gfx950.py | - | pending | pending | TBD | |
| kernels/buffer_ops.py | - | pending | pending | TBD | |
| kernels/causal_conv1d_update.py | - | pending | pending | TBD | |
| kernels/communication_ops_utils.py | - | pending | pending | TBD | |
| kernels/dcp_topk_merge.py | - | pending | pending | TBD | |
| kernels/dpp_utils.py | - | pending | pending | TBD | |
| kernels/fhmoe.py | - | pending | pending | TBD | |
| kernels/flash_attn_func_fp8_gfx950.py | - | pending | pending | TBD | |
| kernels/flash_attn_func_gfx1201.py | - | pending | pending | TBD | |
| kernels/flydsl_dispatch_combine_intranode_kernel.py | - | pending | pending | TBD | |
| kernels/flydsl_dispatch_combine_intranode_op.py | - | pending | pending | TBD | |
| kernels/fused_compress_attn.py | - | pending | pending | TBD | |
| kernels/fused_compress_attn_common.py | - | pending | pending | TBD | |
| kernels/fused_compress_attn_gfx1250.py | yes | pending | pending | TBD | |
| kernels/fused_compress_attn_hca.py | - | pending | pending | TBD | |
| kernels/fused_compress_attn_hca_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gather_gemm_8wave.py | - | pending | pending | TBD | |
| kernels/gather_kv_b_proj_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gdr_common.py | - | pending | pending | TBD | |
| kernels/gdr_decode.py | - | pending | pending | TBD | |
| kernels/gemm_a16w16_gfx950.py | - | pending | pending | TBD | |
| kernels/gemm_a16w16_gfx950_utils.py | - | pending | pending | TBD | |
| kernels/gemm_a16w16_kernel_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gemm_a16wfp4.py | - | pending | pending | TBD | |
| kernels/gemm_a16wfp4_helpers.py | - | pending | pending | TBD | |
| kernels/gemm_a8w8_256x256_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gemm_a8w8_8wave.py | - | pending | pending | TBD | |
| kernels/gemm_a8w8_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gemm_a8w8_splitk_reduce_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gemm_common_gfx1250.py | yes | pending | pending | TBD | |
| kernels/gfx1250_cluster.py | yes | pending | pending | TBD | |
| kernels/herd_topk.py | - | pending | pending | TBD | |
| kernels/kernels_common.py | - | pending | pending | TBD | |
| kernels/layout_utils.py | - | pending | pending | TBD | |
| kernels/mega_mhc.py | - | pending | pending | TBD | |
| kernels/mfma_epilogues.py | - | pending | pending | TBD | |
| kernels/mfma_preshuffle_pipeline.py | - | pending | pending | TBD | |
| kernels/mhc_res_layout.py | - | pending | pending | TBD | |
| kernels/minimax_m3_index_score.py | - | pending | pending | TBD | |
| kernels/minimax_m3_index_score_prefill.py | - | pending | pending | TBD | |
| kernels/mixed_moe_gemm_2stage.py | - | pending | pending | TBD | |
| kernels/mixed_moe_gemm_2stage_common.py | - | pending | pending | TBD | |
| kernels/mla_reduce.py | - | pending | pending | TBD | |
| kernels/moe_contiguous_psum.py | - | pending | pending | TBD | |
| kernels/moe_fused_route_quant_scatter.py | - | pending | pending | TBD | |
| kernels/moe_g2l_lut.py | - | pending | pending | TBD | |
| kernels/moe_gather_reduce.py | - | pending | pending | TBD | |
| kernels/moe_reduce.py | - | pending | pending | TBD | |
| kernels/moe_route_maps.py | - | pending | pending | TBD | |
| kernels/moe_scatter_copy_preshuffle_scale.py | - | pending | pending | TBD | |
| kernels/moe_scatter_copy_token.py | - | pending | pending | TBD | |
| kernels/moe_sorting_kernel.py | - | pending | pending | TBD | |
| kernels/mxfp4_gemm1.py | - | pending | pending | TBD | |
| kernels/mxfp4_gemm2.py | - | pending | pending | TBD | |
| kernels/mxfp4_gemm_common.py | - | pending | pending | TBD | |
| kernels/mxfp4_preshuffle_batched_gemm_gfx1250_tdm.py | yes | pending | pending | TBD | |
| kernels/mxfp4_preshuffle_gfx1250_tdm.py | yes | pending | pending | TBD | |
| kernels/mxfp8_gemm_8wave.py | - | pending | pending | TBD | |
| kernels/mxmoe_dispatcher.py | - | pending | pending | TBD | |
| kernels/mxmoe_gemm_v2.py | - | pending | pending | TBD | |
| kernels/pa_decode_kernel.py | - | pending | pending | TBD | |
| kernels/pa_decode_plan.py | - | pending | pending | TBD | |
| kernels/pa_decode_reduce.py | - | pending | pending | TBD | |
| kernels/preshuffle_gemm.py | - | pending | pending | TBD | |
| kernels/qk_norm_rope_quant.py | - | pending | pending | TBD | |
| kernels/quant_utils.py | - | pending | pending | TBD | |
| kernels/quick_allreduce_int4.py | - | pending | pending | TBD | |
| kernels/silu_and_mul_fq.py | - | pending | pending | TBD | |
| kernels/splitk_epilogue.py | - | pending | pending | TBD | |
| kernels/splitk_fused_epilogue_gfx1250.py | yes | pending | pending | TBD | |
| kernels/swiglu_and_mul.py | - | pending | pending | TBD | |
| kernels/tdm_ops_gfx1250.py | yes | pending | pending | TBD | |
| kernels/tensor_shim.py | - | pending | pending | TBD | |
| kernels/utils.py | - | pending | pending | TBD | |
| linear_attention_kernels.py | - | pending | pending | TBD | |
| linear_attention_prefill_kernels.py | - | pending | pending | TBD | |
| mega_mhc_kernels.py | - | pending | pending | TBD | |
| mega_moe_tp.py | - | pending | pending | TBD | |
| mla_kernels.py | - | pending | pending | TBD | |
| mla_reduce_kernels.py | - | pending | pending | TBD | |
| moe_common.py | - | pending | pending | TBD | |
| moe_kernels.py | - | pending | pending | TBD | |
| moe_sorting.py | - | pending | pending | TBD | |
| mxfp4_gemm1_kernels.py | - | pending | pending | TBD | |
| mxfp4_gemm2_kernels.py | - | pending | pending | TBD | |
| mxfp4_kname.py | - | pending | pending | TBD | |
| mxfp8_bpreshuffle_gemm_gfx1250.py | yes | pending | pending | TBD | |
| mxscale_preshuffle_kernels.py | - | pending | pending | TBD | |
| pa_decode.py | - | pending | pending | TBD | |
| pa_decode_tuning.py | - | pending | pending | TBD | |
| quick_allreduce_int4.py | - | pending | pending | TBD | |
| quick_allreduce_int4_ipc.py | - | pending | pending | TBD | |

## Triton kernel paths (aiter/ops/triton)

| path | strict build | classification | owner | notes |
|------|--------------|----------------|-------|-------|
| _gluon_kernels/ | pending | pending | TBD | |
| _triton_kernels/ | pending | pending | TBD | |
| attention/ | pending | pending | TBD | |
| comms/ | pending | pending | TBD | |
| configs/ | pending | pending | TBD | |
| conv/ | pending | pending | TBD | |
| fusions/ | pending | pending | TBD | |
| gated_delta_net/ | pending | pending | TBD | |
| gemm/ | pending | pending | TBD | |
| gluon/ | pending | pending | TBD | |
| kimi_delta_attn/ | pending | pending | TBD | |
| moe/ | pending | pending | TBD | |
| normalization/ | pending | pending | TBD | |
| quant/ | pending | pending | TBD | |
| rope/ | pending | pending | TBD | |
| utils/ | pending | pending | TBD | |

## Status legend (strict build column)

`optCompilerConfig.json` lists 135 modules; `core.get_args_of_build("all")`
yields 123. The other 12 are not build failures and must not be reported as
such:

- `skip-tune` — 10 modules whose name ends in `tune`; `get_args_of_build`
  skips them by design (they build through the tuning path, not the op path).
- `skip-experimental` — `module_hk_mla_v32_fwd_mi3xx` and
  `module_hk_mla_v40_fwd_mi3xx`, gated behind `AITER_ENABLE_EXPERIMENTAL`.
  Set that variable to include them in the matrix.
- `pass` / `fail` — actual strict-target build result.
- `missing` — requested but absent from the build args; a real gap to chase.

Driver dry-run status (2026-10-22, ROCm 7.0.0 host without a strict target):
module enumeration, child-process plumbing, skip classification and
`--update-md` folding are verified; `pass`/`fail` rows still require the
strict-capable container.

## Container-phase runbook (strict-capable toolchain)

1. Verify the toolchain accepts the strict target:
   `hipcc --offload-arch=gfx1250-strict -c tiny.hip` (must not error)
2. Per-module build matrix:
   `GPU_ARCHS=gfx1250-strict python3 ws2_strict_build_inventory.py`
   then fold results into this file with `--update-md`.
3. Opus .co: `csrc/opus_gemm/gen_co/build_co.py --llvm-bin <custom llvm>`
   requires the custom LLVM with `__builtin_amdgcn_pin_vgpr` and
   `-mllvm -amdgpu-expert-scheduling-mode`; stock ROCm rejects both.
   Gated on Opus owner confirmation per the enablement plan.
4. ASM: `hsa/gfx1250` is B0-only; do not alias into a strict directory.
5. Dispatch sites: `python3 ws2_gfx1250_dispatch_sites.py --update-md`
   refreshes the table below.  Re-run it after every batch of owner decisions;
   a site that gains a strict spelling drops out of the table on its own.
   The `triton-arch` rows cannot be settled until a strict-capable Triton
   reports its target name -- record that spelling here when it is known.

## gfx1250 dispatch sites (A0 decision pending)

<!-- BEGIN generated: gfx1250 dispatch sites -->

Generated by `ws2_gfx1250_dispatch_sites.py`: 150 uses of the exact literal `gfx1250` across 65 files -- 132 runtime comparisons and 18 entries in constant arch allowlists.  On A0 the arch string is `gfx1250-strict`, so a comparison is simply False and an allowlist simply does not contain it: the gfx1250 path is skipped silently, with no error raised.  Each site needs an explicit owner decision -- reuse the gfx1250 path on A0, or keep the fallback.

`arch source` says where the compared string comes from.  `get_gfx` and `get_gfx_runtime` keep the hyphenated spelling, so those sites are known to miss.  `triton-arch` reads Triton's own target name, whose spelling on a strict target is unverified until the toolchain lands; `local` and `unresolved` need a per-site read.

| file | sites | kind | arch source | lines | A0 decision | owner |
|------|-------|------|-------------|-------|-------------|-------|
| aiter/ops/gemm_op_a8w8.py | 8 | compare | get_gfx/local | 181, 231, 780, 1197, 1317, 1503, 2160, 2298 | pending | TBD |
| aiter/ops/flydsl/pa_decode.py | 7 | compare | get_gfx_runtime/local/unresolved | 73, 121, 368, 374, 390, 441, 608 | pending | TBD |
| aiter/ops/triton/attention/unified_attention.py | 7 | compare | unresolved | 54, 385, 439, 480, 508, 530, 549 | pending | TBD |
| aiter/ops/mhc.py | 6 | compare | get_gfx_runtime/local/unresolved | 40, 239, 283, 367, 630, 640 | pending | TBD |
| aiter/ops/triton/attention/pa_mqa_logits.py | 6 | allowlist/compare | get_gfx/unresolved | 43, 333, 694, 702, 740, 752 | pending | TBD |
| aiter/ops/triton/quant/fused_mxfp4_quant.py | 6 | compare | triton-arch | 118, 120, 557, 565, 694, 699 | pending | TBD |
| aiter/mla.py | 5 | compare | get_gfx | 327, 690, 818, 1251, 1421 | pending | TBD |
| aiter/ops/mha.py | 4 | compare | get_gfx | 1949, 1985, 3016, 3779 | pending | TBD |
| aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py | 4 | compare | unresolved | 954, 1272, 1275, 1766 | pending | TBD |
| aiter/ops/triton/fusions/fused_kv_cache.py | 4 | compare | unresolved | 266, 456, 509, 793 | pending | TBD |
| aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py | 4 | allowlist/compare | triton-arch/unresolved | 31, 32, 33, 154 | pending | TBD |
| aiter/ops/triton/utils/_triton/arch_info.py | 4 | compare | triton-arch | 31, 35, 43, 47 | pending | TBD |
| aiter/ops/triton/utils/shuffle.py | 4 | compare | triton-arch | 55, 144, 154, 189 | pending | TBD |
| aiter/ops/asm/mla_decode_v4.py | 3 | compare | unresolved | 289, 352, 701 | pending | TBD |
| aiter/ops/attention.py | 3 | compare | get_gfx | 1283, 1316, 1740 | pending | TBD |
| aiter/ops/pa_sparse_prefill_opus.py | 3 | allowlist/compare | get_gfx_runtime/unresolved | 39, 50, 190 | pending | TBD |
| aiter/ops/triton/attention/mla.py | 3 | compare | unresolved | 42, 117, 388 | pending | TBD |
| aiter/ops/triton/moe/moe_op_gemm_a16w4.py | 3 | compare | triton-arch | 405, 445, 531 | pending | TBD |
| aiter/ops/triton/moe/moe_op_gemm_a8w4.py | 3 | compare | triton-arch | 281, 398, 402 | pending | TBD |
| aiter/aot/flydsl/gemm.py | 2 | compare | local/unresolved | 350, 549 | pending | TBD |
| aiter/fused_moe.py | 2 | compare | get_gfx/get_gfx_runtime | 1075, 1482 | pending | TBD |
| aiter/ops/flydsl/batched_gemm_a8w8_gfx1250.py | 2 | allowlist/compare | get_gfx/unresolved | 36, 189 | pending | TBD |
| aiter/ops/flydsl/batched_gemm_mxfp4.py | 2 | compare | get_gfx | 58, 371 | pending | TBD |
| aiter/ops/flydsl/fmha_kernels.py | 2 | compare | get_gfx | 397, 519 | pending | TBD |
| aiter/ops/flydsl/gather_kv_b_proj.py | 2 | compare | local/unresolved | 193, 484 | pending | TBD |
| aiter/ops/flydsl/gemm_a16w16_policy.py | 2 | allowlist/compare | get_gfx/unresolved | 158, 213 | pending | TBD |
| aiter/ops/flydsl/grouped_moe_gfx1250.py | 2 | compare | get_gfx | 228, 1515 | pending | TBD |
| aiter/ops/flydsl/kernels/fused_compress_attn_hca.py | 2 | compare | get_gfx | 1009, 1259 | pending | TBD |
| aiter/ops/flydsl/kernels/pa_decode_reduce.py | 2 | compare | unresolved | 81, 333 | pending | TBD |
| aiter/ops/shuffle.py | 2 | compare | get_gfx | 140, 431 | pending | TBD |
| aiter/ops/triton/attention/chunk_kda.py | 2 | compare | triton-arch/unresolved | 17, 411 | pending | TBD |
| aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py | 2 | allowlist | unresolved | 33, 34 | pending | TBD |
| aiter/ops/triton/gluon/pa_mqa_logits.py | 2 | compare | unresolved | 102, 580 | pending | TBD |
| aiter/ops/triton/moe/moe_op_gemm_a4w4.py | 2 | compare | triton-arch | 326, 330 | pending | TBD |
| aiter/ops/triton/normalization/fused_add_rmsnorm_pad.py | 2 | compare | triton-arch | 57, 66 | pending | TBD |
| aiter/ops/triton/quant/quant.py | 2 | compare | triton-arch | 319, 567 | pending | TBD |
| aiter/jit/utils/chip_info.py | 1 | compare | get_gfx | 631 | pending | TBD |
| aiter/ops/flydsl/gemm_kernels.py | 1 | compare | get_gfx | 148 | pending | TBD |
| aiter/ops/flydsl/kernels/fused_compress_attn.py | 1 | compare | get_gfx | 2317 | pending | TBD |
| aiter/ops/flydsl/kernels/mega_moe_gfx1250/mega_moe.py | 1 | compare | get_gfx | 585 | pending | TBD |
| aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py | 1 | allowlist | unresolved | 157 | pending | TBD |
| aiter/ops/flydsl/kernels/pa_decode_kernel.py | 1 | compare | get_gfx_runtime | 151 | pending | TBD |
| aiter/ops/flydsl/kernels/qk_norm_rope_quant.py | 1 | compare | get_gfx | 1454 | pending | TBD |
| aiter/ops/flydsl/mla_kernels.py | 1 | compare | local | 67 | pending | TBD |
| aiter/ops/flydsl/pa_decode_tuning.py | 1 | compare | get_gfx_runtime | 563 | pending | TBD |
| aiter/ops/flydsl/topk/topk_per_row.py | 1 | allowlist | unresolved | 157 | pending | TBD |
| aiter/ops/mla_sparse_prefill.py | 1 | compare | get_gfx_runtime | 100 | pending | TBD |
| aiter/ops/norm.py | 1 | compare | get_gfx | 151 | pending | TBD |
| aiter/ops/topk.py | 1 | compare | get_gfx_runtime | 81 | pending | TBD |
| aiter/ops/topk_select.py | 1 | allowlist | unresolved | 63 | pending | TBD |
| aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_3d.py | 1 | compare | unresolved | 20 | pending | TBD |
| aiter/ops/triton/attention/fp8_mqa_logits.py | 1 | compare | triton-arch | 24 | pending | TBD |
| aiter/ops/triton/attention/kda.py | 1 | compare | triton-arch | 19 | pending | TBD |
| aiter/ops/triton/attention/pa_decode_sparse.py | 1 | compare | unresolved | 241 | pending | TBD |
| aiter/ops/triton/attention/pa_prefill_sparse.py | 1 | compare | unresolved | 126 | pending | TBD |
| aiter/ops/triton/fusions/fused_clamp_act_mul.py | 1 | allowlist | unresolved | 23 | pending | TBD |
| aiter/ops/triton/fusions/fused_qk_norm_rope_group_quant.py | 1 | compare | triton-arch | 65 | pending | TBD |
| aiter/ops/triton/fusions/mhc.py | 1 | compare | triton-arch | 557 | pending | TBD |
| aiter/ops/triton/gemm/basic/gemm_a16w16.py | 1 | allowlist | unresolved | 32 | pending | TBD |
| aiter/ops/triton/gemm/basic/gemm_afp8wfp8.py | 1 | allowlist | unresolved | 65 | pending | TBD |
| aiter/ops/triton/gemm/batched/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py | 1 | allowlist | unresolved | 16 | pending | TBD |
| aiter/ops/triton/gemm/batched/batched_gemm_bf16.py | 1 | allowlist | unresolved | 22 | pending | TBD |
| aiter/ops/triton/normalization/fused_rmsnorm_add.py | 1 | compare | triton-arch | 60 | pending | TBD |
| aiter/ops/triton/utils/_triton/tuning/kernels.py | 1 | allowlist | unresolved | 272 | pending | TBD |
| aiter/tuned_gemm.py | 1 | compare | get_gfx | 282 | pending | TBD |

<!-- END generated: gfx1250 dispatch sites -->

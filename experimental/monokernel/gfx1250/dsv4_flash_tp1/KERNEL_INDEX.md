# Kernel symbol index

Includes JIT kernels and inlined JIT helpers across selected, reference, and historical variants. Repeated symbols in separate experiment directories are retained deliberately. See README.md for selected entry points.

| Source | JIT symbols |
|---|---|
| [runtime/app/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/mhc.py](runtime/app/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/mhc.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/aiter-pre-audit/check_pre.py](work/aiter-pre-audit/check_pre.py) | `new`, `old` |
| [work/aiter-pre-audit/check_pre_real.py](work/aiter-pre-audit/check_pre_real.py) | `new`, `old` |
| [work/aiter-pre-audit/model_layer_batch.before.py](work/aiter-pre-audit/model_layer_batch.before.py) | `model_layer` |
| [work/attention-aiter/aiter_adapt.py](work/attention-aiter/aiter_adapt.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/attention-aiter/attention.py](work/attention-aiter/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/attention-aiter/combined.py](work/attention-aiter/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/attention-aiter/device_scope_helpers.py](work/attention-aiter/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/attention-aiter/initial-attention.py](work/attention-aiter/initial-attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/attention-aiter/qa.py](work/attention-aiter/qa.py) | `project_fused`, `finish_fused` |
| [work/attention-aiter/qb.py](work/attention-aiter/qb.py) | `project_qb`, `qb_kernel` |
| [work/attention-aiter/stage.py](work/attention-aiter/stage.py) | `stage_qb` |
| [work/attention-aiter/tail.py](work/attention-aiter/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/attention-aiter/upstream/pa_decode_sparse.py](work/attention-aiter/upstream/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/attention-aiter/vector_original.py](work/attention-aiter/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/attention-direct-kv/aiter_adapt.py](work/attention-direct-kv/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/attention-direct-kv/attention.py](work/attention-direct-kv/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/attention-direct-kv/combined.py](work/attention-direct-kv/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/attention-direct-kv/device_scope_helpers.py](work/attention-direct-kv/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/attention-direct-kv/initial-attention.py](work/attention-direct-kv/initial-attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/attention-direct-kv/lds_overlay.py](work/attention-direct-kv/lds_overlay.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/attention-direct-kv/qa.py](work/attention-direct-kv/qa.py) | `project_fused`, `finish_fused` |
| [work/attention-direct-kv/qb.py](work/attention-direct-kv/qb.py) | `project_qb`, `qb_kernel` |
| [work/attention-direct-kv/register_overlay.py](work/attention-direct-kv/register_overlay.py) | `overlay_register`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/attention-direct-kv/stage.py](work/attention-direct-kv/stage.py) | `stage_qb` |
| [work/attention-direct-kv/tail.py](work/attention-direct-kv/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/attention-direct-kv/upstream/pa_decode_sparse.py](work/attention-direct-kv/upstream/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/attention-direct-kv/vector_original.py](work/attention-direct-kv/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/batch-full/attention_batch.py](work/batch-full/attention_batch.py) | `compute_attention`, `attention_body` |
| [work/batch-full/attention_batch_inline_prefetch.py](work/batch-full/attention_batch_inline_prefetch.py) | `compute_attention`, `attention_body` |
| [work/batch-full/check_attention.py](work/batch-full/check_attention.py) | `batch` |
| [work/batch-full/check_boundary.py](work/batch-full/check_boundary.py) | `output` |
| [work/batch-full/check_compressor.py](work/batch-full/check_compressor.py) | `batch`, `single` |
| [work/batch-full/check_ffn.py](work/batch-full/check_ffn.py) | `fb`, `fs` |
| [work/batch-full/check_ffn_groups.py](work/batch-full/check_ffn_groups.py) | `batch`, `single` |
| [work/batch-full/check_mhc.py](work/batch-full/check_mhc.py) | `c1`, `post_c1` |
| [work/batch-full/check_moe.py](work/batch-full/check_moe.py) | `run` |
| [work/batch-full/check_output.py](work/batch-full/check_output.py) | `batched` |
| [work/batch-full/compressor_batch.py](work/batch-full/compressor_batch.py) | `row`, `compressor_body` |
| [work/batch-full/ffn_batch.py](work/batch-full/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/batch-full/identity_batch.py](work/batch-full/identity_batch.py) | `identity_csa` |
| [work/batch-full/layout_batch.py](work/batch-full/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/batch-full/mhc_batch.py](work/batch-full/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/batch-full/model_layer_batch.py](work/batch-full/model_layer_batch.py) | `model_layer` |
| [work/batch-full/output_batch.py](work/batch-full/output_batch.py) | `split_project`, `inverse_quant`, `reduce_quant`, `output_body` |
| [work/batch-full/output_batch_before_barriers.py](work/batch-full/output_batch_before_barriers.py) | `split_project`, `inverse_quant`, `reduce_quant`, `output_body` |
| [work/batch-full/post_batch.py](work/batch-full/post_batch.py) | `post_body`, `post_batch` |
| [work/batch-full/project_batch.py](work/batch-full/project_batch.py) | `project_body` |
| [work/batch-full/qkv_full_batch.py](work/batch-full/qkv_full_batch.py) | `qkv_batch` |
| [work/batch-full/routing_batch.py](work/batch-full/routing_batch.py) | `add`, `route_body` |
| [work/c16-latency-tuning/aiter_paged.py](work/c16-latency-tuning/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-latency-tuning/attention_paged.py](work/c16-latency-tuning/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-latency-tuning/check_herd.py](work/c16-latency-tuning/check_herd.py) | `run` |
| [work/c16-latency-tuning/edge_sync.py](work/c16-latency-tuning/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-latency-tuning/ffn_batch.py](work/c16-latency-tuning/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-latency-tuning/frozen_before/aiter_paged.py](work/c16-latency-tuning/frozen_before/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-latency-tuning/frozen_before/attention_paged.py](work/c16-latency-tuning/frozen_before/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-latency-tuning/frozen_before/check_herd.py](work/c16-latency-tuning/frozen_before/check_herd.py) | `run` |
| [work/c16-latency-tuning/frozen_before/edge_sync.py](work/c16-latency-tuning/frozen_before/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-latency-tuning/frozen_before/ffn_batch.py](work/c16-latency-tuning/frozen_before/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-latency-tuning/frozen_before/identity_batch.py](work/c16-latency-tuning/frozen_before/identity_batch.py) | `identity_csa` |
| [work/c16-latency-tuning/frozen_before/isolated.py](work/c16-latency-tuning/frozen_before/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-latency-tuning/frozen_before/isolated_compress.py](work/c16-latency-tuning/frozen_before/isolated_compress.py) | `batch` |
| [work/c16-latency-tuning/frozen_before/layout_batch.py](work/c16-latency-tuning/frozen_before/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-latency-tuning/frozen_before/mhc_batch.py](work/c16-latency-tuning/frozen_before/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-latency-tuning/frozen_before/mhc_integrated.py](work/c16-latency-tuning/frozen_before/mhc_integrated.py) | `post_pre` |
| [work/c16-latency-tuning/frozen_before/mhc_native_inline.py](work/c16-latency-tuning/frozen_before/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-latency-tuning/frozen_before/mhc_pre_aiter.py](work/c16-latency-tuning/frozen_before/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-latency-tuning/frozen_before/model_layer_batch.py](work/c16-latency-tuning/frozen_before/model_layer_batch.py) | `model_layer` |
| [work/c16-latency-tuning/frozen_before/native_identity.py](work/c16-latency-tuning/frozen_before/native_identity.py) | `run` |
| [work/c16-latency-tuning/frozen_before/native_routing.py](work/c16-latency-tuning/frozen_before/native_routing.py) | `run` |
| [work/c16-latency-tuning/frozen_before/optimized_compressor.py](work/c16-latency-tuning/frozen_before/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial` |
| [work/c16-latency-tuning/frozen_before/optimized_moe.py](work/c16-latency-tuning/frozen_before/optimized_moe.py) | `phase`, `fused` |
| [work/c16-latency-tuning/frozen_before/optimized_moe_stage.py](work/c16-latency-tuning/frozen_before/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-latency-tuning/frozen_before/optimized_routing.before_subgroup.py](work/c16-latency-tuning/frozen_before/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-latency-tuning/frozen_before/optimized_routing.py](work/c16-latency-tuning/frozen_before/optimized_routing.py) | `add`, `route_body` |
| [work/c16-latency-tuning/frozen_before/output_batch.py](work/c16-latency-tuning/frozen_before/output_batch.py) | `split_project`, `inverse_quant`, `reduce_quant`, `stage_output_weights`, `output_body` |
| [work/c16-latency-tuning/frozen_before/post_batch.py](work/c16-latency-tuning/frozen_before/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-latency-tuning/frozen_before/project_batch.py](work/c16-latency-tuning/frozen_before/project_batch.py) | `project_body` |
| [work/c16-latency-tuning/frozen_before/qa_candidate.py](work/c16-latency-tuning/frozen_before/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-latency-tuning/frozen_before/qkv_candidate.py](work/c16-latency-tuning/frozen_before/qkv_candidate.py) | `qkv_batch` |
| [work/c16-latency-tuning/frozen_before/regions.py](work/c16-latency-tuning/frozen_before/regions.py) | `region` |
| [work/c16-latency-tuning/identity_batch.py](work/c16-latency-tuning/identity_batch.py) | `identity_csa` |
| [work/c16-latency-tuning/isolated.py](work/c16-latency-tuning/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-latency-tuning/isolated_compress.py](work/c16-latency-tuning/isolated_compress.py) | `batch` |
| [work/c16-latency-tuning/layout_batch.py](work/c16-latency-tuning/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-latency-tuning/mhc_batch.py](work/c16-latency-tuning/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-latency-tuning/mhc_integrated.py](work/c16-latency-tuning/mhc_integrated.py) | `post_pre` |
| [work/c16-latency-tuning/mhc_native_inline.py](work/c16-latency-tuning/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-latency-tuning/mhc_pre_aiter.py](work/c16-latency-tuning/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-latency-tuning/model_layer_batch.py](work/c16-latency-tuning/model_layer_batch.py) | `model_layer` |
| [work/c16-latency-tuning/native_identity.py](work/c16-latency-tuning/native_identity.py) | `run` |
| [work/c16-latency-tuning/native_routing.py](work/c16-latency-tuning/native_routing.py) | `run` |
| [work/c16-latency-tuning/optimized_compressor.py](work/c16-latency-tuning/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial`, `pool_sliced`, `hca_ready` |
| [work/c16-latency-tuning/optimized_moe.py](work/c16-latency-tuning/optimized_moe.py) | `phase`, `fused`, `prefetch_next` |
| [work/c16-latency-tuning/optimized_moe_stage.py](work/c16-latency-tuning/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-latency-tuning/optimized_routing.before_subgroup.py](work/c16-latency-tuning/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-latency-tuning/optimized_routing.py](work/c16-latency-tuning/optimized_routing.py) | `add`, `route_body` |
| [work/c16-latency-tuning/output_batch.py](work/c16-latency-tuning/output_batch.py) | `split_project_old`, `inverse_quant`, `reduce_quant`, `stage_output_weights`, `stage_tuned`, `split_project_tuned`, `output_tuned`, `output_body` |
| [work/c16-latency-tuning/post_batch.py](work/c16-latency-tuning/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-latency-tuning/project_batch.py](work/c16-latency-tuning/project_batch.py) | `project_body` |
| [work/c16-latency-tuning/qa_candidate.py](work/c16-latency-tuning/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-latency-tuning/qkv_candidate.py](work/c16-latency-tuning/qkv_candidate.py) | `qkv_batch` |
| [work/c16-latency-tuning/regions.py](work/c16-latency-tuning/regions.py) | `region` |
| [work/c16-monokernel/aiter_paged.py](work/c16-monokernel/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-monokernel/attention_paged.py](work/c16-monokernel/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-monokernel/check_herd.py](work/c16-monokernel/check_herd.py) | `run` |
| [work/c16-monokernel/edge_sync.py](work/c16-monokernel/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-monokernel/ffn_batch.py](work/c16-monokernel/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-monokernel/identity_batch.py](work/c16-monokernel/identity_batch.py) | `identity_csa` |
| [work/c16-monokernel/isolated.py](work/c16-monokernel/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-monokernel/isolated_compress.py](work/c16-monokernel/isolated_compress.py) | `batch` |
| [work/c16-monokernel/layout_batch.py](work/c16-monokernel/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-monokernel/mhc_batch.py](work/c16-monokernel/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-monokernel/mhc_integrated.py](work/c16-monokernel/mhc_integrated.py) | `post_pre` |
| [work/c16-monokernel/mhc_native_inline.py](work/c16-monokernel/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-monokernel/mhc_pre_aiter.py](work/c16-monokernel/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-monokernel/model_layer_batch.py](work/c16-monokernel/model_layer_batch.py) | `model_layer` |
| [work/c16-monokernel/native_identity.py](work/c16-monokernel/native_identity.py) | `run` |
| [work/c16-monokernel/native_routing.py](work/c16-monokernel/native_routing.py) | `run` |
| [work/c16-monokernel/optimized_compressor.py](work/c16-monokernel/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial` |
| [work/c16-monokernel/optimized_moe.py](work/c16-monokernel/optimized_moe.py) | `phase`, `fused` |
| [work/c16-monokernel/optimized_moe_stage.py](work/c16-monokernel/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-monokernel/optimized_routing.before_subgroup.py](work/c16-monokernel/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-monokernel/optimized_routing.py](work/c16-monokernel/optimized_routing.py) | `add`, `route_body` |
| [work/c16-monokernel/output_batch.py](work/c16-monokernel/output_batch.py) | `split_project`, `inverse_quant`, `reduce_quant`, `stage_output_weights`, `output_body` |
| [work/c16-monokernel/post_batch.py](work/c16-monokernel/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-monokernel/project_batch.py](work/c16-monokernel/project_batch.py) | `project_body` |
| [work/c16-monokernel/qa_candidate.py](work/c16-monokernel/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-monokernel/qkv_candidate.py](work/c16-monokernel/qkv_candidate.py) | `qkv_batch` |
| [work/c16-monokernel/regions.py](work/c16-monokernel/regions.py) | `region` |
| [work/c16-native-parity/aiter_paged.py](work/c16-native-parity/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-native-parity/attention_paged.py](work/c16-native-parity/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-native-parity/check_herd.py](work/c16-native-parity/check_herd.py) | `run` |
| [work/c16-native-parity/device_scope_helpers.py](work/c16-native-parity/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/c16-native-parity/edge_sync.py](work/c16-native-parity/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-native-parity/faithful_composed.py](work/c16-native-parity/faithful_composed.py) | `idle`, `qa_prefix`, `qkv`, `inv_worker`, `quant_worker`, `output`, `compact_output` |
| [work/c16-native-parity/faithful_gemm.py](work/c16-native-parity/faithful_gemm.py) | `wslot`, `early_weight`, `operands`, `issue`, `compute`, `gemm`, `entry` |
| [work/c16-native-parity/faithful_ops.py](work/c16-native-parity/faithful_ops.py) | `quant_column`, `qa_norm`, `qk_rope`, `inverse_quant`, `inverse_quant_group`, `quant_entry`, `norm_entry`, `rope_entry`, `inverse_entry` |
| [work/c16-native-parity/ffn_batch.py](work/c16-native-parity/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-native-parity/identity_batch.py](work/c16-native-parity/identity_batch.py) | `identity_csa` |
| [work/c16-native-parity/isolated.py](work/c16-native-parity/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-native-parity/isolated_compress.py](work/c16-native-parity/isolated_compress.py) | `batch` |
| [work/c16-native-parity/layout_batch.py](work/c16-native-parity/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-native-parity/masked_prefetch.py](work/c16-native-parity/masked_prefetch.py) | `masked_line_prefetch` |
| [work/c16-native-parity/mhc_batch.py](work/c16-native-parity/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-native-parity/mhc_integrated.py](work/c16-native-parity/mhc_integrated.py) | `post_pre` |
| [work/c16-native-parity/mhc_native_inline.py](work/c16-native-parity/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-native-parity/mhc_pre_aiter.py](work/c16-native-parity/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-native-parity/model_layer_batch.py](work/c16-native-parity/model_layer_batch.py) | `model_layer` |
| [work/c16-native-parity/native_identity.py](work/c16-native-parity/native_identity.py) | `run` |
| [work/c16-native-parity/native_norm.py](work/c16-native-parity/native_norm.py) | `qa_norm` |
| [work/c16-native-parity/native_quant.py](work/c16-native-parity/native_quant.py) | `quant_groups`, `quant_column` |
| [work/c16-native-parity/native_routing.py](work/c16-native-parity/native_routing.py) | `run` |
| [work/c16-native-parity/native_tiles.py](work/c16-native-parity/native_tiles.py) | `weight_first`, `wslot`, `tile_operands`, `gemm` |
| [work/c16-native-parity/optimized_compressor.py](work/c16-native-parity/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial`, `pool_sliced`, `hca_ready` |
| [work/c16-native-parity/optimized_moe.py](work/c16-native-parity/optimized_moe.py) | `phase`, `fused`, `prefetch_next` |
| [work/c16-native-parity/optimized_moe_stage.py](work/c16-native-parity/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-native-parity/optimized_routing.before_subgroup.py](work/c16-native-parity/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-native-parity/optimized_routing.py](work/c16-native-parity/optimized_routing.py) | `add`, `route_body` |
| [work/c16-native-parity/output_batch.py](work/c16-native-parity/output_batch.py) | `output_body` |
| [work/c16-native-parity/parts.py](work/c16-native-parity/parts.py) | `entry` |
| [work/c16-native-parity/post_batch.py](work/c16-native-parity/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-native-parity/previous/aiter_paged.py](work/c16-native-parity/previous/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-native-parity/previous/attention_paged.py](work/c16-native-parity/previous/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-native-parity/previous/check_herd.py](work/c16-native-parity/previous/check_herd.py) | `run` |
| [work/c16-native-parity/previous/edge_sync.py](work/c16-native-parity/previous/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-native-parity/previous/ffn_batch.py](work/c16-native-parity/previous/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-native-parity/previous/identity_batch.py](work/c16-native-parity/previous/identity_batch.py) | `identity_csa` |
| [work/c16-native-parity/previous/isolated.py](work/c16-native-parity/previous/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-native-parity/previous/isolated_compress.py](work/c16-native-parity/previous/isolated_compress.py) | `batch` |
| [work/c16-native-parity/previous/layout_batch.py](work/c16-native-parity/previous/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-native-parity/previous/mhc_batch.py](work/c16-native-parity/previous/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-native-parity/previous/mhc_integrated.py](work/c16-native-parity/previous/mhc_integrated.py) | `post_pre` |
| [work/c16-native-parity/previous/mhc_native_inline.py](work/c16-native-parity/previous/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-native-parity/previous/mhc_pre_aiter.py](work/c16-native-parity/previous/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-native-parity/previous/model_layer_batch.py](work/c16-native-parity/previous/model_layer_batch.py) | `model_layer` |
| [work/c16-native-parity/previous/native_identity.py](work/c16-native-parity/previous/native_identity.py) | `run` |
| [work/c16-native-parity/previous/native_routing.py](work/c16-native-parity/previous/native_routing.py) | `run` |
| [work/c16-native-parity/previous/optimized_compressor.py](work/c16-native-parity/previous/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial`, `pool_sliced`, `hca_ready` |
| [work/c16-native-parity/previous/optimized_moe.py](work/c16-native-parity/previous/optimized_moe.py) | `phase`, `fused`, `prefetch_next` |
| [work/c16-native-parity/previous/optimized_moe_stage.py](work/c16-native-parity/previous/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-native-parity/previous/optimized_routing.before_subgroup.py](work/c16-native-parity/previous/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-native-parity/previous/optimized_routing.py](work/c16-native-parity/previous/optimized_routing.py) | `add`, `route_body` |
| [work/c16-native-parity/previous/output_batch.py](work/c16-native-parity/previous/output_batch.py) | `split_project_old`, `inverse_quant`, `reduce_quant`, `stage_output_weights`, `stage_tuned`, `split_project_tuned`, `output_tuned`, `output_body` |
| [work/c16-native-parity/previous/post_batch.py](work/c16-native-parity/previous/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-native-parity/previous/project_batch.py](work/c16-native-parity/previous/project_batch.py) | `project_body` |
| [work/c16-native-parity/previous/qa_candidate.py](work/c16-native-parity/previous/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-native-parity/previous/qkv_candidate.py](work/c16-native-parity/previous/qkv_candidate.py) | `qkv_batch` |
| [work/c16-native-parity/previous/regions.py](work/c16-native-parity/previous/regions.py) | `region` |
| [work/c16-native-parity/project_batch.py](work/c16-native-parity/project_batch.py) | `project_body` |
| [work/c16-native-parity/qa_candidate.py](work/c16-native-parity/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-native-parity/qa_deferred.py](work/c16-native-parity/qa_deferred.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-native-parity/qkv_candidate.py](work/c16-native-parity/qkv_candidate.py) | `input_quant`, `qkv_batch` |
| [work/c16-native-parity/regions.py](work/c16-native-parity/regions.py) | `region` |
| [work/c16-native-parity/test_gemm.py](work/c16-native-parity/test_gemm.py) | `run` |
| [work/c16-native-projections/aiter_paged.py](work/c16-native-projections/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-native-projections/attention_paged.py](work/c16-native-projections/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-native-projections/check_herd.py](work/c16-native-projections/check_herd.py) | `run` |
| [work/c16-native-projections/edge_sync.py](work/c16-native-projections/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-native-projections/ffn_batch.py](work/c16-native-projections/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-native-projections/identity_batch.py](work/c16-native-projections/identity_batch.py) | `identity_csa` |
| [work/c16-native-projections/isolated.py](work/c16-native-projections/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-native-projections/isolated_compress.py](work/c16-native-projections/isolated_compress.py) | `batch` |
| [work/c16-native-projections/layout_batch.py](work/c16-native-projections/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-native-projections/mhc_batch.py](work/c16-native-projections/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-native-projections/mhc_integrated.py](work/c16-native-projections/mhc_integrated.py) | `post_pre` |
| [work/c16-native-projections/mhc_native_inline.py](work/c16-native-projections/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-native-projections/mhc_pre_aiter.py](work/c16-native-projections/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-native-projections/model_layer_batch.py](work/c16-native-projections/model_layer_batch.py) | `model_layer` |
| [work/c16-native-projections/native_identity.py](work/c16-native-projections/native_identity.py) | `run` |
| [work/c16-native-projections/native_norm.py](work/c16-native-projections/native_norm.py) | `qa_norm` |
| [work/c16-native-projections/native_quant.py](work/c16-native-projections/native_quant.py) | `quant_groups`, `quant_column` |
| [work/c16-native-projections/native_routing.py](work/c16-native-projections/native_routing.py) | `run` |
| [work/c16-native-projections/native_tiles.py](work/c16-native-projections/native_tiles.py) | `weight_first`, `wslot`, `tile_operands`, `gemm` |
| [work/c16-native-projections/optimized_compressor.py](work/c16-native-projections/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial`, `pool_sliced`, `hca_ready` |
| [work/c16-native-projections/optimized_moe.py](work/c16-native-projections/optimized_moe.py) | `phase`, `fused`, `prefetch_next` |
| [work/c16-native-projections/optimized_moe_stage.py](work/c16-native-projections/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-native-projections/optimized_routing.before_subgroup.py](work/c16-native-projections/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-native-projections/optimized_routing.py](work/c16-native-projections/optimized_routing.py) | `add`, `route_body` |
| [work/c16-native-projections/output_batch.py](work/c16-native-projections/output_batch.py) | `output_body` |
| [work/c16-native-projections/parts.py](work/c16-native-projections/parts.py) | `entry` |
| [work/c16-native-projections/post_batch.py](work/c16-native-projections/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-native-projections/previous/aiter_paged.py](work/c16-native-projections/previous/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c16-native-projections/previous/attention_paged.py](work/c16-native-projections/previous/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c16-native-projections/previous/check_herd.py](work/c16-native-projections/previous/check_herd.py) | `run` |
| [work/c16-native-projections/previous/edge_sync.py](work/c16-native-projections/previous/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/c16-native-projections/previous/ffn_batch.py](work/c16-native-projections/previous/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/c16-native-projections/previous/identity_batch.py](work/c16-native-projections/previous/identity_batch.py) | `identity_csa` |
| [work/c16-native-projections/previous/isolated.py](work/c16-native-projections/previous/isolated.py) | `mhc_pre`, `mhc_mid`, `output`, `shared`, `moe_region`, `tail`, `qkv_cache`, `whole_ffn` |
| [work/c16-native-projections/previous/isolated_compress.py](work/c16-native-projections/previous/isolated_compress.py) | `batch` |
| [work/c16-native-projections/previous/layout_batch.py](work/c16-native-projections/previous/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/c16-native-projections/previous/mhc_batch.py](work/c16-native-projections/previous/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c16-native-projections/previous/mhc_integrated.py](work/c16-native-projections/previous/mhc_integrated.py) | `post_pre` |
| [work/c16-native-projections/previous/mhc_native_inline.py](work/c16-native-projections/previous/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c16-native-projections/previous/mhc_pre_aiter.py](work/c16-native-projections/previous/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/c16-native-projections/previous/model_layer_batch.py](work/c16-native-projections/previous/model_layer_batch.py) | `model_layer` |
| [work/c16-native-projections/previous/native_identity.py](work/c16-native-projections/previous/native_identity.py) | `run` |
| [work/c16-native-projections/previous/native_routing.py](work/c16-native-projections/previous/native_routing.py) | `run` |
| [work/c16-native-projections/previous/optimized_compressor.py](work/c16-native-projections/previous/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial`, `pool_sliced`, `hca_ready` |
| [work/c16-native-projections/previous/optimized_moe.py](work/c16-native-projections/previous/optimized_moe.py) | `phase`, `fused`, `prefetch_next` |
| [work/c16-native-projections/previous/optimized_moe_stage.py](work/c16-native-projections/previous/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c16-native-projections/previous/optimized_routing.before_subgroup.py](work/c16-native-projections/previous/optimized_routing.before_subgroup.py) | `add`, `route_body` |
| [work/c16-native-projections/previous/optimized_routing.py](work/c16-native-projections/previous/optimized_routing.py) | `add`, `route_body` |
| [work/c16-native-projections/previous/output_batch.py](work/c16-native-projections/previous/output_batch.py) | `split_project_old`, `inverse_quant`, `reduce_quant`, `stage_output_weights`, `stage_tuned`, `split_project_tuned`, `output_tuned`, `output_body` |
| [work/c16-native-projections/previous/post_batch.py](work/c16-native-projections/previous/post_batch.py) | `post_body`, `post_batch` |
| [work/c16-native-projections/previous/project_batch.py](work/c16-native-projections/previous/project_batch.py) | `project_body` |
| [work/c16-native-projections/previous/qa_candidate.py](work/c16-native-projections/previous/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-native-projections/previous/qkv_candidate.py](work/c16-native-projections/previous/qkv_candidate.py) | `qkv_batch` |
| [work/c16-native-projections/previous/regions.py](work/c16-native-projections/previous/regions.py) | `region` |
| [work/c16-native-projections/project_batch.py](work/c16-native-projections/project_batch.py) | `project_body` |
| [work/c16-native-projections/qa_candidate.py](work/c16-native-projections/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-native-projections/qa_deferred.py](work/c16-native-projections/qa_deferred.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c16-native-projections/qkv_candidate.py](work/c16-native-projections/qkv_candidate.py) | `input_quant`, `qkv_batch` |
| [work/c16-native-projections/regions.py](work/c16-native-projections/regions.py) | `region` |
| [work/c16-native-projections/test_gemm.py](work/c16-native-projections/test_gemm.py) | `run` |
| [work/c8-ablation/attention_defs.py](work/c8-ablation/attention_defs.py) | `batch` |
| [work/c8-ablation/bench_identity.py](work/c8-ablation/bench_identity.py) | `run` |
| [work/c8-ablation/bench_routing.py](work/c8-ablation/bench_routing.py) | `run` |
| [work/c8-ablation/bench_tail.py](work/c8-ablation/bench_tail.py) | `layouts`, `shuf` |
| [work/c8-ablation/cache_stage.py](work/c8-ablation/cache_stage.py) | `stage` |
| [work/c8-ablation/compress_defs.py](work/c8-ablation/compress_defs.py) | `batch`, `single` |
| [work/c8-ablation/mhc_defs.py](work/c8-ablation/mhc_defs.py) | `c1`, `post_c1` |
| [work/c8-ablation/moe_kernel.py](work/c8-ablation/moe_kernel.py) | `moe_region` |
| [work/c8-ablation/output_defs.py](work/c8-ablation/output_defs.py) | `batched` |
| [work/c8-ablation/shared_defs.py](work/c8-ablation/shared_defs.py) | `batch`, `single` |
| [work/c8-ablation/shared_kernel.py](work/c8-ablation/shared_kernel.py) | `region` |
| [work/c8-ablation/tail_kernel.py](work/c8-ablation/tail_kernel.py) | `combine`, `post_pre` |
| [work/c8-qkv-deps/qa_candidate.before-atomic-fix.py](work/c8-qkv-deps/qa_candidate.before-atomic-fix.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c8-qkv-deps/qa_candidate.py](work/c8-qkv-deps/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/c8-qkv-deps/qkv_candidate.py](work/c8-qkv-deps/qkv_candidate.py) | `qkv_batch` |
| [work/c8-tuning/aiter_paged.py](work/c8-tuning/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/c8-tuning/attention_batch.py](work/c8-tuning/attention_batch.py) | `compute_attention`, `attention_body` |
| [work/c8-tuning/attention_defs.py](work/c8-tuning/attention_defs.py) | `batch` |
| [work/c8-tuning/attention_paged.py](work/c8-tuning/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/c8-tuning/bench_routing.py](work/c8-tuning/bench_routing.py) | `run` |
| [work/c8-tuning/check_routing.py](work/c8-tuning/check_routing.py) | `run` |
| [work/c8-tuning/compress_defs.py](work/c8-tuning/compress_defs.py) | `batch`, `single` |
| [work/c8-tuning/compressor_batch.py](work/c8-tuning/compressor_batch.py) | `row`, `compressor_body`, `pool_partial` |
| [work/c8-tuning/mhc_batch.py](work/c8-tuning/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/c8-tuning/mhc_defs.py](work/c8-tuning/mhc_defs.py) | `c1`, `post_c1` |
| [work/c8-tuning/mhc_native_inline.py](work/c8-tuning/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/c8-tuning/mhc_native_region.py](work/c8-tuning/mhc_native_region.py) | `region` |
| [work/c8-tuning/moe_kernel.py](work/c8-tuning/moe_kernel.py) | `moe_region` |
| [work/c8-tuning/moe_multitoken_prototype.py](work/c8-tuning/moe_multitoken_prototype.py) | `phase`, `fused` |
| [work/c8-tuning/moe_n256.py](work/c8-tuning/moe_n256.py) | `phase`, `fused` |
| [work/c8-tuning/moe_task_stage.py](work/c8-tuning/moe_task_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/c8-tuning/routing_batch.py](work/c8-tuning/routing_batch.py) | `add`, `route_body` |
| [work/c8-tuning/routing_reference.py](work/c8-tuning/routing_reference.py) | `add`, `route_body` |
| [work/c8-tuning/tail_kernel.py](work/c8-tuning/tail_kernel.py) | `combine`, `post_pre` |
| [work/candidate/mhc_kernel.py](work/candidate/mhc_kernel.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/concurrency-rollout/masked_prefetch.py](work/concurrency-rollout/masked_prefetch.py) | `masked_line_prefetch` |
| [work/concurrency-rollout/model_layer_serial.py](work/concurrency-rollout/model_layer_serial.py) | `batched_layer` |
| [work/concurrency-rollout/qa_batch.py](work/concurrency-rollout/qa_batch.py) | `finish`, `qa_batch`, `finish_batch`, `prefetch_only` |
| [work/concurrency-rollout/qkv_batch.py](work/concurrency-rollout/qkv_batch.py) | `qkv_batch` |
| [work/ffn-c1/combine-post/kernel.py](work/ffn-c1/combine-post/kernel.py) | `combine_post` |
| [work/ffn-c1/moe/moe_multitoken_prototype.py](work/ffn-c1/moe/moe_multitoken_prototype.py) | `phase`, `fused` |
| [work/ffn-c1/moe/moe_task_stage.py](work/ffn-c1/moe/moe_task_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/ffn-c64/combine-post/kernel.py](work/ffn-c64/combine-post/kernel.py) | `combine_post` |
| [work/ffn-c64/moe/moe_multitoken_prototype.py](work/ffn-c64/moe/moe_multitoken_prototype.py) | `phase`, `fused` |
| [work/ffn-c64/moe/moe_task_stage.py](work/ffn-c64/moe/moe_task_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/flash/candidate/dsv4_flash_moe_kernel.py](work/flash/candidate/dsv4_flash_moe_kernel.py) | `fused` |
| [work/flash/candidate/dsv4_flash_moe_stage.py](work/flash/candidate/dsv4_flash_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/flash/gemm_stage_prototype.py](work/flash/gemm_stage_prototype.py) | `depreshuffle_b`, `_load_scale_tile`, `_gather_scale_tile`, `_gather_scale_vec`, `_read_tile`, `_gemm_mxfp8_preshuffle_bandwidth_bound_kernel`, `_gemm_mxfp8_preshuffle_compute_bound_kernel` |
| [work/flash/mhc_stage_device_scope.py](work/flash/mhc_stage_device_scope.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel`, `_mono_reduce_apply_main`, `_mhc_one_launch_kernel` |
| [work/flash/mhc_stage_prototype.py](work/flash/mhc_stage_prototype.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel`, `_mono_reduce_apply_main`, `_mhc_one_launch_kernel` |
| [work/flash/mhc_stage_relaxed_poll.py](work/flash/mhc_stage_relaxed_poll.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel`, `_mono_reduce_apply_main`, `_mhc_one_launch_kernel` |
| [work/flash/moe_aiter_original.py](work/flash/moe_aiter_original.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/flash/moe_aiter_scoped.py](work/flash/moe_aiter_scoped.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/flash/moe_multitoken_prototype.py](work/flash/moe_multitoken_prototype.py) | `phase`, `fused` |
| [work/flash/moe_stage_prototype.py](work/flash/moe_stage_prototype.py) | `fused` |
| [work/flash/moe_stage_reduce.py](work/flash/moe_stage_reduce.py) | `fused` |
| [work/flash/moe_stage_scoped.py](work/flash/moe_stage_scoped.py) | `fused` |
| [work/flash/moe_task_stage.py](work/flash/moe_task_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/full-attention-native/full_kernel.py](work/full-attention-native/full_kernel.py) | `full_attention` |
| [work/full-attention-native/layout.py](work/full-attention-native/layout.py) | `unshuffle_body`, `unshuffle_kernel` |
| [work/full-attention-native/output-kernel/output_region.py](work/full-attention-native/output-kernel/output_region.py) | `inverse_quant`, `split_project`, `reduce_quant`, `output_body`, `output_kernel`, `output_stage` |
| [work/full-attention-native/post-kernel/post.py](work/full-attention-native/post-kernel/post.py) | `post_body`, `post` |
| [work/full-attention-native/prefix/aiter_adapt.py](work/full-attention-native/prefix/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/full-attention-native/prefix/attention.py](work/full-attention-native/prefix/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/full-attention-native/prefix/combined.py](work/full-attention-native/prefix/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/full-attention-native/prefix/device_scope_helpers.py](work/full-attention-native/prefix/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-attention-native/prefix/full_prefix.py](work/full-attention-native/prefix/full_prefix.py) | `full_prefix` |
| [work/full-attention-native/prefix/mhc.py](work/full-attention-native/prefix/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/full-attention-native/prefix/optimized_prefix.py](work/full-attention-native/prefix/optimized_prefix.py) | `optimized_prefix` |
| [work/full-attention-native/prefix/optimized_tail.py](work/full-attention-native/prefix/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/full-attention-native/prefix/qa.py](work/full-attention-native/prefix/qa.py) | `project_fused`, `finish_fused` |
| [work/full-attention-native/prefix/qb.py](work/full-attention-native/prefix/qb.py) | `project_qb`, `qb_kernel` |
| [work/full-attention-native/prefix/split_kernels.py](work/full-attention-native/prefix/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/full-attention-native/prefix/stage.py](work/full-attention-native/prefix/stage.py) | `stage_qb` |
| [work/full-attention-native/prefix/tail.py](work/full-attention-native/prefix/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/full-attention-native/prefix/tdm_tail.py](work/full-attention-native/prefix/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/full-attention-native/prefix/vector_original.py](work/full-attention-native/prefix/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/chunk_kda/prepare.py](work/full-layer-c1/aiter-gfx1250-source/attention/chunk_kda/prepare.py) | `chunk_kda_prepare_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/chunk_kda/walk.py](work/full-layer-c1/aiter-gfx1250-source/attention/chunk_kda/walk.py) | `_load_chunk`, `_chunk`, `_step`, `_store_o`, `chunk_kda_walk_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/fp8_mqa_logits.py](work/full-layer-c1/aiter-gfx1250-source/attention/fp8_mqa_logits.py) | `elementwise_max_prop_nan`, `_relu_f32_dual_gfx1250`, `relu_f32`, `_load_kv_scales_block`, `_store_logits_block`, `_mqa_dot`, `mqa_logits_loop_double_buf`, `mqa_logits_loop_pipelined`, `_gluon_fp8_mqa_logits_kernel`, `initialize`, `load_to_shared`, `load_from_shared`, `wait` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/kda_decode.py](work/full-layer-c1/aiter-gfx1250-source/attention/kda_decode.py) | `_state_row`, `_fetch_token`, `_stage_token`, `_staged_token`, `_conv_qkv_load`, `_conv_qkv_roll`, `_conv_qkv_finish`, `_process_token`, `fused_recurrent_kda_packed_decode_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/mla.py](work/full-layer-c1/aiter-gfx1250-source/attention/mla.py) | `fast_exp`, `cdiv_fn`, `apply_softcap`, `_find_seq_idx`, `_mla_decode_lds_pipeline`, `_mla_decode_fwd_kernel`, `_mla_prefill_fwd_kernel_non_pipelined`, `_mla_decode_fwd_kernel_non_pipelined`, `_mla_decode_fwd_reduce_kernel`, `initialize`, `get_next_buffer_id`, `allocate_accumulator`, `load_physical_block_idx_with_mod`, `load_physical_block_idx`, `lds_unshuffle_kv_lora_scales`, `lds_unshuffle_k_rope_scales`, `lds_unshuffle_kv_lora`, `lds_unshuffle_kv_lora_trans`, `lds_unshuffle_k_rope`, `tdm_shared_load_kv_lora_scales`, `tdm_shared_load_k_rope_scales`, `tdm_shared_load_kv_lora`, `tdm_shared_load_kv_lora_trans`, `tdm_shared_load_kv_lora_trans_slice`, `tdm_shared_load_k_rope`, `get_kv_buffer_row_offsets`, `tdm_load_global_to_shared_kv_lora`, `tdm_load_global_to_shared_k_rope`, `compute_qk_lora`, `compute_qk_rope`, `softmax_part0`, `softmax_part1`, `softmax_part1_split_head`, `compute_pkv_lora_trans`, `tdm_shared_load_and_compute_pv_lora_trans_split_head`, `store_L_M`, `store_output_3D`, `store_output_3D_split_head`, `store_output`, `process_lds_tile`, `process_lds_short` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/pa_decode_sparse.py](work/full-layer-c1/aiter-gfx1250-source/attention/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/pa_prefill_sparse.py](work/full-layer-c1/aiter-gfx1250-source/attention/pa_prefill_sparse.py) | `_pa_prefill_sparse` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/unified_attention_2d.py](work/full-layer-c1/aiter-gfx1250-source/attention/unified_attention_2d.py) | `elementwise_max_prop_nan`, `reduce_max_prop_nan`, `attention_loop_standard`, `attention_loop_reordered`, `attention_loop_tensor_subtile_split`, `load_q`, `find_seq_idx`, `_unified_attention_gluon_kernel_2d`, `initialize`, `load_k_to_shared`, `load_v_to_shared`, `load_k_from_shared`, `load_v_from_shared`, `load_block_ids`, `lds_unshuffle_k`, `lds_unshuffle_v`, `initialize`, `load_k_to_shared`, `load_v_to_shared`, `load_k_from_shared`, `load_v_from_shared`, `load_block_ids`, `lds_unshuffle_k`, `lds_unshuffle_v`, `initialize`, `compute_qk`, `apply_mask_qk`, `softmax_part0`, `softmax_part0_w_split`, `combine_ps`, `concat_subtile2`, `softmax_part1`, `compute_pv`, `store_output`, `store_output_split`, `store_partial`, `compute_qk_subtile`, `apply_mask_qk_subtile`, `split_subtile`, `softmax_part1_subtile`, `concat_subtile` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/unified_attention_3d.py](work/full-layer-c1/aiter-gfx1250-source/attention/unified_attention_3d.py) | `apply_softcap`, `find_seq_idx`, `get_q_metadata`, `cdiv_fn`, `get_seq_metadata`, `e2m1_packed_to_fp`, `_unified_attention_gluon_kernel_3d`, `initialize`, `get_next_buffer_id`, `allocate_accumulator`, `load_physical_block_idx_safe`, `load_physical_block_idx`, `load_q_from_global`, `lds_unshuffle_k_scales`, `lds_unshuffle_v_scales`, `lds_unshuffle_k`, `lds_unshuffle_v`, `tdm_shared_load_k_scales`, `tdm_shared_load_v_scales`, `tdm_shared_load_k`, `tdm_shared_load_v`, `tdm_load_global_to_shared_k`, `tdm_load_global_to_shared_v`, `compute_qk`, `apply_softcap`, `apply_addtional_mask_qk`, `softmax_part0`, `softmax_part1`, `softmax_part1_split_head`, `compute_pv`, `tdm_shared_load_and_compute_pv_split_head`, `store_L_M`, `store_output_3D`, `store_output_3D_split_head`, `store_output` |
| [work/full-layer-c1/aiter-gfx1250-source/attention/unified_attention_reduce.py](work/full-layer-c1/aiter-gfx1250-source/attention/unified_attention_reduce.py) | `reduce_segments_gluon` |
| [work/full-layer-c1/aiter-gfx1250-source/fusions/fused_clamp_act_mul.py](work/full-layer-c1/aiter-gfx1250-source/fusions/fused_clamp_act_mul.py) | `_fused_clamp_silu_mul_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/fusions/fused_kv_cache.py](work/full-layer-c1/aiter-gfx1250-source/fusions/fused_kv_cache.py) | `_make_tdm_desc_1d`, `_issue_tdm_load_1d`, `_make_tdm_desc_2d`, `_issue_tdm_load_2d`, `_issue_tdm_gather_2d`, `_store_mla_kv_cache`, `_freq_from_shared_1d`, `_freq_from_shared_2d`, `_rope_pe`, `_fused_qk_rope_cat_and_cache_mla_kernel`, `_get_kv_cache_offsets`, `_get_neox_rotated_x_2d`, `_get_gptj_rotated_x_2d`, `_rope_pe_2d`, `_fused_qk_rope_reshape_and_cache_kernel`, `_qpe_htile_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/fusions/mhc.py](work/full-layer-c1/aiter-gfx1250-source/fusions/mhc.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_a16w16.py](work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_a16w16.py) | `_gemm_a16w16_bandwidth_bound_kernel`, `_gemm_a16w16_compute_bound_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_a16w16_persistent.py](work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_a16w16_persistent.py) | `gemm_a16w16_persistent_kernel_`, `gemm_a16w16_persistent_compute_bound_kernel_` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_a8w8_blockscale.py](work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_a8w8_blockscale.py) | `_gemm_a8w8_blockscale_bandwidth_bound_kernel`, `_gemm_a8w8_blockscale_compute_bound_kernel`, `depreshuffle_b`, `_gemm_a8w8_blockscale_preshuffle_bandwidth_bound_kernel`, `_gemm_a8w8_blockscale_preshuffle_compute_bound_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_mxfp4.py](work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_mxfp4.py) | `depreshuffle_scales`, `depreshuffle_b_raw_to_kn`, `_issue_tdm_stage`, `gemm_mxfp4_preshuffle_gfx1250` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_mxfp8.py](work/full-layer-c1/aiter-gfx1250-source/gemm/basic/gemm_mxfp8.py) | `depreshuffle_b`, `_load_scale_tile`, `_gather_scale_tile`, `_gather_scale_vec`, `_read_tile`, `_gemm_mxfp8_preshuffle_bandwidth_bound_kernel`, `_gemm_mxfp8_preshuffle_compute_bound_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/batched/batched_gemm_a16w8.py](work/full-layer-c1/aiter-gfx1250-source/gemm/batched/batched_gemm_a16w8.py) | `_tile_coords`, `_issue_load`, `_batched_gemm_a16w8_gfx1250_persistent_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/gemm/batched/batched_gemm_bf16.py](work/full-layer-c1/aiter-gfx1250-source/gemm/batched/batched_gemm_bf16.py) | `_batched_gemm_bf16_bandwidth_bound_kernel`, `_batched_gemm_bf16_compute_bound_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/moe/moe_op_gemm_a16w4.py](work/full-layer-c1/aiter-gfx1250-source/moe/moe_op_gemm_a16w4.py) | `xcd_swizzle`, `unswizzle_mx_scale_gfx1250`, `_moe_gemm_a16w4` |
| [work/full-layer-c1/aiter-gfx1250-source/moe/moe_op_gemm_a4w4.py](work/full-layer-c1/aiter-gfx1250-source/moe/moe_op_gemm_a4w4.py) | `unswizzle_scales_gfx1250`, `unshuffle_weights_gfx1250`, `_moe_gemm_a4w4_prefill`, `_moe_gemm_a4w4_decode` |
| [work/full-layer-c1/aiter-gfx1250-source/moe/moe_op_gemm_a8w4.py](work/full-layer-c1/aiter-gfx1250-source/moe/moe_op_gemm_a8w4.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/full-layer-c1/aiter-gfx1250-source/moe/reduce.py](work/full-layer-c1/aiter-gfx1250-source/moe/reduce.py) | `reduce_grouped_gluon` |
| [work/full-layer-c1/aiter-gfx1250-source/normalization/fused_add_rmsnorm_pad.py](work/full-layer-c1/aiter-gfx1250-source/normalization/fused_add_rmsnorm_pad.py) | `_rmsmorm_op`, `_gluon_fused_add_rmsnorm_pad_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/normalization/fused_rmsnorm_add.py](work/full-layer-c1/aiter-gfx1250-source/normalization/fused_rmsnorm_add.py) | `_rmsnorm_op`, `_gluon_fused_rms_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/quant/fused_mxfp4_quant.py](work/full-layer-c1/aiter-gfx1250-source/quant/fused_mxfp4_quant.py) | `_rmsmorm_op`, `_mxfp4_quant_op`, `_gluon_fused_rms_mxfp4_quant_kernel`, `_gluon_fused_reduce_rms_mxfp4_quant_kernel`, `_gluon_fused_dynamic_mxfp4_quant_moe_sort_kernel` |
| [work/full-layer-c1/aiter-gfx1250-source/quant/quant.py](work/full-layer-c1/aiter-gfx1250-source/quant/quant.py) | `_mxfp4_downcast_op`, `_mxfp4_quant_tile`, `gluon_dynamic_mxfp4_quant_kernel_gfx1250`, `_mxfp8_downcast_op`, `_mxfp8_quant_tile`, `gluon_dynamic_mxfp8_quant_kernel_gfx1250` |
| [work/full-layer-c1/all-wmma/device_scope_helpers.py](work/full-layer-c1/all-wmma/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/all-wmma/full_kernel.py](work/full-layer-c1/all-wmma/full_kernel.py) | `all_wmma_layer` |
| [work/full-layer-c1/all-wmma/kernel.py](work/full-layer-c1/all-wmma/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/all-wmma/ops-before-mhc-third.py](work/full-layer-c1/all-wmma/ops-before-mhc-third.py) | `linear`, `mhc_wmma`, `attention_wmma`, `routed_linear` |
| [work/full-layer-c1/all-wmma/ops-rejected-mhc-third.py](work/full-layer-c1/all-wmma/ops-rejected-mhc-third.py) | `linear`, `mhc_wmma`, `attention_wmma`, `routed_linear` |
| [work/full-layer-c1/all-wmma/ops-v1-large-attention.py](work/full-layer-c1/all-wmma/ops-v1-large-attention.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/ops-v2-global-softmax.py](work/full-layer-c1/all-wmma/ops-v2-global-softmax.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/ops.py](work/full-layer-c1/all-wmma/ops.py) | `linear`, `mhc_wmma`, `attention_wmma`, `routed_linear` |
| [work/full-layer-c1/all-wmma/qkv_wmma.py](work/full-layer-c1/all-wmma/qkv_wmma.py) | `pack`, `wmma_matvec`, `partial_norm`, `projection_probe` |
| [work/full-layer-c1/all-wmma/region_kernel.py](work/full-layer-c1/all-wmma/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/all-wmma/split_control.py](work/full-layer-c1/all-wmma/split_control.py) | `split_stage` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/device_scope_helpers.py](work/full-layer-c1/all-wmma/v3-serial-experts/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/full_kernel.py](work/full-layer-c1/all-wmma/v3-serial-experts/full_kernel.py) | `all_wmma_layer` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/kernel.py](work/full-layer-c1/all-wmma/v3-serial-experts/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/ops-v1-large-attention.py](work/full-layer-c1/all-wmma/v3-serial-experts/ops-v1-large-attention.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/ops-v2-global-softmax.py](work/full-layer-c1/all-wmma/v3-serial-experts/ops-v2-global-softmax.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/ops.py](work/full-layer-c1/all-wmma/v3-serial-experts/ops.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/qkv_wmma.py](work/full-layer-c1/all-wmma/v3-serial-experts/qkv_wmma.py) | `pack`, `wmma_matvec`, `partial_norm`, `projection_probe` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/region_kernel.py](work/full-layer-c1/all-wmma/v3-serial-experts/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/all-wmma/v3-serial-experts/split_control.py](work/full-layer-c1/all-wmma/v3-serial-experts/split_control.py) | `split_stage` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/device_scope_helpers.py](work/full-layer-c1/all-wmma/v4-parallel-experts/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/full_kernel.py](work/full-layer-c1/all-wmma/v4-parallel-experts/full_kernel.py) | `all_wmma_layer` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/kernel.py](work/full-layer-c1/all-wmma/v4-parallel-experts/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/ops-v1-large-attention.py](work/full-layer-c1/all-wmma/v4-parallel-experts/ops-v1-large-attention.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/ops-v2-global-softmax.py](work/full-layer-c1/all-wmma/v4-parallel-experts/ops-v2-global-softmax.py) | `linear`, `mhc_wmma`, `attention_wmma` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/ops.py](work/full-layer-c1/all-wmma/v4-parallel-experts/ops.py) | `linear`, `mhc_wmma`, `attention_wmma`, `routed_linear` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/qkv_wmma.py](work/full-layer-c1/all-wmma/v4-parallel-experts/qkv_wmma.py) | `pack`, `wmma_matvec`, `partial_norm`, `projection_probe` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/region_kernel.py](work/full-layer-c1/all-wmma/v4-parallel-experts/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/all-wmma/v4-parallel-experts/split_control.py](work/full-layer-c1/all-wmma/v4-parallel-experts/split_control.py) | `split_stage` |
| [work/full-layer-c1/buffer-scoped/device_scope_helpers.py](work/full-layer-c1/buffer-scoped/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/buffer-scoped/kernel.py](work/full-layer-c1/buffer-scoped/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/buffer-scoped/split_control.py](work/full-layer-c1/buffer-scoped/split_control.py) | `split_stage` |
| [work/full-layer-c1/buffer-scoped/test_sync.py](work/full-layer-c1/buffer-scoped/test_sync.py) | `probe` |
| [work/full-layer-c1/device_scope_helpers.py](work/full-layer-c1/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/fused_glue.py](work/full-layer-c1/fused_glue.py) | `local_quant`, `mhc_local`, `act_quant_local`, `combine_post` |
| [work/full-layer-c1/grid64-native-rounding/kernel.py](work/full-layer-c1/grid64-native-rounding/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/grid64-native-rounding/split_control.py](work/full-layer-c1/grid64-native-rounding/split_control.py) | `split_stage` |
| [work/full-layer-c1/kernel.py](work/full-layer-c1/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/local-glue/control_kernel.py](work/full-layer-c1/local-glue/control_kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/local-glue/device_scope_helpers.py](work/full-layer-c1/local-glue/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/local-glue/kernel.py](work/full-layer-c1/local-glue/kernel.py) | `local_quant`, `mhc_local`, `act_quant_local`, `combine_post`, `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/local-glue/split_control.py](work/full-layer-c1/local-glue/split_control.py) | `split_stage` |
| [work/full-layer-c1/local-glue/test_sync.py](work/full-layer-c1/local-glue/test_sync.py) | `probe` |
| [work/full-layer-c1/packed/device_scope_helpers.py](work/full-layer-c1/packed/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/packed/kernel.py](work/full-layer-c1/packed/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/packed/packed_entry.py](work/full-layer-c1/packed/packed_entry.py) | `get_pointer`, `packed_full_layer` |
| [work/full-layer-c1/packed/split_control.py](work/full-layer-c1/packed/split_control.py) | `split_stage` |
| [work/full-layer-c1/packed/test_sync.py](work/full-layer-c1/packed/test_sync.py) | `probe` |
| [work/full-layer-c1/regions/device_scope_helpers.py](work/full-layer-c1/regions/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/expert-tiles/device_scope_helpers.py](work/full-layer-c1/regions/expert-tiles/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/expert-tiles/kernel.py](work/full-layer-c1/regions/expert-tiles/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/expert-tiles/optimized_kernel.py](work/full-layer-c1/regions/expert-tiles/optimized_kernel.py) | `routed_matvec` |
| [work/full-layer-c1/regions/expert-tiles/region_kernel.py](work/full-layer-c1/regions/expert-tiles/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/expert-tiles/split_control.py](work/full-layer-c1/regions/expert-tiles/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/kernel.py](work/full-layer-c1/regions/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/region_kernel.py](work/full-layer-c1/regions/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/split_control.py](work/full-layer-c1/regions/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/wmma/device_scope_helpers.py](work/full-layer-c1/regions/wmma/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/wmma/kernel.py](work/full-layer-c1/regions/wmma/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/wmma/region_kernel.py](work/full-layer-c1/regions/wmma/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/wmma/split_control.py](work/full-layer-c1/regions/wmma/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/wmma/wmma_ops.py](work/full-layer-c1/regions/wmma/wmma_ops.py) | `pack`, `wmma_matvec`, `partial_norm`, `projection_probe` |
| [work/full-layer-c1/regions/wmma-scaled-diagnostic/device_scope_helpers.py](work/full-layer-c1/regions/wmma-scaled-diagnostic/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/wmma-scaled-diagnostic/kernel.py](work/full-layer-c1/regions/wmma-scaled-diagnostic/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/wmma-scaled-diagnostic/region_kernel.py](work/full-layer-c1/regions/wmma-scaled-diagnostic/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/wmma-scaled-diagnostic/split_control.py](work/full-layer-c1/regions/wmma-scaled-diagnostic/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/wmma-scaled-diagnostic/wmma_ops.py](work/full-layer-c1/regions/wmma-scaled-diagnostic/wmma_ops.py) | `pack`, `wmma_matvec`, `partial_norm`, `projection_probe` |
| [work/full-layer-c1/regions/wmma-v1-software-scales/device_scope_helpers.py](work/full-layer-c1/regions/wmma-v1-software-scales/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/wmma-v1-software-scales/kernel.py](work/full-layer-c1/regions/wmma-v1-software-scales/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/wmma-v1-software-scales/region_kernel.py](work/full-layer-c1/regions/wmma-v1-software-scales/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/wmma-v1-software-scales/split_control.py](work/full-layer-c1/regions/wmma-v1-software-scales/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/wmma-v1-software-scales/wmma_ops.py](work/full-layer-c1/regions/wmma-v1-software-scales/wmma_ops.py) | `wmma_matvec` |
| [work/full-layer-c1/regions/wmma-v2-scaled/device_scope_helpers.py](work/full-layer-c1/regions/wmma-v2-scaled/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/wmma-v2-scaled/kernel.py](work/full-layer-c1/regions/wmma-v2-scaled/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/wmma-v2-scaled/region_kernel.py](work/full-layer-c1/regions/wmma-v2-scaled/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/wmma-v2-scaled/split_control.py](work/full-layer-c1/regions/wmma-v2-scaled/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/wmma-v2-scaled/wmma_ops.py](work/full-layer-c1/regions/wmma-v2-scaled/wmma_ops.py) | `pack`, `wmma_matvec` |
| [work/full-layer-c1/regions/wmma-v3-splitk/device_scope_helpers.py](work/full-layer-c1/regions/wmma-v3-splitk/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/full-layer-c1/regions/wmma-v3-splitk/kernel.py](work/full-layer-c1/regions/wmma-v3-splitk/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/regions/wmma-v3-splitk/region_kernel.py](work/full-layer-c1/regions/wmma-v3-splitk/region_kernel.py) | `region_kernel` |
| [work/full-layer-c1/regions/wmma-v3-splitk/split_control.py](work/full-layer-c1/regions/wmma-v3-splitk/split_control.py) | `split_stage` |
| [work/full-layer-c1/regions/wmma-v3-splitk/wmma_ops.py](work/full-layer-c1/regions/wmma-v3-splitk/wmma_ops.py) | `pack`, `wmma_matvec`, `partial_norm` |
| [work/full-layer-c1/revision1-extra-rounding/kernel.py](work/full-layer-c1/revision1-extra-rounding/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/revision1-extra-rounding/split_control.py](work/full-layer-c1/revision1-extra-rounding/split_control.py) | `split_stage` |
| [work/full-layer-c1/scoped/kernel.py](work/full-layer-c1/scoped/kernel.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/full-layer-c1/scoped/split_control.py](work/full-layer-c1/scoped/split_control.py) | `split_stage` |
| [work/full-layer-c1/scoped/test_sync.py](work/full-layer-c1/scoped/test_sync.py) | `probe` |
| [work/full-layer-c1/split_control.py](work/full-layer-c1/split_control.py) | `split_stage` |
| [work/full-layer-c1/test_sync.py](work/full-layer-c1/test_sync.py) | `probe` |
| [work/full-layer-native/ffn_kernel.py](work/full-layer-native/ffn_kernel.py) | `quant_body`, `router_body`, `route_body`, `_add`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/full-layer-native/layer_kernel.py](work/full-layer-native/layer_kernel.py) | `full_layer` |
| [work/full-layer-native/serving_kernel.py](work/full-layer-native/serving_kernel.py) | `serving_full_layer` |
| [work/integrated-c8/aiter_paged.py](work/integrated-c8/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/integrated-c8/attention_batch.py](work/integrated-c8/attention_batch.py) | `compute_attention`, `attention_body` |
| [work/integrated-c8/attention_batch_inline_prefetch.py](work/integrated-c8/attention_batch_inline_prefetch.py) | `compute_attention`, `attention_body` |
| [work/integrated-c8/attention_paged.py](work/integrated-c8/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/integrated-c8/check_attention.py](work/integrated-c8/check_attention.py) | `batch` |
| [work/integrated-c8/check_boundary.py](work/integrated-c8/check_boundary.py) | `output` |
| [work/integrated-c8/check_compressor.py](work/integrated-c8/check_compressor.py) | `batch`, `single` |
| [work/integrated-c8/check_ffn.py](work/integrated-c8/check_ffn.py) | `fb`, `fs` |
| [work/integrated-c8/check_ffn_groups.py](work/integrated-c8/check_ffn_groups.py) | `batch`, `single` |
| [work/integrated-c8/check_mhc.py](work/integrated-c8/check_mhc.py) | `c1`, `post_c1` |
| [work/integrated-c8/check_moe.py](work/integrated-c8/check_moe.py) | `run` |
| [work/integrated-c8/check_output.py](work/integrated-c8/check_output.py) | `batched` |
| [work/integrated-c8/compressor_batch.py](work/integrated-c8/compressor_batch.py) | `row`, `compressor_body` |
| [work/integrated-c8/edge_sync.py](work/integrated-c8/edge_sync.py) | `poll_edge`, `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/integrated-c8/ffn_batch.py](work/integrated-c8/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/integrated-c8/identity_batch.py](work/integrated-c8/identity_batch.py) | `identity_csa` |
| [work/integrated-c8/layout_batch.py](work/integrated-c8/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/integrated-c8/mhc_batch.py](work/integrated-c8/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/integrated-c8/mhc_integrated.py](work/integrated-c8/mhc_integrated.py) | `post_pre` |
| [work/integrated-c8/mhc_native_inline.py](work/integrated-c8/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/integrated-c8/mhc_pre_aiter.py](work/integrated-c8/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/integrated-c8/model_layer_batch.before-comp-barrier.py](work/integrated-c8/model_layer_batch.before-comp-barrier.py) | `model_layer` |
| [work/integrated-c8/model_layer_batch.py](work/integrated-c8/model_layer_batch.py) | `model_layer` |
| [work/integrated-c8/optimized_compressor.py](work/integrated-c8/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial` |
| [work/integrated-c8/optimized_moe.py](work/integrated-c8/optimized_moe.py) | `phase`, `fused` |
| [work/integrated-c8/optimized_moe_stage.py](work/integrated-c8/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/integrated-c8/optimized_routing.py](work/integrated-c8/optimized_routing.py) | `add`, `route_body` |
| [work/integrated-c8/output_batch.py](work/integrated-c8/output_batch.py) | `split_project`, `inverse_quant`, `reduce_quant`, `stage_output_weights`, `output_body` |
| [work/integrated-c8/output_batch_before_barriers.py](work/integrated-c8/output_batch_before_barriers.py) | `split_project`, `inverse_quant`, `reduce_quant`, `output_body` |
| [work/integrated-c8/post_batch.py](work/integrated-c8/post_batch.py) | `post_body`, `post_batch` |
| [work/integrated-c8/project_batch.py](work/integrated-c8/project_batch.py) | `project_body` |
| [work/integrated-c8/qa_candidate.py](work/integrated-c8/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/integrated-c8/qkv_candidate.py](work/integrated-c8/qkv_candidate.py) | `qkv_batch` |
| [work/integrated-c8/qkv_full_batch.py](work/integrated-c8/qkv_full_batch.py) | `qkv_batch` |
| [work/integrated-c8/routing_batch.py](work/integrated-c8/routing_batch.py) | `add`, `route_body` |
| [work/latency-overlap/atom-overlay/atom/model_ops/dsv4_flash_moe_kernel.py](work/latency-overlap/atom-overlay/atom/model_ops/dsv4_flash_moe_kernel.py) | `fused` |
| [work/latency-overlap/atom-overlay/atom/model_ops/dsv4_flash_moe_stage.py](work/latency-overlap/atom-overlay/atom/model_ops/dsv4_flash_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/latency-overlap/before/aiter_paged.py](work/latency-overlap/before/aiter_paged.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/latency-overlap/before/attention_batch.py](work/latency-overlap/before/attention_batch.py) | `compute_attention`, `attention_body` |
| [work/latency-overlap/before/attention_batch_inline_prefetch.py](work/latency-overlap/before/attention_batch_inline_prefetch.py) | `compute_attention`, `attention_body` |
| [work/latency-overlap/before/attention_paged.py](work/latency-overlap/before/attention_paged.py) | `compute_attention`, `attention_paged` |
| [work/latency-overlap/before/check_attention.py](work/latency-overlap/before/check_attention.py) | `batch` |
| [work/latency-overlap/before/check_boundary.py](work/latency-overlap/before/check_boundary.py) | `output` |
| [work/latency-overlap/before/check_compressor.py](work/latency-overlap/before/check_compressor.py) | `batch`, `single` |
| [work/latency-overlap/before/check_ffn.py](work/latency-overlap/before/check_ffn.py) | `fb`, `fs` |
| [work/latency-overlap/before/check_ffn_groups.py](work/latency-overlap/before/check_ffn_groups.py) | `batch`, `single` |
| [work/latency-overlap/before/check_mhc.py](work/latency-overlap/before/check_mhc.py) | `c1`, `post_c1` |
| [work/latency-overlap/before/check_moe.py](work/latency-overlap/before/check_moe.py) | `run` |
| [work/latency-overlap/before/check_output.py](work/latency-overlap/before/check_output.py) | `batched` |
| [work/latency-overlap/before/compressor_batch.py](work/latency-overlap/before/compressor_batch.py) | `row`, `compressor_body` |
| [work/latency-overlap/before/ffn_batch.py](work/latency-overlap/before/ffn_batch.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/latency-overlap/before/identity_batch.py](work/latency-overlap/before/identity_batch.py) | `identity_csa` |
| [work/latency-overlap/before/layout_batch.py](work/latency-overlap/before/layout_batch.py) | `unshuffle`, `shuffle` |
| [work/latency-overlap/before/mhc_batch.py](work/latency-overlap/before/mhc_batch.py) | `rcp`, `tail_fused`, `mhc_project`, `mhc_batch` |
| [work/latency-overlap/before/mhc_integrated.py](work/latency-overlap/before/mhc_integrated.py) | `post_pre` |
| [work/latency-overlap/before/mhc_native_inline.py](work/latency-overlap/before/mhc_native_inline.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/latency-overlap/before/mhc_pre_aiter.py](work/latency-overlap/before/mhc_pre_aiter.py) | `pre_project`, `pre` |
| [work/latency-overlap/before/model_layer_batch.before-comp-barrier.py](work/latency-overlap/before/model_layer_batch.before-comp-barrier.py) | `model_layer` |
| [work/latency-overlap/before/model_layer_batch.py](work/latency-overlap/before/model_layer_batch.py) | `model_layer` |
| [work/latency-overlap/before/optimized_compressor.py](work/latency-overlap/before/optimized_compressor.py) | `row`, `compressor_body`, `pool_partial` |
| [work/latency-overlap/before/optimized_moe.py](work/latency-overlap/before/optimized_moe.py) | `phase`, `fused` |
| [work/latency-overlap/before/optimized_moe_stage.py](work/latency-overlap/before/optimized_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/latency-overlap/before/optimized_routing.py](work/latency-overlap/before/optimized_routing.py) | `add`, `route_body` |
| [work/latency-overlap/before/output_batch.py](work/latency-overlap/before/output_batch.py) | `split_project`, `inverse_quant`, `reduce_quant`, `output_body` |
| [work/latency-overlap/before/output_batch_before_barriers.py](work/latency-overlap/before/output_batch_before_barriers.py) | `split_project`, `inverse_quant`, `reduce_quant`, `output_body` |
| [work/latency-overlap/before/post_batch.py](work/latency-overlap/before/post_batch.py) | `post_body`, `post_batch` |
| [work/latency-overlap/before/project_batch.py](work/latency-overlap/before/project_batch.py) | `project_body` |
| [work/latency-overlap/before/qa_candidate.py](work/latency-overlap/before/qa_candidate.py) | `finish`, `prefetch_qb`, `qa_candidate` |
| [work/latency-overlap/before/qkv_candidate.py](work/latency-overlap/before/qkv_candidate.py) | `qkv_batch` |
| [work/latency-overlap/before/qkv_full_batch.py](work/latency-overlap/before/qkv_full_batch.py) | `qkv_batch` |
| [work/latency-overlap/before/routing_batch.py](work/latency-overlap/before/routing_batch.py) | `add`, `route_body` |
| [work/latency-overlap/check_edges.py](work/latency-overlap/check_edges.py) | `check` |
| [work/latency-overlap/edge_sync_atomic_poll.py](work/latency-overlap/edge_sync_atomic_poll.py) | `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/latency-overlap/edge_sync_load_poll.py](work/latency-overlap/edge_sync_load_poll.py) | `publish`, `consume`, `edge`, `prefetch_bytes` |
| [work/latency-overlap/tile_qkv.py](work/latency-overlap/tile_qkv.py) | `run_qkv` |
| [work/live-vector/split_control-reference-fp8.py](work/live-vector/split_control-reference-fp8.py) | `split_stage` |
| [work/mhc-native-rms-fix/candidate/aiter_adapt.py](work/mhc-native-rms-fix/candidate/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-native-rms-fix/candidate/attention.py](work/mhc-native-rms-fix/candidate/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-native-rms-fix/candidate/combined.py](work/mhc-native-rms-fix/candidate/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-native-rms-fix/candidate/device_scope_helpers.py](work/mhc-native-rms-fix/candidate/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/aiter_adapt.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/attention.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/combined.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/device_scope_helpers.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/full_prefix.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/full_prefix.py) | `full_prefix` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/mhc.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/qa.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/qb.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/split_kernels.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/stage.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/stage.py) | `stage_qb` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/tail.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/vector_original.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-prefetch/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/aiter_adapt.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/attention.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/combined.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/device_scope_helpers.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/full_prefix.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/full_prefix.py) | `full_prefix` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/mhc.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/ordered.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/ordered.py) | `rcp`, `tail_fused`, `final_region` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/ordered_tail_snapshot.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/ordered_tail_snapshot.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/projection.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/projection.py) | `project`, `region`, `reduce_sq`, `region_sq`, `region_tail`, `project_packed`, `region_packed`, `hint`, `final_region` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/qa.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/qb.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/rcp_snapshot.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/rcp_snapshot.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/split_kernels.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/stage.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/stage.py) | `stage_qb` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/tail.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/tail_snapshot.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/tail_snapshot.py) | `tail_fused`, `optimized` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/vector_original.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-projection/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/aiter_adapt.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/attention.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/candidate.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/candidate.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/candidate_initial.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/candidate_initial.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/combined.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/device_scope_helpers.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/full_prefix.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/full_prefix.py) | `full_prefix` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/mhc.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/qa.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/qb.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/split_kernels.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/stage.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/stage.py) | `stage_qb` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/tail.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/vector_original.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-replicated/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/aiter_adapt.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/attention.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/combined.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/device_scope_helpers.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/full_prefix.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/full_prefix.py) | `full_prefix` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/mhc.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/optimized.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/optimized.py) | `tail_fused`, `optimized` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/optimized_exact.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/optimized_exact.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/optimized_rcp.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/optimized_rcp.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/qa.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/qb.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/split_kernels.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/stage.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/stage.py) | `stage_qb` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/tail.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/vector_original.py](work/mhc-native-rms-fix/candidate/experiments/mhc-opt-tail/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-native-rms-fix/candidate/full_prefix.py](work/mhc-native-rms-fix/candidate/full_prefix.py) | `full_prefix` |
| [work/mhc-native-rms-fix/candidate/mhc.py](work/mhc-native-rms-fix/candidate/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-native-rms-fix/candidate/optimized_prefix.py](work/mhc-native-rms-fix/candidate/optimized_prefix.py) | `optimized_prefix` |
| [work/mhc-native-rms-fix/candidate/optimized_tail.py](work/mhc-native-rms-fix/candidate/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-native-rms-fix/candidate/ordered-gather-ablation/tdm_tail.py](work/mhc-native-rms-fix/candidate/ordered-gather-ablation/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-native-rms-fix/candidate/qa.py](work/mhc-native-rms-fix/candidate/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-native-rms-fix/candidate/qb.py](work/mhc-native-rms-fix/candidate/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-native-rms-fix/candidate/split_kernels.py](work/mhc-native-rms-fix/candidate/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-native-rms-fix/candidate/stage.py](work/mhc-native-rms-fix/candidate/stage.py) | `stage_qb` |
| [work/mhc-native-rms-fix/candidate/tail.py](work/mhc-native-rms-fix/candidate/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-native-rms-fix/candidate/tdm_tail.py](work/mhc-native-rms-fix/candidate/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-native-rms-fix/candidate/upstream-aiter-mhc/gfx1250_mhc.py](work/mhc-native-rms-fix/candidate/upstream-aiter-mhc/gfx1250_mhc.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/mhc-native-rms-fix/candidate/upstream-attention/pa_decode_sparse.py](work/mhc-native-rms-fix/candidate/upstream-attention/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-native-rms-fix/candidate/vector_original.py](work/mhc-native-rms-fix/candidate/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-prefetch/aiter_adapt.py](work/mhc-opt-prefetch/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-prefetch/attention.py](work/mhc-opt-prefetch/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-prefetch/combined.py](work/mhc-opt-prefetch/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-prefetch/device_scope_helpers.py](work/mhc-opt-prefetch/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-prefetch/full_prefix.py](work/mhc-opt-prefetch/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-prefetch/mhc.py](work/mhc-opt-prefetch/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-prefetch/qa.py](work/mhc-opt-prefetch/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-prefetch/qb.py](work/mhc-opt-prefetch/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-prefetch/split_kernels.py](work/mhc-opt-prefetch/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-prefetch/stage.py](work/mhc-opt-prefetch/stage.py) | `stage_qb` |
| [work/mhc-opt-prefetch/tail.py](work/mhc-opt-prefetch/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-prefetch/vector_original.py](work/mhc-opt-prefetch/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-projection/aiter_adapt.py](work/mhc-opt-projection/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-projection/attention.py](work/mhc-opt-projection/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-projection/combined.py](work/mhc-opt-projection/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-projection/device_scope_helpers.py](work/mhc-opt-projection/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-projection/full_prefix.py](work/mhc-opt-projection/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-projection/mhc.py](work/mhc-opt-projection/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-projection/ordered.py](work/mhc-opt-projection/ordered.py) | `rcp`, `tail_fused`, `final_region` |
| [work/mhc-opt-projection/ordered_tail_snapshot.py](work/mhc-opt-projection/ordered_tail_snapshot.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-projection/projection.py](work/mhc-opt-projection/projection.py) | `project`, `region`, `reduce_sq`, `region_sq`, `region_tail`, `project_packed`, `region_packed`, `hint`, `final_region` |
| [work/mhc-opt-projection/qa.py](work/mhc-opt-projection/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-projection/qb.py](work/mhc-opt-projection/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-projection/rcp_snapshot.py](work/mhc-opt-projection/rcp_snapshot.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-projection/split_kernels.py](work/mhc-opt-projection/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-projection/stage.py](work/mhc-opt-projection/stage.py) | `stage_qb` |
| [work/mhc-opt-projection/tail.py](work/mhc-opt-projection/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-projection/tail_snapshot.py](work/mhc-opt-projection/tail_snapshot.py) | `tail_fused`, `optimized` |
| [work/mhc-opt-projection/upstream/pa_decode_sparse.py](work/mhc-opt-projection/upstream/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-projection/vector_original.py](work/mhc-opt-projection/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-replicated/aiter_adapt.py](work/mhc-opt-replicated/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-replicated/attention.py](work/mhc-opt-replicated/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-replicated/candidate.py](work/mhc-opt-replicated/candidate.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-replicated/candidate_initial.py](work/mhc-opt-replicated/candidate_initial.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-replicated/combined.py](work/mhc-opt-replicated/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-replicated/device_scope_helpers.py](work/mhc-opt-replicated/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-replicated/full_prefix.py](work/mhc-opt-replicated/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-replicated/mhc.py](work/mhc-opt-replicated/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-replicated/prefix_exact/aiter_adapt.py](work/mhc-opt-replicated/prefix_exact/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-replicated/prefix_exact/attention.py](work/mhc-opt-replicated/prefix_exact/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-replicated/prefix_exact/combined.py](work/mhc-opt-replicated/prefix_exact/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-replicated/prefix_exact/device_scope_helpers.py](work/mhc-opt-replicated/prefix_exact/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-replicated/prefix_exact/full_prefix.py](work/mhc-opt-replicated/prefix_exact/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-replicated/prefix_exact/mhc.py](work/mhc-opt-replicated/prefix_exact/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-replicated/prefix_exact/optimized_prefix.py](work/mhc-opt-replicated/prefix_exact/optimized_prefix.py) | `optimized_prefix` |
| [work/mhc-opt-replicated/prefix_exact/optimized_tail.py](work/mhc-opt-replicated/prefix_exact/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-replicated/prefix_exact/qa.py](work/mhc-opt-replicated/prefix_exact/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-replicated/prefix_exact/qb.py](work/mhc-opt-replicated/prefix_exact/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-replicated/prefix_exact/split_kernels.py](work/mhc-opt-replicated/prefix_exact/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-replicated/prefix_exact/stage.py](work/mhc-opt-replicated/prefix_exact/stage.py) | `stage_qb` |
| [work/mhc-opt-replicated/prefix_exact/tail.py](work/mhc-opt-replicated/prefix_exact/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-replicated/prefix_exact/tdm_tail.py](work/mhc-opt-replicated/prefix_exact/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-replicated/prefix_exact/vector_original.py](work/mhc-opt-replicated/prefix_exact/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-replicated/prefix_reshape/aiter_adapt.py](work/mhc-opt-replicated/prefix_reshape/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-replicated/prefix_reshape/attention.py](work/mhc-opt-replicated/prefix_reshape/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-replicated/prefix_reshape/combined.py](work/mhc-opt-replicated/prefix_reshape/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-replicated/prefix_reshape/device_scope_helpers.py](work/mhc-opt-replicated/prefix_reshape/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-replicated/prefix_reshape/full_prefix.py](work/mhc-opt-replicated/prefix_reshape/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-replicated/prefix_reshape/mhc.py](work/mhc-opt-replicated/prefix_reshape/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-replicated/prefix_reshape/optimized_prefix.py](work/mhc-opt-replicated/prefix_reshape/optimized_prefix.py) | `optimized_prefix` |
| [work/mhc-opt-replicated/prefix_reshape/optimized_tail.py](work/mhc-opt-replicated/prefix_reshape/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-replicated/prefix_reshape/qa.py](work/mhc-opt-replicated/prefix_reshape/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-replicated/prefix_reshape/qb.py](work/mhc-opt-replicated/prefix_reshape/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-replicated/prefix_reshape/split_kernels.py](work/mhc-opt-replicated/prefix_reshape/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-replicated/prefix_reshape/stage.py](work/mhc-opt-replicated/prefix_reshape/stage.py) | `stage_qb` |
| [work/mhc-opt-replicated/prefix_reshape/tail.py](work/mhc-opt-replicated/prefix_reshape/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-replicated/prefix_reshape/tdm_tail.py](work/mhc-opt-replicated/prefix_reshape/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-replicated/prefix_reshape/vector_original.py](work/mhc-opt-replicated/prefix_reshape/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-replicated/qa.py](work/mhc-opt-replicated/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-replicated/qb.py](work/mhc-opt-replicated/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-replicated/split_kernels.py](work/mhc-opt-replicated/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-replicated/stage.py](work/mhc-opt-replicated/stage.py) | `stage_qb` |
| [work/mhc-opt-replicated/tail.py](work/mhc-opt-replicated/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-replicated/upstream/pa_decode_sparse.py](work/mhc-opt-replicated/upstream/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-replicated/vector_original.py](work/mhc-opt-replicated/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-tail/aiter_adapt.py](work/mhc-opt-tail/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-tail/attention.py](work/mhc-opt-tail/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-tail/combined.py](work/mhc-opt-tail/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-tail/device_scope_helpers.py](work/mhc-opt-tail/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-tail/full_prefix.py](work/mhc-opt-tail/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-tail/mhc.py](work/mhc-opt-tail/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-tail/optimized.py](work/mhc-opt-tail/optimized.py) | `tail_fused`, `optimized` |
| [work/mhc-opt-tail/optimized_exact.py](work/mhc-opt-tail/optimized_exact.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-tail/optimized_rcp.py](work/mhc-opt-tail/optimized_rcp.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-tail/prefix_exact/aiter_adapt.py](work/mhc-opt-tail/prefix_exact/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-tail/prefix_exact/attention.py](work/mhc-opt-tail/prefix_exact/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-tail/prefix_exact/combined.py](work/mhc-opt-tail/prefix_exact/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-tail/prefix_exact/device_scope_helpers.py](work/mhc-opt-tail/prefix_exact/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-tail/prefix_exact/full_prefix.py](work/mhc-opt-tail/prefix_exact/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-tail/prefix_exact/mhc.py](work/mhc-opt-tail/prefix_exact/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-tail/prefix_exact/optimized_prefix.py](work/mhc-opt-tail/prefix_exact/optimized_prefix.py) | `optimized_prefix` |
| [work/mhc-opt-tail/prefix_exact/optimized_tail.py](work/mhc-opt-tail/prefix_exact/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-tail/prefix_exact/qa.py](work/mhc-opt-tail/prefix_exact/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-tail/prefix_exact/qb.py](work/mhc-opt-tail/prefix_exact/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-tail/prefix_exact/split_kernels.py](work/mhc-opt-tail/prefix_exact/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-tail/prefix_exact/stage.py](work/mhc-opt-tail/prefix_exact/stage.py) | `stage_qb` |
| [work/mhc-opt-tail/prefix_exact/tail.py](work/mhc-opt-tail/prefix_exact/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-tail/prefix_exact/tdm_tail.py](work/mhc-opt-tail/prefix_exact/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-tail/prefix_exact/vector_original.py](work/mhc-opt-tail/prefix_exact/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-tail/prefix_reshape/aiter_adapt.py](work/mhc-opt-tail/prefix_reshape/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-tail/prefix_reshape/attention.py](work/mhc-opt-tail/prefix_reshape/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-opt-tail/prefix_reshape/combined.py](work/mhc-opt-tail/prefix_reshape/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-opt-tail/prefix_reshape/device_scope_helpers.py](work/mhc-opt-tail/prefix_reshape/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-opt-tail/prefix_reshape/full_prefix.py](work/mhc-opt-tail/prefix_reshape/full_prefix.py) | `full_prefix` |
| [work/mhc-opt-tail/prefix_reshape/mhc.py](work/mhc-opt-tail/prefix_reshape/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-opt-tail/prefix_reshape/optimized_prefix.py](work/mhc-opt-tail/prefix_reshape/optimized_prefix.py) | `optimized_prefix` |
| [work/mhc-opt-tail/prefix_reshape/optimized_tail.py](work/mhc-opt-tail/prefix_reshape/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-opt-tail/prefix_reshape/qa.py](work/mhc-opt-tail/prefix_reshape/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-tail/prefix_reshape/qb.py](work/mhc-opt-tail/prefix_reshape/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-tail/prefix_reshape/split_kernels.py](work/mhc-opt-tail/prefix_reshape/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-tail/prefix_reshape/stage.py](work/mhc-opt-tail/prefix_reshape/stage.py) | `stage_qb` |
| [work/mhc-opt-tail/prefix_reshape/tail.py](work/mhc-opt-tail/prefix_reshape/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-tail/prefix_reshape/tdm_tail.py](work/mhc-opt-tail/prefix_reshape/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-opt-tail/prefix_reshape/vector_original.py](work/mhc-opt-tail/prefix_reshape/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-opt-tail/qa.py](work/mhc-opt-tail/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-opt-tail/qb.py](work/mhc-opt-tail/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-opt-tail/split_kernels.py](work/mhc-opt-tail/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-opt-tail/stage.py](work/mhc-opt-tail/stage.py) | `stage_qb` |
| [work/mhc-opt-tail/tail.py](work/mhc-opt-tail/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-opt-tail/upstream/pa_decode_sparse.py](work/mhc-opt-tail/upstream/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-opt-tail/upstream-mhc/gfx1250_mhc.py](work/mhc-opt-tail/upstream-mhc/gfx1250_mhc.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/mhc-opt-tail/vector_original.py](work/mhc-opt-tail/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-optimized/aiter_adapt.py](work/mhc-optimized/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-optimized/attention.py](work/mhc-optimized/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-optimized/combined.py](work/mhc-optimized/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-optimized/device_scope_helpers.py](work/mhc-optimized/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/aiter_adapt.py](work/mhc-optimized/experiments/mhc-opt-prefetch/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/attention.py](work/mhc-optimized/experiments/mhc-opt-prefetch/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/combined.py](work/mhc-optimized/experiments/mhc-opt-prefetch/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/device_scope_helpers.py](work/mhc-optimized/experiments/mhc-opt-prefetch/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/full_prefix.py](work/mhc-optimized/experiments/mhc-opt-prefetch/full_prefix.py) | `full_prefix` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/mhc.py](work/mhc-optimized/experiments/mhc-opt-prefetch/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/qa.py](work/mhc-optimized/experiments/mhc-opt-prefetch/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/qb.py](work/mhc-optimized/experiments/mhc-opt-prefetch/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/split_kernels.py](work/mhc-optimized/experiments/mhc-opt-prefetch/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/stage.py](work/mhc-optimized/experiments/mhc-opt-prefetch/stage.py) | `stage_qb` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/tail.py](work/mhc-optimized/experiments/mhc-opt-prefetch/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-optimized/experiments/mhc-opt-prefetch/vector_original.py](work/mhc-optimized/experiments/mhc-opt-prefetch/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-optimized/experiments/mhc-opt-projection/aiter_adapt.py](work/mhc-optimized/experiments/mhc-opt-projection/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-projection/attention.py](work/mhc-optimized/experiments/mhc-opt-projection/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-optimized/experiments/mhc-opt-projection/combined.py](work/mhc-optimized/experiments/mhc-opt-projection/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-optimized/experiments/mhc-opt-projection/device_scope_helpers.py](work/mhc-optimized/experiments/mhc-opt-projection/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-optimized/experiments/mhc-opt-projection/full_prefix.py](work/mhc-optimized/experiments/mhc-opt-projection/full_prefix.py) | `full_prefix` |
| [work/mhc-optimized/experiments/mhc-opt-projection/mhc.py](work/mhc-optimized/experiments/mhc-opt-projection/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-optimized/experiments/mhc-opt-projection/ordered.py](work/mhc-optimized/experiments/mhc-opt-projection/ordered.py) | `rcp`, `tail_fused`, `final_region` |
| [work/mhc-optimized/experiments/mhc-opt-projection/ordered_tail_snapshot.py](work/mhc-optimized/experiments/mhc-opt-projection/ordered_tail_snapshot.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-optimized/experiments/mhc-opt-projection/projection.py](work/mhc-optimized/experiments/mhc-opt-projection/projection.py) | `project`, `region`, `reduce_sq`, `region_sq`, `region_tail`, `project_packed`, `region_packed`, `hint`, `final_region` |
| [work/mhc-optimized/experiments/mhc-opt-projection/qa.py](work/mhc-optimized/experiments/mhc-opt-projection/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-optimized/experiments/mhc-opt-projection/qb.py](work/mhc-optimized/experiments/mhc-opt-projection/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-optimized/experiments/mhc-opt-projection/rcp_snapshot.py](work/mhc-optimized/experiments/mhc-opt-projection/rcp_snapshot.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-optimized/experiments/mhc-opt-projection/split_kernels.py](work/mhc-optimized/experiments/mhc-opt-projection/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-projection/stage.py](work/mhc-optimized/experiments/mhc-opt-projection/stage.py) | `stage_qb` |
| [work/mhc-optimized/experiments/mhc-opt-projection/tail.py](work/mhc-optimized/experiments/mhc-opt-projection/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-optimized/experiments/mhc-opt-projection/tail_snapshot.py](work/mhc-optimized/experiments/mhc-opt-projection/tail_snapshot.py) | `tail_fused`, `optimized` |
| [work/mhc-optimized/experiments/mhc-opt-projection/vector_original.py](work/mhc-optimized/experiments/mhc-opt-projection/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/aiter_adapt.py](work/mhc-optimized/experiments/mhc-opt-replicated/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/attention.py](work/mhc-optimized/experiments/mhc-opt-replicated/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/candidate.py](work/mhc-optimized/experiments/mhc-opt-replicated/candidate.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/candidate_initial.py](work/mhc-optimized/experiments/mhc-opt-replicated/candidate_initial.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/combined.py](work/mhc-optimized/experiments/mhc-opt-replicated/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/device_scope_helpers.py](work/mhc-optimized/experiments/mhc-opt-replicated/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/full_prefix.py](work/mhc-optimized/experiments/mhc-opt-replicated/full_prefix.py) | `full_prefix` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/mhc.py](work/mhc-optimized/experiments/mhc-opt-replicated/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/qa.py](work/mhc-optimized/experiments/mhc-opt-replicated/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/qb.py](work/mhc-optimized/experiments/mhc-opt-replicated/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/split_kernels.py](work/mhc-optimized/experiments/mhc-opt-replicated/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/stage.py](work/mhc-optimized/experiments/mhc-opt-replicated/stage.py) | `stage_qb` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/tail.py](work/mhc-optimized/experiments/mhc-opt-replicated/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-optimized/experiments/mhc-opt-replicated/vector_original.py](work/mhc-optimized/experiments/mhc-opt-replicated/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-optimized/experiments/mhc-opt-tail/aiter_adapt.py](work/mhc-optimized/experiments/mhc-opt-tail/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-tail/attention.py](work/mhc-optimized/experiments/mhc-opt-tail/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-optimized/experiments/mhc-opt-tail/combined.py](work/mhc-optimized/experiments/mhc-opt-tail/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-optimized/experiments/mhc-opt-tail/device_scope_helpers.py](work/mhc-optimized/experiments/mhc-opt-tail/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-optimized/experiments/mhc-opt-tail/full_prefix.py](work/mhc-optimized/experiments/mhc-opt-tail/full_prefix.py) | `full_prefix` |
| [work/mhc-optimized/experiments/mhc-opt-tail/mhc.py](work/mhc-optimized/experiments/mhc-opt-tail/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-optimized/experiments/mhc-opt-tail/optimized.py](work/mhc-optimized/experiments/mhc-opt-tail/optimized.py) | `tail_fused`, `optimized` |
| [work/mhc-optimized/experiments/mhc-opt-tail/optimized_exact.py](work/mhc-optimized/experiments/mhc-opt-tail/optimized_exact.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-optimized/experiments/mhc-opt-tail/optimized_rcp.py](work/mhc-optimized/experiments/mhc-opt-tail/optimized_rcp.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-optimized/experiments/mhc-opt-tail/qa.py](work/mhc-optimized/experiments/mhc-opt-tail/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-optimized/experiments/mhc-opt-tail/qb.py](work/mhc-optimized/experiments/mhc-opt-tail/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-optimized/experiments/mhc-opt-tail/split_kernels.py](work/mhc-optimized/experiments/mhc-opt-tail/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-optimized/experiments/mhc-opt-tail/stage.py](work/mhc-optimized/experiments/mhc-opt-tail/stage.py) | `stage_qb` |
| [work/mhc-optimized/experiments/mhc-opt-tail/tail.py](work/mhc-optimized/experiments/mhc-opt-tail/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-optimized/experiments/mhc-opt-tail/vector_original.py](work/mhc-optimized/experiments/mhc-opt-tail/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-optimized/full_prefix.py](work/mhc-optimized/full_prefix.py) | `full_prefix` |
| [work/mhc-optimized/mhc.py](work/mhc-optimized/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-optimized/optimized_prefix.py](work/mhc-optimized/optimized_prefix.py) | `optimized_prefix` |
| [work/mhc-optimized/optimized_tail.py](work/mhc-optimized/optimized_tail.py) | `rcp`, `tail_fused`, `optimized` |
| [work/mhc-optimized/ordered-gather-ablation/tdm_tail.py](work/mhc-optimized/ordered-gather-ablation/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-optimized/qa.py](work/mhc-optimized/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-optimized/qb.py](work/mhc-optimized/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-optimized/split_kernels.py](work/mhc-optimized/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-optimized/stage.py](work/mhc-optimized/stage.py) | `stage_qb` |
| [work/mhc-optimized/tail.py](work/mhc-optimized/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-optimized/tdm_tail.py](work/mhc-optimized/tdm_tail.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/mhc-optimized/upstream-aiter-mhc/gfx1250_mhc.py](work/mhc-optimized/upstream-aiter-mhc/gfx1250_mhc.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/mhc-optimized/upstream-attention/pa_decode_sparse.py](work/mhc-optimized/upstream-attention/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-optimized/vector_original.py](work/mhc-optimized/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/mhc-through-attention/aiter_adapt.py](work/mhc-through-attention/aiter_adapt.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-through-attention/attention.py](work/mhc-through-attention/attention.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/mhc-through-attention/combined.py](work/mhc-through-attention/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/mhc-through-attention/device_scope_helpers.py](work/mhc-through-attention/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/mhc-through-attention/full_prefix.py](work/mhc-through-attention/full_prefix.py) | `full_prefix` |
| [work/mhc-through-attention/mhc.py](work/mhc-through-attention/mhc.py) | `sinkhorn`, `mhc_pre`, `norm`, `mhc_project`, `mhc_reduce`, `mhc_stage`, `mhc_body`, `mhc_region` |
| [work/mhc-through-attention/qa.py](work/mhc-through-attention/qa.py) | `project_fused`, `finish_fused` |
| [work/mhc-through-attention/qb.py](work/mhc-through-attention/qb.py) | `project_qb`, `qb_kernel` |
| [work/mhc-through-attention/split_kernels.py](work/mhc-through-attention/split_kernels.py) | `qa_reduce`, `qa_norm`, `qb_stage`, `cache_store`, `attention_partials`, `attention_reduce` |
| [work/mhc-through-attention/stage.py](work/mhc-through-attention/stage.py) | `stage_qb` |
| [work/mhc-through-attention/tail.py](work/mhc-through-attention/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/mhc-through-attention/upstream/pa_decode_sparse.py](work/mhc-through-attention/upstream/pa_decode_sparse.py) | `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/mhc-through-attention/vector_original.py](work/mhc-through-attention/vector_original.py) | `clock`, `sync`, `quant`, `matvec`, `norm`, `mhc_mix`, `sinkhorn`, `mhc_pre`, `mhc_post`, `rope`, `attention`, `activation`, `route`, `full_layer` |
| [work/modelwide/aiter_matched.py](work/modelwide/aiter_matched.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/modelwide/aiter_nativeprob.py](work/modelwide/aiter_nativeprob.py) | `overlay_current`, `_pa_decode_sparse`, `_pa_decode_sparse_reduce` |
| [work/modelwide/atom-source/atom/model_ops/dsv4_flash_moe_kernel.py](work/modelwide/atom-source/atom/model_ops/dsv4_flash_moe_kernel.py) | `fused` |
| [work/modelwide/atom-source/atom/model_ops/dsv4_flash_moe_stage.py](work/modelwide/atom-source/atom/model_ops/dsv4_flash_moe_stage.py) | `unswizzle_mx_scale_gfx1250`, `unshuffle_weight_gfx1250`, `_moe_gemm_a8w4_decode_persistent`, `_moe_gemm_a8w4_decode`, `_moe_gemm_a8w4_prefill` |
| [work/modelwide/attention_extended.py](work/modelwide/attention_extended.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/modelwide/attention_matched.py](work/modelwide/attention_matched.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/modelwide/attention_nativeprob.py](work/modelwide/attention_nativeprob.py) | `prefetch_cache`, `compute_attention`, `cache_attention_body`, `attention_kernel`, `qkv_attention` |
| [work/modelwide/combined_scaled.py](work/modelwide/combined_scaled.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/modelwide/compressor.py](work/modelwide/compressor.py) | `row`, `compressor_body` |
| [work/modelwide/compressor_serial.py](work/modelwide/compressor_serial.py) | `row`, `compressor_body` |
| [work/modelwide/device_scope_helpers.py](work/modelwide/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/modelwide/ffn_extended.py](work/modelwide/ffn_extended.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/modelwide/ffn_precise.py](work/modelwide/ffn_precise.py) | `quant_body`, `router_body`, `shared_body`, `activation_quant`, `ffn_body` |
| [work/modelwide/identity_indexer.py](work/modelwide/identity_indexer.py) | `identity_csa` |
| [work/modelwide/mhc_precise.py](work/modelwide/mhc_precise.py) | `rcp`, `tail_fused`, `candidate_body`, `candidate` |
| [work/modelwide/model_layer.py](work/modelwide/model_layer.py) | `model_layer` |
| [work/modelwide/model_layer_matched.py](work/modelwide/model_layer_matched.py) | `model_layer` |
| [work/modelwide/model_layer_precise.py](work/modelwide/model_layer_precise.py) | `model_layer` |
| [work/modelwide/model_layer_scaled.py](work/modelwide/model_layer_scaled.py) | `model_layer` |
| [work/modelwide/output_scaled.py](work/modelwide/output_scaled.py) | `inverse_quant`, `split_project`, `reduce_quant`, `output_body`, `output_kernel`, `output_stage` |
| [work/modelwide/post_precise.py](work/modelwide/post_precise.py) | `post_body`, `post` |
| [work/modelwide/project_compress.py](work/modelwide/project_compress.py) | `project_body`, `project_compress` |
| [work/modelwide/qa_scaled.py](work/modelwide/qa_scaled.py) | `project_fused`, `finish_fused` |
| [work/modelwide/qb_scaled.py](work/modelwide/qb_scaled.py) | `project_qb`, `qb_kernel` |
| [work/modelwide/region_kernels.py](work/modelwide/region_kernels.py) | `qkv`, `output`, `mhc` |
| [work/modelwide/routing.py](work/modelwide/routing.py) | `add`, `route_body` |
| [work/modelwide/sync.py](work/modelwide/sync.py) | `clock`, `poll_read`, `wait_target`, `sync` |
| [work/originals/mhc_kernel.py](work/originals/mhc_kernel.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/qkv-all-tdm/base/fused.py](work/qkv-all-tdm/base/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-all-tdm/base/opt.py](work/qkv-all-tdm/base/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-all-tdm/combined.py](work/qkv-all-tdm/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/qkv-all-tdm/device_scope_helpers.py](work/qkv-all-tdm/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-all-tdm/qa.py](work/qkv-all-tdm/qa.py) | `project_fused`, `finish_fused` |
| [work/qkv-all-tdm/qb.py](work/qkv-all-tdm/qb.py) | `project_qb`, `qb_kernel` |
| [work/qkv-all-tdm/stage.py](work/qkv-all-tdm/stage.py) | `stage_qb` |
| [work/qkv-all-tdm/tail.py](work/qkv-all-tdm/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/qkv-all-tdm-l2/base/fused.py](work/qkv-all-tdm-l2/base/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-all-tdm-l2/base/opt.py](work/qkv-all-tdm-l2/base/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-all-tdm-l2/combined.py](work/qkv-all-tdm-l2/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/qkv-all-tdm-l2/device_scope_helpers.py](work/qkv-all-tdm-l2/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-all-tdm-l2/qa.py](work/qkv-all-tdm-l2/qa.py) | `project_fused`, `finish_fused` |
| [work/qkv-all-tdm-l2/qb.py](work/qkv-all-tdm-l2/qb.py) | `project_qb`, `qb_kernel` |
| [work/qkv-all-tdm-l2/stage.py](work/qkv-all-tdm-l2/stage.py) | `stage_qb` |
| [work/qkv-all-tdm-l2/tail.py](work/qkv-all-tdm-l2/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/qkv-asm-opt/baseline/kernels.py](work/qkv-asm-opt/baseline/kernels.py) | `drain`, `initialize`, `finish_group`, `project`, `reduce_group`, `reduce_norm` |
| [work/qkv-asm-opt/device_scope_helpers.py](work/qkv-asm-opt/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-asm-opt/fused-v1.py](work/qkv-asm-opt/fused-v1.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v2.py](work/qkv-asm-opt/fused-v2.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v3.py](work/qkv-asm-opt/fused-v3.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v4.py](work/qkv-asm-opt/fused-v4.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v5.py](work/qkv-asm-opt/fused-v5.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v6.py](work/qkv-asm-opt/fused-v6.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v7.py](work/qkv-asm-opt/fused-v7.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v8.py](work/qkv-asm-opt/fused-v8.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused-v9.py](work/qkv-asm-opt/fused-v9.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/fused.py](work/qkv-asm-opt/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/initall.py](work/qkv-asm-opt/initall.py) | `initialize_all` |
| [work/qkv-asm-opt/layout_probe.py](work/qkv-asm-opt/layout_probe.py) | `probe` |
| [work/qkv-asm-opt/opt-v1.py](work/qkv-asm-opt/opt-v1.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-asm-opt/opt-v2.py](work/qkv-asm-opt/opt-v2.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-asm-opt/opt-v3.py](work/qkv-asm-opt/opt-v3.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-asm-opt/opt.py](work/qkv-asm-opt/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-asm-opt/packed128_experiment.py](work/qkv-asm-opt/packed128_experiment.py) | `project_fused`, `finish_fused` |
| [work/qkv-asm-opt/prefetch.py](work/qkv-asm-opt/prefetch.py) | `project_prefetch`, `finish_fused` |
| [work/qkv-asm-opt/reduce2d.py](work/qkv-asm-opt/reduce2d.py) | `project_reduce2d`, `finish_fused` |
| [work/qkv-asm-opt/singlewave.py](work/qkv-asm-opt/singlewave.py) | `project_singlewave`, `finish_single` |
| [work/qkv-asm-opt/tdm_opt.py](work/qkv-asm-opt/tdm_opt.py) | `project_tdm`, `finish_fused` |
| [work/qkv-asm-opt/timed.py](work/qkv-asm-opt/timed.py) | `project_timed`, `stamp`, `finish_fused` |
| [work/qkv-atomic-ab/device_scope_helpers.py](work/qkv-atomic-ab/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-atomic-ab/kernels-acqrel.py](work/qkv-atomic-ab/kernels-acqrel.py) | `drain`, `initialize`, `finish_group`, `project`, `reduce_group`, `reduce_norm` |
| [work/qkv-atomic-ab/kernels.py](work/qkv-atomic-ab/kernels.py) | `drain`, `initialize`, `finish_group`, `project`, `reduce_group`, `reduce_norm` |
| [work/qkv-complete-ablation/base/fused.py](work/qkv-complete-ablation/base/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-complete-ablation/base/opt.py](work/qkv-complete-ablation/base/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-complete-ablation/combined.py](work/qkv-complete-ablation/combined.py) | `clock`, `sync`, `complete` |
| [work/qkv-complete-ablation/device_scope_helpers.py](work/qkv-complete-ablation/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-complete-ablation/qa.py](work/qkv-complete-ablation/qa.py) | `project_fused`, `finish_fused` |
| [work/qkv-complete-ablation/qb.py](work/qkv-complete-ablation/qb.py) | `project_qb`, `qb_kernel` |
| [work/qkv-complete-ablation/tail.py](work/qkv-complete-ablation/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/qkv-complete-tuning/base/fused.py](work/qkv-complete-tuning/base/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-complete-tuning/base/opt.py](work/qkv-complete-tuning/base/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-complete-tuning/combined.py](work/qkv-complete-tuning/combined.py) | `clock`, `sync`, `complete` |
| [work/qkv-complete-tuning/device_scope_helpers.py](work/qkv-complete-tuning/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-complete-tuning/qa.py](work/qkv-complete-tuning/qa.py) | `project_fused`, `finish_fused` |
| [work/qkv-complete-tuning/qb.py](work/qkv-complete-tuning/qb.py) | `project_qb`, `qb_kernel` |
| [work/qkv-complete-tuning/tail.py](work/qkv-complete-tuning/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/qkv-load-poll/base/fused.py](work/qkv-load-poll/base/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-load-poll/base/opt.py](work/qkv-load-poll/base/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-load-poll/combined-short-timeout.py](work/qkv-load-poll/combined-short-timeout.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/qkv-load-poll/combined.py](work/qkv-load-poll/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/qkv-load-poll/combined_original.py](work/qkv-load-poll/combined_original.py) | `clock`, `sync`, `complete` |
| [work/qkv-load-poll/device_scope_helpers.py](work/qkv-load-poll/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-load-poll/publication.py](work/qkv-load-poll/publication.py) | `publish_test` |
| [work/qkv-load-poll/publication_acyclic.py](work/qkv-load-poll/publication_acyclic.py) | `once` |
| [work/qkv-load-poll/publication_diagnostic.py](work/qkv-load-poll/publication_diagnostic.py) | `publish_test` |
| [work/qkv-load-poll/publication_long.py](work/qkv-load-poll/publication_long.py) | `publish_test` |
| [work/qkv-load-poll/qa-atomic-version.py](work/qkv-load-poll/qa-atomic-version.py) | `project_fused`, `finish_fused` |
| [work/qkv-load-poll/qa.py](work/qkv-load-poll/qa.py) | `project_fused`, `finish_fused` |
| [work/qkv-load-poll/qb.py](work/qkv-load-poll/qb.py) | `project_qb`, `qb_kernel` |
| [work/qkv-load-poll/tail.py](work/qkv-load-poll/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/qkv-qb-prefetch/base/fused.py](work/qkv-qb-prefetch/base/fused.py) | `project_fused`, `finish_fused` |
| [work/qkv-qb-prefetch/base/opt.py](work/qkv-qb-prefetch/base/opt.py) | `project_opt`, `rg`, `reduce_opt` |
| [work/qkv-qb-prefetch/combined.py](work/qkv-qb-prefetch/combined.py) | `clock`, `poll_read`, `wait_target`, `sync`, `complete` |
| [work/qkv-qb-prefetch/device_scope_helpers.py](work/qkv-qb-prefetch/device_scope_helpers.py) | `dev_load`, `dev_store` |
| [work/qkv-qb-prefetch/qa.py](work/qkv-qb-prefetch/qa.py) | `project_fused`, `finish_fused` |
| [work/qkv-qb-prefetch/qb.py](work/qkv-qb-prefetch/qb.py) | `project_qb`, `qb_kernel` |
| [work/qkv-qb-prefetch/stage.py](work/qkv-qb-prefetch/stage.py) | `stage_qb` |
| [work/qkv-qb-prefetch/tail.py](work/qkv-qb-prefetch/tail.py) | `quant_scale`, `quant`, `norm_rope_body`, `norm_rope`, `norm_only` |
| [work/resume-prep/masked_prefetch.py](work/resume-prep/masked_prefetch.py) | `masked_line_prefetch`, `test_prefetch` |
| [work/resume-prep/qa_batch.before_direct_kv.py](work/resume-prep/qa_batch.before_direct_kv.py) | `finish`, `qa_batch` |
| [work/resume-prep/qkv_batch.before_guard.py](work/resume-prep/qkv_batch.before_guard.py) | `qkv_batch` |
| [work/resume-prep/shadow_metrics.py](work/resume-prep/shadow_metrics.py) | `compare` |
| [work/rollout-c1248/regions.py](work/rollout-c1248/regions.py) | `region` |

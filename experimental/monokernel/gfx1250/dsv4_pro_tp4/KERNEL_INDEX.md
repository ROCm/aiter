# Kernel symbol index

Includes JIT kernels and inlined JIT helpers across selected, reference, and historical variants. Repeated symbols in separate experiment directories are retained deliberately. See README.md for selected entry points.

| Source | JIT symbols |
|---|---|
| [runtime/app/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/mhc.py](runtime/app/aiter/aiter/ops/triton/_gluon_kernels/gfx1250/fusions/mhc.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/aiter-pre-audit/check_pre.py](work/aiter-pre-audit/check_pre.py) | `new`, `old` |
| [work/aiter-pre-audit/check_pre_real.py](work/aiter-pre-audit/check_pre_real.py) | `new`, `old` |
| [work/aiter-pre-audit/model_layer_batch.before.py](work/aiter-pre-audit/model_layer_batch.before.py) | `model_layer` |
| [work/candidate/mhc_kernel.py](work/candidate/mhc_kernel.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/originals/mhc_kernel.py](work/originals/mhc_kernel.py) | `_load_fn_plane`, `_res_offsets`, `_issue_loads`, `_wait_next_stage`, `_read_stage`, `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel`, `_rcp`, `_pin`, `_sigmoid`, `_res_ptrs`, `_load_partials`, `_fold_splits`, `_load_norm_w`, `_reduce_apply_main`, `_post_sinkhorn_ws`, `_mhc_pre_reduce_apply_gfx1250_kernel` |
| [work/round2/candidate/gemm_mxfp8.py](work/round2/candidate/gemm_mxfp8.py) | `depreshuffle_b`, `_load_scale_tile`, `_gather_scale_tile`, `_gather_scale_vec`, `_read_tile`, `_gemm_mxfp8_preshuffle_bandwidth_bound_kernel`, `_gemm_mxfp8_preshuffle_compute_bound_kernel` |
| [work/round2/originals/gemm_mxfp8.py](work/round2/originals/gemm_mxfp8.py) | `depreshuffle_b`, `_load_scale_tile`, `_gather_scale_tile`, `_gather_scale_vec`, `_read_tile`, `_gemm_mxfp8_preshuffle_bandwidth_bound_kernel`, `_gemm_mxfp8_preshuffle_compute_bound_kernel` |
| [work/round2/semaphore_probe.py](work/round2/semaphore_probe.py) | `_publish`, `_reduce` |
| [work/round2/semaphore_probe_cg.py](work/round2/semaphore_probe_cg.py) | `_publish`, `_reduce` |
| [work/round2/semaphore_probe_scoped_asm.py](work/round2/semaphore_probe_scoped_asm.py) | `_publish`, `_reduce` |
| [work/round2/semaphore_probe_scoped_batched.py](work/round2/semaphore_probe_scoped_batched.py) | `_publish`, `_reduce` |
| [work/round2/semaphore_probe_scoped_buffer.py](work/round2/semaphore_probe_scoped_buffer.py) | `_publish`, `_reduce` |
| [work/round2/semaphore_probe_scoped_drain.py](work/round2/semaphore_probe_scoped_drain.py) | `_publish`, `_reduce` |

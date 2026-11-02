# Actual ATOM mHC microbenchmark

C1 DeepSeek V4 Flash hidden4096, 4 residual streams. This region has no sequence-length dimension; 1024 input /1024 output uses these same decode shapes. Checkpoint layer0 weights and embedding fixture. ATOM commit f19778df4054e94d64d6f6bf6dd68f5681bd5945; AITER7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe. GPU selection HIP_VISIBLE_DEVICES=1.

| Scope | Round1 us | Round2 us | Launches |
|---|---:|---:|---:|
| Actual ATOM layer0 mHC+norm, shuffled residual input |7.36760|7.36667|3|
| Optimized core, plain residual input (different boundary) |6.09038|6.08850|1|
| Optimized with required unshuffle adapter (same input boundary) |7.65397|7.66195|2|
| Native later-layer preceding-post + next-pre + norm |4.58421|4.58515|2|

Use round2. Flat graph128 invocations with identical reused tensor addresses,5 warmups,40 alternated-order GPU-event samples per round, median. The later-layer row is separately measured and has a different scope; no matched optimized counterpart is provided. Primary fair layer0 comparison is7.36667 vs7.66195us: optimized is4.01% slower.

Actual ATOM forward enables shuffled residuals by default for gfx1250 C1. Layer0 hc_pre unshuffles them, packs HAF once as BF16 hi/lo during weight loading (packing excluded from timing), and calls AITER HIP pre GEMM plus fused norm. Later-layer fused_post_pre automatically dispatches AITER Gluon with shuffled residuals. `actual_atom_hc_pre.py` is the exact method AST extracted from current ATOM and executed by bench.py; source snapshots/hashes retained.

Layer0 kernel names:

- `mhc_res_unshuffle_h4096_hc4_ks64_b2`
- `void aiter::mhc_pre_gemm_sqrsum_kernel<std::bfloat16_t, 4, 64, 16, 64, true, true>(float*, float*, std::bfloat16_t*, float*, int, int, int, int, int, int, int)`
- `void aiter::mhc_pre_big_fuse_rmsnorm_kernel<std::bfloat16_t, 9, 4, 1, 4096, 1024, 1024, false>(float*, float*, std::bfloat16_t*, float*, float*, float*, float*, std::bfloat16_t*, std::bfloat16_t*, int, int, int, int, float, float, float, float, float, int, int)`

Optimized same-boundary kernels: `mhc_res_unshuffle_h4096_hc4_ks64_b2`, `candidate`.

Later-layer native kernels:

- `_mhc_post_pre_gemm_sqrsum_gfx1250_kernel_C_4096_BLOCK_M_16_K_LOOP_1_SPLIT_K_64_NUM_STAGES_2_RES_SHUFFLED_1_W_PRESHUFFLED_1`
- `_mhc_pre_reduce_apply_gfx1250_kernel_C_4096_ROWS_1_CHUNK_1024_N_CHUNKS_4_S_64_FUSE_RMSNORM_1_RES_SHUFFLED_1_FUSE_QUANT_0`

Numerical caveat: candidate is not fully validated against actual ATOM. 137 inputs tested:9 embeddings plus128 deterministic random inputs. One case71(seed171071) exceeds original0.001 relative RMS threshold, X error0.00106110. Checkpoint embeddings max0.00014515. Postmax7.04e-7, combmax1.49e-7; mailbox health0. Source confirms native HIP RMS sums unrounded FP32 stream mixture squared but normalizes its BF16-rounded value. Candidate RMS sums BF16-rounded mixture squared. This distinction previously hidden by custom unfused control is real and must be addressed before replacement claims.

`native_api.py` exports NativeMHC; input must be shuffled. Run `docker exec -e HIP_VISIBLE_DEVICES=1 vigokhal_dsv4_tp4 python /work/atom-actual-microbench/mhc/bench.py`. `postpre.py` runs later-layer diagnostic. Raw samples, profiler traces and full kernel signatures retained. Profiler durations are one-shot diagnostics and not the graph timing table.

Final graph methodology: capture one invocation, then copy its kernel nodes and dependencies128 times into one flat HIP graph. This reuses all output/scratch addresses as serving graph replay does. No child-graph scheduling overhead. Original fresh-output graph measurements retained in allocation-diagnostic; child-graph experiments retained in childgraph-diagnostic and excluded from final table. flatgraph-verification.json verifies an8-copy graph:24 native /8 core /16 adapter kernel executions (128-copy profiler trace was truncated at254records).

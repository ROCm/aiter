## ROCm/aiter PR #5996 — [Triton/Gluon] [FlyDSL] FlyDSL QSA indexer scorer + sparse GQA

**Adds FlyDSL indexer and sparse-GQA kernels, and a qsa_layer that uses them for two measured query shapes and the pinned vLLM AMD Triton path otherwise.**

Review (advisory): ⚠️ NEEDS WORK — not independently refuted
Validation (deterministic): INCONCLUSIVE — auto-run pytest of op_tests/test_flydsl_qsa.py on gfx950 (AMD Instinct MI355X, HIP 6): 40 passed in 20.06s; execution_receipt pass for aiter.ops.flydsl.kernels.qsa.k2:qsa_k2. Skipped: correctness_s1_grid (no --grid, so coverage is the repository cases only)
Perf (advisory): NOT RUN — stages.perf skipped because this PR adds the target, and a base-vs-head timing requires --perf-control-column, which was not supplied

⚠️ [inferred] aiter/ops/flydsl/kernels/qsa/k1.py sets n_req from context_lens.shape[0] and then indexes page_table[safe_req, logical_page] with no page_table.shape[0] check — at context_lens length 2, a 1-row page_table, and token_to_req=1, is request 1 inside the table? **Author must** reject that pair or clamp the request id to the table. -- late finding: step 7
⚠️ [inferred] aiter/ops/flydsl/kernels/qsa/k1.py _idiv casts query_positions to Uint32 before dividing by _R=4 — at query_positions=-5, does visible become the full context instead of zero blocks? **Author must** reject negative positions or divide them as signed values. -- late finding: step 7
⚠️ [verified] aiter/ops/flydsl/qsa.py _qsa_layer_triton drops score_scale and softmax_scale that _qsa_layer_flydsl passes through, so backend="triton" with score_scale=1.0 cannot apply the caller's scale. **Author must** forward both scales or reject a non-default scale on the Triton path.
⚠️ [inferred] aiter/ops/triton/_triton_kernels/attention/qsa_vllm_amd.py expand_qsa_block_indices_cuda stores while columns < OUTPUT_WIDTH, and aiter/ops/flydsl/qsa.py forwards caller indices unchecked — does a [M, 16] buffer survive token_topk=2048 (width 2051)? **Author must** require that width before expand. -- late finding: step 7
⚠️ [inferred] aiter/ops/flydsl/qsa.py states 1.03x to 1.67x versus live AMD, but this review never timed op_tests/test_flydsl_qsa.py on base and head. **Author must** attach the sweep log for that grid.
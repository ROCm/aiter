# Ticket 02 default screening execution review

Read-only focused review of the active 02.4 screening/bootstrap additions in
`/root/workspace/tmp_dev/aiter-mxfp4-02/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`
and `op_tests/tuning_tests/test_mxfp4_staged_execution.py`. Applied
karpathy-guidelines, using current ticket 02/spec/pipeline audit. Reviewed
`_prepare_screening_case`, `_prepare_gemm1_screening`, `_screen_gemm1`,
`_gemm2_screening_call`, `_screen_gemm2`, `_check_reference_consumer` and
`_check_reference_pair` against the default main production wrappers, direct
G1 entry, Torch reference functions and native-BM layout-population helper.
Ticket 03 FP8/custom-settings work is pending and excluded from findings.
No tests, worker processes or GPU tasks were launched; no repository/tracker
edits were made.

## Result

No concrete default-path execution defect found in this focused review.

G1 case generation, Opus sorting, BM-dependent fused input quantization,
buffer allocation and initial wrapper launch happen before `run_perftest`.
Its timed callable invokes only the direct G1 entry, using the exact
wrapper-created payload/scale buffers. BM16 retains BF16 hidden input,
includes inline quantization in the kernel and supplies empty byte
placeholders for ignored packed-A/scale arguments. Other BMs pass actual FP4
input and fused sorted/shuffled scales. Logical dimensions derive from the
packed weights in the same way as production. Separated gate, no route
weighting in G1, full candidate tuning fields, BM16 native output scales,
default activation limits and Situv2 beta defaults match wrapper execution.

Each G2 attempt generates a fresh reference-producer intermediate during
untimed setup. Its callable includes output zeroing on every atomic
iteration, or the full existing BF16 reduce/scatter wrapper. It passes
sorted route weights, reverse-sorted metadata and actual payload/scales.
Intermediate/callable ownership stays local to the attempt; no cross-candidate
tensor cache was introduced. Default BF16 output does not require the
ticket-03 topk_weights forwarding repair reserved for FP8 reduction.

Consumer self-test independently creates Torch stage1 output, populates
sorted FP4 route payload and native BM scale layout through the established
helper, runs the chosen atomic consumer and compares against ordinary Torch
stage2. Pair bootstrap uses the existing complete pipeline and the same
ordinary Torch reference. Both reject missing/non-finite/above-gate error;
neither times bootstrap nor submits it as a cache/winner result.

The controlled tests assert setup/timing boundaries, default parameters for
all four BMs/three activations, fresh intermediate lifetimes, atomic
initialization, non-atomic complete-call wiring and Torch/pair bootstrap
ordering. These mocked checks do not establish numerical/GPU/production/AOT
acceptance; root owns the subsequent real task after the implementation
commit. `git diff --check` passed for the inspected worktree.

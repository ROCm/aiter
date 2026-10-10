# Ticket 03 configured AOT increment review

Read-only review of `2308ce0bf...9260524a5` in
`/root/workspace/tmp_dev/aiter-mxfp4-03`. Scope: committed
`aiter/aot/flydsl/mxfp4_moe.py`, AOT usage/evidence documentation and
`op_tests/tuning_tests/test_mxfp4_aot_configuration.py`; active fixture manifest
and replay harness were excluded. Followed `karpathy-guidelines` and current
spec/ticket03/environment-options. No tests or GPU tasks were run.

## P2: Documented custom-limit batch collection always fails on shipped Situv2 CSV

Locations: `aiter/aot/flydsl/README.md:110` configured batch example and
`aiter/aot/flydsl/mxfp4_moe.py:170` custom-limit Situv2 rejection.

The new README promises batch `run_aot` can receive limit5 through
`AITER_MXFP4_TUNE_SWIGLU_LIMIT` and the intended CSV through
`AITER_CONFIG_FMOE`. However `common._collect_aot_jobs_for(OpKind.MXFP4_MOE)`
collects `mxfp4_moe.DEFAULT_CSVS`, which always contains the globbed model CSVs
and only appends the active configuration. It includes shipped
`aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv`; its first data row is an
MXMOE Situv2 kernel. With the documented environment, `parse_csv` resolves5,
then raises `ValueError("swiglu_limit is not supported for situv2")` while
collecting that file. `run_aot` consequently stops before any job dispatch,
regardless of the supported Swiglu-only active CSV. The controlled tests cover
single-file parse and standalone main; they do not exercise batch collection of
its actual default set, so they cannot detect this failure.

Make the documented batch configuration usable while preserving rejection of
actual unsupported Situv2/limit combinations and caller-owned configuration
selection. A selected-CSV mechanism or equivalent compatible collection path
needs concrete coverage; simply relaxing the Situv2 rejection would violate the
agreed activation boundary. Single-file standalone `--csv` is unaffected.

## Other reviewed boundaries

No additional concrete defect found in this increment. Omitted/unset/empty
limits retain Swiglu7 and unclamped SiLU/Situv2; explicit positive finite limits
flow to stage1 closures and existing compile keys. Both previous Situv2 beta
variants remain for default jobs. Gate interpretation is exact1 at collection,
matching production. Enabled eligible scatter jobs retain BM128/BN128 or256,
no bias, persistent CU-specific dispatch, uint8 packed `[max_sorted,H/2]`
target and uint8 E8M0 `[max_sorted,H/32]` scale buffers; the host dispatcher
validates those shapes. Bias scatter remains BF16; default/atomic and FP8 reduce
host entries and reduction arguments are preserved. Existing runtime cache
functions are unchanged and jobs flow through the same dispatchers.

Committed documentation reports29 controlled AOT checks; these substitute the
external compiler and establish host-entry behavior only. They do not establish
real compiled-cache/numerical deployment acceptance. Final integration
Standards/Spec review and remaining ticket03 GPU/AOT acceptance remain required.

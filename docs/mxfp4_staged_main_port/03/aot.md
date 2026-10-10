# Configured MXFP4 AOT implementation

The existing public `parse_csv(csv_path, swiglu_limit=None)` discovery seam
preserves omitted callers and supplies a configured positive finite limit to
stage1 compile jobs. Omission reads `AITER_MXFP4_TUNE_SWIGLU_LIMIT`; unset/empty
selects activation defaults. Standalone `--swiglu-limit` takes precedence.
Situv2 custom limits are rejected while both existing parameter variants remain.

Layout-v2 jobs mirror production's exact `1` opt-in gates. FP4 scatter selects
BM128, BN128/BN256, no bias, persistent execution, uint8 payload
`[max_sorted, H/2]` and E8M0 scale bytes `[max_sorted, H/32]`. Bias scatter uses
BF16. FP8 reduce and atomic/default BF16 retain their existing compile entries.
The runtime key functions and winner CSV schema are unchanged.

Tests use the approved discovery and `compile_one_config` seams. They load the
normal ROCm/FlyDSL toolchain with synthetic import discovery and empty physical
device visibility. Only `flydsl.compiler.compile`, an external system boundary,
is substituted; the real host dispatcher validates FP4 dtype and buffer shapes.
GPU initialization, synchronization, events and graphs fail closed. These tests
establish job configuration and host compile-entry behavior; no GPU kernel
compiles or launches, and they provide no numerical/precompiled acceptance.

TDD cycles reproduced an unsupported limit keyword, absent invalid/Situv2
rejection, missing environment forwarding, rejected standalone argument,
incorrect BF16 scatter job and missing FP4 scale-buffer failures before the
corresponding implementation. The final 29 tests cover defaults, custom limit,
unset/empty/enabled gates, both switches with default/custom limit, BN128/BN256
packed scatter, BN512 BF16 fallback, bias and atomic jobs.

Reproduce from this checkout:

```bash
env PATH=/root/workspace/tmp_dev/mxfp4-port-env/bin:$PATH \
  ROCR_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  GPU_ARCHS=gfx950 AITER_TEST_CU_NUM=256 CU_NUM=256 AITER_USE_SYSTEM_TRITON=1 \
  AITER_JIT_DIR=/root/workspace/tmp_dev/mxfp4-port-notes/ticket03-aot-jit \
  CK_DIR=/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel \
  /root/workspace/tmp_dev/mxfp4-port-env/bin/python -m pytest \
  op_tests/tuning_tests/test_mxfp4_aot_configuration.py -q
```

The existing system Triton 3.4 version warning remains visible. AITER core
extension compilation is CPU import preparation. Matching deployment commands
and caller-owned CSV settings are documented in
[the AOT usage guide](../../../../aiter/aot/flydsl/README.md).

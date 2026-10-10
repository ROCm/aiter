# Ordinary fMoE lifecycle regression

Ticket 01 exercises the approved ordinary-tuner seam in
[`test_ordinary_tuner_lifecycle.py`](../../../op_tests/tuning_tests/test_ordinary_tuner_lifecycle.py).
The tests instantiate the real `FmoeTuner`, use real CK registry task enumeration,
and run its input selection, tuning dispatch, `post_process`, result CSV writer,
sorting and final summary. Command checks launch a Python subprocess that runs
`gemm_moe_tune.py` through its actual `__main__` entry point with `runpy.run_path`.
The subprocess receives controlled device properties and measurement dispatch
inside that process.

No GPU kernels or real measurements are needed by this suite. Imports still
require the normal AITER ROCm/FlyDSL environment; this is not evidence of operation
in a GPU-free software environment. The controlled dispatch reads real task tags
without calling their data producers, references, kernels or timers. Device
properties simulate gfx942 with 304 CUs to verify that ordinary fMoE retains its
own supported-device contract rather than inheriting MXFP4's gfx950 restriction.
The fixture uses the build's public `dtypes.fp8` spelling, since dtype aliases are
selected at import time.

The final 17 cases cover:

- Duplicate input removal, final shape ordering/deduplication, and historical
  winner preservation, including NT and numeric fields.
- Ordinary fMoE's exact CSV schema, CK kernel names, `nt=1` selection, separate
  `us1=2.25` and `us2=3.5` stage timings, `us=5.75`, error strings, and numeric
  TFLOPS/bandwidth. These are controlled fixture values, not measured performance.
- The user-selected paired profile and existing raw `profile_fmoe.csv` path.
  Raw profiles retain failed measurements; paired profiles contain usable stage
  combinations. Profile writes are redirected into the test temporary directory.
- Crash (`-1`), timeout (`inf`) and incorrect (`err=0.5`) controlled outcomes,
  a naturally unsupported G1U0 row that remains unfinished, later-success
  persistence, accurate failed-versus-unfinished accounting and status 1.
- `--all` and normal selection, with both same and separate input/output files;
  no-work selections and empty input with absent or historical output.
- Legacy shared rows without gfx/optional flags, per-row gfx942/gfx950 inference,
  removal of obsolete tagged rows, and preserved NT/numeric values.
- Actual command exit 0 on complete success and exit 1 on failed or unfinished
  requests, with valid later winners saved in all three subprocess scenarios.

The failure slice found an actionable shared-summary issue before the fix:
ordinary fMoE failure keys retained enums and dtype objects, while requested CSV
keys were strings. A failed token-4 row was consequently also reported unfinished,
yielding `[4, 6]` instead of the specified unfinished row `[6]`. All three controlled
failure variants failed this assertion. Ticket 01's shared summary repair
normalizes object key values before matching; the same tests passed afterwards.
No ordinary tuning implementation was replaced or patched to hide this mismatch.

An initial fixture used a hard-coded fp8 spelling unsupported by the current
import-time `dtype2str_dict`, and failed before dispatch with a `KeyError`. Using
the public `dtypes.fp8` spelling corrected that fixture assumption. This was a
test-fixture correction, not a new supported-device or kernel change.

The last full runtime check before the incident isolation pause ran on
2026-10-10 with Python 3.12.3, the root-prepared environment and exact-base CK
`96afae1d54fc79c26857de7e1683b0dde1ff3c6f`:

```bash
PATH=/root/workspace/tmp_dev/mxfp4-port-env/bin:$PATH \
CK_DIR=/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel \
AITER_USE_SYSTEM_TRITON=1 \
/root/workspace/tmp_dev/mxfp4-port-env/bin/python -m pytest \
  op_tests/tuning_tests/test_ordinary_tuner_lifecycle.py -q --disable-warnings
```

Result: **16 passed, 36 warnings in 14.61s**. Warnings include the existing
system-Triton version warning (3.4.0 versus the requested 3.6.0) and pandas
concatenation/downcasting future warnings. No Triton kernels are executed by
these tests.

Following the root's incident isolation pause, static review added fail-closed
guards in `prohibit_gpu_work()`. Every parent test enters these guards through an
autouse fixture, and `controlled_environment()` independently enters them in the
subprocess before the real script is executed. The guards prohibit process starts,
uncontrolled measurement dispatch, worker/work-group execution, data producers,
performance timing, CUDA initialization/events/graphs and tensor allocation
factories (explicit CPU dummy allocations are allowed). Device visibility is empty (`ROCR_VISIBLE_DEVICES`,
`HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`) in the parent fixture and explicitly
in the command child's environment, and `multiprocessing.get_context` returns an
inert `None` sentinel. Only the controlled dispatch inside the fixture overrides the blocked
measurement boundary. A guard violation uses `pytest.fail`, which escapes the
tuner's `catch(Exception)` recovery path.

After static isolation review, the final guarded combined run passed **55 tests
in 13.76 s**, including all **17 ordinary fMoE cases**. It is retained in
`isolated-lifecycle-final.txt`. This run uses empty device visibility and the
shared offline import bootstrap. Explicit CPU factory allocations for build
metadata are permitted; default or GPU allocations remain forbidden. It includes
the review regression proving that retuning updates a key with duplicate old
winner rows. Black, Ruff, syntax and diff checks passed.

This controlled regression does not establish kernel correctness, measured
performance, production dispatch or AOT acceptance. Those remain in the separate
GPU acceptance work coordinated by the root.

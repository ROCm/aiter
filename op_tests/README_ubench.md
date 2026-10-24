# MI355X uBench

This ports the dense-operator uBench workflow from `gfx1250/microbench` to
`main`, where the gfx950 FlyDSL GEMM implementation is available.
`bench_combo.py` currently dispatches gfx950 only. The gfx1250 suite remains
on its source branch; this PR does not import unavailable gfx1250 kernels.

## MI355X

Install AITER and its pinned FlyDSL dependency. Clone ROCm/FlyDSL for the
Softmax kernel sources (the compiler wheel alone does not install that checkout).

```bash
python op_tests/bench_combo.py --arch gfx950 --perf \
  --ops softmax a16w16 --flydsl-root /path/to/FlyDSL \
  --data-init norm --seed 0 --dtype bf16 fp16 fp32 \
  --ubench-profiler --smi-monitor --output results.json
```

Omit `--arch` to detect the active GPU. Set device visibility before starting
Python and reserve an idle GPU. The selected SMI ordinal must be the active
benchmark device, and SMI maps HIP ordinal to PCI BDF rather than assuming SMI
enumeration matches HIP enumeration.

Supported gfx950 operators:

- `softmax`: row-wise, out-of-place FlyDSL Softmax, with FP32 accumulation and
  BF16/FP16/FP32 input/output. Includes AITER's 10 unit-test shapes and four
  additional throughput/launch cases.
- `a16w16`: BF16 `A[M,K] @ B[N,K].T`, no weight preshuffle. Imports the exact
  `_TUNED` configurations from the installed AITER HGEMM test using AST parsing,
  without executing that test module's import-time GPU work.

The port covers these two operators only. gfx1250-specific operators such as
Mega MoE are not advertised as working on gfx950.

## Measurement and correctness

Input generation, JSON table output, optional profiler timing and SMI replay
reuse the `gfx1250/microbench` input/JSON helpers (in `ubench_common.py`),
its unchanged `smi_monitor.py`, and main's `aiter.test_common.run_perftest`.
The provenance commit is recorded in `ubench_common.py`.
The additional HIP graph/event timer amortizes host dispatch over 64 operator
calls and reports all samples from five rounds. Compilation, allocation and
reference computation are outside timing. Preallocated output buffers are
poisoned before replay to detect empty captures or incorrect stream selection.
Validation runs both before capture and after replay. Invalid/zero profiler
times are errors, not infinite-throughput results.

`--ubench-profiler` adds the original `run_perftest(testGraph=True)` result as
a separate field. Profiler instrumentation can materially perturb short
kernels; compare the two timing methods separately. Default graph timing reuses
inputs and therefore measures a resident working set, not guaranteed cold HBM.
SMI replay occurs after latency measurement.

Softmax checks normalized output error and row sums against FP32 softmax of the
actual rounded input. The row-sum tolerance includes the row-sum error of the
FP32 reference rounded to the output dtype (important for long FP16 rows). GEMM checks every element with a reference-RMS floor for
cancellation and a normalized RMS bound. These checks are intentionally
stricter than the existing GEMM unit test; a stock configuration can fail, and
that failure is retained in JSON and reflected in the process exit status.

Use `--configs file.json` to load a mapping such as
`{"softmax:0:bf16": {"BLOCK_THREADS": 256, "THREADS_PER_ROW": 64, "ROWS_PER_BLOCK": 4}}`.
`--case-index` selects explicitly indexed cases. `--variant` labels the result.
Experimental implementation modules can be supplied with `--softmax-module`
or `--gemm-module`; they must implement the same operator contract.

## Verification

```bash
pytest -q op_tests/test_bench_combo_dispatch.py
python op_tests/bench_combo.py --arch gfx950 --help
```

CPU tests cover argument forwarding, lazy dispatch, unsupported operators and
help without GPU initialization. Actual gfx1250 runtime regression testing
requires a gfx1250 machine; CPU dispatch tests do not substitute for it.

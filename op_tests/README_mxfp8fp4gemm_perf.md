# gfx1250 ASM F8GEMM performance reproduction

Use an installed aiter environment on gfx1250, with this branch checked out.
From the repository root, run:

```bash
python -m op_tests.test_mxfp8fp4gemm_perf
```

The test uses the same Python environment and checkout as the caller. It has no
machine-specific paths, container requirement, GPU idle polling or clock sampler.
Select the intended GPU using your environment's usual visibility settings and
run without competing GPU work.

Each case/input pair runs in a fresh process. Within that process the complete
native test runs six times, each with Torch reference 2/100, ASM AP0 2/100, then
formal ASM AP1 2/100 (warmup/iterations). All three stages use the native
`aiter.test_common.run_perftest`, including its automatic input rotation.
`--no-reduce` keeps reduction out of the formal GEMM timing. Its output buffer is
allocated once outside each benchmark call and reused within that call.

| Case | M,N,K | Default split-K |
|---|---|---:|
| wqkv_a | 512,2048,7168 | 8 |
| wo_b | 512,7168,16384 | 4 |
| gate_up_proj | 512,6144,7168 | 4 |
| w2 | 512,7168,3072 | 4 |
| wq_b | 512,65536,1536 | 1 |
| indexer_wq_b | 512,8192,1536 | 1 |

Both `constant/constant` and `uniform/auto` data/scale pairs use seed 0. AP1,
split-K and iteration defaults come from `test_mxfp8fp4gemm.py`.
To select one case or inspect the commands without GPU execution:

```bash
python -m op_tests.test_mxfp8fp4gemm_perf --cases indexer_wq_b --data-init constant
python -m op_tests.test_mxfp8fp4gemm_perf --dry-run
```

The underlying native entry also supports direct use and explicit overrides:

```bash
python -m op_tests.test_mxfp8fp4gemm \
  --mode perf --intype a8w8 --shape 512,8192,1536 \
  --pre-benchmark --no-reduce --repeat 6 \
  --data-init constant --scale-init constant --json indexer_constant.json
```

Each execution creates `f8gemm_perf_<timestamp>/`; `--output-dir` can specify a
new directory. `perf.csv` contains individual times and their arithmetic mean;
`summary.json` also preserves correctness verdicts, and `results.json` preserves
all accepted native result fields. Native logs and JSON for **all** attempts,
including incomplete ones, are retained alongside `attempts.json`.
Native stdout/stderr is also streamed to the terminal while each attempt runs.
Each repeat prints its elapsed kernel time, TFLOPS, correctness verdict and raw
GPU event count as soon as it finishes. After all cases finish, the terminal
prints a Markdown summary table with each accepted repeat's time, the arithmetic
mean and correctness counts. Incomplete attempts are marked as excluded from
this final table.

A formal call should produce 100 raw GEMM events. If the profiler returns fewer,
the reproduction test retries the entire six-call group in a fresh process, up
to `--max-attempts 15`. Selection is based on event completeness, never latency.
Failure to obtain a complete group exits with an error. This check applies to
formal timing; prebenchmark profiler counts are retained in native JSON but are
not used to select results. Correctness warnings are retained and printed.

The native timing helper excludes its first profiled iteration and applies IQR
outlier filtering, so 100 raw events do not imply 100 samples in the final mean.
Missing **raw** profiler events are a separate issue: they occur before that
filter. A native `--mode profile` diagnostic on PyTorch
`2.11.0+rocm7.15.0a20260712` (trace metadata: ROCTracer 4.1) recorded 100
`hipDrvLaunchKernelEx` calls followed by `hipDeviceSynchronize`, but only 98 GPU
GEMM events. Correlation IDs identified launches 99 and 100 as the missing GPU
records. This locates the missing records in the profiling collection path;
it does not indicate a shorter Python launch loop. Asynchronous activity delivery
around profiler shutdown/flush is a possible cause, not a confirmed root cause.
The current evidence does not isolate a specific PyTorch/Kineto/ROCTracer bug.

GPU state, runtime versions and clock conditions can affect the numerical times
even when the benchmark protocol is the same.

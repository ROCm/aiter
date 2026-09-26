# GEMM tuning

`tune_gemm.py` tries configs through the wrapper's `get_gemm_config()` lookup.
`gemm_cases.py` builds inputs; `profile_gemm.py` runs one config under rocprofv3.
The parent process never initializes the GPU.

## Run it

Use a Linux AITER source checkout with its test dependencies, Triton, ROCm,
rocprofv3 on `PATH`, and the target GPU. Run on each architecture to be tuned
(gfx942, gfx950, gfx1250, or a future architecture with a working kernel and
`DEFAULT.json`).

```bash
cd aiter/ops/triton/utils/_triton/tuning
python3 tune_gemm.py --list
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 --timeout 900
HIP_VISIBLE_DEVICES=0 python3 tune_gemm.py gemm_a16w16 M=32 N=1024 K=1024 --backend both
```

For wrappers exposing a backend, choose `--backend triton`, `--backend gluon`,
or `--backend both`. `both` tunes Gluon then Triton independently, with separate
configs and logs. An unsupported backend is logged and the other is still tried;
the command exits with a failure status if either could not be tuned.
Omitting the flag preserves the wrapper's default, including its Gluon preference
on gfx1250 where supported. Explicit Triton selection remains available on
wrappers that support it. Fixed-backend cases take no backend flag.
See [COVERAGE.md](COVERAGE.md) for supported routes and previously missing cases.

## The three rules

1. Profile the config the wrapper currently uses, including its default fallback.
2. Try the candidates. Log errors and keep going after a failed config.
3. Replace an existing tuned config only if faster; if no tuned M bucket exists,
   add the best working config. The current default can be the winner.

Each config gets a fresh process. `--timeout` limits its whole run, including
compilation (default 900 seconds). On a timeout, the parent kills the process
group, logs the failure and starts the next config. A crash also loses only that
config. This isolates process failures; it cannot repair a GPU needing a reset.

## What is measured

rocprofv3 records GPU kernel timestamps. For each invocation, the score sums
only the selected GEMM and split-K reduction kernel durations. The final score
is the median over `--runs` invocations (default 250), with a cache clear before
each. Compilation, warmup, Python/wrapper overhead, cache clears, output resets,
and unrelated kernels are excluded. Per-invocation GPU markers keep repeated
launches and reductions together; an incomplete trace is treated as a failure.

Kernel-name filters live in each case's `kernels` setting (`gemm` by default,
`_ff_` for fused feed-forward). Use `--kernel-names` to override the substrings.

Inputs use the same seed in each process. Outputs must be finite and are checked
against the current config when it produces a reference. If the current config
fails before creating that reference, candidates receive finite-output checks
only. Run the kernel's correctness tests on the target GPU before using results.

## Parameters and files

Each architecture/backend's `DEFAULT.json` declares that kernel's own keys.
There is no required common schema for Triton and Gluon. Common launch keys have
shared candidate ranges; other keys use values in that family's JSON files.
The case's `space` constrains them; command-line values override it:

```bash
python3 tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024 \
    --backend triton --space BLOCK_SIZE_M=16,32 BLOCK_SIZE_K=64,128 num_warps=4,8
```

Search keys absent from the selected backend's default are reported and skipped.
Failures, including the config, traceback, compiler/driver output, exit code or
timeout, go to `errors-<op>-<shape>-<backend-or-auto>.txt`. All timings and statuses
go to `results-<op>-<shape>-<backend-or-auto>.jsonl`. These logs are replaced on
rerun. Temporary raw traces are removed after tuning.

The observed lookup selects `configs/<arch>/<backend>/gemm/<family>/`, the
filename and M bucket. Other buckets are preserved. Run jobs writing the same
file sequentially. Modes sharing a lookup share tuned results; give them distinct
lookup keys if they need independent winners. Restart consumers after installing
configs because the runtime caches them.

## Add a GEMM

1. Use `get_gemm_config()` and provide a valid `DEFAULT.json` for each supported
   architecture/backend. The author keeps keys consistent with that kernel.
2. Add a `@gemm_case()` function named for the wrapper or variant. Reuse its test
   input generator; return a callable that invokes the wrapper without `config=`,
   returns every output, and resets accumulators. Keep imports inside the case
   and forward an exposed backend argument.
3. Set `space={...}` for kernel-specific constraints and `kernels=(...)` if the
   default name filter does not cover all its GEMM/reduction launches. New input
   generators must be repeatable with the worker's Torch seed.
4. Update this README, [coverage](COVERAGE.md), the [Triton README](../../../README.md)
   and [Copilot instructions](../../../../../../.github/instructions/aiter-ops-triton.instructions.md)
   when introducing new author requirements. Follow the [config rules](../../../configs/CLAUDE.md).

## Differences from the old tuning scripts

| Old functionality | This workflow |
| --- | --- |
| rocprofv3 timing, cold-cache runs, timeout, continuation after process failure | Retained; a failed config no longer discards its whole batch. |
| Many configs per process (`SCREEN_MAX_BATCH`) | One process per config for simple crash isolation; startup overhead makes sweeps slower. |
| Shared Triton pruning rules and tile blacklisting after resource errors | Not applied across different kernels. Authors constrain their case's `space`; invalid configs are logged individually. |
| `screen-*.log`, separate JSON writer, multi-shape log aggregation | JSONL results and direct updates for one requested shape. Use a shell loop for multiple shapes. Old CLI/log formats are not compatible. |
| Separate installed-config verification command | Baseline timing is part of every run; no standalone verify-only command. |
| Optional GB/s and TFLOPS printing in the trace parser | Kernel microseconds only. |

MoE and grouped GEMMs using other config loaders still need their own lookup and
input integration; see the coverage inventory. No CPU unit-test suite is added.

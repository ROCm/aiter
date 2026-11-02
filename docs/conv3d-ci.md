# FlyDSL Conv3d CI and manual performance checks

## Two intentionally separate entry points

The standard Aiter CI invokes `python op_tests/test_flydsl_conv_implicit.py`
without arguments through the existing test script. This defaults to `--mode ci`:
19 correctness-only cases: six Wan representatives, six keyword regressions,
and seven unfiltered compile-key/static checks. The Wan families are `conv_in`,
`down_0_1`, `resample_96`, `time_conv_192`, `wdec_up_192_L0`, and `wdec_conv_out`.
By default, `conv_in` uses the real `368x544` production shape, requires an exact
(non-borrowed), device-matching tuned lookup, and verifies that the actual compile
parameter uses the returned tile. The other five Wan families use compact
`64x64` spatial extents while preserving channel/filter/temporal/stride/padding
semantics. An explicit `--wan-res` overrides the resolution for every selected
Wan family and disables the exact-tuned requirement, even for `368x544`.
The small checks cover reuse layouts/gating, split-K, six compile-key isolation
shapes, and static block-row reuse. Each CI case uses two direct-call warmups
and ten checked iterations, with no performance timing helper. It does not run
the complete model performance matrix or change shared sharding, timing helpers,
or production kernels. The numerical bar remains zero mismatched elements at
`rtol=atol=2e-2`; non-finite results fail.

The explicit performance entry point is:

```bash
python op_tests/test_flydsl_conv_implicit.py --mode perf
```

It retains the full existing bf16 keyword, Wan2.1 encode/aux/decode, and
Qwen-Image T=1 conv2d matrix, including compile-key isolation and the static
block-row regression. The default model resolutions remain Wan `480x832` and
`368x544` at 81 frames, and Qwen `1024x1024` and `1328x1328`. Performance mode
also checks correctness; it is not permission to relax tolerances.

Existing case and resolution filters remain available for explicitly authorized
manual runs, for example:

```bash
# Focused Wan performance check.
HIP_VISIBLE_DEVICES=0 python op_tests/test_flydsl_conv_implicit.py --mode perf \
  -c down_0_1 --wan-res 368x544 --wan-frames 17

# Focused Qwen performance check; the explicit case filter excludes Wan rows.
HIP_VISIBLE_DEVICES=0 python op_tests/test_flydsl_conv_implicit.py --mode perf \
  -c res_96_L0 --qwen-res 1664x928

# Preserve the existing combined case-selection interface.
HIP_VISIBLE_DEVICES=0 python op_tests/test_flydsl_conv_implicit.py --mode perf \
  -c down_0_1 res_96_L0 --wan-res 480x832 --qwen-res 1024x1024
```

Resolve device visibility in the actual environment before using these examples:
`HIP_VISIBLE_DEVICES=0` is the first device in that process's ROCm inventory,
not necessarily host physical GPU 0 when another visibility mask is inherited.
Do not run GPU commands without authorization and an immediately preceding idle
gate. Keep the existing CLI's small unfiltered regression checks in mind when
estimating a filtered run's work.

## Dedicated manual workflow

[Conv3d Manual Perf](<../.github/workflows/conv3d-perf.yaml>) has only
`workflow_dispatch`. It does not add a schedule, push/PR GPU job, or modify
[standard Aiter CI](<../.github/workflows/aiter-test.yaml>). It becomes dispatchable
after reviewed code is merged into the upstream default branch. Dispatches from
forks or non-default branches are reported on a GitHub-hosted policy job without
allocating a self-hosted GPU runner. There is no arbitrary ref, PR checkout, image,
shell command, or argument input. Repository write access is required by GitHub
to dispatch; the workflow token has only `contents: read`, no OIDC or write scopes,
and checkout does not persist credentials. Repository administrators must still
restrict access to self-hosted runners and review default-branch changes; Docker
and GPU device access are not an untrusted-code security boundary.

The job uses the existing `linux-aiter-mi35x-1` runner label and Python 3.12 image
output from [CI Config](<../.github/workflows/ci-config.yaml>), rather than inventing
a reusable setup workflow. Optional registry authentication uses the existing CI
secret-name outputs; absent credentials use the public image. Registry config is
run-local and removed without logging out other jobs. The image digest is logged.
Aiter is installed into a fresh container from the dispatched checkout, with
pinned submodules and repository requirements. Installed FlyDSL must exactly
match `FLYDSL_VERSION` in [setup.py](<../setup.py>); the workflow reads that value
rather than maintaining another version pin.

`physical_gpu` defaults to `0` and accepts a single non-negative ROCm SMI physical
index. The workflow honors the existing runner's assigned render-device file,
queries the selected GPU's unique ID and PCI address, sets
`ROCR_VISIBLE_DEVICES=GPU-<unique-id>` with `HIP_VISIBLE_DEVICES=0` and
`CUDA_VISIBLE_DEVICES=0`, and verifies that HIP logical 0 has the expected PCI
address, that only one GPU is visible, and that its architecture is `gfx950`.
An inaccessible device or ambiguous mapping fails rather than silently measuring
another GPU. Setup uses empty GPU visibility and a bounded CPU worker budget.

The gate uses the image's existing [ROCm SMI bindings and read-only APIs](https://github.com/ROCm/rocm_smi_lib/blob/master/python_smi_tools/rsmiBindings.py).
It requires zero utilization and **no target-GPU KFD processes** for 30 seconds,
sampled every five seconds, with a ten-minute wait limit. This is deliberately
more conservative than a memory-threshold gate. Missing bindings, unsupported
queries, malformed output, and process-list races fail closed. A process using a
different GPU does not by itself block the selected GPU.

During the workload, five-second samples compare target-GPU KFD PIDs against
`docker top` host PIDs from this run's container. If an external target-GPU process
appears, or monitoring fails, this run fails and stops **only its ownership-label
verified container**. External PIDs are never signaled, killed, paused, or cleaned
up. The full run is invalidated; it is not automatically resumed or counted as a
clean performance result. Re-dispatch after the external work finishes. Sampling
is not an atomic reservation, so short overlaps can occur between samples; inspect
the monitor log and do not interpret a collided run as a regression or speedup.

The job has a 120-minute cap, setup a 30-minute cap, and the perf command a
60-minute timeout with a 30-second kill grace. A separate 105-minute container
lifetime bounds orphaned work if the runner is lost; normal failure/cancellation
cleanup checks the ownership label before removing the unique run/attempt
container. No shared ROCm process-cleanup script, host PID kill, Docker socket
mount, privileged container, host PID namespace, or host-network mode is used.
Hard runner failures can still prevent artifact upload and normal cleanup.

## Logs, caches, and interpreting results

The isolated artifact is `conv3d-perf-log-<run-id>-<attempt>` and is uploaded with
`always()` on success, failure, or cancellation when the runner is reachable.
It never uses the `standard-test-log-*` prefix or feeds the standard benchmark
aggregation. It includes dispatch metadata, image digest, setup output and wall
status, toolchain/device/cache metadata, the complete perf output,
`gpu-monitor.jsonl`, and overall status/wall seconds. Registry credentials and
compiled caches are not uploaded.

FlyDSL, Triton, and Aiter JIT caches start cold in run-local container paths.
There is no shared cache restore/save and no deletion of another task's caches.
Reported setup/perf wall includes import, preparation, reference calculation,
JIT compilation, correctness, and timing; it is **not** kernel latency. Compare
kernel table values only from clean runs on the same physical GPU, commit/toolchain
provenance, and cache policy. Cold and warm wall times are not interchangeable.
The guard status must contain `valid_perf: true`; an absent status or failed
monitor is not a successful benchmark.

## Coverage boundaries and large-address validation

Compact default CI is not complete Wan/Qwen shape coverage, exhaustive tuning,
full layout/dtype combinations, or an end-to-end model test. Performance mode
preserves the full existing matrix, not every possible input extent or kernel
configuration. Neither mode guarantees exercising the `BIG_IN`/`BIG_OUT`
large-address branches or their exact threshold boundaries.

Changes affecting address width, large offsets, or those branches need a separately
authorized boundary-validation plan: cover each relevant threshold just below,
at, and above the boundary, input and output offsets independently, representative
layouts/padding/stride, numerical comparison, allocation feasibility, and watchdogs
on an idle target GPU. No dedicated large-address CLI option is claimed here;
confirm current kernel thresholds and existing test interfaces before designing
the cases. Do not allocate huge tensors implicitly in standard CI or the default
manual matrix.

## Measurements

CPU/static checks and real MI35X runs must be recorded separately. No GPU wall
budget, speedup, or coverage result is claimed by this document. Maintainers should
add measured default-CI/perf wall, tested commit, GPU identity, toolchain, cache
state, exact command, and clean monitor provenance after authorized validation.

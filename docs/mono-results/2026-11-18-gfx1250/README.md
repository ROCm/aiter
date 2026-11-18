# A03 / A09 standalone P256 results: gfx1250, 2026-11-18

GPU correctness and launch checks pass after a small import fix. Standalone
performance acceptance fails: eight configurations regress. The four rows=96
configurations have small differences or overlapping distributions and remain
inconclusive. No model or fused-kernel speedup is claimed.

## Code and environment

- Handoff: `d498cfa2e2d7d16e98ce34d16e88c6125a293e2c`.
- Pinned test base: `efa58c2ccd4525a8287c4f886945fc48ed86a192`.
- Tested code: `d5930aff35d67fa0a2e74ba425909747f3ccef9f`. This commit is retained in this branch's history.
- The fix adds three lines in `aiter/__init__.py`. Triton-only mode must expose
  `dtypes` and initialize the path used by its existing helper imports.
- Native excerpts, kernels, unit tests and benchmark code were not changed.
- GPU 0: gfx1250, wave32, 256 CU, 442368 MiB; host `heliosr-1b114-a07-3`.
- PyTorch `2.14.0+rocm10.3.0a20261103-bkc.20261105`, HIP `7.18.26435`, Triton `3.9.0`.
- Driver `7.1.0.31300009`; image digest `sha256:6a40615d965ec6088217e5682a866875e2c218365c7f135ecb83c4907d6bf2dd`.
- Clock mode stayed `auto`. Sampled sclk was 2222–2256 MHz; mclk was 1900 MHz.
  Socket power cap stayed 2500 W. No framework packages or clock settings changed.

See [environment.json](environment.json) and
[performance-settings.json](performance-settings.json) for the environment.
The test used an isolated checkout and container. It preserved the original
checkout and other jobs. The test container was stopped after validation.

## Correctness

All **36 tests passed, with zero failures and zero skips**: 20 A03 tests,
15 A09 tests and one shared API-contract test. Both ops passed 20 graph replays
with changed inputs. Native/PS byte comparisons, input/weight mutation checks
and output guards passed. The test's separate FP64 K oracle kept its original
BF16 tolerance.

A03 includes shared `[96,1792]` storage, Q1280/KV512 views, and row stride 1792.
A09 includes `[96,2048]`. The two scale recipes remain distinct.
All 12 frozen native hashes and all 12 offline compile variants passed.

Evidence: [correctness XML](correctness-final.xml),
[correctness log](logs/correctness-final.log),
[native hashes](native-hashes.json), and [compile log](logs/compile-fixed.log).

The pinned code first failed test collection because Triton-only mode did not
export `aiter.dtypes`. Adding that export exposed a helper import path error.
Both failures occurred before tensor allocation or kernel launch, so no tensor
inputs or outputs existed. The original logs are
[correctness.log](logs/correctness.log) and
[correctness-fix1.log](logs/correctness-fix1.log). After the fix, the full suite
passed again, including a fresh runtime JIT cache and a profiler run.

An extra existing no-GPU import test failed because this ROCm runtime aborts on
`ROCR_VISIBLE_DEVICES=-1`. The equivalent import check with empty visibility
strings passed. This extra check is separate from the 36-test acceptance suite.
See [the failure](logs/import-regression.log) and
[the compatible check](logs/import-smoke-compatible.log).

## Performance

The table uses [performance-final.json](performance-final.json). Each value is
in microseconds. Each sample is the mean latency per kernel launch in one trial.
P10/P90 use inclusive deciles. A ratio above one means PS is slower.

| Op | Input storage shape | Cache | Native median [P10, P90], us | PS median [P10, P90], us | PS/native | Result |
|---|---|---|---:|---:|---:|---|
| A03 | [96, 1792] | hot | 1.579 [1.570, 1.618] | 1.564 [1.560, 1.573] | 0.9907 | inconclusive |
| A03 | [96, 1792] | rotating | 1.912 [1.910, 1.921] | 1.898 [1.891, 1.913] | 0.9925 | inconclusive |
| A03 | [257, 1792] | hot | 1.692 [1.670, 1.703] | 2.065 [1.960, 2.202] | 1.2206 | regression |
| A03 | [257, 1792] | rotating | 2.054 [2.031, 2.060] | 2.478 [2.322, 2.539] | 1.2065 | regression |
| A03 | [513, 1792] | hot | 1.838 [1.831, 1.846] | 2.690 [2.588, 2.926] | 1.4633 | regression |
| A03 | [513, 1792] | rotating | 2.274 [2.266, 2.283] | 3.121 [3.020, 3.170] | 1.3724 | regression |
| A09 | [96, 2048] | hot | 1.412 [1.392, 1.469] | 1.452 [1.441, 1.510] | 1.0281 | inconclusive |
| A09 | [96, 2048] | rotating | 1.714 [1.692, 1.728] | 1.713 [1.706, 1.741] | 0.9995 | inconclusive |
| A09 | [257, 2048] | hot | 1.512 [1.501, 1.516] | 2.064 [2.022, 2.152] | 1.3649 | regression |
| A09 | [257, 2048] | rotating | 2.038 [2.033, 2.044] | 2.445 [2.438, 2.451] | 1.1999 | regression |
| A09 | [513, 2048] | hot | 1.668 [1.666, 1.684] | 3.078 [3.061, 3.197] | 1.8448 | regression |
| A09 | [513, 2048] | rotating | 2.754 [2.750, 2.758] | 3.648 [3.637, 3.655] | 1.3247 | regression |

A regression requires the PS median increase to exceed the native P90−P10
spread and the PS P10 to exceed native P90. Smaller or overlapping differences
remain inconclusive. All eight large-shape regressions had the same direction
in all three complete runs. A09 rows=96 hot was 2.6%–4.1% slower by median across
those runs, but its final intervals overlap.

The unmodified benchmark used `RotatingBufferBench`, seed 451, shuffle seed 0,
the same GPU and input pool for both providers, and alternating provider order.
Each provider ran 15 trials per configuration. Each graph contained 1000 kernel
calls and each trial used 10 timed graph replays. Capture, allocation, compile
and warmup were outside timing. Hot mode used one input set. Rotating mode used
the gfx1250 default 196 MiB pool target. JSON records the actual pool lengths.
No profiler was enabled for any timing run.

Three complete matrices are retained:

1. [performance.json](performance.json): the first run had another GPU job near
   its start. It is not the source of the final table.
2. [performance-repeat.json](performance-repeat.json): a repeat after an idle
   interval, using the guide's original command.
3. [performance-final.json](performance-final.json): the final run after five
   seconds with no KFD processes and GPU busy=0. An external wrapper recorded
   each case's start/end time without changing the benchmark implementation.

During all final case windows, 100 ms KFD process samples found only this
benchmark. This is sampling evidence; it cannot exclude shorter activity.
See [case windows](performance-final-windows.jsonl),
[load audit](performance-load-audit.json) and [performance logs](logs/).
No best-of-run selection or mixing was used.

## Actual launches and JIT

The installed rocprofv3 captured 92 PS dispatches in the complete test suite.
All used work-item grid `(2048,16,1)` and workgroup `(128,1,1)`: **16×16 TGs,
128 threads per TG, four wave32 waves**. See [launch-summary.json](launch-summary.json).

The separate [target-launches.csv](target-launches.csv) contains 12 actual
wrapper launches: native and PS for both ops at rows=96/257/513.
[target-resources.json](target-resources.json) maps each launch to its real
JIT code hash, artifact hashes, input strides and resource use.

| Op | Native TG counts for rows=96/257/513 | PS TG count | Native / PS VGPR | Scratch bytes |
|---|---|---:|---:|---:|
| A03 | 96 / 257 / 513 | 256 | 83 / 83 | 0 / 0 |
| A09 | 192 / 514 / 1026 | 256 | 47 / 51 | 0 / 0 |

A03 PS workers process rows at stride 256. At rows=257/513, the busiest worker
processes 2/3 rows. A09 uses 1024-element tasks; at rows=257/513, its busiest
worker processes 3/5 tasks. Later tasks in the same worker execute serially.
A09 also adds loop control: its actual ISA has two conditional branches, versus
zero in native. No target variant spills to scratch. These confirmed scheduling
and ISA differences are consistent with the regressions. Hardware counters were
not collected to assign an exact fraction of the slowdown to each cause.

## Scope and retained artifacts

GPU sanitizer validation was not established. The image has compiler ASAN
support, but no matched separate ASAN HIP/HSA runtime was found.
Concurrent residency of all 256 TGs was not measured.
Full mono DEV/HT, s104/s105 preservation, inter-op joins, device-call interfaces
and model integration remain unimplemented and unverified.

This directory contains the portable performance data and small supporting
records. CSV newlines and one log line's trailing whitespace were normalized
for Git. The three performance JSON files are unchanged copies. Full runtime TTIR, TTGIR, LLVM IR, ISA, code objects, metadata and raw
profiler traces remain in the validation result archive on the test host:

- Directory: `/home/weizhu15/workspace/mono-acceptance-20261118`.
- Archive: `/home/weizhu15/workspace/mono-acceptance-20261118-results.tar.gz`.
- Archive SHA256: `ec38c863944cc59bef4980f5409b3ba05aac8c33defde8bc89b0e9b9dd40808a`.

The archive predates this publication commit. Its tested checkout was clean at
`d5930aff35d67fa0a2e74ba425909747f3ccef9f`. The table values here come from those saved measurements;
no benchmark was rerun just to publish the data.

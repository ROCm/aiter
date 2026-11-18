# A03/A09 standalone P256 validation

## Four-task PS experiment: 2026-11-18

This section records the current user-requested four-task experiment. The
sections below record the earlier one-task implementation and its handoff.
The older static resource limits describe that earlier selection. The new
experiment deliberately increases per-wave work and does not meet those old
VGPR/code-size limits. It is not a successful performance acceptance.

Tested code: `25c1092044557ab3e0c5f90374cb2e4b06cc6142`. Baseline PS1: `6f85fc6d8dbc4c01cf4d40a6e48f1990c7952056`.
A03 handles four adjacent rows per TG iteration. A09 handles four adjacent
1024-element tasks (two rows at width 2048). Both keep 256 TGs and 128 threads
per TG. The actual grid is one-dimensional, with work-item grid `(32768,1,1)`.

All 52 GPU tests passed with zero skips. These retain all original 45 cases
and add seven A03 cases at the new 1024-row loop boundary. Frozen native hashes
and all 12 compile variants passed. The first A03 attempt failed six byte
checks because its automatic reduction layout changed. Explicit Gluon layouts
fixed these cases; no native source or tolerance changed. Non-PS A03 is unchanged.

Actual Graph tracing recorded 48 dispatches for each of A03 PS, A09 FP8 PS and
A09 QAT PS. All used 256 TGs x 128 threads. Actual ISA issues four A09 loads
before the first load-result wait. A03 issues four rows of Q/K loads before
that wait. PS4 VGPR/shared/scratch are A03 168/64/0, FP8 76/0/0 and QAT 60/0/0.
These resource fields do not measure residency.

Timing used the shared GPU lock and idle checks, without profiling. Each
selected case has 15 trials and no foreign GPU PID in its 100 ms process
samples. Two contaminated paired cases were discarded and repeated in full.
Native/PS4 use one input pool and alternating order. Values are microseconds.

| Op | Rows | Cache | Native median / P10 / P90 (us) | PS4 median / P10 / P90 (us) | PS4/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| a03 | 96 | hot | 1.556748 / 1.553161 / 1.562145 | 2.892640 / 2.878984 / 2.896413 | 1.858130 | regression |
| a03 | 96 | rotating | 1.918209 / 1.915257 / 1.920165 | 3.276872 / 3.269959 / 3.365023 | 1.708298 | regression |
| a03 | 257 | hot | 1.685472 / 1.661242 / 1.691207 | 2.963542 / 2.950575 / 2.971481 | 1.758286 | regression |
| a03 | 257 | rotating | 2.036438 / 2.026278 / 2.053536 | 3.508891 / 3.504066 / 3.515869 | 1.723053 | regression |
| a03 | 513 | hot | 1.840280 / 1.787212 / 1.846371 | 3.001283 / 2.998181 / 3.013520 | 1.630883 | regression |
| a03 | 513 | rotating | 2.242006 / 2.234880 / 2.245511 | 3.675697 / 3.672838 / 3.678854 | 1.639468 | regression |
| a09 | 96 | hot | 1.399276 / 1.392892 / 1.408441 | 1.444631 / 1.323942 / 1.454359 | 1.032414 | inconclusive |
| a09 | 96 | rotating | 1.732855 / 1.725823 / 1.741467 | 1.719751 / 1.717479 / 1.722751 | 0.992438 | inconclusive |
| a09 | 257 | hot | 1.538324 / 1.492725 / 1.540745 | 1.507490 / 1.500089 / 1.512898 | 0.979956 | inconclusive |
| a09 | 257 | rotating | 2.002499 / 1.984849 / 2.009294 | 1.739041 / 1.737788 / 1.740148 | 0.868435 | improvement |
| a09 | 513 | hot | 1.637156 / 1.631263 / 1.647279 | 1.452247 / 1.431756 / 1.456366 | 0.887055 | improvement |
| a09 | 513 | rotating | 2.707783 / 2.699110 / 2.715536 | 2.026919 / 2.015178 / 2.044917 | 0.748553 | improvement |
| a09-qat | 96 | hot | 1.257856 / 1.255068 / 1.258948 | 1.302819 / 1.301787 / 1.307620 | 1.035746 | regression |
| a09-qat | 96 | rotating | 1.614831 / 1.613323 / 1.616599 | 1.679771 / 1.678194 / 1.681111 | 1.040215 | regression |
| a09-qat | 257 | hot | 1.314621 / 1.313751 / 1.316103 | 1.341153 / 1.339088 / 1.343135 | 1.020182 | regression |
| a09-qat | 257 | rotating | 1.692847 / 1.691781 / 1.693656 | 1.754295 / 1.753258 / 1.756638 | 1.036299 | regression |
| a09-qat | 513 | hot | 1.427337 / 1.425034 / 1.431802 | 1.728108 / 1.712330 / 1.747755 | 1.210722 | regression |
| a09-qat | 513 | rotating | 1.808997 / 1.805118 / 1.811625 | 2.019332 / 1.999071 / 2.059459 | 1.116272 | regression |

A03 regresses. A09 FP8 improves some cases but does not pass all six cases.
A09 QAT remains slower than native. The separate three-provider comparison
shows that four-task PS helps A09 at 257/513 rows relative to old PS. That gain
does not override a native regression. A03 gains no benefit from this mapping.
For A03, useful TG counts at 96/257/513 rows fall to 24/65/129; registers and
per-TG work increase. A09 issues more independent loads and reduces loop trips.
PS4 ATT, per-WGP residency and exact residual cycle attribution remain untested.

Raw evidence and full before/after statistics: `/home/weizhu15/workspace/mono-ps4-20261118`.
See `REPORT.zh.md`, `performance-accepted.json`, `performance-selections.json`,
`performance-load-audit.json`, `trace-jit.jsonl`, `jit-review.json`, and `logs/`.
No full mono ABI, DEV/HT, reserved-register, join or model integration was added.

Use this guide only after the external runner's offline gate and Claude's
independent local review. Neither certifies GPU correctness or performance.
The [review TODO](mono-persistent-review-todo.md) records the selected exact
power-of-two reciprocal change, its static cost and the unresolved request cost.

## Follow-up: A03 timing and A09 diagnosis

The user requested A03 retesting and further A09 analysis after the first
acceptance. All GPU work used the same fixed delivery `5446016c` and image.
Production kernel, wrapper, benchmark, tests and native source did not change.
The diagnostic wrappers and raw evidence stay outside the checkout.

A03 passed all 20 selected tests again, with zero skips. Its Graph trace passed;
all 24 PS dispatches used grid `(32768,1,1)` and workgroup `(128,1,1)`.
The original benchmark then ran all six A03 cases with 15 alternating trials.
The 101 process samples during the case windows saw only this benchmark.

| Rows | Cache | Native median / P10 / P90 (us) | PS median / P10 / P90 (us) | PS/native | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| 96 | hot | 1.564855 / 1.554226 / 1.589715 | 1.585209 / 1.567499 / 1.604769 | 1.013007 | inconclusive |
| 96 | rotating | 1.919946 / 1.917698 / 1.925404 | 1.916001 / 1.912472 / 1.919784 | 0.997945 | inconclusive |
| 257 | hot | 1.667741 / 1.656719 / 1.701918 | 1.920399 / 1.917161 / 1.967340 | 1.151497 | regression |
| 257 | rotating | 2.043015 / 2.025434 / 2.046702 | 2.349874 / 2.278817 / 2.467875 | 1.150199 | regression |
| 513 | hot | 1.821483 / 1.816528 / 1.851258 | 2.701228 / 2.663911 / 2.783161 | 1.482983 | regression |
| 513 | rotating | 2.252620 / 2.244447 / 2.263607 | 3.190436 / 3.169208 / 3.216171 | 1.416322 | regression |

A03 performance fails: four regressions and two inconclusive rows. These are
current-delivery measurements, not the historical two-dimensional-grid data.

A09 diagnostics kept the task math fixed. The external P256 replica has
instruction-identical ISA to selected P256. P512, P1024 and P1026 retain the
same instruction sequence and resources; only four stride immediates change.
At `[513,2048]`, all cases below had no observed concurrent GPU process.
These values are median microseconds from 15 trials without profiling.

| Diagnostic | FP8 hot | QAT hot | FP8 rotating | QAT rotating |
| --- | ---: | ---: | ---: | ---: |
| Frozen native | 1.666988 | 1.420895 | 2.766726 | 1.778635 |
| Selected non-PS | 1.628382 | 1.351147 | 2.778071 | 1.765415 |
| P256 replica | 3.046068 | 2.559087 | 3.503421 | 3.186402 |
| Fixed descriptor, P256 | 2.918942 | 2.560750 | 3.363139 | 3.181355 |
| Same loop body, P512 | 2.028084 | 1.828533 | 3.052226 | 2.261622 |
| Same loop body, P1024 | 1.674411 | 1.528544 | 2.588580 | 1.791406 |
| Same loop body, P1026 | 1.649970 | 1.359507 | 2.799002 | 1.777249 |

Raw JSON has full precision and P10/P90. Removing descriptor reuse/XACK waits
at fixed P256 barely changes QAT and only partly helps FP8. Thus those waits
are not the main cause. The fixed-descriptor diagnostic also changes address
arithmetic and is limited to inputs below 2 GiB; it is not a production fix.

An actual QAT ATT trace at SE0, target CU/WGP 1, SIMD0 shows four overlapping
waves for native, one wave for P256, and four for P1026. In this sample, native
and P1026 issue four loads across four wave contexts. P256 issues four loads
serially in one context. Its wave lasts 3655 GFX cycles, with 1682 cycles at
load-result waits and 233 at XACK waits. This is local trace evidence, not a
whole-GPU occupancy percentage or a substitute for benchmark timing.
Both the timed controls and trace identify the main mechanism: the current
P256 loop exposes fewer independent waves and does not pipeline the next tile's
load ahead of the current tile's work. Hardware workgroup scheduling can overlap
independent wave contexts; this software loop cannot provide the same overlap.
Equal task counts do not imply equal latency hiding. Exact cache-level and
whole-kernel cycle attribution remain open.

Tail work cannot explain the clean boundary regression. At exactly 512 rows, every
P256 worker has four tasks, yet QAT hot is 2.392025 us versus native 1.419148 us.
There is already a large regression without a fifth-iteration tail.

The external diagnostic wrappers passed 64 strict inherited assertion cases,
plus four exact-boundary cases, including special BF16 values and Graph replay.
One of 12 main diagnostic cases and four of eight boundary cases overlapped
other GPU jobs. They are marked unusable; no favorable samples were selected
from them. SQ_WAVES matched 1024/4104 for P256/native at 513 rows. Four other
SQ instruction/wait/cycle counters returned zero on all variants and are unusable.
The final ATT sample had no other GPU process in adjacent 100 ms observations.

Evidence root: `/home/weizhu15/workspace/mono-followup-5446016c-20261118/`.
See `REPORT.zh.md`, `a03-performance.json`, `diagnostic-performance.json`,
`per-case-load-audit.json`, `review/worker-count-isa.diff`, and
`diagnostic-att-clean-summary.json`. `att-wave-timeline.svg` plots the actual
wave trace. No production fix or upstream push was made. The original A09
acceptance result remains failed.

## MI400 result: 2026-11-18

Delivery `5446016c426910d96e2b13996586dfe875863da0` was tested in a clean,
independent checkout on `heliosr-1b114-a07-3`. The machine had recovered from
the earlier driver block. No kernel, wrapper, test, benchmark or frozen native
source changed. All 12 frozen symbol hashes and 12 offline builds passed.

The numerical suite passed once without profiling: **45 passed, zero skips**.
This includes 20 A03 cases and 25 A09 cases. Native/PS bytes, input/weight
preservation, guards, special BF16 fixtures and all Graph checks passed.
A03 includes shared `[96,1792]` storage with Q1280/KV512 views and row stride
1792. A09 includes `[96,2048]`, `[513,2048]` and `[513,2080]`.

Both A09 Graph nodes then passed under rocprofv3. All 96 PS dispatches
(48 FP8, 48 QAT) used grid `(32768,1,1)` and workgroup `(128,1,1)`:
**256 TGs x 128 threads**. Actual wrapper JIT IR, ISA, code objects and metadata
were saved. Actual PS FP8 is 808 B / 21 VGPR; QAT is 844 B / 20 VGPR.
Both use 68 SGPR, shared=0 and scratch=0. These differ from the earlier local
binder numbers below, but pass the fixed resource gates. Vector widths, masks,
loop dependencies and FP modes passed review. No general division, LDS,
scratch or memory-mask EXEC branch was found. Non-PS has no task loop.
The raw profiler VGPR/SGPR fields are retained separately from compiler counts.

The one required timing matrix ran without a profiler at 12:01:37--12:01:52 UTC.
Each row has 15 alternating native/PS trials on the same input pool. Each trial
uses 10 replays of a 1000-kernel Graph. Hot pool length is 1; rotating pool
lengths are 523, 196 and 98 for 96, 257 and 513 rows. Each sample is mean
latency per launch. Decisions use the rules in section 3, before rounding.

| Path | Rows | Input reuse | Native median / P10 / P90 (us) | PS median / P10 / P90 (us) | PS/native | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| FP8 | 96 | hot | 1.386007 / 1.374745 / 1.414506 | 1.403738 / 1.387628 / 1.428432 | 1.012792 | inconclusive |
| FP8 | 96 | rotating | 1.717423 / 1.692109 / 1.730208 | 1.688139 / 1.679708 / 1.709433 | 0.982949 | inconclusive |
| FP8 | 257 | hot | 1.480265 / 1.474389 / 1.548339 | 2.034931 / 1.946374 / 2.066170 | 1.374707 | regression |
| FP8 | 257 | rotating | 2.087570 / 2.077553 / 2.207312 | 2.492392 / 2.398856 / 2.518862 | 1.193920 | regression |
| FP8 | 513 | hot | 1.663527 / 1.658817 / 1.695893 | 3.053400 / 2.986177 / 3.085353 | 1.835498 | regression |
| FP8 | 513 | rotating | 2.738817 / 2.608372 / 2.744959 | 3.523660 / 3.503407 / 3.546427 | 1.286563 | regression |
| QAT | 96 | hot | 1.252347 / 1.237009 / 1.256956 | 1.227174 / 1.225645 / 1.228566 | 0.979899 | improvement |
| QAT | 96 | rotating | 1.603641 / 1.601850 / 1.604787 | 1.581236 / 1.578911 / 1.582523 | 0.986028 | improvement |
| QAT | 257 | hot | 1.315974 / 1.312891 / 1.571942 | 1.714587 / 1.689177 / 1.753119 | 1.302903 | regression |
| QAT | 257 | rotating | 1.692194 / 1.689980 / 1.694664 | 2.343633 / 2.328332 / 2.364742 | 1.384967 | regression |
| QAT | 513 | hot | 1.421809 / 1.418243 / 1.425017 | 2.465512 / 2.430558 / 2.489546 | 1.734067 | regression |
| QAT | 513 | rotating | 1.793473 / 1.792476 / 1.795698 | 3.187413 / 3.182001 / 3.196832 | 1.777229 | regression |

**FP8 fails:** four regressions and two inconclusive rows.
**QAT fails:** four regressions and two improvements. Neither path passes all
six rows. No automatic rerun, tuning or ablation matrix followed that run.
The later user-requested follow-up above adds current A03 timing and A09
diagnostics. Historical A03 timings do not certify either run. No upstream push
was made because the requested success gate failed.

The GPU was ordinal 0, gfx1250/wave32, with 256 CUs. The image digest was
`sha256:6a40615d965ec6088217e5682a866875e2c218365c7f135ecb83c4907d6bf2dd`.
It used PyTorch `2.14.0+rocm10.3.0a20261103-bkc.20261105`, HIP `7.18.26435`,
Triton `3.9.0+gitb7196d97` and amdgpu `7.1.0.31300009`.
Packages, clocks and power settings were preserved. Performance mode was AUTO.
The 100 ms process log observed only the benchmark GPU process during timing.
Seven metric samples at about 2.177 s intervals recorded GFX 2213--2240 MHz,
HBM 1900 MHz and socket power 1038--1177 W. No sampled thermal or power
throttle was active. These samples cannot exclude shorter unobserved events.

The measured regression grows when P256 workers must repeat tasks. There are
192, 514 and 1026 tiles at the tested row counts. PS workers execute up to
one, three and five serial iterations; native dispatches one TG per tile.
This is a concrete scheduling and loop-cost difference. The timing pattern is
consistent with that cost, but does not isolate its cycle contribution.
Cache traffic, scale-store request combining, residency and runtime retry state
remain unmeasured. Smaller code and fewer VGPRs do not establish a speedup.

Full logs, JSON/XML, maps, JIT artifacts and private review are on the runner at
`/home/weizhu15/workspace/mono-acceptance-5446016c-resumed/`.
Start with `REPORT.zh.md`, `performance.json`, `performance-decisions.json`,
`jit-launch-gate.json`, `load-audit.json` and `logs/`. Memory-sanitizer coverage
and full mono ABI/DEV/HT/reserved-register/join/model work remain unverified.

## Source and local status

Selected code commit is `0d54cbf3e7b16ac7563c34df6e5d744945f6a495`.
This is the exact kernel and test source approved by Claude Opus 5.5 in local
run `20261118-155450-63f32955`. The fixed outer acceptance passed, exit 0.
The published delivery adds only guide updates after that code commit. Use
`Wanzizhu/aiter`, branch `mono-dev`, and the exact delivery SHA supplied in the
handoff. Read this guide from that checkout and keep the delivery checkout for
all tests; do not reset to an older test commit or apply historical patches.
Record HEAD, status, dirty diff and source hashes. Check the two selected-source
hashes below and the frozen native manifest before execution.

If using the offline patch bundle instead, apply its single patch to a clean
independent checkout at public base
`a8a913883d7be9837c2f46f2e92de8fc701bac26`. Verify that bundle's manifest;
HEAD remains at the public base after patch application. The offline bundle and
the published delivery have identical reviewed kernel/test bytes; later guide
updates can differ. Preserve framework packages and unrelated tasks.
The isolated review clone omitted one unrelated gitlink for runner support;
both delivery methods preserve production submodule entries and declarations.

| Reference | Fixed source |
| --- | --- |
| Public comparison base | ROCm/aiter `a8a913883d7be9837c2f46f2e92de8fc701bac26` |
| A09 original native | ATOM `10b99a085ba46e5b3a125f0114cec7c71e401552` |
| A03, unchanged | `7108dbec079595026f76b7b1b8e082fe1669796d` |
| Frozen excerpts / hashes | `op_tests/triton_tests/utils/mono_native.py` and `mono_native_manifest.json` |

Selected code SHA256:

```text
0393295fc04b699224174e77ed5ede74f28d46ea588afed986f48cd87c2a2849  aiter/ops/triton/_gluon_kernels/gfx1250/quant/fp8_group32.py
6e64b2b37adc63e55a9d5306ae5c5d9a6ca9d5b18efa436374490a5908c71e60  op_tests/triton_tests/quant/test_mono_persistent.py
```

The worker completed 12 offline native/candidate builds and checked all 12
frozen symbol hashes. The collection-only CPU import harness found the same
45 node IDs. All original assertions remain. Black checks on individual files,
Ruff and diff whitespace checks passed. The worker also built 22 real-binder
variants: 12 selected, six native and four previous-code comparisons. These
cover FP8/QAT, PS/non-PS and `[1,32]`, `[513,2048]`, `[513,2080]` tails. QAT
uses the wrapper's actual `S=None` specialization.

The unchanged external CPU checker was copied into private workspace artifacts
and run there. It checked 65,536 BF16 bit patterns against 165 scale codes:
10,813,440 pairs, with zero post-clamp FP8 or QAT byte mismatches. Pre-clamp
bits match except that NaNs are compared as both NaN. This checks CPU algebra
and reference semantics; it is not GPU numerical proof. At the local handoff,
the new GPU fixtures had not run. They passed in the hardware result above.
Full worker logs, IR/ISA, code objects, source manifest, patch and dependency
review are in private `build/a09-local/`.

The outer runner's fixed command remains:

```text
/home/weizhu15/workspace/aiter-mono-tools/bin/python /home/weizhu15/workspace/mono-a09-finalize-20261118/acceptance.py
```

It checks 12 builds, frozen hashes, Black/Ruff, the exact 45 node IDs, four real
binder variants, exhaustive CPU arithmetic and ISA/resource selection gates.
The outer runner executed this command successfully. Claude independently
reviewed the recorded result and the exact candidate, and returned `approve`
with no blockers. Its two advisory items are handled in this guide: special-value
failure triage below and line wrapping. No reviewed kernel or test bytes changed
after approval. This approval covers local readiness only. The separate
hardware result above records GPU correctness, actual JIT/launch and the
failed performance gate.
Never use the CPU collection harness to run numerical tests or bypass the GPU
gate below.

The [historical report](mono-results/2026-11-18-gfx1250/README.md) covers
`d5930aff35d67fa0a2e74ba425909747f3ccef9f`, before this implementation. Its
36 passes, eight regressions and four inconclusive rows are historical only.

## 1. Prepare the supplied checkout

Use the machine's existing compatible gfx1250 ROCm environment. Set
MONO_RESULTS to the authorized artifact directory before these commands.
Preserve its packages, other tasks, clocks and power settings. Stop on a failed
command and retain the full error. Do not repair the environment by installing
frameworks as part of this handoff.

```bash
set -euo pipefail
: "${MONO_RESULTS:?Set the authorized artifact directory}"
mkdir -p "$MONO_RESULTS"
export AITER_TRITON_ONLY=1
export TRITON_DEFAULT_FP_FUSION=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$MONO_RESULTS/commit.txt"
git status --porcelain > "$MONO_RESULTS/git-status.txt"
git diff --binary > "$MONO_RESULTS/source.diff"
sha256sum aiter/ops/triton/_gluon_kernels/gfx1250/quant/fp8_group32.py \
  aiter/ops/triton/quant/fp8_group32.py \
  op_tests/triton_tests/quant/test_mono_persistent.py \
  op_tests/op_benchmarks/triton/bench_mono_persistent.py \
  op_tests/triton_tests/utils/mono_native.py \
  op_tests/triton_tests/utils/mono_native_manifest.json \
  > "$MONO_RESULTS/source-sha256.txt"
python - <<'PY' | tee "$MONO_RESULTS/environment.txt"
import platform
import torch
import triton
from triton.runtime import driver
assert torch.version.hip is not None and torch.cuda.is_available()
target = driver.active.get_current_target()
assert target.arch in ("gfx1250", "gfx1250-strict") and target.warp_size == 32
print(platform.platform())
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
print("target", target)
print(torch.cuda.get_device_properties(torch.cuda.current_device()))
PY
```

Check the selected hashes above and the frozen native manifest. For a patch
bundle, also check its delivery manifest. Record image digest,
host, selected GPU UUID/ordinal, driver, clock/power settings and other GPU
workloads. Retain load/clock observations with timestamps and sampling interval.
Runtime retry state is unknown unless the machine provides evidence for it.
Do not infer it from MODE.REPLAY_MODE in ISA.

## 2. Correctness and actual launch evidence first

The offline compiler checks frozen symbols before compiling. Its metadata and
ISA do not replace the real GPU wrapper specializations saved by pytest.
Run all 45 numerical tests once without a profiler. Then trace only the two
A09 Graph nodes. A profiler failure must not discard the numerical result.
Profiler durations are not timing results.

```bash
python -m op_tests.compile_mono_persistent --output "$MONO_RESULTS/compile" \
  2>&1 | tee "$MONO_RESULTS/compile.log"
export TRITON_CACHE_DIR="$MONO_RESULTS/jit-cache"
export TRITON_DUMP_DIR="$MONO_RESULTS/jit-dump"
export TRITON_KERNEL_DUMP=1
python -m pytest \
  op_tests/triton_tests/quant/test_mono_persistent.py -ra -v \
  --junitxml="$MONO_RESULTS/correctness.xml" \
  2>&1 | tee "$MONO_RESULTS/correctness.log"
python - "$MONO_RESULTS/correctness.xml" <<'PYXML'
import sys
import xml.etree.ElementTree as ET
cases = list(ET.parse(sys.argv[1]).iter("testcase"))
assert len(cases) == 45, f"Expected 45 tests, found {len(cases)}"
assert all(c.find("skipped") is None and c.find("failure") is None
           and c.find("error") is None for c in cases), "Correctness gate failed"
PYXML
```

After the numerical gate passes, collect launch evidence. The selected nodes
cover FP8 and QAT at 257 and 513 rows. Reuse the JIT cache from the full run.

```bash
if command -v rocprofv3 >/dev/null 2>&1; then
  if rocprofv3 --kernel-trace --output-format csv \
    --output-directory "$MONO_RESULTS/launch-trace" -- \
    python -m pytest \
    'op_tests/triton_tests/quant/test_mono_persistent.py::test_graph_replay_changes_inputs[a09]' \
    'op_tests/triton_tests/quant/test_mono_persistent.py::test_graph_replay_changes_inputs[a09-five-iterations]' \
    -ra -v --junitxml="$MONO_RESULTS/launch-correctness.xml" \
    > "$MONO_RESULTS/launch.log" 2>&1; then
    printf 'Trace command completed; inspect grid and workgroup before timing.\n'
  else
    printf 'Launch gate pending; keep correctness.xml and inspect launch.log.\n'
    exit 1
  fi
else
  printf 'Launch gate pending: rocprofv3 is unavailable.\n'
  exit 1
fi
```

Coverage retains all existing nodes and assertions. A09 retains native and
independent finite-value FP8/E8M0/QAT oracles, XOR-prefilled outputs, guards and
unchanged inputs. The existing scale-boundary node also builds integer bits
on GPU for `[65536,32]` (all BF16 patterns, 4 MiB) and `[8193,32]` (mixed groups,
512 KiB plus 64 B). The fixtures cover normal/subnormal values, signed zeros,
finite extremes, Inf and signed quiet/signalling NaNs. Mixed groups rotate
values through all lane slices and waves; finite groups force underflow with
large maxima. Both fixtures compare PS and non-PS FP8/E8M0/QAT bytes against
frozen native and check input/output guards. Nonfinite data never enters the
finite-only oracle. There is no relaxed NaN or byte tolerance. Shapes also
include `[513,2048]` with five iterations and `[513,2080]` with one valid group
in the last task. Scale ownership spans all four waves.
The two A09 Graph nodes check both FP8 and QAT after each of 20 input changes.
A03 tests and its existing FP64 K tolerance remain unchanged.

Pass requires 45 passes and zero skips, byte/guard mismatches or GPU errors.
For a failure, retain node ID, seed, input, outputs, source manifest and complete
log. Stop before timing. Do not loosen numerical checks or change frozen native.
GPU memory-sanitizer coverage remains unverified unless separately evidenced.

For the new special-value fixtures, record fixture name, row/group, raw BF16
bits, native/candidate E8M0 codes and FP8/QAT output bytes. If scales differ,
inspect the existing amax reduction, input masks and scale ownership/storage
before attributing the mismatch to the new multiply. If scales match, inspect
the arithmetic, clamp/conversion and payload storage paths. An sNaN-only failure
would motivate a reduction-order check; it is not proof of that cause. No such
GPU failure has been observed locally. Retain strict assertions and do not run
an automatic variant sweep.

Map A09 `PERSISTENT_1` dispatches to actual JIT artifacts and host calls.
Every nonempty PS launch must have 256 TGs x 128 threads, or work-item grid
`(32768,1,1)` and workgroup `(128,1,1)`. This does not prove concurrent residency.
For each changed FP8/QAT path, save actual IR, ISA, code object and metadata.
Check buffer access widths, masks, dynamic metadata.shared=0, scratch=0 and
loop dependencies. A zero fixed LDS field alone is insufficient. Non-PS has
no task loop. If launch tracing is unavailable, report the P256 runtime gate
as pending and stop before performance acceptance.

The selected real-binder FP8/QAT PS uses 21/18 VGPR and 892/876 code bytes.
Frozen native uses 49/44 VGPR and 1388/1264 bytes. The previous PS used 53/47
VGPR and 1436/1420 bytes. All use 68 SGPR. Selected PS compiler VGPRBlocks
are 1/1, versus native 3/2; these are allocation fields, not measured occupancy.
The reduction removes eight general division sequences per lane in both paths.
Scale derivation, clamp/conversion order and stored E8M0 bytes remain fixed.
The [review TODO](mono-persistent-review-todo.md) gives the arithmetic proof
and the non-PS comparison.

Expected accesses remain buffer_load_b128, FP8 buffer_store_b64 plus
buffer_store_b8, and QAT buffer_store_b128. Local IR/ISA has no general divide,
LDS, scratch or memory-mask EXEC branch. It retains a single scale writer,
global tail masks, zero-filled masked loads and safe descriptor/source reuse.
FP32 denorm mode remains 3, round mode 0, with `allow_flush_denorm=false`.
Review actual JIT for these properties before timing. A new spill, LDS copy,
mask branch, lost vector width, changed FP mode or unsafe dependency requires
review. Static selection requires FP8 code < 1388 B and VGPR <= 49, QAT code
< 1264 B and VGPR <= 44, with shared=0 and scratch=0 on both paths. These are
local readiness gates, not a runtime speed guarantee.

## 3. Time only frozen native and final PS

After correctness and launch checks pass, run this one matrix without a
profiler. QAT is required because the shared arithmetic change affects both
paths.
The benchmark now defaults to A09 FP8 and QAT; it does not include A03 unless
`--ops a03` is supplied. This handoff selects both A09 paths explicitly.

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --ops a09 a09-qat --rows 96 257 513 --cache both --trials 15 \
  --output "$MONO_RESULTS/performance.json" \
  2>&1 | tee "$MONO_RESULTS/performance.log"
```

This retains RotatingBufferBench, the common input pool, seeded shuffle,
alternating native/PS order and 15 trials. Each sample is mean latency per
launch over Graph replays. Compile, allocate, capture and warm up before timed
replays. Wrappers allocate outputs during capture; the providers do not share
a preallocated output pool. Hot/rotating labels input reuse, not measured cache
hits. These are standalone timings, not model TPOT.

JSON records each round's order, provider times, PS-minus-native microseconds,
start/end timestamps, all samples, median/P10/P90, pool and graph lengths,
median difference and PS/native ratio. Report each of the 12 op/rows/cache
rows separately. Use these conservative decision rules:

```text
delta = PS median - native median
noise = native P90 - native P10
regression = delta > noise and PS P10 > native P90
improvement = -delta > noise and PS P90 < native P10
otherwise = inconclusive
```

An improvement passes that row. A regression fails that row. Overlap,
equal medians or insufficient separation remain inconclusive; they do not
establish equivalence. No positive slowdown allowance is assumed. A path
passes only if all six required rows are improvements, and its correctness
and P256 gates also pass. Report FP8 and QAT status separately. A gain against
historical PS cannot offset a native regression. Return failed or inconclusive
rows with the evidence; do not automatically rerun or start another tuning or ablation
matrix.

## 4. Return the evidence and exact status

Return the recorded SHA/source hashes, dirty diff, environment, correctness
XML/log, dispatch mapping, actual JIT artifacts, offline artifacts and timing
JSON/log. Full review explanations belong in private runner artifacts. Update
these two existing guides with verified status; add no tracked result report.
Separate implementation, code/algebra review, CPU exhaustive checks, offline
ISA/resource/collection, Claude local review, GPU numerical/Graph/JIT/launch
and performance status. Lower-level scale request combining, cache traffic,
retry state and cycle attribution remain unknown without evidence. Full mono
ABI, DEV/HT, reserved registers, joins and model integration are outside this
standalone acceptance.

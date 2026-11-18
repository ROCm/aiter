# A09: one final native / P256 acceptance

Use this guide only after the external runner's offline gate and Claude's
independent local review. Neither certifies GPU correctness or performance.
The [review TODO](mono-persistent-review-todo.md) records the selected buffer
candidate, its resource cost and the unresolved 8 B/wave request cost.

## Source and local status

Use the final source supplied by the runner, based on production `08e20f6`
with A09 `ff4eec1` and this buffer-access change. Record the supplied source
manifest, HEAD and dirty diff; HEAD alone does not identify this candidate.
Do not replay historical patches or change the working framework packages.

| Reference | Fixed source |
| --- | --- |
| Public comparison base | ROCm/aiter `a8a913883d7be9837c2f46f2e92de8fc701bac26` |
| A09 original native | ATOM `10b99a085ba46e5b3a125f0114cec7c71e401552` |
| A03, unchanged | `7108dbec079595026f76b7b1b8e082fe1669796d` |
| Frozen excerpts / hashes | `op_tests/triton_tests/utils/mono_native.py` and `mono_native_manifest.json` |

The worker compiled 12 offline variants, checked the 12 frozen hashes and
collected the same 45 tests with a labelled CPU-only import harness. Additional
offline builds used real binders for FP8/QAT and PS/non-PS, including tails.
Black checks on individual files, Ruff and diff whitespace checks passed.
No GPU test, actual GPU JIT, launch trace or timing was executed. The external
runner must record its own acceptance result and Claude's actual verdict.
Claude's verdict is pending at build-0, with no findings supplied yet.

The runner's fixed offline command is:

```text
/home/weizhu15/workspace/aiter-mono-tools/bin/python /home/weizhu15/workspace/mono-a09-claude-review-20261118/offline_acceptance.py
```

It checks 12 builds, frozen hashes, Black/Ruff, exactly 45 unchanged node IDs
and four binder specializations. The CPU import harness is for collection
only. Never use it to run numerical tests or bypass the GPU gate below.
Private runner artifacts hold full review notes, offline IR/ISA and logs.

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

Compare source hashes with the runner's supplied manifest. Record image digest,
host, selected GPU UUID/ordinal, driver, clock/power settings and other GPU
workloads. Retain load/clock observations with timestamps and sampling interval.
Runtime retry state is unknown unless the machine provides evidence for it.
Do not infer it from MODE.REPLAY_MODE in ISA.

## 2. Correctness and actual launch evidence first

The offline compiler checks frozen symbols before compiling. Its metadata and
ISA do not replace the real GPU wrapper specializations saved by pytest.
Run all 45 numerical tests once. If rocprofv3 is present, collect the launch
trace during this correctness run. Profiler durations are not timing results.

```bash
python -m op_tests.compile_mono_persistent --output "$MONO_RESULTS/compile" \
  2>&1 | tee "$MONO_RESULTS/compile.log"
export TRITON_CACHE_DIR="$MONO_RESULTS/jit-cache"
export TRITON_DUMP_DIR="$MONO_RESULTS/jit-dump"
export TRITON_KERNEL_DUMP=1
mono_trace=()
if command -v rocprofv3 >/dev/null 2>&1; then
  mono_trace=(rocprofv3 --kernel-trace --output-format csv \
    --output-directory "$MONO_RESULTS/launch-trace" --)
fi
"${mono_trace[@]}" python -m pytest \
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

Coverage retains all existing nodes and assertions. A09 checks native and
independent FP8/E8M0/QAT oracles, XOR-prefilled outputs, guards and unchanged
inputs. Shapes include `[513,2048]` with five iterations and `[513,2080]` with
one valid group in the last task. Scale ownership spans all four waves.
The two A09 Graph nodes check both FP8 and QAT after each of 20 input changes.
A03 tests and its existing FP64 K tolerance remain unchanged.

Pass requires 45 passes and zero skips, byte/guard mismatches or GPU errors.
For a failure, retain node ID, seed, input, outputs, source manifest and complete
log. Stop before timing. Do not loosen numerical checks or change frozen native.
GPU memory-sanitizer coverage remains unverified unless separately evidenced.

Map A09 `PERSISTENT_1` dispatches to actual JIT artifacts and host calls.
Every nonempty PS launch must have 256 TGs x 128 threads, or work-item grid
`(32768,1,1)` and workgroup `(128,1,1)`. This does not prove concurrent residency.
For each changed FP8/QAT path, save actual IR, ISA, code object and metadata.
Check buffer access widths, masks, dynamic metadata.shared=0, scratch=0 and
loop dependencies. A zero fixed LDS field alone is insufficient. Non-PS has
no task loop. If launch tracing is unavailable, report the P256 runtime gate
as pending and stop before performance acceptance.

## 3. Time only frozen native and final PS

After correctness and launch checks pass, run this one matrix without a
profiler. QAT is included because the shared memory path changed.

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
rows separately. Apply the existing regression rule:

```text
delta = PS median - native median
noise = native P90 - native P10
regression = delta > noise and PS P10 > native P90
```

Overlapping ranges or insufficient evidence are inconclusive, not automatic
passes. Both P256 and better/non-regressing performance against frozen native
are required. Report FP8 and QAT status separately. A gain against historical
PS cannot offset a native regression. Return failed or inconclusive rows with
the evidence; do not start another tuning or ablation matrix.

## 4. Return the evidence and exact status

Return the supplied source manifest, dirty diff, environment, correctness
XML/log, dispatch mapping, actual JIT artifacts, offline artifacts and timing
JSON/log. Full review explanations belong in private runner artifacts. Update
these two existing guides with verified status; add no tracked result report.
Separate implementation, local compiler/collection, Claude local review,
GPU numerical/Graph/launch and performance status. Lower-level scale request
combining, cache traffic, retry state and cycle attribution remain unknown
without evidence. Full mono ABI, DEV/HT, reserved registers, joins and model
integration are outside this standalone acceptance.

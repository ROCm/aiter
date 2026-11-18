# A09: one final native / P256 acceptance

Use this guide only after the external runner's offline gate and Claude's
independent local review. Neither certifies GPU correctness or performance.
The [review TODO](mono-persistent-review-todo.md) records the selected exact
power-of-two reciprocal change, its static cost and the unresolved request cost.

## Source and local status

Production HEAD is `f4317a49eb72ef43525e2331c73db515a8edd5cd`.
The code baseline is `1777a3f1e67220ff0450f46f1d179b6076358733`.
The selected source is that baseline plus this arithmetic and test patch; it
is not code commit `1777a3f`. Use the supplied patch and final source manifest.
The runner checkout starts at `7bdd28762c6248f67e0babce8bc25e08ff82a61d`.
The read-only proof at
`/home/weizhu15/workspace/mono-a09-finalize-20261118/standalone-source.json`
confirms that only an unrelated gitlink was removed from the source clone.
All other index modes and blobs match production. Submodule declarations stay
unchanged. Record the manifest, HEAD and dirty diff; HEAD alone is insufficient.
Do not replay historical patches or change the working framework packages.

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
and reference semantics; it is not GPU numerical proof. The new GPU fixtures
are implemented but have not run. Full worker logs, IR/ISA, code objects,
source manifest, patch and dependency review are in private `build/a09-local/`.

The outer runner's fixed command remains:

```text
/home/weizhu15/workspace/aiter-mono-tools/bin/python /home/weizhu15/workspace/mono-a09-finalize-20261118/acceptance.py
```

It checks 12 builds, frozen hashes, Black/Ruff, the exact 45 node IDs, four real
binder variants, exhaustive CPU arithmetic and ISA/resource selection gates.
The worker did not run this command because it writes outside its allowed
workspace. The outer runner owns its execution and result. Independent Claude
review of this new arithmetic is pending until the outer workflow records it.
The previous approval does not cover this patch. Neither local checks nor a
local approval certify GPU correctness, actual JIT, launch or performance.
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

Compare source hashes with the runner's supplied manifest. Record image digest,
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
and P256 gates also pass. Report FP8 and QAT status separately. A gain against historical
PS cannot offset a native regression. Return failed or inconclusive rows with
the evidence; do not automatically rerun or start another tuning or ablation
matrix.

## 4. Return the evidence and exact status

Return the supplied source manifest, dirty diff, environment, correctness
XML/log, dispatch mapping, actual JIT artifacts, offline artifacts and timing
JSON/log. Full review explanations belong in private runner artifacts. Update
these two existing guides with verified status; add no tracked result report.
Separate implementation, code/algebra review, CPU exhaustive checks, offline
ISA/resource/collection, Claude local review, GPU numerical/Graph/JIT/launch
and performance status. Lower-level scale request combining, cache traffic,
retry state and cycle attribution remain unknown without evidence. Full mono ABI, DEV/HT, reserved registers, joins and model
integration are outside this standalone acceptance.

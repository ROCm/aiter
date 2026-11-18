# A03 and A09: gfx1250 native / persistent validation

The branch is `ROCm/aiter:asm_mon_dev`. These are standalone persistent stages.
Run correctness before timing. No GPU result or speedup has been measured on
the development host. Do not install these stages into a model for this task.

The [2026-11-18 gfx1250 hardware report](mono-results/2026-11-18-gfx1250/README.md)
records 36 passing GPU tests after an import fix. Eight benchmark configurations
regress, and four remain inconclusive. Standalone performance acceptance fails.

For the next run, use the [review follow-up TODO](mono-persistent-review-todo.md).
It keeps the import fix and adds correctness cases, runtime JIT evidence,
boundary measurements and separate optimization candidates. Its checkout
instructions describe the separate baseline and candidate runs.

The local code candidate is `ff4eec11474431671047122be058ad43df5f8be0`. It includes A03
`7108dbec079595026f76b7b1b8e082fe1669796d` and the A09 explicit-layout change.
Both PS wrappers use grid `(256,)`. A03 keeps the native device source; A09
uses a shared Gluon tile helper and runs a task loop only for PS. Each wave
stores its own scales without LDS rearrangement. Quantization math is unchanged.
Local compilation and host checks pass; the expanded 45 GPU tests and timing
are pending. The report above covers the previous implementation only.

## Code and native references

The original handoff was `efa58c2ccd4525a8287c4f886945fc48ed86a192`.
For the current candidate, apply the code patches below and record the resulting SHA.

| Item | Fixed source |
| --- | --- |
| Development base and A03 native | ROCm/aiter `df6a53e19a6d4bbd434c1700b83a862d79215e9e` |
| A09 native | ROCm/ATOM `10b99a085ba46e5b3a125f0114cec7c71e401552` |
| Native excerpts | `op_tests/triton_tests/utils/mono_native.py` |
| Excerpt provenance and SHA256 | `op_tests/triton_tests/utils/mono_native_manifest.json` |

The 12 native symbols are frozen source excerpts. They are separate from the
candidate to prevent a candidate edit from changing its own reference.
Do not format or refactor them. The ATOM FP8 binding is resolved to E4M3FN for
gfx1250. No ATOM installation or model checkpoint is required.

| Stage | Input and output | Scheduling change |
| --- | --- | --- |
| A03 | Shared BF16 `[96,1792]`; Q width 1280 and K width 512, both row stride 1792 elements. Q output E4M3FN `[96,1280]`, uint8 E8M0 `[96,40]`; K output BF16 `[96,512]`. Separate weights, eps `1e-20`, amax floor `1e-10`. | Add `persistent=True` to `fused_dual_rmsnorm_mxfp8_quant`. One row per task; original math and row loop remain. |
| A09 | BF16 `[96,2048]` to E4M3FN `[96,2048]` and E8M0 `[96,64]`. Group32, amax floor `1e-10`. | `quantize_fp8_group32(..., persistent=True)`. ATOM's 1024-element task, 192 tasks for this shape. |

Both PS grids are `(256,)`. Worker `pid0` processes tasks at stride 256.
Each TG has four wave32 waves. A09 nonpersistent mode calls the same tile
helper once. A03 and A09 use different native scale recipes. Do not replace
either recipe with the other or loosen byte equality to make a mismatch pass.
A03 keeps the actual row count M and uses `NUM_PRGMS=256` only for PS.

## Prepare the machine

Use a working gfx1250 ROCm image with its matched PyTorch and Triton versions.
The development host's CPU PyTorch is not a GPU environment recommendation.
Use an isolated checkout and test environment. Stop on any failed command.
Preserve the machine's working
framework packages and model checkout.

```bash
set -euo pipefail
git clone --branch asm_mon_dev git@github.com:ROCm/aiter.git aiter-mono-validation
cd aiter-mono-validation
git switch -c mono-p256-candidate a8a913883d7be9837c2f46f2e92de8fc701bac26
# Save this updated guide outside the checkout first. Receive both code patches.
# Set MONO_PATCH_DIR to their absolute directory before running these commands.
: "${MONO_PATCH_DIR:?Set the directory containing the received code patches}"
git am "$MONO_PATCH_DIR/a03-native-p256.patch"
git am "$MONO_PATCH_DIR/a09-explicit-layout.patch"
python3 -m venv --system-site-packages .venv-mono-test
source .venv-mono-test/bin/activate
python -m pip install pytest
export AITER_TRITON_ONLY=1
export TRITON_DEFAULT_FP_FUSION=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MONO_RESULTS="$PWD/../mono-results-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$MONO_RESULTS"
git rev-parse HEAD > "$MONO_RESULTS/commit.txt"
git status --porcelain > "$MONO_RESULTS/git-status.txt"
python -m pip freeze > "$MONO_RESULTS/pip-freeze.txt"
python - <<'PY' | tee "$MONO_RESULTS/environment.txt"
import platform
import torch
import triton
from triton.runtime import driver
assert torch.version.hip is not None, "A ROCm PyTorch build is required"
assert torch.cuda.is_available(), "No ROCm GPU is visible"
target = driver.active.get_current_target()
assert target.arch in ("gfx1250", "gfx1250-strict"), target
assert target.warp_size == 32, target
print(platform.platform())
print("torch", torch.__version__, "hip", torch.version.hip, "triton", triton.__version__)
print("target", target)
print(torch.cuda.get_device_properties(torch.cuda.current_device()))
PY
```

Record the container image digest, host name, selected GPU, driver version,
clock/power settings and other GPU workloads beside these files. Use the same
GPU and settings for both providers. No build of CK or HIP extensions is needed
for these Triton-only commands. If a package import fails, record the complete
traceback before changing dependencies.

## Compile and run correctness

The compile command also rejects changes to the frozen native excerpts. It
builds 12 variants with an explicit gfx1250 / wave32 target and saves LLVM IR,
Triton/Gluon IR, assembly, code objects and metadata. It does not run a kernel.
These offline signatures do not include the real wrapper's pointer alignment
and integer/stride specialization. Their ISA and resource counts are examples,
not a substitute for the actual JIT artifacts saved during pytest below.

```bash
python -m op_tests.compile_mono_persistent --output "$MONO_RESULTS/compile" \
  2>&1 | tee "$MONO_RESULTS/compile.log"
export TRITON_CACHE_DIR="$MONO_RESULTS/jit-cache"
export TRITON_DUMP_DIR="$MONO_RESULTS/jit-dump"
export TRITON_KERNEL_DUMP=1
python -m pytest op_tests/triton_tests/quant/test_mono_persistent.py -ra -v \
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

| Coverage | Cases and checks |
| --- | --- |
| A03 task count | 0, 1, 95, 96, 255, 256, 257, 513 rows; both 1280/512 and 96/33 widths |
| A03 layout and math | Shared storage, actual strides, padded outputs, independent Q/K eps, floors 0 / `1e-10` / `1e-4`; K also checked against FP64 RMSNorm |
| A09 shape | `(0,32)`, `(1,32)`, `(3,288)`, `(17,5120)`, `(2,3,128)`, `(96,2048)`, `(96,5120)`, `(96,576)`, `(128,2048)`, `(129,2048)`, `(256,2048)`, `(257,2048)`, `(512,2048)`, `(513,2048)`, `(513,2080)`, `(255,1024)`, `(256,1024)`, `(257,1024)`, `(513,1024)`, `(8193,32)` |
| A09 values | Zero and negative zero, small and large finite BF16 values, E8M0 scale boundaries; separate PyTorch quant and QAT oracle; `[513,2048]` scale ownership patterns across waves and five iterations |
| Output writes | Before each provider, fill output bytes with the native result XOR `0xFF`; compare all result bytes and guards |
| Replay | 20 Graph replays for A03 and A09 `[257,2048]` / `[513,2048]`; change input, assert reference output changes, then compare all bytes to frozen native |
| API limits | Reject invalid output shape and noncontiguous A09 input |

Pass requires all 45 candidate tests (including the original 36) to pass,
with no skips, byte mismatch, guard change,
input/weight mutation or GPU error. FP64 K comparison has its stated BF16
tolerance; native/PS output comparison has zero tolerance. Zero-size tensors
have no payload; A09 skips their launch. Do not treat skipped tests as success.

For a mismatch, preserve the failing node ID, seed, source commit and complete
log. Save the input and both outputs with `torch.save` before a repair. Record
the changed commit and rerun the failing case plus the full matrix after fixing
it. If memory checking is available in the installed ROCm toolchain, also run
the target and tail cases under that tool and retain its log.

Save a separate kernel trace to verify the actual grid and workgroup size. For
a machine with `rocprofv3`, use the command below. It is a configuration check;
do not use instrumented durations for the performance comparison.

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory "$MONO_RESULTS/launch-trace" -- \
  python -m pytest op_tests/triton_tests/quant/test_mono_persistent.py -q
```

Check that each nonempty PS launch has 256 TGs with 128 threads per TG.
The current candidate has work-item grid `(32768,1,1)` and workgroup
`(128,1,1)`, which gives TG grid `(256,1,1)`. Keep
the raw CSV and actual JIT metadata. New A03 kernel names no longer contain
a `PERSISTENT` field. Trace each provider separately or map dispatch correlation
IDs to host calls, including M, NUM_PRGMS and grid. Grid alone cannot identify
the provider at M=256. If the profiler is absent, record that the
runtime launch trace is pending; the wrapper code alone is not a measured trace.

## Compare performance after correctness passes

```bash
python -m op_tests.op_benchmarks.triton.bench_mono_persistent \
  --rows 96 257 513 --cache both --trials 15 \
  --output "$MONO_RESULTS/performance.json" \
  2>&1 | tee "$MONO_RESULTS/performance.log"
```

This uses the repository's `RotatingBufferBench`. Both providers use the same
input pool, seeded shuffle and graph method. Hot mode uses one input set;
rotating mode uses the repository's gfx1250 pool defaults. Compilation,
allocation, graph construction and warmup precede the timed replays. Provider
order alternates native/PS and PS/native. The benchmark does not run correctness
checks; the pytest gate above is required.

Each sample is the mean latency per kernel launch in one trial. JSON records
all samples, their median, P10/P90, pool length, graph length, versions and the
PS/native median ratio. These are standalone graph timings, not model TPOT.
Compare each op/shape/cache row separately. Report any PS slowdown larger than
native run-to-run variation as a regression. If variation prevents a decision,
reduce external GPU load and repeat; report the result as inconclusive until
then. Do not use a hypothetical fusion benefit to accept a standalone regression.

## Mono integration boundary

This branch completes the standalone P256 scheduling change. It does not expose
a callable SP3/FlyDSL device function and does not implement inter-op joins.
An empty worker exits this standalone kernel; in a fused kernel it must return
to the outer controller and participate in its join.

The full mono contract also requires activation loads with DEV scope, all
output/scale stores with DEV + HT, and preservation of s104/s105. Ordinary
Triton loads/stores and its register allocator do not express all these
requirements here. An offline Triton 3.9 probe maps `.cg` loads to DEV, but
`.wb` plus `evict_last` stores do not emit HT. No claim of those properties is
made for this branch.

After standalone acceptance, use the mono controller's SP3/FlyDSL memory and
call interfaces, keeping each op's numerical body and task mapping. Explicitly
set DEV + HT for payload and scale stores, DEV for activation loads, preserve
s104/s105 across the call, and join before the consumer loads. Audit the actual
code object and test producer/consumer visibility and empty workers before
marking integration complete. A Triton standalone pass does not validate this
future integration or concurrent residency of all 256 TGs.

## Development validation and return package

On 2026-11-18, the development host used CPU PyTorch 2.15.0 and Triton 3.9.0.
All 12 offline builds passed with four wave32 waves and `num_ctas=1`. The 12
frozen symbol hashes passed. Black, Ruff 0.16.0 and Python syntax checks passed.
GPU pytest collection stops in the existing `utils/mla_decode_ref.py` driver
initialization on this host. No GPU correctness, replay or timing result exists
from this host. Independent Codex review and its repairs are recorded in the
project's operator note.

Return the complete small result directory, including versions, exact commit,
correctness XML/logs, compile metadata/assembly and timing JSON. Include a table
with op, shape, cache mode, native median/P10/P90, PS median/P10/P90, ratio and
pass/regression/inconclusive status. Separate implementation, GPU correctness,
performance and full mono integration status in the report.

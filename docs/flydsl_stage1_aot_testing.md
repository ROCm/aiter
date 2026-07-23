# FlyDSL Stage1 ABI-driven AOT testing

## 1. Scope

This guide validates the AOT4 implementation on
`zhimding/refactor_aot`:

- the shared Stage1 provider resolves `CompileUnit` objects;
- Stage1 compiles directly from each `KernelSignature`;
- pointer, scalar, stream, and tensor ABI arguments are materialized without
  constructing Stage1 FakeTensors or entering the full Stage1 runtime host;
- runtime strict loading uses the existing FlyDSL disk cache with
  `FLYDSL_RUNTIME_RUN_ONLY=1` and cannot fall back to JIT compilation;
- the CK-Tile Stage1 FlyDSL epilogue uses the metadata-only tensor ABI adapter;
- the existing Stage2 AOT path still works after stream ownership moved to
  `LaunchContext`.

AOT4 does **not** implement:

- a Stage2 `CompilePlan`;
- AOT sorting;
- Manifest generation or persistence;
- the final upstream FlyDSL AOT API.

Stage2 remains explicitly transitional: AOT drives its full host under
`FakeTensorMode`, but passes `LaunchContext(None)` instead of letting kernel
internals inspect `COMPILE_ONLY`.

All commands below use Bash and assume the repository root is the current
directory.

## 2. Prerequisites

### Repository and software

Run from the repository root and activate the required virtual environment
before every Python-related command:

```bash
source /opt/venv/bin/activate
test -f setup.py
test -f aiter/aot/flydsl/moe.py
git branch --show-current
```

The real GPU flow requires:

- a visible gfx950 GPU with 256 compute units;
- a working ROCm PyTorch installation;
- FlyDSL 0.2.3, which is the version pinned by `setup.py`;
- several GiB of free temporary disk space for compiler intermediates and the
  isolated FlyDSL cache.

Check the imported versions:

```bash
source /opt/venv/bin/activate
python - <<'PY'
from importlib.metadata import version

import flydsl
import torch

print("torch:", torch.__version__)
print("torch.version.hip:", torch.version.hip)
print("flydsl:", version("flydsl"))
print("flydsl module:", flydsl.__file__)

assert torch.version.hip, "PyTorch is not a ROCm build"
assert version("flydsl") == "0.2.3", "setup.py requires flydsl==0.2.3"
PY
```

If necessary, install the pinned wheel into the active environment:

```bash
source /opt/venv/bin/activate
python -m pip install "flydsl==0.2.3"
```

### GPU selection and clean environment

`GPU_INDEX` is the physical GPU index to isolate. The process sees that GPU as
`cuda:0` after `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set.

```bash
source /opt/venv/bin/activate

export GPU_INDEX="${GPU_INDEX:-0}"
export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export GPU_ARCHS=gfx950
export CU_NUM=256

unset ARCH
unset COMPILE_ONLY
unset FLYDSL_GPU_ARCH
unset FLYDSL_RUNTIME_RUN_ONLY
unset FLYDSL_COMPILE_OPT_LEVEL
unset FLYDSL_COMPILE_BACKEND
unset FLYDSL_COMPILE_LLVM_DIR
unset FLYDSL_DEBUG_ENABLE_DEBUG_INFO
unset FLYDSL_EXTRA_SOURCE_DIRS
unset AITER_CONFIG_FMOE
unset AITER_FLYDSL_USE_STAGE1_COMPILE_PLAN
unset AITER_FLYDSL_FORCE_REDUCE

python - <<'PY'
import torch

assert torch.cuda.is_available(), "No visible ROCm GPU"
assert torch.cuda.device_count() == 1, "GPU isolation should expose one device"

properties = torch.cuda.get_device_properties(0)
arch = str(properties.gcnArchName).split(":", 1)[0]
print("visible device:", properties.name)
print("arch:", properties.gcnArchName)
print("compute units:", properties.multi_processor_count)

assert arch == "gfx950", f"expected gfx950, got {arch}"
assert properties.multi_processor_count == 256
PY
```

Create a private work directory and cache. Keep the same shell open for the
remaining commands so these exported paths remain available:

```bash
source /opt/venv/bin/activate

export WORK_DIR
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aiter-aot4-gfx950.XXXXXX")"
export CACHE_DIR="$WORK_DIR/flydsl-cache"
export ONE_ROW_CSV="$WORK_DIR/gfx950-one-row.csv"
mkdir -p "$CACHE_DIR"

printf 'WORK_DIR=%s\nCACHE_DIR=%s\nONE_ROW_CSV=%s\n' \
  "$WORK_DIR" "$CACHE_DIR" "$ONE_ROW_CSV"
df -h "$WORK_DIR"
```

## 3. Fast CPU suite

The tests isolate project imports and mock compiler, launcher, tensor-allocation,
stream, and CUDA boundaries. They do not require a GPU.

Run the complete AOT1-AOT4 CPU suite:

```bash
source /opt/venv/bin/activate
AITER_AOT_IMPORT=1 \
  python -m pytest -q aiter/aot/flydsl/tests
```

The AOT4 reference result is:

```text
19 passed, 14 subtests passed
```

The important focused nodes are:

```bash
source /opt/venv/bin/activate

# AOT1 compile-request golden.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_moe_compile_baseline.py::TestMoeCompileBaseline::test_gfx950_requests_match_golden_and_are_deterministic

# Callable-bound CompilePlan DSL and immutable CompileContext.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_compile_plan.py

# Pointer/scalar/stream/tensor signature materialization.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_aot_backend.py::TestContextsAndMaterialization::test_signature_materializes_pointer_scalars_stream_and_tensor_metadata

# Compile versus strict load, environment restoration, and no fallback.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_aot_backend.py::TestCompileAndStrictLoad

# Default runtime provider/backend routing.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_moe_stage1_compile_plan.py::TestStage1GoldenParity::test_default_runtime_calls_resolver_and_backend

# Direct Stage1/CK-Tile AOT must not cross FakeTensor, allocation, stream, CUDA,
# build_stage1_compile_inputs, or full flydsl_moe_stage1 boundaries.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_aot_backend.py::TestDirectStage1Aot::test_stage1_and_cktile_jobs_never_enter_fake_or_runtime_host
```

Run formatting, lint, and whitespace checks on the AOT4 Python files:

```bash
source /opt/venv/bin/activate

PY_FILES=(
  aiter/aot/flydsl/moe.py
  aiter/aot/flydsl/tests/moe_compile_recorder.py
  aiter/aot/flydsl/tests/test_aot_backend.py
  aiter/aot/flydsl/tests/test_compile_plan.py
  aiter/aot/flydsl/tests/test_moe_stage1_compile_plan.py
  aiter/ops/flydsl/aot_backend.py
  aiter/ops/flydsl/compile_plan.py
  aiter/ops/flydsl/launch_context.py
  aiter/ops/flydsl/moe_compile_plan.py
  aiter/ops/flydsl/moe_kernels.py
  op_tests/test_moe_2stage.py
)

black --check "${PY_FILES[@]}"
ruff format --check "${PY_FILES[@]}"
ruff check "${PY_FILES[@]}"
git diff --check
```

Expected results are that Black leaves all 11 files unchanged, Ruff reports all
files formatted and all checks passed, and `git diff --check` emits no output.

## 4. Isolated real gfx950 flow

### 4.1 Generate a deterministic one-row config

Do not edit a tracked tuned config. The following command resolves the same
merged FMoE config used by runtime, selects the known gfx950 row that was used
for the AOT4 gate, validates both FlyDSL kernel names, and writes one row under
`WORK_DIR`.

By default, `AITER_CONFIGS.AITER_CONFIG_FMOE_FILE` merges
`aiter/configs/tuned_fmoe.csv` with matching files under
`aiter/configs/model_configs/`. Set `SOURCE_FMOE_CSV` only if an equivalent
already-merged **file** should be used instead.

```bash
source /opt/venv/bin/activate
unset AITER_CONFIG_FMOE

python - <<'PY'
import os
from pathlib import Path

import pandas as pd

from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

kernel1 = "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_bnt0_fp4"
kernel2 = "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic"

source_override = os.environ.get("SOURCE_FMOE_CSV")
source = (
    Path(source_override).expanduser().resolve()
    if source_override
    else Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE).resolve()
)
destination = Path(os.environ["ONE_ROW_CSV"]).resolve()

if not source.is_file():
    raise FileNotFoundError(f"FMoE source must be a CSV file: {source}")

frame = pd.read_csv(source)
tags = (
    frame["_tag"].fillna("").astype(str).str.strip()
    if "_tag" in frame.columns
    else pd.Series("", index=frame.index)
)
mask = (
    (frame["cu_num"].astype(int) == 256)
    & (frame["token"].astype(int) == 16)
    & (frame["model_dim"].astype(int) == 3072)
    & (frame["inter_dim"].astype(int) == 256)
    & (frame["expert"].astype(int) == 256)
    & (frame["topk"].astype(int) == 8)
    & (frame["kernelName1"].astype(str) == kernel1)
    & (frame["kernelName2"].astype(str) == kernel2)
    & (tags == "")
)
selected = frame.loc[mask].drop_duplicates().iloc[:1].copy()
if len(selected) != 1:
    raise RuntimeError(
        f"expected the known gfx950 row in {source}, found {int(mask.sum())}"
    )

assert get_flydsl_kernel_params(kernel1)["stage"] == 1
assert get_flydsl_kernel_params(kernel2)["stage"] == 2

destination.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(destination, index=False)

columns = [
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "q_type",
    "q_dtype_a",
    "q_dtype_w",
    "kernelName1",
    "kernelName2",
]
print("source:", source)
print("one-row config:", destination)
print(selected[columns].to_string(index=False))
PY
```

Verify that the generated file has exactly a header and one data row:

```bash
test "$(wc -l < "$ONE_ROW_CSV")" -eq 2
```

For `AITER_CONFIG_FMOE`, a single file path is used directly and a
colon-separated list of files is merged. A directory path is not supported.
The strict runtime command below deliberately sets `AITER_CONFIG_FMOE` to the
generated single file.

### 4.2 Compile Stage1 and transitional Stage2

Run the normal MoE AOT entry point with one worker and the isolated cache:

```bash
source /opt/venv/bin/activate
set -o pipefail

AITER_AOT_IMPORT=1 \
AITER_FLYDSL_AOT_WORKERS=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
COMPILE_ONLY=0 \
FLYDSL_RUNTIME_RUN_ONLY=0 \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
  python -m aiter.aot.flydsl.moe --csv "$ONE_ROW_CSV" \
  |& tee "$WORK_DIR/aot.log"
status=${PIPESTATUS[0]}
test "$status" -eq 0
```

This selected row expands to two Stage1 jobs and two Stage2 jobs because the
current AOT coverage includes both supported bias choices. Expected evidence:

- the banner reports `Stage1 jobs:    2` and `Stage2 jobs:    2`;
- every Stage1 success line contains `direct_units=1`;
- every Stage2 success line contains `stage2_transitional=True`;
- the summary reports four successful compilations and zero failures;
- the final line is `All compilations succeeded. Cache is ready.`

Check the distinguishing markers:

```bash
grep -q "direct_units=1" "$WORK_DIR/aot.log"
grep -q "stage2_transitional=True" "$WORK_DIR/aot.log"
grep -q "All compilations succeeded. Cache is ready." "$WORK_DIR/aot.log"
! grep -q "\[FAIL\]" "$WORK_DIR/aot.log"
```

`direct_units=1` is emitted only by the direct Stage1/CK-Tile plan branch.
The CPU boundary test in section 3 is the stronger proof that this branch does
not enter `FakeTensorMode`, `build_stage1_compile_inputs`, tensor allocation,
stream lookup, CUDA target lookup, or full `flydsl_moe_stage1`.

### 4.3 Strict runtime and numerical gate

Keep `FLYDSL_RUNTIME_RUN_ONLY=1` active for the complete runtime process. Do
not pass `--csv-filter`: that option intentionally disables `check_aot_cache`
for targeted development rows.

```bash
source /opt/venv/bin/activate
set -o pipefail

AITER_CONFIG_FMOE="$ONE_ROW_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/tuned_op_bench.csv" \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_2stage.py --no-legacy \
  |& tee "$WORK_DIR/strict-runtime.log"
status=${PIPESTATUS[0]}
test "$status" -eq 0
```

Expected criteria:

- the invocation shows `strict_accuracy = True` and
  `check_aot_cache = True`;
- it scans one CSV case and records one result;
- no `AotCacheMissError` or `no usable AOT cache` message appears;
- the process exits zero;
- the output contains no NaN;
- the existing strict criterion passes: if elementwise `checkAllclose` reports
  an error, `logits_diff` must still be at most `0.01`.

The AOT4 reference run exited zero with `logits_diff` approximately
`0.000577`. That number is not a portable golden; the criteria above are the
actual assertions in `op_tests/test_moe_2stage.py`.

## 5. Negative controls

### 5.1 Empty-cache strict failure

Create a genuinely empty cache and run the same one-row test. The command must
fail before numerical validation:

```bash
source /opt/venv/bin/activate

export EMPTY_CACHE="$WORK_DIR/empty-cache"
rm -rf -- "$EMPTY_CACHE"
mkdir -p "$EMPTY_CACHE"

if AITER_CONFIG_FMOE="$ONE_ROW_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/negative-tuned-op.csv" \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$EMPTY_CACHE" \
FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_2stage.py --no-legacy \
  >"$WORK_DIR/empty-cache.log" 2>&1; then
  status=0
else
  status=$?
fi

cat "$WORK_DIR/empty-cache.log"
test "$status" -ne 0
grep -Eq "AotCacheMissError|no usable AOT cache" "$WORK_DIR/empty-cache.log"
```

The structured error should identify
`aiter.flydsl.moe.stage1.mixed_gemm.v1`, target `gfx950/256`, its signature,
and the empty cache directory. `FLYDSL_RUNTIME_RUN_ONLY=1` makes this a hard
failure; no JIT fallback is permitted.

### 5.2 Incompatible cache key

FlyDSL includes compile options and the target in its cache key. Changing a
compile option while loading the otherwise valid cache is a cheap incompatible
cache control:

```bash
source /opt/venv/bin/activate

if AITER_CONFIG_FMOE="$ONE_ROW_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/incompatible-tuned-op.csv" \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
FLYDSL_RUNTIME_RUN_ONLY=1 \
FLYDSL_COMPILE_OPT_LEVEL=1 \
  python op_tests/test_moe_2stage.py --no-legacy \
  >"$WORK_DIR/incompatible-cache.log" 2>&1; then
  status=0
else
  status=$?
fi

cat "$WORK_DIR/incompatible-cache.log"
test "$status" -ne 0
grep -Eq "AotCacheMissError|no usable AOT cache" \
  "$WORK_DIR/incompatible-cache.log"
```

This is a cache-key incompatibility test, not a numerical failure: Stage1 must
fail to load before a kernel executes. A real cache produced for another ROCm
target behaves the same way because the target is also part of the FlyDSL key.
The structured unit test for an explicitly mismatched `RocmTarget` is:

```bash
source /opt/venv/bin/activate
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_aot_backend.py::TestCompileAndStrictLoad::test_invalid_abi_fields_and_compiler_mismatch_are_structured
```

## 6. CK-Tile tensor ABI test

This focused gfx950 command compiles the CK-Tile SwiGLU FlyDSL epilogue
directly, then loads it strictly with real runtime tensors and checks a zero
input/output smoke:

```bash
source /opt/venv/bin/activate

AITER_AOT_IMPORT=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
FLYDSL_RUNTIME_RUN_ONLY=0 \
  python - <<'PY'
import torch

from aiter.aot.flydsl.common import run_only_env
from aiter.aot.flydsl.moe import compile_one_config
from aiter.ops.flydsl.moe_kernels import flydsl_swiglu_and_mul_interleaved

result = compile_one_config(
    kernel_name="cktile_epilogue_swiglu",
    model_dim=0,
    inter_dim=256,
    experts=0,
    topk=8,
    cu_num=256,
    stage="epilogue",
    act="swiglu",
    split_k=2,
    post_activation_layout="interleaved",
    enable_bias=False,
)
print("CKTILE_DIRECT_RESULT", result, flush=True)
assert result["compile_time"] is not None
assert result["compile_units"] == 1
assert result["direct_stage1_aot"] is True

x = torch.zeros((2, 512), dtype=torch.bfloat16, device="cuda")
out = torch.empty((2, 256), dtype=torch.bfloat16, device="cuda")
with run_only_env():
    flydsl_swiglu_and_mul_interleaved(x, out)
torch.cuda.synchronize()

assert torch.count_nonzero(out).item() == 0
print("CKTILE_STRICT_TENSOR_ABI_PASS", flush=True)
PY
```

This proves that:

- the two declared bf16 tensor ABI fields are compiled from metadata only;
- the resulting cache signature matches normal contiguous runtime tensors;
- strict loading and launch work without JIT fallback.

FlyDSL 0.2.x does **not** provide a public metadata-only tensor ABI
constructor. AOT4 therefore uses a minimal internal `TorchTensorJitArg` shim.
This command must not be described as testing a public FlyDSL tensor AOT API.

## 7. Stage2 transitional smoke

The full flow in section 4 already compiles Stage2. This focused command
isolates the two Stage2 bias variants in a separate cache:

```bash
source /opt/venv/bin/activate

export STAGE2_CACHE_DIR="$WORK_DIR/stage2-cache"
rm -rf -- "$STAGE2_CACHE_DIR"
mkdir -p "$STAGE2_CACHE_DIR"

AITER_AOT_IMPORT=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
ONE_ROW_CSV="$ONE_ROW_CSV" \
FLYDSL_RUNTIME_CACHE_DIR="$STAGE2_CACHE_DIR" \
FLYDSL_RUNTIME_RUN_ONLY=0 \
  python - <<'PY'
import os

from aiter.aot.flydsl.moe import compile_one_config, parse_csv

jobs = [
    job
    for job in parse_csv(os.environ["ONE_ROW_CSV"])
    if job["stage"] == 2
]
print(f"transitional_stage2_jobs={len(jobs)}", flush=True)
assert len(jobs) == 2

results = [compile_one_config(**job) for job in jobs]
print("STAGE2_SMOKE_RESULTS", results, flush=True)
assert all(result["compile_time"] is not None for result in results)
assert not any(result.get("direct_stage1_aot") for result in results)
PY
```

Expected behavior:

- both jobs print `stage2_transitional=True`;
- `COMPILE_ONLY=1` is applied by the existing transitional host;
- no kernel is launched during AOT;
- Stage2 receives an explicit null `LaunchContext`, so it performs no runtime
  stream lookup during compilation.

This smoke does not claim that Stage2 is ABI-driven.

## 8. Troubleshooting

### FlyDSL 0.2.2 versus 0.2.3

Some development machines still import FlyDSL 0.2.2. The relevant
`TorchTensorJitArg`, run-only, and cache interfaces were inspected on both
0.2.2 and 0.2.3, and the focused backend tests pass on both. Nevertheless,
`setup.py` pins 0.2.3; use 0.2.3 for the real gate and release evidence.

Always inspect the package that Python actually imports:

```bash
source /opt/venv/bin/activate
python - <<'PY'
from importlib.metadata import version

import flydsl

print(version("flydsl"))
print(flydsl.__file__)
PY
```

Remove a stale `PYTHONPATH` entry or rebuild a local FlyDSL checkout if it
shadows the installed 0.2.3 wheel.

### Cache or target mismatch

- Use the same `CACHE_DIR`, kernel source, FlyDSL version, compile options,
  gfx target, and CU count for AOT and strict runtime.
- The selected row has `cu_num=256`, which maps to gfx950 in the MoE AOT
  adapter.
- Do not reuse a cache from gfx942 or a different FlyDSL/compiler build.
- A target/cache-key mismatch produces `AotCacheMissError` or
  `no usable AOT cache` before numerical output; it is not a numerical
  correctness failure.

### Quantization error reporting

The selected MXFP4 row can print a failed elementwise `checkAllclose` summary
while the process still passes the repository's strict criterion. The current
criterion requires:

- no NaN; and
- not both `err != 0` and `logits_diff > 0.01`.

Treat process exit zero plus those assertions as the current gate. Do not
report elementwise allclose success when the log says otherwise.

### OOM or a shared GPU

- Keep `AITER_FLYDSL_AOT_WORKERS=1`.
- Choose an idle `GPU_INDEX`.
- Confirm that only one device is visible.
- Use the one-row config rather than the complete merged model config.
- Remove stale tensors/processes before retrying.

### Stale environment or cache

Unset `COMPILE_ONLY`, `ARCH`, `FLYDSL_GPU_ARCH`, and
`FLYDSL_RUNTIME_RUN_ONLY` before starting. Use a new `mktemp` work directory
for every gate. Do not point `AITER_CONFIG_FMOE` at a directory.

### Run-only prevention of JIT fallback

`FLYDSL_RUNTIME_RUN_ONLY=1` must remain active for both positive and negative
strict runtime processes. On a miss, `load_aot(..., strict=True)` raises; it
never invokes the compile fallback. If a supposedly strict command compiles a
new Stage1 artifact, first verify that this variable reached the Python
process.

### Internal tensor shim limitation

`AotArtifact` currently wraps the FlyDSL launcher and existing FlyDSL cache.
There is no independent artifact file format or public tensor ABI API.
The internal metadata-only `TorchTensorJitArg` adapter is intentionally
isolated in `aiter/ops/flydsl/aot_backend.py` so a future upstream API can
replace it mechanically.

## 9. Cleanup and acceptance checklist

Remove generated files and restore the shell environment:

```bash
if [[ -n "${WORK_DIR:-}" && -d "$WORK_DIR" ]]; then
  rm -rf -- "$WORK_DIR"
fi

unset WORK_DIR CACHE_DIR ONE_ROW_CSV EMPTY_CACHE STAGE2_CACHE_DIR
unset SOURCE_FMOE_CSV
unset AITER_CONFIG_FMOE AITER_TUNED_OP_BENCH_CSV
unset AITER_AOT_IMPORT AITER_FLYDSL_AOT_WORKERS
unset ARCH COMPILE_ONLY FLYDSL_GPU_ARCH FLYDSL_RUNTIME_RUN_ONLY
unset FLYDSL_RUNTIME_CACHE_DIR FLYDSL_COMPILE_OPT_LEVEL
unset FLYDSL_COMPILE_BACKEND FLYDSL_COMPILE_LLVM_DIR
unset FLYDSL_DEBUG_ENABLE_DEBUG_INFO FLYDSL_EXTRA_SOURCE_DIRS
unset AITER_FLYDSL_FORCE_REDUCE
unset GPU_ARCHS CU_NUM HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_INDEX
```

Acceptance checklist:

- [ ] FlyDSL 0.2.3 and a gfx950/256-CU device were verified.
- [ ] The CPU suite reported `19 passed, 14 subtests passed`.
- [ ] The generated CSV contains exactly the known header and one supported
      data row.
- [ ] Stage1 AOT lines contain `direct_units=1`.
- [ ] Stage2 AOT lines contain `stage2_transitional=True`.
- [ ] Strict runtime ran with `FLYDSL_RUNTIME_RUN_ONLY=1` and exited zero.
- [ ] Strict numerical output contains no NaN and meets the current logits
      criterion.
- [ ] Empty and incompatible caches fail non-zero before numerical execution.
- [ ] CK-Tile metadata-only tensor ABI compilation and strict launch pass.
- [ ] Stage2 transitional smoke passes after the `LaunchContext` refactor.
- [ ] Black, Ruff format, Ruff lint, and `git diff --check` pass.
- [ ] The work directory was removed and tracked config files were unchanged.

# FlyDSL MoE ABI-driven AOT testing

## Scope

This guide validates the FlyDSL MoE `OperationPlan` compile projection and
runtime execution: explicit sorting, Stage1, Stage2, and Stage2 reduction on
`gfx950`. Providers own node selection/order and compile overrides; runtime
adapters retain allocation, ABI packing, grids, and stream launch.

The current Aiter compatibility backend:

- resolves callable-bound `CompileUnit` objects with explicit `RocmTarget`;
- materializes pointer/scalar/stream ABI arguments without FakeTensors;
- compiles and strictly loads the concrete oneshot, combined P0v2+P23, and
  four-kernel sorting launchers through the same `AotBackend` as Stage1,
  Stage2, plain reduction, and masked EP reduction;
- routes normal runtime sorting, Stage1, and Stage2 through the same providers;
- raises structured `AotCacheMissError` in run-only mode without JIT fallback.

Sorting is deliberately explicit. Ordinary tuned Stage1/Stage2 CSV rows do not
enable or emit sorting jobs, even if an ad-hoc `uses_flydsl_sorting` column is
present. Callers must provide `MoeSortingCompileCase(max_tokens, num_experts,
topk, has_mask, unit_size, ...)` and opt into FlyDSL sorting at runtime.

This milestone does not add a Manifest, global deduplication, artifact
packaging, or a public FlyDSL save/load API. `AotBackend` remains an Aiter
compatibility layer around FlyDSL 0.2.x launchers and its disk cache.

All commands use Bash from the repository root.

## CPU suite and static checks

Activate the project virtual environment before every Python command:

```bash
source /opt/venv/bin/activate
AITER_AOT_IMPORT=1 python -m pytest -q aiter/aot/flydsl/tests
```

The AOT6 reference result is `36 passed, 53 subtests passed`.

Important focused tests:

```bash
source /opt/venv/bin/activate

# AOT1 golden request baseline.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_moe_compile_baseline.py

# Ordered Stage2 units, op IDs, bound builder kwargs, ABI, persistence, and
# default runtime provider/backend adoption.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_moe_stage2_compile_plan.py

# Metadata-only direct compile/load, strict misses, and AOT boundary guards.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_aot_backend.py

# Concrete sorting units, pure target/token policies, aggregate ordering,
# direct-AOT CPU boundaries, and runtime provider/backend adoption.
AITER_AOT_IMPORT=1 python -m pytest -q \
  aiter/aot/flydsl/tests/test_moe_sorting_compile_plan.py
```

Format and lint all changed Python:

```bash
source /opt/venv/bin/activate

PY_FILES=(
  aiter/aot/flydsl/moe.py
  aiter/aot/flydsl/tests/moe_compile_recorder.py
  aiter/aot/flydsl/tests/test_aot_backend.py
  aiter/aot/flydsl/tests/test_moe_sorting_compile_plan.py
  aiter/aot/flydsl/tests/test_moe_stage2_compile_plan.py
  aiter/ops/flydsl/kernels/moe_sorting_kernel.py
  aiter/ops/flydsl/moe_compile_plan.py
  aiter/ops/flydsl/moe_kernels.py
  aiter/ops/flydsl/moe_sorting.py
)

black --check "${PY_FILES[@]}"
ruff format --check "${PY_FILES[@]}"
ruff check "${PY_FILES[@]}"
git diff --check
```

## Isolated gfx950 setup

Choose one physical GPU. It becomes `cuda:0` inside the isolated processes.

```bash
source /opt/venv/bin/activate

export GPU_INDEX="${GPU_INDEX:-0}"
export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export GPU_ARCHS=gfx950
export CU_NUM=256

unset ARCH COMPILE_ONLY FLYDSL_GPU_ARCH FLYDSL_RUNTIME_RUN_ONLY
unset AITER_CONFIG_FMOE AITER_FLYDSL_FORCE_REDUCE

python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
properties = torch.cuda.get_device_properties(0)
arch = str(properties.gcnArchName).split(":", 1)[0]
print(properties.name, properties.gcnArchName, properties.multi_processor_count)
assert arch == "gfx950"
assert properties.multi_processor_count == 256
PY

export WORK_DIR
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aiter-aot5-gfx950.XXXXXX")"
export CACHE_DIR="$WORK_DIR/flydsl-cache"
export ONE_ROW_CSV="$WORK_DIR/gfx950-one-row.csv"
mkdir -p "$CACHE_DIR"
```

## Explicit sorting direct AOT

Compile one case for every concrete launcher family, then strictly load the
same units. The mask state and token bucket are compile metadata; a dynamic
`num_local_tokens` value is only consumed later while packing a launch.

```bash
source /opt/venv/bin/activate

export SORTING_CACHE="$WORK_DIR/flydsl-sorting-cache"
rm -rf -- "$SORTING_CACHE"
mkdir -p "$SORTING_CACHE"

AITER_AOT_IMPORT=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$SORTING_CACHE" \
  python - <<'PY'
from aiter.aot.flydsl.moe import compile_moe_sorting_case
from aiter.ops.flydsl.aot_backend import create_compile_context, load_aot
from aiter.ops.flydsl.compile_plan import RocmTarget
from aiter.ops.flydsl.moe_compile_plan import (
    MoeSortingCompileCase,
    resolve_moe_sorting_compile_plan,
)

target = RocmTarget("gfx950", 256)
cases = [
    MoeSortingCompileCase(8, 256, 8, False),
    MoeSortingCompileCase(128, 384, 8, False),
    MoeSortingCompileCase(4096, 256, 8, True),
]
compile_context = create_compile_context(target)
for case in cases:
    artifact, = compile_moe_sorting_case(case, context=compile_context)
    print("SORTING_AOT_COMPILE_PASS", artifact.unit.spec.op_id)

load_context = create_compile_context(target, strict_runtime=True)
for case in cases:
    plan = resolve_moe_sorting_compile_plan(case, context=load_context)
    artifact = load_aot(plan.units[0], context=load_context, strict=True)
    print("SORTING_AOT_STRICT_LOAD_PASS", artifact.unit.spec.op_id)
PY
```

Expected operation IDs:

- `aiter.flydsl.moe.sorting.oneshot.v1`
- `aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1`
- `aiter.flydsl.moe.sorting.multiphase.k4_fused.v1`

Run the repository correctness harness against its torch reference under
strict load. These three commands cover oneshot unmasked, E384
`k4_block=512` P0v2+P23, and masked four-kernel numerical execution.

```bash
source /opt/venv/bin/activate
set -o pipefail

unset AITER_AOT_IMPORT COMPILE_ONLY ARCH FLYDSL_GPU_ARCH

HIP_VISIBLE_DEVICES="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
GPU_ARCHS=gfx950 CU_NUM=256 \
FLYDSL_RUNTIME_CACHE_DIR="$SORTING_CACHE" FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_sorting.py \
    -m 8 -e 256 -t 8 -em f -p 0 -dp 0 -md 64 -id 64 \
  |& tee "$WORK_DIR/sorting-oneshot-strict.log"

HIP_VISIBLE_DEVICES="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
GPU_ARCHS=gfx950 CU_NUM=256 \
FLYDSL_RUNTIME_CACHE_DIR="$SORTING_CACHE" FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_sorting.py \
    -m 128 -e 384 -t 8 -em f -p 0 -dp 0 -md 64 -id 64 \
  |& tee "$WORK_DIR/sorting-p0v2-strict.log"

HIP_VISIBLE_DEVICES="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
GPU_ARCHS=gfx950 CU_NUM=256 \
FLYDSL_RUNTIME_CACHE_DIR="$SORTING_CACHE" FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_sorting.py \
    -m 4096 -e 256 -t 8 -em t -p 0 -dp 0 -md 64 -id 64 \
  |& tee "$WORK_DIR/sorting-4k-masked-strict.log"

for log in "$WORK_DIR"/sorting-*-strict.log; do
  grep -q "moe_sorting summary (markdown)" "$log"
  ! grep -Eq "AotCacheMissError|no usable AOT cache" "$log"
done
```

An empty sorting cache must fail structurally and must not JIT:

```bash
source /opt/venv/bin/activate

export EMPTY_SORTING_CACHE="$WORK_DIR/empty-sorting-cache"
rm -rf -- "$EMPTY_SORTING_CACHE"
mkdir -p "$EMPTY_SORTING_CACHE"

if HIP_VISIBLE_DEVICES="$GPU_INDEX" CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
GPU_ARCHS=gfx950 CU_NUM=256 \
FLYDSL_RUNTIME_CACHE_DIR="$EMPTY_SORTING_CACHE" FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_sorting.py \
    -m 8 -e 256 -t 8 -em f -p 0 -dp 0 -md 64 -id 64 \
  >"$WORK_DIR/empty-sorting-cache.log" 2>&1; then
  status=0
else
  status=$?
fi

cat "$WORK_DIR/empty-sorting-cache.log"
test "$status" -ne 0
grep -Eq "AotCacheMissError|no usable AOT cache" \
  "$WORK_DIR/empty-sorting-cache.log"
echo "SORTING_EMPTY_CACHE_NEGATIVE_PASS status=$status"
```

## Direct Stage1 + atomic Stage2 flow

Generate one deterministic row from the merged runtime configuration:

```bash
source /opt/venv/bin/activate

python - <<'PY'
import os
from pathlib import Path

import pandas as pd

from aiter.jit.core import AITER_CONFIGS

source = Path(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE).resolve()
destination = Path(os.environ["ONE_ROW_CSV"]).resolve()
frame = pd.read_csv(source)
tags = (
    frame["_tag"].fillna("").astype(str).str.strip()
    if "_tag" in frame
    else pd.Series("", index=frame.index)
)
mask = (
    (frame["cu_num"].astype(int) == 256)
    & (frame["token"].astype(int) == 16)
    & (frame["model_dim"].astype(int) == 3072)
    & (frame["inter_dim"].astype(int) == 256)
    & (frame["expert"].astype(int) == 256)
    & (frame["topk"].astype(int) == 8)
    & (
        frame["kernelName1"].astype(str)
        == "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_bnt0_fp4"
    )
    & (
        frame["kernelName2"].astype(str)
        == "flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic"
    )
    & (tags == "")
)
selected = frame.loc[mask].drop_duplicates().iloc[:1]
assert len(selected) == 1, int(mask.sum())
selected.to_csv(destination, index=False)
print(selected[[
    "cu_num", "token", "model_dim", "inter_dim", "expert", "topk",
    "kernelName1", "kernelName2",
]].to_string(index=False))
PY

test "$(wc -l < "$ONE_ROW_CSV")" -eq 2
```

Compile into the isolated cache:

```bash
source /opt/venv/bin/activate
set -o pipefail

AITER_AOT_IMPORT=1 \
AITER_FLYDSL_AOT_WORKERS=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
FLYDSL_RUNTIME_RUN_ONLY=0 \
  python -m aiter.aot.flydsl.moe --csv "$ONE_ROW_CSV" \
  |& tee "$WORK_DIR/aot.log"
status=${PIPESTATUS[0]}
test "$status" -eq 0

grep -q "direct_stage1_units=1" "$WORK_DIR/aot.log"
grep -q "direct_stage2_units=1" "$WORK_DIR/aot.log"
grep -q "All compilations succeeded. Cache is ready." "$WORK_DIR/aot.log"
! grep -q "stage2_transitional" "$WORK_DIR/aot.log"
! grep -q "\[FAIL\]" "$WORK_DIR/aot.log"
```

The row expands to two Stage1 and two Stage2 jobs because eligible MXFP4 rows
compile both bias-presence variants. The CSV does not imply sorting. Add its
explicit sorting case to the same cache:

```bash
source /opt/venv/bin/activate

AITER_AOT_IMPORT=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
  python - <<'PY'
from aiter.aot.flydsl.moe import compile_moe_sorting_case
from aiter.ops.flydsl.aot_backend import create_compile_context
from aiter.ops.flydsl.compile_plan import RocmTarget
from aiter.ops.flydsl.moe_compile_plan import MoeSortingCompileCase

context = create_compile_context(RocmTarget("gfx950", 256))
artifact, = compile_moe_sorting_case(
    MoeSortingCompileCase(
        max_tokens=16,
        num_experts=256,
        topk=8,
        has_mask=False,
        unit_size=32,
    ),
    context=context,
)
print("EXPLICIT_SORTING_AOT_PASS", artifact.unit.spec.op_id)
PY
```

Run the matching full numerical case with FlyDSL sorting explicitly selected
and strict cache loading:

```bash
source /opt/venv/bin/activate
set -o pipefail

AITER_CONFIG_FMOE="$ONE_ROW_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/tuned-op-bench.csv" \
AITER_USE_FLYDSL_MOE_SORTING=1 \
AITER_USE_CK_MOE_SORTING=0 \
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

grep -q "check_aot_cache              = True" "$WORK_DIR/strict-runtime.log"
! grep -Eq "AotCacheMissError|no usable AOT cache" \
  "$WORK_DIR/strict-runtime.log"
```

This proves the explicit sorting, Stage1, and atomic Stage2 artifacts coexist
in one cache. The MXFP4 elementwise check may report differences. The
repository gate is process exit zero, no NaN, and its existing `logits_diff`
criterion; do not claim elementwise allclose if the log reports otherwise.

## Plain reduction AOT and numerical gate

Generate a supported reduction row:

```bash
source /opt/venv/bin/activate

export REDUCE_CSV="$WORK_DIR/gfx950-reduce-one-row.csv"
export REDUCE_CACHE="$WORK_DIR/flydsl-reduce-cache"
mkdir -p "$REDUCE_CACHE"

python - <<'PY'
import os
import pandas as pd

from aiter.jit.core import AITER_CONFIGS

frame = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_FMOE_FILE)
tags = (
    frame["_tag"].fillna("").astype(str).str.strip()
    if "_tag" in frame
    else pd.Series("", index=frame.index)
)
mask = (
    (frame["cu_num"].astype(int) == 256)
    & (frame["token"].astype(int) == 256)
    & (frame["model_dim"].astype(int) == 3072)
    & (frame["inter_dim"].astype(int) == 512)
    & (frame["expert"].astype(int) == 128)
    & (frame["topk"].astype(int) == 4)
    & (
        frame["kernelName1"].astype(str)
        == "flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_fp4"
    )
    & (
        frame["kernelName2"].astype(str)
        == "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_reduce_bnt2"
    )
    & (tags == "")
)
selected = frame.loc[mask].drop_duplicates().iloc[:1]
assert len(selected) == 1, int(mask.sum())
selected.to_csv(os.environ["REDUCE_CSV"], index=False)
PY

test "$(wc -l < "$REDUCE_CSV")" -eq 2
```

Compile and require two direct Stage2 units: GEMM plus plain reduction.

```bash
source /opt/venv/bin/activate
set -o pipefail

AITER_AOT_IMPORT=1 \
AITER_FLYDSL_AOT_WORKERS=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$REDUCE_CACHE" \
FLYDSL_RUNTIME_RUN_ONLY=0 \
  python -m aiter.aot.flydsl.moe --csv "$REDUCE_CSV" \
  |& tee "$WORK_DIR/reduce-aot.log"
status=${PIPESTATUS[0]}
test "$status" -eq 0

grep -q "direct_stage1_units=1" "$WORK_DIR/reduce-aot.log"
grep -q "direct_stage2_units=2" "$WORK_DIR/reduce-aot.log"
! grep -q "\[FAIL\]" "$WORK_DIR/reduce-aot.log"
```

Add the row's explicit P0v2+P23 sorting artifact. Its empty `(0, 0)` atomic
buffer placeholder also exercises the dynamic two-dimensional tensor ABI used
by reduction mode:

```bash
source /opt/venv/bin/activate

AITER_AOT_IMPORT=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$REDUCE_CACHE" \
  python - <<'PY'
from aiter.aot.flydsl.moe import compile_moe_sorting_case
from aiter.ops.flydsl.aot_backend import create_compile_context
from aiter.ops.flydsl.compile_plan import RocmTarget
from aiter.ops.flydsl.moe_compile_plan import MoeSortingCompileCase

context = create_compile_context(RocmTarget("gfx950", 256))
artifact, = compile_moe_sorting_case(
    MoeSortingCompileCase(256, 128, 4, False, unit_size=64),
    context=context,
)
print("REDUCE_SORTING_AOT_PASS", artifact.unit.spec.op_id)
PY
```

Run sorting, reduction, and the full numerical path strictly:

```bash
source /opt/venv/bin/activate
set -o pipefail

AITER_CONFIG_FMOE="$REDUCE_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/reduce-tuned-op-bench.csv" \
AITER_USE_FLYDSL_MOE_SORTING=1 \
AITER_USE_CK_MOE_SORTING=0 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$REDUCE_CACHE" \
FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_2stage.py --no-legacy \
  |& tee "$WORK_DIR/reduce-strict-runtime.log"
status=${PIPESTATUS[0]}
test "$status" -eq 0

! grep -Eq "AotCacheMissError|no usable AOT cache" \
  "$WORK_DIR/reduce-strict-runtime.log"
```

## Masked EP direct compile/load

A one-GPU numerical run does not reproduce distributed expert-parallel routing.
Compile and strictly load the masked graph directly, while CPU tests cover its
ordered graph and exact ABI:

```bash
source /opt/venv/bin/activate

export MASKED_CACHE="$WORK_DIR/flydsl-masked-cache"
mkdir -p "$MASKED_CACHE"

AITER_AOT_IMPORT=1 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$MASKED_CACHE" \
  python - <<'PY'
from aiter.ops.flydsl.aot_backend import (
    compile_aot,
    create_compile_context,
    load_aot,
)
from aiter.ops.flydsl.compile_plan import RocmTarget
from aiter.ops.flydsl.moe_compile_plan import resolve_moe_stage2_compile_plan

target = RocmTarget("gfx950", 256)
compile_context = create_compile_context(target)
plan = resolve_moe_stage2_compile_plan(
    context=compile_context,
    model_dim=3072,
    inter_dim=512,
    experts=32,
    topk=4,
    tile_m=64,
    tile_n=128,
    tile_k=256,
    doweight_stage2=True,
    a_dtype="fp4",
    b_dtype="fp4",
    out_dtype="bf16",
    mode="reduce",
    accumulate=False,
    return_per_slot=False,
    persist=None,
    token_num=256,
    routing_block_count=None,
    dtype_str="bf16",
    use_mask=True,
    topk_ids_available=True,
    num_experts=256,
    sort_block_m=0,
    waves_per_eu=None,
    use_async_copy=False,
    cu_num_mul=1,
    b_nt=2,
    model_dim_pad=0,
    inter_dim_pad=0,
    xcd_swizzle=0,
    enable_bias=False,
)
expected = [
    "aiter.flydsl.moe.stage2.mixed_gemm.v1",
    "aiter.flydsl.moe.stage2.reduction.masked.v1",
]
assert [unit.spec.op_id for unit in plan.units] == expected
assert dict(plan.units[0].spec.call.arguments)["persist_m"] == -1
[compile_aot(unit, context=compile_context) for unit in plan.units]

load_context = create_compile_context(target, strict_runtime=True)
loaded = [
    load_aot(unit, context=load_context, strict=True)
    for unit in plan.units
]
assert all(artifact.loaded for artifact in loaded)
print("MASKED_DIRECT_COMPILE_LOAD_PASS", len(loaded))
PY
```

## Empty-cache negative control

The same strict runtime command must fail against a genuinely empty cache:

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
grep -Eq "AotCacheMissError|no usable AOT cache" \
  "$WORK_DIR/empty-cache.log"
```

The structured exception identifies the missing op ID, `gfx950/256` target,
kernel signature, and cache directory. It must not invoke `compile_aot`.

## Expected operation IDs

Sorting plans emit exactly one of:

- `aiter.flydsl.moe.sorting.oneshot.v1`
- `aiter.flydsl.moe.sorting.multiphase.p0v2_p23.v1`
- `aiter.flydsl.moe.sorting.multiphase.k4_fused.v1`

Stage2 plans use:

- `aiter.flydsl.moe.stage2.mixed_gemm.v1`
- `aiter.flydsl.moe.stage2.int4_gemm.v1`
- `aiter.flydsl.moe.stage2.reduction.plain.v1`
- `aiter.flydsl.moe.stage2.reduction.masked.v1`

Stage1 operation IDs are unchanged.

## Aggregate CPU-only plan

`MoeCompilePlanCase` accepts already-bound Stage1 and Stage2 plans and requires
the caller to say explicitly whether sorting is present. The aggregate is only
an ordered input to future Manifest generation; it does not serialize,
deduplicate, or package artifacts.

```python
from aiter.ops.flydsl.moe_compile_plan import (
    MoeCompilePlanCase,
    MoeSortingCompileCase,
    resolve_moe_compile_plan,
)

# stage1_plan and stage2_plan come from the existing Stage1/Stage2 providers
# using the same explicit CompileContext.
aggregate = resolve_moe_compile_plan(
    MoeCompilePlanCase(
        sorting=MoeSortingCompileCase(128, 256, 8, False, unit_size=32),
        stage1=stage1_plan,
        stage2=stage2_plan,
    ),
    context=compile_context,
)
assert aggregate.units[0].spec.op_id.startswith("aiter.flydsl.moe.sorting.")

without_sorting = resolve_moe_compile_plan(
    MoeCompilePlanCase(
        sorting=None,
        stage1=stage1_plan,
        stage2=stage2_plan,
    ),
    context=compile_context,
)
```

## Cleanup

```bash
if [[ -n "${WORK_DIR:-}" && -d "$WORK_DIR" ]]; then
  rm -rf -- "$WORK_DIR"
fi

unset WORK_DIR CACHE_DIR ONE_ROW_CSV SORTING_CACHE EMPTY_SORTING_CACHE
unset REDUCE_CSV REDUCE_CACHE MASKED_CACHE EMPTY_CACHE
unset AITER_CONFIG_FMOE AITER_TUNED_OP_BENCH_CSV
unset AITER_USE_FLYDSL_MOE_SORTING AITER_USE_CK_MOE_SORTING
unset AITER_AOT_IMPORT AITER_FLYDSL_AOT_WORKERS
unset ARCH COMPILE_ONLY FLYDSL_GPU_ARCH FLYDSL_RUNTIME_RUN_ONLY
unset FLYDSL_RUNTIME_CACHE_DIR AITER_FLYDSL_FORCE_REDUCE
unset GPU_ARCHS CU_NUM HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_INDEX
```

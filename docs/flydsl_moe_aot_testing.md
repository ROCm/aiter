# FlyDSL MoE compile-request AOT testing

## Scope

The standard two-stage MoE path shares only CPU-only compile decisions and
immutable `CompileRequest` values between runtime and AOT:

- `stage1_compile_requests(metadata, target)`
- `stage2_compile_requests(metadata, runtime_metadata, target)`
- `sorting_compile_request(case, target)`
- `cktile_epilogue_compile_requests(metadata, target)`

Each request carries a stable operation ID, explicit `RocmTarget`, normalized
builder kwargs, and a `KernelSignature`. `AotBackend` materializes ABI metadata
without allocating tensors or constructing FakeTensors. Runtime retains tensor
allocation, argument packing, grids, streams, and launches.

Ordinary Stage1/Stage2 CSV rows never infer sorting. Sorting remains an explicit
`MoeSortingCompileCase` API.

This implementation does not yet provide a Manifest, artifact packaging,
cross-process deduplication, or a public FlyDSL save/load API. It uses the
FlyDSL 0.2.x disk cache as a compatibility boundary.

## CPU checks

Activate the project virtual environment before Python commands:

```bash
source /opt/venv/bin/activate

AITER_AOT_IMPORT=1 python -m pytest -q aiter/aot/flydsl/tests

black --check \
  aiter/ops/flydsl/compile_request.py \
  aiter/ops/flydsl/moe_compile_requests.py \
  aiter/ops/flydsl/aot_backend.py \
  aiter/ops/flydsl/moe_kernels.py \
  aiter/ops/flydsl/moe_sorting.py \
  aiter/aot/flydsl/moe.py \
  aiter/aot/flydsl/tests
ruff format --check \
  aiter/ops/flydsl/compile_request.py \
  aiter/ops/flydsl/moe_compile_requests.py \
  aiter/ops/flydsl/aot_backend.py \
  aiter/ops/flydsl/moe_kernels.py \
  aiter/ops/flydsl/moe_sorting.py \
  aiter/aot/flydsl/moe.py \
  aiter/aot/flydsl/tests
ruff check \
  aiter/ops/flydsl/compile_request.py \
  aiter/ops/flydsl/moe_compile_requests.py \
  aiter/ops/flydsl/aot_backend.py \
  aiter/ops/flydsl/moe_kernels.py \
  aiter/ops/flydsl/moe_sorting.py \
  aiter/aot/flydsl/moe.py \
  aiter/aot/flydsl/tests
git diff --check
```

The golden contract is
`aiter/aot/flydsl/tests/data/moe_compile_requests_gfx950.json`. Runtime recording
and direct request-factory tests independently compare builder identity, fully
defaulted kwargs, order, operation IDs, and ABI declarations against it.

## Request API example

```python
from aiter.ops.flydsl.aot_backend import compile_aot, create_compile_context
from aiter.ops.flydsl.compile_request import RocmTarget
from aiter.ops.flydsl.moe_compile_requests import stage1_compile_requests

target = RocmTarget("gfx950", 256)
context = create_compile_context(target)
requests = stage1_compile_requests(stage1_metadata, target)
artifacts = [
    compile_aot(request, context=context)
    for request in requests
]
```

Explicit sorting uses:

```python
from aiter.aot.flydsl.moe import compile_moe_sorting_case
from aiter.ops.flydsl.moe_compile_requests import MoeSortingCompileCase

artifact, = compile_moe_sorting_case(
    MoeSortingCompileCase(128, 384, 8, False),
    context=context,
)
print(artifact.request.op_id)
```

## Bounded standard gfx950 strict test

This check covers only the standard `flydsl_moe1` and `flydsl_moe2` path. It
does not cover or imply `flydsl_mxmoe` support.

Choose one idle physical GPU and create a dedicated temporary cache:

```bash
source /opt/venv/bin/activate

export GPU_INDEX="${GPU_INDEX:-0}"
export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export GPU_ARCHS=gfx950
export CU_NUM=256

unset ARCH COMPILE_ONLY FLYDSL_GPU_ARCH FLYDSL_RUNTIME_RUN_ONLY
unset AITER_CONFIG_FMOE AITER_FLYDSL_FORCE_REDUCE

export WORK_DIR
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aiter-compile-request-e2e-XXXXXX")"
export CACHE_DIR="$WORK_DIR/flydsl-cache"
export EMPTY_CACHE="$WORK_DIR/empty-cache"
export ONE_ROW_CSV="$WORK_DIR/gfx950-one-row.csv"
mkdir -p "$CACHE_DIR" "$EMPTY_CACHE"

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
```

Generate the documented standard row from the merged runtime config:

```bash
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

Compile from an empty isolated cache:

```bash
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
test "${PIPESTATUS[0]}" -eq 0

rg -q "direct_stage1_requests=1" "$WORK_DIR/aot.log"
rg -q "direct_stage2_requests=1" "$WORK_DIR/aot.log"
rg -q "All compilations succeeded" "$WORK_DIR/aot.log"
! rg -q "\[FAIL\]" "$WORK_DIR/aot.log"
```

Run the matching standard path strictly. FlyDSL sorting is disabled here so the
test is specifically Stage1/Stage2 coverage:

```bash
set -o pipefail
AITER_CONFIG_FMOE="$ONE_ROW_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/tuned-op-bench.csv" \
AITER_USE_FLYDSL_MOE_SORTING=0 \
GPU_ARCHS=gfx950 \
CU_NUM=256 \
HIP_VISIBLE_DEVICES="$GPU_INDEX" \
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
FLYDSL_RUNTIME_CACHE_DIR="$CACHE_DIR" \
FLYDSL_RUNTIME_RUN_ONLY=1 \
  python op_tests/test_moe_2stage.py --no-legacy \
  |& tee "$WORK_DIR/strict-runtime.log"
test "${PIPESTATUS[0]}" -eq 0

rg -q "check_aot_cache[[:space:]]*=[[:space:]]*True" \
  "$WORK_DIR/strict-runtime.log"
! rg -q "output contains NaN|has_nan[[:space:]]*=[[:space:]]*True" \
  "$WORK_DIR/strict-runtime.log"
! rg -q "AotCacheMissError|no usable AOT cache" \
  "$WORK_DIR/strict-runtime.log"
```

The repository correctness gate permits elementwise differences only when its
existing logits criterion passes. Require process exit zero, no NaN, and no
strict cache miss; do not claim elementwise allclose unless the log does.

An empty-cache strict run must fail structurally:

```bash
if AITER_CONFIG_FMOE="$ONE_ROW_CSV" \
AITER_TUNED_OP_BENCH_CSV="$WORK_DIR/negative-tuned-op.csv" \
AITER_USE_FLYDSL_MOE_SORTING=0 \
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

test "$status" -ne 0
rg -q "AotCacheMissError|no usable AOT cache" "$WORK_DIR/empty-cache.log"
echo "EMPTY_CACHE_NEGATIVE_PASS status=$status"
```

The structured exception includes the missing operation ID, target, ABI
signature, and cache directory and does not fall back to JIT.

## Cleanup

```bash
if [[ -n "${WORK_DIR:-}" && -d "$WORK_DIR" ]]; then
  rm -rf -- "$WORK_DIR"
fi

unset WORK_DIR CACHE_DIR EMPTY_CACHE ONE_ROW_CSV GPU_INDEX
unset AITER_CONFIG_FMOE AITER_TUNED_OP_BENCH_CSV
unset AITER_USE_FLYDSL_MOE_SORTING AITER_AOT_IMPORT
unset AITER_FLYDSL_AOT_WORKERS ARCH COMPILE_ONLY FLYDSL_GPU_ARCH
unset FLYDSL_RUNTIME_RUN_ONLY FLYDSL_RUNTIME_CACHE_DIR
unset GPU_ARCHS CU_NUM HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

# Triton Upgrade & GEMM Kernel Tuning Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an orchestrated pipeline to baseline GEMM kernels on Triton 3.4, tune them for latest Triton with LDS-aware config filtering, and validate performance parity.

**Architecture:** A set of Python scripts under `aiter/ops/triton/utils/_triton/tunning/` that orchestrate the existing `screen.py`/`view-screen.py`/`verify-perf.py` tuning infrastructure. New `ut_*.py` scripts fill gaps for kernels lacking tuning harnesses. An LDS filter pre-prunes the config search space to avoid out-of-resources failures with latest Triton's async copy.

**Tech Stack:** Python 3.12, PyTorch, Triton, rocprofv3, AMD MI355X (gfx950)

**Spec:** `docs/superpowers/specs/2026-03-17-triton-upgrade-gemm-tuning-design.md`

---

## File Map

### New Tuning Scripts (7 files)
All under `aiter/ops/triton/utils/_triton/tunning/`:

| File | Responsibility |
|------|---------------|
| `ut_a16w16_gemm_gated.py` | Tuning harness for gated A16W16 GEMM |
| `ut_a16w16_gemm_atomic.py` | Tuning harness for atomic A16W16 GEMM |
| `ut_a16w16_gemm_agnostic.py` | Tuning harness for agnostic A16W16 GEMM |
| `ut_a16wfp4_gemm.py` | Tuning harness for A16WFP4 GEMM |
| `ut_a8wfp4_gemm.py` | Tuning harness for A8WFP4 GEMM |
| `ut_afp4wfp4_gemm_pre_quant_atomic.py` | Tuning harness for AFP4WFP4 pre-quant atomic GEMM |
| `ut_a16w8_gemm_blockscale.py` | Tuning harness for A16W8 blockscale GEMM (non-preshuffle) |

### Orchestration Scripts (6 files)
All under `aiter/ops/triton/utils/_triton/tunning/`:

| File | Responsibility |
|------|---------------|
| `collect_shapes.py` | Collects (M,N,K) shapes from configs, model_shapes.json, test scripts |
| `lds_filter.py` | Computes LDS-safe block-size ranges per kernel given 160KB limit |
| `collect_baseline.py` | Runs rocprofv3 benchmarks on current Triton, parses kernel_trace CSV |
| `run_tuning.py` | Dispatches `screen.py` across multiple GPUs with a work queue |
| `compare_results.py` | Compares baseline vs new results, generates regression report |
| `orchestrate.py` | Top-level CLI that drives the full baseline → tune → validate pipeline |

---

## Chunk 1: New Tuning Scripts

These 7 files follow the identical pattern established by `ut_template.py` and existing scripts like `ut_a8w8_gemm.py`. Each is ~30-45 lines.

### Task 1: ut_a16w16_gemm_gated.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm_gated.py`

- [ ] **Step 1: Write the tuning script**

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_a16w16_gated import gemm_a16w16_gated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16_gated import (
    generate_gemm_a16w16_gated_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# Returns (x, weight, out_dtype, y) — 4 values
x, w, _, y = generate_gemm_a16w16_gated_inputs(
    M, N, K,
    dtype,
    output=True,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    def fn():
        ############################################################
        # <run API>
        gemm_a16w16_gated(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
```

- [ ] **Step 2: Verify script loads correctly**

Run from the `tunning/` directory:
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_a16w16_gemm_gated.py 64 4096 7168
```
Expected: Script runs the kernel 250 times then exits (or errors on missing `config` param, which is fine — confirms imports work).

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm_gated.py
git commit -m "feat(tunning): add ut_a16w16_gemm_gated.py tuning script"
```

### Task 2: ut_a16w16_gemm_atomic.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm_atomic.py`
- Reference: `aiter/ops/triton/gemm/basic/gemm_a16w16_atomic.py:111` — the public API is `gemm_a16w16_atomic(x, w, dtype, y, config)` (NO bias parameter)

- [ ] **Step 1: Write the tuning script**

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# gemm_a16w16_atomic does NOT take bias — only (x, w, dtype, y, config)
x, w, _, _, y = generate_gemm_a16w16_inputs(
    M, N, K,
    dtype,
    output=True,
    bias=False,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    def fn():
        ############################################################
        # <run API>
        gemm_a16w16_atomic(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
```

- [ ] **Step 2: Verify script loads correctly**
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_a16w16_gemm_atomic.py 64 4096 7168
```

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm_atomic.py
git commit -m "feat(tunning): add ut_a16w16_gemm_atomic.py tuning script"
```

### Task 3: ut_a16w16_gemm_agnostic.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm_agnostic.py`
- Reference: `aiter/ops/triton/gemm/basic/gemm_a16w16_agnostic.py` — public API is `gemm_a16w16(x, w, dtype, y, config=config)`

- [ ] **Step 1: Write the tuning script**

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_a16w16_agnostic import gemm_a16w16
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
x, w, bias, _, y = generate_gemm_a16w16_inputs(
    M, N, K,
    dtype,
    output=True,
    bias=False,
)
############################################################

for config in config_list:
    if config is not None:
        config = config.copy()
        config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    def fn():
        ############################################################
        # <run API>
        gemm_a16w16(x, w, dtype, y, config=config)
        ############################################################

    run_profile(fn)
```

- [ ] **Step 2: Verify script loads correctly**
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_a16w16_gemm_agnostic.py 64 4096 7168
```

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_a16w16_gemm_agnostic.py
git commit -m "feat(tunning): add ut_a16w16_gemm_agnostic.py tuning script"
```

### Task 4: ut_a16wfp4_gemm.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_a16wfp4_gemm.py`
- Reference: `aiter/ops/triton/gemm/basic/gemm_a16wfp4.py:176` — `gemm_a16wfp4(x, w, w_scales, atomic_add, dtype, y, config)`
- Reference: `op_tests/triton_tests/gemm/basic/test_gemm_a16wfp4.py:15` — `generate_gemm_a16wfp4_inputs(M, N, K, output, atomic_add, dtype, layout, shuffle)` returns `(x, w, w_shuffled, x_scales, w_scales, w_scales_shuffled, y)`

- [ ] **Step 1: Write the tuning script**

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_a16wfp4 import (
    generate_gemm_a16wfp4_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
# Signature: generate_gemm_a16wfp4_inputs(M, N, K, output, atomic_add, dtype, layout, shuffle)
# Returns: (x, w, w_shuffled, x_scales, w_scales, w_scales_shuffled, y)
x, w, _, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
    M, N, K,
    output=True,
    atomic_add=False,
    dtype=dtype,
    layout="TN",
    shuffle=False,
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        # Signature: gemm_a16wfp4(x, w, w_scales, atomic_add, dtype, y, config)
        gemm_a16wfp4(x, w, w_scales, False, dtype, y, config=config)
        ############################################################

    run_profile(fn)
```

- [ ] **Step 2: Verify script loads correctly**
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_a16wfp4_gemm.py 64 4096 7168
```

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_a16wfp4_gemm.py
git commit -m "feat(tunning): add ut_a16wfp4_gemm.py tuning script"
```

### Task 5: ut_a8wfp4_gemm.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_a8wfp4_gemm.py`
- Reference: `aiter/ops/triton/gemm/basic/gemm_a8wfp4.py:26` — `gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=config)`
- Reference: `op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py:49` — `generate_gemm_a8wfp4_inputs`

- [ ] **Step 1: Write the tuning script**

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_a8wfp4 import (
    generate_gemm_a8wfp4_inputs,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
_, e4m3_type = get_fp8_dtypes()
dtype = torch.float16
x, w, x_scales, w_scales, _, _, y = generate_gemm_a8wfp4_inputs(
    M, N, K,
    e4m3_type,
    dtype,
    layout="TN",
    output=True,
)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_a8wfp4(x, w, y, x_scales, w_scales, dtype, config=config)
        ############################################################

    run_profile(fn)
```

**Important**: Read `test_gemm_a8wfp4.py:49-120` to confirm the return value order of `generate_gemm_a8wfp4_inputs`. The function returns `(x, w, x_scales, w_scales, x_fp32, w_fp32, y)` — adjust the unpacking to match.

- [ ] **Step 2: Verify script loads correctly**
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_a8wfp4_gemm.py 64 4096 7168
```

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_a8wfp4_gemm.py
git commit -m "feat(tunning): add ut_a8wfp4_gemm.py tuning script"
```

### Task 6: ut_afp4wfp4_gemm_pre_quant_atomic.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_afp4wfp4_gemm_pre_quant_atomic.py`
- Reference: `aiter/ops/triton/gemm/basic/gemm_afp4wfp4_pre_quant_atomic.py` — `gemm_afp4wfp4_pre_quant(x, w, w_scale, dtype, y)`
- **NOTE**: `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4_pre_quant_atomic.py` does NOT exist in the repo. The bench script `bench_gemm_afp4wfp4_pre_quant_atomic.py` imports from it, suggesting it's a planned file. We must either create a minimal test/input-generator or inline the input generation.

- [ ] **Step 1: Write the tuning script with inline input generation**

Since the test file doesn't exist, generate inputs inline based on the bench script pattern at `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4_pre_quant_atomic.py`:

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
    gemm_afp4wfp4_pre_quant,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
# Inline generation since test_gemm_afp4wfp4_pre_quant_atomic.py does not exist.
# Read gemm_afp4wfp4_pre_quant_atomic.py and the bench script to determine
# the correct input format (bf16 activations, packed fp4 weights, e8m0 scales).
# Then generate accordingly. This is a placeholder — adapt after reading the kernel source:
dtype = torch.float32
SCALE_GROUP_SIZE = 32
x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
w = w_low | (w_high << 4)
w_scale = torch.randint(124, 128, (N, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda")
y = torch.zeros((M, N), device="cuda", dtype=dtype)
############################################################

for config in config_list:

    def fn():
        ############################################################
        # <run API>
        gemm_afp4wfp4_pre_quant(x, w, w_scale, dtype, y, config=config)
        ############################################################

    run_profile(fn)
```

**Important**: Before running, read `gemm_afp4wfp4_pre_quant_atomic.py` to confirm the exact function signature and expected input shapes/dtypes. The inline generation above is a best-guess based on the bench script. Also verify `config=` kwarg is accepted.

- [ ] **Step 2: Verify script loads correctly**
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_afp4wfp4_gemm_pre_quant_atomic.py 64 4096 7168
```

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_afp4wfp4_gemm_pre_quant_atomic.py
git commit -m "feat(tunning): add ut_afp4wfp4_gemm_pre_quant_atomic.py tuning script"
```

### Task 7: ut_a16w8_gemm_blockscale.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/ut_a16w8_gemm_blockscale.py`
- Reference: `aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py:21` — `gemm_a16w8_blockscale(..., config=config)`
- Reference: `op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py:105` — `generate_gemm_a16w8_blockscale_inputs`

- [ ] **Step 1: Write the tuning script**

Modeled directly on the existing `ut_a16w8_gemm_blockscale_preshuffle.py`, but using `gemm_a16w8_blockscale` (non-preshuffle) and `shuffle=False`.

Actual signatures (verified from source):
- `generate_gemm_a16w8_blockscale_inputs(M, N, K, block_shape_n, block_shape_k, dtype, layout, output, shuffle)` returns `(x, weight, weight_shuffled, w_scale, y)` — 5 values
- `gemm_a16w8_blockscale(x, w, w_scale, dtype, y, prequant, config, skip_reduce)` — NO x_scale, NO bias

```python
import sys
from _utils import (
    run_profile,
    get_input_shape_and_config_list,
)

############################################################
# <import>
import torch
from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import gemm_a16w8_blockscale
from op_tests.triton_tests.gemm.basic.test_gemm_a16w8_blockscale import (
    generate_gemm_a16w8_blockscale_inputs,
)

############################################################

input_shape, config_list = get_input_shape_and_config_list(sys.argv, shape_size=3)
M, N, K = input_shape

############################################################
# <generate input>
dtype = torch.bfloat16
shuffle = False
block_shape_n, block_shape_k = 128, 128
# Returns: (x, weight, weight_shuffled, w_scale, y) — 5 values
x, weight, weight_triton, w_scale, y = generate_gemm_a16w8_blockscale_inputs(
    *input_shape,
    block_shape_n,
    block_shape_k,
    dtype=dtype,
    output=True,
    shuffle=shuffle,
)
############################################################

for config in config_list:
    assert config is None or config["BLOCK_SIZE_K"] == 128

    def fn():
        ############################################################
        # <run API>
        gemm_a16w8_blockscale(
            x, weight_triton, w_scale, dtype, y, prequant=False, config=config
        )
        ############################################################

    run_profile(fn)
```

- [ ] **Step 2: Verify script loads correctly**
```bash
cd aiter/ops/triton/utils/_triton/tunning/
python ut_a16w8_gemm_blockscale.py 64 4096 7168
```

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/ut_a16w8_gemm_blockscale.py
git commit -m "feat(tunning): add ut_a16w8_gemm_blockscale.py tuning script"
```

---

## Chunk 2: Core Orchestration Utilities

### Task 8: collect_shapes.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/collect_shapes.py`

This script collects all (M,N,K) shapes for a given kernel from three sources.

- [ ] **Step 1: Write `collect_shapes.py`**

The script should:
1. Accept `--kernel <name>` (e.g., `a8w8`, `a16w16_gated`, `all`) and `--output-dir <path>`
2. Parse config files: glob `aiter/ops/triton/configs/gemm/gfx950-GEMM-*.json`, extract N,K from filenames with regex `N=(\d+)-K=(\d+)`, extract M values from JSON keys (`M_LEQ_(\d+)` -> that number, `any` -> standard M list)
3. Parse model shapes: read `model_shapes.json`, for each model find keys matching the kernel name, extract N,K, generate M values from standard batch sizes `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`
4. Round all M values to next power of 2 (for `screen.py` compatibility)
5. Deduplicate and sort
6. Write `results/shapes/shapes_gemm_<kernel>.json`

Key mapping from kernel names to config file patterns and model_shapes.json keys:
```python
KERNEL_CONFIG_MAP = {
    "a16w16": {"config_pattern": "A16W16", "model_key": "gemm_a16w16"},
    "a8w8": {"config_pattern": "A8W8", "model_key": None},  # not in model_shapes
    "a8w8_blockscale": {"config_pattern": "A8W8_BLOCKSCALE", "model_key": "gemm_a8w8_blockscale"},
    "a8w8_per_token_scale": {"config_pattern": "A8W8_PER_TOKEN_SCALE", "model_key": "gemm_a8w8_per_token_scale"},
    "afp4wfp4": {"config_pattern": "AFP4WFP4", "model_key": "gemm_afp4wfp4"},
    # ... etc for all 12 kernels
}
```

- [ ] **Step 2: Test shape collection for a known kernel**

```bash
cd aiter/ops/triton/utils/_triton/tunning/
python collect_shapes.py --kernel a8w8 --output-dir results/shapes/
cat results/shapes/shapes_gemm_a8w8.json | python -m json.tool | head -20
```
Expected: JSON array of `{"M": ..., "N": ..., "K": ...}` objects.

- [ ] **Step 3: Test `--kernel all`**

```bash
python collect_shapes.py --kernel all --output-dir results/shapes/
ls results/shapes/
```
Expected: One `shapes_gemm_<kernel>.json` file per kernel.

- [ ] **Step 4: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/collect_shapes.py
git commit -m "feat(tunning): add collect_shapes.py for shape collection"
```

### Task 9: lds_filter.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/lds_filter.py`

- [ ] **Step 1: Write `lds_filter.py`**

The script should:
1. Accept `--kernel <name>` and `--num-stages <int list>` (default `[2, 3]`; also accepts single value)
2. Define per-kernel parameters table (from spec):
   ```python
   KERNEL_PARAMS = {
       "a16w16": {"a_dtype_size": 2, "w_dtype_size": 2, "scale_overhead": lambda bm,bn,bk: 0, "bk_constraint": None, "preshuffle_operand": None},
       "a8w8_blockscale": {"a_dtype_size": 1, "w_dtype_size": 1, "scale_overhead": lambda bm,bn,bk: (bm*bk//128 + bn*bk//128), "bk_constraint": [128], "preshuffle_operand": None},
       # ... all 12 kernels from spec table
   }
   ```
3. Enumerate `(BM, BN, BK)` from screen.py defaults: BM in `[4,8,16,32,64,128,256,512]`, BN in `[16,32,64,128,256]`, BK in `[128,256,512,1024]`
4. For each combo, compute: `lds_a = BM * BK * a_dtype_size` (0 if preshuffle_operand == "A"), `lds_w = BN * BK * w_dtype_size` (0 if preshuffle_operand == "W"), `lds_total = (lds_a + lds_w) * num_stages + scale_overhead(BM, BN, BK)`
5. Filter: keep only where `lds_total <= 163840` AND BK satisfies `bk_constraint`
6. Output: the filtered BM, BN, BK ranges as space-separated values suitable for `screen.py` CLI args
7. Also support `--print-cli` mode that outputs `--block-size-m-range ... --block-size-n-range ... --block-size-k-range ...`

- [ ] **Step 2: Test LDS filter for a8w8 (should allow large blocks)**

```bash
cd aiter/ops/triton/utils/_triton/tunning/
python lds_filter.py --kernel a8w8 --num-stages 2 --print-cli
```
Expected: BM up to 256+, BN up to 256, BK up to 512+ (fp8 is small, 1 byte each).

- [ ] **Step 3: Test LDS filter for a16w16 (should be more restrictive)**

```bash
python lds_filter.py --kernel a16w16 --num-stages 2 --print-cli
```
Expected: Smaller ranges since bf16 is 2 bytes each — should exclude the largest (BM=256, BN=256, BK=256) combos.

- [ ] **Step 4: Test LDS filter for a8w8_blockscale (BK=128 only)**

```bash
python lds_filter.py --kernel a8w8_blockscale --num-stages 2 --print-cli
```
Expected: Only BK=128 in the output.

- [ ] **Step 5: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/lds_filter.py
git commit -m "feat(tunning): add lds_filter.py for LDS-aware config filtering"
```

### Task 10: collect_baseline.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/collect_baseline.py`

- [ ] **Step 1: Write `collect_baseline.py`**

The script should:
1. Accept `--kernel <name>`, `--shapes-file <path>`, `--output <path>`, `--gpu <int>`, `--bench-dir <path>` (path to `op_tests/op_benchmarks/triton/`)
2. For each shape in the shapes file, run:
   ```
   HIP_VISIBLE_DEVICES=<gpu> rocprofv3 --kernel-trace -f csv -o <output_dir>/trace_<kernel>_<M>_<N>_<K> -- python <bench_script> --shape <M> <N> <K> --metric time --layout TN
   ```
3. If no bench script exists for the kernel, fall back to the ut script:
   ```
   HIP_VISIBLE_DEVICES=<gpu> rocprofv3 --kernel-trace -f csv -o <output_dir>/trace_<kernel>_<M>_<N>_<K> -- python <ut_script> <M> <N> <K>
   ```
4. Parse the `*_kernel_trace.csv` using the pattern from `rprof.py`: filter by kernel name substring, compute p50 runtime, sum kernel + reduce kernel if split-K
5. Kernel name patterns per kernel type (e.g., `_gemm_a8w8_kernel` for a8w8, `_gemm_a16w16_kernel` for a16w16, etc.)
6. Write results to JSON: `{"kernel": {"M-N-K": {"time_ns": ..., "kernel_name": ...}}}`

Mapping from kernel name to bench script and kernel name pattern:
```python
KERNEL_BENCH_MAP = {
    "a8w8": {"bench": "bench_gemm_a8w8.py", "ut": "ut_a8w8_gemm.py", "kernel_pattern": "_gemm_a8w8_kernel"},
    "a16w16": {"bench": "bench_gemm_a16w16.py", "ut": "ut_a16w16_gemm.py", "kernel_pattern": "_gemm_a16w16_kernel"},
    # ... all 12 kernels
}
```

- [ ] **Step 2: Test baseline collection for one kernel, one shape**

```bash
cd aiter/ops/triton/utils/_triton/tunning/
echo '[{"M": 64, "N": 4096, "K": 7168}]' > /tmp/test_shapes.json
python collect_baseline.py --kernel a8w8 --shapes-file /tmp/test_shapes.json --output results/baseline/ --gpu 0
cat results/baseline/baseline_a8w8.json
```
Expected: JSON with `"64-4096-7168": {"time_ns": <some number>, ...}`.

- [ ] **Step 3: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/collect_baseline.py
git commit -m "feat(tunning): add collect_baseline.py for rocprofv3 benchmarking"
```

### Task 11: run_tuning.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/run_tuning.py`

- [ ] **Step 1: Write `run_tuning.py`**

The script should:
1. Accept `--kernel <name>`, `--shapes-file <path>`, `--gpus 0-7` (or `0,1,2`), `--output-dir <path>`, `--lds-args <string>` (pre-computed by lds_filter.py), `--dry-run`
2. Build work queue: list of `(M, N, K)` from shapes file
3. For each shape, construct the screen.py command:
   ```
   HIP_VISIBLE_DEVICES=<gpu> python screen.py <M> <N> <K> 0 ut_<kernel>.py <lds_args> --num-stages-range 2 3
   ```
   (GPU ID passed to screen.py is always 0 since HIP_VISIBLE_DEVICES remaps. `--num-stages-range 2 3` overrides screen.py's default `[1, 2]` per spec recommendation.)
4. Launch up to `len(gpus)` concurrent subprocesses
5. Use a simple process pool: when a process completes, launch next item from queue
6. Track progress in `progress.json` — skip completed items on resume
7. After all shapes complete, collect (N,K) pairs and run `view-screen.py` to generate JSON configs
8. `--dry-run` prints the work queue and estimated time without executing

- [ ] **Step 2: Test dry-run mode**

```bash
cd aiter/ops/triton/utils/_triton/tunning/
python run_tuning.py --kernel a8w8 --shapes-file results/shapes/shapes_gemm_a8w8.json --gpus 0,1 --output-dir results/tuning/ --lds-args "--block-size-m-range 16 32 64 128 --block-size-n-range 32 64 128 --block-size-k-range 128 256" --dry-run
```
Expected: Prints list of commands that would be run, grouped by GPU.

- [ ] **Step 3: Test single shape tuning**

```bash
echo '[{"M": 64, "N": 4096, "K": 7168}]' > /tmp/test_shapes.json
python run_tuning.py --kernel a8w8 --shapes-file /tmp/test_shapes.json --gpus 0 --output-dir results/tuning/ --lds-args "--block-size-k-range 128 256"
```
Expected: screen.py runs for M=64, N=4096, K=7168 and produces log file.

- [ ] **Step 4: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/run_tuning.py
git commit -m "feat(tunning): add run_tuning.py for parallel GPU dispatch"
```

### Task 12: compare_results.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/compare_results.py`

- [ ] **Step 1: Write `compare_results.py`**

The script should:
1. Accept `--baseline <path>` (JSON), `--new <path>` (JSON), `--output <path>` (report text file), `--regression-threshold 0.03`
2. For each kernel present in both files, for each shape:
   - Compute delta: `(new_time - baseline_time) / baseline_time`
   - Flag if delta > threshold
3. Compute geomean of `(baseline_time / new_time)` across all shapes per kernel
4. Print and write report in the format from the spec:
   ```
   === gemm_a8w8 ===
   Shape           Baseline(ns)  New(ns)   Delta    Status
   16-4096-7168    12345         11800     -4.4%    OK
   ...
   Geomean ratio: 1.02x (PASS)
   ```
5. Exit code 0 if all kernels pass, 1 if any fails

- [ ] **Step 2: Test with mock data**

Create two small JSON files with known values and verify the report output:
```bash
cd aiter/ops/triton/utils/_triton/tunning/
echo '{"gemm_a8w8": {"64-4096-7168": {"time_ns": 1000}}}' > /tmp/baseline.json
echo '{"gemm_a8w8": {"64-4096-7168": {"time_ns": 950}}}' > /tmp/new.json
python compare_results.py --baseline /tmp/baseline.json --new /tmp/new.json --output /tmp/report.txt
cat /tmp/report.txt
```
Expected: Shows -5.0% improvement, PASS.

- [ ] **Step 3: Test regression detection**

```bash
echo '{"gemm_a8w8": {"64-4096-7168": {"time_ns": 1000}}}' > /tmp/baseline.json
echo '{"gemm_a8w8": {"64-4096-7168": {"time_ns": 1100}}}' > /tmp/new.json
python compare_results.py --baseline /tmp/baseline.json --new /tmp/new.json --output /tmp/report.txt
echo "Exit code: $?"
```
Expected: Shows +10.0% REGRESSION, exit code 1.

- [ ] **Step 4: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/compare_results.py
git commit -m "feat(tunning): add compare_results.py for performance comparison"
```

---

## Chunk 3: Top-Level Orchestrator

### Task 13: orchestrate.py

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/orchestrate.py`

- [ ] **Step 1: Write `orchestrate.py`**

This is the main entry point. It should:

1. Use `argparse` with subcommands: `baseline`, `tune`, `validate`, `full`
2. Common args: `--kernels <all|a8w8,a16w16,...>`, `--gpus <0-7|0,1,2>`, `--force`, `--dry-run`
3. `baseline` subcommand:
   - Run `collect_shapes.py` for each kernel
   - Run `collect_baseline.py` for each kernel, parallelized across GPUs
   - Merge per-kernel baseline JSONs into `baseline_triton34.json`
4. `tune` subcommand:
   - Run `collect_shapes.py` for each kernel (if not already done)
   - Run `lds_filter.py` for each kernel to get filtered block-size ranges
   - Run `run_tuning.py` for each kernel with filtered ranges
   - Copy generated JSON configs to `aiter/ops/triton/configs/gemm/`
5. `validate` subcommand:
   - Run `collect_baseline.py` for each kernel (same as baseline but on new Triton)
   - Run `compare_results.py` with baseline and new results
   - Print summary report
6. `full` subcommand: runs baseline → tune → validate in sequence, with `--skip-baseline` option
7. Track progress via `results/progress.json`

Parse `--gpus` argument:
- `0-7` -> `[0,1,2,3,4,5,6,7]`
- `0,2,4` -> `[0,2,4]`
- `0` -> `[0]`

Resolve kernel names:
- `all` -> list all kernels from `KERNEL_CONFIG_MAP`
- `a8w8,a16w16` -> split by comma

- [ ] **Step 2: Test `orchestrate.py baseline --dry-run`**

```bash
cd aiter/ops/triton/utils/_triton/tunning/
python orchestrate.py baseline --kernels a8w8 --gpus 0 --dry-run
```
Expected: Prints what would be run without executing.

- [ ] **Step 3: Test `orchestrate.py tune --dry-run`**

```bash
python orchestrate.py tune --kernels a8w8 --gpus 0 --dry-run
```
Expected: Prints the tuning work queue with LDS-filtered ranges.

- [ ] **Step 4: End-to-end test with one kernel, one shape**

Run the full pipeline for `a8w8` with a single small shape to verify the whole flow:
```bash
python orchestrate.py baseline --kernels a8w8 --gpus 0
# (install latest triton here if not already done)
python orchestrate.py tune --kernels a8w8 --gpus 0
python orchestrate.py validate --kernels a8w8 --gpus 0
cat results/report.txt
```

- [ ] **Step 5: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/orchestrate.py
git commit -m "feat(tunning): add orchestrate.py top-level pipeline driver"
```

### Task 14: Create results directory and .gitignore

**Files:**
- Create: `aiter/ops/triton/utils/_triton/tunning/results/.gitkeep`
- Create: `aiter/ops/triton/utils/_triton/tunning/results/.gitignore`

- [ ] **Step 1: Create results directory with gitignore**

```bash
mkdir -p aiter/ops/triton/utils/_triton/tunning/results
touch aiter/ops/triton/utils/_triton/tunning/results/.gitkeep
cat > aiter/ops/triton/utils/_triton/tunning/results/.gitignore << 'EOF'
# Ignore all results except this file
*
!.gitignore
!.gitkeep
EOF
```

- [ ] **Step 2: Commit**
```bash
git add aiter/ops/triton/utils/_triton/tunning/results/.gitkeep aiter/ops/triton/utils/_triton/tunning/results/.gitignore
git commit -m "feat(tunning): add results directory structure"
```

---

## Chunk 4: Integration Testing & Execution

### Task 15: Full Pipeline Smoke Test on Triton 3.4

- [ ] **Step 1: Run baseline collection for all basic GEMM kernels**

```bash
cd aiter/ops/triton/utils/_triton/tunning/
python orchestrate.py baseline --kernels all --gpus 0-7
```

Verify:
- `results/shapes/` has one file per kernel
- `results/baseline/baseline_triton34.json` has timing data
- No crashes or missing kernels

- [ ] **Step 2: Spot-check baseline numbers**

Pick a known kernel/shape combo and verify the time is reasonable:
```bash
python -c "
import json
d = json.load(open('results/baseline/baseline_triton34.json'))
for kernel, shapes in d.items():
    print(f'{kernel}: {len(shapes)} shapes')
    for shape, data in list(shapes.items())[:2]:
        print(f'  {shape}: {data[\"time_ns\"]:.0f} ns')
"
```

- [ ] **Step 3: Commit baseline results (optional, for reference)**

```bash
# Only if you want to preserve baseline in repo
cp results/baseline/baseline_triton34.json docs/superpowers/
git add docs/superpowers/baseline_triton34.json
git commit -m "data: capture Triton 3.4 baseline timings"
```

### Task 16: Install Latest Triton and Run Tuning

- [ ] **Step 1: Build and install latest Triton**

```bash
git clone https://github.com/triton-lang/triton.git /tmp/triton-latest
cd /tmp/triton-latest
pip install -e python
```

Verify: `python -c "import triton; print(triton.__version__)"`

- [ ] **Step 2: Run tuning for one kernel first (smoke test)**

```bash
cd /app/aiter/aiter/ops/triton/utils/_triton/tunning/
python orchestrate.py tune --kernels a8w8 --gpus 0-7
```

Check:
- screen.py logs appear in `results/tuning/`
- JSON config files generated in `results/configs/`
- No LDS out-of-resources errors (or they're handled gracefully)

- [ ] **Step 3: Run tuning for all kernels**

```bash
python orchestrate.py tune --kernels all --gpus 0-7
```

This will take significant time (hours). Monitor progress via `results/progress.json`.

- [ ] **Step 4: Copy generated configs**

```bash
cp results/configs/*.json /app/aiter/aiter/ops/triton/configs/gemm/
```

### Task 17: Validation

- [ ] **Step 1: Run validation**

```bash
cd /app/aiter/aiter/ops/triton/utils/_triton/tunning/
python orchestrate.py validate --kernels all --gpus 0-7
cat results/report.txt
```

- [ ] **Step 2: Review report**

Check:
- All kernels show geomean ratio >= 1.0
- No individual shape has > 3% regression
- If any failures, re-tune those specific shapes with more config exploration

- [ ] **Step 3: If regressions exist, iterate**

For any kernel with regressions:
```bash
# Re-tune with wider search space or num_stages=3
python orchestrate.py tune --kernels <regressed_kernel> --gpus 0-7 --force
python orchestrate.py validate --kernels <regressed_kernel> --gpus 0-7
```

- [ ] **Step 4: Commit final configs**

```bash
cd /app/aiter
git add aiter/ops/triton/configs/gemm/gfx950-*.json
git commit -m "perf(configs): retune gfx950 GEMM configs for latest Triton"
```

- [ ] **Step 5: Run unit tests to verify correctness**

```bash
ulimit -c 0
for gpu in 0 1 2 3 4 5 6 7; do
    HIP_VISIBLE_DEVICES=$gpu timeout 600 python -m pytest op_tests/triton_tests/gemm/basic/ -x -q --tb=line &
done
wait
```

All tests should pass with the new configs.

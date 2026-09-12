---
name: aiter-triton-kernel
description: >
 Author or modify Triton kernels inside AITER. Enforces the `aiter/ops/triton/`
 layout (category subfolders: `gemm/`, `attention/`, `moe/`, `normalization/`,
 `quant/`, `rope/`, `fusions/`, `utils/`), the matching `_triton_kernels/`
 mirror, config selection via `get_gemm_config()`, config-aware kernel naming
 via `make_kernel_repr()`, arch-string conventions (`gfx942` / `gfx950` — never
 product names), JSON config layout (`M_LEQ_x` / `M_GEQ_x` / `any`), and the
 placement of matching tests under `op_tests/triton_tests//`.
 Use when writing new Triton kernels, porting ones from triton-lang/triton-ops,
 or refactoring an existing AITER Triton op.
 Usage: /aiter-triton-kernel 
allowed-tools: Bash Read Edit Grep Glob
---

# Author a Triton Kernel in AITER

AITER's Triton ops live under `aiter/ops/triton/` in a category layout. The
corresponding raw `@triton.jit` kernels live under
`aiter/ops/triton/_triton_kernels/` mirroring the same category tree.

Rules of thumb, lifted from `aiter/ops/triton/README.md`:

- **Wrapper file** (`aiter/ops/triton/ /kernel.py`) exposes a Python
  function that reshapes inputs, picks a config, and launches the kernel.
- **Kernel file** (`aiter/ops/triton/_triton_kernels/ /kernel.py`)
  contains the `@triton.jit` body and an internal `_get_config(M, N, K)`.
- **Config JSON** (`aiter/ops/triton/configs/ /{arch}-NAME.json`)
  stores tuned parameters keyed by `M_LEQ_x` / `M_GEQ_x` / `any`.
- **Test** (`op_tests/triton_tests/ /test_kernel.py`) is a `pytest` file.

## 1. Pick the category

| Category | Use for | Path |
|----------|---------|------|
| `gemm/basic` | Plain A*W* matmul | `aiter/ops/triton/gemm/basic/` |
| `gemm/batched` | `bmm` style | `aiter/ops/triton/gemm/batched/` |
| `gemm/feed_forward` | MLP-specific GEMMs (fused activation, glu) | `aiter/ops/triton/gemm/feed_forward/` |
| `gemm/fused` | Fused GEMM (+quant, +epilogue) | `aiter/ops/triton/gemm/fused/` |
| `attention` | MHA / MQA / GQA / MLA | `aiter/ops/triton/attention/` |
| `moe` | Fused MoE variants | `aiter/ops/triton/moe/` |
| `normalization` | RMSNorm / LayerNorm | `aiter/ops/triton/normalization/` |
| `quant` | FP8 / FP4 / INT4 quantizers | `aiter/ops/triton/quant/` |
| `rope` | Rotary embeddings | `aiter/ops/triton/rope/` |
| `fusions` | Other fused ops | `aiter/ops/triton/fusions/` |
| `utils` | Shared helpers (`gemm_config_utils.py`, `kernel_repr.py`, logger) | `aiter/ops/triton/utils/` |

The **flat-file layout is deprecated**. A `_BACKWARD_COMPAT_MAP` in
`aiter/ops/triton/__init__.py` keeps old imports working, but new code should
use the category paths directly.

## 2. Skeleton for a new kernel

Using `gemm_a16w16` as the reference. Replace `my_op` + `MY-OP-CFG` as needed.

### 2.1 Kernel body — `aiter/ops/triton/_triton_kernels/category/my_op.py`

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config


_kernel_repr = make_kernel_repr(
    "_my_op_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"],
)


@triton.jit(repr=_kernel_repr)
def _my_op_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """My kernel: Y = X @ W^T.

    BLOCK_SIZE_M / _N / _K tile the problem; GROUP_SIZE_M controls the
    super-group schedule for better L2 reuse.
    """
    # ... standard GEMM body ...
    pass


def _get_config(M: int, N: int, K: int):
    return get_gemm_config("MY-OP", M, N, K)
```

### 2.2 Wrapper — `aiter/ops/triton/category/my_op.py`

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
from typing import Optional
import torch
import triton

from aiter.ops.triton._triton_kernels.category.my_op import (
    _my_op_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def my_op(
    x: torch.Tensor,
    w: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """Compute Y = X @ W^T with Triton.

    Args:
        x: (M, K) input.
        w: (N, K) weight (internally transposed).
        y: optional pre-allocated (M, N) output.
        config: override the tuned config.

    Returns:
        torch.Tensor of shape (M, N).
    """
    _LOGGER.info(f"MY_OP: x={tuple(x.shape)} w={tuple(w.shape)}")

    M, K = x.shape
    N, _ = w.shape
    if config is None:
        config, _tuned = _get_config(M, N, K)
    if y is None:
        y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _my_op_kernel[grid](
        x, w, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        **config,
    )
    return y
```

### 2.3 Config — `aiter/ops/triton/configs/category/gfx950-MY-OP.json`

```json
{
  "M_LEQ_64":   { "BLOCK_SIZE_M": 16,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2 },
  "M_LEQ_512":  { "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2 },
  "M_GEQ_4096": { "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2 },
  "any":        { "BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,  "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2 }
}
```

Key rules from `aiter/ops/triton/README.md`:

- `any` is **mandatory** when any config exists.
- Keys are `M_LEQ_x` / `M_GEQ_x` — never the deprecated `large`/`small`.
- File name pattern: `{arch}-{CONFIG_NAME}.json`. Specialized: `{arch}-{CONFIG_NAME}-N={N}-K={K}.json`.
- `{arch}` is the gfx string (`gfx942`, `gfx950`, `gfx1100`, …), NOT a product name.

### 2.4 Test — `op_tests/triton_tests/category/test_my_op.py`

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
from aiter.ops.triton.category.my_op import my_op


def torch_ref(x, w):
    return x @ w.T


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M,N,K", [(128, 1280, 8192), (1024, 1024, 1024)])
def test_my_op(dtype, M, N, K):
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    w = torch.randn(N, K, dtype=dtype, device="cuda")
    got = my_op(x, w)
    ref = torch_ref(x, w)
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
```

Run:

```bash
pytest op_tests/triton_tests/category/test_my_op.py -v
# or the whole category:
pytest op_tests/triton_tests/category/
```

## 3. Arch-specific code

**Always** compare `DEVICE_ARCH` to the gfx string:

```python
from aiter.ops.triton.utils.arch_info import DEVICE_ARCH

if DEVICE_ARCH in ("gfx942",):
    ...
elif DEVICE_ARCH in ("gfx950",):
    ...
```

Never parse product names. This anti-pattern breaks on new archs and is
explicitly called out in the README:

```python
# DON'T DO THIS
if int(DEVICE_ARCH.split("MI")[1].replace("X", "")) < 300:
    ...
```

## 4. Trace-friendly kernel names

Decorate every `@triton.jit` with a `repr` that embeds the compile-time
constants — this makes `rocprofv3` stats readable at a glance and lets the
test-selection script (`.github/scripts/select_triton_tests.py`) track
config dependencies.

```python
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_kernel_repr = make_kernel_repr(
    "_my_op_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"],
)

@triton.jit(repr=_kernel_repr)
def _my_op_kernel(...):
    ...
```

## 5. Split-K / reduce pattern

If you support `NUM_KSPLIT > 1`, follow `gemm_a16w16.py`:

1. The main kernel writes `(NUM_KSPLIT, M, N) float32` partials.
2. A separate `_kernel_reduce` sums them back into `(M, N)`.
3. Wrapper decides whether to call the reduce based on `skip_reduce` (useful
   for fusing into a downstream quant op).
4. Use `compute_splitk_params(config, K)` from `gemm_config_utils` when
   appropriate.

## 6. Config-loader integration (for the test-selection script)

`.github/scripts/select_triton_tests.py` statically parses the tree to map
kernel files -> config files. Stick to these patterns so the script can follow
you:

```python
# String literal for the config name
get_gemm_config("MY-OP", M, N, K)

# f-string referring to the configs path
f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MY-OP.json"
```

## 7. Checklist for a new Triton kernel

- [ ] Wrapper placed under the correct `aiter/ops/triton//` folder.
- [ ] Kernel body placed under matching `aiter/ops/triton/_triton_kernels//`.
- [ ] Config files named `{arch}-NAME.json` with `M_LEQ_x` / `M_GEQ_x` / `any` keys (never `large`/`small`).
- [ ] `@triton.jit(repr=...)` using `make_kernel_repr` for trace-friendly names.
- [ ] Wrapper has a docstring covering args / returns / config knobs.
- [ ] `DEVICE_ARCH` comparisons use gfx strings, never product names.
- [ ] Test under `op_tests/triton_tests//test_.py`, `pytest`-compatible.
- [ ] `ruff check aiter/ops/triton/ op_tests/triton_tests/` is clean.
- [ ] Correctness vs a torch reference at `rtol/atol ~= 1e-2` for fp16/bf16.

## 8. Quick checklist for arch-specific exceptions

- Compare `DEVICE_ARCH` to `("gfx950", "gfx942", ...)`.
- Do not parse product names.
- When adding a new arch: add the config JSON files for that arch, then add
  a `DEVICE_ARCH` branch in the wrapper only if dispatch logic actually differs.

## 9. Common pitfalls

| Symptom | Fix |
|---------|-----|
| Kernel works at M=1024 but fails at M=3 | Missing `M_LEQ_4` / `M_LEQ_8` entry; `any` fallback has unsuitable block sizes. |
| `KeyError: 'any'` | Every config file must include an `any` block. |
| Triton recompile on every call | You're passing a tensor value as a constexpr; keep constexprs to shapes/strides/block sizes. |
| Config file not found | Filename uses product name (`MI300X-...`) instead of arch (`gfx942-...`). |
| `AttributeError: 'DEVICE_ARCH' ...` | Import from `aiter.ops.triton.utils.arch_info`, not a local helper. |
| Old flat-file imports fail after refactor | Extend `_BACKWARD_COMPAT_MAP` in `aiter/ops/triton/__init__.py`. |

# Triton Ops

This folder contains Triton-based kernels and wrappers used by AITER.

## GEMM config loading (single utility)

GEMM kernels now use a single utility function instead of managing their own config loading.

- Location: `aiter/ops/triton/utils/gemm_config_utils.py`
- Function: `get_gemm_config(config_name, M, N=None, K=None, bounds=None, specialized_filename=None)`

### Config file formats

**Deprecated format**

```json
{
  "large": { "...": "..." },
  "small": { "...": "..." }
}
```

**New format (required)**

```json
{
  "M_LEQ_64": { "...": "..." },
  "M_GEQ_4096": { "...": "..." },
  "any": { "...": "..." }
}
```

Meaning:
- `M_LEQ_64`: applies when `M <= 64`
- `M_GEQ_4096`: applies when `M >= 4096`
- `any`: fallback for all other `M` values (**must exist if a config exists**)

### How config selection works

Config files are loaded from:
- Default configs: `aiter/ops/triton/configs/gemm/{arch}-{config_name}.json`
- Specialized configs (optional): `{arch}-{config_name}-N={N}-K={K}.json`

Selection rules for the `M` dimension:
1. Searches `M_LEQ_x` keys in ascending order using this default list:
   `[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`
2. If no match, searches `M_GEQ_x` keys in descending order.
3. Falls back to `any` if no bounds match.

### Usage patterns

**Normal case (most common)**

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-A16W16", M, N, K)
```

**With Split-K**

```python
def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
    return compute_splitk_params(config, K), is_tuned
```

**With custom bounds**

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("FF-A16W16-fused", M, N, K, bounds=[4, 8, 64, 4096])
```

**With specialized filename (mostly for fused kernels)**

```python
def _get_config(M: int, N_fp4: int, N_bf16: int, K: int, shuffle: bool = False):
    config_name = "FUSED-GEMM-AFP4WFP4-A16W16"
    specialized_filename = f"N4={N_fp4}-N16={N_bf16}-K={2*K}"
    return get_gemm_config(config_name, M, specialized_filename=specialized_filename)
```

### Config file naming convention

**Default config**
- Pattern: `{arch}-{CONFIG_NAME}.json`
- Example: `gfx950-GEMM-A16W16.json`

**Specialized config (N, K specific)**
- Pattern: `{arch}-{CONFIG_NAME}-N={N}-K={K}.json`
- Example: `gfx950-GEMM-A16W16-N=256-K=7168.json`

**Specialized config (custom)**
- Pattern: `{arch}-{CONFIG_NAME}-{custom_suffix}.json`
- Example: `gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json`

### Config name patterns

- Basic GEMM: `GEMM-A{x}W{y}` (e.g. `GEMM-A16W16`, `GEMM-A8W8`)
- Batched GEMM: `BATCHED_GEMM-A{x}W{y}` (e.g. `BATCHED_GEMM-A16W16`)
- Fused ops: `FUSED-GEMM-{operation}` (e.g. `FUSED-GEMM-A8W8_BLOCKSCALE-A16W16`)
- Feed-forward: `FF-A{x}W{y}-fused` (e.g. `FF-A16W16-fused`)

Variants: append suffix like `_BLOCKSCALE`, `_PRESHUFFLED`, etc.

### Adding a new GEMM kernel (checklist)

- Import and use `get_gemm_config`.
- Add a `_get_config(M, N, K)` wrapper that calls `get_gemm_config(...)`.
- Use config keys `M_LEQ_x`, `M_GEQ_x`, `any` (do **not** use `large`/`small`).
- Name config files per the conventions above: `{arch}-{CONFIG_NAME}.json`.
- Add specialized configs when needed using the proper suffix.
- If the kernel supports split-K, use `compute_splitk_params()`.

## Config-aware kernel names in traces (repr)

Triton kernel names in traces now embed compile-time configuration parameters, making it easier to correlate performance results with the exact tuned config.

- Repr helper: `aiter/ops/triton/utils/_triton/kernel_repr.py`
- Function: `make_kernel_repr(base_name, config_keys)`

### Example usage

```python
kernel_repr = make_kernel_repr(
    "_gemm_a16_w16_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "..."],
)

@triton.jit(repr=kernel_repr)
def _gemm_a16_w16_kernel(...):
    ...
```

How it works:
- Reads `specialization.constants` and appends `KEY_VALUE` pairs to the kernel name.
- Values are sanitized for readability:
  - `None -> NONE`
  - `bool -> 0/1`
  - strings are uppercased and non-alnum characters become `_`

## Kernel doc comments

Kernels should include a concise doc comment (preferably a docstring near the wrapper):

Include:
- What the kernel computes
- Which configuration parameters are included (and what they control)
- What it returns
- Any special considerations

Example docstring template:

```python
"""
Computes batched 8-bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
X is quantized to INT8 during computation using per-token grouped quantization.
W is pre-quantized INT8 with per-batch-element scaling.

Args:
    X (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True.
        Quantized to INT8 on-the-fly during GEMM.
    WQ (torch.Tensor): Pre-quantized INT8 weight batch with shape (B, N, K), internally transposed.
    w_scale (torch.Tensor): Per-batch scale for WQ with shape (1,).
    group_size (int): Group size for per-token grouped quantization of X. Must be power of 2.
    bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
    dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
    splitK (Optional[int]): Not supported. Must be None.
    YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N) or (M, B, N) if transpose_bm=True.
    transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.
    transpose_bm_in (Optional[bool]): Transpose batch and M dimensions in input.
    config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M).

Returns:
    torch.Tensor: Output batch with shape (B, M, N) or (M, B, N) if transpose_bm=True.
"""
```

## Architecture naming convention

Across Triton configuration files and related code, key behavior by GPU architecture identifiers (like `gfx950`) rather than product names.

- Config filenames must use `{arch}` like `gfx950`.
- Conditionals should compare against `DEVICE_ARCH` values like `"gfx950"`.

Preferred:

```python
if impl == "gluon" and DEVICE_ARCH not in ("gfx950",):
    ...
```

Avoid product-name parsing like:

```python
# Don't do this
if impl == "gluon" and int(DEVICE_ARCH.split("MI")[1].replace("X", "")) < 350:
    ...
```

## Triton test organization

`op_tests/triton_tests` has been reorganized into categorized subfolders for easier maintenance and targeted test runs.

Categories include: `gemm/`, `moe/`, `attention/`, `quant/`, etc.

### How to run

- All Triton tests:
  - `pytest op_tests/triton_tests/`
- GEMM tests only:
  - `pytest op_tests/triton_tests/gemm/`
- Basic GEMM subset:
  - `pytest op_tests/triton_tests/gemm/basic/`
- MOE tests only:
  - `pytest op_tests/triton_tests/moe/`

---

### Quick checklist when adding a new kernel

- Config filenames use `{arch}` like `gfx950`.
- Add config-aware trace naming via `make_kernel_repr(...)` and `@triton.jit(repr=...)`.
- Add a kernel docstring describing behavior + config params + returns + notes.

### Quick checklist when adding arch-specific exceptions

- Compare `DEVICE_ARCH` to `("gfx950", "gfx942", ...)`.
- Do not parse product names.

### Quick checklist when adding Triton tests

- Place tests under the appropriate category in `op_tests/triton_tests/<category>/`.

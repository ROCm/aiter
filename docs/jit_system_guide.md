# AITER JIT Compilation System Guide

This guide documents AITER's just-in-time kernel compilation system — the core infrastructure that builds HIP/C++ operator kernels on demand, caches them, and optionally pre-compiles them at install time.

---

## Quick Reference

| Task | How |
|------|-----|
| **Use an operator** | Just call it — JIT compiles on first use |
| **Force rebuild** | `AITER_REBUILD=1` |
| **Verbose logging** | `AITER_LOG_MORE=1` (or `2` for call args) |
| **Target specific GPU** | `GPU_ARCHS="gfx942"` |
| **Pre-compile at install** | `PREBUILD_KERNELS=2 python3 setup.py install` |
| **Override JIT cache dir** | `AITER_JIT_DIR=/path/to/dir` |

---

## 1. Architecture Overview

```
Python function call (e.g., aiter.silu_and_mul())
    │
    ▼
@compile_ops("module_activation") decorator
    │
    ▼
Try importlib.import_module("module_activation")
    │
    ├── SUCCESS → call C++ function via pybind11 → return
    │
    └── ModuleNotFoundError →
            │
            ▼
        Load build config from optCompilerConfig.json
            │
            ▼
        build_module() — acquire lock, compile via Ninja
            │
            ▼
        Copy .so to JIT dir, import, cache, call function
```

### Core Components

| Component | File | Role |
|---|---|---|
| Core orchestrator | `aiter/jit/core.py` | Module definition, build orchestration, `compile_ops` decorator |
| HIP compilation | `aiter/jit/utils/cpp_extension.py` | Ninja-based C++/HIP compilation (forked from PyTorch) |
| Multiprocess lock | `aiter/jit/utils/file_baton.py` | File-based synchronization for parallel builds |
| Module config | `aiter/jit/optCompilerConfig.json` | JSON registry of ~55 compilable operator modules |
| GPU detection | `aiter/jit/utils/chip_info.py` | AMD GPU arch detection via `rocminfo` |
| Version tracking | `aiter/jit/utils/_cpp_extension_versioner.py` | Content-hash-based recompilation detection |
| torch.compile guard | `aiter/jit/utils/torch_guard.py` | Registers ops into `torch.ops.aiter` namespace |

---

## 2. The `@compile_ops` Decorator

The `compile_ops` decorator is the interface between Python operator wrappers and the JIT system.

```python
def compile_ops(
    _md_name: str,                    # Module name (key in optCompilerConfig.json)
    fc_name: Optional[str] = None,    # C++ function name (defaults to Python func name)
    gen_func: Optional[Callable] = None,  # Dynamic build-args generator
    gen_fake: Optional[Callable] = None,  # Fake tensor generator for torch.compile
):
```

### Pattern 1: Simple Static Binding (most common)

Multiple functions share the same compiled module:

```python
@compile_ops("module_activation")
def silu_and_mul(out: Tensor, input: Tensor) -> None: ...

@compile_ops("module_activation")
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...
```

### Pattern 2: Explicit Name + torch.compile Support

When the C++ function name differs from the Python name:

```python
@compile_ops("module_gemm_a8w8", fc_name="gemm_a8w8", gen_fake=gen_fake_tensors)
def gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out, bias=None, splitK=0): ...
```

### Pattern 3: Dynamic Code Generation

For operators that need per-dtype compilation (e.g., binary ops):

```python
def cmdGenFunc(op_name: str, input: Tensor, other: Tensor) -> dict:
    dtype_str = str(input.dtype).split(".")[1] + "_" + str(other.dtype).split(".")[1]
    return {
        "md_name": f"module_aiter_{op_name}_{dtype_str}",
        "blob_gen_cmd": [f"{AITER_CSRC_DIR}/kernels/generate_binaryop.py ..."],
    }

@compile_ops("module_aiter_operator", gen_func=partial(cmdGenFunc, "add"), gen_fake=binary_fake_shape)
def add(input: Tensor, other: Tensor) -> Tensor: ...
```

### How the Decorator Works

1. On first call, tries `importlib.import_module(md_name)`.
2. If `.so` exists (prebuilt or previously JIT-compiled), import succeeds immediately.
3. If `ModuleNotFoundError`, loads config from JSON, calls `build_module()`, re-imports.
4. The C++ function is retrieved via `getattr(module, func_name)` and called.
5. On first success, validates Python argument types against the pybind11 signature.
6. The wrapper is also registered into `torch.ops.aiter` for `torch.compile` compatibility.

---

## 3. Module Configuration

### optCompilerConfig.json

Each entry maps a module name to its build configuration:

```json
{
  "module_activation": {
    "srcs": ["f'{AITER_CSRC_DIR}/pybind/activation_pybind.cu'",
             "f'{AITER_CSRC_DIR}/kernels/activation_kernels.cu'"],
    "flags_extra_cc": [],
    "flags_extra_hip": ["-DCK_TILE_..."],
    "extra_ldflags": "None",
    "extra_include": [],
    "verbose": "False",
    "is_python_module": "True",
    "blob_gen_cmd": "''"
  }
}
```

| Field | Description |
|---|---|
| `srcs` | Source file paths (f-string expressions evaluated at load time) |
| `flags_extra_cc` | Extra C++ compiler flags |
| `flags_extra_hip` | Extra HIP compiler flags |
| `extra_include` | Additional include directories |
| `blob_gen_cmd` | Code generation script (CK instance generators, etc.) |
| `hipify` | Whether to run CUDA-to-HIP conversion |
| `hip_clang_path` | Optional custom HIP compiler path |

All values are stored as string expressions and `eval()`'d at load time, enabling dynamic path construction using variables like `AITER_CSRC_DIR`, `CK_DIR`, etc.

---

## 4. Build Flow

When `build_module()` is called:

1. **Lock**: Acquires `FileBaton` at `{build_dir}/lock_{module_name}`.
2. **Clean** (if `AITER_REBUILD`): Deletes `.so` and/or build directory.
3. **Stage sources**: Copies into `{build_dir}/{module}/build/srcs/`, optionally renaming `.cpp` → `.cu`.
4. **Compile flags**: Assembles flags including:
   - Base: `-O3 -std=c++20`
   - HIP: AMD platform defines, `--offload-arch={arch}` per target
   - Version-conditional: LLVM-version-specific flags
   - Each flag is tested with `hip_flag_checker()` before use
5. **Blob generation**: Executes code-gen scripts that produce additional `.cu` files.
6. **Ninja build**: `_jit_compile()` generates a `build.ninja` and runs `ninja -v -j {workers}`.
7. **Output**: Copies the `.so` to the JIT directory.

---

## 5. Cache Directory Structure

```
{jit_dir}/
    build/
        lock_{module_name}          # FileBaton lock files
        {module_name}/
            build/
                srcs/               # Staged .cu source files
                build.ninja         # Ninja build file
                *.cuda.o            # Object files
                {module_name}.so    # Built shared library
            blob/                   # Code-generated sources
    {module_name}.so                # Final .so for import
```

### JIT Directory Resolution

Priority order:
1. `AITER_JIT_DIR` environment variable (if set)
2. `aiter/jit/` directory (if writable)
3. `~/.aiter/jit/` (fallback)

### Cache Invalidation

- **Automatic**: Content-hash-based — hashes source files + build flags. Recompiles only when something changes.
- **Manual**: `AITER_REBUILD=1` deletes `.so` + build dir; `AITER_REBUILD=2` deletes `.so` only.
- **Prebuild**: `setup.py` with `PREBUILD_KERNELS` wipes everything and rebuilds from scratch.

---

## 6. GPU Architecture Detection

### Detection Flow

1. Check `GPU_ARCHS` env var — if set (and not `"native"`), use directly.
2. Otherwise, run `rocminfo` and parse for `gfx` arch strings.
3. Also extracts compute unit (CU) count for hardware differentiation.

### Supported Architectures

`native` (auto-detect), `gfx90a` (MI250X), `gfx940`, `gfx941`, `gfx942` (MI300X), `gfx950` (MI350X), `gfx1100`–`gfx1201`

Multi-target: `GPU_ARCHS="gfx942;gfx950"` compiles for both.

---

## 7. PREBUILD_KERNELS Interaction

When set during `setup.py`, kernels are pre-compiled at install time:

| Level | What Gets Built |
|---|---|
| `0` (default) | Nothing — all kernels JIT on first use |
| `1` | Core kernels (excludes tuning modules, most MHA variants) |
| `2` | Inference kernels (excludes backward and tuning modules) |
| `3` | MHA kernels only (minimal build) |

At runtime, `@compile_ops` simply tries `import_module()`. Pre-built `.so` files load instantly; missing ones trigger JIT as usual. The two modes are fully transparent.

---

## 8. Environment Variables

### Core JIT

| Variable | Default | Description |
|---|---|---|
| `AITER_REBUILD` | `0` | `1` = full rebuild; `2` = rebuild .so only |
| `AITER_JIT_DIR` | (auto) | Override JIT output directory |
| `AITER_LOG_MORE` | `0` | `1` = verbose logging; `2` = log call arguments |
| `AITER_FP4x2` | `1` | Enable FP4x2 support on gfx950 |

### Build

| Variable | Default | Description |
|---|---|---|
| `GPU_ARCHS` | `"native"` | Target GPU arch(s), semicolon-separated |
| `MAX_JOBS` | (auto) | Max parallel compilation jobs |
| `PREBUILD_KERNELS` | `0` | Prebuild level (0–3) |
| `CK_DIR` | `3rdparty/composable_kernel` | Composable Kernel source root |

### Compiler Path Overrides

| Variable | Used By |
|---|---|
| `GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH` | A4W4 blockscale, MOE CK 2-stage |
| `FLATMM_HIP_CLANG_PATH` | DeepGEMM, A8W8 bpreshuffle CKTile |
| `MHA_HIP_CLANG_PATH` | MHA forward/backward kernels |

---

## 9. Adding a New Operator

### Step 1: Create C++ source

Add `.cu` files under `csrc/pybind/` (pybind interface) and `csrc/kernels/` (implementation).

### Step 2: Register in JSON config

Add an entry to `aiter/jit/optCompilerConfig.json`:

```json
"module_my_op": {
    "srcs": ["f'{AITER_CSRC_DIR}/pybind/my_op_pybind.cu'",
             "f'{AITER_CSRC_DIR}/kernels/my_op.cu'"],
    "flags_extra_cc": [],
    "flags_extra_hip": [],
    "extra_ldflags": "None",
    "extra_include": [],
    "verbose": "False",
    "blob_gen_cmd": "''"
}
```

### Step 3: Create Python wrapper

```python
from ..jit.core import compile_ops

@compile_ops("module_my_op")
def my_function(input: torch.Tensor, output: torch.Tensor) -> None: ...
```

### Step 4: Add torch.compile support (optional)

```python
def my_function_fake(input, output):
    return output

@compile_ops("module_my_op", gen_fake=my_function_fake)
def my_function(input: torch.Tensor, output: torch.Tensor) -> torch.Tensor: ...
```

### Guidelines

- **Function body should be `...` (Ellipsis)** — the Python body is never executed; it's replaced by C++.
- **Multiple functions per module**: Multiple `@compile_ops` decorators can share the same module name.
- **Type annotations are required**: Used for runtime type checking against the C++ signature.
- **Blob generation**: For template-heavy ops, use `blob_gen_cmd` pointing to a code-gen script. The `{{}}` placeholder is replaced with the output directory.

---

## 10. Debugging Tips

| Problem | Solution |
|---------|----------|
| Kernel won't compile | Set `AITER_LOG_MORE=1` for verbose output |
| Stale cached kernel | Set `AITER_REBUILD=1` to force rebuild |
| Check exact build commands | Inspect `{jit_dir}/build/{module}/build/build.ninja` |
| Log function call arguments | Set `AITER_LOG_MORE=2` |
| Wrong GPU target | Check `GPU_ARCHS` and verify with `rocminfo` |

---

## 11. Source Files

| Component | Path |
|---|---|
| Core orchestrator | `aiter/jit/core.py` |
| HIP compilation | `aiter/jit/utils/cpp_extension.py` |
| Multiprocess lock | `aiter/jit/utils/file_baton.py` |
| Module config | `aiter/jit/optCompilerConfig.json` |
| GPU detection | `aiter/jit/utils/chip_info.py` |
| Version tracking | `aiter/jit/utils/_cpp_extension_versioner.py` |
| torch.compile guard | `aiter/jit/utils/torch_guard.py` |

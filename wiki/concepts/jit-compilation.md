---
title: "JIT Compilation"
last_verified: 2026-04-06
source_files:
  - aiter/jit/core.py
  - aiter/jit/utils/chip_info.py
tags: [jit, compilation, hip, build]
---

# JIT Compilation

## Overview
aiter uses a JIT (Just-In-Time) compilation system to build HIP C++ extensions on first use. The `compile_ops()` decorator in `aiter/jit/core.py` handles compilation, caching, and loading of native kernels.

## How It Works
1. Python operator functions are decorated with `@compile_ops("module_name")`
2. On first call, the JIT system compiles the corresponding C++ source from `csrc/`
3. The compiled module is cached in `aiter/jit/build/` for subsequent uses
4. The compiled functions are bound via pybind11 or ctypes FFI

## Key Components
- `compile_ops(module_name, fc_name, gen_fake, ffi_type)` -- decorator that registers a JIT-compiled function
- `AITER_CSRC_DIR` -- path to C++ source files
- `AITER_ROOT_DIR` -- project root
- `AITER_CONFIGS` -- paths to config CSV files

## GPU Detection
`aiter/jit/utils/chip_info.py` provides:
- `get_cu_num()` -- number of compute units on current GPU
- `get_gfx()` -- GPU architecture string (e.g., "gfx942")
- `get_num_sms()` -- same as CU count (used by attention selector)

## torch.compile Compatibility
`aiter/jit/utils/torch_guard.py` provides `torch_compile_guard` for operators that need special handling under torch.compile.

## Fake Tensor Support
Many operators provide `gen_fake` functions for torch.compile tracing. These return tensors with correct shapes/dtypes without executing the actual kernel.

## Related Pages
- [[concepts/backend-selection]] -- JIT compilation enables CK/HIP backends
- [[ck/architecture]] -- CK kernels compiled through this system

## Source Files
- `aiter/jit/core.py` -- JIT compilation infrastructure
- `aiter/jit/utils/chip_info.py` -- GPU architecture detection
- `aiter/jit/utils/torch_guard.py` -- torch.compile compatibility

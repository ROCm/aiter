---
name: aiter-jit-debug
description: >
 Diagnose AITER JIT compilation failures: "ModuleNotFoundError: module_*"
 at import time, `hipcc` errors during `@compile_ops`, stale `.so` caches,
 hipify mishandling of ROCm intrinsics, arch mismatch (gfx942 vs gfx950),
 and module-config typos in `optCompilerConfig.json`. Use when an AITER op
 fails to load or rebuild, or when `AITER_REBUILD=1` doesn't pick up code
 changes.
 Usage: /aiter-jit-debug []
allowed-tools: Bash Read Edit Grep Glob
---

# Debug AITER JIT Compilation

AITER compiles most kernels on-demand through `aiter/jit/core.py` + the
`@compile_ops` decorator. This skill walks through the common failure modes
in order of likelihood.

## Step 0: Turn on verbose logging

Before anything else, reproduce the failure with maximum verbosity:

```bash
AITER_LOG_MORE=1 AITER_LOG_LEVEL=DEBUG python3 op_tests/test_my_op.py 2>&1 | tee /tmp/aiter_jit.log
```

Environment variables you'll use throughout:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AITER_LOG_MORE` | `0` | Adds compile command lines, timings, file paths. |
| `AITER_LOG_LEVEL` | `INFO` | `DEBUG` shows every JIT decision. |
| `AITER_REBUILD` | `0` | Force-rebuild every JIT module. |
| `GPU_ARCHS` | auto | Comma/semicolon list, e.g. `"gfx942"` or `"gfx942;gfx950"`. |
| `MAX_JOBS` | auto | Reduce when builds OOM. |
| `CK_DIR` | `3rdparty/composable_kernel` | Alternate CK source path. |

## Step 1: Classify the error

| Symptom | Likely cause | Go to |
|---------|--------------|-------|
| `ModuleNotFoundError: No module named 'module_xxx'` at import | Missing/typo'd key in `optCompilerConfig.json` | §2 |
| `ImportError: cannot import name 'xxx' from 'module_xxx'` | pybind name mismatch | §3 |
| `hipcc: command not found` or `/opt/rocm/bin/hipcc: error` | ROCm toolchain not on PATH | §4 |
| `undefined symbol: _Z ...` at load | Missing source in `srcs` | §5 |
| `hipErrorNoBinaryForGpu` at run | Arch mismatch | §6 |
| Code changes ignored | Stale cache / wrong mtime | §7 |
| `hipify` mangled an intrinsic / fails to parse | hipify should be disabled for this module | §8 |
| OOM during compile / `killed signal 9` | Too many parallel jobs | §9 |
| CK build hangs / takes >30 min | Template-heavy instance file; needs splitting | §10 |

## 2. ModuleNotFoundError: module_xxx

Three files must agree on the module name:

```bash
grep -r '"module_xxx"' aiter/jit/optCompilerConfig.json
grep -r 'PYBIND11_MODULE(module_xxx'  csrc/pybind/
grep -rn '@compile_ops("module_xxx")' aiter/ops/
```

**All three must match character-for-character.** A single typo (`module_xxx` vs
`module_xx`) causes this error.

The JSON key is read in `aiter/jit/core.py::get_args_of_build()`. If it's
missing, `build_module()` never runs and the `.so` is never produced.

Fix: add the missing key (see `aiter-add-operator` skill, Step 3) or correct
the typo.

## 3. ImportError: cannot import name 'xxx'

The module loaded but the symbol isn't exported. Check the pybind definition:

```cpp
PYBIND11_MODULE(module_xxx, m) {
  m.def("xxx", &xxx, "...");   // <-- the name "xxx" here must match
}
```

`@compile_ops("module_xxx")` binds the wrapper function **by name**. So a
`def fast_path(...)` in Python tied to pybind `m.def("fast_path", ...)` is OK,
but `m.def("fast_path_v2", ...)` would break.

## 4. hipcc not found / wrong path

AITER relies on `torch.utils.cpp_extension` (forked under
`aiter/jit/utils/cpp_extension.py`) to find `hipcc`. Check:

```bash
which hipcc
hipcc --version          # expect ROCm 6.x or 7.x
python3 -c "import torch; print(torch.version.hip)"
```

Common fixes:

```bash
# Add ROCm to PATH
export PATH=/opt/rocm/bin:$PATH
# Match hip version to torch
python3 -c "import torch.utils.cpp_extension as c; print(c.ROCM_HOME)"
```

If `torch.version.hip` is `None`, you're on a CPU-only PyTorch build — reinstall
a ROCm-enabled wheel.

## 5. undefined symbol at load time

`import aiter.ops.my_op` succeeds, but calling `my_op(...)` crashes with
`undefined symbol: _Z5my_op...`. The `.so` compiled, but a referenced function
wasn't in any `srcs` entry.

```bash
nm -D aiter/jit/module_my_op.so | grep -i my_op
# Shows the undefined symbol (U marker)
```

Fix: add the missing `.cu` / `.cpp` to `srcs` in `optCompilerConfig.json`.

## 6. hipErrorNoBinaryForGpu

The kernel was compiled for a different arch than the GPU running it.

```bash
rocm-smi --showproductname   # e.g. MI300X (gfx942)
python3 -c "import torch; print(torch.cuda.get_device_properties(0))"
```

Force a rebuild targeting the right arch:

```bash
rm -rf aiter/jit/build aiter/jit/module_my_op.so
GPU_ARCHS="gfx942" AITER_REBUILD=1 python3 op_tests/test_my_op.py
```

For multi-arch wheels, use `GPU_ARCHS="gfx942;gfx950"`. `setup.py` respects
this in PREBUILD mode.

## 7. Stale cache / code change not picked up

AITER caches compiled `.so` files in `aiter/jit/` keyed on source file mtime +
flags. If NFS / Docker bind-mount mtimes are unstable, the cache may persist
across edits.

**Nuclear option**:

```bash
rm -rf aiter/jit/build aiter/jit/*.so
AITER_REBUILD=1 python3 op_tests/test_my_op.py
```

**Surgical option** (just one module):

```bash
rm -f aiter/jit/module_my_op.so aiter/jit/build/module_my_op/
AITER_REBUILD=1 python3 op_tests/test_my_op.py
```

**Debug which source triggered or blocked a rebuild**:

```bash
AITER_LOG_MORE=1 python3 -c "from aiter.ops.my_op import my_op; my_op"
# Search the log for "build finished" / "build skipped" entries.
```

## 8. hipify mangles an intrinsic

AITER's hipify pass rewrites CUDA calls to HIP equivalents
(`aiter/jit/utils/hipify/`). When your source is **already** HIP (e.g. uses
`__builtin_amdgcn_*`, ROCm-specific types, CK-tile code), hipify can mis-edit
it.

Disable hipify for that module by adding `"hipify": "False"` to the JSON entry:

```json
"module_pa_ragged": {
  "srcs": [...],
  "flags_extra_hip": [...],
  "hipify": "False"
}
```

Signs you need this:
- Compile fails on a line that looks fine in the source.
- `grep cuda ` shows hipify already ran over the file (look in
  `aiter/jit/build/module_xxx/` for the hipified copy).

## 9. Compile OOM

Template-heavy CK instance files can consume 4–8 GB each. With `MAX_JOBS=64`
that's >200 GB of RAM.

```bash
MAX_JOBS=8 AITER_REBUILD=1 python3 op_tests/test_my_op.py
# or, for the whole install:
MAX_JOBS=8 PREBUILD_KERNELS=1 python3 setup.py develop
```

`setup.py::getMaxJobs()` already caps based on free memory, but it assumes
0.5 GB/job which is optimistic for CK.

## 10. CK compile takes forever

If `csrc/ck_*` instances take >30 min per file, split them:

```bash
ls csrc/ck_gemm_a8w8/  # look for gen_instances.py and *_part{1,2,3}.cpp naming
```

CK's own generator (`example/ck_tile/*/generate.py`) supports `--part`
splitting. After regenerating, make sure the new files are listed in the
`blob_gen_cmd` output directory.

## Useful inspection commands

```bash
# Show the full JIT command for a module (works even after it compiled):
AITER_LOG_MORE=1 AITER_REBUILD=1 python3 -c "
import aiter.ops.my_op as m; m.my_op
" 2>&1 | grep -E '(hipcc|g\+\+|-o .*\.so)'

# Show which archs a .so contains:
/opt/rocm/bin/roc-obj-ls aiter/jit/module_my_op.so

# Validate JSON syntax after editing:
python3 -c "import json; json.load(open('aiter/jit/optCompilerConfig.json'))"
```

## Sanity checklist when an op is "broken"

1. `git status` — uncommitted changes to `optCompilerConfig.json`?
2. `rm -rf aiter/jit/build aiter/jit/*.so` — stale cache cleared?
3. `AITER_REBUILD=1 AITER_LOG_MORE=1 python3 op_tests/test_my_op.py 2>&1 | head -50`
4. Confirm module name matches across JSON, pybind, and `@compile_ops`.
5. Confirm `GPU_ARCHS` covers the target GPU.
6. For CK ops: confirm `3rdparty/composable_kernel` is at the expected submodule SHA.

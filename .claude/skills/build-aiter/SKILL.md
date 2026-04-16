---
name: build-aiter
description: >
 Install and build AITER from source on a ROCm host or inside a Docker container.
 Covers recursive submodule checkout (Composable Kernel), `python setup.py develop`
 vs `pip install -e .`, `PREBUILD_KERNELS` levels (0/1/2/3), `GPU_ARCHS`
 selection (gfx942/gfx950), optional FlyDSL and Iris/Triton-comms deps, and how
 to verify the install. Use when the user wants to set up AITER, rebuild after a
 ROCm/torch upgrade, or switch between develop and install modes.
 Usage: /build-aiter [container@host]
allowed-tools: Bash Read
---

# Build and Install AITER

Build AITER from source. AITER is normally installed in **editable (develop) mode**
because most kernels are JIT-compiled on first use.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `[container@host]` | No | Target in the form `container@hostname`. If omitted, build locally. |

## Prerequisites

- **ROCm**: 6.x or 7.x, with `hipcc` on PATH and a working GPU (`rocm-smi`).
- **PyTorch**: ROCm-enabled build, matching the installed ROCm (e.g. `rocm/vllm-dev:nightly`).
- **Python**: 3.9 – 3.12.
- **C++ toolchain**: `ninja`, a C++17 compiler, `cmake >= 3.20`.
- **Disk space**: ~10 GB for build artifacts under `aiter/jit/`.
- **Submodules**: The repo embeds `3rdparty/composable_kernel`. A non-recursive
  clone will fail the build.

## Step 1: Clone with submodules

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
```

If you forgot `--recursive`:

```bash
cd aiter
git submodule sync && git submodule update --init --recursive
```

## Step 2: Pick an install mode

AITER supports three install flavors:

| Mode | Command | When to use |
|------|---------|-------------|
| **JIT (default)** | `python3 setup.py develop` | Fastest setup; kernels are compiled lazily on first use. |
| **Prebuild tuned kernels** | `PREBUILD_KERNELS=1 GPU_ARCHS="gfx942;gfx950" python3 setup.py develop` | Build all tuned kernels up-front; slow install (~30–60 min) but zero first-call latency. |
| **Pip editable** | `pip install -e .` | Same as `setup.py develop` but uses PEP 517. Preferred for CI. |

`PREBUILD_KERNELS` levels (see `setup.py`):

| Value | Effect |
|-------|--------|
| `0` (default) | Skip all `_tune` modules; JIT everything on demand. |
| `1` | Prebuild tuned kernels; skip most `_tune` modules and most `mha` variants. |
| `2` | Prebuild everything except `_bwd` and `_tune`. |
| `3` | Prebuild only `module_fmha_v3*`. |

Common env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_ARCHS` | auto | Semicolon-separated arch list, e.g. `"gfx942;gfx950"`. |
| `ENABLE_CK` | `1` | Set `0` to build without Composable Kernel (no `mha`/CK-GEMM ops). |
| `AITER_REBUILD` | `0` | Force re-compile of every JIT module. |
| `AITER_LOG_MORE` | `0` | Verbose JIT logs (command lines, compile times). |
| `MAX_JOBS` | auto (CPU\*0.8) | Parallel compile jobs. |

## Step 3: Optional extras

```bash
# FlyDSL (required for A4W4 FusedMoE kernels; optional otherwise)
pip install --pre flydsl

# OR: install all optional deps listed in requirements.txt
pip install -r requirements.txt

# Iris / Triton-based GPU communication
pip install -r requirements-triton-comms.txt
```

## Step 4: Verify the install

```bash
python3 -c "import aiter; print('aiter OK')"
python3 -c "from aiter.ops.activation import silu_and_mul; print('activation OK')"

# Smoke test that JIT works (will compile module_activation on first call)
python3 op_tests/test_activation.py --help
```

For a full smoke run:

```bash
python3 op_tests/test_layernorm2d.py
python3 op_tests/test_gemm_a8w8.py -m 128 -n 1536 -k 7168
```

## Step 5: Remote / Docker build

When building inside a container on a remote host, wrap every command in
`ssh ... 'docker exec ... bash -c "..."'`:

```bash
HOST=hjbog-srdc-39.amd.com
CONTAINER=my_aiter_dev

# Checkout + build
ssh -o LogLevel=ERROR $HOST "docker exec $CONTAINER bash -c '
    cd / && git clone --recursive https://github.com/ROCm/aiter.git &&
    cd aiter && python3 setup.py develop
'"

# Rebuild after code change
ssh -o LogLevel=ERROR $HOST "docker exec $CONTAINER bash -c '
    cd /aiter && AITER_REBUILD=1 python3 -c \"import aiter; from aiter.ops.activation import silu_and_mul\"
'"
```

## Rebuild after code changes

```bash
# Python-only edits: no rebuild needed (editable install).

# C++ / HIP kernel edits: force-rebuild the affected JIT module(s).
AITER_REBUILD=1 python3 op_tests/test_activation.py

# Wipe all JIT artifacts and start fresh:
rm -rf aiter/jit/build aiter/jit/*.so
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'aiter'` | Re-run `python3 setup.py develop`; confirm the venv is activated. |
| `fatal error: composable_kernel/...` | Submodules not initialized — run `git submodule update --init --recursive`. |
| `hipcc: not found` | ROCm not on PATH; `source /opt/rocm/bin/rocm_env.sh` or reinstall ROCm. |
| Build OOM / `killed (signal 9)` | Lower `MAX_JOBS` (e.g. `MAX_JOBS=4`). See `getMaxJobs()` in `setup.py`. |
| `gfx ` not compiled | Re-run with explicit `GPU_ARCHS="gfxXXX"` and `AITER_REBUILD=1`. |
| `import flydsl` fails | FlyDSL is optional; `pip install --pre flydsl` or ignore if you don't need A4W4 MoE. |
| Wheel build produces `amd-aiter` but import needs `aiter` | The package name in `setup.py` is `amd-aiter`; the import name is `aiter`. Both are expected. |
| Windows build fails | Windows sets `ENABLE_CK = False` and `PREBUILD_KERNELS = False`. Only a small Python subset builds. Use WSL or a Linux container for full builds. |

## Reference env for reproducibility

| Component | Version |
|-----------|---------|
| Base image | `rocm/vllm-dev:nightly` (ROCm 7.0, PyTorch 2.9, rocprofv3). |
| AITER branch | `main` (use a SHA in CI). |
| CK submodule | Tracked at the SHA committed in `.gitmodules`. |
| FlyDSL | `flydsl==0.1.1.dev409` (pinned in `pyproject.toml`). |
| Formatter pins | `black==26.3.0`, `ruff==0.15.7`, `clang-format-18`. |

---
name: build-aiter
description: >
 Install and build AITER from source on a ROCm host (Linux; Windows disables
 CK and prebuild). This skill is a thin hook: `setup.py` and
 `docs/installation.rst` are the authoritative sources. Use this skill to get
 the right env-var cheatsheet, pick the correct `PREBUILD_KERNELS` mode, and
 avoid the handful of recurring footguns. Triggers: "install aiter",
 "prebuild kernels", "setup.py develop", "GPU_ARCHS", "pip install -e aiter".
allowed-tools: Bash Read
---

# Build and Install AITER

Authoritative references (read these first if the user wants depth):
- `setup.py` — source of truth for all build behavior.
- `docs/installation.rst` — prose walkthrough.
- `CONTRIBUTE.md` — CK submodule / optional deps.

This skill encodes only the parts agents keep getting wrong.

## Platform
- Linux + ROCm only for a full build. On Windows `setup.py` sets
  `ENABLE_CK = False` and `PREBUILD_KERNELS = False` (search `IS_WINDOWS` in
  `setup.py`). Use WSL2 / Linux / container for real builds.

## Install modes

| Mode | Command | When |
|------|---------|------|
| JIT (default, fastest install) | `python3 setup.py develop` | Dev iteration. Kernels built lazily on first use. |
| Prebuild + editable | `PREBUILD_KERNELS=<N> GPU_ARCHS="gfx942;gfx950" pip install -e . --no-build-isolation` | CI / reproducible perf runs. |

Editable is preferred; `setup.py develop` is equivalent and works too.

## `PREBUILD_KERNELS` modes (read straight from `setup.py::get_exclude_ops`)

| Value | What is excluded from prebuild |
|-------|--------------------------------|
| `0` (default) | only `*_tune`; everything else is JIT on first use. |
| `1` | `*_tune`, plus every `mha*` except `module_fmha_v3_fwd` and `module_fmha_v3_varlen_fwd`. |
| `2` | `*_tune` and `*_bwd`. Forward-only inference build (common). |
| `3` | Everything except `module_fmha_v3*`. Attention-only. |

**There is no mode that strictly prebuilds "GEMM + attention only".** The
closest is `=2` (forward everything) or a custom edit to `get_exclude_ops()`.

## Key env vars

| Var | Default | Effect |
|-----|---------|--------|
| `GPU_ARCHS` | auto-detect | Semicolon-separated arch list, e.g. `"gfx942;gfx950"`. Set explicitly in CI. |
| `PREBUILD_KERNELS` | `0` | See table above. |
| `ENABLE_CK` | `1` | Set `0` to build without Composable Kernel (drops CK GEMM / MHA). |
| `MAX_JOBS` | # cores | Parallel compile jobs. Lower if RAM-constrained (CK instances are heavy). |
| `AITER_REBUILD` | `0` | Runtime: force a JIT rebuild of a module on next import. |

## Sanity-check after install

```bash
python3 -c "import aiter; print(aiter.__version__)"
python3 -c "from aiter.jit.utils.chip_info import get_gfx; print(get_gfx())"
```

If arch detection disagrees with the `GPU_ARCHS` used at install time,
expect `hipErrorNoBinaryForGpu` at first kernel call.

## Recurring footguns

| Symptom | Cause / fix |
|---------|-------------|
| Package named `amd-aiter`, import is `aiter` | Expected; both names are correct. |
| Windows build succeeds but no kernels available | `ENABLE_CK/PREBUILD_KERNELS` forced to False. Use WSL/Linux. |
| `error: Microsoft Visual C++ 14.0 required` | Wrong platform; AITER needs hipcc not MSVC. |
| CK submodule missing | Run `git submodule update --init --recursive`. |
| Prebuild skipped with "torch not installed" | Install a ROCm PyTorch wheel first, then rerun install. |
| First import triggers long compile | JIT mode behaving normally; set `PREBUILD_KERNELS=1` (or 2) for up-front cost. |
| Stale `.so` after arch change | `export AITER_REBUILD=1`, re-import. |

## When to escalate
- Build errors during CK instance compilation → `aiter-jit-debug` skill.
- Runtime `hipErrorNoBinaryForGpu` → `aiter-jit-debug`.
- Tuning data for a specific op after a fresh install → `aiter-ck-tune` /
  `aiter-moe-tuning`.

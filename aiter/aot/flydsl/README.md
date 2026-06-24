# FlyDSL AOT Pre-compilation & Tests

This directory holds the **AOT (Ahead-Of-Time) pre-compilation entry points** for
FlyDSL kernels. Each module extracts every unique FlyDSL kernel name from aiter's
tuned CSV configs and compiles them into the cache up front, so that at runtime
the JIT path hits the cache instead of compiling again.

| Module | OpKind | Description |
| --- | --- | --- |
| `moe.py` | `MOE` | MoE / Mixed-MoE kernels (stage1 + stage2) |
| `gemm.py` | `GEMM` | GEMM kernels |
| `grouped_moe.py` | `GROUPED_MOE` | gfx1250 grouped MoE GEMM kernels |
| `chunk_gdn_h.py` | `CHUNK_GDN_H` | chunk-gdn-h kernels |
| `common.py` | — | Shared job collection / process pool / cache-hit checking logic |

---

## 0. Set up the environment

Run everything inside your Python virtualenv, e.g.:

```bash
source /path/to/venv/bin/activate
```

All commands below assume you run them from the repo root (the top-level `aiter`
directory of your checkout).

---

## 1. Run AOT pre-compilation (compile smoke test)

The most direct "test" is to run each module as a `python -m` entry point and
confirm every kernel compiles. Each module prints `Compiled: N ok, M failed` at
the end and exits 0 when all succeed, 1 on any failure — so it plugs straight
into CI.

```bash
# MoE / Mixed-MoE (default CSVs)
python -m aiter.aot.flydsl.moe

# GEMM
python -m aiter.aot.flydsl.gemm

# grouped MoE (gfx1250)
python -m aiter.aot.flydsl.grouped_moe

# chunk-gdn-h
python -m aiter.aot.flydsl.chunk_gdn_h
```

### Common arguments

```bash
# Custom CSV(s) — every module supports --csv and accepts multiple paths
python -m aiter.aot.flydsl.moe --csv /path/to/config1.csv /path/to/config2.csv

# chunk_gdn_h also supports overriding the arch column for cross-compiling
python -m aiter.aot.flydsl.chunk_gdn_h --target-arch gfx942
```

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `AITER_AOT_IMPORT` | Set to `1` so `import aiter` only loads the lightweight JIT core and skips the full top-level op namespace — faster and avoids heavy import side effects during AOT compilation (this is what `setup.py` sets while pre-compiling). | `0` |
| `FLYDSL_RUNTIME_CACHE_DIR` | Cache directory | `~/.flydsl/cache` |
| `AITER_FLYDSL_AOT_WORKERS` | Process-pool concurrency (each worker uses ~1.5–2.5 GB RSS; lower it on memory-tight containers) | `min(available CPUs, 64)` |
| `AITER_CONFIGS` | Resolves the default CSV lookup path (same as the runtime JIT) | repo built-in |
| `ARCH` / `GPU_ARCHS` | **Banner/logging only** — printed as the "Target arch" line. Does **not** control the compiled target. | auto-detect |

> **About the compile target arch.** The arch each kernel is actually compiled
> for is derived per-job from the CSV's `cu_num` column (`cu_num_to_arch(...)`)
> and applied internally via `FLYDSL_GPU_ARCH`. That internal var is overwritten
> for every job, so setting `ARCH` / `GPU_ARCHS` / `FLYDSL_GPU_ARCH` in your shell
> does **not** change what gets built. To cross-compile, use
> `chunk_gdn_h --target-arch <arch>` (the only module that exposes an override),
> or edit the `cu_num` column in the CSV.

Example:

```bash
AITER_FLYDSL_AOT_WORKERS=16 python -m aiter.aot.flydsl.moe
```

---

## 2. Run the "AOT cache hit" test

Compiling successfully is not enough — you also want to verify that the **runtime
actually hits the AOT cache** (no cache miss). That is done by
`op_tests/test_moe_2stage.py`, which wraps test cases with
`aiter.aot.flydsl.common.fail_on_aot_cache_miss`: if the runtime falls back to
JIT compilation, the case fails.

Full flow:

```bash
source /path/to/venv/bin/activate

# (1) First compile the kernels into the cache
python -m aiter.aot.flydsl.moe

# (2) Then run the MoE 2stage test with cache checking.
#     When a case has check_aot_cache=True it routes through
#     test_fmoe_with_aot_cache_check, which raises AssertionError on a cache miss.
python op_tests/test_moe_2stage.py
```

> Note: both steps must use the **same** `FLYDSL_RUNTIME_CACHE_DIR` and run on
> (or target) the **same GPU arch**, otherwise step 2 will be treated as a miss
> because the cache dir / arch don't line up.

---

## 3. Troubleshooting

- **`CSV file not found`**: check the `--csv` path, or whether `AITER_CONFIGS`
  points at a valid config directory.
- **Lots of `[FAIL]` prints + exit code 1**: an individual kernel failed to
  compile; stdout has per-kernel diagnostics. The exception message inlines at
  most 10 entries (`_MAX_ERRORS_IN_MSG` in `common.py`), the rest are elided as
  `(... N more)`.
- **Worker OOM / killed**: lower `AITER_FLYDSL_AOT_WORKERS`.
- **Step 2 reports a cache miss**: confirm step 1 actually ran, the cache dir and
  arch match, and the CSV config hasn't changed.

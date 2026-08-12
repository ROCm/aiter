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
| `common.py` | — | Shared job collection, the deadlock-free fork pool, and cache-hit checking logic |

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
| `AITER_FLYDSL_AOT_WORKERS` | Max concurrent worker processes. Set explicitly to honor it verbatim (bypasses the memory cap below); `0`/negative clamps to 1. Each worker uses ~1.5–2.5 GB RSS. | `min(affinity-aware CPUs, 64)`, then capped by available memory |
| `AITER_FLYDSL_AOT_MEM_PER_WORKER_GB` | Assumed GiB/worker for the **auto memory cap** that keeps the OOM-killer from firing. Only applies when `AITER_FLYDSL_AOT_WORKERS` is **not** set; `0` disables the cap. | `2.0` |
| `AITER_FLYDSL_AOT_TIMEOUT` | Per-kernel wall-clock cap (seconds). A worker stuck *alive* past this is killed (and retried); `0` disables. | `1200` |
| `AITER_FLYDSL_AOT_MAX_RETRIES` | Retries for a worker that **died abnormally** (OOM-kill / segfault / timeout-kill). A clean compile error is never retried. `0` disables. | `2` |
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
- **Every kernel fails with the same error** (e.g. `'ArithValue' object has no
  attribute 'ir_value'`): this is a **FlyDSL version mismatch**, not a per-kernel
  problem. Check the *imported* FlyDSL:
  ```bash
  python -c "import flydsl, os; print(flydsl.__version__, os.path.dirname(flydsl.__file__))"
  ```
  If the version is older than the project's `FLYDSL_VERSION` (top-level
  `setup.py`), a stale build is winning on `PYTHONPATH`. Either `pip install` the
  matching version *and* drop the shadowing entry from `PYTHONPATH`, or rebuild
  your local FlyDSL checkout (`scripts/build.sh`, after `pip install
  nanobind==2.12.0` if CMake reports it missing) so the on-`PYTHONPATH` build dir
  is refreshed to the right version.
- **Worker OOM / killed (exitcode -9)**: abnormal exits are auto-retried
  (`AITER_FLYDSL_AOT_MAX_RETRIES`) and the default worker count is already
  memory-capped (`AITER_FLYDSL_AOT_MEM_PER_WORKER_GB`). If it still happens,
  lower `AITER_FLYDSL_AOT_WORKERS` or raise the assumed GiB/worker.
- **A kernel hangs / never finishes**: it is killed once it exceeds
  `AITER_FLYDSL_AOT_TIMEOUT` (default 1200 s) and then retried. Lower the timeout
  to fail faster, or raise it for genuinely slow kernels.
- **`hipModuleLoadData ... hipErrorNoBinaryForGpu` printed but the kernel still
  shows `[OK]`**: expected when AOT-compiling for an arch that is **not** the
  machine's GPU (e.g. building `gfx950` artifacts on a different card). MLIR
  compilation and the cache write succeed; only the *load* step fails, which AOT
  does not need. It is noise, not a failure.
- **Step 2 reports a cache miss**: confirm step 1 actually ran, the cache dir and
  arch match, and the CSV config hasn't changed.

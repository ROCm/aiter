# FlyDSL AOT Pre-compilation & Tests

This directory holds the **AOT (Ahead-Of-Time) pre-compilation entry points** for
FlyDSL kernels. Each module extracts every unique FlyDSL kernel name from aiter's
tuned CSV configs and compiles them into the cache up front, so that at runtime
the JIT path hits the cache instead of compiling again.

| Module | OpKind | Description |
| --- | --- | --- |
| `moe.py` | `MOE` | Every FlyDSL kernel the tuned fused-MoE / FHMoE rows reach, by replaying `fused_moe` itself (see below) |
| `gemm.py` | `GEMM` | GEMM kernels |
| `conv.py` | `CONV` | conv3d implicit-GEMM kernels, plus the NCDHW->NDHWC pre-transpose each NCDHW caller reaches first |
| `grouped_moe.py` | `GROUPED_MOE` | gfx1250 grouped MoE GEMM kernels |
| `chunk_gdn_h.py` | `CHUNK_GDN_H` | chunk-gdn-h opt (K5) kernels |
| `mega_moe.py` | `MEGA_MOE` | MegaMoE A8W4 profile bundles for MTPR 8192/16384/32768 |
| `fmha_fp8.py` | `FMHA_FP8` | gfx950 FP8 flash-attention forward, per head shape in `DEFAULT_SHAPES` |
| `common.py` | — | Shared job collection, the deadlock-free fork pool, and cache-hit checking logic |
| `dry_run.py` | — | Runs aiter's real host code on CPU (FakeTensors, `COMPILE_ONLY=1`, GPU runtime forbidden) |

### How `moe.py` works

`moe.py` does not restate any MoE launch logic. For every tuned row it builds
what a caller passes `fused_moe` (bf16 activations, routing, and weights
quantized and preshuffled the way `op_tests/test_moe_2stage.py` prepares them)
as FakeTensors and calls the runtime's own implementation (`_fused_moe_impl`,
or `_fhmoe_impl` for shared-expert rows) inside `dry_run`. The runtime looks the
row up, picks and parameterizes its kernels, sizes its buffers and launches its
auxiliary kernels as at inference; every FlyDSL launch on that path is
compiled into the cache instead of run. Kernel or wrapper changes therefore
need no AOT change.

- Each row is replayed at its tuned token count and at the smallest M that
  `get_padded_M` maps to it, because some compile-time parameters follow the
  actual M. `--row-tokens-only` replays the tuned count only.
- Caller-side choices a row does not fix are enumerated: expert bias where the
  quant family supports it (on both stages, and on stage 1 alone, since a
  stage-2 bias makes the runtime re-select configs whose stage 2 cannot take
  one), both gate/up layouts for MXMOE stage-1 rows, and the default and model
  SiTUv2 betas. Combinations the runtime rejects are reported as `[SKIP]`.
- aiter fixes some arch-dependent values at import, so the replay runs in one
  process per target arch with the GPUs hidden and `GPU_ARCHS` set before
  aiter is imported. `python -m aiter.aot.flydsl.moe` starts those processes
  side by side and splits `AITER_FLYDSL_AOT_WORKERS` between them by job count.
- A worker replays up to 12 jobs of one row in turn, so the kernels they share
  compile once. `AITER_FLYDSL_AOT_TIMEOUT` applies to the worker's whole batch.
- `--check` compiles nothing and fails on any kernel the replay would need but
  the cache lacks; it is the CPU-side counterpart of the step-2 test below.

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

# conv3d
python -m aiter.aot.flydsl.conv

# grouped MoE (gfx1250)
python -m aiter.aot.flydsl.grouped_moe

# chunk-gdn-h
python -m aiter.aot.flydsl.chunk_gdn_h

# MegaMoE profile bundles (all token buckets and all eight ranks)
python -m aiter.aot.flydsl.mega_moe

# Restrict the deployment profiles when building a smaller custom image.
python -m aiter.aot.flydsl.mega_moe --experts-per-rank 48
```

The FP8 flash-attention kernel picks its tile, rescale threshold and split-K
factor per call, so a serving process JIT-compiles each new combination on the
request path. `fmha_fp8` enumerates every combination the wrapper's heuristics
can reach for each head shape in its `DEFAULT_SHAPES` table. Add a model by
adding an entry (`num_heads, num_kv_heads, head_dim, head_dim_v, layout`, heads
per rank, layout `varlen_cross` for the chunked-prefill serving path).

```bash
# every configured shape (Kimi-K3 at TP8: 12:12:192:128 -> 92 kernels)
python -m aiter.aot.flydsl.fmha_fp8
# ad hoc shape, without editing DEFAULT_SHAPES
python -m aiter.aot.flydsl.fmha_fp8 --shape 12:12:192:128
```

MegaMoE defaults to all three DeepSeek-V4-Pro deployment profiles: r0/r32/r64
(`experts_per_rank=48/52/56`). The expert count is part of the compiled ABI, so
one profile cannot safely reuse another profile's bundle.

### Common arguments

```bash
# Custom CSV(s) — every module supports --csv and accepts multiple paths
python -m aiter.aot.flydsl.moe --csv /path/to/config1.csv /path/to/config2.csv
python -m aiter.aot.flydsl.chunk_gdn_h --csv /path/to/tuned.csv

# MoE: verify an existing cache on CPU instead of compiling
python -m aiter.aot.flydsl.moe --csv /path/to/config.csv --check
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
| `ARCH` / `GPU_ARCHS` | Selects which jobs to build, not what arch a job compiles *for* (that comes from the CSV's `cu_num`). `conv.py` applies it inside `parse_csv`, so both `python -m` and the `setup.py` path (`run_aot`) honour it. `gemm.py` still filters in `main()` only, so `run_aot` builds all of its archs. | auto-detect |

> **About the compile target arch.** The arch each kernel is actually compiled
> for is derived per-job from the CSV's `cu_num` column (`cu_num_to_arch(...)`)
> and applied internally via `FLYDSL_GPU_ARCH`. That internal var is overwritten
> for every job, so setting `ARCH` / `GPU_ARCHS` / `FLYDSL_GPU_ARCH` in your shell
> does **not** change what gets built. To cross-compile, edit the `cu_num`
> column in the CSV. `moe.py` additionally runs each target arch in its own
> process with `GPU_ARCHS` pinned and the GPUs hidden, so it builds the same
> kernels on a machine with or without GPUs.

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

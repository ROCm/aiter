---
name: aiter-ck-tune
description: >
 Tune and integrate a Composable-Kernel (CK-tile) kernel in AITER. Covers the
 untuned.csv / tuned.csv round-trip, the `csrc/ck_*/gen_instances.py` +
 `gemm_*_tune.py` flow, per-model tuned configs under
 `aiter/configs/model_configs/`, how the merged config at
 `/tmp/aiter_configs/...` is built, and how to use `AITER_REBUILD=1` after
 retuning. Use when a user says "tune a8w8 GEMM", "retune MoE for qwen3",
 "integrate a new CK tile kernel", or needs to add new GEMM shapes.
 Usage: /aiter-ck-tune  [--model ]
allowed-tools: Bash Read Edit Grep Glob
---

# Tune & Integrate CK Kernels in AITER

AITER uses CK-tile kernels for most GEMM-like ops (`gemm_a8w8`,
`gemm_a4w4_blockscale`, `batched_gemm_*`, `fmoe`, …). Each tuned op follows
the same 4-file pattern:

```
csrc/ck_/
├── gen_instances.py       # Emits *_instance_part{1,2,3}.cpp at JIT build time
├── gemm__tune.cu       # Tune-time pybind: takes (M,N,K,kernelId) -> latency
├── gemm__tune.py       # Python harness that runs the sweep
├── gemm_.cu            # Runtime pybind: takes (X, W, Y, config) -> Y
└── README.md                # Per-op tuning guide (follow the pattern in ck_gemm_a8w8)

aiter/configs/
├── _untuned_.csv    # Shapes you want tuned
├── _tuned_.csv      # Best kernelId per shape (auto-filled)
└── model_configs/
    └── _tuned__.csv
```

On first call, `@compile_ops` builds both the runtime module
(`module_gemm_a8w8`) and the tune module (`module_gemm_a8w8_tune`) via
`aiter/jit/optCompilerConfig.json`. `PREBUILD_KERNELS=1` prebuilds the runtime
module only; `_tune` modules remain JIT.

---

## Step 1: Add your GEMM shapes

Edit the untuned CSV in `aiter/configs/`. Example for
`aiter/configs/a8w8_untuned_gemm.csv`:

```csv
M,N,K
128,1536,7168
256,1536,7168
1024,8192,1024
```

Column order: as given in the per-op README (usually `M,N,K`; batched GEMM
adds a `B` column). Don't reorder — the parser is positional.

For a per-model tune, put the untuned shapes under
`aiter/configs/model_configs/_untuned__.csv`.

## Step 2: Run the tuner

For `gemm_a8w8`:

```bash
python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
    -i aiter/configs/a8w8_untuned_gemm.csv \
    -o aiter/configs/a8w8_tuned_gemm.csv
```

This triggers a JIT build of `module_gemm_a8w8_tune` (builds ALL CK instances —
takes a few minutes). Then for every row in `-i` it sweeps every instance,
picks the fastest that satisfies `errRatio < 0.05`, and writes the winner
to `-o`.

Key tuner flags (see `csrc/ck_gemm_a8w8/README.md` for the full list):

| Flag | Default | Purpose |
|------|---------|---------|
| `-k, --splitK` | off | Include split-K variants |
| `-o2, --profile_file` | "" | Dump all candidates (not just the best) for offline analysis |
| `--errRatio` | `0.05` | Tolerable `|got - ref|/|ref|` |
| `--mp` | #GPUs | Parallel processes across GPUs |
| `--batch` | `100` | Shapes per batch (memory management) |
| `--warmup` | `5` | Warmup iters before timing |
| `--iters` | `101` | Benchmark iters |
| `--timeout` | none | Per-task timeout |
| `--all` | off | Retune everything (otherwise only new rows) |
| `-v, --verbose` | off | Verbose progress |
| `--sort` | on | Keep tuned CSV sorted by `cu_num,N,M,K` |

Multi-GPU sweep:

```bash
python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i ... -o ... --mp 8
```

The output CSV schema:

```
cu_num,M,N,K,q_dtype_w,kernelId,splitK,us,kernelName,tflops,bw,errRatio
```

`cu_num` encodes the chip (e.g. `80` for MI300X). Rows are keyed on
`(cu_num, M, N, K, q_dtype_w)`.

## Step 3: Per-model tuning (optional)

Under `aiter/configs/model_configs/` each model has its own tuned pairs:

```
a8w8_blockscale_tuned_gemm_dsv3.csv
a8w8_blockscale_untuned_gemm_dsv3.csv
a8w8_blockscale_tuned_gemm_qwen3_235b.csv
...
```

Point the tuner at the untuned variant:

```bash
python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py \
    -i aiter/configs/model_configs/a8w8_blockscale_untuned_gemm_qwen3_235b.csv \
    -o aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_qwen3_235b.csv
```

At runtime, AITER merges the generic tuned CSV with **all** matching
`model_configs/*_tuned_*.csv` entries into
`/tmp/aiter_configs/_tuned_.csv`. That merged file is what the
runtime module consumes. See `aiter/jit/core.py::AITER_CONFIG` for the
resolution order.

Override the merged path via the matching env var:

```bash
# Examples from aiter/jit/core.py
AITER_CONFIG_GEMM_A8W8=/path/to/my_a8w8_tuned_gemm.csv python3 op_tests/test_gemm_a8w8.py
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=/path/to/custom.csv  python3 bench.py
```

## Step 4: Rebuild the runtime module with the new config

If `PREBUILD_KERNELS=0` (default), the next test call JIT-builds the winning
kernels on demand — no extra step needed.

If you had `PREBUILD_KERNELS=1` during `setup.py develop`, pre-existing `.so`
files were built against the **old** tuned CSV. Force a rebuild:

```bash
AITER_REBUILD=1 python3 op_tests/test_gemm_a8w8.py
# Or wipe first:
rm -rf aiter/jit/build aiter/jit/module_gemm_a8w8.so
python3 op_tests/test_gemm_a8w8.py
```

## Step 5: Verify correctness and perf

```bash
# Correctness across the tuned shapes:
python3 op_tests/test_gemm_a8w8.py

# Single shape, verbose:
AITER_LOG_MORE=1 AITER_LOG_TUNED_CONFIG=1 python3 op_tests/test_gemm_a8w8.py -m 128 -n 1536 -k 7168

# Perf bench vs torch reference:
python3 op_tests/op_benchmarks/triton/bench_gemm_a8w8.py
```

`AITER_LOG_TUNED_CONFIG=1` prints which config file was used and which
kernelId won for each call — crucial for spotting cases where the merged CSV
lost a row.

---

## Integrating a brand-new CK op

If you're adding `ck_my_gemm`:

1. **Copy a template**. Start from `csrc/ck_gemm_a8w8/` (contains
   `gen_instances.py`, `gemm_a8w8_tune.{cu,py}`, `gemm_a8w8.cu`, plus the
   header under `csrc/ck_gemm_a8w8/include/`).
2. **Regenerate instances**. CK's own generator (`example/ck_tile/*/generate.py`)
   in `3rdparty/composable_kernel/` produces instances. `gen_instances.py`
   wraps that call with the right `--filter` to keep the part-files small.
3. **Add 2 JIT modules** to `aiter/jit/optCompilerConfig.json`:
   - `module_my_gemm` (runtime) and `module_my_gemm_tune` (tuner). Only the
     tune module should appear in `_tune` filter in `setup.py::get_exclude_ops`
     so PREBUILD_KERNELS skips it.
4. **Python wrapper** in `aiter/ops/gemm_op_my_gemm.py` using `@compile_ops`.
5. **Tuned CSV pair** in `aiter/configs/my_gemm_{un,}tuned_gemm.csv`.
6. **Per-op README** in `csrc/ck_my_gemm/README.md` — copy the layout of
   `ck_gemm_a8w8/README.md` so users know what flags you support.

## Tips for CK tuning

- CK instance files are template-heavy. If compile is slow, split
  `gen_instances.py` output into `*_part{1,2,3}.cpp`. See
  `3rdparty/composable_kernel/library/src/tensor_operation_instance/gpu/
  gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_*`.
- `MAX_JOBS=8` is a good starting point for CK compile; each instance
  can eat 4–8 GB of RAM.
- `--profile_file` dumps every candidate; inspect it to spot cases where
  the winner is only 1% faster than #2 — your tuned config is noise-limited.
- Keep `errRatio` at `0.05` for FP16/BF16 and tighten to `0.01` for FP32.
  FP8 /FP4 workloads may need larger tolerances — check the test's own
  `rtol/atol`.
- Model-specific tuned configs (`model_configs/*_qwen3_235b.csv`) are
  **additive**. Don't remove rows from the generic `a8w8_tuned_gemm.csv`
  just because the model config has them.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Tuner silently skips most shapes | `errRatio` too tight, or kernels all fail correctness. Re-run with `-v` and widen `--errRatio`. |
| New shape added but still "kernel not found" at runtime | Merged config at `/tmp/aiter_configs/...` is stale. `rm -rf /tmp/aiter_configs && AITER_REBUILD=1 python3 ...` |
| Tune module OOMs during compile | Split instances (`gen_instances.py` → more `*_part*.cpp`) and/or drop `MAX_JOBS`. |
| Perf regressed after retune | Check `errRatio` of the new winner vs the old one in `--profile_file`. The new pick may be 0.1% faster at tune time but noisy in the real workload. |
| Different CU counts produce different winners | Expected. `cu_num` is part of the key; tune on each SKU you care about (MI300X=80, MI300A=38, MI325X=88, MI350X=?). |

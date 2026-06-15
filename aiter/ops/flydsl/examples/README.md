# FlyDSL op examples

A small, repeatable format for each FlyDSL op: a **PyTorch reference**
implementation, a **FlyDSL kernel** invocation, and a machine-readable
`config.json` of shapes/params that drives both. Each script is standalone and
prints what it ran (shapes, dtype, basic output stats).

## Directory convention

Every op is one subfolder with the **same file layout**:

```
examples/
  _common.py            # shared helpers (availability, config load, inputs, timing, JSON)
  README.md             # this file
  <op>/
    pytorch_ref.py      # PyTorch reference; runs over the shapes in config.json
    flydsl_run.py       # FlyDSL kernel; runs over the shapes in config.json
    config.json         # shapes + tiling/params for this op (real AITER configs)
    kernel_sources.md   # pointers to the authoritative FlyDSL kernel files
    README.md           # per-op docs (math, how to run, config schema)
```

Currently implemented:

- `hgemm/` — plain bf16/f16 GEMM, `out = a @ b.T` with fp32 accumulation.
- `moe/` — fused Mixture-of-Experts (routing + gate/up GEMM + gated activation +
  down GEMM + weighted combine); FlyDSL path uses the quantized a4w4 two-stage
  kernels.

## Running

Both scripts are runnable as modules and directly:

```bash
# from the repo root
python -m aiter.ops.flydsl.examples.hgemm.pytorch_ref
python -m aiter.ops.flydsl.examples.hgemm.flydsl_run

# or directly (the scripts add the repo root to sys.path if needed)
cd aiter/ops/flydsl/examples/hgemm
python pytorch_ref.py        # CPU-friendly reference
python flydsl_run.py         # needs ROCm + flydsl
```

Each accepts `--config <path>`, `--case <name>` (repeatable), `--seed <int>`,
and `--json <path>` to emit machine-readable run outputs. `flydsl_run.py` also
takes `--time` (with `--warmup`/`--iters`) for median time and TFLOP/s.

If ROCm/CUDA or the optional `flydsl` package is missing, `flydsl_run.py` prints
a clear message and exits 0 (it does not crash). The PyTorch reference runs on
CPU when no device is available.

## Adding a new op

1. Create `examples/<op>/` and copy the file layout above.
2. `pytorch_ref.py`: a standalone, documented reference function plus a
   `main()` that loads `config.json` and runs every case, printing stats.
3. `flydsl_run.py`: import the op from `aiter.ops.flydsl` (guard the import so
   the file loads without `flydsl`), run every case from `config.json`, print
   stats.
4. `config.json`: list representative shapes plus known-valid kernel params.
   Prefer **real shapes** from AITER's shipped configs under `aiter/configs/`
   (and `aiter/configs/model_configs/`) and record each case's `source`.
   Include a `_schema` block describing each field (JSON has no comments).
5. `kernel_sources.md`: list the repo-relative path(s) of the authoritative
   FlyDSL kernel file(s) and their entry symbols, so the implementation is easy
   to find without searching.
6. `README.md`: the math, how to run, expected output, the config schema, and a
   link to `kernel_sources.md`.

Reuse `_common.py` for anything shared; only add op-specific logic in the op
folder. If `_common` needs a new generic helper (e.g. a different FLOP formula),
add it there so later ops benefit.

> **Note on `config.json`:** it is **required for shape-heavy ops** (attention,
> MoE) where the shape/layout space is large and must be enumerated explicitly.
> For simple ops it can be minimal (a couple of shapes); keeping it is
> recommended for consistency.

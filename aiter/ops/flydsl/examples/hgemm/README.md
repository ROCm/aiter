# HGEMM example

Plain half-precision GEMM where the second operand is stored transposed
(`[N, K]`), matching the `flydsl_hgemm` wrapper contract.

## Math

For `a` of shape `[M, K]` and `b` of shape `[N, K]`:

```
out[m, n] = sum_k a[m, k] * b[n, k]      # out = a @ b.T
```

- dtype: `bf16` or `f16` inputs/outputs.
- accumulation: **fp32**, result cast back to the input dtype.

## Files

| File             | Role                                                        |
| ---------------- | ----------------------------------------------------------- |
| `pytorch_ref.py` | `hgemm_ref(a, b)` + runner over the shapes in `config.json`.|
| `flydsl_run.py`  | runs `flydsl_hgemm` over the shapes in `config.json`.       |
| `config.json`    | representative `(m, n, k)` shapes + valid tiling params.     |

## Running

```bash
# PyTorch reference (runs on CPU or GPU)
python pytorch_ref.py
python pytorch_ref.py --case square_m256_n256_k512_default --json ref_out.json
python -m aiter.ops.flydsl.examples.hgemm.pytorch_ref

# FlyDSL kernel (needs ROCm + flydsl)
python flydsl_run.py
python flydsl_run.py --time --json flydsl_out.json
python -m aiter.ops.flydsl.examples.hgemm.flydsl_run
```

### Expected output

Each script prints a per-case row with the input shape, output shape, and basic
output stats (min/max/mean/std):

```
CASE                              SHAPE(MxNxK)   OUT       MIN      MAX      MEAN     STD
square_m256_n256_k512_default     256x256x512    256x256   ...      ...      ...      ...
...
```

With `--time`, `flydsl_run.py` adds `MEDIAN_ms` and `TFLOPs` columns. With
`--json`, the same per-case data plus an `environment` block is written as JSON.

When ROCm/CUDA or `flydsl` is unavailable, `flydsl_run.py` prints a one-line
skip message and exits 0.

## Config schema (`config.json`)

Top-level keys:

- `op` — `"hgemm"`.
- `description` — the math summary.
- `dtype` — `"bf16"` or `"f16"` (shared by all cases).
- `default_tiling` — kwargs passed to `flydsl_hgemm`: `tile_m`, `tile_n`,
  `tile_k`, `split_k`, `block_m_warps`, `block_n_warps` (the validated wrapper
  defaults: `128/128/64`, `split_k=1`, `2x2` warps).
- `cases` — list of entries; each needs `name`, `m`, `n`, `k` and may override
  any tiling key (e.g. `tile_m`, `split_k`, `b_to_lds`).

A `_schema` object at the top of the file documents every field inline (JSON
has no comments).

### Valid-shape constraints (enforced by the wrapper)

- `n % tile_n == 0` and `n >= tile_n`.
- `k % split_k == 0`, and `(k / split_k) % tile_k == 0` with `tile_k >= 32`.
- `tile_m % (block_m_warps * 16) == 0`, `tile_n % (block_n_warps * 16) == 0`.

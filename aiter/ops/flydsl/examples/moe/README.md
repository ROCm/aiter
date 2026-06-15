# MoE example

Fused token-choice Mixture-of-Experts: route each token to its top-k experts,
apply a per-expert gate/up projection with a gated activation, a down
projection, and combine the expert outputs with the routing weights.

Kernel source: see [`kernel_sources.md`](kernel_sources.md) for the exact
FlyDSL kernel files and entry points.

## What this example covers

`pytorch_ref.py` — the **unquantized** reference (the standalone "source"
formulation):

- standard fused-MoE math: softmax router over gate logits, top-k selection
  with **renormalized** weights;
- per-expert gate/up GEMM (`w1`, `[E, 2*I, D]`) producing `[gate | up]`, followed
  by the **silu**-gated activation `silu(gate) * up`;
- per-expert down GEMM (`w2`, `[E, D, I]`), then the weighted top-k combine;
- **fp32** accumulation, **bf16** inputs and output. No quantization.

`flydsl_run.py` — drives the real **two-stage quantized a4w4** kernels:

- **stage1** `flydsl_moe_stage1`: gate/up GEMM fused with the silu-gated
  activation → `[T, topk, I]`.
- **stage2** `flydsl_moe_stage2`: down GEMM + weighted top-k combine → `[T, D]`.
- quantization is **a4w4**: activations and weights are mxfp4
  (`torch.float4_e2m1fn_x2`), with **e8m0** block scales and
  `QuantType.per_1x32`; weights/scales are **pre-shuffled**; routing uses a
  **sorted** token/expert dispatch built by `moe_sorting`.
- dtype boundaries: **bf16** inputs → quantized to **mxfp4 + e8m0 scales** for
  each GEMM → **bf16** output. The stage1 output is **re-quantized to mxfp4**
  before stage2 (the stage2 kernel consumes mxfp4 activations).

The two sides are **not bit-exact**: the reference is bf16-fidelity unquantized,
the kernel path is mxfp4. They are not compared in-repo; each script runs its
own side and prints output stats.

## Math

For `T` tokens, model dim `D`, intermediate dim `I`, `E` experts, `topk`
selected per token, with `w1` of shape `[E, 2*I, D]` and `w2` of shape
`[E, D, I]`:

```
scores      = router(hidden)                 # [T, E]
g           = softmax(scores)
w, ids      = topk(g, topk)                   # per-token weights + expert ids
gate_up     = x @ w1[e].T                      # [., 2*I]  for tokens routed to e
h           = silu(gate) * up                  # split [gate | up], gated act
y_e         = h @ w2[e].T                       # [., D]
out         = sum over topk of  w * y_e         # [T, D]
```

- accumulation: **fp32**, result cast back to the input dtype.
- activation: `silu` (default) or `gelu`, applied to the gate half.

## Files

| File             | Role                                                            |
| ---------------- | --------------------------------------------------------------- |
| `pytorch_ref.py` | `moe_ref(...)` standard fused-MoE + runner over `config.json`.  |
| `flydsl_run.py`  | runs the FlyDSL a4w4 stage1 + stage2 path over `config.json`.   |
| `config.json`    | representative MoE shapes + the kernel params.                   |
| `kernel_sources.md` | pointers to the authoritative FlyDSL kernel files.           |

## Running

```bash
# PyTorch reference (runs on CPU or GPU, torch only)
python pytorch_ref.py
python pytorch_ref.py --case dsv3_t16_e257_k9 --json ref_out.json
python -m aiter.ops.flydsl.examples.moe.pytorch_ref

# FlyDSL a4w4 kernels (needs ROCm + flydsl)
python flydsl_run.py
python flydsl_run.py --json flydsl_out.json
python -m aiter.ops.flydsl.examples.moe.flydsl_run
```

### Expected output

A per-case row with the token count, dims, output shape `[T, D]`, and basic
output stats (min/max/mean/std):

```
CASE                TOKENS  DIMS                 OUT       MIN   MAX   MEAN   STD
dsv3_t16_e257_k9    16      D7168/I256/E257/k9   16x7168   ...   ...   ...    ...
...
```

With `--json`, the same per-case data plus an `environment` block is written as
JSON. When ROCm/CUDA or `flydsl` is unavailable, `flydsl_run.py` prints a
one-line skip message and exits 0.

## Config schema (`config.json`)

Top-level keys:

- `op` — `"moe"`.
- `description` — the math summary.
- `dtype` — reference/output dtype, `"bf16"` or `"f16"`.
- `default_params` — shared params (per-case keys override): `activation`,
  `a_dtype`/`b_dtype` (`fp4`), `quant_type` (`per_1x32`), `block_m`, `tile_n`,
  `tile_k`, `mode`.
- `cases` — each needs `name`, `tokens`, `model_dim`, `inter_dim`, `experts`,
  `topk`, and a `source` (the CSV/model the row came from).

A `_schema` object at the top of the file documents every field inline (JSON
has no comments).

The bundled cases are **real a4w4 fused-MoE configs** from AITER's shipped CSVs
(rows with `q_dtype_a`/`q_dtype_w = torch.float4_e2m1fn_x2` and
`q_type = QuantType.per_1x32`), each tagged with its `source`:

- `aiter/configs/model_configs/dsv3_fp4_untuned_fmoe.csv` (DeepSeek-V3) —
  `D=7168, I=256, E=257, topk=9`.
- `aiter/configs/model_configs/kimik2_fp4_untuned_fmoe.csv` (Kimi-K2) —
  `D=7168, I=256, E=384, topk=8`.

### Valid-shape constraints

- `model_dim % tile_k == 0`.
- `2 * inter_dim % tile_n == 0`.
- `experts >= topk`.

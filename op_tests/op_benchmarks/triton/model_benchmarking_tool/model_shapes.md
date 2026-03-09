# model_shapes.json — Structure and fields

This file defines benchmark shapes per model and kernel for the model benchmarking script. It is keyed by **model name**; each model has **kernel names** mapping to a **list of shape objects**.

For a new kernel to be run by the script, the corresponding benchmark must be imported in `bench_models.py` and the kernel name (as used in this JSON) must be added to `KERNEL_DICT` there.

## Top-level structure

```text
{
  "<model_name>": {
    "<kernel_name>": [ { <shape_fields> }, ... ],
    ...
  },
  ...
}
```

- **Model name**: string (e.g. `"Llama3 405B"`, `"GPT-OSS 120B"`).
- **Kernel name**: string identifying the benchmark (see below).
- **Shape list**: one or more objects; each object describes one shape variant.

## Kernel types and shape fields

### GEMM kernels (e.g. `gemm_a16w16`, `gemm_a8w8_per_token_scale`, `gemm_a8w8_blockscale`, `gemm_afp4wfp4`)

| Field     | Type   | Description |
|----------|--------|-------------|
| `N`      | int    | Output/inner dimension N. |
| `K`      | int    | Inner dimension K. |
| `TP_dim` | string \| null | `"N"`, `"K"`, or `null`. |

### Batched GEMM kernels (`batched_gemm_a8w8`, `batched_gemm_afp4wfp4`)

| Field     | Type   | Description |
|----------|--------|-------------|
| `B`      | int    | Batch size. |
| `N`      | int    | Dimension N. |
| `K`      | int    | Dimension K. |
| `TP_dim` | string \| null | `"B"`, `"N"`, `"K"`, or `"null"`. |

### MoE GEMM kernels (e.g. `moe_op_gemm_a8w8`, `moe_op_gemm_a8w8_blockscale`, `moe_op_gemm_a8w4`, `moe_op_gemm_a4w4`)

| Field   | Type | Description |
|--------|------|-------------|
| `E`     | int | Number of experts. |
| `Dim1`  | int | First dimension (i.e. hidden_size). |
| `Dim2`  | int | Second dimension (i.e. moe_intermediate_size*2). |
| `TopK`  | int | Number of experts per token (top-k). |

### RMSNorm (`rmsnorm`)

| Field | Type | Description |
|-------|------|-------------|
| `N`   | int  | Normalization dimension. |

### RoPE (`rope`)

| Field          | Type   | Description |
|---------------|--------|-------------|
| `num_heads`   | int    | Total number of query heads. |
| `num_kv_heads`| int    | Number of key/value heads. |
| `head_dim`    | int    | Head dimension. |
| `two_inputs`  | string | `"true"` or `"false"`. |
| `positions`   | string | `"true"` or `"false"`. |
| `rotate_style`| string | `"neox"` or `"gptj"`. |

### MHA — Multi-Head Attention (`mha`)

| Field     | Type   | Description |
|----------|--------|-------------|
| `hq`     | int    | Number of query heads. |
| `hkv`    | int    | Number of key/value heads. |
| `dqk`    | int    | Query/key head dimension. |
| `dv`     | int    | Value head dimension. |
| `comment`| string | Optional label (e.g. `"Prefill"`, `"Text"`, `"Vision"`). |

### MLA — Multi-head Latent Attention (`mla`)

Uses the same shape fields as `mha`:

| Field     | Type   | Description |
|----------|--------|-------------|
| `hq`     | int    | Number of query heads. |
| `hkv`    | int    | Number of key/value heads. |
| `dqk`    | int    | Query/key head dimension. |
| `dv`     | int    | Value head dimension. |
| `comment`| string | Optional label (e.g. `"Decode"`). |

## Example

```json
{
  "Llama3 405B": {
    "gemm_a8w8_per_token_scale": [
      { "N": 106496, "K": 16384, "TP_dim": "N" },
      { "N": 16384, "K": 53248, "TP_dim": "K" }
    ],
    "rmsnorm": [
      { "N": 16384 }
    ],
    "rope": [
      {
        "num_heads": 128,
        "num_kv_heads": 8,
        "head_dim": 128,
        "two_inputs": "true",
        "positions": "true",
        "rotate_style": "neox"
      }
    ],
    "mha": [
      { "hq": 128, "hkv": 8, "dqk": 128, "dv": 128 }
    ]
  }
}
```

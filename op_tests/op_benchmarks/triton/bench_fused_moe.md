# User guide: `bench_fused_moe.py`

This document describes how to run the **fused MoE SiLU** Triton benchmark (`fused_moe_silu`) using the same model-driven workflow as `bench_moe.py`.

## What this script does

- Reads architecture fields from **`utils/model_configs.json`** (via `get_model_configs`).
- For each selected **MoE** model, builds one or two **GEMM-shaped** MoE problem sizes `(M, N, K)` with expert count `E` and routing depth `top_k`, then times **`fused_moe_silu`** with `triton.testing.do_bench` (warmup 25, repeat 100).
- Reports **time (ms)**, **TFLOPS**, and **effective bandwidth (GB/s)** in a Triton `perf_report` table, and optionally writes a **CSV** in the current directory when **`-o`** is passed.

Unlike `bench_moe.py`, this script **only** exercises the **`fused_moe_silu`** kernel path (no non-fused `fused_moe` variant).

## Requirements

- **Environment:** AITER tree with Triton op tests (e.g. repo layout where `op_tests.op_benchmarks.triton.utils` and `op_tests.triton_tests.moe.test_moe` resolve).
- **`PYTHONPATH`** must include the AITER package root (for example `/workspace/aiter` in the standard container layout).
- **Working directory:** Run from `op_tests/op_benchmarks/triton` so default paths like `utils/model_configs.json` resolve, **or** pass an explicit **`-model_configs`** / **`--model-configs`** path.
- **Hardware:** A GPU is expected; the benchmark executes real kernels.

## Which models are valid?

Entries in `model_configs.json` must include MoE fields used by the script:

- `hidden_size`, `intermediate_size`
- `num_expert`, `top_k`

Dense-only configs (no MoE) will not work with this benchmark. Use MoE families such as **Mixtral**, **Llama 3** (with experts), **DeepSeek-V3**, **Qwen3** MoE variants, etc., as defined in your JSON.

To see names the loader knows about, run:

```bash
python3 bench_fused_moe.py --help
```

The help text lists dynamically discovered **`--model`** values.

## Command-line reference

| Option | Description |
|--------|-------------|
| **`--model`**, **`-model`** | Comma-separated model keys (e.g. `llama3-8B`, `mixtral-7B`) or `all`. If omitted, the default selection matches `bench_moe.py` (**mixtral**). |
| **`--dtype`**, **`-dtype`** | Dtype key for `str_to_torch_dtype` (e.g. `fp16`, `bf16`, `float16`). Default: `fp16`. |
| **`-M`** | Token batch size(s), **comma-separated**. Default: **`1,4,16,64,256,1024,4096`**. Examples: `-M 4096` (single size) or `-M 1,64,4096`. |
| **`-model_configs`**, **`--model-configs`** | Path to `model_configs.json` (default: `utils/model_configs.json`). |
| **`-o`** | Write Triton CSV output into the **current working directory** (same behavior family as other `bench_*.py` scripts). |
| **`-routed_weight`** | Enable routed-weight path in the MoE helper. |
| **`-fp8_w8a8`**, **`-int8_w8a16`**, **`-int4_w4a16`** | Quantization paths (when supported by helpers and kernel). |
| **`-group_size`** | Required for **`-int4_w4a16`** (GPTQ/AWQ-style block size). |
| **`-has_zp`** | Zero-point flag for int4 path. |
| **`-fp8_type`** | FP8 storage type key (default `e5m2fnuz`) when FP8 is used. |
| **`-print_time`** | Restrict reported lines to time only. |
| **`-print_vgpr`** | Dump VGPR-related autotune output (see `benchmark_utils.print_vgpr`). |
| **`-no_bench_stage2`** | Flips whether a **second** `(N, K)` line per model is included (same semantics as `bench_moe.py`; default keeps both stage-1 and stage-2 style shapes when the flag is left at its default). |

Use **`python3 bench_fused_moe.py -h`** for the full, up-to-date list.

## Examples

From `op_tests/op_benchmarks/triton` with `PYTHONPATH` set to your AITER root:

```bash
# Single model, FP16 (uses default M sweep: 1,4,16,64,256,1024,4096)
python3 bench_fused_moe.py --model mixtral-7B --dtype fp16

# Explicit batch size and bfloat16
python3 bench_fused_moe.py --model llama3-8B --dtype bf16 -M 2048

# Multiple models and CSV output
python3 bench_fused_moe.py --model mixtral-7B,llama3-70B --dtype fp16 -M 4096 -o

# All MoE entries in the JSON (long run)
python3 bench_fused_moe.py --model all --dtype fp16 -o
```

## Output

- **Stdout:** A pretty-printed table with columns such as `model`, `M`, `N`, `K`, `E`, `top_k`, and metrics (unless `-print_time` narrows it).
- **CSV:** With **`-o`**, Triton writes a CSV named from the benchmark plot name (typically tied to the script name) in the **current directory**.

## Relationship to `bench_moe.py`

| Topic | `bench_moe.py` | `bench_fused_moe.py` |
|--------|----------------|-------------------------|
| Kernel | `fused_moe` or `fused_moe_silu` (via **`-silu_fused`**) | **`fused_moe_silu` only** |
| Model JSON | Same | Same |
| Dtype | Parsed; note legacy `bench_moe` may hard-code FP16 in some call paths | **`--dtype` / `-dtype`** applied consistently to helpers and kernel dtype |

For architecture documentation of fields in `model_configs.json`, see **`utils/new_models_config.md`** (if present in your tree).

## Troubleshooting

- **`KeyError` on `num_expert` / `top_k`:** The chosen model is not a MoE entry in `model_configs.json`; pick a MoE model or extend the JSON.
- **Import errors:** Set **`PYTHONPATH`** to the AITER root and run from the Triton benchmarks directory (or fix `-model_configs`).
- **Empty or wrong CSV location:** Remember **`-o`** writes relative to the **process current working directory**, not the script path.

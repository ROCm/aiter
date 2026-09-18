# User guide: `bench_fp8_quantization.py`

This document describes how to run the **FP8 quantization** microbenchmark that compares an **AITER Triton** path (`fused_flatten_fp8_group_quant`) with an optional **PyTorch** row-wise reference (`torch_row`).

## What this script does

- Builds activation tensors of shape **`(M, K)`** on CUDA, reshapes to **`(M, 1, K)`** for the AITER **flatten + per-group FP8** kernel (see below).
- Times the chosen kernel with **`triton.testing.do_bench`** (warmup 25, repeat 100), wrapped in **`triton.testing.perf_report`**.
- Reports **time (ms)**, a rough **TFLOPS** estimate, and an **effective bandwidth (GB/s)** heuristic (bytes moved per iteration / time).
- With **`-o`**, writes a CSV in the **current working directory** (Triton naming, typically tied to the script stem).

## Kernels

| `--kernel` | Implementation |
|------------|------------------|
| **`aiter_flatten`** (default) | `aiter.ops.triton.quant.fused_fp8_quant.fused_flatten_fp8_group_quant` on **`x.view(M, 1, K)`**, FP8 dtype **`aiter.dtypes.fp8`**, group size **`--group-size`** (default **128**). |
| **`torch_row`** | PyTorch: row-wise `abs` → `amax` on the last dim, scale by **`torch.finfo(fp8).max`**, cast activations to **`aiter.dtypes.fp8`**. Useful as a simple baseline; **not** numerically identical to per-group AITER quantization. |

## Model-driven vs synthetic shapes

- **`--model` / `-model`**: loads **`hidden_size`** from **`utils/model_configs.json`** (via `get_model_configs`) and uses it as **`K`** for each selected model. **`M`** comes from **`-M`**. **`-K` is ignored** when `--model` is set.
- **No `--model`**: uses synthetic labels like **`synthetic-K7168`**. **`K`** is taken from **`-K`** (comma-separated; default **`7168`**). **`M`** from **`-M`** (default sweep **`1,4,16,64,256,1024,4096`**).

Any model entry that defines **`hidden_size`** can be used; run **`python3 bench_fp8_quantization.py --help`** to see dynamically listed ids.

## Requirements

- **AITER** installed and importable; **`PYTHONPATH`** should include the package root (e.g. **`/workspace/aiter`** in the usual container layout).
- Run from **`op_tests/op_benchmarks/triton`** so **`utils/model_configs.json`** resolves, or pass **`-model_configs`** / **`--model-configs`**.
- **GPU** required (CUDA tensors and Triton).

## Command-line reference

| Option | Description |
|--------|-------------|
| **`--model`**, **`-model`** | Model id(s) from `model_configs.json` (comma-separated / `all` per `get_model_configs` rules). Sets **`K = hidden_size`**. |
| **`-M`** | Row size(s) **`M`**, comma-separated. Default: **`1,4,16,64,256,1024,4096`**. |
| **`-K`** | Hidden size(s) **`K`** when **`--model` is omitted**. Default: **`7168`**. Ignored if **`--model`** is set. |
| **`--kernel`**, **`-kernel`** | **`aiter_flatten`** or **`torch_row`**. |
| **`--group-size`**, **`-group-size`** | Group size for **`aiter_flatten`** (default **128**). Ignored for **`torch_row`**. |
| **`--dtype`**, **`-dtype`** | Activation dtype for `str_to_torch_dtype` (e.g. **`bf16`**, **`fp16`**). Default: **`bf16`**. |
| **`-model_configs`**, **`--model-configs`** | Path to JSON (default **`utils/model_configs.json`**). |
| **`-o`** | Write Triton CSV to the **current directory**. |
| **`-print_time`** | Report only time column. |
| **`-print_vgpr`** | VGPR / autotune dump path (see `benchmark_utils.print_vgpr`). |

Use **`python3 bench_fp8_quantization.py -h`** for the full list.

## Examples

From `op_tests/op_benchmarks/triton` with `PYTHONPATH` set to your AITER root:

```bash
# AITER flatten+group FP8 quant, synthetic M×K
python3 bench_fp8_quantization.py --kernel aiter_flatten --dtype bf16 -M 1 -K 7168

# PyTorch row-wise reference, same shape
python3 bench_fp8_quantization.py --kernel torch_row --dtype bf16 -M 1 -K 7168

# Use DeepSeek-V3 hidden size as K, sweep M
python3 bench_fp8_quantization.py --model deepseek-V3 --kernel aiter_flatten --dtype bf16 -o

# CSV output
python3 bench_fp8_quantization.py -M 1,4 --kernel aiter_flatten -o
```

## Output

- **Stdout:** Triton table with columns **`model`**, **`M`**, **`K`**, **`kernel`**, plus metrics.
- **CSV:** With **`-o`**, filename follows Triton’s plot naming (typically **`bench_fp8_quantization.csv`** in the cwd).

## Notes

- **TFLOPS** here is a **rough** heuristic (`6 * M * K` ops / time), not a strict FLOP definition for quantization.
- **Bandwidth** is an approximate **memory traffic** model (activations + FP8 output + block scales for **`aiter_flatten`**).
- For **numerical correctness** of AITER ops, use the official **`op_tests/triton_tests/quant/test_fused_fp8_quant.py`** tests; this script is for **performance exploration**.

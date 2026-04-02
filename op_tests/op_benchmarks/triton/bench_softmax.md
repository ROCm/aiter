# `bench_softmax.py` user guide

Benchmarks **softmax** on `(M, N)` tensors along the last dimension:

- **`aiter`** — `aiter.ops.triton.softmax.softmax` (Triton)
- **`torch`** — `torch.nn.functional.softmax`

Requires **CUDA**, **PyTorch**, **Triton**, and the **aiter** package on `PYTHONPATH` (run from repo root or set `PYTHONPATH` to `/workspace/aiter`).

## Where to run

From the Triton benchmarks directory (so `utils/model_configs.json` resolves):

```bash
cd /workspace/aiter/op_tests/op_benchmarks/triton
python3 bench_softmax.py [options]
```

Or from elsewhere with `PYTHONPATH`:

```bash
export PYTHONPATH=/workspace/aiter
python3 /workspace/aiter/op_tests/op_benchmarks/triton/bench_softmax.py [options]
```

## Quick examples

```bash
# AITER kernel, bf16, single shape
python3 bench_softmax.py --kernel aiter --dtype bf16 -M 1 -N 7168

# PyTorch reference, multiple M
python3 bench_softmax.py --kernel torch -M 1,4 -N 7168

# Model-driven N (from model_configs.json), write CSV
python3 bench_softmax.py --kernel aiter --model deepseek-V3 --dtype bf16 -o

# Only hidden_size as N
python3 bench_softmax.py --model llama3-405B --config-n hidden -o
```

## Main options

| Option | Default | Meaning |
|--------|---------|---------|
| `--kernel` / `-kernel` | `aiter` | `aiter` or `torch` |
| `--dtype` / `-dtype` | `bf16` | Activation dtype (`str_to_torch_dtype`, e.g. `fp16`, `bf16`) |
| `-M` | `1,4,16,64,256,1024,4096` | Row counts (comma-separated) |
| `-N` | `7168` | Last dim when **no** `--model` (comma-separated) |
| `--model` / `-model` | *(unset)* | Model id(s) from JSON; **N** comes from `--config-n`, not `-N` |
| `--config-n` | `both` | With `--model`: `hidden`, `intermediate`, or `both` for N |
| `--model-configs` | `utils/model_configs.json` | Path to model JSON (relative to cwd) |
| `-print_time` | off | Print only time column in the report |
| `-print_vgpr` | off | Print VGPR usage for compiled Triton kernel |
| `-o` | off | Write Triton **CSV** in the **current working directory** |

## Output

- **Stdout:** Triton `perf_report` table (time, and optionally TFLOPS and bandwidth).
- **With `-o`:** CSV in `.` (working directory), not necessarily next to the script unless you `cd` there first.

## Shapes

- **Synthetic:** omit `--model`; sweep `-M` × `-N`.
- **From models:** set `--model` (comma-separated ids allowed via `get_model_configs`); **N** = `hidden_size` / `intermediate_size` / both per `--config-n`.

## Help

```bash
python3 bench_softmax.py --help
```

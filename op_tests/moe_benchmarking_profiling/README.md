# MOE Benchmarking & Profiling Pipeline

MOE kernel benchmarking, profiling, and performance analysis.

## Scripts

| Script | Purpose |
|--------|---------|
| `extract_configs.py` | Extract config from trace files (JSON) → configs/ |
| `benchmark.py` | Benchmark + select best kernels (ASM, CK, Triton) → results/ (use -h for help) |
| `profile_kernels.py` | Profile with rocprofv3 → results/profiling/ |
| `analyze_profiling.py` | Generate performance analyses → results/analysis/ |
| `analysis_notebook.ipynb` | Interactive Jupyter notebook for visualization |
| `moe_utils.py` | Shared utilities |

## Quick Start

```bash
cd aiter/op_tests/moe_benchmarking_profiling

# Run with defaults (uses configs/ and saves to results/)
python benchmark.py -i configs/sample.csv -o results/benchmark
python profile_kernels.py -i results/benchmark_best_kernels.csv
python analyze_profiling.py -i results/profiling/kernels_with_counters.csv

# Or use the interactive notebook
jupyter notebook analysis_notebook.ipynb
```

## Outputs

- `results/benchmark_*.csv` - Benchmark results, combinations, best kernels
- `results/profiling/kernels_with_counters.csv` - Profiling data
- `results/analysis/*.html` - Roofline plots

## Useful Options

### benchmark.py

```bash
# Custom config file (default output: results/benchmark)
python benchmark.py -i configs/example_config.csv

# Include Triton e2e kernels (default: only ASM/CK)
python benchmark.py -i configs/sample.csv -o results/out --include-triton

# Quick single config test
python benchmark.py --config "1,4096,1536,16,8,ActivationType.Silu,torch.bfloat16,torch.bfloat16,torch.bfloat16,QuantType.No,1,0" -o results/test

# Resume interrupted run
python benchmark.py -i configs/sample.csv -o results/out --resume

# Force re-run (ignore existing results)
python benchmark.py -i configs/sample.csv -o results/out --force

# Use specific number of GPUs
python benchmark.py -i configs/sample.csv -o results/out --gpus 4

# Adjust error threshold (default 50%)
python benchmark.py -i configs/sample.csv -o results/out --error-threshold 10.0
```

### Other scripts

```bash
# Use MI350 specs
python analyze_profiling.py --gpu MI350 -i results/profiling/kernels_with_counters.csv

# Keep temp files for debugging
python profile_kernels.py --keep-temp-files -i results/benchmark_best_kernels.csv
```

## Dependencies

Profiling requires `rocprofv3` on system.

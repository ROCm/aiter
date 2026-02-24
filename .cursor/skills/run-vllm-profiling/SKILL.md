---
name: run-vllm-profiling
description: Run vLLM serving benchmarks to profile model performance. Use when the user wants to benchmark, profile, or measure latency/throughput of an LLM served via vLLM, or asks about TTFT, TPOT, E2EL, token throughput, or serving performance.
---

# Run vLLM Serving Profiler

## Overview

The profiling script is bundled with this skill at [run_profiling.sh](run_profiling.sh). It is self-contained: starts a vLLM server, runs benchmarks across a matrix of (input_tokens, output_tokens, max_concurrency) configs, and outputs a human-readable table + CSV.

## Setup

Copy the script to the project's scripts directory (or wherever appropriate):
```bash
cp /path/to/this/skill/run_profiling.sh /app/scripts/run_profiling.sh
chmod +x /app/scripts/run_profiling.sh
```

## Quick Start

```bash
# Minimal: just specify the model
./run_profiling.sh --model /data/MyModel

# Full example with ROCm env vars
export VLLM_ROCM_USE_AITER=1
./run_profiling.sh \
    --model /data/Kimi-K2-Thinking-MXFP4 \
    --tp 8 \
    --max-model-len 16384 \
    --combos "1024:1024 1024:8192 8192:1024" \
    --concurrencies "4 8 16 32 64" \
    --server-extra-args "--enable-auto-tool-choice --tool-call-parser kimi_k2 --reasoning-parser kimi_k2" \
    --output-dir /app/scripts
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model PATH` | **required** | Model path or HuggingFace name |
| `--tp N` | auto-detect GPUs | Tensor parallel size |
| `--max-model-len N` | 16384 | Must be >= max(input + output) across combos |
| `--max-num-seqs N` | 1024 | Max concurrent sequences on server |
| `--gpu-mem-util F` | 0.95 | GPU memory utilization fraction |
| `--host H` | localhost | Server bind host |
| `--port P` | 8000 | Server bind port |
| `--server-extra-args "..."` | "" | Verbatim extra args to `vllm serve` |
| `--combos "I:O ..."` | "1024:1024 1024:8192 8192:1024" | Space-separated input:output pairs |
| `--concurrencies "N ..."` | "4 8 16 32 64" | Space-separated concurrency values |
| `--prompts-multiplier N` | 8 | num_prompts = N * concurrency |
| `--no-warmup` | off | Skip JIT warmup run (not recommended) |
| `--output-dir DIR` | script dir | Where to save the CSV |

## Running and Monitoring

1. **Pre-flight**: Export any env vars (e.g. `VLLM_ROCM_USE_AITER=1`) before running.
2. **Run in background** for long sweeps:
   ```bash
   ulimit -c 0
   ./run_profiling.sh --model /data/M > profiling_out.txt 2>&1 &
   ```
3. **Monitor progress**: Config headers print as each config starts.
   ```bash
   tail -f profiling_out.txt          # live progress
   grep "^===" profiling_out.txt      # completed configs
   ```
4. **Check interim results**: Results accumulate in a temp CSV file:
   ```bash
   for f in /tmp/tmp.*; do head -1 "$f" 2>/dev/null | grep -q "input_tokens" && cat "$f"; done
   ```
5. **Segfault handling**: The script exits immediately on segfault (exit codes 132/134/139 or log pattern match). On server startup failure, it skips the config and continues.

## Output

- Human-readable table printed to stdout on completion.
- CSV saved to `<output-dir>/profile_results_YYYYMMDD_HHMMSS.csv`.
- Metrics collected: **TTFT** (ms), **TPOT** (ms), **E2EL** (ms), **Output token throughput** (tok/s), **Total token throughput** (tok/s).
- Prefers Median values; falls back to Mean if Median unavailable.

## Time Estimates

Per config: server startup (3-5 min) + warmup run + measurement run.

| Output tokens | ~Time per config (conc=4) | ~Time per config (conc=64) |
|---------------|--------------------------|---------------------------|
| 1024          | 8-12 min                 | 15-25 min                 |
| 8192          | 35-70 min                | 60-120 min                |

A full 3x5 sweep with 8192-output configs can take 6-10 hours.

## Safety Features

- `ulimit -c 0` prevents expensive GPU core dumps.
- Caches cleared between server restarts (triton, vllm, comgr, torchinductor).
- Warmup run absorbs JIT compilation overhead so measurement is clean.
- Server killed and restarted per config for isolation.
- Segfault detection on both server and client.

---
name: vllm-profiling-development
description: Modify, extend, or recreate the vLLM profiling script from scratch. Use when the user wants to add metrics, change benchmark behavior, add new sweep dimensions, fix issues with the profiler, or build a new profiling script for vLLM serving benchmarks.
---

# vLLM Profiling Script - Development Guide

Reference script lives in the sibling skill directory at [run_profiling.sh](../run-vllm-profiling/run_profiling.sh). Read it for the full implementation. This skill covers the architecture, design decisions, and how to extend or recreate it.

## Architecture

The script has five sections. When recreating, implement them in this order:

### 1. Argument Parsing & Defaults

Flags parsed via `while/case` loop. Required: `--model`. Everything else has defaults.

Key design choice: `--server-extra-args` is a single string passed verbatim via shell word splitting (requires `# shellcheck disable=SC2086`). This avoids needing to enumerate every possible vLLM flag.

### 2. Helper Functions

| Function | Purpose |
|----------|---------|
| `clear_caches` | Remove triton/vllm/comgr/torchinductor caches between runs |
| `check_segfault` | Check exit code (132/134/139) and grep log for fault patterns |
| `parse_client_output` | Grep `vllm bench serve` output for Median (preferred) or Mean metrics |
| `start_server` | Launch `vllm serve` in background, capture PID |
| `wait_for_server` | Poll for "Application startup complete." or `/v1/models` responding (30 min timeout) |
| `kill_server` | SIGTERM + wait + segfault check |
| `run_client` | Run `vllm bench serve` with given input/output/concurrency |

### 3. Temp Files & EXIT Trap

`dump_results` is registered as an EXIT trap. It:
- Prints a human-readable aligned table via `printf`
- Copies the results CSV to `<output-dir>/profile_results_<timestamp>.csv`
- Cleans up all temp files

This ensures results are always dumped, even on error or SIGTERM.

### 4. Main Loop

Nested loop: outer over `INPUT_OUTPUT_COMBOS`, inner over `CONCURRENCIES`. Per iteration:
1. Clear caches
2. Start server + wait for ready
3. Run warmup client (absorbs JIT, discarded)
4. Run measurement client (results parsed)
5. Kill server
6. Append row to results CSV

On server failure: record FAIL row, `continue` to next config.

### 5. Output Format

CSV header: `input_tokens,output_tokens,max_concurrency,ttft_ms,tpot_ms,e2el_ms,output_throughput_tok_s,total_throughput_tok_s`

## vLLM Bench Output Parsing

`vllm bench serve` prints a block like:
```
============ Serving Benchmark Result ============
Output token throughput (tok/s):         1234.56
Total Token throughput (tok/s):          2345.67
---------------Time to First Token----------------
Mean TTFT (ms):          123.45
Median TTFT (ms):        120.00
-----Time per Output Token (Excl. 1st Token)------
Mean TPOT (ms):          12.34
Median TPOT (ms):        11.50
------------------End to End Latency--------------
Mean E2EL (ms):          45678.90
Median E2EL (ms):        44000.00
```

Parse with: `grep -E "Median TTFT|Mean TTFT" | head -1 | awk '{print $NF}'`. Prefer Median, fall back to Mean (head -1 picks whichever appears first; Median appears after Mean in output, so grep both and take first match by putting Median pattern first).

## Common Extensions

### Adding a new metric (e.g. ITL)

1. Add variable to `parse_client_output`:
   ```bash
   ITL_MS=$(grep -E "Median ITL \(ms\):|Mean ITL \(ms\):" "$out" 2>/dev/null | head -1 | awk '{print $NF}' | tr -d '\r')
   ```
2. Add column to CSV header in the `echo ... > "$RESULTS_FILE"` line.
3. Add `${ITL_MS:-}` to the result row `echo` in the main loop.
4. Add column to `printf` format strings in `dump_results`.

### Adding a new sweep dimension (e.g. batch size)

Add an outer loop level. Wrap the combo/concurrency loops in a new `for` loop. Add the dimension as a column in CSV header and result rows.

### Adding dataset support (non-random)

The current script uses `--dataset-name random`. To support real datasets:
1. Add `--dataset` flag pointing to a ShareGPT/other dataset file.
2. Replace `--dataset-name random --random-input-len ... --random-output-len ...` with `--dataset-name sharegpt --dataset-path $DATASET`.
3. Token length combos become irrelevant; sweep only concurrency.

### Adding server env var presets

For model-specific env vars (e.g. ROCm flags), add a `--env-preset` flag:
```bash
case "$ENV_PRESET" in
    rocm-mla)
        export VLLM_ROCM_USE_AITER=1
        export VLLM_ROCM_USE_AITER_MLA=1
        ;;
    rocm-no-mla)
        export VLLM_ROCM_USE_AITER=1
        export VLLM_ROCM_USE_AITER_MLA=0
        ;;
esac
```

### Resuming a partial run

To add resume support:
1. Accept `--resume-csv` flag pointing to a partial CSV.
2. Load completed (input, output, concurrency) tuples from it.
3. In the main loop, `continue` if the tuple already exists in the loaded set.
4. Append new results to the same file.

## Recreating From Scratch

If the script is lost or needs a full rewrite, follow this skeleton:

```bash
#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0

# 1. Parse args: --model (required), --tp, --combos, --concurrencies, etc.
# 2. Define: clear_caches, check_segfault, parse_client_output,
#            start_server, wait_for_server, kill_server, run_client
# 3. Create temp files (mktemp). Register EXIT trap to dump results + cleanup.
# 4. Write CSV header to results file.
# 5. Nested loop: for combo in COMBOS; for conc in CONCURRENCIES:
#      clear_caches -> start_server -> wait -> warmup -> measure -> kill -> parse -> record
# 6. EXIT trap dumps human-readable table + saves CSV copy.
```

Critical invariants:
- **Always kill server between configs** for isolation.
- **Always clear caches** before server start for reproducibility.
- **Always run warmup** before measurement (JIT compilation on first run inflates latency).
- **EXIT trap must always fire** to avoid losing partial results.
- **Segfault = immediate exit** (GPU memory corruption makes further results unreliable).
- Server ready detection: both log grep AND HTTP health check (some vLLM versions differ in log format).

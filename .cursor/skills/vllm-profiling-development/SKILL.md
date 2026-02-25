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
| `start_server` | Launch `vllm serve` in background, capture PID. Accepts optional trace dir arg; when provided, adds `--profiler-config` with torch profiler settings and sets `VLLM_RPC_TIMEOUT=1800000` |
| `wait_for_server` | Poll for "Application startup complete." or `/v1/models` responding (30 min timeout) |
| `kill_server` | SIGTERM + wait + segfault check |
| `run_client` | Run `vllm bench serve` with given input/output/concurrency. Accepts optional 5th arg (`1`/`0`) to enable `--profile` flag for trace collection |

### 3. Temp Files & EXIT Trap

`dump_results` is registered as an EXIT trap. It:
- Prints a human-readable aligned table via `printf`
- Copies the results CSV to `<output-dir>/profile_results_<timestamp>.csv`
- Cleans up all temp files

This ensures results are always dumped, even on error or SIGTERM.

### 4. Main Loop

Nested loop: outer over `INPUT_OUTPUT_COMBOS`, inner over `CONCURRENCIES`. Per iteration:
1. Clear caches
2. Compute per-config trace directory (if `--trace` enabled)
3. Start server + wait for ready (with `--profiler-config` if tracing)
4. Run warmup client WITHOUT `--profile` (absorbs JIT, discarded; profiling NOT active)
5. Run measurement client WITH `--profile` if tracing (results parsed; uses `TRACE_PROMPTS_MULTIPLIER=1`)
6. Kill server
7. Append row to results CSV

On server failure: record FAIL row, `continue` to next config.

**Tracing flow detail:** The server starts with `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "...", "torch_profiler_with_stack": false}'` which loads the profiler infrastructure. The warmup `vllm bench serve` run omits `--profile`, so no recording occurs. The measurement `vllm bench serve` run includes `--profile`, which signals the server to start/stop recording around the benchmark. This separation ensures warmup JIT overhead never appears in traces.

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

### Modifying trace collection behavior

The tracing subsystem uses vLLM's built-in torch profiler support:

- **Server**: `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "...", ...}'` — tells vLLM to load profiler infrastructure and where to write trace files. Additional config keys control trace content (stacks, memory, shapes, FLOPs, gzip).
- **Client**: `--profile` flag on `vllm bench serve` — signals the server to start/stop recording around the benchmark run automatically.
- **`VLLM_RPC_TIMEOUT=1800000`**: Set when tracing to allow 30 minutes for trace flushing on large models.

Key design decisions:
1. **Per-config trace directories**: `<trace-dir>/in<I>_out<O>_conc<N>/` ensures trace files from different configs never collide.
2. **Client-gated recording**: Warmup `vllm bench serve` runs without `--profile`, so no recording occurs. Measurement run uses `--profile` to capture only steady-state behavior.
3. **Reduced prompts**: `TRACE_PROMPTS_MULTIPLIER=1` (hardcoded when `--trace` is active) keeps traces small. The normal `PROMPTS_MULTIPLIER` is saved/restored around the measurement run.
4. **Minimal trace content**: `torch_profiler_with_stack: false` keeps traces lean. Can be re-enabled for debugging.
5. **Server restart per config**: Each config gets a fresh server process, providing clean trace isolation.

To customize:
- Change `TRACE_PROMPTS_MULTIPLIER` to adjust trace size vs. representativeness.
- Add `--trace-prompts-multiplier N` flag if you want user control over this.
- Enable stack traces by setting `torch_profiler_with_stack: true` in the `--profiler-config` JSON (increases trace size significantly).
- To collect traces only for specific configs, add filtering logic in the main loop before calling `run_client` with profile=1.

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

# 1. Parse args: --model (required), --tp, --combos, --concurrencies, --trace, --trace-dir, etc.
# 2. Define: clear_caches, check_segfault, parse_client_output,
#            start_server (accepts optional trace_dir), wait_for_server, kill_server,
#            run_client, start_profiling, stop_profiling
# 3. Create temp files (mktemp). Register EXIT trap to dump results + cleanup.
# 4. Write CSV header to results file.
# 5. Nested loop: for combo in COMBOS; for conc in CONCURRENCIES:
#      trace_dir = <trace-dir>/in<I>_out<O>_conc<N> (if --trace)
#      clear_caches -> start_server(trace_dir) -> wait ->
#      warmup via run_client without --profile ->
#      measure via run_client with --profile (if --trace, reduced prompts) ->
#      kill -> parse -> record
# 6. EXIT trap dumps human-readable table + saves CSV copy + lists trace dirs.
```

Critical invariants:
- **Always kill server between configs** for isolation.
- **Always clear caches** before server start for reproducibility.
- **Always run warmup** before measurement (JIT compilation on first run inflates latency).
- **EXIT trap must always fire** to avoid losing partial results.
- **Segfault = immediate exit** (GPU memory corruption makes further results unreliable).
- Server ready detection: both log grep AND HTTP health check (some vLLM versions differ in log format).
- **Tracing: warmup must never be profiled.** Only pass `--profile` to the measurement `vllm bench serve` run, never to warmup. This keeps traces clean and representative of steady-state behavior.
- **Tracing: each config gets its own trace directory.** Never share trace dirs across configs — trace files can collide and traces from different workloads become impossible to distinguish.

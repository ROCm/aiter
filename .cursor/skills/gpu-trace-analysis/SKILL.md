---
name: gpu-trace-analysis
description: Analyze GPU kernel traces from ML serving frameworks (PyTorch Profiler .pt.trace.json.gz files). Identifies prefill/decode/mixed sections, disambiguates kernel invocations by runtime variants, measures GPU idle time. Use when the user has GPU trace files, asks about kernel performance, prefill vs decode analysis, GPU utilization, or ML serving framework profiling.
---

# GPU Kernel Trace Analysis for ML Serving Frameworks

## Overview

Analyzes PyTorch Profiler traces (`.pt.trace.json.gz`) from ML serving frameworks that support prefill-decode batching. The analysis:

1. **Detects the framework** from trace event patterns
2. **Identifies logical sections** (prefill, decode, mixed iterations)
3. **Maps GPU kernels to sections** using CPU-GPU correlation IDs
4. **Disambiguates kernel invocations** — same kernel name called N times per iteration with different shapes/roles are split into distinct "variants" by runtime clustering
5. **Measures GPU idle time** on the global timeline with per-section attribution

## Quick Start: Using the Existing Script

Check for `analyze_trace_v2.py` — either in the workspace or at the path below:

```bash
python3 scripts/analyze_trace_v2.py <trace_file.json.gz>
```

The script at [scripts/analyze_trace_v2.py](scripts/analyze_trace_v2.py) auto-detects the framework and produces a full report. If the script is missing, follow the methodology below to recreate it. If the script exists but doesn't support a new framework, follow "Adding a New Framework" below.

## Methodology (With or Without Script)

### Step 1: Load and Explore the Trace

Traces are Chrome Trace Event format JSON (often gzipped). Load with:

```python
import json, gzip
with gzip.open(path, 'rt') as f:
    data = json.load(f)
events = data['traceEvents']
```

First, understand the trace structure. Examine event categories:

```python
from collections import Counter
cats = Counter(e.get('cat','') for e in events)
```

Key categories to look for — see [trace-format-reference.md](trace-format-reference.md) for full details:

| Category | What it contains |
|----------|-----------------|
| `kernel` | GPU kernel executions (the main analysis target) |
| `gpu_memcpy`, `gpu_memset` | GPU memory operations |
| `python_function` | Python call stack (used by Atom framework) |
| `user_annotation` | CPU-side logical markers (used by vLLM/SGLang) |
| `gpu_user_annotation` | GPU-side logical markers (maps kernels to sections) |
| `cuda_runtime` | CUDA/HIP API calls (`hipGraphLaunch` for graph replay) |
| `cpu_op` | PyTorch operator dispatches |
| `ac2g` | Activity-to-GPU correlation records |

### Step 2: Detect the Framework

Scan events for distinguishing patterns:

| Framework | Detection pattern |
|-----------|-------------------|
| **Atom** (DeepSeek MLA) | `python_function` events containing `model_runner` + `forward` |
| **vLLM/SGLang** (annotation-based) | `user_annotation` matching `execute_context_N(T)_generation_M(T)` |
| **Unknown / new** | Look for any `user_annotation`, `python_function`, or `gpu_user_annotation` events that mark iteration boundaries |

For a new framework: look for repeating patterns in `user_annotation` or `python_function` that delineate forward-pass iterations. The key is finding events whose time ranges encompass one complete model forward pass.

### Step 3: Identify Sections (Prefill / Decode / Mixed)

**Goal**: Find time ranges for each forward-pass iteration and classify as prefill, decode, or mixed.

**Atom framework**: `python_function` events ending in `: forward` from `model_runner.py` define iterations. Sub-functions `_forward_prefill` and `prepare_decode` within each forward determine the type.

**vLLM/SGLang**: `user_annotation` events matching `execute_context_N(tokens)_generation_M(tokens)` directly encode the section type:
- `context > 0, generation == 0` → prefill
- `context == 0, generation > 0` → decode
- Both > 0 → mixed

**New frameworks**: Look for iteration-boundary markers. Common patterns:
- Annotations with "step", "iteration", "forward", "batch" in the name
- Python function events from the model runner
- Repeating sequences of GPU kernels with consistent structure

### Step 4: Map GPU Kernels to Sections

This is the hardest step. GPU execution is asynchronous — kernels dispatched by CPU section N may execute on the GPU during CPU section N+1.

**Method 1 — `External id` (non-graph kernels)**: Each `kernel` event has `args.External id` linking to a `cpu_op` event. Find the CPU op's timestamp, then find which section contains that timestamp.

**Method 2 — `correlation` + `hipGraphLaunch` (graph-replayed kernels)**: Graph kernels show as "Dispatch Task" with `args.correlation` but no useful `External id`. Match `correlation` to the `cuda_runtime` event (`hipGraphLaunch` / `cudaGraphLaunch`) that has the same correlation. Use the *CPU timestamp* of that launch event to find the section.

**Method 3 — `gpu_user_annotation` (vLLM/SGLang)**: GPU-side annotation events have the same names as CPU `user_annotation`. Binary-search kernels into these GPU-side time ranges. Match GPU annotations to CPU sections by name+order.

**Method 4 — Fallback**: Use the kernel's GPU timestamp to find the containing CPU section. Less accurate for async execution but works as last resort.

**Critical pitfall**: NEVER assign graph-replayed kernels to sections based on their GPU timestamp alone. Async execution means graph kernels from decode iteration N often run during decode iteration N+1's CPU time. Always use the CPU-side launch timestamp via correlation IDs.

### Step 5: Disambiguate Kernel Names

GPU kernel names are often long template instantiations. Create a `short_kernel_name()` function that maps full names to human-readable short names. Key patterns:

| Pattern in full name | Short name |
|---------------------|------------|
| `Cijk_` | `tensile_gemm_MT{dims}` (AMD Tensile GEMM) |
| `kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale` | `ck_blockscale_gemm` (CK blockscale) |
| `fmoe_bf16_blockscaleFp8` | `fused_moe_gemm` |
| `_moe_gemm_a8w4` | `moe_gemm_a8w4` |
| `_gemm_a16_w16_kernel` | `triton_gemm_a16w16_M{}_N{}_K{}` |
| `Rmsnorm2dFwd` | `rmsnorm2d_fwd` |
| `fmha_fwd` | `flash_attention_fwd` |
| `paged_attention` | `paged_attention` |
| `ncclDevKernel` | `nccl_collective` |
| `cross_device_reduce_*stage` | `allreduce_Nstage` |

For new models/frameworks: inspect kernel names in the trace, identify template parameters that distinguish shapes (grid dims, tile sizes, M/N/K), and add new patterns.

### Step 6: Detect Invocation Variants

**Problem**: The same kernel name may be called N times per iteration for different roles (e.g., 3 GEMMs per transformer layer: Q-proj, K-proj, O-proj) with distinct runtimes.

**Solution — Ordinal Position + Recursive Runtime Clustering**:

1. Within each iteration, sort kernels of the same name by GPU timestamp. Assign ordinal index (0, 1, 2, ...).
2. Across iterations of the same type, group durations by ordinal index. The i-th call always represents the same "role" (guaranteed by graph replay; approximately true for non-graph kernels).
3. Compute the mean duration for each ordinal position across iterations.
4. Sort ordinal positions by mean duration. Recursively find the largest gap where:
   - Both resulting sub-clusters have >= 3 members
   - Cluster mean ratio >= 1.10 (10% difference)
   - **Statistical significance**: Gap between cluster means > 3 * sqrt(pooled_within_ordinal_variance / n_iterations). This prevents over-splitting noisy kernels (e.g., allreduce with high per-call variance) while correctly splitting clean kernels (e.g., GEMMs with consistent per-call runtimes).

**Example result for DeepSeek V2 decode**: `ck_blockscale_gemm` (189 calls/iter) splits into 3 variants: 64 calls at ~21us, 60 calls at ~8.4us, 61 calls at ~7.2us — corresponding to 3 distinct GEMM roles across 63 transformer layers.

### Step 7: GPU Idle Time Analysis

Compute idle on the **global GPU timeline** (not per-section):

1. Collect ALL GPU events (`kernel`, `gpu_memcpy`, `gpu_memset`), sort by timestamp.
2. Compute gaps between consecutive events.
3. Classify each gap:
   - **Intra-section**: both neighboring kernels belong to the same section
   - **Inter-section**: neighbors belong to different sections
   - **Unassigned**: one or both neighbors not mapped to any section
4. Report: overall utilization, idle by section type, top N largest gaps, idle hotspots by kernel-pair transitions.

**Critical pitfall**: Do NOT compute idle per-section independently. GPU async execution means kernels from different CPU sections interleave on the GPU. Per-section idle will massively over-count because "gaps" within one section's kernels are filled by another section's kernels.

## Adding a New Framework

When the script doesn't recognize a trace:

1. **Explore event categories**: `Counter(e.get('cat','') for e in events)`
2. **Look for section markers**: Search `user_annotation`, `python_function`, `gpu_user_annotation` for repeating patterns that delimit forward-pass iterations
3. **Determine section classification**: Find sub-events or naming patterns that indicate prefill vs decode
4. **Choose kernel mapping strategy**: If `gpu_user_annotation` exists, use Method 3. If `cuda_runtime` has graph launches, use Method 2. Otherwise use Method 1 or 4.
5. **Add to the script**:
   - Add a detection clause in `detect_framework()`
   - Add `find_sections_<name>()` function
   - Add `map_kernels_<name>()` function
   - Update `short_kernel_name()` with new kernel patterns
   - Update the `analyze()` function to call the new functions

## Common Pitfalls

- **Graph-replayed kernels**: Show as "Dispatch Task" in `args.kind`. They lack `grid`/`block` args and sometimes lack `External id`. Use `correlation` → `hipGraphLaunch`/`cudaGraphLaunch` mapping.
- **Warmup iterations**: The first decode iteration after profiler start may have abnormally high CPU time (JIT compilation, caching). Detect by comparing kernel counts.
- **Multi-GPU traces**: Filter by device (`args.device`) or process (`pid`/`tid`). Each GPU has its own timeline.
- **Overlapping GPU events**: Some traces show slight timestamp overlaps between consecutive kernels. Use a small epsilon (0.01us) when computing gaps.
- **Annotation naming varies**: vLLM uses `execute_context_N(T)_generation_M(T)`. Other frameworks may use different patterns. Always inspect actual event names first.

## Output Format

The analysis should produce tables with these sections:

1. **Section Summary**: CPU/GPU duration per section type, per-iteration detail
2. **Kernel Breakdown** (per section type): Each kernel variant as a row with `Var` (label), `#/It` (calls per iteration), `Cnt` (total count), `Total`, `Avg`, `Min`, `Max`, `%GPU`
3. **Aggregated by kernel type**: All variants merged, showing total time and percentage
4. **GEMM Disambiguation**: Focused view of GEMM-family kernels with variant breakdown
5. **Prefill vs Decode Comparison**: Side-by-side average runtimes with ratio
6. **GPU Idle Analysis**: Utilization, idle breakdown by section type, top gaps, idle hotspots by kernel-pair

## Additional Resources

- For detailed Chrome Trace Event format and field descriptions: [trace-format-reference.md](trace-format-reference.md)
- Analysis script: [scripts/analyze_trace_v2.py](scripts/analyze_trace_v2.py)

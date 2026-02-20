# Chrome Trace Event Format Reference for PyTorch Profiler Traces

## File Format

PyTorch Profiler outputs Chrome Trace Event format as `.pt.trace.json.gz` (gzipped JSON).

```python
import json, gzip
with gzip.open(path, 'rt') as f:
    data = json.load(f)
events = data['traceEvents']
```

Each event is a dict. The key fields:

| Field | Type | Description |
|-------|------|-------------|
| `cat` | string | Event category (see below) |
| `name` | string | Event name (kernel name, function name, etc.) |
| `ph` | string | Phase: `X` (complete), `B`/`E` (begin/end), `i` (instant) |
| `ts` | float | Timestamp in microseconds (us) |
| `dur` | float | Duration in microseconds (for `ph=X` events) |
| `pid` | int | Process ID (often represents device or CPU thread group) |
| `tid` | int | Thread ID (often represents stream for GPU events) |
| `args` | dict | Additional metadata (varies by category) |

## Event Categories

### `kernel` — GPU Kernel Executions

The primary analysis target. Each event represents one GPU kernel launch.

```json
{
  "cat": "kernel",
  "name": "void at::native::...<kernel_template_params>",
  "ph": "X",
  "ts": 1234567.89,
  "dur": 42.5,
  "pid": 0,
  "tid": 7,
  "args": {
    "External id": 12345,
    "correlation": 67890,
    "device": 0,
    "stream": 7,
    "grid": [128, 1, 1],
    "block": [256, 1, 1],
    "kind": "Dispatch Task"
  }
}
```

Key `args` fields:
- **`External id`**: Links to the CPU-side `cpu_op` that launched this kernel. Primary method for mapping non-graph kernels to CPU sections.
- **`correlation`**: Links to the `cuda_runtime` event that dispatched this kernel. Critical for graph-replayed kernels where `External id` is missing or unhelpful.
- **`device`**: GPU device index.
- **`stream`**: GPU stream index.
- **`grid`** / **`block`**: Launch dimensions. Empty `()` for graph-replayed kernels.
- **`kind`**: If `"Dispatch Task"`, this kernel was replayed from a CUDA graph.

### `gpu_memcpy` / `gpu_memset` — GPU Memory Operations

Same structure as `kernel`. Include in GPU timeline analysis (idle time, utilization).

### `python_function` — Python Call Stack

Represents Python function calls. Used by the **Atom framework** to identify forward-pass iterations.

```json
{
  "cat": "python_function",
  "name": "model_runner.py(814): forward",
  "ph": "X",
  "ts": 100000.0,
  "dur": 17000.0
}
```

Atom-specific patterns:
- `model_runner.py(...): forward` — Top-level forward pass (one per iteration)
- `_forward_prefill` in name — Indicates prefill sub-step
- `prepare_decode` in name — Indicates decode sub-step
- `prepare_sampled_ids` — Post-processing (often dominates first decode iteration)

### `user_annotation` — CPU-Side Logical Markers

Marks high-level phases on the CPU timeline. Used by **vLLM/SGLang** frameworks.

```json
{
  "cat": "user_annotation",
  "name": "execute_context_0(0)_generation_4(4)",
  "ph": "X",
  "ts": 200000.0,
  "dur": 620.0
}
```

The vLLM naming pattern: `execute_context_N(T)_generation_M(T)` where:
- `N` = number of context (prefill) requests, `T` = total context tokens
- `M` = number of generation (decode) requests, `T` = total generation tokens
- Only context > 0 → prefill; only generation > 0 → decode; both > 0 → mixed

### `gpu_user_annotation` — GPU-Side Logical Markers

Same names as `user_annotation` but on the GPU timeline. These directly contain the GPU kernels. Very useful for mapping kernels to sections — binary-search kernel timestamps into these ranges.

### `cuda_runtime` — CUDA/HIP Runtime API Calls

Represents calls to the CUDA or HIP runtime from the CPU.

```json
{
  "cat": "cuda_runtime",
  "name": "hipGraphLaunch",
  "ph": "X",
  "ts": 150000.0,
  "dur": 5.2,
  "args": {
    "correlation": 67890
  }
}
```

**Critical for graph replay**: `hipGraphLaunch` (AMD) / `cudaGraphLaunch` (NVIDIA) events have a `correlation` ID that matches the `correlation` field of the GPU `kernel` events they dispatched. The CPU timestamp of the launch event tells you which CPU section triggered the graph replay.

### `cpu_op` — PyTorch Operator Dispatches

CPU-side operator calls. Link to GPU kernels via `External id`.

```json
{
  "cat": "cpu_op",
  "name": "aten::mm",
  "args": { "External id": 12345 }
}
```

### `ac2g` — Activity-to-GPU Correlation

Linking records that map CPU `External id` to GPU `correlation`. Usually not needed if you use the methods above directly.

## Kernel-to-Section Mapping Methods

### Method 1: External id (non-graph kernels)

```
kernel.args["External id"] → cpu_op with matching External id → cpu_op.ts → find section
```

### Method 2: correlation + GraphLaunch (graph-replayed kernels)

```
kernel.args["correlation"] → cuda_runtime (hipGraphLaunch) with matching correlation
  → cuda_runtime.ts (CPU timestamp) → find section
```

### Method 3: gpu_user_annotation (vLLM/SGLang)

```
kernel.ts → binary search into gpu_user_annotation time ranges
  → match gpu_user_annotation name to cpu user_annotation by name+order → section
```

### Method 4: Fallback (GPU timestamp)

```
kernel.ts → binary search into section CPU time ranges
```

Less accurate due to async GPU execution but works as last resort.

## Kernel Name Patterns by Framework

### AMD ROCm Kernels (Common in MI300 traces)

| Full name pattern | Meaning |
|-------------------|---------|
| `Cijk_Ailk_Bljk_*_MT{N}x{M}x{K}` | Tensile GEMM with specific tile sizes |
| `kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale<...Sequence<1, 32, 1, 8>...>` | CK blockscale GEMM, large tile |
| `kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale<...Sequence<1, 16, 1, 16>...>` | CK blockscale GEMM, small tile |
| `fmoe_bf16_blockscaleFp8_*` | Fused MoE GEMM |
| `cross_device_reduce_1stage` / `_2stage` | Allreduce (1-stage or 2-stage) |
| `__amd_rocclr_copyBuffer` | ROCm buffer copy |

### Triton Kernels

| Full name pattern | Meaning |
|-------------------|---------|
| `*_gemm_a16_w16_kernel_*_SIZE_M_N_N_K_K_*` | Triton GEMM (extract M, N, K from name) |
| `*_moe_gemm_a8w4*` | MoE GEMM with A8W4 quantization |
| `triton_poi_fused_*` | Triton fused pointwise operations |
| `*_fwd_kernel*` | Triton attention forward |
| `*_combined_routing_fused*` | MoE routing kernel (fused variant) |

### Common PyTorch / CUDA Kernels

| Full name pattern | Meaning |
|-------------------|---------|
| `Rmsnorm2dFwd` | RMSNorm forward |
| `fmha_fwd*` | Flash attention forward |
| `paged_attention*` | Paged attention (vLLM) |
| `reshape_and_cache_kernel` | KV cache update |
| `SoftMaxForward` | Softmax |
| `ncclDevKernel*` | NCCL collective communication |
| `CatArrayBatchedCopy` | Tensor concatenation |

### MoE-Specific Kernels

| Full name pattern | Meaning |
|-------------------|---------|
| `MoeSortingKernel` | MoE token sorting |
| `MoeSortingMultiPhaseKernel_P{0,1,23}` | Multi-phase MoE sorting |
| `MoeSortingClearWorkspace` | MoE sort workspace init |
| `grouped_topk_opt_sort` | MoE top-k sorting |
| `*_topk*` | Top-k selection |
| `*_reduce_grouped*` | Grouped reduction |

### DeepSeek MLA-Specific Kernels

| Full name pattern | Meaning |
|-------------------|---------|
| `*mla_a16w16*` | MLA decode attention |
| `kn_mla_reduce` | MLA reduce |
| `concat_and_cache_mla` | MLA KV cache update |
| `fused_qk_rope_cat_and_cache_mla` | Fused QK RoPE + KV cache |
| `kn_entry_2c_sbhd_cached_indirect` | Cached attention indirect |
| `kn_get_mla_metadata` | MLA metadata computation |
| `batched_gemm_a8w8_*_SIZE_M_N_N_K_K_*` | Batched A8W8 GEMM (extract M, N, K) |

## CUDA Graph Replay Detection

Graph-replayed kernels have these characteristics:
- `args.kind` == `"Dispatch Task"`
- `args.grid` and `args.block` may be empty `[]`
- `args.External id` is often missing or points to the graph-capture event (not the replay)
- `args.correlation` links to the `hipGraphLaunch` that replayed the graph

Always check for `hipGraphLaunch` / `cudaGraphLaunch` in `cuda_runtime` events when graph replay is present. The CPU timestamp of the launch event is what determines which section the kernels belong to.

# Triton GEMM Kernel Reference

## XCD Remapping Logic (AMD-specific)

All GEMM kernels use XCD (Execution Compute Die) remapping for multi-chiplet AMD GPUs:

```python
pid = tl.program_id(axis=0)
pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
tall_xcds = GRID_MN % NUM_XCDS
tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
xcd = pid % NUM_XCDS
local_pid = pid // NUM_XCDS
if xcd < tall_xcds:
    pid = xcd * pids_per_xcd + local_pid
else:
    pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
```

Use `get_num_xcds()` from `aiter.ops.triton.utils.device_info`.

## Grouped Ordering for L2 Cache Reuse

```python
if GROUP_SIZE_M == 1:
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
else:
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

## Config JSON Format

File: `aiter/ops/triton/configs/gemm/{arch}-GEMM-{NAME}.json`

```json
{
  "M_LEQ_128": {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16
  },
  "M_GEQ_256": {
    "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2
  },
  "any": {
    "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2
  }
}
```

Selection priority: M_LEQ/M_GEQ first, then "any" as fallback. Shape-specific files `-N={N}-K={K}.json` override general configs.

## Existing GEMM Variants

| Variant | Kernel File | Key Differences |
|---------|------------|-----------------|
| A8W8 (INT8) | `basic/gemm_a8w8.py` | Per-tensor row-wise scale |
| A8W8 Blockscale | `basic/gemm_a8w8_blockscale.py` | Per-block 2D scale factors |
| A16W16 (BF16) | `basic/gemm_a16w16.py` | No quantization, direct BF16 matmul |
| AFP4WFP4 | `basic/gemm_afp4wfp4.py` | 4-bit weight + activation |
| Batched BF16 | `batched/batched_gemm_bf16.py` | Batch dimension, 3D tensors |
| Batched A8W8 | `batched/batched_gemm_a8w8.py` | Batched + quantized |
| Feed-forward | `feed_forward/ff_a16w16.py` | Two GEMMs + activation (SiLU gate) |
| Fused split-cat | `fused/fused_gemm_*_split_cat.py` | GEMM + tensor split + concat |

## Blockscale Pattern

Block-scale GEMM uses 2D scale factors instead of 1D:

```python
# Scale shapes: a_scale (M // BLOCK_M, K // BLOCK_K), b_scale (N // BLOCK_N, K // BLOCK_K)
# Apply per-block scaling inside the K-loop
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a_scale_val = tl.load(a_scale_ptr + ...) 
    b_scale_val = tl.load(b_scale_ptr + ...)
    a = tl.load(a_ptrs, ...).to(tl.float32) * a_scale_val
    b = tl.load(b_ptrs, ...).to(tl.float32) * b_scale_val
    accumulator += tl.dot(a, b)
```

## Batched GEMM Pattern

Batched GEMMs add a batch dimension:

```python
@triton.jit
def _batched_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, B,  # B = batch size
    stride_ab, stride_am, stride_ak,  # batch stride first
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    ...
):
    pid_batch = tl.program_id(axis=1)  # 2D grid: (MN_tiles, batch)
    # Offset pointers by batch
    a_ptr += pid_batch * stride_ab
    b_ptr += pid_batch * stride_bb
    c_ptr += pid_batch * stride_cb
    # Rest follows standard GEMM pattern
```

Grid: `(triton.cdiv(M, BM) * triton.cdiv(N, BN), batch_size)`

## Feed-Forward GEMM Pattern

Two-stage GEMM with gated activation:

```python
# Stage 1: gate = x @ w1_gate, up = x @ w1_up (or single fused w1)
# Activation: out = silu(gate) * up
# Stage 2: result = out @ w2
```

## Benchmark Utilities

### Shape-based benchmark
```python
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_shape_benchmark_object
benchmark = get_shape_benchmark_object(get_caller_name_no_ext(), args)
```

### Model-based benchmark
```python
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_model_benchmark_object
benchmark = get_model_benchmark_object(get_caller_name_no_ext(), args)
```

Model configs in `op_tests/op_benchmarks/triton/utils/model_configs.json` map model names to hidden_dim and intermediate_dim.

### VGPR Reporting
```python
if args.print_vgpr:
    fun = lambda: run_benchmark(args, defaults)
    print_vgpr(fun, get_caller_name_no_ext())
```

Requires `AMDGCN_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 TRITON_PRINT_AUTOTUNING=1`.

## Common Imports

```python
# Kernel file
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

# Wrapper file
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.logger import AiterTritonLogger

# Test file
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

# Benchmark file
from op_tests.op_benchmarks.triton.utils.argparse import get_parser, add_argparse_ff, get_ff_args
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_model_benchmark_object, get_shape_benchmark_object,
    print_vgpr, get_caller_name_no_ext,
)
```

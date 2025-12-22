# Triton-based Communication (Iris)

AITER supports GPU-initiated communication using the [Iris library](https://github.com/ROCm/iris). This enables high-performance Triton-based communication primitives like reduce-scatter and all-gather.

## Installation

**Install with Triton communication support:**
```bash
# Option 1: Install via extras
pip install -e ".[triton_comms]"

# Option 2: Install all optional dependencies
pip install -e ".[all]"
```

## Basic Usage

```python
from aiter import IrisCommContext, reduce_scatter, all_gather
import torch.distributed as dist

# Initialize PyTorch distributed
dist.init_process_group(backend="nccl")

# Use Iris-based communication
with IrisCommContext(heap_size=2**30) as ctx:  # 1GB heap
    input_tensor = ctx.iris_ctx.empty((4096, 4096), dtype=torch.float32)
    output = reduce_scatter(input_tensor, ctx)
    result = all_gather(output, ctx)
```

## Automatic Heap Size Calculation

```python
from aiter import IrisCommContext, calculate_heap_size
import torch

# Automatically calculate required heap size for your operations
M, N = 8192, 7168  # Your tensor dimensions
heap_size = calculate_heap_size(
    M=M,
    N=N,
    dtype=torch.float32,
    world_size=2,  # Number of GPUs
    quant_mode="fp8_per_token",  # "none", "fp8_per_token", or "fp4_per_token"
    all_gather=True,
)

# Guaranteed to have enough memory
with IrisCommContext(heap_size=heap_size) as ctx:
    # Your operations here
    pass
```


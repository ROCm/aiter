---
title: "Communication Operators"
last_verified: 2026-04-06
source_files:
  - aiter/ops/communication.py
  - aiter/ops/custom_all_reduce.py
  - aiter/ops/quick_all_reduce.py
  - docs/triton_comms.md
tags: [communication, allreduce, iris, triton, distributed]
---

# Communication Operators

## Overview
aiter provides distributed communication primitives for multi-GPU inference and training. There are three communication subsystems: custom all-reduce (for small tensor parallelism), quick all-reduce, and Iris-based Triton communication.

## Custom All-Reduce
- Module: `custom_all_reduce.py`
- Optimized for tensor parallelism within a node
- Uses custom signal-based synchronization

## Quick All-Reduce
- Module: `quick_all_reduce.py`
- Lightweight all-reduce implementation

## Iris-based Communication (Optional)
Iris is AMD's GPU-initiated communication library. When installed, it enables:
- `reduce_scatter(input_tensor, ctx)` -- distributed reduce-scatter
- `all_gather(output, ctx)` -- distributed all-gather
- `reduce_scatter_rmsnorm_quant_all_gather(...)` -- fused reduce-scatter + RMSNorm + quant + all-gather
- `IrisCommContext(heap_size)` -- context manager for Iris communication
- `calculate_heap_size(M, N, dtype, world_size, quant_mode, all_gather)` -- compute required heap

### Installation
```bash
pip install -e ".[triton_comms]"
# or
pip install -r requirements-triton-comms.txt
```

### Usage
```python
from aiter import IrisCommContext, reduce_scatter, all_gather
with IrisCommContext(heap_size=2**30) as ctx:
    output = reduce_scatter(input_tensor, ctx)
    result = all_gather(output, ctx)
```

## Distributed Environment Setup
`communication.py` provides `init_dist_env()` for initializing PyTorch distributed with tensor parallelism, using NCCL backend and custom all-reduce.

## Related Pages
- [[operators/norm]] -- fused norm+communication (reduce_scatter_rmsnorm_quant_all_gather)
- [[concepts/backend-selection]] -- Iris availability affects communication path

## Source Files
- `aiter/ops/communication.py` -- distributed env setup and utilities
- `aiter/ops/custom_all_reduce.py` -- custom all-reduce implementation
- `aiter/ops/quick_all_reduce.py` -- quick all-reduce
- `docs/triton_comms.md` -- Iris communication documentation

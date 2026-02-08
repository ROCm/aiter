# AITER Distributed Infrastructure Guide

This guide documents AITER's multi-GPU distributed computing infrastructure, including tensor parallelism, custom all-reduce implementations, and communication primitives.

---

## Quick Reference

| Use Case | Recommended API | Backend |
|----------|----------------|---------|
| **Initialize distributed env** | `aiter.init_dist_env(tp_size, rank)` | NCCL/RCCL + custom |
| **Tensor-parallel all-reduce** | `tensor_model_parallel_all_reduce(input)` | Auto-selected |
| **Fused all-reduce + RMSNorm** | `tensor_model_parallel_fused_allreduce_rmsnorm(...)` | Custom IPC |
| **ASM-level all-reduce** | `aiter.all_reduce_asm(input)` | HIP ASM |
| **GPU-initiated comms (Triton)** | `IrisCommContext` + `reduce_scatter`/`all_gather` | [Iris](https://github.com/ROCm/iris) |
| **Tear down** | `aiter.destroy_dist_env()` | — |

---

## 1. Architecture Overview

```
Layer 4: User API
  aiter/ops/communication.py      — init_dist_env(), all_reduce_asm()

Layer 3: Parallel State (Group Management)
  aiter/dist/parallel_state.py    — GroupCoordinator, process group lifecycle

Layer 2: Device Communicator
  aiter/dist/device_communicators/communicator_cuda.py — dispatch point

Layer 1: Backend Implementations
  custom_all_reduce.py   — IPC-based (small-medium tensors, intra-node)
  quick_all_reduce.py    — Quantized (large tensors, MI300 series)
  communicator_pynccl.py — NCCL/RCCL wrapper (general purpose)

Layer 0: Native Ops (JIT-compiled HIP kernels)
  aiter/ops/custom_all_reduce.py, aiter/ops/quick_all_reduce.py

Separate Path: Iris/Triton Communications
  aiter/ops/triton/comms/         — GPU-initiated reduce-scatter/all-gather
```

---

## 2. Initialization

```python
import aiter

aiter.init_dist_env(
    tensor_model_parallel_size=8,    # TP degree
    rankID=0,                        # Global rank
    backend="cpu:gloo,cuda:nccl",    # Communication backends
    distributed_init_method="env://",
    local_rank=-1,                   # Auto-detect from LOCAL_RANK env
    data_parallel_size=1,
    data_parallel_rank=0,
)
```

### What `init_dist_env()` Does

1. Enables custom all-reduce globally.
2. Calls `torch.distributed.init_process_group()`.
3. Creates the world `GroupCoordinator`.
4. Creates TP, PP, DP, and EP process groups with device communicators.
5. Registers IPC signal buffers for custom all-reduce.

### Teardown

```python
aiter.destroy_dist_env()
```

---

## 3. Parallelism Dimensions

| Dimension | Global | Description |
|-----------|--------|-------------|
| **TP** (Tensor Parallelism) | `_TP` | Adjacent ranks share the same node; used for attention/GEMM splits |
| **PP** (Pipeline Parallelism) | `_PP` | Ranks across pipeline stages |
| **DP** (Data Parallelism) | `_DP` | Ranks processing different data shards |
| **EP** (Expert Parallelism) | `_EP` | Combined DP × TP; used for MOE all-to-all |

Layout order: `ExternalDP × DP × PP × TP` — adjacent GPU ranks are in the same TP group for optimal XGMI/NVLink bandwidth.

### Accessors

```python
from aiter.dist import parallel_state as ps

tp_group = ps.get_tp_group()              # GroupCoordinator
tp_size  = ps.get_tensor_model_parallel_world_size()
tp_rank  = ps.get_tensor_model_parallel_rank()
```

---

## 4. Communication Primitives

### All-Reduce

```python
from aiter.dist.communication_op import tensor_model_parallel_all_reduce

output = tensor_model_parallel_all_reduce(input_tensor)
```

**Dispatch priority** (automatic selection):

| Priority | Backend | When Used |
|----------|---------|-----------|
| 1 | **Quick All-Reduce** | Large tensors on MI300 series, quantized comm enabled |
| 2 | **Custom All-Reduce** | Small-medium tensors, IPC-based, intra-node |
| 3 | **PyNccl** | General purpose NCCL/RCCL |
| 4 | **PyTorch distributed** | Fallback |

### Fused All-Reduce + RMSNorm

```python
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm

output = tensor_model_parallel_fused_allreduce_rmsnorm(
    input, residual, weight, eps
)
```

Fuses all-reduce with RMSNorm in a single kernel for small tensors.

### Reduce-Scatter / All-Gather

```python
from aiter.dist.communication_op import (
    tensor_model_parallel_reduce_scatter,
    tensor_model_parallel_all_gather,
)

output = tensor_model_parallel_reduce_scatter(input, use_custom=True)
output = tensor_model_parallel_all_gather(input, use_custom=False)
```

### ASM-Level All-Reduce (Direct Kernel)

```python
aiter.all_reduce_asm(input_tensor)
```

Bypasses the GroupCoordinator stack entirely — maximum performance for CUDA graph scenarios.

---

## 5. Custom All-Reduce (IPC-Based)

**File:** `aiter/dist/device_communicators/custom_all_reduce.py`

Uses GPU IPC shared memory for direct inter-rank communication without NCCL.

### Requirements
- Single-node only
- World sizes: 2, 4, 6, 8
- Input size: multiple of 16 bytes, max ~128MB

### CUDA Graph Support

```python
with ca_comm.capture() as graph_capture:
    # Operations are captured into a CUDA graph
    output = all_reduce(input)
# After capture, register_graph_buffers() is called automatically
```

---

## 6. Quick All-Reduce (Quantized)

**File:** `aiter/dist/device_communicators/quick_all_reduce.py`

Designed for large tensors on AMD MI300 series. Supports quantized communication for bandwidth reduction.

### Quantization Modes

| Mode | Env Variable Value | Description |
|------|-------------------|-------------|
| `FP` | `FP` | Full precision (float16/bfloat16) |
| `FP8` | `FP8` | 8-bit floating point |
| `INT6` | `INT6` | 6-bit integer (Q6) |
| `INT4` | `INT4` | 4-bit integer (Q4) |
| `NONE` | `NONE` (default) | Disabled |

Enable with: `AITER_QUICK_REDUCE_QUANTIZATION=FP8`

### Requirements
- World sizes: 2, 4, 8
- Input dtype: float16 or bfloat16
- ROCm MI300/MI350 series only (`gfx94*`, `gfx95*`)

---

## 7. Iris/Triton Communications (Separate Path)

The Iris path is a **separate** communication system that enables GPU-initiated collectives from within Triton kernels. It does NOT use the GroupCoordinator or any IPC/NCCL mechanisms.

```python
from aiter import IrisCommContext, reduce_scatter, all_gather

with IrisCommContext(heap_size=2**30) as ctx:
    input_shard = ctx.iris_ctx.empty((4096, 4096), dtype=torch.float32)
    output = reduce_scatter(input_shard, ctx)
    result = all_gather(output, ctx)
```

### Fused Operations

```python
from aiter.ops.triton.comms.fused import reduce_scatter_rmsnorm_quant_all_gather

# RS + RMSNorm + Quant + AG in one Triton kernel
output = reduce_scatter_rmsnorm_quant_all_gather(...)
```

See the [Triton Comms Guide](triton_comms.md) for details.

---

## 8. All-to-All (Expert Parallelism)

The EP communicator supports multiple all-to-all backends for MOE dispatch:

| Backend | Env/Config | Description |
|---------|-----------|-------------|
| `mori` (default) | — | MoRI library for AMD GPUs |
| `naive` | — | Simple all-to-all |
| `allgather_reducescatter` | — | AG+RS-based all-to-all |
| `deepep_high_throughput` | — | DeepEP high-throughput mode |
| `deepep_low_latency` | — | DeepEP low-latency mode |

---

## 9. Shared Memory Broadcast

**File:** `aiter/dist/shm_broadcast.py`

A lock-free shared memory ring buffer (`ShmRingBuffer`) + ZMQ pub-sub (`MessageQueue`) for efficient metadata broadcast in the TP group.

- **Local readers** (same node): Shared memory for small data, ZMQ IPC for overflow.
- **Remote readers** (different nodes): ZMQ TCP pub-sub.

---

## 10. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AITER_QUICK_REDUCE_QUANTIZATION` | `NONE` | Quick all-reduce quantization mode |
| `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16` | `1` | Cast bf16→fp16 for faster kernels |
| `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | `0` (2GB) | Max tensor size for quick all-reduce |
| `VLLM_NCCL_SO_PATH` | (auto) | Path to NCCL/RCCL shared library |
| `HIP_VISIBLE_DEVICES` | (auto) | GPU device visibility |
| `LOCAL_RANK` | (auto) | Local rank assignment |

---

## 11. Source Files

| Component | Path |
|---|---|
| User API | `aiter/ops/communication.py` |
| Parallel state | `aiter/dist/parallel_state.py` |
| Communication ops | `aiter/dist/communication_op.py` |
| CUDA communicator | `aiter/dist/device_communicators/communicator_cuda.py` |
| Custom all-reduce | `aiter/dist/device_communicators/custom_all_reduce.py` |
| Quick all-reduce | `aiter/dist/device_communicators/quick_all_reduce.py` |
| PyNccl wrapper | `aiter/dist/device_communicators/communicator_pynccl.py` |
| All-to-all backends | `aiter/dist/device_communicators/all2all.py` |
| Shared memory broadcast | `aiter/dist/shm_broadcast.py` |
| Custom AR native ops | `aiter/ops/custom_all_reduce.py` |
| Quick AR native ops | `aiter/ops/quick_all_reduce.py` |
| Iris/Triton comms | `aiter/ops/triton/comms/` |

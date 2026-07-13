# IFOE cross-node custom all-reduce (gfx1250)

Runs a custom all-reduce **across nodes** over the UALink-over-Ethernet (IFOE)
fabric by sharing peer buffers with HIP **fabric handles** instead of IPC
handles. Packaged as the `module_custom_all_reduce_ifoe` JIT module.

## Layout

| file | role |
|---|---|
| `csrc/include/custom_all_reduce_ifoe.cuh` | device kernels (opt fp32 + bf16 + fp8), sync, structs |
| `csrc/include/custom_all_reduce_ifoe.h` | host API declarations |
| `csrc/kernels/custom_all_reduce_ifoe.cu` | fabric alloc/import, context, launch dispatch |
| `csrc/pybind/custom_all_reduce_ifoe_pybind.cu` | pybind entry |
| `aiter/ops/custom_all_reduce_ifoe.py` | `@compile_ops` stubs |
| `aiter/dist/device_communicators/ifoe_custom_all_reduce.py` | `IfoeCustomAllreduce` communicator |
| `op_tests/multigpu_tests/ifoe_custom_ar/test_ifoe_allreduce.py` | this driver |

## Why this works with no kernel change

On gfx1250 the fabric (IFOE) makes a remote GPU's memory addressable like a local
peer's. Two primitives the all-reduce relies on both work cross-node:

- **peer data read/write** — a remote GPU's fabric buffer is read/written like a
  local peer's.
- **peer atomic signals** — the `start_sync`/`end_sync` barrier (system-scope
  atomic store into a peer's flag + spin-load on own flag) completes cross-node.

So the 2-stage reduce-scatter/allgather structure is reused; only the
buffer-sharing mechanism changes:

| | intra-node (IPC) | cross-node (this) |
|---|---|---|
| export | `hipIpcGetMemHandle` | `hipMemExportToShareableHandle` (fabric) |
| import | `hipIpcOpenMemHandle` | `hipMemImportFromShareableHandle` (fabric) |
| buffer alloc | any `hipMalloc` | VMM (`hipMemCreate` + map, fabric-exportable) |
| handle exchange | 1 node | `torch.distributed` all-gather (any node) |

The 64-byte fabric handles are exchanged over `torch.distributed`, so the same
code path runs TP4 (1 node) and TP8 (2 nodes).

> **Requirement:** HIP fabric export (`hipMemExportToShareableHandle` with
> `hipMemHandleTypeFabric`) needs **ROCm ≥ 7.15**. Older runtimes (e.g. the
> ROCm 7.12 / HIP 7.2 bundled in some aiter containers) return `invalid
> argument` on export. Run inside an image whose ROCm matches the host driver.

## Run

```bash
# TP4, single node (4 GPUs):
torchrun --nproc_per_node=4 test_ifoe_allreduce.py --mb 256

# TP8, two nodes x 4 GPUs (run on each node, node_rank 0 then 1):
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=<node0-ip> --master_port=29500 test_ifoe_allreduce.py --mb 256
```

The first run JIT-compiles `module_custom_all_reduce_ifoe`. The driver runs both
the `fp32`, `bf16`, and `fp8` modes and checks correctness.

### API

```python
from aiter.dist.device_communicators.ifoe_custom_all_reduce import IfoeCustomAllreduce
comm = IfoeCustomAllreduce(group, torch.device("cuda", local_rank), max_bytes=1 << 30)
out = comm.all_reduce(x)                  # fp32, lossless
out = comm.all_reduce(x, mode="bf16")     # bf16 on the wire (lossy)
out = comm.all_reduce(x, mode="fp8")      # fp8 e4m3 on the wire (lossier, fastest)
comm.dispose()
```

`all_reduce` copies `inp` into the registered fabric buffer and writes the result
into `out` (a production integration would allocate the model buffer directly as
the fabric buffer to skip the copy-in). `unroll` / `blocks` override the launch
geometry; both default to tuned values.

## Modes

- **`fp32`** (`allreduce2_opt`) — lossless. MLP-unrolled reduce-scatter (U float4
  in flight per thread) + per-peer contiguous allgather.
- **`bf16`** (`allreduce2_bf16`) — casts each rank's fp32 input to bf16 (a
  separate kernel, for grid-wide visibility), does the reduce-scatter/allgather
  in bf16 (half the fabric bytes), and accumulates in fp32. Lossy (bf16 rounds
  the addends before summing — standard for ML gradient all-reduce); the test's
  correctness check passes because the small-integer test values are exact in bf16.
- **`fp8`** (`allreduce2_fp8`) — same idea with fp8 e4m3 (quarter the fabric
  bytes, fp32 accumulate); lossier than bf16 but the highest throughput.
  Requires the tensor byte size to be a multiple of 64.

## Benchmark / reproduce

`--bench` reports `us/iter` and busbw; `--sweep` runs a latency sweep over a
range of element counts (`--elems N1 N2 ...` for custom sizes). Example — the
TP4 latency sweep across all three modes:

```bash
torchrun --nproc_per_node=4 test_ifoe_allreduce.py --sweep --bench --modes fp32 bf16 fp8
```

TP8 sweep (two nodes, run on each with the coordinator on node 0):

```bash
torchrun --nnodes=2 --node_rank=<0|1> --nproc_per_node=4 \
    --master_addr=<node0-ip> --master_port=29500 \
    test_ifoe_allreduce.py --sweep --bench
```

Each line prints `us/iter` per (size, mode); compare against your reference
all-reduce (e.g. RCCL / the stock aiter path) at the same shapes.

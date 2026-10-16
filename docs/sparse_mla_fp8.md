# FP8 sparse MLA on gfx950

`sparse_mla_fp8_fwd` implements sparse attention over a shared FP8 latent K/V
pool, with BF16 query conversion inside attention and BF16 output. It targets
H64/D512 on gfx950 (MI355X), without appended RoPE. The API is opt-in; an engine
must explicitly route eligible calls and provide its own fallback.

```python
import torch
from aiter import sparse_mla_fp8_fwd, allocate_sparse_mla_fp8_workspace

q = torch.randn((8, 64, 512), device="cuda", dtype=torch.bfloat16)
kv = torch.randn((8192, 512), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
indices = torch.randint(8192, (8, 2112), device="cuda", dtype=torch.int32)
indices[:, 2051:] = -1
out = torch.empty_like(q)
workspace = allocate_sparse_mla_fp8_workspace(q)

# Warm the JIT before capture. Supplying storage avoids repeated allocation.
out, _ = sparse_mla_fp8_fwd(q, kv, indices, 1 / 16, out=out, workspace=workspace)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out, _ = sparse_mla_fp8_fwd(q, kv, indices, 1 / 16, out=out, workspace=workspace)
# Update q, kv and indices in place, then replay for the same shapes.
graph.replay()
```

## Input and numerical contract

| Argument | Contract |
|---|---|
| Q | BF16 or `float8_e4m3fn`, `[Q,64,512]`, Q from 0 through 2048 |
| Q layout | Unit inner stride, non-overlapping rows, pointer and row strides aligned to 16 bytes |
| KV | Contiguous `float8_e4m3fn`, `[slots,512]`, `[pages,page_size,512]`, or `[slots,1,1,512]` |
| Slot IDs | Contiguous int32 `[Q,K]` or `[Q,1,K]`, capacity K from 0 through 2176 |
| Scales | Finite positive Python scalars: `softmax_scale`, `q_scale`, `kv_scale` |
| O | Contiguous BF16 `[Q,64,512]` |
| Optional LSE | Contiguous FP32 `[Q,64]`, natural logarithm |
| Workspace | Opaque flat BF16 partial output and FP32 partial LSE from the allocator |

Slot IDs address physical rows in the flattened pool. Negative and out-of-range
IDs are masked; duplicate IDs retain their multiplicity. K is the array capacity,
including padding. Empty selections produce O=0 and LSE=-infinity. KV addresses
use 64-bit arithmetic, including for pools larger than 4 GiB.

BF16 Q is encoded as `FP8(Q.float() / q_scale)`. FP8 Q is already encoded and is
not quantized again. Decoding scales apply to Q, K and V; `kv_scale` is used in
both the score and value paths. The E4M3FN conversion uses round-to-nearest-even,
signed zero and gradual underflow; overflow produces NaN, matching the PyTorch
cast. Choose a scale that keeps model activations representable. P is encoded
as FP8 with decoding scale 1/256. There is one FP8 Q component and one QK
computation. Accumulators, softmax and LSE are FP32; split partial O and final O
are BF16.

Numerical tests compare against independent FP64 attention on the quantized Q,
with per-head relative RMS <=0.07 and elementwise atol/rtol=0.07. This measures
implementation error for FP8 attention; it does not imply equivalence to BF16
attention or establish model accuracy.

The operator uses the current HIP stream. Validation reads tensor metadata,
not GPU values. Graph capture fixes the shape-dependent dispatch and replay
reads updated tensor contents. Outputs and workspace must not overlap any
input or each other. Unsupported devices and metadata raise errors.

## Dispatch and implementation

| Q | Heads per CTA | K tile | Splits | Main loading path |
|---|---:|---:|---:|---|
| 1–6 | 16 | 128 | 17 | Small-query head tiles |
| 7–12 | 32 | 128 | 9 | Two KV tiles prefetched around Q conversion |
| 13–21 | 32 | 128 | 6 | Rotating KV buffers |
| 22–23 | 32 | 128 | 4 | Rotating KV buffers |
| 24–42 | 64 | 128 | 6 | Rotating KV buffers |
| 43–64 | 64 | 128 | 4 | Rotating KV buffers |
| 65–128 | 64 | 128 | 2 | Streaming double buffer |
| 129–2048 | 64 | 128 | 1 | Streaming double buffer, direct output |

Q8 uses head-major partial LSE and a packed FP32 reducer. Other split paths use
their corresponding split-major layout. Workspace is opaque so callers do not
need to reproduce these layouts. No workspace initialization kernel is required.
The 2112-entry selection capacity has a specialization in the prefetch path;
other supported capacities use the general path.

## Reproduction

The HIP implementation uses AITER's ctypes JIT and HIP intrinsics/assembly; it
does not require HipKittens. Build with `GPU_ARCHS=gfx950`. Warm the operator
before Graph capture. Run correctness and replay checks with:

```bash
GPU_ARCHS=gfx950 python -m pytest -q op_tests/test_sparse_mla_fp8.py
```

The benchmark accepts BF16 Q for both HIP and Gluon, includes conversion inside
attention, and ends with BF16 O ready. It uses a random 5,608,046,592-byte FP8 KV
pool, 2051 valid slots per query, capacity 2112, and `-1` padding. The default
queries are every power of two from 1 through 2048; `--all-small` additionally
covers Q1–64 individually and Q65.

```bash
GPU_ARCHS=gfx950 python op_tests/op_benchmarks/bench_sparse_mla_fp8.py \
  --all-small --output /tmp/sparse-mla-fp8-results
```

Each of two fresh Q seeds is measured in forward and reverse order. Each phase
uses 30 timing samples, with 20 calls captured per HIP Graph. The reported time
is the median of the four phase medians. Compilation, host allocation and
Python execution are excluded; captured GPU work is included. Raw samples,
launch tile size, configuration and per-phase numerical checks are saved.

HIP uses fixed Q scale one. Gluon uses dynamic tile quantization, checked against
its source-order quantized-Q reference. That reference does not assert bitwise
identity between Torch division and the hardware reciprocal instruction.

`speedup = Gluon_us / HIP_us - 1`. Logical KV bandwidth is
`Q * valid_K * 512 / HIP_us / 1e6` TB/s: one selected FP8 KV payload per query.
It excludes repeated head reads, Q/O and intermediate/index traffic, so it is
not measured HBM traffic. Operator timings do not establish serving throughput.

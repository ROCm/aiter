# BF16 sparse MLA on gfx950

`aiter.sparse_mla_bf16_fwd` computes attention over a per-query CSR list of
global cache slots. K and V are the same BF16 latent, D=512, with no appended
RoPE dimensions. Both prefill and decode use this contract. H16 and H64 are
supported; GLM-5.3-Flash is one caller of this geometry.

This is an explicit HIP entry point. The existing Gluon `sparse_mla_fwd`
continues to handle its broader set of formats and geometries. An engine can
select this entry point after checking its device, dtype, head count and
geometry, and retain its existing implementation for other configurations.

## API and storage

```python
import aiter
import torch

# q: [Q,H,512] BF16; kv: [slots,512] BF16
# ptr: [Q+1] int32; idx: [nnz] int32 GLOBAL physical slot ids
splits = 8  # choose outside capture; supported: 1,2,4,8,16,32
out = torch.empty_like(q, memory_format=torch.contiguous_format)
lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
workspace = (
    torch.empty((q.shape[0], splits, q.shape[1], 512), device=q.device),
    torch.empty((q.shape[0], splits, q.shape[1]), device=q.device),
)
result, result_lse = aiter.sparse_mla_bf16_fwd(
    q, kv, ptr, idx, softmax_scale,
    version=2, kv_splits=splits,
    out=out, return_lse=True, lse=lse, workspace=workspace,
)
```

O is contiguous BF16 `[Q,H,512]`. LSE is optional, contiguous FP32 `[Q,H]`
and uses the natural logarithm. Split-K workspace contains normalized FP32
partial O and natural-log partial LSE, contiguous `[Q,S,H,512]` and `[Q,S,H]`.
It is unused for S=1. The convenience wrapper allocates omitted buffers;
supplying all buffers makes the call allocation-free after JIT warmup.
Warm the extension before capturing a graph. Calls use the query device's
current stream and do not synchronize to inspect CSR values.

Q/KV allow padded outer strides and BF16 storage offsets, with stride one
in the last dimension. KV also accepts flat-view-compatible
`[pages,page_size,512]` and `[slots,1,1,512]` layouts. Paged inputs are viewed,
not copied. Out/workspace storage must not overlap inputs or one another.
An output with a larger physical head stride requires the engine to use a
contiguous latent output and then its existing projection/copy path.

The caller supplies monotone CSR offsets starting at zero and ending at
`idx.numel()`. Slot ids outside `[0,slots)` are masked; duplicates retain
their multiplicity. Empty or fully masked rows yield O=0, LSE=-inf.
Causal/window selection is expressed through CSR. There is no implicit
top-k truncation, attention sink, cache quantization, or separate RoPE input.
In particular, a 2051-entry row preserves all 2051 entries.

## Implementations

| Version | H16 main | H64 main | Split-K reduction |
|---|---|---|---|
| 1 | One wave, Q in VGPR, 33,024 B LDS | Full140: eight waves, Q in LDS, 139,520 B LDS | Fully unrolled one-wave H64; original H16 reducer |
| 2 | Four waves, 68,352 B LDS | Four head tiles when Q×S≤128 (68,352 B/CTA), otherwise four waves (65,792 B/CTA) | Fully unrolled one-wave H16/H64 |

The version selector keeps both implementations reproducible under one API.
The default version is 2; the default split count is the explicit S=1,
not an autotuned choice. Use the supplied sweep to choose a split count
for the deployment workload. No automatic routing changes to the existing
Gluon API are part of this operator.

The extension reuses AITER's existing HipKittens dependency and pinned register
helpers from [PR #3459](https://github.com/ROCm/aiter/pull/3459). No new external
dependency is introduced. This module uses AITER's CK-free headers. Like the
other HK modules, it is excluded from the
ordinary all-op prebuild unless `AITER_ENABLE_EXPERIMENTAL=1`; explicit JIT
calls can build it directly. The module preserves FP32 denormals, overriding
AITER's usual flush-to-zero flag. Compiler-managed VGPRs must stay below v40.

## Reproduce correctness and timing

```bash
python -m pytest -q op_tests/test_sparse_mla_bf16.py
python -m op_tests.op_benchmarks.bench_sparse_mla_bf16 \
  --verify --output sparse_mla_o_only.jsonl
python -m op_tests.op_benchmarks.bench_sparse_mla_bf16 \
  --verify --return-lse --output sparse_mla_o_lse.jsonl
# Independent repeat with reversed case/variant order and fresh allocations:
python -m op_tests.op_benchmarks.bench_sparse_mla_bf16 \
  --verify --reverse --output sparse_mla_o_only_repeat.jsonl
```

The benchmark defaults to H16/H64, Q=1/32/128/256/512/1024/2048, K=2048/2051,
S=1/2/4/8/16/32. Each query owns a disjoint 32K-slot pool. Q=2048 uses a
64 GiB KV allocation, so run one benchmark on an otherwise idle MI355X.
Native v1/v2 share input, final output and workspace addresses. The Gluon
comparison comes from the same checkout and includes explicit splits and
its shipped split policy. Internal Gluon workspaces differ from the native
workspace; their traffic is part of the timed operation.

Timing uses alternating variant order and 60 graph/event samples per point.
Warm and cold results are separate. Cold flushes 512 MiB before the graph's
start event. JIT and Python allocation/dispatch time are excluded. Every
case checks sampled query rows against independent FP64 before and after
timing, and hashes all inputs before/after. Correctness failures are reported
and excluded from ratios; a native failure aborts the run.

Report the complete per-shape table, matched splits, best tested splits and
Gluon's shipped policy separately. JSONL includes raw samples, median/p90,
source/binary hashes, logical GB/s and QK+PV TFLOP/s. Logical traffic counts
Q, one logical gather of KV, indices, O and optional LSE; it excludes repeated
reads and partial buffers and is not measured HBM traffic. A roofline derived
from it is a lower-bound model, not proof of a memory bottleneck.

For execution-path validation, run a small sweep under `rocprofv3
--kernel-trace --hip-trace` with `--purpose trace --samples 1 --warmup 0`.
Trace durations are diagnostic and must not be used as performance results.
Build `sparse_mla_bf16.cu` with `--save-temps` and check the retained assembly:

```bash
python op_tests/check_sparse_mla_bf16_isa.py path/to/sparse_mla_bf16-hip-amdgcn-amd-amdhsa-gfx950.s
```

Operator correctness and latency are distinct from model quality and serving
performance. An engine integration should validate its logical-to-physical
index mapping, selected-set policy (including any tail fitting), graph buffer
reuse and fallback, then measure the model with a stable baseline.

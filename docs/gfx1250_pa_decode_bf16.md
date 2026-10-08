# Native BF16 in gfx1250 planned PA decode

The planned gfx1250 backend accepts native BF16 K/V caches and uses
`v_wmma_f32_16x16x32_bf16` for QK and PV, with FP32 accumulation and softmax
statistics. BF16 and FP16 queries/outputs are supported; FP16 queries are
converted to BF16 operands. BF16 probabilities are stored directly, without
FP8 normalization or conversion.

`compute_type=None` infers compute from the cache dtype. Passing
`compute_type=torch.bfloat16` selects native BF16 explicitly. K and V must both
have the selected dtype. BF16 decode rejects `key_scale` and `value_scale`;
it neither allocates nor reads quantization-scale buffers. Native BF16 compute
is available on gfx1250. gfx942/gfx950 retain their FP8 paths.

BF16 caches use eight-element packing, keeping each contiguous chunk at 16 bytes:

| Cache | Shape |
|---|---|
| K | `[pages, Hkv, D // 8, page_size, 8]` |
| Transposed V | `[pages, Hkv, page_size // 8, D, 8]` |
| Plain V | `[pages, Hkv, D, page_size]` |

FP8 retains its 16-element packing. BF16 uses twice the K/V storage of FP8.
The caches must satisfy the existing contiguous-layout and finite-padding
requirements. In particular, unwritten tokens in a transposed V cache's last
owned page must contain finite values.

Given BF16 `raw_key` and `raw_value` in `[pages, Hkv, page_size, D]` order,
queries in `[batch * query_length, Hq, D]` order, and valid GPU lengths/tables:

```python
import torch
from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

pages, kv_heads, page_size, head_dim = raw_key.shape
key_cache = (
    raw_key.reshape(pages, kv_heads, page_size, head_dim // 8, 8)
    .permute(0, 1, 3, 2, 4)
    .contiguous()
)
value_cache = (
    raw_value.reshape(pages, kv_heads, page_size // 8, 8, head_dim)
    .permute(0, 1, 2, 4, 3)
    .contiguous()
)
plan = plan_pa_decode(context_lengths, kv_heads, query_length=query_length)
output = torch.empty_like(query)
pa_decode(
    output, query, key_cache, value_cache, context_lengths, block_tables,
    head_dim**-0.5, query_length,
    compute_type=torch.bfloat16,
    work_plan=plan,
)
```

The plan contract remains 256 tokens, with four wave32 waves computing 64 tokens
at a time. BF16 TDM loads power-of-two D64/128/256/512 tiles directly into padded
LDS rows; the padding interval is at most 1024 bytes. Other supported dimensions,
including D384/768/1024, use vector DMA. Double buffering overlaps the next K/V
loads with current compute when LDS capacity permits. Byte extents determine
wide-address dispatch, including BF16 caches crossing 2/4 GiB.

The widest BF16 MTP case reuses one Q tile and one K/V buffer to fit in 320 KiB
LDS. D1024/G16/MTP4 passes correctness for both V layouts, but its large
accumulator state spills registers (1508 private bytes per work-item in the
checked specialization). Its performance needs further tuning.

Build the plan before graph capture and supply scratch buffers during capture.
Refresh the plan after changing lengths, as for FP8. The current
`pa_decode_tuning` CLI/cache generates FP8 workloads; use the default plan budget
or an explicitly measured `workgroup_budget` for BF16. Timing results below
search budgets without changing production defaults or installing cache entries.

Validation on gfx1250 passed 163 GPU cases: the existing 114 FP8 cases and 49
native BF16 cases. BF16 coverage includes both query dtypes and V layouts,
page16/64/128, D64/128/256/384/512/768/1024, empty and partial contexts, MTP,
windows, sinks, probability tails, small queries, refreshed graphs, odd query
and output strides, poisoned scratch, wide cache addresses, and scale rejection.
gfx942/gfx950 MTP4 tile/reducer compile checks also passed; numerical validation
on those architectures was not performed.

```bash
GPU_ARCHS=gfx1250 python3 -m pytest -q \
  op_tests/test_flydsl_pa_decode.py \
  op_tests/test_flydsl_pa_decode_gfx1250.py \
  op_tests/test_flydsl_pa_decode_bf16.py
```

Fresh, single-specialization ISA checks on D128/G8/MTP1 show unchanged FP8
resources relative to `4b4246125`: 106 VGPRs, 90 SGPRs, 44096 static LDS bytes,
zero register spills and zero private bytes. Native BF16 uses 111 VGPRs,
70 SGPRs and 79936 static LDS bytes, also with zero spills/private bytes.

## Performance versus FP8 and PR #3256

Measured on 2026-10-08 with gfx1250, 256 CUs, 320 KiB LDS and 23.347 TB/s reported
peak bandwidth, using FlyDSL 0.3.3.dev886, PyTorch 2.11.0+rocm10.1 and HIP 7.16.
GQA uses Hq64/Hkv8; MQA uses Hq16/Hkv1. All rows use D128, page128,
BF16 queries/outputs, shuffled unique physical pages and dense contexts.

Native BF16, FP8 and PR #3256 ran in one process with exactly equivalent K/V
values. FP8 has per-token power-of-two scales, and its dequantized values equal
the BF16 caches. Each candidate was checked against an FP32 reference on sampled
sequences before timing. Each graph contains 32 decode/reduce calls; seven
randomized timing rounds use 12 replay windows per candidate. Initialization,
planning, quantization, compilation and Python launch overhead are excluded.

| Workload | FP8 µs | Native BF16 µs | PR #3256 BF16 µs | BF16 speedup vs PR | BF16 effective KV TB/s |
|---|---:|---:|---:|---:|---:|
| MQA B4/4K | 13.73 | 12.33 | 24.97 | 2.025× | 0.680 |
| GQA B16/4K | 24.55 | 33.31 | 38.15 | 1.146× | 8.060 |
| GQA B64/4K | 69.46 | 111.12 | 117.03 | 1.053× | 9.663 |
| GQA B64/16K | 225.03 | 417.98 | 458.30 | 1.096× | 10.275 |
| GQA B64/16K/MTP4 | 338.13 | 509.00 | — | — | 8.438 |

Rows compare each implementation's best searched configuration. New backends
search budgets 128/256/512/1024/2048/4096; PR #3256 searches partition sizes
256/512/1024/2048/4096 and compute tiles 128/256. Best BF16 budgets are
128/1024/4096/4096/2048 in row order. PR #3256 lacks the MTP path used here.
Its checkout is `495b57fb7f3a70180af80b129c1d19b0d876b895` with the recorded
FlyDSL API compatibility patch; these are local measurements with the installed
compiler. The PR implementation already uses TDM.

BF16 is faster than PR #3256 in these cases, but slower than FP8 for the larger
workloads because K/V storage doubles and native BF16 contracts K32 instead of
FP8 K128/K64. Effective bandwidth counts useful K/V bytes once across GQA/MTP
reuse, excluding scales, Q/O, scratch and metadata. B64/16K reads 4 GiB of useful
BF16 K/V, yielding 10.275 TB/s (44.0% of reported peak). This metric is distinct
from a physical memory-bus measurement.

The paired scripts, raw timing samples, ISA, correctness logs and physical-read
counters are kept in the validation workspace at
`/home/sixifang/gfx1250-dev/pa-gfx1250-validation/BF16_PERFORMANCE.md`.
Resource verification follows FlyDSL's
[isa-resource-diff](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/isa-resource-diff/SKILL.md)
workflow; staging preserves the existing
[prefetch-data-load](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/prefetch-data-load/SKILL.md)
pipeline.

A separate paired FP8 regression sweep against `4b4246125` measured
70.97→70.90 µs at B64/4K, 230.51→230.77 µs at B64/16K, and
344.07→344.54 µs at B64/16K/MTP4. Changes are within 0.15% in this run,
consistent with the unchanged checked FP8 ISA resources. This sweep uses
different input values from the equivalent-data BF16/PR comparison.

Separate ten-replay ROCprofv3 `FETCH_SIZE` passes at B64/16K/budget4096
measured 3.8437 GiB of physical reads per BF16 call and 1.9867 GiB per FP8
call, including the reducer. Using the independent unprofiled latencies gives
9.874 TB/s for BF16 (42.3% of reported peak) and 9.480 TB/s for FP8. Counters
include cache/memory effects and exclude writes; profiler latency is not used
for speedups.

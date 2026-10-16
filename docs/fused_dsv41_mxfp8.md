# Fused DSV4.1 sparse attention and MXFP8 output

`aiter.fused_dsv41_mxfp8` combines optional forward Q RoPE, sparse attention,
inverse output RoPE, and group32 MXFP8 quantization in one HIP launch on gfx950.
It returns `(out, out_scale, max_logits, lse)` for the grouped output projection.
The operator accepts tensors and has no dependency on the ATOM Python runtime.

## Tensor contract

| Argument | Dtype and shape | Meaning |
| --- | --- | --- |
| `q` | BF16 `[T,H,512]` | Contiguous head-major Q; H = 8, 16, 32, 64 or 128 |
| `pool` | uint8 `[bytes]` or `[bytes,1]` | FP8 data backing; legacy rows or paged data |
| `prefix` | int64 (legacy) / int32 (paged) `[capacity]` | Tagged byte addresses / physical slots |
| `prefix_indptr` | int32 `[T+1]` | CSR pointers into prefix |
| `cos`, `sin` | BF16 `[P,32]` | Adjacent-pair RoPE tables for the last 64 channels |
| `positions` | int32/int64 `[T]` | Positions in `[0,P)` |
| `extend_kv` | optional BF16 `[S,512]` | Current-chunk KV; already rotated |
| `extend`, `extend_indptr` | optional int32 `[capacity]`, `[T+1]` | Extend row indices and CSR |
| `attn_sink` | optional FP32 `[H]` | Per-head extra logit in the denominator |
| `out`, `out_scale` | E4M3FN `[T,H/8,4096]`, E8M0 `[T,H/8,128]` | Grouped quantized output |
| `max_logits`, `lse` | FP32 `[T,H]` | Statistics excluding the sink |

In legacy mode (`cache_page_size=0`), each physical KV row is 528 bytes: 512 E4M3FN values followed by 16 E8M0
scales, one for each 32 values. Addresses are relative to `pool.data_ptr()`;
the pool base and live row offsets must be aligned to four bytes. Unsupported
tags and out-of-range addresses/extend indices are masked. All CSR pointers
must be monotonic and within capacity. Value-dependent preconditions (CSR,
positions, finite values) are caller responsibilities; no host synchronization
is inserted to inspect GPU contents.

All tensors are contiguous on the same GPU. Q, BF16 extend, quantized output
and optional debug output require 16-byte alignment. Output buffers must not
overlap inputs or each other. Pass all three extend tensors together, or omit
all three for pure-cache attention. FP4-tagged cache rows are unsupported.

Q and selected KV values must be finite; sink may also be negative infinity.
Attention rounds to BF16 before inverse RoPE and quantization. Empty rows
produce zero output, negative-infinity max, and positive-infinity LSE. Optional
FP32 `debug_out[T,H,512]` receives the result before MXFP8 quantization.

## Independent paged data and scales

Set `cache_page_size=B>0` to read int32 physical slots from `prefix` using the
same CSR pointers. Every slot has 512 E4M3FN data bytes and 16 E8M0 scales.
Supply these keyword arguments together:

| Argument | Meaning |
| --- | --- |
| `cache_scales` | Contiguous uint8 `[bytes]` view starting at the scale base |
| `cache_page_size` | Slots per physical page, positive int32 |
| `cache_num_rows` | Valid physical slots, nonnegative int32 |
| `cache_data_page_stride` | Byte distance between data pages, at least `B*512` |
| `cache_scale_page_stride` | Byte distance between scale pages, at least `B*16` |

For physical slot `s`, both addresses are calculated independently:

```text
page = s // B; row = s % B
data  = pool.data_ptr()         + page * data_page_stride  + row * 512
scale = cache_scales.data_ptr() + page * scale_page_stride + row * 16
```

Both bases and both page strides require four-byte alignment. Strides and
address arithmetic are 64-bit. Negative slots, slots outside `cache_num_rows`,
and rows truncated by either byte view are masked. The host computes the
number of complete addressable rows from each view's size and caps the valid
range before launch; this does not read GPU contents or synchronize the GPU.
Data rows cannot address the page's scale section or padding. Only live CSR
ranges are read. A zero valid-row count is supported.

For vLLM-style page partitions `[B*512 data | B*16 scales | padding]`, pass
two views of the same allocation, with equal page strides:

```python
flat = cache.view(-1)  # contiguous uint8 storage, no copy
page_bytes = cache.stride(0)  # for uint8 [num_pages, page_bytes]
paged_options = dict(
    cache_scales=flat[B * 512:],
    cache_page_size=B,
    cache_num_rows=num_valid_slots,
    cache_data_page_stride=page_bytes,
    cache_scale_page_stride=page_bytes,
)
out, scales, max_logits, lse = fused_dsv41_mxfp8(
    q, flat, slots_int32, prefix_indptr, cos, sin, positions,
    q_is_rotated=False, **paged_options,
)
```

For separate allocations, pass the data backing as `pool`, the scale backing
as `cache_scales`, and their respective page strides. Neither case repacks KV.
Warm up and retain these views and options when capturing/replaying a graph.
All cache metadata must be zero/omitted in legacy mode. The low-level entry
has the same five new parameters appended with legacy-compatible defaults.

This repairs a **single FP8 cache with CSR indexing**. It is not a full mega
interface: two independently paged FP8 SWA/compressed sources, dense slot
matrices plus lengths, Q16/O32 interleaving, FP32 RoPE, MN-major packed output
scales, token-slice output writes and complete model forward are separate
integration work. Optional `extend_kv` remains BF16. The numerical path still
rounds attention to BF16 before inverse RoPE and quantization.

## Contiguous BF16 cache / prefill workspace

```python
from aiter import fused_dsv41_bf16

out, scales, max_logits, lse = fused_dsv41_bf16(
    q, kv_bf16, row_indices_int32, indptr_int32, cos, sin, positions,
    q_is_rotated=False, attn_sink=sink,
)
```

`kv_bf16` is contiguous, 16-byte-aligned BF16 `[S,512]` or `[S,1,512]` on
Q's device, with KV RoPE already applied. Indices are flat int32 logical row
indices with CSR `indptr[T+1]`. Negative/out-of-range indices are masked;
empty rows use the same zero/max/LSE convention. FP8 page metadata must be
omitted. The singleton-head view performs no copy. The general
`fused_dsv41_mxfp8` entry also accepts a two-dimensional BF16 pool.

Optional output buffers and stream/graph behavior match the FP8 entry. No
gather/dequant or index conversion occurs here; callers can supply a native
BF16 cache or a previously decoded workspace. Q/RoPE/output formats remain
the existing AITER formats. This is not the full FlashMLA mega tensor ABI.

See [decode profiling and optimized FP8/BF16 comparison](fused_dsv41_decode_prefill_20261016.md)
for the measured changes, input reuse assumptions and timing boundaries.

## Usage and graph replay

```python
from aiter import fused_dsv41_mxfp8

# KV rows already include RoPE; fuse Q RoPE by setting q_is_rotated=False.
out, scales, max_logits, lse = fused_dsv41_mxfp8(
    q, pool, prefix, prefix_indptr, cos, sin, positions,
    q_is_rotated=False, attn_sink=sink,
)

# Warm up before capture; keep tensor addresses fixed for graph replay.
fused_dsv41_mxfp8(
    q, pool, prefix, prefix_indptr, cos, sin, positions,
    q_is_rotated=False, attn_sink=sink,
    out=out, out_scale=scales, max_logits=max_logits, lse=lse,
)
```

The low-level `fused_dsv41_mxfp8_fwd` entry accepts only caller-allocated
outputs. Call it with Q's device current, following AITER's stream convention.
The convenience entry above selects Q's device automatically.

## Kernel dispatch and compilation

Legacy and paged addressing use separate compile-time kernel variants. B32/B128
pages are specialized; other page sizes use the general paged variant.

H64/H128 pure-cache attention uses the specialized pipeline for all token
counts, for either FP8 cache or BF16 workspace. H64 prefetches FP8 before QK
and commits after QK; H128 commits after PV. BF16 loads directly to LDS at
the corresponding stage. H64 retains eight lane-specific scale bytes; H128
reads KV validity flags from LDS. Scheduling barriers protect PV consumers.
H8/H16/H32 and mixed prefix/extend retain the established shared-KV kernels;
mixed-source H128 switches to its late loader at `T >= 2048`.

The module uses AITER's `compile_ops(..., develop=True)`, `aiter_tensor_t`,
pybind and current HIP stream. `-ffp-contract=off` preserves the inverse-RoPE
rounding sequence. The device source disables flush-to-zero to preserve the
finite E8M0 codec's subnormal range. Explicit fused operations in Q RoPE remain
fused. There is no separate custom compiler or runtime cache loader.

Q is already head-major: its layout can be produced by the preceding GEMM.
Standalone layout preparation, GEMM and cache writes are outside operator
timing. The operator does not switch an entire model forward to this path.

## Validation

```bash
GPU_ARCHS=gfx950 python -m pytest op_tests/test_fused_dsv41_mxfp8.py op_tests/test_fused_dsv41_paged.py op_tests/test_fused_dsv41_bf16.py -q
GPU_ARCHS=gfx950 python op_tests/test_fused_dsv41_mxfp8.py --benchmark --tokens 560 --heads 64 --kv 640
```

Tests use an independent PyTorch dense oracle and physical packed rows, with
no external model checkout. Coverage includes all supported head counts,
raw/rotated Q, BF16 extend, long/ragged rows, empty/invalid interior tiles,
codec extremes, four-byte row alignment, graph replay after CSR changes,
preallocated outputs, non-default streams, metadata rejection and the
1024/2048-token dispatch thresholds. The CLI benchmark is a synthetic
repeated-row fixture for local smoke/performance checks.


The paged-cache repair passes 183 native tests (49 existing and 134 paged).
Paged coverage includes independent allocation, same-page sections, padding,
non-power-of-two pages, tail rows, invalid interior tiles, both byte-view bounds,
large strides, a within-page data offset above 2 GB, all heads, raw/rotated Q,
BF16 extend, non-default streams, preallocated graph replay, and dispatch
boundaries at T=1024/1025/2047/2048/2049. The independent oracle decodes actual
cache bytes. Separate integration diagnostics directly consume the fixed vLLM
writer (revision `9f6e18fe642a187bf5e8aa9a5bf21e96217fedb4`) for B32/B128.

### Historical legacy-layout measurements

The following measurements predate the paged repair and validate only legacy
contiguous 528-byte rows. They are not paged-cache performance results.

The integrated build was validated on MI355X (gfx950), ROCm 7.2.4: 49 native
tests and 66 existing ATOM codec/cache/adapter tests passed. The latter suite
replaced only the tensor attention entry with this AITER operator; it is not
a full-model forward validation. H64 raw/rotated-Q scratch remains 84/96
bytes per thread (256 VGPRs, 67,584-byte LDS per block).

The following fixed CPU-generated decode fixtures used preallocated outputs,
raw Q, eight calls per graph, ten replays per event sample, five samples per
round, and six rounds alternating implementation order. Values are medians of
round medians; clocks were not locked. E4M3FN output and E8M0 scales were
bit-exact against the optimized standalone implementation in all six cases;
max/LSE passed at `rtol=atol=1e-6`. S denotes each request's long-cache length.

| T | H | K | S | Standalone (us) | AITER (us) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 560 | 64 | 640 | 8192 | 206.00 | 208.42 |
| 560 | 64 | 640 | 32768 | 209.27 | 211.98 |
| 960 | 64 | 640 | 8192 | 257.62 | 252.92 |
| 960 | 64 | 640 | 32768 | 260.94 | 256.07 |
| 320 | 128 | 1152 | 8192 | 287.00 | 294.27 |
| 320 | 128 | 1152 | 32768 | 296.44 | 302.88 |

These measurements include Q RoPE, attention, inverse output RoPE and MXFP8
quantization. They exclude the producer GEMM, cache writer, standalone layout
preparation and the rest of the model.


### Paged repair measurements (2026-10-16)

The same logical six fixtures were constructed directly in a single B128
page-partition cache. Results below use the final source and a fresh JIT build;
they are not a measurement of a dual-cache mega integration. Three variants
ran in all six execution-order permutations, with five event samples each.
Each sample contains eight calls per graph and ten replays. All six quantized
outputs/scales are bit-exact against the legacy standalone baseline; sampled
dense max/LSE/debug oracles and 108 poison/replay checks pass. There are 540
event samples. Q RoPE is included; producer GEMM and Q rearrangement are excluded.

| T | H | K | S | Legacy standalone (us) | Candidate legacy (us) | Candidate paged (us) | paged TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 64 | 640 | 8192 | 206.21 | 211.75 | 207.14 | 226.78 |
| 560 | 64 | 640 | 32768 | 210.04 | 215.84 | 211.33 | 222.29 |
| 960 | 64 | 640 | 8192 | 258.14 | 258.53 | 253.40 | 317.80 |
| 960 | 64 | 640 | 32768 | 262.15 | 262.70 | 257.01 | 313.34 |
| 320 | 128 | 1152 | 8192 | 288.14 | 334.58 | 315.07 | 306.71 |
| 320 | 128 | 1152 | 32768 | 296.49 | 342.28 | 334.80 | 288.64 |

H64 paged latency is within approximately -2% to +1% of the legacy standalone
baseline in this run. H128 paged latency is approximately 9-13% higher; the
candidate's H128 legacy mode is also slower than the standalone baseline.
Correctness is established; performance parity is not claimed. B128 paged
raw/rotated-Q scratch is 84/96 bytes per thread for H64 and 16/8 for H128;
both have 256 VGPRs, 0 AGPRs and 67,584 bytes LDS per block.


### H128 load-stage and scale-lifetime optimization (2026-10-16)

H128 restores the lane range after opaque inline assembly, keeps lane address
calculation local to prefetch/commit, and retains only four lane-specific scale
bytes across QK/PV instead of all sixteen scale bytes. Cache addressing and
the public/numerical contracts are unchanged. B32/B128 raw/rotated-Q scratch
falls from 16/8 to 0/0 bytes per thread; VGPRs fall from 256/256 to 254/252.
Generic-page scratch falls from 48/48 to 8/8. H64 resources are unchanged.
LDS remains 67,584 bytes per block; no occupancy increase is claimed.

The selected source and fresh JIT passed 183 native tests. The source and
binary were unchanged for this final run, so that regression was retained
and checked by hash rather than rerun. A fresh pinned vLLM writer run passes
all 20 cases with zero error, including the low-level legacy defaults check.
All six fixtures pass bit-exact quantized-output/scale comparisons, max/LSE
checks and 18 sampled dense oracles; 300 poison/replay checks pass.

The following same-run comparison uses MI355X HIP6, five implementations,
ten forward/reverse cyclic execution orders and 1,500 event samples total.
Each sample contains eight calls per graph and ten replays. Medians are the
median of each visit's five-sample median. Clocks were not locked. Scope is
the single B128 page-partition cache, including Q RoPE and excluding GEMM
and Q rearrangement; it is not a dual-cache mega/model-forward measurement.
Fixed means the independent-address repair at `8983a1026`.

| T | H | K | S | Standalone us | Fixed paged us | Optimized paged us | Paged latency reduction | TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 64 | 640 | 8192 | 206.60 | 206.86 | 207.54 | -0.33% | 226.35 |
| 560 | 64 | 640 | 32768 | 209.85 | 211.37 | 211.26 | 0.05% | 222.36 |
| 960 | 64 | 640 | 8192 | 258.04 | 253.30 | 253.01 | 0.11% | 318.29 |
| 960 | 64 | 640 | 32768 | 261.84 | 256.02 | 255.99 | 0.01% | 314.58 |
| 320 | 128 | 1152 | 8192 | 286.93 | 313.98 | 281.35 | 10.39% | 343.47 |
| 320 | 128 | 1152 | 32768 | 296.25 | 333.66 | 293.52 | 12.03% | 329.24 |

H128 paged latency drops by 10.39%/12.03% relative to the address repair and
is 1.94%/0.92% below the standalone baseline. H64 paged varies from -0.11%
to +0.33% in latency relative to the repair. H128 legacy mode also improves,
but remains about 8.8-9.2% slower than the standalone baseline:

| S | Standalone us | Fixed legacy us | Optimized legacy us |
|---:|---:|---:|---:|
| 8192 | 286.93 | 333.35 | 313.36 |
| 32768 | 296.25 | 341.98 | 322.27 |

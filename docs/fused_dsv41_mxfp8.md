# Fused DSV4.1 sparse attention and MXFP8 output

`aiter.fused_dsv41_mxfp8` combines optional forward Q RoPE, sparse attention,
inverse output RoPE, and group32 MXFP8 quantization in one HIP launch on gfx950.
It returns `(out, out_scale, max_logits, lse)` for the grouped output projection.
The operator accepts tensors and has no dependency on the ATOM Python runtime.

## Tensor contract

| Argument | Dtype and shape | Meaning |
| --- | --- | --- |
| `q` | BF16 `[T,H,512]` | Contiguous head-major Q; H = 8, 16, 32, 64 or 128 |
| `pool` | uint8 `[bytes]` or `[bytes,1]` | Packed physical FP8 KV backing |
| `prefix` | int64 `[capacity]` | Packed addresses `(byte_offset << 1) \| 1` |
| `prefix_indptr` | int32 `[T+1]` | CSR pointers into prefix |
| `cos`, `sin` | BF16 `[P,32]` | Adjacent-pair RoPE tables for the last 64 channels |
| `positions` | int32/int64 `[T]` | Positions in `[0,P)` |
| `extend_kv` | optional BF16 `[S,512]` | Current-chunk KV; already rotated |
| `extend`, `extend_indptr` | optional int32 `[capacity]`, `[T+1]` | Extend row indices and CSR |
| `attn_sink` | optional FP32 `[H]` | Per-head extra logit in the denominator |
| `out`, `out_scale` | E4M3FN `[T,H/8,4096]`, E8M0 `[T,H/8,128]` | Grouped quantized output |
| `max_logits`, `lse` | FP32 `[T,H]` | Statistics excluding the sink |

Each physical KV row is 528 bytes: 512 E4M3FN values followed by 16 E8M0
scales, one for each 32 values. Addresses are relative to `pool.data_ptr()`;
the pool base and live row offsets must be aligned to four bytes. Unsupported
tags and out-of-range addresses/extend indices are masked. All CSR pointers
must be monotonic and within capacity. Value-dependent preconditions (CSR,
positions, finite values) are caller responsibilities; no host synchronization
is inserted to inspect GPU contents.

All tensors are contiguous on the same GPU. Q, BF16 extend, quantized output
and optional debug output require 16-byte alignment. Output buffers must not
overlap inputs or each other. Pass all three extend tensors together, or omit
all three for the FP8-only decode path. FP4-tagged cache rows are unsupported.

Q and selected KV values must be finite; sink may also be negative infinity.
Attention rounds to BF16 before inverse RoPE and quantization. Empty rows
produce zero output, negative-infinity max, and positive-infinity LSE. Optional
FP32 `debug_out[T,H,512]` receives the result before MXFP8 quantization.

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

H64/H128 with `T <= 1024` and no BF16 extend use raw FP8 prefetch. H64 issues
the next raw KV loads before QK and decodes after QK; H128 decodes after PV.
H64 rematerializes bounded lane indices at each phase to shorten address
register lifetimes and reduce scratch traffic. Scheduling barriers protect
PV accumulators while KV panels are reused. H8/H16/H32 and BF16 extend use
the established shared-KV pipelines. H128 switches to its late-load pipeline
at `T >= 2048`.

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
GPU_ARCHS=gfx950 python op_tests/test_fused_dsv41_mxfp8.py
GPU_ARCHS=gfx950 python op_tests/test_fused_dsv41_mxfp8.py --benchmark --tokens 560 --heads 64 --kv 640
```

Tests use an independent PyTorch dense oracle and physical packed rows, with
no external model checkout. Coverage includes all supported head counts,
raw/rotated Q, BF16 extend, long/ragged rows, empty/invalid interior tiles,
codec extremes, four-byte row alignment, graph replay after CSR changes,
preallocated outputs, non-default streams, metadata rejection and the
1024/2048-token dispatch thresholds. The CLI benchmark is a synthetic
repeated-row fixture for local smoke/performance checks.


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

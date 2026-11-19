# R1 NVFP4 MLA on gfx1250

`mla_decode_nvfp4_fwd` decodes one query per request with 128 heads, a shared
page64 latent KV cache, and BF16 output. The backend includes GPU producers
for BF16 queries and active KV tiles. It requires gfx1250 B0+ and does not
support split-K, MTP, context parallelism, logit capping or other head counts.

## Packed format and numerical contract

| Tensor | Contiguous shape and dtype | Meaning |
|---|---|---|
| `q_nope` | uint8 `[B,128,256]` | 512 NoPE E2M1 values, low nibble first |
| `q_aux` | uint8 `[B,128,128]` | RoPE base/residual, E4M3 scales, padding |
| `kv_nope` | uint8 `[P,16384]` | Page64 NoPE in 16-token x 32-channel bricks |
| `kv_aux` | uint8 `[P,8192]` | Page64 RoPE, residual and scales, N16 bricks |
| `kv_indptr` | int32 `[B+1]` | Page-count prefix sum |
| `kv_indices` | int32 `[capacity]` | Physical page IDs, valid up to `kv_indptr[-1]` |
| `kv_last_page_lens` | int32 `[B]` | 1..64 for nonempty rows, 0 for empty rows |
| `out` | bfloat16 `[B,128,512]` | Caller-owned final output |
| `lse` | float32 `[B,128]` | Natural-log LSE; also required as scratch |
| `params` | int32 `[4]` | Opaque ABI bits created by `prepare_mla_nvfp4_params` |

All tensors must share a device. KV base pointers require 256-byte alignment;
Q/output require 16-byte alignment. Normal PyTorch allocations satisfy these
requirements, while arbitrary offset views may not. Q's two planes have fixed
row strides of 256 and 128 bytes: slices of an interleaved row are unsupported.
The byte layout is distinct from FP8 MLA, MXFP4 and other NVFP4 operators.
Each token16 AUX tile occupies 2048 bytes: two 256-byte RoPE bricks, two
residual bricks, two NoPE-scale bricks, one RoPE/residual-scale brick and one
padding brick. Each brick has 16 rows of 16 bytes. The scale brick stores
four RoPE and four residual scales in each row; its remaining bytes are padding.

Gains are fixed: `G_Q=6.72`, `G_KV=89.6`, `G_P=2688`. Each reconstructed value
is `E2M1(code) * E4M3(scale) / gain`. The query producer takes absorbed,
RoPE-rotated BF16 `[B,128,576]`; it does not apply projections or rotary embedding.
Q NoPE and RoPE use per-row blocks of 16 channels. RoPE has an additional
independently quantized residual. The score includes
`QN*C + QR*KR + QE*KR + QR*KE`, excluding `QE*KE`.

C uses a shared scale for each **16-token x 16-channel** block. The GPU cache
writer quantizes C codes using `amax/6` and stores the rounded E4M3 scale.
Q and RoPE codes use the rounded stored scale. Scales clamp to finite E4M3
with a positive floor of `2**-9`; all-zero blocks use an ideal scale of 1.
Valid input values must be finite. This is a quantized attention operator:
agreement with its quantized oracle does not establish full-model accuracy.

The shader uses N128 tiles, FP4 probabilities and delayed threshold
rescaling. Each N16 group computes a local log2 maximum `b` and
`e = exp2(fma(raw_score, beta, -b))`. With the common N128 online anchor `a`,
`w = exp2(b - a)`. The numerator uses `FP4(6*e) * FP8(5*w)` and the denominator
accumulates unquantized FP32 `e*w`. Fully masked groups contribute zero.
The pipeline overlaps KV prefetch with QK/PV and uses separate startup and drain
paths. Output uses LDS staging and tensor stores. The producer and decoder use the same N16 AUX byte layout and fixed-gain
numerical contract.
The only supported log2 rescale threshold is 6; the parameter helper rejects
other values.
Its fixed PSF multiplier 5 folds in the KV gain. Arbitrary global gains or
unquantized probabilities are not supported by this code object.

## Minimal example

```python
import torch
import aiter

device = "cuda"
q = torch.randn((1, 128, 576), device=device, dtype=torch.bfloat16)
q_nope, q_aux = aiter.mla_nvfp4_quantize_q(q)
kv_nope = torch.empty((1, 16384), device=device, dtype=torch.uint8)
kv_aux = torch.empty((1, 8192), device=device, dtype=torch.uint8)

# A whole original BF16 token16 tile; here all 16 rows are valid.
tail = torch.randn((1, 16, 576), device=device, dtype=torch.bfloat16)
page_ids = torch.tensor([0], device=device, dtype=torch.int32)
tile_ids = torch.tensor([0], device=device, dtype=torch.int32)
valid = torch.tensor([16], device=device, dtype=torch.int32)
aiter.mla_nvfp4_update_cache(tail, page_ids, tile_ids, valid, kv_nope, kv_aux)

indptr = torch.tensor([0, 1], device=device, dtype=torch.int32)
indices = torch.tensor([0], device=device, dtype=torch.int32)
last = torch.tensor([16], device=device, dtype=torch.int32)
params = aiter.prepare_mla_nvfp4_params(
    device=device, physical_pages=1, sm_scale=576**-0.5
)
out = torch.empty((1, 128, 512), device=device, dtype=torch.bfloat16)
lse = torch.empty((1, 128), device=device, dtype=torch.float32)
out, lse = aiter.mla_decode_nvfp4_fwd(
    q_nope, q_aux, kv_nope, kv_aux, out, indptr, indices, last,
    params=params, lse=lse, return_lse=True,
)
```

Pass preallocated producer outputs and LSE for an allocation-free steady state.
The shader always writes LSE, including when `return_lse=False`; omitting `lse`
makes the wrapper allocate scratch. Empty requests return O=0 and LSE=-inf.
Keep at least one allocated physical page, including all-empty batches.

## Cache updates and graph replay

`mla_nvfp4_update_cache` receives BF16 `[T,16,576]` original tiles and int32
`page_ids`, `tile_ids`, `valid_tokens`, all `[T]`. Tile IDs are 0..3 within a
page, and valid lengths are 0..16. Invalid BF16 rows may be NaN. Each update
must name a distinct `(page_id,tile_id)`; concurrent writes to a tile are invalid.

The caller retains the BF16 active tail. Appending an outlier changes a shared
scale and therefore requires requantizing earlier tokens in the same token16
group from their original values. Do not rebuild the tail by dequantizing old
FP4. Completed groups need no repeated work. The writer directly updates the
native bricks and needs no global V copy or full-history shuffle.

The caller owns page allocation, tail retention, shared-prefix copy-on-write
and metadata publication. Publish lengths/CSR after the tile write on the same
stream, or establish an event dependency across streams. Shape checks do not
read GPU metadata back: valid CSR, page IDs, params capacity and live finite
scales remain caller contracts. Per-request KV length must fit signed int32.

The cache writer only emits N16 AUX. Its `aux_layout` keyword and native layout
flag have been removed. Existing calls should omit the keyword, and old N32 AUX
caches must be rebuilt from original BF16 data. The layouts are incompatible
despite having the same tensor shape. Q and NoPE layouts are unchanged.
The ctypes module is `module_mla_nvfp4_v5`, so existing v1/v2/v3/v4 JIT binaries
cannot reuse an obsolete argument list, CO name or directory. No manual cache
deletion is needed.

Prepare immutable params and warm up all selected operators before graph
capture. The current stream is selected from the input tensor's device.
Capture can include producers, metadata conversion and decode. Update the
contents of caller-owned buffers before replay; preserve their addresses and
lifetimes. Recreate params if physical cache capacity or softmax scale changes.

Dense lengths/page tables can use the existing GPU adapter:

```python
aiter.mla.mla_decode_prepare_page_csr(
    seq_lens, block_tables, indptr, indices, last
)
```

The adapter requires `indices` capacity at least `B * block_tables.shape[1]`.
It converts metadata only, not KV bytes.

## Unified MLA entry

The same backend is available through `aiter.mla.mla_decode_fwd` using
`kv_layout="nvfp4_n128"`, `q=q_nope`, `kv_buffer=kv_nope`, `q_aux=q_aux`,
`kv_aux=kv_aux`, `nvfp4_params=params` and `nvfp4_lse=lse`.
Use page CSR, `page_size=64`, `max_seqlen_q=1`, `nhead_kv=1` and
`num_kv_splits=None` or 1. `qo_indptr` may be None; if supplied it must contain
`[0,1,...,B]`. The result is `(out, lse_or_None)`.

Leave `sm_scale`, `q_scale` and `kv_scale` unset on the unified call: they are
already encoded in params. Planner metadata, dense metadata, CP, partial
logits and incompatible shapes raise errors. Unsupported NVFP4 inputs cannot
fall back to FP8 kernels without an explicit conversion.

## Build and verification

The shipped `mla_a4w4_qh128_1tg_32mx4_64nx1_r1_np.co` has an
80-byte, ten-pointer ABI, fixed LDS of 262144 bytes and a 16KiB aligned
instruction entry. Dynamic LDS is zero.
It contains 94568 instruction bytes, uses 1024 VGPRs / 106 SGPRs and retains
SPI prefetch 255. Its public symbol is `mla_a4w4_qh128_1tg_32mx4_64nx1_r1_np`.
Both HIP producer kernels also request 16KiB instruction alignment.
The kernel object/descriptor itself is a different address from the entry.

The precompiled ASM CO is in the shared `hsa/gfx1250/mla/` directory.
AITER loads the shipped CO and compiles the C++ wrapper and HIP producer
kernels when needed. ASM source and offline build tooling are maintained
separately and are not distributed with AITER.

```bash
python3 -m pytest op_tests/test_mla_nvfp4.py -q
AITER_NVFP4_TEST_LARGE=1 python3 -m pytest op_tests/test_mla_nvfp4.py -q
python3 -m pytest op_tests/test_mla_decode_pagesize64.py -k Nvfp4 -q
python3 op_tests/test_mla_decode_pagesize64.py --kv-layout nvfp4_n128 -b 5120 6144 7168 -c 5120 8192 10240
```

The page64 tests start from BF16 Q/KV, call both GPU producers, and check the
unified decode entry against full FP32 attention on those BF16 inputs. Output
and finite LSE use `atol=rtol=0.3` with the page64 benchmark's existing 5%
element-mismatch budget. A per-request normalized RMS output error limit of
0.3 also rejects zero outputs for long histories. These are
end-to-end quantization tolerances, separate from the stricter quantized
oracle checks in `test_mla_nvfp4.py`. Invalid tail tokens are NaN; empty requests
must return exact zero output and negative-infinity LSE.

The CLI reports mean decode time from 30 HIP-event samples of a 32-dispatch
graph, with a same-format warm graph before each sample. It excludes
quantization, cache writes, reference checks and transfers.

Add `--compare-fp8` to pack the same original BF16 Q/KV and physical page mapping
for the shipped FP8 QH128 shuffled-page64 kernel as well. The paired run
alternates NVFP4/FP8 and FP8/NVFP4 order, retains all 30 samples per format,
and checks that graph replay is bitwise equal to eager output/LSE. Both formats
are checked against the same original-BF16 FP32 reference: NVFP4 uses tolerance
0.3 and FP8 uses 0.06, with the 5% element-mismatch budget and a per-request NRMS
bound equal to the respective tolerance. The table includes both mean times,
`fp8 / nvfp4` time ratio and each format's error metrics.

```bash
python3 op_tests/test_mla_decode_pagesize64.py --kv-layout nvfp4_n128 --compare-fp8 -b 5120 6144 7168 -c 5120 8192 10240
```

FP8 uses unit Q/KV scales and the existing `mla_decode_fwd_qh128_asm` API with
preallocated output/LSE; NVFP4 uses the unified MLA API with fixed gains and B6.
Both launch their shipped ASM decode CO. The default FP8 CLI behavior is unchanged.

Tests cover an independent NumPy N128 oracle, all four score contributions,
per-token RoPE scales, cold rescaling, empty/tail/ring boundaries, native/public
and unified dispatch, repeatability, guards, producer byte layouts, active-tail
requantization, BF16-to-decode, dense-to-CSR, stream/graph and fullgraph compile.
The optional test allocates approximately 12GiB to put both KV streams above
4GiB offsets. Full-model accuracy and shape-specific performance tuning remain
separate from this first integration.

### Paired NVFP4 / FP8 benchmark (2026-11-19)

QH128, one query, page64, no split and threshold 6. Each pair uses the same
random BF16 Q/KV and physical pages. Times are full decode dispatch means in
microseconds under the paired graph32/30-sample protocol above. FP8 and NVFP4
use different quantized representations; the ratio is not an equal-precision
kernel comparison. The FP8 CO is
`hsa/gfx1250/mla/mla_a8w8_qh128_1tg_32mx4_64nx1_ps64.co`
(SHA256 `ec9bcb29971bf18d9c347afbfebebac47170165438a4be74d4c8f27f623df85c`).

The measurements use the N16-only producer; packing is outside timing.
The original decoder instruction stream is preserved by the final CO rename.

**a07-3, gfx1250 B0 — complete rerun**

| Batch | KV tokens | NVFP4 us | FP8 us | FP8 / NVFP4 |
|---:|---:|---:|---:|---:|
| 5120 | 5120 | 1299.77 | 1635.09 | 1.2580x |
| 5120 | 8192 | 2037.16 | 2564.09 | 1.2587x |
| 5120 | 10240 | 2544.24 | 3191.48 | 1.2544x |
| 6144 | 5120 | 1561.91 | 1967.70 | 1.2598x |
| 6144 | 8192 | 2461.99 | 3082.30 | 1.2520x |
| 6144 | 10240 | 3065.00 | 3837.38 | 1.2520x |
| 7168 | 5120 | 1825.46 | 2295.34 | 1.2574x |
| 7168 | 8192 | 2879.88 | 3605.19 | 1.2519x |
| 7168 | 10240 | 3582.31 | 4497.11 | 1.2554x |

All 18 completed pairs passed output, LSE, per-request NRMS, guard and
eager/graph checks. At each case's maximum request NRMS, NVFP4 spans
0.117564–0.133525 and FP8 spans 0.0322255–0.0360402 against the same
original-BF16 FP32 reference. Both formats had zero element mismatches at
their respective tolerances; this does not imply equal accuracy.

The first a07-3 sweep stopped at B7168/KV10240 with a GPU memory fault
reported in a PyTorch reference GEMM. An isolated maximum case and this
fresh complete sweep then passed with the same code, image and BLAS backend.
The failed sweep is excluded from the table; its root cause remains unverified.

Measurements used the NVFP4 CO before the final file/symbol rename. The complete
`.text` section is byte-identical after renaming, and the renamed v4 path passed
GPU regression tests on d07-2. No performance change is attributed to the rename.

### Accuracy relative to PR #7987

The requested historical baseline is [#7987](https://github.com/ROCm/aiter/pull/7987),
commit `25b4d9b5c51ad444f399989b6ae6658c6b25b5a1`. Its original NVFP4 CO has SHA256
`6b856a0577ed30b3e24eaf1ad7d48ad59fe2ceca6f4b7304cb469747ff7044f1` and uses N32 AUX.
The comparison ran its original checkout/producers/decoder on d07-2, using the
same BF16 generator, seeds, nine shapes and FP32 reference as the current NVFP4 sweep.
Both use threshold **6** for this matched comparison; #7987's original default
was 1. The runtime-loaded baseline instruction bytes and entry alignment were
also verified.

| Batch | KV tokens | #7987 max request NRMS | Current NVFP4 max request NRMS |
|---:|---:|---:|---:|
| 5120 | 5120 | 0.132764 | 0.132764 |
| 5120 | 8192 | 0.123605 | 0.123605 |
| 5120 | 10240 | 0.117564 | 0.117564 |
| 6144 | 5120 | 0.133525 | 0.133525 |
| 6144 | 8192 | 0.122352 | 0.122352 |
| 6144 | 10240 | 0.117959 | 0.117959 |
| 7168 | 5120 | 0.132510 | 0.132510 |
| 7168 | 8192 | 0.122234 | 0.122234 |
| 7168 | 10240 | 0.117759 | 0.117759 |

Every case's maximum request NRMS agrees to six decimal places. Both versions
pass the output/LSE checks, with a largest request NRMS of 0.133525. No accuracy
regression was observed on these cases. This is a metric-level comparison;
complete old/new output tensors were not compared bit for bit. Full-model
accuracy and other input distributions remain unverified. The separate FP8
comparison above uses AITER's original N64-tile FP8 kernel, not the #7987 NVFP4 CO.

### Current validation (2026-11-19)

The final public test suite passed **43 selected tests on d07-2**, with no skips,
using the restored maximum absolute output error of 0.025.

The N16-only implementation was validated on gfx1250 B0 hardware on d07-2 and
a07-3 with the same pinned ROCm/PyTorch image. The quantized-oracle checks use
NRMS 0.003, maximum absolute output error 0.025 and LSE absolute error 2e-4.

Coverage includes the independent quantized oracle, all score contributions,
per-token RoPE scales, threshold 6 and rejection of other values, every tail
after the four-slot wrap, empty rows, guards, byte-exact BF16 producer layouts,
active-tail requantization, graph replay, fullgraph compile, KV addresses above
4GiB and existing FP8 QH128 behavior. The obsolete N32 writer branch is removed;
its earlier layout-selection keyword is explicitly rejected.

The symbol is `mla_a4w4_qh128_1tg_32mx4_64nx1_r1_np`; it uses N128 tiles,
N16 AUX and an 80-byte ABI. Moving the CO into the shared MLA directory
preserves its instruction stream. N16 producer quantization arithmetic and
address formulas are unchanged by the cleanup.

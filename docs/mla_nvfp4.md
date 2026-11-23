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
| `lse` | optional float32 `[B,128]` | Reserved compatibility buffer, never written |
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

Input gains are fixed: `G_Q=6.72`, `G_KV=89.6`. Each reconstructed value
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

The decoder uses N128 tiles and local N16 probability quantization. Each
N16 group computes a local log2 maximum `b` and
`e = exp2(fma(raw_score, beta, -b))`. With the common N128 online anchor `a`,
`w = exp2(b - a)` and `P = FP4(6*e) * FP8(5*w)`. Both the PV numerator `R`
and denominator `Lq` use this **same quantized probability mass**. Fully
masked groups contribute zero. When the anchor changes, both accumulated
`R` and `Lq` are rescaled by the same FP32 `alpha = exp2(old_a - new_a)`.
The final output is `R / Lq / 89.6`.

The supported log2 rescale threshold is **1 (T1)**. The parameter helper rejects
other values. Its four words contain FP32 bits for
`beta = sm_scale * log2(e) / (6.72 * 89.6)`, **`1/89.6`**, and `1.0`, followed by
the integer physical page capacity. Recreate old parameter buffers: the previous
`1/2688` multiplier and threshold 6 are incompatible with this integration.
The fixed probability factors 6 and 5 cancel between numerator and denominator;
the output multiplier removes only the KV gain.

The decoder does not produce a natural-log LSE. The public wrappers reject
`return_lse=True`; the ABI retains its LSE pointer slot without accessing it.
The pipeline overlaps KV prefetch with QK/PV, with separate startup/drain paths
and LDS-staged tensor output stores.

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
out, _ = aiter.mla_decode_nvfp4_fwd(
    q_nope, q_aux, kv_nope, kv_aux, out, indptr, indices, last,
    params=params,
)
```

Pass preallocated producer outputs for an allocation-free steady state.
Decode allocates no LSE scratch. An optional legacy `lse` buffer remains
unchanged. Empty requests return exact zero output.
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
The ctypes module is `module_mla_nvfp4_v6`, so older JIT binaries cannot reuse
an obsolete argument list or the wrapper that requires an LSE buffer. No manual cache
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
`kv_aux=kv_aux` and `nvfp4_params=params`. Leave `return_lse=False`;
`nvfp4_lse` is optional and remains untouched.
Use page CSR, `page_size=64`, `max_seqlen_q=1`, `nhead_kv=1` and
`num_kv_splits=None` or 1. `qo_indptr` may be None; if supplied it must contain
`[0,1,...,B]`. The result is `(out, None)`.

Leave `sm_scale`, `q_scale` and `kv_scale` unset on the unified call: they are
already encoded in params. Planner metadata, dense metadata, CP, partial
logits and incompatible shapes raise errors. Unsupported NVFP4 inputs cannot
fall back to FP8 kernels without an explicit conversion.

## Build and verification

The shipped `mla_a4w4_qh128_1tg_32mx4_64nx1_r1_np.co` has an
80-byte, ten-pointer ABI, fixed LDS of 262144 bytes and a 16KiB aligned
instruction entry. Dynamic LDS is zero.
It contains 73484 instruction bytes, uses 1024 VGPRs / 106 SGPRs and retains
SPI prefetch 255. Its public symbol is `mla_a4w4_qh128_1tg_32mx4_64nx1_r1_np`.
Both HIP producer kernels also request 16KiB instruction alignment.
The kernel object/descriptor itself is a different address from the entry.

The precompiled ASM CO is in the shared `hsa/gfx1250/mla/` directory.
AITER loads the shipped CO and compiles the C++ wrapper and HIP producer
kernels when needed. ASM source and offline build tooling are maintained
separately and are not distributed with AITER.

```bash
python3 -m pytest op_tests/test_mla_decode_pagesize64.py -k Nvfp4 -q
AITER_NVFP4_TEST_LARGE=1 python3 -m pytest op_tests/test_mla_nvfp4.py -q
python3 op_tests/test_mla_decode_pagesize64.py --kv-layout nvfp4_n128 -b 5120 6144 7168 -c 5120 8192 10240
```

The **primary operator accuracy reference** is the page64 suite's standard
high-level FP32 `softmax(Q @ K.T * sm_scale) @ V` on the **original BF16 Q/KV**.
It calls both GPU producers and the unified decode entry, and includes input
and probability quantization error. It does not reproduce the kernel's N128
loop, T1 rebasing, FP4 softmax probabilities or quantized denominator.
The existing NVFP4 limits remain `atol=rtol=0.3`, a 5% element-mismatch budget,
and per-request normalized RMS output error at most `0.3`. Invalid tails are
NaN and empty requests must return exact zero output. These operator tolerances
alone do not prove that model accuracy is unchanged.

`test_mla_nvfp4.py` additionally checks the specified quantized algorithm using
logical NumPy matrices. Its maximum absolute output error remains `0.025` and
normalized RMS error `0.003`, without an element-mismatch allowance. This is
an implementation regression check, **not a model-accuracy reference**. The
constant-V test checks exact BF16 preservation under normalization, and legacy
LSE buffers must remain untouched. Full-model acceptance uses the complete
MLPerf evaluations below.

The CLI reports mean decode time from 30 HIP-event samples of a 32-dispatch
graph, with a same-format warm graph before each sample. It excludes
quantization, cache writes, reference checks and transfers.

Add `--compare-fp8` to pack the same original BF16 Q/KV and physical page mapping
for the shipped FP8 QH128 shuffled-page64 kernel as well. The paired run
alternates NVFP4/FP8 and FP8/NVFP4 order, retains all 30 samples per format,
and checks that graph replay is bitwise equal to eager output (and FP8 LSE). Both formats
are checked against the same original-BF16 FP32 reference: NVFP4 uses tolerance
0.3 and FP8 uses 0.06, with the 5% element-mismatch budget and a per-request NRMS
bound equal to the respective tolerance. The table includes both mean times,
`fp8 / nvfp4` time ratio and each format's error metrics.

```bash
python3 op_tests/test_mla_decode_pagesize64.py --kv-layout nvfp4_n128 --compare-fp8 -b 5120 6144 7168 -c 5120 8192 10240
```

FP8 uses unit Q/KV scales and the existing `mla_decode_fwd_qh128_asm` API with
preallocated output/LSE; NVFP4 uses the unified MLA API with fixed gains and T1.
Both launch their shipped ASM decode CO. The default FP8 CLI behavior is unchanged.

Tests cover an independent NumPy N128 oracle, all four score contributions,
per-token RoPE scales, cold rescaling, empty/tail/ring boundaries, native/public
and unified dispatch, repeatability, guards, producer byte layouts, active-tail
requantization, BF16-to-decode, dense-to-CSR, stream/graph and fullgraph compile.
The optional test allocates approximately 12GiB to put both KV streams above
4GiB offsets. The validation and measurements below describe the current decoder;
the earlier full-model result is identified separately.

### Public integration validation (2026-11-23)

On gfx1250 B0 hardware, **44 selected GPU tests passed, with no skips** using a
fresh v6 JIT build. Coverage includes the standard BF16-input FP32 attention
reference, supplemental quantized-algorithm checks, constant V, guards, producer
layouts, graph/fullgraph paths, KV offsets beyond 4 GiB and FP8 QH128 regression.
Existing test tolerances are unchanged. The actual loaded decoder's 73,484
instruction bytes match the version measured below; decoder and both HIP
producer entries are 16 KiB aligned.

The current decoder also passed 1,042 randomized/constant-input requests around
short-sequence, tail and loop boundaries against standard FP32 attention on the
original BF16 inputs. All 22,528 requests in the performance cases below matched
the preceding published decoder bitwise. Guards, empty outputs, retained LSE
buffers and graph/eager consistency passed. These operator checks do not replace
a full-model evaluation of this revision.

### Full-model accuracy history

The preceding published decoder, at commit
`fbe834626cf7ee91ff07ccdcdb05cf43c14b5960`, completed one DeepSeek-R1 MLPerf
accuracy evaluation using N16 producers, threshold 1, output multiplier `1/89.6`
and one KV split:

| Correct | Exact match | Tokens/sample | Result |
|---:|---:|---:|---|
| 3541 / 4388 | 80.697356% | 3931.095260 | PASS |

The supplied gate requires exact match >=80.545% and 3497.6–4274.9 tokens/sample.
This is historical evidence for the preceding decoder. The current revision has
not had a new full-model evaluation; its bitwise operator checks above are not
a claim of a new MLPerf score, repeated-run accuracy stability or model-throughput
improvement.

### Decode timing (2026-11-22)

Single-GPU measurements on gfx1250 B0 used the current decoder instructions,
threshold 1, QH128 and page size 64. Both formats shared original BF16 Q/KV,
request lengths and randomized physical page mappings, with their respective
native shuffled layouts. NVFP4 produced output only; the existing FP8 API also
produced LSE.

Each row contains two measurement blocks with opposite initialization orders.
Each block uses 8 eager warmups, 32 dispatches per graph, 5 initial graph replays,
and 30 alternating NVFP4/FP8 and FP8/NVFP4 rounds. A same-format warm graph
precedes every sample. The table retains all 60 samples per format and reports
the arithmetic mean in microseconds per dispatch. Input preparation, quantization,
cache writes and reference checks are outside the timed region.

The first nine rows use a pool of **199,945 physical pages**. The target mean is
not necessarily the actual mean after clipping:

| Batch | Target mean KV | Length distribution | NVFP4 mean (us) | FP8 mean (us) | FP8 / NVFP4 | NVFP4 faster rounds |
|---:|---:|---|---:|---:|---:|---:|
| 1536 | 2500 | Uniform | 240.100 | 247.501 | 1.0308x | 59/60 |
| 1536 | 2500 | Lognormal | 285.748 | 280.372 | 0.9812x | 1/60 |
| 1536 | 2500 | Few long | 242.439 | 254.359 | 1.0492x | 60/60 |
| 1536 | 4000 | Uniform | 353.809 | 386.797 | 1.0932x | 59/60 |
| 1536 | 4000 | Lognormal | 427.594 | 428.989 | 1.0033x | 38/60 |
| 1536 | 4000 | Few long | 306.189 | 324.113 | 1.0585x | 60/60 |
| 1024 | 4000 | Uniform | 240.659 | 251.416 | 1.0447x | 60/60 |
| 1024 | 4000 | Lognormal | 293.424 | 287.514 | 0.9799x | 2/60 |
| 1024 | 4000 | Few long | 222.339 | 232.322 | 1.0449x | 60/60 |

Uniform means every request has exactly the target length. Lognormal lengths
use NumPy `default_rng(0)`, log-space sigma 0.8, rescaling to the target mean,
clipping to [200, 20000], and integer conversion; the three shapes are generated
in the order shown. Their actual means are 2498.698, 3970.786 and 3955.996.
For the few-long cases, `floor(batch/32)` requests receive the remaining target
token budget after the others receive half the target mean; lengths are clipped
to [200, 30000]. Their actual means are 2148.438, 2875 and 2875. Different
distributions therefore do not have identical total token counts; compare the
two formats within each row.

Seven of these nine row means favor NVFP4. The 1.0033x result is close to parity,
and FP8 remains faster in two lognormal cases. Performance is workload-dependent.
The two blocks are from the same run, rather than independent repeated runs.

The larger uniform workloads use **600,000 physical pages**, accommodating
409,600 and 491,520 independent KV pages respectively:

| Batch | Fixed KV | NVFP4 mean (us) | FP8 mean (us) | FP8 / NVFP4 | NVFP4 time reduction | NVFP4 faster rounds |
|---:|---:|---:|---:|---:|---:|---:|
| 5120 | 5120 | 1390.817 | 1692.650 | 1.2170x | 17.83% | 60/60 |
| 5120 | 6144 | 1643.160 | 2007.923 | 1.2220x | 18.17% | 60/60 |

Both initialization-order blocks favor NVFP4 for both large workloads. Pool size
changes between the two tables, so absolute timing differences cannot be
attributed solely to batch size. These are decode-kernel comparisons with FP8,
not a direct paired comparison with the preceding NVFP4 binary, and they do not
establish end-to-end model-throughput gains.

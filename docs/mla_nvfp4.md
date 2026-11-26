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
The ctypes module is `module_mla_nvfp4_v7`. This version loads the N128-named CO
and prevents reuse of cached v6 wrappers that load the old N64-named file.
Older argument lists and wrappers that require an LSE buffer also cannot be
reused. No manual cache deletion is needed.

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

## Fused decode prep

`mla_nvfp4_decode_prep` does a decode step's Q RoPE quantization and KV append
in one HIP launch (its own module, `module_mla_nvfp4_decode_prep`; no ASM code
object). For B tokens, one per request:

```python
aiter.mla_nvfp4_decode_prep(
    q_rope, q_aux, ring, req_ids, k_nope, k_rope, positions, slots,
    kv_nope, kv_aux, cos_cache, sin_cache,
)
```

- `q_aux [B,128,128]` gets bytes 0..63 and 96..127 exactly as
  `mla_nvfp4_quantize_q` writes them. Bytes 64..95, the NoPE scales, are left
  as written, so the NoPE producer may run before or after.
- `ring [R,16,576]` is the caller's BF16 tail: one 16-token tile per request
  (512 NoPE channels, then 64 rotated RoPE channels). The token's row
  `positions % 16` of `ring[req_ids]` is set, and rows 0..that row are
  requantized into page `slots // 64`, tile `positions % 64 // 16`, exactly as
  `mla_nvfp4_update_cache` writes that tile. A tile's first token also writes
  the zero codes and 0x7F scales of its later rows.
- With BF16 `cos_cache`/`sin_cache` (GPT-J pairs), `q_rope [B,128,64]` and
  `k_rope [B,64]` are rotated at `positions`, rounding as aiter's Triton RoPE
  (`forward_triton`) does; without them they are already rotated.
- Slot -1 marks a padding row: zero Q codes, 0x7F scales, no KV write. Tokens
  whose page or request id is outside the cache or the ring write no KV either.

`req_ids` is int32, `positions` and `slots` are int64, all `[B]`. `q_rope` rows
must be 128 head strides apart: contiguous, or the RoPE slice of a
`[B,128,D]` Q. A batch must not repeat a request. The written tensors
(`q_aux`, the ring, both cache planes) must not overlap each other or the
inputs; a ring whose rows overlap is rejected. The caller owns tail
recycling: a tile's first token overwrites row 0. The op is graph-safe; warm
it up before capture.

For fused RoPE at B >= 1024, each 384-thread block handles one request: four Q
waves each load 32 heads through TDM and process four groups of eight heads,
while eight K waves handle the tail tile. Q and K use disjoint LDS and output
regions, and K RoPE processing overlaps the historical NoPE tile load. Batch
size is a runtime argument: non-aligned sizes such as 1537, 2049 and 4097 use
the same kernel. A fixed-stride DSR1 entry checks all specialized strides
before dispatch and uses a compact 116-byte kernel argument list for gfx1250
kernarg preload. Other layouts at B >= 1024 use the generic-stride joint
entry; smaller batches and the already-rotated path retain the 256-thread
implementation. A captured graph keeps its launch batch size; changes to the
live request count use the existing negative-slot padding convention.

For the fixed-stride entry at B >= 2048, Q RoPE keeps cos/sin packed and uses
BF16 instruction operand selectors for coefficient broadcast and GPT-J rotation.
The BF16 multiply/FMA rounding is unchanged. This lowers VGPR allocation from
76 to 60 without changing LDS or kernel arguments; padding writes are unrolled
in this specialization. Smaller batches and other strides retain their previous
scheduling; K and quantization arithmetic are unchanged.

On a four-gfx1250 DeepSeek-R1 MLPerf trace using ATOM 1117 dev10 and the
`gfx1250-atom-20261117-dsr1` image, logical rank dp0's complete decode-prep
kernel medians were 10.494, 10.815 and 10.895 us in the 300/600/900-second
windows. Actual 61-layer per-graph sums had medians of 0.647, 0.665 and
0.670 ms. These are model trace observations, not a bandwidth benchmark.
They predate the runtime-batch extension and packed Q RoPE optimization.

On 2026-11-24, full 4388-sample model accuracy runs passed with the latest
K-up and decode-prep optimizations, attention #8200 at b3bef9970, and the
same image:

| Maximum graph batch | Exact match | Tokens/sample |
| ---: | ---: | ---: |
| 1536 | 80.674567% | 3927.594348 |
| 2048 | 80.674567% | 3927.594348 |

All four ranks replayed each run's largest graph bucket. At the first
observed replay, active requests ranged from 1026 to 1091 per rank for the
1536 run, and from 1042 to 1091 for the 2048 run. The other rows were padding;
these runs do not establish full live-request occupancy at those batch sizes.
Op tests cover full batches and changing padding. No new model trace was run.

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

The shipped `mla_a4w4_qh128_1tg_32mx4_128nx1_r1_np.co` has an
80-byte, ten-pointer ABI, fixed LDS of 262144 bytes and a 16KiB aligned
instruction entry. Dynamic LDS is zero.
It contains 64768 instruction bytes, uses 1024 VGPRs / 106 SGPRs and retains
SPI prefetch 255. Its public symbol is `mla_a4w4_qh128_1tg_32mx4_128nx1_r1_np`.
Both HIP producer kernels also request 16KiB instruction alignment.
The kernel object/descriptor itself is a different address from the entry.
The decoder keeps output completion inline, compacts helper paths and uses
bounded instruction prefetch. Its entry schedule overlaps query completion
and scale-loading work. Full first N128 tiles retain the mask fast path; partial
first tiles are masked before the softmax maximum reduction. These changes
preserve the T1 numerical contract and public ABI.

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
4GiB offsets. Current public integration validation and matched FP8 decode
measurements are reported below; full-model results retain their historical
revision and validation scope.

### Public integration validation (2026-11-24)

On gfx1250 B0 hardware, **44 selected GPU tests passed, with no skips** using a
fresh v7 JIT build and the refreshed public CO:

```bash
AITER_NVFP4_TEST_LARGE=1 python3 -m pytest op_tests/test_mla_nvfp4.py op_tests/test_mla_decode_pagesize64.py -k 'Nvfp4 or nvfp4 or TestMlaQh128'
```

Coverage includes standard FP32 attention on original BF16 inputs, the existing
strict quantized-algorithm oracle, constant V, guards, producer layouts,
graph/fullgraph execution, KV offsets beyond 4 GiB and FP8 QH128 regression.
The tests and their accuracy limits are unchanged. All 64,768 loaded decoder
instruction bytes match the refreshed CO, and the decoder and both HIP producer
entries are 16 KiB aligned. The public N128 filename, symbol, ten-pointer ABI and
v7 loader interface remain compatible; this refresh requires no module rename.

The two current FP8 benchmark runs below cover 43,008 requests whose complete
NVFP4 outputs match the validated T1 baseline bitwise. Graph/eager equality,
finite values, output guards and the untouched reserved LSE buffer passed.
The first, middle and last request in each row were additionally compared with
standard FP32 attention on the original BF16 inputs: 45 requests per format,
90 checks in total. Maximum NRMS was 0.179018 for NVFP4 and 0.047535 for FP8;
no checked elements exceeded their existing tolerance budgets. These operator
checks do not replace a full-model evaluation of this revision.

### Full-model accuracy history

An earlier published decoder, at commit
`fbe834626cf7ee91ff07ccdcdb05cf43c14b5960`, completed one DeepSeek-R1 MLPerf
accuracy evaluation using N16 producers, threshold 1, output multiplier `1/89.6`
and one KV split:

| Correct | Exact match | Tokens/sample | Result |
|---:|---:|---:|---|
| 3541 / 4388 | 80.697356% | 3931.095260 | PASS |

The supplied gate requires exact match >=80.545% and 3497.6–4274.9 tokens/sample.
This is historical evidence for that earlier decoder. The current revision has
not had a new full-model evaluation; its bitwise operator checks above are not
a claim of a new MLPerf score, repeated-run accuracy stability or model-throughput
improvement.

### Decode timing (2026-11-24)

The following measurements use the instruction image shipped in the current
CO on a single gfx1250 B0 GPU, with threshold 1, QH128 and page size 64.
Both formats share original BF16 Q/KV, request lengths and random physical page
mapping, using their respective native packed layouts. NVFP4 produces output
only; the existing FP8 API also produces LSE.

Each row has two blocks with opposite initialization orders. Each block uses
8 eager warmups, 32 dispatches per graph, 5 initial graph replays, and 30 AB/BA
rounds with a same-format warm graph before each timed sample. All 60 samples
per format are retained; the tables show arithmetic means in microseconds per
decode dispatch. Preparation, quantization, cache writes and correctness checks
are outside timing. The two blocks belong to one run, rather than independent
repeated runs. GPU occupancy sampling observed no concurrent task during timing.

Physical page pool: **199,945 pages**.

| Batch | Target mean KV | Distribution | NVFP4 mean (us) | FP8 mean (us) | FP8 / NVFP4 | NVFP4 faster rounds |
|---:|---:|---|---:|---:|---:|---:|
| 1536 | 2500 | Uniform | 250.680 | 262.032 | 1.0453x | 58/60 |
| 1536 | 2500 | Lognormal | 302.718 | 314.822 | 1.0400x | 60/60 |
| 1536 | 2500 | Few long | 269.034 | 284.654 | 1.0581x | 60/60 |
| 1536 | 4000 | Uniform | 370.483 | 406.853 | 1.0982x | 59/60 |
| 1536 | 4000 | Lognormal | 435.527 | 479.816 | 1.1017x | 60/60 |
| 1536 | 4000 | Few long | 308.745 | 336.980 | 1.0914x | 60/60 |
| 1024 | 4000 | Uniform | 252.862 | 264.841 | 1.0474x | 60/60 |
| 1024 | 4000 | Lognormal | 319.833 | 329.296 | 1.0296x | 60/60 |
| 1024 | 4000 | Few long | 233.473 | 247.047 | 1.0581x | 60/60 |

Physical page pool: **322,560 pages**.

| Batch | Target mean KV | Distribution | NVFP4 mean (us) | FP8 mean (us) | FP8 / NVFP4 | NVFP4 faster rounds |
|---:|---:|---|---:|---:|---:|---:|
| 5120 | 2500 | Uniform | 792.386 | 938.756 | 1.1847x | 60/60 |
| 5120 | 2500 | Lognormal | 860.328 | 980.381 | 1.1395x | 60/60 |
| 5120 | 2500 | Few long | 698.368 | 819.397 | 1.1733x | 60/60 |
| 5120 | 4000 | Uniform | 1175.578 | 1438.787 | 1.2239x | 60/60 |
| 5120 | 4000 | Lognormal | 1258.903 | 1487.265 | 1.1814x | 60/60 |
| 5120 | 4000 | Few long | 882.745 | 1066.741 | 1.2084x | 60/60 |

The batch-5120 uniform KV=4000 case requires exactly 322,560 pages and fills
its physical pool. The original 199,945-page pool cannot hold this workload.
Other cases use fewer pages than their pool capacity. Pool size differs between
the tables, so cross-table timing changes cannot be attributed solely to batch.

Uniform gives every request the target length. Lognormal uses NumPy
`default_rng(0)`, log-space sigma 0.8, rescaling to the target mean, clipping to
[200, 20000], then integer conversion. Each table starts its own generator;
shapes are generated in the order shown. Actual lognormal means are 2498.698,
3970.786 and 3955.996 in the first table, and 2498.390 and 3957.895 in the second.
For few-long cases, `floor(batch/32)` requests receive the remaining target
token budget after other requests receive half the target mean; lengths are
clipped to [200, 30000]. Their actual means are 2148.438 for target 2500 and
2875 for target 4000. Different distributions therefore have different token
totals; compare the formats within each row.

All 15 row means favor NVFP4, with FP8/NVFP4 ratios of 1.0296x to 1.2239x.
For batch 5120, NVFP4 reduces decode time by 12.25% to 18.29% and wins all
60 paired rounds in every row. Both initialization-order blocks favor NVFP4
for every row. Performance remains workload-dependent.

The public-symbol CO preserves the complete measured `.text` and kernel
descriptor byte for byte; repackaging was not timed separately. Its SHA256 is
`21881918157398e443aa3041953a9969f56d676e4eb2e7cfc4071388aa5f5063`.
These compare current NVFP4 with FP8, rather than directly measuring the
incremental gain over the preceding NVFP4 CO. They do not establish an
end-to-end model-throughput improvement.

# MHA V4 Entrypoint And FMHA V4 Engine

> Engineering reference for contributors. Keep current contracts here; preserve detailed history
> only where it explains an ABI, correctness constraint, or measured performance decision.

## Current Status

Dense BF16-output MHA v4 is implemented and validated on gfx950. Sorted block-sparse dispatch
(mask/LUT APIs, `mode=1` manifest rows) is wired on the same family; sparse `.co` files are
deployed next to the dense objects. Gfx942 native FP8/FP8 and signed INT8/FP8 have both dense
and sorted-sparse rows under v4 (256×64 tiles).

The public raw and packed APIs support eight dense combinations:

| Q/K | V | Output |
|---|---|---|
| BF16 | BF16 | BF16 |
| INT8 | FP8 | BF16 |
| FP8 | FP8 | BF16 |
| MXFP8 | FP8 | BF16 |
| MXFP6 E2M3 | FP8 | BF16 |
| MXFP4 E2M1 | FP8 | BF16 |
| MXFP6 E2M3 | MXFP4 E2M1 | BF16 |
| MXFP4 E2M1 | MXFP4 E2M1 | BF16 |

Current scope is batched, non-causal MHA with BF16 raw inputs, head dimension 128, and BF16
output. Dense and sorted block-sparse execution both support grouped-query head ratios; sparse
LUT rows are one per query head. Sparse ships on gfx950 (all eight packed recipes, 256×128)
and gfx942 (native FP8/FP8 and INT8/FP8, 256×64). It is inference-only: no backward,
dropout, RNG state, LSE, or varlen. Unsupported requests fail explicitly and never fall back
to `aiter.ops.mha`.

## Stable Decisions And Ownership

- `aiter.ops.mha_v4` owns mixed-precision preprocessing, packed-layout reconstruction, format and
    scale validation, and the raw/packed Python APIs. `aiter.ops.mha` and `fmha_v3_fwd` retain their
    generic ownership.
- `fmha_v4_fwd` is the internal JIT, launcher, manifest, and HSA family. V4 identifies an extensible
    dispatch and ABI generation, not a universal replacement for v3.
- Dispatch is explicit in Q/K/V formats and scale modes. Tensor dtype, packed width, stride, and
    storage size validate a selected row; they never select one.
- Format IDs are stable and distinguish encodings and integer signedness. `FP6_E2M3` is the active
    FP6 encoding (`MXFP6` alias); `FP6_E3M2` is reserved. Scale granularity remains a separate
    `AttentionScaleMode`, allowing MXFP8 or NVFP4-style recipes without inventing value formats.
- Q, K, and V preprocessing remain separate custom ops for distributed overlap. Exotic layouts
    cross custom-op boundaries as contiguous raw buffers and are rebuilt by MHA v4 view helpers in
    the final launch boundary.
- The public name is not Sage-branded because the supported combinations do not map exactly to one
    SageAttention version.
- Preserve `Optional[T]` annotations in entrypoints and fake implementations. `T | None` caused a
    measured Inductor regression in end-to-end model execution.

The current implementation is intentionally one module, `aiter/ops/mha_v4.py`; a speculative
subpackage split is not part of the design. It exports:

- `mha_v4`, `mha_v4_mxfp8`, and `mha_v4_packed`;
- `AttentionFormat`, `AttentionScaleMode`, `native_fp8_format`, `mha_v4_kv_tile`, and
  `scale_modes_for_formats`;
- canonical per-tensor, MX Q/K, and V quantizers;
- `mxfp4_k_view`, `mxfp6_k_view`, and `mxfp4_v_view` for raw-buffer reconstruction;
- `mha_v4_q_multiplier` for the MX Q scaling recipe.

## Authoritative References

- API and preprocessing ownership: `aiter/ops/mha_v4.py`.
- Host launcher: `csrc/py_itfs_cu/asm_mha_v4_fwd.cu`.
- Manifests and binaries: `hsa/<arch>/fmha_v4_fwd/`.
- Benchmark integration: `op_tests/op_benchmarks/triton/bench_sage.py`.

## Validated Baseline

Dense extraction, dedicated dispatch, six raw preprocessing paths, packed launch, benchmark
migration, and distributed integration are complete. Callers can delegate quantization, MX Q
scaling, scale recipes, and packed views to MHA v4 while retaining separate Q/K/V custom ops for
communication overlap.

Validation includes eager accuracy for all eight combinations, fullgraph eager/compiled parity,
finite outputs, allocator churn with downstream consumers, explicit code-object dispatch,
unaligned and unequal sequence lengths, retained model captures, and balanced multi-GPU target-shape
benchmarks. Focused coverage lives in `op_tests/test_mha_v4.py`.

Still deferred:

- VSA/Sparge compatibility adapters and 128x128 sparse tiles;
- low-precision output with an explicit data/scale ABI;
- additional BF16 kernel variants with distinct manifest identities;
- causal, varlen, other head dimensions, and more Q/K/V/O combinations;
- remaining gfx942 recipes (MX, BF16 sparse), plus CDNA5 and RDNA coverage.

## Current Dense Performance

Current gfx950 long-sequence dense ASM kernel throughput, excluding Q/K/V preprocessing:

| Q/K format | V format | Throughput (TFLOP/s) |
|---|---|---:|
| INT8 | FP8 | 2315 |
| FP8 | FP8 | 3050 |
| MXFP6 | FP8 | 3450 |
| MXFP6 | MXFP4 | 3700 |
| MXFP4 | FP8 | 3695 |
| MXFP4 | MXFP4 | 4000 |

These values are the current optimization baselines, not portable performance guarantees. Attach
the exact benchmark shape, harness revision, GPU count, and code-object hashes when promoting them
to release-facing documentation.

## Public API Levels

MHA v4 exposes raw and packed levels. Direct code-object launch remains private.

### Raw QKV API

This is the default application API:

```python
output = mha_v4(
    query,
    key,
    value,
    q_format=AttentionFormat.MXFP6,
    k_format=AttentionFormat.MXFP6,
    v_format=native_fp8_format(),
    softmax_scale=None,
    return_lse=False,
    out=None,
    block_mask=None,
)
```

Inputs are contiguous BF16 BSHD tensors. The requested formats select canonical per-operand
preprocessing and an explicit ASM row; unsupported combinations fail. Q/K must currently match.
Output is BF16, and a supplied `out` must match Q's shape/device. Q, K, and V preprocessing remain
separate custom ops so distributed schedulers can overlap each with its input communication.
The canonical FP8 Q/K recipe applies normalized hd128 Walsh-Hadamard rotation before per-tensor
quantization on both gfx942 and gfx950; V uses unrotated per-tensor FP8 quantization.
Optional `block_mask` is a boolean tile mask at the architecture's sparse geometry (256×128 on
gfx950, 256×64 on gfx942): `[B, H, Qtiles, KVtiles]` or `[B, Qtiles, KVtiles]` (broadcast across
heads). Use `mha_v4_kv_tile()` for the KV dimension. It is converted internally to a ragged LUT;
the host work table is not a Python argument.

#### Grouped-Query Attention

Both raw entrypoints (`mha_v4` and `mha_v4_mxfp8`) and `mha_v4_packed` accept GQA directly. Q uses
shape `[batch, query_length, query_heads, 128]`; K and V use
`[batch, key_value_length, kv_heads, 128]`. K and V must have the same head count, `query_heads`
must be divisible by `kv_heads`, and the ratio `query_heads / kv_heads` must be one of
`1, 2, 4, 8, 16`. Ratio 1 is ordinary multi-head attention. The kernel maps each contiguous group
of query heads to one K/V head; callers must not expand K or V to `query_heads`. Output retains Q's
batch, sequence, and head dimensions.

For example, Q with 32 heads and K/V with 8 heads selects GQA ratio 4. Q and K still use the same
number format and canonical quantization recipe; "Q/K formats must match" refers to their encoding,
not their head counts. Ratios outside the supported power-of-two set fail explicitly.

### Packed Expert API

This API supports benchmarks, distributed integrations, preprocessing reuse, and callers that
already own packed operands:

```python
output = mha_v4_packed(
    q=packed_query,
    k=packed_key,
    v=packed_value,
    q_descale=q_scale,
    k_descale=k_scale,
    v_descale=v_scale,
    q_format=AttentionFormat.MXFP6,
    k_format=AttentionFormat.MXFP6,
    v_format=native_fp8_format(),
    q_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
    k_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
    v_scale_mode=AttentionScaleMode.F32_PER_CHANNEL,
    softmax_scale=1.0,
    return_lse=False,
    out=None,
    kv_block_indices=None,
    lut_start=None,
    lut_count=None,
)
```

The packed API takes each operand's data, descale, format, and scale mode explicitly. It validates
the complete recipe plus dtype, shape, and layout before launching. Call
`scale_modes_for_formats()` for the production recipe rather than duplicating mode triples.
The optional LUT triple (`kv_block_indices`, `lut_start`, `lut_count`) must be all set or all
omitted; do not pass a dataclass and do not pass a mask to the packed API. Sparse launch uses
manifest `mode=1`; the work table is built inside the sparse custom op.

MX Q/K/V producers return contiguous raw buffers where the ASM layout is not an ordinary tensor
layout. `mxfp4_k_view`, `mxfp6_k_view`, and `mxfp4_v_view` reconstruct logical views. Raw buffers,
not exotic strided views, cross custom-op boundaries; final launch ops rebuild the views.

### MXFP4 V Contract

The F4F4 and F6F4 rows use true MXFP4 V: E2M1 values with one E8M0 scale for every
`(channel, 32-token)` block. `quantize_v_mxfp4` fuses amax, ceil-power-of-two scale generation,
normalization, E2M1 encoding, and the final col-major ASM layout. It returns a contiguous raw FP4
buffer plus a uint8 scale image shaped `[batch, heads, ceil(sequence / 128) * 512]`; ragged loads
are masked and the 64-byte launch slack is zero. The scale image is already in ASM gather order,
not generic row-major metadata. Packed launch uses `E8M0_PER_1X32`; FP8 V uses
`F32_PER_CHANNEL`.

One single-warp Triton program owns each `(32-token, 32-channel)` block, eliminating overlapping
writers. The deployed trailing-underscore F4F4/F6F4 kernels load V scales at QK exit so softmax
hides their VMEM latency, retain 95 SGPR and 256 VGPR, and use 66,048 and 43,008 bytes LDS
respectively. F4F4 keeps next-K0 prefetch under the penultimate PV MFMA; F6F4 keeps split-FP6 K0
prefetch at the PV tail because earlier placement was flat in balanced eight-GPU testing.

Any producer dtype, shape, or layout change requires a versioned custom-op name. Promotion requires
byte equality against the independent Torch payload/scale reference at sequences
`1, 127, 128, 129, 257`, deterministic output, zero slack, eager/fullgraph parity, allocator churn,
focused coverage, and repeated retained model captures. At
`b=1,hq=hk=5,sq=sk=65536,d=dv=128`, final eight-GPU e2e medians were
`3574.8 TFLOP/s` for F4F4 versus `3459.0` for F4F8, and `3351.2 TFLOP/s` for F6F4 versus
`3205.1` for F6F8. The deployed code-object SHA256 values are
`212981592d1e4801f93db1cb8cc37db1ed7335e3fdadf53c0d01e7bd53917d72` (F4F4) and
`a5046f1dcc0d51033122310efab70796e690086391285b9e5cdeaa5496d292a9` (F6F4).

### MXFP6 K Contract

MXFP6 K preprocessing fuses hd128 Hadamard rotation, E2M3 quantization, and final ASM-order packing
in one HIP launch. Each 128-token/head tile contains 12,288 data bytes, a 4,096-byte reserved
region, and a 1,024-byte scale tail. Partial tiles are zero-filled, and the public custom op returns
contiguous raw data and scale buffers so compiled callers never carry the exotic logical view.

Changes to this path require byte equality against `reorder_fp6_k_lds_order_triton` for compact
data, scale tails, and valid scale bytes at aligned and ragged sequence lengths. Keep the raw-buffer
custom-op ABI unchanged unless the op name is versioned with the layout.

## Formats And Scales

Format and scale granularity are separate concepts:

```python
class AttentionFormat(IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8_E4M3 = 3
    FP8_E4M3_FNUZ = 4
    FP8_E5M2 = 5
    FP8_E5M2_FNUZ = 6
    FP6_E2M3 = 7
    FP6_E3M2 = 8
    FP4_E2M1 = 9
    INT8 = 10
    UINT8 = 11
    INT4 = 12
    UINT4 = 13


class AttentionScaleMode(IntEnum):
    NONE = 0
    F32_PER_TENSOR = 1
    F32_PER_HEAD = 2
    F32_PER_TOKEN = 3
    F32_PER_CHANNEL = 4
    E8M0_PER_1X32 = 5
```

An FP8, FP6, FP4, or INT8 format does not imply a scale mode. The manifest explicitly records the
scale mode and scale storage format for Q, K, V, and O. This permits future kernels to reuse the
same number format with different quantization granularities without changing the public enum.

The raw API chooses the production recipe through `scale_modes_for_formats`; the packed API requires
that exact recipe explicitly. Add configurable scale modes only when multiple kernels support the
same Q/K/V formats.

## Output Contract

The API returns a BF16 tensor. If `out` is supplied, the kernel writes and returns that same tensor.
Low-precision output will require an explicit data/scale ownership contract and a versioned ABI;
do not add an output record before a kernel and downstream consumer require it.

`return_lse=False` is reserved in both APIs; `True` currently fails clearly. Once supported, use:

```python
output = mha_v4(..., return_lse=False)
output, lse = mha_v4(..., return_lse=True)
```

LSE must be contiguous FP32 `[batch, query_heads, query_length]`, representing the natural-log
log-sum-exp of the selected kernel's scaled logits. Use a versioned or dedicated LSE custom op so
compiled output arity remains stable; do not add dropout or RNG outputs.

## Explicit Kernel Dispatch

The host launcher receives an explicit, compile-time-specializable key containing at least:

```text
architecture
q_format
q_scale_mode
k_format
k_scale_mode
v_format
v_scale_mode
output_format
output_scale_mode
head_dim_qk
head_dim_v
mask_mode
sparse_mode
sequence_mode
layout
bf16_conversion
```

Tensor dtype, shape, stride, and storage size validate the selected row. They never select it.
Unsupported Q/K/V/O combinations fail at manifest lookup with the requested key in the error.

Manifest rows also own:

```text
query_tile
kv_tile
workgroup_size
kernarg_abi
kernel_symbol
code_object
```

Kernel cache identity is `(kernel_symbol, code_object)`, never the symbol alone.

BF16 dispatch uses the same explicit format and scale-mode key as other rows. Each architecture
owns its manifest row and code object under `hsa/<arch>/fmha_v4_fwd/`; adding gfx942 BF16 support
does not require a Python-side architecture branch.

## Sparse Contract

Sorted block-sparse execution is implemented for gfx950 hd128 rows (256×128) and for gfx942
native FP8/FP8 plus INT8/FP8 (256×64). Other gfx942 recipes stay dense-only.

Selection is an explicit manifest dimension (`mode=0` dense, `mode=1` sorted-sparse), not
inferred from pointers or redirected from a dense request. Dense and sparse use separate
launchers so the dense kernarg layout stays frozen.

Raw API: optional boolean `block_mask` at query-tile 256 × `mha_v4_kv_tile()` (128 on gfx950,
64 on gfx942). Convert with
`block_attn_mask_to_ragged_lut(..., num_heads=q.shape[2], return_none_if_dense=False)`.
An all-True mask still takes the sparse row. GQA uses the same ratio as dense; LUT and work-table
rows are one per query head. A 3-D mask broadcasts across query heads; a 4-D mask may give grouped
query heads different KV-tile lists.

Packed API: optional int32 LUT triple. `lut_start` / `lut_count` have one entry per
`(batch, query_head, query_block)`. `kv_block_indices` is 1-D and may be over-allocated to
`B*H*Qtiles*KVtiles` to avoid data-dependent allocations. Key length must be a multiple of the
architecture KV tile (128 on gfx950, 64 on gfx942).

The host builds a work table inside the sparse custom op. If every `lut_count` is equal
(uniform / top-k sparsity), visit order stays raster; otherwise rows are ordered
longest-LUT-first (LPT).

Up to 8192 entries one fused kernel ranks and packs the table; past that the sort falls back to
ATen. The limit is where the 8-byte keys fill the 64 KB of LDS a workgroup gets.

A LUT row may select nothing. `lut_count == 0` is a no-op that writes a zero output tile, so an
all-False `block_mask` row is valid input: the ASM clamps the row's prologue reads in bounds and
skips the KV traversal, and the epilogue's zero-row-sum path zeroes the tile. That makes the entry
count unbounded below, so the only bound the launcher can check without reading device data is that
`kv_block_indices` is non-empty (the kernels dereference the row base even for an empty row, and
read speculatively up to one entry past the row they traverse). Set `AITER_MHA_V4_VALIDATE_LUT=1` to
also check starts, counts, and index ranges device-side, which costs a synchronization per launch
and is off by default.

Sparse code objects live next to dense ones: `hsa/gfx950/fmha_v4_fwd/` (for example
`fwd_hd128_fp8_sparse.co`) and `hsa/gfx942/fmha_v4_fwd/MI300/` for the two gfx942 recipes.

Do not add optional LUT arguments to the dense MXFP4/MXFP6 launch custom ops; sparse MX goes
through `mha_v4_packed` after reconstructing views.

### VSA Compatibility

AITER VSA supplies delta-encoded fixed-capacity rows plus counts at 128-query-token granularity;
the proposed MHA v4 descriptor uses flat absolute indices and explicit start/count. Encoding
conversion is cheap, but geometry is not: current 256x128 ASM workgroups share one KV list across
two 128-row halves, while adjacent VSA rows may differ. Exact support therefore follows:

1. Directly use an existing 256x128 sparse kernel when adjacent 128-query VSA rows are identical or
    when the policy natively emits 256-query rows.
2. Add a manifest-selected 128x128 ASM sparse kernel for arbitrary VSA rows. This is the primary
    exact compatibility path and must be benchmarked because reducing the query tile changes the
    eight-wave load/compute balance.
3. Optionally add a 256x128 union kernel carrying per-half membership bits if VSA masks have enough
    overlap to make union overcompute cheaper than the 128x128 kernel. This is a separate optimized
    ABI, not the default conversion.

A compatibility helper may decode existing VSA tensors into the common descriptor and reuse the
same packed executor. It must not create another quantization or dispatch stack. Ordered-prefix
optimizations such as `freeze_after` are optional manifest-selected extensions, not prerequisites
for compatibility.

## Output ABI Evolution

Existing kernels write BF16 through the v1 argument layout. Low-precision output requires a
versioned extension rather than repurposed fields, with explicit metadata for at least:

```text
output scale pointer
output data format
output scale format and mode
output scale strides or contiguous-layout metadata
```

Fix offsets with the first implementing kernel; existing v1 binaries retain their original size.

## `torch.compile` Rules

1. Keep Q, K, and V preprocessing as separate custom ops; keep ASM launch behind a custom op.
2. Pass exotic layouts across custom-op boundaries as contiguous raw buffers and rebuild views at
    launch. Fake implementations must expose exact public shapes and dtypes.
3. Version custom-op names whenever output shape, packed layout, or ABI changes.
4. Validate compiled paths with allocator churn and a downstream consumer.
5. Avoid data-dependent sparse allocations.
6. Use `Optional[T]`, not `T | None`, in public/fake/custom-op declarations because the latter
    caused a measured end-to-end Inductor regression.

## Forward Roadmap

1. Add VSA/Sparge adapters over the shared sparse LUT and packed executor, plus a 128x128 sparse
    tile if adjacent VSA 128-query rows differ.
2. Add LSE under a stable output schema for ring attention.
3. Add approximate BF16 under a distinct symbol and code object from generic v3 BF16.
4. Add a versioned low-precision-output ABI once data/scale ownership is concrete.
5. Expand architectures, head dimensions, sequence modes, and format combinations only through
    explicit manifest rows.

## Required Validation

Every dense change must preserve eager/fullgraph parity, finite output, allocator-churn safety,
explicit dispatch, unsupported-contract rejection, deterministic fixed-input behavior, and BF16
reference accuracy. Layout or quantizer changes additionally require byte-level tests at aligned
and ragged sequences. Synchronization or performance changes require repeated retained captures
and balanced multi-GPU target-shape benchmarking.

Sparse work adds LUT validation for partial KV tails, varied row counts, empty-row policy, explicit
sparse dispatch, and correctness against BF16. ABI or output-shape changes require versioned custom
ops and compatibility tests for existing binaries.
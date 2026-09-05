# MHA v4

MHA v4 is the BF16-output attention path backed by explicit format, scale, packing, and sparse
dispatch metadata. Unsupported recipes fail instead of falling back to another attention engine.

## Scope

- Contiguous BF16 BSHD inputs with head dimension 128.
- BF16 BSHD output.
- Dense and sorted block-sparse inference.
- Grouped-query ratios `1, 2, 4, 8, 16`.
- No backward, dropout, RNG state, LSE, causal, or varlen support yet.

Supported dense recipes:

| Q/K | V |
|---|---|
| BF16 | BF16 |
| BF16 | FP8 |
| INT8 | FP8 |
| MXFP8 | FP8 |
| FP8 | FP8 |
| FP8 | MXFP6 |
| MXFP6 | FP8 |
| MXFP6 | MXFP6 (dense only) |
| MXFP6 | MXFP4 |
| MXFP4 | FP8 |
| MXFP4 | MXFP4 |

## Ownership

`aiter.ops.mha_v4` owns:

- `AttentionFormat`, `AttentionScaleMode`, and `AttentionPack`;
- raw recipe selection and validation;
- dense/sparse manifest dispatch;
- `mha_v4` and `mha_v4_packed`;
- final launch wrappers that rebuild packed views.

`aiter.ops.mha_v4_quant` owns:

- rotation and quantization producers;
- packed-buffer allocation and sizing;
- MXFP4/MXFP6 layout constants;
- `mxfp4_k_view`, `mxfp6_k_view`, and `mxfp4_v_view`.

The dependency is one-way: `mha_v4` imports `mha_v4_quant`. The entrypoint re-exports the
established producer API for compatibility, but new implementation-facing code should import
producers from `mha_v4_quant`.

Q, K, and V remain separate custom ops so distributed runtimes can overlap preprocessing with
communication. Nonstandard layouts cross custom-op boundaries as contiguous raw buffers and are
rebuilt only at the launch boundary.

### Producer Backends

Backend choice is private to `mha_v4_quant`; recipe selection does not branch on it.

| Producer | Backend |
|---|---|
| Per-tensor INT8/FP8 | Triton |
| Rotated FP8 and FP8 V | Triton |
| Canonical MXFP6 V | Triton |
| Canonical MXFP4 V | Triton |
| MXFP8/MXFP6/MXFP4 Q and K | HIP `module_fmha_v4_fwd` |
| FP6-P MXFP6 V | HIP `module_fmha_v4_fwd` |
| FP6-P MXFP4 V | HIP `module_fmha_v4_fwd` |

## APIs

Use `mha_v4` for BF16 inputs and canonical preprocessing:

```python
output = mha_v4(
    query,
    key,
    value,
    q_format=AttentionFormat.MXFP6,
    k_format=AttentionFormat.MXFP6,
    v_format=native_fp8_format(),
    block_mask=None,
)
```

Use `mha_v4_packed` when preprocessing is external or overlapped:

```python
output = mha_v4_packed(
    packed_query,
    packed_key,
    packed_value,
    q_scale,
    k_scale,
    v_scale,
    q_format,
    k_format,
    v_format,
    q_scale_mode,
    k_scale_mode,
    v_scale_mode,
    v_pack=AttentionPack.DEFAULT,
)
```

Formats and scale modes are independent manifest dimensions. Tensor dtype, shape, stride, and
storage validate a selected row; they never select one. Omitting raw scale modes selects the
canonical recipe from `scale_modes_for_formats()`; supplying them requires all three modes and
selects another explicitly supported recipe such as MXFP8.

## Packed Layouts

MX producers return contiguous raw buffers when the ASM layout is not representable as an ordinary
contiguous tensor. Rebuild logical views with the helpers in `mha_v4_quant` immediately before
calling `mha_v4_packed`.

MXFP4 V uses E2M1 values with one E8M0 scale per `(channel, 32-token)` block. Each 128-token tile
contributes 8,192 data bytes and 512 scale bytes. The data buffer includes 64 bytes of launch slack.

`AttentionPack.DEFAULT` is the canonical V token order used by sparse kernels and FP8-P rows.
`AttentionPack.V_FOR_FP6_P` selects the token order consumed by dense FP6-P rows. Numeric format and
packing order are separate contracts.

Changing a custom op's output shape or packed layout requires a versioned custom-op name.

## Sparse Contract

Raw callers pass an optional boolean `block_mask`:

- gfx950 geometry: 256 query tokens by 128 KV tokens;
- gfx942 geometry: 256 query tokens by 64 KV tokens;
- shape `[B, H, Qtiles, KVtiles]` or `[B, Qtiles, KVtiles]` with head broadcast.

Packed callers pass all or none of the int32 LUT triple: `kv_block_indices`, `lut_start`, and
`lut_count`. LUT/work-table rows are per query head, including under GQA. Dense uses manifest
`mode=0`; sorted sparse uses `mode=1` and a separate launcher/code object.

An empty sparse row is valid and writes a zero output tile. Set `AITER_MHA_V4_VALIDATE_LUT=1` for
device-side start/count/index validation; it synchronizes and is disabled by default.

Dense and sparse code objects may use different reduction schedules. Compare their outputs with a
strict numerical tolerance or cosine threshold, not bit equality. Comparisons between two launches
of the same code object may remain exact where determinism is part of the test.

## Compile And ABI Rules

1. Keep Q, K, V preprocessing and ASM launch behind separate custom ops.
2. Pass exotic layouts across custom-op boundaries as contiguous raw buffers.
3. Fake implementations must expose exact output shapes and dtypes.
4. Version custom-op names when output shape, packed layout, or ABI changes.
5. Preserve `Optional[T]` in public/fake/custom-op declarations; `T | None` caused a measured
   Inductor regression.
6. Do not infer dispatch from tensor metadata or redirect unsupported recipes.

## Validation

Run `pytest op_tests/test_mha_v4.py` for entrypoint changes. Quantizer/layout changes additionally
require byte-level checks at aligned and ragged sequence lengths, eager/fullgraph parity, allocator
churn, and downstream-consumer coverage. Kernel performance changes require the relevant retained
model captures and balanced multi-GPU target-shape benchmarks.

Key implementation locations:

- Python dispatch: `aiter/ops/mha_v4.py`
- Producers and layouts: `aiter/ops/mha_v4_quant.py`
- HIP quantization: `csrc/kernels/mha_v4_quant.cu`
- Host launcher: `csrc/py_itfs_cu/asm_mha_v4_fwd.cu`
- Manifests and binaries: `hsa/<arch>/fmha_v4_fwd/`

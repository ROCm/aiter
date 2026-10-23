# MHA v4

MHA v4 is the BF16-output attention path backed by explicit format, scale, packing, and sparse
dispatch metadata. Unsupported recipes fail instead of falling back to another attention engine.

## Scope

- Contiguous BF16 BSHD inputs with head dimension 128.
- BF16 BSHD output.
- Dense, sorted block-sparse, and Sol-Attn inference.
- Each sparse and Sol-Attn row declares the KV tiles per sequence it can traverse (`lut_max`);
  the launcher refuses longer sequences instead of overrunning a staged LUT.
- Grouped-query ratios `1, 2, 4, 8, 16`.
- Per-batch key lengths via `seqlens_k`, on the dense GFX950 `BF16 Q/K` rows only. Every other
  recipe, architecture, and the sorted-sparse path reject it.
- Log-sum-exp via `return_lse` on GFX950, wherever the selected manifest row declares `lse`, which
  today is every dense, sparse and Sol-Attn row. GFX942 rejects it until its exported value is
  measured.
- No backward, dropout, RNG state, causal, or Q-side varlen support yet.

Supported GFX950 recipes. Every recipe is available in both dense and sorted-sparse mode with the
same V packing and scale modes, and every Sol-Attn row uses that packing too.

| Q/K | V | Modes |
|---|---|---|
| BF16 | BF16 | dense, sparse, Sol-Attn |
| BF16 | FP8 | dense, sparse, Sol-Attn |
| INT8 | FP8 | dense, sparse, Sol-Attn |
| MXFP8 | FP8 | dense, sparse, Sol-Attn |
| FP8 | FP8 | dense, sparse, Sol-Attn |
| FP8 | MXFP6 | dense, sparse, Sol-Attn |
| MXFP6 | FP8 | dense, sparse |
| MXFP6 | MXFP6 | dense, sparse, Sol-Attn |
| MXFP6 | MXFP4 | dense, sparse, Sol-Attn |
| MXFP4 | MXFP4 | dense, sparse, Sol-Attn |

GFX942 ships per-tensor FP8/FP8 and INT8/FP8 in all three modes.

MXFP4 Q/K requires MXFP4 V. The FP8-V variant is retired.

## Ownership

`aiter.ops.mha_v4` owns:

- `AttentionFormat`, `AttentionScaleMode`, and `AttentionPack`;
- raw recipe selection and validation;
- dense/sparse/Sol-Attn manifest dispatch and the geometry queries (`mha_v4_block_tile`,
  `mha_v4_block_tiles`, `mha_v4_kv_tile`, `mha_v4_operands`);
- `mha_v4`, `mha_v4_sol`, and `mha_v4_packed`;
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
| MXFP8/MXFP6/MXFP4 Q and K | HIP `module_mha_v4_quant` |
| FP6-P MXFP6 V | HIP `module_mha_v4_quant` |
| FP6-P MXFP4 V | HIP `module_mha_v4_quant` |

No manifest row consumes canonical MX V: every MXFP6-V and MXFP4-V row selects the FP6-P pack.
`quantize_v_mxfp6` is retained only as the reference the FP6-P layout test permutes against.

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

`AttentionPack.DEFAULT` is the canonical V token order. `AttentionPack.V_FOR_FP6_P` selects the V
token order that FP6-P and FP4-P consumers require, and a row's packing no longer depends on the
mode: dense and sparse rows for the same recipe select the same pack. Numeric format and consumer
pairing remain separate dispatch contracts even when the physical V layout is identical.

Changing a custom op's output shape or packed layout requires a versioned custom-op name.

## Sparse Contract

Raw callers pass an optional boolean `block_mask`:

- shape `[B, H, Qtiles, KVtiles]` or `[B, Qtiles, KVtiles]` with head broadcast;
- geometry `block_tile`, defaulting to the recipe's own: on gfx950 256x64 for BF16 and BF16/FP8
  Q/K and 256x128 for the rest, on gfx942 256x64 throughout;
- gfx950 also ships 64x64 sparse and Sol-Attn rows for FP8, INT8/FP8, BF16, BF16/FP8, MXFP8,
  FP8/MXFP6, MXFP6, MXFP6/MXFP4, MXFP4 (the MX V rows in FP6-P order) and MXFP6/FP8. MXFP6/FP8's
  Sol-Attn row is 64x64 only, so a Sol-Attn launch of it must pass `block_tile=(64, 64)`.

The geometry is a property of the manifest row, so the rows no longer agree on one KV tile per
arch. Ask `mha_v4_block_tile(mha_v4_operands(...), mode)` or `mha_v4_block_tiles(...)` with the
operands being launched; `mha_v4_kv_tile()` without operands raises where the rows disagree rather
than returning a tile the mask may not match.

Packed callers pass all or none of the int32 LUT triple: `kv_block_indices`, `lut_start`, and
`lut_count`, plus `block_tile` for a non-default geometry. LUT/work-table rows are per query head,
including under GQA. Dense uses manifest `mode=0`; sorted sparse uses `mode=1` and a separate
launcher/code object. A row declaring `ragged_kv` accepts a key length that is not a multiple of
its KV tile; the others refuse it.

An empty sparse row is valid and writes a zero output tile. Set `AITER_MHA_V4_VALIDATE_LUT=1` for
device-side start/count/index validation; it synchronizes and is disabled by default.

Dense and sparse code objects may use different reduction schedules. Compare their outputs with a
strict numerical tolerance or cosine threshold, not bit equality. Comparisons between two launches
of the same code object may remain exact where determinism is part of the test.

## Sol-Attn Contract

Sol-Attn (arXiv 2607.24027) is manifest `mode=2`. It runs the same block-sparse exact pass as
`mode=1`, then a second pass over pooled per-block K/V that masks off the blocks the LUT already
covered, so a below-threshold block contributes its zeroth-order term instead of nothing. Both
passes share one online-softmax state, and `return_lse` reports that one softmax.

Raw API: `mha_v4_sol(..., beta=0.4)`. It takes no selection, because routing has to see the
quantized K the kernel will read; `beta` sets the per-query-tile threshold at
`mean_j(proxy) + beta * std_j(proxy)`, so it selects a block density rather than a block count.
It always uses the canonical scale modes and does not apply `mha_v4`'s K-mean smoothing. It
serves every recipe with a mode-2 row except MXFP8, whose formats are FP8's; MXFP8 runs through
the packed API.

Packed API: the LUT triple plus `mean_k`, `mean_v`, and `block_bitmap`, all set or all omitted, and
rejected without a LUT triple. `aiter.ops.triton.attention.utils.sol_prepare()` produces all
of them from one selection, either routed from `beta` or supplied as `block_attn_mask`, so the
bitmap and the LUT cannot disagree. A supplied mask changes only the LUT and the bitmap; the
unselected blocks are still swept from the pooled K/V, which is what separates Sol-Attn from a
`mode=1` launch over the same mask. The pooling block is the row's KV tile and is not passed to the
kernel, so pool at `mha_v4_block_tile()`'s answer.

Pooled scales: a per-tensor or per-channel descale survives pooling over the sequence, so BF16,
BF16/FP8, FP8, and INT8/FP8 leave `mean_k_scale` and `mean_v_scale` unset. An E8M0 1x32 operand
pools in dequantized space and is requantized, so its pooled scale must be passed: an MX Q/K
recipe (including MXFP8) passes `mean_k_scale`, and an MX V recipe passes `mean_v_scale`. The requirement is checked against the scale
modes, because an unset slot means "read the source scale", not "error". MX operands are not
element addressable, so name them in `k_packed_format` / `v_packed_format` and pass the
pre-quantization tensors as `k_source` / `v_source`; FP6-P V rows take `mxfp4_fp6_p` or
`mxfp6_fp6_p`. A pooled V keeps V's 128-token packing whatever the KV tile, so at 64x64 its scale
image covers the pooled rows rounded up to 128.

Rows declaring `sorted` launch over a work table ordered by `lut_count` (`sorted_dispatch`); the
order changes only scheduling, and the output is bitwise identical either way. `kv_range_tokens`
splits the softmax into LSE-merged key ranges on rows declaring `kv_range`.

## Compile And ABI Rules

1. Keep Q, K, V preprocessing and ASM launch behind separate custom ops.
2. Pass exotic layouts across custom-op boundaries as contiguous raw buffers.
3. Fake implementations must expose exact output shapes and dtypes.
4. Version custom-op names when output shape, packed layout, or ABI changes.
5. Preserve `Optional[T]` in public/fake/custom-op declarations; `T | None` caused a measured
   Inductor regression.
6. Do not infer dispatch from tensor metadata or redirect unsupported recipes.

## Validation

Run `pytest op_tests/test_mha_v4.py op_tests/test_mha_v4_sparse.py` for entrypoint changes, and
`op_tests/triton_tests/attention/test_sol_prepare.py` for Sol-Attn routing or pooling. Quantizer/layout changes additionally
require byte-level checks at aligned and ragged sequence lengths, eager/fullgraph parity, allocator
churn, and downstream-consumer coverage. Kernel performance changes require the relevant retained
model captures and balanced multi-GPU target-shape benchmarks.

Key implementation locations:

- Python dispatch: `aiter/ops/mha_v4.py`
- Producers and layouts: `aiter/ops/mha_v4_quant.py`
- HIP quantization: `csrc/kernels/mha_v4_quant.cu`
- Host launcher: `csrc/py_itfs_cu/asm_mha_v4_fwd.cu`
- Manifests and binaries: `hsa/<arch>/fmha_v4_fwd/`

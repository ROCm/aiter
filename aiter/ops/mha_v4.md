# MHA V4 Entrypoint And FMHA V4 Engine

> Evolving engineering plan for contributors and coding agents. This is not release documentation.
> Carefully update the status, decisions, and checklist whenever you change the design.

## Status

- Last updated: 2026-08-08.
- Development branch: `mha_v4`, forked from `mxfp6_fmha_gfx950` at `8ccca033`.
- Preserve `mxfp6_fmha_gfx950` as the validated integration baseline; do not add MHA v4 work to it.
- Phase: dense BF16-output extraction and xDiT migration implemented and validated on gfx950.
- `mha_v4` and `mha_v4_packed` support the six initial gfx950 combinations.
- A gfx942/CDNA3 signed INT8/FP8 manifest row and code object are also preserved under v4.
- Keep upstream `aiter/ops/mha.py` annotation style unchanged. New MHA v4 entrypoints deliberately
    use `Optional[T]`: the equivalent `T | None` annotations caused measured Inductor regressions in
    end-to-end model execution.

## Fixed First-Release Decisions

- Public module and entrypoint: `aiter.ops.mha_v4.mha_v4`.
- Internal JIT, CSV, HSA directory, and launcher family: `fmha_v4_fwd`.
- Six dense format combinations listed in [Current Dense Performance](#current-dense-performance).
- Batched, non-causal MHA only; head dimension 128; BF16 output.
- Forward/inference only: no backward, dropout, dropout mask, or RNG state.
- `return_lse=False` is reserved in the API, but `True` is unsupported until an LSE-writing kernel
    is implemented.
- Explicit format dispatch; never infer the kernel from packed width or V dtype.
- Stable format IDs distinguish signed and unsigned integer operands (`INT8`, `UINT8`, `INT4`,
  `UINT4`). Future RDNA3/RDNA4 IU8/IU4 kernels map these IDs to the WMMA NEG-bit signedness fields;
  signedness is never inferred from packed storage dtype.
- Value formats are ordered floating-point largest-to-smallest, then integer: FP32, FP16, BF16,
  explicit FP8 encodings, explicit FP6 encodings, FP4 E2M1, then signed/unsigned INT8/INT4.
- FP16 is reserved even though the initial manifest has no FP16 row.
- FP6 encodings are explicit: ID 7 is `FP6_E2M3` (the current kernels; `MXFP6` remains an alias),
  while ID 8 is reserved for `FP6_E3M2` (`MXBF6` shorthand). Packed width alone must not select
  between them because both encode four values in three bytes.
- NVFP4 is deferred. It uses the same FP4 E2M1 values as MXFP4 but a different dual-scale recipe,
  so future support belongs in `AttentionScaleMode` and manifest rows, not a new value format.
- MXFP8 likewise uses an FP8 value encoding plus an E8M0 block-scale mode; it does not need a new
    `AttentionFormat`. TF32 is an FP32 compute mode rather than a stored operand format. FP64, NF4,
    INT2, and other formats stay out of the first enum until an attention kernel and ABI require them.
- Unsupported capabilities fail clearly; never fall back to `aiter.ops.mha`.
- No Sage branding in the public API because the six combinations do not map cleanly to Sage
    versions.

## Authoritative References

- Public implementation: `aiter/ops/mha_v4.py`.
- Dedicated host launcher: `csrc/py_itfs_cu/asm_mha_v4_fwd.cu`.
- Explicit manifests and binaries: `hsa/<arch>/fmha_v4_fwd/`.
- Generic `aiter.ops.mha` and `fmha_v3_fwd` are restored to non-mixed-format ownership.
- Benchmark and production preprocessing reference: `op_tests/op_benchmarks/triton/bench_sage.py`.
- Compile-safe production integration reference:
    `/app/xDiT/xfuser/core/distributed/attention_backend.py`.
- Canonical PyISA sources: `/workspace/diffusion-models-inference-private/asm/fmha_sage_fwd/gfx950/`.
- Approximate BF16 source for a later phase:
    `/workspace/diffusion-models-inference-private/asm/fmha_v3_fwd/mi350/fwd_hd128_bf16_block.py`.

## Implementation Checklist

### Dense Extraction

- [x] Define format IDs without binding format to scale granularity.
- [x] Add the `fmha_v4_fwd` manifest with explicit Q/K/V/O formats and kernel identity.
- [x] Add a dedicated C++/HIP launcher with no dtype- or shape-based format inference.
- [x] Add the final ASM launch custom op and fake implementation.
- [x] Move or wrap Q, K, and V preprocessing as independent compile-safe custom ops.
- [x] Keep exotic K/V views inside the final custom-op boundary; pass contiguous backing buffers.
- [x] Implement `mha_v4(...) -> torch.Tensor` for the six dense combinations.
- [x] Expose `mha_v4_packed` for kernel-only benchmarks and integrations with packed operands.
- [x] Migrate `bench_sage.py` kernel-only and `--e2e` paths.
- [x] Migrate xDiT callers to `aiter.ops.mha_v4` in the xDiT repository without mixing
    cross-repository changes into the AITER PR.
- [x] Remove branch-added mixed-format wrappers and automatic I8FP8 routing from `aiter.ops.mha`.
- [x] Restore mixed-format ownership out of the v3 host, manifest, and code-object slots.

### Validation Gates

- [x] Eager accuracy for all six combinations against the pure BF16 reference through
    `bench_sage.py`.
- [x] `torch.compile(fullgraph=True)` eager/compiled bitwise parity for every raw preprocessing and
    launch path.
- [x] Allocator churn plus a downstream `.contiguous()` consumer for every raw path.
- [x] Finite-output checks for packed and raw paths.
- [x] Explicit manifest dispatch observed for every gfx950 symbol/code object.
- [ ] Add automated rejection tests for unsupported format, mask, layout, head-dimension, and
    head-count requests.
- [x] Run the requested long-context command:
    `python op_tests/op_benchmarks/triton/bench_sage.py --b 1 --hq 5 --sq 8192 --d 128 --kernel all`.
- [x] `python -m pytest -q op_tests/test_mha_v4.py`: 23 passed.
- [ ] Run the balanced real target-shape benchmark and record GPU count plus code-object hashes.

### Deferred Phases

- [ ] Sparse 256x128 ragged-LUT kernels and Sparge integration.
- [ ] VSA compatibility and, if needed, an exact 128x128 sparse kernel.
- [ ] LSE-writing kernels and ring-attention integration.
- [ ] FP8 output with an explicit data/scale contract.
- [ ] Approximate BF16-input kernel under a distinct identity from v3 BF16.
- [ ] GQA, causal, varlen, additional head dimensions, and other Q/K/V/O combinations.
- [ ] Add more gfx942/CDNA3 combinations, gfx1250/CDNA5, and RDNA3/RDNA4 manifest rows/code objects.
    RDNA integer rows must encode A/B signedness explicitly for IU8/IU4 WMMA selection.

## Design Index

- [Naming And Layers](#naming-and-layers)
- [Goal](#goal)
- [Current Dense Performance](#current-dense-performance)
- [Package Boundary](#package-boundary)
- [Public API Levels](#public-api-levels)
- [Formats And Scales](#formats-and-scales)
- [Output Contract](#output-contract)
- [Explicit Kernel Dispatch](#explicit-kernel-dispatch)
- [Sparse Contract](#sparse-contract)
- [VSA Compatibility](#vsa-compatibility)
- [Output ABI Evolution](#output-abi-evolution)
- [`torch.compile` Rules](#torchcompile-rules)
- [Migration](#migration)
- [Open Decisions](#open-decisions)
- [Required Validation](#required-validation)

## Naming And Layers

Use `fmha_v4_fwd` for the internal launch family, generated manifest, JIT module, and HSA directory:

```text
aiter/hsa/<arch>/fmha_v4_fwd/
```

This name is recognizable beside `fmha_v3_fwd`, but it is not the primary application API. A v4
engine denotes a more extensible dispatch and kernarg contract, not universally better accuracy or
a replacement for every v3 kernel.

In AITER terminology, FMHA means fused multi-head attention; it does not mean forward-only. The
direction is expressed by the `_fwd` or `_bwd` suffix, as in the existing `fmha_v3_fwd` and
`fmha_v3_bwd` families. Therefore `fmha_v4` would not communicate the lack of backward support more
clearly than `mha_v4`.

The first public API is `aiter.ops.mha_v4`. A future unification may move stable functionality into
`aiter.ops.mha`, but the initial separation avoids adding more format-specific routing to that
already broad module.

The public function remains `mha_v4(...)`; its contract explicitly states inference-only and
forward-only. The low-level custom op, JIT module, CSV, and code-object directory use
`fmha_v4_fwd`, where the direction suffix is useful and consistent with AITER conventions.

Do not brand the first API as Sage. INT8/FP8 resembles SageAttention v1 and MXFP4/FP8 is related to
later low-precision attention work, but the supported format combinations do not map exactly onto
SageAttention versions. `mha_v4` describes the explicit engine generation without making a
potentially misleading algorithm claim.

## Goal

Create an FMHA v4 engine independent of `aiter.ops.mha` that can grow without encoding kernel
identity in tensor dtype, packed width, or incidental storage layout.

The initial release includes six dense, non-causal, head-dimension-128 MHA kernels:

- INT8 Q/K with FP8 V;
- FP8 Q/K with FP8 V;
- MXFP6 Q/K with FP8 V;
- MXFP4 Q/K with FP8 V;
- MXFP6 Q/K with MXFP4 V;
- MXFP4 Q/K with MXFP4 V.

All six initially write BF16 output. Unsupported combinations return an explicit error; there is no
fallback to `aiter.ops.mha`.

Head dimension 128 is an initial manifest capability, not a permanent public-API restriction. The
API derives logical head dimensions from its inputs and dispatch key; future manifest rows may add
other dimensions without introducing another entrypoint.

Sparse execution, VSA compatibility, Sparge policy, causal attention, grouped-query attention,
varlen attention, approximate BF16 input, and low-precision output are follow-up work.

This entrypoint is inference-only. It does not expose dropout, a dropout mask, backward state, or an
RNG state. RNG state in the existing generic FMHA API exists to reproduce training-time dropout;
without dropout it has no role in MHA v4.

When the approximate BF16 kernel is added, it must not replace or silently dispatch from AITER's
existing BF16 FMHA implementation.

## Current Dense Performance

Current gfx950 long-sequence dense ASM kernel throughput, excluding Q/K/V preprocessing:

| Q/K format | V format | Throughput (TFLOP/s) |
|---|---|---:|
| INT8 | FP8 | 2315 |
| FP8 | FP8 | 3118 |
| MXFP6 | FP8 | 3430 |
| MXFP4 | FP8 | 3540 |
| MXFP6 | MXFP4 | 3790 |
| MXFP4 | MXFP4 | 4000 |

These values are the current optimization baselines, not portable performance guarantees. Attach
the exact benchmark shape, harness revision, GPU count, and code-object hashes when promoting them
to release-facing documentation.

## Package Boundary

```text
aiter/ops/mha_v4/
    __init__.py       stable raw-QKV and packed exports
    api.py            raw-QKV preprocessing and packed orchestration
    types.py          formats, scale modes, LUT, and operand/output records
    quant.py          shared compile-safe quantization custom ops
    _ops.py           launch custom op and fake implementation
    _manifest.py      generated or loaded kernel capability table
```

This package must not import the high-level dispatch machinery in `aiter.ops.mha`. The final
custom op calls a dedicated C++/HIP FMHA v4 launcher. Existing code-object loading utilities may be
shared where their contracts match.

## Public API Levels

Establish two public levels and keep direct code-object launch private.

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
)
```

The input tensors are unquantized BF16 in the first implementation. `q_format`, `k_format`, and
`v_format` specify the formats prepared for the ASM kernel. Output is BF16 in the initial API.

Each operand format is independent at the API level. The manifest defines supported combinations;
for example, an MXFP4 Q plus MXFP6 K combination returns a clear unsupported-kernel error until a
matching kernel exists. No combination is inferred from tensor dtype or shape.

The first release rejects causal mode, grouped-query head counts, sparse metadata, `return_lse=True`,
head dimensions other than 128, and formats not listed in the initial kernel matrix.

Q, K, and V preprocessing remain separate custom ops. This lets `torch.compile` and distributed
schedulers run each operation as soon as its corresponding input is available.

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
)
```

The packed API takes each operand's data tensor, scale tensor, value format, and scale mode
explicitly. The launcher validates this complete key against one manifest row.

Packed layout is also part of that explicit contract. The MXFP4-QK/FP8-V row uses the
chunk-major coalesced K layout produced by `quantize_mxfp4_k`; the initial F4F4 row still uses
contiguous token-strided K until its binary is migrated separately. Passing one layout to the
other row is rejected.

Exotic LDS-order tensors are represented by contiguous backing buffers plus metadata. An
`as_strided` view never crosses a custom-op boundary; the final launch op reconstructs it while
populating the kernarg.

Arbitrary code-object paths and symbols are not a production API. Kernel-development tools may
retain a separate direct launcher.

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

The raw API initially chooses the production scale mode associated with the selected manifest row.
The packed API requires it explicitly in each operand descriptor. A future raw API option may
request a non-default scale mode when more than one kernel supports the same Q/K/V/O formats.

## Output Contract

The initial API supports BF16 output only and does not expose an `output_format` argument yet. It
returns a plain BF16 `torch.Tensor`. If `out` is supplied, the kernel writes it and returns the
same tensor.

This matches current AITER behavior: low-level `fmha_v3_fwd` returns its internal four-tensor tuple,
but user-facing `flash_attn_func`, FP8/I8FP8 wrappers, and the current MX-packed wrapper return only
the output tensor. xDiT likewise consumes a tensor directly.

When FP8 output is added, the API will need to return or accept both data and scale. That extension
may introduce an `AttentionOutput` record or a separate quantized-output API. Do not add the record
to the BF16-only release before its data/scale ownership and downstream use are concrete.

Reserve `return_lse: bool = False` in both raw and packed APIs. The initial manifest has no
LSE-writing rows, so `return_lse=True` returns a clear unsupported-capability error. Once kernels
write LSE, the return convention is:

```python
output = mha_v4(..., return_lse=False)
output, lse = mha_v4(..., return_lse=True)
```

LSE is contiguous FP32 with shape `[batch, query_heads, query_length]`. It is the natural-log
log-sum-exp of the exact scaled logits used by the selected kernel, before output quantization. This
is the state ring attention needs to merge partial outputs from different KV shards.

Adding LSE must not add dropout or RNG outputs. The launch custom op should use a versioned schema
or a dedicated LSE-returning op so its output arity remains stable under `torch.compile`; the Python
wrapper may select that op using the specialized `return_lse` boolean.

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

The approximate BF16 kernel uses a distinct symbol, code-object slot, and manifest row, for example
`fwd_hd128_bf16_approx.co`. It must not overwrite or reuse generic `fwd_hd128_bf16.co` dispatch.

## Sparse Contract

Deferred to the sparse follow-up PR. The first release does not accept sparse metadata.

The primary sparse input is a ragged LUT:

```python
@dataclass(frozen=True)
class AttentionBlockSparseLut:
    kv_block_indices: torch.Tensor
    lut_start: torch.Tensor
    lut_count: torch.Tensor
    query_block_size: int = 256
    kv_block_size: int = 128
```

All three tensors are contiguous device `int32` tensors. `lut_start` and `lut_count` contain one
entry per `(batch, query_head, query_block)`. Every active query block must contain at least one KV
block until kernels define an empty-row result.

`block_mask_to_lut()` is a convenience custom op. It may overallocate `kv_block_indices` to avoid
data-dependent output shapes and graph breaks. The packed expert API accepts a prebuilt LUT.

Sparse selection is explicit and resolves a sparse manifest row and code object. A non-null LUT
must never silently redirect a dense kernel, and sparse selection is never inferred from extra
kernarg pointers.

Current sparse PyISA kernels append these pointers to the v3 kernarg:

```text
0x290  kv_block_indices
0x2a0  lut_start
0x2b0  lut_count
```

Dense v1 kernels retain the 656-byte kernarg and current sparse v1 kernels retain the 704-byte
kernarg. Each manifest row declares its ABI and size.

### VSA Compatibility

AITER's existing `vsa_sparse_attention` is primarily a sparse execution API. It does not discover
the sparse pattern. Its caller supplies a fixed-capacity LUT and a count for every
`(batch, query_head, 128-query-token block)`.

Its metadata differs from the FMHA v4 ragged ABI:

- the VSA LUT row has capacity `ceil(kv_len / 128)`;
- entry zero is an absolute KV-block index and later entries are delta encoded;
- `block_counts` gives the active prefix and the final row slot is reserved for CK lookahead;
- FMHA v4 uses flat absolute indices plus `lut_start` and `lut_count`;
- current VSA selection granularity is 128 query tokens, while the existing PyISA sparse kernels
  share one KV list across a 256-query-token workgroup.

The encoding conversion is cheap and belongs in a compile-safe GPU custom op. The query-block
geometry is not merely an encoding difference. Two adjacent VSA rows may select different KV
blocks, whereas the current eight-wave PyISA kernel cooperatively stages one selected KV block for
both 128-row wavegroups. Merging the two lists would either change semantics or require computing
their union and masking membership separately for each wavegroup.

FMHA v4 therefore treats VSA as another producer of the common ragged sparse descriptor, with the
descriptor retaining `query_block_size`. Exact VSA support follows this order:

1. Directly use an existing 256x128 sparse kernel when adjacent 128-query VSA rows are identical or
    when the policy natively emits 256-query rows, as current xDiT Sparge recipes do.
2. Add a manifest-selected 128x128 PyISA sparse kernel for arbitrary VSA rows. This is the primary
    exact compatibility path and must be benchmarked because reducing the query tile changes the
    eight-wave load/compute balance.
3. Optionally add a 256x128 union kernel carrying per-half membership bits if VSA masks have enough
    overlap to make union overcompute cheaper than the 128x128 kernel. This is a separate optimized
    ABI, not the default conversion.

The public compatibility helper may accept the existing `(block_lut, block_counts)` tensors,
decode them to an `AttentionBlockSparseLut`, and call the same `fmha_v4_packed` executor. It must
not maintain a second Q/K/V quantization or code-object dispatch stack.

VSA-specific ordered-prefix optimizations, such as processing high-priority blocks with live
running-max updates and freezing the max for a tail, are optional kernel metadata. They can extend
the ragged descriptor with a per-row `freeze_after` tensor and select a matching manifest row.
Plain VSA compatibility does not require this optimization; AITER's current CK API exposes no
freeze metadata.

## Output ABI Evolution

Existing kernels write BF16 output through the v1 FMHA argument layout. Low-precision-output
kernels require a versioned extension rather than repurposed fields.

A v2 layout reserves explicit slots after the sparse extension for at least:

```text
output scale pointer
output data format
output scale format and mode
output scale strides or contiguous-layout metadata
```

The exact offsets are fixed with the first low-precision-output kernel. Existing v1 binaries
continue to launch with their original argument sizes.

## `torch.compile` Rules

1. Q, K, and V preprocessing are separate custom ops so distributed scheduling can overlap them.
2. The ASM launch is always a custom op, including variants with ordinary dense storage.
3. Custom ops return contiguous backing buffers for exotic K or V layouts. Required views are
   reconstructed only inside the final launch op.
4. Fake implementations return exact public data and scale shapes and dtypes without loading a
   code object.
5. Custom-op names are versioned whenever output shape, packed storage layout, or ABI changes.
6. Compile validation includes allocator churn and a downstream consumer such as
   `output.data.contiguous()`.
7. Sparse LUT creation avoids data-dependent allocations.
8. Public functions, fake implementations, and custom-op declarations use `Optional[T]`, not
    `T | None`. The union-operator annotation style caused a measured `torch.compile` performance
    regression in the current branch. Preserve the existing `aiter.ops.mha` annotation rewrite and
    apply the same convention throughout the new entrypoints.

## Migration

Remove all branch-added custom-kernel wrappers and automatic I8FP8 routing from `aiter.ops.mha`.
The benchmark and xDiT call `aiter.ops.mha_v4` directly. Avoid compatibility aliases unless an
external downstream consumer requires a deprecation window.

Migration is staged:

1. Add format and scale types, `fmha_v4_fwd` manifest, dedicated host launcher, packed
    BF16-output launch op, and fake implementation.
2. Move production Q/K/V preprocessing for the six dense combinations behind MHA v4 custom ops.
3. Add raw-QKV and packed APIs with BF16 tensor output.
4. Move `bench_sage.py` and xDiT callers, then remove custom-kernel logic from `aiter.ops.mha`.
5. In a later PR, add sparse manifest rows, code objects, LUT validation, and sparse launch tests.
6. Later add VSA compatibility and Sparge policy over the shared ragged-LUT executor.
7. Later add the approximate BF16 code object under its distinct identity.
8. Later add the versioned FP8-output ABI; consider other output formats afterward.

## Open Decisions

The first-release public contract is fixed: `aiter.ops.mha_v4`, six dense format combinations,
non-causal MHA, head dimension 128, and plain BF16 tensor output. Remaining implementation choices
that do not change this public contract are:

1. Decide whether the packed expert API is public in the first release or kept private until a
   second caller needs it. The raw-QKV API is required for xDiT either way.
2. Finalize the manifest schema and whether it is generated from a dedicated CSV or represented by
   a small static table for the first six rows. A dedicated CSV is preferred because sparse and
   output-format dimensions are planned.
3. Decide whether Q/K/V preprocessing custom ops live in `aiter.ops.mha_v4.quant` immediately or
   initially reuse implementations from current quant modules behind private wrappers.
4. Attach exact shape, harness, GPU-count, and code-object hashes to the performance baseline.

## Required Validation

- eager and compiled parity for every supported Q/K/V/O combination;
- compiled allocator-churn tests with downstream consumers;
- dense and sparse correctness against a pure BF16 reference;
- sparse LUT validation, including partial KV tails and varied per-query-block counts;
- dispatch tests proving every explicit key resolves to the intended symbol and code object;
- rejection tests for unsupported combinations and descriptor mismatches;
- fixed-input repeated determinism and all-GPU long-context tests for synchronization changes;
- balanced multi-GPU target-shape performance tests after correctness gates pass.
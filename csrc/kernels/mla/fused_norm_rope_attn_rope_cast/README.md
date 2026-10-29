# Fused norm / RoPE / attention / cast

All native implementation files for this operator live in this directory.
The public Python API is `aiter/ops/fused_norm_rope_attn_rope_cast.py`.

| File | Responsibility |
| --- | --- |
| `fused_norm_rope_attn_rope_cast.cu` | Torch-free ctypes entrypoints, tensor validation, argument conversion and dispatch between attention implementations. |
| `attention_32rows.cu` | 32-row KV kernels and launches, including shared-KV FP8 and legacy/BF16 paths. |
| `attention_32rows.cuh` | Private cache readers, tile helpers and pipelines used only by `attention_32rows.cu`. |
| `attention_64rows.cuh` | Shared 64-row KV implementation, including prefill, split attention and split reduction. |
| `attention_64rows_h16.cu` | Native 16-head FP8 attention, also used to fill small H32 split grids. |
| `attention_64rows_h32.cu` | Shared 32-head FP8 path for H32. |
| `attention_64rows_h64.cu` | H64 attention, shared split reducers, and the host dispatcher. |
| `attention_64rows_h128.cu` | H128 instantiations compiled independently from H64. |
| `params.hpp` | KV layout descriptions, both internal argument structures and launch declarations; no device pipeline definitions. |
| `dispatch_policy.hpp` | Shared performance selection rules. |
| `q_norm.cuh` | 16-byte-aligned Q loads and RMSNorm helpers shared by both attention implementations. |

`32rows` / `64rows` means KV rows per tile. The h16/h32/h64/h128 translation
units group implementations by query-head count. H16 uses a native 16-head
pipeline; H32 uses a 32-head tile. These file names do not select
prepared/unprepared or decode/prefill modes.

The 32-row helpers use the `fused_mla_32rows` namespace and
`FusedMla32RowsTraits`. They are shared by BF16 QK/PV and FP8 QK with BF16 PV,
so their names do not encode a fixed input precision or wave count.
The BF16 path remains available through
`fused_norm_rope_attn_rope_cast_bf16`; the main two-source FP8 API quantizes
raw BF16 Q before FP8 QK. The upstream helper attribution stays in the source.

The entrypoint first tries the 64-row/split dispatcher; otherwise it launches
the 32-row implementation. The selection rules stay in `dispatch_policy.hpp`.
The entrypoint does not include either device implementation header.

The module has six HIP translation units. H64 and H128 remain separate so
they can compile in parallel. H16 is compiled separately to preserve H32 raw
performance; shared implementation headers are not duplicated. `aiter/jit/optCompilerConfig.json` lists these files and retains
different compiler options for the 32-row and 64-row implementations.

Execution requires gfx950. Device-body guards allow builds that also target
gfx942; the native entrypoints reject unsupported devices before launch.

## LDS reads in the PV pipeline

The local `lds_read_b128_p` / `lds_read_b128_v` helpers intentionally keep
inline assembly. Opus `smem::load<4>` on i32 elements can express the same
128-bit read, but it exposes the read to compiler scheduling. This pipeline
uses explicit LDS issue ordering and `s_waitcnt_lgkmcnt` thresholds.
A ROCm 7.2.4 comparison retained the ds_read instruction counts but changed
wait scheduling; it did not establish a consistent performance advantage.
The existing assembly and dispatch are therefore retained.

## Public API and private diagnostics

Only `fused_norm_rope_attn_rope_cast`, its `_legacy` wrapper and its `_bf16`
wrapper are exported from `aiter`. None accepts `debug_out`. The underscored
Python wrappers and ctypes entrypoints retain FP32 pre-quantization output
for internal correctness checks. Diagnostic dispatch controls remain internal.
Performance timing calls the public API; diagnostic calls are outside timing.

BF16 Q and internal FP32 diagnostic views accept 16-byte-aligned storage
offsets. Vector readers/stores honor that alignment. Both native entrypoints
reject softmax scales whose FP32 base-2 conversion is nonfinite.
Runtime detection uses the existing shared helpers without adding a device
parameter to `get_gfx_runtime`.

## Validation

Run the same command as CI:

```sh
python op_tests/test_fused_norm_rope_attn_rope_cast.py
```

Direct execution runs 3,230 correctness cases, including ragged inputs,
BF16/FP8 modes, prepared/raw/normalized Q, graph replay, dispatch boundaries,
storage offsets and invalid arguments. It then runs the public-API benchmark
matrix against an independent Torch reference and prints Markdown tables.
No pytest collection is required. The default matrix has 144 cases covering
T=1/128/1024, H16/H32/H64/H128, topk=512/1024/2048, prepared/unprepared and
decode/prefill, with a 65,536-row prefix pool and a 128-row extend window.
Prefill extends are causal; decode uses independent request histories.

The gfx942+gfx950 build and full gfx950 regression include the H16/H32
paths, compact split storage and the causal prefix/extend transition.

## Earlier output-consistency failures

Two historical layout-refactor runs each had 3 failures among 2,428 tests:
H64 graph replay and H128 normalized-Q common/general eager comparisons.
These were bytewise output mismatches, not yet traced to a root cause.

Fresh full runs with the preceding binary, the exact layout-refactor binary
and the later shared-helper binary each passed all 2,428 tests. The six
historically failing cases also passed with each comparison phase expanded
to 32 iterations. These reruns did not reproduce the issue; the alignment
and scale-validation fixes above must not be taken as proof of its cause.

## Automatic split selection

The default `num_splits=0` uses the same batch/grid and index-capacity rules
for prefill and decode. Small prefill batches can use split attention; large
grids retain the single-kernel paths. `num_splits=1` explicitly selects one
kernel. Explicit multi-split counts and caller-supplied workspace continue
to require `is_decode=True`; automatic prefill allocates its own scratch.
This changes prefill scheduling without changing the decode rules or CSR
semantics. It is not an ATOM end-to-end performance claim.

## Small decode split reduction

Output-only decode uses parallel split statistics on small grids. The smallest
grids distribute each head across a CTA; larger split=8 grids keep coalesced
partial reads and divide the output dimensions across four CTAs. These reducers
share the existing partial layout and are instantiated once for both head counts.
Optional diagnostic/statistic outputs and larger grids use the existing reducer.
Very small attention grids allow more work splits to shorten each CTA's KV loop.
Split=1 still writes the final output inside attention; split>1 uses attention
and one reduction launch. Raw Q preparation stays inside attention. The kernels
build with standard ROCm 7.2.4 and do not use register-pinning compiler extensions.

The two-source FP8 API supports H16/H32. H16 uses native 16-head QK/PV tiles;
H32 keeps the shared 32-head pipeline. Both preserve compact workspace
`[T, num_splits, H, 516]`. Raw Q preparation and final output remain fused;
no head-padding or reorder launch is added. Split selection uses the existing
grid/work rule rather than a per-shape table.

For split H32 attention, two 16-head CTAs are used when the resulting grid fits
within 256 CTAs. Larger grids retain the 32-head tile. Both paths write the same
partial workspace layout and use the same reducer. H16/H32 output-only reducer
instantiations live with the shared reducers in `attention_64rows_h64.cu`; this
keeps their extra code out of the H32 attention translation unit. Diagnostic
outputs continue to use the general reducer.

H16 automatic split selection uses the existing work and grid limit directly,
including non-power-of-two counts. Other head counts retain power-of-two
rounding. Explicit split requests and the 32-split limit are unchanged.

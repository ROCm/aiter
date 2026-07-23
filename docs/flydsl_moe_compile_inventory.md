# FlyDSL standard two-stage MoE compile inventory

## Status and scope

This document records the standard host path and its CPU-only compile-request
resolver on `zhimding/refactor_aot`. It is a behavioral baseline and
CompilePlan inventory. It is **not** a Manifest, persistence format, global
deduplication policy, or packaged-artifact design.

The inventory covers the standard
`fused_moe -> fused_moe_ -> fused_moe_2stages` path and every FlyDSL artifact
reachable from it:

- optional FlyDSL MoE sorting, including oneshot and both multiphase launchers;
- a FlyDSL stage1 or stage2 selected independently by a tuned row or a host
  heuristic;
- stage1 split-K activation/quantization helpers;
- FlyDSL activation epilogues reached from a CK-Tile stage1;
- stage2 plain and expert-parallel masked reduction.

It intentionally excludes the one-stage assembly path, the separate
`flydsl_mxmoe_*`/`aiter.aot.flydsl.mxfp4_moe` port, gfx1250 grouped MoE,
Triton MoE, and standalone kernel tests. Mixed rows are still in scope: for
example, CK stage1 + FlyDSL stage2 and FlyDSL stage1 + CK-Tile or Opus stage2.

Primary implementation references:

- [`aiter/fused_moe.py`](../aiter/fused_moe.py)
- [`aiter/ops/flydsl/moe_kernels.py`](../aiter/ops/flydsl/moe_kernels.py)
- [`aiter/ops/flydsl/moe_sorting.py`](../aiter/ops/flydsl/moe_sorting.py)
- [`aiter/ops/flydsl/kernels/moe_sorting_kernel.py`](../aiter/ops/flydsl/kernels/moe_sorting_kernel.py)
- [`aiter/aot/flydsl/moe.py`](../aiter/aot/flydsl/moe.py)

## Runtime call graph

```text
fused_moe
└─ fused_moe_                         # torch_compile_guard custom-op body
   ├─ derive M/topk/E/model_dim/inter_dim, quant dtypes and layout
   ├─ get_2stage_cfgs
   │  ├─ look up tuned_fmoe CSV row
   │  ├─ decode kernelName1/kernelName2 into MOEMetadata
   │  └─ otherwise apply host fallback heuristics
   ├─ moe_sorting
   │  ├─ _flydsl_moe_sorting         # opt-in and gate-compatible only
   │  │  └─ flydsl_moe_sorting_fwd
   │  │     ├─ resolve_moe_sorting_compile_plan
   │  │     ├─ AotBackend.resolve_aot(selected concrete unit)
   │  │     └─ moe_sorting_flydsl
   │  │        └─ tensor_shim._run_compiled -> strict-load/JIT selected launcher
   │  ├─ _moe_sorting_impl -> Opus or CK sorting
   │  └─ specialized FLAT/output_aux paths
   └─ fused_moe_2stages
      ├─ prepare/quantize stage1 input
      ├─ metadata.stage1
      │  ├─ _flydsl_stage1_wrapper
      │  │  └─ flydsl_moe_stage1
      │  │     ├─ compile_flydsl_moe_stage1
      │  │     │  ├─ compile_mixed_moe_gemm1  # fp4/fp8 weights
      │  │     │  └─ compile_moe_gemm1        # bf16 x int4
      │  │     ├─ _run_compiled -> flyc.compile
      │  │     └─ optional split-K FlyDSL activation/quant helper
      │  ├─ cktile_moe_stage1
      │  │  └─ optional interleaved FlyDSL activation epilogue
      │  └─ CK/HIP/assembly stage1
      ├─ prepare or consume the inter-stage quantized tensor
      └─ metadata.stage2
         ├─ _flydsl_stage2_wrapper
         │  └─ flydsl_moe_stage2
         │     ├─ compile_flydsl_moe_stage2
         │     │  ├─ compile_mixed_moe_gemm2  # fp4/fp8 weights
         │     │  └─ compile_moe_gemm2        # bf16 x int4
         │     ├─ _run_compiled -> flyc.compile
         │     └─ reduce mode only:
         │        _run_moe_reduction
         │        └─ compile_moe_reduction
         │           └─ _run_compiled -> flyc.compile
         └─ CK, CK-Tile, Opus, or assembly stage2
```

Builder construction and artifact compilation are distinct. Sorting now emits
one concrete `CompileUnit` for the selected launcher: oneshot, combined
P0v2+P23, or combined ClearWS+P0+P1+P23. The private multiphase factory still
constructs its internal launcher tuple, but the registered compile operation
returns exactly the launcher passed to FlyDSL compile/cache/load. The CPU
baseline records this agreed builder-request boundary, not private cache keys.

## Dispatch and trigger conditions

### Two-stage configuration selection

`get_2stage_cfgs()` owns backend selection.

1. It builds a lookup key from:
   `gfx`, CU count, padded token tier, `model_dim`, `inter_dim`, local expert
   count, routed `topk`, activation, output/activation/weight dtypes,
   quantization type, gate/up shape, and `doweight_stage1`.
   Below 32768 tokens the tier is `nextPow2(M)`; 32768 and 131072 are explicit
   large tiers. In expert-parallel mode the always-masked fake top-k slot is
   removed from the CSV lookup key.
2. It reads `AITER_CONFIGS.AITER_CONFIG_FMOE_FILE`, first trying an exact
   `act_type` row and then the activation-disabled fallback. Rows tagged
   `_tag=flydsl_fallback` are excluded from runtime lookup. A missing large
   tier may fall back to a smaller configured tier.
3. A `kernelName1` or `kernelName2` beginning with `flydsl_` selects the
   corresponding FlyDSL wrapper only when FlyDSL is available. Stages are
   selected independently; the other stage may be CK, CK-Tile, Opus, or
   assembly/HIP.
4. Without a usable tuned row, current heuristics can synthesize FlyDSL names:
   - per-1x32 bf16-activation/int4-weight (`a16wi4`) selects FlyDSL stage1 and
     atomic stage2 when FlyDSL is available;
   - the MX path requires bf16/fp16 output, per-1x32 quantization, fp4/fp8
     activation and weight dtypes, preshuffled gate/up weights (`use_g1u1`),
     `doweight_stage1=False`, FlyDSL availability, and either SwiGLU or
     `AITER_FLYDSL_FORCE=1`. It selects FlyDSL stage1 and atomic stage2.
     `AITER_FLYDSL_FORCE` currently defaults to `1`, so this heuristic is not
     limited to SwiGLU.

The `flydsl_mxmoe_*` name family is intercepted earlier by its separate port
and is outside this inventory.

### Optional FlyDSL sorting

FlyDSL sorting is reached only when all of these are true:

- `AITER_USE_CK_MOE_SORTING != 1`;
- `AITER_USE_FLYDSL_MOE_SORTING == 1`;
- FlyDSL is available;
- no local-top-k-id result is requested;
- the route is not FLAT;
- the specialized `output_aux` route is not requested;
- `moe_sorting_dispatch_policy == 0`.

Otherwise standard sorting defaults to Opus, or to CK when
`AITER_USE_CK_MOE_SORTING=1`. Requesting local top-k IDs or `output_aux`
also forces a non-FlyDSL sorting path. Expert masking is supported by FlyDSL
sorting and becomes the compile-time `has_mask` variant.

Let `T_bucket` be the explicit maximum token count; normal runtime uses
`topk_ids.shape[0]`. A dynamic `num_local_tokens` tensor is only read while
packing the selected launch and must not exceed the bucket. `E` is the global
routed expert count. With `BLOCK_SIZE=256`:

```text
sub_tokens =
  floor((LDS_ints / 2 / (E + 1) - 2) / 8) * 8

ONESHOT_MAX_T =
  min(sub_tokens, max(16, 256 // max(topk, E // 8)))
```

The LDS size is 163840 bytes on gfx95*, 65536 bytes on gfx94* and the
conservative fallback.

- `T_bucket <= min(sub_tokens, ONESHOT_MAX_T)`: compile and launch the oneshot
  artifact. Its launcher bound is `max(T_bucket, 8)` rounded to a multiple of 8.
- above that threshold and `T_bucket <= 2048`: compile and launch the combined
  multiphase `P0v2 + P23` artifact.
- `T_bucket > 2048`: compile and launch the combined four-kernel
  `ClearWS + P0 + P1 + P23` artifact.

Multiphase requests use `k4_block=256` for `E <= 256`; for
`256 < E <= 512` they use 512 through `T=8192` and 256 above it; larger
expert counts use 256. The other static request fields are `E`, `topk`,
sorting `unit_size`/block-M, and `has_mask`. The provider requires an explicit
`RocmTarget`; gfx94, gfx95, and supported RDNA-family LDS capacities and wave
size are resolved by kernel-owned pure helpers. Missing or unsupported
architecture metadata is an error.

Sorting also owns zero-initialization for atomic stage2. For unmasked reduce
mode it receives an empty `(0, 0)` `moe_buf`, so the sorting kernel has no
output-zeroing work.

## Stage1 compile requests

### Primary GEMM

`_flydsl_stage1_wrapper()` resolves the full kernel name through
`get_flydsl_kernel_params()` and calls `flydsl_moe_stage1()`.
The registry is pre-populated with supported named variants carrying complete
static parameter dictionaries; its fallback suffix parser overlays `_kwN`,
`_fp4`, `_fp8`, and `_sbmN` fields on a known base name.
The stage entry derives the exact tensor dimensions and calls
`compile_flydsl_moe_stage1()` with:

- shape: `model_dim`, `inter_dim`, local `experts`, and runtime `topk`;
- tile/type: `tile_m`, `tile_n`, `tile_k`, `a_dtype`, `b_dtype`,
  and effective `out_dtype`;
- route weighting: `doweight_stage1 = (sorted_weights is not None)`;
- activation/layout: `act`, `gate_mode`, and compile-time bias presence;
- scheduling: `persist_m`, `use_async_copy`, `k_batch`, `waves_per_eu`,
  `b_nt`, `a_scale_one`, `xcd_swizzle`, and `k_wave`;
- padding: `model_dim_pad` and `inter_dim_pad`.

The wrapper hard-codes `use_async_copy=True`; stage1 `persist_m` normalizes to
1 unless a direct caller overrides it. Bias disables wrapper-level padding via
`_get_padding_for_flydsl()`. `swiglu_limit` is normalized to 7.0 for an unset
SwiGLU limit and `+inf` for an unset SiLU limit, but remains a runtime scalar
rather than a builder parameter.

For fp4/fp8 weights, the next builder is
`compile_mixed_moe_gemm1()`. For bf16 activations with int4 weights it is
`compile_moe_gemm1()` with `in_dtype="int4_bf16"`, group size 32, bf16 scale,
and CShuffle enabled automatically only for split-K.

For non-split-K `_fp4`/`_fp8` stage1 names, activation, optional route
weighting, quantization, and sorted E8M0-scale emission are fused into the
primary GEMM artifact.

### Split-K auxiliary activation and quantization

`k_batch > 1` makes the primary FlyDSL GEMM write bf16/fp16 gate/up partials
into a zeroed `[token, topk, 2 * inter_dim]` buffer. Therefore the primary
builder receives the base output dtype, even when the requested final stage1
output is fp4 or fp8. Post-processing is selected as follows:

| Runtime predicate | Post-processing artifact |
| --- | --- |
| interleaved gate/up and final fp4/fp8 | `build_silu_and_mul_fq_module(inter_dim, topk, quant_mode="fp4" or "fp8", gui_layout=True, act, enable_bias)` |
| interleaved gate/up and no final quantization | the same builder with `quant_mode="none"` and `gui_layout=True` |
| separated gate/up and final fp4 | the same builder with `quant_mode="fp4"` and `gui_layout=False` |
| every other split-K case | CK/HIP `silu_and_mul`, `swiglu_and_mul`, or bias variant at runtime; no auxiliary FlyDSL artifact |

The FlyDSL helper owns activation, split-K result consumption, optional bias,
optional fp4/fp8 quantization, and sorted scale output. Split-K bias requires
`topk_ids`, because `sorted_token_ids` encodes token/slot but not expert ID.
The bias-presence boolean is a compile parameter; the bias shape and
`swiglu_limit` are runtime arguments.

Under `COMPILE_ONLY=1`, the non-FlyDSL plain split-K activation is deliberately
skipped after the primary GEMM compile.

## CK-Tile stage1 with a FlyDSL epilogue

`cktile_moe_stage1()` itself calls the CK-Tile/HIP
`aiter.moe_cktile2stages_gemm1`; it is not a FlyDSL GEMM.
A separate post-activation step exists only when `split_k > 1`.

The result is treated as interleaved when
`post_activation_layout="interleaved"`, or when
`post_activation_layout="auto"` and `w1.dtype` is
`torch.float4_e2m1fn_x2`. `"standard"` explicitly disables the interleaved
route.

- Interleaved SwiGLU calls
  `flydsl_swiglu_and_mul_interleaved()` and builds
  `build_swiglu_and_mul_module(inter_dim)`. Its request does not include
  `topk`.
- Interleaved SiLU calls
  `flydsl_silu_and_mul_interleaved()` and builds
  `build_silu_and_mul_fq_module(inter_dim, topk, quant_mode="none",
  gui_layout=True, act="silu", enable_bias=False)`.
- Interleaved GELU is implemented with PyTorch tensor operations, not FlyDSL.
- Standard-layout SiLU/SwiGLU/GELU and bias variants call AITER activation
  operators, not FlyDSL. Interleaved split-K bias is rejected.
- Non-split-K CK-Tile keeps activation in the CK-Tile epilogue and does not
  reach either FlyDSL helper.

The current CSV AOT collector is intentionally broader than this runtime
predicate: any `kernelName1` beginning with `cktile_` emits an epilogue job,
without checking `split_k` or the runtime weight layout, and maps every
non-SwiGLU activation to SiLU. That is an old-path over-approximation, not a
new resolver contract.

## Stage2 and reduction compile requests

`_flydsl_stage2_wrapper()` resolves the stage2 kernel name and forwards its
registered parameters to `flydsl_moe_stage2()`. The primary
`compile_flydsl_moe_stage2()` request contains:

- `model_dim`, `inter_dim`, local `experts`, runtime `topk`;
- `tile_m`, `tile_n`, `tile_k`, activation/weight/output dtype tags;
- `doweight_stage2 = (sorted_weights is not None)`;
- `accumulate`, derived from configured mode and `return_per_slot`;
- `persist_m`, `sort_block_m`, `waves_per_eu`, `use_async_copy`,
  `cu_num_mul`, `b_nt`, and `xcd_swizzle`;
- model/intermediate padding and compile-time bias presence.

`doweight_stage1` determines routing-weight ownership: stage1 receives
`sorted_weights` when true; otherwise stage2 receives them. A configured
`mode="reduce"` makes `accumulate=False`. Atomic mode uses
`accumulate=True`, writes directly into the sorting-zeroed `[T, model_dim]`
buffer, and has no reduction artifact. `AITER_FLYDSL_FORCE_REDUCE=1` can
override an atomic name at runtime.

`persist_m` is runtime-derived:

- explicit `persist=True` -> `-1` (persistent CU grid);
- explicit `persist=False` -> 4 when `m_blocks > 256`, otherwise 1;
- unspecified -> `-1` when `m_blocks > 256`, otherwise 1;
- fp8 activation forces 1.

For reduce mode, GEMM2 writes a contiguous
`[token, topk, model_dim]` temporary and `_run_moe_reduction()` emits a second
FlyDSL request:

```text
compile_moe_reduction(
    topk=runtime_topk,
    model_dim=model_dim,
    dtype_str="f16" | "bf16" | "f32",
    use_mask=(expert_mask is not None),
    num_experts=expert_mask.numel() if masked else 0,
)
```

- Plain reduction sums every top-k slot.
- Expert-parallel masked reduction additionally gathers
  `expert_mask[topk_ids[token, slot]]` and sums only valid slots. Both
  `expert_mask` and `topk_ids` are normalized to contiguous int32 runtime
  inputs. Missing `topk_ids` is an error.
- An unsupported output dtype may use `torch.sum` only for the unmasked case;
  masked reduction rejects it. The standard output dtypes above use FlyDSL.

The runtime CSV lookup removes the EP fake slot from its key, but the actual
stage2 and reduction requests receive the runtime `topk`, including that slot.

## Parameter ownership summary

| Owner/source | Parameters and decisions |
| --- | --- |
| User/model tensors | `M`, runtime `topk`, local `E`, dimensions, tensor dtypes/layout, scales, biases, `expert_mask`, `num_local_tokens`, `swiglu_limit` |
| `fused_moe_` host logic | activation/quant dtype remapping, padded token tier, sorting backend eligibility, route-weight stage, inter-stage quantization, empty vs zeroed `moe_buf` |
| Tuned CSV row | `block_m`, `ksplit`, `kernelName1`, `kernelName2`; row match also depends on architecture, shape, activation, dtypes, quant type, gate/up shape, and route-weight stage |
| Kernel-name registry | stage, tile sizes, dtype tags, stage1 `k_batch`/gate mode/schedule knobs, stage2 atomic/reduce mode, `sort_block_m`, persistence and schedule variants |
| Stage entry functions | exact compile dimensions, presence booleans, effective split-K output dtype, persistent-mode threshold, argument packing, auxiliary helper selection |
| Sorting host code | oneshot threshold, `max_tokens` bucket, multiphase family, workspace size, `k4_block`, mask variant |
| Environment/runtime architecture | FlyDSL availability, sorting backend flags, force-reduce/heuristic debug flags, CU count, ROCm architecture and wave size |
| Current AOT adapter | Explicit target, declared launch ABI materialization, CSV stage enumeration, explicit sorting cases, and compile-only execution; it does not own Manifest policy |

Pointer signatures do not encode scale or routing tensor lengths, and tensor
**presence** can still select compile-time booleans. One host-level exception
is stage2: `sorted_expert_ids.shape[0]` contributes to `m_blocks`, which can
change the derived `persist_m` request. Sorting also uses its token bucket,
expert count, top-k, mask presence, unit size, and architecture to select an
artifact.

## FlyDSL and non-FlyDSL boundary

- **FlyDSL:** selected standard stage1/stage2 GEMMs, split-K
  `silu_and_mul_fq`, CK-Tile interleaved SiLU/SwiGLU epilogues, stage2
  reduction, and opt-in sorting.
- **CK/HIP:** opt-in CK sorting, CK/CK-Tile stage GEMMs, ordinary split-K
  activation helpers, and the `get_hip_quant` family used before stage1 and
  between stages when quantization is not fused by FlyDSL stage1.
- **Opus:** default standard sorting when CK sorting is not forced, local-ID
  and some auxiliary sorting routes, plus tuned Opus stage2 rows.
- **Host/PyTorch:** tensor allocation/view logic and CK-Tile interleaved GELU.

Backend mixing is expected. A FlyDSL stage1 does not imply a FlyDSL stage2,
and FlyDSL sorting is an independent opt-in.

## ABI-driven AOT route and baseline contract

[`aiter/aot/flydsl/moe.py`](../aiter/aot/flydsl/moe.py) constructs the same
explicit sorting, Stage1, and Stage2 `OperationCase` values as runtime and
compiles each operation plan's `CompileUnit` projection:

1. It enumerates CSV rows and resolves `flydsl_` kernel names through the
   current registry. It also emits CK-Tile epilogue jobs and both supported
   bias-presence variants for eligible fp4-weight rows.
2. It derives an explicit `RocmTarget` from `cu_num` and creates the shared
   Aiter `AotBackend`.
3. Stage1 and Stage2 providers bind existing compile-wrapper signatures and
   defaults. Each unit carries a stable versioned op ID and exact launch ABI.
4. `compile_moe_sorting_case()` accepts explicit token bucket, global expert
   count, top-k, unit size, mask state, and optional path/k4 assertions. It does
   not infer sorting from ordinary tuned rows.
5. The backend materializes tensor/pointer/scalar/stream metadata and invokes
   the FlyDSL launcher in compile-only mode. No FakeTensor, runtime tensor
   allocation, stream lookup, or full sorting/stage host call is involved.

Stage2's kernel-owned persistence helper accepts either the real runtime
routing-block count or the AOT token bucket and standard routing capacity.
Both paths therefore share the automatic threshold and fp8 override rather
than mirroring the `m_blocks > 256` decision. CSV reduce rows emit the plain
reduction unit. Masked EP reduction remains an explicit provider operation
case requiring top-k-ID semantics and a positive global expert count.

`MoeOperationCase` composes explicitly optional sorting, Stage1, and
Stage2/reduction nodes with ordered dependencies. `MoeCompilePlanCase` remains
a compatibility projection. Neither is a Manifest, persistence format, global
deduplication policy, or packaged artifact source.

The CPU baseline recorder enters the runtime stage and sorting hosts with
test-owned fake inputs while mocking compile/launch/CUDA boundaries. It
normalizes complete public builder kwargs plus a stable trigger ID and compares
the exact request set with the checked-in gfx950 golden. Separate provider
tests compare ordered Stage2 units, op IDs, bound builder requests, and ABIs
against the same golden. Neither test snapshots FlyDSL-private cache keys.

### Checked-in trigger matrix

The golden's stable trigger IDs map to this inventory as follows. A trigger can
emit more than one request when a stage builds a primary GEMM plus an auxiliary
artifact. Each sorting trigger emits only its selected concrete builder.

- Stage1 primary builders:
  `stage1.main.non_split.bias.route_weighted`,
  `stage1.int4.splitk`,
  `stage1.splitk.fp4.silu.separated`,
  `stage1.splitk.fp8.swiglu.interleaved.bias`, and
  `stage1.splitk.none.silu.interleaved`.
  These cover mixed-MX and standard int4 GEMM1, route weighting, primary bias,
  both gate layouts, and all three split-K post-activation quant modes.
- CK-Tile FlyDSL epilogues:
  `cktile.epilogue.silu` and `cktile.epilogue.swiglu`.
- Stage2:
  `stage2.atomic.bias`, `stage2.int4.atomic`,
  `stage2.reduce.plain`, `stage2.reduce.plain.large_auto_persist`, and
  `stage2.reduce.masked_ep`.
  These cover mixed-MX and standard int4 GEMM2, atomic and reduce modes, the
  automatic persistence threshold, and plain/masked reduction.
- Sorting:
  `sorting.oneshot.unmasked`, `sorting.oneshot.masked`,
  `sorting.multiphase.p0v2.unmasked.e384`, and
  `sorting.multiphase.4k.masked`.
  The E384 case records the reachable `k4_block=512` multiphase specialization;
  the E256 cases retain `k4_block=256`.

Requests are sorted by stable ID before serialization, dictionaries are emitted
with sorted keys, and non-finite floats use explicit string spellings. Repeated
recordings therefore produce byte-identical canonical JSON without depending
on import or scenario declaration order.

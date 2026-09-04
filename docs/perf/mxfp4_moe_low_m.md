# Low-M MXFP4 fused-MoE paths

Decode-shaped fused-MoE calls (`M <= 64`, one graph replay per token step) are
latency bound rather than throughput bound. The generic two-stage path pays a
fixed cost that does not shrink with `M`: an `E`-wide sort whose padded extent
is dominated by `E * block_m`, and a launch grid sized for that extent. This
document describes the two low-M paths that avoid that cost, how a tuned row
selects them, and the measured result.

Everything here is driven by ordinary tuned-config metadata. There is no path
registry, no `path` CSV column, no model-name dispatch, and no M-threshold
environment variable.

## Paths

### Padded-row quant launch bound (all paths)

For `R = M * topk` route slots, `A = min(E, R)` occupied blocks and sort block
`BM`, the distribution-independent upper bound on padded sorted rows is

```text
BM * (A + floor((R - A) / BM))
```

The fused quant/sort launcher uses `min(legacy_extent, padded_row_bound)`, so a
caller that supplies a trustworthy host-side expert bound can only shrink the
grid, never grow it. Callers that omit the bound keep the legacy extent; buffer
allocation, layout and ABI are unchanged.

At `M=1, E=896, BM=32` roughly 98% of the legacy workgroups were empty.

Entry points: `moe_quant_padded_rows_upper_bound` (`csrc/include/quant.h`),
`fused_dynamic_mx_quant_moe_sort_hip_bounded` (`aiter/ops/quant.py`), and the
`routing_num_experts` argument threaded through `aiter/fused_moe.py`.

### Direct M1 executor (`M == 1`)

At `M == 1` sorting buys nothing: every route slot is a distinct
`(token, expert)` pair and, for any realistic `E`, almost every one lands on a
distinct expert. The direct executor consumes raw `topk_ids`/`topk_weights`,
binds one route per workgroup, and accumulates into a pre-zeroed output with
atomics.

Three kernels per call:

1. BF16 -> MXFP4 activation quantization fused with output zeroing
2. route-centric stage1 over raw `(token, expert)` routes
3. weighted atomic stage2

Selected by `flat=1, run_1stage=0` plus the canonical kernel-name pair

```text
flydsl_moe1_direct_m1_afp4_wfp4_bf16_t32x32x256_w4_kw2_fp4
flydsl_moe2_direct_m1_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32
```

`flat=1, run_1stage=1` keeps its pre-existing one-stage meaning. A flat
two-stage row that does not name the canonical pair is rejected.

The quant/zero fusion is a private pybind target
(`_dynamic_per_group_scaled_quant_fp4_direct_m1_internal`); it is the canonical
quant template plus an output-zero loop, and is not exported as a general API.
Splitting it back into four kernels costs about 7.5% at graph level.

Implementation: `aiter/ops/flydsl/moe_direct_m1.py`, with route addressing in
`mixed_moe_gemm_2stage_common.py` and `mxmoe_dispatcher.py`. Both `silu` and
`situv2` stage1 activations are supported.

### BM16 inline-sort two-stage (`2 <= M <= 32`)

For small but non-unit `M` the sort is still worth doing, but at `block_m=16`
rather than 32, with activation quantization folded into stage1 so the frontend
collapses to a single fused sort/zero kernel. Three kernels per call: fused
sort/zero, BM16 stage1 with inline quant, BM16 atomic stage2.

Selected purely from metadata: `run_1stage=0`, `flat=0`, `output_aux`,
`fuse_quant=fp4`, an `inline_quant` stage1 name, and a stage2 whose
`sort_block_m` matches stage1 `BM`. `_is_mxfp4_inline_sort` in
`aiter/fused_moe.py` performs that check; there is no name allowlist.

The path stops at the `token=32` bucket. Stage1 inline quant re-quantizes the
`BM` rows of every block, so its cost scales with padded rows while the saving
(two frontend launches, about 8 us) is fixed. At `E=896, topk=16` the two cross
near `M=57`; since a bucket must be chosen for its worst case, `token=64` keeps
the legacy frontend.

## Dispatch and fallback

`aiter/ops/flydsl/mxfp4_moe_capability.py` holds what the two paths share: the
`MoeCall` invocation record, `check_a4w4_lowm` (dtype, quant type, gate mode,
activation, g1u1 with two-sided preshuffle, e8m0 scale extents, contiguity and
device, and rejection of EP / bias / explicit activation scales / padding), and
kernel-name accessors.

Validation happens in two places:

- **Static**, against the shape key alone, inside `get_2stage_cfgs`. A tuned
  row that cannot apply is dropped and the lookup falls through to the default
  heuristics.
- **Runtime**, against the actual `MoeCall`, in `_fused_moe_impl`. If a
  selected fast row cannot run this invocation, the metadata is re-resolved
  with that family disabled.

Neither ever raises: an unsupported input degrades to the legacy path with a
warning. `op_tests/test_moe_flat_dispatch.py` covers the rejection cases.

## Tuned-row policy

`get_padded_M` is `nextPow2`, so a tuned `token` row covers a bucket:
`token=32` serves `M=17..32` and `token=64` serves `M=33..64`. A row must be
chosen for its whole bucket, not for its nominal `M`.

Current MXFP4 rows:

| bucket | path |
|---|---|
| `M == 1` | direct M1 |
| `M = 2..16` | BM16 inline-sort |
| `M = 17..32` | BM16 inline-sort |
| `M = 33..64` | legacy two-stage, `t32x64x256` stage1 |
| `M > 64` | legacy |

The `token=1` row of every shipped MXFP4 tuned config uses the direct pair.

CSV timing fields are kernel medians, not graph latency: `us1` and `us2` are
the stage1/stage2 kernel durations and `us = us1 + us2`. Sort/quant and host
overhead are excluded by construction.

## Measured result

Kimi-K3 A4W4 SiTUv2, `E=896`, `topk=16`, `H=3584`, TP8 local `I=384`, BF16
in/out, MXFP4 per-1x32, gfx950 / MI355X. Compared against the same tree with
the pre-existing tuned rows, under HIP Graph, 9 alternating AB/BA rounds of 300
replays each, over 7 route distributions.

| M | geomean | worst |
|---|---|---|
| 1 | 1.403x | 1.391x |
| 2 | 1.207x | 1.188x |
| 4 | 1.238x | 1.186x |
| 8 | 1.129x | 1.110x |
| 16 | 1.162x | 1.125x |
| 17 | 1.142x | 1.077x |
| 24 | 1.102x | 1.029x |
| 32 | 1.059x | 0.966x |
| 40 | 1.018x | 1.011x |
| 48 | 1.016x | 1.009x |
| 64 | 1.012x | 1.003x |

Overall geomean 1.140x, i.e. 12.3% end-to-end latency reduction across the
`M <= 64` matrix. Cosine similarity against a BF16 oracle is unchanged.

The single sub-1.0 cell is `M=32` under a synthetic route that assigns all 512
slots to 512 distinct experts; real grouped-topk routing at `M=32` produces
about 76.5% distinct experts, where the same configuration gains 3-7%.
Reverting the `token=32` row to legacy would cost `M=17..31` a 3-14% gain.

Direct M1 was measured against the tuned `token=1` row of every other shipped
MXFP4 config (11 shapes across three models, 3 routes each): geomean 1.189x
with no shape regressing. The margin shrinks as `inter_dim` grows (1.23x at
`inter=256`, 1.02x at `inter=2048`) because weight traffic comes to dominate.

## Headroom

At `M=64` the graph is 130.1 us stage1 + 67.3 us stage2 + 22.5 us quant/sort +
3.8 us launch gap. The two GEMMs are 88% of it, at 83% and 82% of HBM peak over
a weight working set fixed by the routing (618 of 896 experts active), and the
tuned rows are rank 1 of 96 and rank 3 of 144 in their candidate sweeps. So
even a free frontend would only be 1.13x, and the GEMMs are not the remaining
opportunity. What is left:

- **Quant/zero fusion into stage1.** Each route-centric block would redundantly
  quantize the same activation row, costing about 0.26 us of extra (L2-resident)
  reads while removing a kernel and its launch gap. Measured ceiling by gating
  the dispatch out: 1.33x on top of the current `M=1` result. It requires
  porting the DPP/e8m0 inline-quant sequence into the raw-MLIR stage1 emitter
  shared by every other tuned config, so it belongs in its own change.
- **Launch gap at low M.** 5.1 us of the 20.4 us `M=1` graph is inter-kernel
  gap; the fusion above removes most of it.

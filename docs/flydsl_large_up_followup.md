# Exact large-up HTI: nine-period loop and LDS-only output-tail waits

## Scope and deployed change

Parent: `231edf879f25270855b0bc9cb2acb68195724fa5`. This continuation optimizes
**M16384,N7168,K3584**, BF16 input/output, row-major A and K-contiguous weight,
no bias/scaling/preshuffle, using the already-selected gfx950 HTI policy:
256x256x64, stage2, split-K1, waves2x4x1, group4.

This is not another WS promotion. The regressing small-up WS CSV promotion
remains reverted. The other three requested shapes retain their previous
device code and CSV rows. The four-shape task is not fully solved.

Two changes are applied together under the existing exact-shape/config-derived
`n_grouped_loads` flag:

1. Expand nine two-tile HTI periods per loop iteration. K3584/BK64 has 56
   tiles: 54 steady-state tiles are exactly three iterations of 18 tiles,
   followed by the unchanged two-tile drain. DMA, MFMA and workgroup rendezvous
   ordering and each output's K-addition order are preserved.
2. At the three C-only epilogue boundaries, wait for `lgkmcnt(0)` instead of
   `vmcnt(0)`, then execute the same workgroup barrier. The final K drain
   already retires all A/B DMA before the final tile's computation. The
   output phase depends on LDS C stores/reads, not on completion of disjoint
   global output stores. Those stores remain part of the kernel and are
   complete when its dependent consumer executes; none is omitted.

All other policies retain the original loop and tail. Callers cannot force
the internal flag through `user_kwargs`. `16383/16385` neighbors and altered
K, waves, hints, bias/output type, group and split configurations do not
select it. Runtime strides remain supported, not hard-coded.

The device name suffix changes from `_ng4sk` to `_ng4sk_u9lg` to invalidate
old JIT binaries. The stable CSV kernelName, solution83 and splitK1 remain
unchanged because they already select this policy. Exactly one model CSV
row's measured `us/tflops/bw` is refreshed.

## Matched before/after microseconds

Both sides are complete `aiter.tuned_gemm.gemm_a16w16_out` calls. The before
device module is copied from the parent; after is the integrated repository
source, not a prototype. Implementation selection is outside timing with no
per-call mock. Each pair uses the same A/B/output addresses. No output or
input-transform cache is used.

Hardware: two MI355X gfx950/256-CU cards, A=BDF0000:05:00.0 (HIP1) and
B=BDF0000:15:00.0 (HIP3), selected with zero observed graphics activity.
This does not guarantee exclusive ownership throughout the run. No clock,
power or other-user workload manipulation.

Six fresh processes:3421/3422/3423 on A and3431/3432/3433 on B,16 alternating
AB/BA rounds per shape/mode, all four requested shapes tested.
- `rotation`: standard run_perftest rotating-input profiler mode,
  10 warmups/101 iterations; its reported aggregate is device time, not
  Python end-to-end wall time.
- `rotate_graph8`: eight distinct A/B/output sets captured into one graph,
  5 warmups/30 event-timed replays, divided by8.
- `graph4`: four complete calls using fixed addresses per graph,
  5 warmups/30 event-timed replays, divided by4.
- Entries below are medians of three process medians on each card.
  All individual rounds, including noise and reversals within rounds, are kept.

### Changed shape: M16384,N7168,K3584

| Card | Mode | Before us | After us | Delta us |
|:---|:---|---:|---:|---:|
| 0000:05:00.0 | rotation | 574.393 | 572.379 | -2.014 |
| 0000:05:00.0 | rotate_graph8 | 570.551 | 568.157 | -2.394 |
| 0000:05:00.0 | graph4 | 558.395 | 556.654 | -1.741 |
| 0000:15:00.0 | rotation | 562.498 | 560.907 | -1.591 |
| 0000:15:00.0 | rotate_graph8 | 560.635 | 558.474 | -2.161 |
| 0000:15:00.0 | graph4 | 548.919 | 546.152 | -2.768 |

All six process-level comparisons improve in all three modes:
rotation0.265–0.428%, rotate_graph8 0.343–0.532%, graph4 0.306–0.588%.
This is a **small measured improvement**, not a large speedup or a guarantee
for every invocation/hardware/load condition.

The CSV `us=560.9072` is card B's rotation median (seeds3431..3433).
`tflops=1500.81` and `bw=719.73` GB/s are recomputed from the same measurement.
The previous CSV's historical564.9705us is **not** used as the before baseline;
the matched before value on B is562.498us. Graph latency is not substituted
into the rotation column.

### Every final target process

| Seed | Mode | Before us | After us | Reduction |
|---:|:---|---:|---:|---:|
| 3421 | rotation | 572.573 | 570.502 | 0.362% |
| 3421 | rotate_graph8 | 569.472 | 567.370 | 0.369% |
| 3421 | graph4 | 557.946 | 555.237 | 0.486% |
| 3422 | rotation | 574.393 | 572.800 | 0.277% |
| 3422 | rotate_graph8 | 570.988 | 568.157 | 0.496% |
| 3422 | graph4 | 558.395 | 556.654 | 0.312% |
| 3423 | rotation | 574.644 | 572.379 | 0.394% |
| 3423 | rotate_graph8 | 570.551 | 568.231 | 0.407% |
| 3423 | graph4 | 558.702 | 556.994 | 0.306% |
| 3431 | rotation | 562.332 | 560.132 | 0.391% |
| 3431 | rotate_graph8 | 559.650 | 557.691 | 0.350% |
| 3431 | graph4 | 548.510 | 546.213 | 0.419% |
| 3432 | rotation | 562.498 | 561.007 | 0.265% |
| 3432 | rotate_graph8 | 560.635 | 558.711 | 0.343% |
| 3432 | graph4 | 548.919 | 545.693 | 0.588% |
| 3433 | rotation | 563.317 | 560.907 | 0.428% |
| 3433 | rotate_graph8 | 561.460 | 558.474 | 0.532% |
| 3433 | graph4 | 549.288 | 546.152 | 0.571% |

### Four-shape controls, including unfavorable fluctuations

| Card | M,N,K | Mode | Before us | After us | Implementation |
|:---|:---|:---|---:|---:|:---|
| a | 128,3584,7168 | rotation | 21.081 | 21.114 | identical machine code |
| a | 128,3584,7168 | graph4 | 18.998 | 19.010 | identical machine code |
| a | 128,7168,3584 | rotation | 20.178 | 20.152 | identical machine code |
| a | 128,7168,3584 | graph4 | 19.311 | 19.281 | identical machine code |
| a | 16384,3584,7168 | rotation | 595.255 | 595.607 | identical machine code |
| a | 16384,3584,7168 | graph4 | 587.929 | 587.690 | identical machine code |
| a | 16384,7168,3584 | rotation | 574.393 | 572.379 | changed |
| a | 16384,7168,3584 | graph4 | 558.395 | 556.654 | changed |
| b | 128,3584,7168 | rotation | 21.119 | 21.325 | identical machine code |
| b | 128,3584,7168 | graph4 | 18.949 | 18.906 | identical machine code |
| b | 128,7168,3584 | rotation | 20.286 | 20.188 | identical machine code |
| b | 128,7168,3584 | graph4 | 19.354 | 19.329 | identical machine code |
| b | 16384,3584,7168 | rotation | 587.621 | 587.163 | identical machine code |
| b | 16384,3584,7168 | graph4 | 579.197 | 580.054 | identical machine code |
| b | 16384,7168,3584 | rotation | 562.498 | 560.907 | changed |
| b | 16384,7168,3584 | graph4 | 548.919 | 546.152 | changed |

Unchanged controls' timing fluctuations are not new optimizations. In
particular, some small-down rotation samples are slower although independently
compiled machine code is identical. They are retained rather than labeled
zero-delta measurements. No claim that all four shapes improved is made.

## ISA and thread trace

Independent, separate-cache compilations before/after confirm:
- small-down, small-up, large-down and the ordinary large-neighbor kernel
  have byte-identical `.text`;
- only the exact large-up kernel changes;
- target remains232 VGPR,0 AGPR,128KiB LDS and zero private scratch;
- target text grows from11,712 to27,008 bytes due to unrolling. Code size
  increase was validated in both rotating-address and fixed-address modes.

Integrated-code rocprofv3 ATT on CU0/SE0, invocation2,56 sampled waves each,
all fully stitched by the decoder:

| Kernel | Median wave cycles | Median post-last-MFMA tail cycles | Tail vmcnt(0) stall cycles | Tail lgkmcnt(0) stall cycles |
|:---|---:|---:|---:|---:|
| Before | 131,004 | 5,776 | 532 | 610 |
| After | 129,336 | 5,094 | 0 | 616 |

These are sampled, profiled-wave statistics, not whole-kernel microseconds
or exclusive stall shares. They support the removed unnecessary output-store
wait; the unprofiled public-call measurements establish the scoped speedup.

## Correctness, isolation and rejected trials

Final sources passed:
- strict FP32-reference checks at rtol=atol=.01, zero violating elements,
  and bitwise before/after equality for the changed non-split kernel;
- row padding0/16 with aligned nonzero offsets, input halos/output canaries,
  six changed-input replay rounds (normal/scaled/dyadic), independent streams,
  explicit non-current-stream launches and allocator pressure;
-42 single-nonzero-K dyadic probes at loop-period boundaries and both drain
  tiles, using K offsets0/32/63; poisoned output and guards;
- `python op_tests/test_gemm_a16w16_latent.py --seed 3447`:PASS, including
  existing target/neighbor and exact-input down-M128 ownership tests;
- `python -m pytest -q op_tests/tuning_tests/test_a16w16_tuned_config_lookup.py op_tests/tuning_tests/test_config_shape_collision.py`:
  57 tests/4605 subtests passed;
- normal merged CSV resolves the existing policy and updated metrics, and
  its compiled name selects `_ng4sk_u9lg`;
- separate-cache before/after compilation and non-target `.text` identity.

The unchanged down-M128 BF16 atomic path retains its documented strict
random-reference limitation. Its exact-input test does not certify a
precision repair; no tolerance or reference was changed to qualify this work.

This continuation rejected many plausible but slower/unsafe options:
different LDS read widths compiled to identical machine code; early fragment
loads increased register pressure; wave remapping and packed LDS epilogues
required different convergence and did not win; direct register output,
global-to-LDS DMA, lower-LDS BK32, non-power-of-two tiles, transpose-fused
small GEMMs, scheduling masks/groups, fixed AGPR and full-tile AGPR variants
failed correctness, regressed or lacked reproducible gains. Failures and
negative results remain in the host experiment directory.

Importantly, **tail-only** initially looked promising but the integrated
rotation comparison regressed in2/3 processes; it was withdrawn locally and
never committed. The final combination was fixed before qualification,
passed six independent prototype processes, then six additional integrated
processes above. Only the combined exact large-up schedule is selected.

Host evidence: `flydsl_only/hti_followup/`, especially `combo_qualify/`
(pre-integration), `release/results_342*.json`, `release/results_343*.json`,
`release/trace_summary.json`, independent binary hashes, contract/latent logs
and before/after CSV snapshots. No full-model gain, completed remote GPU CI,
universal hardware speedup or global optimality is claimed.

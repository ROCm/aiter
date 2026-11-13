# Reject regressing WS promotion; continue cold-input optimization

## Deployment correction

Commit71976a2 reverts8e0a68b in full. The small-up model CSV, dispatcher and
tests are restored to bf6094e. The previous promotion was wrong for the
rotation workload:19.592 ->20.493us. Graph gains alone did not justify that
regression. No candidate from this continuation replaces the default.

This continuation focuses on M128,N7168,K3584, the regressing row. It does
not claim new wins on either large-M shape or the down-M128 shape. The three
other task defaults stay unchanged. Existing opt-in WS remains available.

## Measured public-call comparison (microseconds)

Two gfx950 MI355X cards,256 CU, selected with0% observed activity/free VRAM,
BDF0000:95:00.0 (HIP7) and0000:85:00.0 (HIP5). This is an initial activity
observation, not guaranteed exclusive ownership. No clocks/power changes or
termination of other users' work.

Three fresh processes per card, seeds3051..3053 and3071..3073.12 cyclic/
reversed four-way rounds per process:
- Rotation:default run_perftest rotating-input profiler mode,10 warmups/
  101 iterations. Complete public operation; its device-time aggregation
  is not Python wall time.
- Graph4:5 warmups/30 event-timed replays of4 complete calls, divided by4.
- All use the public gemm_a16w16_out chain. Experimental device implementation
  selection occurs outside timing, not as a per-invocation mock.
- Strict FP32 reference, rtol=atol=.01, zero violating elements, changed-input
  graph checks. Ring and trimmed WS must be bitwise-equal to default; FP32
  slice-K changes addition association and is checked against the reference.
- Entries below are medians of3 process medians. Lower is better.

Candidates:
- Ring:original64x64x64/stage7/w4x2x1 FT, static7-slot K-loop expansion,
  unchanged arithmetic and DMA/barrier ordering.
- FP32 slice-K:64x64x128/stage3/w2x2x2 FT, FP32 LDS reduction before the only
  BF16 output conversion, static3-slot expansion.
- Trimmed WS:64x64x128/stage4/2 loader+2x2 compute waves, counters; skip
  unused initial released polls/final release publication, no final barrier.

| GPU BDF suffix | Mode | Original us | FT ring us | FP32 slice-K us | Trimmed WS us |
|:---|:---|---:|---:|---:|---:|
| 95:00.0 | rotation | 19.799 | 19.918 | 19.755 | 20.006 |
| 95:00.0 | graph4 | 19.508 | 17.263 | 17.945 | 18.662 |
| 85:00.0 | rotation | 20.102 | 20.285 | 20.054 | 20.417 |
| 85:00.0 | graph4 | 19.419 | 17.233 | 17.848 | 18.515 |

**No candidate passes the no-regression gate.** Ring regressed rotation in5/6
processes; trimmed WS in6/6. FP32 slice-K was marginally better on the first
card but regressed in2/3 processes on the second card. Its tiny first-card
rotation advantage did not reproduce. Graph4 improvements are real in these
runs but are not sufficient to change the default again. All six processes,
including reversals, remain in the evidence.

## Cold-address thread trace

Earlier hot-address graph traces could not explain rotation behavior. The new
capture cycles32 distinct A/B/output sets (about1.7GB) and samples invocation32,
CU0/SE0. rocprofv3 --att --kernel-trace and the decoder produce fully stitched
timelines for all8 sampled waves in each representative.

| Kernel | Median sampled wave cycles | Mean buffer-load stall cycles/wave | Mean waitcnt stall cycles/wave | Mean barrier stall cycles/wave |
|:---|---:|---:|---:|---:|
| Original FT | 38,858 | 7,065 | 5,980 | 7,821.5 |
| Static7-slot FT | 39,154 | 12,185.5 | 6,630 | 11,078.5 |
| FP32 slice-K | 38,686 | 8,570.5 | 9,724 | 8,334.5 |

The expanded FT's cold-load dispatch stalls increased, offsetting its control
flow savings. FP32 slice-K reduces barrier count but still waits for DMA
(e.g. vmcnt(4)). This is observed stall attribution, not a proof of one sole
hardware cause. Wave stall aggregates are not exclusive kernel time shares.
Profiling cycles are **not** substituted for the unprofiled microseconds above.

## Experiments retained, not deployed

-211 validated WS tile/role/stage configurations:smaller tiles/more CTAs,
  BK64/128/256 and1/2/4 loaders. No stable rotation+graph winner.
-72 WS cache-policy/K-stagger comparisons on small up/down,18 epilogue/unused
  handshake comparisons. Scalar/packed or LDS-vector output, no final barrier,
  first/last-generation handshake omission were tested separately.
-98 FT static-ring tile/stage/wave configurations;85 full-M128 narrow-N panels;
  60 FT cache/stagger/group settings. A few sub-percent rotation screening
  gains did not reproduce in independent runs.
-Original BF16 slice-K screening failed strict numerical checks (65 cases).
  Code inspection found BF16 rounding before LDS slice reduction. The FP32
  reduction prototype passed all65 corresponding checks; it still did not
  provide a reproducible rotation improvement and is not installed.
-66 independent FP32 split-K complete-operation cases passed strict checks,
  but per-call partial allocation and separate reduction erased the gain.
  Those costs were included, not excluded as setup.
-Additional A/B tests:SGPR offsets, incremental ring indices,2/4/7 unrolling,
  A/B DMA order, compute-before-prefetch, scheduling-barrier omission and
  B cache-control aux0/1/2/3. No default winner.
-XCD/tile bijections were checked by enumeration and bitwise GPU results.
  An invalid4-N grouping for28 CTAs/XCD was rejected and archived, then removed
  before rerunning valid mappings. The retained mappings did not win rotation.
-Development compile failures are archived, not treated as numerical passes.
  No tolerance was widened, stale output accepted, or failed candidate routed
  through the CSV.

Host evidence:flydsl_only/ws_no_regression/ contains screens, failures,
per-round samples, both card selections, final six-process qualification,
cold_trace/decoded waves and summary.json.

## Restored-default validation

-Source tree after rollback is identical to bf6094e before this report.
-CPU tuning/lookup/collision suite:56 tests/4605 subtests passed.
-python op_tests/test_gemm_a16w16_latent.py --seed3091:PASS, including the
  three ordinary targets, padded-M neighbors and original down-M128 exact-input
  atomic ownership checks. The latter is not a repair of its known strict
  random-input precision limitation.
-No ongoing GPU job from this experiment remains at report completion.
-No full-model gain, global optimum, all-shape improvement, or completed
  remote GPU CI is claimed.

The remaining direction is a materially different cold-data feed/tiling
strategy, not installing another hot-graph-only win. A new default must beat
the restored baseline in repeated rotation measurements and retain graph
performance and strict correctness.

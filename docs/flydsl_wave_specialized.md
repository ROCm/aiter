# Experimental gfx950 BF16 wave-specialized GEMM

Most configurations remain an **explicit experimental path**. One exact
M128 up-projection model CSV row now selects the counter variant for measured
graph latency gains, with a documented rotation-mode regression; see
[CSV before/after results](flydsl_ws_csv_results.md). It implements dedicated loader and
compute waves on gfx950. It does not port NVIDIA TMA/WGMMA or dynamically
redistribute registers. All other CSV rows keep their previous kernels.

    from aiter.ops.flydsl import gemm_a16w16_wave_specialized

    gemm_a16w16_wave_specialized(
        a, weight, out=out,
        block_m=64, block_n=64, block_k=128,
        stages=3, producer_waves=2, m_waves=2, n_waves=2,
        split_k=1, prefetch=True, counter_handshake=True, stream=stream,
    )

A and weight are BF16 `[M,K]` and `[N,K]` row-major CUDA tensors, on the same
gfx950 device. `out` is contiguous BF16 `[M,N]`; if absent, it is allocated
on the launch stream. Unit K stride, 16-byte-aligned input pointers and row
strides are required. N must be a multiple of 4 and K divisible by
`split_k * block_k`. M/N edge tiles are masked. There is no bias/scaling or
activation support. Invalid inputs raise before dispatch.

## Roles and synchronization

A workgroup contains `producer_waves + m_waves*n_waves` waves of 64 threads:

- Each loader wave issues disjoint 16-byte async global-to-LDS stripes for
  both A and B. Their aggregate stripes cover each input tile once.
- Compute waves read LDS and execute BF16 MFMA16x16x32 with FP32 accumulation.
- Every ring slot has a ready sequence for each loader and a released sequence
  for each compute wave. A loader may overwrite a slot only after all its
  consumers release the old generation. A consumer may read only after all
  loader stripes publish the new generation.
- `counter_handshake=False` retains these per-wave sequence arrays as a
  control. `True` uses two cumulative atomic counters per slot: a producer
  waits for `released >= generation * consumers` and contributes one ready
  increment after its DMA completes; a consumer waits for
  `ready >= (generation+1) * producers` and contributes one release after
  retiring its LDS reads. Counters start at zero per workgroup/invocation
  and are **never reset during ring reuse**. A faster wave cannot contribute
  to the next slot generation until the other role has completed the current
  generation. The validated tensor-span limits also keep counts within i32.
- Workgroup-scope acquire/release operations, explicit wave-local completion
  waits and leader publication order each handoff. Polls include `s_sleep(1)`.
  Only waves in the **same resident workgroup** are polled. There is no
  dependency on an unscheduled workgroup.
- The only whole-workgroup barriers are initial flag initialization and final
  convergence. No role-local branch executes a full-workgroup barrier.
- `prefetch=False` waits for each loader's current K tile before publication.
  `prefetch=True` queues up to `stages-1` DMA groups and retires/publishes them
  in order. A short partition uses the guarded serial loop.

This is loader/compute specialization for one output tile per workgroup.
It does **not** yet implement a cross-output-tile persistent pipeline,
two-consumer ping-pong scheduling or cluster multicast.

For split_k=1 the compute waves store BF16 output directly. For split_k>1
they write disjoint FP32 partials; a separate deterministic reduction sums
partitions and rounds once. Its per-call allocation and reduction launch
are included in measurements. The original default down-M128 BF16 atomic
path has a different intermediate-rounding contract and its known strict
random-reference errors remain on that default; this new path's checks use
FP32 reference at rtol=atol=.01 with zero violating elements.

## Resources, validation and cache contract

LDS bytes:

    stages * (block_m + block_n) * block_k * 2
    + stages * (2 if counter_handshake else producer_waves + m_waves*n_waves) * 4

Input stripes must divide among producers. LDS <=160 KiB, producer waves in
{1,2,4}, compute waves in {2,4,8}, stages in [2,5], block_k in {64,128,256},
and split_k <=16. The outstanding DMA bound is checked conservatively.
All addressed tensor and FP32 partial spans must fit signed-i32 bytes.

Scratch is per invocation on its launch stream, never cached by tensor
address. Graph captures own their scratch, and independent streams do not
share it. Compilation is cached per geometry/prefetch/handshake; handles are keyed by
device and dtype. M/N/K, leading strides and tensor pointers remain runtime
ABI values; rank, unit stride and output layout are checked on every call.

`op_tests/test_flydsl_hgemm.py` contains the new path's tests, including the
four requested shapes, short K, ragged M/N, multi-generation ring reuse,
padded strides, guards, cold capture, concurrent graph replay and rejection
checks. The existing CPU tuning test suite covers geometry limits, DMA
ownership and sequence generations. Default GEMM sources and CSVs are not
changed.

    python op_tests/test_flydsl_hgemm.py -s 128,3584,7168 128,7168,3584 16384,3584,7168 16384,7168,3584
    python op_tests/test_gemm_a16w16_latent.py --seed 2747
    python -m pytest -q op_tests/tuning_tests/test_a16w16_tuned_config_lookup.py op_tests/tuning_tests/test_config_shape_collision.py

## Initial four-shape tuning experiment (ecaba5c)

Measured against the parent PR's default public `gemm_a16w16_out` path,
commit `1b52b6f`, on a gfx950 MI355X selected by free VRAM, BDF0000:75:00.0.
No clock changes or termination of other users' workloads.

Initial 120 configurations swept:
- Small M: tiles64x64x64,32x64x128,64x64x128,128x64x64,64x128x64;
  consumer wave layouts2x2 or1x4, producer waves1/2, stages2/3 and selected
  stage4 controls, split_k1/2.
- Large M: tiles128x128x64,128x256x64,256x128x64,256x256x64,128x128x128;
  consumer layouts2x2/2x4/4x2, producer waves1/2, stages2/3 and selected
  stage4 controls, split_k1.
- Configurations failing the explicit DMA/LDS limits were not launched.

A second 47-configuration screen added the loader prefetch queue on/off,
stage3 and four-loader-wave controls around the better first-round shapes.
All 167 screened launches passed strict numerical checks.

The chosen candidate **per shape** was then repeated in three fresh processes,
seeds2731/2732/2733, eight alternating AB/BA rounds:
- rotation: default `run_perftest` rotating-input profiler mode, 10 warmups,
  101 iterations, including allocations and reduction;
- graph4: four complete invocations captured, CUDA events, 5 warmups,
  30 iterations, divided by four.

The selected candidate for every shape uses `prefetch=True`.
Positive percentages below mean **slower**, not speedup.

| M,N,K | BMxBNxBK | Stages | Loader + compute waves | Split K | Rotation slower | Graph4 slower |
|:---|:---|---:|:---|---:|---:|---:|
| (128, 3584, 7168) | 64x64x128 | 3 | 2 + 2x2 | 2 | 20.5–23.8% | 14.9–25.8% |
| (128, 7168, 3584) | 64x64x128 | 3 | 2 + 2x2 | 1 | 3.7–4.3% | 1.1–10.1% |
| (16384, 3584, 7168) | 256x128x64 | 3 | 2 + 4x2 | 1 | 35.2–42.2% | 42.3–42.8% |
| (16384, 7168, 3584) | 256x128x64 | 3 | 2 + 4x2 | 1 | 43.2–46.2% | 47.8–48.0% |

**No winner was selected for production.** Queueing DMA helps the new path
relative to its serial-loader version, but has not beaten the existing
default. ISA inspection confirms two whole-WG barriers and no private spills
in the inspected representatives. Polling, load concentration, wave-resource
cost and scalar output traffic remain hypotheses for further profiling;
this experiment does not prove a single bottleneck.

These are scoped measurements under shared-node conditions, not a universal
claim that wave specialization is slower on AMD. They do establish that
this implementation/tuning batch should not replace the four defaults.


## Thread-trace-guided continuation

This continuation keeps all default GEMM code and CSVs unchanged and adds only
`counter_handshake=True` to this explicit API. Leaving it out retains the
previous per-wave protocol. Both protocols retain sleep, completion waits
and workgroup-scope release/acquire; this is **not** a relaxed-ordering shortcut.

### Observed hotspots and change

`rocprofv3 --att --kernel-trace` produced decoded wave timelines and ISA,
not just raw ATT files. The initial small-M trace split waves by whether they
executed MFMA. Ready polling was repeatedly `ds_read_b32; s_waitcnt lgkmcnt(0)`
plus a compare/branch/sleep loop. Loader release checks scanned all compute
waves. DMA was concentrated into two loader waves. In the original prefetch
steady state, `s_waitcnt vmcnt(16)` remained before publication: the hypothesis
that every ready publication drained the **whole** DMA queue was not supported.

The implemented change replaces P ready flags and C released flags per ring
slot with one cumulative ready and one cumulative released counter. Each wave
leader performs a workgroup-scope release atomic increment once per K tile;
waiting waves acquire the role-count threshold. No live counter is reset.

Same-capture prototype controls, compute-wave duration median on the traced CU:

| M,N,K | Same WS tile | Per-wave flags (cycles) | Counters (cycles) | Reduction |
|:---|:---|---:|---:|---:|
| (128,7168,3584) | 64x64x128 | 33,154 | 30,088 | 9.25% |
| (16384,7168,3584) | 256x128x64 | 115,364 | 97,498 | 15.49% |

All sampled waves in these captures were fully stitched by the decoder.
These are sampled-wave timings **under profiling**, not whole-kernel speedups.
The large default computes 256x256 per workgroup, versus WS 256x128; comparing
their raw wave durations without accounting for twice the output is invalid.
Also, summed stall cycles across concurrent waves are not a kernel time share.

Single-variable experiments not adopted:
- Removing `s_sleep` alone did not improve the small-M control. Batched flag
  polling helped graph timing but less than the cumulative-counter protocol.
- Scalar DMA offsets and simple ring-index changes were not consistent wins.
- Loading all fragments and releasing LDS earlier increased register work;
  it did not improve the control. Register-shuffle/vector output stores did
  not reliably improve on the counter version either.
- Increasing queue depth from `stages-1` to the DMA-safe `stages` bound was
  checked with exact old/new results and changed-input graph replays. It
  regressed several configurations, including the nearest small-M candidate.
  Eighteen shape/config comparisons included 2/3-stage and 2/4-loader controls
  and larger tiles. No deeper-queue/larger-tile candidate was adopted.
- Large-M output tail was about 5.2–5.5k cycles in the sampled WS waves, much
  smaller than their roughly 97–115k total duration. This does not prove
  stores are optimal, but they were not the dominant observed delay.

### Repeated unprofiled measurements

MI355X gfx950, qualification/integrated BDF0000:15:00.0 (HIP3), same
selected device within each paired batch, shared-node
conditions, no clock or other-user workload changes. The prototype used equal
host wrappers (no timed mock construction), three fresh seeds2861/2862/2863,
12 cyclic/reversed three-way rounds. Rotation:10 warmups/101 iterations in
`run_perftest` rotating-input profiler mode. Graph4:5 warmups/30 event-timed
replays of four complete calls, divided by four. Split-K allocation and
reduction are included. Candidate geometries are those in the initial table.

Each cell is the median across three per-process ratios, followed by their
full range. **WS improvement positive = faster than old WS. Default delta
positive = still slower than default.** No samples or slow processes removed.

| M,N,K | Rotation WS improvement | Rotation default delta | Graph4 WS improvement | Graph4 default delta |
|:---|---:|---:|---:|---:|
| (128, 3584, 7168) | +1.92% [+1.51, +2.30] | +18.89% [+18.77, +19.66] | +6.42% [+6.29, +10.61] | +7.79% [+3.83, +8.75] |
| (128, 7168, 3584) | +3.42% [+3.09, +6.91] | +2.84% [+0.03, +34.22] | +6.79% [-1.24, +7.02] | -5.51% [-6.55, +10.08] |
| (16384, 3584, 7168) | +4.71% [-4.96, +7.03] | +38.82% [+29.41, +53.20] | +6.18% [+4.62, +7.53] | +32.21% [+31.13, +35.36] |
| (16384, 7168, 3584) | +4.28% [-5.71, +7.01] | +41.09% [+36.61, +55.44] | +4.94% [+4.51, +6.82] | +41.05% [+39.17, +41.20] |

A fourth process (seed2881) benchmarked the integrated public API, not the
prototype wrappers, with the same protocol. Complete-operation microseconds:

| M,N,K | Mode | Default us | Old WS us | Counter WS us |
|:---|:---|---:|---:|---:|
| (128, 3584, 7168) | rotation | 20.882 | 25.475 | 24.875 |
| (128, 3584, 7168) | graph4 | 18.891 | 21.907 | 20.560 |
| (128, 7168, 3584) | rotation | 19.878 | 20.999 | 20.430 |
| (128, 7168, 3584) | graph4 | 19.241 | 19.453 | 18.124 |
| (16384, 3584, 7168) | rotation | 588.006 | 800.481 | 761.995 |
| (16384, 3584, 7168) | graph4 | 959.360 | 1263.503 | 1053.359 |
| (16384, 7168, 3584) | rotation | 571.918 | 818.657 | 781.798 |
| (16384, 7168, 3584) | graph4 | 548.745 | 813.393 | 775.294 |

The integrated large-down graph batch changed operating regime within the
batch (default ~959us, counter ~1053us). It is retained, not promoted as a
stable 16.6% improvement over old WS. Absolute timings and some ratios varied
substantially under co-tenancy.

**Conclusion:** cumulative counters improve this WS implementation, generally
by several percent in these runs. The small-up graph case was faster than
default in two of three prototype processes and in the integrated process,
but regressed in one prototype process and did not consistently win rotation.
The other shapes remain slower than default. No production/default winner
is declared; the original four-shape objective is not fully solved.

Evidence files on the experiment host are under `flydsl_only/ws_trace/`:
`att/`, `att_counter/`, `att_large/`, `qualify/results_286*.json`,
`full_slots/results.json`, and `integrated/`. ATT sampled CU0/SE0, iteration2;
use the matching unprofiled results for performance conclusions.


### Integrated trace and validation

A new capture of the integrated API, using `counter_handshake=False/True`
in the same process, confirmed the reduction in sampled compute-wave duration:

| WS tile | Per-wave flags (cycles) | Counters (cycles) | Reduction |
|:---|---:|---:|---:|
| 64x64x128 | 46,508 | 42,138 | 9.40% |
| 256x128x64 | 116,482 | 105,174 | 9.71% |

All sampled waves were fully stitched. In this capture, average compute-wave
`s_waitcnt` stall cycles decreased from19,637 to17,015 (small tile) and from
35,381 to26,853 (large tile). These are per-wave aggregates, not exclusive
whole-kernel stall percentages. `ds_add_u32` implements the new counter
publication. Both protocols have two static whole-WG barriers; metadata has
zero private scratch and disassembly has no scratch instructions. Registers
are unchanged at46 VGPR for the small tile and100 VGPR/16 AGPR for the large
tile. LDS drops only48/96 bytes, so this is not an occupancy-by-LDS-size gain.

Validation on the integrated sources:
-11 geometry cases x2 protocols: PASS. Four targets, short-K, serial producer,
  2/3/4/5-stage wraparound, 1/2/4 loaders, 2/4/8 compute waves, ragged M/N,
  aligned padded strides, cold captures, concurrent streams, modified inputs,
  NaN-poisoned outputs, input/output/workspace canaries and allocation pressure.
  Counter cases additionally require bitwise equality with per-wave flags.
-24 randomized counter-protocol configurations with6 changed-input replays
  each of a3-call graph: PASS. Additional Split-K1/2/4 equivalence: PASS.
-`python op_tests/test_gemm_a16w16_latent.py --seed 2893`: PASS; this retains
  the documented exact-input contract for the default atomic down-M128 case.
-`python -m pytest -q op_tests/tuning_tests/test_a16w16_tuned_config_lookup.py op_tests/tuning_tests/test_config_shape_collision.py`:
  56 tests/4605 subtests passed. Includes an abstract randomized interleaving
  model checking all producers published before any read, and all consumers
  released before overwrite. That model does not itself prove GPU ordering.
-Black/Ruff/diff checks: PASS. Existing default GEMM and CSV files unchanged.

For only the WS GPU contract cases (without running all older HGEMM paths):

    from op_tests.test_flydsl_hgemm import (
        _WS_CASES, test_hgemm_wave_specialized, check_wave_specialized_rejects,
    )
    for case in _WS_CASES:
        for counter in (False, True):
            test_hgemm_wave_specialized(*case, counter_handshake=counter)
    check_wave_specialized_rejects()

Full-model benefit, exhaustive configurations, isolated-node performance and
completed remote GPU CI are **not** established by these tests.

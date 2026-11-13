# Experimental gfx950 BF16 wave-specialized GEMM

This is an **explicit experimental path**, not a replacement for
`flydsl_hgemm` or tuned GEMM dispatch. It implements dedicated loader and
compute waves on gfx950. It does not port NVIDIA TMA/WGMMA or dynamically
redistribute registers. No existing CSV row selects this function.

    from aiter.ops.flydsl import gemm_a16w16_wave_specialized

    gemm_a16w16_wave_specialized(
        a, weight, out=out,
        block_m=64, block_n=64, block_k=128,
        stages=3, producer_waves=2, m_waves=2, n_waves=2,
        split_k=1, prefetch=True, stream=stream,
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
    + stages * (producer_waves + m_waves*n_waves) * 4

Input stripes must divide among producers. LDS <=160 KiB, producer waves in
{1,2,4}, compute waves in {2,4,8}, stages in [2,5], block_k in {64,128,256},
and split_k <=16. The outstanding DMA bound is checked conservatively.
All addressed tensor and FP32 partial spans must fit signed-i32 bytes.

Scratch is per invocation on its launch stream, never cached by tensor
address. Graph captures own their scratch, and independent streams do not
share it. Compilation is cached per geometry/prefetch; handles are keyed by
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

## Four-shape tuning experiment

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

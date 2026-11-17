# MegaMoE FP4 pull dispatch on gfx1250

This directory contains MegaMoE's expert-parallel dispatch, planning, and
combine kernels. The experimental pull path targets non-S1F prefill and writes
GEMM1's preshuffled activation layout directly. It is opt-in; ordinary push
dispatch remains the default.

## Enable pull

Set these variables before importing AITER:

```bash
export AITER_MEGA_DISPATCH_PULL=1
export AITER_MEGA_PULL_TDM_GATHER=1
export AITER_MEGA_S1F=0
export AITER_MEGA_QUANT_FUSE=128
export AITER_MEGA_PLAN_PREFETCH=0
```

The recipe must use compact dispatch with an FP4 wire and fused-scatter
combine. Pull requires positive `AITER_MEGA_QUANT_FUSE` and the supported
preshuffle dimensions. It does not replace S1F: if S1F is selected for a step,
that step keeps the existing S1F path. Explicitly disabling S1F makes a pull
comparison unambiguous.

`AITER_MEGA_PULL_TDM_GATHER=1` selects the 128-thread non-persistent kernel.
Without it, pull uses the earlier 512-thread persistent async-copy kernel.
The TDM kernel's symbol ends in `_tdmgather_t128_g2x8_grid`.

## Data flow

1. `stages/dispatch_compact.py` selects `stages/pull.py` for a non-S1F step.
2. The existing `compact_plan.py` launch runs routing workgroups alongside
   quantization workgroups. Quantization processes each local input token once
   and writes FP4 payload and E8M0 scales into symmetric source storage.
   Planning exchanges histograms and writes receiver-side `ep_rowmap` entries:
   packed source rank/route plus the routing weight bits. The packed route
   identifies both the source token and its top-k slot.
3. A stream event makes local dispatch wait for the entire local plan/quant
   launch. Dispatch publishes the source-ready epoch with system release
   ordering and acquires every peer's epoch before reading its data. Histogram
   completion alone is insufficient. Both push and pull require data readiness.
4. `dispatch_pull.py` gathers remote source rows into private per-wave LDS,
   then stores preshuffled payload and row-major scales into local output.
5. `ExpertGrouping` carries `a_preshuffled=True`, `psum`, `masked_m`, and the
   inverse route map to GEMM1 and fused GEMM2/scatter. The GEMM input contract
   is `a_preshuffled=1`, `row_major_ascale=1`, `wire_row_stride=0`.
6. The existing combine barrier protects source storage from reuse by the
   next layer while peers still read it.

Quantization overlaps the histogram/routing work within the plan launch.
Pull waits for that whole launch; it does not consume partially quantized
source rows. Ordinary next-layer plan prefetch is suppressed for pull steps
because it cannot supply their quantized inputs.

## TDM geometry and layouts

For DSV4.1 Flash, `H=5120`, `I=2304`, EP4, 384 total experts, and top-k 6:

| Item | Value |
|---|---|
| Source payload | 2560 bytes/token, packed FP4 |
| Source scales | 160 bytes/token, one E8M0 byte per 32 values |
| Wire row | 2816 bytes: payload + scales + 96 bytes padding |
| Workgroup | 128 threads, four wave32s |
| Work per wave | One M16 tile; two eight-row TDM gathers issued before waiting |
| Work per workgroup | At most 64 destination rows, with no persistent tile loop |
| Grid | `max(1, ceil(row_capacity / 64))` workgroups |
| LDS per workgroup | `4 * 16 * 2816 + 96 * 4 = 180608` bytes |
| Output payload | `[M/16, K/32, M%16, 16 packed bytes]` |
| Output scales | Row-major `[M, K/32]` |

At 16384 tokens/rank and alignment 256, the existing conservative capacity is
417792 rows, so the grid contains 6528 workgroups. Uniform routing produces
about `16384 * 6 = 98304` live routes per rank, before expert padding.
Device-side `psum` skips unused tiles and empty experts. Grid sizing does not
assume balanced routing and does not read GPU counters back to the host.
More grid workgroups do not by themselves increase simultaneous residency;
the LDS footprint still matters.

Each wave owns its LDS slice and output vectors, eliminating per-tile
workgroup barriers. Startup readiness and metadata barriers remain. Payload
writeback uses 16-byte LDS loads and global stores; scales use four-byte
loads/stores. The local and legacy async paths reuse the extracted
`make_apre_payload_ops` helper in `../moe_apreshuffled_quant_scatter.py`.
The TDM path has its own per-wave writeback to avoid a 512-thread shuffle
barrier.

## Reproduce the prefill workload

Use the repository-pinned FlyDSL (0.5.0 at the development baseline) and a
compatible MORI installation exposing `mori.cco`. On this development host,
the tested MORI checkout is `/workspace/mori-plan-describe`; selecting that
checkout is a host-specific setup, not a portable installation requirement.

From the repository root, after selecting four idle gfx1250 GPUs:

```bash
PYTHONUNBUFFERED=1 ENABLE_CK=0 \
MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 \
MORI_AMD_SMI_PATH=/opt/rocm/lib/libamd_smi.so.27 \
AITER_MEGA_DISPATCH_PULL=1 AITER_MEGA_PULL_TDM_GATHER=1 \
AITER_MEGA_S1F=0 AITER_MEGA_QUANT_FUSE=128 AITER_MEGA_PLAN_PREFETCH=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_CONFIG_GROUPED_FMOE=aiter/configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv \
torchrun --standalone --nproc_per_node=4 \
  op_tests/multigpu_tests/bench_mega_moe.py \
  -tpr 16384 --layers 8 -e 384 -k 6 -hd 5120 -id 2304 \
  --fused_shared_experts 0 --routing uniform \
  -q a4w4_mxfp4 --dispatch_wire fp4 \
  --stage1_fused true --stage2_fused true \
  --profile_table 1 --prof_replays 32 --acc_verify 0 \
  --warmup 5 --iters 32 --seed 0 --data-init uniform
```

For the push control, change only `AITER_MEGA_DISPATCH_PULL=0`. Keep S1F off,
routing, CSV, timing method, and all other workload settings identical.
`-tpr` means tokens **per rank**: this case has 65536 global tokens. The EP
CSV lookup uses 96 local experts, `topk=-1`, and token key 65536 for this case.
Do not equate an EP token key with the non-EP local-token workload.

## Validation

- CPU recipe tests: `python3 -m pytest -q tests/test_mega_moe_gfx1250_recipe.py`
  passed, 33 tests, on 2026-11-17.
- The existing multi-GPU test now includes `_check_pull`. It compares full
  output with push over four changing-input graph replays, alternating random
  routes and routes concentrated on rank 0. It also checks exact valid-row
  payload, scales, routing weight bits, and the shared local preshuffle path.
- The current non-persistent TDM implementation passed that check at EP4,
  16384 tokens/rank, H5120/I2304/E384/top-k 6, with preshuffle enabled.
  Saved log on the development host:
  `/workspace/benchmarks/ds41_pull_grid_20261116/check_target_shuffle1/test.log`.
- The standard suite is
  `python3 op_tests/multigpu_tests/test_mega_moe_gfx1250.py --world 4`.
  Set `AITER_MEGA_PULL_TDM_GATHER=1`, `AITER_MEGA_QUANT_FUSE=128`, and
  `AITER_MEGA_S1F=0`, but leave `AITER_MEGA_DISPATCH_PULL=0`: the pull check
  enables it internally, while the suite also exercises FP8 push recipes.
  Its default shape differs from the saved DSV4.1 Flash validation shape. GPU jobs should run
  only after an idle check; use the shared `gpu-lock` wrapper when installed.

## Performance evidence and limits

The following are **historical persistent-kernel measurements**, not timings
of the current non-persistent TDM kernel. Workload and routing match the
prefill command above. Units are microseconds; E2E is latency per layer.

| Version | Runs | Plan / plan+quant | Dispatch | GEMM1 | GEMM2 | E2E/layer |
|---|---:|---:|---:|---:|---:|---:|
| Push | 1 | 63.39 | 252.04 | 620.62 | 585.34 | 1931.84 |
| Original 512-thread persistent pull, shuffle | 2 | 126.07 | 340.43 | 504.27 | 578.35 | 1942.69 |
| 128-thread persistent pull, two gathers, shuffle | 2 | 126.45 | 306.39 | 503.59 | 578.08 | 1904.61 |
| Original 512-thread persistent pull, no shuffle | 1 | 125.87 | 335.57 | 631.30 | 584.15 | 2068.29 |
| 128-thread persistent pull, two gathers, no shuffle | 1 | 125.38 | 290.10 | 633.79 | 584.62 | 2024.81 |
| Current 128-thread non-persistent TDM pull | 0 | pending | pending | pending | pending | pending |

Kernel timings used HIP event nodes inside a graph: median over 256 samples
per GPU, averaged over four GPUs, then averaged over successful runs. E2E
used the separate uninstrumented native graph. Push's plan column excludes
its separate quantization; pull's plan column includes quantization and
inverse-map writes. Columns therefore do not sum to E2E.

The earlier two-gather persistent change reduced shuffled dispatch by 10.0%
and observed E2E by 2.0% versus original pull. Its dispatch was still 21.6%
slower than push. The small E2E difference versus push is not sufficient to
claim a stable win from one push run. No speedup is claimed for removing the
persistent loop until a valid measurement completes.

Raw historical records and both historical sources are preserved on the
development host in `/workspace/benchmarks/ds41_pull_inflight_20261115/`.
These local artifact paths are not bundled with the repository.

A follow-up comparison in
`/workspace/benchmarks/ds41_pull_dispatch_only_20261117/` uses identical
source/map/output buffers for the 128-thread persistent and non-persistent
kernels. It prepares real plan/quant inputs once and times eight dispatches
per graph replay with external HIP events, over 32 interleaved trials.
It checks exact valid-row bytes and requires eight real captured kernel
nodes before accepting timing. This is isolated dispatch with producers
already complete, not E2E or in-pipeline timing. The first attempt captured
empty graphs; all those timings are invalid. Full-pipeline stalled attempts
are also excluded. As of this submission, the corrected comparison is
waiting for idle GPUs and has no accepted result.

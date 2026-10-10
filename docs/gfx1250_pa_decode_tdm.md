# TDM staging for gfx1250 FP8 PA decode

The gfx1250 FP8 decode kernel uses TDM for per-token scales with head dimensions
128, 256, 512 and 1024. Each wave gathers 16 K tokens from the packed cache into
padded token-major LDS rows through a rank-3 descriptor. V copies split head rows
across waves for page64/128, and split pages for page16. Both public V layouts
are supported. TDM supplies the row padding without a separate LDS rearrangement.

The pipeline retains the 64-token compute tile, 256-token plan contract and
alternating LDS buffers. `tensor_wait(0)` followed by a CTA barrier retires the
current tile; the next tile loads while the current tile computes. The D1024
one-buffer path loads after all current reads retire. Global offsets remain
64-bit, and inactive slices use safe owned-page addresses with zero extents.
Transposed V retains the existing requirement that the last owned page's
unwritten tail contain finite values.

Single-wave atoms use explicit slices because the installed collective rank-3
lowering did not preserve padded wave offsets in an initial GPU experiment.
D64, scalar scales and other supported head dimensions retain vector DMA:
unconditional TDM measured slower for D64 and scalar D128.

## Measured performance

Measured on 2026-10-08 against the pre-TDM FP8 commit `0f3509e7c`, on gfx1250
with 256 CUs, 320 KiB LDS and 23.347 TB/s reported peak bandwidth. Software:
FlyDSL 0.3.3.dev886, PyTorch 2.11.0+rocm10.1 and HIP 7.16.

Both implementations ran in one process using identical inputs, unique shuffled
pages, randomized candidate/backend order and seven timing rounds. Each graph
contains 32 decode/reduce calls. Initialization, planning, quantization,
compilation and Python launch time are excluded. Budgets
128/256/512/1024/2048/4096 were searched; the default remains 512.

GQA uses Hq64/Hkv8 and MQA uses Hq16/Hkv1. Queries/outputs are BF16, caches are
FP8 E4M3FN, with per-token FP32 scales and page128. D128 unless specified.

| Workload | Before µs | TDM µs | Speedup | Effective KV TB/s |
|---|---:|---:|---:|---:|
| MQA B4/4K | 14.19 | 13.62 | 1.042× | 0.308 |
| GQA B64/4K | 73.77 | 70.97 | 1.039× | 7.565 |
| GQA B64/16K | 251.57 | 230.79 | 1.090× | 9.305 |
| GQA B64/16K/MTP4 | 367.46 | 344.17 | 1.068× | 6.240 |
| GQA B64/4K/plain V | 77.28 | 72.76 | 1.062× | 7.379 |
| GQA B16/4K/D256 | 51.40 | 47.53 | 1.081× | 5.647 |

Rows compare each implementation's best configuration. D256 changes budget from
512 to 2048; at the same budget512 it improves 51.40 to 48.17 µs (1.067×).
The 16 selected workloads improved by 1.023–1.090×, with a 1.051× geometric mean.
Effective bandwidth counts useful K/V bytes once across GQA/MTP reuse, excluding
scales, Q/O, scratch and metadata. It is not a physical bus measurement.

Separate ROCprofv3 `FETCH_SIZE` passes show B64/16K physical memory reads remain
approximately 2034 MiB. Normalized by independent unprofiled latency, read
bandwidth increases from 8.48 to 9.24 TB/s. Traffic changes by less than 0.04%.
Profiling serializes dispatches, so profiled latency is not used for speedups.

Fresh same-specialization ISA and extracted executed HSACO objects confirm
`tensor_load_to_lds` replaces the K/V vector DMA loads. VGPR metadata decreases
129 to 106, SGPRs increase 68 to 90, and static LDS stays 44096 bytes. Both versions
have zero VGPR/SGPR spills and zero private segment bytes. The resource-diff
workflow flags increased SGPRs; measured performance improves despite that change.
No CDNA occupancy formula is applied to gfx1250.

A fresh comparison against PR #3256 on exactly equivalent dequantized attention
data measured 69.57 versus 117.34 µs at B64/4K (1.687×) and 225.15 versus
458.37 µs at B64/16K (2.036×), with both implementations tuned. PR #3256 already
uses TDM and BF16 caches, so those numbers compare full kernels. They do not
measure the isolated TDM gain. This comparison uses different input values from
the before/after sweep.

## Validation

The original 114-case GPU validation covered BF16/FP16 queries,
scalar/per-token scales, both V layouts,
page16/64/128, MTP, windows, sinks, partition boundaries, refreshed graphs, odd
strides, poisoned scratch, 2/4 GiB offsets, partial pages, D384/768 fallback and
one-buffer D1024. gfx942/gfx950 MTP4 tile and reducer compile checks passed;
those architectures were not tested numerically or benchmarked.

```bash
GPU_ARCHS=gfx1250 python3 -m pytest -q \
  op_tests/test_flydsl_pa_decode.py
```

The paired benchmark, raw timing samples, counters, ISA and plots are recorded
in `/home/sixifang/gfx1250-dev/pa-gfx1250-validation/TDM_PERFORMANCE.md` in the
validation workspace. The benchmark loads the unchanged baseline worktree at
`0f3509e7c`; it does not modify tuning defaults or install tuning entries.

The analysis followed FlyDSL's
[prefetch-data-load](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/prefetch-data-load/SKILL.md)
and
[isa-resource-diff](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/isa-resource-diff/SKILL.md)
workflows.

## D128 page-sized FP8 tiles

The packed per-token D128 path now uses 128-token staging and K128 PV WMMA
for page128, full attention, and at most 16 query rows per KV head. This extends
the existing D256 path. The 256-token planner contract and workgroup budget
remain the same. Other storage modes and windowed rows retain their tile sizes.

FlyDSL ATT traces identified global-load waits around page-table lookup and
scale staging as the dominant stalls. Matched B64/4K equal-context traces
showed `s_wait_loadcnt` stalls decrease from 1,176,507 to 411,586 cycles (65%),
with total sampled cycles decreasing 53%. The shipped analyzer's CDNA occupancy
estimate does not apply to gfx1250; interpretation uses its actual split-counter
instructions. Exact-specialization ISA reports VGPR118 to146, SGPR89 to102,
and static LDS21056 to38976 bytes, with no register spills or private scratch.
The increased resources are accepted because paired unprofiled timing improved.

A same-process baseline/candidate comparison of B16/B64, context limits4K/16K,
and uniform/log-uniform/bimodal random contexts improved all 12 cases by
1.13–2.09× at budget512. An idle-GPU rerun measured B64/16K uniform at
210.378 µs, 5.170 TB/s useful KV bandwidth and approximately5.134 TB/s external
read bandwidth. Separate single-pass `FETCH_SIZE` collection normalizes read
bytes by unprofiled decode+reduce graph latency. This is warm-cache read traffic,
not total bus utilization; the device reports23.347 TB/s peak.

The benchmark's four sampled reference checks passed in each case. Checking
every sequence exposed a short-context FP8 Q/P rounding error of0.008108 on a
13-token row with atol=rtol=0.005. The full results preserve this limitation and
the failed validation attempt; they do not claim strict accuracy for every row.
The new boundary tests cover FP16/BF16 queries, decode/MTP2, page/token tails,
NaN cache padding, poisoned scratch and graph plan refresh.

Timings, counters, traces and resource reports are saved in
`/home/sixifang/gfx1250-dev/benchmark-results/fp8-optimization/`.
`BANDWIDTH.md` describes the rerun and measurement limitations;
`bandwidth-summary.json`/`.csv` contain all12 combined rows.

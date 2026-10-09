# FP8 decode bandwidth tuning on gfx1250

The FP8 kernel retains 64-token compute steps, 256-token plan tiles, four
wave32 waves and double-buffered K/V staging. The optimization reduces work
around those operations:

- Full 64-token tiles use constant TDM extents; partial tiles retain clamped
  extents and safe addresses into owned pages.
- Page64/128 scale loads share the tile's page-table lookup. Page16 retains
  per-token page selection.
- Each wave reduces the same 64 value scales locally. This replaces a
  cross-wave normalization barrier without changing the normalization range.
- K/V scales load in vectors, and independent V operands load before PV WMMA.
- FP8 D128 with one 16-row query tile retains its Q operand and scale in
  registers across KV iterations.
- The next iteration's CTA barrier retires the final query tile's reads in
  the double-buffer path. Intermediate query tiles and the single-buffer path
  keep their explicit retirement barrier.

The cache format, quantization, plan/scratch contracts and FP32 softmax statistics
remain compatible. BF16 paths pass the existing tests. FP8 D128/G8/MTP1 ISA
uses 108 VGPRs versus 106 before, 90 SGPRs in both, and 44096 static LDS bytes
in both, with zero register spills/private bytes. It has six fewer LDS read
sites. Full/tail branching duplicates TDM instruction sites in the ISA;
only the selected branch executes at runtime.

Correctness validation passed 163 GPU cases spanning FP8 and BF16, query dtypes,
page sizes, both V layouts, partial/empty contexts, MTP, windows, sinks,
probability tails, graph refresh, odd strides, poisoned scratch and 2/4 GiB
cache boundaries. Black, Ruff and whitespace checks passed.

## Measurement

The target is 50% of the device's reported 23.347 TB/s peak, or 11.674 TB/s.
Useful KV bandwidth counts K/V once across GQA/MTP reuse and excludes scales,
Q/O, scratch and metadata. Physical reads are measured separately with
ROCprofv3 `FETCH_SIZE`, including cache/memory effects and excluding writes.
The rates use independent unprofiled latency because profiler collection
serializes dispatches.

Measured on 2026-10-09 against commit `91672d2f3` on gfx1250 with 256 CUs and
320 KiB LDS, using FlyDSL 0.3.3.dev886, PyTorch 2.11.0+rocm10.1 and HIP 7.16.
Both kernels run in the same process with identical FP8 caches, per-token FP32
scales unless specified, BF16 queries/outputs, shuffled unique physical pages,
and FP32 sampled correctness checks for every candidate. GQA uses Hq64/Hkv8;
MQA uses Hq16/Hkv1. D128/page128 unless specified.

Seven randomized timing rounds use graphs of 32 decode/reduce calls and 12 replay
windows per candidate. Initialization, quantization, planning, compilation and
Python launch overhead are excluded. The expanded search covers budgets
128/256/512/1024/2048/4096/8192/16384/32768. The default production plan budget
remains twice the CU count. Supply a measured budget when building the plan:

```python
plan = plan_pa_decode(
    context_lengths, num_kv_heads,
    query_length=query_length,
    workgroup_budget=8192,
)
```

The budget above is appropriate for the measured dense D128 streaming workloads;
measure the budget for a different workload, or use the offline tuner. Decode
never runs an online tuning search.

ATT collection was attempted with FlyDSL debug information, but the installed
trace decoder failed to initialize (`Error loading decoder: 37`). The findings
here use ISA inspection, paired timing and physical-read counters.

Full results, raw samples, experiments, counters and reproduction commands are
in `/home/sixifang/gfx1250-dev/pa-gfx1250-validation/BANDWIDTH_PERFORMANCE.md`.
The analysis follows FlyDSL's
[kernel-trace-analysis](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/kernel-trace-analysis/SKILL.md)
and
[isa-resource-diff](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/isa-resource-diff/SKILL.md)
workflows, with the ATT limitation stated above.

## Results

| Workload | Before µs | After µs | Speedup | Useful KV TB/s | % peak | Best budget before/after |
|---|---:|---:|---:|---:|---:|---:|
| mqa-b4-q1 | 13.78 | 11.47 | 1.201× | 0.366 | 1.57% | 128/128 |
| gqa-b16-4k | 24.62 | 20.79 | 1.184× | 6.456 | 27.65% | 1024/1024 |
| gqa-b64-4k | 66.68 | 62.32 | 1.070× | 8.614 | 36.90% | 16384/8192 |
| gqa-b64-16k | 218.05 | 197.62 | 1.103× | 10.867 | 46.54% | 8192/8192 |
| gqa-b64-16k-q4 | 336.86 | 256.59 | 1.313× | 8.369 | 35.85% | 8192/4096 |
| gqa-b64-4k-scalar | 57.61 | 56.43 | 1.021× | 9.514 | 40.75% | 1024/1024 |
| gqa-b64-4k-plain-v | 68.69 | 61.66 | 1.114× | 8.706 | 37.29% | 8192/1024 |
| gqa-b16-4k-d256 | 47.86 | 39.14 | 1.223× | 6.859 | 29.38% | 2048/512 |
| gqa-b256-16k | 826.38 | 744.21 | 1.110× | 11.542 | 49.44% | 8192/16384 |
| gqa-b64-64k | 822.63 | 740.87 | 1.110× | 11.594 | 49.66% | 16384/16384 |
| gqa-b128-16k | 421.09 | 379.29 | 1.110× | 11.324 | 48.50% | 8192/8192 |
| gqa-b64-32k | 420.38 | 379.18 | 1.109× | 11.327 | 48.52% | 8192/8192 |

The twelve selected workloads improved by 1.021–1.313×, with a 1.137× geometric
mean. A refined B64/64K sweep measured 740.40 µs at budget 16384, yielding
11.602 TB/s useful KV bandwidth (49.69% of reported peak).

| Workload | Backend | Read GiB/call | Unprofiled µs | Physical read TB/s | % peak |
|---|---|---:|---:|---:|---:|
| gqa-b64-16k | before | 1.9943 | 218.05 | 9.820 | 42.06% |
| gqa-b64-64k | before | 7.9402 | 823.20 | 10.357 | 44.36% |
| gqa-b64-16k | after | 1.9939 | 197.62 | 10.834 | 46.40% |
| gqa-b64-64k | after | 7.9421 | 740.40 | 11.518 | 49.33% |

The 50% physical-read target was approached but not reached. B64/64K measured
11.518 TB/s (49.33% of peak); about 729.5 µs is needed for 50% at its measured
traffic, compared with the current latency of 740.4 µs. B64/16K reaches 46.40%
physical-read bandwidth. Counter traffic changed less than 0.03% at the same
budget, so the improvement comes from reduced runtime.

At budget 4096, B64/16K improves 230.76→211.38 µs (1.092×). The expanded
search selects 8192 for both implementations: 218.05→197.62 µs (1.103×).
The runtime default remains 512; the gfx1250 offline tuner default search now
includes 8192/16384.

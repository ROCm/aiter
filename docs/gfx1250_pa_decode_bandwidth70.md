# FP8 PA decode: 70% bandwidth target on gfx1250

Packed FP8 staging now supports D256 as well as D128. D256 full attention with
page128 or larger uses 128-token compute steps, native K128 FP8 QK/PV WMMA,
and a single packed LDS buffer. D128 and windowed D256 retain 64-token steps.
The four-wave launch and 256-token plan contract remain shared across paths.

Full-tile K descriptors collapse token/byte axes to rank two; contiguous V
transfers use rank one. Tail descriptors keep exact token extents and zero-fill
invalid K/V bytes. D256 K staging is `[16, 128, 16]` (dimension chunk, token,
byte); V is `[8, 256, 16]` (token chunk, dimension, byte). Each wave transfers
four K dimension chunks and two V token chunks. WMMA operands read the packed
chunks directly. D256 Q operands stay in registers when one query tile fits.

The packed path carries the previous V normalization factor with its FP32
output. It rescales that output before PV WMMA and uses it as the accumulator,
removing separate post-WMMA output scaling and addition. Empty planned tail
tiles preserve the prior factor so zero new attention mass leaves the output
intact. The final store applies the factor and softmax denominator together.

Native BF16 and multi-tile queries use the existing staging/accumulation path.
Windowed D256 uses 64-token steps: a 128-token pilot exceeded the existing
0.005 tolerance by one output element on a varying-scale MTP4 tail test.
The tolerance was kept unchanged.

## Measurement

Baseline: `62aab3321076ce506a5c8b3d9f6fea85d7403578`, the previous 60% optimization.
Measured 2026-10-09 on gfx1250, 256 CUs, 320 KiB LDS, reported peak bandwidth
23.347 TB/s, FlyDSL 0.3.3.dev886, PyTorch 2.11.0+rocm10.1, HIP 7.16.
GQA uses Hq64/Hkv8, FP8 E4M3FN caches, per-token FP32 scales, BF16 queries/outputs,
page128 and unique shuffled physical pages, unless a row says otherwise.

Seven randomized paired rounds run both versions on identical inputs in one
process. Each timing graph contains 32 decode/reduce calls; each round uses
12 replay windows. Each candidate checks sampled output against an FP32
reference and verifies all output is finite. Planning, quantization,
compilation and Python launch overhead are excluded. Budget searches cover
128/256/512/1024/2048/4096/8192/16384/32768, grouping equivalent plans.
Tests, compilation and profiling do not overlap final timing.

Useful KV bandwidth counts every K/V byte once, excluding scales, queries,
outputs and scratch. Physical reads use ROCprofv3 `FETCH_SIZE` (KiB × 1024),
averaged over ten warmed decode and ten reducer dispatches. Read rates use
independent unprofiled timing because counter collection serializes dispatches.
Physical reads include cache/memory effects and exclude writes.

The 70% physical-read target is **16.3429 TB/s**. Results are specific to the
listed workloads; D128 and small/short workloads have lower utilization.

| Workload | Before µs | After µs | Speedup | Useful KV TB/s | Best budget before/after |
|---|---:|---:|---:|---:|---:|
| MQA B4/4K/D128 | 11.13 | 10.94 | 1.018× | 0.383 | 128/128 |
| GQA B16/4K/D128 | 17.37 | 16.95 | 1.025× | 7.920 | 2048/2048 |
| GQA B64/4K/D128 | 50.63 | 50.76 | 0.997× | 10.576 | 1024/1024 |
| GQA B64/16K/D128 | 151.50 | 145.28 | 1.043× | 14.782 | 2048/2048 |
| GQA B64/16K/D128/MTP4 | 243.52 | 243.70 | 0.999× | 8.812 | 4096/4096 |
| GQA B64/4K/D128 scalar scales | 56.22 | 57.11 | 0.984× | 9.401 | 1024/1024 |
| GQA B64/4K/D128 plain V | 60.29 | 59.28 | 1.017× | 9.056 | 1024/1024 |
| GQA B16/4K/D256 | 38.81 | 27.65 | 1.404× | 9.707 | 512/1024 |
| GQA B64/4K/D256 | 109.19 | 75.95 | 1.438× | 14.138 | 2048/1024 |
| GQA B64/16K/D256/MTP4 | 501.50 | 501.69 | 1.000× | 8.561 | 2048/2048 |
| GQA B64/16K/D256 | 400.66 | 260.64 | 1.537× | 16.478 | 4096/1024 |
| GQA B64/64K/D256 | 1553.63 | 1009.11 | 1.540× | 17.025 | 16384/1024 |
| GQA B64/64K/D128 | 579.93 | 555.09 | 1.045× | 15.475 | 2048/2048 |

| Workload | Backend | Read bytes/call | Unprofiled µs | Physical-read TB/s | % reported peak |
|---|---|---:|---:|---:|---:|
| GQA B64/16K/D128 | before | 2,129,058,288 | 151.50 | 14.053 | 60.19% |
| GQA B64/16K/D128 | after | 2,128,443,523 | 145.28 | 14.651 | 62.75% |
| GQA B64/64K/D128 | before | 8,495,914,141 | 579.93 | 14.650 | 62.75% |
| GQA B64/64K/D128 | after | 8,495,981,165 | 555.09 | 15.306 | 65.56% |
| GQA B64/16K/D256 | before | 4,202,023,341 | 400.66 | 10.488 | 44.92% |
| GQA B64/16K/D256 | after | 4,187,604,403 | 260.64 | 16.066 | 68.82% |
| GQA B64/64K/D256 | before | 16,795,621,670 | 1553.63 | 10.811 | 46.30% |
| GQA B64/64K/D256 | after | 16,729,459,507 | 1009.11 | 16.578 | 71.01% |

D256/B64/64K exceeds the target at **71.01% / 16.578 TB/s**, improving
1,553.63→1,009.11 µs (1.540×). D128/B64/64K improves 579.93→555.09 µs
and reaches 65.56%; it remains below 70%. D256/B64/16K reaches 68.82%.

A second seven-round paired pass confirms D256/B64/64K at 1012.43 µs,
16.524 TB/s and **70.78% physical reads**, using the same measured
read bytes. The scalar-scale control measures 55.45→54.93 µs on that pass;
its earlier 1.6% slowdown does not repeat.

At budget 1024 for both versions, D256/B64/64K improves
1924.14→1009.11 µs (1.907×). The gain also holds
before selecting each version's best budget. Total physical-read bytes fall
by 0.39%, chiefly from fewer reducer slots; most speedup comes from the kernel.

The runtime plan budget remains twice the CU count. The measurements use the
best offline-measured budget for each row; decode performs no online search.
For the measured D256/B64/64K case, use a plan with budget 1024:

```python
plan = plan_pa_decode(
    context_lengths, num_kv_heads,
    query_length=query_length,
    workgroup_budget=1024,
)
```

Measure or use the offline tuner for other workloads.

## Validation and resources

The regression matrix covers native BF16/FP8, BF16/FP16 queries, D64–1024,
page16/64/128, both V layouts, partial/empty contexts, MTP, windows/sinks,
probability tails, graph refresh, odd strides, poisoned scratch and 2/4 GiB
cache boundaries. Packed-tail coverage now spans D128/D256, BF16/FP16 queries,
query lengths 1/4, page64/128, and full/windowed attention with independent
K/V scale patterns varying 256× and NaNs in invalid cache tokens.

The final suite passes **195 tests in 254.54 seconds**. The 32 packed-tail tests
pass separately after the window dispatch change. gfx942/gfx950 MTP4 tile and
reducer compile checks, Black, Ruff and whitespace checks pass. Native BF16
paired checks are effectively unchanged: B16/4K 33.572→33.508 µs and
B64/16K 427.343→427.002 µs. These BF16 values are not FP8 bandwidth results.

| Specialization | VGPR before/after | SGPR before/after | Static LDS before/after | Spills/private bytes |
|---|---:|---:|---:|---:|
| FP8 D128/G8/MTP1 | 114/118 | 87/94 | 37,952/21,056 | 0/0 |
| FP8 D256/G8/MTP1 | 158/228 | 90/90 | 83,008/73,792 | 0/0 |
| FP8 D128/G8/MTP4 | 158/158 | 90/90 | 46,400/46,400 | 0/0 |
| BF16 D128/G8/MTP1 | 104/104 | 78/78 | 79,936/79,936 | 0/0 |

Single-specialization ISA captures use fresh directories with FlyDSL runtime
caching disabled. The resource-diff helper flags increased registers for the
packed paths; they are retained based on correctness and measured performance.
No CDNA occupancy formula is applied to gfx1250. ATT remains unavailable with
the installed decoder (`Error loading decoder: 37`); no stall attribution is
claimed. Analysis follows FlyDSL's
[isa-resource-diff](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/isa-resource-diff/SKILL.md),
[prefetch-data-load](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/prefetch-data-load/SKILL.md),
and
[lds-optimization](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/lds-optimization/SKILL.md)
workflows.

Raw paired samples, counter CSVs, experiment snapshots, resource comparisons,
and reproduction commands are recorded in
`/home/sixifang/gfx1250-dev/pa-gfx1250-validation/BANDWIDTH70_PERFORMANCE.md`.
The earlier 50%/60%, BF16, TDM, PR5809 and PR3256 reports are preserved.

# FP8 PA decode: exceeding 60% physical-read bandwidth on gfx1250

The FP8 D128 decode kernel now retains the cache's packed 16-byte chunks in
LDS. This lets TDM transfer contiguous spans without transposing rows or
inserting padding. WMMA operands assemble directly from packed LDS chunks.
For the measured B64/64K workload, decode plus reduction improves from
740.32 to 579.20 µs, and physical-read bandwidth increases from 49.34% to
62.83% of the reported 23.347 TB/s peak.

The packed K tile is `[4, 8, 16, 16]` (wave, dimension chunk, token, byte);
V is `[4, 128, 16]` (token chunk, dimension, byte). Four single-wave TDM descriptors transfer one
16-token chunk each. The exact tail extent zero-fills invalid V bytes.

Packed staging applies to per-token FP8 caches, D128, transposed V, page sizes
64 or larger, and one 16-row query tile. Other specializations retain padded
staging. A paired MTP4 experiment with packed staging was about 1% slower;
retaining padded staging improves the final MTP4 result by 5.6%.

The kernel also loads independent V operands before the probability exchange
barrier, and uses `afn` for softmax exponentials to avoid library underflow
fixups. Infinity handling remains enabled for masked rows and empty history.
K/V remain double-buffered, with 64-token compute steps and 256-token plan tiles.
Queries/outputs can use BF16 or FP16; native BF16 caches retain their existing
WMMA/TDM path.

## Measurement

Measured on 2026-10-09 against `da68eb89956400b96daa66a0e0836443ebe632c9`,
on gfx1250 with 256 CUs, 320 KiB LDS, FlyDSL 0.3.3.dev886, PyTorch
2.11.0+rocm10.1 and HIP 7.16. GQA uses Hq64/Hkv8, D128/page128 unless
specified, FP8 E4M3FN caches, per-token FP32 scales, BF16 queries/outputs and
shuffled physical pages owned by distinct sequences. Every candidate checks
sampled output against an FP32 reference and checks all output for finiteness.

Both implementations use the same inputs in one process. Seven randomized
rounds use graphs of 32 decode/reduce calls and 12 replay windows per candidate.
Planning, quantization, compilation and Python launch overhead are excluded.
Budget searches cover 128/256/512/1024/2048/4096/8192/16384/32768, with equivalent
plans grouped. GPU tests and profiling do not overlap final timing.

Useful KV bandwidth counts each K/V byte once, excluding scales, queries,
outputs and scratch. Physical reads use ROCprofv3 `FETCH_SIZE` (KiB × 1024),
averaged over ten decode and ten reducer dispatches. Rates use independently
measured unprofiled latency because counter collection serializes dispatches.
Physical reads include memory/cache effects and exclude writes.

The 60% target is **14.0082 TB/s**. The results below exceed it for the two
specified streaming workloads. B64/16K is only 0.11 percentage points above
the target, so B64/64K provides the stronger margin. Other shapes have lower
bandwidth utilization.

| Workload | Before µs | After µs | Speedup | Useful KV TB/s | % reported peak | Best budget before/after |
|---|---:|---:|---:|---:|---:|---:|
| MQA B4/4K | 11.48 | 11.26 | 1.019× | 0.372 | 1.60% | 128/128 |
| GQA B16/4K | 20.88 | 17.31 | 1.206× | 7.752 | 33.20% | 1024/2048 |
| GQA B64/4K | 63.07 | 50.52 | 1.248× | 10.626 | 45.51% | 8192/1024 |
| GQA B64/16K | 197.88 | 151.64 | 1.305× | 14.162 | 60.66% | 8192/2048 |
| GQA B64/16K/MTP4 | 256.89 | 243.33 | 1.056× | 8.826 | 37.80% | 4096/4096 |
| GQA B64/4K scalar scales | 58.08 | 55.96 | 1.038× | 9.594 | 41.09% | 1024/1024 |
| GQA B64/4K plain V | 61.11 | 59.30 | 1.031× | 9.053 | 38.77% | 1024/1024 |
| GQA B16/4K/D256 | 39.02 | 38.80 | 1.006× | 6.918 | 29.63% | 512/512 |
| GQA B64/64K | 740.32 | 579.20 | 1.278× | 14.831 | 63.52% | 16384/2048 |

| Workload | Backend | Read bytes/call | Unprofiled µs | Physical-read TB/s | % reported peak |
|---|---|---:|---:|---:|---:|
| GQA B64/16K | before | 2,141,904,835 | 197.88 | 10.824 | 46.36% |
| GQA B64/16K | after | 2,128,215,830 | 151.64 | **14.035** | **60.11%** |
| GQA B64/64K | before | 8,528,080,490 | 740.32 | 11.519 | 49.34% |
| GQA B64/64K | after | 8,496,649,238 | 579.20 | **14.670** | **62.83%** |

Selecting fewer workgroups reduces reducer reads: B64/64K falls from 33.44 MB
to 4.31 MB per call. Decode reads change by less than 0.04%. At the same
budget 8192, B64/16K improves 197.88→159.17 µs; at budget 16384,
B64/64K improves 740.32→588.42 µs. The gain therefore persists before
selecting the lower final budget.

The runtime default budget remains twice the CU count. For these measured
streaming shapes, build a plan with the selected budget:

```python
plan = plan_pa_decode(
    context_lengths, num_kv_heads,
    query_length=query_length,
    workgroup_budget=2048,
)
```

Use the offline tuner or measure a budget for other shapes. Decode does not
perform an online tuning search.

## Validation and resources

The final 167-case GPU suite passes FP8/native BF16, BF16/FP16 queries,
page16/64/128, D64–1024, both V layouts, partial/empty contexts, MTP,
windows/sinks, probability tails, graph refresh, odd strides, poisoned scratch
and 2/4 GiB cache boundaries. Four added tests exercise page64/128 packed tails
with NaNs in invalid cache tokens, independent K/V scale patterns spanning
256× and query lengths 1/4 that fit one query tile. A rank-2 V TDM descriptor
clamps exact valid-token extents and zero-fills invalid bytes, preventing
`0 * NaN` in WMMA. The retained selection policy is checked against the full
shared, gfx1250 and BF16 suites.

Single-specialization ISA comparisons use fresh dump directories with the
FlyDSL runtime cache disabled:

| Specialization | VGPR before/after | SGPR before/after | Static LDS before/after | Spills/private bytes |
|---|---:|---:|---:|---:|
| FP8 D128/G8/MTP1 | 108/114 | 90/87 | 44,096/37,952 | 0/0 |
| FP8 D128/G8/MTP4 | 154/158 | 86/90 | 46,400/46,400 | 0/0 |
| BF16 D128/G8/MTP1 | 102/104 | 78/78 | 79,936/79,936 | 0/0 |

The resource-diff tool flags the FP8 and BF16 register increases. MTP4 timing
improves; BF16 paired measurements are effectively unchanged (B16/4K:
33.60→33.52 µs; B64/16K: 428.48→427.27 µs). These BF16 bandwidth numbers
are not used for the FP8 target. Existing BF16 D1024/G16/MTP4 spilling remains
documented in the BF16 report.

Compile checks pass for gfx942/gfx950 MTP4 tile/reducer specializations.
The full suite passed 167 tests in 224.48 seconds. Black, Ruff and whitespace
checks pass.

A separate contiguous TDM read benchmark reaches 19.44 TB/s (83.3% of reported
peak) for 4 GiB of reads. It establishes headroom in raw data delivery; it is
not the PA decode result and includes a small store to keep the loads live.
ATT remains unavailable with the installed decoder (`Error loading decoder:
37`); no instruction-stall attribution is claimed. Analysis follows FlyDSL's
[isa-resource-diff](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/isa-resource-diff/SKILL.md)
and
[prefetch-data-load](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/prefetch-data-load/SKILL.md)
workflows.

Raw samples, experiment snapshots, counter CSVs and reproduction commands are
in `/home/sixifang/gfx1250-dev/pa-gfx1250-validation/BANDWIDTH60_PERFORMANCE.md`.
Earlier PR5809/PR3256 comparisons are preserved in the previous reports;
the before/after numbers here compare against the immediately preceding
implementation, `da68eb899`.

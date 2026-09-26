# GLM-5.3 IQ2R: isolated TP-rank MoE analysis

The serving goal remains unmet. Production sweeps are paused by user request.
This work uses one MI355X GPU to represent one TP8 or TP4 rank, without loading
a complete model. E285 additionally stages individual real checkpoint slices. Dense E199+ candidate kernels remain isolated. E225 restores a previously
selected C4/C8 frontend omitted from the consolidated production source. Official comparison remains unchanged ATOM
`benchmark_serving` at 1k/1k and 8k/1k, C1 through C256.

## What is measured

BF16 hidden states plus supplied top-9 routes → task sorting/input quantization
→ gate/up/SwiGLU/intermediate quantization → down → BF16 route reduction.
Router projection/top-k and TP all-reduce are excluded. These are microseconds
per local MoE call, not concurrency measurements or serving throughput.

There are 257 experts (256 routed and one always-on shared). TP8 gate shape is
[257,512,6144] and down is [257,6144,256]; TP4 doubles the intermediate width.
Inputs/routes are identical across arms. MXFP4 is quantized from materialized
synthetic IQ2R values. Actual MXFP4 dispatch is A4W4; IQ2R decodes to FP8 and
uses FP8×FP8 MFMA. Smaller stored weights do not imply half the total traffic.

Clean measurements rotate 32 distinct weight banks, use seven alternating
rounds, and bracket separate rocprof traces/counter passes. Candidates must
match the original IQ2R output exactly as graph inputs and routes change.
Cross-format model quality is not established by synthetic tensors.

## M4 latency checkpoint — E312–E322

The TP8 M4 candidate now reaches local synthetic hot-route parity with MXFP4
while retaining a spread-route advantage. The 0.5% hot difference is small;
it should be treated as parity, not a robust lead. TP4 M4 also has fresh stable
comparisons ahead of MXFP4. Each row uses one fixed candidate policy for both
route patterns, complete clean bookends and 32 rotating weight banks.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
| 8 | 4 | spread | 25.26 | 23.24 | -8.0% |
| 8 | 4 | hot | 19.99 | 19.89 | -0.5% |
| 4 | 4 | spread | 36.93 | 32.60 | -11.7% |
| 4 | 4 | hot | 28.47 | 25.78 | -9.4% |

These are complete single-GPU TP-rank MoE-call times, not serving concurrency.
TP8 uses E317 wave sorting/paired FP8 conversion plus E321 two-token down reuse
when a GPU ballot confirms identical expert slots. Other routes use the timed
per-token fallback. TP4 uses the same frontend with E319 wave-private codebook
completion. Down retains the final cross-wave reduction barrier.

At TP8 hot routing, E321 pair2 halves down MFMA instructions (27,648 to 13,824)
and reduces VALU from 1.42M to 0.82M. Spread MFMA work is unchanged. All native
kernels have zero scratch. E321 uses five-second warmup after several earlier
attempts had excessive timing drift. E319 TP8, E314 and E316 drift is retained
and those timing tables are provisional.

E318 validates input FP8/scales, gate FP8/scales and final BF16 at TP8/TP4;
240 separate frontend checks include invalid IDs and scale extremes. E322 adds
changing reuse guards, reordered expert slots and invalid routes: 1,536 exact
graph/eager checks and 1,152 comparisons with original IQ2R across both TPs.
Its first attempt failed only in post-validation invalid-ID count reporting;
the corrected checker and original failure are preserved, with no runtime or
tolerance change. E320 independently qualifies E317 frontend on 16 actual TP8
captures and six TP8/TP4 weight slices. TP4 reuses TP8 captures. Captured
qualification of E321 down is still required.

E312/E313 fused gate/frontend attempts regress. E314 replaces repeated large
histogram scans with wave sorting but fusion still loses. E315 moves the wave
sort to the standalone frontend and wins; E316 cooperative quantization loses.
E317 paired FP8 conversion adds a small gain; early loads regress. E319 removes
an unnecessary initial workgroup barrier. E321 reuses weights across matching
tokens and assigns their reductions to separate waves.

The overall goal remains unmet. Larger-token and dense gaps remain, including
the previously measured 9–34% dense TP4 deficit. No new production runtime is
integrated and serving sweeps remain paused. Next: actual-capture qualification
of the new down path and fresh M32/M64/M128/M256 MoE attribution, then qualified
integration and unchanged ATOM benchmark_serving acceptance.

## TP4 follow-up and dense layout qualification — E304–E311

The E302 combined small-token candidate has fresh qualified TP4 M16 results.
TP4 M4 also passed correctness, but its 4–5% timing drift prevents qualification.
TP4 M8's earlier unchanged-MXFP4 bound failure remains unresolved.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
| 4 | 16 | spread | 91.83 | 79.70 | -13.2% |
| 4 | 16 | hot | 31.13 | 26.03 | -16.4% |

E308 changes dense down records from groups of three N16 blocks to groups of
four, with N512 workgroup tiles and matching reduction indexing. Separate
transformed tensors preserve exact decoded weights/scales, original allocation
size, M32 rows and one/two interleaved column batches. No legacy decoder receives
quad records. E262 gate and routing policy remain unchanged.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
| 4 | 1024 | spread | 292.89 | 319.97 | +9.2% |
| 4 | 1024 | hot | 187.53 | 204.78 | +9.2% |
| 4 | 4096 | spread | 601.03 | 803.23 | +33.6% |
| 4 | 4096 | hot | 499.58 | 650.75 | +30.3% |

Each row averages fresh complete clean bookends over 32 rotating weight banks.
The dense N512 candidate improves previous E262 by 0.7–3.5%; all four cases
have less than 2.1% timing drift. The dense MXFP4 gap remains about 9–34%.
These are synthetic local MoE-call times, not serving concurrency.

E311 qualifies unchanged E308 and E262 across 39 eligible TP4 shape/routing
cases and 312 changing steps, including zero/16x inputs and dense/chunk
boundaries: 936 exact checks across clean/trace/clean. Final BF16 and intermediate
FP8 bytes/scales match original IQ2R after restoring original route order.
Scatter permutations and gather/scatter inverses are checked. The initial raw-row
checker failed on E262 because the sorters use different valid row orders;
the diagnostic and both checkers are preserved. No runtime or tolerance changed.
Native E308 gate/down/reduction dispatch is verified with zero scratch. Actual
dense captures and production fallback handling remain unqualified.

E304 stages persistent-down metadata and cuts vector reads by 62%, but remains
much slower than E261. E305/E306 shorten live state or change compiler residency
hints without meaningful speedups. E307 TP8 quad down gives mixed small gains;
E309/E310 M4 quad adaptations do not improve the selected combination. These
remain unselected. E312 is a separate fused frontend/gate POC in progress.

No new production runtime is integrated. Serving sweeps remain paused. Earlier
model-quality, TP4 baseline-bound and independent scalar-MFMA bound issues remain
open; these checks do not clear them or establish serving parity.

## Combined small-token qualification — E293–E303

E302 measures the unchanged E293 binary with the E297 combined policy:
E280 exact packing/gate scheduling, E286 independent down waves only at TP8
M16, E287 static M16 routing, E289 batched reduction, E292 M4 down epilogue,
and E293 four-column-partition input quantization.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
| 8 | 4 | spread | 25.34 | 24.41 | -3.7% |
| 8 | 4 | hot | 20.43 | 22.18 | +8.5% |
| 8 | 8 | spread | 36.46 | 35.59 | -2.4% |
| 8 | 8 | hot | 26.76 | 22.49 | -16.0% |
| 8 | 16 | spread | 48.95 | 46.64 | -4.7% |
| 8 | 16 | hot | 28.18 | 24.71 | -12.3% |

Each row averages complete clean bookends over 32 rotating banks. One second
of synchronized rotating-graph warmup precedes each case and profiling pass;
measurement rounds and fixed correctness bounds are unchanged. Every baseline
and candidate bookend has less than 1% drift. All seven phases pass with exact
IQ2R checks and native dispatch, zero scratch. Five of six cases beat fresh
MXFP4; TP8 M4 hot remains 8.5% slower. These are synthetic one-rank MoE times,
not serving concurrency or checkpoint model-quality comparisons.

E297 qualifies 30 synthetic TP8/TP4 shape/routing cases, 240 changing steps,
720 exact checks and 45 frontend ordering/invalid-route tests. E299 qualifies
the same combination on 16 actual TP8 M4/M8 captures and six real weight
slices: 32 capture/TP cases, 256 changing steps and 768 exact checks. Final
BF16 and intermediate FP8 bytes/scales match original IQ2R. TP4 reuses TP8
hidden/routes with TP4 weights; M16 remains synthetic-only.

On these captures, combined versus original IQ2R is 23.15 versus 24.80 µs
at TP8 M4, 31.48 versus 33.21 at TP8 M8, 30.66 versus 32.83 at TP4 M4,
and 43.76 versus 48.84 at TP4 M8. These are not MXFP4 comparisons.

E294 compact M4 weight reuse, E295 dense batched reduction, E296 temporary
FP8 gate expansion, E298 global M4 codebook reads and E300–E303 persistent
fused down remain unselected. E303 removes 96.5% of measured LDS bank
conflicts without improving time; repeated route metadata loads are the
next candidate. E293/E294/E295 scout timing drift remains documented and
does not replace the stable E302 results.

No new production runtime is integrated. TP4 M8's unchanged-MXFP4 fixed-bound
failure remains unresolved; no new TP4 comparison is claimed here. Earlier
model-quality and independent scalar-MFMA error-bound issues remain open.
Dense TP8/TP4 gaps and serving qualification remain. Serving sweeps stay paused.

## Follow-up results — E285–E292

E289 combines E280 packing/gate scheduling, independent down-projection waves at TP8 M16, the static M16 frontend and batched final reduction reads. E292 separately parallelizes the existing M4 fused-down epilogue. These are shape-specific isolated experiments, with no new production integration.

| Experiment | TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R latency overhead |
|:---|---:|---:|:---|---:|---:|---:|
| E292 | 8 | 4 | spread | 25.30 | 24.47 | -3.3% |
| E292 | 8 | 4 | hot | 19.97 | 22.23 | +11.3% |
| E289 | 8 | 16 | spread | 49.86 | 48.88 | -2.0% |
| E289 | 8 | 16 | hot | 28.59 | 25.51 | -10.8% |
| E292 | 4 | 4 | spread | 36.20 | 32.64 | -9.8% |
| E292 | 4 | 4 | hot | 27.86 | 26.32 | -5.5% |

Each row averages complete clean bookends over32 rotating banks; all baseline/candidate drift in this table is below3%. E289 TP8 M16 and E292 TP4 M4 include full counters. The selected M16 spread case is now faster than its fresh MXFP4 control. TP8 M4 hot remains behind. Token lists and RNG progression differ between experiments; compare arms within a row. These are synthetic one-rank MoE calls, not serving concurrency.

E285 qualifies unchanged E280 using16 actual TP8 captures and six real TP8/TP4 checkpoint slices:32 capture/TP cases and256 changing steps, with exact final BF16 and intermediate FP8 bytes/scales. TP4 reuses TP8 hidden/routes with TP4 slices. E280 reduces local latency against original IQ2R by4.9/4.7% at TP8 M4/M8 and5.3/6.1% at TP4 M4/M8. These real-capture comparisons do not include MXFP4.

E291 qualifies the E289 combination across30 synthetic shape/routing cases and240 changing steps, including zero/16× inputs and12/13 boundaries. Final outputs and intermediate values/scales are exact. Native dispatch is verified, zero scratch, and all45 stable-order/invalid-route/changing-graph frontend tests pass. E285 and E291 cover different candidates; they do not substitute for captured qualification of the newly combined path.

E286 down removes cross-wave partial-sum transfers: TP8 M16 spread down22.66→16.69µs, unchanged DRAM/MFMA counts,31% fewer LDS instructions and49% fewer LDS-wait cycles. E289 reduction batches independent reads: memory instruction counts and traffic remain essentially unchanged, while WAIT_ANY cycles fall56%. Neither result establishes higher achieved occupancy.

E288 TP4 adaptive fusion and E290 wider route9 tiles are unselected due regressions. E289 TP4 r1 is incomplete because unchanged MXFP4 exceeded the fixed graph/eager bound (0.0153846 versus0.015); no TP4 M8 comparison is qualified. E292 r1 TP4 has elevated candidate drift and is retained; r2 supplies the stable table above. Dense E261/E262 results below remain unchanged. Serving sweeps stay paused.

## Small-token scheduling checkpoint — E278–E283

E280 combines exact index/sign packing, shorter sign-temporary lifetimes, and
four independent codebook reads per batch. Native gate VGPRs fall from E275's
129 to 102, restoring two theoretical CTAs/CU with no spills. E280 TP8/TP4 r2
include clean bookends and all four rocprof counter groups. Original IQ2R
checks remain exact across changing inputs/routes. These are synthetic local
MoE-call times; tokens are not serving concurrency.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
| 8 | 4 | spread | 25.21 | 24.67 | -2.2% |
| 8 | 4 | hot | 20.06 | 22.37 | +11.5% |
| 8 | 8 | spread | 36.52 | 35.66 | -2.4% |
| 8 | 8 | hot | 26.92 | 22.56 | -16.2% |
| 8 | 16 | spread | 48.72 | 54.72 | +12.3% |
| 8 | 16 | hot | 28.29 | 26.84 | -5.1% |
| 4 | 4 | spread | 36.03 | 32.83 | -8.9% |
| 4 | 4 | hot | 27.74 | 26.51 | -4.4% |
| 4 | 8 | spread | 55.12 | 54.23 | -1.6% |
| 4 | 8 | hot | 23.15 | 27.40 | +18.3% |
| 4 | 16 | spread | 93.41 | 81.75 | -12.5% |
| 4 | 16 | hot | 31.04 | 28.85 | -7.1% |

The same E280 policy is used for every small-token row; no route-pattern
selector or best-of-variant table is used. E278 isolated sign scheduling fixed
the residency cliff; E280 batching adds a further gain. Batching does not lower
every wait counter: at TP8 M8 spread, E278→E280 gate trace is17.17→16.35µs while
LDS-wait cycles rise414,794→514,652. Scheduling and residency must be assessed
using completed work/time, rather than treating a single wait count as latency.

E279 adaptive down extension and E281 route9 extension are unselected because
of spread-route or broad regressions. E282 applies the new decoder schedule to
the dense gate: exact, but slightly slower than E261, so dense selection is
unchanged. E283 down scheduling fails correctness in its TP8 four-read and TP4
one-read arms and is ineligible pending diagnosis. E284 passes30 TP8/TP4 shape/pattern cases and240 changing steps with
exact final outputs and intermediate FP8 bytes/scales; actual captures remain. None of E275–E283 is integrated into production.

The goal remains unmet. TP8 M4 hot/M16 spread and TP4 M8 hot remain behind in
this small fixture. Dense gaps, boundary/intermediate/captured-input checks,
production integration, official serving bookends and real-MTP quality remain.

## Current measured position

TP8 uses E261 and TP4 uses E262: adjacent9-bit index packing, exact static
normalization of codebook signs, and an M32 down tile. At1024 tokens down and
reduction use one column batch; at4096 they use two interleaved batches. Original
reference weights have signed codebooks, and candidate transformations retain
all decoded FP8 bytes and storage sizes. This is a synthetic fixture change
from earlier positive-book tables, with matched references within every run.
Each row averages complete E261 r3 or E262 r2 clean bookends. Neither candidate
is integrated; the shape policy needs wider qualification.

| TP | Tokens | Routes | MXFP4 µs | IQ2R candidate µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
|8|1024|spread|213.38|215.14|+0.8%|
|8|1024|hot|124.76|143.96|+15.4%|
|8|4096|spread|452.02|518.98|+14.8%|
|8|4096|hot|395.27|441.25|+11.6%|
|4|1024|spread|294.69|323.04|+9.6%|
|4|1024|hot|190.81|212.45|+11.3%|
|4|4096|spread|606.06|817.97|+35.0%|
|4|4096|hot|500.69|678.25|+35.5%|

No automatic route-pattern selector is qualified. Original TP4 dense samples
have unresolved intermittent outliers and are not a qualified denominator.
The first E218 trace has host-marker clock-alignment failures; its clean bookends
are valid. E218 r2 and E220 r2 use strict trace containment with a host-only
marker margin and provide valid stage attribution.

## Where the time goes

E235 full counters on 4,096 spread tokens reduce gate DRAM from 990.6 to 543.3 MB
while trace time barely changes: 232.55 to 231.61 µs. About 78.1M VALU and 3.714M
FP8 MFMA instructions remain. This supports reducing decoder and instruction
work as well as traffic. FP8-specific counters are zero for MXFP4 and do not
measure FP4 work. Device metadata reports 160 KiB LDS per CU/block and eight
maximum waves per SIMD. E243/E244 full counter follow-ups are complete. E243 M4096 spread gate
reduces VALU instructions 78.12M → 70.51M and all-wait cycles 58.87M → 41.50M;
trace time falls 226.70 → 217.65 µs. LDS-specific waits rise 4.04M → 11.41M;
occupancy stays near 19%. The useful metric is completed work/time, not a
single wait category.

E244 M4096 spread down reduces VALU 121.68M → 97.72M, vector reads
2.18M → 1.37M and DRAM 1226.85 → 849.52 MB. It increases registers
108+4 → 20+132 and measured occupancy falls 40.45% → 25.10%. Its down trace
is slower (365.39 → 390.46 µs), while the complete clean block improves
941.25 → 910.21 µs and the gate trace improves 411.70 → 375.96 µs.
These data do not establish a standalone down-stage speedup; they motivate
trading some record reloads for fewer live accumulators. Counter passes and
trace timestamps are diagnostic; clean block time decides candidate wins.


E252's full follow-up at TP4 M4096 spread reduces down VALU instructions
97.72M → 83.32M, with unchanged 3.716M FP8 MFMA instructions and 1.374M
vector reads. Down trace falls399.66 → 383.77µs; measured occupancy stays
about25.2%. The paired clean block improves917.12 → 904.01µs. All four
paired cases improve about1.3–1.6%, but the dense TP4 gap is still large.

At TP8 M4096 spread, E243 full-MoE profiled DRAM is1.789GB versus1.923GB
for MXFP4: only7.0% fewer bytes at the complete boundary. Down plus route
reduction already account for1.211GB of IQ2R traffic. The large per-route
BF16 output is written and then gathered for reduction, so halving stored
weight bytes cannot halve total MoE traffic. These counters come from
separate profiling passes and are not an instantaneous bandwidth reading.

E220 profile-r2, 4096 spread, rocprof attribution:

| Stage | MXFP4 trace µs | IQ2R candidate trace µs | IQ2R DRAM MB |
|:---|---:|---:|---:|
|Gate/up|about 141|229.60|990.9|
|Down|165.10|222.11|1075.7|
|Reduction|107.86|115.50|about 506|

The original one-workgroup sorter was a separate large bottleneck; the parallel
sorter removes most of that cost on dense batches. Sharing activations and wider
tiles reduce repeated traffic. E214 drops gate wait cycles from 195.04M to 60.27M
while VALU instructions stay around 78M, supporting latency overlap. E220 down
still executes about 77M VALU instructions versus MXFP4's 14.2M and transfers
1075.7 MB versus 725.6 MB. Current gate/down winners have zero scratch spills.
Both decoder/instruction work and repeated traffic remain material costs.

E218 r2 TP4 gate improves from 646.37 to 419.81 µs versus MXFP4 262.11 µs.
Its down remains 558.09 µs versus MXFP4 201.69 µs, with 2384.9 versus 1017.9 MB
of DRAM traffic. E222 reduces whole-block TP4 time further; it remains behind.

Chunked occupancy correction: original E261/E262 summaries added occupancy
rates across two sequential down/reduction launches. The preserved
`summary-v2.json` files instead use duration-weighted rates from the same
occupancy pass. At4096 spread, down occupancy is36.80% atTP8 and38.51% atTP4.
Clean times, trace times, additive counters and the published comparison table
are unchanged. Original summaries remain preserved for artifact audit; use v2
for occupancy rates. See `experiments/single-gpu-report/occupancy-correction-r1.json`.

## Corrected small-token baseline

E225 found that the consolidated source omitted the E167 C4/C8 static frontend.
The environment flag was silently ignored; output-equality tests missed it.
Official E180 r8 logs did use the static frontend. The restoration is now pushed,
with tests of actual GPU kernel names: before 2 failed/3 passed, after 54 GPU
tests passed across dispatch, quantization/routing, and complete TP8/TP4 MoE.

Fresh corrected TP8, four spread tokens: MXFP4 25.04 µs, generic IQ2R 27.30 µs,
static IQ2R 26.47 µs. The gate is already faster than MXFP4 in this case;
frontend/down account for the remaining gap. Eight spread tokens: MXFP4 37.26,
static IQ2R 37.70 µs. These are synthetic token counts, not serving concurrency.
TP4 static-vs-generic checks and clean bookends pass exactly. A full small-token
MXFP4 comparison is withheld: two runs exceeded the unchanged atomic-output
validation threshold at M8 spread (max normalized 0.0151515 vs limit 0.015).
Dense E199+ results are unaffected by this small-token dispatch correction.

## Experiment index

| Experiments | Change and finding |
|:---|:---|
|E199|Rebuilt isolated baseline; full boundary and rocprof attribution. Original dense IQ2R has excess traffic and instructions, not merely low occupancy.|
|E200|Parallel histogram/prefix/scatter sorter. Strong dense gain; inappropriate as an unconditional small-token policy.|
|E201|Wider down register loading. Rejected.|
|E202|Shared down activations across N waves, preserving two K128 partials. Useful dense foundation.|
|E203–E204|Wider gate tiles and shorter live ranges. E204 M64/N128 without lookahead was retained as an experimental control.|
|E205|Extend dense eligibility to 512–2047 tokens. Large synthetic gain; boundary/real-capture qualification pending.|
|E206|Vector output stores through LDS. Exact but slower; rejected.|
|E207|Shared gate activations across N waves. Cuts DRAM substantially, but little spread gain and hot regression without a pipeline.|
|E208|Finish one down N16 atom at a time. Shorter lifetimes improve M32; exact, no spills.|
|E209|Larger vectorized route-reduction tiles, same ordered FP32 FMAs. Small gains; combined with E214/E220 in the latest TP8 candidate.|
|E210|TP4 gate/down adaptation. Loading only the current K record removes spills; still behind MXFP4.|
|E211|Task grouping and XCD remapping. Small configuration-dependent gains only.|
|E212|Reduce gate accumulators by narrowing N work per wave. Strong initial regression; second bookend invalid due overlap, not a qualified selection.|
|E213|Materialized FP8 diagnostic excludes decode time and expands storage. Independent FP8 arithmetic reference is unresolved; not an IQ2R optimization or a lower bound.|
|E214|Two LDS activation slots plus one compressed-weight lookahead. Current leading TP8 gate, exact synthetic outputs, no spills.|
|E215|Buffer input/output addressing. R1 bounds bug fixed; r2 exact and slightly faster.|
|E216|MFMA operand transposition enables adjacent vector output stores. Exact, modest gain with four waves/M32.|
|E217|Reuse down activations across more N48 groups while keeping only current weights/accumulators live. Exact; M64/two groups wins spread and M32/two groups favors hot.|
|E218|TP4 adaptation of E214 gate and transposed down. Exact, faster than stable E210 control; still behind MXFP4.|
|E219|M32/M16 slabs reduce gate live state but repeat weight decode. Exact, slower; rejected.|
|E220|Async down activation/scales loads, wider output groups, next-group compressed weights. Current leading TP8 down; exact, no spills. Combined vector8 reduction improves further.|
|E221|Keep activation fragments in registers across atoms/groups. Exact, neutral or slower; not selected.|
|E222|TP4 down adaptation with four K128 partial sums and per-K lookahead. Exact, faster than E218 control; still behind MXFP4.|
|E223|Exact reciprocal/division epilogue changes; exhaustive BF16 checks passed but no general gain. Not selected.|
|E224|Finish one M16 row atom at a time in down. Exact, slower; rejected.|
|E225|Restore omitted E167 static frontend and test actual dispatch. Pushed production correction; fresh TP8 small-token comparison and exact TP4 IQ2R checks.|
|E226|M128 gate with half-quad N32 waves. Exact and spill-free, but gate/whole-block slower; rejected.|
|E227|Transpose dense gate MFMA for direct wave-local SwiGLU/quantization. Exact, no broad gain; not selected.|
|E228|LDS sign-mask lookup replaces integer expansion. Gate slower; down roughly neutral. Not selected.|
|E229–E230|Retune only down grid size on current TP8/TP4 kernels. Exact, small shape-dependent gains; no general selector selected.|
|E231|Corrected MFMA32 primitive matches current MFMA16 bit for bit across 262144 FP32 outputs each for unit and varying scales. Independent scalar roundoff-bound issue affects both and remains open.|
|E232|Exact compressed-record repacking and MFMA32 down. Whole MoE exact; slower than E220.|
|E233|Add next-group weight lookahead to MFMA32 down. A harness data-selector bug was caught and fixed; corrected run exact, no performance win.|
|E234|MFMA32 gate with K64 activation pipeline and direct epilogue. Exact, no spills, slower; rejected.|
|E235–E236|Revisit XCD/task ordering on current TP8/TP4 kernels. Exact, modest gains. TP8 full counters show gate DRAM nearly halved without a meaningful gate-time change; decoder/instruction work remains. TP4 clean gain 3.5–10.3%, still behind MXFP4.|
|E237|Decode weights once into LDS and share among M waves at M64/M128. Exact but slower in every tested shape; rejected.|
|E238|Decode each active expert into call-local FP8 scratch and pay decode inside every timed call. Exact, no overall gain; rejected.|
|E239|Last producer performs ordered route reduction after release/acquire signaling. Completed checks exact; severe regression, clean screen stopped intentionally. Separate trace-only evidence retained.|
|E240–E241|Remove branches before padded-row MFMAs. Exact; small broad TP8 gate gain, TP4 shape-dependent.|
|E242|Sign nibbles via wave shuffle or 64-byte LDS lookup. Exact, slower or neutral; rejected.|
|E243|Use a single activation scale byte and unroll the gate K loop by two; combine with XCD/task ordering. Exact, modest broad TP8 gains. Larger unrolls regress.|
|E244|TP4 gate adaptation; move down K loop outside the three output atoms so each record is consumed once. Exact, broader dense gains; full counters show fewer instructions/bytes but lower down occupancy. Further qualification pending.|

|E245|Batch independent codebook reads, then load each activation fragment close to use. Exact; small shape-dependent gains, no broad selector. Compiler fences regress.|
|E246|Full sign-expanded codebook, with its copy cost included. Exact, larger LDS allocation, slower; rejected.|
|E247|Half-expanded sign codebook with smaller copy/LDS cost. Exact, still slower; rejected.|
|E248|Decode active down experts into call-local FP8 scratch on every timed call. Exact, slower; initial decoder launch imbalance and missing consumer lookahead motivated E250.|
|E249|TP4 processes two N atoms per K loop and tests full short-K unrolling. Exact, modest broad improvement with M64/unroll4; smaller unrolls regress.|
|E250|Direct expert-indexed decode plus FP8 weight-fragment lookahead repairs E248 scheduling costs. Exact, much faster than E248 but still slower than fused IQ2R.|
|E251|Ordinary and scaled FP8 MFMAs match bit for bit; CPU/GPU double references agree. Controlled probes localize extra rounding to eight-product groups. Exact random-dot semantics and justified bound remain unresolved.|
|E252|Fully unroll TP4 down's four K128 iterations while retaining all three N atoms. Exact, modest broad gain over E244; full counters show about15% fewer down VALU instructions with similar occupancy.|
|E253|Store/reduce the BF16 route buffer in column-tile-major order without a transpose. Exact, no consistent gain; not selected.|
|E254|Precompute 16-bit codebook offsets and final scale bytes in static gate records. Exact representation, 3.25 rather than 2.25 bits/weight excluding codebook. Exact; about4% fewer gate VALU instructions but more traffic and a large M1024 spread regression. Unselected.|

|E255|Independent M32 tasks with current ordering. Exact and lower register pressure, but hot/4096-token regressions outweigh a small M1024 spread gain. Unselected.|
|E256|Separate low/high LDS codebook planes and three-dword entries. Exact, zero scratch; both layouts are slower in all four shapes. Unselected.|
|E257|Reduce a token/N384 tile on ninth-producer arrival. Exact and zero scratch, but synchronization/reduction costs cause a large regression. Rejected.|

|E258|Interleave down and ordered reduction in two column batches. Improves4096-token cases; more batches and1024-token cases regress. Reduction DRAM counters do not explain the gain.|
|E259|Combine sign masking/insertion for nonnegative codebooks; exact static normalization supports signed books without storage growth. Modest gate gains; CPU byte-reference checks pass.|
|E260|Pack four adjacent9-bit indices in the same low32/high4 fields, eliminating most index reconstruction. Exact and unchanged payload size; combined gate/down gains.|
|E261|Combine normalized signs, packed indices and one/two column batches at TP8. Signed-book whole-MoE checks pass; gate VALU drops22% at4096 spread. Delaying both reductions until after both down chunks loses the gain.|
|E262|Adapt packing, normalized signs and column batches to TP4. M32 down becomes preferable in these tests. Exact, faster than E252; substantial dense gap remains.|
|E263|Read codebooks through global memory without LDS staging. Exact, much slower in every tested shape; rejected.|
|E264|N-atom LDS lookahead with unroll1/2 and compiler scheduling barriers. Exact, but r2 gains are small/mixed and registers increase; no broad selection.|
|E265|M64/N32 gate with the current packed decoder and pipeline. Exact and lower register allocation, but only hot1024 improves; no broad selection.|
|E266|MFMA16/32 repeated accumulation matches bit for bit in six K128/K768/K6144 tests,1572864 outputs. No performance claim; independent scalar bound remains open.|
|E267|MFMA32 gate with K128 activation buffering and current decoder/order. Exact but slower in all four shapes; no selection.|
|E268|Batch two/three K128 activation/weight tiles in the N32 gate. Exact but slower on dense4096 cases; no selection.|
|E269–E270|Persistent gate grids1/2/3/4/6/8/12 at TP8/TP4. Exact; larger grids give small/mixed changes and no broad selection.|
|E271|TP4 packed MFMA32 down, M32/M64 with/without next-N-group lookahead. Exact, small hot gain but spread regressions; no broad selection.|
|E272|Single K6144 accumulation chain changes FP32 rounding: relative L2 about1e-7–3e-7 and78 BF16 differences across524288 outputs. Diagnostic only; no acceptance or quality claim.|
|E273|Single FP32 gate accumulation chain changes rounding and has no broad speedup. Native registers246→181 still permit only two CTAs/CU; unselected.|
|E274|Exact late-activation gate with compiler minimum3-CTA hint. Forcing168 registers spills224–308 bytes/thread and regresses; rejected.|
|E275|Apply exact packing/normalized signs to C4 gate and route9 down at TP8/TP4. Smaller decoder instruction count and useful gains; TP4 spread gate regresses. No broad selector yet.|
|E276|Packed compact-route and register down at8/16 tokens, TP8/TP4. Exact; down helps, packed gate regresses spread routes.|
|E277|Single scale byte and minimum-two-CTA hint both remain at129 VGPRs/one CTA. Resource-only; no timing or correctness claim.|
|E278|Bound sign-temporary lifetimes in small gate. Exact;129→104 VGPRs, two-CTA residency restored. Atom barriers do not broadly win.|
|E279|Extend TP8 adaptive down reuse to4/16 tokens. Exact; helps hot routing, regresses M4 spread; unselected.|
|E280|Batch two/four codebook reads with shorter sign lifetimes. Four-read gate uses102 VGPRs, exact and faster across small TP8/TP4 cases; isolated.|
|E281|Extend token-owned route9 down to8/16 tokens at both TPs. Exact but slower; rejected.|
|E282|Apply E278/E280 decoder schedule to dense M64 gate. Exact, slightly slower than E261; unselected.|
|E283|Small down sign scheduling/batching. TP8 four-read and TP4 one-read arms fail correctness; ineligible pending diagnosis.|
|E284|Unchanged E280 passes30 TP8/TP4 shape/pattern cases and240 changing steps, including intermediate FP8 bytes/scales and12/13-task boundaries.|
|E285|Qualify unchanged E280 on16 actual captures and six real layer/rank weight slices:32 cases/256 changes, exact final/intermediate values.|
|E286|Assign down columns to independent waves, removing cross-wave partial sums. Exact, useful at TP8 M16 spread; TP4/hot mixed.|
|E287|Instantiate static ballot routing for M16. Exact; saves about0.3–0.4µs. Invalid-route/stable-order tests pass under E291.|
|E288|TP4 four/eight-token adaptive fused down. Exact but spread regressions outweigh hot gains; unselected.|
|E289|Batch nine independent reduction reads. Exact; combined TP8 M16 spread48.88µs beats fresh MXFP449.86µs. TP4 comparison fails unchanged-baseline tolerance.|
|E290|Widen route9 output tiles to96/192 columns. Exact, slower at TP8 M4; rejected.|
|E291|Unchanged E289 combination passes30 boundary/shape cases,240 changing steps and45 frontend tests; captured qualification remains.|
|E292|Map48 fused-down output columns to48 reducing lanes. Exact and modestly faster at TP8/TP4 M4; TP8 hot gap remains.|

| E293 | Partition small input quantization across two/four column groups. Exact; original timing drift prevents qualification. E302 supplies stable combined results. |
| E294 | Reuse M4 down weights across compact expert tasks. Exact N16/N48 variants regress on hot routes; unselected, scout drift retained. |
| E295 | Batch independent reads in dense chunk reduction. Exact, no broad improvement over E261; unselected. |
| E296 | Expand active gate weights to temporary FP8 each call. All preparation counted; exact but slower with or without prefetch; rejected. |
| E297 | Combined policy passes 30 TP8/TP4 cases, 240 changing steps, 720 exact checks and 45 frontend tests. |
| E298 | Read M4 codebooks directly from global memory. Exact but down time nearly doubles; rejected. TP4 compiled, not measured. |
| E299 | Combined policy passes 32 real-capture/TP cases and 768 exact checks using six actual weight slices. TP4 reuses TP8 inputs; M16 unqualified on captures. |
| E300 | Retain dense hot-route down weights in registers and fuse ordered route reduction. Dynamic guards/fallbacks timed; exact but much slower. |
| E301 | Store each persistent expert result once; map routes at reduction. Exact; recovers part of E300 regression but remains unselected. |
| E302 | One-second graph warmup yields stable TP8 combined comparisons. Five of six small-token cases beat fresh MXFP4; M4 hot remains +8.5%. |
| E303 | Swizzle persistent-down LDS output columns. Bank conflicts fall 96.5%, time does not improve; exact, zero scratch, unselected. |

| E304 | Stage route metadata once per persistent M32 tile. Exact; vector reads fall 62%, but total time still trails E261 badly. Unselected. |
| E305 | Retire persistent down accumulators one N16 result at a time. Exact, fewer registers, no meaningful gain; unselected. |
| E306 | Disable M16-half unrolling and request two-block compiler residency. Both variants use 112 registers and about 27% measured occupancy; no gain. |
| E307 | Exact N64 down records and N256/N512 tiles at TP8. N512 gives mixed small gains; no broad selection. First compile failure preserved. |
| E308 | TP4 quad down with N512/batch4. Exact; 0.7–3.5% better than E262 in all four dense cases, still 9–34% behind MXFP4. |
| E309 | Use N64 quads in M4 token-owned down. Exact but slower at spread/hot routes; unselected. |
| E310 | Load only a compact N16 quad record, reuse it across M4, and share a 384-workgroup grid with the N64 fallback. Exact; no gain over the selected path. |
| E311 | Qualify E308 and E262 at 39 TP4 dense shape/routing cases, 312 changes and 936 exact checks. Correct the raw-row checker using verified route permutations; no tolerance change. |

| E312 | Fuse route preparation and BF16 quantization into gate. Exact, but repeats work and regresses; unselected. |
| E313 | Quantize unique inputs once before route-fused gate. Exact; partial recovery, still slower than separate frontend. |
| E314 | Replace each gate workgroup histogram/scans with one-wave stable sorting. Exact but fusion still slower; excessive drift retained. |
| E315 | Use one-wave stable sorting in standalone frontend. Exact; useful M4 gain at two column partitions. |
| E316 | Cooperative 32-lane scale-group quantization. Exact, slower; drift makes timings provisional. |
| E317 | Pair native FP8 conversions and pack bytes directly. Exact, small additional gain; early input loads regress. |
| E318 | Qualify E315/E317 input/intermediate/final outputs at M4 TP8/TP4, plus 240 CPU-route/native-quant frontend checks. |
| E319 | Complete each route wave’s private codebook without the initial workgroup barrier; test direct VMEM-to-LDS copy. Exact, modest gains; TP8 timing provisional, TP4 stable. |
| E320 | Qualify E317 frontend with 16 actual captures and six real TP8/TP4 weight slices; exact intermediate/final results. |
| E321 | GPU-guarded two/four-token weight reuse with separate reducing waves. Two-token candidate reaches M4 TP8 hot parity and beats spread MXFP4. |
| E322 | Qualify wave-private codebook and token reuse/fallback on changing guards, reordered slots and invalid IDs; exact. Correct reporting-only invalid bincount without changing runtime/tolerances. |

Each experiment lives in `experiments/eNNN/`, with preserved source, module
identity, and results. E207 profile-r2 and E212 profile-r2/clean-b have explicit
INVALID_TIMING markers because they overlapped. E207 was rerun serially as
profile-r3. A controller-wide GPU lock now prevents concurrent experiments.

## Remaining acceptance work

1. Beat the local MXFP4 boundary with exact IQ2R arithmetic across relevant shapes.
2. Improve TP4 further and resolve its original dense timing instability.
3. Check boundary/partial tasks, skewed routes, intermediate values/scales, real
   captures, and graph reuse before changing production dispatch.
4. Integrate qualified winners and compare with fresh same-node MXFP4 bookends
   using unchanged ATOM benchmark_serving, with zero failed requests.
5. Complete model-quality and agentic qualification with real MTP acceptance.

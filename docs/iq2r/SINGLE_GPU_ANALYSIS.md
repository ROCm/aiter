# GLM-5.3 IQ2R: isolated TP-rank MoE analysis

The serving goal remains unmet. Production sweeps are paused by user request.
This work uses one MI355X GPU to represent one TP8 or TP4 rank, without loading
a model checkpoint. Dense E199+ candidate kernels remain isolated. E225 restores a previously
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

## Current measured position

TP8 uses E214 gate + E220 four-wave/M32/two-group down + E209 vector8 reduction.
TP4 uses E218 four-wave gate + E222 four-wave/M64/two-group down. Each row
averages two clean bookends from its own run.

| TP | Tokens | Routes | MXFP4 µs | IQ2R candidate µs | IQ2R latency overhead |
|---:|---:|:---|---:|---:|---:|
|8|1024|spread|211.73|231.96|+9.6%|
|8|1024|hot|125.45|170.39|+35.8%|
|8|4096|spread|455.48|596.75|+31.0%|
|8|4096|hot|394.94|500.54|+26.7%|
|4|1024|spread|296.16|365.85|+23.5%|
|4|1024|hot|190.68|268.31|+40.7%|
|4|4096|spread|612.45|1006.79|+64.4%|
|4|4096|hot|496.24|854.72|+72.2%|

No automatic route-pattern selector is qualified. Original TP4 dense samples
have unresolved intermittent outliers and are not a qualified denominator.
The first E218 trace has host-marker clock-alignment failures; its clean bookends
are valid. E218 r2 and E220 r2 use strict trace containment with a host-only
marker margin and provide valid stage attribution.

## Where the time goes

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

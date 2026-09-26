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
|E265|In progress: M64/N32-per-wave gate with packed indices/signs, buffered activations, weight lookahead and current ordering. Test lower accumulator state at four/eight waves and two grid sizes.|

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

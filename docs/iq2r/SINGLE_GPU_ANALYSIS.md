# GLM-5.3 IQ2R: isolated TP-rank MoE analysis

The serving goal remains unmet. Production sweeps are paused by user request.
This work uses one MI355X GPU to represent one TP8 or TP4 rank, without loading
a model checkpoint. Production ATOM/AITER kernels have not been changed by
these E199+ experiments. Official comparison remains unchanged ATOM
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

TP8 uses E214 gate + E217 M64/two-group down. TP4 uses E218 four-wave gate
and M64 transposed down. Each row averages two clean bookends from its own run.

| TP | Tokens | Routes | MXFP4 µs | IQ2R candidate µs |
|---:|---:|:---|---:|---:|
|8|1024|spread|209.17|225.34|
|8|1024|hot|121.89|173.10|
|8|4096|spread|454.68|629.40|
|8|4096|hot|387.45|543.99|
|4|1024|spread|291.87|403.27|
|4|1024|hot|187.60|270.27|
|4|4096|spread|607.26|1109.18|
|4|4096|hot|502.92|946.63|

TP8's M32/two-group down alternative is faster on hot routes: 169.53/525.90 us
at 1024/4096. No automatic routing-policy selector is qualified. Original TP4
dense samples have unresolved intermittent outliers and are not a qualified
denominator. The first E218 trace has host-marker clock-alignment failures;
no stage attribution is accepted from it. Its separate clean bookends are valid.

## Where the time goes

E214 profile-r2, 4096 spread, rocprof attribution:

| Stage | MXFP4 trace µs | IQ2R candidate trace µs | MXFP4 DRAM MB | IQ2R DRAM MB |
|:---|---:|---:|---:|---:|
|Gate/up|141.07|230.26|609.20|990.07|
|Down|163.20|294.28|724.96|1137.02|
|Reduction|107.82|114.66|503.33|505.70|

The original one-workgroup sorter was a separate large bottleneck; the parallel
sorter removes most of that cost on dense batches. Sharing activations and wider
tiles reduce repeated traffic. The E214 pipeline drops gate wait cycles from
195.04M to 60.27M while VALU instructions stay around 78M, directly supporting
latency overlap. There are no spills in the E214 gate or E208 down winner.
Down is now the largest remaining stage gap. Its activation reuse, MFMA output
orientation, store instructions, and decoder work are the next targets.

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
|E209|Larger vectorized route-reduction tiles, same ordered FP32 FMAs. Small gains; combination with latest gate pending.|
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

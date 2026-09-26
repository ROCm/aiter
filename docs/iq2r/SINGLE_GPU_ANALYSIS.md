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

## TP4 scheduling and real-weight qualification — E415–E419

All eight selected TP8 rows still beat matched MXFP4; five of six selected compact TP4 rows beat baseline. E416 improves its matched dense TP4 control by 0.52–1.44% and passes real-weight qualification, but dense parity and TP4 M256 hot remain open. No production or serving parity is claimed.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 119.15 | 101.60 | -14.7% |
| 8 | 128 | hot | 43.35 | 42.53 | -1.9% |
| 8 | 256 | spread | 131.03 | 111.48 | -14.9% |
| 8 | 256 | hot | 61.32 | 58.53 | -4.5% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 223.90 | 179.49 | -19.8% |
| 4 | 256 | hot | 68.93 | 86.72 | +25.8% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 594.30 | 730.14 | +22.9% |
| 4 | 4096 | hot | 488.88 | 602.89 | +23.3% |

E415 scalar wave addressing cuts static M16 register use to 104 and improves the overlap control modestly, but retains a cold-route tradeoff against E400. E416 compiler-visible dense codebook reads pass 420 synthetic checks and improve matched E386 by 0.52–1.44%; E418 adds 2,160 exact real-weight checks using the same binary. Replace only the two selected M4096 rows; M1024 stays on E363, which was not a fresh E416 arm. The new M4096 MXFP4 denominators differ from previous sessions, so cross-session percentage changes are not matched regressions or speedups. E417 corrected M32 buffering and E419 bounded unrolling are rejected: lower aggregate instructions/waits do not offset about half the measured wave occupancy. Together E415–E419 add 1,554 synthetic and 2,160 real-weight checks. All completed timing rows meet the unchanged 3% rule. Frozen failures are retained. Correct the earlier evolving inventory shorthand for E413 hot from 4.6% to 4.5% (unrounded 4.5457%); raw and published R24 evidence remain unchanged.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## TP8 M256 isolated parity and real-weight qualification — E413–E414

All eight selected TP8 comparison rows now beat their matched MXFP4 baselines. E413 closes M256 hot with 58.53 versus 61.32 us (4.5% faster), while preserving wins on spread and mixed using one fixed policy. Five of six compact TP4 rows beat baseline; TP4 M256 hot and dense gaps remain. This is isolated operator parity within the selected TP8 coverage, not complete TP8 coverage or serving parity.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 119.15 | 101.60 | -14.7% |
| 8 | 128 | hot | 43.35 | 42.53 | -1.9% |
| 8 | 256 | spread | 131.03 | 111.48 | -14.9% |
| 8 | 256 | hot | 61.32 | 58.53 | -4.5% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 223.90 | 179.49 | -19.8% |
| 4 | 256 | hot | 68.93 | 86.72 | +25.8% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E413 passes 315 synthetic checks and all fresh-bookend rows remain within the unchanged 3% drift limit. The combined activation/codebook pipeline is 14.9% faster than MXFP4 on TP8 M256 spread, 4.5% faster on hot and 10.9% faster on mixed. It gives up some previous-control cold speed (6.1% spread, 4.2% mixed) to close hot parity with one policy. E414 then passes 1,440 exact real-weight checks on three layer/rank slices using the identical binary, including input/intermediate FP8/scales and final BF16. Whole-model quality and serving remain unqualified.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Activation and compiler scheduling follow-up — E408–E412

The selected comparison is unchanged: seven of eight selected TP8 rows and five of six compact TP4 rows beat matched MXFP4. TP8 M256 hot remains 0.9% behind, TP4 M256 hot 25.8% behind, and dense TP4 M4096 about 22% behind. Complete isolated and serving parity remain unmet.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 119.15 | 101.60 | -14.7% |
| 8 | 128 | hot | 43.35 | 42.53 | -1.9% |
| 8 | 256 | spread | 135.83 | 104.72 | -22.9% |
| 8 | 256 | hot | 62.80 | 63.36 | +0.9% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 223.90 | 179.49 | -19.8% |
| 4 | 256 | hot | 68.93 | 86.72 | +25.8% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E408 repairs the tested M32 carry but does not improve performance; its tied variant still fails and is preserved. E409 M16 activation lookahead reduces matched TP4 M256 hot time from 88.17 to 83.13 us, but regresses spread/mixed. E410 explicit-drain removal is neutral, and bounded unrolling regresses. E411 compiler-visible M16 codebook loads pass exactness and gain 1.1% on hot with essentially unchanged cold timing; its M32 variant loses. E412 combined overlap reaches 83.05 us versus 68.93 us MXFP4 on hot, still 20.5% behind. No broad replacement or serving parity is claimed. These experiments add 1,890 full-run checks and 27 diagnostics; all completed full-run rows meet the unchanged 3% drift limit.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Scheduling and LDS-layout follow-up — E402, E404–E407

The selected comparison is unchanged: seven of eight selected TP8 rows and five of six compact TP4 rows beat matched MXFP4. TP8 M256 hot remains 0.9% behind, TP4 M256 hot 25.8% behind, and dense TP4 M4096 about 22% behind. Complete isolated and serving parity remain unmet.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 119.15 | 101.60 | -14.7% |
| 8 | 128 | hot | 43.35 | 42.53 | -1.9% |
| 8 | 256 | spread | 135.83 | 104.72 | -22.9% |
| 8 | 256 | hot | 62.80 | 63.36 | +0.9% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 223.90 | 179.49 | -19.8% |
| 4 | 256 | hot | 68.93 | 86.72 | +25.8% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E402 passes 378 checks and stable timing, but its four-wave grid4 gives only a hot/spread tradeoff. E404 passes 315 checks and stable timing; DWORD-plane codebooks double the bank-conflict counter and regress. E405 stops before GPU work because its alignment probe does not establish the intended single 64-bit LDS read. E406 passes 315 checks and stable timing, but both modern M32 schedules regress. E407 stops on nonfinite M32 output; native assembly exposes a codebook-result copy before completion. All failures and raw artifacts are preserved. E408 is investigating that unsafe carry and remains outside this publication. No arithmetic or tolerance is relaxed.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## M128 gate parity and TP4 barrier qualification — E397–E403

The tested TP8 M128 hot gap is closed: IQ2R is 1.9% faster than fresh MXFP4, with exact real-weight operator qualification. Seven of eight selected TP8 rows and five of six compact TP4 rows now beat matched MXFP4. TP8 M256 hot remains 0.9% behind; TP4 M256 hot remains 25.8% behind its latest baseline, and dense TP4 M4096 remains about 22% behind. Complete isolated and serving parity remain unmet.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 119.15 | 101.60 | -14.7% |
| 8 | 128 | hot | 43.35 | 42.53 | -1.9% |
| 8 | 256 | spread | 135.83 | 104.72 | -22.9% |
| 8 | 256 | hot | 62.80 | 63.36 | +0.9% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 223.90 | 179.49 | -19.8% |
| 4 | 256 | hot | 68.93 | 86.72 | +25.8% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E397 token-major M128 visits regress and are rejected. E398 balanced visits modestly improve spread/mixed, but hot remains provisional at 3.0066% maximum drift; it is unselected. E399 transfers retained gate scheduling to M128, passes 252 checks and stable timing, and closes the tested hot gap. E400 removes the obsolete TP4 gate cache barrier, passes 315 checks and stable timing, and improves its fixed control by 0.7–1.5%. E401 and E403 add 1,080 exact real-weight checks each. All setup failures are preserved, including missing support modules, the default baseline linker path and an omitted qualification helper. No arithmetic or tolerance was relaxed. E402 is tracked separately and excluded from this publication. Earlier small-token baseline/coverage issues and TP8 dense gaps remain outside this selected table.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Ordered-down sign planes and dense epilogues — E392–E396

Complete isolated and serving parity remain unmet. Six of eight TP8 rows and five of six compact TP4 rows beat matched MXFP4. New ordered-down sign planes improve the selected M256 kernels. TP8 M256 hot remains 0.9% behind; TP4 M256 hot is 25.8% behind its fresh baseline. Dense TP4 M4096 remains about 22% behind.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 135.83 | 104.72 | -22.9% |
| 8 | 256 | hot | 62.80 | 63.36 | +0.9% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 232.20 | 181.33 | -21.9% |
| 4 | 256 | hot | 70.37 | 88.52 | +25.8% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E392 concurrent filtered gates are rejected, with the profiler stall preserved. E393 passes 315 checks and stable timing; plane4 is retained. E394 passes 315 r1 checks, preserves the 3.19% hot drift failure, then passes 72 new checks and independent narrowed stable bookends. Both reduce VALU instructions without consistently reducing wait cycles. E395 passes 2,160 real-weight checks for these two modules. Its timestamp attribution failure is preserved and recovered using unique explicit dispatch/marker correlation IDs. E396 passes 420 exact checks after fixing input-layout comparison in the harness, but both direct dense epilogues regress and are rejected. No native arithmetic or numerical tolerance was relaxed. Earlier small-token and TP8 dense deficits remain outside this selected table.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Tile reuse and real-weight qualification — E388–E391

Complete isolated and serving parity remain unmet. The selected comparison is unchanged: six of eight TP8 rows and five of six compact TP4 rows beat matched MXFP4. Dense TP4 M4096 remains 22.5%/22.1% slower. E388–E390 do not improve the selected policy.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 135.27 | 106.07 | -21.6% |
| 8 | 256 | hot | 63.12 | 64.07 | +1.5% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 232.97 | 183.05 | -21.4% |
| 4 | 256 | hot | 72.54 | 89.55 | +23.4% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E388 modern MFMA32 passes 336 exact checks but regresses. E389 pairs its records, passes 420 checks and reduces load instructions, but still loses to the selected control. E390 M128 × N64 passes 336 checks and reduces decoding for full tasks, yet loses 12.2–59.3%; occupancy roughly halves and spread padding increases. All completed timing rows meet the unchanged 3% drift limit. E391 passes 1,080 real-weight changing graph/eager checks for retained E386, including exact input/intermediate scales and native dispatch. This qualifies operator correctness; tiled small captures do not establish native large-batch or whole-model quality. Earlier small-token and TP8 dense deficits remain outside this selected table.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Activation-ring rejection and decoder instructions — E385–E387

Complete isolated and serving parity remain unmet. The selected table still has six of eight TP8 rows and five of six compact TP4 rows faster than matched MXFP4. Direct codebook byte addresses improve both dense TP4 M4096 routes by 0.3–0.6% against their current control, leaving 22.5%/22.1% MXFP4 gaps. Extra activation lookahead is rejected; the compact TP8 address change is unselected.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 135.27 | 106.07 | -21.6% |
| 8 | 256 | hot | 63.12 | 64.07 | +1.5% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 232.97 | 183.05 | -21.4% |
| 4 | 256 | hot | 72.54 | 89.55 | +23.4% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.20 | 738.76 | +22.5% |
| 4 | 4096 | hot | 501.69 | 612.41 | +22.1% |

E385 implements and checks actual selective VM waits, passes 420 numerical checks and stable bookends, but both three-slot activation rings regress. E386 passes 420 checks and stable timing: direct byte addresses are retained only at M4096; M1024 is mixed and paired sign shifts are unselected. E387 passes 252 checks, loses qualified spread, and retains hot as provisional at 3.2529% drift; the 3% ceiling is unchanged. R17 real-weight qualifications still apply to the earlier kernels; E386 needs new real-weight qualification. Earlier small-token and TP8 dense deficits remain outside this selected table. No production or serving claim.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Compact gate scheduling and real-weight qualification — E380–E384

The reported selected kernels beat MXFP4 on six of eight TP8 rows and five of six compact TP4 rows. New gate scheduling reduces the reported TP8 M256 hot gap to 1.5% and TP4 M256 hot gap to 23.4%. Dense TP4 remains 3.3–23.5% behind. Latest compact and dense candidates pass 4,320 real-weight operator checks. Complete isolated and serving parity remain unmet.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 135.27 | 106.07 | -21.6% |
| 8 | 256 | hot | 63.12 | 64.07 | +1.5% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 232.97 | 183.05 | -21.4% |
| 4 | 256 | hot | 72.54 | 89.55 | +23.4% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 598.92 | 739.46 | +23.5% |
| 4 | 4096 | hot | 496.62 | 612.02 | +23.2% |

E380 carries codebook lookups across K iterations; E381 reuses a codebook within one expert; E382 transfers register loading and cross-K scheduling to TP4; E383 removes an obsolete weight-cache barrier. E383 r1 hot drift of 3.31% remains provisional; selected r2 is an independent pair at the unchanged 3% limit. E384 verifies nine real-weight slice/configuration runs with eight input/route changes and native dispatch. Its capture_tiled inputs repeat small saved captures and do not establish native large-batch or whole-model quality. All attempts are preserved. Earlier small-token and TP8 dense deficits remain outside this selected table; cross-session times are not interchangeable.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Dense reuse and task scheduling — E374–E379

The reported selected kernels beat MXFP4 on six of eight TP8 rows and five of six compact TP4 rows. TP8 M256 token-major routing improves its control by 0.8–1.7%, but hot remains 4.3% behind fresh MXFP4. TP4 M256 hot remains 29.7% behind, and the reported dense TP4 rows remain 3.3–23.5% behind. N1024 down provides a small M4096 improvement. The full isolated and serving goal remains unmet.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 136.49 | 111.22 | -18.5% |
| 8 | 256 | hot | 63.64 | 66.38 | +4.3% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 232.58 | 186.30 | -19.9% |
| 4 | 256 | hot | 71.31 | 92.46 | +29.7% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 598.92 | 739.46 | +23.5% |
| 4 | 4096 | hot | 496.62 | 612.02 | +23.2% |

E374 earlier codebook lookups are rejected, while its unchanged Pipeline4 arm independently qualifies a TP4 M256 tradeoff. E375 reduces padded MFMA work without broad improvement. E376 separate task-fill launches lose qualified spread; drifting hot/boundary rows remain provisional. E377 reduces registers but adds global-read instructions and loses every row. E378 retains plain token-major route visits after 315 exact checks; optional wave aggregation is not selected. E379 retains N1024/grid8 down only at M4096 after 420 exact checks, preserving identical MFMA counts and ordered arithmetic. All failures and prior results remain available. Cross-session timings are not interchangeable. Fleet health and the six saved real-weight shards were rechecked, but no new real-weight or model-quality qualification is claimed.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Ordered quad down and gate pipelines — E365–E373

The selected kernels beat MXFP4 on six of the eight reported TP8 rows and five of six compact TP4 rows. TP8 hot gaps are 0.4% at 128 tokens and 2.3% at 256. Newly qualified TP4 M256 spread is 19.9% faster, but hot is 29.7% slower. The four reported dense TP4 rows remain 3.3–23.5% slower. The full isolated and serving goal is not achieved.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 135.15 | 111.86 | -17.2% |
| 8 | 256 | hot | 62.64 | 64.10 | +2.3% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 256 | spread | 232.58 | 186.30 | -19.9% |
| 4 | 256 | hot | 71.31 | 92.46 | +29.7% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.04 | 744.74 | +23.5% |
| 4 | 4096 | hot | 500.70 | 610.02 | +21.8% |

E365 qualifies one-ahead register records with a prospectively narrowed repeat; E366 vector gate output qualifies independently, but E367 does not establish an additive gain. E368 extends exact ordered down to TP4 M256 and exposes a hot gap. E369 adapts quad down to TP8 M256 and E371 adjusts its grid, keeping one policy across both routing patterns. E370/E372 M32 and split-K gate variants pass exact checks but lose qualified spread; drifting hot rows remain provisional. E373 vector8/batch3 qualifies in a prospectively narrowed r2 with 168 exact checks; it improves hot at a 0.28% spread cost. Disassembly confirms E308 already overlaps all nine payload reads, correcting the initial source-level interpretation; E373 changes metadata handling, workgroup size and registers. No new real-weight or model-quality qualification is claimed.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Shared activations and workgroup ordering checkpoint — E354–E364

The selected isolated kernels beat MXFP4 on six of eight TP8 rows and all four compact TP4 rows. TP8 hot gaps are 0.4% at 128 tokens and 3.1% at 256. The combined dense TP4 kernel remains 3.3–23.5% behind. The full performance goal is not achieved.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.73 | 103.33 | -15.1% |
| 8 | 128 | hot | 44.16 | 44.32 | +0.4% |
| 8 | 256 | spread | 134.87 | 113.61 | -15.8% |
| 8 | 256 | hot | 62.76 | 64.71 | +3.1% |
| 4 | 64 | spread | 191.78 | 159.06 | -17.1% |
| 4 | 64 | hot | 42.53 | 42.02 | -1.2% |
| 4 | 128 | spread | 212.21 | 169.64 | -20.1% |
| 4 | 128 | hot | 60.88 | 59.32 | -2.6% |
| 4 | 1024 | spread | 290.48 | 300.02 | +3.3% |
| 4 | 1024 | hot | 187.38 | 196.84 | +5.0% |
| 4 | 4096 | spread | 603.04 | 744.74 | +23.5% |
| 4 | 4096 | hot | 500.70 | 610.02 | +21.8% |

E354 reduces route metadata and vector-load overhead. E355 shares activation fragments and E356 applies four-read compact TP4 decoding. E357/E360 pass expanded synthetic correctness; E361 passes 6480 real-weight operator checks, with larger capture_tiled batches explicitly constructed from small captures. E358 full M32 pipeline is unselected, with hot timing drift retained. E359 MFMA32 is exact but slower. E362 improves M256 TP8 through workgroup remapping; LDS wait counts do not fall. E363 combines qualified dense kernels; its r1 tensor-selection failure is preserved and corrected before a fresh full r2. E364 paired MFMA32 records pass 588 checks but do not produce a broad speedup. Every listed comparison uses one fixed arm for both routing patterns.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Ordered down and dense decoding checkpoint — E340–E353

The selected isolated kernels beat MXFP4 on six of eight TP8 rows and three of eight TP4 rows shown. TP8 hot gaps are down to 0.6% at 128 tokens and 4.5% at 256; TP4 M64 hot remains 2.0% behind and dense cases remain 4.5–27.4% behind. The full performance goal is not achieved.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 32 | spread | 78.78 | 66.57 | -15.5% |
| 8 | 32 | hot | 28.56 | 28.09 | -1.7% |
| 8 | 64 | spread | 103.36 | 86.56 | -16.3% |
| 8 | 64 | hot | 40.97 | 34.38 | -16.1% |
| 8 | 128 | spread | 121.47 | 102.79 | -15.4% |
| 8 | 128 | hot | 43.79 | 44.04 | +0.6% |
| 8 | 256 | spread | 135.48 | 116.80 | -13.8% |
| 8 | 256 | hot | 62.69 | 65.48 | +4.5% |
| 4 | 64 | spread | 192.76 | 160.17 | -16.9% |
| 4 | 64 | hot | 42.39 | 43.25 | +2.0% |
| 4 | 128 | spread | 213.04 | 170.07 | -20.2% |
| 4 | 128 | hot | 60.96 | 60.28 | -1.1% |
| 4 | 1024 | spread | 292.24 | 309.81 | +6.0% |
| 4 | 1024 | hot | 188.55 | 197.06 | +4.5% |
| 4 | 4096 | spread | 604.94 | 770.63 | +27.4% |
| 4 | 4096 | hot | 501.74 | 625.64 | +24.7% |

E347 restores exact ordered M256 down accumulation; E349 passes576 extended correctness checks. E350 four-read dense lookahead crosses the256-register boundary and loses despite fewer LDS waits. E351 limits lookahead to two reads and gives a small gain with occupancy retained. E352 four-read lookahead improves compact TP8 with stable occupancy; LDS waits do not fall. E346/E348 shared-expert fusion and E353 direct global activation loads are exact but slower and rejected. The E342 TP4 r1/r2 drift and harness-edit error remain visible; corrected r3 asserts its timed arms and qualifies all rows. These are explicit measured policies by TP/token range; no route-pattern-based selector is proposed.

Microseconds per complete isolated TP-rank MoE call; lower is better. Tokens
are not serving concurrency. Measurements use 32 rotating banks, five-second
warmup and fresh MXFP4 bookends; every selected row passes the unchanged 3%
maximum-drift rule. Failed attempts and provisional rows remain preserved.
The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. MXFP4 uses A4W4; IQ2R retains FP8
activations and decodes weights to FP8. Synthetic MXFP4 is requantized from
materialized IQ2R and does not establish original-checkpoint model quality.

Exact changing graph/eager checks, route-aligned gate FP8/scales, native
kernel dispatch and zero scratch are audited. The original MXFP4 numerical
bounds are unchanged. No new production integration or serving qualification
is claimed. Dense/hot gaps, broader real-capture qualification, safe packing
and fallbacks, model quality and final ATOM benchmark_serving acceptance remain.
Serving sweeps stay paused while these candidates are qualified and integrated.

## Compact gate and decoding checkpoint — E327–E339

The fixed compact-gate policy beats MXFP4 in six of eight stable M64/M128 synthetic cases. TP8 M128 hot remains 1.4% slower; TP4 M64 hot remains 0.3% slower. These small remaining gaps are still open. TP4 r2 repeats the frozen selected candidate after the original hot rows exceeded the unchanged drift limit.

| TP | Tokens | Routes | MXFP4 µs | IQ2R µs | IQ2R time difference |
|---:|---:|:---|---:|---:|---:|
| 8 | 64 | spread | 103.76 | 86.67 | -16.5% |
| 8 | 64 | hot | 41.06 | 34.49 | -16.0% |
| 8 | 128 | spread | 122.56 | 104.61 | -14.6% |
| 8 | 128 | hot | 44.29 | 44.90 | +1.4% |
| 4 | 64 | spread | 193.15 | 159.39 | -17.5% |
| 4 | 64 | hot | 42.51 | 42.64 | +0.3% |
| 4 | 128 | spread | 213.66 | 170.10 | -20.4% |
| 4 | 128 | hot | 61.10 | 60.61 | -0.8% |

Microseconds per complete isolated TP-rank MoE call; lower is better. Token
counts are not serving concurrency. Clean measurements use 32 rotating banks,
five-second warmup and fresh MXFP4 bookends. Every selected row meets the
unchanged 3% maximum drift limit. All attempts and unstable rows remain in
their experiment reports.

The comparison includes input quantization/task sorting, gate/up/SwiGLU and
intermediate quantization, down, and final route reduction. Router projection,
top-k and TP all-reduce are excluded. The MXFP4 native path uses A4W4; IQ2R
preserves FP8 activations and decoded FP8 weights. Synthetic MXFP4 is requantized
from materialized IQ2R, so these fixtures do not establish original-checkpoint
model quality.

Static sign planes remove nibble expansion/multiplication without adding weight
bytes or changing decoded values. Gate vector instructions fall by 16–19%.
Component-major gate partial sums reduce LDS instruction count; total measured
bank conflicts do not fall. Compact M16 task lists avoid unused half-task slots
and lower gate register pressure while keeping M32 tasks for down. Encoding
both task prefixes in one scan targets the extra dual-list frontend latency.

Native dispatch, zero scratch, changing graph/eager outputs, exact token-major
input FP8/scales, route-aligned gate FP8/scales and final BF16 outputs are audited.
The independent task-list checker validates every emitted compact gate tile.
The original MXFP4 atomic-output bounds remain unchanged. E339 adds a new
weight/input seed, eight changes, zeros, 16× inputs, slot permutations, and
spread/hot/skew/mixed routing; it makes no performance claim.

No new production runtime integration or serving qualification is claimed.
The overall goal remains unmet: dense and other outstanding shape gaps, real
weight/captured-input qualification of these changes, safe packing/fallbacks,
model quality, and the final ATOM 1k/1k and 8k/1k sweeps remain. Serving sweeps
stay paused until useful candidates are qualified and integrated.

## Medium-token checkpoint — E323–E326

IQ2R now beats MXFP4 in three of four freshly measured M64/M128 cases.
M128 hot remains 10.8% slower. One fixed policy is used for both routing
patterns: E235 N384/grid4 down plus E326 variant3 combined frontend.

| TP | Tokens | Routes | MXFP4 µs | Original IQ2R µs | IQ2R candidate µs | IQ2R overhead |
|---:|---:|:---|---:|---:|---:|---:|
| 8 | 64 | spread | 103.71 | 133.41 | 99.25 | -4.3% |
| 8 | 64 | hot | 41.47 | 43.87 | 38.43 | -7.3% |
| 8 | 128 | spread | 122.51 | 160.67 | 119.47 | -2.5% |
| 8 | 128 | hot | 44.62 | 60.63 | 49.44 | +10.8% |

Complete synthetic single-GPU TP-rank MoE calls, not serving concurrency.
Both clean bookends completed with 32 rotating weight banks and five-second
warmup; maximum drift is 1.03%. All 504 graph/eager checks pass, candidate
input FP8/scales, route-aligned gate FP8/scales and final BF16 match original
IQ2R exactly, and native traces show the intended kernels with zero scratch.
The MXFP4 atomic-output bounds are unchanged.

E323 identified M128 spread down as the main gap: about 78 versus 40 µs,
despite 27% less profiled DRAM traffic. The M32 tasks use only 14% of their
computed rows. E325 assigns waves to output columns, reuses activations and
skips empty M16 subtiles. It roughly halves spread MFMA work and cuts VMEM-read
instructions by 71%, bringing down near MXFP4. E326 overlaps independent route
sorting and token-major input quantization in one launch, saving another
4–5 µs. Paired FP8 conversion and 256 active quantization threads provide a
small further gain.

E325's first checker compared buffers in a changing atomic-sort order. The
corrected checker first verifies inverse gather/scatter and expert IDs, then
compares intermediates in original route order. The failure is retained;
runtime, input generation and numerical tolerances were unchanged. E323 and
E325 rows exceeding the existing 3% drift limit remain provisional.

E324 qualifies the earlier TP8 M4 two-token reuse and TP4 M4 wave-private
codebook changes on six real weight slices and 16 captured inputs: 32 cases,
256 changing steps and 768 checks, exact intermediate/final results, zero
scratch. All 122 collected artifact hashes match. TP4 reuses TP8 captured
inputs with TP4 weights; this adds no MXFP4 or serving comparison.

The overall goal remains unmet. M128 hot, M32/M256, TP4 medium and dense gaps,
broader qualification, model quality and production integration remain.
Serving sweeps stay paused. Next POCs test packed down decoding/final reduction
and packed batched codebook reads in the medium-token gate.

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

| E323 | Profile original M32/M64/M128/M256 MoE. Down dominates M64/M128 spread despite lower DRAM bytes; padding and decode instruction volume are material. Excessive drift remains visible. |
| E324 | Qualify E321 TP8 M4 reuse and E319 TP4 M4 codebook completion on real slices/captures. Exact, 768 checks; no new MXFP4/serving comparison. |
| E325 | Use existing independent-output-wave down kernels at M64/M128. N384 halves sparse MFMA work and nearly matches MXFP4 down; correct intermediate checks for atomic-sort permutations. |
| E326 | Overlap route sorting and identity input quantization in one launch; pair FP8 conversions. Combined policy beats MXFP4 at M64 spread/hot and M128 spread; M128 hot still behind. |

| E327 | Normalize codebook signs and repack adjacent 9-bit down indices; compare final-reduction variants. E289b9 is best among these arms, but M128 hot remains behind. Excessive bookend drift is retained. |
| E328 | Batch independent packed gate codebook reads at M32. Batch4 helps spread modestly; register weight-record prefetch regresses. All numerical checks pass; unstable timings remain provisional. |
| E329 | Direct gate epilogue removes the shared gate/up transfer and barrier while retaining BF16 rounding. LDS drops by 4 KiB; the hot gap remains. |
| E330 | Split M32 gate tasks into M16 subtiles, testing workgroup counts and ordering. M16 improves hot routes but badly regresses spread; no route-specific CPU selector is selected. |
| E331 | Use E326 frontend plus TP4 N256/N512 quad down tiles. Four ordered K128 sums remain exact. N256 at M64 and N512 at M128 beats spread MXFP4 but leaves hot 3–5% behind. |
| E332 | Transpose static sign bits into four planes so shift plus AND/OR replaces nibble extraction and multiplication. Same bytes/decoded values, 16–19% fewer gate VALU instructions. Gate-only gains; down-only unselected. |
| E333 | Adapt packed/direct/sign-plane gate to TP4. E333 sign-plane gate with fixed E331 down policy beats spread MXFP4 by 14–15%; hot remains about 4% behind. |
| E334 | Ablate component-major gate reduction storage and all-eight-wave M32 epilogues. Component storage reduces LDS instructions without lowering total bank conflicts. The combined version helps; plain parallel reduction regresses. M64 hot timing drift is retained. |
| E335 | Emit compact M16 gate tasks while retaining M32 down tasks. Stable fixed grid2 beats M64/M128 spread and M64 hot MXFP4; M128 hot remains +4.0%. All 504 checks and independent task-list checks pass. The second task-prefix scan costs about 1.7 µs. |
| E336 | TP4 compact M16 gate tasks with unchanged N256/N512 down shape policy. All 504 exact/task-list checks pass. Stable spread gains of 16–19%; hot gaps remain 3.3% at M64 and 1.5% at M128. |
| E337 | Extend medium TP8 candidates to M32/M256 with explicit short-K/scheduled-large entry wrappers so packed tensors reach only their matching decoder. Pending GPU qualification. |
| E338 | Combine component-major compact M16 gate reduction and one packed task-prefix scan. Six of eight stable medium-token cases beat fresh MXFP4; TP8 M128 hot +1.4%, TP4 M64 hot +0.3%. TP4 r1 hot drift is retained; frozen selected-arm r2 qualifies with max 0.61% drift. |
| E339 | Frozen E338 passes 1,152 exact arm checks across TP8/TP4 with a new weight/input seed, zeros, 16x inputs, reordered slots, and spread/hot/skew/mixed routes. No performance or actual-weight claim. |

| E337 / E344 | The M256 extension exposed changed down rounding: two independent K128 partial sums differed from the original sequential accumulator. Preserve the failure; E344 restores the ordered chain and passes 504 checks without changing tolerances. Spread cases improve; hot gaps remain. |
| E340 | Independent route-reduction wave grouping remains unselected; all attempted results are retained. |
| E341 | Aggregate periodic-route histogram/scatter atomics. Exact at TP8 but slower than E338 in all four M64/M128 cases; TP4 remains compiled and unmeasured. |
| E342 | Parallel task-record emission improves TP8 medium cases modestly. TP4 r1 drift and the r2 dispatcher-edit mistake are preserved. Correctly asserted r3 qualifies all rows: both spread cases and M128 hot beat MXFP4; M64 hot remains 2.0% slower. |
| E343 | Transpose sign bits into static planes in the dense TP4 gate and group four independent codebook reads. Exact and modestly faster; gate VALU counts fall, while total all-wait cycles do not improve against E308. |
| E345 | Apply sign planes and grouped codebook reads to TP4 dense down while fixing the E343 gate. Original-IQ2R timing drift invalidates different r1/r2 rows. Prospective r3 retains original IQ2R for exact correctness/native checks and times MXFP4/candidate only: every row qualifies, but dense gaps remain 6.1–28.0%. |
| E346 | Fuse shared-expert down into final route reduction, retaining BF16 rounding and route order. All 504 poisoned-scratch checks pass. Scattered output ownership nearly doubles physical traffic at M4096; much slower, rejected. |
| E347 | Combine compact component-major M16 gate, E342 frontend and exact M256 down. R1 timing drift is preserved. Frozen r2 qualifies all rows: M32 spread/hot and M256 spread beat MXFP4; M256 hot remains7.4% slower. |
| E348 | Transpose shared results locally and give reduction lanes contiguous eight-column reads. All 588 checks and timing rows qualify; traffic falls sharply versus E346, but every new fusion arm still loses to E345. Rejected. |
| E349 | Frozen E347 passes576 extended checks with new seed, eight changes, zeros, large inputs, slot permutations and four routing patterns. Native gate/frontend/ordered-down and zero scratch verified; no performance claim. |
| E350 | Explicit four-read codebook groups and one-atom lookahead in TP4 dense gate. All504 checks and timing rows pass, but all candidates lose. At M4096 hot, four-read lookahead lowers LDS waits yet measured occupancy roughly halves and gate time rises270→358µs. Unroll1 stays above256 registers and loses further. |
| E351 | Two-read dense gate lookahead holds static allocation at256 registers. All420 checks and all timing rows pass. Improves E345 control by0.3–1.2%, with occupancy retained and DRAM essentially unchanged; still4.5–27.4% behind MXFP4. |
| E352 | Compact TP8 codebook schedule ablation: all504 checks and timing rows pass. Four-read lookahead is fastest across all four cases, saving0.7–2.5% versus control. Spread wins; M128/M256 hot remain0.6%/4.5% slower than MXFP4. LDS waits rise slightly, so this is not a fewer-LDS-waits claim. |
| E353 | Replace TP4 down shared activation cache with direct register loads, M32/M64 and optional one-K128 activation lookahead. All588 checks and timing rows pass, but every variant loses. Global-read instruction count rises sharply while physical DRAM bytes change modestly; rejected. |

| E354 | Widen batched route reduction to vector8/batch3. Exact; reduces metadata shuffles, global reads and VALU instructions. Retained, with small TP8 full-block gains; hot parity remains open. |
| E355 | Cache activations cooperatively and reuse each K128 fragment across four output atoms. M64 without extra lookahead improves dense TP4; M64 lookahead crosses 256 registers and loses. All 588 exact checks pass. |
| E356 | Apply four-read codebook lookahead to compact TP4. All 504 checks pass and all four M64/M128 spread/hot rows beat fresh MXFP4. |
| E357 | Frozen E354 TP8 expanded correctness/native qualification:1152 exact checks across M32/M64/M128/M256, four routing patterns, new seed and eight input/route changes. |
| E358 | Reuse the dense activation/weight pipeline at M32 TP8. Exact, but no broad gain; both hot timing rows exceed 3% drift. Lower traffic and waits do not compensate for low useful workgroup occupancy. |
| E359 | MFMA32 with bounded current/next K128 records passes 672 exact checks and stable timing, but every variant loses to E355. More VALU and global-read instructions offset wider matrix instructions. |
| E360 | Frozen E356 TP4 expanded correctness/native qualification:576 exact checks at M64/M128 with spread/hot/skew/mixed routes and eight input changes. |
| E361 | Qualify frozen E354/E356 on three real layer/rank weight slices at TP8 and TP4: 6480 exact checks. capture_tiled repeats saved M4/M8 rows; remaining patterns use generated inputs. Operator correctness only. |
| E362 | Enable XCD remapping at M256 with grid multiplier2. Both patterns improve, all 336 exact checks pass; hot falls to 64.71us versus fresh MXFP4 62.76us (3.1% gap). Actual grid geometry and zero scratch verified. |
| E363 | Combine E351 two-read gate and E355 M64 down: 504 exact checks, all rows stable, 0.6–3.6% faster than E345. Dense MXFP4 gaps remain 3.3–23.5%. Incorrect r1 tensor selection is preserved; corrected fresh r2 qualifies. |
| E364 | Pair MFMA32 records and reuse activations: 588 exact checks and all stable rows. M32 improves one hot case but loses on the other three; M64 lookahead crosses 256 registers. No broad selection; initial compilation error retained. |

| E365 | One-ahead register weight records and gate grid3. R1 hot drift 3.56% is provisional; narrowed r2 passes 168 exact checks and stable timing. Register/grid2 improves hot versus its matched control; grid3 is unselected. |
| E366 | Pair partial reads and emit four FP8 output bytes in one dword store, preserving all rounding. 168 exact checks and stable timing; independently useful. Four byte stores become one dword and data-permutation instructions fall. |
| E367 | Combine register-record loading with vector gate output. 252 exact checks; qualified spread loses to both components and hot drift is 4.46%. No additive gain established; unselected. |
| E368 | Extend compact TP4 to M256 with a matching quad decoder preserving four sequential K128 accumulations. 168 exact checks and stable rows. Remapping helps both patterns; spread beats MXFP4 but hot remains 29.7% behind. |
| E369 | Adapt ordered quad down to TP8 K256 with N256/N512 outputs and matching tile-major reduction. 210 exact checks and stable rows. N512 improves both patterns against the matched triplet control; hot remains 1.7% behind that run's MXFP4. |
| E370 | Reuse gate weights over M32 rows, separately testing split-K read4 and full activation/weight pipelines. 252 exact checks. All new variants lose qualified spread; hot drift 3.19% prevents qualification. |
| E371 | Hold N512 down arithmetic/layout fixed and compare persistent grid8/4/2. 210 exact checks, both rows stable. Grid4 improves hot 1.9% for a 0.24% spread cost; its matched MXFP4 hot gap is 2.3%. |
| E372 | Use four K waves with two separate K768 accumulators each, retaining the original eight-part ordered sum. 252 exact checks, zero scratch. Both M16/M32 versions lose qualified spread; MXFP4 hot drift 3.61% is retained. |
| E373 | Adapt vector8/batch3, vector8/batch9 and vector16/batch3 reduction to N512 tile-major output. R1 hot drift 3.47% is retained. Narrowed r2 passes 168 exact checks with stable rows; vector8/b3 improves hot 1.4% for a 0.28% spread cost. Native E308 already overlaps nine payload loads; no missing-load-overlap claim. |

| E374 | Move first codebook lookups before activation completion. 210 exact checks and stable rows, but slower. The unchanged Pipeline4 arm independently qualifies a TP4 M256 tradeoff: hot improves 9.7% versus compact control while spread loses 10.6%. |
| E375 | Pass actual active M16 fragments to the dense M64 gate. 336 exact checks and stable rows. Spread MFMA counts fall 36.6%/13.1%, but only M1024 spread improves (1.2%); other rows regress. Keep dense control. |
| E376 | Choose compact/Pipeline4 work per actual M32 task count in two disjoint launches. 315 exact checks include 16/17-row boundaries. Qualified spread loses 19.9%; hot/boundary drift remains provisional. Reject extra sparse pipeline launch. |
| E377 | Halve dense N atoms per wave and load only the matching packed-record half. 336 exact checks and stable rows. Registers fall 256 to 182 with identical MFMA counts, but global-read instructions rise about 62% and all cases regress 11.7–24.1%. |
| E378 | Visit route slots across adjacent tokens; separately test guarded uniform-wave atomics. Corrected r2 passes 315 exact checks and all rows stabilize. Plain token-major visits improve their control 0.8–1.7% and are retained for qualification; hot remains 4.3% behind fresh MXFP4. Setup/preparation failures are preserved. |
| E379 | Widen dense down to N1024 by reusing each M64 activation cache across four sequential N groups. 420 exact checks and stable rows; unchanged 220 registers and identical MFMA counts. Retain grid8 at M4096 only (0.2–0.9% gains); M1024 stays N512. Dense MXFP4 gaps remain about 23%. |

| E380 | Carry four next-K codebook reads across the compact gate loop without deeper weight prefetch. 252 exact checks, max drift 2.63%; improves E378 by 2.9–4.0%. Retained at TP8 M256; hot remains behind MXFP4. |
| E381 | Keep the shared codebook while a persistent workgroup stays on the same expert. 252 exact checks and stable timing. Global-read/LDS instruction savings match the task model exactly; 0.5–2.6% gains, with small spread/mixed differences. |
| E382 | Move TP4 compact gate records from LDS to registers and add cross-K lookup scheduling. 378 exact checks, max drift 2.91%; cross-K improves E368 by 2.7–4.5%. Register-only loses mixed; codebook reuse adds no consistent benefit. TP4 M256 hot remains 23.4% behind MXFP4. |
| E383 | Remove the pre-partial-store barrier inherited from the unused LDS weight cache in the register-only gate. 252 r1 checks pass but hot drifts 3.31%. Independent r2 passes 72 new checks and stable bookends, improving E381 by 0.7–2.2%; TP8 M256 hot remains 1.5% behind. |
| E384 | Qualify E381/E383 TP8 M256, E382 TP4 M256 and E363/E379 TP4 M1024/M4096 on three real layer/rank slices each. All 4,320 exact changing graph/eager checks, input/intermediate scales, native dispatch and zero scratch pass. Small captures are tiled to larger shapes; no serving or timing claim. |

| E385 | Three-slot activation ring with actual selective VM retirement passes 420 checks and stable bookends, but both variants lose. LDS waits fall while VALU instructions rise about 12%. Compiler-drain r2/r3 and scratch-descriptor r4 attempts are preserved without GPU timing. Rejected. |
| E386 | Direct codebook byte addressing removes one VALU instruction per atom. All 420 checks and timing rows pass. Retain at TP4 M4096 only for 0.3–0.6% gains; M1024 is mixed and paired 64-bit sign shifts are unselected. Dense gaps remain about 22%. |
| E387 | Transfer direct byte addresses to selected compact TP8 M256. All 252 checks pass; VALU counts fall about 2.8%, but qualified spread regresses. Hot drift 3.2529% remains provisional at the unchanged 3% ceiling. Unselected. |

| E388 | Modern sign-plane/byte-address MFMA32 gate passes 336 exact checks and stable bookends, but loses 5.4–10.5%. Wider matrix instructions reduce counts without reducing arithmetic; waits rise. Failed padded fixture retained. |
| E389 | Pair exact K64 records into three K128 loads. All 420 checks and stable bookends pass; VMEM instruction savings recover 1.3–2.9% versus E388, but remain slower than the selected kernel. Unselected. |
| E390 | Double gate M reuse to 128 rows and retain N64 per wave. All 336 checks pass, with zero scratch and stable bookends. Register allocation rises to 446, occupancy roughly halves, padding grows on spread routes, and whole-MoE regresses 12.2–59.3%. Rejected; build failures preserved. |
| E391 | Qualify frozen E386 at TP4 M4096 on three real layer/rank slices, five route patterns and eight input changes. All 1,080 exact checks pass, including input/intermediate/final results, native dispatch and zero scratch. Operator qualification only; no timing or whole-model-quality claim. |

| E392 | Concurrent disjoint compact/pipeline gates pass 90 clean exact/bounded checks but lose to selected E382. The original rocprof graph stall and independent clean bookends are preserved; no native overlap qualification is claimed. Rejected. |
| E393 | Transfer sign planes and grouped codebook reads to TP8 ordered K256 down. All 315 checks and stable rows pass; retain plane4 for 1.0–1.7% complete-MoE gains. MFMA/read counts are unchanged; wait counters are mixed. Hot remains 0.9% behind fresh MXFP4. |
| E394 | Apply sign planes to TP4 down while preserving four ordered K128 updates. 315 r1 checks pass; hot drift is 3.19%. Narrowed unchanged-binary r2 adds 72 checks and stable bookends. Retain plane4: hot improves 1.5% versus control, but remains 25.8% behind fresh MXFP4. |
| E395 | Qualify E393/E394 plane4 on three actual layer/rank slices each, five patterns and eight input changes. All 2,160 exact input/intermediate/final graph/eager checks and native dispatch pass. Preserve and repair timestamp attribution using unique explicit correlation IDs; no GPU arithmetic or tolerance change. |
| E396 | Remove the dense gate/up shared-memory epilogue transfer using rounded registers and byte/vector4 stores. All 420 exact checks and stable rows pass after a route-versus-token comparison repair. Both candidates regress despite unchanged 256-register allocation and zero scratch/spills; rejected. |

| E397 | M128 one-thread-per-token route visits pass 252 checks but regress all three cases; half the waves are idle during visits. Rejected. |
| E398 | M128 balanced two-thread-per-token visits pass 315 checks and modestly improve spread/mixed. Hot drift is 3.0066%, above the unchanged 3% ceiling. Preserve the provisional row; balanced and aggregated variants are unselected. |
| E399 | Transfer register records, cross-K codebook carry, same-expert reuse and obsolete-barrier removal to TP8 M128. All 252 checks and stable rows pass. Hot is 42.53 us versus fresh MXFP4 43.35 us; spread/mixed also win. Retained and qualified by E401. |
| E400 | Remove the obsolete register-weight cache barrier from TP4 M256 gate, keeping selected plane4 down fixed. All 315 checks and stable rows pass; retain removal for 0.7–1.5% whole-MoE gains. Hot remains 25.8% behind fresh MXFP4. Reuse is an alternative; E403 qualifies removal. |
| E401 | The frozen E399 TP8 M128 module passes 1,080 exact real-weight input/intermediate/final graph/eager checks across three slices, five patterns and eight changes, with native dispatch and zero scratch. Correctness only. |
| E403 | The frozen E400 TP4 M256 removal module passes 1,080 exact real-weight checks and native dispatch after restoring a missing harness packing helper. Preserve the failed first attempt; module, arithmetic and tolerances are unchanged. |

| E402 | Four-wave M16 gate with grid2/4/8 passes 378 exact checks and stable timing. Grid4 trades a 0.9% hot gain for 0.7–0.8% spread/mixed regressions; E400 remains selected. |
| E404 | DWORD-plane codebooks retain exact values but double LDS bank-conflict counts. All 315 checks and stable rows pass; full-wait and cross-K variants regress 3.3–11.4% and are rejected. Native audit setup failure is preserved. |
| E405 | Padded 12-byte codebook proposal stops at a CPU alignment probe: compiler selects two 32-bit reads, so the intended single 64-bit read is not established. No GPU timing or promotion. |
| E406 | Modern M32 register/cross-K reuse passes 315 exact checks and stable timing. Bounded unrolling reduces 142 to 128 VGPRs, but every row still loses; rejected. |
| E407 | Two-word cross-K lookahead fails the initial M32 finite-output check before timings. Native code copies an outstanding DS result without a wait. Failure, source and binary are frozen for E408 diagnosis; no candidate selected. |

| E408 | Draining two-word cross-K carries fixes the tested M32 kernel: 315 full-run plus 27 diagnostic checks pass, but all timings lose. Tied-output repair fails correctness; all failures retained. |
| E409 | Two-slot M16 activation staging reuses existing LDS, passes 378 checks and improves TP4 hot by 5.7%, but loses spread/mixed. Single-slot M32 loses throughout. |
| E410 | Remove unnecessary M16 other-slot drain; compiler inserts a later wait and complete latency is unchanged. Bounded K loop reduces registers 126 to 108 but regresses. All 378 checks pass. |
| E411 | Ordinary compiler-visible LDS loads make two-word cross-K dependencies trackable. M16 gains 1.1% hot with effectively unchanged cold timing; M32 loses. All 378 checks pass. |
| E412 | Combine activation staging and compiler-visible codebook loads. No-drain form reaches 83.05 us hot against 68.93 us MXFP4, with cold regressions against E400. All 441 checks pass; retain for TP8 transfer, no TP4 broad promotion. |

| E413 | TP8 M256 activation/codebook overlap beats matched MXFP4 on spread/hot/mixed by 14.9/4.5/10.9%, with 315 exact synthetic checks and stable fresh bookends. Trades previous cold speed for one policy that closes tested hot parity. |
| E414 | The unchanged E413 binary passes 1,440 real-weight graph/eager checks across three TP8 layer/rank slices, five patterns and eight changes. No whole-model or serving claim. |

| E415 | Scalar wave identity lowers M16 VGPRs 120 to 104 and hot VALU instructions 6.1%; metadata-first adds no general benefit. 378 exact checks and stable rows. Retain scalar as an ingredient; E400 cold routes stay faster. Best hot remains 18.4% behind MXFP4. |
| E416 | Dense scalar identity plus compiler-visible codebook loads passes 420 exact checks and stable rows, improving matched E386 by 0.52–1.44%. Static 247/248 VGPRs do not change hardware residency. Retain tracked loads at M4096 after E418; dense parity remains open. |
| E417 | Corrected M32 one/two-slot activation staging passes 378 exact checks and stable rows but regresses. Hot double-slot VALU counts fall 36.1% versus M16 while measured mean active-CU wave occupancy drops 3.80 to 2.00. Reject; preserve the initial cross-wave union ownership failure and repair. |
| E418 | Unchanged E416 binary passes 2,160 exact real-weight checks at TP4 M1024/M4096 over layers 3/40/77 and five patterns. Capture ranks 0/3/7 map to actual TP4 weight ranks 0/3/3. Operator correctness only. |
| E419 | Bound M32 two-slot K-loop unrolling to one/two. 378 exact checks and stable rows, but static VGPRs remain 130/135 and measured occupancy stays near two waves per active CU. Both regress and are rejected. |

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

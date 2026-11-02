# Fully fused Q/KV preparation: device-scope load polling

## Result and runnable API

Implemented one-launch Flash C1 inputquant -> QA/KV WMMA -> split-K reduction -> first weighted RMSNorms -> latentquant -> QB WMMA -> Qheadnorm -> Q/KV RoPE. Default `region.QKVRegion(..., mode='single')`; matched two-launch control `mode='two'`. No attention, KVcache publication, MoE or serving integration.

Default polling emits `buffer_load_b32 ... scope:SCOPE_DEV` with default RT temporal policy and NV=0. No hardware atomic RMW is used for polling. Counters still use atomic arrival increments. Compiled source/ISA/LLVM/environment saved under scratch-benchmark/.

## ISA finding and correction to earlier explanation

[AMD CDNA5 ISA section4.1.1](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf), printed pp35-37: SCOPE_DEV provides device-wide coherence; TH_RT is regular temporal retention, while TH_HT prioritizes retention. A temporal hint alone does not establish coherence. Scalar WGP-scope loads are not coherent with vector stores/atomics. Both scalar RT/HT device-scope encodings were accepted by the gfx1250 assembler. Default final implementation uses vector device-scope loads; outputs are published with explicit completion drains and CTA barriers before signaling.

IMPORTANT: inspection of the older actual compiled kernel shows the `atomic_add(Counter,0)` polling source already lowers to `global_load_b32 ... scope:SCOPE_DEV`. In the previous artifact `qkv-complete-ablation/final-benchmark/single256-complete.amdgcn`, poll loads occur near lines1580/1813; real returning atomic increments remain near1534. Thus the prior assertion that each poll issued an atomicRMW and caused RMW contention was incorrect. Pure-load spelling does not remove hardware atomics that were already optimized away. The new controls compare explicit scalar/vector loads and barrier topology, not hardware atomic polling versus loads.

## Numerical issue found and fixed

Expanded stress intentionally included inputcase163 repeatedly amid other inputs. Data-atomic split-K showed a final Q error0.004022694 (0.402269%) versus the unchanged0.1% gate, in BOTH the previous two-launch control and new fused paths.

Diagnostic proof: a small Q-A norm difference(relativeRMS1.27144749e-5) changed one latent FP8 channel. Stored latent FP8/scales matched quantization of the current Q-A result. Q-B output matched an independent projection using that actual latent input to relativeRMS6.589767e-10. This reproduces atomic-order sensitivity amplified by FP8 quantization, not a Q-B publication-visibility error for these cases. Initial atomic-path timing numbers in final-benchmark/ are therefore diagnostic only, not accepted performance.

Fix:192QA/KV producers store128FP32 split partials each into a16x1536 scratch buffer (96KiB). Their existing group completion counters elect the last Q/KV CTA. That CTA adds16partials in fixed FP32 order before BF16 rounding, weighted norm and latentquant/KVRoPE. No additional kernel launch. The final default no longer uses FP32 atomic adds for projection data; completion counters remain atomic. FP64 sum was also tested but slower and unnecessary for the tested accuracy gate. GEMMs remain BF16 WMMA with FP32 accumulation, FP8 checkpoint storage.

## Accepted same-pass second-round timings

Median microseconds per full preparation invocation,128invocations/graph,40interleaved samples. Fixed-order FP32 reduction, identical QB N128/Kunroll4 arithmetic across rows.

| Working set | Two launches | Single/vector RT grid barrier | Single/scalar RT grid barrier | Single/vector RT Qready semaphore |
|---|---:|---:|---:|---:|
| 1 copies | 10.472 | 11.091 | 11.036 | 11.450 |
| 16 copies | 11.979 | 12.382 | 12.416 | 12.932 |

Onecopy=38MiB projection weights;16copies=608MiB identical weights at different addresses, not16different layers and not a forced-cold-cache claim. Gamma/scale tables shared. Single-launch Q-B is implemented and validated, but it does not outperform the matched two-launch control. Gap cannot be attributed solely to synchronization: fusion also changes register allocation and stage scheduling.

Default uses vectorRT because it is slightly better on the larger working set; the single-copy difference versus scalarRT is small. No HT default. Qready semaphore removes all-grid rendezvous but adds readiness/epoch bookkeeping and did not improve latency here. All per-invocation updates/reset are timed; allocations excluded.

Resources: matched separate QA143VGPR/8192B LDS, QB273VGPR/2048B LDS; fullkernel289VGPR/8192B LDS. ZeroSGPR/VGPRspills and zero private scratch. Grid256CTAs,4waves/CTA. First192 perform QA128CTAs+KV64CTAs, split16/N128/K256. After gridbarrier,256QB CTAs produce N128 each/fullK1024,4CTAs/head. Last head producer performs norm/RoPE.

## Publication ordering and synchronization

First-stage partial stores use device scope, then store/load completion waits and CTAbarrier before group completion increment. Last group producer reads completed partials and publishes its norm/quant/rope outputs. All256CTAs drain stores and signal a cumulative gridcounter. Polling loads use device scope and are present inside the emitted retryloop. Countertarget selects the current256arrival generation. Counter increments by256per invocation, no reset.

Polling timeout is bounded10millioniterations or1billion100MHzticks(~10s). Timeout sets healthbit2; allwaves continue uniformly, so output is INVALID and must be rejected if health!=0. This is a diagnostic bound, not proof of forward progress. Require256available CUs, no concurrent GPUkernels, independent state for overlapping contexts, sequential calls sharing state.

Qready alternative: final Qproducer drains latentquant stores and incrementsReady. Each CTA expects its own saved epoch+1, waits onReady then updates its epoch. This avoids latearrivals accidentally waiting for the nextinvocation. KV is not a global dependency, although each KVproducer still finishes its own work before doing its QBtile. It is slower here and not the default.

## Validation and retained failed experiments

Accepted fixed-order variants:268inputs x3replays x7variants, zero failures. Final outputs checked against independent FP64 GEMMs using exact quantized operands, BF16 projection rounding, FP32 norms and BF16 rotary outputs. Same0.1% relative-RMS gate for Q and KV separately. Default max over268cases: Q9.160966e-5, KV1.247117e-4. Controlled rotary tables vary by inputcase; no claim of live YaRN table integration or full-model quality.

Default fixed-order vector-grid variant passed10240checked changing-input graph invocations, including the atomic-failing input among32changing cases. Other fixed-order variants have evidence in scratch-stress/; scalarRT also passed268cases and wrap tests. All tested counters reset/epochs consistent, healthzero. Wrap tests cover signed/unsigned rollover for grid and semaphore variants;9cases x512graph replays plusinitialcalls passed. API100replay smoke passed forbothmodes.

Acyclic publication test:10240publications per polling style(sourceadd-zero/scalarRT/vectorRT),255consumers x512changing payloadvalues each, zerohealth errors. This matches onepublication per dispatch with payload reuse across capturedgraph launches. Each style checked1,336,934,400payload values.

Initial long CYCLIC handshake test (128producer/consumer roundtrips inside one dispatch) with100ms watchdog produced timeouts and subsequent invalidpayload flags in scalar and vector paths. Diagnostic waits sometimes exceeded100ms with0/few polling iterations; rootcause not established. Longer10s watchdog completed3passes each for sourceadd-zero and scalarRT. HT cyclic test was interrupted after prolonged execution; vectorlong test was not reached. No claim that this cyclic test passed allstyles or that timeout was definitively a cache/coherence bug. Failed/interrupted logs retained; default is the validated one-publication fused workload. HT not promoted.

Old atomic-chain artifacts remain unchanged for provenance. Resume record marks their extended-chain numerical limitation. No serving benchmark, imagepublication, branchpush, or GSM8K in this task.

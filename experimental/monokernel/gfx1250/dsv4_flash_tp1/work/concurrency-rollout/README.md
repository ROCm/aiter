# Flash C2/C4/C8 rollout

Requested after C1 correctness closure: 1024 input/1024 output, TP1, full model, include MoE. Native reference stays authoritative during diagnosis. Every benchmark configuration uses one server lifetime and client runs1 and2; report run2 only. C2 preliminary QKV-hybrid results are saved; C4/C8 candidate long-run faults remain unresolved. See REPORT.md for the final stopped state.

`server.sh` derives from the existing image-native settings, max sequences8, capture sizes[1,2,4,8]. `clients.sh baseline 10200 C` runs exactly two clients with8*C requests. All concurrencies share the same baseline server lifetime. Preserve the same max sequences/capture policy for the candidate comparison. `run_baselines.py` starts no servers; it waits for the existing server and invokes the clients sequentially.

## Installed AITER study

Source commit7cddd8e632e8d07eebdf9e1cab7e7a9f1e8c65fe, `/app/aiter/aiter/ops/triton/_gluon_kernels/gfx1250`:
- `gemm/basic/gemm_mxfp8.py`: TDM pipelining with NUM_BUFFERS-1 prologue slots; operand tiles read from LDS in their own stage to avoid keeping them live in VGPRs. Whole-span activation-scale staging with32KiB LDS budget and compact weight-scale staging. Cluster layouts partition M/N; compatible descriptor loads multicast operand tiles. Do not assume its large-M measured gains apply to M2/4/8; benchmark these shapes.
- `attention/pa_decode_sparse.py`: independent sequence/head/split dimensions; QK and PV use distinct WMMA layouts; split reduction separate in the baseline. Increase useful parallelism via sequence dimension and sweep split counts while preserving native Q scaling/rounding and BF16 probability conversion.
- Existing C1 preserves QA/QB L2 prefetch, early TDM loads, fixed resident256-CTA synchronization and duplicated small computation. A batched version must avoid a grid barrier across nonresident CTAs; no assumption that coherentL2 alone supplies ordering.

## Acceptance and reporting

For each group, save native/candidate outputs on identical inputs and private cache/state. Validate stage outputs, state/cache updates, then generation before timing. Model-wide C1 currently being rechecked with late weight binding, precise mHC, matched attention, and corrected indexer packed-row stride. Do not extrapolate C1 diagnostic comparisons to batched serving correctness.

Report second-run medianTTFT(ms), medianTPOT(ms), and total throughput(input+output tokens/s), alongside actual completed requests and output length. Save raw JSON and logs.

## Current experiments

`model_layer_serial.py` is REJECTED for serving: loop liveness pushes VGPRs to1024 with spills; C8 showed barrier stalls. Its initial run also had an incorrect assumption of row-contiguous residuals; native residual layout is[K/64,HC,M,64]. The later gather/scatter correction is retained for reference, but this does not resolve residency. `runtime.py` now rejects enabling ATOM_DSV4_BATCH_SERIAL.

`qa_batch.py` is a genuinely batched region: input quantization + scaled-WMMA QA/KV projection + split reduction +QA RMSNorm/FP8 +KV norm/RoPE. All M<=8 rows reuse each TDM weight tile, early weight TDM overlaps quantization, and four per-thread cache-line prefetches cover32MiB of QB weights. Fixed256-CTA producer grid. RMS reduction mirrors AITER's8 contiguous values/thread and ascending pair reduction; explicit FMAs preserve its local accumulation.

`qa_integration.py` hooks this region across43layers with `ATOM_DSV4_BATCH_QA=shadow|substitute`, experimental full-layer flags OFF. Prefill remains native. Native QB and native final Q/KV norm/RoPE remain; the extra fused KV norm/RoPE output is currently diagnostic, so this integration does not yet remove that native operation. QB prefetch targets the prepared native QB weight storage. This is partial QA-region integration, NOT the completed batched full-layer monokernel.

Microchecks32 varied inputs per C2/4/8: QA projection and quant outputs exact; scales exact; health0. `qa-batch-native-results.json` second-round microseconds(nativeQA+norm only / candidateQA+norm+KVrope): C2 16.7051/8.7326; C4 16.7158/8.7216; C8 16.9857/8.9141. This comparator intentionally gives native fewer operations; it is not an end-to-end result.

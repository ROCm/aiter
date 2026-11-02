# AITER-derived attention integrated into the QKV monokernel

Implemented an adaptation of the requested [AITER gfx1250 sparse decode kernel](https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_gluon_kernels/gfx1250/attention/pa_decode_sparse.py). Exact downloaded source and MIT license are preserved in upstream/, with SHA256/provenance in UPSTREAM.json. No AITER runtime dependency is needed by the adapted kernel.

## What is integrated

- AITER's grouped-head BF16 WMMA QK and PV, online softmax, sparse slot handling, two-deep TDM slot/KV gather pipeline, and split-KV softmax merge.
- Q tiles also use TDM. Scalar/small-vector accesses remain direct. Sparse KV indices can be -1; valid indices address a unified BF16 cache whose rows supply both K and V.
- KV cache update is performed byCTA0 and published with a device-scope grid barrier before attention reads. Non-speculative L2 prefetch covers requested KV rows, using safe addresses for masked indices. Existing all-weight TDM and non-speculative QB L2 prefetch remain enabled in the QKV stage.
- Split partials m/l/acc are published with **device-scope buffer stores**, drained before an arrival counter. The last CTA for a head group invokes AITER's TDM partial reducer inside the same kernel and resets the group counter. The original upstream split/reduce kernel boundary is therefore removed. No polling is needed for this last-arriver merge; the initial cache publication still uses a grid barrier.
- The sink is included once: inside unsplit attention or in the final split merge. All-invalid tiles are guarded against -inf-minus--inf NaNs. The API requires each configured split to own a tile and validates index bounds at construction.
- Query scaling happens after QK accumulation in FP32 rather than rounding a scaled query to BF16. PV uses high and low BF16 probability terms in two WMMA operations to preserve the existing0.001 relative-RMS gate. This is an adaptation, not a claim of bitwise identity to unmodified upstream.

## Launch integration

`AttentionRegion`: one launch combining cache update, attention and (when split) merge.

`QKVAttentionRegion(fused=True)`: one256-CTA launch combining our entire inputquant→QA/KV→norm→latentquant→QB→norm→RoPE region, cache update and attention. A grid publication barrier after QKV precedes the cache-update/publication barrier. There are no child kernel launches. At128tokens, default BH16/split1 gives4 active attention CTAs, each handling16 heads. Split4 gives16 active attention CTAs. All256 CTAs participate in the QKV/cache publication barriers.

`QKVAttentionRegion(fused=False)`: the same QKV and attention operations in two region launches. Profiler traces in trace-verification.json confirm exactly1 versus2 kernels.

Default BH16/BK32. Auto split selects1 below1024 entries and4 at/above1024, based on tested128/1024 endpoints; other lengths are not claimed to have an optimal split. Manual1/2/4 overrides are supported when all segments are active. Wider BH32/64 configurations were swept but did not beat BH16 in this C1 test. Two-region integration is currently faster than full fusion; both paths are available, with fused=True retained as the explicit monokernel API default.

## Performance: second round only

| QKV + cache update + attention | Reused weights (us) | 16 rotating QKV copies (us) |
|---|---:|---:|
| Old vector attention, two launches | 22.726949 | 22.898777 |
| AITER-derived attention, two launches | 19.551422 | 20.402070 |
| AITER-derived attention, one launch | 20.025410 | 21.645184 |

Isolated cache-update+attention at128tokens: old vector implementation **10.904903us**, adapted AITER split1 **7.162426us**. The original historical14.56us number used a different profiler harness; the matched remeasurement above is the valid control. The old control retains its vector attention math under the current cache-update/barrier wrapper.

Split sweep (separate interleaved experiment, so do not subtract its timings from the chain experiment):

| Context | Heads/CTA | KV splits | Time (us) |
|---|---:|---:|---:|
| 128 | 16 | 1 | 7.082465 |
| 128 | 16 | 2 | 10.302746 |
| 128 | 16 | 4 | 10.315113 |
| 1024 | 16 | 1 | 34.981368 |
| 1024 | 16 | 2 | 24.153456 |
| 1024 | 16 | 4 | 17.672364 |

At128tokens, merging partials costs more than splitting saves. At1024tokens, split4 substantially improves over split1. The default split heuristic is based on these endpoints, not a full production sequence-length tuning table.

All performance figures use round2 only,128 calls per captured graph,5 warmup replays and40 alternating-order samples per round. Rotating mode cycles through16 identical QKV weight copies at distinct addresses,608MiB total, while the attention cache remains a controlled128-token window. It is a larger-weight-working-set proxy, not actual model-layer serving. Original round1 and all samples are retained. No TTFT/TPOT/token-throughput claim.

## Correctness and assembly

1152 attention checks:127/128/256/1024 entries,32 cases per length,3 split choices,3 replays. Cases include random order, -1 masks, entirely invalid slots, varied query/cache magnitudes and sinks. Reference is independent FP64 attention with BF16 output. Max relativeRMS=0.00017310922; threshold remains0.001.

64 full-chain cases check Q/KV against the previously validated QKV control, and attention against FP64. Six stress configurations each check10240 changing-input graph invocations: attention split1/2/4; two-region split1; fused split1 andsplit4. All pass with clean health/counters. Max stress relativeRMS=0.000149633808. These checks do not replace serving/GSM8K evaluation.

Assembly and LLVM IR are saved for measured paths. ISA verification checks TDM, WMMA, prefetch instructions and explicit device scope on buffer stores. Selected paths have zero spills. Exact register/LDS counts are in resources.json. Publication scope was corrected during integration; earlier exploratory results remain clearly labeled initial-* and are not the final comparison.

## Scope and limitations

SingleGPU FlashC1,64 heads,512 dimensions, unified uncompressed BF16 K/V cache. Sparse slot indexing is supported, but this does not implement the full Flash compressed-cache/index-selection machinery or quantized KV formats. Current-token destination is fixed per object for capture; callers rebuild/select an object for a different slot. No TP/EP, no MoE, no inverse-RoPE/output-projection integration, no ATOM serving integration, no git/image push. Fixture uses checkpoint layer0 projection weights and real sink, with controlled cache histories; stress additionally uses synthetic queries/cache. No full checkpoint required to replay the saved artifact.

Read README.md for reproduction. GPU idle after final checks. All sources, fixtures, provenance/license, traces, assembly, raw timings, validation and environment are included in the archive.

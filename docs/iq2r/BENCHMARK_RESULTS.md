## Current MXFP4 vs IQ2R benchmark results

Updated 2026-09-25 21:11 UTC. Unchanged ATOM `benchmark_serving`, one MI355X node, TP8. **Output tokens/s; higher is better.** The measured E180 r8 runtime predates the E190 router guard included in this source snapshot. These measurements are historical evidence, not a qualification of the complete snapshot.

**TP8 — 1k input / 1k output**

| Concurrency | MXFP4 before | IQ2R | MXFP4 after | IQ2R vs MXFP4 mean |
|---:|---:|---:|---:|---:|
| 1 | 80.89 | 84.00 | 79.73 | +4.59% |
| 2 | 153.13 | 159.33 | 152.13 | +4.39% |
| 4 | 327.93 | 323.41 | 326.69 | -1.19% |
| 8 | 592.67 | 592.86 | 595.56 | -0.21% |
| 16 | unavailable | 1052.71 | unavailable | — |
| 32 | unavailable | 1636.81 | unavailable | — |
| 64 | unavailable | 2549.67 | unavailable | — |
| 128 | unavailable | 3879.19 | unavailable | — |
| 256 | unavailable | 5662.29 | unavailable | — |

The MXFP4 mean is the average of the before/after measurements. At C1/C2 IQ2R is above both samples; at C4 it is below both; C8 falls between them. **The full comparison is incomplete:** both MXFP4 arms failed before producing a C16 result, so no high-concurrency baseline is available.

**TP8 — 8k input / 1k output**

| Concurrency | MXFP4 | IQ2R output tokens/s | IQ2R vs MXFP4 |
|---:|---:|---:|---:|
| 1 | unavailable | 77.14 | — |
| 2 | unavailable | 150.12 | — |
| 4 | unavailable | 284.58 | — |
| 8 | unavailable | 478.93 | — |
| 16 | unavailable | 741.55 | — |
| 32 | unavailable | 1022.59 | — |
| 64 | unavailable | 1296.99 | — |
| 128 | unavailable | 1593.88 | — |
| 256 | unavailable | 1840.95 | — |

**TP4 — both workloads:** no completed official-script measurements yet. The fresh TP4 comparison is queued after the correctness diagnostics.

All 18 completed IQ2R points have zero failed requests, and the TP8 IQ2R native-dispatch audit passed. The eight completed MXFP4 samples also have zero failed requests, but both MXFP4 arms later failed; the complete run is not failure-free. Answer-quality qualification remains separate.

Protocol: random length ratio 0.8, 10×concurrency measured requests, 2×concurrency warmups, seed 0, FP8 KV, no EP, no MTP. The final agentic workload will use real MTP acceptance.

**Full methodology, latency, failure evidence, and historical results:** [ATOM status comment](https://github.com/ROCm/ATOM/pull/2335#issuecomment-5837890842) · [AITER status comment](https://github.com/ROCm/aiter/pull/5728#issuecomment-5837891193).

## Correctness status at this snapshot

E183 normal-EOS diagnostics completed 352 requests at each TP width without
request failures. TP4 passed 340/352 answer checks; TP8 passed 339/352. Thus
transport completion and native equivalence have passed these checks, while
the strict answer-quality qualification remains incomplete. These custom
checks are not a standard accuracy benchmark or a matched MXFP4 quality study.
E191 is validating the router guard under normal full-model serving before
fresh TP4/TP8 benchmark runs can proceed.

The build and tests in [README.md](README.md) validate this source snapshot;
they do not replace the required full-model comparison. The original client
hash, raw-result hashes, request validation, and latency metrics are retained in
[E180_R8_METRICS.json](E180_R8_METRICS.json).

## Current MXFP4 vs IQ2R benchmark results

Snapshot: 2026-09-25T22:54:14.977017+00:00.

E197 uses the frozen unguarded E180 r8 MXFP4 baseline, unchanged official ATOM benchmark client and workload. This is a later baseline collection on the same node; fresh bookends around the IQ2R run are still required. Only fully validated, zero-failure points have throughput values.

## 1024/1024

| C | Existing MXFP4 A / B tok/s | New MXFP4 tok/s | IQ2R r8 tok/s | IQ2R vs new baseline | New collection status |
|---:|---:|---:|---:|---:|---|
| 1 | 80.89 / 79.73 | — | 84.00 | — | already measured |
| 2 | 153.13 / 152.13 | — | 159.33 | — | already measured |
| 4 | 327.93 / 326.69 | — | 323.41 | — | already measured |
| 8 | 592.67 / 595.56 | — | 592.86 | — | already measured |
| 16 | — / — | — | 1052.71 | — | failed |
| 32 | — / — | 1673.35 | 1636.81 | -2.18% | 320 / 0 requests; native pass |
| 64 | — / — | 2622.04 | 2549.67 | -2.76% | 640 / 0 requests; native pass |
| 128 | — / — | 4070.54 | 3879.19 | -4.70% | 1280 / 0 requests; native pass |
| 256 | — / — | 5786.88 | 5662.29 | -2.15% | 2560 / 0 requests; native pass |

## 8192/1024

| C | Existing MXFP4 A / B tok/s | New MXFP4 tok/s | IQ2R r8 tok/s | IQ2R vs new baseline | New collection status |
|---:|---:|---:|---:|---:|---|
| 1 | — / — | 74.48 | 77.14 | +3.57% | 10 / 0 requests; native pass |
| 2 | — / — | 146.39 | 150.12 | +2.55% | 20 / 0 requests; native pass |
| 4 | — / — | 296.91 | 284.58 | -4.15% | 40 / 0 requests; native pass |
| 8 | — / — | 510.50 | 478.93 | -6.19% | 80 / 0 requests; native pass |
| 16 | — / — | 841.10 | 741.55 | -11.84% | 160 / 0 requests; native pass |
| 32 | — / — | 1164.83 | 1022.59 | -12.21% | 320 / 0 requests; native pass |
| 64 | — / — | 1545.03 | 1296.99 | -16.05% | 640 / 0 requests; native pass |
| 128 | — / — | — | 1593.88 | — | running |
| 256 | — / — | — | 1840.95 | — | pending |

New-baseline ratios use identical input/output length arrays (SHA-256 verified) but a later run. They are provisional, not final bookend-qualified gains.

C32/C64/C128/C256 at 1k/1k ran consecutively on one loaded server (PID78978),
with no reloads between them. C16 failed on the original unguarded native MXFP4
sort/quant path after 115/160 reported completions, even as the first point on
a fresh server. That failed attempt has no accepted throughput. Each subsequent
pass requires the full request count, zero per-request errors, positive output
lengths, and native MXFP4 dispatch. Original source/module identities and the
unchanged official-client hash are verified. The experimental E190 router
guard is absent from these MXFP4 measurements.

Protocol: unchanged `atom.benchmarks.benchmark_serving`, random ratio 0.8,
seed 0, 10*C measured requests and 2*C warmups; TP8, FP8 KV, no EP/MTP/prefix
caching, maximum 256 sequences, 4096 batched tokens, 12288 model length,
all nine graph sizes, prefill chunk128, GPU memory fraction0.75.

TP4 official-script measurements remain pending and deferred while the missing
TP8 MXFP4 results are collected. The complete performance/quality goal remains
unmet. Final agentic qualification will use real MTP acceptance.

[ATOM status comment](https://github.com/ROCm/ATOM/pull/2335#issuecomment-5837890842) ·
[AITER status comment](https://github.com/ROCm/aiter/pull/5728#issuecomment-5837891193).

## Correctness status at this snapshot

E183 normal-EOS diagnostics completed 352 requests at each TP width without
request failures. TP4 passed 340/352 answer checks; TP8 passed 339/352. Thus
transport completion and native equivalence have passed these checks, while
the strict answer-quality qualification remains incomplete. These custom
checks are not a standard accuracy benchmark or a matched MXFP4 quality study.
E191 completed guarded-router normal serving diagnostics: TP4 335/352
answer checks and TP8 337/352, with zero failed requests. These remain
custom diagnostics, not final answer-quality qualification against MXFP4.

The build and tests in [README.md](README.md) validate this source snapshot;
they do not replace the required full-model comparison. The original client
hash, raw-result hashes, request validation, and latency metrics are retained in
[E180_R8_METRICS.json](E180_R8_METRICS.json).

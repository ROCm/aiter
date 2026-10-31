# Round MERGE: integrate measured TDM wins, disable default BM64

Merged R16 `f64eedaf` into TDM-4 `bef193aa`, common champion `3e3555b3`.
No push, GPU dispatch or frozen-directory changes.

## Routing

| Shared QSA_BF16_K_LOAD | bf16 decode/draft | bf16 prefill | fp8 decode | fp8 prefill |
|---|---|---|---|---|
| auto (default) | r16 c32 TDM | r32 c64 TDM | r16 c64 raw | r32 c64 padded TDM |
| raw | r16 c64 raw | r32 c64 raw | r16 c64 raw | r32 c64 raw |
| tdm | r16 c32 TDM | r32 c64 TDM | r16 c64 raw | r32 c64 padded TDM |

`QSA_BF16_R64=1` explicitly permits bf16 r64 c64 TDM when rows >= 64*requests;
without it even forced TDM keeps r32. Raw/cop disables r64 even with opt-in.
The private explicit-r64 builder remains available for compile-only evidence.
`QSA_BF16_R16_TDM_COLS=64` restores legacy r16 TDM geometry.
`QSA_FP8_K_LOAD`, if explicitly set, overrides the shared switch for fp8;
otherwise it inherits the shared switch so `QSA_BF16_K_LOAD=raw` is all-raw.

Unaligned cache bases still route to actual raw r16 c64, not a TDM-labelled fallback.
Partial r16 blocks and mixed/partial prefill retain exact raw paths, i64 plane
extents, physical-page-zero selection, predicated/clamped stores and exactly-once
logit/metadata writes. The public wrapper API is unchanged.

## Source-matched no-device evidence

[Full merged evidence](<aiter/ops/flydsl/kernels/mqa_logits/_g2/merge_isa_evidence.json>)
records every one of 45 real final-ISA specializations over seven mode directories:
auto/raw/tdm/tdm64/r64/r64raw/r64tdm. Stock FlyDSL 0.3.3.dev886, target gfx1250,
image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, network none, no devices.
Remote stage: `/data/blyu/qsa/work/opt-merge-cpu-trace-20261031`,
source `source-merge.aveGXL`, results `results-merge.FZDBSv`.

Kernel SHA256: `9428b7efdc1a9cd95588af0da0e2053c74720cde698994cc03819f6ccc43398e`.
All eleven reused TDM-4 kernel names are byte-identical to its ISA evidence.
New auto/forced r16 c32 are byte-identical to measured R16:
`d04e3b64d5d7bcd06f8ef1dceb39d9aea91cd8d96721ac8dcf1b16e5f01ed666`.

Static instruction counts (including fallbacks, not dynamic work):

| Kernel | LDS B | VGPR | physical SGPR | pitch B | tensor | waits | wait(0) | ds_load_b128 | barrier pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 r16 c32 TDM | 34816 | 97 | 105 | 272 | 16 | 16 | 4 | 64 | 8 |
| bf16 r32 c64 TDM | 17408 | 183 | 79 | 272 | 2 | 2 | 1 | 32 | 1 |
| fp8 r32 c64 TDM | 9216 | 193 | 83 | 144 | 2 | 2 | 1 | 32 | 1 |
| bf16 r64 c64 TDM (opt-in) | 17408 | 235 | 103 | 272 | 2 | 2 | 1 | 64 | 1 |
| bf16 r16 c64 TDM (legacy opt-in) | 69632 | 193 | 105 | 272 | 32 | 32 | 4 | 128 | 8 |

Every specialization has zero `v_readfirstlane_b32`, `v_max_i64` and
`ds_load_b128_addtid` markers. All WMMA kernels have zero VGPR spills/private
scratch. r16 c32 retains 11 SGPR-to-VGPR spills; legacy c64 retains 29.
Final `.amdhsa_next_free_vgpr/sgpr` directives give physical register counts:
r16 c32 YAML logical counts are 80/107, not the final 97/105 allocation. The
checker reports both instead of wrongly treating YAML80 as a failed97 gate.
The unchanged scalar bf16 diagnostic fallback retains 22 VGPR spills/92B scratch.
Tensor descriptors are SGPR-only and outside clauses, workgroup mask0, bf16
padding encoding5/3, fp8 encoding4/3; completion waits/barriers are retained.

All CPU gates passed: contract111 checks, address26 cases, Q-LDS31 checks,
trace16/trace32 each16 cases, K-TDM37 tests, routing12, fp8 K-TDM18, BM6412;
79 unittest tests total. Strict TDM c32/r32/r64 and legacyc64 gates, bf16/fp8 ISA
routing equalities and r64 opt-in exclusions passed. Initial resource gate
failure was investigated and corrected without any kernel change; final
source-matched complete re-verification returned `ALL_GATES_RC=0`.

Reproduce with [compile driver](<aiter/ops/flydsl/kernels/mqa_logits/_g2/host_compile_merge_cpu.sh>)
and [final verifier](<aiter/ops/flydsl/kernels/mqa_logits/_g2/host_verify_merge_cpu.sh>).
[Summarizer](<aiter/ops/flydsl/kernels/mqa_logits/_g2/summarize_merge_cpu.py>) checks
all CPU logs, source hash, all45 real ISA dumps, byte-identities and waterfall markers.

## Measured expectations, not a new timing claim

From parent measured tables: decode bf16 c65536 92.2->79.7us (-13.6%), draft
bf16 c65536 86.6->82.7us (-4.6%), fp8 prefill bucket 942.7->910.5us (-3.4%).
Turning BM64 off avoids its +7.3% bm16 and +7.0% bmge64 bf16 prefill regression.
Decode bf16 c32768 previously +2.7% (noise) remains a held-out risk.

No merged-tree GPU correctness or latency validation was performed. On an
explicitly authorized healthy gfx1250 host acquire cooperative gpu-lock, run
first-touch and the complete22-case correctness suite plus mixed/partial/tail,
invalid-page and unaligned subview cases. Confirm dispatch names, fp8 all256-code
and scale behavior, then randomized unprofiled A/A and A/B against champion/raw
controls for every decode/draft/prefill bucket. Keep r64 opt-in separate.

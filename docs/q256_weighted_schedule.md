# Qwen Q256 weighted M128 scheduling

The gfx950 Q256 integration includes the `weighted-m128` producer and matching
GPU job-map builder from `qwen-aiter-package-20261016` (PyISA kernels commit
`32a1d2f80250118c8b4f25e57eb1becbff24eb75`).

Weighted execution requires an explicit `_route_weighted` Q256 configuration
and exactly 8192 actual tokens (not merely the 8192-token config bucket), with
model width 8192, 513 experts, and topk 11. Adjacent actual token counts use the
compact producer for `[tokens, topk, model]` contributions.

Loading the weighted CO alone does not establish shared-M128 scheduling. The
GPU planner qualifies actual 8192-token geometry only with exactly 32 trailing
256-row physical tiles for expert 512, a shared boundary >= 1 whose preceding
expert is not 512, and `shared_boundary + 64 <= 1024`. Qualified captures use a
nonidentity map; other geometry retains the identity fallback.

Each weighted call builds a fresh 1024-entry int32 map on the GPU using eight programs.
The producer receives its pointer at kernarg byte offset 128; the ABI remains
448 bytes. Shared-expert work uses M128 (64 shared jobs), matching the
scheduler's fixed `ABI_SHARED_ROWS=128` geometry. Contributions already include routing
weights and use the existing compact reducer.

## Deployment

Deploy these together:

- `hsa/gfx950/fmoe/silu/fused_moe_mxfp4_prefill_1tg_4w_256mx1_128nx1_ps_fp32gate_route_weighted.co`
- `aiter/ops/triton/moe/q256_schedule.py` (authoritative category wrapper)
- `aiter/ops/triton/_triton_kernels/moe/q256_schedule.py`
- The updated Q256 Python/C++ launcher and `q256_producer_args.hpp`.

The M128 CO SHA256 is
`b2f06e524f590571b8cdf0a395432fa0d915f20d04c91ef5590b13f7bfee881e`;
the original package scheduler SHA256 was
`bcc7f029274584b75177d07cbf118c52af7a15dee78569f1b8fda2c5955368db`.
M64 and M128 packages use the same weighted filename and symbol, so always
replace the producer and its scheduler as a pair.

Resolve the active loader's `AITER_ASM_DIR` (including environment overrides)
when updating an installed package. Clear its Q256 AITER JIT build and Triton
scheduler cache entries, then restart workers.

## Validation

Package provenance covers captured Qwen routes 029/046/049 on gfx950 with
synthetic activations/weights and a sampled independent reference. M128 was
fastest on retained routes 046/049; M64 was faster on 029.

For this integration, the package hashes, eight established CPU contract tests,
Python syntax, and C++ ABI assertions passed. Fresh full-API validation on
SMCI355 gfx950 GPU PCI `0000:05:00.0` passed with the updated launcher compiled
from this checkout and fresh JIT/Triton caches:

- Exact compact/sorted contribution equality and finite output.
- Independent 64-token reference NRMS: 0.00273150 (threshold 0.05).
- Dynamic-route graph/eager, prequantized-input, and caller-output parity.
- Profile confirmed `make_job_map` and the weighted producer symbol.
- Full-API graph median: 1156.51 us (seven samples of 50 replays).

Validation artifacts: `/tmp/opencode/q256-m128-validation-smci355/report.md`.

## 2026-10-23 production-path recovery (latest evidence)

The original M64/M128 package provenance, hashes, and validation above remain
historical evidence. The latest comparison uses actual main
`18946ab5870e1468ca8b0f0ddfa9124f4e861550` versus feature HEAD
`36d49f102ac3fd548b408c37eb4c0ea731581bfd` plus the verified working delta,
FlyDSL **0.4.0**, and CK `96afae1d54fc79c26857de7e1683b0dde1ff3c6f`.
Backend source restoration/recovery on this main/FlyDSL 0.4 baseline is complete:
recovered NT streaming reducer loads apply **only** when
`(tokens, model, topk) == (8192, 8192, 11)`. All other tested shapes retain the
default load policy, including D4096/8192 tokens and adjacent D8192 boundaries.
This reducer predicate is separate from shared-M128 planner qualification.

Current real captured routing IDs/weights 029/046/049 were replayed with
synthetic activations and expert weights (seed 1234, expert weights ×0.02).
The public `aiter.fused_moe.fused_moe` API used each tree's real model configs,
without runtime overrides. Seven alternating main/candidate rounds, 50 graph
replays per sample, and a fixed caller output buffer ran under one continuous
GPU lock on MI355X/gfx950 (physical GPU 1, logical GPU 0), with idle checks.

| Capture | Actual main median, µs | Recovered production median, µs | Latency reduction |
| --- | ---: | ---: | ---: |
| 029 | 1254.08 | 1155.80 | 7.84% |
| 046 | 1251.30 | 1171.79 | 6.35% |
| 049 | 1236.19 | 1169.40 | 5.40% |

All full outputs were bitwise equal across main/candidate and every round;
independent 64-token references passed with NRMS 0 and mismatch ratio 0. Fresh
correctness gates passed six shared shapes, a random control, and five token
boundaries (4096/4097/8191/8192/8193). Actual profiler receipts confirmed the
producer/planner and `_nt` reducer at the exact target, default policy elsewhere,
and nonidentity maps on all three qualified captures. The random control did
not qualify and used the identity fallback.

The previous generic-random comparison did not force a universal shared expert.
It remains a valid fallback measurement, but is not meaningful evidence for
the target shared-M128 path. Publishing this 8K recovery now requires the
FlyDSL cache-policy prerequisite as well as the paired scheduler/producer;
the recovered target does not use the earlier default-cache final tree.

Final artifacts:
`/tmp/opencode/aiter-5988-8k-recovery/production-validation/report-final.json`
and `/tmp/opencode/aiter-5988-8k-recovery/production-validation/SUMMARY.md`.

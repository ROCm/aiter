# Round TDM-AUTO — route bf16 TDM by geometry: default TDM for r32 prefill, raw for r16 decode/draft

You are the ROUND-TDMAUTO optimizer. **Time-box ~15 minutes.** This is a small routing change on top of the
measured round-TDM2 kernel. Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-tdmauto`
(branch `qsa-index-scorer/opt-tdmauto`, base = `6802aec5` = TDM-2: padded 272 B LDS TDM for bf16 r32 and r16,
default raw). Never push; never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`,
`records/` except `records/g2/`) or `records/sealed/`.

## Why

Round TDM-2 measured on a07-3 with `QSA_BF16_K_LOAD=tdm`: bf16 r32 prefill improved
(prefill bf16 bm16 956.4 → **760.2 µs**, −20.5 %; decode_bf16_c65536 97.1 → 81.0, −16.6 %), but the r16 TDM
path (used by decode c32768 and draft) regressed (decode_bf16_c32768 +8 %, draft_c65536 +10 %,
draft_c32768 +16 %) because of its 69,632 B LDS footprint. We want the r32 win as the **default** without the
r16 regressions.

## Task

Change the routing default: TDM is used for bf16 **r32** blocks by default (`rows_per_block == 32` shapes,
i.e. prefill) and the **r16** path stays raw by default. Provide `QSA_BF16_K_LOAD=raw|tdm|auto` with
`auto` the new default (`raw` forces the old per-lane path everywhere, `tdm` forces both r32 and r16 TDM as
today's opt-in). Keep the r16 TDM code itself available under the forced `tdm` mode but not selected by
default. Everything else stays byte-identical: the r32 TDM padded path, the raw path, fp8, descriptors,
safety contract, host guards, wrapper API. Keep the dispatch compile-time selection so the raw kernel
specialisations remain in the binary for A/B.

## Verification (no GPU) + deliverable

- CPU gates green: existing contract/trace16/address/Q-LDS/padded-layout 28 tests/strict ISA; add a routing
  test that `auto` selects the TDM r32 specialisation for prefill shapes and the raw r16 specialisation for
  decode/draft shapes, and that `raw`/`tdm` force the old behaviour.
- No-device real-ISA trace (CPU-only container, image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no
  `--device`; SSH user `blyu`, host `heliosr-1b114-a07-3.mnb.dcgpu`, key
  `C:\Users\blyu\.ssh\id_ed25519`, staging dir `/data/blyu/qsa/work/opt-tdmauto-cpu-trace-20261031`):
  both `auto` and `raw` builds `[COMPILED]`; in `auto`, the r16 decode kernel must be the raw specialisation
  and the r32 prefill kernel the padded-TDM one (report dispatch names and kernel names).
- Commit on `qsa-index-scorer/opt-tdmauto`; report JSON {status, base_commit, commits[], files_sha256,
  what_changed, routing_evidence{auto/raw/tdm dispatch table, kernel names, ISA markers}, expected_gain{case:%},
  risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done.

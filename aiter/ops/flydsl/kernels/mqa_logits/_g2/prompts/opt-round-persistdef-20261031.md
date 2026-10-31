# Round PERSISTDEF — make the persistent prefill kernel the default (QSA_BF16_PERSIST=1, P=8)

You are the ROUND-PERSISTDEF optimizer. **Time-box ~15 minutes. Small routing/default change.** Base =
`e0d85129` (opt-in persistent bf16 r32 prefill kernel; branch `qsa-index-scorer/opt-persist`). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-persistdef`
(branch `qsa-index-scorer/opt-persistdef`). Never push; never touch frozen dirs (`golden_baseline/`,
`baseline/`, `reference/`, `analysis/`, `records/` except `records/g2/`) or `records/sealed/`.

## Measured facts (root, a07-3 idle windows, correctness 22/22 PASS with the persistent kernel on)

The opt-in persistent kernel (`QSA_BF16_PERSIST=1`, `QSA_PERSIST_COLS=8`, kernel sha `57a43523…`) measured:
- prefill512_bf16_c65536: 133.0 → 110.3 µs (P=4, −17.0 %) → **105.2 µs (P=8)**;
- gate protocol (pinned chunked prefill): prefill_bf16_bm16 777.1 → 638.8 (P=4) → **606.0 µs (P=8)**;
  prefill_bf16_bmge64 770.7 → 633.7 → **597.5 µs**;
- decode/draft/fp8 unchanged (routing does not select the persistent kernel there; observed deltas are
  idle-window noise). P=8 beat P=4 on both prefill buckets.

## Task

Flip the defaults so the persistent variant is used by default for the guarded bf16 r32 prefill shapes:
- `QSA_BF16_PERSIST` default **1** (0 still forces the single-tile r32 TDM kernel for A/B);
- `QSA_PERSIST_COLS` default **8** (2/4/8 still accepted; validate values);
- keep all existing guards exactly: persistent only for bf16 r32 TDM prefill with raw access, 128 B-aligned
  K, `columns >= 64*QSA_PERSIST_COLS`; unaligned/mixed/partial/narrow shapes must fall back to the current
  single-tile r32 TDM or raw r16 paths exactly as today; fp8, r16 decode/draft, r64 and cop paths untouched.
- update the routing/env-validation tests and the CPU/ISA gates so they assert the new defaults (default-on
  persistent for eligible shapes, `QSA_BF16_PERSIST=0` still yields the single-tile kernel, P=2/4/8 all
  compile, unaligned/raw/cop still exclude persistent). Keep the source of every pre-existing function
  byte-identical to the base where possible; note any test-only changes.
- No-device ISA evidence for `auto` default at P=8 and for the forced-single-tile A/B mode; report dispatch
  names (`..._persist8`) and that all non-persistent kernels stay byte-identical to base.

Commit on `qsa-index-scorer/opt-persistdef`; report JSON {status, base_commit, commits[], files_sha256,
what_changed, routing_table{default/off}, tdm_evidence{persist8 + single-tile dispatch, LDS bytes, ISA counts},
expected_gain{prefill_bf16:% measured}, risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done.

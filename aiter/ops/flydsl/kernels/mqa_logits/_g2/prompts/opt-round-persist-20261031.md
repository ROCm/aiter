# Round PERSIST — persistent blocks looping column chunks to overlap TDM with WMMA (prefill bf16, champion adea96d5)

You are the ROUND-PERSIST optimizer. **Time-box ~45 minutes.** Base = champion `adea96d5` (kernel sha
`9428b7ef…`: bf16 r32 prefill padded TDM 272 B pitch/17,408 B LDS, bf16 r16 c32 TDM decode/draft, fp8 r32
padded TDM 144 B pitch, fp8 r16 raw, r64 opt-in). Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-persist`
(branch `qsa-index-scorer/opt-persist`). Never push; never touch frozen dirs (`golden_baseline/`, `baseline/`,
`reference/`, `analysis/`, `records/` except `records/g2/`) or `records/sealed/`.

## Why this round

Measured: prefill bf16 bm16 is 777 µs = 30.9 % of the 15.7 TB/s gate, but its actual traffic (~1.5 GB K +
0.76 GB logits over the chunked protocol) works out to only ~1.9 TB/s effective. The TDM path exposes a
`tensor_wait(0)` + workgroup barrier once per block and every block stages exactly ONE 64-column K tile that
both M-tile pairs consume — there is nothing to overlap inside a block, and the PIPE round showed that
double-buffering that single-shot tile (69,632 B LDS) loses to occupancy (+12 % decode, +5.7 % draft). The
missing structure is **persistent blocks that loop over several column chunks**, so chunk c+1's K gathers can
be issued while chunk c's WMMA runs, at the cost of only 2 × 17,408 B = 34,816 B LDS.

## Task

Add a **bf16 r32 persistent prefill kernel**: one workgroup owns 32 query rows and loops over a contiguous span
of column tiles (e.g. `QSA_PERSIST_COLS` = 4 or 8 chunks of 64 columns, i.e. 256/512 columns per block), with
**two K-tile buffers** (2 × 17,408 B, 272 B pitch) pipelined: on each loop iteration, issue the next chunk's
TDM gathers into the idle buffer, consume the current chunk (WMMA + store), then `tensor_wait(0)` for the next
buffer and swap. Barrier discipline: WG barrier before cross-wave LDS consumption of a buffer and before
overwriting an in-flight buffer (buffer-local phases; do not serialise the whole block at every iteration).
Grid: `ceil(columns / (64*QSA_PERSIST_COLS))` by `ceil(rows/32)` with the shape guards unchanged; store masks
and exactly-once logit writes preserved. Dispatch it for bf16 prefill shapes with enough columns
(e.g. `columns >= 64*QSA_PERSIST_COLS`, full r32 single-request blocks); keep the current r32 single-tile
kernel for other shapes and the raw fallbacks for mixed/partial/unaligned. Keep the frp8/fp16 r16 paths
untouched.

## Hard constraints

Stock flydsl 0.3.3.dev886 only. TDM rules: uniform SGPR D#/ids, tensor ops not in clauses, `tensor_wait(0)`
observes the issued ops (per-buffer exact waits are fine when only that buffer's ops are outstanding), per-wave
issue-to-XACK ≤3 and per-SIMD ≤6, completion ≤63, never issue into or overwrite an in-flight/being-read buffer,
WG barrier before cross-wave LDS consumption, invalid ids clamp to physical page 0. Keep the safety contract
(`_plane_bytes` i64 extents, clamped/predicated stores, host guards, wrapper API, every logit written once).
Keep the current default (single-tile r32 TDM) selectable via an env switch so the root can A/B; the new
persistent variant should be opt-in until measured.

## Verification (no GPU) + deliverable

- CPU gates green on the new tree (contract/address/Q-LDS/trace16+32/K-TDM/routing/fp8/BM64/strict ISA +
  pipeline tests); add tests for the persistent schedule: two-buffer plan, exact wait/barrier ordering per
  iteration, no overwrite while in flight, byte-exact K fragments for every chunk/wave/lane/k-chunk, store
  coverage exactly-once, LDS bounds, and that the single-tile default path is byte-identical to base.
- No-device real-ISA trace (CPU-only container, image `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no
  `--device`; SSH user `blyu`, host `heliosr-1b114-a07-3.mnb.dcgpu`, key
  `C:\Users\blyu\.ssh\id_ed25519`, staging dir `/data/blyu/qsa/work/opt-persist-cpu-trace-20261031`): report
  tensor loads/waits per iteration, barrier pairs, LDS bytes, VGPR, zero waterfall markers.
- Commit on `qsa-index-scorer/opt-persist`; report JSON {status, base_commit, commits[], files_sha256,
  what_changed, tdm_evidence{geometry, buffers, chunk count, LDS bytes, ISA counts, waits/barriers},
  expected_gain{case:%}, risks, correctness_evidence{cpu_tests}, gpu_plan}. Report when done; a clean opt-in
  candidate with all gates green is the deliverable (root measures).

# Round J — prefill K reuse: two WMMA M tiles per wave (32 rows), bf16 + fp8

You are the ROUND-J optimizer (disposable, fresh context). **No GPU**; the root measures. Worktree:
`C:\Users\blyu\Documents\Workspace\MI455_QSA_INDEX_SCORER\repo\aiter\worktrees\qsa-opt-j`
(branch `qsa-index-scorer/opt-j`, base = final champion commit `cde2c821`; kernel sha `a28f9e03…`).
Never push. Never touch frozen dirs (`golden_baseline/`, `baseline/`, `reference/`, `analysis/`, `records/`
except `records/g2/`) or `records/sealed/`. **Time-boxed round: land the smallest safe change and commit
early; a clean partial result is better than an unfinished rewrite.**

## Champion context (measured on a07-3; correctness 22/22 + 9/9 bucket shapes PASS)

`cde2c821` = `1644eeb5` baseline + bf16 WMMA 16x16x32 + fp8 LDS-Q + fp8 WMMA + descriptor waterfall-loop fix.
Measured (median): decode_bf16_c65536 91.1 µs, decode_bf16_c32768 71.7, draft_bf16 89.5, prefill512_bf16
198.6 µs; decode_fp8_c65536 79.1, decode_fp8_c32768 70.1, draft_fp8 89.5, prefill512_fp8 192.0.
Bucket diagnostic (full 2880-row prefill launches): prefill_bf16 846 µs (4.44 TB/s modelled / 22 % of the
20 TB/s fallback), prefill_fp8 900 µs (1.27 TB/s / 6.4 %), while decode/draft sit at 15 %/7 %.

## Why prefill: K is re-read once per 16-row M tile

Both WMMA kernels use one wave = 4 query rows × 4 heads = one 16-row WMMA M tile, and a 64-column N tile. For
prefill buckets (2880 rows / 2 requests, i.e. ~1440 rows per request) the K rows for a 64-column tile are the
same for every 16-row group of the same request, so K is fetched `rows/16` times per request. Doubling the M
tile to **32 rows per wave** halves K traffic; on the bucket shapes that is the dominant term
(K ≈ 2.9 GB for prefill_bf16 vs ~1.5 GB modelled floor).

## Task (smallest safe slice, both dtypes, prefill-relevant shapes only)

Give a wave **two WMMA M tiles** (rows r0..r15 and r16..r31 of the same request), sharing the K fragments:
one K load/gather per step per wave, two accumulator sets, both stored with the existing predicated clamped
stores. Keep the current 16-row behaviour as the fallback for shapes where 32 rows cannot be formed
(rows < 32 in a block, mixed requests inside a 32-row group, tails). Host picks the 32-row specialisation
from shape scalars (compile-time `_m32` kernel name suffix, e.g. `..._r32_c64_...`), with a
request-uniformity guard and a per-16-row fallback path. Decode/draft paths must stay unchanged
(they are `rows/requests` ≤ 4, so 32-row groups would mix requests — just don't select the new path there).
bf16 and fp8 versions: reuse the existing builders; bf16 first, fp8 only if trivially safe within the time-box.

Constraints: keep the waterfall-free descriptor form (`_plane_bytes` i32-clamped extents), in-bounds
predicated/clamped stores, i64 extents, host guards, wrapper API; every output written exactly once in
[0, visible), -inf elsewhere; stock flydsl **0.3.3.dev886** only. CPU gates must stay green (contract incl.
WMMA lane emulator, trace_check 16/16, audit_addressing, test_q_lds, test_fp8_wmma, descriptor tests) plus
new tests: 32-row fast path == 16-row fallback bitwise on all fixtures, mixed-request 32-row groups fall back,
rows∤32 tails, and the M-tile store masks. `COMPILE_ONLY=1` trace of all specialisations must `[COMPILED]`
(you may SSH to a07-3 for the no-device trace: user `blyu`, host `heliosr-1b114-a07-3.mnb.dcgpu`, key
`C:\Users\blyu\.ssh\id_ed25519`, unique staging dir `/data/blyu/qsa/work/opt-j-cpu-trace-20261031`, image
`rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, no `--device` flags, no gpu-lock; see
`records/prompts/opt-round-i-20261031.md` for the exact dump recipe).

Commit on `qsa-index-scorer/opt-j` and report JSON {status, base_commit, commits[], files_sha256,
what_changed, expected_gain{case:%}, risks, correctness_evidence{cpu_tests}, gpu_plan}. State clearly whether
the new path covers prefill only and whether fp8 is included or deliberately deferred. Report when done.

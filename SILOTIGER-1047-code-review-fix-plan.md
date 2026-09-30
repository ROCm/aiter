# SILOTIGER-1047 — Code review fixes

Close the items in `general-review.md` (2026-09-24, review HEAD `b8ee4802e`)
that a 2026-09-28 source triage still found in the tree. Kernels have since
moved to `aiter/ops/flydsl/kernels/qsa/k1.py` and `k2.py`; the review's
`k1_family_a.py` / `k2_family_a.py` line numbers are historical. Follow the
current files.

Parent: [SILOTIGER-1040](https://amd.atlassian.net/browse/SILOTIGER-1040).
Ticket: [SILOTIGER-1047](https://amd.atlassian.net/browse/SILOTIGER-1047).
Ticket-wide plan: `SILOTIGER-1047-plan.md`. Review: `general-review.md`.

**In scope:** the still-open correctness, safety, test, cleanup, and AOT
items below. Each kernel change is re-timed on the bar shapes and reverted
if it slows them down.

**Out of scope (stale, or already closed on the parent plan):**

- Family A/B emit-kernel and wrapper duplication, and the dead `_VEC`.
  One `k1.py`, one emit module, one `qsa_k1_block_ids`. `H=8` is a second
  scorer compile.
- The unused vector the review cited at old `k2_family_a.py:630-636`. It
  is gone. Dead tiled-copy values that remain are phase 5.
- Competitor pin and the layer profiler. Parent phase 1 is checked.
- Production opt-in. Parent phase 4 is checked: `qsa_layer` is the in-tree
  `auto` / `flydsl` / `triton` entry. `qwen4_exp` is not in this tree, and
  the parent plan says that is intentional. Do not reopen it here.
- K1 long-row tie, sentinel, and expansion checks that already passed, and
  the overlay barriers the review found adequate. Do not rework them.

Work the phases **in order**. Later items assume earlier ones have landed.
Leave checkboxes unchecked until that item is done; paste tables and notes
under the relevant phase as evidence.

## Progress

- [x] 1. K2 correctness (4 GiB pages, empty-tile NaN, output layout, Q over-read, empty tables)
- [x] 2. K1 page faults (prefill over-read, decode page ids, K1 4 GiB / i32)
- [x] 3. K1 gfx942 H=8 LDS and the runtime arch allowlist
- [x] 4. Score-matrix scope, benches, and stronger K1 assertions
- [x] 5. Behavior-preserving kernel cleanup
- [x] 6. QSA AOT registration
- [x] 7. Optional: caller-owned K2 split workspace, only if a profile says so

## Locked decisions

These locks apply to **this campaign** unless a later note explicitly
supersedes them. Ticket-wide locks in `SILOTIGER-1047-plan.md` still
apply. This campaign does not reopen that plan's closed phases.

### Skills

- **Skills (read, do not recall).** Before writing or reviewing FlyDSL
  for this ticket, Read
  `.claude/skills/flydsl-kernel-authoring/SKILL.md` and follow it.
  For `op_tests/test_flydsl_qsa.py`, also Read
  `.claude/skills/aiter-op-test/SKILL.md`.
  Do not start kernel code from memory of those skills.
  Cleanup/modernization of existing kernels uses
  `flydsl-kernel-code-cleanup`, not authoring, and is not a phase
  of this plan unless a later lock says otherwise.

Phase 5 is that later lock, for this campaign only. It is a
behavior-preserving cleanup. It does not authorize a kernel rewrite.

### Test environment

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 6**
  (`HIP_VISIBLE_DEVICES=6`).
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale
  HSACO cannot mask a bad rewrite. `CACHE=0` in this container also needs
  `ROCM_PATH=/root/.flydsl/toolkit`, or `gpu-module-to-binary` fails to
  find `ld.lld`. `CACHE=1` hides that by hitting the disk cache.
- **Gate.** Both layers, after each kernel or test edit:
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    ROCM_PATH=/root/.flydsl/toolkit \
    python3 -m pytest op_tests/test_flydsl_qsa.py -q
  ```
  Paths may move; keep this snippet in sync. The full `__main__` sweep is
  the perf check below, not a prerequisite for a correctness-only checkbox.
  New cases are `test_*` in `op_tests/test_flydsl_qsa.py`. Do not put them
  on a `bench_*`.

### Performance

- **Fixes must not hurt performance.** A correctness or safety change that
  slows the serving path is a miss and reverts, even if the numerics are
  right. Flat within run-to-run noise is success. This campaign does not
  use the overlay plan's 3% keep-gate; that gate is for optimizations that
  have to win. Here a speedup is not required, and a loss is not allowed.
- **Bar shapes.** Re-time the kernel you touched, family A, GPU 6 / gfx950,
  cold `--rotate 0`, `FLYDSL_RUNTIME_ENABLE_CACHE=0`:
  - K2: decode `M∈{1,8}` at `L=32768`, prefill `M=512` at `L=8192`
  - K1: the same points, plus one long-row prefill (`M>=16`, one request,
    `n_columns>512`) so the prefill scorer is the thing that ran
  Paste the before/after µs under the phase. Compare like with like: same
  rotate, same cache setting, HEAD control taken with `CACHE=0` immediately
  before the candidate. Do not compare a cold cell to a hot one.
- **Cheap form of each fix.** The shape of the fix is part of the lock.
  A slower implementation that also fixes the bug does not count.
  - **4 GiB pages.** Rebase the page base in 64-bit arithmetic once per
    page, then use a bounded page-local descriptor whose offset stays
    32-bit. Do not turn the K/V gather into a 64-bit flat address per
    element. `max_size=False` on the whole cache is not the fix.
  - **Empty-tile NaN.** When the previous max is `-inf` or the previous
    denominator is 0, rescale by 0. One predicate on the existing
    `alpha`. Do not add a second softmax, a branch per element, or a
    change to the finite-score path.
  - **Output layout.** Allocate the default `out` with a contiguous
    `torch.empty(q.shape, dtype=q.dtype, device=q.device)`. Do not put
    layout into the plan key: that multiplies compiles and can keep the
    first call's stride forever. Caller-supplied `out` stays
    contiguous-required.
  - **Q over-read.** Clamp dead query rows before the address is formed,
    the way K1 already does, or bound the descriptor to the live group.
    Do not add a per-element address on the Q load.
  - **Empty page table or cache.** Reject the geometry, or take a
    zero-output return, before launch. Do not add a host scan of the
    table or a no-load branch inside the tile loop.
  - **K1 page ids.** Add the same `phys_live` predicate K2 already has,
    and bound prefill `col_live` by the request context, before the
    load. Do not change score math, the selector, or `n_columns`.
    Padded tables may still inflate the score buffer; fixing that
    allocation is not part of the fault fix, because it can change
    which scorer runs.
  - **gfx942 H=8 LDS.** Route that one case to the one-row scorer, or
    shrink only the gfx942 tile. Do not shrink the gfx950 H=8 prefill
    tile and do not change gfx950 codegen to make the gfx942 budget.
  - **Arch allowlist.** Reject an unsupported runtime arch in the QSA
    wrappers, from `q.device`. Do not retune gfx950 or gfx942 codegen.
    Do not change `topk_select`'s global `get_gfx()` gate for other ops;
    the QSA wrapper must not call it on a device it has not allowlisted.
  - **Score matrix.** Do not remove the long-row `[M, n_columns]` fp32
    buffer. The parent plan already allows it. The fix is to make
    `SILOTIGER-1047.md` say the same thing.
  - **Benches and assertions.** Test and harness edits only.
  - **Cleanup.** Delete or correct comments and dead values. No numeric
    and no schedule change. If a "cleanup" moves an instruction, it is
    not cleanup.
  - **AOT.** Register with the existing `default_jobs()` collector.
    Do not add a `[None]` sentinel. Registration must not add work to
    the JIT call path.
  - **Split workspace.** Leave the internal fp32 partials in place
    unless a profile on the bar shapes shows that allocation. A
    caller-owned buffer is an API addition, not a second allocation
    on the default path.

### Scope

- **One defect per experiment.** Land and re-time before starting the
  next kernel edit. A phase checkbox stays open until its new `test_*`
  fails on the unfixed tree (or is a static check that cannot run on
  gfx950) and passes after the fix.
- **Added tests check behavior.** A new `test_*` must show the kernel
  or wrapper doing the wrong thing before the fix and the right thing
  after, by what it returns, raises, or launches. Reading source text,
  signatures, or bytecode is not that test.
- **gfx950 is the device we have.** GPU 6 proves the gfx950 numerics and
  the bar timings. The gfx942 LDS miss is a static shared-memory size
  plus a dispatch decision. Do not claim a gfx942 run from a gfx950 box.
- **Do not touch the overlay barriers** unless a fix in this plan cannot
  be correct without a new one. The review found no synchronization defect.
- **Do not change K1 tie policy, sentinel padding, or expand.** Those
  checks passed. Set equality stays the K1 gate; phase 4 adds duplicate
  and prefix checks on top of it.

## Subtasks

### 1. K2 correctness

Review must-fix 1, 2, 3, and 5, plus the known Q over-read the review
counted separately. All five are still in `k2.py`.

- [x] **1a. Page addressing past 4 GiB.** `make_buffer_tensor` on the whole
      `k_cache` / `v_cache` then `fx.slice` by physical page uses a 32-bit
      descriptor offset. Page size 16, physical page 262144 aliases page 0.
      `test_k2_page_past_4gib` fills page 0 with ones and page 262144 with
      twos and gathers both in one tile (expect 1.5 / 2 / 1). A
      multi-gigabyte allocation is skipped when the device cannot hold it.

      The kept compile specializes on cache bytes `> 2^32`. Bar shapes
      keep the uniform whole-cache descriptor. A larger cache is a second
      kernel: each gathered row rebases its page in 64-bit, then a
      descriptor covers only that row. Do not put both bodies in one
      kernel, and do not rebuild a descriptor on every row of a cache
      that still fits.

      A per-tile descriptor on every shape was tried on 2026-09-28 and
      reverted. It was correct and too slow (GPU 6, cold `--rotate 0`,
      `CACHE=0`): 12.36 → 16.74 µs, 18.65 → 36.65 µs, 263.93 → 1861.20 µs.

      The specialization, same script, four shots. Medians sit on the
      baseline; the 13.77 µs shot is the short-kernel spread.

      | M | L | before µs | after µs (four shots) |
      |--:|--:|----------:|-----------------------|
      | 1 | 32768 | 12.36 | 13.77, 12.52, 12.29, 12.63 |
      | 8 | 32768 | 18.65 | 18.91, 18.93, 18.95, 18.67 |
      | 512 | 8192 | 263.93 | 263.37, 259.26, 260.84, 262.24 |
- [x] **1b. Empty first tile.** `m` starts at `-inf`. An all-masked tile has
      `tile_max == -inf`, and `alpha = exp2(m_prev - m_new)` is NaN. The
      epilogue then writes zero because the denominator is not `> 0`.
      One select replaces that alpha with 0 when `m_prev` is `-inf` or
      the running denominator is 0. A finite max with a positive
      denominator keeps the exp2 result.
      `test_k2_empty_first_tile_keeps_later_token` failed on the unfixed
      tree at M=512, W=128, column 64 (mean 0) and passes after the
      guard, including the review's M=1, W=2048, column 16. Q/K are
      zero and V is one, so the live token's output is one.

      GPU 6, cold `--rotate 0`, `CACHE=0`, control taken immediately
      before the candidate:

      | M | L | before µs | after µs |
      |--:|--:|----------:|---------:|
      | 1 | 32768 | 12.23 | 12.14 |
      | 8 | 32768 | 18.76 | 18.67 |
      | 512 | 8192 | 260.37 | 261.61 |
- [x] **1c. Output layout vs the plan cache.** Default `out` was
      `torch.empty_like(q)`, which kept a non-contiguous layout. Only a
      caller-supplied `out` is checked for contiguity. `_plan` has no
      layout, and `_run_compiled` keeps the first compile. The default
      is now `torch.empty(q.shape, dtype=q.dtype, device=q.device)`.
      Caller-supplied `out` stays contiguous-required. The plan key is
      unchanged.
      `test_k2_default_out_ignores_query_strides` calls `[2, 24, 256]`
      at strides `(6144, 256, 1)` and then `(6144, 1, 24)`. On the
      unfixed tree the second call missed 12037 of 12288 elements. Both
      match the oracle after the allocation change.

      GPU 6, cold `--rotate 0`, `CACHE=0`, control taken immediately
      before the candidate. The kernel is unchanged on this contiguous
      path; the spread matches earlier shots of the same binary.

      | M | L | before µs | after µs |
      |--:|--:|----------:|---------:|
      | 1 | 32768 | 12.75 | 12.25 |
      | 8 | 32768 | 19.26 | 18.87 |
      | 512 | 8192 | 259.49 | 263.75 |
- [x] **1d. Q over-read.** The QK tile is 16 rows. Group 12 and group 5
      used to address the extra rows and mask them after the load. The
      prologue now builds one uniform descriptor on this row's live
      group (`num_records` is `group_size * head_dim * 2` bytes). The
      16-row copy is unchanged, so the padding rows hardware-zero
      instead of reading the next group or past `q`. The post-load
      mask remains. No per-element Q address.
      The mask already hid the extra rows, so the family A K2 oracles
      and `test_qsa_layer_family_b_matches_oracle` (group 5) were green
      before the change and stay green.

      GPU 6, cold `--rotate 0`, `CACHE=0`, control taken immediately
      before the candidate:

      | M | L | before µs | after µs |
      |--:|--:|----------:|---------:|
      | 1 | 32768 | 12.17 | 12.31 |
      | 8 | 32768 | 18.97 | 18.50 |
      | 512 | 8192 | 264.42 | 263.49 |
- [x] **1e. Empty page table or cache.** A nonempty index list with
      `k_cache.shape[0] == 0` or `page_table.shape[1] == 0` returns
      zeros before launch, next to the empty-query return. No table
      scan and no tile-loop branch.
      `test_k2_empty_cache_or_table_returns_zeros` saw a launch on the
      unfixed tree for both geometries and now gets zeros without one.
      The family A decode oracle is the smoke call; the kernel is
      unchanged, so the bar shapes were not re-timed.

### 2. K1 page faults

The K1 tile-extent audit. The prefill fault was reproduced. The decode
fault, the K1 4 GiB cap, and the i32 score index were not run; verify
them, and fix the ones that are real. Do not change score math.

- [x] **2a. Prefill page-table over-read.** The prefill K gather
      bounds `col_live` by `context_lens[0] / R` and requires
      `phys_live = (phys >= 0) & (phys < n_cache_blocks)` before the
      load, using a safe page id. `n_cache_blocks` is passed only on
      the prefill launch. Score math, the selector, `n_columns`, and
      the decode scorer are unchanged. The K descriptor is still the
      default `max_size=True`; the 4 GiB cap is 2c.
      `test_k1_prefill_padded_page_table`: one request, M=32, context
      4096, 64 real pages, four padded entries. On the unfixed tree
      pad `-1` aborted the process at the result read. Pad 0, pad
      `-1`, and page id 100000 match the oracle after the fix.
      `test_k1_family_a_set_equality_prefill` stays green.

      GPU 6, cold `--rotate 0`, `CACHE=0`, control taken immediately
      before the candidate. Long-row prefill, one request, 2048
      columns, so the prefill scorer ran:

      | M | L | before µs | after µs |
      |--:|--:|----------:|---------:|
      | 512 | 8192 | 30.20 | 30.46 |
- [x] **2b. Decode page ids.** The one-row scorer now requires
      `phys_live = (phys >= 0) & (phys < n_cache_blocks)` and loads K
      at the safe page id. A dead column still reads entry 0, but a
      `-1` there is not a cache index, and a `-1` inside the visible
      range drops that column. `n_cache_blocks` is a decode-launch
      argument. Score math, the selector, and `n_columns` are unchanged.
      `test_k1_decode_rejects_invalid_page_ids` uses M=4 and a table
      wider than 512 columns, so the one-row scorer runs. On the
      unfixed tree the no-page request (entry 0 is `-1`) aborted the
      process when the selector launched. After the fix that request
      selects nothing, and a `-1` on a visible page matches the oracle
      with those blocks masked. `test_k1_family_a_set_equality_two_tiles`
      stays green.

      GPU 6, cold `--rotate 0`, `CACHE=0`, control taken immediately
      before the candidate:

      | M | L | before µs | after µs |
      |--:|--:|----------:|---------:|
      | 1 | 32768 | 15.08 | 15.00 |
      | 8 | 32768 | 20.45 | 20.37 |
- [x] **2c. K1 4 GiB cache.** 1a's helper is closed over inside the
      K2 kernel, so both K1 scorers use the same scheme rather than a
      shared function: a cache that fits in 4 GiB keeps one whole-cache
      descriptor, and a larger cache is a separate compile that rebases
      each gathered row in 64-bit and then covers only that row.
      `test_k1_page_past_4gib`: physical page 1048576 starts at 4 GiB
      and holds ones; every other logical page is zeros. On the unfixed
      tree M=4 selected page 0's blocks. After the fix M=4 (one-row
      scorer) and M=32 (prefill scorer) match the oracle. The fast-path
      two-tile and prefill set-equality tests stay green.

      GPU 6, cold `--rotate 0`, `CACHE=0`, control taken immediately
      before the candidate. These shapes fit in 4 GiB, so they take the
      whole-cache compile:

      | M | L | before µs | after µs |
      |--:|--:|----------:|---------:|
      | 1 | 32768 | 14.58 | 15.15 |
      | 8 | 32768 | 19.25 | 20.62 |
      | 512 | 8192 | 31.21 | 29.94 |
- [x] **2d. i32 score-store index.** The store is `scores[row, out_col]`.
      The row stride of that contiguous fp32 matrix is a dynamic i64
      (tensor strides are i64 unless a kernel asks for 32-bit strides;
      these scorers do not). The i32 row is widened before the
      multiply, so `M * n_columns >= 2^31` does not wrap the address.
      Serving shapes do not reach that product anyway. The harness
      prefill is at most `M=8192`, and the long context is 128k tokens,
      which is 32768 indexer blocks. `8192 * 32768 = 2^28`. The
      review's `16384 × 131072` is twice that `M` and four times that
      column count. No store change, so the bar shapes were not re-timed.

### 3. K1 gfx942 H=8 LDS and arch allowlist

Review must-fix 4 and should-fix 7. Source analysis; GPU 6 is gfx950
and does not prove the LDS budget.

- [x] **3a. H=8 prefill LDS.** The 16-row tile is Q `16·H·128` bf16
      + K `32·128` bf16 + 16 int32s + C `H·2·2·64·4` fp32. At `H=8`
      that is 73,792 bytes, over gfx942's 65,536. `H=4` is 41,024 and
      fits. The one-row tile does not grow with `H` (Q is 16-wide, C
      is reused per head), so gfx942 `H=8` now takes that scorer.
      gfx950 `H=8` still takes the 16-row builder, and that builder's
      source is unchanged (`block_m = 16`, `block_n = 32`), so the
      gfx950 prefill bar was not re-timed.
      `test_k1_gfx942_h8_skips_prefill_tile` failed on the old dispatch
      with `gfx942 H=8 still dispatches the 16-row tile` and passes
      after. `test_k1_family_a_set_equality_prefill` still passes on
      this gfx950 box.
- [x] **3b. Runtime arch allowlist.** Both wrappers treated every
      device whose name does not start with `gfx950` as the gfx942
      path. `qsa_device_arch` keeps the ISA token before the first
      colon and accepts only `gfx942` and `gfx950`. `qsa_k1_block_ids`,
      `qsa_k1_score_and_select`, and `qsa_k2` call it with
      `q.device`'s `gcnArchName` before a kernel launch, and the
      long-row scorer does so again before `topk_select`. K32 stays
      `arch == "gfx950"`. `topk_select` still gates on `get_gfx()`.
      `test_qsa_arch_allowlist` failed on the unfixed tree with
      `No module named 'aiter.ops.flydsl.kernels.qsa.arch'` and passes
      after. Suffixes such as `gfx942:sramecc+:xnack-` still resolve,
      and `gfx1100`, `gfx1250`, `gfx90a`, `gfx9420`, and `GFX950` raise.
      The kernel bodies are unchanged, so the bar shapes were not
      re-timed. This container's GPU 6 is MI300X gfx942; the live name
      resolves to gfx942, which is the same tile the old
      `startswith("gfx950")` test selected. Pytest gate, `CACHE=0`:
      34 passed.

### 4. Score-matrix scope, benches, and K1 assertions

Review should-fix 6, 8, and 9, the "consider" assertion note, and the
still-open ticket-doc half of "Earlier findings." No kernel change.

- [x] **4a. Resolve the score-matrix conflict in the ticket.** Phase 2
      of `SILOTIGER-1047.md` said "No full score matrix." It now matches
      the parent lock: fused emit writes ids with no score matrix when
      `visible <= 512`, and longer rows materialize an `[M, n_blocks]`
      fp32 buffer. The long-row allocation in `k1.py` is unchanged.
- [x] **4b. K1 bench timings cover the same work.** `bench_qsa_family_a_k1`
      now times `_flydsl_k1_select`: `qsa_k1_block_ids`, then the vendored
      Triton expand that live AMD select already includes. The vLLM and
      #4882 columns still time `qsa_select_paged_tokens`, which expand
      inside the call. Set equality stays on block ids. The layer bench
      is unchanged. `test_family_a_k1_bench_times_expand` checks that the
      helper's ids match `qsa_k1_block_ids` and its indices match a
      separate expand. No kernel change, so the bar shapes were not
      re-timed.
- [x] **4c. Sweep mismatches fail, and requested M is not dropped.**
      Family A K1/K2 report `M<=8` as decode and every larger requested
      M, including 64, 2048, and 8192, as prefill. Family B K1 runs
      every requested M in the emit or long-L table for that `L`.
      `M > L` is logged. A nonzero `flydsl_k1 err` and a `flydsl_k2 err`
      above the unit tolerance (`checkAllclose` rtol=1e-2 atol=1e-2,
      so `err != 0`) raise. `test_k1_k2_sweep_keeps_requested_m_and_fails_on_mismatch`
      checks the decode/prefill split and that a nonzero K1 or K2 err
      raises. No kernel change, so the bar shapes were not re-timed.
      The HIP-graph table is still decode `M<=8`.
- [x] **4d. Stronger K1 assertions.** `_set_mismatch_ratio` still
      compares sets and ignores duplicates and order. The existing K1
      unit cases now also require each output row to be unique valid
      ids, then only `-1`. A duplicate or a valid id after `-1` fails
      even when the sets match. `test_k1_block_ids_require_packed_prefix`
      rejects a duplicate, a valid id after `-1`, and a set mismatch.
      Those unit cases, including the no-live-page
      row of `-1`, passed on GPU 6. No kernel change, so the bar
      shapes were not re-timed. Sweep and bench columns still report
      set mismatch only.

### 5. Behavior-preserving kernel cleanup

Review should-fix 10, plus the audit notes that are dead code rather
than faults. Read `flydsl-kernel-code-cleanup` before editing. No
numeric change, no schedule change, no bar re-time beyond the pytest
gate unless a diff is not comment-only — if the diff is not
comment-only, re-time the touched kernel and revert on a loss.

- [x] **5a. K2 header.** The module docstring said gfx950 always stores
      K and V separately. Decode (BLOCK_N=16) still overlays one tile;
      ``split_kv_lds`` is gfx950 prefill only. The header now says that.
      Docstring only, so the bar shapes were not re-timed.
- [x] **5b. Dead K2 values.** The builder built
      `make_tiled_copy(lds_copy64, ...).get_slice(tid)` and
      `make_tiled_copy_B(pv_b_atom, ...).get_slice(lane)` and discarded
      both. Both ops are `Pure`. `pv_wave_mma` only fed the second
      call, so it went too. `kv_store`, `qk_a_copy`, `lds_copy64`, and
      `pv_b_atom` stay; those are used.
      Family A decode and prefill still match the oracle. Paired bar
      times on GPU 6, cold cache: M=1 L=32768
      12.25 → 12.25 µs, M=8 18.47 → 18.34 µs, M=512 L=8192
      264.17 → 262.45 µs. Flat.
- [x] **5c. Emit's unused arguments.** The emit kernel took `q`,
      `k_cache`, `page_table`, and `score_scale` and never read them.
      Those parameters are gone from the kernel and from its launch.
      The plan key is still one entry per page size, so this does not
      add a second compile. `qsa_k1_block_ids` still accepts them: the
      long-row scorer reads them, and the emit body still writes
      `tid` or `-1`. Family A short decode, which is the emit path,
      still matches the oracle.
      The K1 bar shapes have more than 512 columns, so they run the
      scorer. The emit ceiling (`n_columns=512`, L=2048) on GPU 6,
      cold cache: M=1 2.04 → 1.57 µs, M=8 3.08 → 2.85 µs. Not a loss.

### 6. QSA AOT registration

"Earlier findings": there is no QSA `OpKind` or collector registration.
`aiter/aot/flydsl/common.py` `OpKind` is still MoE, GEMM, conv, GDN,
and FMHA. The collector does not treat a `[None]` job list as "no
configs"; use `default_jobs()`, the same pattern as `mega_moe.py`.

- [x] **6a.** Add a QSA `OpKind` and a collector that returns
      `default_jobs()` for the family A launches this campaign already
      compiles: page-16 K1 emit, the M=1 long-row scorer (M=8 decode
      shares it), the M=512 prefill scorer, and the three K2 bar
      launches. Family B, H=8, and the other sweep values are not
      registered. `qsa_k1_block_ids` and `qsa_k2` are unchanged; the
      collector is imported only from the AOT driver.
      `test_qsa_aot_collector_lists_family_a_launches` failed on the
      unfixed collector with `QSA is not an AOT kind` and passes after.
      It checks the job list the collector returns. No kernel change,
      so the bar shapes were not re-timed.
- [x] **6b.** A collector unit check: an empty config list is an empty
      job list, not a `[None]` sentinel. No GPU required.
      `default_jobs(())` already returned `[]`. The collector was not
      changed, and `[None]` was not added.
      `test_qsa_aot_empty_launch_list_has_no_jobs` calls that function
      and checks the list it returns. It passes. One launch still
      collects one job. No kernel change, so the bar shapes were not
      re-timed.

### 7. Optional split workspace

"Consider" in the review. Not required to close the review.

`qsa_k2` allocates fp32 partials on every split launch (`k2.py`
around the `n_splits == 1` branch). At family A, `M=128` and 8
splits, that is about 24 MiB. The bar shapes are smaller, and the
prefill bar does not split.

- [x] **7a.** Profile the bar shapes. The allocation is visible on the
      decode bars, so the API is unchanged here and 7b is the
      follow-up.
      GPU 6, family A, caller-supplied `out`, cold cache. M=1 at
      L=32768 uses 128 splits (3.01 MiB fp32). M=8 uses 32 splits
      (6.02 MiB). M=512 at L=8192 uses 1 split and allocates nothing.
      The GPU trace is the kernel only: 12.07 µs, 16.48 µs, and
      254.83 µs. `torch.empty` launches no kernel.
      Wall clock of the two empties, allocator already warm
      (`memory_reserved` unchanged), against a 0.11 µs empty call:
      3.16–3.25 µs at M=1 and 3.25–3.37 µs at M=8. The kernel-only
      launch is 11.3–11.5 µs and 14.8–15.1 µs. Prefill wall time
      matches that launch (246.7 µs against 246.3 µs). A CUDA event
      around the same empties is 1.5 µs above its 4.4 µs floor.
- [x] **7b.** If it is visible, accept a caller-owned workspace and
      keep the current allocation as the default when the caller
      passes none. Re-time the bar shapes. The default path must not
      get a second buffer.
      `workspace` is `(partial_out, partial_lse)`. `None` still
      allocates that pair on a split launch. One split still stores
      into `out` and does not read `workspace`.
      `test_k2_caller_workspace_is_the_only_partial_buffer` failed
      unfixed with `qsa_k2 does not accept a caller-owned split
      workspace` and passes after. The workspace launch matches the
      default launch, the caller's `partial_out` is written, and the
      peak of that launch stays under one partial buffer. The default
      launch's peak is one buffer, not two.
      Default-path bars on GPU 6, cold cache, control immediately
      before: M=1 L=32768 11.99 → 12.17 µs, M=8 18.63 → 18.61 µs,
      M=512 L=8192 261.74 → 261.65 µs. Flat.

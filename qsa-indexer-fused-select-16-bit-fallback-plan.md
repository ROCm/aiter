# QSA indexer 16-bit score buffer: implementation plan

Living plan for the 16-bit fallback in [SILOTIGER-1481](SILOTIGER-1481-description.md). Check a box when a step is done. Put measured numbers in [Results](#results) and dated notes in [Log](#log). Leave the prose in place when a step finishes.

[qsa-indexer-fused-select-plan.md](qsa-indexer-fused-select-plan.md) is the master document. Its fixed decisions carry over to this plan: test environment, hardware, geometry, correctness reference, timing protocol, acceptance gates, coordination, and the vLLM checkouts. Change one of those only by editing the master and adding a log line there. This document locks the decisions that belong to the fallback, and it does not reopen a decision the master has closed.

The master records that option 1, the fused split-local top-512, misses gate 1 after tuning, and that option 2, scoring twice and storing no scores, is out. This plan is the fallback the ticket names for that outcome: keep one score pass, store order-preserving 16-bit keys instead of fp32, halve the score traffic, double the rows per chunk, and rescore the boundary bucket in fp32.

## Locked decisions

Change these only by editing this section and adding a log line. Code, tests, and the write-up follow this section as it stands. Anything not listed here is read from the master.

### What carries over

- The interpreter, `gpuq` rules, idle-grant rule, and session record in the master's [Test environment](qsa-indexer-fused-select-plan.md#test-environment). GPU work for this plan uses that prefix. Branch work stays on `samaario/qsa-indexer-fused-select` unless a log line names a worktree.
- gfx950 is the acceptance device. gfx942 is measured, and this path is enabled there only where it beats the current indexer. LDS budgets are unchanged.
- Real vLLM geometry, the 1.7e-6 tie bound, the oracle, and the set gate. Another production kernel is not the oracle.
- The timing protocol: CUDA events, five warmup steps, ten timed steps, median, arms interleaved, warm and cold. The kernel gate is the master's gate 1. At 8K the rule is 1.05× the faster of vllm#65188 and aiter#5996. At 32K and above the rule is strictly faster. The vLLM arms stay the sum of their score, top-k, and expand kernels. aiter#5996 and this path are whole calls, matching the probe-free harness in the master.
- Gate 5 does not apply again. The master already applied it. The shares are 0.454 at `p128k` and 0.334 at `p256k`, and the profile-only close was not taken.
- `_fused_select_wins` stays false. This plan does not dispatch the fused select kernel.

### Mechanism

- One score pass, on today's 16-row indexer tile. The pass writes an order-preserving 16-bit key per visible block and does not write an fp32 score buffer.
- The stored key starts from `_score_key16` in `aiter/ops/flydsl/kernels/qsa/indexer_fused_select.py`: the high 16 bits of the order-preserving fp32 bit pattern. Positive scores keep fp32 order in those bits. Phase 1 may replace that map only if it fails the collision check below, and the replacement stays 16 bits and order-preserving.
- Two scores that land in different keys keep fp32 order. Two scores that land in the same key are not ordered by the stored buffer. The boundary rescore is what makes that exact.
- Selection keeps every block whose key is strictly above the bucket that contains the 512th key, drops every block whose key is strictly below it, and rescores the boundary bucket in fp32. The rescore reads the compressed keys again for those blocks only, computes the same bf16-in fp32-accumulate score the oracle uses, and picks the exact remainder of the top 512. Exact ties at the 512th score may resolve to either block. The rescore may run inside the select kernel. What makes it correct is that formula and the 1.7e-6 tie gate, not where the rescore runs.
- The fp32 tie bound stays 1.7e-6. A block outside the oracle set passes only when its oracle score is within that bound of the 512th score. A coarser key is allowed only when this rescore still meets that gate.
- Expand stays the existing expand. This plan does not fold expand into the scorer.
- When `n_i <= 512`, take all visible blocks without scoring them, on the existing emit path.

### Score buffer and chunks

- The workspace cap stays `_SCORE_WORKSPACE_BYTES` (128 MiB), the same cap as today's fp32 buffer.
- A stored row is `n_columns` uint16 keys, 2 bytes each. Today's row is `n_columns` fp32 scores, 4 bytes each. Rows per chunk are `128 MiB / (n_columns * 2)`, against today's `128 MiB / (n_columns * 4)`.
- `n_columns` is the width today's `qsa_indexer_block_ids` already scores for that step. This plan does not switch the indexer from the padded table to a trimmed prefix. Trim is a separate change, and mixing it in would make a win unreadable.
- A caller-owned buffer is sized from the 2-byte row. It is overwritten, reused across calls without a memset, and not returned.
- The scorer and the selector chosen for the whole batch stay in force for a short last chunk, as they do on the fp32 path.

### Dispatch

- The public entry stays on today's indexer until a measured step beats the faster arm under the kernel gate.
- The fp32 path keeps its numerics, its launches, and its buffer check. A 16-bit buffer is a separate path, not a reinterpret of the fp32 workspace.
- Shapes and archs this path does not win keep `qsa_indexer_block_ids`.

### Out of scope

- Option 1 and option 2. Do not retune the fused select kernel in this plan.
- Serving, the decode table, and vLLM wiring. Those start only after the kernel gate passes, and they follow the master's Phases 8 and 9.
- FP8 and MXFP4 keys, `stacked_decode`, and the specialized decode kernels.

## Phases

### 0. Budget from the master's baseline

- [x] From the master's [Baseline](qsa-indexer-fused-select-plan.md#baseline), write the fp32 chunk count, score bytes, and `T_score` / `T_topk` / `T_expand` for `p8k`, `p32k`, `p128k`, and `p256k` into [Budget](#budget).
- [x] Compute the 16-bit rows-per-chunk and chunk count at the same `n_columns` the baseline used. Record both. Do not treat the doubled count as a measured win.
- [x] Name the gap this path can move: score-buffer traffic, and the extra passes over K that come from the chunk loop. Name the gap it cannot move by itself: a top-k that still reads every stored key, and expand.
- [x] Stop and record it in [Decision record](#decision-record) if the score-buffer share of the faster arm, at both `p128k` and `p256k`, is smaller than the miss this path would have to close. Otherwise Phase 1 starts. Applied 2026-11-19: the stop does not fire. Phase 1 is next.

### 1. Key and boundary check

- [x] On the seeded 8K, 32K, and padded-prefix cases from the master's oracle check, count how often two scores that differ by more than 1.7e-6 share a `_score_key16` value, and how wide the boundary bucket is at the 512th key.
- [x] Keep `_score_key16` if a torch rescore of that boundary bucket reproduces the oracle set on those three cases. Otherwise pick another order-preserving 16-bit map, log it, and rerun the same check. Kept. The rescore matched the oracle set on all three cases.
- [x] Record the key, the boundary rule, and the collision counts in [Key check](#key-check).

### 2. Store 16-bit keys from the existing scorer

- [x] Add a scorer path that writes uint16 keys for the visible prefix and leaves the fp32 store unchanged. Same 16-row tile, same bf16 MFMA, same 4-head ReLU sum, same masking of invisible, incomplete, and missing-page blocks.
- [x] Faithfulness: decoding the stored keys with the Phase 1 rule matches the fp32 scorer's visible scores on the 32K case, within the tie bound, and the kernel time stays within 5% of the same-session fp32 scorer when the store width is not the thing under test. The 5% check uses a same-width comparison only if one is cheap to run; the real comparison is the halved buffer in Phase 4. Passed. Same-width ratio 1.007. The set match is exact.
- [x] Do not change the production dispatch. `qsa_indexer_block_ids` still launches the fp32 store. `_fused_select_wins` stays false.

### 3. Select on keys, rescore the boundary

- [x] Select the top 512 from the uint16 buffer by the Phase 1 rule. Rescore only the boundary bucket, in fp32, from the compressed K. Passed on the seeded 32K case. A row with at most 512 visible blocks keeps those blocks and does not rescore.
- [x] The rescore computes the oracle's formula, bf16 in and fp32 accumulate, from compressed K. It may run in the select kernel, and the 1.7e-6 tie gate grades it. Another production kernel's stored score is not the reference. The bucket is gathered from compressed K and accumulated in fp32.
- [x] Expand with the existing expand. Token ids match the set gate's contract: `[M, 2051]`, tail included, `-1` padded. Passed on the 32K case and on a short row.

### 4. Chunk the 2-byte buffer

- [x] Size the workspace at 2 bytes per column under the 128 MiB cap. A full step of 16384 rows reuses one buffer across chunks. 1019 rows per chunk, 17 chunks, one buffer.
- [x] A short last chunk keeps the scorer and the selector chosen for the full batch. An 8-row tail of a 1027-row step kept the full-batch scorer.
- [x] Record chunk count and bytes written and read per case. Recorded under [16-bit chunks](#16-bit-chunks).

### 5. Set gate

- [x] Run the master's set-gate cases: padded tables, missing pages, a cache view spanning more than 4 GiB, row tiles spanning requests, mixed steps, MTP rows, and `n_i <= 512`. All eight of the master's `_SELECT_CASES` ran, including `workspace`.
- [x] `select err` is the fraction of rows outside the 1.7e-6 window. The gate passes only at 0 on every case. 0 on every case.
- [x] Tests skip on archs where this path is off. `qsa_indexer_key16_supported` is gfx950 only, and every int16 check that launches the path skips on that reason.

### 6. Kernel gate

- [x] Before the gate, time the int16 scorer alone at `p128k` and `p256k` against the fp32 scorer and the aiter#5996 whole call in one idle session, and measure the boundary bucket there. Recorded under [16-bit select budget](#16-bit-select-budget). The budget is 1901.5 µs at `p128k` and 2405.0 µs at `p256k`.
- [x] Replace the host select with a GPU int16 select that keeps the Phase 3 rule and passes the Phase 5 set gate. The existing radix takes fp32 only, so this is its own kernel. `qsa_indexer_key16_select` now launches that kernel. `select err` is 0 on the seeded 32K case, the short row, both wide-bucket cases, and all eight set-gate cases.
- [x] One idle gfx950 session, probe-free, interleaved with aiter#5996, vllm#65188, and vllm#62589. Same ten cases as the master's Phase 6 matrix. Five warmup steps, ten timed steps, warm and cold. `op_tests/bench_qsa_indexer_select_abc.py --key16`, bus 0000:f5:00.0.
- [x] Fill [Kernel gate](#kernel-gate). Apply gate 1 per case. Every case misses.
- [x] If `p128k` or `p256k` misses, record the miss and stop. Do not start serving. Both miss: 6084.9 µs against 3611.5 µs at `p128k`, and 9468.2 µs against 5734.1 µs at `p256k`. Serving does not start. gfx942 is not measured.
- [ ] Measure gfx942. Enable this path there only for the cases where it wins.

### 7. Dispatch

- [ ] Add a host predicate, default false. Turn it on only for shapes that won in Phase 6.
- [ ] The fp32 entry and `_fused_select_wins` stay as they are.
- [ ] Run the existing `op_tests/test_flydsl_qsa.py` suite and the new tests through `gpuq`.

### 8. Close or hand back

- [ ] If Phase 6 passes, serving and the decode table follow the master's Phases 8 and 9, and the numbers land in this document's tables.
- [x] If Phase 6 misses, write the outcome in [Decision record](#decision-record). The fused select stays off. Today's indexer stays the public path. Recorded 2026-11-20.
- [x] Log the date.

## Results

### Environment confirmation

| Session | Date | `get_gfx()` | gpuq / rocm-smi / bus | Device | Idle |
| --- | --- | --- | --- | --- | --- |
| Phase 2 store | 2026-11-19 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:75:00.0 | MI355X | rocm-smi GPU 3, use 0%, no other pids |
| Phase 3 select | 2026-11-19 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:65:00.0 | MI355X | rocm-smi GPU 2, use 0%, no other pids |
| Phase 4 chunks | 2026-11-19 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:f5:00.0 | MI355X | rocm-smi GPU 7, use 0%, no other pids |
| Phase 5 set gate | 2026-11-19 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:75:00.0 | MI355X | rocm-smi GPU 3, use 0%, no other pids |
| Phase 6 select budget | 2026-11-19 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:65:00.0 | MI355X | rocm-smi GPU 2, use 0%, no other pids |
| Phase 6 GPU select | 2026-11-20 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:65:00.0 | MI355X | rocm-smi GPU 2, use 0%, no other pids. One earlier grant was busy and was discarded |
| Phase 6 kernel gate | 2026-11-20 | gfx950 | gpuq, ROCm 7.2.1, bus 0000:f5:00.0 | MI355X | rocm-smi GPU 7, use 0%, no other pids |

### Budget

fp32 chunks, score bytes, and stage times are the aiter#5996 rows of the master's baseline (2026-11-12, bus 0000:f5:00.0). That is the path whose buffer this plan changes. The bench scores `COLUMNS` = 168 × 392 = 65856 on every case, and it counts chunks with `_score_workspace_rows(COLUMNS)`. That is 509 fp32 rows per chunk (`128 MiB / (65856 × 4)`). The same width at 2 bytes is 1019 rows per chunk (`128 MiB / (65856 × 2)`). Score bytes are `rows × 65856 × element_size`, which is the figure the baseline prints. The 16-bit chunk count and byte count are that formula, not a timed result.

| Case | Rows | fp32 chunks | fp32 score bytes | `T_score` | `T_topk` | `T_expand` | `T_step` | 16-bit chunks | 16-bit score bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p8k | 8192 | 17 | 2058.0 MiB | 328.0 | 438.4 | 83.5 | 849.9 | 9 | 1029.0 MiB |
| p32k | 16384 | 33 | 4116.0 MiB | 954.3 | 820.6 | 146.9 | 1921.8 | 17 | 2058.0 MiB |
| p128k | 16384 | 33 | 4116.0 MiB | 2072.9 | 1600.1 | 124.3 | 3797.3 | 17 | 2058.0 MiB |
| p256k | 16384 | 33 | 4116.0 MiB | 3905.9 | 1833.2 | 122.3 | 5861.4 | 17 | 2058.0 MiB |

What this path can move. The stored buffer is half the bytes, 2058.0 MiB written instead of 4116.0 MiB on the 16384-row cases, and the top-k's read of that buffer halves with it. The chunk loop goes from 33 passes over K to 17 on those cases, and from 17 to 9 at `p8k`. Each pass re-reads the same compressed keys.

What it cannot move by itself. Top-k still reads every stored key, and expand does not touch the score buffer (83.5 to 146.9 µs here). `T_score` is the whole score kernel, compute included, so halving the store does not halve `T_score`. The 16-bit chunk counts above are not a measured speedup.

The faster arm at `p128k` and `p256k` is aiter#5996, both in this baseline (`T_step` 3797.3 and 5861.4) and in the master's probe-free gate (3623.9 and 5845.4). `T_score` is 0.546 and 0.666 of the baseline step. The miss against that arm is a strict improvement on today's indexer, and the score stage is larger than that miss. The Phase 0 stop does not fire. At `p8k` the faster arm is vllm#65188 at 207.3 µs, and the whole aiter score stage is 328.0 µs against a 642.6 µs gap, so the buffer inside that stage cannot be the whole of the 8K miss. That case is not the stop rule. The stop rule is `p128k` and `p256k`.

### Key check

`_score_key16` stays the key. gfx950, gpuq GPU 0, the seeded inputs from `run_qsa_indexer_select_oracle`: 256 rows at context 8192, 128 rows at context 32768, and 128 rows at context 8192 padded to 65856 columns. A colliding key holds two or more visible scores whose span exceeds 1.7e-6. A colliding pair is two such scores inside one key. The boundary width is the number of visible blocks whose key equals the key of the 512th score. The rescore keeps every strictly higher key and fills the rest from that bucket by the fp32 score, smaller index on an exact tie.

| Case | Rows | Columns | Colliding keys | Colliding pairs | Max span | Boundary min / median / max | Oracle set matches |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 8K | 256 | 2048 | 101944 | 975004 | 0.0308 | 2 / 8 / 18 | yes |
| 32K | 128 | 8192 | 85240 | 8062017 | 0.0311 | 3 / 11 / 19 | yes |
| padded prefix | 128 | 65856 | 51048 | 497783 | 0.0307 | 2 / 8 / 17 | yes |

The key is coarser than 1.7e-6: one key spans up to about 0.031 in score. The boundary bucket stays under 20 blocks, so the fp32 rescore reads that many blocks per row on these cases, not the whole row. The check is `test_qsa_score_key16` in `op_tests/test_flydsl_qsa.py`.

### 16-bit store

The 16-row scorer has a second store, `store_key16`, launched only by `qsa_indexer_prefill_score_buffer`. The fp32 store is the default, and `qsa_indexer_block_ids` does not call the new entry. The key buffer is int16. A positive score's key is 0..32767, the same bits as the Phase 1 high-16 map. An invisible or missing-page column stores -1, which sorts below those keys. Columns past the last visited tile are left unwritten.

The check is `test_qsa_score_key16_store`, the seeded 32K case: 128 rows, context 32768, page 392, identity page table, 8232 columns. Both launches use that width. The kernel wrote every visible column. On those columns the stored key equals `_score_key16` of the fp32 scorer, and -1 where that scorer wrote `-inf`. Treating the unwritten tail as not visible, the Phase 1 boundary rescore matches the fp32 scorer's top-512 on every row. Same-session CUDA-event means, 5 warmup and 50 timed, on bus 0000:75:00.0: fp32 21.867 µs, key16 22.018 µs, ratio 1.007. That is inside the 5% bar. The halved buffer is Phase 4.

### 16-bit select

`qsa_indexer_key16_select` launches one workgroup per row. The workgroup stages the visible int16 keys in LDS, finds the 512th key with two 8-bit histogram passes, and keeps every key strictly above it. The boundary bucket is rescored from compressed K, in fp32, with the oracle's bf16-in accumulation. A bucket of at most 1024 blocks is ranked in LDS. A wider bucket takes four more 8-bit passes over the recomputed fp32 score. A row with at most 512 live keys keeps those blocks and does not rescore. Block order within a row is unspecified. `qsa_expand_block_indices` then writes token ids `[M, 2051]`. `qsa_indexer_block_ids` does not call this.

The checks are `test_qsa_key16_select` and `test_qsa_key16_select_bucket`. On the seeded 32K case the block set matches the oracle within 1.7e-6 on every row, and the token ids match the expand of those blocks. The boundary bucket holds 1365 blocks across the 128 rows. A 16-row context-2048 case has no boundary bucket, and its set matches too. A crafted bucket of 900 blocks takes the LDS rank, and one of 8092 blocks takes the recomputed radix. Both match the oracle within 1.7e-6. CUDA-event means, 5 warmup and 10 timed, on bus 0000:65:00.0: 45.69 µs for the 32K select, 37.61 µs for the short row, 136.19 µs for the 900-block bucket, and 1815.30 µs for the 8092-block bucket. The wide radix is a crafted overflow, not the bench's bucket, which stays under 40 blocks. These times are the select, not the kernel gate.

### 16-bit chunks

`qsa_indexer_key16_score_and_select` stores int16 keys in one buffer of `_key16_workspace_rows` rows and reuses it for every chunk. That is 1019 rows at 65856 columns, `128 MiB / (65856 × 2)`. The fp32 helper stays at 509 rows. A short last chunk passes the full batch into the scorer, so an 8-row tail does not fold the heads. The selector is the same key select on every chunk. `qsa_indexer_block_ids` does not call this.

Key bytes are `rows × 65856 × 2`, written once and read once by the select. K read is one compressed-K pass per chunk, `chunks × visible_blocks × 128 × 2`. The scorer also rereads K inside a pass; that extra traffic is not in this column.

| Case | Rows | Chunks | Key written | Key read | K read |
| --- | ---: | ---: | ---: | ---: | ---: |
| p8k | 8192 | 9 | 1029 MiB | 1029 MiB | 4.5 MiB |
| p32k | 16384 | 17 | 2058 MiB | 2058 MiB | 34 MiB |
| p128k | 16384 | 17 | 2058 MiB | 2058 MiB | 136 MiB |
| p256k | 16384 | 17 | 2058 MiB | 2058 MiB | 272 MiB |

The short-tail check is `test_qsa_key16_chunks`: 1027 rows, 65856 columns, chunks of 1019 and 8. The tail's keys match a direct score that was told the full batch, the first chunk's block ids match its own select, and the whole step matches the oracle within 1.7e-6. Token ids are the expand, width 2051. CUDA-event mean, 2 warmup and 5 timed, on bus 0000:65:00.0: 210.41 µs for the score plus the GPU select over both chunks. That is not the kernel gate.

### 16-bit set gate

`test_qsa_key16_select_gate` runs `qsa_indexer_key16_score_and_select` on the master's `_SELECT_CASES` and grades it with the master's tie gate against the fp32 oracle. Every row must also be packed, valid ids then `-1`, and the token ids must equal the expand of those blocks. The key buffer starts filled with 12345, a valid positive key, so a column the scorer did not write would be ranked and fail the gate. `workspace` runs twice, refilling the buffer with -7 between runs, and the token ids must not change.

Three changes let every case reach this path. The score entry now runs the 16-row tile on any batch except gfx942 at 8 heads. The tile masks rows past the batch, and the 8-row `mtp` case uses the 4-row-by-4-head fold. A table of at most 512 columns, the `emit` case, takes the existing emit path and is not scored. `qsa_indexer_key16_supported` gates the path to gfx950.

| Case | Rows | Columns | Path | `select err` | µs |
| --- | ---: | ---: | --- | ---: | ---: |
| pad | 32 | 65856 | key16 | 0 | 68.63 |
| missing | 16 | 2352 | key16 | 0 | 68.26 |
| wide | 16 | 2352 | key16 | 0 | 69.70 |
| span | 16 | 2352 | key16 | 0 | 68.10 |
| mixed | 16 | 2352 | key16 | 0 | 68.64 |
| mtp | 8 | 2352 | key16 | 0 | 67.52 |
| emit | 16 | 392 | emit | 0 | 42.51 |
| workspace | 32 | 65856 | key16 | 0 | 68.24 |

CUDA-event means, 2 warmup and 10 timed, on bus 0000:65:00.0, after the host select was replaced by the GPU select. They are not the kernel gate. The Phase 5 session, on bus 0000:75:00.0, had already passed these cases with the host select, and in that session the Phase 2 to 4 checks passed again. The same-width store ratio was 0.999 (21.576 µs against 21.604 µs fp32).

### 16-bit select budget

This measures how much time is left for an int16 select once the int16 scorer and expand have run. The geometry is the bench's: `_build` on the wide indexer cache, 65856 columns, live K filled. The script, `/tmp/key16_budget_probe.py`, is not in the repo. Four arms run interleaved on each case. Each is warm, with 5 warmup and 10 timed rounds, and the figure is the median of one CUDA-event pair per arm.

- `A`: `qsa_indexer_block_ids` plus expand, the aiter#5996 whole call.
- `E`: expand alone on A's block ids.
- `S32`: `qsa_indexer_prefill_score_buffer` with the fp32 store, in 509-row chunks.
- `S16`: the same entry with `store_key16`, in 1019-row chunks.

Both scorer arms pass the full batch as `batch_rows`. The budget is `A - S16 - E`. Today's top-k is shown as `A - S32 - E`. That is derived, not probed, and it assumes `S32` matches the score stage inside `A`. The bucket is the number of visible blocks whose key equals the key of the 512th entry. It is counted on the last 1019-row chunk, on rows with more than 512 visible blocks.

| Case | Rows | `A` | `E` | `S32` (chunks) | `S16` (chunks) | `S16 / S32` | Budget | Today's top-k | Bucket min / median / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| p8k | 8192 | 833.4 | 81.5 | 328.7 (17) | 176.9 (9) | 0.538 | 575.0 | 423.2 | 3 / 9 / 13 |
| p32k | 16384 | 1521.7 | 140.6 | 666.4 (33) | 455.8 (17) | 0.684 | 925.3 | 714.7 | 4 / 11 / 20 |
| p128k | 16384 | 3551.9 | 140.7 | 1979.8 (33) | 1509.7 (17) | 0.763 | 1901.5 | 1431.4 | 12 / 23 / 35 |
| p256k | 16384 | 5520.2 | 149.7 | 3818.4 (33) | 2965.5 (17) | 0.777 | 2405.0 | 1552.1 | 14 / 26 / 40 |

At `p128k` and `p256k` the faster arm is aiter#5996, timed here in the same session. An int16 select only as fast as today's fp32 top-k would already beat it: about 3082 µs against 3551.9 (0.868), and 4667 µs against 5520.2 (0.846). The select reads half the bytes of the fp32 top-k. The budget does not include the boundary rescore. That bucket is at most 40 blocks per row, so the rescore reads at most 40 compressed keys per row.

The smaller cases are against vllm#65188, timed in the master's probe-free run and not in this session. At `p32k` that arm is 926.2 µs. The select would have 926.2 − 455.8 − 140.6 = 329.8 µs, about 2.2 times faster than today's top-k. At `p8k` the bar is 1.05 × 204.5 = 214.7 µs, and `S16` plus `E` is already 258.4 µs. That case misses whatever the select does.

### Kernel gate

Median microseconds per QSA layer per step, gfx950. Margin is this path divided by the faster of aiter#5996 and vllm#65188.

| Case | 16-bit path | aiter#5996 | vllm#65188 | Faster arm | Margin | Pass |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| p8k | 1067.8 | 872.3 | 211.0 | vllm#65188 | 5.060 | no |
| p32k | 2968.8 | 1632.8 | 926.8 | vllm#65188 | 3.203 | no |
| p128k | 6084.9 | 3611.5 | 4175.7 | aiter#5996 | 1.685 | no |
| p256k | 9468.2 | 5734.1 | 66481.2 | aiter#5996 | 1.651 | no |
| two128k | 6254.7 | 3710.2 | 4320.6 | aiter#5996 | 1.686 | no |
| four128k | 6380.9 | 3826.8 | 4360.2 | aiter#5996 | 1.667 | no |
| mix3 | 6244.4 | 3762.7 | 4306.8 | aiter#5996 | 1.660 | no |
| mix16 | 6762.3 | 4315.5 | 4329.2 | aiter#5996 | 1.567 | no |
| mix64 | 7656.5 | 4793.8 | 4377.6 | vllm#65188 | 1.749 | no |
| mtp | 6149.2 | 3659.2 | 4308.8 | aiter#5996 | 1.680 | no |

Warm-2 medians from one interleaved session, `op_tests/bench_qsa_indexer_select_abc.py --key16`. Five warmup steps, ten timed steps. The 16-bit figure is the whole `qsa_indexer_key16_score_and_select` call, expand included. aiter#5996 is `qsa_indexer_block_ids` plus expand. The vLLM arms are the sum of their score, top-k, and expand kernels. vllm#62589 ran in the same session and is slower than the faster arm on every case. Cold 16-bit repeats stay within about 1% of these warm-2 medians. At `p8k` the rule is 1.05×. At every other case the rule is strictly faster. Every case misses. `p128k` and `p256k` are 1.685× and 1.651× aiter#5996.

The score-only probe left 1901.5 µs and 2405.0 µs for a select. This whole call is 6084.9 µs and 9468.2 µs, about 4575 µs and 6500 µs past that session's int16 scorer. The select at the bench width does not fit the budget. Serving does not start, and gfx942 is not measured. `_fused_select_wins` stays false. `qsa_indexer_block_ids` stays the public path.

### Serving gate

Filled only if Phase 6 passes. The protocol is the master's Phase 8.

| Input | Concurrency | TTFT 16-bit | TTFT vllm#65188 | ITL 16-bit | ITL vllm#65188 | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| | | | | | | |

### Decode table

No gate. Filled only if Phase 6 passes.

| Rows | Context | 16-bit score | 16-bit select | Current path | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| | | | | | |

## Decision record

Phase 0, 2026-11-19: the stop does not fire. At `p128k` and `p256k` the faster arm is aiter#5996, and this path is a change to that arm, so the miss is a strict improvement on its step. The baseline score stage is 2072.9 µs of 3797.3 µs (0.546) and 3905.9 µs of 5861.4 µs (0.666). That share is larger than the miss. Phase 1 is next. The 16-bit counts, 1019 rows per chunk and 17 chunks on the 16384-row cases, are the workspace formula and are not a measured win.

Outcome: miss. Option 1 stays a miss in the master. The 16-bit path passes the set gate and misses gate 1 on every case. At `p128k` it is 6084.9 µs against aiter#5996 at 3611.5 µs. At `p256k` it is 9468.2 µs against 5734.1 µs. The fused select stays off. Today's indexer stays the public path. Serving does not start.

Gates met: none. Gate 5 stays closed, as the master applied it. Gate 1 misses on all ten cases. The margins are in [Kernel gate](#kernel-gate).

gfx942 enablement: not measured.

Phase 1, 2026-11-19: keep `_score_key16`. On the three seeded cases the boundary rescore matches the oracle set, and the boundary bucket is at most 19 blocks.

Phase 2, 2026-11-19: the 16-row scorer can store int16 keys. On the seeded 32K case the stored keys match the fp32 scorer, the boundary rescore matches that scorer's top-512, and the same-width time is 1.007× the fp32 scorer (22.018 µs vs 21.867 µs). The public dispatch is unchanged. Phase 3 is next.

Phase 3, 2026-11-19: `qsa_indexer_key16_select` keeps every key above the 512th and rescores that bucket with the oracle score. On the seeded 32K case the block set matches the oracle within 1.7e-6, the token ids are the expand of those blocks, and the bucket holds 1365 blocks across 128 rows. A short row keeps every visible block and does not rescore. The public dispatch is unchanged. Phase 4 is next.

Phase 4, 2026-11-19: the int16 workspace holds 1019 rows at 65856 columns. A 16384-row step reuses that one buffer across 17 chunks, and an 8-row tail keeps the full-batch scorer. Key traffic is 1029 MiB at `p8k` and 2058 MiB on the 16384-row cases. The public dispatch is unchanged. Phase 5 is next.

Phase 5, 2026-11-19: the int16 path passes the set gate. `select err` is 0 on all eight of the master's cases: padded, missing page, a cache view past 4 GiB, a tile spanning requests, mixed, MTP, `n_i <= 512`, and workspace reuse. The path is gfx950 only, and its tests skip elsewhere. The public dispatch is unchanged. Phase 6 is next.

Phase 6 budget, 2026-11-19: build a GPU int16 select. In one idle session the int16 scorer is 0.763× and 0.777× the fp32 scorer at `p128k` and `p256k`, 1509.7 µs and 2965.5 µs. That leaves 1901.5 µs and 2405.0 µs under aiter#5996 for the select, and today's fp32 top-k takes 1431.4 µs and 1552.1 µs. The boundary bucket is at most 35 and 40 blocks. The existing radix takes fp32 only. Widening the keys back to fp32 would give back the halved buffer. So the select is a new kernel, and the shared radix stays as it is. `p32k` needs that select under about 330 µs, and `p8k` misses regardless.

Phase 6 select, 2026-11-20: `qsa_indexer_key16_select` is a GPU kernel. One workgroup per row stages the keys, finds the boundary with two 8-bit passes, and rescores that bucket in fp32. On the seeded 32K case the set matches within 1.7e-6 in 45.69 µs, against 1247.49 µs for the host select it replaced. A 900-block bucket and an 8092-block bucket both match. The set gate is 0 on all eight cases, and the 1027-row chunked step is 210.41 µs. The public dispatch is unchanged. The kernel gate is next.

Phase 6 gate, 2026-11-20: stop. The 16-bit whole call misses gate 1 on every case. Warm-2 `p128k` is 6084.9 µs against aiter#5996 at 3611.5 µs (1.685×), and `p256k` is 9468.2 µs against 5734.1 µs (1.651×). The closest case is `mix16` at 1.567×. `p8k` is 5.060× vllm#65188. The int16 scorer was fast enough. The full-width select is not. gfx942 is not measured. `_fused_select_wins` stays false, and `qsa_indexer_block_ids` stays the public path.

Fallback key: `_score_key16`, the high 16 bits of the order-preserving fp32 key.

## Open questions

- On the seeded 8K, 32K, and padded-prefix cases the boundary bucket is at most 19 blocks. Whether it stays that small at 128K and 256K, where a row has many more visible blocks, is still open. Answered 2026-11-19: on the bench's `p128k` and `p256k` rows it is 12 / 23 / 35 and 14 / 26 / 40.
- Does the existing radix consume uint16 keys, or does the boundary path need its own select? Answered 2026-11-19: it needs its own. `flydsl_top_k_per_row_decode` raises on anything but 2D fp32 logits, and `topk_select` serves bf16 and fp16 only at topk=1.
- The master's aiter arm chunks the padded 65,856-column table, and vllm#65188 trims to `max_seq_len`. This plan keeps the aiter width. A later trim is out of scope here.

## Log

- 2026-11-19: Plan opened. The master is [qsa-indexer-fused-select-plan.md](qsa-indexer-fused-select-plan.md), and its fixed decisions carry over. Option 1 misses gate 1 on the probe-free interleaved run: `p128k` fused 3814.2 µs against aiter#5996 at 3623.9 µs, `p256k` 7934.8 µs against 5845.4 µs, and every other case misses, including `p8k` at 1.13× vllm#65188. Option 2 stays out, per the master. This document is the ticket's 16-bit score-buffer fallback. No kernel work has started. `_fused_select_wins` stays false.
- 2026-11-19: Phase 0 budget, from the master's baseline. No GPU run. aiter#5996 scores 65856 columns, 509 fp32 rows per 128 MiB chunk: 17 chunks at `p8k` (8192 rows) and 33 at `p32k`, `p128k`, and `p256k` (16384 rows). Score bytes are 2058.0 MiB and 4116.0 MiB. The 2-byte formula gives 1019 rows per chunk, 9 chunks at `p8k` and 17 on the 16384-row cases, and half the bytes. Stage times are in [Budget](#budget). The faster arm at `p128k` and `p256k` is aiter#5996, and its score stage is 0.546 and 0.666 of `T_step`, larger than the strict improvement this path has to find. The stop does not fire. Phase 1 is next. At `p8k` the faster arm is vllm#65188 at 207.3 µs and the whole aiter score stage is 328.0 µs, so that miss is not a score-buffer miss. `_fused_select_wins` stays false.
- 2026-11-19: Phase 1 key check, gfx950, gpuq GPU 0. `test_qsa_score_key16` uses the oracle's seeded 8K, 32K, and padded-prefix inputs. `_score_key16` stays. Colliding pairs, scores that share a key and differ by more than 1.7e-6, are 975004, 8062017, and 497783. The widest key span is 0.0311. Boundary width is 2/8/18, 3/11/19, and 2/8/17 (min/median/max). The fp32 rescore of that bucket matches the oracle top-512 on every row. Phase 2 is next. `_fused_select_wins` stays false.
- 2026-11-19: Phase 2 store, gfx950, ROCm 7.2.1, bus 0000:75:00.0, rocm-smi GPU 3 at 0% with no other pids. `test_qsa_score_key16_store` on the seeded 32K case, 128 rows, 8232 columns. The int16 keys match `_score_key16` of the fp32 16-row scorer on every visible column, and the boundary rescore matches that scorer's top-512. Same-width CUDA-event means, 5 warmup and 50 timed, are 21.867 µs (fp32) and 22.018 µs (key16), ratio 1.007. `qsa_indexer_block_ids` still stores fp32. Phase 3 is next. `_fused_select_wins` stays false.
- 2026-11-19: Phase 3 select, gfx950, ROCm 7.2.1, bus 0000:65:00.0, rocm-smi GPU 2 at 0% with no other pids. `test_qsa_key16_select` on the seeded 32K case and a 16-row context-2048 case. The 32K block set matches the oracle within 1.7e-6, and both cases' token ids match the existing expand, width 2051. The 32K boundary bucket holds 1365 blocks. The short case does not rescore. CUDA-event means, 5 warmup and 10 timed, are 1247.49 µs and 243.90 µs. `qsa_indexer_block_ids` is unchanged. Phase 4 is next. `_fused_select_wins` stays false.
- 2026-11-19: Phase 4 chunks, gfx950, ROCm 7.2.1, bus 0000:f5:00.0, rocm-smi GPU 7 at 0% with no other pids. The int16 workspace is 1019 rows at 65856 columns. `p8k` is 9 chunks and 1029 MiB of keys. `p32k`, `p128k`, and `p256k` are 17 chunks and 2058 MiB. K is read once per chunk: 4.5, 34, 136, and 272 MiB. An 8-row tail of 1027 rows kept the full-batch scorer, and that step matched the oracle within 1.7e-6. `qsa_indexer_block_ids` is unchanged. Phase 5 is next. `_fused_select_wins` stays false.
- 2026-11-19: Phase 5 set gate, gfx950, ROCm 7.2.1, bus 0000:75:00.0, rocm-smi GPU 3 at 0% with no other pids. `test_qsa_key16_select_gate` on `pad`, `missing`, `wide`, `span`, `mixed`, `mtp`, `emit`, and `workspace`. `select err` is 0 on every case, every row is packed, and the token ids equal the expand. The key buffer started at 12345, a valid key, and `workspace` gave the same ids after a refill to -7. The score entry now runs the 16-row tile on any batch except gfx942 at 8 heads, so the 8-row `mtp` case uses the head fold. `emit` takes the existing emit path. `qsa_indexer_key16_supported` gates the path to gfx950, and `run_qsa_key16_checks` skips every launching check elsewhere; a patched gfx942 lookup skipped all four runners. The Phase 2 to 4 checks passed again in that session, with a same-width store ratio of 0.999. `qsa_indexer_block_ids` is unchanged. Phase 6 is next. `_fused_select_wins` stays false.
- 2026-11-19: Locked decision edit, approved by the user. Under Mechanism, the boundary rescore may run inside the select kernel, as long as it computes the oracle's bf16-in, fp32-accumulate formula and passes the 1.7e-6 tie gate. The Phase 3 box now says the same thing. The Phase 3 host rescore already meets it. No code changed.
- 2026-11-19: Phase 6 select budget, gfx950, ROCm 7.2.1, bus 0000:65:00.0, rocm-smi GPU 2 at 0% with no other pids. `/tmp/key16_budget_probe.py` ran four warm, interleaved arms on the bench's `p8k`, `p32k`, `p128k`, and `p256k`, with 5 warmup and 10 timed rounds, median. Arms: the aiter#5996 whole call, expand, the fp32 scorer in 33 chunks, and the int16 scorer in 17 chunks (9 and 17 at `p8k`). The int16 scorer is 1509.7 µs at `p128k` and 2965.5 µs at `p256k`, 0.763× and 0.777× fp32. That leaves 1901.5 µs and 2405.0 µs for the select. Today's fp32 top-k, derived, is 1431.4 µs and 1552.1 µs, inside that budget. The boundary bucket is 12 / 23 / 35 and 14 / 26 / 40. The radix is fp32 only, so a GPU int16 select is next. `p32k` needs it under about 330 µs against vllm#65188, a number from another session. `p8k` misses regardless. `_fused_select_wins` stays false.
- 2026-11-20: Phase 6 GPU select, gfx950, ROCm 7.2.1, bus 0000:65:00.0, rocm-smi GPU 2 at 0% with no other pids. One earlier grant on bus 0000:05:00.0 was busy and was discarded. `qsa_indexer_key16_select` now launches one workgroup per row. It stages the int16 keys, finds the 512th key with two 8-bit histogram passes, and rescores that bucket in fp32 from compressed K. A bucket of at most 1024 blocks is ranked in LDS. A wider one takes four more 8-bit passes over the recomputed score. `test_qsa_key16_select` matches the oracle within 1.7e-6: 45.69 µs on the seeded 32K case and 37.61 µs on the short row. `test_qsa_key16_select_bucket` matches too, at 136.19 µs for 900 blocks and 1815.30 µs for 8092. The set gate is 0 on all eight cases, 67.52 to 69.70 µs, and the 1027-row chunked step is 210.41 µs. `qsa_indexer_block_ids` is unchanged. The kernel gate is next. `_fused_select_wins` stays false.
- 2026-11-20: Phase 6 kernel gate, gfx950, ROCm 7.2.1, bus 0000:f5:00.0, rocm-smi GPU 7 at 0% with no other pids. `op_tests/bench_qsa_indexer_select_abc.py --key16` interleaved the 16-bit whole call with aiter#5996, vllm#65188, and vllm#62589. Ten cases, five warmup steps, ten timed steps, warm and cold, probe-free. Warm-2 `p128k` is 6084.9 µs against aiter#5996 at 3611.5 µs, and `p256k` is 9468.2 µs against 5734.1 µs. Every case misses gate 1. The closest is `mix16` at 1.567×, and `p8k` is 5.060× vllm#65188. Cold stays within about 1% of warm-2. The earlier score-only probe left about 1900 µs for the select. This whole call is about 4575 µs past that session's int16 scorer at `p128k`, so the full-width select does not fit. Serving does not start. gfx942 is not measured. `_fused_select_wins` stays false. `qsa_indexer_block_ids` stays the public path.

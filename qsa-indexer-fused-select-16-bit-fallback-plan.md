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
- Selection keeps every block whose key is strictly above the bucket that contains the 512th key, drops every block whose key is strictly below it, and rescores the boundary bucket in fp32. The rescore reads the compressed keys again for those blocks only, computes the same bf16-in fp32-accumulate score the oracle uses, and picks the exact remainder of the top 512. Exact ties at the 512th score may resolve to either block.
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

- [ ] Add a scorer path that writes uint16 keys for the visible prefix and leaves the fp32 store unchanged. Same 16-row tile, same bf16 MFMA, same 4-head ReLU sum, same masking of invisible, incomplete, and missing-page blocks.
- [ ] Faithfulness: decoding the stored keys with the Phase 1 rule matches the fp32 scorer's visible scores on the 32K case, within the tie bound, and the kernel time stays within 5% of the same-session fp32 scorer when the store width is not the thing under test. The 5% check uses a same-width comparison only if one is cheap to run; the real comparison is the halved buffer in Phase 4.
- [ ] Do not change the production dispatch.

### 3. Select on keys, rescore the boundary

- [ ] Select the top 512 from the uint16 buffer by the Phase 1 rule. Rescore only the boundary bucket, in fp32, from the compressed K.
- [ ] The rescore uses the oracle's score, not another production kernel's score.
- [ ] Expand with the existing expand. Token ids match the set gate's contract: `[M, 2051]`, tail included, `-1` padded.

### 4. Chunk the 2-byte buffer

- [ ] Size the workspace at 2 bytes per column under the 128 MiB cap. A full step of 16384 rows reuses one buffer across chunks.
- [ ] A short last chunk keeps the scorer and the selector chosen for the full batch.
- [ ] Record chunk count and bytes written and read per case.

### 5. Set gate

- [ ] Run the master's set-gate cases: padded tables, missing pages, a cache view spanning more than 4 GiB, row tiles spanning requests, mixed steps, MTP rows, and `n_i <= 512`.
- [ ] `select err` is the fraction of rows outside the 1.7e-6 window. The gate passes only at 0 on every case.
- [ ] Tests skip on archs where this path is off.

### 6. Kernel gate

- [ ] One idle gfx950 session, probe-free, interleaved with aiter#5996, vllm#65188, and vllm#62589. Same ten cases as the master's Phase 6 matrix. Five warmup steps, ten timed steps, warm and cold.
- [ ] Fill [Kernel gate](#kernel-gate). Apply gate 1 per case.
- [ ] If `p128k` or `p256k` misses, record the miss and stop. Do not start serving.
- [ ] Measure gfx942. Enable this path there only for the cases where it wins.

### 7. Dispatch

- [ ] Add a host predicate, default false. Turn it on only for shapes that won in Phase 6.
- [ ] The fp32 entry and `_fused_select_wins` stay as they are.
- [ ] Run the existing `op_tests/test_flydsl_qsa.py` suite and the new tests through `gpuq`.

### 8. Close or hand back

- [ ] If Phase 6 passes, serving and the decode table follow the master's Phases 8 and 9, and the numbers land in this document's tables.
- [ ] If Phase 6 misses, write the outcome in [Decision record](#decision-record). The fused select stays off. Today's indexer stays the public path.
- [ ] Log the date.

## Results

### Environment confirmation

| Session | Date | `get_gfx()` | gpuq / rocm-smi / bus | Device | Idle |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

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

### Kernel gate

Median microseconds per QSA layer per step, gfx950. Margin is this path divided by the faster of aiter#5996 and vllm#65188.

| Case | 16-bit path | aiter#5996 | vllm#65188 | Faster arm | Margin | Pass |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| p8k | | | | | | |
| p32k | | | | | | |
| p128k | | | | | | |
| p256k | | | | | | |
| two128k | | | | | | |
| four128k | | | | | | |
| mix3 | | | | | | |
| mix16 | | | | | | |
| mix64 | | | | | | |
| mtp | | | | | | |

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

Outcome: open. Option 1 stays a miss in the master. This fallback has not been built.

Gates met: none. Gate 5 stays closed, as the master applied it.

Margins: not measured. The budget is the table under [Budget](#budget).

gfx942 enablement: not measured.

Phase 1, 2026-11-19: keep `_score_key16`. On the three seeded cases the boundary rescore matches the oracle set, and the boundary bucket is at most 19 blocks.

Fallback key: `_score_key16`, the high 16 bits of the order-preserving fp32 key.

## Open questions

- On the seeded 8K, 32K, and padded-prefix cases the boundary bucket is at most 19 blocks. Whether it stays that small at 128K and 256K, where a row has many more visible blocks, is still open.
- Does the existing radix consume uint16 keys, or does the boundary path need its own select?
- The master's aiter arm chunks the padded 65,856-column table, and vllm#65188 trims to `max_seq_len`. This plan keeps the aiter width. A later trim is out of scope here.

## Log

- 2026-11-19: Plan opened. The master is [qsa-indexer-fused-select-plan.md](qsa-indexer-fused-select-plan.md), and its fixed decisions carry over. Option 1 misses gate 1 on the probe-free interleaved run: `p128k` fused 3814.2 µs against aiter#5996 at 3623.9 µs, `p256k` 7934.8 µs against 5845.4 µs, and every other case misses, including `p8k` at 1.13× vllm#65188. Option 2 stays out, per the master. This document is the ticket's 16-bit score-buffer fallback. No kernel work has started. `_fused_select_wins` stays false.
- 2026-11-19: Phase 0 budget, from the master's baseline. No GPU run. aiter#5996 scores 65856 columns, 509 fp32 rows per 128 MiB chunk: 17 chunks at `p8k` (8192 rows) and 33 at `p32k`, `p128k`, and `p256k` (16384 rows). Score bytes are 2058.0 MiB and 4116.0 MiB. The 2-byte formula gives 1019 rows per chunk, 9 chunks at `p8k` and 17 on the 16384-row cases, and half the bytes. Stage times are in [Budget](#budget). The faster arm at `p128k` and `p256k` is aiter#5996, and its score stage is 0.546 and 0.666 of `T_step`, larger than the strict improvement this path has to find. The stop does not fire. Phase 1 is next. At `p8k` the faster arm is vllm#65188 at 207.3 µs and the whole aiter score stage is 328.0 µs, so that miss is not a score-buffer miss. `_fused_select_wins` stays false.
- 2026-11-19: Phase 1 key check, gfx950, gpuq GPU 0. `test_qsa_score_key16` uses the oracle's seeded 8K, 32K, and padded-prefix inputs. `_score_key16` stays. Colliding pairs, scores that share a key and differ by more than 1.7e-6, are 975004, 8062017, and 497783. The widest key span is 0.0311. Boundary width is 2/8/18, 3/11/19, and 2/8/17 (min/median/max). The fp32 rescore of that bucket matches the oracle top-512 on every row. Phase 2 is next. `_fused_select_wins` stays false.
